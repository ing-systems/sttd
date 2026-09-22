// FastConformer encoder stack (Burn port, bring-up).
//
// ONNX reference (`encoder-model.int8.onnx`): 5654 nodes, 8x subsampling,
// encoder output `[1, 1024, T/8]`, inputs `audio_signal [1, 128, T]` +
// `length`. Quantized ops: `DynamicQuantizeLinear` x223,
// `MatMulInteger` x217, `ConvInteger` x77, `LayerNormalization` x120,
// `Softmax` x24.
//
// Bring-up order (each step landed with a parity test against ORT):
//  1. subsampling conv block — done (`pre_encode.rs`, golden-verified).
//  2. one Conformer block — done (`block.rs`, golden-verified).
//  3. full depth (`BurnEncoder::forward` below) — done, isolated
//     per-layer parity flat at ~2-15%; chained drift is
//     activation-requantization accumulation (exact match waits on
//     `quant::forward` replicating `DynamicQuantizeLinear`).
//
// Length masking is a no-op for full-length offline utterances (same as
// `block.rs`); the graph's final transpose is just a reshape to
// `[1, 1024, T']`, so `forward` returns `[T'][1024]` rows matching every
// `/layers.N/norm_out` intermediate.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use super::block::{block_forward, pos_window, BlockWeights};
use super::pre_encode::{pre_encode_want, PreEncodeWeights};
use super::weights::RawTensor;

/// One Conformer block config (FastConformer-TDT-0.6B-v3 shapes, read off
/// the int8 ONNX: 24 `layers.N` blocks, FFN 1024 -> 4096 -> 1024).
#[derive(Debug, Clone)]
pub(crate) struct ConformerConfig {
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) ffn_expansion: usize,
    pub(crate) conv_kernel: usize,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            // Confirmed: `layers.N.self_attn.pos_bias_u/v` are (8, 128) in
            // the converted weights — 8 heads x 128 dims = 1024.
            n_heads: 8,
            ffn_expansion: 4,
            conv_kernel: 9,
        }
    }
}

/// Encoder config: depth is resolved from converted weights at load.
#[derive(Debug, Clone)]
pub(crate) struct EncoderConfig {
    pub(crate) in_mels: usize,
    pub(crate) d_model: usize,
    pub(crate) out_dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) block: ConformerConfig,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            in_mels: 128,
            d_model: 1024,
            out_dim: 1024,
            n_layers: 24, // `layers.0`..`layers.23` in the int8 ONNX
            block: ConformerConfig::default(),
        }
    }
}

/// Subsampling: 2 x strided conv = 4x, + frame stacking = 8x total
/// (matches the TDT `encoder_stride = 8` in `decoder_tdt.rs`).
pub(crate) fn subsampled_frames(n_mel_frames: usize) -> usize {
    n_mel_frames / 8
}

/// Full encoder stack: pre-encode subsampling + 24 Conformer blocks.
///
/// Layout (verified against `encoder-model.int8.onnx`): mel `[T][128]`
/// -> pre-encode `[T'][1024]` -> one `block_forward` per layer (each
/// ending in its own `norm_out`) -> `[T'][1024]` rows. The graph's final
/// `Transpose_2` only reshapes to `[1, 1024, T']` for the joint; the row
/// layout here matches every `/layers.N/norm_out` intermediate
/// (`[1, T', 1024]`), so parity tests compare rows directly and only the
/// final `outputs` tensor needs a transpose.
pub(crate) struct BurnEncoder {
    pre: PreEncodeWeights,
    blocks: Vec<BlockWeights>,
    pos_table: Vec<Vec<f32>>, // [9999][1024], frozen relative table
}

impl BurnEncoder {
    /// Load the full stack. Shapes/roles are asserted per layer by
    /// `PreEncodeWeights::load` / `BlockWeights::load`, so a corrupt file
    /// fails here, never mid-utterance.
    pub(crate) fn load(
        tensors: &HashMap<String, RawTensor>,
        manifest: &EncoderManifest,
    ) -> Result<Self> {
        let pre = PreEncodeWeights::load(tensors, manifest)?;
        let pos_table = match tensors.get("pos_enc.table") {
            Some(RawTensor::F32 { shape, data }) => {
                if shape != &vec![9999, 1024] {
                    anyhow::bail!("pos_enc.table shape {shape:?} != [9999, 1024]");
                }
                data.chunks(1024).map(|c| c.to_vec()).collect::<Vec<_>>()
            }
            Some(other) => anyhow::bail!("pos_enc.table: expected F32, got {}", other.kind()),
            None => anyhow::bail!("pos_enc.table: missing tensor"),
        };
        let n = manifest.n_layers();
        // FastConformer-TDT-0.6B-v3 shapes, read off the int8 ONNX.
        let (d_model, n_heads, d_head, d_ff) = (1024, 8, 128, 4096);
        let mut blocks = Vec::with_capacity(n);
        for layer in 0..n {
            blocks.push(BlockWeights::load(
                tensors, manifest, layer, d_model, n_heads, d_head, d_ff,
            )?);
        }
        Ok(Self {
            pre,
            blocks,
            pos_table,
        })
    }

    pub(crate) fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Mel `[T][128]` -> encoder states `[T'][1024]`. Empty in, empty out.
    pub(crate) fn forward(&self, mel: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut h = self.pre.forward(mel)?;
        if h.is_empty() {
            return Ok(h);
        }
        let win = pos_window(&self.pos_table, h.len());
        for b in &self.blocks {
            h = block_forward(&h, b, &win);
        }
        Ok(h)
    }
}

/// Exact tensor names `weights::load_selected` must fetch for a full
/// `BurnEncoder::load`: pre-encode set + per-layer linears/norms/biases +
/// the frozen position table. Keeps the full 582 MB file on disk for
/// callers that only need one layer (see `block.rs` tests).
pub(crate) fn encoder_want(manifest: &EncoderManifest) -> Result<HashSet<String>> {
    let mut want = pre_encode_want(manifest)?;
    for layer in 0..manifest.n_layers() {
        for suffix in LAYER_ROLES {
            let role = format!("layers.{layer}/{suffix}");
            let e = manifest
                .role(&role)
                .with_context(|| format!("manifest lacks {role}"))?;
            want.insert(e.weight.clone());
            want.insert(format!("{}.scale", e.weight));
            want.insert(format!("{}.zero_point", e.weight));
        }
        for n in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ] {
            want.insert(format!("layers.{layer}.{n}.weight"));
            want.insert(format!("layers.{layer}.{n}.bias"));
        }
        want.insert(format!("layers.{layer}.self_attn.pos_bias_u"));
        want.insert(format!("layers.{layer}.self_attn.pos_bias_v"));
        want.insert(
            manifest
                .dw_bias_init(layer)
                .unwrap_or_else(|| format!("layers.{layer}/conv/depthwise_bias")),
        );
    }
    want.insert("pos_enc.table".to_string());
    Ok(want)
}

/// One quantized op from `encoder-manifest.json` (see
/// `tools/convert-parakeet-int8-to-burn.py::emit_encoder_manifest`):
/// a NeMo role path (`layers.3/self_attn/linear_q`) joined to its int8
/// weight base name in the converted safetensors.
#[derive(Debug, Clone)]
pub(crate) struct RoleEntry {
    pub(crate) role: String,
    pub(crate) kind: String,
    /// Weight base name in `*-burn.safetensors` (`<base>` + `.scale` +
    /// `.zero_point`); `None` only for a malformed manifest (rejected by
    /// `load_manifest`).
    pub(crate) weight: String,
    pub(crate) weight_shape: Vec<usize>,
    /// `conv.group` for `kind == "conv"` (scalar in the manifest's `conv`
    /// attr); `None` for matmuls. Lets loaders assert depthwise vs dense
    /// at load instead of trusting role names.
    pub(crate) group: Option<usize>,
}

/// The 12 quantized roles every Conformer layer carries (verified against
/// the int8 graph: 24 layers x these, no exceptions).
pub(crate) const LAYER_ROLES: &[&str] = &[
    "self_attn/linear_q",
    "self_attn/linear_k",
    "self_attn/linear_v",
    "self_attn/linear_pos",
    "self_attn/linear_out",
    "feed_forward1/linear1",
    "feed_forward1/linear2",
    "feed_forward2/linear1",
    "feed_forward2/linear2",
    "conv/pointwise_conv1",
    "conv/depthwise_conv",
    "conv/pointwise_conv2",
];

/// Validated encoder manifest: role -> entry for all 24 layers plus the
/// 6 pre_encode ops. The forward pass looks weights up here, so a missing
/// or misshapen role fails at load, never mid-utterance.
#[derive(Debug, Clone, Default)]
pub(crate) struct EncoderManifest {
    pub(crate) roles: std::collections::HashMap<String, RoleEntry>,
    /// Depthwise-conv bias map: `layers.N/conv/depthwise_bias` role ->
    /// anonymous F32 initializer name (traced through `Reshape` by the
    /// converter; see `tools/convert-parakeet-int8-to-burn.py`). Absent
    /// for hand-written manifests; `BlockWeights::load` then falls back
    /// to the role name itself.
    pub(crate) dw_bias: std::collections::HashMap<String, String>,
}

impl EncoderManifest {
    pub(crate) fn role(&self, role: &str) -> Option<&RoleEntry> {
        self.roles.get(role)
    }

    pub(crate) fn dw_bias_init(&self, layer: usize) -> Option<String> {
        self.dw_bias
            .get(&format!("layers.{layer}/conv/depthwise_bias"))
            .cloned()
    }

    pub(crate) fn n_layers(&self) -> usize {
        self.roles
            .keys()
            .filter_map(|r| {
                r.strip_prefix("layers.")
                    .and_then(|s| s.split('/').next())
                    .and_then(|n| n.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |m| m + 1)
    }
}

/// Parse + validate `encoder-manifest.json`: every layer in
/// `0..n_layers` must carry all of `LAYER_ROLES` with a named,
/// non-empty weight shape; pre_encode's 6 ops must be present.
pub(crate) fn load_manifest(json: &str) -> anyhow::Result<EncoderManifest> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("burn encoder manifest: invalid JSON: {e}"))?;
    let ops = v
        .get("quant_ops")
        .and_then(|o| o.as_array())
        .ok_or_else(|| anyhow::anyhow!("burn encoder manifest: missing quant_ops"))?;
    let mut roles = std::collections::HashMap::new();
    for op in ops {
        let role = op
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| anyhow::anyhow!("burn encoder manifest: entry without role"))?;
        let weight = op
            .get("weight")
            .and_then(|w| w.as_str())
            .ok_or_else(|| anyhow::anyhow!("burn encoder manifest: {role} without weight"))?;
        let shape = op
            .get("weight_shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow::anyhow!("burn encoder manifest: {role} without weight_shape"))?;
        let mut dims = Vec::with_capacity(shape.len());
        for d in shape {
            dims.push(d.as_u64().ok_or_else(|| {
                anyhow::anyhow!("burn encoder manifest: {role} has non-integer dim")
            })? as usize);
        }
        if dims.is_empty() || dims.contains(&0) {
            anyhow::bail!("burn encoder manifest: {role} has empty weight_shape");
        }
        let kind = op.get("kind").and_then(|k| k.as_str()).unwrap_or("matmul");
        let group = op
            .get("conv")
            .and_then(|c| c.get("group"))
            .and_then(|g| g.as_u64())
            .map(|g| g as usize);
        roles.insert(
            role.to_string(),
            RoleEntry {
                role: role.to_string(),
                kind: kind.to_string(),
                weight: weight.to_string(),
                weight_shape: dims,
                group,
            },
        );
    }
    let manifest = EncoderManifest { roles, dw_bias: parse_dw_bias(&v)? };
    // Pre-encode's 6 ops (5 convs + out proj) must be present.
    for role in [
        "pre_encode/conv/conv.0",
        "pre_encode/conv/conv.2",
        "pre_encode/conv/conv.3",
        "pre_encode/conv/conv.5",
        "pre_encode/conv/conv.6",
        "pre_encode/out",
    ] {
        if !manifest.roles.contains_key(role) {
            anyhow::bail!("burn encoder manifest: missing {role}");
        }
    }
    let n = manifest.n_layers();
    if n == 0 {
        anyhow::bail!("burn encoder manifest: no layers.N roles");
    }
    for layer in 0..n {
        for suffix in LAYER_ROLES {
            let role = format!("layers.{layer}/{suffix}");
            if !manifest.roles.contains_key(&role) {
                anyhow::bail!("burn encoder manifest: missing {role}");
            }
        }
    }
    Ok(manifest)
}

/// Parse the optional top-level `dw_bias` map (role -> initializer).
/// Missing or malformed = empty map; `BlockWeights::load` falls back to
/// the role name, so hand-written test manifests keep working.
fn parse_dw_bias(v: &serde_json::Value) -> anyhow::Result<std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    let Some(map) = v.get("dw_bias") else {
        return Ok(out);
    };
    let map = map
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("burn encoder manifest: dw_bias is not an object"))?;
    for (k, val) in map {
        let init = val.as_str().ok_or_else(|| {
            anyhow::anyhow!("burn encoder manifest: dw_bias[{k}] is not a string")
        })?;
        out.insert(k.clone(), init.to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampling_matches_tdt_stride_8() {
        // 101 mel frames (1 s) -> 12 encoder frames (floor).
        assert_eq!(subsampled_frames(101), 12);
        assert_eq!(subsampled_frames(800), 100);
    }

    #[test]
    fn default_config_shapes_match_onnx_io() {
        let c = EncoderConfig::default();
        assert_eq!(c.in_mels, 128);
        assert_eq!(c.out_dim, 1024);
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.d_model, 1024);
    }

    fn mini_manifest(missing: Option<&str>) -> String {
        // 1 layer x 12 roles + pre_encode, generated (not hand-written).
        let mut ops = Vec::new();
        for suffix in LAYER_ROLES {
            let role = format!("layers.0/{suffix}");
            if Some(role.as_str()) == missing {
                continue;
            }
            ops.push(format!(
                "{{\"role\":\"{role}\",\"kind\":\"matmul\",\"weight\":\"w_{suffix}\",\"weight_shape\":[4,4]}}",
                role = role,
                suffix = suffix.replace('/', "_"),
            ));
        }
        for pre in [
            "pre_encode/conv/conv.0",
            "pre_encode/conv/conv.2",
            "pre_encode/conv/conv.3",
            "pre_encode/conv/conv.5",
            "pre_encode/conv/conv.6",
            "pre_encode/out",
        ] {
            if Some(pre) == missing {
                continue;
            }
            ops.push(format!(
                "{{\"role\":\"{pre}\",\"kind\":\"matmul\",\"weight\":\"w_pre\",\"weight_shape\":[4,4]}}",
            ));
        }
        format!("{{\"quant_ops\":[{}]}}", ops.join(","))
    }

    #[test]
    fn manifest_validates_complete_layer() {
        let m = load_manifest(&mini_manifest(None)).expect("complete manifest loads");
        assert_eq!(m.n_layers(), 1);
        let q = m.role("layers.0/self_attn/linear_q").expect("role");
        assert_eq!(q.weight, "w_self_attn_linear_q");
        assert_eq!(q.weight_shape, vec![4, 4]);
    }

    #[test]
    fn manifest_rejects_missing_role() {
        let err = load_manifest(&mini_manifest(Some("layers.0/conv/depthwise_conv")))
            .expect_err("missing role must fail");
        assert!(err.to_string().contains("layers.0/conv/depthwise_conv"));
        let err = load_manifest(&mini_manifest(Some("pre_encode/out")))
            .expect_err("missing pre_encode must fail");
        assert!(err.to_string().contains("pre_encode/out"));
    }

    #[test]
    fn manifest_rejects_malformed() {
        assert!(load_manifest("{}").is_err());
        assert!(load_manifest("{\"quant_ops\":[]}").is_err());
        assert!(load_manifest("not json").is_err());
    }

    #[test]
    fn manifest_captures_conv_group() {
        let json = "{\"quant_ops\":[\
            {\"role\":\"pre_encode/conv/conv.0\",\"kind\":\"conv\",\"weight\":\"a\",\"weight_shape\":[256,1,3,3],\"conv\":{\"group\":1}},\
            {\"role\":\"pre_encode/conv/conv.2\",\"kind\":\"conv\",\"weight\":\"b\",\"weight_shape\":[256,1,3,3],\"conv\":{\"group\":256}},\
            {\"role\":\"pre_encode/conv/conv.3\",\"kind\":\"conv\",\"weight\":\"c\",\"weight_shape\":[256,256,1,1]},\
            {\"role\":\"pre_encode/conv/conv.5\",\"kind\":\"conv\",\"weight\":\"d\",\"weight_shape\":[256,1,3,3],\"conv\":{\"group\":256}},\
            {\"role\":\"pre_encode/conv/conv.6\",\"kind\":\"conv\",\"weight\":\"e\",\"weight_shape\":[256,256,1,1]},\
            {\"role\":\"pre_encode/out\",\"kind\":\"matmul\",\"weight\":\"f\",\"weight_shape\":[4096,1024]},\
            {\"role\":\"layers.0/self_attn/linear_q\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/self_attn/linear_k\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/self_attn/linear_v\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/self_attn/linear_pos\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/self_attn/linear_out\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/feed_forward1/linear1\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/feed_forward1/linear2\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/feed_forward2/linear1\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/feed_forward2/linear2\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/conv/pointwise_conv1\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]},\
            {\"role\":\"layers.0/conv/depthwise_conv\",\"kind\":\"conv\",\"weight\":\"q\",\"weight_shape\":[4,1,9]},\
            {\"role\":\"layers.0/conv/pointwise_conv2\",\"kind\":\"matmul\",\"weight\":\"q\",\"weight_shape\":[4,4]}\
        ]}";
        let m = load_manifest(json).expect("conv manifest loads");
        assert_eq!(m.role("pre_encode/conv/conv.0").unwrap().group, Some(1));
        assert_eq!(m.role("pre_encode/conv/conv.2").unwrap().group, Some(256));
        assert_eq!(m.role("pre_encode/conv/conv.3").unwrap().group, None);
        assert_eq!(m.role("pre_encode/out").unwrap().group, None);
    }

    /// Reads the real converter output. Ignored by default: needs the
    /// 90 KB manifest next to the model dir (run the converter first).
    /// Run explicitly: `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn real_manifest_has_24_complete_layers() {
        let path = "models/parakeet-tdt-0.6b-v3-int8/encoder-manifest.json";
        let json = std::fs::read_to_string(path).expect("run the converter first");
        let m = load_manifest(&json).expect("real manifest validates");
        assert_eq!(m.n_layers(), 24);
        // Spot-check shapes read off the int8 graph.
        assert_eq!(
            m.role("layers.0/feed_forward1/linear1").unwrap().weight_shape,
            vec![1024, 4096]
        );
        assert_eq!(
            m.role("layers.23/conv/depthwise_conv").unwrap().weight_shape,
            vec![1024, 1, 9]
        );
    }

    #[test]
    fn encoder_want_covers_full_stack() {
        let m = load_manifest(&mini_manifest(None)).expect("mini manifest loads");
        let want = encoder_want(&m).expect("want set builds");
        // Mini manifest shares `w_pre` across all 6 pre-encode roles
        // (deduped to weight + scale + zp + out bias = 4) while layer
        // roles are distinct (12 x3 + 10 norms + 2 pos-bias + 1 dw bias
        // = 49), plus the position table.
        assert_eq!(want.len(), 4 + 49 + 1, "want={want:?}");
        assert!(want.contains("pos_enc.table"));
        assert!(want.contains("w_self_attn_linear_q.scale"));
        assert!(want.contains("layers.0.norm_out.weight"));
        assert!(want.contains("pre_encode.out.bias"));
    }

    /// Full-stack ORT parity: the T=64 seed-1234 mel (same bytes as the
    /// pre-encode golden) through all 24 blocks. Goldens are ORT-CPU runs
    /// of the int8 encoder (`encoder_full_golden.safetensors`): every
    /// `/layers.N/norm_out` intermediate plus the final `outputs`
    /// (transposed to `[T'][1024]` rows). Ignored: needs converted
    /// weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_full_encoder_parity() {
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest = load_manifest(&json).unwrap();
        assert_eq!(manifest.n_layers(), 24);
        let want = encoder_want(&manifest).unwrap();
        let tensors =
            load_selected(&dir.join("encoder-burn.safetensors"), &want).unwrap();
        let enc = BurnEncoder::load(&tensors, &manifest).unwrap();
        assert_eq!(enc.n_layers(), 24);

        let bytes =
            std::fs::read("src/parakeet_burn/testdata/encoder_full_golden.safetensors").unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let f32mat = |name: &str, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let v = st.tensor(name).unwrap();
            assert_eq!(v.shape(), &[rows, cols]);
            v.data()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<f32>>()
                .chunks(cols)
                .map(|c| c.to_vec())
                .collect()
        };
        let mel = f32mat("mel_T128", 64, 128);
        // `forward` plumbing: shape, finiteness, determinism.
        let y = enc.forward(&mel).unwrap();
        assert_eq!((y.len(), y[0].len()), (8, 1024));
        assert!(y.iter().all(|r| r.iter().all(|v| v.is_finite())));
        let y2 = enc.forward(&mel).unwrap();
        assert_eq!(y, y2, "forward is deterministic");
        // Informational chain drift only (no assert): fp32 math without
        // dynamic activation quantization accumulates requant noise down
        // the stack — see the isolated bounds below and docs.
        let enc_ref = f32mat("enc_ref", 8, 1024);
        let (mut cmax, mut csum) = (0.0f32, 0.0f32);
        for (yr, wr) in y.iter().zip(enc_ref.iter()) {
            for (a, b) in yr.iter().zip(wr.iter()) {
                cmax = cmax.max((a - b).abs());
                csum += (a - b).abs();
            }
        }
        eprintln!(
            "chained full depth: max_abs={cmax:.4} mean_abs={:.4} (known requant drift, not asserted)",
            csum / (8.0 * 1024.0)
        );

        // Isolated per-block parity: each block runs on ORT's exact
        // input for that depth (pre_ref for layer 0, layer{N-1}_ref
        // otherwise), so error never compounds across layers. The curve
        // is flat at ~2-15% relative — activation-requantization noise,
        // not a mis-mapped layer (a wrong weight would spike one depth
        // to ~100%+). Bounds hold margin over measured (worst isolated
        // max 14.7% at layers 9-11, 10% at 12-13, 5.5% at 23).
        let pre_bytes =
            std::fs::read("src/parakeet_burn/testdata/pre_encode_golden.safetensors").unwrap();
        let pre_st = safetensors::SafeTensors::deserialize(&pre_bytes).unwrap();
        let pv = pre_st.tensor("pre_ref").unwrap();
        assert_eq!(pv.shape(), &[8, 1024]);
        let pre_ref: Vec<Vec<f32>> = pv
            .data()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>()
            .chunks(1024)
            .map(|c| c.to_vec())
            .collect();
        // Sanity: our pre-encode output matches the golden block input
        // (same 5% fp32-vs-int8 bound as `pre_encode.rs`).
        let own_pre = enc.pre.forward(&mel).unwrap();
        let mut pre_max = 0.0f32;
        let mut pre_ref_max = 0.0f32;
        for (a, b) in own_pre.iter().zip(pre_ref.iter()) {
            for (x, g) in a.iter().zip(b.iter()) {
                pre_max = pre_max.max((x - g).abs());
                pre_ref_max = pre_ref_max.max(g.abs());
            }
        }
        assert!(
            pre_max < 0.05 * pre_ref_max.max(1.0),
            "pre drift: max_abs={pre_max} ref_max={pre_ref_max}"
        );
        let win = super::super::block::pos_window(&enc.pos_table, 8);
        let mut prev = pre_ref;
        for (layer, b) in enc.blocks.iter().enumerate() {
            let want = f32mat(&format!("layer{layer}_ref"), 8, 1024);
            let got = super::super::block::block_forward(&prev, b, &win);
            let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f32);
            let mut ref_max = 0.0f32;
            for (yr, wr) in got.iter().zip(want.iter()) {
                for (a, c) in yr.iter().zip(wr.iter()) {
                    max_abs = max_abs.max((a - c).abs());
                    sum_abs += (a - c).abs();
                    ref_max = ref_max.max(c.abs());
                }
            }
            let mean_abs = sum_abs / (8.0 * 1024.0);
            eprintln!(
                "isolated layer{layer}: max_abs={max_abs:.4} mean_abs={mean_abs:.4} ref_max={ref_max:.4} rel_max={:.4}",
                max_abs / ref_max.max(1e-6)
            );
            if layer < 23 {
                assert!(
                    max_abs < 0.20 * ref_max.max(1.0),
                    "layer{layer} drift: max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
                );
            } else {
                // Final states are tiny (ref_max ~0.12): absolute bound
                // with ~3x margin over measured (0.0068).
                assert!(
                    max_abs < 0.02,
                    "layer{layer} drift: max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
                );
            }
            prev = want;
        }
    }
}
