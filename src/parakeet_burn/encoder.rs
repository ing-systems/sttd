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
use super::tensor::TensorCtx;
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

    /// Build every cached tensor in the stack now (load-time warming):
    /// pre-encode first, then all 24 layers in parallel over rayon's
    /// pool (each layer warms independently — see `BlockWeights::warm`).
    /// Without this the first utterance pays the full dequant + `from_data`
    /// build (~4 s); warmed, every forward reuses the tensors.
    pub(crate) fn warm(&self, device: &super::backend::CpuDevice) {
        use rayon::prelude::*;
        self.pre.warm(device);
        self.blocks.par_iter().for_each(|b| b.warm(device));
    }

    /// Mel `[T][128]` -> encoder states `[T'][1024]`. Empty in, empty out.
    pub(crate) fn forward(&self, mel: &[Vec<f32>], ctx: &TensorCtx) -> Result<Vec<Vec<f32>>> {
        let mut h = self.pre.forward(mel, ctx)?;
        if h.is_empty() {
            return Ok(h);
        }
        let win = pos_window(&self.pos_table, h.len());
        for b in &self.blocks {
            h = block_forward(&h, b, &win, ctx);
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
        // `forward` plumbing: shape, finiteness, determinism (scalar path).
        let sctx = TensorCtx::scalar();
        let y = enc.forward(&mel, &sctx).unwrap();
        assert_eq!((y.len(), y[0].len()), (8, 1024));
        assert!(y.iter().all(|r| r.iter().all(|v| v.is_finite())));
        let y2 = enc.forward(&mel, &sctx).unwrap();
        assert_eq!(y, y2, "forward is deterministic");
        // Informational chained residual only (no assert): with the
        // integer quantized paths the full-depth drift is down to ~0.01
        // absolute (was 0.05 pre-quantization). Isolated per-layer bounds
        // below are the real gates; the file-mode smoke test covers e2e.
        let enc_ref = f32mat("enc_ref", 8, 1024);
        let (mut cmax, mut csum) = (0.0f32, 0.0f32);
        for (yr, wr) in y.iter().zip(enc_ref.iter()) {
            for (a, b) in yr.iter().zip(wr.iter()) {
                cmax = cmax.max((a - b).abs());
                csum += (a - b).abs();
            }
        }
        eprintln!(
            "chained full depth: max_abs={cmax:.4} mean_abs={:.4} (informational, not asserted)",
            csum / (8.0 * 1024.0)
        );

        // Isolated per-block parity: each block runs on ORT's exact
        // input for that depth (pre_ref for layer 0, layer{N-1}_ref
        // otherwise), so error never compounds across layers. With the
        // integer quantized paths + f64 layernorm + voted shift, most
        // layers land <= 2e-4 absolute; layers 10/11/16/21 keep <= 0.94
        // (0.6% rel, last-ULP accumulation inside ORT's own reductions).
        // A mis-mapped weight would spike one depth to ~100%+ instead.
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
        // Isolated per-block parity runs on BOTH paths (scalar fallback
        // + tensor): each block runs on ORT's exact input for its depth,
        // so error never compounds. Tensor bounds match scalar per
        // policy (b) — loosened only if Flex summation order exceeds
        // them while transcripts stay word-exact.
        let win = super::super::block::pos_window(&enc.pos_table, 8);
        let tctx = TensorCtx {
            device: super::super::backend::cpu_device(),
            enabled: true,
        };
        for (tag, ctx) in [("scalar", sctx), ("tensor", tctx)] {
            // Sanity per path: pre-encode output matches the golden
            // block input (same 5% fp32-vs-int8 bound as `pre_encode.rs`;
            // this also covers the pre-encode out-projection tensor twin).
            let own_pre = enc.pre.forward(&mel, &ctx).unwrap();
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
                "{tag} pre drift: max_abs={pre_max} ref_max={pre_ref_max}"
            );
            let mut prev = pre_ref.clone();
            for (layer, b) in enc.blocks.iter().enumerate() {
                let want = f32mat(&format!("layer{layer}_ref"), 8, 1024);
                let got = super::super::block::block_forward(&prev, b, &win, &ctx);
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
                "{tag} isolated layer{layer}: max_abs={max_abs:.4} mean_abs={mean_abs:.4} ref_max={ref_max:.4} rel_max={:.4}",
                max_abs / ref_max.max(1e-6)
            );
            if layer < 23 {
                // Near-exact integer paths: most layers land <= 2e-4;
                // layers 10/11/16/21 keep <= 0.94 (0.6% rel, last-ULP
                // accumulation in ORT's own reductions). Bound holds
                // ~3x margin over the worst measured.
                assert!(
                    max_abs < 0.02 * ref_max.max(1.0),
                    "{tag} layer{layer} drift: max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
                );
            } else {
                // Final states are tiny (ref_max ~0.12): absolute bound
                // with ~3x margin over measured (0.0063).
                assert!(
                    max_abs < 0.02,
                    "{tag} layer{layer} drift: max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
                );
            }
            prev = want;
            }
        }
    }

    /// Timing breakdown on real weights (release only — debug times are
    /// meaningless). No asserts on speed: prints a per-stage table so
    /// optimization targets are measured, not guessed. Ignored: needs
    /// converted weights on disk.
    /// `cargo test --release --features parakeet-burn -- --ignored profile --nocapture`.
    #[test]
    #[ignore]
    fn profile_breakdown() {
        use std::time::Instant;
        use super::super::block::{
            block_forward, conv_block, ffn, layernorm_batch, pos_window, rel_attention,
        };
        use super::super::weights::load_selected;
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;

        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest = load_manifest(&json).unwrap();
        let t0 = Instant::now();
        let tensors =
            load_selected(&dir.join("encoder-burn.safetensors"), &encoder_want(&manifest).unwrap())
                .unwrap();
        let enc = BurnEncoder::load(&tensors, &manifest).unwrap();
        eprintln!("profile: encoder load+warm = {:.0} ms", ms(t0));
        // Steady-state only: the binary warms at load (see `mod.rs`), so
        // the first forward must not pay cache builds in these numbers.
        let cpu = super::super::backend::cpu_device();
        enc.warm(&cpu);

        let bytes =
            std::fs::read("src/parakeet_burn/testdata/encoder_full_golden.safetensors").unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let mel_v = st.tensor("mel_T128").unwrap();
        let mel: Vec<Vec<f32>> = mel_v
            .data()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>()
            .chunks(128)
            .map(|c| c.to_vec())
            .collect();

        let sctx = TensorCtx::scalar();
        let tctx = TensorCtx {
            device: super::super::backend::cpu_device(),
            enabled: true,
        };
        for (tag, ctx) in [("scalar", sctx), ("tensor", tctx)] {
            let pre = enc.pre.forward(&mel, &ctx).unwrap();
            let mut pre_ms = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let _ = enc.pre.forward(&mel, &ctx).unwrap();
                pre_ms = pre_ms.min(ms(t));
            }
            let win = pos_window(&enc.pos_table, pre.len());
            let b = &enc.blocks[0];
            // Stage breakdown on layer 0 (replicates `block_forward` staging).
            let mut stage = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    f();
                    best = best.min(ms(t));
                }
                best
            };
            let ffn1_ms = stage(&|| {
                let n1 = layernorm_batch(&pre, &b.norm_ff1);
                let _ = ffn(&n1, &b.ffn1_l1, &b.ffn1_l2, &ctx);
            });
            let norm_att_ms = stage(&|| {
                let _ = layernorm_batch(&pre, &b.norm_att);
            });
            let attn_ms = stage(&|| {
                // NOTE: attention input here is norm(pre), not norm(pre+f1)
                // as in the real chain — timing only, values discarded.
                let na = layernorm_batch(&pre, &b.norm_att);
                let _ = rel_attention(&na, b, &win, &ctx);
            });
            let conv_ms = stage(&|| {
                let nc = layernorm_batch(&pre, &b.norm_conv);
                let _ = conv_block(&nc, b, &ctx);
            });
            let ffn2_ms = stage(&|| {
                let n2 = layernorm_batch(&pre, &b.norm_ff2);
                let _ = ffn(&n2, &b.ffn2_l1, &b.ffn2_l2, &ctx);
            });
            let t = Instant::now();
            let reps = 5;
            let mut blocks_ms = f64::INFINITY;
            for _ in 0..reps {
                let t1 = Instant::now();
                let mut h = pre.clone();
                for b in &enc.blocks {
                    h = block_forward(&h, b, &win, &ctx);
                }
                blocks_ms = blocks_ms.min(ms(t1));
            }
            let _ = ms(t);
            eprintln!(
                "profile {tag}: T=64 mel -> T'={} | pre={pre_ms:.1}ms blocks24={blocks_ms:.1}ms/layer={:.2}ms | layer0: ffn1={ffn1_ms:.2}ms norm_att={norm_att_ms:.2}ms attn={attn_ms:.2}ms conv={conv_ms:.2}ms ffn2={ffn2_ms:.2}ms",
                pre.len(),
                blocks_ms / 24.0,
            );
        }

        // Linear microbench on real FFN weights (T=8, the golden shape).
        let b = &enc.blocks[0];
        let mut seed: u64 = 0xB00;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let x1024: Vec<Vec<f32>> = (0..8).map(|_| (0..1024).map(|_| rnd()).collect()).collect();
        let x4096: Vec<Vec<f32>> = (0..8).map(|_| (0..4096).map(|_| rnd()).collect()).collect();
        let x1024_32: Vec<Vec<f32>> =
            (0..32).map(|_| (0..1024).map(|_| rnd()).collect()).collect();
        let x4096_32: Vec<Vec<f32>> =
            (0..32).map(|_| (0..4096).map(|_| rnd()).collect()).collect();
        let dev = super::super::backend::cpu_device();
        for (name, lin, xs) in [
            ("ffn1_l1 1024->4096", &b.ffn1_l1, &x1024),
            ("ffn1_l2 4096->1024", &b.ffn1_l2, &x4096),
            ("wout 1024->1024", &b.wout, &x1024),
        ] {
            let t = Instant::now();
            let reps = 20;
            for _ in 0..reps {
                let _ = lin.forward_batch_t(xs, &dev);
            }
            eprintln!(
                "profile linear {name}: {:.3} ms/call (T=8, {reps} reps)",
                ms(t) / reps as f64
            );
        }
        for (name, lin, xs) in [
            ("ffn1_l1 1024->4096 T32", &b.ffn1_l1, &x1024_32),
            ("ffn1_l2 4096->1024 T32", &b.ffn1_l2, &x4096_32),
            ("wout 1024->1024 T32", &b.wout, &x1024_32),
        ] {
            let t = Instant::now();
            let reps = 20;
            for _ in 0..reps {
                let _ = lin.forward_batch_t(xs, &dev);
            }
            eprintln!(
                "profile linear {name}: {:.3} ms/call (20 reps)",
                ms(t) / reps as f64
            );
        }
        // Quantize-only cost at T32 (flat copy + min/max + map).
        {
            let flat: Vec<f32> =
                x1024_32.iter().flat_map(|r| r.iter().copied()).collect();
            let t = Instant::now();
            let reps = 50;
            for _ in 0..reps {
                let _ = super::super::igemm::quantize_rows_i16(&flat);
            }
            eprintln!(
                "profile quantize 32x1024: {:.3} ms/call (50 reps)",
                ms(t) / reps as f64
            );
            let flat4: Vec<f32> =
                x4096_32.iter().flat_map(|r| r.iter().copied()).collect();
            let t = Instant::now();
            for _ in 0..reps {
                let _ = super::super::igemm::quantize_rows_i16(&flat4);
            }
            eprintln!(
                "profile quantize 32x4096: {:.3} ms/call (50 reps)",
                ms(t) / reps as f64
            );
        }

        // Elementwise microbench: swish over FFN-width rows (T=8 x 4096)
        // isolates the libm exp cost inside ffn1/ffn2 timings above.
        {
            use super::super::block::swish_inplace;
            let mut rows = x4096.clone();
            let t = Instant::now();
            let reps = 20;
            for _ in 0..reps {
                for r in rows.iter_mut() {
                    swish_inplace(r);
                }
            }
            eprintln!(
                "profile swish 8x4096 serial: {:.3} ms/call ({reps} reps)",
                ms(t) / reps as f64
            );
            let t = Instant::now();
            for _ in 0..reps {
                use rayon::prelude::*;
                rows.par_iter_mut().for_each(|r| swish_inplace(r));
            }
            eprintln!(
                "profile swish 8x4096 rayon-rows: {:.3} ms/call ({reps} reps)",
                ms(t) / reps as f64
            );
        }

        // Attention Flex-vs-scalar on the same T=8 input.
        {
            let b = &enc.blocks[0];
            let pre = enc.pre.forward(&mel, &sctx).unwrap();
            let win = pos_window(&enc.pos_table, pre.len());
            let natt = layernorm_batch(&pre, &b.norm_att);
            let t = Instant::now();
            for _ in 0..10 {
                let _ = rel_attention(&natt, b, &win, &sctx);
            }
            let s_ms = ms(t) / 10.0;
            let t = Instant::now();
            for _ in 0..10 {
                let _ = rel_attention(&natt, b, &win, &tctx);
            }
            let t_ms = ms(t) / 10.0;
            eprintln!("profile attn T=8/layer: scalar={s_ms:.2}ms tensor={t_ms:.2}ms");
        }

        // Realistic-utterance shapes: random mel T=256 -> T'=32 (what a
        // 2.5 s utterance looks like after the frontend). Values are
        // random — timing has no data-dependent branches — but shapes,
        // quantization windows and attention O(T^2) match production.
        {
            let mut seed: u64 = 0x32;
            let mut rnd = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
            };
            let mel256: Vec<Vec<f32>> =
                (0..256).map(|_| (0..128).map(|_| rnd()).collect()).collect();
            for (tag, ctx) in [("scalar", sctx), ("tensor", tctx)] {
                if tag == "scalar" {
                    continue; // scalar path timing already covered; skip 1 s+ burn
                }
                let mut pre_ms = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    let _ = enc.pre.forward(&mel256, &ctx).unwrap();
                    pre_ms = pre_ms.min(ms(t));
                }
                let pre = enc.pre.forward(&mel256, &ctx).unwrap();
                assert_eq!(pre.len(), 32);
                let win = pos_window(&enc.pos_table, pre.len());
                let b = &enc.blocks[0];
                let mut stage = |f: &dyn Fn()| {
                    let mut best = f64::INFINITY;
                    for _ in 0..5 {
                        let t = Instant::now();
                        f();
                        best = best.min(ms(t));
                    }
                    best
                };
                let attn_ms = stage(&|| {
                    let na = layernorm_batch(&pre, &b.norm_att);
                    let _ = rel_attention(&na, b, &win, &ctx);
                });
                let conv_ms = stage(&|| {
                    let nc = layernorm_batch(&pre, &b.norm_conv);
                    let _ = conv_block(&nc, b, &ctx);
                });
                let ffn_ms = stage(&|| {
                    let n1 = layernorm_batch(&pre, &b.norm_ff1);
                    let _ = ffn(&n1, &b.ffn1_l1, &b.ffn1_l2, &ctx);
                });
                let mut h = pre.clone();
                let mut full_ms = f64::INFINITY;
                for _ in 0..5 {
                    let t1 = Instant::now();
                    h = pre.clone();
                    for b in &enc.blocks {
                        h = block_forward(&h, b, &win, &ctx);
                    }
                    full_ms = full_ms.min(ms(t1));
                }
                eprintln!(
                    "profile {tag} T=256 mel -> T'=32 | pre={pre_ms:.1}ms blocks24={full_ms:.1}ms/layer={:.2}ms | layer0: attn={attn_ms:.2}ms conv={conv_ms:.2}ms ffn={ffn_ms:.2}ms",
                    full_ms / 24.0,
                );
            }
        }

        // Frontend on 2.5 s synthetic audio.
        {
            let fe = super::super::frontend::MelFrontend::tdt_v3();
            let mut seed: u64 = 0xFE;
            let pcm: Vec<f32> = (0..40_000)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.6
                })
                .collect();
            let mut best = f64::INFINITY;
            let reps = 5;
            let mut n_frames = 0;
            for _ in 0..reps {
                let t = Instant::now();
                n_frames = fe.extract(&pcm).unwrap().len();
                best = best.min(ms(t));
            }
            eprintln!(
                "profile frontend 2.5s -> {n_frames} frames: {best:.1} ms/call (min of {reps})",
            );
        }

        // Joint decode on the real T=8 encoder output (e2e tripwire: tokens).
        {
            use super::super::joint::{BurnJoint, joint_want};
            let want = joint_want();
            let jt = load_selected(&dir.join("decoder_joint-burn.safetensors"), &want).unwrap();
            let mut joint = BurnJoint::load(&jt, 8193).unwrap();
            let enc_out = enc.forward(&mel, &tctx).unwrap();
            let t = Instant::now();
            let reps = 5;
            let mut toks = Vec::new();
            for _ in 0..reps {
                let (tk, _, _) = joint.decode(&enc_out, &tctx);
                toks = tk;
            }
            eprintln!(
                "profile joint decode T={}: {:.2} ms/call ({reps} reps) tokens={toks:?}",
                enc_out.len(),
                ms(t) / reps as f64
            );
            // Realistic T=32 decode on random encoder-shaped frames.
            let mut seed: u64 = 0xE70;
            let frame32: Vec<Vec<f32>> = (0..32)
                .map(|_| {
                    (0..1024)
                        .map(|_| {
                            seed = seed
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
                        })
                        .collect()
                })
                .collect();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let _ = joint.decode(&frame32, &tctx);
                best = best.min(ms(t));
            }
            eprintln!("profile joint decode T=32: {best:.2} ms/call (min of 5)");
        }

        // Aligned-vs-split 64B vector loads in the integer micro-kernel:
        // Vec<i16> is only 2B-aligned, so every `_mm512_loadu` splits.
        // If aligned buffers win clearly, the hot path allocates them.
        {
            use super::super::igemm::linear_i16;
            use super::super::igemm::widen_weights;
            let (t, k, n) = (32usize, 1024usize, 1024usize);
            let wu: Vec<u8> = (0..n * k).map(|i| (i * 37 % 256) as u8).collect();
            let bw = widen_weights(&wu, n, k, &[100.0]);
            let mut seed: u64 = 0xA6;
            let a: Vec<i16> = (0..t * k)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed % 511) as i16 - 255
                })
                .collect();
            // 64B-aligned twin of `a` (same values).
            let mut abuf = vec![0i16; a.len() + 32];
            let off = abuf.as_ptr() as usize % 64;
            let shift = if off == 0 { 0 } else { (64 - off) / 2 };
            abuf[shift..shift + a.len()].copy_from_slice(&a);
            let aa = &abuf[shift..shift + a.len()];
            assert_eq!(aa.as_ptr() as usize % 64, 0);
            // Forced-split twin: 2B offset (rows stay equally shifted —
            // every real K is a multiple of 32, so the stride keeps all
            // rows 2-mod-64 together).
            let mut sbuf = vec![0i16; a.len() + 32];
            let base = sbuf.as_ptr() as usize % 64;
            let soff = (((2 + 64 - base) % 64) / 2).max(1);
            sbuf[soff..soff + a.len()].copy_from_slice(&a);
            let sp = &sbuf[soff..soff + a.len()];
            assert_ne!(sp.as_ptr() as usize % 64, 0);
            assert_eq!(sp, &a[..]);
            let sw = vec![0.02f32; n];
            let mut best_split = f64::INFINITY;
            let mut best_al = f64::INFINITY;
            for _ in 0..10 {
                let mut out = vec![0.0f32; t * n];
                let t0 = Instant::now();
                linear_i16(sp, t, k, &bw, n, &sw, None, 0.05, &mut out);
                best_split = best_split.min(ms(t0));
                let t0 = Instant::now();
                linear_i16(aa, t, k, &bw, n, &sw, None, 0.05, &mut out);
                best_al = best_al.min(ms(t0));
            }
            eprintln!("profile gemm T32/K1024/N1024: split={best_split:.3}ms aligned={best_al:.3}ms");
        }
    }
}
