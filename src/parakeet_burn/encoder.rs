// FastConformer encoder stack (Burn port, bring-up).
//
// ONNX reference (`encoder-model.int8.onnx`): 5654 nodes, 8x subsampling,
// encoder output `[1, 1024, T/8]`, inputs `audio_signal [1, 128, T]` +
// `length`. Quantized ops: `DynamicQuantizeLinear` x223,
// `MatMulInteger` x217, `ConvInteger` x77, `LayerNormalization` x120,
// `Softmax` x24.
//
// Bring-up order (each step lands with a parity test against ORT):
//  1. subsampling conv block (2 x Conv1d stride 2 + norms)
//  2. one Conformer block (FFN + MHSA + conv + norms, int8 Linear)
//  3. full depth + length masking + 8x output projection to 1024
//
// Until step 3 completes, `forward` is unreachable: `load` fails fast
// (see `weights.rs`) so no partial encoder can mistranscribe.

use super::quant::Int8Linear;

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

/// Placeholder encoder holding int8 linears once converted.
pub(crate) struct BurnEncoder {
    #[allow(dead_code)]
    config: EncoderConfig,
    #[allow(dead_code)]
    linears: Vec<Int8Linear>,
}

impl BurnEncoder {
    pub(crate) fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            linears: Vec::new(),
        }
    }
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
}

impl EncoderManifest {
    pub(crate) fn role(&self, role: &str) -> Option<&RoleEntry> {
        self.roles.get(role)
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
        roles.insert(
            role.to_string(),
            RoleEntry {
                role: role.to_string(),
                kind: kind.to_string(),
                weight: weight.to_string(),
                weight_shape: dims,
            },
        );
    }
    let manifest = EncoderManifest { roles };
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
}
