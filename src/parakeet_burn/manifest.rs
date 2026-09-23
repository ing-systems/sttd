// Validated encoder manifest: role -> weight mapping for the converted int8 files.
//
// Extracted from `encoder.rs` to break the `encoder <-> block` module cycle:
// `BurnEncoder` needs `conformer_forward`, while `BlockWeights::load` and
// `PreEncodeWeights::load` need the manifest types. All three now depend
// down on this leaf module; nothing here depends on `encoder`, `block`,
// or `pre_encode`.
//
// ONNX reference (`encoder-model.int8.onnx`, see
// `tools/convert-parakeet-int8-to-burn.py::emit_encoder_manifest`).

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
