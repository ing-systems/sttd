// Converted-weight gate: the Burn engine only runs on weights produced
// by `tools/convert-parakeet-int8-to-burn.py` from the int8 ONNX files.
// Same `models/parakeet-tdt-0.6b-v3-int8/` dir, new files:
// `encoder-burn.safetensors` + `decoder_joint-burn.safetensors`.
//
// Rationale: `burn-import` cannot ingest the quantized graph
// (`DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`,
// `DynamicQuantizeLSTM` have no Burn equivalent), so the converter
// extracts int8 weights + per-row scales into safetensors and the encoder
// (`encoder.rs`) / joint (`joint.rs`) consume those directly. Until the
// files exist, `load` fails fast with the exact command to run.

use anyhow::{Context, Result};
use std::path::Path;

pub(crate) const ENCODER_BURN_FILE: &str = "encoder-burn.safetensors";
pub(crate) const JOINT_BURN_FILE: &str = "decoder_joint-burn.safetensors";

/// `(name, shape, f32-data)` rows from a converted weights file.
pub(crate) type WeightRows = Vec<(String, Vec<usize>, Vec<f32>)>;

/// Fail fast unless converted Burn weights exist next to the ONNX files.
pub(crate) fn require_converted_weights(dir: &Path) -> Result<()> {
    let enc = dir.join(ENCODER_BURN_FILE);
    let joint = dir.join(JOINT_BURN_FILE);
    let mut missing = Vec::new();
    if !enc.is_file() {
        missing.push(ENCODER_BURN_FILE);
    }
    if !joint.is_file() {
        missing.push(JOINT_BURN_FILE);
    }
    if missing.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "burn parakeet weights missing in {} ({}): convert the int8 ONNX first:\n\
         \n  python3 tools/convert-parakeet-int8-to-burn.py \\\n    --model-dir {}\n\
         \nthen retry with `--parakeet-engine burn` (or use `--parakeet-engine ort` today)",
        dir.display(),
        missing.join(", "),
        dir.display()
    )
}

/// Load one safetensors file into `(name, shape, f32-data)` rows,
///
/// dequantizing 8-bit weights and passing through `F32`/`F16`/`BF16`.
/// Written by `tools/convert-parakeet-int8-to-burn.py` from the int8 ONNX:
/// quantized weights are stored as `<base>` (U8 asymmetric + `.scale`
/// F32 + `.zero_point` U8, or I8 + `.scale` + optional `.zero_point`)
/// next to plain F32 biases. Used by the encoder/joint loaders as weight
/// mapping lands.
#[allow(dead_code)]
pub(crate) fn load_safetensors_f32(path: &Path) -> Result<WeightRows> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut out = Vec::new();
    for name in st.names() {
        if name.ends_with(".scale") || name.ends_with(".zero_point") {
            continue; // consumed alongside their weight below
        }
        let v = st
            .tensor(name)
            .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
        let shape: Vec<usize> = v.shape().to_vec();
        match v.dtype() {
            safetensors::Dtype::F32 => {
                let raw = v.data();
                let (chunks, _) = raw.as_chunks::<4>();
                let mut data = Vec::with_capacity(chunks.len());
                for c in chunks {
                    data.push(f32::from_le_bytes(*c));
                }
                out.push((name.to_string(), shape, data));
            }
            safetensors::Dtype::F16 => {
                let raw = v.data();
                let (chunks, _) = raw.as_chunks::<2>();
                let mut data = Vec::with_capacity(chunks.len());
                for c in chunks {
                    data.push(half_f16_to_f32(u16::from_le_bytes(*c)));
                }
                out.push((name.to_string(), shape, data));
            }
            safetensors::Dtype::BF16 => {
                let raw = v.data();
                let (chunks, _) = raw.as_chunks::<2>();
                let mut data = Vec::with_capacity(chunks.len());
                for c in chunks {
                    let bits = (u16::from_le_bytes(*c) as u32) << 16;
                    data.push(f32::from_bits(bits));
                }
                out.push((name.to_string(), shape, data));
            }
            safetensors::Dtype::U8 => {
                // Asymmetric static quantization (encoder MatMuls/Convs,
                // decoder embed + joint): `w = (u8 - zp) * scale`.
                let (scales, zps) = aux_pair(&st, path, name)?;
                let rows = shape.first().copied().unwrap_or(1).max(1);
                let cols = v.data().len() / rows;
                let mut data = Vec::with_capacity(v.data().len());
                for r in 0..rows {
                    let sc = row_value(&scales, r);
                    let zp = row_value(&zps, r);
                    for c in 0..cols {
                        data.push((v.data()[r * cols + c] as f32 - zp) * sc);
                    }
                }
                out.push((name.to_string(), shape, data));
            }
            safetensors::Dtype::I8 => {
                // Symmetric-ish (decoder LSTMs): `w = (i8 - zp) * scale`.
                let (scales, zps) = aux_pair(&st, path, name)?;
                let rows = shape.first().copied().unwrap_or(1).max(1);
                let cols = v.data().len() / rows;
                let mut data = Vec::with_capacity(v.data().len());
                for r in 0..rows {
                    let sc = row_value(&scales, r);
                    let zp = row_value(&zps, r);
                    for c in 0..cols {
                        data.push((v.data()[r * cols + c] as i8 as f32 - zp) * sc);
                    }
                }
                out.push((name.to_string(), shape, data));
            }
            other => anyhow::bail!(
                "unsupported dtype {other:?} for tensor {name} in {}",
                path.display()
            ),
        }
    }
    Ok(out)
}

/// Read `<name>.scale` (F32, scalar or per-row) and `<name>.zero_point`
/// for an 8-bit weight. Zero-point is required for U8 (stored U8) and
/// optional for I8 (stored I8, defaults to 0). Returns `(scales, zps)` as
/// f32 vectors, each scalar-or-per-row (broadcast by the caller).
fn aux_pair(
    st: &safetensors::SafeTensors<'_>,
    path: &Path,
    name: &str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let weight = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
    let scale_key = format!("{name}.scale");
    let zp_key = format!("{name}.zero_point");
    let st32 = st.tensor(&scale_key).with_context(|| {
        format!(
            "8-bit tensor {name} needs {scale_key} in {}",
            path.display()
        )
    })?;
    let mut scales = Vec::with_capacity(st32.data().len() / 4);
    let (chunks, _) = st32.data().as_chunks::<4>();
    for c in chunks {
        scales.push(f32::from_le_bytes(*c));
    }
    let zps = match weight.dtype() {
        safetensors::Dtype::U8 => {
            let t = st.tensor(&zp_key).with_context(|| {
                format!(
                    "uint8 tensor {name} needs {zp_key} in {}",
                    path.display()
                )
            })?;
            t.data().iter().map(|&b| b as f32).collect()
        }
        safetensors::Dtype::I8 => match st.tensor(&zp_key) {
            Ok(t) => t.data().iter().map(|&b| b as i8 as f32).collect(),
            Err(_) => vec![0.0],
        },
        other => anyhow::bail!("{name} has unexpected dtype {other:?} in {}", path.display()),
    };
    Ok((scales, zps))
}

/// Pick the scalar-or-per-row entry for row `r`.
fn row_value(v: &[f32], r: usize) -> f32 {
    v[r.min(v.len().saturating_sub(1))]
}

fn half_f16_to_f32(bits: u16) -> f32 {
    // Minimal IEEE-754 binary16 -> binary32 (handles normals; subnormals
    // flush toward zero — fine for weight inspection, not training).
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    let fbits = if exp == 0 {
        (sign << 31) | (mant << 13)
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(fbits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_weights_error_names_converter() {
        let tmp = std::env::temp_dir().join(format!(
            "sttd-burn-weights-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let err = require_converted_weights(&tmp).expect_err("must fail without files");
        let msg = err.to_string();
        assert!(msg.contains("convert-parakeet-int8-to-burn"), "msg={msg}");
        assert!(msg.contains("encoder-burn.safetensors"), "msg={msg}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn present_weights_pass_gate() {
        let tmp = std::env::temp_dir().join(format!(
            "sttd-burn-weights-ok-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join(ENCODER_BURN_FILE), b"x").unwrap();
        std::fs::write(tmp.join(JOINT_BURN_FILE), b"x").unwrap();
        require_converted_weights(&tmp).expect("must pass with files");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Synthetic U8 (+scale/zp) + F32 roundtrip through the loader, the
    /// same layout `convert-parakeet-int8-to-burn.py` writes.
    #[test]
    fn safetensors_u8_dequantizes_with_zero_point() {
        use safetensors::tensor::{Dtype, TensorView};
        // weight [[121+10, 121-5]] u8, zp=121, scale=0.5 -> [[5.0, -2.5]]
        let w = vec![131u8, 116];
        let scale = 0.5f32.to_le_bytes().to_vec();
        let zp = vec![121u8];
        let bias = 1.0f32.to_le_bytes().to_vec();
        let bytes = safetensors::tensor::serialize(
            vec![
                ("w".to_string(), TensorView::new(Dtype::U8, vec![1, 2], &w).unwrap()),
                (
                    "w.scale".to_string(),
                    TensorView::new(Dtype::F32, vec![1], &scale).unwrap(),
                ),
                (
                    "w.zero_point".to_string(),
                    TensorView::new(Dtype::U8, vec![1], &zp).unwrap(),
                ),
                ("b".to_string(), TensorView::new(Dtype::F32, vec![1], &bias).unwrap()),
            ],
            &None,
        )
        .expect("serialize");
        let tmp = std::env::temp_dir().join(format!(
            "sttd-burn-safetensors-test-{}.st",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&tmp, bytes).unwrap();
        let rows = load_safetensors_f32(&tmp).expect("load");
        std::fs::remove_file(&tmp).ok();
        let wrow = rows.iter().find(|(n, _, _)| n == "w").expect("w");
        assert_eq!(wrow.1, vec![1, 2]);
        assert!((wrow.2[0] - 5.0).abs() < 1e-6, "row={wrow:?}");
        assert!((wrow.2[1] + 2.5).abs() < 1e-6, "row={wrow:?}");
        let brow = rows.iter().find(|(n, _, _)| n == "b").expect("b");
        assert!((brow.2[0] - 1.0).abs() < 1e-6, "row={brow:?}");
    }
}
