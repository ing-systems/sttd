// One FastConformer block, matching the int8 ONNX graph op-for-op in fp32.
//
// Graph facts read off `encoder-model.int8.onnx` (see module docs in
// `encoder.rs`; verified with the probe scripts, not assumed):
// - Pre-norm sandwich with half-step FFNs:
//     x += 0.5 * FFN1(LN_ff1(x))      (`/layers.N/Mul` constant is 0.5)
//     x += MHSA(LN_att(x), pos)       (relative XL-style, see below)
//     x += Conv(LN_conv(x))           (GLU + depthwise k=9 + Swish)
//     x += 0.5 * FFN2(LN_ff2(x))      (`Mul_1` constant is 0.5)
//     y = LN_out(x)
// - All 120 LayerNormalization epsilons are uniformly 1e-5 (biased
//   variance, i.e. divide by N — ONNX semantics).
// - Quantized linears carry NO bias (only the depthwise conv, norms and
//   pos_bias do); FFN activation is Swish (`activation/Mul` =
//   linear * sigmoid(linear)); conv gate is GLU + Swish after depthwise.
// - Depthwise pads are the constant [4, 4] (SAME); the data-side Where is
//   a lengths mask, a no-op for full-length offline utterances.
// - Relative attention: frozen table (9999, 1024), window
//   table[5000-T .. 5000+T-1], distance-0 at window index T-1;
//   B[i,j] = (q_i + v) . P[(i-j) + (T-1)], scores / sqrt(128).

use std::collections::HashMap;

use anyhow::{Context, Result};

use super::encoder::EncoderManifest;
use super::quant::Uint8Linear;
use super::weights::RawTensor;

/// LayerNorm params (gamma/beta, eps fixed 1e-5 per graph).
#[derive(Debug, Clone)]
pub(crate) struct LayerNormP {
    pub(crate) gamma: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

pub(crate) const LN_EPS: f32 = 1e-5;

pub(crate) fn layernorm_row(x: &[f32], p: &LayerNormP) -> Vec<f32> {
    assert_eq!(x.len(), p.gamma.len());
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * inv * p.gamma[i] + p.beta[i])
        .collect()
}

pub(crate) fn layernorm_batch(x: &[Vec<f32>], p: &LayerNormP) -> Vec<Vec<f32>> {
    x.iter().map(|r| layernorm_row(r, p)).collect()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn swish_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x * sigmoid(*x);
    }
}

pub(crate) fn softmax_rows(scores: &mut [Vec<f32>]) {
    for row in scores.iter_mut() {
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut s = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - m).exp();
            s += *x;
        }
        let inv = 1.0 / s;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

/// Depthwise conv (groups == channels): `w[c][k]`, symmetric padding.
#[derive(Debug, Clone)]
pub(crate) struct DwConv {
    pub(crate) weight_u8: Vec<u8>,
    pub(crate) scale: f32,
    pub(crate) zero_point: f32,
    pub(crate) bias: Vec<f32>,
    pub(crate) channels: usize,
    pub(crate) kernel: usize,
    pub(crate) pad_left: usize,
    pub(crate) pad_right: usize,
}

impl DwConv {
    pub(crate) fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let c = self.channels;
        assert!(x.iter().all(|r| r.len() == c));
        // dequantize once per call (weights are small: 1024x9)
        let w: Vec<f32> = self
            .weight_u8
            .iter()
            .map(|&q| (q as f32 - self.zero_point) * self.scale)
            .collect();
        let mut padded = vec![vec![0.0f32; c]; self.pad_left + t + self.pad_right];
        for (i, row) in x.iter().enumerate() {
            padded[self.pad_left + i].copy_from_slice(row);
        }
        let out_len = padded.len() - self.kernel + 1;
        let mut out = vec![vec![0.0f32; c]; out_len];
        for (t_out, row) in out.iter_mut().enumerate() {
            for ch in 0..c {
                let mut acc = self.bias[ch];
                let base = ch * self.kernel;
                for k in 0..self.kernel {
                    acc += w[base + k] * padded[t_out + k][ch];
                }
                row[ch] = acc;
            }
        }
        out
    }
}

/// GLU gate: split 2d in halves, `a * sigmoid(b)`.
pub(crate) fn glu(v: &[f32]) -> Vec<f32> {
    assert!(v.len() % 2 == 0);
    let (a, b) = v.split_at(v.len() / 2);
    a.iter().zip(b.iter()).map(|(&x, &g)| x * sigmoid(g)).collect()
}

/// Full per-layer weights (dequantized lazily inside the linears).
pub(crate) struct BlockWeights {
    pub(crate) n_heads: usize,
    pub(crate) d_head: usize,
    pub(crate) norm_ff1: LayerNormP,
    pub(crate) norm_att: LayerNormP,
    pub(crate) norm_conv: LayerNormP,
    pub(crate) norm_ff2: LayerNormP,
    pub(crate) norm_out: LayerNormP,
    pub(crate) ffn1_l1: Uint8Linear,
    pub(crate) ffn1_l2: Uint8Linear,
    pub(crate) ffn2_l1: Uint8Linear,
    pub(crate) ffn2_l2: Uint8Linear,
    pub(crate) wq: Uint8Linear,
    pub(crate) wk: Uint8Linear,
    pub(crate) wv: Uint8Linear,
    pub(crate) wpos: Uint8Linear,
    pub(crate) wout: Uint8Linear,
    pub(crate) pos_bias_u: Vec<Vec<f32>>, // [heads][d_head]
    pub(crate) pos_bias_v: Vec<Vec<f32>>,
    pub(crate) pw1: Uint8Linear, // 1024 -> 2048
    pub(crate) pw2: Uint8Linear, // 1024 -> 1024
    pub(crate) dw: DwConv,
}

fn take_f32(tensors: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    match tensors.get(name) {
        Some(RawTensor::F32 { shape, data }) => Ok((shape.clone(), data.clone())),
        Some(other) => anyhow::bail!("{name}: expected F32, got {}", other.kind()),
        None => anyhow::bail!("{name}: missing tensor"),
    }
}

fn take_norm(tensors: &HashMap<String, RawTensor>, base: &str) -> Result<LayerNormP> {
    let (gs, gamma) = take_f32(tensors, &format!("{base}.weight"))?;
    let (bs, beta) = take_f32(tensors, &format!("{base}.bias"))?;
    if gs.len() != 1 || bs.len() != 1 || gs[0] != bs[0] {
        anyhow::bail!("{base}: norm shape mismatch {gs:?} vs {bs:?}");
    }
    Ok(LayerNormP { gamma, beta })
}

fn take_linear(
    tensors: &HashMap<String, RawTensor>,
    manifest: &EncoderManifest,
    role: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Uint8Linear> {
    let entry = manifest
        .role(role)
        .with_context(|| format!("manifest lacks {role}"))?;
    let raw = tensors
        .get(&entry.weight)
        .with_context(|| format!("{}: missing weight {}", role, entry.weight))?;
    let (shape, data, scale, zp) = match raw {
        RawTensor::U8 {
            shape,
            data,
            scale,
            zp,
        } => (shape, data, scale, zp),
        other => anyhow::bail!("{}: expected U8, got {}", role, other.kind()),
    };
    // ONNX MatMul weight (K, N) = (in, out) is transposed to the
    // `[out][in]` store `forward` reads; 1x1 convs (out, in, 1) already
    // match and pass through.
    let (k, n, store) = match shape.as_slice() {
        [k, n] => (*k, *n, super::quant::transpose_u8(data, *k, *n)),
        [n, k, 1] => (*k, *n, data.clone()),
        _ => anyhow::bail!("{role}: unexpected weight shape {shape:?}"),
    };
    if k != in_dim || n != out_dim {
        anyhow::bail!("{role}: shape {shape:?} != ({in_dim}, {out_dim})");
    }
    let bcast = |v: &[f32]| {
        if v.len() == 1 {
            vec![v[0]; n]
        } else {
            v.to_vec()
        }
    };
    Ok(Uint8Linear {
        weight_u8: store,
        scale: bcast(scale),
        zero_point: bcast(zp),
        bias: None, // verified: quantized linears carry no bias
        in_dim,
        out_dim,
    })
}

impl BlockWeights {
    /// Load one layer's weights. `tensors` holds the converted safetensors
    /// entries (see `weights::load_selected`); shapes are asserted here so
    /// a corrupt/shuffled file fails at load, never mid-utterance.
    pub(crate) fn load(
        tensors: &HashMap<String, RawTensor>,
        manifest: &EncoderManifest,
        layer: usize,
        d_model: usize,
        n_heads: usize,
        d_head: usize,
        d_ff: usize,
    ) -> Result<Self> {
        let r = |s: &str| format!("layers.{layer}/{s}");
        let norm = |s: &str| take_norm(tensors, &format!("layers.{layer}.{s}"));
        let lin = |s: &str, i: usize, o: usize| take_linear(tensors, manifest, &r(s), i, o);
        let (_, pu) = take_f32(tensors, &format!("layers.{layer}.self_attn.pos_bias_u"))?;
        let (_, pv) = take_f32(tensors, &format!("layers.{layer}.self_attn.pos_bias_v"))?;
        if pu.len() != n_heads * d_head || pv.len() != n_heads * d_head {
            anyhow::bail!("layer {layer}: pos_bias shape mismatch");
        }
        let pos_bias_u = pu.chunks(d_head).map(|c| c.to_vec()).collect();
        let pos_bias_v = pv.chunks(d_head).map(|c| c.to_vec()).collect();

        // Depthwise conv: weight (C, 1, K) + scalar-or-row scale/zp + bias.
        let dw_entry = manifest
            .role(&r("conv/depthwise_conv"))
            .with_context(|| format!("manifest lacks layers.{layer}/conv/depthwise_conv"))?;
        let (dw_shape, dw_u8, dw_sc, dw_zp) = match tensors.get(&dw_entry.weight) {
            Some(RawTensor::U8 {
                shape,
                data,
                scale,
                zp,
            }) => (shape, data, scale, zp),
            other => anyhow::bail!("dw conv: expected U8, got {:?}", other.map(|t| t.kind())),
        };
        if dw_shape.len() != 3 || dw_shape[0] != d_model || dw_shape[1] != 1 {
            anyhow::bail!("dw conv shape {dw_shape:?}, expected [{d_model}, 1, K]");
        }
        let kernel = dw_shape[2];
        let dw_bias_name = format!("layers.{layer}/conv/depthwise_bias");
        let dw_bias_w = manifest
            .role(&dw_bias_name)
            .map(|e| e.weight.clone())
            .unwrap_or_else(|| format!("layers.{layer}/conv/depthwise_bias"));
        // dw bias recorded in manifest's dw_bias map (initializer name).
        let dw_bias_init = manifest
            .dw_bias_init(layer)
            .unwrap_or(dw_bias_w);
        let (_, dw_bias) = take_f32(tensors, &dw_bias_init)?;
        if dw_bias.len() != d_model {
            anyhow::bail!("dw bias len {}, expected {d_model}", dw_bias.len());
        }

        Ok(Self {
            n_heads,
            d_head,
            norm_ff1: norm("norm_feed_forward1")?,
            norm_att: norm("norm_self_att")?,
            norm_conv: norm("norm_conv")?,
            norm_ff2: norm("norm_feed_forward2")?,
            norm_out: norm("norm_out")?,
            ffn1_l1: lin("feed_forward1/linear1", d_model, d_ff)?,
            ffn1_l2: lin("feed_forward1/linear2", d_ff, d_model)?,
            ffn2_l1: lin("feed_forward2/linear1", d_model, d_ff)?,
            ffn2_l2: lin("feed_forward2/linear2", d_ff, d_model)?,
            wq: lin("self_attn/linear_q", d_model, d_model)?,
            wk: lin("self_attn/linear_k", d_model, d_model)?,
            wv: lin("self_attn/linear_v", d_model, d_model)?,
            wpos: lin("self_attn/linear_pos", d_model, d_model)?,
            wout: lin("self_attn/linear_out", d_model, d_model)?,
            pos_bias_u,
            pos_bias_v,
            pw1: lin("conv/pointwise_conv1", d_model, 2 * d_model)?,
            pw2: lin("conv/pointwise_conv2", d_model, d_model)?,
            dw: DwConv {
                weight_u8: dw_u8.clone(),
                scale: dw_sc[0],
                zero_point: dw_zp[0],
                bias: dw_bias,
                channels: d_model,
                kernel,
                pad_left: kernel / 2,
                pad_right: kernel / 2,
            },
        })
    }
}

/// Slice the relative window for length T from the full table:
/// rows [5000-T .. 5000+T-1] (length 2T-1, distance 0 at index T-1).
/// Table layout (verified): `table[i]` = PE(4999-i), interleaved sin/cos.
pub(crate) fn pos_window(table: &[Vec<f32>], t: usize) -> Vec<Vec<f32>> {
    assert!(table.len() >= 9999);
    assert!(t >= 1 && 5000 - t + (2 * t - 1) <= table.len());
    table[5000 - t..5000 + t - 1].to_vec()
}

fn linear_batch(lin: &Uint8Linear, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter().map(|r| lin.forward(r)).collect()
}

fn add_inplace(a: &mut [Vec<f32>], b: &[Vec<f32>]) {
    for (ra, rb) in a.iter_mut().zip(b.iter()) {
        for (x, y) in ra.iter_mut().zip(rb.iter()) {
            *x += *y;
        }
    }
}

fn scale_inplace(a: &mut [Vec<f32>], s: f32) {
    for r in a.iter_mut() {
        for x in r.iter_mut() {
            *x *= s;
        }
    }
}

fn split_heads(x: &[Vec<f32>], n_heads: usize, d_head: usize) -> Vec<Vec<Vec<f32>>> {
    x.iter()
        .map(|row| {
            (0..n_heads)
                .map(|h| row[h * d_head..(h + 1) * d_head].to_vec())
                .collect()
        })
        .collect()
}

/// Relative multi-head attention (Transformer-XL shift):
/// B[i,j] = (q_i + v) . P[(i-j) + (T-1)], scores / sqrt(dh).
pub(crate) fn rel_attention(
    x: &[Vec<f32>],
    w: &BlockWeights,
    pos_window: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let t = x.len();
    let (h, dh) = (w.n_heads, w.d_head);
    let q = split_heads(&linear_batch(&w.wq, x), h, dh);
    let k = split_heads(&linear_batch(&w.wk, x), h, dh);
    let v = split_heads(&linear_batch(&w.wv, x), h, dh);
    let p = split_heads(&linear_batch(&w.wpos, pos_window), h, dh);
    assert_eq!(p.len(), 2 * t - 1);
    let scale = 1.0 / (dh as f32).sqrt();
    let mut heads_out = Vec::with_capacity(h);
    for hh in 0..h {
        let mut scores = vec![vec![0.0f32; t]; t];
        // BD[k] for all window positions once per query row.
        let mut bd = vec![vec![0.0f32; 2 * t - 1]; t];
        for i in 0..t {
            let qv: Vec<f32> = q[i][hh]
                .iter()
                .zip(w.pos_bias_v[hh].iter())
                .map(|(a, b)| a + b)
                .collect();
            for (kk, pk) in p.iter().enumerate() {
                bd[i][kk] = qv.iter().zip(pk[hh].iter()).map(|(a, b)| a * b).sum();
            }
        }
        for i in 0..t {
            let qu: Vec<f32> = q[i][hh]
                .iter()
                .zip(w.pos_bias_u[hh].iter())
                .map(|(a, b)| a + b)
                .collect();
            for j in 0..t {
                let ac: f32 = qu.iter().zip(k[j][hh].iter()).map(|(a, b)| a * b).sum();
                let kk = (i as isize - j as isize + t as isize - 1) as usize;
                scores[i][j] = (ac + bd[i][kk]) * scale;
            }
        }
        softmax_rows(&mut scores);
        let mut o = vec![vec![0.0f32; dh]; t];
        for i in 0..t {
            for j in 0..t {
                let a = scores[i][j];
                for d in 0..dh {
                    o[i][d] += a * v[j][hh][d];
                }
            }
        }
        heads_out.push(o);
    }
    // merge heads + out proj
    let merged: Vec<Vec<f32>> = (0..t)
        .map(|i| {
            (0..h)
                .flat_map(|hh| heads_out[hh][i].clone())
                .collect()
        })
        .collect();
    linear_batch(&w.wout, &merged)
}

pub(crate) fn ffn(x: &[Vec<f32>], l1: &Uint8Linear, l2: &Uint8Linear) -> Vec<Vec<f32>> {
    let mut h = linear_batch(l1, x);
    for row in h.iter_mut() {
        swish_inplace(row);
    }
    linear_batch(l2, &h)
}

pub(crate) fn conv_block(x: &[Vec<f32>], w: &BlockWeights) -> Vec<Vec<f32>> {
    let gated: Vec<Vec<f32>> = linear_batch(&w.pw1, x).iter().map(|r| glu(r)).collect();
    let mut dwed = w.dw.forward(&gated);
    // SAME padding (k=9, pads 4/4) preserves length; trim defensively.
    dwed.truncate(x.len());
    while dwed.len() < x.len() {
        dwed.push(vec![0.0; dwed[0].len()]);
    }
    for row in dwed.iter_mut() {
        swish_inplace(row);
    }
    linear_batch(&w.pw2, &dwed)
}

/// One full block: half-step FFN sandwich + final norm.
pub(crate) fn block_forward(
    x: &[Vec<f32>],
    w: &BlockWeights,
    pos_window: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut h = x.to_vec();
    let mut f = ffn(&layernorm_batch(&h, &w.norm_ff1), &w.ffn1_l1, &w.ffn1_l2);
    scale_inplace(&mut f, 0.5);
    add_inplace(&mut h, &f);
    let a = rel_attention(&layernorm_batch(&h, &w.norm_att), w, pos_window);
    add_inplace(&mut h, &a);
    let c = conv_block(&layernorm_batch(&h, &w.norm_conv), w);
    add_inplace(&mut h, &c);
    let mut f2 = ffn(&layernorm_batch(&h, &w.norm_ff2), &w.ffn2_l1, &w.ffn2_l2);
    scale_inplace(&mut f2, 0.5);
    add_inplace(&mut h, &f2);
    layernorm_batch(&h, &w.norm_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet_burn::quant::Uint8Linear;

    fn tiny_linear(w: Vec<u8>, zp: f32, s: f32, in_dim: usize, out_dim: usize) -> Uint8Linear {
        // hand-quantized stand-in: exact when values fit (w-zp)*s
        Uint8Linear {
            weight_u8: w,
            scale: vec![s; out_dim],
            zero_point: vec![zp; out_dim],
            bias: None,
            in_dim,
            out_dim,
        }
    }

    #[test]
    fn layernorm_matches_biased_definition() {
        let p = LayerNormP {
            gamma: vec![1.0, 1.0, 1.0],
            beta: vec![0.0, 0.0, 0.0],
        };
        let y = layernorm_row(&[1.0, 2.0, 3.0], &p);
        // mean 2, biased var (1+0+1)/3 = 2/3
        let inv = 1.0 / ((2.0f32 / 3.0) + 1e-5).sqrt();
        assert!((y[0] + inv).abs() < 1e-5, "y={y:?}");
        assert!(y[1].abs() < 1e-5, "y={y:?}");
        assert!((y[2] - inv).abs() < 1e-5, "y={y:?}");
    }

    #[test]
    fn swish_and_glu_gate() {
        let mut v = vec![0.0f32, 1.0];
        swish_inplace(&mut v);
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 0.7310586).abs() < 1e-6, "v={v:?}");
        // GLU([2, 0]) = 2 * sigmoid(0) = 1
        assert_eq!(glu(&[2.0, 0.0]), vec![1.0]);
    }

    #[test]
    fn dw_identity_kernel_preserves_interior() {
        // k=3, center tap zp+1 with side taps at zp (dequant 0):
        // out == in everywhere (zero pad contributes nothing).
        let dw = DwConv {
            weight_u8: vec![128, 129, 128],
            scale: 1.0,
            zero_point: 128.0,
            bias: vec![0.0],
            channels: 1,
            kernel: 3,
            pad_left: 1,
            pad_right: 1,
        };
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = dw.forward(&x);
        assert_eq!(y.len(), 4);
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-5, "y={y:?}");
        }
    }

    #[test]
    fn pos_window_slices_centered_distances() {
        // synthetic table[i] = i: window for T=4 is rows 4996..5002.
        let table: Vec<Vec<f32>> = (0..9999).map(|i| vec![i as f32]).collect();
        let w = pos_window(&table, 4);
        assert_eq!(w.len(), 7);
        assert_eq!(w[0][0], 4996.0);
        assert_eq!(w[3][0], 4999.0); // distance 0 at index T-1
        assert_eq!(w[6][0], 5002.0);
    }

    #[test]
    fn attention_relative_index_uses_distance() {
        // 1 head, dh=2, d=2, T=2: identity projections, zero bias.
        // pos window rows are distinct markers: P[k] = [10+k, 0].
        // q=k=v=x=[[1,0],[0,1]].
        let d = 2;
        let ident = tiny_linear(vec![129, 128, 128, 129], 128.0, 1.0 / 127.0, d, d);
        // dequant check: row0 = [(129-128)/127, 0] = [1/127, 0] — NOT identity!
        // Build exact identity instead via scale trick below.
        let _ = ident;
        // exact identity: zp=0, s=1, w=[[1,0],[0,1]] needs u8 0/1 — fine.
        let eye = Uint8Linear {
            weight_u8: vec![1, 0, 0, 1],
            scale: vec![1.0, 1.0],
            zero_point: vec![0.0, 0.0],
            bias: None,
            in_dim: 2,
            out_dim: 2,
        };
        let w = BlockWeights {
            n_heads: 1,
            d_head: 2,
            norm_ff1: LayerNormP { gamma: vec![1.0, 1.0], beta: vec![0.0, 0.0] },
            norm_att: LayerNormP { gamma: vec![1.0, 1.0], beta: vec![0.0, 0.0] },
            norm_conv: LayerNormP { gamma: vec![1.0, 1.0], beta: vec![0.0, 0.0] },
            norm_ff2: LayerNormP { gamma: vec![1.0, 1.0], beta: vec![0.0, 0.0] },
            norm_out: LayerNormP { gamma: vec![1.0, 1.0], beta: vec![0.0, 0.0] },
            ffn1_l1: eye.clone(),
            ffn1_l2: eye.clone(),
            ffn2_l1: eye.clone(),
            ffn2_l2: eye.clone(),
            wq: eye.clone(),
            wk: eye.clone(),
            wv: eye.clone(),
            wpos: eye.clone(),
            wout: eye.clone(),
            pos_bias_u: vec![vec![0.0, 0.0]],
            pos_bias_v: vec![vec![0.0, 0.0]],
            pw1: Uint8Linear {
                weight_u8: vec![1, 0, 0, 1, 0, 0, 0, 0],
                scale: vec![1.0; 4],
                zero_point: vec![0.0; 4],
                bias: None,
                in_dim: 2,
                out_dim: 4,
            },
            pw2: Uint8Linear {
                weight_u8: vec![1, 0, 0, 1],
                scale: vec![1.0, 1.0],
                zero_point: vec![0.0, 0.0],
                bias: None,
                in_dim: 2,
                out_dim: 2,
            },
            dw: DwConv {
                weight_u8: vec![0, 1, 0, 0, 1, 0],
                scale: 1.0,
                zero_point: 0.0,
                bias: vec![0.0, 0.0],
                channels: 2,
                kernel: 3,
                pad_left: 1,
                pad_right: 1,
            },
        };
        // layernorm first: use pre-normalized inputs to keep them exact.
        // x rows already zero-mean/unit-var-ish; attention on raw rows:
        let x = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let pw = vec![vec![10.0, 0.0], vec![20.0, 0.0], vec![30.0, 0.0]];
        let y = rel_attention(&x, &w, &pw);
        assert_eq!(y.len(), 2);
        assert!(y.iter().all(|r| r.len() == 2 && r.iter().all(|v| v.is_finite())));
        // Row 0 must attend more to j=0 (q0.k0=2 > q0.k1=-2, plus pos terms
        // bounded by marker gaps): output closer to v0=[1,-1] than v1.
        assert!(y[0][0] > 0.0 && y[0][1] < 0.0, "y={y:?}");
        // Full block preserves shape and stays finite.
        let table: Vec<Vec<f32>> = (0..9999).map(|i| vec![i as f32, 0.0]).collect();
        let win = pos_window(&table, 2);
        let z = block_forward(&x, &w, &win);
        assert_eq!(z.len(), 2);
        assert!(z.iter().all(|r| r.len() == 2 && r.iter().all(|v| v.is_finite())));
    }

    fn load_layer0(dir: &std::path::Path) -> (BlockWeights, Vec<Vec<f32>>) {
        use super::super::encoder::EncoderManifest;
        use super::super::weights::load_selected;
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest: EncoderManifest =
            crate::parakeet_burn::encoder::load_manifest(&json).unwrap();
        let mut want = std::collections::HashSet::new();
        for suffix in crate::parakeet_burn::encoder::LAYER_ROLES {
            let role = format!("layers.0/{suffix}");
            let e = manifest.role(&role).unwrap();
            want.insert(e.weight.clone());
            want.insert(format!("{}.scale", e.weight));
            want.insert(format!("{}.zero_point", e.weight));
        }
        for n in [
            "layers.0.norm_feed_forward1.weight",
            "layers.0.norm_feed_forward1.bias",
            "layers.0.norm_self_att.weight",
            "layers.0.norm_self_att.bias",
            "layers.0.norm_conv.weight",
            "layers.0.norm_conv.bias",
            "layers.0.norm_feed_forward2.weight",
            "layers.0.norm_feed_forward2.bias",
            "layers.0.norm_out.weight",
            "layers.0.norm_out.bias",
            "layers.0.self_attn.pos_bias_u",
            "layers.0.self_attn.pos_bias_v",
            "pos_enc.table",
        ] {
            want.insert(n.to_string());
        }
        // dw bias initializer name comes from the manifest map.
        want.insert(manifest.dw_bias_init(0).expect("dw bias for layer 0"));
        let tensors =
            load_selected(&dir.join("encoder-burn.safetensors"), &want).unwrap();
        let bw = BlockWeights::load(&tensors, &manifest, 0, 1024, 8, 128, 4096).unwrap();
        let table = match tensors.get("pos_enc.table").unwrap() {
            RawTensor::F32 { shape, data } => {
                assert_eq!(shape, &vec![9999, 1024]);
                data.chunks(1024).map(|c| c.to_vec()).collect::<Vec<_>>()
            }
            _ => panic!("pos table must be F32"),
        };
        (bw, table)
    }

    /// Loads real layer-0 weights + pos table. Ignored: needs converted
    /// weights on disk. `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn real_layer0_block_runs_finite() {
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let (bw, table) = load_layer0(dir);
        // deterministic pseudo-random input, T=8
        let mut seed: u64 = 0x12345678;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let x: Vec<Vec<f32>> = (0..8).map(|_| (0..1024).map(|_| rnd()).collect()).collect();
        let win = pos_window(&table, 8);
        assert_eq!(win.len(), 15);
        let y = block_forward(&x, &bw, &win);
        assert_eq!(y.len(), 8);
        assert!(y.iter().all(|r| r.len() == 1024 && r.iter().all(|v| v.is_finite())));
        // norm_out invariant: (y - beta) / gamma has ~zero mean and
        // ~unit variance (LayerNorm normalizes before the affine map,
        // so the raw row mean is NOT zero — it carries mean(beta)).
        let z: Vec<f32> = y[0]
            .iter()
            .enumerate()
            .map(|(i, v)| (v - bw.norm_out.beta[i]) / bw.norm_out.gamma[i])
            .collect();
        let mean: f32 = z.iter().sum::<f32>() / z.len() as f32;
        let var: f32 = z.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / z.len() as f32;
        assert!(mean.abs() < 1e-3, "mean={mean}");
        assert!((var - 1.0).abs() < 5e-2, "var={var}");
    }

    /// ORT parity for one full Conformer block: golden input is the real
    /// pre-encode output for the T=64 seed-1234 mel (`layer0_golden`,
    /// `/layers.0/norm_out/LayerNormalization_output_0` from onnxruntime).
    /// This is the test that pins the MatMul `(K, N)` -> `[out][in]`
    /// transpose (see `Uint8Linear` docs): without it every Conformer
    /// linear silently computes `W @ x` instead of `W^T @ x`.
    /// Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_layer0_parity() {
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let (bw, table) = load_layer0(dir);
        let bytes =
            std::fs::read("src/parakeet_burn/testdata/layer0_golden.safetensors").unwrap();
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
        let x = f32mat("blk_in_T1024", 8, 1024);
        let want = f32mat("blk_ref", 8, 1024);
        let win = pos_window(&table, 8);
        let y = block_forward(&x, &bw, &win);
        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f32);
        let mut ref_max = 0.0f32;
        for (yr, wr) in y.iter().zip(want.iter()) {
            for (a, b) in yr.iter().zip(wr.iter()) {
                max_abs = max_abs.max((a - b).abs());
                sum_abs += (a - b).abs();
                ref_max = ref_max.max(b.abs());
            }
        }
        let mean_abs = sum_abs / (8.0 * 1024.0);
        // fp32 math vs dynamic int8 activations through 12 quantized
        // linears: ~1% relative is activation-requantization noise.
        // Bound holds ~2.5x margin over measured (max 0.85, mean 0.12
        // at ref_max 75.8).
        assert!(
            max_abs < 0.03 * ref_max.max(1.0),
            "max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
        );
    }
}
