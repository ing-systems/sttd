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
//   B[i,j] = (q_i + v) . P[(j-i) + (T-1)] (sign voted off ORT's own
//   Add_2 = AC + shift(BD); the table is position-reversed, see
//   `relative_position_window`), scores / sqrt(128). QK^T/BD/AV run in fp32 — the
//   graph uses plain `MatMul` there (only the projections quantize).

use std::collections::HashMap;

use anyhow::{Context, Result};

use super::runtime::CpuBackend;
use super::manifest::EncoderManifest;
use super::int_gemm::{sigmoid, Epilogue};
use super::ops::{glu, layernorm_batch, softmax_rows, swish_inplace, LayerNormP};
use super::quant_linear::Uint8Linear;
use super::tensor_ctx::TensorCtx;
use super::weights::RawTensor;

use rayon::prelude::*;

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
    /// Lazily-built dequantized weight + bias tensors (same rationale as
    /// `quant_linear::CachedT`: the `[C,1,K]` store is tiny but `from_data` per
    /// call still costs an allocation + copy per block per forward).
    pub(crate) cache: std::sync::OnceLock<(
        burn::tensor::Tensor<CpuBackend, 3>,
        burn::tensor::Tensor<CpuBackend, 1>,
    )>,
}

impl DwConv {
    pub(crate) fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use super::quant_linear::dynamic_quantize_u8;
        let t = x.len();
        let c = self.channels;
        assert!(x.iter().all(|r| r.len() == c));
        // Per-tensor dynamic quantization over the whole `[T,C]` input,
        // integer MAC per tap (mirrors `ConvInteger`; see `quant` docs).
        let flat: Vec<f32> = x.iter().flat_map(|r| r.iter().copied()).collect();
        let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
        let zpw = self.zero_point.round() as i64;
        let mut padded = vec![zpx; c * (self.pad_left + t + self.pad_right)];
        for (i, _row) in x.iter().enumerate() {
            let base = (self.pad_left + i) * c;
            let qbase = i * c;
            padded[base..base + c].copy_from_slice(&xq[qbase..qbase + c]);
        }
        // NOTE: zero-padding in quantized space is `zpx` (= real 0), so
        // padded taps contribute nothing after zp subtraction.
        let out_len = self.pad_left + t + self.pad_right - self.kernel + 1;
        let mut out = vec![vec![0.0f32; c]; out_len];
        for (t_out, row) in out.iter_mut().enumerate() {
            for ch in 0..c {
                let mut acc = 0i64;
                let base = ch * self.kernel;
                for k in 0..self.kernel {
                    let q = padded[(t_out + k) * c + ch] as i64 - zpx as i64;
                    acc += (self.weight_u8[base + k] as i64 - zpw) * q;
                }
                row[ch] = acc as f32 * (self.scale * sx) + self.bias[ch];
            }
        }
        out
    }

    /// Build the cached weight + bias tensors now (load-time warming).
    pub(crate) fn prime_cache(&self, device: &super::runtime::CpuDevice) {
        let _ = self.cached(device);
    }

    /// Dequantized `[C,1,K]` weight + `[C]` bias tensors, built once.
    /// Bit-identical to the old per-call dequant (same per-element
    /// `(q - zp) * scale`, pure reorder into the conv layout).
    fn cached(
        &self,
        device: &super::runtime::CpuDevice,
    ) -> &(
        burn::tensor::Tensor<CpuBackend, 3>,
        burn::tensor::Tensor<CpuBackend, 1>,
    ) {
        self.cache.get_or_init(|| {
            use burn::tensor::{Tensor, TensorData};
            let zpw = self.zero_point;
            let wflat: Vec<f32> = self
                .weight_u8
                .iter()
                .map(|&q| (q as f32 - zpw) * self.scale)
                .collect();
            let wt: Tensor<CpuBackend, 3> = Tensor::from_data(
                TensorData::new(wflat, [self.channels, 1, self.kernel]),
                device,
            );
            let bt: Tensor<CpuBackend, 1> = Tensor::from_data(
                TensorData::new(self.bias.clone(), [self.channels]),
                device,
            );
            (wt, bt)
        })
    }

    /// Tensor twin of `forward`: identical per-tensor activation
    /// quantization, dequantized to f32, one Flex depthwise conv1d
    /// (`groups == channels`) against the cached dequantized weights with
    /// bias. Zero-padding is real 0 on both sides (quantized `zpx`
    /// subtracts to 0, Flex pads `Constant(0.0)`), and pads are symmetric
    /// (`kernel / 2` each side), so plain `ConvOptions` suffices.
    /// Residual is f32 summation order only.
    pub(crate) fn forward_t(
        &self,
        x: &[Vec<f32>],
        device: &super::runtime::CpuDevice,
    ) -> Vec<Vec<f32>> {
        use burn::tensor::{module::conv1d, ops::ConvOptions, Tensor, TensorData};
        use super::quant_linear::dynamic_quantize_u8;
        let t = x.len();
        assert!(t >= 1, "dw forward_t: empty input");
        let c = self.channels;
        assert!(x.iter().all(|r| r.len() == c));
        assert_eq!((self.pad_left, self.pad_right), (self.kernel / 2, self.kernel / 2));
        let flat: Vec<f32> = x.iter().flat_map(|r| r.iter().copied()).collect();
        let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
        // [T][C] rows -> [1, C, T], dequantizing straight into the planar
        // layout (no intermediate dequant Vec).
        let mut ncl = vec![0.0f32; c * t];
        for (i, row) in xq.chunks(c).enumerate() {
            for (ch, &q) in row.iter().enumerate() {
                ncl[ch * t + i] = (q as f32 - zpx as f32) * sx;
            }
        }
        let xt: Tensor<CpuBackend, 3> = Tensor::from_data(TensorData::new(ncl, [1, c, t]), device);
        let (wt, bt) = self.cached(device);
        let y = conv1d(
            xt,
            wt.clone(),
            Some(bt.clone()),
            ConvOptions::new([1], [self.pad_left], [1], c),
        );
        // [1, C, T] -> [T][C] rows.
        let back: Vec<f32> = y.into_data().to_vec::<f32>().expect("dw forward_t: to_vec");
        let mut out = vec![vec![0.0f32; c]; t];
        for (ch, plane) in back.chunks(t).enumerate() {
            for (i, v) in plane.iter().enumerate() {
                out[i][ch] = *v;
            }
        }
        out
    }

    /// Exact parallel twin of `forward`: the same `dynamic_quantize_u8`
    /// call (identical q/s/zpx by construction — same function, same
    /// input), the same padded layout, and the same per-element tap-sum
    /// in the same serial order — rayon only spreads independent output
    /// rows over threads, so outputs match `forward` bit-for-bit (pinned
    /// by `dw_par_matches_scalar`). Replaces `forward_t` (Flex grouped
    /// conv1d + dequant/re-planar copies per call) on the tensor path:
    /// the integer MACs here are ~300k per layer, so Flex dispatch and
    /// its copies dominate — and Flex additionally drifts by f32
    /// summation order, which this removes.
    /// `swish` fuses the conv's Swish activation into the same output
    /// write (`v * sigmoid(v)` — the exact expression `swish_inplace`
    /// applies afterwards), saving a full pass over the `[T][C]` output.
    pub(crate) fn forward_par(&self, x: &[Vec<f32>], swish: bool) -> Vec<Vec<f32>> {
        use super::quant_linear::dynamic_quantize_u8;
        let t = x.len();
        let c = self.channels;
        assert!(x.iter().all(|r| r.len() == c));
        let flat: Vec<f32> = x.iter().flat_map(|r| r.iter().copied()).collect();
        let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
        let zpw = self.zero_point.round() as i64;
        let mut padded = vec![zpx; c * (self.pad_left + t + self.pad_right)];
        for (i, _row) in x.iter().enumerate() {
            let base = (self.pad_left + i) * c;
            let qbase = i * c;
            padded[base..base + c].copy_from_slice(&xq[qbase..qbase + c]);
        }
        let out_len = self.pad_left + t + self.pad_right - self.kernel + 1;
        let scale_sx = self.scale * sx;
        let (w, kernel, bias) = (&self.weight_u8, self.kernel, &self.bias);
        (0..out_len)
            .into_par_iter()
            .map(|t_out| {
                let mut row = vec![0.0f32; c];
                for ch in 0..c {
                    let mut acc = 0i64;
                    let base = ch * kernel;
                    for k in 0..kernel {
                        let q = padded[(t_out + k) * c + ch] as i64 - zpx as i64;
                        acc += (w[base + k] as i64 - zpw) * q;
                    }
                    let mut v = acc as f32 * scale_sx + bias[ch];
                    if swish {
                        v = v * sigmoid(v);
                    }
                    row[ch] = v;
                }
                row
            })
            .collect()
    }

    /// Dispatch: exact parallel integer `forward_par` on the tensor path,
    /// exact serial integer `forward` on the scalar path. (`forward_t`,
    /// the Flex grouped-conv twin, is kept for its parity test only.)
    pub(crate) fn dispatch(&self, x: &[Vec<f32>], ctx: &super::tensor_ctx::TensorCtx) -> Vec<Vec<f32>> {
        if ctx.enabled {
            self.forward_par(x, false)
        } else {
            self.forward(x)
        }
    }
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
        [k, n] => (*k, *n, super::quant_linear::transpose_u8(data, *k, *n)),
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
        cache: std::sync::OnceLock::new(),
        w_i16: std::sync::OnceLock::new(),
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
                cache: std::sync::OnceLock::new(),
            },
        })
    }
}

impl BlockWeights {
    /// Build every cached tensor in this layer now (load-time warming —
    /// see `Uint8Linear::prime_cache`): the 11 quantized linears plus the
    /// depthwise conv. Each warms independently (no shared state), so the
    /// encoder warms all layers in parallel via rayon's pool.
    pub(crate) fn prime_cache(&self, device: &super::runtime::CpuDevice) {
        self.ffn1_l1.prime_cache(device);
        self.ffn1_l2.prime_cache(device);
        self.ffn2_l1.prime_cache(device);
        self.ffn2_l2.prime_cache(device);
        self.wq.prime_cache(device);
        self.wk.prime_cache(device);
        self.wv.prime_cache(device);
        self.wpos.prime_cache(device);
        self.wout.prime_cache(device);
        self.pw1.prime_cache(device);
        self.pw2.prime_cache(device);
        self.dw.prime_cache(device);
    }
}

/// Slice the relative window for length T from the full table:
/// rows [5000-T .. 5000+T-1] (length 2T-1, distance 0 at index T-1).
/// Table layout (verified): `table[i]` = PE(4999-i), interleaved sin/cos.
pub(crate) fn relative_position_window(table: &[Vec<f32>], t: usize) -> Vec<Vec<f32>> {
    assert!(table.len() >= 9999);
    assert!(t >= 1 && 5000 - t + (2 * t - 1) <= table.len());
    table[5000 - t..5000 + t - 1].to_vec()
}

fn linear_batch(lin: &Uint8Linear, x: &[Vec<f32>], ctx: &TensorCtx) -> Vec<Vec<f32>> {
    // Batched: the activation quantization spans the whole `[T,K]` input
    // tensor, as ORT's `DynamicQuantizeLinear` does (see `quant` docs).
    // Tensor path replicates that span exactly (see `forward_batch_t`).
    lin.batch_dispatch(x, ctx)
}

fn add_inplace(a: &mut [Vec<f32>], b: &[Vec<f32>]) {
    a.par_iter_mut().enumerate().for_each(|(i, ra)| {
        for (x, y) in ra.iter_mut().zip(b[i].iter()) {
            *x += *y;
        }
    });
}

fn scale_inplace(a: &mut [Vec<f32>], s: f32) {
    a.par_iter_mut().for_each(|r| {
        for x in r.iter_mut() {
            *x *= s;
        }
    });
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

/// Pre-softmax relative scores `[T][H][T]`: AC + XL-shifted BD, scaled.
/// B[i,j] = (q_i + u) . k_j + (q_i + v) . P[(j-i) + (T-1)].
/// Split out of `relative_attention` so parity tests can compare scores
/// directly against the graph's pre-softmax `Div` output.
pub(crate) fn relative_scores(
    x: &[Vec<f32>],
    w: &BlockWeights,
    relative_position_window: &[Vec<f32>],
    ctx: &TensorCtx,
) -> Vec<Vec<Vec<f32>>> {
    let t = x.len();
    let (h, dh) = (w.n_heads, w.d_head);
    let q = split_heads(&linear_batch(&w.wq, x, ctx), h, dh);
    let k = split_heads(&linear_batch(&w.wk, x, ctx), h, dh);
    let p = split_heads(&linear_batch(&w.wpos, relative_position_window, ctx), h, dh);
    assert_eq!(p.len(), 2 * t - 1);
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        out.push(Vec::with_capacity(h));
        for hh in 0..h {
            let qu: Vec<f32> = q[i][hh]
                .iter()
                .zip(w.pos_bias_u[hh].iter())
                .map(|(a, b)| a + b)
                .collect();
            let qv: Vec<f32> = q[i][hh]
                .iter()
                .zip(w.pos_bias_v[hh].iter())
                .map(|(a, b)| a + b)
                .collect();
            let mut row = vec![0.0f32; t];
            for j in 0..t {
                let ac: f32 = qu.iter().zip(k[j][hh].iter()).map(|(a, b)| a * b).sum();
                // Window index runs opposite to the query-key distance:
                // voted off ORT's own Add_2 = AC + shift(BD) during staged
                // bring-up (see `ort_layer0_stages_parity`), kk = (j-i)+(T-1).
                // The table itself is position-reversed
                // (table[i] = PE(4999-i)), which is why the sign looks
                // backwards.
                let kk = (j as isize - i as isize + t as isize - 1) as usize;
                let bd: f32 = qv.iter().zip(p[kk][hh].iter()).map(|(a, b)| a * b).sum();
                row[j] = (ac + bd) * scale;
            }
            out[i].push(row);
        }
    }
    out
}
/// Fused core of relative attention: one rayon task per query row
/// computing AC + shifted BD, softmax and value mixing in the *scalar*
/// path's exact accumulation order — each dot is a sequential f32 sum,
/// the softmax follows `softmax_rows` step for step, AV accumulates over
/// `j` in order — so the result is bit-identical to the scalar core
/// (pinned by `fused_attention_matches_scalar`). What it removes is the
/// Flex path's ~100 tiny op dispatches per layer (and the scalar path's
/// per-(row, head) `Vec`s); T rows × H heads in one parallel pass.
fn attention_core_into(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    p: &[Vec<f32>],
    n_heads: usize,
    d_head: usize,
    pos_bias_u: &[Vec<f32>],
    pos_bias_v: &[Vec<f32>],
    out: &mut [Vec<f32>],
) {
    let t = q.len();
    assert_eq!(k.len(), t);
    assert_eq!(v.len(), t);
    assert_eq!(p.len(), 2 * t - 1);
    assert_eq!(out.len(), t);
    let scale = 1.0 / (d_head as f32).sqrt();
    out.par_iter_mut().enumerate().for_each(|(i, orow)| {
        // Reused across heads: no per-(row, head) allocations.
        let mut scores = vec![0.0f32; t];
        let mut qu = vec![0.0f32; d_head];
        let mut qv = vec![0.0f32; d_head];
        for hh in 0..n_heads {
            let qs = &q[i][hh * d_head..(hh + 1) * d_head];
            for d in 0..d_head {
                qu[d] = qs[d] + pos_bias_u[hh][d];
            }
            for d in 0..d_head {
                qv[d] = qs[d] + pos_bias_v[hh][d];
            }
            for j in 0..t {
                let kj = &k[j][hh * d_head..(hh + 1) * d_head];
                let mut ac = 0.0f32;
                for d in 0..d_head {
                    ac += qu[d] * kj[d];
                }
                // Same window index as `relative_scores` (see its docs).
                let kk = (j as isize - i as isize + t as isize - 1) as usize;
                let pj = &p[kk][hh * d_head..(hh + 1) * d_head];
                let mut bd = 0.0f32;
                for d in 0..d_head {
                    bd += qv[d] * pj[d];
                }
                scores[j] = (ac + bd) * scale;
            }
            // `softmax_rows`' exact sequence.
            let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut s = 0.0f32;
            for x in scores.iter_mut() {
                *x = (*x - m).exp();
                s += *x;
            }
            let inv = 1.0 / s;
            for x in scores.iter_mut() {
                *x *= inv;
            }
            let ov = &mut orow[hh * d_head..(hh + 1) * d_head];
            for o in ov.iter_mut() {
                *o = 0.0;
            }
            for j in 0..t {
                let a = scores[j];
                let vj = &v[j][hh * d_head..(hh + 1) * d_head];
                for d in 0..d_head {
                    ov[d] += a * vj[d];
                }
            }
        }
    });
}
/// Relative multi-head attention (Transformer-XL shift):
/// B[i,j] = (q_i + v) . P[(j-i) + (T-1)], scores / sqrt(dh).
/// Scores come from `relative_scores`; this adds softmax + value mixing.
/// Scalar reference / `STTD_BURN_TENSOR=0` path; the tensor path goes
/// through [`relative_attention_into`].
pub(crate) fn relative_attention(
    x: &[Vec<f32>],
    w: &BlockWeights,
    relative_position_window: &[Vec<f32>],
    ctx: &TensorCtx,
) -> Vec<Vec<f32>> {
    if ctx.enabled {
        let t = x.len();
        let mut out = vec![vec![0.0f32; w.n_heads * w.d_head]; t];
        relative_attention_into(x, w, relative_position_window, ctx, &mut out, None);
        return out;
    }
    relative_attention_scalar(x, w, relative_position_window, ctx)
}

/// Exact scalar core of [`relative_attention`] (softmax + value mixing over
/// `relative_scores`), used as the reference the fused core is pinned to.
fn relative_attention_scalar(
    x: &[Vec<f32>],
    w: &BlockWeights,
    relative_position_window: &[Vec<f32>],
    ctx: &TensorCtx,
) -> Vec<Vec<f32>> {
    let t = x.len();
    let (h, dh) = (w.n_heads, w.d_head);
    let v = split_heads(&linear_batch(&w.wv, x, ctx), h, dh);
    let scores = relative_scores(x, w, relative_position_window, ctx);
    // scores is [T][H][T]; softmax over keys per (frame, head).
    let mut heads_out = Vec::with_capacity(h);
    for hh in 0..h {
        let mut per_head: Vec<Vec<f32>> = (0..t).map(|i| scores[i][hh].clone()).collect();
        softmax_rows(&mut per_head);
        let mut o = vec![vec![0.0f32; dh]; t];
        for i in 0..t {
            for j in 0..t {
                let a = per_head[i][j];
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
    linear_batch(&w.wout, &merged, ctx)
}

/// [`relative_attention`] writing straight into `out`: `None` overwrites,
/// `Some(scale)` accumulates `out += scale * attn`. The tensor path runs
/// the shared-A Q/K/V projections (one activation quantization for all
/// three — exact, they see the same input), the fused
/// [`attention_core_into`], and the out-projection with the residual add
/// fused into its GEMM epilogue.
pub(crate) fn relative_attention_into(
    x: &[Vec<f32>],
    w: &BlockWeights,
    relative_position_window: &[Vec<f32>],
    ctx: &TensorCtx,
    out: &mut [Vec<f32>],
    add: Option<f32>,
) {
    let t = x.len();
    let (h, dh) = (w.n_heads, w.d_head);
    let d = h * dh;
    assert_eq!(out.len(), t);
    if !ctx.enabled {
        let a = relative_attention_scalar(x, w, relative_position_window, ctx);
        match add {
            None => {
                for (o, s) in out.iter_mut().zip(&a) {
                    o.copy_from_slice(s);
                }
            }
            Some(sc) => {
                out.par_iter_mut().zip(a.par_iter()).for_each(|(o, s)| {
                    for (x, &y) in o.iter_mut().zip(s.iter()) {
                        *x += sc * y;
                    }
                });
            }
        }
        return;
    }
    let qkv = Uint8Linear::shared_activation(ctx, x);
    let mut q = vec![vec![0.0f32; d]; t];
    let mut k = vec![vec![0.0f32; d]; t];
    let mut v = vec![vec![0.0f32; d]; t];
    let mut proj = |lin: &Uint8Linear, dst: &mut [Vec<f32>]| match &qkv {
        Some(a) => lin.forward_quantized_into(a, ctx, dst, Epilogue::None),
        None => lin.forward_batch_into(x, ctx, dst, Epilogue::None),
    };
    proj(&w.wq, &mut q);
    proj(&w.wk, &mut k);
    proj(&w.wv, &mut v);
    let mut p = vec![vec![0.0f32; d]; relative_position_window.len()];
    w.wpos.forward_batch_into(relative_position_window, ctx, &mut p, Epilogue::None);
    let mut merged = vec![vec![0.0f32; d]; t];
    attention_core_into(
        &q,
        &k,
        &v,
        &p,
        h,
        dh,
        &w.pos_bias_u,
        &w.pos_bias_v,
        &mut merged,
    );
    let epi = match add {
        Some(sc) => Epilogue::AddScale(sc),
        None => Epilogue::None,
    };
    w.wout.forward_batch_into(&merged, ctx, out, epi);
}

pub(crate) fn ffn(x: &[Vec<f32>], l1: &Uint8Linear, l2: &Uint8Linear, ctx: &TensorCtx) -> Vec<Vec<f32>> {
    let mut h = linear_batch(l1, x, ctx);
    h.par_iter_mut().for_each(|row| swish_inplace(row));
    linear_batch(l2, &h, ctx)
}

/// Tensor-path [`ffn`] accumulating the branch straight into `out`:
/// `out += scale * l2(swish(l1(x)))`. The Swish runs in l1's GEMM
/// epilogue and the residual add in l2's (`AddScale(scale)`), so there is
/// no activation pass, no scale pass, no add pass and no branch buffer.
/// Bit-identical to `ffn` + `scale_inplace(0.5)` + `add_inplace`.
pub(crate) fn ffn_into(
    x: &[Vec<f32>],
    l1: &Uint8Linear,
    l2: &Uint8Linear,
    ctx: &TensorCtx,
    out: &mut [Vec<f32>],
    scale: f32,
) {
    let t = x.len();
    assert_eq!(out.len(), t);
    let mut hidden = vec![vec![0.0f32; l1.out_dim]; t];
    l1.forward_batch_into(x, ctx, &mut hidden, Epilogue::Swish);
    l2.forward_batch_into(&hidden, ctx, out, Epilogue::AddScale(scale));
}

pub(crate) fn conv_block(x: &[Vec<f32>], w: &BlockWeights, ctx: &TensorCtx) -> Vec<Vec<f32>> {
    let gated: Vec<Vec<f32>> = linear_batch(&w.pw1, x, ctx)
        .into_par_iter()
        .map(|r| glu(&r))
        .collect();
    let mut dwed = w.dw.dispatch(&gated, ctx);
    // SAME padding (k=9, pads 4/4) preserves length; trim defensively.
    dwed.truncate(x.len());
    while dwed.len() < x.len() {
        dwed.push(vec![0.0; dwed[0].len()]);
    }
    dwed.par_iter_mut().for_each(|row| swish_inplace(row));
    linear_batch(&w.pw2, &dwed, ctx)
}

/// Tensor-path [`conv_block`] accumulating into `out`: pw1's GEMM writes
/// the GLU result directly (the 2048-wide pre-activation never exists as
/// a buffer), the depthwise conv's Swish is fused into its output write,
/// and pw2 accumulates into `out` (`AddScale`). Bit-identical to
/// `conv_block` + `add_inplace`.
pub(crate) fn conv_block_into(
    x: &[Vec<f32>],
    w: &BlockWeights,
    ctx: &TensorCtx,
    out: &mut [Vec<f32>],
    scale: f32,
) {
    let t = x.len();
    assert_eq!(out.len(), t);
    let mut gated = vec![vec![0.0f32; w.dw.channels]; t];
    w.pw1.forward_batch_into(x, ctx, &mut gated, Epilogue::Glu);
    let mut dwed = if ctx.enabled {
        w.dw.forward_par(&gated, true)
    } else {
        let mut y = w.dw.forward(&gated);
        y.par_iter_mut().for_each(|r| swish_inplace(r));
        y
    };
    // SAME padding (k=9, pads 4/4) preserves length; trim defensively.
    dwed.truncate(t);
    while dwed.len() < t {
        dwed.push(vec![0.0; w.dw.channels]);
    }
    w.pw2.forward_batch_into(&dwed, ctx, out, Epilogue::AddScale(scale));
}

/// One full block: half-step FFN sandwich + final norm.
pub(crate) fn conformer_forward(
    x: &[Vec<f32>],
    w: &BlockWeights,
    relative_position_window: &[Vec<f32>],
    ctx: &TensorCtx,
) -> Vec<Vec<f32>> {
    let mut h = x.to_vec();
    if ctx.enabled {
        // Fused tensor path: each branch accumulates straight into `h`
        // through its GEMM epilogues (no branch buffers, no separate
        // scale/add passes).
        ffn_into(
            &layernorm_batch(&h, &w.norm_ff1),
            &w.ffn1_l1,
            &w.ffn1_l2,
            ctx,
            &mut h,
            0.5,
        );
        let na = layernorm_batch(&h, &w.norm_att);
        relative_attention_into(&na, w, relative_position_window, ctx, &mut h, Some(1.0));
        let nc = layernorm_batch(&h, &w.norm_conv);
        conv_block_into(&nc, w, ctx, &mut h, 1.0);
        let n2 = layernorm_batch(&h, &w.norm_ff2);
        ffn_into(&n2, &w.ffn2_l1, &w.ffn2_l2, ctx, &mut h, 0.5);
        return layernorm_batch(&h, &w.norm_out);
    }
    let mut f = ffn(&layernorm_batch(&h, &w.norm_ff1), &w.ffn1_l1, &w.ffn1_l2, ctx);
    scale_inplace(&mut f, 0.5);
    add_inplace(&mut h, &f);
    let a = relative_attention(&layernorm_batch(&h, &w.norm_att), w, relative_position_window, ctx);
    add_inplace(&mut h, &a);
    let c = conv_block(&layernorm_batch(&h, &w.norm_conv), w, ctx);
    add_inplace(&mut h, &c);
    let mut f2 = ffn(&layernorm_batch(&h, &w.norm_ff2), &w.ffn2_l1, &w.ffn2_l2, ctx);
    scale_inplace(&mut f2, 0.5);
    add_inplace(&mut h, &f2);
    layernorm_batch(&h, &w.norm_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ops::layernorm_row;
    use crate::parakeet_burn::quant_linear::Uint8Linear;

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
        // k=3, center tap zp+1 with side taps at zp (dequant 0).
        // Constant input quantizes exactly (guard: s=1, zp=0), so out == in
        // everywhere (zero pad contributes nothing, and the quantized pad
        // value `zpx` likewise subtracts to zero).
        let dw = DwConv {
            weight_u8: vec![128, 129, 128],
            scale: 1.0,
            zero_point: 128.0,
            bias: vec![0.0],
            channels: 1,
            kernel: 3,
            pad_left: 1,
            pad_right: 1,
            cache: std::sync::OnceLock::new(),
        };
        let x = vec![vec![2.0], vec![2.0], vec![2.0], vec![2.0]];
        let y = dw.forward(&x);
        assert_eq!(y.len(), 4);
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-5, "y={y:?}");
        }
    }

    #[test]
    fn dw_par_matches_scalar_exactly() {
        // Parallel integer twin must equal the serial integer path
        // bit-for-bit (independent outputs, identical per-element order),
        // over odd lengths, kernel sizes and mixed-sign weights/biases.
        let mk = |channels: usize, kernel: usize| DwConv {
            weight_u8: (0..channels * kernel).map(|i| (i * 37 % 256) as u8).collect(),
            scale: 0.02,
            zero_point: 121.0,
            bias: (0..channels).map(|i| i as f32 * 0.125 - 1.0).collect(),
            channels,
            kernel,
            pad_left: kernel / 2,
            pad_right: kernel / 2,
            cache: std::sync::OnceLock::new(),
        };
        let mut seed: u64 = 0xD1E;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 6.0
        };
        for (channels, kernel, t) in [(1, 3, 4), (2, 3, 5), (8, 9, 32), (16, 5, 17)] {
            let dw = mk(channels, kernel);
            let x: Vec<Vec<f32>> = (0..t)
                .map(|_| (0..channels).map(|_| rnd()).collect())
                .collect();
            let s = dw.forward(&x);
            assert_eq!(dw.forward_par(&x, false), s, "c={channels} k={kernel} t={t}");
            // Swish-fused twin: same values, activation applied in the
            // output write (exact vs swish_inplace over `forward`).
            let mut want = s.clone();
            want.iter_mut().for_each(|r| swish_inplace(r));
            assert_eq!(dw.forward_par(&x, true), want, "c={channels} k={kernel} t={t}");
        }
    }

    #[test]
    fn dw_tensor_matches_scalar() {
        use super::super::runtime::cpu_device;
        // 2 channels, k=3, non-trivial taps/scales/biases, mixed signs.
        let dw = DwConv {
            weight_u8: vec![3, 250, 7, 200, 10, 44],
            scale: 0.02,
            zero_point: 121.0,
            bias: vec![0.5, -0.25],
            channels: 2,
            kernel: 3,
            pad_left: 1,
            pad_right: 1,
            cache: std::sync::OnceLock::new(),
        };
        let x = vec![
            vec![1.0, -2.0],
            vec![0.5, 4.0],
            vec![-3.0, 0.25],
            vec![2.0, -0.75],
            vec![-1.0, 1.5],
        ];
        let dev = cpu_device();
        let s = dw.forward(&x);
        let t = dw.forward_t(&x, &dev);
        assert_eq!((s.len(), t.len()), (5, 5));
        let (mut mx, mut rm) = (0.0f32, 0.0f32);
        for (a, b) in s.iter().zip(t.iter()) {
            for (p, q) in a.iter().zip(b.iter()) {
                mx = mx.max((p - q).abs());
                rm = rm.max(q.abs());
            }
        }
        eprintln!("dw tensor-vs-scalar: max_abs={mx:.2e} ref_max={rm:.3}");
        assert!(mx < 1e-4 * rm.max(1.0), "mx={mx} rm={rm}");
    }

    #[test]
    fn relative_position_window_slices_centered_distances() {
        // synthetic table[i] = i: window for T=4 is rows 4996..5002.
        let table: Vec<Vec<f32>> = (0..9999).map(|i| vec![i as f32]).collect();
        let w = relative_position_window(&table, 4);
        assert_eq!(w.len(), 7);
        assert_eq!(w[0][0], 4996.0);
        assert_eq!(w[3][0], 4999.0); // distance 0 at index T-1
        assert_eq!(w[6][0], 5002.0);
    }

    #[test]
    fn attention_relative_index_uses_distance() {
        // 1 head, dh=1, d=1, T=2: identity projections, zero bias.
        // Markers increase with window index: P = [10, 20, 30].
        // x = [[1], [0]] kills AC/BD on row 1 (uniform -> mean) while row
        // 0 sees AC = [1, 0], BD = [P[1], P[2]] = [20, 30] under the
        // correct kk = (j-i)+(T-1). Row 0 must therefore prefer key 1
        // (output near v1 = 0); the flipped sign would put P[0] = 10 on
        // key 1 and prefer key 0 instead. This pins the shift direction
        // voted off ORT's Add_2 (see `relative_scores`).
        let eye = Uint8Linear {
            weight_u8: vec![1],
            scale: vec![1.0],
            zero_point: vec![0.0],
            bias: None,
            in_dim: 1,
            out_dim: 1,
            cache: std::sync::OnceLock::new(),
            w_i16: std::sync::OnceLock::new(),
        };
        let one = LayerNormP { gamma: vec![1.0], beta: vec![0.0] };
        let w = BlockWeights {
            n_heads: 1,
            d_head: 1,
            norm_ff1: one.clone(),
            norm_att: one.clone(),
            norm_conv: one.clone(),
            norm_ff2: one.clone(),
            norm_out: one.clone(),
            ffn1_l1: eye.clone(),
            ffn1_l2: eye.clone(),
            ffn2_l1: eye.clone(),
            ffn2_l2: eye.clone(),
            wq: eye.clone(),
            wk: eye.clone(),
            wv: eye.clone(),
            wpos: eye.clone(),
            wout: eye.clone(),
            pos_bias_u: vec![vec![0.0]],
            pos_bias_v: vec![vec![0.0]],
            pw1: Uint8Linear {
                weight_u8: vec![1, 0],
                scale: vec![1.0, 1.0],
                zero_point: vec![0.0, 0.0],
                bias: None,
                in_dim: 1,
                out_dim: 2,
                cache: std::sync::OnceLock::new(),
                w_i16: std::sync::OnceLock::new(),
            },
            pw2: Uint8Linear {
                weight_u8: vec![1],
                scale: vec![1.0],
                zero_point: vec![0.0],
                bias: None,
                in_dim: 1,
                out_dim: 1,
                cache: std::sync::OnceLock::new(),
                w_i16: std::sync::OnceLock::new(),
            },
            dw: DwConv {
                weight_u8: vec![0, 1, 0],
                scale: 1.0,
                zero_point: 0.0,
                bias: vec![0.0],
                channels: 1,
                kernel: 3,
                pad_left: 1,
                pad_right: 1,
                cache: std::sync::OnceLock::new(),
            },
        };
        let x = vec![vec![1.0], vec![0.0]];
        let pw = vec![vec![10.0], vec![20.0], vec![30.0]];
        let y = relative_attention(&x, &w, &pw, &TensorCtx::scalar());
        assert_eq!(y.len(), 2);
        // Row 0: scores [1+20, 0+30] -> attends to key 1 (v=0).
        assert!(y[0][0] < 0.1, "y={y:?}");
        // Row 1: all-zero scores -> uniform mean of v = 0.5.
        assert!((y[1][0] - 0.5).abs() < 1e-5, "y={y:?}");
        // Tensor core must agree with the scalar path (and therefore
        // pick the same shift direction): any gather/indexing bug shows
        // up here before real weights are involved. On the integer
        // fast path the whole comparison is bit-exact (shared-A Q/K/V
        // quantize identically, the fused core replicates the scalar
        // accumulation order).
        let yt = relative_attention(
            &x,
            &w,
            &pw,
            &TensorCtx {
                device: super::super::runtime::cpu_device(),
                enabled: true,
            },
        );
        assert_eq!(yt.len(), 2);
        let exact = crate::parakeet_burn::int_gemm::has_avx512_int_gemm();
        for (a, b) in y.iter().zip(yt.iter()) {
            if exact {
                assert_eq!(a, b, "scalar={y:?} tensor={yt:?}");
            } else {
                assert!((a[0] - b[0]).abs() < 1e-5, "scalar={y:?} tensor={yt:?}");
            }
        }
        // Full block preserves shape and stays finite.
        let table: Vec<Vec<f32>> = (0..9999).map(|i| vec![i as f32]).collect();
        let win = relative_position_window(&table, 2);
        let z = conformer_forward(&vec![vec![1.0], vec![0.0]], &w, &win, &TensorCtx::scalar());
        assert_eq!(z.len(), 2);
        assert!(z.iter().all(|r| r.len() == 1 && r.iter().all(|v| v.is_finite())));
    }

    /// The fused attention core replicates the scalar core bit-for-bit:
    /// identical `qu/qv`, sequential f32 dots, `softmax_rows`' sequence
    /// and in-order AV accumulation, over multiple T and heads.
    #[test]
    fn fused_attention_core_matches_scalar_core() {
        let mut seed: u64 = 0xA77;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 4.0
        };
        for (t, h, dh) in [(1usize, 1usize, 4usize), (2, 2, 4), (5, 2, 4), (8, 2, 8)] {
            let d = h * dh;
            let q: Vec<Vec<f32>> = (0..t).map(|_| (0..d).map(|_| rnd()).collect()).collect();
            let k: Vec<Vec<f32>> = (0..t).map(|_| (0..d).map(|_| rnd()).collect()).collect();
            let v: Vec<Vec<f32>> = (0..t).map(|_| (0..d).map(|_| rnd()).collect()).collect();
            let p: Vec<Vec<f32>> = (0..2 * t - 1)
                .map(|_| (0..d).map(|_| rnd()).collect())
                .collect();
            let pu: Vec<Vec<f32>> = (0..h).map(|_| (0..dh).map(|_| rnd()).collect()).collect();
            let pv: Vec<Vec<f32>> = (0..h).map(|_| (0..dh).map(|_| rnd()).collect()).collect();
            // Scalar reference: the exact code `relative_scores` + `softmax_rows`
            // + AV run (same per-head slice copies, same orders).
            let scale = 1.0 / (dh as f32).sqrt();
            let mut want = vec![vec![0.0f32; d]; t];
            for i in 0..t {
                for hh in 0..h {
                    let qs = &q[i][hh * dh..(hh + 1) * dh];
                    let qu: Vec<f32> =
                        qs.iter().zip(pu[hh].iter()).map(|(a, b)| a + b).collect();
                    let qv: Vec<f32> =
                        qs.iter().zip(pv[hh].iter()).map(|(a, b)| a + b).collect();
                    let mut row = vec![0.0f32; t];
                    for j in 0..t {
                        let ac: f32 = qu
                            .iter()
                            .zip(k[j][hh * dh..(hh + 1) * dh].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let kk = (j as isize - i as isize + t as isize - 1) as usize;
                        let bd: f32 = qv
                            .iter()
                            .zip(p[kk][hh * dh..(hh + 1) * dh].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        row[j] = (ac + bd) * scale;
                    }
                    let mut rows = vec![row];
                    softmax_rows(&mut rows);
                    for j in 0..t {
                        let a = rows[0][j];
                        for dd in 0..dh {
                            want[i][hh * dh + dd] += a * v[j][hh * dh + dd];
                        }
                    }
                }
            }
            let mut got = vec![vec![0.0f32; d]; t];
            attention_core_into(&q, &k, &v, &p, h, dh, &pu, &pv, &mut got);
            assert_eq!(got, want, "t={t} h={h} dh={dh}");
        }
    }

    fn load_layer0(dir: &std::path::Path) -> (BlockWeights, Vec<Vec<f32>>) {
        use super::super::manifest::EncoderManifest;
        use super::super::weights::load_selected;
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest: EncoderManifest =
            crate::parakeet_burn::manifest::load_manifest(&json).unwrap();
        let mut want = std::collections::HashSet::new();
        for suffix in crate::parakeet_burn::manifest::LAYER_ROLES {
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
        let win = relative_position_window(&table, 8);
        assert_eq!(win.len(), 15);
        let y = conformer_forward(&x, &bw, &win, &TensorCtx::scalar());
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

    /// ORT parity for one full Conformer block: golden input is the real    /// pre-encode output for the T=64 seed-1234 mel (`layer0_golden`,
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
        let win = relative_position_window(&table, 8);
        let y = conformer_forward(&x, &bw, &win, &TensorCtx::scalar());
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
        eprintln!("layer0: max_abs={max_abs:.6} mean_abs={mean_abs:.6} ref_max={ref_max:.3}");
        // Full block is now numerically near-exact (integer quantized
        // paths + f64 layernorm + voted shift; see the staged test).
        // Measured 1.5e-5 at ref_max 75.8; bound holds wide margin.
        assert!(
            max_abs < 0.001 * ref_max.max(1.0),
            "max_abs={max_abs} mean_abs={mean_abs} ref_max={ref_max}"
        );
    }

    /// Staged ORT parity: each branch output (0.5*FFN1, attention,
    /// conv, 0.5*FFN2) against ORT residual differences, plus pre-softmax
    /// scores against the graph's `Div` output (`layer0_stages`,
    /// `/layers.0/Add*` + `/layers.0/self_attn/Div`). Every branch runs on
    /// ORT's own intermediate as input, so error never compounds: a
    /// divergent sub-block (e.g. a flipped relative shift, or a new
    /// rounding path) shows up at its own stage. Bounds hold ~20x margin
    /// over measured (branches <= 4.3e-4, scores 2e-5).
    /// Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_layer0_stages_parity() {
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let (bw, table) = load_layer0(dir);
        let gbytes = std::fs::read("src/parakeet_burn/testdata/layer0_golden.safetensors").unwrap();
        let gst = safetensors::SafeTensors::deserialize(&gbytes).unwrap();
        let f32mat = |st: &safetensors::SafeTensors<'_>, name: &str, rows: usize, cols: usize| -> Vec<Vec<f32>> {
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
        let x = f32mat(&gst, "blk_in_T1024", 8, 1024);
        let sbytes = std::fs::read("src/parakeet_burn/testdata/layer0_stages.safetensors").unwrap();
        let sst = safetensors::SafeTensors::deserialize(&sbytes).unwrap();
        let add0 = f32mat(&sst, "Add", 8, 1024);
        let add1 = f32mat(&sst, "Add_1", 8, 1024);
        let add2 = f32mat(&sst, "Add_2", 8, 1024);
        let add3 = f32mat(&sst, "Add_3", 8, 1024);
        let sub = |a: &[Vec<f32>], b: &[Vec<f32>]| -> Vec<Vec<f32>> {
            a.iter()
                .zip(b.iter())
                .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(p, q)| p - q).collect())
                .collect()
        };
        let check = |tag: &str, got: &[Vec<f32>], want: &[Vec<f32>]| {
            let (mut mx, mut sm) = (0.0f32, 0.0f32);
            let mut rm = 0.0f32;
            for (a, b) in got.iter().zip(want.iter()) {
                for (p, q) in a.iter().zip(b.iter()) {
                    mx = mx.max((p - q).abs());
                    sm += (p - q).abs();
                    rm = rm.max(q.abs());
                }
            }
            eprintln!("{tag}: max_abs={mx:.6} mean_abs={:.6} ref_max={rm:.3}", sm / 8192.0);
            assert!(mx < 0.01, "{tag}: max_abs={mx} ref_max={rm}");
        };
        // Residual chain: add0 = x+f1, add1 = add0+a, add2 = add1+c,
        // add3 = add2+f2. Each branch input is layernormed above.
        let ctx = TensorCtx::scalar();
        let mut f1 = ffn(&layernorm_batch(&x, &bw.norm_ff1), &bw.ffn1_l1, &bw.ffn1_l2, &ctx);
        for r in f1.iter_mut() {
            for v in r.iter_mut() {
                *v *= 0.5;
            }
        }
        check("ffn1", &f1, &sub(&add0, &x));
        let win = relative_position_window(&table, 8);
        let a = relative_attention(&layernorm_batch(&add0, &bw.norm_att), &bw, &win, &ctx);
        check("attn", &a, &sub(&add1, &add0));
        let c = conv_block(&layernorm_batch(&add1, &bw.norm_conv), &bw, &ctx);
        check("conv", &c, &sub(&add2, &add1));
        let mut f2 = ffn(&layernorm_batch(&add2, &bw.norm_ff2), &bw.ffn2_l1, &bw.ffn2_l2, &ctx);
        for r in f2.iter_mut() {
            for v in r.iter_mut() {
                *v *= 0.5;
            }
        }
        check("ffn2", &f2, &sub(&add3, &add2));
        // Pre-softmax scores: ORT's Div is [H,T,T]; `relative_scores` returns
        // [T,H,T], transposed here for the comparison.
        let div_v = sst.tensor("self_attn_Div").unwrap();
        assert_eq!(div_v.shape(), &[8, 8, 8]);
        let divf: Vec<f32> = div_v
            .data()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let sc = relative_scores(&layernorm_batch(&add0, &bw.norm_att), &bw, &win, &ctx);
        let mut flat_htt = vec![0.0f32; 512];
        for i in 0..8 {
            for hh in 0..8 {
                for j in 0..8 {
                    flat_htt[(hh * 8 + i) * 8 + j] = sc[i][hh][j];
                }
            }
        }
        let (mut mx, mut sm) = (0.0f32, 0.0f32);
        let mut rm = 0.0f32;
        for (p, q) in flat_htt.iter().zip(divf.iter()) {
            mx = mx.max((p - q).abs());
            sm += (p - q).abs();
            rm = rm.max(q.abs());
        }
        eprintln!("scores: max_abs={mx:.6} mean_abs={:.6} ref_max={rm:.3}", sm / 512.0);
        assert!(mx < 0.001, "scores: max_abs={mx} ref_max={rm}");
    }

    /// Tensor-path layer-0 parity: same golden as `ort_layer0_parity`,
    /// but through `conformer_forward` with the tensor linears. Compared
    /// against ORT (same 0.1% bound — policy (b) allows re-baselining
    /// here if Flex summation order exceeds it while transcripts stay
    /// word-exact) and against the scalar twin (tight: f32 order only).
    /// Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_layer0_parity_tensor() {
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
        let win = relative_position_window(&table, 8);
        let tctx = TensorCtx {
            device: super::super::runtime::cpu_device(),
            enabled: true,
        };
        let y = conformer_forward(&x, &bw, &win, &tctx);
        let ys = conformer_forward(&x, &bw, &win, &TensorCtx::scalar());
        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f32);
        let (mut max_vs, mut ref_max) = (0.0f32, 0.0f32);
        for ((yt, ys), wr) in y.iter().zip(ys.iter()).zip(want.iter()) {
            for ((a, s), b) in yt.iter().zip(ys.iter()).zip(wr.iter()) {
                max_abs = max_abs.max((a - b).abs());
                sum_abs += (a - b).abs();
                max_vs = max_vs.max((a - s).abs());
                ref_max = ref_max.max(b.abs());
            }
        }
        eprintln!(
            "layer0 tensor: max_abs={max_abs:.6} mean_abs={:.6} vs_scalar={max_vs:.2e} ref_max={ref_max:.3}",
            sum_abs / (8.0 * 1024.0)
        );
        assert!(
            max_abs < 0.001 * ref_max.max(1.0),
            "max_abs={max_abs} ref_max={ref_max}"
        );
        assert!(max_vs < 1e-4 * ref_max.max(1.0), "vs_scalar={max_vs}");
    }

    }
