// Pre-encode subsampling: mel `[T][128]` -> `[T'][1024]`.
//
// Graph facts read off `encoder-model.int8.onnx` (verified with probe
// scripts, not assumed):
// - `audio_signal [B,128,T]` -> `Transpose[0,2,1]` -> `[B,T,128]` ->
//   `Unsqueeze` axis 1 -> `[B,1,T,128]` (convs run over time x freq).
// - `conv.0`: Conv 1 -> 256, k3 s2 p1 + bias, then Relu.
// - `conv.2`: depthwise (group 256) k3 s2 p1 + bias, NO activation.
// - `conv.3`: pointwise 1x1 + bias, then Relu.
// - `conv.5`: depthwise k3 s2 p1 + bias, NO activation.
// - `conv.6`: pointwise 1x1 + bias, then Relu.
// - `Transpose[0,2,1,3]` -> `[B,T',256,16]` -> `Reshape [B,T',4096]`
//   (freq folded into channels: 256 * 16) -> `MatMul [4096,1024]` +
//   `pre_encode.out.bias` -> `[B,T',1024]` -> `Transpose[0,2,1]`.
// - The lengths chain computes `floor((length + 1) / 2)` per stride-2
//   stage (Add -1, Div 2, Add 1, Floor); the Where mask is a no-op for
//   full-length offline utterances, so it is not reproduced here.
//
// Like `conformer_block.rs`, weights stay 8-bit in RAM (scalar scale/zp in the
// converted files) and dequantize inside the fp32 forward. The conv
// weights are tiny (~138 K params); the `pre_encode/out` projection
// reuses `Uint8Linear`.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use rayon::prelude::*;

use super::runtime::CpuBackend;
use super::manifest::EncoderManifest;
use super::quant_linear::Uint8Linear;
use super::tensor_ctx::TensorCtx;
use super::weights::RawTensor;

/// Exact staged output length: three stride-2 (k=3, p=1) stages, each
/// `floor((t + 1) / 2)`. This is NOT `t / 8` for small `t` (e.g. 50 ->
/// 25 -> 13 -> 7, not 6); `encoder::coarse_subsampled_frames` stays as the
/// coarse `encoder_stride = 8` estimate.
pub(crate) fn exact_subsampled_len(t: usize) -> usize {
    let mut n = t;
    for _ in 0..3 {
        n = (n + 1) / 2;
    }
    n
}

/// One 2D conv weight set (u8 + scalar scale/zp + f32 bias).
pub(crate) struct Conv2dW {
    pub(crate) weight_u8: Vec<u8>, // [out_c][in_c/group][k][k], row-major
    pub(crate) scale: f32,
    pub(crate) zero_point: f32,
    pub(crate) bias: Vec<f32>, // [out_c]
    pub(crate) out_c: usize,
    pub(crate) in_c: usize,
    pub(crate) group: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    /// Lazily-built dequantized `[out_c, in_c/group, k, k]` weight tensor
    /// (same rationale as `quant_linear::CachedT`: the pre-encode convs run once
    /// per forward, and per-call `from_data` is pure overhead).
    pub(crate) cache: std::sync::OnceLock<burn::tensor::Tensor<CpuBackend, 4>>,
}

impl Conv2dW {
    fn dequant(&self) -> Vec<f32> {
        self.weight_u8
            .iter()
            .map(|&q| (q as f32 - self.zero_point) * self.scale)
            .collect()
    }

    /// Dequantized `[out_c, in_c/group, k, k]` weight tensor, built once
    /// (bit-identical to the old per-call dequant: same per-element
    /// arithmetic, only built once instead of once per forward).
    fn cached(&self, device: &super::runtime::CpuDevice) -> &burn::tensor::Tensor<CpuBackend, 4> {
        self.cache.get_or_init(|| {
            burn::tensor::Tensor::from_data(
                burn::tensor::TensorData::new(
                    self.dequant(),
                    [self.out_c, self.in_c / self.group, self.kernel, self.kernel],
                ),
                device,
            )
        })
    }

    /// Build the cached weight tensor now (load-time warming).
    fn prime_cache(&self, device: &super::runtime::CpuDevice) {
        let _ = self.cached(device);
    }
}

/// `[C][H][W]` in, `[C'][H'][W']` out. `H' = (H + 2*pad - k)/s + 1`.
/// Mirrors `ConvInteger -> Mul(scales)`: the input is dynamically
/// quantized per-tensor (see `quant` docs) and the MAC runs in integer
/// arithmetic; the weight scale applies after. Bias lands later via
/// `add_bias_relu`, matching the graph's Add order.
fn conv2d_forward(w: &Conv2dW, x: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    use super::quant_linear::dynamic_quantize_u8;
    assert_eq!(x.len(), w.in_c);
    let h = x[0].len();
    let wd = x[0][0].len();
    assert!(x.iter().all(|c| c.len() == h && c.iter().all(|r| r.len() == wd)));
    let oh = (h + 2 * w.pad - w.kernel) / w.stride + 1;
    let ow = (wd + 2 * w.pad - w.kernel) / w.stride + 1;
    let flat: Vec<f32> = x.iter().flat_map(|c| c.iter().flat_map(|r| r.iter().copied())).collect();
    let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
    let xq3: Vec<&[u8]> = xq.chunks(h * wd).collect();
    let zpw = w.zero_point.round() as i64;
    let icg = w.in_c / w.group; // input channels per group
    let k2 = w.kernel * w.kernel;
    // weight base offset for (oc, ic): ((oc * icg) + ic) * k2
    let mut y = vec![vec![vec![0.0f32; ow]; oh]; w.out_c];
    for oc in 0..w.out_c {
        let g = oc / (w.out_c / w.group);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = 0i64;
                for ic in 0..icg {
                    let c = g * icg + ic;
                    let base = (oc * icg + ic) * k2;
                    for ky in 0..w.kernel {
                        let iy = oy * w.stride + ky;
                        if iy < w.pad || iy >= h + w.pad {
                            continue;
                        }
                        for kx in 0..w.kernel {
                            let ix = ox * w.stride + kx;
                            if ix < w.pad || ix >= wd + w.pad {
                                continue;
                            }
                            acc += (xq3[c][(iy - w.pad) * wd + (ix - w.pad)] as i64 - zpx as i64)
                                * (w.weight_u8[base + ky * w.kernel + kx] as i64 - zpw);
                        }
                    }
                }
                y[oc][oy][ox] = acc as f32 * (sx * w.scale);
            }
        }
    }
    y
}

/// Exact parallel twin of `conv2d_forward`: the same
/// `dynamic_quantize_u8` call (identical q/s/zpx), the same padded
/// layout, and the same per-output accumulation order
/// (`ic -> ky -> kx`, serial) — rayon only spreads independent output
/// channels over threads, so outputs match `forward` bit-for-bit.
/// Replaces `conv2d_t` (Flex conv2d + dequant/re-layout copies per call)
/// on the tensor path for k>1 or strided/grouped convs.
fn conv2d_par(w: &Conv2dW, x: &[Vec<Vec<f32>>], relu: bool) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(x.len(), w.in_c);
    let h = x[0].len();
    let wd = x[0][0].len();
    assert!(x.iter().all(|c| c.len() == h && c.iter().all(|r| r.len() == wd)));
    let oh = (h + 2 * w.pad - w.kernel) / w.stride + 1;
    let ow = (wd + 2 * w.pad - w.kernel) / w.stride + 1;
    // Planar quantize: same window/q values as flatten-then-quantize, but
    // no intermediate f32 copy (this input reaches 2M elements per conv).
    let (xq, sx, zpx) = super::int_gemm::quantize_u8_3d(x);
    let xq3: Vec<&[u8]> = xq.chunks(h * wd).collect();
    let zpw = w.zero_point.round() as i64;
    let icg = w.in_c / w.group;
    let k2 = w.kernel * w.kernel;
    let sxs = sx * w.scale;
    let (wu, kk, st, pd) = (&w.weight_u8, w.kernel, w.stride, w.pad);
    let bias = &w.bias;
    (0..w.out_c)
        .into_par_iter()
        .map(|oc| {
            let g = oc / (w.out_c / w.group);
            let mut omap = vec![vec![0.0f32; ow]; oh];
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0i64;
                    for ic in 0..icg {
                        let c = g * icg + ic;
                        let base = (oc * icg + ic) * k2;
                        for ky in 0..kk {
                            let iy = oy * st + ky;
                            if iy < pd || iy >= h + pd {
                                continue;
                            }
                            for kx in 0..kk {
                                let ix = ox * st + kx;
                                if ix < pd || ix >= wd + pd {
                                    continue;
                                }
                                acc += (xq3[c][(iy - pd) * wd + (ix - pd)] as i64 - zpx as i64)
                                    * (wu[base + ky * kk + kx] as i64 - zpw);
                            }
                        }
                    }
                    // Bias + ReLU fused into the output write (same
                    // arithmetic `add_bias_relu` applies afterwards).
                    let mut v = acc as f32 * sxs + bias[oc];
                    if relu && v < 0.0 {
                        v = 0.0;
                    }
                    omap[oy][ox] = v;
                }
            }
            omap
        })
        .collect()
}

/// Exact 1x1-conv fast path (conv3/conv6: 256 -> 256, k1 s1, dense):
/// a 1x1 stride-1 conv IS a per-position matmul, so this runs it as one
/// blocked integer GEMM (`int_gemm::int_gemm_epi`) over H*W rows instead of
/// a direct convolution. Same per-tensor activation quantization window as
/// `conv2d_forward` (whole input tensor), same integers, same scale-up;
/// the graph's bias + ReLU ride the GEMM's epilogue (same arithmetic
/// `add_bias_relu` applies afterwards) — bit-identical to both twins.
/// Weights widen per call (65k u8, ~10 us); no cache needed.
///
/// VNNI (`ctx.use_vnni()`): same gather, u8 activations through
/// `vnni_gemm_epi` with per-call requantized s8 weights (65k params —
/// requant ~10 us, amortized over the single call per forward).
/// Approximate (requant noise); `STTD_BURN_VNNI=0` keeps the i16 path.
fn conv1x1_gemm(
    w: &Conv2dW,
    x: &[Vec<Vec<f32>>],
    relu: bool,
    ctx: &TensorCtx,
) -> Vec<Vec<Vec<f32>>> {
    use super::int_gemm::{int_gemm_epi, quantize_rows_i16, widen_weights, Epilogue};
    assert_eq!(x.len(), w.in_c);
    let h = x[0].len();
    let wd = x[0][0].len();
    assert!(x.iter().all(|c| c.len() == h && c.iter().all(|r| r.len() == wd)));
    debug_assert!(w.kernel == 1 && w.stride == 1 && w.pad == 0 && w.group == 1);
    // Rows are positions (h*wd of them), columns are channels: gather
    // straight into row-major (no separate flat copy — the window spans
    // the same values either way).
    let t = h * wd;
    let mut rows = vec![0.0f32; t * w.in_c];
    for (c, cmap) in x.iter().enumerate() {
        for (i, row) in cmap.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                rows[(i * wd + j) * w.in_c + c] = v;
            }
        }
    }
    let epi = if relu {
        Epilogue::Relu
    } else {
        Epilogue::None
    };
    let mut out = vec![0.0f32; t * w.out_c];
    if ctx.use_vnni() {
        use super::int_gemm::{requant_u8_to_s8, vnni_gemm_epi};
        use super::quant_linear::dynamic_quantize_u8;
        let (q, sx, zpx) = dynamic_quantize_u8(&rows);
        let vw = requant_u8_to_s8(&w.weight_u8, w.out_c, w.in_c, &[w.scale], &[w.zero_point]);
        vnni_gemm_epi(
            &q,
            t,
            w.in_c,
            &vw.w_s8,
            &vw.rowsum,
            w.out_c,
            &vw.s_prime,
            Some(&w.bias),
            sx,
            zpx as i32,
            &mut out,
            epi,
        );
    } else {
        let (a, sx, _) = quantize_rows_i16(&rows);
        // k=1 store is already `[out][in]` row-major (see `take_conv`).
        let bw = widen_weights(&w.weight_u8, w.out_c, w.in_c, &[w.zero_point]);
        int_gemm_epi(
            &a,
            t,
            w.in_c,
            &bw,
            w.out_c,
            &[w.scale],
            Some(&w.bias),
            sx,
            &mut out,
            epi,
        );
    }
    // Rows back to `[out_c][h][wd]` maps.
    let mut y = vec![vec![vec![0.0f32; wd]; h]; w.out_c];
    for (i, orow) in out.chunks(w.out_c).enumerate() {
        let (r, c_) = (i / wd, i % wd);
        for (oc, &v) in orow.iter().enumerate() {
            y[oc][r][c_] = v;
        }
    }
    y
}

/// Dispatch: exact parallel integer kernels on the tensor path
/// (`conv1x1_gemm` for dense 1x1, `conv2d_par` otherwise) with bias +
/// optional ReLU fused into the output write, exact serial integer
/// `conv2d_forward` + `add_bias_relu` on the scalar path. (`conv2d_t`,
/// the Flex twin, is kept for its parity test only.)
fn conv_dispatch(
    w: &Conv2dW,
    x: &[Vec<Vec<f32>>],
    ctx: &TensorCtx,
    relu: bool,
) -> Vec<Vec<Vec<f32>>> {
    if ctx.enabled {
        if w.kernel == 1 && w.stride == 1 && w.pad == 0 && w.group == 1 {
            conv1x1_gemm(w, x, relu, ctx)
        } else {
            // Direct strided/grouped convs stay on the exact integer path
            // (u8 activations + i64 MACs — already 1 B/param, so VNNI would
            // not halve traffic there); only the GEMM-backed 1x1s and all
            // `Uint8Linear`s requantize under option (b).
            conv2d_par(w, x, relu)
        }
    } else {
        let mut y = conv2d_forward(w, x);
        add_bias_relu(&mut y, &w.bias, relu);
        y
    }
}

/// Tensor twin of `conv2d_forward`: identical per-tensor activation
/// quantization, dequantized to f32, one Flex conv2d against the cached
/// dequantized weights. No bias here: the caller runs the shared
/// `add_bias_relu` tail exactly as the scalar path (matches the
/// graph's Add order and keeps one bias site). Residual is f32
/// summation order only. Pads are symmetric throughout the graph, so
/// plain `ConvOptions` suffices (no `PaddedConvOptions`).
fn conv2d_t(
    w: &Conv2dW,
    x: &[Vec<Vec<f32>>],
    device: &super::runtime::CpuDevice,
) -> Vec<Vec<Vec<f32>>> {
    use burn::tensor::{module::conv2d, ops::ConvOptions, TensorData};
    assert_eq!(x.len(), w.in_c);
    let h = x[0].len();
    let wd = x[0][0].len();
    assert!(x.iter().all(|c| c.len() == h && c.iter().all(|r| r.len() == wd)));
    let flat: Vec<f32> = x.iter().flat_map(|c| c.iter().flat_map(|r| r.iter().copied())).collect();
    let (xq, sx, zpx) = super::quant_linear::dynamic_quantize_u8(&flat);
    let xd: Vec<f32> = xq.iter().map(|&q| (q as f32 - zpx as f32) * sx).collect();
    let xt: burn::tensor::Tensor<CpuBackend, 4> = burn::tensor::Tensor::from_data(
        TensorData::new(xd, [1, w.in_c, h, wd]),
        device,
    );
    // Dequantized weights, built once (see `Conv2dW::cached`).
    let wt = w.cached(device);
    let y = conv2d(
        xt,
        wt.clone(),
        None,
        ConvOptions::new([w.stride, w.stride], [w.pad, w.pad], [1, 1], w.group),
    );
    let dims = y.dims();
    y.into_data()
        .to_vec::<f32>()
        .expect("conv2d_t: to_vec")
        .chunks(dims[2] * dims[3])
        .map(|c| c.chunks(dims[3]).map(|r| r.to_vec()).collect())
        .collect()
}

fn add_bias_relu(y: &mut [Vec<Vec<f32>>], bias: &[f32], relu: bool) {
    assert_eq!(y.len(), bias.len());
    for (map, b) in y.iter_mut().zip(bias.iter()) {
        for row in map.iter_mut() {
            for v in row.iter_mut() {
                *v += *b;
                if relu && *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

/// Full pre-encode weights: 5 convs + the 4096 -> 1024 projection.
pub(crate) struct PreEncodeWeights {
    pub(crate) conv0: Conv2dW,
    pub(crate) conv2: Conv2dW,
    pub(crate) conv3: Conv2dW,
    pub(crate) conv5: Conv2dW,
    pub(crate) conv6: Conv2dW,
    pub(crate) out: Uint8Linear,
}

fn take_conv(
    tensors: &HashMap<String, RawTensor>,
    manifest: &EncoderManifest,
    role: &str,
    out_c: usize,
    in_c: usize,
    group: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> Result<Conv2dW> {
    let entry = manifest
        .role(role)
        .with_context(|| format!("manifest lacks {role}"))?;
    if entry.kind != "conv" {
        anyhow::bail!("{role}: expected kind conv, got {}", entry.kind);
    }
    if entry.group != Some(group) {
        anyhow::bail!("{role}: expected group {group}, got {:?}", entry.group);
    }
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
    let icg = in_c / group;
    if shape.as_slice() != [out_c, icg, kernel, kernel] {
        anyhow::bail!("{role}: shape {shape:?} != [{out_c}, {icg}, {kernel}, {kernel}]");
    }
    if scale.len() != 1 || zp.len() != 1 {
        anyhow::bail!("{role}: expected scalar scale/zp, got {} / {}", scale.len(), zp.len());
    }
    // Bias follows the weight-base convention (`….weight` -> `….bias`),
    // matching the graph's `pre_encode.conv.N.bias` initializers.
    let bias_name = entry
        .weight
        .strip_suffix(".weight")
        .map(|b| format!("{b}.bias"))
        .with_context(|| format!("{role}: weight name has no .weight suffix"))?;
    let (_, bias) = match tensors.get(&bias_name) {
        Some(RawTensor::F32 { data, .. }) => ((), data.clone()),
        Some(other) => anyhow::bail!("{bias_name}: expected F32, got {}", other.kind()),
        None => anyhow::bail!("{bias_name}: missing tensor"),
    };
    if bias.len() != out_c {
        anyhow::bail!("{bias_name}: len {} != {out_c}", bias.len());
    }
    Ok(Conv2dW {
        weight_u8: data.clone(),
        scale: scale[0],
        zero_point: zp[0],
        bias,
        out_c,
        in_c,
        group,
        kernel,
        stride,
        pad,
        cache: std::sync::OnceLock::new(),
    })
}

impl PreEncodeWeights {
    /// Load all pre-encode weights. Shapes/groups/acts are asserted here
    /// so a corrupt file fails at load, never mid-utterance.
    pub(crate) fn load(
        tensors: &HashMap<String, RawTensor>,
        manifest: &EncoderManifest,
    ) -> Result<Self> {
        let conv0 = take_conv(tensors, manifest, "pre_encode/conv/conv.0", 256, 1, 1, 3, 2, 1)?;
        let conv2 = take_conv(tensors, manifest, "pre_encode/conv/conv.2", 256, 256, 256, 3, 2, 1)?;
        let conv3 = take_conv(tensors, manifest, "pre_encode/conv/conv.3", 256, 256, 1, 1, 1, 0)?;
        let conv5 = take_conv(tensors, manifest, "pre_encode/conv/conv.5", 256, 256, 256, 3, 2, 1)?;
        let conv6 = take_conv(tensors, manifest, "pre_encode/conv/conv.6", 256, 256, 1, 1, 1, 0)?;

        let out_entry = manifest
            .role("pre_encode/out")
            .context("manifest lacks pre_encode/out")?;
        if out_entry.kind != "matmul" {
            anyhow::bail!("pre_encode/out: expected kind matmul, got {}", out_entry.kind);
        }
        if out_entry.weight_shape != vec![4096, 1024] {
            anyhow::bail!(
                "pre_encode/out: shape {:?} != [4096, 1024]",
                out_entry.weight_shape
            );
        }
        let out_raw = tensors.get(&out_entry.weight).with_context(|| {
            format!("pre_encode/out: missing weight {}", out_entry.weight)
        })?;
        let (o_shape, o_data, o_scale, o_zp) = match out_raw {
            RawTensor::U8 {
                shape,
                data,
                scale,
                zp,
            } => (shape, data, scale, zp),
            other => anyhow::bail!("pre_encode/out: expected U8, got {}", other.kind()),
        };
        if o_shape.as_slice() != [4096, 1024] {
            anyhow::bail!("pre_encode/out: stored shape {o_shape:?} != [4096, 1024]");
        }
        // Graph name is fixed (`pre_encode.out.bias` on the out Add), not
        // derived from the anonymous `onnx::MatMul_*` weight base.
        let (_, out_bias) = match tensors.get("pre_encode.out.bias") {
            Some(RawTensor::F32 { data, .. }) => ((), data.clone()),
            Some(other) => anyhow::bail!("pre_encode.out.bias: expected F32, got {}", other.kind()),
            None => anyhow::bail!("pre_encode.out.bias: missing tensor"),
        };
        if out_bias.len() != 1024 {
            anyhow::bail!("pre_encode.out.bias: len {} != 1024", out_bias.len());
        }
        let bcast = |v: &[f32]| {
            if v.len() == 1 {
                vec![v[0]; 1024]
            } else {
                v.to_vec()
            }
        };
        // ONNX MatMul weight (K, N) = (in, out): transpose to the
        // `[out][in]` store `forward` reads (see `Uint8Linear` docs).
        if o_data.len() != 4096 * 1024 {
            anyhow::bail!("pre_encode/out: weight len {} != 4096*1024", o_data.len());
        }
        let out = Uint8Linear {
            weight_u8: super::quant_linear::transpose_u8(o_data, 4096, 1024),
            scale: bcast(o_scale),
            zero_point: bcast(o_zp),
            bias: Some(out_bias),
            in_dim: 4096,
            out_dim: 1024,
            cache: std::sync::OnceLock::new(),
            w_i16: std::sync::OnceLock::new(),
            w_vnni: std::sync::OnceLock::new(),
        };

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    /// Build every cached tensor now (load-time warming): the 5 convs
    /// plus the out-projection.
    pub(crate) fn prime_cache(&self, device: &super::runtime::CpuDevice) {
        for w in [&self.conv0, &self.conv2, &self.conv3, &self.conv5, &self.conv6] {
            w.prime_cache(device);
        }
        self.out.prime_cache(device);
    }

    /// Mel `[T][128]` -> `[T'][1024]`. Empty in, empty out.
    pub(crate) fn forward(&self, mel: &[Vec<f32>], ctx: &TensorCtx) -> Result<Vec<Vec<f32>>> {
        if mel.is_empty() {
            return Ok(Vec::new());
        }
        if !mel.iter().all(|r| r.len() == 128) {
            anyhow::bail!("pre_encode: mel width {} != 128", mel[0].len());
        }
        // Unsqueeze channel: [1][T][128].
        let mut x = vec![mel.to_vec()];
        for (w, relu) in [
            (&self.conv0, true),
            (&self.conv2, false),
            (&self.conv3, true),
            (&self.conv5, false),
            (&self.conv6, true),
        ] {
            // Convs follow `ctx`: exact parallel integer kernels with the
            // bias (+ ReLU) fused into the output write on the tensor
            // path, exact serial integer + `add_bias_relu` on the scalar
            // path (same arithmetic, same Add order).
            x = conv_dispatch(w, &x, ctx, relu);
        }
        // x is [256][Ht][Wf]; Transpose[0,2,1,3] + flatten rows.
        let ht = x[0].len();
        let wf = x[0][0].len();
        if ht != exact_subsampled_len(mel.len()) {
            anyhow::bail!(
                "pre_encode: T={} gave Ht={ht}, expected {}",
                mel.len(),
                exact_subsampled_len(mel.len())
            );
        }
        let mut flat = vec![vec![0.0f32; 256 * wf]; ht];
        for (t, row) in flat.iter_mut().enumerate() {
            for c in 0..256 {
                row[c * wf..(c + 1) * wf].copy_from_slice(&x[c][t]);
            }
        }
        if flat[0].len() != self.out.in_dim {
            anyhow::bail!(
                "pre_encode: flattened {} != out in_dim {}",
                flat[0].len(),
                self.out.in_dim
            );
        }
        // Batched: the out-projection's activation quantization spans
        // the whole `[T',4096]` input like ORT's (see `quant` docs).
        // `batch_dispatch_auto` selects scalar / i16 / VNNI per `ctx`;
        // the five convs run their own dispatch (`conv1x1_gemm` follows
        // VNNI via GEMM, direct convs stay exact integer).
        Ok(self.out.batch_dispatch_auto(&flat, ctx))
    }
}

/// Exact tensor names `weights::load_selected` must fetch for `load`.
pub(crate) fn required_tensors(manifest: &EncoderManifest) -> Result<HashSet<String>> {
    let mut want = HashSet::new();
    for role in [
        "pre_encode/conv/conv.0",
        "pre_encode/conv/conv.2",
        "pre_encode/conv/conv.3",
        "pre_encode/conv/conv.5",
        "pre_encode/conv/conv.6",
        "pre_encode/out",
    ] {
        let e = manifest
            .role(role)
            .with_context(|| format!("manifest lacks {role}"))?;
        want.insert(e.weight.clone());
        want.insert(format!("{}.scale", e.weight));
        want.insert(format!("{}.zero_point", e.weight));
        if let Some(base) = e.weight.strip_suffix(".weight") {
            // Conv biases follow the weight base; harmless if absent for
            // `out` (its bias has a fixed graph name, added below).
            want.insert(format!("{base}.bias"));
        }
    }
    want.insert("pre_encode.out.bias".to_string());
    Ok(want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet_burn::quant_linear::Uint8Linear;

    #[test]
    fn staged_length_is_exact() {
        assert_eq!(exact_subsampled_len(0), 0);
        assert_eq!(exact_subsampled_len(1), 1); // (1+1)/2 x3
        assert_eq!(exact_subsampled_len(50), 7); // 25 -> 13 -> 7, NOT 50/8=6
        assert_eq!(exact_subsampled_len(64), 8);
        assert_eq!(exact_subsampled_len(101), 13); // 51 -> 26 -> 13
        assert_eq!(exact_subsampled_len(800), 100);
    }

    #[test]
    fn conv_stride_pad_geometry_is_exact() {
        // 1ch 4x4 constant 2.0, all-ones k3 s2 p1, no bias. Constant input
        // quantizes exactly, so the hand-computed window sums hold:
        // window [0,0] covers 4 real cells, [0,1]/[1,0] cover 6, [1,1] 9.
        let w = Conv2dW {
            weight_u8: vec![1; 9],
            scale: 1.0,
            zero_point: 0.0,
            bias: vec![0.0],
            out_c: 1,
            in_c: 1,
            group: 1,
            kernel: 3,
            stride: 2,
            pad: 1,
            cache: std::sync::OnceLock::new(),
        };
        let x = vec![vec![vec![2.0; 4]; 4]];
        let y = conv2d_forward(&w, &x);
        assert_eq!(y.len(), 1);
        assert_eq!(y[0].len(), 2);
        // Approximate: the integer path scales by max/255, which is not
        // exactly representable in f32 (1-ULP error).
        for (got, want) in y[0][0].iter().zip([8.0, 12.0]) {
            assert!((got - want).abs() < 1e-4, "y={y:?}");
        }
        for (got, want) in y[0][1].iter().zip([12.0, 18.0]) {
            assert!((got - want).abs() < 1e-4, "y={y:?}");
        }
    }

    #[test]
    fn depthwise_routes_channels() {
        // group=2, k1 s1: out[ch] = w_ch * in[ch]. Zero-centered values
        // quantize without saturation (see `quant` docs); the 1-LSB
        // rounding rides through the small weights.
        let w = Conv2dW {
            weight_u8: vec![3, 5],
            scale: 1.0,
            zero_point: 0.0,
            bias: vec![0.0, 0.0],
            out_c: 2,
            in_c: 2,
            group: 2,
            kernel: 1,
            stride: 1,
            pad: 0,
            cache: std::sync::OnceLock::new(),
        };
        let x = vec![vec![vec![-2.0]], vec![vec![4.0]]];
        let y = conv2d_forward(&w, &x);
        assert!((y[0][0][0] + 6.0).abs() < 0.1, "y={y:?}");
        assert!((y[1][0][0] - 20.0).abs() < 0.1, "y={y:?}");
    }

    #[test]
    fn conv_tensor_matches_scalar() {
        use super::super::runtime::cpu_device;
        // Dense k3s2p1 + depthwise k3s2p1 over small maps with mixed
        // signs; tensor twin must match the integer chain to f32 order.
        let dense = Conv2dW {
            weight_u8: vec![7, 250, 3, 40, 128, 200, 1, 9, 77, 5, 5, 5, 0, 255, 128, 64, 32, 16],
            scale: 0.02,
            zero_point: 121.0,
            bias: vec![0.0],
            out_c: 2,
            in_c: 1,
            group: 1,
            kernel: 3,
            stride: 2,
            pad: 1,
            cache: std::sync::OnceLock::new(),
        };
        let x = vec![vec![
            vec![1.0, -2.0, 3.0, -0.5],
            vec![0.5, 4.0, -1.0, 2.0],
            vec![-3.0, 0.25, 1.5, -2.5],
            vec![2.0, -0.75, 0.0, 1.25],
        ]];
        let dev = cpu_device();
        let s = conv2d_forward(&dense, &x);
        let t = conv2d_t(&dense, &x, &dev);
        assert_eq!((s.len(), t.len()), (2, 2));
        let (mut mx, mut rm) = (0.0f32, 0.0f32);
        for (a, b) in s.iter().zip(t.iter()) {
            for (r, q) in a.iter().zip(b.iter()) {
                for (p, v) in r.iter().zip(q.iter()) {
                    mx = mx.max((p - v).abs());
                    rm = rm.max(v.abs());
                }
            }
        }
        eprintln!("conv tensor-vs-scalar: max_abs={mx:.2e} ref_max={rm:.3}");
        assert!(mx < 1e-4 * rm.max(1.0), "mx={mx} rm={rm}");
    }

    #[test]
    fn bias_relu_adds_then_clamps() {
        let mut y = vec![vec![vec![-1.0, 2.0]]];
        add_bias_relu(&mut y, &[1.0], true);
        assert_eq!(y[0][0], vec![0.0, 3.0]);
        let mut z = vec![vec![vec![-1.0]]];
        add_bias_relu(&mut z, &[0.0], false);
        assert_eq!(z[0][0], vec![-1.0]);
    }

    fn tiny_weights() -> PreEncodeWeights {
        // All identity convs (k1 s1, w=1, b=0), 1 channel, mel width 2;
        // out sums the 2 flattened values + 10 bias.
        let id = |bias: Vec<f32>| Conv2dW {
            weight_u8: vec![1],
            scale: 1.0,
            zero_point: 0.0,
            bias,
            out_c: 1,
            in_c: 1,
            group: 1,
            kernel: 1,
            stride: 1,
            pad: 0,
            cache: std::sync::OnceLock::new(),
        };
        PreEncodeWeights {
            conv0: id(vec![0.0]),
            conv2: id(vec![0.0]),
            conv3: id(vec![0.0]),
            conv5: id(vec![0.0]),
            conv6: id(vec![0.0]),
            out: Uint8Linear {
                weight_u8: vec![1, 1],
                scale: vec![1.0],
                zero_point: vec![0.0],
                bias: Some(vec![10.0]),
                in_dim: 2,
                out_dim: 1,
                cache: std::sync::OnceLock::new(),
                w_i16: std::sync::OnceLock::new(),
            w_vnni: std::sync::OnceLock::new(),
            },
        }
    }

    #[test]
    fn forward_plumbing_preserves_order() {
        let w = tiny_weights();
        // Constant mel quantizes exactly, so the identity stages preserve
        // it bit-for-bit; the width-128 check is bypassed by staging
        // manually (see `forward_rejects_bad_width` for the real entry).
        let mel = vec![vec![3.0, 3.0]; 4];
        // forward() asserts width 128, so call stages manually here.
        let mut x = vec![mel.clone()];
        for (c, relu) in [
            (&w.conv0, true),
            (&w.conv2, false),
            (&w.conv3, true),
            (&w.conv5, false),
            (&w.conv6, true),
        ] {
            let mut y = conv2d_forward(c, &x);
            add_bias_relu(&mut y, &c.bias, relu);
            x = y;
        }
        assert_eq!(x, vec![mel]);
        // flatten [1][4][2] -> [4][2], out sums + bias.
        let flat: Vec<Vec<f32>> = (0..4).map(|t| x[0][t].clone()).collect();
        let y: Vec<Vec<f32>> = flat.iter().map(|r| w.out.forward(r)).collect();
        assert_eq!(y, vec![vec![16.0]; 4]);
    }

    #[test]
    fn forward_rejects_bad_width() {
        use super::super::tensor_ctx::TensorCtx;
        let w = tiny_weights();
        assert!(w.forward(&[vec![0.0; 127]], &TensorCtx::scalar()).is_err());
        assert!(w.forward(&[], &TensorCtx::scalar()).unwrap().is_empty());
    }

    fn conv_par_case(
        out_c: usize,
        in_c: usize,
        group: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        h: usize,
        wd: usize,
    ) {
        // Non-trivial taps/scales/biases, mixed signs; outputs must match
        // the serial integer path bit-for-bit.
        let icg = in_c / group;
        let w = Conv2dW {
            weight_u8: (0..out_c * icg * kernel * kernel)
                .map(|i| (i * 37 % 256) as u8)
                .collect(),
            scale: 0.02,
            zero_point: 121.0,
            bias: (0..out_c).map(|i| i as f32 * 0.125 - 1.0).collect(),
            out_c,
            in_c,
            group,
            kernel,
            stride,
            pad,
            cache: std::sync::OnceLock::new(),
        };
        let mut seed: u64 = 0xC0E | ((out_c * 7919 + h * 104729) as u64);
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 6.0
        };
        let x: Vec<Vec<Vec<f32>>> = (0..in_c)
            .map(|_| (0..h).map(|_| (0..wd).map(|_| rnd()).collect()).collect())
            .collect();
        for relu in [false, true] {
            // Fused bias (+ReLU) twins must equal the serial path plus the
            // production post-pass, bit-for-bit.
            let mut s = conv2d_forward(&w, &x);
            add_bias_relu(&mut s, &w.bias, relu);
            let p = conv2d_par(&w, &x, relu);
            assert_eq!(
                p, s,
                "out_c={out_c} group={group} k={kernel} s={stride} relu={relu}"
            );
            if kernel == 1 && stride == 1 && pad == 0 && group == 1 {
                let g = conv1x1_gemm(
                    &w,
                    &x,
                    relu,
                    &super::super::tensor_ctx::TensorCtx::scalar(),
                );
                assert_eq!(g, s, "1x1 out_c={out_c} relu={relu}");
            }
        }
    }

    #[test]
    fn conv_par_matches_scalar_exactly() {
        // Dense k3s2p1, depthwise k3s2p1, pointwise 1x1, and odd shapes —
        // the same geometries the encoder actually runs.
        conv_par_case(2, 1, 1, 3, 2, 1, 4, 4);
        conv_par_case(4, 4, 4, 3, 2, 1, 8, 6);
        conv_par_case(4, 4, 1, 1, 1, 0, 7, 5);
        conv_par_case(8, 8, 1, 1, 1, 0, 9, 9);
        conv_par_case(6, 3, 1, 3, 1, 1, 5, 7);
    }

    /// Loads real pre-encode weights. Ignored: needs converted weights on
    /// disk. `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn real_pre_encode_shape_finite_deterministic() {
        use super::super::manifest::load_manifest;
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest = load_manifest(&json).unwrap();
        let want = required_tensors(&manifest).unwrap();
        // 6 weights + 6 scales + 6 zps + 5 conv biases + out bias.
        assert_eq!(want.len(), 6 * 3 + 6);
        let tensors =
            load_selected(&dir.join("encoder-burn.safetensors"), &want).unwrap();
        let w = PreEncodeWeights::load(&tensors, &manifest).unwrap();
        // Deterministic pseudo-random mel, T=64.
        let mut seed: u64 = 0xABCDEF;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let mel: Vec<Vec<f32>> = (0..64).map(|_| (0..128).map(|_| rnd()).collect()).collect();
        let y1 = w.forward(&mel, &super::super::tensor_ctx::TensorCtx::scalar()).unwrap();
        assert_eq!(y1.len(), 8, "T=64 -> T'=8");
        assert!(y1.iter().all(|r| r.len() == 1024));
        assert!(y1.iter().all(|r| r.iter().all(|v| v.is_finite())));
        let y2 = w.forward(&mel, &super::super::tensor_ctx::TensorCtx::scalar()).unwrap();
        assert_eq!(y1, y2, "forward is deterministic");
        // Output spans a sane range (projection of normalized mel).
        let max: f32 = y1.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max > 0.1 && max < 1e4, "max={max}");
    }

    /// ORT parity: golden input/output produced by running the int8
    /// encoder ONNX in onnxruntime (CPU) with `audio_signal = mel^T`
    /// (seed 1234, T=64) and reading `/pre_encode/out/Add_output_0`
    /// (golden committed under `testdata/`).
    /// Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_golden_parity() {
        use super::super::manifest::load_manifest;
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let json = std::fs::read_to_string(dir.join("encoder-manifest.json")).unwrap();
        let manifest = load_manifest(&json).unwrap();
        let want = required_tensors(&manifest).unwrap();
        let tensors =
            load_selected(&dir.join("encoder-burn.safetensors"), &want).unwrap();
        let w = PreEncodeWeights::load(&tensors, &manifest).unwrap();
        let bytes =
            std::fs::read("src/parakeet_burn/testdata/pre_encode_golden.safetensors").unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let mel_v = st.tensor("mel_T128").unwrap();
        let ref_v = st.tensor("pre_ref").unwrap();
        assert_eq!(mel_v.shape(), &[64, 128]);
        assert_eq!(ref_v.shape(), &[8, 1024]);
        let f32row = |v: safetensors::tensor::TensorView<'_>| -> Vec<f32> {
            v.data()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let mel_f = f32row(mel_v);
        let ref_f = f32row(ref_v);
        let mel: Vec<Vec<f32>> = mel_f.chunks(128).map(|c| c.to_vec()).collect();
        let tctx = super::super::tensor_ctx::TensorCtx {
            device: super::super::runtime::cpu_device(),
            enabled: true,
            vnni: false,
        };
        for (tag, ctx) in [
            ("scalar", super::super::tensor_ctx::TensorCtx::scalar()),
            ("tensor", tctx),
        ] {
            let y = w.forward(&mel, &ctx).unwrap();
            assert_eq!((y.len(), y[0].len()), (8, 1024));
            let flat: Vec<f32> = y.iter().flat_map(|r| r.iter().copied()).collect();
            let max_abs = flat
                .iter()
                .zip(ref_f.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let ref_max = ref_f.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            eprintln!("pre_encode {tag}: max_abs={max_abs:.5} ref_max={ref_max:.3}");
            // Integer emulation of the graph's DynamicQuantizeLinear /
            // ConvInteger chain (see `quant` docs): residual is f32 rounding
            // only (measured < 5e-6 at ref_max 3921). Bound holds wide margin.
            assert!(
                max_abs < 0.001 * ref_max.max(1.0),
                "{tag}: max_abs={max_abs} ref_max={ref_max}"
            );
        }
    }
}
