// int8/uint8 weight containers for the Burn port.
//
// The exported ONNX uses *static* weight quantization (uint8 asymmetric
// for encoder MatMuls/Convs and the decoder embed/joint, int8 for the
// decoder LSTMs) plus *dynamic* activation quantization
// (`DynamicQuantizeLinear` + `MatMulInteger` + `ConvInteger`,
// `DynamicQuantizeLSTM`). This module stores weights as 8-bit + scale
// (+ zero-point) — 8-bit-now on disk/VRAM — and mirrors the graph's
// dynamic path in integer arithmetic (`dynamic_quantize_u8` +
// i64 MAC + one f32 scale-up), so the residual vs ORT is f32 rounding
// only (1-op probe: 1.5e-6). Swapping the inner loops for native Burn
// tensor ops later changes speed, not numerics.

/// One row-quantized int8 Linear: `y = ((w_i8 - zp) * scale) @ x + bias`.
#[derive(Debug, Clone)]
pub(crate) struct Int8Linear {
    /// `[out][in]` int8 weights.
    pub(crate) weight_i8: Vec<i8>,
    /// Per-output-row scale (ONNX `MatMulInteger` scale input).
    pub(crate) scale: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl Int8Linear {
    pub(crate) fn new(
        weight_i8: Vec<i8>,
        scale: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Self {
        assert_eq!(weight_i8.len(), in_dim * out_dim);
        assert_eq!(scale.len(), out_dim);
        if let Some(b) = &bias {
            assert_eq!(b.len(), out_dim);
        }
        Self {
            weight_i8,
            scale,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Dequantize one row to f32 (used by the Burn matmul until native
    /// int8 kernels land per-op).
    pub(crate) fn dequant_row(&self, row: usize) -> Vec<f32> {
        let s = self.scale[row];
        self.weight_i8[row * self.in_dim..(row + 1) * self.in_dim]
            .iter()
            .map(|&q| q as f32 * s)
            .collect()
    }

    /// `y = W x + b`, `x: [in]`, returns `[out]`.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim);
        let mut y = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let s = self.scale[o];
            let base = o * self.in_dim;
            let mut acc = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                acc += self.weight_i8[base + i] as f32 * s * xi;
            }
            if let Some(b) = &self.bias {
                acc += b[o];
            }
            y[o] = acc;
        }
        y
    }

    /// Quantize an f32 row to int8 with a single scale (converter helper).
    pub(crate) fn quantize_row(row_f32: &[f32]) -> (Vec<i8>, f32) {
        let amax = row_f32.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            return (vec![0; row_f32.len()], 1.0);
        }
        let scale = amax / 127.0;
        let q = row_f32.iter().map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8).collect();
        (q, scale)
    }
}

/// Transpose row-major `[rows][cols]` u8 -> `[cols][rows]`. Applied at
/// load to ONNX `MatMul` `(K, N)` weights (see `Uint8Linear` docs).
pub(crate) fn transpose_u8(d: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(d.len(), rows * cols);
    let mut t = vec![0u8; d.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = d[r * cols + c];
        }
    }
    t
}

/// Dynamically quantize activations the way ORT's `DynamicQuantizeLinear`
/// does (pinned by 1-node probe models: asymmetric uint8, per-tensor over
/// the whole input, with the range forced to include zero —
/// `lo = min(0, min)`, `hi = max(0, max)`, `s = (hi - lo) / 255`,
/// `zp = clamp(round(-lo / s))`, `q = clamp(round(x / s) + zp)`).
/// All-zero input yields `s = 1.0`, `zp = 0` (also probed).
/// Returns `(quantized, scale, zero_point)`.
///
/// One deliberate unknown: the exact .5 rounding mode (Rust `round`
/// rounds half away from zero; no probe produced an exact .5 to tell).
/// At most 1 LSB on a negligible share of values.
pub(crate) fn dynamic_quantize_u8(xs: &[f32]) -> (Vec<u8>, f32, u8) {
    assert!(!xs.is_empty());
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in xs {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let (lo, hi) = (mn.min(0.0), mx.max(0.0));
    let range = hi - lo;
    if !(range > 0.0) {
        return (vec![0; xs.len()], 1.0, 0);
    }
    let s = range / 255.0;
    let zp = ((-lo / s).round() as i32).clamp(0, 255) as u8;
    let q = xs
        .iter()
        .map(|&v| ((v / s).round() as i32 + zp as i32).clamp(0, 255) as u8)
        .collect();
    (q, s, zp)
}

/// One asymmetrically-quantized uint8 Linear (encoder MatMuls/Convs,
/// decoder embed + joint). Mirrors the graph's
/// `DynamicQuantizeLinear -> MatMulInteger -> Mul(scales) -> Add(bias)`
/// chain in integer arithmetic: `y = Σ(Xq-zpx)(Wq-zpw) * sx*sw + b`,
/// zp scalar-or-per-row.
////// Storage is ALWAYS `[out][in]` row-major. ONNX `MatMul` weights are
/// `(K, N) = (in, out)` — transpose them with `transpose_u8` at load
/// (a no-op only for square shapes, which is why square-only tests
/// never caught a missing transpose). ONNX 1x1 `Conv` weights are
/// `[out][in][1]` and already match.
#[derive(Debug, Clone)]
pub(crate) struct Uint8Linear {
    pub(crate) weight_u8: Vec<u8>,
    pub(crate) scale: Vec<f32>,
    pub(crate) zero_point: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl Uint8Linear {
    pub(crate) fn dequant_row(&self, row: usize) -> Vec<f32> {
        let s = self.scale[row.min(self.scale.len() - 1)];
        let z = self.zero_point[row.min(self.zero_point.len() - 1)];
        self.weight_u8[row * self.in_dim..(row + 1) * self.in_dim]
            .iter()
            .map(|&q| (q as f32 - z) * s)
            .collect()
    }

    /// Integer MAC of one quantized activation row against weight row
    /// `o`: `Σ (xq - zpx) * (wq - zpw)` in i64 (no overflow: at most
    /// 4096 terms of ±127×±255).
    fn mac_row(&self, xq: &[u8], zpx: u8, o: usize) -> i64 {
        let zpw = self.zero_point[o.min(self.zero_point.len() - 1)].round() as i64;
        let base = o * self.in_dim;
        let mut acc = 0i64;
        for (k, &q) in xq.iter().enumerate() {
            acc += (q as i64 - zpx as i64) * (self.weight_u8[base + k] as i64 - zpw);
        }
        acc
    }

    /// `y = W x + b` with `x` dynamically quantized per-tensor over the
    /// single row (matches the graph for 1-row inputs such as the joint's
    /// `[1,1,K]` frames). Multi-row callers must use `forward_batch` so
    /// the quantization spans the whole input tensor like ORT's does.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim);
        let (xq, sx, zpx) = dynamic_quantize_u8(x);
        self.mac_all(&xq, sx, zpx)
    }

    /// Batched `forward`: the activation quantization spans every row
    /// (per-tensor, as ORT quantizes the whole `[T,K]` input at once).
    pub(crate) fn forward_batch(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        assert!(xs.iter().all(|r| r.len() == self.in_dim));
        let flat: Vec<f32> = xs.iter().flat_map(|r| r.iter().copied()).collect();
        if flat.is_empty() {
            return Vec::new();
        }
        let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
        xq.chunks(self.in_dim)
            .map(|row| self.mac_all(row, sx, zpx))
            .collect()
    }

    /// Scale one quantized row's MACs into fp32 outputs (+ bias).
    fn mac_all(&self, xq: &[u8], sx: f32, zpx: u8) -> Vec<f32> {
        let rows = self.weight_u8.len() / self.in_dim;
        let mut y = vec![0.0f32; rows];
        for (o, yo) in y.iter_mut().enumerate() {
            let s = self.scale[o.min(self.scale.len() - 1)];
            let mut acc = self.mac_row(xq, zpx, o) as f32 * (sx * s);
            if let Some(b) = &self.bias {
                acc += b[o];
            }
            *yo = acc;
        }
        y
    }

    /// Dequantize the full `[out][in]` weight store to f32 row-major.
    /// Transient per call (the 8-bit store stays canonical in RAM: a
    /// cached f32 copy of the 582 MB encoder would cost ~2.3 GB).
    pub(crate) fn dequant_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.weight_u8.len());
        for o in 0..self.out_dim {
            let s = self.scale[o.min(self.scale.len() - 1)];
            let z = self.zero_point[o.min(self.zero_point.len() - 1)];
            let base = o * self.in_dim;
            w.extend(
                self.weight_u8[base..base + self.in_dim]
                    .iter()
                    .map(|&q| (q as f32 - z) * s),
            );
        }
        w
    }

    /// Tensor twin of `forward_batch`: identical per-tensor activation
    /// quantization, but the quantized activations are dequantized back
    /// to f32 and the MAC runs as one Flex matmul against dequantized
    /// weights. Activation-requantization noise is therefore replicated
    /// exactly; the residual vs the integer path is f32 summation order
    /// only (measured 9e-8 relative on synthetic inputs).
    pub(crate) fn forward_batch_t<B: burn::tensor::backend::Backend>(
        &self,
        xs: &[Vec<f32>],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<Vec<f32>> {
        use burn::tensor::Tensor;
        assert!(xs.iter().all(|r| r.len() == self.in_dim));
        if xs.is_empty() {
            return Vec::new();
        }
        let flat: Vec<f32> = xs.iter().flat_map(|r| r.iter().copied()).collect();
        let (xq, sx, zpx) = dynamic_quantize_u8(&flat);
        let xd: Vec<f32> = xq.iter().map(|&q| (q as f32 - zpx as f32) * sx).collect();
        let x_rows: Vec<Vec<f32>> = xd.chunks(self.in_dim).map(|c| c.to_vec()).collect();
        let x = super::tensor::rows_to_tensor::<B>(&x_rows, device);
        let w_rows: Vec<Vec<f32>> = self
            .dequant_weights()
            .chunks(self.in_dim)
            .map(|c| c.to_vec())
            .collect();
        // Store is `[out][in]`; matmul wants `[K, N] = [in, out]`.
        let w: Tensor<B, 2> = super::tensor::rows_to_tensor::<B>(&w_rows, device).transpose();
        let mut y = x.matmul(w);
        if let Some(b) = &self.bias {
            // `[1, out]` broadcasts over the `[T, out]` rows.
            let bt = super::tensor::rows_to_tensor::<B>(&[b.clone()], device);
            y = y + bt;
        }
        super::tensor::tensor_to_rows(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_quantize_matches_ort_probe() {
        // Values from the 1-node ORT probe (`probe_dql.py`): asymmetric
        // uint8, per-tensor — s=(3-(-2))/255, zp=102, rows as observed.
        let (q, s, zp) = dynamic_quantize_u8(&[1.0, -2.0, 3.0, 0.5, 0.5, 0.5]);
        assert!((s - 0.019607844).abs() < 1e-7, "s={s}");
        assert_eq!(zp, 102);
        assert_eq!(q, vec![153, 0, 255, 128, 128, 128]);
    }

    #[test]
    fn dynamic_quantize_constant_input_stays_exact() {
        // Positive-constant input: range forced to [0, max], so
        // s = max/255, zp = 0, q saturates high — and reconstructs max.
        let (q, s, zp) = dynamic_quantize_u8(&[1.0, 1.0]);
        assert!((s - 1.0 / 255.0).abs() < 1e-9, "s={s}");
        assert_eq!((q, zp), (vec![255, 255], 0));
    }

    #[test]
    fn dynamic_quantize_all_zero_is_guarded() {
        // Probed ORT behavior: s = 1.0, zp = 0, all-zero output.
        assert_eq!(dynamic_quantize_u8(&[0.0, 0.0]), (vec![0, 0], 1.0, 0));
    }

    #[test]
    fn forward_batch_quantizes_per_tensor() {
        // Identity weights, zp 0: outputs reconstruct the inputs up to
        // 1-LSB rounding. Two rows quantized jointly (min=-2, max=3 over
        // all 4 elements): row0 -> [153, 0], row1 -> [255, 128].
        let lin = Uint8Linear {
            weight_u8: vec![1, 0, 0, 1],
            scale: vec![1.0, 1.0],
            zero_point: vec![0.0, 0.0],
            bias: None,
            in_dim: 2,
            out_dim: 2,
        };
        let batched = lin.forward_batch(&[vec![1.0, -2.0], vec![3.0, 0.5]]);
        assert_eq!(batched.len(), 2);
        for (got, want) in batched.iter().zip([vec![1.0, -2.0], vec![3.0, 0.5]]) {
            for (a, b) in got.iter().zip(want.iter()) {
                assert!((a - b).abs() < 0.02, "got={batched:?}");
            }
        }
        // Solo spans only its own row (min=-2, max=1): same values back
        // within its own coarser scale.
        let solo = lin.forward(&[1.0, -2.0]);
        assert!((solo[0] - 1.0).abs() < 0.02 && (solo[1] + 2.0).abs() < 0.02);
    }

    #[test]
    fn int8_linear_matches_fp32_matmul() {
        // W = [[1,2],[3,4]] fp32, quantized per-row.
        let rows = [vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let mut w = Vec::new();
        let mut scales = Vec::new();
        for r in &rows {
            let (q, s) = Int8Linear::quantize_row(r);
            w.extend(q);
            scales.push(s);
        }
        let lin = Int8Linear::new(w, scales, None, 2, 2);
        let y = lin.forward(&[1.0, 1.0]);
        // Row scales: 2/127, 4/127; rounding error < 1 LSB.
        assert!((y[0] - 3.0).abs() < 0.05, "y={y:?}");
        assert!((y[1] - 7.0).abs() < 0.05, "y={y:?}");
    }

    #[test]
    fn zero_row_quantizes_without_nan() {
        let (q, s) = Int8Linear::quantize_row(&[0.0, 0.0]);
        assert_eq!(q, vec![0, 0]);
        assert!(s.is_finite() && s > 0.0);
        let lin = Int8Linear::new(q, vec![s], None, 2, 1);
        assert_eq!(lin.forward(&[5.0, -3.0]), vec![0.0]);
    }

    #[test]
    fn dequant_row_roundtrips() {
        let (q, s) = Int8Linear::quantize_row(&[0.5, -0.25, 0.125]);
        let lin = Int8Linear::new(q, vec![s], None, 3, 1);
        let dq = lin.dequant_row(0);
        assert!((dq[0] - 0.5).abs() < 0.01, "dq={dq:?}");
        assert!((dq[1] + 0.25).abs() < 0.01, "dq={dq:?}");
    }

    #[test]
    fn uint8_linear_matches_fp32_with_zero_point() {
        // W = [[1,2],[250,4]] u8, zp=121 scalar, scale=0.02:
        // row0 -> [(1-121)*.02, (2-121)*.02] = [-2.4, -2.38]
        let lin = Uint8Linear {
            weight_u8: vec![1, 2, 250, 4],
            scale: vec![0.02],
            zero_point: vec![121.0],
            bias: None,
            in_dim: 2,
            out_dim: 2,
        };
        let y = lin.forward(&[1.0, 1.0]);
        assert!((y[0] + 4.78).abs() < 1e-4, "y={y:?}");
        assert!((y[1] - 0.24).abs() < 1e-4, "y={y:?}");
        // dequant_row broadcasts the scalar scale/zp.
        assert_eq!(lin.dequant_row(1).len(), 2);
    }

    #[test]
    fn transpose_u8_fixes_matmul_layout() {
        // ONNX (K=3, N=2) storage; forward reads [out][in], so load must
        // transpose. Non-square on purpose: a missing transpose still
        // runs for square shapes and only scrambles values.
        let kn: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // rows K, cols N
        let oi = transpose_u8(&kn, 3, 2);
        assert_eq!(oi, vec![1, 3, 5, 2, 4, 6]);
        let lin = Uint8Linear {
            weight_u8: oi,
            scale: vec![1.0, 1.0],
            zero_point: vec![0.0, 0.0],
            bias: None,
            in_dim: 3,
            out_dim: 2,
        };
        // y[0] = 1*x0 + 3*x1 + 5*x2 (column 0 of the (K,N) store).
        // Approximate: the integer path scales by 1/255, which is not
        // exactly representable in f32 (1-ULP error).
        let y = lin.forward(&[1.0, 1.0, 1.0]);
        assert!((y[0] - 9.0).abs() < 1e-4 && (y[1] - 12.0).abs() < 1e-4, "y={y:?}");
    }

    #[test]
    fn tensor_linear_matches_integer_path() {
        use crate::parakeet_burn::backend::{cpu_device, CpuBackend};
        // Non-square (3 -> 2) with per-row scales, distinct zero-points
        // and bias, so a transposed weight read or a dropped bias shows
        // up. Inputs span signs and magnitudes.
        let lin = Uint8Linear {
            weight_u8: vec![1, 2, 250, 4, 5, 6],
            scale: vec![0.02, 0.03],
            zero_point: vec![121.0, 5.0],
            bias: Some(vec![0.5, -0.25]),
            in_dim: 3,
            out_dim: 2,
        };
        let xs = vec![
            vec![1.0, -2.0, 3.0],
            vec![0.5, 0.25, -1.5],
            vec![10.0, 0.0, -7.0],
        ];
        let s = lin.forward_batch(&xs);
        let t = lin.forward_batch_t::<CpuBackend>(&xs, &cpu_device());
        assert_eq!(s.len(), 3);
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|r| r.len() == 2));
        let (mut max_rel, mut ref_max) = (0.0f32, 0.0f32);
        for (a, b) in s.iter().zip(t.iter()) {
            for (p, q) in a.iter().zip(b.iter()) {
                ref_max = ref_max.max(q.abs());
                max_rel = max_rel.max((p - q).abs() / q.abs().max(1.0));
            }
        }
        eprintln!("tensor linear: max_rel={max_rel:.2e} ref_max={ref_max:.3}");
        // Activation requantization noise is replicated exactly, so the
        // residual is f32 summation order only.
        assert!(max_rel < 1e-4, "max_rel={max_rel} ref_max={ref_max}");
    }

    #[test]
    fn tensor_linear_empty_matches_scalar() {
        use crate::parakeet_burn::backend::{cpu_device, CpuBackend};
        let lin = Uint8Linear {
            weight_u8: vec![1, 2],
            scale: vec![1.0],
            zero_point: vec![0.0],
            bias: None,
            in_dim: 2,
            out_dim: 1,
        };
        let empty: Vec<Vec<f32>> = Vec::new();
        assert_eq!(lin.forward_batch_t::<CpuBackend>(&empty, &cpu_device()), empty);
    }
}
