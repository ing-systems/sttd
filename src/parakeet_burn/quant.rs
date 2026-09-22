// int8/uint8 weight containers for the Burn port.
//
// The exported ONNX uses *static* weight quantization (uint8 asymmetric
// for encoder MatMuls/Convs and the decoder embed/joint, int8 for the
// decoder LSTMs) plus *dynamic* activation quantization
// (`DynamicQuantizeLinear` + `MatMulInteger` + `ConvInteger`,
// `DynamicQuantizeLSTM`). Burn's native 8-bit GEMM coverage is still
// maturing, so this module stores weights as 8-bit + scale (+ zero-point)
// — 8-bit-now on disk/VRAM — and dequantizes to f32 for the Burn matmul.
// Swapping the inner matmul for a native quantized kernel later changes
// no call sites — only `forward`.

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

/// Dynamically quantize activations to int8 (mirrors ONNX
/// `DynamicQuantizeLinear`: per-row scale, no zero-point).
pub(crate) fn dynamic_quantize_row(x: &[f32]) -> (Vec<i8>, f32) {
    Int8Linear::quantize_row(x)
}

/// One asymmetrically-quantized uint8 Linear (encoder MatMuls/Convs,
/// decoder embed + joint):
/// `y = (((w_u8 - zp) * scale)) @ x + bias`, zp scalar-or-per-row.
///
/// Storage is ALWAYS `[out][in]` row-major. ONNX `MatMul` weights are
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

    /// `y = W x + b`, `x: [in]`, returns `[out]`.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim);
        let rows = self.weight_u8.len() / self.in_dim;
        let mut y = vec![0.0f32; rows];
        for (o, yo) in y.iter_mut().enumerate() {
            let dq = self.dequant_row(o);
            let mut acc: f32 = dq.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            if let Some(b) = &self.bias {
                acc += b[o];
            }
            *yo = acc;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(lin.forward(&[1.0, 1.0, 1.0]), vec![9.0, 12.0]);
    }
}
