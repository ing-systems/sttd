// Integer GEMM for the tensor path's quantized linears.
//
// `Uint8Linear` weights and dynamically-quantized activations are already
// u8; the scalar path multiplies them as integers (`mac_row`, i64
// accumulator). The tensor path used to dequantize both sides to f32 and
// run a Flex matmul — at ~240 GFLOPS effective on the skinny `[T,K]x[K,N]`
// shapes that dominate the encoder (T=25-125), that f32 GEMM was 80%+ of
// forward time.
//
// This module runs the same products as integers through AVX-512
// `vpmaddwd` (32 MACs/instruction) on i16-widened rows, parallelized over
// output rows (encoder) or output-column blocks via rayon. Integer
// addition is exact, so blocking and reordering change nothing: the i32
// accumulator (terms <= 65025, at most 4096 of them, so <= 266M — no
// overflow) holds the same value the scalar path's i64 holds, and the
// single f32 scale-up + bias uses the same formula. Outputs match
// `forward_batch` bit-for-bit while running several times faster than
// the f32 Flex matmul.
//
// Weights arrive pre-widened (`Uint8Linear::w_i16`, built once at warm);
// activations widen inside `quantize_rows_i16`, which replicates the
// probe-pinned `dynamic_quantize_u8` math exactly (same min/max window,
// same zp, same q) and then subtracts zp as i16.

use rayon::prelude::*;

/// AVX-512 integer path needs zmm word ops (`vpmaddwd`): F + BW. Checked
/// once per call (a static load); machines without it run the scalar
/// integer path, which is equally exact and merely slower.
pub(crate) fn has_fast_path() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
}

/// Widen one `[out][in]` u8 weight store to i16 with the row zero-point
/// subtracted: `B[o,k] = w - zp[o].round()`, the same integers the scalar
/// `mac_row` multiplies (range [-255, 255], exact).
pub(crate) fn widen_weights(w: &[u8], out_dim: usize, in_dim: usize, zp: &[f32]) -> Vec<i16> {
    assert_eq!(w.len(), out_dim * in_dim);
    let mut b = vec![0i16; w.len()];
    // Rows are independent: exact in any order, so parallelize.
    b.par_chunks_mut(in_dim)
        .enumerate()
        .for_each(|(o, row)| {
            let z = zp[o.min(zp.len() - 1)].round() as i16;
            for (k, v) in row.iter_mut().enumerate() {
                *v = w[o * in_dim + k] as i16 - z;
            }
        });
    b
}

/// Quantize activations exactly like `dynamic_quantize_u8` (same window,
/// same zp, same q values) and widen to i16 with zp subtracted:
/// `A[i] = (q - zpx)`. Returns `(a_i16, scale, zpx)`.
/// Parallel min/max + parallel map: min/max are associative and exact,
/// per-element q is independent, so this is bit-identical to the serial
/// version (pinned by `quantize_twin_matches_serial`).
pub(crate) fn quantize_rows_i16(xs: &[f32]) -> (Vec<i16>, f32, i64) {
    assert!(!xs.is_empty());
    let n_threads = rayon::current_num_threads().max(1);
    let (mn, mx) = xs
        .par_chunks(xs.len().div_ceil(n_threads).max(1))
        .map(|c| {
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in c {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            (mn, mx)
        })
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(a, b), (c, d)| (a.min(c), b.max(d)),
        );
    let (lo, hi) = (mn.min(0.0), mx.max(0.0));
    let range = hi - lo;
    if !(range > 0.0) {
        return (vec![0; xs.len()], 1.0, 0);
    }
    let s = range / 255.0;
    let zp = ((-lo / s).round() as i32).clamp(0, 255);
    let a: Vec<i16> = xs
        .par_iter()
        .map(|&v| ((v / s).round() as i32 + zp).clamp(0, 255) as i16 - zp as i16)
        .collect();
    (a, s, zp as i64)
}

/// `C[T,N] = A[T,K] @ B[N,K]^T`, integer MACs in i32, one f32 scale-up +
/// bias per element with the same formula as `Uint8Linear::mac_all`:
/// `out = acc as f32 * (sx * sw[o]) + bias[o]` (`sw` honors the same
/// scalar-or-per-row broadcast, `bias` is `None` or full-length).
/// `out` must hold exactly `T*N` f32s.
pub(crate) fn linear_i16(
    a: &[i16],
    t: usize,
    k: usize,
    b: &[i16],
    n: usize,
    sw: &[f32],
    bias: Option<&[f32]>,
    sx: f32,
    out: &mut [f32],
) {
    assert_eq!(a.len(), t * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(out.len(), t * n);
    assert!(k >= 1 && t >= 1 && n >= 1);
    if let Some(bb) = bias {
        assert_eq!(bb.len(), n);
    }
    let fast = has_fast_path();
    if t >= 16 {
        // Row-major C: rows are contiguous, so parallelize over rows;
        // each row runs the blocked micro-kernel over all N columns.
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(ti, row)| {
                let arow = &a[ti * k..(ti + 1) * k];
                if fast {
                    // SAFETY: AVX-512 gated on `fast`; slice lengths
                    // asserted on entry.
                    unsafe { row_kernel(arow, b, k, n, 0, n, sw, bias, sx, row) };
                } else {
                    row_scalar(arow, b, k, n, 0, n, sw, bias, sx, row);
                }
            });
    } else {
        // Few rows (joint single-step calls): rows are too few to balance
        // threads, so parallelize over column tiles into thread-local
        // buffers, then stitch (C is small here: T<16 rows).
        let tb = (n.div_ceil(64).div_ceil(OB) * OB).max(OB);
        let ranges: Vec<(usize, usize)> = (0..n)
            .step_by(tb)
            .map(|o0| (o0, (o0 + tb).min(n)))
            .collect();
        let tiles: Vec<Vec<f32>> = ranges
            .into_par_iter()
            .map(|(o0, o1)| {
                let mut tile = vec![0.0f32; t * (o1 - o0)];
                for ti in 0..t {
                    let arow = &a[ti * k..(ti + 1) * k];
                    let seg = &mut tile[ti * (o1 - o0)..(ti + 1) * (o1 - o0)];
                    if fast {
                        // SAFETY: as above.
                        unsafe { row_kernel(arow, b, k, n, o0, o1, sw, bias, sx, seg) };
                    } else {
                        row_scalar(arow, b, k, n, o0, o1, sw, bias, sx, seg);
                    }
                }
                tile
            })
            .collect();
        // Serial stitch: tiles cover disjoint column ranges (rayon
        // `collect` preserves range order).
        let mut o0 = 0;
        for tile in &tiles {
            let w = tile.len() / t;
            for ti in 0..t {
                out[ti * n + o0..ti * n + o0 + w]
                    .copy_from_slice(&tile[ti * w..(ti + 1) * w]);
            }
            o0 += w;
        }
    }
}

/// Scalar row-segment fallback (and exactness reference for the
/// micro-kernel): columns `[o0, o1)` of one output row.
fn row_scalar(
    a_row: &[i16],
    b: &[i16],
    k: usize,
    _n: usize,
    o0: usize,
    o1: usize,
    sw: &[f32],
    bias: Option<&[f32]>,
    sx: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), o1 - o0);
    for (oi, o) in (o0..o1).enumerate() {
        let s = dot_scalar(a_row, &b[o * k..(o + 1) * k], k);
        let w = sw[o.min(sw.len() - 1)];
        let mut y = s as f32 * (sx * w);
        if let Some(bb) = bias {
            y += bb[o];
        }
        out[oi] = y;
    }
}

/// Micro-block width: columns computed together, sharing one A-row
/// stream (A traffic drops ~OBx) with accumulators held in registers.
const OB: usize = 8;

/// Blocked micro-kernel: columns `[o0, o1)` of one output row from one A
/// row. `OB` columns at a time (A streamed once per block, `OB`
/// accumulators in zmm), scalar tail for the ragged edge. Integer
/// arithmetic throughout, so blocking is bit-exact vs `dot_scalar`.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_kernel(
    a_row: &[i16],
    b: &[i16],
    k: usize,
    _n: usize,
    o0: usize,
    o1: usize,
    sw: &[f32],
    bias: Option<&[f32]>,
    sx: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(out.len(), o1 - o0);
    let ap = a_row.as_ptr();
    let bp = b.as_ptr();
    let swp = sw.as_ptr();
    let swlen = sw.len();
    let bbp = bias.map(|bb| bb.as_ptr());
    let outp = out.as_mut_ptr();
    let k_main = k - (k % 32);
    let mut o = o0;
    while o + OB <= o1 {
        let mut acc = [_mm512_setzero_si512(); OB];
        let mut kk = 0;
        while kk < k_main {
            let va = _mm512_loadu_si512(ap.add(kk) as *const _);
            let mut j = 0;
            while j < OB {
                let vb = _mm512_loadu_si512(bp.add((o + j) * k + kk) as *const _);
                acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(va, vb));
                j += 1;
            }
            kk += 32;
        }
        let mut j = 0;
        while j < OB {
            let mut s = _mm512_reduce_add_epi32(acc[j]);
            let mut kk2 = k_main;
            while kk2 < k {
                s += *ap.add(kk2) as i32 * *bp.add((o + j) * k + kk2) as i32;
                kk2 += 1;
            }
            let w = *swp.add((o + j).min(swlen - 1));
            let mut y = s as f32 * (sx * w);
            if let Some(p) = bbp {
                y += *p.add(o + j);
            }
            *outp.add(o - o0 + j) = y;
            j += 1;
        }
        o += OB;
    }
    // Ragged columns: one SIMD dot each (same code as the old per-dot
    // path — exact, and out of the hot loop).
    while o < o1 {
        let s = dot_avx512(ap, bp.add(o * k), k);
        let w = *swp.add(o.min(swlen - 1));
        let mut y = s as f32 * (sx * w);
        if let Some(p) = bbp {
            y += *p.add(o);
        }
        *outp.add(o - o0) = y;
        o += 1;
    }
}

/// One integer dot product over `k` i16 pairs, i32 accumulator.
/// Exact for our ranges (terms <= 65025, k <= 4096 in this model).
fn dot_scalar(a: &[i16], b: &[i16], k: usize) -> i32 {
    let mut s = 0i32;
    for i in 0..k {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// AVX-512 twin of `dot_scalar`: 32 MACs per `vpmaddwd`, i32
/// accumulation, scalar tail. Bit-identical (integer arithmetic is
/// exact, so vector width and tails change nothing).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_avx512(a: *const i16, b: *const i16, k: usize) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    while i + 32 <= k {
        let va = _mm512_loadu_si512(a.add(i) as *const _);
        let vb = _mm512_loadu_si512(b.add(i) as *const _);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(va, vb));
        i += 32;
    }
    let mut sum = _mm512_reduce_add_epi32(acc);
    while i < k {
        sum += *a.add(i) as i32 * *b.add(i) as i32;
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_path_active_on_host() {
        // This port builds with `-C target-cpu=native` (see
        // `.cargo/config.toml`) for a local AVX-512 host; the integer
        // path must actually engage there.
        assert!(has_fast_path(), "expected AVX-512F+BW on this host");
    }

    /// LCG f32s in [-3, 3], deterministic (no rand dependency).
    fn rnd_vec(len: usize, mut seed: u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 6.0
            })
            .collect()
    }

    #[test]
    fn quantize_twin_matches_serial() {
        use super::super::quant::dynamic_quantize_u8;
        // Shapes incl. non-multiples of 32/64, constants, all-zero (guard
        // path), and single-element inputs.
        for (len, seed) in [(1, 7), (3, 8), (31, 9), (64, 10), (1000, 11), (4096, 12)] {
            let xs = rnd_vec(len, seed);
            let (q, s, zp) = dynamic_quantize_u8(&xs);
            let (a, st, zt) = quantize_rows_i16(&xs);
            assert_eq!((st, zt), (s, zp as i64), "len={len}");
            let expect: Vec<i16> = q.iter().map(|&v| v as i16 - zp as i16).collect();
            assert_eq!(a, expect, "len={len}");
        }
        // Guard path: all-zero and constant inputs.
        for xs in [vec![0.0; 16], vec![2.5; 100], vec![-1.0; 7]] {
            let (q, s, zp) = dynamic_quantize_u8(&xs);
            let (a, st, zt) = quantize_rows_i16(&xs);
            assert_eq!((s, zp as i64), (st, zt));
            let expect: Vec<i16> = q.iter().map(|&v| v as i16 - zp as i16).collect();
            assert_eq!(a, expect);
        }
    }

    #[test]
    fn widen_matches_scalar_integers() {
        // Row zp broadcast: scalar scale/zp + per-row scale/zp.
        let w: Vec<u8> = (0..2 * 70).map(|i| (i * 37 % 256) as u8).collect();
        for zp in [vec![0.0], vec![121.0], vec![3.0, 200.0]] {
            let b = widen_weights(&w, 2, 70, &zp);
            assert_eq!(b.len(), 140);
            for o in 0..2 {
                let z = zp[o.min(zp.len() - 1)].round() as i16;
                for k in 0..70 {
                    assert_eq!(b[o * 70 + k], w[o * 70 + k] as i16 - z, "o={o} k={k}");
                }
            }
        }
    }

    /// Full driver vs the scalar integer path, exact equality, over odd
    /// shapes (tails!), per-row scales/zps, bias/None, and extreme values.
    #[test]
    fn gemm_matches_scalar_mac_exactly() {
        use super::super::quant::Uint8Linear;
        let mut seed: u64 = 0x16E44;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 6.0
        };
        // (t, k, n) incl. non-multiples of 32 (tail), t < 16 (serial
        // branch) and t >= 16 (parallel branch).
        for (t, k, n) in [(1, 70, 3), (3, 128, 65), (16, 100, 40), (20, 1024, 33), (5, 640, 8198)] {
            let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..k).map(|_| rnd()).collect()).collect();
            let wu: Vec<u8> = (0..n * k).map(|_| (rnd() * 41.0 + 128.0) as u8).collect();
            // Per-row scales/zps (including a scalar-broadcast case).
            let (sw, zpw): (Vec<f32>, Vec<f32>) = if n % 2 == 0 {
                ((0..n).map(|i| 0.01 + 0.001 * (i % 7) as f32).collect(), vec![100.0])
            } else {
                (
                    (0..n).map(|i| 0.02 + 0.002 * (i % 5) as f32).collect(),
                    (0..n).map(|i| (i * 13 % 256) as f32).collect(),
                )
            };
            let bias: Option<Vec<f32>> = if k % 2 == 0 {
                Some((0..n).map(|_| rnd()).collect())
            } else {
                None
            };
            // Reference: the scalar integer path under test.
            let lin = Uint8Linear {
                weight_u8: super::super::quant::transpose_u8(&wu, k, n),
                scale: sw.clone(),
                zero_point: zpw.clone(),
                bias: bias.clone(),
                in_dim: k,
                out_dim: n,
                cache: std::sync::OnceLock::new(),
                w_i16: std::sync::OnceLock::new(),
            };
            let expect = lin.forward_batch(&xs);
            // Driver under test: quantize twin + integer GEMM.
            let flat: Vec<f32> = xs.iter().flat_map(|r| r.iter().copied()).collect();
            let (a, sx, _) = quantize_rows_i16(&flat);
            let b = widen_weights(&lin.weight_u8, n, k, &lin.zero_point);
            let mut out = vec![0.0f32; t * n];
            linear_i16(&a, t, k, &b, n, &lin.scale, lin.bias.as_deref(), sx, &mut out);
            let got: Vec<Vec<f32>> = out.chunks(n).map(|c| c.to_vec()).collect();
            assert_eq!(got, expect, "t={t} k={k} n={n}");
        }
    }

    #[test]
    fn dot_avx512_matches_scalar_on_tails() {
        // K values around vector widths (31/32/33, 63/64/65, 0) plus a
        // large K, extreme i16 values (saturation corners).
        if !has_fast_path() {
            return;
        }
        let mut seed: u64 = 0xD07;
        for k in [0, 1, 31, 32, 33, 63, 64, 65, 100, 4096] {
            let a: Vec<i16> = (0..k)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed % 511) as i16 - 255
                })
                .collect();
            let b: Vec<i16> = (0..k)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed % 511) as i16 - 255
                })
                .collect();
            let s = dot_scalar(&a, &b, k);
            let v = unsafe { dot_avx512(a.as_ptr(), b.as_ptr(), k) };
            assert_eq!(v, s, "k={k}");
        }
        // Saturation corner: all max-magnitude products at max K.
        let a = vec![255i16; 4096];
        let b = vec![-255i16; 4096];
        let expect = 4096i32 * 255 * -255;
        assert_eq!(dot_scalar(&a, &b, 4096), expect);
        assert_eq!(unsafe { dot_avx512(a.as_ptr(), b.as_ptr(), 4096) }, expect);
    }
}
