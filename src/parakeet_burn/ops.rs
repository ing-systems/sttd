// Shared elementwise ops for the Conformer stack.
//
// Extracted from `block.rs`: layernorm / Swish / softmax / GLU are used by
// `block`, `pre_encode` tests, and `igemm` epilogue tests. Single owner,
// no behavior change.
//
// Exactness contract (Phase 4 decision, preserved): per-element math is
// scalar with the same libm calls ORT was verified against; rayon only
// spreads independent rows (same intra-row order, bit-exact vs serial).
// Layernorm uses f64 accumulators — sequential f32 sums round enough to
// matter 4096-wide (bisected during bring-up).

use rayon::prelude::*;

use super::int_gemm::sigmoid;

/// LayerNorm params (gamma/beta, eps fixed 1e-5 per graph).
#[derive(Debug, Clone)]
pub(crate) struct LayerNormP {
    pub(crate) gamma: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

pub(crate) const LN_EPS: f32 = 1e-5;

pub(crate) fn layernorm_row(x: &[f32], p: &LayerNormP) -> Vec<f32> {
    // Stays scalar by decision (Phase 4): the mean/variance sums span
    // 1024+ elements and a sequential f32 sum rounds enough to matter
    // downstream (~1e-6 normalized error amplifies 4096-wide through
    // the FFN to ~1.0 — bisected during bring-up). f32 tensor
    // reductions (parallel order) regress this; f64 tensor layernorm
    // would cost more traffic than the scalar loop saves. The op is
    // memory-bound O(T*D) beside the matmuls, so nothing is lost.
    assert_eq!(x.len(), p.gamma.len());
    // f64 accumulators: the mean/variance sums span 1024 elements, and a
    // sequential f32 sum rounds enough to matter downstream (a ~1e-6
    // normalized error amplifies 4096-wide through the FFN to ~1.0 at
    // ref_max 600 — bisected via the staged parity test below). ORT's own
    // vectorized reduction lands within 1e-7 of this.
    let n = x.len() as f64;
    let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + LN_EPS as f64).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| ((v as f64 - mean) * inv * p.gamma[i] as f64 + p.beta[i] as f64) as f32)
        .collect()
}

pub(crate) fn layernorm_batch(x: &[Vec<f32>], p: &LayerNormP) -> Vec<Vec<f32>> {
    // Rows are independent (same per-row f64 math as `layernorm_row`, same
    // order within a row): rayon only spreads rows over threads, so this
    // is bit-exact vs the serial loop.
    x.par_iter().map(|r| layernorm_row(r, p)).collect()
}

// Swish/GLU/add/scale stay scalar *per element* by decision (Phase 4):
// each element is computed exactly (same libm exp the graph was verified
// against — no SIMD-exp tensor twins, which would drift); independent
// rows additionally run in parallel via rayon (same intra-row order, so
// bit-exact vs the serial loops). Serial exp was the bottleneck for
// short utterances (a 4096-wide swish row costs ~3x its own matmul).
pub(crate) fn swish_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x * sigmoid(*x);
    }
}

pub(crate) fn softmax_rows(scores: &mut [Vec<f32>]) {
    scores.par_iter_mut().for_each(|row| {
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
    });
}

/// GLU gate: split 2d in halves, `a * sigmoid(b)`.
pub(crate) fn glu(v: &[f32]) -> Vec<f32> {
    assert!(v.len() % 2 == 0);
    let (a, b) = v.split_at(v.len() / 2);
    a.iter().zip(b.iter()).map(|(&x, &g)| x * sigmoid(g)).collect()
}
