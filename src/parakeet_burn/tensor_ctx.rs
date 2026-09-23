// Burn Tensor helpers for the tensor-ops swap (step 4, Phase 1+).
//
// The scalar int8 forward (`quant_linear`, `conformer_block`, `subsampler`, `joint`)
// stays the default path during bring-up (see `runtime::parse_tensor_flag`);
// the tensor variants built on these helpers must reproduce it within
// f32 rounding so the ORT goldens keep passing under policy (b).
// Everything here is generic over `B: Backend` so the concrete CPU
// backend (`Flex` today) remains a one-line choice in `runtime.rs`.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn::tensor::backend::BackendTypes;

use super::runtime::CpuBackend;

/// `&[Vec<f32>]` rows (`[T][K]`, non-empty, rectangular) -> `Tensor<B, 2>`.
/// Panics on empty/ragged input: callers return early on empty (as the
/// scalar forwards do) instead of guessing a width.
pub(crate) fn rows_to_tensor<B: Backend>(rows: &[Vec<f32>], device: &<B as BackendTypes>::Device) -> Tensor<B, 2> {
    assert!(!rows.is_empty(), "rows_to_tensor: empty input");
    let cols = rows[0].len();
    assert!(
        rows.iter().all(|r| r.len() == cols),
        "rows_to_tensor: ragged rows"
    );
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_data(
        TensorData::new(flat, [rows.len(), cols]),
        device,
    )
}

/// Threaded handle for the tensor path: where to allocate + kill-switch.
/// `enabled = false` dispatches every op back to its scalar twin, so the
/// scalar fallback (decision 2: kept for debugging parity regressions)
/// is always one flag away. `FlexDevice` is `Copy`, so this rides along
/// by value through every forward signature from Phase 2b on.
///
/// `vnni` selects the Phase-11 VNNI requantized path (`vpdpbusd` u8*s8,
/// approximate — see `int_gemm::requant_u8_to_s8`) vs the bit-exact
/// `vpmaddwd` i16 path. Kept explicit (not env-global) so tests can run
/// both paths in one process: production sets it from
/// `runtime::vnni_enabled()`, tests set `false` for golden pinning and
/// `true` for the loose-tolerance VNNI goldens.
#[derive(Clone, Copy)]
pub(crate) struct TensorCtx {
    pub(crate) device: <CpuBackend as BackendTypes>::Device,
    pub(crate) enabled: bool,
    pub(crate) vnni: bool,
}

impl TensorCtx {
    pub(crate) fn scalar() -> Self {
        Self {
            device: super::runtime::cpu_device(),
            enabled: false,
            vnni: false,
        }
    }

    /// Tensor path with explicit VNNI choice (production: `vnni` from
    /// `runtime::vnni_enabled()`; tests: `false` = bit-exact pinning).
    pub(crate) fn with_vnni(device: <CpuBackend as BackendTypes>::Device, vnni: bool) -> Self {
        Self {
            device,
            enabled: true,
            vnni,
        }
    }

    /// Whether this dispatch should run the VNNI kernels: tensor path on,
    /// VNNI requested, and the CPU actually has `avx512vnni`.
    #[inline]
    pub(crate) fn use_vnni(&self) -> bool {
        self.enabled && self.vnni && super::int_gemm::has_avx512_vnni()
    }
}

/// `Tensor<B, 2>` -> `Vec<Vec<f32>>` rows.
pub(crate) fn tensor_to_rows<B: Backend>(t: Tensor<B, 2>) -> Vec<Vec<f32>> {
    let dims = t.dims();
    t.into_data()
        .to_vec::<f32>()
        .expect("tensor_to_rows: to_vec")
        .chunks(dims[1])
        .map(|c| c.to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet_burn::runtime::CpuBackend;

    #[test]
    fn tensor_roundtrip_preserves_non_square() {
        let device = crate::parakeet_burn::runtime::cpu_device();
        let rows = vec![
            vec![1.0f32, -2.0, 3.5],
            vec![0.0, 100.0, -0.25],
            vec![7.0, 8.0, 9.0],
            vec![-1.0, -2.0, -3.0],
            vec![0.5, 0.25, 0.125],
        ];
        let t: Tensor<CpuBackend, 2> = rows_to_tensor(&rows, &device);
        assert_eq!(t.dims(), [5, 3]);
        assert_eq!(tensor_to_rows(t), rows);
    }

    #[test]
    #[should_panic]
    fn tensor_rejects_empty() {
        let device = crate::parakeet_burn::runtime::cpu_device();
        let _: Tensor<CpuBackend, 2> = rows_to_tensor(&[], &device);
    }

    #[test]
    #[should_panic]
    fn tensor_rejects_ragged() {
        let device = crate::parakeet_burn::runtime::cpu_device();
        let _: Tensor<CpuBackend, 2> = rows_to_tensor(&[vec![1.0], vec![1.0, 2.0]], &device);
    }
}
