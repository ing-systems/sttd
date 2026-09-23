// Burn Tensor helpers for the tensor-ops swap (step 4, Phase 1+).
//
// The scalar int8 forward (`quant`, `block`, `pre_encode`, `joint`)
// stays the default path during bring-up (see `backend::parse_tensor_flag`);
// the tensor variants built on these helpers must reproduce it within
// f32 rounding so the ORT goldens keep passing under policy (b).
// Everything here is generic over `B: Backend` so the concrete CPU
// backend (`Flex` today) remains a one-line choice in `backend.rs`.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn::tensor::backend::BackendTypes;

use super::backend::CpuBackend;

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
#[derive(Clone, Copy)]
pub(crate) struct TensorCtx {
    pub(crate) device: <CpuBackend as BackendTypes>::Device,
    pub(crate) enabled: bool,
}

impl TensorCtx {
    pub(crate) fn scalar() -> Self {
        Self {
            device: super::backend::cpu_device(),
            enabled: false,
        }
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
    use crate::parakeet_burn::backend::CpuBackend;

    #[test]
    fn tensor_roundtrip_preserves_non_square() {
        let device = crate::parakeet_burn::backend::cpu_device();
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
        let device = crate::parakeet_burn::backend::cpu_device();
        let _: Tensor<CpuBackend, 2> = rows_to_tensor(&[], &device);
    }

    #[test]
    #[should_panic]
    fn tensor_rejects_ragged() {
        let device = crate::parakeet_burn::backend::cpu_device();
        let _: Tensor<CpuBackend, 2> = rows_to_tensor(&[vec![1.0], vec![1.0, 2.0]], &device);
    }
}
