// Burn backend selection: NdArray (CPU) always with `parakeet-burn`,
// CubeCL CUDA with `parakeet-burn-cuda` + `--gpu`.

#[cfg(feature = "parakeet-burn")]
pub(crate) type CpuBackend = burn_ndarray::NdArray;

#[cfg(feature = "parakeet-burn-cuda")]
pub(crate) type CudaBackend = burn_cuda::Cuda;

/// Which Burn device the engine runs on (log-only today; the encoder
/// bring-up in `encoder.rs` threads this through to tensor allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BurnDevice {
    Cpu,
    #[cfg(feature = "parakeet-burn-cuda")]
    Cuda,
}

pub(crate) fn select_device(request_gpu: bool) -> BurnDevice {
    #[cfg(feature = "parakeet-burn-cuda")]
    if request_gpu {
        return BurnDevice::Cuda;
    }
    let _ = request_gpu;
    BurnDevice::Cpu
}

/// Size rayon's process-wide thread pool from `--threads` (already
/// resolved: `0` means physical cores, minimum 1 — see
/// `asr::resolve_n_threads`). The pool is global and one-shot: the
/// first `build_global` wins and later calls (reloads after pause)
/// are no-ops via `CAP`. If something else sized it first (today only
/// `tokenizers` on the ORT path could, and it never loads alongside
/// Burn), the existing size is kept and logged, never overridden.
///
/// Nothing in the Burn path uses this pool yet: `burn-ndarray` is
/// built without its `multi-threads` feature (its `iter_par!` degrades
/// to sequential) and the int8 forward loops are scalar. So today this
/// only pins the size for the future tensor-ops swap; it changes no
/// numerics and no current speed.
pub(crate) fn cap_thread_pool(n_threads: usize) {
    use std::sync::Once;
    static CAP: Once = Once::new();
    let want = n_threads.max(1);
    CAP.call_once(|| {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(want)
            .build_global()
        {
            Ok(()) => log::info!("burn parakeet: rayon pool capped at {want} threads (--threads)"),
            Err(e) => log::info!("burn parakeet: rayon pool already sized, keeping it ({e})"),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_default_without_cuda_build() {
        // Without the cuda feature every request lands on CPU.
        let d = select_device(false);
        assert_eq!(d, BurnDevice::Cpu);
    }

    #[test]
    fn rayon_cap_is_idempotent_and_effective() {
        super::cap_thread_pool(4);
        // Reloads re-call this: the Once gate makes every later call a
        // no-op that must never panic, even with a different request.
        super::cap_thread_pool(2);
        // First call wins the process-wide pool. Nothing else in this
        // test binary initializes it first (the only other rayon user,
        // `tokenizers`, never loads here), so the size must be ours.
        // If some future test beats us to it, build_global reports
        // AlreadyInitialized and this assert needs revisiting.
        assert_eq!(rayon::current_num_threads(), 4);
    }

    #[test]
    fn burn_ndarray_matmul_runs() {
        use burn::tensor::Tensor;
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let a: Tensor<NdArray, 2> = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let b: Tensor<NdArray, 2> = Tensor::from_data([[1.0f32, 0.0], [0.0, 1.0]], &device);
        let c = a.matmul(b);
        let data: Vec<f32> = c.into_data().to_vec().expect("to_vec");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "parakeet-burn-cuda")]
    #[test]
    fn burn_cuda_device_probes() {
        // Probe only: CUDA may be absent on CI; constructing the device
        // handle must not fail the suite.
        let device = burn_cuda::CudaDevice::default();
        let _ = device;
    }
}
