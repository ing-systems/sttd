// Burn backend selection: Flex (CPU) always with `parakeet-burn`,
// CubeCL CUDA with `parakeet-burn-cuda` + `--gpu`.

#[cfg(feature = "parakeet-burn")]
pub(crate) type CpuBackend = burn_flex::Flex;

/// Concrete CPU device handle type (see `cpu_device`).
#[cfg(feature = "parakeet-burn")]
pub(crate) type CpuDevice =
    <CpuBackend as burn::tensor::backend::BackendTypes>::Device;

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

/// Concrete CPU device handle for tensor allocation. Threaded through
/// to every tensor op as phases land (Phase 2+); the scalar path never
/// touches it.
pub(crate) fn cpu_device() -> <CpuBackend as burn::tensor::backend::BackendTypes>::Device {
    <CpuBackend as burn::tensor::backend::BackendTypes>::Device::default()
}

/// Parse the tensor-path selector (see `use_tensor_path`). Pure for tests.
pub(crate) fn parse_tensor_flag(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true"))
}

/// Scalar int8 loops stay the default path during bring-up; `STTD_BURN_TENSOR=1`
/// (or `=true`) selects the tensor variants once a phase wires them in.
/// Phase 7 flips the default. Read once at load in Phase 2 (never in
/// tests: env vars are process-global and test threads run in parallel).
pub(crate) fn use_tensor_path() -> bool {
    parse_tensor_flag(std::env::var("STTD_BURN_TENSOR").ok().as_deref())
}
/// Size rayon's process-wide thread pool from `--threads` (already
/// resolved: `0` means physical cores, minimum 1 — see
/// `asr::resolve_n_threads`). The pool is global and one-shot: the
/// first `build_global` wins and later calls (reloads after pause)
/// are no-ops via `CAP`. If something else sized it first (today only
/// `tokenizers` on the ORT path could, and it never loads alongside
/// Burn), the existing size is kept and logged, never overridden.
///
/// Nothing in the Burn path uses this pool yet beyond flex's own
/// multithreaded ops (on by default): the int8 forward loops are still
/// scalar. So today this pins the size for the tensor-ops swap; it
/// changes no numerics and no current speed.
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
    fn tensor_flag_parsing() {
        assert!(super::parse_tensor_flag(Some("1")));
        assert!(super::parse_tensor_flag(Some("true")));
        assert!(!super::parse_tensor_flag(None));
        assert!(!super::parse_tensor_flag(Some("0")));
        assert!(!super::parse_tensor_flag(Some("false")));
        assert!(!super::parse_tensor_flag(Some("yes")));
    }

    #[test]
    fn cpu_device_allocates() {
        // The handle threads through to every tensor op from Phase 2 on.
        let d = super::cpu_device();
        let _ = d;
    }

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
    fn burn_flex_matmul_runs() {
        use burn::tensor::Tensor;
        use burn_flex::{Flex, FlexDevice};
        let device = FlexDevice;
        let a: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let b: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 0.0], [0.0, 1.0]], &device);
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
