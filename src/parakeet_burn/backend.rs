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
