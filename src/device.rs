//! Inference device selection (CPU or CUDA) for the ONNX Runtime sessions.
//!
//! The GPU path exists only in a `cuda` build. Without that Cargo feature the
//! `Cuda` variant stays in the enum (callers match on it) but nothing can
//! produce it, and [`apply`] hands back the builder unchanged.

use ort::session::builder::SessionBuilder;

/// The device an ONNX Runtime session runs on. `Copy`, so callers pass it by
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Cuda` is never built without the `cuda` feature.
pub(crate) enum Device {
    /// Default CPU execution provider (no `with_execution_providers` call).
    Cpu,
    /// CUDA execution provider on device 0, TF32 off.
    Cuda,
}

/// Resolve the device a session must run on.
///
/// Returns `Device::Cuda` only when the caller asked for the GPU (`--gpu`)
/// *and* the binary was built with the `cuda` feature. A `--gpu` request in a
/// non-CUDA build is not an error: it falls back to the CPU here, and the
/// flag-ignoring warning in `main` reports it.
pub(crate) fn requested(use_gpu: bool) -> Device {
    if use_gpu && cfg!(feature = "cuda") {
        Device::Cuda
    } else {
        Device::Cpu
    }
}

/// Is the CUDA execution provider usable in this build and on this machine?
///
/// The answer does not change during a run, so it is probed once and cached.
/// Any failure (the ONNX Runtime environment cannot start, the provider is
/// not compiled in, no GPU present) reads as "no", because a missing GPU must
/// not stop transcription.
///
/// Without the `cuda` feature this is a constant `false`.
pub(crate) fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use ort::ep::ExecutionProvider;

        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *AVAILABLE.get_or_init(|| match ort::environment::Environment::current() {
            Ok(_) => ort::ep::CUDA::default().is_available().unwrap_or(false),
            Err(e) => {
                log::warn!("CUDA probe failed (ONNX Runtime environment): {e}");
                false
            }
        });
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Attach the execution provider for `device` to `builder`.
///
/// `Device::Cpu` returns the builder unchanged (the CPU provider is the ONNX
/// Runtime default). `Device::Cuda` registers CUDA on device 0 with TF32 off,
/// which keeps results reproducible across runs. A registration failure is a
/// warn, not an error: the builder is returned unchanged so the session still
/// loads on the CPU.
pub(crate) fn apply(builder: SessionBuilder, device: Device) -> SessionBuilder {
    match device {
        Device::Cpu => builder,
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let ep = ort::ep::CUDA::default()
                .with_device_id(0)
                .with_tf32(false)
                .build()
                .error_on_failure();
            match builder.with_execution_providers([ep]) {
                Ok(b) => b,
                Err(e) => {
                    let msg = e.to_string();
                    log::warn!("CUDA execution provider unavailable ({msg}); using CPU");
                    e.recover()
                }
            }
        }
        // No `cuda` feature: the session cannot use the GPU. `requested`
        // never returns `Cuda` here, so this only guards a direct caller.
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            log::warn!("CUDA requested but this build has no `cuda` feature; using CPU");
            builder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_follows_gpu_flag_and_feature() {
        assert_eq!(requested(false), Device::Cpu);
        assert_eq!(requested(true), requested_matches_feature());
    }

    /// Without the `cuda` feature the GPU flag cannot select CUDA; with it,
    /// the flag selects CUDA regardless of whether a GPU is present (a
    /// missing GPU is a `cuda_available` question, not a selection one).
    const fn requested_matches_feature() -> Device {
        if cfg!(feature = "cuda") {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}
