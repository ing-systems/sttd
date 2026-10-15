//! Inference device selection (CPU or CUDA) for the ONNX Runtime sessions.
//!
//! One owner for the whole policy: what `--gpu` resolves to ([`resolve`]),
//! whether a CUDA session is worth attempting ([`cuda_available`]), how the
//! provider is registered ([`apply`]), and what happens when a CUDA load does
//! not work out ([`commit_with_cpu_fallback`]). The engines supply their own
//! session settings and nothing else. It also owns what `--threads` resolves
//! to ([`resolve_n_threads`], [`auto_threads`]), which every `ort` user shares.
//!
//! The GPU path exists only in a `cuda` build. Without that Cargo feature the
//! `Gpu` variant stays in the enum (callers match on it) but nothing can
//! produce it, and [`apply`] hands back the builder unchanged.

use ort::session::builder::SessionBuilder;

use crate::config::DEFAULT_THREADS;

/// The device an ONNX Runtime session runs on. `Copy`, so callers pass it by
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Gpu` is never built without the `cuda` feature.
pub(crate) enum Device {
    /// Default CPU execution provider (no `with_execution_providers` call).
    Cpu,
    /// CUDA execution provider on device 0, TF32 off.
    Gpu,
}

impl Device {
    /// Is this the GPU? For ready lines and log fields, which report the device
    /// a session actually got rather than the one that was asked for.
    pub(crate) fn is_gpu(self) -> bool {
        matches!(self, Device::Gpu)
    }
}

/// Resolve the device a run uses, from the `--gpu` flag.
///
/// Returns `Device::Gpu` only when the caller asked for the GPU *and* the
/// binary was built with the `cuda` feature. A `--gpu` request in a non-CUDA
/// build is not an error: it warns here and the run goes on with CPU sessions.
///
/// Whether a CUDA build has a usable device is a separate question, asked per
/// session by [`commit_with_cpu_fallback`].
pub(crate) fn resolve(use_gpu: bool) -> Device {
    let device = if use_gpu && cfg!(feature = "cuda") {
        Device::Gpu
    } else {
        Device::Cpu
    };
    if use_gpu && !device.is_gpu() {
        log::warn!(
            "--gpu is set but this build has no CUDA support (build with --features cuda); running on CPU"
        );
    }
    device
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
/// Runtime default). `Device::Gpu` registers CUDA on device 0 with TF32 off,
/// which keeps results reproducible across runs. A registration failure is a
/// warn, not an error: the builder is returned unchanged so the session still
/// loads on the CPU.
pub(crate) fn apply(builder: SessionBuilder, device: Device) -> SessionBuilder {
    match device {
        Device::Cpu => builder,
        #[cfg(feature = "cuda")]
        Device::Gpu => {
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
        // No `cuda` feature: the session cannot use the GPU. `resolve`
        // never returns `Gpu` here, so this only guards a direct caller.
        #[cfg(not(feature = "cuda"))]
        Device::Gpu => {
            log::warn!("CUDA requested but this build has no `cuda` feature; using CPU");
            builder
        }
    }
}

/// Commit a session on `device`, falling back to the CPU rather than failing.
///
/// `build` receives the device to place the session on, and must be able to
/// build it twice: a `SessionBuilder` is consumed by the commit, so the CPU
/// retry needs a fresh one. `what` names the session in the warnings.
///
/// There are two ways the GPU can fail and neither of them stops the run:
///
/// - this build or this machine has no usable device. That is probed *before*
///   the first attempt ([`cuda_available`]), so a CUDA build on a machine
///   without a GPU never reaches the CUDA load path at all;
/// - a CUDA session that will not commit: out of memory, a driver mismatch, or
///   an op the provider has no kernel for. Retried once on the CPU.
///
/// Both warn rather than fail, because a missing GPU must not stop
/// transcription. The two are separate decisions, so the probe lives here and
/// the retry in [`commit_with_cpu_retry`], which takes the device as given.
pub(crate) fn commit_with_cpu_fallback<T, E, F>(
    device: Device,
    what: &str,
    build: F,
) -> Result<T, E>
where
    F: Fn(Device) -> Result<T, E>,
    E: From<ort::Error> + std::fmt::Display,
{
    if device.is_gpu() && !cuda_available() {
        log::warn!("no usable CUDA device for {what}; running it on the CPU");
        return commit_with_cpu_retry(Device::Cpu, what, build);
    }
    commit_with_cpu_retry(device, what, build)
}

/// The second half of [`commit_with_cpu_fallback`]: build on `device`, and if
/// that was the GPU and it failed, build once more on the CPU.
///
/// The retry is not itself retried. A load that fails on the CPU has nothing
/// left to fall back to, so its error is the real one and it is returned.
fn commit_with_cpu_retry<T, E, F>(device: Device, what: &str, build: F) -> Result<T, E>
where
    F: Fn(Device) -> Result<T, E>,
    E: From<ort::Error> + std::fmt::Display,
{
    match build(device) {
        Ok(session) => Ok(session),
        Err(e) if device.is_gpu() => {
            log::warn!("cannot load {what} on CUDA ({e}); retrying on the CPU");
            build(Device::Cpu)
        }
        Err(e) => Err(e),
    }
}

/// Query physical cores for `--threads 0` (auto). Falls back to
/// `available_parallelism` and then `DEFAULT_THREADS` when the query fails.
pub(crate) fn auto_threads() -> i32 {
    pick_auto_threads(
        num_cpus::get_physical(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    )
}

/// Pick an auto thread count from a physical-core count and a logical-core
/// count (`0` = query failed). Prefers physical (ORT `intra_op` guidance:
/// one thread per physical core); falls back to logical, then
/// `DEFAULT_THREADS`. Pure over its inputs so tests don't touch the
/// scheduler. Always >= 1.
pub(crate) fn pick_auto_threads(physical: usize, logical: usize) -> i32 {
    let n = if physical > 0 {
        physical
    } else if logical > 0 {
        logical
    } else {
        DEFAULT_THREADS.max(1) as usize
    };
    (n.min(i32::MAX as usize) as i32).max(1)
}

/// Resolve a `--threads` value: `0` means auto, anything else is used as-is.
/// Pure over its inputs so tests don't touch the scheduler.
pub(crate) fn resolve_n_threads(requested: i32, auto: i32) -> i32 {
    if requested == 0 {
        auto.max(1)
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_follows_gpu_flag_and_feature() {
        assert_eq!(resolve(false), Device::Cpu);
        assert_eq!(resolve(true), resolve_matches_feature());
    }

    /// Without the `cuda` feature the GPU flag cannot select CUDA; with it, the
    /// flag selects CUDA regardless of whether a GPU is present (a missing GPU
    /// is a `cuda_available` question, and `commit_with_cpu_fallback` asks it
    /// per session).
    const fn resolve_matches_feature() -> Device {
        if cfg!(feature = "cuda") {
            Device::Gpu
        } else {
            Device::Cpu
        }
    }

    /// The helper takes an `Fn`, so a test that wants to record the calls
    /// records them through a `RefCell`. The closure's borrow ends when the
    /// helper returns, which is why `into_inner` works afterwards.
    use std::cell::RefCell;

    /// A CPU request is never rewritten into a GPU attempt, and never retried:
    /// the fallback is a GPU-only concern, so the closure runs exactly once and
    /// a failure is the caller's own error.
    #[test]
    fn a_cpu_request_is_left_alone() {
        let seen = RefCell::new(Vec::new());
        let result: Result<(), ort::Error> = commit_with_cpu_retry(Device::Cpu, "s", |device| {
            seen.borrow_mut().push(device);
            Err(ort::Error::new("no kernel"))
        });
        assert_eq!(seen.into_inner(), vec![Device::Cpu]);
        assert!(result.is_err());
    }

    /// A GPU commit that fails is retried on the CPU, so a session that will not
    /// load on the provider still loads.
    #[test]
    fn a_failed_gpu_attempt_is_retried_on_cpu() {
        let seen = RefCell::new(Vec::new());
        let result: Result<&str, ort::Error> = commit_with_cpu_retry(Device::Gpu, "s", |device| {
            seen.borrow_mut().push(device);
            match device {
                Device::Gpu => Err(ort::Error::new("no kernel")),
                Device::Cpu => Ok("loaded"),
            }
        });
        assert_eq!(seen.into_inner(), vec![Device::Gpu, Device::Cpu]);
        assert_eq!(result.expect("the CPU retry must succeed"), "loaded");
    }

    /// A GPU commit that succeeds is not retried: the point of the retry is a
    /// failed load, not a second opinion.
    #[test]
    fn a_successful_gpu_attempt_is_not_retried() {
        let seen = RefCell::new(Vec::new());
        let result: Result<&str, ort::Error> = commit_with_cpu_retry(Device::Gpu, "s", |device| {
            seen.borrow_mut().push(device);
            Ok("loaded")
        });
        assert_eq!(seen.into_inner(), vec![Device::Gpu]);
        assert_eq!(result.expect("the GPU attempt must stand"), "loaded");
    }

    /// The retry does not retry: a load that fails on the CPU has nothing left
    /// to fall back to, so its error is the one the caller sees.
    #[test]
    fn a_failed_cpu_retry_is_the_final_error() {
        let seen = RefCell::new(Vec::new());
        let result: Result<(), ort::Error> = commit_with_cpu_retry(Device::Gpu, "s", |device| {
            seen.borrow_mut().push(device);
            Err(ort::Error::new(if device.is_gpu() {
                "gpu failed"
            } else {
                "cpu failed"
            }))
        });
        assert_eq!(seen.into_inner(), vec![Device::Gpu, Device::Cpu]);
        assert!(
            result
                .expect_err("both attempts failed")
                .to_string()
                .contains("cpu failed"),
            "the CPU error must be the one reported"
        );
    }

    /// With no usable device the GPU load is never attempted at all, so the
    /// closure sees the CPU on its only call. Without the `cuda` feature
    /// `cuda_available` is a constant false, so this is the branch a CPU-only
    /// build always takes.
    #[test]
    fn a_gpu_request_without_a_device_goes_straight_to_cpu() {
        if cuda_available() {
            // This machine has a device, so the downgrade cannot be observed
            // here; `commit_with_cpu_retry` covers the other branch.
            return;
        }
        let seen = RefCell::new(Vec::new());
        let result: Result<(), ort::Error> = commit_with_cpu_fallback(Device::Gpu, "s", |device| {
            seen.borrow_mut().push(device);
            Ok(())
        });
        assert_eq!(seen.into_inner(), vec![Device::Cpu]);
        assert!(result.is_ok());
    }

    #[test]
    fn thread_count_zero_means_auto() {
        assert_eq!(DEFAULT_THREADS, 8);
        assert_eq!(resolve_n_threads(0, 16), 16);
        assert_eq!(resolve_n_threads(8, 16), 8);
        // Auto of 0/negative can never happen via `auto_threads`, but clamp
        // defensively so the model never gets 0 threads.
        assert_eq!(resolve_n_threads(0, 0), 1);
        assert!(auto_threads() >= 1);
        // Physical preferred; logical fallback; then DEFAULT_THREADS.
        assert_eq!(pick_auto_threads(6, 12), 6);
        assert_eq!(pick_auto_threads(0, 12), 12);
        assert_eq!(pick_auto_threads(0, 0), DEFAULT_THREADS);
    }
}
