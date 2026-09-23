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

/// Tensor path is the default since Phase 7 (parity proven on goldens
/// + file-mode); `STTD_BURN_TENSOR=0` (or `=false`) selects the scalar
/// fallback for debugging regressions. Pure for tests.
pub(crate) fn parse_tensor_flag(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false"))
}

/// Read once at load (never in tests: env vars are process-global).
pub(crate) fn tensor_path_enabled() -> bool {
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
/// Worker `i` is additionally pinned to the first CPU of physical core
/// `i % n_cores` (Linux sysfs topology; no SMT siblings share a worker),
/// so AVX-512-heavy workers stop migrating. On by default since Phase 10
/// (measured ~85 ms vs ~95 ms per 2.5 s utterance on the Ryzen 9 9950X);
/// `STTD_BURN_PIN=0` disables it. If `--threads` exceeds the physical
/// core count, pinning is skipped (two pinned workers must not share a
/// CPU) and logged.
pub(crate) fn size_thread_pool(n_threads: usize) {
    use std::sync::Once;
    static CAP: Once = Once::new();
    let want = n_threads.max(1);
    let pin = parse_pin_flag(std::env::var("STTD_BURN_PIN").ok().as_deref());
    CAP.call_once(|| {
        let mut cpus = if pin { physical_cpus() } else { Vec::new() };
        if !cpus.is_empty() && want > cpus.len() {
            // More workers than cores would stack two pinned threads on
            // one CPU; don't pin in that case.
            log::info!(
                "burn parakeet: STTD_BURN_PIN ignored ({want} threads > {} physical cores)",
                cpus.len()
            );
            cpus.clear();
        }
        if !cpus.is_empty() {
            log::info!("burn parakeet: pinning rayon workers to physical cores {cpus:?}");
        }
        let builder = rayon::ThreadPoolBuilder::new().num_threads(want);
        let builder = if cpus.is_empty() {
            builder
        } else {
            builder.start_handler(move |i| pin_current_thread(cpus[i % cpus.len()]))
        };
        match builder.build_global() {
            Ok(()) => log::info!("burn parakeet: rayon pool capped at {want} threads (--threads)"),
            Err(e) => log::info!("burn parakeet: rayon pool already sized, keeping it ({e})"),
        }
    });
}

/// `STTD_BURN_PIN` parsing (pure, testable): default on, `0`/`false`
/// opts out (same shape as `parse_tensor_flag`).
pub(crate) fn parse_pin_flag(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false"))
}

/// VNNI path selector (Phase 11, option (b)): `STTD_BURN_VNNI=0`/`false`
/// forces the bit-exact `vpmaddwd` i16 path even on VNNI-capable CPUs.
/// Default on: when the CPU reports `avx512vnni`, production uses the
/// requantized `vpdpbusd` u8*s8 kernels; otherwise it falls back
/// automatically. Pure for tests (env is read once at load).
pub(crate) fn parse_vnni_flag(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false"))
}

/// Read once at load (never in tests: env vars are process-global).
pub(crate) fn vnni_enabled() -> bool {
    parse_vnni_flag(std::env::var("STTD_BURN_VNNI").ok().as_deref())
}

/// First CPU of every physical core, from the Linux sysfs topology
/// (`cpuN/topology/thread_siblings_list`); falls back to `0..physical`
/// when the topology is unreadable. Sorted ascending, duplicates removed.
pub(crate) fn physical_cpus() -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for cpu in 0..num_cpus::get() {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let Ok(list) = std::fs::read_to_string(&path) else {
            break;
        };
        let first = list
            .trim()
            .split([',', '-'])
            .next()
            .and_then(|s| s.parse::<usize>().ok());
        if let Some(f) = first {
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }
    if out.is_empty() {
        out = (0..num_cpus::get_physical()).collect();
    }
    out.sort_unstable();
    out
}

fn pin_current_thread(_cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(_cpu, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_flag_parsing() {
        // Tensor is the default since Phase 7; only explicit 0/false
        // opts out to the scalar fallback.
        assert!(super::parse_tensor_flag(None));
        assert!(super::parse_tensor_flag(Some("1")));
        assert!(super::parse_tensor_flag(Some("true")));
        assert!(super::parse_tensor_flag(Some("yes")));
        assert!(!super::parse_tensor_flag(Some("0")));
        assert!(!super::parse_tensor_flag(Some("false")));
    }

    #[test]
    fn cpu_device_allocates() {
        // The handle threads through to every tensor op from Phase 2 on.
        let d = super::cpu_device();
        let _ = d;
    }

    #[test]
    fn pin_flag_parsing() {
        // Default on; only explicit 0/false opts out.
        assert!(super::parse_pin_flag(None));
        assert!(super::parse_pin_flag(Some("1")));
        assert!(super::parse_pin_flag(Some("true")));
        assert!(!super::parse_pin_flag(Some("0")));
        assert!(!super::parse_pin_flag(Some("false")));
    }

    #[test]
    fn vnni_flag_parsing() {
        // Default on; only explicit 0/false opts out (same shape as tensor/pin).
        assert!(super::parse_vnni_flag(None));
        assert!(super::parse_vnni_flag(Some("1")));
        assert!(super::parse_vnni_flag(Some("true")));
        assert!(!super::parse_vnni_flag(Some("0")));
        assert!(!super::parse_vnni_flag(Some("false")));
    }

    #[test]
    fn physical_cpus_are_unique_and_nonempty() {
        let cpus = super::physical_cpus();
        assert!(!cpus.is_empty(), "physical_cpus must fall back to cpu ids");
        let mut sorted = cpus.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(cpus, sorted, "physical_cpus must be unique and sorted");
        assert!(cpus.iter().all(|&c| c < num_cpus::get()));
    }

    #[test]
    fn cpu_is_default_without_cuda_build() {
        // Without the cuda feature every request lands on CPU.
        let d = select_device(false);
        assert_eq!(d, BurnDevice::Cpu);
    }

    #[test]
    fn rayon_cap_is_idempotent_and_effective() {
        // The pool is process-wide and one-shot. Several tests in this
        // binary use rayon (frontend, block/igemm kernels), so the first
        // `build_global` may already have happened by the time this runs;
        // what must hold either way is that repeated calls never panic and
        // never change an existing size. (The winning-first case —
        // `--threads` actually sizing the pool — is what the binary
        // relies on, and it calls `size_thread_pool` before any parallel
        // work; it is not observable from a shared test process.)
        let before = rayon::current_num_threads();
        super::size_thread_pool(4);
        // Reloads re-call this: the Once gate makes every later call a
        // no-op that must never panic, even with a different request.
        super::size_thread_pool(2);
        assert_eq!(rayon::current_num_threads(), before);
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
