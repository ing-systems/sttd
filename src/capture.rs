use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use rtrb::Producer;

use crate::dsp::{i16_to_f32, mono_mix_frame, pick_louder_channel, resolve_channel, u16_to_f32};

/// Capture buffer requested from cpal, in frames. The backend-default size
/// can silently drop a large share of capture periods, heard as crackling.
/// Mirrors Murmure's `CAPTURE_BUFFER_FRAMES`.
pub(crate) const CAPTURE_BUFFER_FRAMES: u32 = 4096;

/// Clone of `config` asking for the fixed capture buffer.
fn fixed_buffer_config(config: &cpal::StreamConfig) -> cpal::StreamConfig {
    let mut fixed = config.clone();
    fixed.buffer_size = cpal::BufferSize::Fixed(CAPTURE_BUFFER_FRAMES);
    fixed
}

/// Build with the fixed capture buffer first, falling back to the backend
/// default when the device rejects it. `build` runs up to twice, so callbacks
/// it captures must be rebuildable (e.g. `Arc` clones) — the realtime
/// `rtrb` producer path cannot use this (see `supports_fixed_buffer`).
fn build_with_buffer_fallback<S>(
    base: &cpal::StreamConfig,
    mut build: impl FnMut(&cpal::StreamConfig) -> Result<S, cpal::BuildStreamError>,
) -> Result<S, cpal::BuildStreamError> {
    let fixed = fixed_buffer_config(base);
    build(&fixed).or_else(|e| {
        log::debug!(
            "fixed capture buffer ({CAPTURE_BUFFER_FRAMES} frames) rejected: {e}, falling back to default"
        );
        build(base)
    })
}

/// Whether `device` accepts the fixed capture buffer for `sample_format`:
/// one throwaway probe build with no-op callbacks. Lets `build_input_stream`
/// pick the winning config *before* moving the realtime `rtrb` producer into
/// its callback — `Producer` is not `Clone`, so a failed fixed-buffer build
/// could not be retried afterwards.
fn supports_fixed_buffer(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
) -> bool {
    let fixed = fixed_buffer_config(config);
    fn builds<S: cpal::SizedSample>(device: &cpal::Device, cfg: &cpal::StreamConfig) -> bool {
        device
            .build_input_stream(cfg, |_: &[S], _| {}, |_: cpal::StreamError| {}, None)
            .is_ok()
    }
    let ok = match sample_format {
        cpal::SampleFormat::F32 => builds::<f32>(device, &fixed),
        cpal::SampleFormat::I16 => builds::<i16>(device, &fixed),
        cpal::SampleFormat::U16 => builds::<u16>(device, &fixed),
        _ => false,
    };
    log::debug!(
        "capture buffer: {}",
        if ok {
            format!("fixed {CAPTURE_BUFFER_FRAMES} frames")
        } else {
            "backend default (fixed rejected)".to_string()
        }
    );
    ok
}

/// Probe the loudest input channel: open a short-lived input stream on
/// `device`/`config`, collect ~`probe_ms` of interleaved audio, and return
/// the highest-RMS channel index. Returns `None` for mono input, silence,
/// or too few samples — the caller then falls back to mix. Only used for
/// `--channel auto`; `mix`/`pick` never probe.
pub(crate) fn probe_loudest_channel(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    sample_format: cpal::SampleFormat,
    probe_ms: u64,
) -> Result<Option<usize>> {
    if channels <= 1 {
        return Ok(None);
    }
    let collected = Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
    // Temporary stream: same downmix-scale as the realtime callback in
    // `build_input_stream` (shared `i16_to_f32`/`u16_to_f32` closures), but
    // appending interleaved f32 into a Mutex Vec instead of an rtrb ring.
    // Short-lived (hundreds of ms), so a lock per callback is fine — the
    // realtime path never takes it, and never allocates.
    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_probe_stream(device, config, &collected, |s: f32| s)?,
        cpal::SampleFormat::I16 => build_probe_stream(device, config, &collected, i16_to_f32)?,
        cpal::SampleFormat::U16 => build_probe_stream(device, config, &collected, u16_to_f32)?,
        fmt => anyhow::bail!("unsupported sample format: {fmt:?}"),
    };
    use cpal::traits::StreamTrait;
    stream
        .play()
        .context("failed to start auto-channel probe")?;
    std::thread::sleep(std::time::Duration::from_millis(probe_ms.max(100)));
    drop(stream);
    let data = collected
        .lock()
        .map_err(|_| anyhow::anyhow!("probe buffer poisoned"))?;
    // Whole frames only; need at least ~50 ms to judge.
    let frames = data.len() / channels.max(1);
    if frames < 16 {
        log::warn!("auto-channel probe heard too little audio, falling back to mix");
        return Ok(None);
    }
    let usable = &data[..frames * channels];
    match pick_louder_channel(usable, channels) {
        Some(i) => {
            log::info!(
                "auto-channel: picked channel {i} of {channels} (loudest in {probe_ms} ms probe)"
            );
            Ok(Some(i))
        }
        None => {
            log::warn!("auto-channel probe heard only silence, falling back to mix");
            Ok(None)
        }
    }
}

/// The `--channel auto` probe stream for one sample format: appends each
/// callback's interleaved samples, scaled by `conv`, to `collected`; a stream
/// error clears what was collected.
fn build_probe_stream<S>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    collected: &Arc<std::sync::Mutex<Vec<f32>>>,
    conv: fn(S) -> f32,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    S: cpal::SizedSample + Send + Sync + 'static,
{
    build_with_buffer_fallback(config, |cfg| {
        let collected = Arc::clone(collected);
        let err_collected = Arc::clone(&collected);
        device.build_input_stream(
            cfg,
            move |data: &[S], _| {
                if let Ok(mut v) = collected.lock() {
                    v.extend(data.iter().map(|&s| conv(s)));
                }
            },
            move |err| {
                log::warn!("auto-channel probe stream error: {err}");
                let _ = err_collected.lock().map(|mut v| v.clear());
            },
            None,
        )
    })
}

// ---------------------------------------------------------------------------
// Input device: discovery, selection, capture
// ---------------------------------------------------------------------------

/// Print every available input device and its default config to stdout as an
/// aligned table (mirroring `--list-models`): `NAME` left-aligned, `CONFIG`
/// (`rate, channels, sample format`, or `config unavailable`), with a
/// two-character leading marker: `*` = system default input device,
/// `>` = matches `wanted` (`--device`/`STTD_DEVICE`, substring,
/// case-insensitive, same rule as `select_input_device`). A row can carry
/// both (`*>`). Enumeration order is kept so the first `>` row is what
/// selection would pick.
pub(crate) fn print_input_devices(wanted: Option<&str>) -> Result<()> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    let mut entries = Vec::new();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());
        let config = match device.default_input_config() {
            Ok(cfg) => format!(
                "{} Hz, {} ch, {:?}",
                cfg.sample_rate().0,
                cfg.channels(),
                cfg.sample_format()
            ),
            Err(_) => "config unavailable".to_string(),
        };
        // Names are the only comparable handle; duplicates (if any) share
        // the default marker.
        let is_default = default_name.as_deref() == Some(name.as_str());
        entries.push(DeviceEntry {
            name,
            config,
            is_default,
        });
    }
    print!("{}", format_input_devices(&entries, wanted));
    Ok(())
}

/// One enumerated input device for `--list-devices` display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceEntry {
    pub(crate) name: String,
    pub(crate) config: String,
    pub(crate) is_default: bool,
}

/// Case-insensitive substring match, shared by selection and display so the
/// `>` marker shows exactly what `--device` would pick.
pub(crate) fn device_name_matches(name: &str, wanted: &str) -> bool {
    name.to_lowercase().contains(&wanted.to_lowercase())
}

/// Render the `--list-devices` table: header plus one row per entry, `NAME`
/// left-aligned, leading `*`/`>` marker. Empty lists render a single
/// `no input devices found` line; a non-empty `wanted` filter that matches
/// nothing appends a `no device matches ...` hint (mirroring
/// `--list-models`' `selected: ... (not in catalog)` line).
pub(crate) fn format_input_devices(entries: &[DeviceEntry], wanted: Option<&str>) -> String {
    if entries.is_empty() {
        return "no input devices found\n".to_string();
    }
    let name_w = entries
        .iter()
        .map(|e| e.name.chars().count())
        .max()
        .unwrap_or(0)
        .max("NAME".len());
    let mut out = String::new();
    out.push_str(&format!("   {0:<1$}  CONFIG\n", "NAME", name_w));
    for e in entries {
        let is_match = wanted
            .map(|w| device_name_matches(&e.name, w))
            .unwrap_or(false);
        let marker = match (e.is_default, is_match) {
            (true, true) => "*>",
            (true, false) => "* ",
            (false, true) => " >",
            (false, false) => "  ",
        };
        out.push_str(&format!(
            "{marker} {0:<1$}  {2}\n",
            e.name, name_w, e.config
        ));
    }
    if let Some(w) = wanted {
        if !w.trim().is_empty() && !entries.iter().any(|e| device_name_matches(&e.name, w)) {
            out.push_str(&format!(
                "no device matches {w:?} (substring, case-insensitive)\n"
            ));
        }
    }
    out
}

/// Pick the input device: `device` (substring match, case-insensitive)
/// if set, otherwise the system default. Without this the tool is stuck on
/// whatever the OS considers the default mic (e.g. a webcam capsule).
pub(crate) fn select_input_device(host: &cpal::Host, wanted: Option<&str>) -> Result<cpal::Device> {
    if let Some(wanted) = wanted {
        let mut devices = host
            .input_devices()
            .context("failed to enumerate input devices")?;
        let found = devices.find(|d| {
            d.name()
                .map(|n| device_name_matches(&n, wanted))
                .unwrap_or(false)
        });
        return found.with_context(|| {
            format!("--device/STTD_DEVICE={wanted:?} matched no input device (see --list-devices)")
        });
    }
    host.default_input_device()
        .context("no default input device available")
}

/// Shared capture telemetry, written by the cpal callback / error handler and
/// read by the main loop's throttled warn line. Atomics only — the callback
/// must never block or allocate.
#[derive(Debug, Clone, Default)]
pub(crate) struct StreamTelemetry {
    /// Callbacks that overflowed the raw sample ring (capture outruns
    /// resampling; see `run::microphone::RAW_RING_SECS`).
    pub(crate) dropped_chunks: Arc<AtomicUsize>,
    /// `err_fn` invocations (xruns, backend errors).
    pub(crate) errors: Arc<AtomicUsize>,
    /// Set on `DeviceNotAvailable`: the main loop rebuilds the stream.
    pub(crate) device_gone: Arc<AtomicBool>,
}

/// Build the cpal input stream for the device's native sample format.
///
/// Real-time callback: per-sample mono mix/convert + `rtrb` push only — no
/// `Vec`, no allocation, no blocking, no logging. On a full ring the sample
/// is dropped and the callback counted once in `dropped_chunks` instead of
/// stalling the audio thread — the main loop reports the total.
pub(crate) fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    channel: Option<usize>,
    sample_format: cpal::SampleFormat,
    raw_tx: Producer<f32>,
    telemetry: &StreamTelemetry,
) -> Result<cpal::Stream> {
    // Fail fast on a bad `--channel` rather than capturing silence.
    let channel = resolve_channel(channel, channels)?;
    let dropped = Arc::clone(&telemetry.dropped_chunks);
    let errors = Arc::clone(&telemetry.errors);
    let device_gone = Arc::clone(&telemetry.device_gone);
    let err_fn = move |err: cpal::StreamError| {
        errors.fetch_add(1, Ordering::Relaxed);
        match err {
            cpal::StreamError::DeviceNotAvailable => {
                device_gone.store(true, Ordering::Release);
                log::error!("input device disconnected: {err}");
            }
            _ => log::error!("stream error: {err}"),
        }
    };

    // Fixed capture buffer when the device accepts it. Probed up front: the
    // realtime producer below moves into its callback and cannot be rebuilt
    // for a retry, unlike the probe path above.
    let fixed;
    let config: &cpal::StreamConfig = if supports_fixed_buffer(device, config, sample_format) {
        fixed = fixed_buffer_config(config);
        &fixed
    } else {
        config
    };
    // One callback body for all three sample formats: `mono_mix_frame` owns the
    // mix/pick/silence rule and `conv` owns the per-format scale, so the
    // F32/I16/U16 arms below differ only in their type arguments.
    struct RtArgs {
        channels: usize,
        channel: Option<usize>,
        dropped: Arc<AtomicUsize>,
    }
    fn build_for_format<S, C, E>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        args: RtArgs,
        conv: C,
        raw_tx: Producer<f32>,
        err_fn: E,
    ) -> Result<cpal::Stream>
    where
        S: Copy + Send + Sync + 'static + cpal::SizedSample,
        C: Fn(S) -> f32 + Send + Sync + 'static,
        E: FnMut(cpal::StreamError) + Send + Sync + 'static,
    {
        let RtArgs {
            channels,
            channel,
            dropped,
        } = args;
        let mut tx = raw_tx;
        Ok(device.build_input_stream(
            config,
            move |data: &[S], _| {
                let mut overflowed = false;
                for frame in data.chunks_exact(channels.max(1)) {
                    if tx
                        .push(mono_mix_frame(frame, channels, channel, &conv))
                        .is_err()
                    {
                        overflowed = true;
                    }
                }
                if overflowed {
                    // RT thread: count only, never log/allocate. The main loop
                    // reports `dropped_chunks`.
                    dropped.fetch_add(1, Ordering::Relaxed);
                }
            },
            err_fn,
            None,
        )?)
    }

    // Built once: only one arm runs, so each arm can move `raw_tx` (which is
    // not `Clone`) and `err_fn`.
    let args = RtArgs {
        channels,
        channel,
        dropped,
    };
    match sample_format {
        cpal::SampleFormat::F32 => {
            build_for_format::<f32, _, _>(device, config, args, |s: f32| s, raw_tx, err_fn)
        }
        cpal::SampleFormat::I16 => {
            build_for_format::<i16, _, _>(device, config, args, i16_to_f32, raw_tx, err_fn)
        }
        cpal::SampleFormat::U16 => {
            build_for_format::<u16, _, _>(device, config, args, u16_to_f32, raw_tx, err_fn)
        }
        fmt => anyhow::bail!("unsupported sample format: {fmt:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, config: &str, is_default: bool) -> DeviceEntry {
        DeviceEntry {
            name: name.to_string(),
            config: config.to_string(),
            is_default,
        }
    }

    #[test]
    fn devices_table_marks_default_only() {
        let entries = vec![
            entry("HDA Intel PCH", "48000 Hz, 2 ch, F32", true),
            entry("USB Mic", "16000 Hz, 1 ch, I16", false),
        ];
        let table = format_input_devices(&entries, None);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("NAME") && lines[0].contains("CONFIG"));
        assert!(lines[1].starts_with("* "), "line={:?}", lines[1]);
        assert!(lines[2].starts_with("  "), "line={:?}", lines[2]);
    }

    #[test]
    fn devices_table_marks_device_filter_match() {
        let entries = vec![
            entry("HDA Intel PCH", "48000 Hz, 2 ch, F32", true),
            entry("USB Mic", "16000 Hz, 1 ch, I16", false),
        ];
        let table = format_input_devices(&entries, Some("usb"));
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3, "table={table:?}");
        assert!(lines[1].starts_with("* "), "line={:?}", lines[1]);
        assert!(lines[2].starts_with(" >"), "line={:?}", lines[2]);
    }

    #[test]
    fn devices_table_combines_default_and_match() {
        let entries = vec![entry("USB Mic", "16000 Hz, 1 ch, I16", true)];
        let table = format_input_devices(&entries, Some("USB"));
        let row = table.lines().nth(1).expect("row must exist");
        assert!(row.starts_with("*>"), "row={row:?}");
    }

    #[test]
    fn devices_table_aligns_name_column() {
        let entries = vec![
            entry("a", "48000 Hz, 2 ch, F32", false),
            entry("much-longer-name", "config unavailable", false),
        ];
        let table = format_input_devices(&entries, None);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3);
        // NAME starts at the same offset and CONFIG at the same offset.
        let name_start = lines[0].find("NAME").expect("header has NAME");
        for line in lines.iter().skip(1) {
            assert_eq!(&line[..2], "  ");
            let config_at = line.find("48000").or_else(|| line.find("config"));
            assert!(config_at.is_some(), "line={line:?}");
        }
        let config_offsets: Vec<usize> = lines
            .iter()
            .skip(1)
            .map(|l| {
                l.find("48000")
                    .or_else(|| l.find("config"))
                    .expect("config must exist")
            })
            .collect();
        assert_eq!(config_offsets[0], config_offsets[1]);
        assert!(name_start == 3);
    }

    #[test]
    fn devices_table_reports_empty_and_no_match() {
        assert_eq!(format_input_devices(&[], None), "no input devices found\n");
        let entries = vec![entry("USB Mic", "16000 Hz, 1 ch, I16", false)];
        let table = format_input_devices(&entries, Some("webcam"));
        assert!(table.contains("no device matches"), "table={table:?}");
        let table = format_input_devices(&entries, Some("usb"));
        assert!(!table.contains("no device matches"), "table={table:?}");
    }

    #[test]
    fn device_name_match_is_case_insensitive_substring() {
        assert!(device_name_matches("HDA Intel PCH", "intel"));
        assert!(device_name_matches("USB Mic", "USB"));
        assert!(!device_name_matches("USB Mic", "webcam"));
    }

    #[test]
    fn fixed_buffer_config_requests_4096_frames() {
        let base = cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(48_000),
            buffer_size: cpal::BufferSize::Default,
        };
        let fixed = fixed_buffer_config(&base);
        assert_eq!(
            fixed.buffer_size,
            cpal::BufferSize::Fixed(CAPTURE_BUFFER_FRAMES)
        );
        assert_eq!(CAPTURE_BUFFER_FRAMES, 4096);
        assert_eq!(fixed.channels, 2);
        assert_eq!(fixed.sample_rate.0, 48_000);
    }

    #[test]
    fn buffer_fallback_uses_fixed_first_then_default() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let base = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16_000),
            buffer_size: cpal::BufferSize::Default,
        };
        let calls = AtomicUsize::new(0);
        let out = build_with_buffer_fallback(&base, |cfg| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                assert_eq!(
                    cfg.buffer_size,
                    cpal::BufferSize::Fixed(CAPTURE_BUFFER_FRAMES)
                );
                Err(cpal::BuildStreamError::StreamConfigNotSupported)
            } else {
                assert_eq!(cfg.buffer_size, cpal::BufferSize::Default);
                Ok(())
            }
        });
        assert!(out.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn buffer_fallback_propagates_when_both_rejected() {
        let base = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16_000),
            buffer_size: cpal::BufferSize::Default,
        };
        let out: Result<(), cpal::BuildStreamError> =
            build_with_buffer_fallback(&base, |_| Err(cpal::BuildStreamError::DeviceNotAvailable));
        assert!(matches!(
            out,
            Err(cpal::BuildStreamError::DeviceNotAvailable)
        ));
    }
}
