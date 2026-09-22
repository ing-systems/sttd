use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use crossbeam_channel::Sender;

use crate::dsp::{
    downmix_i16_to_mono_with_channel, downmix_to_mono_with_channel,
    downmix_u16_to_mono_with_channel, resolve_channel,
};

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
/// must never block or allocate beyond the mono buffer.
#[derive(Debug, Clone, Default)]
pub(crate) struct StreamTelemetry {
    /// Chunks dropped because the bounded `raw_rx` was full (capture outruns
    /// resampling; see `RAW_QUEUE_BOUND`).
    pub(crate) dropped_chunks: Arc<AtomicUsize>,
    /// `err_fn` invocations (xruns, backend errors).
    pub(crate) errors: Arc<AtomicUsize>,
    /// Set on `DeviceNotAvailable`: the main loop rebuilds the stream.
    pub(crate) device_gone: Arc<AtomicBool>,
}

/// Build the cpal input stream for the device's native sample format.
///
/// The callback only selects/mixes to mono and `try_send`s to `raw_tx`: it
/// never calls `unwrap`/`expect`, never blocks, and does no heavy work. On a
/// full queue the chunk is dropped and counted (`dropped_chunks`) instead of
/// stalling the audio thread — the main loop reports the total.
pub(crate) fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    channel: Option<usize>,
    sample_format: cpal::SampleFormat,
    raw_tx: Sender<Vec<f32>>,
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

    match sample_format {
        cpal::SampleFormat::F32 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono_with_channel(data, channels, channel);
                    if let Err(e) = tx.try_send(mono) {
                        // RT thread: count only, never log/allocate. The main
                        // loop reports `dropped_chunks`.
                        dropped.fetch_add(1, Ordering::Relaxed);
                        let _ = e;
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    // Single alloc: convert + downmix in one pass.
                    let mono = downmix_i16_to_mono_with_channel(data, channels, channel);
                    if let Err(e) = tx.try_send(mono) {
                        dropped.fetch_add(1, Ordering::Relaxed);
                        let _ = e;
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    // Single alloc: convert + downmix in one pass.
                    let mono = downmix_u16_to_mono_with_channel(data, channels, channel);
                    if let Err(e) = tx.try_send(mono) {
                        dropped.fetch_add(1, Ordering::Relaxed);
                        let _ = e;
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
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
}
