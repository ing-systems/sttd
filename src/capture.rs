use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use crossbeam_channel::Sender;

use crate::dsp::downmix_to_mono;

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

/// Build the cpal input stream for the device's native sample format.
///
/// The callback only downmixes to mono and forwards to `raw_tx`. It never
/// calls `unwrap`/`expect`, never blocks, and does no heavy work.
pub(crate) fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    sample_format: cpal::SampleFormat,
    raw_tx: Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let err_fn = |err| log::error!("stream error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono(data, channels);
                    if let Err(e) = tx.send(mono) {
                        log::warn!("raw queue closed: {e}");
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
                    let mut tmp = Vec::with_capacity(data.len());
                    for s in data {
                        tmp.push(*s as f32 / 32768.0);
                    }
                    let mono = downmix_to_mono(&tmp, channels);
                    if let Err(e) = tx.send(mono) {
                        log::warn!("raw queue closed: {e}");
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
                    let mut tmp = Vec::with_capacity(data.len());
                    for s in data {
                        tmp.push(*s as f32 / 65535.0 * 2.0 - 1.0);
                    }
                    let mono = downmix_to_mono(&tmp, channels);
                    if let Err(e) = tx.send(mono) {
                        log::warn!("raw queue closed: {e}");
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
