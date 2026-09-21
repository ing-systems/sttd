use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use crossbeam_channel::Sender;

use crate::dsp::downmix_to_mono;

// ---------------------------------------------------------------------------
// Input device: discovery, selection, capture
// ---------------------------------------------------------------------------

/// Print every available input device and its default config to stdout.
pub(crate) fn print_input_devices() -> Result<()> {
    let host = cpal::default_host();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());
        match device.default_input_config() {
            Ok(cfg) => println!("{name}  ({} Hz, {} ch)", cfg.sample_rate().0, cfg.channels()),
            Err(_) => println!("{name}  (config unavailable)"),
        }
    }
    Ok(())
}

/// Pick the input device: `STTD_DEVICE` (substring match, case-insensitive)
/// if set, otherwise the system default. Without this the tool is stuck on
/// whatever the OS considers the default mic (e.g. a webcam capsule).
pub(crate) fn select_input_device(host: &cpal::Host) -> Result<cpal::Device> {
    if let Ok(wanted) = std::env::var("STTD_DEVICE") {
        let wanted_lower = wanted.to_lowercase();
        let mut devices = host
            .input_devices()
            .context("failed to enumerate input devices")?;
        let found = devices.find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&wanted_lower))
                .unwrap_or(false)
        });
        return found
            .with_context(|| format!("STTD_DEVICE={wanted:?} matched no input device (see --list-devices)"));
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
    let err_fn = |err| eprintln!("[sttd] stream error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono(data, channels);
                    if let Err(e) = tx.send(mono) {
                        eprintln!("[sttd] raw queue closed: {e}");
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
                        eprintln!("[sttd] raw queue closed: {e}");
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
                        eprintln!("[sttd] raw queue closed: {e}");
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
