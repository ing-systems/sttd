use anyhow::{Context, Result};
use cpal::traits::DeviceTrait;
use crossbeam_channel::Sender;

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Downmix an interleaved multi-channel buffer to mono by averaging.
pub(crate) fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
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

/// Install a Ctrl-C handler that flips `running` to false.
///
/// Uses the `ctrlc` crate so the main loop can flush trailing speech before
/// exiting. Failures are logged, never fatal.
pub(crate) fn ctrlc_handler(cb: impl Fn() + Send + 'static) -> Result<()> {
    ctrlc::set_handler(cb).context("failed to install Ctrl-C handler")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(downmix_to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }
}
