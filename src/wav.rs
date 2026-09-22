use std::path::Path;

use anyhow::{Context, Result};

use crate::dsp::{downmix_to_mono, downmix_to_mono_with_channel, pick_louder_channel, ChannelSel};

// ---------------------------------------------------------------------------
// WAV I/O (utterance dumps + `--file` mode) via `hound`.
// ---------------------------------------------------------------------------

pub(crate) fn write_wav_mono16(path: &Path, pcm_f32: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for &s in pcm_f32 {
        let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(clamped)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    Ok(())
}

#[derive(Debug)]
pub(crate) struct WavAudio {
    pub(crate) sample_rate: u32,
    /// Mono, f32 in [-1, 1].
    pub(crate) samples: Vec<f32>,
}

/// WAV reader: integer PCM (8/16/24/32-bit) or IEEE-float32, any sample
/// rate/channel count, honoring `--channel`: `mix` averages, `pick` takes
/// one channel verbatim, `auto` keeps the loudest channel (highest RMS,
/// mix fallback on silence/mono). Anything else (e.g. compressed WAV) is
/// rejected with a message suggesting an `ffmpeg` conversion.
pub(crate) fn read_wav_mono_f32_with_channel(path: &Path, sel: ChannelSel) -> Result<WavAudio> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, bits @ (8 | 16 | 24 | 32)) => {
            let scale = (1u32 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.with_context(|| format!("failed to decode {}", path.display()))
                        .map(|v| v as f32 / scale)
                })
                .collect::<Result<Vec<_>>>()?
        }
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.with_context(|| format!("failed to decode {}", path.display())))
            .collect::<Result<Vec<_>>>()?,
        (fmt, bits) => anyhow::bail!(
            "{}: unsupported WAV format ({fmt:?}, bitsPerSample={bits}); convert with e.g. \
             `ffmpeg -i in.wav -ac 1 -sample_fmt s16 out.wav`",
            path.display()
        ),
    };
    let samples = match sel {
        ChannelSel::Mix => downmix_to_mono(&interleaved, channels),
        ChannelSel::Pick(i) => {
            crate::dsp::resolve_channel(Some(i), channels).with_context(|| {
                format!(
                    "--channel={i} out of range for {channels}-channel file {}",
                    path.display()
                )
            })?;
            downmix_to_mono_with_channel(&interleaved, channels, Some(i))
        }
        ChannelSel::Auto => match pick_louder_channel(&interleaved, channels) {
            Some(i) => {
                log::info!(
                    "auto-channel: picked channel {i} of {channels} for {}",
                    path.display()
                );
                downmix_to_mono_with_channel(&interleaved, channels, Some(i))
            }
            None => downmix_to_mono(&interleaved, channels),
        },
    };
    Ok(WavAudio {
        sample_rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WHISPER_RATE;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sttd_test_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn wav_roundtrip_mono16() {
        let dir = test_dir("roundtrip");
        let path = dir.join("roundtrip.wav");
        let pcm: Vec<f32> = (0..1600)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        write_wav_mono16(&path, &pcm, WHISPER_RATE).unwrap();
        let wav = read_wav_mono_f32_with_channel(&path, ChannelSel::Mix).unwrap();
        assert_eq!(wav.sample_rate, WHISPER_RATE);
        assert_eq!(wav.samples.len(), pcm.len());
        for (a, b) in wav.samples.iter().zip(pcm.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_reads_stereo_i16_downmixed() {
        let dir = test_dir("stereo");
        let path = dir.join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: WHISPER_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..160 {
            writer.write_sample(16384i16).unwrap(); // L: 0.5
            writer.write_sample(-16384i16).unwrap(); // R: -0.5
        }
        writer.finalize().unwrap();
        let wav = read_wav_mono_f32_with_channel(&path, ChannelSel::Mix).unwrap();
        assert_eq!(wav.sample_rate, WHISPER_RATE);
        assert_eq!(wav.samples.len(), 160);
        for s in wav.samples {
            assert!(s.abs() < 1e-6, "{s}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_auto_channel_picks_louder() {
        use crate::dsp::ChannelSel;
        let dir = test_dir("autoch");
        let path = dir.join("auto.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: WHISPER_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..160 {
            writer.write_sample(3277i16).unwrap(); // L: ~0.1 quiet
            writer.write_sample(16384i16).unwrap(); // R: 0.5 loud
        }
        writer.finalize().unwrap();
        // Auto keeps the loud (R) channel verbatim instead of averaging.
        let wav = read_wav_mono_f32_with_channel(&path, ChannelSel::Auto).unwrap();
        assert_eq!(wav.samples.len(), 160);
        for s in wav.samples {
            assert!((s - 0.5).abs() < 1e-3, "{s}");
        }
        // Explicit pick still works.
        let wav = read_wav_mono_f32_with_channel(&path, ChannelSel::Pick(0)).unwrap();
        for s in wav.samples {
            assert!((s - 0.1).abs() < 1e-3, "{s}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_reads_float32() {
        let dir = test_dir("float32");
        let path = dir.join("float.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: WHISPER_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..160 {
            writer.write_sample(0.25f32).unwrap();
        }
        writer.finalize().unwrap();
        let wav = read_wav_mono_f32_with_channel(&path, ChannelSel::Mix).unwrap();
        assert_eq!(wav.samples.len(), 160);
        for s in wav.samples {
            assert!((s - 0.25).abs() < 1e-6, "{s}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_rejects_truncated_file() {
        let dir = test_dir("truncated");
        let path = dir.join("truncated.wav");
        let pcm = vec![0.1f32; 160];
        write_wav_mono16(&path, &pcm, WHISPER_RATE).unwrap();
        // Cut the file mid-data so the data chunk is truncated.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 100);
        std::fs::write(&path, &bytes).unwrap();
        let res = read_wav_mono_f32_with_channel(&path, ChannelSel::Mix);
        assert!(res.is_err(), "truncated file must fail");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wav_rejects_non_wav() {
        let dir = test_dir("nonwav");
        let path = dir.join("empty.wav");
        std::fs::write(&path, b"not a wav file").unwrap();
        let res = read_wav_mono_f32_with_channel(&path, ChannelSel::Mix);
        assert!(res.is_err(), "non-WAV must fail");
        std::fs::remove_dir_all(&dir).ok();
    }
}
