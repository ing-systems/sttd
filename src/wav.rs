use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::downmix_to_mono;

// ---------------------------------------------------------------------------
// WAV I/O (utterance dumps + `--file` mode); no extra dependency.
// ---------------------------------------------------------------------------

pub(crate) fn write_wav_mono16(path: &Path, pcm_f32: &[f32], sample_rate: u32) -> Result<()> {
    let mut file =
        std::fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;

    let num_samples = pcm_f32.len() as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    let data_bytes = num_samples * 2;

    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_bytes).to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for &s in pcm_f32 {
        let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        file.write_all(&clamped.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) struct WavAudio {
    pub(crate) sample_rate: u32,
    /// Mono, f32 in [-1, 1].
    pub(crate) samples: Vec<f32>,
}

/// Minimal RIFF/WAVE reader: PCM16 or IEEE-float32, any sample rate/channel
/// count. Anything else (e.g. compressed WAV) is rejected with a message
/// suggesting an `ffmpeg` conversion.
pub(crate) fn read_wav_mono_f32(path: &Path) -> Result<WavAudio> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        anyhow::bail!("{}: not a RIFF/WAVE file", path.display());
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut samples: Option<Vec<f32>> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_size).min(data.len());
        if body_start > data.len() {
            break;
        }

        if chunk_id == b"fmt " {
            let body = &data[body_start..body_end];
            if body.len() < 16 {
                anyhow::bail!("{}: truncated fmt chunk", path.display());
            }
            audio_format = u16::from_le_bytes(body[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if chunk_id == b"data" {
            let body = &data[body_start..body_end];
            samples = Some(decode_pcm(body, audio_format, bits_per_sample, channels)?);
        }

        // Chunks are word-aligned.
        pos = body_start + chunk_size + (chunk_size % 2);
    }

    let samples = samples.with_context(|| format!("{}: no data chunk found", path.display()))?;
    if sample_rate == 0 {
        anyhow::bail!("{}: no fmt chunk found", path.display());
    }
    Ok(WavAudio { sample_rate, samples })
}

fn decode_pcm(body: &[u8], audio_format: u16, bits_per_sample: u16, channels: u16) -> Result<Vec<f32>> {
    let channels = channels.max(1) as usize;
    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => body
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => body
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        (fmt, bits) => anyhow::bail!(
            "unsupported WAV format (audioFormat={fmt}, bitsPerSample={bits}); convert with e.g. \
             `ffmpeg -i in.wav -ac 1 -sample_fmt s16 out.wav`"
        ),
    };
    Ok(downmix_to_mono(&interleaved, channels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WHISPER_RATE;

    #[test]
    fn wav_roundtrip_mono16() {
        let dir = std::env::temp_dir().join(format!("sttd_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.wav");
        let pcm: Vec<f32> = (0..1600).map(|i| ((i % 100) as f32 / 100.0) - 0.5).collect();
        write_wav_mono16(&path, &pcm, WHISPER_RATE).unwrap();
        let wav = read_wav_mono_f32(&path).unwrap();
        assert_eq!(wav.sample_rate, WHISPER_RATE);
        assert_eq!(wav.samples.len(), pcm.len());
        for (a, b) in wav.samples.iter().zip(pcm.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
