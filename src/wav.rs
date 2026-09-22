use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use crate::dsp::downmix_to_mono;

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

#[derive(Debug)]
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
    let mut raw_data: Option<Vec<u8>> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        // SAFETY: pos+8 <= data.len() checked at loop head, so both 4-byte slices are exact len.
        let chunk_size =
            u32::from_le_bytes(data[pos + 4..pos + 8].try_into().expect("len checked above"))
                as usize;
        let body_start = pos + 8;
        let Some(body_end) = body_start.checked_add(chunk_size) else {
            anyhow::bail!("{}: chunk size overflows", path.display());
        };
        if body_end > data.len() {
            anyhow::bail!(
                "{}: truncated chunk (claims {chunk_size} bytes, file ends early)",
                path.display()
            );
        }

        if chunk_id == b"fmt " {
            let body = &data[body_start..body_end];
            if body.len() < 16 {
                anyhow::bail!("{}: truncated fmt chunk", path.display());
            }
            // SAFETY: body.len() >= 16 checked above, so all slices below are exact len.
            audio_format = u16::from_le_bytes(body[0..2].try_into().expect("len checked above"));
            channels = u16::from_le_bytes(body[2..4].try_into().expect("len checked above"));
            sample_rate = u32::from_le_bytes(body[4..8].try_into().expect("len checked above"));
            bits_per_sample =
                u16::from_le_bytes(body[14..16].try_into().expect("len checked above"));
        } else if chunk_id == b"data" {
            // Stash raw bytes; decode after the loop once `fmt` is known so
            // `data`-before-`fmt` order also works. Last data chunk wins.
            raw_data = Some(data[body_start..body_end].to_vec());
        }

        // Chunks are word-aligned.
        pos = body_end + (chunk_size % 2);
    }

    let raw_data = raw_data.with_context(|| format!("{}: no data chunk found", path.display()))?;
    if sample_rate == 0 {
        anyhow::bail!("{}: no fmt chunk found", path.display());
    }
    let samples = decode_pcm(&raw_data, audio_format, bits_per_sample, channels)?;
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

    fn write_wav_bytes(path: &Path, fmt_first: bool, truncate_data: bool) {
        // Minimal PCM16 mono 16kHz WAV with 160 samples, optionally with
        // data-before-fmt order or a lying (truncated) data chunk size.
        let samples = vec![0.1f32; 160];
        let mut data_bytes = Vec::new();
        for s in &samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let fmt_body: Vec<u8> = {
            let mut b = Vec::new();
            b.extend_from_slice(&1u16.to_le_bytes()); // PCM
            b.extend_from_slice(&1u16.to_le_bytes()); // mono
            b.extend_from_slice(&WHISPER_RATE.to_le_bytes());
            b.extend_from_slice(&(WHISPER_RATE * 2).to_le_bytes()); // byte rate
            b.extend_from_slice(&2u16.to_le_bytes()); // block align
            b.extend_from_slice(&16u16.to_le_bytes()); // bits
            b
        };
        let claimed = if truncate_data {
            data_bytes.len() + 100
        } else {
            data_bytes.len()
        } as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes()); // size placeholder
        out.extend_from_slice(b"WAVE");
        let mut chunk = |id: &[u8], body: &[u8], size: u32| {
            out.extend_from_slice(id);
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(body);
        };
        if fmt_first {
            chunk(b"fmt ", &fmt_body, 16);
            chunk(b"data", &data_bytes, claimed);
        } else {
            chunk(b"data", &data_bytes, claimed);
            chunk(b"fmt ", &fmt_body, 16);
        }
        if !truncate_data {
            // Only write the real body; truncated case writes short body
            // with a lying header (already done above).
        }
        std::fs::write(path, &out).unwrap();
    }

    #[test]
    fn wav_rejects_truncated_chunk() {
        let dir = std::env::temp_dir().join(format!("sttd_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.wav");
        write_wav_bytes(&path, true, true);
        let res = read_wav_mono_f32(&path);
        assert!(res.is_err(), "truncated chunk must fail");
        assert!(res.unwrap_err().to_string().contains("truncated"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wav_accepts_data_before_fmt() {
        let dir = std::env::temp_dir().join(format!("sttd_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_first.wav");
        write_wav_bytes(&path, false, false);
        let wav = read_wav_mono_f32(&path).expect("data-before-fmt must parse");
        assert_eq!(wav.sample_rate, WHISPER_RATE);
        assert_eq!(wav.samples.len(), 160);
        std::fs::remove_file(&path).ok();
    }
}
