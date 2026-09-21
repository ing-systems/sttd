use anyhow::{Context, Result};
use rubato::{FftFixedInOut, Resampler};

// ---------------------------------------------------------------------------
// Signal primitives: channel downmix + resampler construction
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

/// Build an hw_rate -> `target_rate` resampler with a fixed ~10 ms input
/// chunk size, so its output lands on a round 10 ms frame count at
/// `target_rate` (480 for 48 kHz, 160 for 16 kHz). Shared by both
/// `FrameProcessor` resamplers (`resampler_hw48`, and the `DirectTo16k`
/// variant of `whisper_resample`) — one formula, so their input chunk sizes
/// agree whenever that's mathematically possible (see `WhisperResample`).
pub(crate) fn resampler_for(
    hw_rate: u32,
    target_rate: u32,
) -> Result<(FftFixedInOut<f32>, usize)> {
    let desired_in = ((hw_rate as f64 / 100.0).round() as usize).max(1);
    let resampler = FftFixedInOut::<f32>::new(hw_rate as usize, target_rate as usize, desired_in, 1)
        .with_context(|| format!("failed to create rubato resampler (hw -> {target_rate})"))?;
    let input_frames = resampler.input_frames_next();
    Ok((resampler, input_frames))
}

/// How the whisper branch gets from hw_rate down to 16 kHz.
///
/// `DirectTo16k` is one resampling hop instead of two (hw -> 48k -> 16k): the
/// whisper branch never uses the 48 kHz intermediate for anything (it
/// exists only so RNNoise, which is fixed at 48 kHz, can run for the VAD
/// branch), so paying for it was pure waste. It's only valid when a
/// hw_rate -> 16 kHz resampler consumes the *same* input chunk size as the
/// hw_rate -> 48 kHz one (verified for 16k/44.1k/48k/96k hw rates in
/// `dual_resamplers_agree_on_input_frames`) — otherwise the two branches
/// would desync in time. `Via48k` is the original hw->48k->16k path,
/// reusing the VAD branch's already-resampled 48 kHz frame, kept as a
/// fallback for any hw rate where that assumption doesn't hold.
pub(crate) enum WhisperResample {
    DirectTo16k {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
    Via48k {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, WHISPER_RATE};
    use nnnoiseless::DenoiseState;

    /// Test oracle for the raw hw-rate chunk size `resampler_for` settles on:
    /// a 10 ms slice of the hardware rate.
    fn expected_input_chunk_for(hw_rate: u32) -> usize {
        ((hw_rate as f64 * FRAME_48K as f64 / DENOISE_RATE as f64).round() as usize).max(1)
    }

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(downmix_to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn chunk_sizes_yield_480_out() {
        assert_eq!(expected_input_chunk_for(44_100), 441);
        assert_eq!(expected_input_chunk_for(48_000), 480);
        assert_eq!(expected_input_chunk_for(16_000), 160);
        assert_eq!(expected_input_chunk_for(96_000), 960);
    }

    #[test]
    fn nnnoiseless_frame_size_is_480() {
        assert_eq!(DenoiseState::FRAME_SIZE, FRAME_48K);
    }

    #[test]
    fn rubato_produces_480_for_common_rates() {
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let chunk = expected_input_chunk_for(hw);
            let resampler = FftFixedInOut::<f32>::new(hw as usize, 48_000, chunk, 1).unwrap();
            assert_eq!(resampler.output_frames_next(), 480, "hw={hw}");
        }
    }

    #[test]
    fn rubato_48k_to_16k_produces_160() {
        let resampler = FftFixedInOut::<f32>::new(48_000, 16_000, FRAME_48K, 1).unwrap();
        assert_eq!(resampler.input_frames_next(), FRAME_48K);
        assert_eq!(resampler.output_frames_next(), FRAME_16K);
    }

    #[test]
    fn dual_resamplers_agree_on_input_frames() {
        // `FrameProcessor::new` only takes the `WhisperResample::DirectTo16k`
        // (single-hop hw->16k) path when a hw->16k resampler consumes the
        // same input chunk size as the hw->48k one, since both are fed from
        // the same raw stream in lockstep. Confirms that holds for common
        // rates, and that both branches still land on a round 10 ms frame
        // count (480 @ 48k, 160 @ 16k) — i.e. that `resampler_for`'s single
        // formula (`hw_rate / 100`) is the right one for both targets.
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let (r48, in48) = resampler_for(hw, DENOISE_RATE).unwrap();
            let (r16, in16) = resampler_for(hw, WHISPER_RATE).unwrap();
            assert_eq!(in48, in16, "hw={hw}: input chunk sizes must match");
            assert_eq!(r48.output_frames_next(), FRAME_48K, "hw={hw}");
            assert_eq!(r16.output_frames_next(), FRAME_16K, "hw={hw}");
        }
    }
}
