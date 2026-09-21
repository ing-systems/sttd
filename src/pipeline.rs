use std::collections::VecDeque;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};
use webrtc_vad::{Vad, VadMode};

use crate::config::{
    DENOISE_RATE, DEFAULT_MIN_SPEECH_RMS, FRAME_16K, FRAME_48K, RING_FRAMES, WHISPER_RATE,
    env_f32,
};
use crate::utterance::{Clip, Utterance, buffer_frame, flush_utterance};

/// Build an hw_rate -> `target_rate` resampler with a fixed ~10 ms input
/// chunk size, so its output lands on a round 10 ms frame count at
/// `target_rate` (480 for 48 kHz, 160 for 16 kHz). Shared by both
/// `FrameProcessor` resamplers (`resampler_hw48`, and the `Direct` variant
/// of `whisper_resample`) — one formula, so their input chunk sizes agree
/// whenever that's mathematically possible (see `WhisperResample`).
pub(crate) fn make_resampler(
    hw_rate: u32,
    target_rate: u32,
) -> Result<(FftFixedInOut<f32>, usize)> {
    let desired_in = ((hw_rate as f64 / 100.0).round() as usize).max(1);
    let resampler = FftFixedInOut::<f32>::new(hw_rate as usize, target_rate as usize, desired_in, 1)
        .with_context(|| format!("failed to create rubato resampler (hw -> {target_rate})"))?;
    let input_frames = resampler.input_frames_next();
    Ok((resampler, input_frames))
}

// ---------------------------------------------------------------------------
// Frame processing: denoise/VAD branch + whisper branch + utterance buffer
// ---------------------------------------------------------------------------

/// Owns all per-frame DSP/VAD/buffering state, including the hw-rate ->
/// 48 kHz / hw-rate -> 16 kHz resamplers (previously built and driven by the
/// caller). Fed one raw hw-rate chunk at a time (`input_frames()` samples)
/// by either the live capture loop or file mode, so both share identical
/// resampling and utterance-detection behavior.
pub(crate) struct FrameProcessor {
    // hw_rate -> 48 kHz (RNNoise's required rate). Always used to feed the
    // VAD branch, since denoise needs 48 kHz; its input chunk size
    // (`hw_input_frames`) is the raw hw-rate frame size every caller must
    // feed `process_frame` with.
    resampler_hw48: FftFixedInOut<f32>,
    resample_hw48_in: Vec<Vec<f32>>,
    resample_hw48_out: Vec<Vec<f32>>,
    hw_input_frames: usize,

    // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
    resampler_vad: FftFixedInOut<f32>,
    resample_vad_in: Vec<Vec<f32>>,
    resample_vad_out: Vec<Vec<f32>>,

    /// Whisper branch resampling: a direct hw_rate -> 16 kHz hop when the
    /// hw rate allows it (see `WhisperResample`), avoiding the redundant
    /// 48 kHz intermediate whisper's (non-denoised) signal never needed.
    whisper_resample: WhisperResample,

    denoise: Box<DenoiseState<'static>>,
    denoise_in: Vec<f32>,
    denoise_out: Vec<f32>,
    clean48_vad: Vec<f32>,

    vad: Vad,
    vad_i16: Vec<i16>,

    /// Rolling ~300 ms of whisper-branch audio, prepended to an utterance
    /// the instant it opens so onsets aren't clipped by VAD/onset latency.
    preroll: VecDeque<f32>,
    utterance: Utterance,
    /// `STTD_MIN_RMS` override (or `DEFAULT_MIN_SPEECH_RMS`), read once at
    /// startup and reused for every utterance.
    min_speech_rms: f32,

    whisper_tx: Sender<Clip>,
}

/// How the whisper branch gets from hw_rate down to 16 kHz.
///
/// `Direct` is one resampling hop instead of two (hw -> 48k -> 16k): the
/// whisper branch never uses the 48 kHz intermediate for anything (it
/// exists only so RNNoise, which is fixed at 48 kHz, can run for the VAD
/// branch), so paying for it was pure waste. It's only valid when a
/// hw_rate -> 16 kHz resampler consumes the *same* input chunk size as the
/// hw_rate -> 48 kHz one (verified for 16k/44.1k/48k/96k hw rates in
/// `dual_resamplers_agree_on_input_frames`) — otherwise the two branches
/// would desync in time. `Chained` is the original hw->48k->16k path,
/// reusing the VAD branch's already-resampled 48 kHz frame, kept as a
/// fallback for any hw rate where that assumption doesn't hold.
enum WhisperResample {
    Direct {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
    Chained {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
}

impl FrameProcessor {
    pub(crate) fn new(hw_rate: u32, whisper_tx: Sender<Clip>) -> Result<Self> {
        let (resampler_hw48, hw_input_frames) = make_resampler(hw_rate, DENOISE_RATE)
            .context("failed to create rubato resampler (hw -> 48k)")?;
        eprintln!(
            "[sttd] rubato: {hw_rate} -> {DENOISE_RATE}, in={hw_input_frames} out={}",
            resampler_hw48.output_frames_next()
        );
        let resample_hw48_in = resampler_hw48.input_buffer_allocate(true);
        let resample_hw48_out = resampler_hw48.output_buffer_allocate(true);

        let resampler_vad = FftFixedInOut::<f32>::new(
            DENOISE_RATE as usize,
            WHISPER_RATE as usize,
            FRAME_48K,
            1,
        )
        .context("failed to create rubato resampler (48k -> 16k, vad branch)")?;
        debug_assert_eq!(resampler_vad.input_frames_next(), FRAME_48K);
        debug_assert_eq!(resampler_vad.output_frames_next(), FRAME_16K);
        let resample_vad_in = resampler_vad.input_buffer_allocate(true);
        let resample_vad_out = resampler_vad.output_buffer_allocate(true);

        // Try a direct hw -> 16k hop for the whisper branch; only usable
        // when it consumes the same-sized raw hw chunk as the hw -> 48k
        // resampler above, since both are fed from the same raw stream in
        // lockstep (see `WhisperResample`).
        let whisper_resample = match make_resampler(hw_rate, WHISPER_RATE) {
            Ok((resampler, hw16_input_frames)) if hw16_input_frames == hw_input_frames => {
                debug_assert_eq!(resampler.output_frames_next(), FRAME_16K);
                let in_buf = resampler.input_buffer_allocate(true);
                let out_buf = resampler.output_buffer_allocate(true);
                WhisperResample::Direct {
                    resampler,
                    in_buf,
                    out_buf,
                }
            }
            other => {
                match &other {
                    Ok((_, hw16_input_frames)) => eprintln!(
                        "[sttd] hw->16k and hw->48k resamplers disagree on input chunk size \
                         ({hw16_input_frames} vs {hw_input_frames} samples) at {hw_rate} Hz; \
                         falling back to hw->48k->16k for the whisper branch"
                    ),
                    Err(e) => eprintln!(
                        "[sttd] failed to build a direct hw->16k resampler ({e}); falling back \
                         to hw->48k->16k for the whisper branch"
                    ),
                }
                let resampler = FftFixedInOut::<f32>::new(
                    DENOISE_RATE as usize,
                    WHISPER_RATE as usize,
                    FRAME_48K,
                    1,
                )
                .context("failed to create rubato resampler (48k -> 16k, whisper branch)")?;
                debug_assert_eq!(resampler.input_frames_next(), FRAME_48K);
                debug_assert_eq!(resampler.output_frames_next(), FRAME_16K);
                let in_buf = resampler.input_buffer_allocate(true);
                let out_buf = resampler.output_buffer_allocate(true);
                WhisperResample::Chained {
                    resampler,
                    in_buf,
                    out_buf,
                }
            }
        };

        let mut vad =
            Vad::new(WHISPER_RATE as i32).map_err(|_| anyhow::anyhow!("invalid VAD rate"))?;
        // Aggressive (not VeryAggressive): with pre-roll (1.4) and the fix
        // that stops discarding mid-utterance non-speech (1.1), the cost of
        // a false positive is a few extra ms of silence, while a false
        // negative here used to mean lost phonemes.
        vad.fvad_set_mode(VadMode::Aggressive)
            .map_err(|_| anyhow::anyhow!("failed to set VAD mode"))?;

        Ok(Self {
            resampler_hw48,
            resample_hw48_in,
            resample_hw48_out,
            hw_input_frames,
            resampler_vad,
            resample_vad_in,
            resample_vad_out,
            whisper_resample,
            denoise: DenoiseState::new(),
            denoise_in: vec![0.0f32; FRAME_48K],
            denoise_out: vec![0.0f32; FRAME_48K],
            clean48_vad: vec![0.0f32; FRAME_48K],
            vad,
            vad_i16: vec![0i16; FRAME_16K],
            preroll: VecDeque::with_capacity(RING_FRAMES * FRAME_16K),
            utterance: Utterance::default(),
            min_speech_rms: env_f32("STTD_MIN_RMS", DEFAULT_MIN_SPEECH_RMS),
            whisper_tx,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) fn input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Process one raw hw-rate chunk (`input_frames()` samples).
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        // ---- hw -> 48 kHz: shared basis for the VAD branch (denoise needs
        // 48 kHz), and for the whisper branch too when using the `Chained`
        // fallback -----------------------------------------------------------
        self.resample_hw48_in[0][..frame_hw_raw.len()].copy_from_slice(frame_hw_raw);
        let frame48_produced = match self.resampler_hw48.process_into_buffer(
            &self.resample_hw48_in,
            &mut self.resample_hw48_out,
            None,
        ) {
            Ok((_, produced)) => produced,
            Err(e) => {
                eprintln!("[sttd] hw->48k resample error: {e}");
                return;
            }
        };
        if frame48_produced != FRAME_48K {
            // Denoise's frame API is fixed-size; an unexpected chunk count
            // (resampler startup/edge case) would desync it. Skip this
            // frame rather than feed denoise a mismatched buffer.
            eprintln!(
                "[sttd] hw->48k produced {frame48_produced} samples, expected {FRAME_48K}; \
                 skipping frame"
            );
            return;
        }
        // Owned: needed below both for denoise and, in the `Chained` case,
        // for the whisper branch's own resample call, which would otherwise
        // alias this borrow of `self`.
        let frame_48k_raw = self.resample_hw48_out[0][..FRAME_48K].to_vec();

        // ---- VAD branch: denoise, then resample to 16 kHz -----------------
        for (i, s) in frame_48k_raw.iter().enumerate() {
            self.denoise_in[i] = (s * 32768.0).clamp(-32768.0, 32767.0);
        }
        self.denoise
            .process_frame(&mut self.denoise_out, &self.denoise_in);
        for (i, s) in self.denoise_out.iter().enumerate() {
            self.clean48_vad[i] = s / 32768.0;
        }
        self.resample_vad_in[0][..FRAME_48K].copy_from_slice(&self.clean48_vad);
        let vad_produced = match self.resampler_vad.process_into_buffer(
            &self.resample_vad_in,
            &mut self.resample_vad_out,
            None,
        ) {
            Ok((_, produced)) => produced,
            Err(e) => {
                eprintln!("[sttd] vad resample error: {e}");
                return;
            }
        };

        // ---- whisper branch: raw (no denoise) -------------------------------
        let whisper16 = match &mut self.whisper_resample {
            WhisperResample::Direct {
                resampler,
                in_buf,
                out_buf,
            } => {
                in_buf[0][..frame_hw_raw.len()].copy_from_slice(frame_hw_raw);
                match resampler.process_into_buffer(in_buf, out_buf, None) {
                    Ok((_, produced)) => out_buf[0][..produced].to_vec(),
                    Err(e) => {
                        eprintln!("[sttd] whisper resample error: {e}");
                        return;
                    }
                }
            }
            WhisperResample::Chained {
                resampler,
                in_buf,
                out_buf,
            } => {
                in_buf[0][..FRAME_48K].copy_from_slice(&frame_48k_raw);
                match resampler.process_into_buffer(in_buf, out_buf, None) {
                    Ok((_, produced)) => out_buf[0][..produced].to_vec(),
                    Err(e) => {
                        eprintln!("[sttd] whisper resample error: {e}");
                        return;
                    }
                }
            }
        };

        // ---- VAD decision --------------------------------------------------
        // `vad_i16` is pre-sized for the steady-state FRAME_16K case; guard
        // against a resampler startup/edge call producing a different count
        // instead of panicking on an out-of-bounds write.
        if self.vad_i16.len() < vad_produced {
            self.vad_i16.resize(vad_produced, 0);
        }
        for (i, s) in self.resample_vad_out[0][..vad_produced].iter().enumerate() {
            self.vad_i16[i] = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        }
        let is_speech = match self.vad.is_voice_segment(&self.vad_i16[..vad_produced]) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("[sttd] VAD rejected frame length");
                false
            }
        };

        // ---- utterance buffering --------------------------------------------
        let should_flush = buffer_frame(&mut self.utterance, &self.preroll, &whisper16, is_speech);
        if should_flush {
            self.flush();
        }

        // ---- pre-roll ring update -------------------------------------------
        self.preroll.extend(whisper16.iter().copied());
        while self.preroll.len() > RING_FRAMES * FRAME_16K {
            self.preroll.pop_front();
        }
    }

    fn flush(&mut self) {
        flush_utterance(&mut self.utterance, &self.whisper_tx, self.min_speech_rms);
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    pub(crate) fn finish(&mut self) {
        if !self.utterance.buf.is_empty() {
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, WHISPER_RATE};

    fn input_chunk_for(hw_rate: u32) -> usize {
        ((hw_rate as f64 * FRAME_48K as f64 / DENOISE_RATE as f64).round() as usize).max(1)
    }

    #[test]
    fn chunk_sizes_yield_480_out() {
        assert_eq!(input_chunk_for(44_100), 441);
        assert_eq!(input_chunk_for(48_000), 480);
        assert_eq!(input_chunk_for(16_000), 160);
        assert_eq!(input_chunk_for(96_000), 960);
    }

    #[test]
    fn denoise_frame_is_480() {
        assert_eq!(DenoiseState::FRAME_SIZE, FRAME_48K);
    }

    #[test]
    fn rubato_produces_480_for_common_rates() {
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let chunk = input_chunk_for(hw);
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
        // `FrameProcessor::new` only takes the `WhisperResample::Direct`
        // (single-hop hw->16k) path when a hw->16k resampler consumes the
        // same input chunk size as the hw->48k one, since both are fed from
        // the same raw stream in lockstep. Confirms that holds for common
        // rates, and that both branches still land on a round 10 ms frame
        // count (480 @ 48k, 160 @ 16k) — i.e. that `make_resampler`'s single
        // formula (`hw_rate / 100`) is the right one for both targets.
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let (r48, in48) = make_resampler(hw, DENOISE_RATE).unwrap();
            let (r16, in16) = make_resampler(hw, WHISPER_RATE).unwrap();
            assert_eq!(in48, in16, "hw={hw}: input chunk sizes must match");
            assert_eq!(r48.output_frames_next(), FRAME_48K, "hw={hw}");
            assert_eq!(r16.output_frames_next(), FRAME_16K, "hw={hw}");
        }
    }

    #[test]
    fn vad_accepts_10ms_16k_frame() {
        let mut vad = Vad::new(16_000).unwrap();
        vad.fvad_set_mode(VadMode::Aggressive).unwrap();
        let silence = vec![0i16; FRAME_16K];
        assert!(vad.is_voice_segment(&silence).is_ok());
    }
}
