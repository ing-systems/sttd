use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};
use webrtc_vad::{Vad, VadMode};

use crate::config::{
    DENOISE_RATE, DEFAULT_MIN_SPEECH_RMS, FRAME_16K, FRAME_48K, WHISPER_RATE, f32_from_env,
};
use crate::dsp::{WhisperResample, resampler_for};
use crate::utterance::{FrameAction, Utterance, UtteranceBuffer};

// ---------------------------------------------------------------------------
// Frame processing: denoise/VAD branch + whisper branch + utterance buffer
// ---------------------------------------------------------------------------

/// Owns all per-frame DSP/VAD/buffering state, including the hw-rate ->
/// 48 kHz / hw-rate -> 16 kHz resamplers (previously built and driven by the
/// caller). Fed one raw hw-rate chunk at a time (`required_input_frames()`
/// samples) by either the microphone loop or file mode, so both share
/// identical resampling and utterance-detection behavior.
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

    utterance: UtteranceBuffer,
    /// `STTD_MIN_RMS` override (or `DEFAULT_MIN_SPEECH_RMS`), read once at
    /// startup and reused for every utterance.
    min_speech_rms: f32,

    whisper_tx: Sender<Utterance>,
}

impl FrameProcessor {
    pub(crate) fn new(hw_rate: u32, whisper_tx: Sender<Utterance>) -> Result<Self> {
        let (resampler_hw48, hw_input_frames) = resampler_for(hw_rate, DENOISE_RATE)
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
        let whisper_resample = match resampler_for(hw_rate, WHISPER_RATE) {
            Ok((resampler, hw16_input_frames)) if hw16_input_frames == hw_input_frames => {
                debug_assert_eq!(resampler.output_frames_next(), FRAME_16K);
                let in_buf = resampler.input_buffer_allocate(true);
                let out_buf = resampler.output_buffer_allocate(true);
                WhisperResample::DirectTo16k {
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
                WhisperResample::Via48k {
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
            utterance: UtteranceBuffer::new(),
            min_speech_rms: f32_from_env("STTD_MIN_RMS", DEFAULT_MIN_SPEECH_RMS),
            whisper_tx,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) fn required_input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Process one raw hw-rate chunk (`required_input_frames()` samples).
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        // ---- hw -> 48 kHz: shared basis for the VAD branch (denoise needs
        // 48 kHz), and for the whisper branch too when using the `Via48k`
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
        // Owned: needed below both for denoise and, in the `Via48k` case,
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
            WhisperResample::DirectTo16k {
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
            WhisperResample::Via48k {
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
        if self.utterance.push_frame(&whisper16, is_speech) == FrameAction::Flush {
            self.enqueue_utterance();
        }
    }

    /// Close the open utterance and hand it to the whisper thread if it
    /// cleared the acoustic gates. Separated from `UtteranceBuffer::close` so
    /// the accept/reject decision stays testable without a live channel.
    fn enqueue_utterance(&mut self) {
        if let Some(utterance) = self.utterance.close(self.min_speech_rms) {
            if let Err(e) = self.whisper_tx.send(utterance) {
                eprintln!("[sttd] whisper queue closed: {e}");
            }
        }
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    pub(crate) fn flush_trailing_speech(&mut self) {
        if self.utterance.is_open() {
            self.enqueue_utterance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FRAME_16K;

    #[test]
    fn webrtc_vad_accepts_10ms_16k_frame() {
        let mut vad = Vad::new(16_000).unwrap();
        vad.fvad_set_mode(VadMode::Aggressive).unwrap();
        let silence = vec![0i16; FRAME_16K];
        assert!(vad.is_voice_segment(&silence).is_ok());
    }
}
