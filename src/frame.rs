use std::collections::VecDeque;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;

use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, WHISPER_RATE};
use crate::denoise::{Denoiser, DenoiserKind};
use crate::dsp::{FftStream, hw_input_frames};
use crate::utterance::{FrameAction, Utterance, UtteranceBuffer};
use crate::vad::SpeechDetector;

// ---------------------------------------------------------------------------
// Frame processing: VAD branch + whisper branch + utterance buffer
// ---------------------------------------------------------------------------

/// Owns all per-frame DSP/VAD/buffering state. Fed one raw hw-rate chunk at
/// a time (`required_input_frames()` samples) by either the microphone loop
/// or file mode, so both share identical resampling and
/// utterance-detection behavior.
pub(crate) struct FrameProcessor {
    hw_input_frames: usize,
    vad_path: VadPath,

    /// Completed 16 kHz frames waiting to be paired. In `Direct` mode both
    /// queues share ownership of each frame via `Arc` (one shared resampler
    /// feeds them together); in `Denoised` mode the two `FftStream`s complete
    /// on different cadences, so output frames pair up by order below instead
    /// of by call.
    pending_vad: VecDeque<std::sync::Arc<[f32]>>,
    pending_whisper: VecDeque<std::sync::Arc<[f32]>>,

    vad: SpeechDetector,

    utterance: UtteranceBuffer,
    /// `min_speech_rms`, read once at startup and reused for every utterance.
    min_speech_rms: f32,

    whisper_tx: Sender<Utterance>,
}

/// The VAD branch's signal path, chosen once at startup from `--denoiser`
/// and fixed for the process's lifetime.
enum VadPath {
    /// Default (`none`): a single hw_rate -> 16 kHz hop feeds both the VAD
    /// verdict and the utterance buffer — no 48 kHz intermediate, no
    /// denoiser object. Each completed 16 kHz frame serves both roles, so
    /// VAD/whisper pairing is trivially 1:1.
    Direct { resampler: FftStream },
    /// Opt-in (`deepfilter`): hw_rate -> 48 kHz -> denoise -> 48 kHz ->
    /// 16 kHz for VAD, plus an independent direct hw_rate -> 16 kHz hop for
    /// the whisper branch (which stays raw — the model was trained on raw
    /// audio, and the 48 kHz intermediate exists only so the denoiser, which
    /// is fixed at 48 kHz, can run for the VAD branch).
    Denoised {
        // hw_rate -> 48 kHz (the denoiser's required rate). Its resampler's
        // fixed FFT chunks complete on their own cadence, so drained
        // 480-frames are denoised one by one as they arrive rather than one
        // per input chunk.
        resampler_hw48: FftStream,
        // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
        resampler_vad: FftStream,
        resampler_whisper: FftStream,
        denoiser: Denoiser,
        clean48_vad: Vec<f32>,
    },
}

impl FrameProcessor {
    pub(crate) fn new(
        hw_rate: u32,
        whisper_tx: Sender<Utterance>,
        denoiser_kind: DenoiserKind,
        vad_threshold: f32,
        min_speech_rms: f32,
    ) -> Result<Self> {
        let hw_input_frames = hw_input_frames(hw_rate);
        let vad_path = match denoiser_kind {
            DenoiserKind::Bypass => {
                let resampler = FftStream::new(hw_rate, WHISPER_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {WHISPER_RATE})"))?;
                eprintln!("[sttd] denoiser: none");
                VadPath::Direct { resampler }
            }
            DenoiserKind::DeepFilter => {
                let resampler_hw48 = FftStream::new(hw_rate, DENOISE_RATE, FRAME_48K)
                    .with_context(|| format!("failed to create resampler (hw -> {DENOISE_RATE})"))?;
                let (chunk_in, chunk_out) = resampler_hw48.chunk_sizes();
                eprintln!(
                    "[sttd] resampler: {hw_rate} -> {DENOISE_RATE}, \
                     fft_chunk_in={chunk_in} fft_chunk_out={chunk_out} frame_out={FRAME_48K}"
                );

                let resampler_vad = FftStream::new(DENOISE_RATE, WHISPER_RATE, FRAME_16K)
                    .context("failed to create resampler (48k -> 16k, vad branch)")?;

                let resampler_whisper = FftStream::new(hw_rate, WHISPER_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {WHISPER_RATE})"))?;

                let denoiser = Denoiser::new(DenoiserKind::DeepFilter)?;
                VadPath::Denoised {
                    resampler_hw48,
                    resampler_vad,
                    resampler_whisper,
                    denoiser,
                    clean48_vad: vec![0.0f32; FRAME_48K],
                }
            }
        };

        // Minimum earshot score for a window to count as speech (see `vad`).
        let vad = SpeechDetector::new(vad_threshold);

        Ok(Self {
            hw_input_frames,
            vad_path,
            pending_vad: VecDeque::new(),
            pending_whisper: VecDeque::new(),
            vad,
            utterance: UtteranceBuffer::new(),
            min_speech_rms,
            whisper_tx,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) fn required_input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Process one raw hw-rate chunk (`required_input_frames()` samples).
    /// Each branch's `FftStream` may complete zero, one, or (on catch-up)
    /// several 10 ms frames per call; completed VAD/whisper frames are
    /// paired by order, so the VAD verdict always describes the whisper
    /// audio it is attached to.
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        match &mut self.vad_path {
            // ---- direct: one shared hop feeds both queues together --------
            VadPath::Direct { resampler } => {
                for frame16 in resampler.feed(frame_hw_raw) {
                    let shared: std::sync::Arc<[f32]> = frame16.into();
                    self.pending_whisper.push_back(std::sync::Arc::clone(&shared));
                    self.pending_vad.push_back(shared);
                }
            }
            // ---- denoised: 48 kHz detour for VAD, direct hop for whisper --
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_whisper,
                denoiser,
                clean48_vad,
            } => {
                // ---- hw -> 48 kHz -> denoise -> 16 kHz (VAD branch) -------
                for frame48 in resampler_hw48.feed(frame_hw_raw) {
                    denoiser.denoise_frame(&frame48, clean48_vad);
                    self.pending_vad.extend(
                        resampler_vad
                            .feed(clean48_vad)
                            .into_iter()
                            .map(std::sync::Arc::<[f32]>::from),
                    );
                }

                // ---- whisper branch: raw (no denoise) ---------------------
                self.pending_whisper.extend(
                    resampler_whisper
                        .feed(frame_hw_raw)
                        .into_iter()
                        .map(std::sync::Arc::<[f32]>::from),
                );
            }
        }

        self.drain_paired_frames();
    }

    /// Pair up completed VAD/whisper frames by order: VAD verdict plus
    /// utterance buffering for every pair where both sides are ready. A
    /// branch running ahead simply waits in its queue; both average
    /// 100 frames/s, so the queues stay bounded at a frame or two.
    fn drain_paired_frames(&mut self) {
        while let (Some(vad16), Some(whisper16)) =
            (self.pending_vad.pop_front(), self.pending_whisper.pop_front())
        {
            // ---- VAD decision --------------------------------------------------
            let is_speech = self.vad.verdict(&vad16);

            // ---- utterance buffering --------------------------------------------
            if self.utterance.push_frame(&whisper16, is_speech) == FrameAction::Flush {
                self.enqueue_utterance();
            }
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
    /// Zero-pads each resampler's trailing partial chunk (see
    /// `FftStream::flush`) so tail audio stuck in an incomplete FFT chunk
    /// still reaches the utterance instead of being lost with the shutdown.
    pub(crate) fn flush_trailing_speech(&mut self) {
        match &mut self.vad_path {
            VadPath::Direct { resampler } => {
                for frame16 in resampler.flush() {
                    let shared: std::sync::Arc<[f32]> = frame16.into();
                    self.pending_whisper.push_back(std::sync::Arc::clone(&shared));
                    self.pending_vad.push_back(shared);
                }
            }
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_whisper,
                denoiser,
                clean48_vad,
            } => {
                for frame48 in resampler_hw48.flush() {
                    denoiser.denoise_frame(&frame48, clean48_vad);
                    self.pending_vad.extend(
                        resampler_vad
                            .feed(clean48_vad)
                            .into_iter()
                            .map(std::sync::Arc::<[f32]>::from),
                    );
                }
                self.pending_vad.extend(
                    resampler_vad
                        .flush()
                        .into_iter()
                        .map(std::sync::Arc::<[f32]>::from),
                );
                self.pending_whisper.extend(
                    resampler_whisper
                        .flush()
                        .into_iter()
                        .map(std::sync::Arc::<[f32]>::from),
                );
            }
        }
        self.drain_paired_frames();
        if self.utterance.is_open() {
            self.enqueue_utterance();
        }
    }
}
