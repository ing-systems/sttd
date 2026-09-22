use anyhow::Result;
use crossbeam_channel::Sender;

use crate::denoise::DenoiserKind;
use crate::dsp::hw_input_frames;
use crate::gate::GateConfig;
use crate::signal::SignalPath;
use crate::utterance::{FrameAction, Utterance, UtteranceBuffer};
use crate::vad::SpeechDetector;

// ---------------------------------------------------------------------------
// Frame processing: VAD verdicts + utterance buffer, fed by the signal path
// ---------------------------------------------------------------------------

/// Owns per-frame VAD/buffering state. Fed one raw hw-rate chunk at a time
/// (`required_input_frames()` samples) by either the microphone loop or file
/// mode, so both share identical resampling and utterance-detection behavior.
/// Resampling / denoising / pairing lives behind `SignalPath`; this module
/// only turns paired frames into VAD verdicts and utterances.
pub(crate) struct FrameProcessor {
    hw_input_frames: usize,
    signal: SignalPath,
    vad: SpeechDetector,
    utterance: UtteranceBuffer,
    gate: GateConfig,
    whisper_tx: Sender<Utterance>,
}

impl FrameProcessor {
    pub(crate) fn new(
        hw_rate: u32,
        whisper_tx: Sender<Utterance>,
        denoiser_kind: DenoiserKind,
        vad_threshold: f32,
        vad_adapt: bool,
        gate: GateConfig,
        hangover_frames: u32,
    ) -> Result<Self> {
        let hw_input_frames = hw_input_frames(hw_rate);
        let signal = SignalPath::new(hw_rate, denoiser_kind)?;

        // Minimum earshot score for a window to count as speech (see `vad`).
        let vad = SpeechDetector::new(vad_threshold, vad_adapt);

        Ok(Self {
            hw_input_frames,
            signal,
            vad,
            utterance: UtteranceBuffer::new().with_hangover_frames(hangover_frames),
            gate,
            whisper_tx,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) fn required_input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Utterances waiting on the whisper thread. Unbounded by design (speech
    /// must not be dropped under load); the main loop warns when it grows,
    /// so a slow GPU shows up in the log instead of silent memory growth.
    pub(crate) fn queue_depth(&self) -> usize {
        self.whisper_tx.len()
    }

    /// Unpaired frames currently waiting inside `SignalPath`, per branch
    /// `(vad, whisper)`. Exposed for the main-loop backpressure warn so a
    /// resampler divergence shows up in the log instead of silent drift.
    pub(crate) fn signal_pending_depths(&self) -> (usize, usize) {
        self.signal.pending_depths()
    }

    /// Process one raw hw-rate chunk (`required_input_frames()` samples).
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        for (vad16, whisper16) in self.signal.feed(frame_hw_raw) {
            let is_speech = self.vad.verdict(&vad16);
            if self.utterance.push_frame(&whisper16, is_speech) == FrameAction::Flush {
                self.enqueue_utterance();
            }
        }
    }

    /// Close the open utterance and hand it to the whisper thread if it
    /// cleared the acoustic gates. Separated from `UtteranceBuffer::close` so
    /// the accept/reject decision stays testable without a live channel.
    /// Stale-backlog shedding lives in the whisper worker (see
    /// `WHISPER_SHED_KEEP`), which owns the queue's receiver — this side
    /// never blocks the processing loop.
    fn enqueue_utterance(&mut self) {
        if let Some(utterance) = self.utterance.close(&self.gate) {
            if let Err(e) = self.whisper_tx.send(utterance) {
                log::warn!("whisper queue closed: {e}");
            }
        }
    }


    /// Discard all in-flight audio for a pause transition: signal-path
    /// queues + resampler state, VAD staging/latch, and the utterance buffer
    /// including pre-roll. Idempotent; discarding an already-empty pipeline
    /// is a no-op.
    pub(crate) fn discard_pending(&mut self) {
        self.signal.clear();
        self.vad.reset();
        self.utterance.discard();
    }

    /// Observe-only background feed for pause: resample + RMS -> noise floor,
    /// no earshot, no utterance, no latch change. Keeps resamplers continuous
    /// (no `clear()` here) so the floor estimate has no chunk gaps.
    pub(crate) fn observe_pause(&mut self, frame_hw_raw: &[f32]) {
        for (vad16, _) in self.signal.feed(frame_hw_raw) {
            self.vad.observe_background(&vad16);
        }
    }

    /// Drop pre-pause speech state on pause entry without touching warm
    /// state: VAD inference reset (noise floor preserved, see `vad`) +
    /// utterance discard (pre-roll included). Resamplers are kept continuous
    /// for `observe_pause`. Idempotent.
    pub(crate) fn discard_speech_keep_warm(&mut self) {
        self.vad.reset();
        self.utterance.discard();
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    pub(crate) fn flush_trailing_speech(&mut self) {
        for (vad16, whisper16) in self.signal.flush() {
            let is_speech = self.vad.verdict(&vad16);
            if self.utterance.push_frame(&whisper16, is_speech) == FrameAction::Flush {
                self.enqueue_utterance();
            }
        }
        if self.utterance.is_open() {
            self.enqueue_utterance();
        }
    }
}
