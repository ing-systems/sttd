use anyhow::Result;

use crate::agc::Agc;
use crate::denoise::DenoiserKind;
use crate::dsp::hw_input_frames;
use crate::gate::GateConfig;
use crate::signal::SignalPath;
use crate::utterance::{FrameAction, Utterance, UtteranceBuffer};
use crate::vad::SpeechDetector;

// ---------------------------------------------------------------------------
// Frame processing: Hp (in SignalPath) -> Agc (shared-gain, VAD-gated) ->
// VAD verdicts + utterance buffer, fed by the signal path
// ---------------------------------------------------------------------------

/// Owns per-frame VAD/buffering state. Fed one raw hw-rate chunk at a time
/// (`required_input_frames()` samples) by either the microphone loop or file
/// mode, so both share identical resampling and utterance-detection behavior.
/// Resampling / denoising / Hp / pairing lives behind `SignalPath`; this module
/// applies shared-gain Agc then turns paired frames into VAD verdicts and
/// utterances.
pub(crate) struct FrameProcessor {
    hw_input_frames: usize,
    signal: SignalPath,
    vad: SpeechDetector,
    utterance: UtteranceBuffer,
    gate: GateConfig,
    agc: Agc,
    /// Previous frame's verdict. Gates Agc (1-frame delay breaks the
    /// VAD<->gain feedback loop): silence passes at unity, speech adapts.
    prev_speech: bool,
}

impl FrameProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        hw_rate: u32,
        denoiser_kind: DenoiserKind,
        vad_threshold: f32,
        vad_adapt: bool,
        gate: GateConfig,
        hangover_frames: u32,
        target_rms: f32,
        max_gain: f32,
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
            agc: Agc::new(target_rms, max_gain),
            prev_speech: false,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) fn required_input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Unpaired frames currently waiting inside `SignalPath`, per branch
    /// `(vad, whisper)`. Exposed for the main-loop backpressure warn so a
    /// resampler divergence shows up in the log instead of silent drift.
    pub(crate) fn signal_pending_depths(&self) -> (usize, usize) {
        self.signal.pending_depths()
    }

    /// Process one raw hw-rate chunk (`required_input_frames()` samples).
    /// Returns the closed utterance, if any — the caller owns the `rtrb`
    /// handoff to the whisper thread. At most one utterance can close per
    /// call (reopening needs `ONSET_SPEECH_FRAMES` fresh speech frames).
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) -> Option<Utterance> {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        for (vad16, whisper16) in self.signal.feed(frame_hw_raw) {
            let mut vad_frame = (*vad16).clone();
            let mut whisper_frame = (*whisper16).clone();
            self.agc
                .process_pair(&mut vad_frame, &mut whisper_frame, self.prev_speech);
            let is_speech = self.vad.verdict(&vad_frame);
            self.prev_speech = is_speech;
            if self.utterance.push_frame(&whisper_frame, is_speech) == FrameAction::Flush {
                return self.close_current();
            }
        }
        None
    }

    /// Close the open utterance and return it if it cleared the acoustic
    /// gates. Separated from `UtteranceBuffer::close` so the accept/reject
    /// decision stays testable without a live channel. Stale-backlog
    /// shedding lives in the whisper worker (see `WHISPER_SHED_KEEP`), which
    /// owns the queue's consumer — this side never blocks the processing
    /// loop.
    fn close_current(&mut self) -> Option<Utterance> {
        self.utterance.close(&self.gate)
    }


    /// Discard all in-flight audio for a pause/hold transition: signal-path
    /// queues + resampler/Hp state, Agc gain, VAD staging/latch, and the
    /// utterance buffer including pre-roll. Idempotent; discarding an
    /// already-empty pipeline is a no-op. Noise floor preserved.
    pub(crate) fn discard_pending(&mut self) {
        self.signal.clear();
        self.vad.reset();
        self.utterance.discard();
        self.agc.reset();
        self.prev_speech = false;
    }

    /// Full-pipeline hold (pause): same as `discard_pending` — no background
    /// learning, no resampling, frozen floor. The main loop parks (drains and
    /// drops raw) after calling this once on the pause edge.
    pub(crate) fn hold_pipeline(&mut self) {
        self.discard_pending();
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    /// Returns the trailing utterance, if any.
    pub(crate) fn flush_trailing_speech(&mut self) -> Option<Utterance> {
        for (vad16, whisper16) in self.signal.flush() {
            let mut vad_frame = (*vad16).clone();
            let mut whisper_frame = (*whisper16).clone();
            self.agc
                .process_pair(&mut vad_frame, &mut whisper_frame, self.prev_speech);
            let is_speech = self.vad.verdict(&vad_frame);
            self.prev_speech = is_speech;
            if self.utterance.push_frame(&whisper_frame, is_speech) == FrameAction::Flush {
                return self.close_current();
            }
        }
        if self.utterance.is_open() {
            return self.close_current();
        }
        None
    }
}
