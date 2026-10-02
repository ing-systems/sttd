use anyhow::Result;

use crate::agc::Agc;
use crate::denoise::DenoiserKind;
use crate::dsp::hw_input_frames;
use crate::gate::GateConfig;
use crate::noise::frame_rms;
use crate::signal::{PairedFrames, SignalPath};
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

    /// The open utterance so far, when a partial transcript is due (see
    /// `UtteranceBuffer::partial_snapshot`). Call after `process_frame`: an
    /// utterance that frame closed is no longer open, so it yields `None`.
    pub(crate) fn take_partial(&mut self) -> Option<Vec<f32>> {
        self.utterance.partial_snapshot(self.gate.min_speech_rms)
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    pub(crate) const fn required_input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Unpaired frames currently waiting inside `SignalPath`, per branch
    /// `(vad, model)`. Exposed for the main-loop backpressure warn so a
    /// resampler divergence shows up in the log instead of silent drift.
    pub(crate) fn signal_pending_depths(&self) -> (usize, usize) {
        self.signal.pending_depths()
    }

    /// Current warm noise floor (for carrying it across a device rebuild —
    /// see `adopt_noise_floor`; a rebuild constructs a brand new
    /// `FrameProcessor`, so `discard_pending`'s "floor preserved" guarantee
    /// doesn't apply across that boundary on its own).
    pub(crate) const fn noise_floor(&self) -> f32 {
        self.vad.noise_floor()
    }

    /// Whether an utterance is currently buffered (opened, not yet closed).
    /// The beep-leak guard (`main::cue_guard`) skips its discard when speech
    /// is in flight so a cue never cuts real audio.
    pub(crate) const fn has_open_utterance(&self) -> bool {
        self.utterance.is_open()
    }

    /// Adopt a floor learned by a previous `FrameProcessor` (device rebuild).
    /// Without this, a rebuild silently drops back to `NoiseTracker`'s
    /// default (0.001), running the VAD at the un-lifted base threshold for
    /// the ~2 s the EMA needs to relearn a noisy room — exactly the window
    /// where false triggers turn into hallucinated utterances.
    pub(crate) const fn adopt_noise_floor(&mut self, floor: f32) {
        self.vad.set_floor(floor);
    }

    /// Process one raw hw-rate chunk (`required_input_frames()` samples).
    /// Returns the closed utterance, if any — the caller owns the `rtrb`
    /// handoff to the ASR thread. At most one utterance can close per
    /// call (reopening needs `ONSET_SPEECH_FRAMES` fresh speech frames, which
    /// a chunk's few-pairs surplus past a hangover close can never supply).
    ///
    /// Keeps iterating past a close instead of returning immediately: `feed`
    /// occasionally completes two pairs for one hw chunk (non-integer rate
    /// ratios), and returning early used to silently drop the second pair —
    /// the first ~10 ms of the *next* utterance's pre-roll, plus its VAD
    /// verdict and noise-floor update.
    pub(crate) fn process_frame(&mut self, frame_hw_raw: &[f32]) -> Option<Utterance> {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        let mut closed = None;
        for pair in self.signal.feed(frame_hw_raw) {
            if let Some(done) = self.absorb_pair(pair) {
                closed = Some(done);
            }
        }
        closed
    }

    /// Agc + VAD verdict + buffering for one completed pair. Returns the
    /// closed utterance when this frame asked for a flush.
    ///
    /// `Shared` is the default topology: one allocation serves both roles, so
    /// the Agc applies the shared gain once and a single post-gain RMS serves
    /// both the VAD floor and the utterance buffer. `evidence_rms` stays the
    /// pre-AGC measurement either way — evidence must describe what the
    /// microphone heard, not what the AGC did to it (see
    /// `UtteranceBuffer::push_frame_evidenced`).
    fn absorb_pair(&mut self, pair: PairedFrames) -> Option<Utterance> {
        match pair {
            PairedFrames::Shared(mut frame) => {
                let evidence_rms = self.agc.process_shared(&mut frame, self.prev_speech);
                let loudness = frame_rms(&frame);
                let is_speech = self.vad.verdict(&frame, loudness);
                self.prev_speech = is_speech;
                self.absorb_frame(&frame, is_speech, evidence_rms, loudness)
            }
            PairedFrames::Split(mut vad_frame, mut model_frame) => {
                let evidence_rms =
                    self.agc
                        .process_pair(&mut vad_frame, &mut model_frame, self.prev_speech);
                let is_speech = self.vad.verdict(&vad_frame, frame_rms(&vad_frame));
                self.prev_speech = is_speech;
                let loudness = frame_rms(&model_frame);
                self.absorb_frame(&model_frame, is_speech, evidence_rms, loudness)
            }
        }
    }

    /// Hand one gained frame to the utterance buffer with its two loudness
    /// measurements: `evidence_rms` (pre-AGC, what the mic heard) and
    /// `loudness` (post-Agc, what the buffer and the model get).
    fn absorb_frame(
        &mut self,
        model_frame: &[f32],
        is_speech: bool,
        evidence_rms: f32,
        loudness: f32,
    ) -> Option<Utterance> {
        if self
            .utterance
            .push_frame_evidenced(model_frame, is_speech, evidence_rms, loudness)
            == FrameAction::Flush
        {
            self.close_current()
        } else {
            None
        }
    }

    /// Close the open utterance and return it if it cleared the acoustic
    /// gates. Separated from `UtteranceBuffer::close` so the accept/reject
    /// decision stays testable without a live channel. Stale-backlog
    /// shedding lives in the ASR worker (see `ASR_SHED_KEEP`), which
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
    /// Returns the trailing utterance, if any. Keeps iterating past a
    /// mid-flush close instead of returning immediately (see
    /// `process_frame`'s doc comment for why).
    pub(crate) fn flush_trailing_speech(&mut self) -> Option<Utterance> {
        let mut closed = None;
        for pair in self.signal.flush() {
            if let Some(done) = self.absorb_pair(pair) {
                closed = Some(done);
            }
        }
        if closed.is_none() && self.utterance.is_open() {
            closed = self.close_current();
        }
        closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> FrameProcessor {
        FrameProcessor::new(
            16_000,
            DenoiserKind::Bypass,
            0.5,
            true,
            GateConfig::new(0.003, 0.25),
            80,
            0.0,
            8.0,
        )
        .unwrap()
    }

    #[test]
    fn a_partial_needs_an_open_utterance() {
        // 3 s of silence never opens an utterance, so there is nothing to cut a
        // partial from, however long the frame processor runs.
        let mut idle = processor();
        let hop = idle.required_input_frames();
        for _ in 0..200 {
            idle.process_frame(&vec![0.0; hop]);
            assert!(idle.take_partial().is_none());
        }
    }

    #[test]
    fn noise_floor_round_trips_through_adopt() {
        // Regression for the device-rebuild fix: a rebuild constructs a
        // brand new FrameProcessor (hence a brand new SpeechDetector /
        // NoiseTracker at the default 0.001 floor), so the warm floor must
        // be carried across explicitly rather than assumed preserved.
        let mut fp = processor();
        assert!(
            (fp.noise_floor() - 0.001).abs() < 1e-6,
            "fresh floor default"
        );
        fp.adopt_noise_floor(0.02);
        assert!(
            (fp.noise_floor() - 0.02).abs() < 1e-6,
            "adopted floor must read back"
        );

        let mut rebuilt = processor();
        assert!(
            (rebuilt.noise_floor() - 0.001).abs() < 1e-6,
            "new instance starts cold"
        );
        rebuilt.adopt_noise_floor(fp.noise_floor());
        assert!(
            (rebuilt.noise_floor() - 0.02).abs() < 1e-6,
            "rebuild must carry the old instance's floor forward"
        );
    }
}
