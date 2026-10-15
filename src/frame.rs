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
    /// drops raw) after calling this once on the pause edge; `flush_on_pause`
    /// folds it into the pause-edge sequence.
    pub(crate) fn hold_pipeline(&mut self) {
        self.discard_pending();
    }

    /// Pause edge: cut the open utterance and hand it back so the last thing
    /// said survives the hold, then hold the pipeline as `hold_pipeline` does.
    /// Returns the flushed utterance (marked as a pause flush), or `None` when
    /// nothing was buffered or the acoustic gates dropped it.
    ///
    /// Order is the point: the hold discards the utterance buffer, so the cut
    /// has to happen first. Marking after the trailing flush covers an
    /// utterance that `signal`'s zero-padded tail closed rather than the open
    /// buffer. The caller queues the result (see `asr::worker::keep_while_paused`),
    /// because the worker drops untagged backlog while a pause is held.
    pub(crate) fn flush_on_pause(&mut self) -> Option<Utterance> {
        self.flush_and_hold(true)
    }

    /// Speaking edge (`--speak`): the same cut and hold as the pause edge, but
    /// **not** marked as a pause flush.
    ///
    /// The distinction matters because the mark is about the *worker's* policy
    /// for a held pause (`asr::worker::keep_while_paused` keeps a pause flush and drops
    /// an untagged backlog). Speaking holds the pipeline without setting the
    /// pause flag, so the worker is not in that branch at all and an untagged
    /// utterance here is an ordinary one — it is transcribed and emitted like
    /// any other, which is what the user wants: their last words before sttd
    /// started talking still get transcribed.
    pub(crate) fn flush_on_speaking(&mut self) -> Option<Utterance> {
        self.flush_and_hold(false)
    }

    /// Cut the open utterance, optionally mark it a pause flush, then hold.
    fn flush_and_hold(&mut self, mark_pause: bool) -> Option<Utterance> {
        let mut flushed = self.flush_trailing_speech();
        if mark_pause {
            if let Some(utterance) = flushed.as_mut() {
                utterance.mark_pause_flush();
            }
        }
        self.hold_pipeline();
        flushed
    }

    /// Flush any trailing speech. Call once at shutdown / end of input, and on
    /// the pause edge (see `flush_on_pause`). Returns the trailing utterance,
    /// if any. Keeps iterating past a mid-flush close instead of returning
    /// immediately (see `process_frame`'s doc comment for why).
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
    use crate::config::MIN_SPEECH_FRAMES;

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

    /// Same, with the acoustic gates off (rms gate and density floor at 0), so
    /// a test sees only `MIN_SPEECH_FRAMES` deciding whether the audio counts
    /// as speech enough to be transmitted.
    fn processor_without_gates() -> FrameProcessor {
        FrameProcessor::new(
            16_000,
            DenoiserKind::Bypass,
            0.5,
            true,
            GateConfig::new(0.0, 0.0),
            80,
            0.0,
            8.0,
        )
        .unwrap()
    }

    /// Voiced-ish audio: a harmonic stack under a 4 Hz envelope, which is what
    /// the detector expects speech to look like.
    fn voiced(hop: usize, t: f32) -> Vec<f32> {
        (0..hop)
            .map(|i| {
                let s = t + i as f32 / 16_000.0;
                let env = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 4.0 * s).sin();
                let tone = (2.0 * std::f32::consts::PI * 140.0 * s).sin()
                    + 0.6 * (2.0 * std::f32::consts::PI * 700.0 * s).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 2400.0 * s).sin();
                0.25 * env * tone / 1.9
            })
            .collect()
    }

    /// Feed syllable-like bursts (40 ms of voice, 20 ms of near silence). The
    /// gaps hold the adaptive VAD floor down, so the bursts keep being flagged
    /// as speech; one steady tone is learned as background instead and never
    /// reaches `MIN_SPEECH_FRAMES`.
    fn speak(fp: &mut FrameProcessor, bursts: usize) {
        let hop = fp.required_input_frames();
        let mut t = 0.0f32;
        let step = hop as f32 / 16_000.0;
        for _ in 0..bursts {
            for _ in 0..4 {
                fp.process_frame(&voiced(hop, t));
                t += step;
            }
            for _ in 0..2 {
                let quiet: Vec<f32> = voiced(hop, t).iter().map(|s| s * 0.02).collect();
                fp.process_frame(&quiet);
                t += step;
            }
        }
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
    fn flush_on_pause_holds_the_pipeline_with_nothing_buffered() {
        // Silence never opens an utterance, so the pause edge has nothing to
        // save — and must still hold the pipeline.
        let mut fp = processor();
        let hop = fp.required_input_frames();
        for _ in 0..50 {
            fp.process_frame(&vec![0.0; hop]);
        }
        assert!(fp.flush_on_pause().is_none(), "nothing was buffered");
        assert!(
            !fp.has_open_utterance(),
            "the pause must leave no open buffer"
        );
    }

    #[test]
    fn flush_on_pause_hands_back_the_open_utterance_marked_as_a_flush() {
        // The regression this guards: the pause edge used to hold the pipeline
        // first, which discarded the open utterance and lost the last thing
        // said. The cut must come first, and what it returns must be marked so
        // the worker keeps it while the pause holds.
        let mut fp = processor_without_gates();
        speak(&mut fp, 30);
        assert!(
            fp.has_open_utterance(),
            "the trailing gap is short of the hangover, so the utterance is still open"
        );
        let flushed = fp
            .flush_on_pause()
            .expect("the open utterance must be cut, not discarded");
        assert!(!flushed.pcm.is_empty());
        assert!(flushed.pause_flush, "the worker must keep this while held");
        assert!(
            flushed.evidence.speech_frames >= MIN_SPEECH_FRAMES,
            "the synthetic bursts must reach the speech-frame floor"
        );
        // The hold still ran: nothing is left open, and a second pause edge
        // has nothing to hand back.
        assert!(!fp.has_open_utterance());
        assert!(fp.flush_on_pause().is_none());
    }

    #[test]
    fn the_speaking_edge_cuts_the_utterance_without_marking_it_a_pause() {
        // Same cut as the pause edge — what was said before sttd started
        // talking is real speech — but **not** marked. The mark means "keep this
        // while a pause is held" (`asr::worker::keep_while_paused`), and speaking holds
        // the pipeline without setting the pause flag, so the mark would claim
        // a policy that is not in force.
        let mut fp = processor_without_gates();
        speak(&mut fp, 30);
        assert!(fp.has_open_utterance());
        let flushed = fp
            .flush_on_speaking()
            .expect("the open utterance must be cut, not discarded");
        assert!(
            !flushed.pcm.is_empty(),
            "the words before the speech are kept"
        );
        assert!(
            !flushed.pause_flush,
            "speaking is not a pause, so the worker's held-pause policy must not claim it"
        );
        assert!(flushed.evidence.speech_frames >= MIN_SPEECH_FRAMES);
        // The hold ran, so nothing is left open.
        assert!(!fp.has_open_utterance());
        assert!(fp.flush_on_speaking().is_none());
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
