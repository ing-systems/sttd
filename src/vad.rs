use earshot::Detector;

// ---------------------------------------------------------------------------
// VAD: bridges the pipeline's 10 ms frame cadence to earshot's fixed 16 ms
// (256-sample @ 16 kHz) prediction window. Everything outside this module
// only ever sees a per-10ms-frame VAD verdict; it doesn't know earshot's
// window doesn't line up with that cadence.
// ---------------------------------------------------------------------------

/// Samples earshot requires per prediction, at `WHISPER_RATE` (16 kHz): 16 ms.
const EARSHOT_WINDOW: usize = 256;

/// Turns the VAD branch's 16 kHz stream into a per-10ms-frame speech verdict.
///
/// earshot only predicts on exactly `EARSHOT_WINDOW` (256) samples, while the
/// rest of the pipeline runs on 160-sample (10 ms) frames (`FRAME_16K`), so
/// incoming samples are staged until a full window is available. 8 frames
/// (1280 samples) yield exactly 5 windows, landing on frames 2, 4, 5, 7, 8 of
/// each group of 8; a frame that doesn't complete a window re-reports the
/// last verdict instead of guessing. The longest such re-report run is one
/// frame, so a single positive window still marks at most 2 consecutive
/// frames as speech — `ONSET_SPEECH_FRAMES` (3) still requires evidence from
/// more than one independent prediction.
///
/// Adds at most `EARSHOT_WINDOW`'s worth of latency (16 ms) before a verdict
/// reflects newly arrived audio, absorbed by pre-roll and silence hangover
/// the same way the denoiser's own lookahead is.
pub(crate) struct SpeechDetector {
    detector: Box<Detector>,
    /// 16 kHz samples not yet consumed by a full `EARSHOT_WINDOW`.
    staging: Vec<f32>,
    threshold: f32,
    /// Hysteresis state: in-speech or not. Re-reported for frames that
    /// didn't themselves complete a window. Starts `false`.
    in_speech: bool,
    /// Last earshot score, for hysteresis across windows.
    last_score: f32,
    /// Warm background estimate: survives `reset()` on purpose (see below).
    noise: crate::noise::NoiseTracker,
    adapt: bool,
}

impl SpeechDetector {
    pub(crate) fn new(threshold: f32, adapt: bool) -> Self {
        Self {
            detector: Detector::default_boxed(),
            staging: Vec::with_capacity(EARSHOT_WINDOW),
            threshold,
            in_speech: false,
            last_score: 0.0,
            noise: crate::noise::NoiseTracker::new(),
            adapt,
        }
    }

    /// Current noise floor (for logging/tuning).
    #[allow(dead_code)]
    pub(crate) fn noise_floor(&self) -> f32 {
        self.noise.floor()
    }

    /// Feed one 10 ms VAD-branch frame (16 kHz, `[-1, 1]`); returns the
    /// current speech verdict with hysteresis: entering speech needs
    /// `score >= enter` (base lifted by background), staying in speech only
    /// needs `score >= exit` (enter minus hysteresis). Non-speech frames
    /// keep the background estimate warm; speech frames freeze it.
    pub(crate) fn verdict(&mut self, frame_16k: &[f32]) -> bool {
        self.staging.extend_from_slice(frame_16k);

        let mut consumed = 0;
        let mut scored = false;
        while self.staging.len() - consumed >= EARSHOT_WINDOW {
            let window = &self.staging[consumed..consumed + EARSHOT_WINDOW];
            let score = self.detector.predict_f32(window);
            self.last_score = score;
            scored = true;
            let (enter, exit) = if self.adapt {
                (
                    self.noise.enter_threshold(self.threshold),
                    self.noise.exit_threshold(self.threshold),
                )
            } else {
                (self.threshold, self.threshold)
            };
            if self.in_speech {
                if score < exit {
                    self.in_speech = false;
                }
            } else if score >= enter {
                self.in_speech = true;
            }
            consumed += EARSHOT_WINDOW;
        }
        self.staging.drain(..consumed);

        // Warm the floor only on background: speech must never raise it.
        // Frames without a fresh score reuse the latched state (same as
        // before) and still count as background when not in speech.
        if !self.in_speech {
            // `scored` frames carry a fresh earshot verdict; unscored frames
            // are 10 ms slices of the same background — both are safe.
            let _ = scored;
            self.noise.update_background(frame_16k);
        }

        self.in_speech
    }

    /// Observe-only background update for pause: RMS -> noise floor, no
    /// earshot, no utterance, no latch change. Cheap (no neural pass) so a
    /// 10 s pause on fan noise actually teaches the threshold instead of
    /// idling. Never marks speech.
    pub(crate) fn observe_background(&mut self, frame_16k: &[f32]) {
        self.noise.update_background(frame_16k);
    }

    /// Drop staged partial-window samples, the hysteresis latch, and the
    /// earshot detector's internal state (ring buffer + feature context +
    /// RNN state) for a pause transition, so resume starts from silence,
    /// not pre-pause audio. The noise floor is deliberately kept: it is the
    /// warm background Faza 2 learns from.
    pub(crate) fn reset(&mut self) {
        self.detector.reset();
        self.staging.clear();
        self.in_speech = false;
        self.last_score = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_VAD_THRESHOLD, FRAME_16K};

    #[test]
    fn silence_is_not_speech() {
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let silence = vec![0.0f32; FRAME_16K];
        // A full second of silence: enough frames to complete several
        // earshot windows regardless of how staging aligns.
        for _ in 0..100 {
            assert!(!vad.verdict(&silence));
        }
    }

    #[test]
    fn prediction_cadence_is_5_per_8_frames() {
        // 8 frames of FRAME_16K (160) samples = 1280 = 5 * EARSHOT_WINDOW
        // (256): staging must be empty again after the 8th frame, with no
        // leftover samples carried past a clean group boundary.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        for _ in 0..8 {
            vad.verdict(&frame);
        }
        assert!(vad.staging.is_empty());
    }

    #[test]
    fn never_predicts_on_a_short_frame() {
        // earshot's predict_f32 debug_asserts frame.len() == 256 and returns
        // -1.0 (never a valid [0,1] score) if fed anything else. Staging
        // must never let more than EARSHOT_WINDOW - 1 samples build up
        // between windows, which is what guarantees every slice handed to
        // earshot is exactly EARSHOT_WINDOW long.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        for _ in 0..40 {
            vad.verdict(&frame);
            assert!(vad.staging.len() < EARSHOT_WINDOW);
        }
    }

    #[test]
    fn latches_between_windows() {
        // Force a verdict with a single frame, then confirm a same-verdict
        // frame that completes no new window still reports it.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        // Frame 1: 160 samples staged, no window completed (160 < 256).
        let v1 = vad.verdict(&frame);
        assert_eq!(v1, vad.in_speech);
        assert_eq!(vad.staging.len(), FRAME_16K);
    }

    #[test]
    fn honours_custom_threshold() {
        // A threshold of exactly 1.0 can never be exceeded by an in-range
        // score, so silence (and any real, non-clipping audio) always reads
        // as not-speech.
        let mut vad = SpeechDetector::new(1.0, true);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..20 {
            assert!(!vad.verdict(&silence));
        }
    }

    #[test]
    fn adapt_off_keeps_single_threshold() {
        // Kill-switch path: adapt=false must behave like the pre-adapt
        // single-threshold verdict — silence never opens, threshold 1.0 never
        // fires, and pause observe still warms the floor for a later re-enable.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, false);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..20 {
            assert!(!vad.verdict(&silence));
        }
        vad.observe_background(&silence);
        let mut never = SpeechDetector::new(1.0, false);
        for _ in 0..20 {
            assert!(!never.verdict(&silence));
        }
    }

    #[test]
    fn reset_drops_staging_and_latch() {
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        // One frame stages 160 samples without completing a window.
        vad.verdict(&frame);
        assert!(!vad.staging.is_empty());
        vad.reset();
        assert!(vad.staging.is_empty());
        assert!(!vad.in_speech);
    }
}
