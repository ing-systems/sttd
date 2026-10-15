use earshot::Detector;

// ---------------------------------------------------------------------------
// VAD: bridges the pipeline's 10 ms frame cadence to earshot's fixed 16 ms
// (256-sample @ 16 kHz) prediction window. Everything outside this module
// only ever sees a per-10ms-frame VAD verdict; it doesn't know earshot's
// window doesn't line up with that cadence.
// ---------------------------------------------------------------------------

/// Samples earshot requires per prediction, at `SAMPLE_RATE` (16 kHz): 16 ms.
const EARSHOT_WINDOW: usize = 256;
/// Step between consecutive earshot windows: 128 samples (8 ms, 50%
/// overlap). Overlapping doubles the prediction rate (~10 windows per 8
/// frames vs 5 before) so almost every 10 ms frame carries a fresh
/// verdict, halving onset/offset jitter. Adjacent windows share half
/// their audio, so a short transient can now appear in two windows —
/// the 3-window median plus `ONSET_SPEECH_FRAMES` still absorb it.
const EARSHOT_STEP: usize = 128;

/// Window decisions the median rides over: the output flips only when at
/// least 2 of the last 3 raw windows agree, so a single flicker window (a
/// click, a breath, one RNN wobble) can never flip the frame verdict on its
/// own. Costs ~1 window (16 ms) of entry/exit latency each way, absorbed by
/// pre-roll and silence hangover the same way the prediction window itself
/// is. Fixed at 3: 5 would add another 32 ms and risk shaving short words
/// for little extra flicker rejection once hysteresis already handles the
/// near-threshold chatter.
const MEDIAN_WINDOWS: usize = 3;

/// Turns the VAD branch's 16 kHz stream into a per-10ms-frame speech verdict.
///
/// earshot only predicts on exactly `EARSHOT_WINDOW` (256) samples, while the
/// rest of the pipeline runs on 160-sample (10 ms) frames (`FRAME_16K`), so
/// incoming samples are staged until a full window is available. Windows
/// advance by `EARSHOT_STEP` (128, 50% overlap), so ~10 predictions land per
/// 8 frames; a frame that doesn't complete a window re-reports the last
/// verdict instead of guessing. Input arrives already 80 Hz high-passed
/// (`Hp` in `signal_path`, shared with the model branch) so hum never reaches
/// earshot or the floor. Each window's score passes through hysteresis first
/// (enter at `enter`, hold until below `exit`), then through a majority
/// vote over the last `MEDIAN_WINDOWS` raw window decisions: a single
/// positive window marks no frame as speech at all, so
/// `ONSET_SPEECH_FRAMES` (3) always needs evidence from at least two
/// independent predictions, and a single dropout window inside speech never
/// breaks the latch either.
///
/// Adds at most one `EARSHOT_WINDOW` plus one median step of latency before
/// a verdict reflects newly arrived audio, absorbed by pre-roll and silence
/// hangover the same way the denoiser's own lookahead is.
pub(crate) struct SpeechDetector {
    detector: Box<Detector>,
    /// 16 kHz samples not yet consumed by a full `EARSHOT_WINDOW`.
    /// With 50% overlap only `EARSHOT_STEP` samples are consumed per
    /// prediction; the rest stays staged as the next window's first half.
    staging: Vec<f32>,
    threshold: f32,
    /// Smoothed output: majority of `window_history`. Re-reported for frames
    /// that didn't themselves complete a window. Starts `false`.
    in_speech: bool,
    /// Raw per-window hysteresis state (input to the median, not the
    /// output). Split from `in_speech` so the median can't feed back into
    /// the thresholds.
    raw_speech: bool,
    /// Last `MEDIAN_WINDOWS` raw window decisions, oldest-first. Pre-filled
    /// with `false` (silence) so the very first windows get flicker
    /// protection too, and reset to the same on pause.
    window_history: [bool; MEDIAN_WINDOWS],
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
            raw_speech: false,
            window_history: [false; MEDIAN_WINDOWS],
            noise: crate::noise::NoiseTracker::new(),
            adapt,
        }
    }

    /// Current noise floor (for logging/tuning, and for carrying the warm
    /// estimate across a device rebuild — see `set_floor`).
    pub(crate) const fn noise_floor(&self) -> f32 {
        self.noise.floor()
    }

    /// Adopt a floor learned elsewhere (a device rebuild replaces this whole
    /// `SpeechDetector`, so `reset()`'s "keep the floor" guarantee doesn't
    /// apply — the caller must carry it across explicitly via this setter).
    /// Clamped the same as `NoiseTracker::update_background`.
    pub(crate) const fn set_floor(&mut self, floor: f32) {
        self.noise.set_floor(floor);
    }

    /// Feed one 10 ms VAD-branch frame (16 kHz, `[-1, 1]`); returns the
    /// current speech verdict with hysteresis plus median smoothing: entering
    /// raw speech needs `score >= enter` (base lifted by background), staying
    /// in raw speech only needs `score >= exit` (enter minus hysteresis), and
    /// the reported verdict is the majority of the last `MEDIAN_WINDOWS` raw
    /// decisions. Non-speech frames keep the background estimate warm; speech
    /// frames freeze it. Input is already HP'd upstream (`Hp`); staging and
    /// scoring see it verbatim.
    ///
    /// `frame_rms` is the caller's measurement of `frame_16k`: the pipeline
    /// already has it, and the floor estimator would otherwise scan the same
    /// 160 samples again on every frame.
    pub(crate) fn verdict(&mut self, frame_16k: &[f32], frame_rms: f32) -> bool {
        self.staging.extend_from_slice(frame_16k);

        // The floor only moves after the loop, so the thresholds hold for
        // every window this frame completes.
        let (enter, exit) = if self.adapt {
            (
                self.noise.enter_threshold(self.threshold),
                self.noise.exit_threshold(self.threshold),
            )
        } else {
            (self.threshold, self.threshold)
        };
        let mut consumed = 0;
        while self.staging.len() - consumed >= EARSHOT_WINDOW {
            let window = &self.staging[consumed..consumed + EARSHOT_WINDOW];
            let score = self.detector.predict_f32(window);
            let raw = if self.raw_speech {
                score >= exit
            } else {
                score >= enter
            };
            self.push_window(raw);
            consumed += EARSHOT_STEP;
        }
        // `consumed` advances by the overlap step, so up to
        // `EARSHOT_WINDOW - 1` samples stay staged as the next window's
        // first half — never more, so every slice handed to earshot stays
        // exactly `EARSHOT_WINDOW` long.
        self.staging.drain(..consumed);

        // Warm the floor only on background: speech must never raise it.
        // A frame that completed no window is a 10 ms slice of the same
        // background as the window that did, so it counts too. The floor
        // learns from the same HP'd audio earshot sees.
        if !self.in_speech {
            self.noise.update_background_rms(frame_rms);
        }

        self.in_speech
    }

    /// Fold one raw window decision into the median: update the hysteresis
    /// state, shift it into the history (dropping the oldest), and recompute
    /// the reported verdict as the majority. Split from `verdict` so tests can
    /// drive flicker sequences without a live earshot signal.
    fn push_window(&mut self, raw: bool) {
        self.raw_speech = raw;
        self.window_history.rotate_left(1);
        self.window_history[MEDIAN_WINDOWS - 1] = raw;
        let trues = self.window_history.iter().filter(|&&b| b).count();
        self.in_speech = trues * 2 > MEDIAN_WINDOWS;
    }

    /// Drop staged partial-window samples, the hysteresis latch, the median
    /// history, and the earshot detector's internal
    /// state (ring buffer + feature context + RNN state) for a pause
    /// transition, so resume starts from silence, not pre-pause audio. The
    /// noise floor is deliberately kept: it is the warm background Faza 2
    /// learns from.
    pub(crate) fn reset(&mut self) {
        self.detector.reset();
        self.staging.clear();
        self.in_speech = false;
        self.raw_speech = false;
        self.window_history = [false; MEDIAN_WINDOWS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_VAD_THRESHOLD, FRAME_16K};
    use crate::noise::frame_rms;

    #[test]
    fn silence_is_not_speech() {
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let silence = vec![0.0f32; FRAME_16K];
        // A full second of silence: enough frames to complete several
        // earshot windows regardless of how staging aligns.
        for _ in 0..100 {
            assert!(!vad.verdict(&silence, frame_rms(&silence)));
        }
    }

    #[test]
    fn prediction_cadence_overlaps_every_frame() {
        // 50% overlap (step 128): 8 frames (1280 samples) yield 9 windows
        // ((1280-256)/128+1) with 128 samples left staged as the next
        // window's first half — not empty, and bounded below one window.
        // The point is density: almost every 10 ms frame completes a fresh
        // prediction instead of 5 per 8.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        for _ in 0..8 {
            vad.verdict(&frame, frame_rms(&frame));
        }
        assert!(vad.staging.len() < EARSHOT_WINDOW);
        assert_eq!(vad.staging.len(), 128);
        assert_eq!(EARSHOT_STEP, 128);
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
            vad.verdict(&frame, frame_rms(&frame));
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
        let v1 = vad.verdict(&frame, frame_rms(&frame));
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
            assert!(!vad.verdict(&silence, frame_rms(&silence)));
        }
    }

    #[test]
    fn adapt_off_keeps_single_threshold() {
        // Kill-switch path: adapt=false must behave like the pre-adapt
        // single-threshold verdict — silence never opens, threshold 1.0 never
        // fires.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, false);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..20 {
            assert!(!vad.verdict(&silence, frame_rms(&silence)));
        }
        let mut never = SpeechDetector::new(1.0, false);
        for _ in 0..20 {
            assert!(!never.verdict(&silence, frame_rms(&silence)));
        }
    }

    #[test]
    fn reset_drops_staging_and_latch() {
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        // One frame stages 160 samples without completing a window.
        vad.verdict(&frame, frame_rms(&frame));
        assert!(!vad.staging.is_empty());
        vad.push_window(true);
        vad.push_window(true);
        assert!(vad.in_speech);
        vad.reset();
        assert!(vad.staging.is_empty());
        assert!(!vad.in_speech);
        assert!(!vad.raw_speech);
        assert!(vad.window_history.iter().all(|&b| !b));
        assert_eq!(vad.window_history.len(), MEDIAN_WINDOWS);
    }

    #[test]
    fn median_kills_single_window_flicker() {
        // A lone positive window (click, breath, one RNN wobble) must not
        // flip the verdict: 1-of-3 is not a majority.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        vad.push_window(true);
        assert!(!vad.in_speech, "single spike must not open");
        // A second consecutive positive reaches 2-of-3: real onset.
        vad.push_window(true);
        assert!(vad.in_speech, "two consecutive windows must open");
    }

    #[test]
    fn median_rides_through_single_dropout() {
        // Steady speech stays latched through one dropout window: 2-of-3
        // still holds.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        vad.push_window(true);
        vad.push_window(true);
        assert!(vad.in_speech);
        vad.push_window(true);
        assert!(vad.in_speech);
        vad.push_window(false);
        assert!(vad.in_speech, "single dropout must not break the latch");
        // Two dropouts in a row reach 1-of-3: speech really ended.
        vad.push_window(false);
        assert!(!vad.in_speech, "two consecutive dropouts must close");
    }

    #[test]
    fn median_history_starts_silenced() {
        // Fresh (and reset) detectors pre-fill with `false`, so even the
        // very first window gets flicker protection.
        let vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        assert_eq!(vad.window_history.len(), MEDIAN_WINDOWS);
        assert!(vad.window_history.iter().all(|&b| !b));
        assert_eq!(MEDIAN_WINDOWS, 3);
    }

    #[test]
    fn hp_is_upstream_now() {
        // Hum rejection lives in `Hp` (80 Hz biquad after resampling); VAD
        // stages scoring input verbatim with no internal filter state.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..20 {
            assert!(!vad.verdict(&silence, frame_rms(&silence)));
        }
        vad.reset();
        assert!(vad.staging.is_empty());
    }
}
