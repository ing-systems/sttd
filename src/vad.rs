use std::collections::VecDeque;

use earshot::Detector;

// ---------------------------------------------------------------------------
// VAD: bridges the pipeline's 10 ms frame cadence to earshot's fixed 16 ms
// (256-sample @ 16 kHz) prediction window. Everything outside this module
// only ever sees a per-10ms-frame VAD verdict; it doesn't know earshot's
// window doesn't line up with that cadence.
// ---------------------------------------------------------------------------

/// Samples earshot requires per prediction, at `WHISPER_RATE` (16 kHz): 16 ms.
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
/// verdict instead of guessing. Each frame is high-passed at 60 Hz first
/// (same cutoff as the whisper-branch conditioning, streaming state) so
/// mains hum can't inflate scores or the noise floor — the VAD sees what
/// whisper receives. Each window's score passes through hysteresis first
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
    window_history: VecDeque<bool>,
    /// Last earshot score, for hysteresis across windows.
    last_score: f32,
    /// Warm background estimate: survives `reset()` on purpose (see below).
    noise: crate::noise::NoiseTracker,
    adapt: bool,
    /// Streaming one-pole high-pass (60 Hz, same cutoff as the
    /// whisper-branch conditioning): `y[n] = x[n] - x[n-1] + a * y[n-1]`.
    /// Runs on every VAD-branch frame before staging so hum is gone before
    /// it can bias earshot or the noise floor. Zero-state start; `reset()`
    /// clears it so resume doesn't blend pre-pause audio.
    hp_a: f32,
    hp_x_prev: f32,
    hp_y_prev: f32,
    hp_init: bool,
}

impl SpeechDetector {
    pub(crate) fn new(threshold: f32, adapt: bool) -> Self {
        let hp_a = (-2.0 * std::f32::consts::PI * crate::config::CONDITION_HP_CUTOFF_HZ
            / crate::config::WHISPER_RATE as f32)
            .exp();
        Self {
            detector: Detector::default_boxed(),
            staging: Vec::with_capacity(EARSHOT_WINDOW),
            threshold,
            in_speech: false,
            raw_speech: false,
            window_history: VecDeque::from(vec![false; MEDIAN_WINDOWS]),
            last_score: 0.0,
            noise: crate::noise::NoiseTracker::new(),
            adapt,
            hp_a,
            hp_x_prev: 0.0,
            hp_y_prev: 0.0,
            hp_init: false,
        }
    }

    /// Current noise floor (for logging/tuning).
    #[allow(dead_code)]
    pub(crate) fn noise_floor(&self) -> f32 {
        self.noise.floor()
    }

    /// Feed one 10 ms VAD-branch frame (16 kHz, `[-1, 1]`); returns the
    /// current speech verdict with hysteresis plus median smoothing: entering
    /// raw speech needs `score >= enter` (base lifted by background), staying
    /// in raw speech only needs `score >= exit` (enter minus hysteresis), and
    /// the reported verdict is the majority of the last `MEDIAN_WINDOWS` raw
    /// decisions. Non-speech frames keep the background estimate warm; speech
    /// frames freeze it. The frame is high-passed (60 Hz, streaming) before
    /// anything else so hum never reaches earshot or the floor estimate.
    pub(crate) fn verdict(&mut self, frame_16k: &[f32]) -> bool {
        let filtered = self.highpass_frame(frame_16k);
        self.staging.extend_from_slice(&filtered);

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
        // Frames without a fresh score reuse the latched state (same as
        // before) and still count as background when not in speech.
        // The floor learns from the same high-passed audio earshot sees.
        if !self.in_speech {
            // `scored` frames carry a fresh earshot verdict; unscored frames
            // are 10 ms slices of the same background — both are safe.
            let _ = scored;
            self.noise.update_background(&filtered);
        }

        self.in_speech
    }

    /// Streaming 60 Hz one-pole high-pass over one 10 ms frame, preserving
    /// filter state across calls. First sample after `new`/`reset` starts
    /// from zero state (pre-roll silence is near zero, so no click).
    fn highpass_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(frame.len());
        for &x in frame {
            if !self.hp_init {
                // Zero-state start: no transient from a stale `x_prev`.
                self.hp_x_prev = x;
                self.hp_y_prev = 0.0;
                self.hp_init = true;
                out.push(0.0);
                continue;
            }
            let y = x - self.hp_x_prev + self.hp_a * self.hp_y_prev;
            self.hp_x_prev = x;
            self.hp_y_prev = y;
            out.push(y);
        }
        out
    }

    /// Fold one raw window decision into the median: update the hysteresis
    /// state, push it onto the history (capped at `MEDIAN_WINDOWS`), and
    /// recompute the reported verdict as the majority. Split from `verdict`
    /// so tests can drive flicker sequences without a live earshot signal.
    fn push_window(&mut self, raw: bool) {
        self.raw_speech = raw;
        self.window_history.push_back(raw);
        while self.window_history.len() > MEDIAN_WINDOWS {
            self.window_history.pop_front();
        }
        let trues = self.window_history.iter().filter(|&&b| b).count();
        self.in_speech = trues * 2 > self.window_history.len();
    }

    /// Observe-only background update for pause: RMS -> noise floor, no
    /// earshot, no utterance, no latch change. Cheap (no neural pass) so a
    /// 10 s pause on fan noise actually teaches the threshold instead of
    /// idling. Never marks speech. High-passed like `verdict` so the floor
    /// stays consistent with what earshot will see after resume.
    pub(crate) fn observe_background(&mut self, frame_16k: &[f32]) {
        let filtered = self.highpass_frame(frame_16k);
        self.noise.update_background(&filtered);
    }

    /// Drop staged partial-window samples, the hysteresis latch, the median
    /// history, the high-pass state, and the earshot detector's internal
    /// state (ring buffer + feature context + RNN state) for a pause
    /// transition, so resume starts from silence, not pre-pause audio. The
    /// noise floor is deliberately kept: it is the warm background Faza 2
    /// learns from.
    pub(crate) fn reset(&mut self) {
        self.detector.reset();
        self.staging.clear();
        self.in_speech = false;
        self.raw_speech = false;
        self.window_history.clear();
        self.window_history.extend(vec![false; MEDIAN_WINDOWS]);
        self.last_score = 0.0;
        self.hp_x_prev = 0.0;
        self.hp_y_prev = 0.0;
        self.hp_init = false;
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
    fn prediction_cadence_overlaps_every_frame() {
        // 50% overlap (step 128): 8 frames (1280 samples) yield 9 windows
        // ((1280-256)/128+1) with 128 samples left staged as the next
        // window's first half — not empty, and bounded below one window.
        // The point is density: almost every 10 ms frame completes a fresh
        // prediction instead of 5 per 8.
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.0f32; FRAME_16K];
        for _ in 0..8 {
            vad.verdict(&frame);
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
    fn highpass_attenuates_hum_keeps_speech() {
        // 50 Hz hum must shrink through the streaming HP; 1 kHz tone must
        // pass nearly untouched (steady middle, past startup transient).
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let hum: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let mut filtered = Vec::new();
        for chunk in hum.chunks(FRAME_16K) {
            filtered.extend(vad.highpass_frame(chunk));
        }
        let rms = |v: &[f32]| {
            (v.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let before = rms(&hum[400..1200]);
        let after = rms(&filtered[400..1200]);
        assert!(after < before * 0.7, "hum before={before} after={after}");

        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let tone: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let mut filtered = Vec::new();
        for chunk in tone.chunks(FRAME_16K) {
            filtered.extend(vad.highpass_frame(chunk));
        }
        let before = rms(&tone[400..1200]);
        let after = rms(&filtered[400..1200]);
        let ratio = after / before;
        assert!((0.9..=1.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn highpass_state_resets_on_pause() {
        let mut vad = SpeechDetector::new(DEFAULT_VAD_THRESHOLD, true);
        let frame = vec![0.5f32; FRAME_16K];
        let _ = vad.highpass_frame(&frame);
        assert!(vad.hp_init);
        vad.reset();
        assert!(!vad.hp_init);
        assert_eq!(vad.hp_x_prev, 0.0);
        assert_eq!(vad.hp_y_prev, 0.0);
    }
}
