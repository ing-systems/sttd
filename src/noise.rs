// ---------------------------------------------------------------------------
// Background noise tracker: the warm state that survives pause.
//
// Split from `SpeechDetector`'s inference state on purpose: the detector's
// RNN/ring buffer is reset on pause (resume starts from silence, see `vad`),
// but the background estimate must stay warm — otherwise a 10 s pause on fan
// noise teaches nothing and resume false-triggers on the same noise.
//
// Fed RMS-only (no earshot) so pause observe is cheap: resample -> RMS ->
// EMA. During live mode only non-speech frames update it, so speech never
// raises the floor. Pause frames are all background by definition.
// ---------------------------------------------------------------------------

/// EMA speed for the noise floor (~2 s at 100 fps). Slow on purpose: a door
/// slam must not move the floor, a fan ramp-up must.
pub(crate) const NOISE_EMA_ALPHA: f32 = 0.005;
/// How much RMS floor lifts the VAD enter threshold: `enter = base + floor *
/// GAIN`. Floor 0.01 (loud room) -> +0.1; floor 0.001 (quiet) -> +0.01.
pub(crate) const NOISE_GAIN: f32 = 10.0;
/// Max lift above base, so a very loud room can't wedge the VAD shut.
pub(crate) const NOISE_MAX_LIFT: f32 = 0.2;
/// Hysteresis between enter/exit: exit = enter - this. Kills single-window
/// flicker without touching the base threshold.
pub(crate) const VAD_HYSTERESIS: f32 = 0.1;
/// Absolute floor for `exit_threshold`, independent of `base`. A `--vad-
/// threshold`/`STTD_VAD_THRESHOLD` below `VAD_HYSTERESIS` (0.1) — legal input,
/// `parse_fraction` only rejects outside `0.0..=1.0` — would otherwise clamp
/// `enter_threshold(base) - VAD_HYSTERESIS` to exactly `0.0`; earshot's score
/// is never negative, so `score >= 0.0` is then always true and `raw_speech`
/// latches permanently open once it first triggers (see
/// `SpeechDetector::verdict`), silencing the noise floor forever and forcing
/// every utterance to the 30 s hard cap. `0.02` sits well below real speech
/// scores but above silence-level earshot noise, so the latch can always
/// break regardless of how `--vad-threshold` is configured.
pub(crate) const EXIT_THRESHOLD_FLOOR: f32 = 0.02;

// ---------------------------------------------------------------------------
// Frame measurement: one definition of frame loudness for the whole pipeline.
// ---------------------------------------------------------------------------
//
// `agc` (envelope follower, pre-gain evidence), `utterance` (quiet run, early
// split, weighted density) and this module's floor estimator all read frame
// loudness, and their thresholds are only comparable because they all measure
// the same way. It lived in three copies (one per module) until they shared
// this one. Accumulation is in `f64` — a 160-sample `f32` sum loses ~7 digits
// of precision — and the result is narrowed once, at the end.

/// RMS of one frame, linear (not dB). An empty frame is silence.
///
/// `#[inline]`: the per-frame loudness measure, called two or three times per
/// 10 ms frame from `frame`, `utterance` and `agc`.
#[inline]
pub(crate) fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / frame.len() as f64).sqrt() as f32
}

/// Largest absolute sample (0 for an empty buffer).
#[inline]
pub(crate) fn frame_peak(frame: &[f32]) -> f32 {
    frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// Slow background RMS estimate. Warm across pause, frozen across speech.
#[derive(Debug, Clone)]
pub(crate) struct NoiseTracker {
    floor: f32,
}

impl NoiseTracker {
    pub(crate) const fn new() -> Self {
        Self { floor: 0.001 }
    }

    /// Current floor estimate (linear RMS, not dB).
    pub(crate) const fn floor(&self) -> f32 {
        self.floor
    }

    /// Adopt a floor learned elsewhere (see `SpeechDetector::set_floor`),
    /// clamped the same as `update_background`.
    pub(crate) const fn set_floor(&mut self, floor: f32) {
        self.floor = floor.clamp(0.0001, 0.05);
    }

    /// Update from a background frame, measured here. Test-only: production
    /// (`SpeechDetector::verdict`) passes the RMS it already has.
    #[cfg(test)]
    pub(crate) fn update_background(&mut self, frame_16k: &[f32]) {
        self.update_background_rms(frame_rms(frame_16k))
    }

    /// Update from a background frame's own RMS: the caller already measured the
    /// frame (the VAD gets its RMS from the pipeline), so measuring it again
    /// here would scan the same 160 samples twice.
    pub(crate) fn update_background_rms(&mut self, rms: f32) {
        if rms < self.floor {
            // 4x faster down: leaving a noisy room adapts in ~0.5 s.
            let a = (NOISE_EMA_ALPHA * 4.0).min(1.0);
            self.floor += a * (rms - self.floor);
        } else {
            self.floor += NOISE_EMA_ALPHA * (rms - self.floor);
        }
        self.floor = self.floor.clamp(0.0001, 0.05);
    }

    /// Enter threshold for the current background: base lifted by floor.
    pub(crate) fn enter_threshold(&self, base: f32) -> f32 {
        let lifted = base + (self.floor * NOISE_GAIN).min(NOISE_MAX_LIFT);
        lifted.clamp(0.0, 1.0).max(base.min(1.0))
    }

    /// Exit threshold: enter minus hysteresis. Once open, an utterance holds
    /// on through a weaker window instead of chopping on one flicker. Never
    /// below `EXIT_THRESHOLD_FLOOR` — see its doc comment for why a bare
    /// `clamp(0.0, 1.0)` here isn't safe for every valid `base`.
    pub(crate) fn exit_threshold(&self, base: f32) -> f32 {
        (self.enter_threshold(base) - VAD_HYSTERESIS)
            .clamp(0.0, 1.0)
            .max(EXIT_THRESHOLD_FLOOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_room_keeps_base_threshold() {
        let mut n = NoiseTracker::new();
        let quiet = vec![0.001f32; 160];
        for _ in 0..200 {
            n.update_background(&quiet);
        }
        let enter = n.enter_threshold(0.5);
        assert!(
            (enter - 0.5).abs() < 0.03,
            "quiet floor={} enter={enter}",
            n.floor()
        );
        assert!(n.exit_threshold(0.5) < enter);
    }

    #[test]
    fn exit_threshold_never_hits_zero_on_a_low_base() {
        // Regression: a quiet room (near-zero floor lift) with a low
        // --vad-threshold used to clamp exit_threshold to exactly 0.0, and
        // any valid earshot score (>= 0.0) then satisfies `score >= exit`
        // unconditionally, latching the VAD permanently open. A low base is
        // legal input (parse_fraction only rejects outside 0.0..=1.0), so
        // the floor must hold regardless of configuration.
        let n = NoiseTracker::new(); // fresh: floor lift ~0
        for base in [0.0, 0.02, 0.05, 0.09] {
            let exit = n.exit_threshold(base);
            assert!(
                exit >= EXIT_THRESHOLD_FLOOR,
                "base={base}: exit={exit} below floor"
            );
            assert!(
                exit > 0.0,
                "base={base}: exit={exit} must never be exactly 0"
            );
        }
        // A high base is unaffected by the floor (hysteresis alone clears it).
        assert!(n.exit_threshold(0.5) > EXIT_THRESHOLD_FLOOR + 0.1);
    }

    #[test]
    fn loud_room_lifts_threshold_with_cap() {
        let mut n = NoiseTracker::new();
        let loud = vec![0.02f32; 160];
        for _ in 0..2000 {
            n.update_background(&loud);
        }
        let enter = n.enter_threshold(0.5);
        assert!(enter > 0.5, "enter={enter} floor={}", n.floor());
        assert!(enter <= 0.5 + NOISE_MAX_LIFT + 1e-6, "enter={enter}");
        assert_eq!(n.exit_threshold(0.5), enter - VAD_HYSTERESIS);
    }

    #[test]
    fn set_floor_adopts_and_clamps() {
        let mut n = NoiseTracker::new();
        n.set_floor(0.01);
        assert!((n.floor() - 0.01).abs() < 1e-6);
        // Clamped the same as update_background: out-of-range input can't
        // wedge the tracker outside its valid range.
        n.set_floor(-1.0);
        assert_eq!(n.floor(), 0.0001);
        n.set_floor(10.0);
        assert_eq!(n.floor(), 0.05);
    }

    #[test]
    fn floor_drops_fast_rises_slow() {
        let mut n = NoiseTracker::new();
        let loud = vec![0.02f32; 160];
        for _ in 0..2000 {
            n.update_background(&loud);
        }
        let high = n.floor();
        let quiet = vec![0.0005f32; 160];
        for _ in 0..100 {
            n.update_background(&quiet);
        }
        let after_drop = n.floor();
        assert!(after_drop < high, "high={high} after={after_drop}");
        // Rising back takes longer than dropping: 100 loud frames from quiet
        // must not reach the 2000-frame level.
        let mut n2 = NoiseTracker::new();
        for _ in 0..100 {
            n2.update_background(&loud);
        }
        assert!(n2.floor() < high, "fast rise would be {}", n2.floor());
    }

    #[test]
    fn empty_frame_is_silence() {
        assert_eq!(frame_rms(&[]), 0.0);
        let mut n = NoiseTracker::new();
        n.update_background(&[]);
        assert!(n.floor() < 0.001);
    }
}
