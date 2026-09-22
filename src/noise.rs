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

/// Slow background RMS estimate. Warm across pause, frozen across speech.
#[derive(Debug, Clone)]
pub(crate) struct NoiseTracker {
    floor: f32,
}

impl NoiseTracker {
    pub(crate) fn new() -> Self {
        Self { floor: 0.001 }
    }

    /// Current floor estimate (linear RMS, not dB).
    pub(crate) fn floor(&self) -> f32 {
        self.floor
    }

    /// RMS of one 16 kHz frame.
    pub(crate) fn frame_rms(frame_16k: &[f32]) -> f32 {
        if frame_16k.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = frame_16k.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / frame_16k.len() as f64).sqrt() as f32
    }

    /// Update from a background frame (pause, or live non-speech). Asymmetric:
    /// drops fast (new quiet room trusted quickly), rises slow (a loud burst
    /// must persist before it lifts the threshold).
    pub(crate) fn update_background(&mut self, frame_16k: &[f32]) {
        let rms = Self::frame_rms(frame_16k);
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
    /// on through a weaker window instead of chopping on one flicker.
    pub(crate) fn exit_threshold(&self, base: f32) -> f32 {
        (self.enter_threshold(base) - VAD_HYSTERESIS).clamp(0.0, 1.0)
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
        assert_eq!(NoiseTracker::frame_rms(&[]), 0.0);
        let mut n = NoiseTracker::new();
        n.update_background(&[]);
        assert!(n.floor() < 0.001);
    }
}
