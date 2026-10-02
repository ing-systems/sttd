// ---------------------------------------------------------------------------
// Agc: VAD-gated shared-gain automatic gain control, after Hp, before VAD +
// utterance.
// ---------------------------------------------------------------------------
//
// One instance applies the same `cur_gain` to both the VAD and model frames
// so the verdict always describes the audio being buffered. Gating uses the
// previous frame's verdict (1-frame delay breaks the VAD<->gain feedback
// loop): silence frames pass through at unity and never adapt, speech frames
// are amplified toward the target and adapt the envelope.
//
// Freeze-on-silence + max cap + slow release keep noise from pumping; initial
// gain is unity (G.169). Silence tail therefore stays raw while speech is
// normalized — the RMS/density gates then see rescued quiet speech, as
// intended.

use crate::config::{AGC_ATTACK_MS, AGC_RELEASE_MS, FRAME_16K, SAMPLE_RATE};
use crate::noise::frame_rms;

/// Below this frame RMS the frame never adapts (activation threshold, ~-50 dBFS).
pub(crate) const AGC_MIN_RMS: f32 = crate::config::DEFAULT_MIN_SPEECH_RMS;

/// Shared-gain AGC state. Mono only; 16 kHz framing.
pub(crate) struct Agc {
    target: f32,
    max_gain: f32,
    /// Per-frame smoothing coefficients: the continuous time constants
    /// (`AGC_ATTACK_MS`/`AGC_RELEASE_MS` at `SAMPLE_RATE`) raised to the power
    /// of one frame. Precomputed once in `new` because every frame is the same
    /// fixed `FRAME_16K` samples, so the two `powf` calls used to run per frame
    /// for a constant.
    frame_attack: f32,
    frame_release: f32,
    envelope: f32,
    cur_gain: f32,
}

impl Agc {
    pub(crate) fn new(target: f32, max_gain: f32) -> Self {
        let sr = SAMPLE_RATE as f32;
        let attack_coef = (-1.0 / (AGC_ATTACK_MS * 0.001 * sr)).exp();
        let release_coef = (-1.0 / (AGC_RELEASE_MS * 0.001 * sr)).exp();
        let frame_len = FRAME_16K as f32;
        Self {
            target: target.max(0.0),
            max_gain: max_gain.max(1.0),
            frame_attack: attack_coef.powf(frame_len),
            frame_release: release_coef.powf(frame_len),
            envelope: target.max(1e-6),
            cur_gain: 1.0,
        }
    }

    /// Current applied gain (test-only reader; production never asks).
    #[cfg(test)]
    pub(crate) const fn gain(&self) -> f32 {
        self.cur_gain
    }

    /// Advance the envelope and the shared gain from `rms` (the model frame's
    /// RMS before any gain is applied). Returns `None` when the frame must
    /// pass through untouched: silence (gated off), a disabled target, or a
    /// frame below the activation threshold. Never adapts on silence.
    fn adapt(&mut self, rms: f32, prev_speech: bool) -> Option<f32> {
        if !prev_speech || self.target <= 0.0 {
            return None;
        }
        if rms < AGC_MIN_RMS || !rms.is_finite() {
            return None;
        }
        // Envelope follower: fast attack on rise, slow release on fall.
        if rms > self.envelope {
            self.envelope = self.frame_attack * self.envelope + (1.0 - self.frame_attack) * rms;
        } else {
            self.envelope = self.frame_release * self.envelope + (1.0 - self.frame_release) * rms;
        }
        let desired = (self.target / self.envelope.max(1e-6)).clamp(1.0, self.max_gain);
        // Smooth gain toward desired with the same time constants to avoid clicks.
        // Rising gain (louder output) uses release (slow); falling uses attack (fast).
        let (frame_coef, goal) = if desired > self.cur_gain {
            (self.frame_release, desired)
        } else {
            (self.frame_attack, desired)
        };
        self.cur_gain =
            (frame_coef * self.cur_gain + (1.0 - frame_coef) * goal).clamp(1.0, self.max_gain);
        Some(self.cur_gain)
    }

    /// Scale one buffer by `gain`, limited so its peak stays below 0.99.
    /// `peak` is the caller's already-measured peak of this buffer. A single
    /// hot sample only limits this frame's application; the loop state keeps
    /// the smoother gain.
    fn apply_gain(buf: &mut [f32], gain: f32, peak: f32) {
        let applied = if peak > 0.0 {
            gain.min(0.99 / peak).max(1.0)
        } else {
            gain
        };
        if applied > 1.0 {
            for s in buf.iter_mut() {
                *s *= applied;
            }
        }
    }

    /// Process a VAD/model pair with shared gain, gated by the previous
    /// verdict (1-frame delay breaks the VAD<->gain loop). Silence passes
    /// through at unity with no adaptation; speech adapts the envelope from
    /// the model (stored) branch then applies the same gain to both so the
    /// verdict always describes the buffered audio.
    ///
    /// Returns the model frame's RMS *before* any gain is applied, on every code
    /// path (including the silence/gated-off early returns). Callers use
    /// this as the utterance buffer's acoustic evidence (`Evidence.rms`,
    /// `gate::floor_for`'s weak/strong split) instead of `frame_rms` on the
    /// (possibly gained) buffered audio — evidence should describe what the
    /// microphone actually heard, not what this AGC did to it afterward.
    pub(crate) fn process_pair(
        &mut self,
        vad: &mut [f32],
        model: &mut [f32],
        prev_speech: bool,
    ) -> f32 {
        debug_assert_eq!(model.len(), FRAME_16K, "frames are fixed size");
        let rms = frame_rms(model);
        if let Some(gain) = self.adapt(rms, prev_speech) {
            for buf in [vad, model] {
                let peak = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                Self::apply_gain(buf, gain, peak);
            }
        }
        rms
    }

    /// Same as `process_pair`, for the default `Direct` topology where one
    /// allocation serves both roles: the gain is measured and applied once,
    /// and the caller reads the same samples for the VAD verdict and for the
    /// utterance buffer. Applying the pair's per-buffer gain twice to one
    /// buffer would be a double gain, so the shared path scales once.
    pub(crate) fn process_shared(&mut self, frame: &mut [f32], prev_speech: bool) -> f32 {
        debug_assert_eq!(frame.len(), FRAME_16K, "frames are fixed size");
        let rms = frame_rms(frame);
        if let Some(gain) = self.adapt(rms, prev_speech) {
            // Clip safety: never push peak past 0.99.
            let peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            Self::apply_gain(frame, gain, peak);
        }
        rms
    }

    /// Reset to unity gain + target envelope for pause/discard transitions.
    pub(crate) const fn reset(&mut self) {
        self.envelope = self.target.max(1e-6);
        self.cur_gain = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FRAME_16K;

    /// Feed one speech-gated frame pair with the same content on both
    /// branches (the default `Direct` topology) and hand both buffers back
    /// for inspection.
    fn adapt_pair(agc: &mut Agc, level: f32, speech: bool) -> (Vec<f32>, Vec<f32>) {
        let mut vad = vec![level; FRAME_16K];
        let mut model = vec![level; FRAME_16K];
        agc.process_pair(&mut vad, &mut model, speech);
        (vad, model)
    }

    #[test]
    fn silence_never_adapts_or_amplifies() {
        let mut agc = Agc::new(0.04, 8.0);
        let silence = vec![0.0005f32; FRAME_16K];
        for _ in 0..50 {
            let (vad, model) = adapt_pair(&mut agc, 0.0005, false);
            assert_eq!(vad, silence);
            assert_eq!(model, silence);
        }
        assert_eq!(agc.gain(), 1.0);
        // Even speech-gated, sub-threshold frames never adapt.
        adapt_pair(&mut agc, 0.0005, true);
        assert_eq!(agc.gain(), 1.0);
    }

    #[test]
    fn speech_converges_toward_target() {
        let mut agc = Agc::new(0.04, 8.0);
        for _ in 0..200 {
            // Fresh frame each time to avoid compounding; gain state is what we
            // assert.
            adapt_pair(&mut agc, 0.01, true);
        }
        assert!(
            agc.gain() > 1.5,
            "quiet speech must lift gain, got {}",
            agc.gain()
        );
        assert!(agc.gain() <= 8.0);
    }

    #[test]
    fn loud_speech_stays_at_unity() {
        let mut agc = Agc::new(0.04, 8.0);
        let loud = vec![0.2f32; FRAME_16K];
        let (vad, model) = adapt_pair(&mut agc, 0.2, true);
        assert_eq!(agc.gain(), 1.0);
        assert_eq!(vad, loud);
        assert_eq!(model, loud);
    }

    #[test]
    fn reset_returns_to_unity() {
        let mut agc = Agc::new(0.04, 8.0);
        for _ in 0..50 {
            adapt_pair(&mut agc, 0.01, true);
        }
        assert!(agc.gain() > 1.0);
        agc.reset();
        assert_eq!(agc.gain(), 1.0);
    }

    #[test]
    fn process_pair_returns_pre_gain_rms_not_post_gain() {
        // Regression for the strong/weak evidence collapse (M1): callers use
        // this return value to score acoustic evidence (see
        // UtteranceBuffer::push_frame_evidenced), so it must describe the
        // microphone, not what this frame ends up amplified to.
        let mut agc = Agc::new(0.04, 8.0);
        // Warm the envelope toward a quiet target first so the first real
        // speech frame below gets a real (>1x) gain applied, not the initial
        // unity gain.
        for _ in 0..50 {
            let mut vad = vec![0.005f32; FRAME_16K];
            let mut model = vec![0.005f32; FRAME_16K];
            agc.process_pair(&mut vad, &mut model, true);
        }
        assert!(agc.gain() > 1.0, "gain must have risen, got {}", agc.gain());

        let mut vad = vec![0.005f32; FRAME_16K];
        let mut model = vec![0.005f32; FRAME_16K];
        let pre_gain_rms = agc.process_pair(&mut vad, &mut model, true);
        // The buffer itself was amplified...
        let post_gain_rms = frame_rms(&model);
        assert!(
            post_gain_rms > pre_gain_rms * 1.2,
            "sanity: frame must actually have been amplified, pre={pre_gain_rms} post={post_gain_rms}"
        );
        // ...but the returned value must still be the original, pre-gain RMS.
        assert!(
            (pre_gain_rms - 0.005).abs() < 1e-4,
            "process_pair must return the pre-gain RMS, got {pre_gain_rms}"
        );
    }

    #[test]
    fn process_pair_returns_rms_even_when_gated_off() {
        // Both early-return paths (not prev_speech, or target <= 0) must
        // still report the frame's RMS rather than 0.0/garbage — a caller
        // scoring acoustic evidence on a silence-gated frame needs the real
        // value just as much as on an adapted one.
        let mut agc = Agc::new(0.04, 8.0);
        let mut vad = vec![0.01f32; FRAME_16K];
        let mut model = vec![0.01f32; FRAME_16K];
        let rms = agc.process_pair(&mut vad, &mut model, false); // prev_speech=false
        assert!((rms - 0.01).abs() < 1e-4, "got {rms}");
        assert_eq!(
            model,
            vec![0.01f32; FRAME_16K],
            "gated-off frame must pass through untouched"
        );

        let mut agc_off = Agc::new(0.0, 8.0); // target <= 0 disables entirely
        let mut vad2 = vec![0.01f32; FRAME_16K];
        let mut model2 = vec![0.01f32; FRAME_16K];
        let rms2 = agc_off.process_pair(&mut vad2, &mut model2, true);
        assert!((rms2 - 0.01).abs() < 1e-4, "got {rms2}");
    }
}
