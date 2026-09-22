// ---------------------------------------------------------------------------
// Agc: VAD-gated shared-gain automatic gain control, after Hp, before VAD +
// utterance.
// ---------------------------------------------------------------------------
//
// One instance applies the same `cur_gain` to both the VAD and whisper frames
// so the verdict always describes the audio being buffered. Gating uses the
// previous frame's verdict (1-frame delay breaks the VAD<->gain feedback
// loop): silence frames pass through at unity and never adapt, speech frames
// are amplified toward the target and adapt the envelope.
//
// Freeze-on-silence + max cap + slow release keep noise from pumping; initial
// gain is unity (G.169). Silence tail therefore stays raw while speech is
// normalized — the RMS/density gates then see rescued quiet speech, as
// intended.

use crate::config::{AGC_ATTACK_MS, AGC_RELEASE_MS, WHISPER_RATE};

/// Below this frame RMS the frame never adapts (activation threshold, ~-50 dBFS).
pub(crate) const AGC_MIN_RMS: f32 = crate::config::DEFAULT_MIN_SPEECH_RMS;

/// Shared-gain AGC state. Mono only; 16 kHz framing.
pub(crate) struct Agc {
    target: f32,
    max_gain: f32,
    attack_coef: f32,
    release_coef: f32,
    envelope: f32,
    cur_gain: f32,
}

impl Agc {
    pub(crate) fn new(target: f32, max_gain: f32) -> Self {
        let sr = WHISPER_RATE as f32;
        let attack_coef = (-1.0 / (AGC_ATTACK_MS * 0.001 * sr)).exp();
        let release_coef = (-1.0 / (AGC_RELEASE_MS * 0.001 * sr)).exp();
        Self {
            target: target.max(0.0),
            max_gain: max_gain.max(1.0),
            attack_coef,
            release_coef,
            envelope: target.max(1e-6),
            cur_gain: 1.0,
        }
    }

    /// Current applied gain (for logging/tests).
    #[allow(dead_code)]
    pub(crate) fn gain(&self) -> f32 {
        self.cur_gain
    }

    /// Process one 10 ms frame in place (single-branch helper; production uses
    /// `process_pair`). When `speech` is false the frame passes untouched and
    /// nothing adapts — never on silence.
    #[allow(dead_code)]
    pub(crate) fn process(&mut self, frame: &mut [f32], speech: bool) {
        if !speech || self.target <= 0.0 {
            return;
        }
        let rms = frame_rms(frame);
        if rms < AGC_MIN_RMS || !rms.is_finite() {
            return;
        }
        let n = frame.len().max(1) as f32;
        let frame_attack = self.attack_coef.powf(n);
        let frame_release = self.release_coef.powf(n);
        // Envelope follower: fast attack on rise, slow release on fall.
        if rms > self.envelope {
            self.envelope = frame_attack * self.envelope + (1.0 - frame_attack) * rms;
        } else {
            self.envelope = frame_release * self.envelope + (1.0 - frame_release) * rms;
        }
        let desired = (self.target / self.envelope.max(1e-6)).clamp(1.0, self.max_gain);
        // Smooth gain toward desired with the same time constants to avoid clicks.
        // Rising gain (louder output) uses release (slow); falling uses attack (fast).
        let (frame_coef, goal) = if desired > self.cur_gain {
            (frame_release, desired)
        } else {
            (frame_attack, desired)
        };
        self.cur_gain = frame_coef * self.cur_gain + (1.0 - frame_coef) * goal;
        self.cur_gain = self.cur_gain.clamp(1.0, self.max_gain);
        // Clip safety: never push peak past 0.99.
        let peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        if peak > 0.0 {
            let max_for_clip = 0.99 / peak;
            let applied = self.cur_gain.min(max_for_clip).max(1.0);
            // Don't let a single hot sample collapse the loop state; only limit
            // this frame's application.
            for s in frame.iter_mut() {
                *s *= applied;
            }
            return;
        }
        for s in frame.iter_mut() {
            *s *= self.cur_gain;
        }
    }

    /// Process a VAD/whisper pair with shared gain, gated by the previous
    /// verdict (1-frame delay breaks the VAD<->gain loop). Silence passes
    /// through at unity with no adaptation; speech adapts the envelope from
    /// the whisper (stored) branch then applies the same gain to both so the
    /// verdict always describes the buffered audio.
    ///
    /// Returns `whisper`'s RMS *before* any gain is applied, on every code
    /// path (including the silence/gated-off early returns). Callers use
    /// this as the utterance buffer's acoustic evidence (`Evidence.rms`,
    /// `gate::floor_for`'s weak/strong split) instead of `frame_rms` on the
    /// (possibly gained) buffered audio — evidence should describe what the
    /// microphone actually heard, not what this AGC did to it afterward.
    pub(crate) fn process_pair(
        &mut self,
        vad: &mut [f32],
        whisper: &mut [f32],
        prev_speech: bool,
    ) -> f32 {
        let rms = frame_rms(whisper);
        if !prev_speech || self.target <= 0.0 {
            return rms;
        }
        if rms < AGC_MIN_RMS || !rms.is_finite() {
            return rms;
        }
        let n = whisper.len().max(1) as f32;
        let frame_attack = self.attack_coef.powf(n);
        let frame_release = self.release_coef.powf(n);
        if rms > self.envelope {
            self.envelope = frame_attack * self.envelope + (1.0 - frame_attack) * rms;
        } else {
            self.envelope = frame_release * self.envelope + (1.0 - frame_release) * rms;
        }
        let desired = (self.target / self.envelope.max(1e-6)).clamp(1.0, self.max_gain);
        let (frame_coef, goal) = if desired > self.cur_gain {
            (frame_release, desired)
        } else {
            (frame_attack, desired)
        };
        self.cur_gain = (frame_coef * self.cur_gain + (1.0 - frame_coef) * goal)
            .clamp(1.0, self.max_gain);
        for buf in [vad, whisper] {
            let peak = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            let applied = if peak > 0.0 {
                self.cur_gain.min(0.99 / peak).max(1.0)
            } else {
                self.cur_gain
            };
            if applied > 1.0 {
                for s in buf.iter_mut() {
                    *s *= applied;
                }
            }
        }
        rms
    }

    /// Reset to unity gain + target envelope for pause/discard transitions.
    pub(crate) fn reset(&mut self) {
        self.envelope = self.target.max(1e-6);
        self.cur_gain = 1.0;
    }
}

fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / frame.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FRAME_16K;

    #[test]
    fn silence_never_adapts_or_amplifies() {
        let mut agc = Agc::new(0.04, 8.0);
        let mut silence = vec![0.0005f32; FRAME_16K];
        let before = silence.clone();
        for _ in 0..50 {
            agc.process(&mut silence, false);
        }
        assert_eq!(agc.gain(), 1.0);
        assert_eq!(silence, before);
        // Even speech-gated, sub-threshold frames never adapt.
        let mut quiet = vec![0.0005f32; FRAME_16K];
        agc.process(&mut quiet, true);
        assert_eq!(agc.gain(), 1.0);
    }

    #[test]
    fn speech_converges_toward_target() {
        let mut agc = Agc::new(0.04, 8.0);
        let frame = vec![0.01f32; FRAME_16K];
        for _ in 0..200 {
            agc.process(&mut frame.clone(), true);
            // Recompute fresh frame each time to avoid compounding; gain state
            // is what we assert.
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
        let mut loud = vec![0.2f32; FRAME_16K];
        let before = loud.clone();
        agc.process(&mut loud, true);
        assert_eq!(agc.gain(), 1.0);
        assert_eq!(loud, before);
    }

    #[test]
    fn reset_returns_to_unity() {
        let mut agc = Agc::new(0.04, 8.0);
        let frame = vec![0.01f32; FRAME_16K];
        for _ in 0..50 {
            agc.process(&mut frame.clone(), true);
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
            let mut whisper = vec![0.005f32; FRAME_16K];
            agc.process_pair(&mut vad, &mut whisper, true);
        }
        assert!(agc.gain() > 1.0, "gain must have risen, got {}", agc.gain());

        let mut vad = vec![0.005f32; FRAME_16K];
        let mut whisper = vec![0.005f32; FRAME_16K];
        let pre_gain_rms = agc.process_pair(&mut vad, &mut whisper, true);
        // The buffer itself was amplified...
        let post_gain_rms = frame_rms(&whisper);
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
        let mut whisper = vec![0.01f32; FRAME_16K];
        let rms = agc.process_pair(&mut vad, &mut whisper, false); // prev_speech=false
        assert!((rms - 0.01).abs() < 1e-4, "got {rms}");
        assert_eq!(whisper, vec![0.01f32; FRAME_16K], "gated-off frame must pass through untouched");

        let mut agc_off = Agc::new(0.0, 8.0); // target <= 0 disables entirely
        let mut vad2 = vec![0.01f32; FRAME_16K];
        let mut whisper2 = vec![0.01f32; FRAME_16K];
        let rms2 = agc_off.process_pair(&mut vad2, &mut whisper2, true);
        assert!((rms2 - 0.01).abs() < 1e-4, "got {rms2}");
    }
}
