// ---------------------------------------------------------------------------
// Hp: 80 Hz high-pass after resampling, before VAD + utterance.
// ---------------------------------------------------------------------------
//
// Single fixed-rate (16 kHz) biquad so no per-hw-rate coefficients are needed.
// `Direct` mode filters once (shared allocation); `Denoised` mode filters each
// 16 kHz branch with its own state (denoised vs raw must never share a delay
// line). Replaces the old 60 Hz one-poles in `vad` + `condition_utterance`.

use biquad::{Biquad, Coefficients, DirectForm2Transposed, Q_BUTTERWORTH_F32, Type};

/// High-pass cutoff (Hz). Removes mains hum/rumble/plosive wind while keeping
/// Polish speech intact (male F0 ~85 Hz+).
pub(crate) const HP_HZ: f32 = 80.0;
/// Rate the filter is designed for. All Hp instances run at 16 kHz (post-
/// resample); no hw-rate instances exist.
pub(crate) const HP_RATE_HZ: u32 = crate::config::SAMPLE_RATE;

/// Streaming 80 Hz biquad high-pass (Butterworth Q, 12 dB/oct).
pub(crate) struct Hp {
    inner: DirectForm2Transposed<f32>,
}

impl Hp {
    pub(crate) fn new() -> Self {
        let coeffs = Coefficients::<f32>::from_params(
            Type::HighPass,
            HP_RATE_HZ.hz(),
            HP_HZ.hz(),
            Q_BUTTERWORTH_F32,
        )
        .expect("80 Hz HP coeffs must be valid at 16 kHz");
        Self {
            inner: DirectForm2Transposed::<f32>::new(coeffs),
        }
    }

    /// Filter one 10 ms frame in place, preserving state across calls.
    pub(crate) fn process(&mut self, frame: &mut [f32]) {
        for s in frame.iter_mut() {
            *s = self.inner.run(*s);
        }
    }

    /// Drop filter memory so post-pause audio isn't blended from pre-pause.
    pub(crate) fn reset(&mut self) {
        self.inner.reset_state();
    }
}

use biquad::ToHertz;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FRAME_16K;

    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn hp_consts_are_80hz_at_16k() {
        assert_eq!(HP_HZ, 80.0);
        assert_eq!(HP_RATE_HZ, 16_000);
    }

    #[test]
    fn hp_attenuates_hum_keeps_speech() {
        let mut hp = Hp::new();
        let hum: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let mut filtered = hum.clone();
        for chunk in filtered.chunks_mut(FRAME_16K) {
            hp.process(chunk);
        }
        let before = rms(&hum[2000..14_000]);
        let after = rms(&filtered[2000..14_000]);
        assert!(after < before * 0.5, "before={before} after={after}");

        let mut hp = Hp::new();
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let mut filtered = tone.clone();
        for chunk in filtered.chunks_mut(FRAME_16K) {
            hp.process(chunk);
        }
        let before = rms(&tone[2000..14_000]);
        let after = rms(&filtered[2000..14_000]);
        let ratio = after / before;
        assert!((0.9..=1.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn hp_blocks_dc() {
        let mut hp = Hp::new();
        let dc = vec![0.5f32; 16_000];
        let mut out = dc.clone();
        for chunk in out.chunks_mut(FRAME_16K) {
            hp.process(chunk);
        }
        assert!(rms(&out[2000..]) < 1e-3);
    }

    #[test]
    fn hp_reset_clears_memory() {
        let mut hp = Hp::new();
        let loud = vec![0.5f32; FRAME_16K];
        hp.process(&mut loud.clone());
        hp.reset();
        let mut hp2 = Hp::new();
        let mut a = vec![0.0f32; FRAME_16K];
        let mut b = a.clone();
        hp.process(&mut a);
        hp2.process(&mut b);
        assert_eq!(a, b);
    }
}
