// ---------------------------------------------------------------------------
// Slaney mel filterbank, shared by both speech frontends.
// ---------------------------------------------------------------------------
//
// Moved out of `parakeet_core/frontend.rs`, which owned it before a second
// frontend needed it: Parakeet-TDT (`n_fft` 512, 128 mels) and Whisper
// large-v3-turbo (`n_fft` 400, 128 mels) share the librosa recipe below and
// differ only in those two numbers. One table, one owner, so neither engine's
// mel scale can drift from the other's.
//
// The scale is librosa's: Slaney mel (`htk = false`) with `norm = "slaney"`
// area weighting, `fmax = sample_rate / 2`. Parakeet's `nemo128` filterbank and
// Whisper's `mel_filters.npz` are both this table.
//
// `Stft`, the power-spectrogram step both frontends run before the filterbank,
// lives here for the same reason. The frontends differ only in the window
// (symmetric vs periodic Hann) and how they pad the signal, which stay theirs.

use std::sync::Arc;

use ndarray17::{Array2, ArrayView2};
use realfft::num_complex::Complex;
use realfft::{FftError, RealToComplex};

/// Slaney mel scale (librosa).
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

/// Triangular mel filterbank as `[n_mels, n_fft / 2 + 1]`, one row per mel
/// band, area-normalized (librosa `norm = "slaney"`).
pub(crate) fn create_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);

    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }
    // Slaney area normalization.
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for k in 0..freq_bins {
            filterbank[[i, k]] *= enorm as f32;
        }
    }
    filterbank
}

/// Power spectrogram (`|FFT|^2` of each windowed frame) with its FFT plan and
/// every buffer built once, so a frontend allocates nothing per utterance
/// beyond growing the spectrogram to the longest one seen.
pub(crate) struct Stft {
    plan: Arc<dyn RealToComplex<f32>>,
    /// Analysis window, at most `n_fft` long; a shorter one is zero-padded.
    window: Vec<f32>,
    /// One windowed frame (n_fft samples).
    frame: Vec<f32>,
    out: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    /// `[n_fft / 2 + 1, frames]` row-major, grown on demand and never shrunk:
    /// a shorter utterance is a prefix of it.
    spectrogram: Vec<f32>,
}

impl Stft {
    pub(crate) fn new(n_fft: usize, window: Vec<f32>) -> Self {
        assert!(window.len() <= n_fft, "analysis window longer than n_fft");
        let plan = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);
        Self {
            out: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            frame: vec![0.0; n_fft],
            plan,
            window,
            spectrogram: Vec::new(),
        }
    }

    /// Power spectrogram of `n_frames` frames of `padded`, frame `i` starting
    /// at `i * hop`, as `[n_fft / 2 + 1, n_frames]`. `padded` must hold a full
    /// window for every frame.
    pub(crate) fn power(
        &mut self,
        padded: &[f32],
        hop: usize,
        n_frames: usize,
    ) -> Result<ArrayView2<'_, f32>, FftError> {
        let win_length = self.window.len();
        let freq_bins = self.out.len();
        let len = freq_bins * n_frames;
        if self.spectrogram.len() < len {
            self.spectrogram.resize(len, 0.0);
        }
        let spectrogram = &mut self.spectrogram[..len];
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            for (dst, (&sample, &taper)) in self.frame[..win_length]
                .iter_mut()
                .zip(padded[start..start + win_length].iter().zip(&self.window))
            {
                *dst = sample * taper;
            }
            // The FFT uses its input as scratch, so a window shorter than
            // `n_fft` needs its zero tail restored every frame; the head was
            // just rewritten.
            self.frame[win_length..].fill(0.0);
            self.plan
                .process_with_scratch(&mut self.frame, &mut self.out, &mut self.scratch)?;
            for (k, value) in self.out.iter().enumerate() {
                spectrogram[k * n_frames + frame_idx] = value.norm_sqr();
            }
        }
        Ok(
            ArrayView2::from_shape((freq_bins, n_frames), &self.spectrogram[..len])
                .expect("spectrogram buffer holds freq_bins * n_frames"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mel band must be a non-negative, non-empty triangle, and the
    /// bands must march upward in frequency: the lowest mel owns the lowest
    /// bin, the highest mel the highest one.
    #[test]
    fn filterbank_is_slaney_triangles_in_ascending_frequency() {
        for (n_fft, n_mels) in [(512usize, 128usize), (400, 128)] {
            let bank = create_mel_filterbank(n_fft, n_mels, 16_000);
            let freq_bins = n_fft / 2 + 1;
            assert_eq!(bank.shape(), &[n_mels, freq_bins]);
            let mut last_peak = 0usize;
            for mel in 0..n_mels {
                let row: Vec<f32> = (0..freq_bins).map(|k| bank[[mel, k]]).collect();
                assert!(
                    row.iter().all(|&v| v >= 0.0),
                    "mel {mel} has a negative weight"
                );
                let sum: f32 = row.iter().sum();
                assert!(sum > 0.0, "mel {mel} is an empty band");
                let peak = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(k, _)| k)
                    .unwrap();
                assert!(peak >= last_peak, "mel {mel} peaks below the previous band");
                last_peak = peak;
            }
            // The highest mel must own the top of the spectrum. Its peak sits
            // at the second-to-last filter edge — the edges run from `mel_min`
            // to `mel_max` in `n_mels + 2` steps, so the last band's centre is
            // one step below `mel_max`, and the topmost bins only carry its
            // tail. Assert on the peak's *frequency* (at least 80% of Nyquist),
            // not on a bin index: how far below the top that lands depends on
            // `n_fft`, and 250 of 256 is correct for this table.
            let top: Vec<f32> = (0..freq_bins).map(|k| bank[[n_mels - 1, k]]).collect();
            let top_peak = top
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k)
                .unwrap();
            let nyquist_bin = freq_bins - 1;
            assert!(
                top_peak * 10 >= nyquist_bin * 8,
                "n_fft={n_fft}: top mel peaks at {}/{} of Nyquist",
                top_peak,
                nyquist_bin
            );
            // The lowest mel must own the bottom of the spectrum: the scale is
            // linear below 1 kHz, so band 0's centre is only tens of hertz up.
            // Its weight at exactly 0 Hz is zero — a triangle is zero at its own
            // edges — so assert on the peak, not on bin 0.
            let bottom: Vec<f32> = (0..freq_bins).map(|k| bank[[0, k]]).collect();
            let bottom_peak = bottom
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k)
                .unwrap();
            assert!(
                bottom_peak * 10 <= (freq_bins - 1),
                "n_fft={n_fft}: bottom mel peaks at bin {bottom_peak}, too high"
            );
        }
    }

    /// Parakeet and Whisper share this table but use different `n_fft`, so
    /// their bin grids differ: bin `k` of a 400-point FFT sits at
    /// `k * 40` Hz while bin `k` of a 512-point one sits at `k * 31.25` Hz.
    /// Weighting is a function of *frequency*, not bin index, so the two
    /// banks must agree wherever they sample the same frequency — that is,
    /// at multiples of 40 Hz the 400-point grid hits exactly and the
    /// 512-point grid hits exactly when `k * 40 / 31.25` is an integer.
    ///
    /// This guards the shared-table move: it fails if either engine ever gets a
    /// filterbank built on a different mel scale or a different `fmax`.
    #[test]
    fn parakeet_and_whisper_banks_agree_at_shared_frequencies() {
        let parakeet = create_mel_filterbank(512, 128, 16_000);
        let whisper = create_mel_filterbank(400, 128, 16_000);
        let mut checked = 0;
        for whisper_bin in 0..=200usize {
            let hz = whisper_bin as f64 * 16_000.0 / 400.0;
            // Only frequencies both grids represent exactly.
            if (hz / (16_000.0 / 512.0)).fract().abs() > 1e-9 {
                continue;
            }
            let parakeet_bin = (hz / (16_000.0 / 512.0)).round() as usize;
            for mel in 0..128 {
                let a = parakeet[[mel, parakeet_bin]];
                let b = whisper[[mel, whisper_bin]];
                assert!(
                    (a - b).abs() < 1e-5,
                    "mel {mel} at {hz} Hz (bins {parakeet_bin} vs {whisper_bin}): {a} vs {b}"
                );
            }
            checked += 1;
        }
        assert!(
            checked > 8,
            "only {checked} shared frequencies; test is too weak"
        );
    }
}
