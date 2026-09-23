// Mel frontend matching `parakeet-rs` TDT-v3 preprocessing exactly:
// preemphasis 0.97 -> reflect-pad STFT (n_fft 512, hop 160, win 400,
// Hann) -> 128-bin Slaney mel -> ln(x + 2^-24) -> per-feature 0-mean,
// unit-variance (Bessel N-1, eps 1e-5). Pure Rust + `realfft` so the CPU
// and CUDA paths share bit-identical features.

use anyhow::{Context, Result};
use rayon::prelude::*;

/// TDT-v3 preprocessor constants (see `parakeet_rs::ParakeetTDT::from_pretrained`).
pub(crate) const TDT_FEATURE_SIZE: usize = 128;
pub(crate) const TDT_N_FFT: usize = 512;
pub(crate) const TDT_HOP: usize = 160;
pub(crate) const TDT_WIN: usize = 400;
pub(crate) const TDT_PREEMPHASIS: f32 = 0.97;
pub(crate) const TDT_LOG_GUARD: f32 = 5.9604645e-8; // 2^-24

pub(crate) struct MelFrontend {
    mel_basis: Vec<Vec<f32>>, // [n_mels][n_fft/2+1]
    window: Vec<f32>,        // cached Hann (rebuilt per call before)
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>, // cached plan
    n_mels: usize,
    n_fft: usize,
    hop: usize,
    win: usize,
    preemphasis: f32,
}

impl MelFrontend {
    pub(crate) fn tdt_v3() -> Self {
        Self::new(
            TDT_FEATURE_SIZE,
            TDT_N_FFT,
            TDT_HOP,
            TDT_WIN,
            TDT_PREEMPHASIS,
        )
    }

    pub(crate) fn new(n_mels: usize, n_fft: usize, hop: usize, win: usize, preemphasis: f32) -> Self {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        Self {
            mel_basis: slaney_mel_basis(n_fft, n_mels, 16_000),
            window: hann_window(win),
            fft: planner.plan_fft_forward(n_fft),
            n_mels,
            n_fft,
            hop,
            win,
            preemphasis,
        }
    }

    /// 16 kHz mono -> `[frames][n_mels]` normalized log-mel.
    pub(crate) fn extract(&self, pcm_16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut feats = self.unnormalized_log_mel(pcm_16k)?;
        normalize_per_feature(&mut feats);
        Ok(feats)
    }

    /// Log-mel *before* per-feature normalization (for parity tests that
    /// compare spectral energy across bins — normalization destroys that).
    /// Frames are independent (same per-bin f64 accumulation in the same
    /// inner-k order as the old serial loops), so rayon over frames is
    /// bit-exact — and the `[frames][mels]` layout lands directly, with no
    /// `[mels][frames]` transpose roundtrip.
    fn unnormalized_log_mel(&self, pcm_16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        if pcm_16k.is_empty() {
            return Ok(Vec::new());
        }
        let emphasized = apply_preemphasis(pcm_16k, self.preemphasis);
        let power = self
            .stft_power(&emphasized)
            .context("burn frontend: stft failed")?;
        let n_frames = power.first().map(|_| power[0].len()).unwrap_or(0);
        Ok((0..n_frames)
            .into_par_iter()
            .map(|f| {
                let mut row = vec![0.0f32; self.n_mels];
                for (m, basis) in self.mel_basis.iter().enumerate() {
                    let mut acc = 0.0f64;
                    for (k, w) in basis.iter().enumerate() {
                        acc += *w as f64 * power[k][f] as f64;
                    }
                    row[m] = (acc as f32 + TDT_LOG_GUARD).ln();
                }
                row
            })
            .collect())
    }
}

fn apply_preemphasis(x: &[f32], coef: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    out.push(x[0]);
    for i in 1..x.len() {
        out.push(x[i] - coef * x[i - 1]);
    }
    out
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (n as f32 - 1.0)).cos())
        .collect()
}

/// Power spectrogram via the cached `realfft` plan: `[freq_bins][frames]`,
/// reflect-padded like `parakeet-rs` (`pad = n_fft/2` zeros each side).
/// Frames are fully independent (same window, same per-frame FFT), so
/// rayon over frames is bit-exact; results collect in order.
impl MelFrontend {
    /// Power spectrogram via the cached `realfft` plan (see above).
    fn stft_power(&self, x: &[f32]) -> Result<Vec<Vec<f32>>> {
        let pad = self.n_fft / 2;
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(x);
        padded.resize(padded.len() + pad, 0.0);
        if padded.len() < self.n_fft {
            return Ok(vec![vec![0.0; 1]; self.n_fft / 2 + 1]);
        }
        let n_frames = (padded.len() - self.n_fft) / self.hop + 1;
        let bins = self.n_fft / 2 + 1;
        let per_frame: Result<Vec<Vec<f32>>> = (0..n_frames)
            .into_par_iter()
            .map(|f| {
                let start = f * self.hop;
                let mut input = vec![0.0f32; self.n_fft];
                let avail = (padded.len() - start).min(self.win);
                for i in 0..avail {
                    input[i] = padded[start + i] * self.window[i];
                }
                let mut spectrum = self.fft.make_output_vec();
                let mut scratch = self.fft.make_scratch_vec();
                self.fft
                    .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                    .map_err(|e| anyhow::anyhow!("fft failed: {e}"))?;
                Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
            })
            .collect();
        // Transpose [frames][bins] -> [bins][frames] (pure reorder).
        let mut out = vec![vec![0.0f32; n_frames]; bins];
        for (f, spec) in per_frame?.iter().enumerate() {
            for (k, &v) in spec.iter().enumerate() {
                out[k][f] = v;
            }
        }
        Ok(out)
    }
}

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

fn slaney_mel_basis(n_fft: usize, n_mels: usize, sr: usize) -> Vec<Vec<f32>> {
    let bins = n_fft / 2 + 1;
    let fmax = sr as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    let points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|i| i as f64 * sr as f64 / n_fft as f64)
        .collect();
    let fdiff: Vec<f64> = points.windows(2).map(|w| w[1] - w[0]).collect();
    let mut fb = vec![vec![0.0f32; bins]; n_mels];
    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - points[i]) / fdiff[i];
            let upper = (points[i + 2] - freq) / fdiff[i + 1];
            fb[i][k] = 0.0f64.max(lower.min(upper)) as f32;
        }
        let enorm = 2.0 / (points[i + 2] - points[i]);
        for v in fb[i].iter_mut() {
            *v *= enorm as f32;
        }
    }
    fb
}

fn normalize_per_feature(feats: &mut [Vec<f32>]) {
    let n = feats.len();
    if n <= 1 {
        return;
    }
    let d = feats[0].len();
    for j in 0..d {
        let mean: f32 = feats.iter().map(|r| r[j]).sum::<f32>() / n as f32;
        let var: f32 = feats.iter().map(|r| (r[j] - mean).powi(2)).sum::<f32>() / (n as f32 - 1.0);
        let std = var.sqrt() + 1e-5;
        for r in feats.iter_mut() {
            r[j] = (r[j] - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_frame_count_matches_hop_arithmetic() {
        // 1 s @16k, n_fft 512, pad 256*2, hop 160:
        // frames = (16000 + 512 - 512)/160 + 1 = 101
        let fe = MelFrontend::tdt_v3();
        let feats = fe.extract(&vec![0.0; 16_000]).expect("extract");
        assert_eq!(feats.len(), 101, "frames={}", feats.len());
        assert_eq!(feats[0].len(), 128);
    }

    #[test]
    fn silence_is_finite_and_normalized() {
        let fe = MelFrontend::tdt_v3();
        // dithered silence (pipeline pads with +-1e-4, never exact zero)
        let mut seed: u32 = 0x9E3779B9;
        let pcm: Vec<f32> = (0..8000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f32 / u32::MAX as f32 - 0.5) * 2e-4
            })
            .collect();
        let feats = fe.extract(&pcm).expect("extract");
        assert!(!feats.is_empty());
        for row in &feats {
            for &v in row {
                assert!(v.is_finite(), "non-finite frontend output {v}");
            }
        }
        // Near-constant input has tiny per-bin variance, so the norm
        // divides by ~1e-5 and f32 cancellation leaves a small residual
        // mean (same formula as parakeet-rs) — bound it loosely.
        let n = feats.len() as f32;
        let mean0: f32 = feats.iter().map(|r| r[0]).sum::<f32>() / n;
        assert!(mean0.abs() < 0.05, "mean0={mean0}");
    }

    #[test]
    fn parallel_frontend_is_deterministic() {
        // Frames run on rayon's pool: same input must give bit-identical
        // features every time (collect preserves order, per-element math
        // is unchanged from the serial loops).
        let fe = MelFrontend::tdt_v3();
        let mut seed: u32 = 0xF207;
        let pcm: Vec<f32> = (0..16_000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f32 / u32::MAX as f32 - 0.5) * 0.6
            })
            .collect();
        let a = fe.extract(&pcm).expect("extract");
        let b = fe.extract(&pcm).expect("extract");
        assert_eq!(a, b);
    }

    #[test]
    fn sine_concentrates_mel_energy() {
        // 1 kHz sine must peak (pre-norm) in the low-mid mels: with 128
        // Slaney mels over 0-8 kHz, 1 kHz lands at mel ~15/45 -> bin ~42.
        // A pure tone lights only ~3 of 128 mels, so compare the peak bin
        // against the high-frequency floor, not band sums (guard log
        // dominates any wide sum). Pre-norm: normalization would erase
        // the cross-bin comparison.
        let fe = MelFrontend::tdt_v3();
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let feats = fe.unnormalized_log_mel(&pcm).expect("extract");
        let interior = &feats[2..feats.len() - 2];
        let n = interior.len() as f32;
        let mut mean = vec![0.0f32; 128];
        for row in interior {
            for (m, &v) in row.iter().enumerate() {
                mean[m] += v / n;
            }
        }
        let (peak, _) = mean
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!((30..=60).contains(&peak), "peak={peak}");
        let floor: f32 = mean[90..120].iter().sum::<f32>() / 30.0;
        assert!(mean[peak] - floor > 5.0, "peak={} floor={floor}", mean[peak]);
    }
}
