// ---------------------------------------------------------------------------
// ort_engine frontend: 16 kHz mel features without ONNX.
//
// Ported from `parakeet-rs` 0.3.7 `src/audio.rs` + `src/config.rs`
// (MIT OR Apache-2.0, Copyright altunenes): STFT via `realfft`, Slaney mel
// filterbank, `ln(x + 2^-24)`, per-feature mean/var normalization. Dropped:
// `load_audio` (sttd has `wav`), the `channels` argument (always mono here).
//
// This replaces Murmure's `nemo128.onnx` preprocessor session: same
// numerical recipe (NeMo `ParakeetFeatureExtractor`: 128 mels, n_fft 512,
// hop 160, win 400, preemphasis 0.97), one less ONNX file to ship.
// ---------------------------------------------------------------------------

use ndarray17::Array2;
use realfft::RealToComplex;
use std::f32::consts::PI;
use std::sync::Arc;

use super::types::ParakeetError;

/// NeMo `ParakeetFeatureExtractor` parameters for the TDT 0.6B model.
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub preemphasis: f32,
    pub sampling_rate: usize,
    pub win_length: usize,
}

impl PreprocessorConfig {
    pub fn tdt_0_6b() -> Self {
        Self {
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            preemphasis: 0.97,
            sampling_rate: 16_000,
            win_length: 400,
        }
    }
}

/// Cached mel filterbank + FFT plan, built once at model load.
pub struct Frontend {
    pub config: PreprocessorConfig,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
}

impl Frontend {
    pub fn new(config: PreprocessorConfig) -> Self {
        let mel_basis =
            create_mel_filterbank(config.n_fft, config.feature_size, config.sampling_rate);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(config.n_fft);
        Self {
            config,
            mel_basis,
            fft_plan,
        }
    }

    /// Raw 16 kHz mono samples -> `[time_steps, 128]` log-mel features.
    pub fn extract(&self, audio: &[f32]) -> Result<Array2<f32>, ParakeetError> {
        let audio = apply_preemphasis(audio, self.config.preemphasis);
        let spectrogram = stft_with_plan(
            &audio,
            &self.fft_plan,
            self.config.n_fft,
            self.config.hop_length,
            self.config.win_length,
        )?;
        let mel_spectrogram = self.mel_basis.dot(&spectrogram);
        // NeMo: log_zero_guard_type="add", value=2^-24.
        let log_zero_guard: f32 = 2.0f32.powi(-24);
        let mel_spectrogram = mel_spectrogram.mapv(|x| (x + log_zero_guard).ln());
        let mut mel_spectrogram = mel_spectrogram.t().to_owned();

        // Per-feature normalization (mean=0, std=1, Bessel's correction).
        let num_frames = mel_spectrogram.shape()[0];
        let num_features = mel_spectrogram.shape()[1];
        if num_frames <= 1 {
            return Ok(mel_spectrogram);
        }
        for feat_idx in 0..num_features {
            let mut column = mel_spectrogram.column_mut(feat_idx);
            let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
            let variance: f32 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames as f32 - 1.0);
            let std = variance.sqrt() + 1e-5;
            for val in column.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        Ok(mel_spectrogram)
    }
}

fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);
    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }
    result
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

fn stft_with_plan(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>, ParakeetError> {
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let window = hann_window(win_length);
    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut input = vec![0.0f32; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        input.fill(0.0);
        for i in 0..win_length.min(padded.len() - start) {
            input[i] = padded[start + i] * window[i];
        }
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| ParakeetError::Frontend(format!("FFT failed: {e}")))?;
        for k in 0..freq_bins {
            spectrogram[[k, frame_idx]] = output[k].norm_sqr();
        }
    }
    Ok(spectrogram)
}

// Slaney mel scale (librosa).
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

fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
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
