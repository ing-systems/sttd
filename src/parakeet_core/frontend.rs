// ---------------------------------------------------------------------------
// parakeet_core frontend: 16 kHz mel features without ONNX.
//
// Ported from `parakeet-rs` 0.3.7 `src/audio.rs` + `src/config.rs`
// (MIT OR Apache-2.0, Copyright altunenes): STFT via `realfft`, Slaney mel
// filterbank, `ln(x + 2^-24)`, per-feature mean/var normalization. Dropped:
// `load_audio` (sttd has `wav`), the `channels` argument (always mono here).
//
// This replaces Murmure's `nemo128.onnx` preprocessor session: same
// numerical recipe (NeMo `ParakeetFeatureExtractor`: 128 mels, n_fft 512,
// hop 160, win 400, preemphasis 0.97), one less ONNX file to ship. The mel
// filterbank itself now lives in `crate::mel`, shared with the Whisper
// frontend.
//
// The window, the padded signal, the FFT input/output/scratch and the
// spectrogram scratch buffer are built once in `new` and reused by every
// `extract` call, so a transcription allocates only the returned features.
// ---------------------------------------------------------------------------

use ndarray17::linalg::general_mat_mul;
use ndarray17::{Array2, Array3, Axis};
use realfft::RealToComplex;
use realfft::num_complex::Complex;
use std::f32::consts::PI;
use std::sync::Arc;

use super::types::ParakeetError;
use crate::mel::create_mel_filterbank;

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

/// Cached mel filterbank, FFT plan and scratch buffers, built once at model
/// load. `extract` takes `&mut self` because it reuses the buffers.
pub struct Frontend {
    pub config: PreprocessorConfig,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    /// Hann analysis window (win_length), precomputed.
    window: Vec<f32>,
    /// Zero-padded, preemphasized signal of the current utterance.
    padded: Vec<f32>,
    /// One windowed frame (n_fft samples).
    frame: Vec<f32>,
    /// Reusable FFT input/output/scratch buffers.
    fft_out: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    /// Power spectrogram scratch: [n_fft / 2 + 1, time_steps].
    spectrogram: Array2<f32>,
    /// Feature scratch: [1, feature_size, time_steps] in encoder layout.
    features: Array3<f32>,
}

impl Frontend {
    pub fn new(config: PreprocessorConfig) -> Self {
        let mel_basis =
            create_mel_filterbank(config.n_fft, config.feature_size, config.sampling_rate);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(config.n_fft);
        let freq_bins = config.n_fft / 2 + 1;
        let window = hann_window(config.win_length);
        let frame = vec![0.0; config.n_fft];
        let features = Array3::zeros((1, config.feature_size, 0));
        Self {
            config,
            mel_basis,
            fft_out: fft_plan.make_output_vec(),
            fft_scratch: fft_plan.make_scratch_vec(),
            fft_plan,
            window,
            padded: Vec::new(),
            frame,
            spectrogram: Array2::zeros((freq_bins, 0)),
            features,
        }
    }

    /// Raw 16 kHz mono samples -> `[1, 128, time_steps]` log-mel features in
    /// the layout the encoder expects. The returned array owns its data; the
    /// cached scratch buffers stay valid for the next call.
    pub fn extract(&mut self, audio: &[f32]) -> Result<Array3<f32>, ParakeetError> {
        self.pad_and_preemphasize(audio);
        let time_steps = self.stft()?;
        self.mel_features(time_steps);
        Ok(std::mem::take(&mut self.features))
    }

    /// Center the signal with `n_fft / 2` zeros on each side and apply the
    /// NeMo preemphasis, writing straight into the reusable padded buffer.
    fn pad_and_preemphasize(&mut self, audio: &[f32]) {
        let pad_amount = self.config.n_fft / 2;
        let coef = self.config.preemphasis;
        self.padded.clear();
        self.padded.resize(pad_amount, 0.0);
        if let Some(&first) = audio.first() {
            self.padded.push(first);
            for i in 1..audio.len() {
                self.padded.push(audio[i] - coef * audio[i - 1]);
            }
        }
        self.padded.resize(self.padded.len() + pad_amount, 0.0);
    }

    /// Power spectrogram of the padded signal into `self.spectrogram`.
    /// Returns the frame count.
    fn stft(&mut self) -> Result<usize, ParakeetError> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.config.win_length;
        let num_frames = (self.padded.len() - n_fft) / hop_length + 1;
        let freq_bins = n_fft / 2 + 1;
        if self.spectrogram.shape() != [freq_bins, num_frames] {
            self.spectrogram = Array2::zeros((freq_bins, num_frames));
        }

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            self.frame.fill(0.0);
            for i in 0..win_length.min(self.padded.len() - start) {
                self.frame[i] = self.padded[start + i] * self.window[i];
            }
            self.fft_plan
                .process_with_scratch(&mut self.frame, &mut self.fft_out, &mut self.fft_scratch)
                .map_err(|e| ParakeetError::Frontend(format!("FFT failed: {e}")))?;
            for k in 0..freq_bins {
                self.spectrogram[[k, frame_idx]] = self.fft_out[k].norm_sqr();
            }
        }
        Ok(num_frames)
    }

    /// Mel projection, `ln(x + 2^-24)` and per-feature normalization (mean=0,
    /// std=1, Bessel's correction) into `self.features`.
    fn mel_features(&mut self, time_steps: usize) {
        let num_features = self.config.feature_size;
        if self.features.shape() != [1, num_features, time_steps] {
            self.features = Array3::zeros((1, num_features, time_steps));
        }
        // NeMo: log_zero_guard_type="add", value=2^-24.
        let log_zero_guard: f32 = 2.0f32.powi(-24);
        // C <- 1 * mel_basis * spectrogram + 0 * C: the same gemm call the
        // former `mel_basis.dot(&spectrogram)` made, writing into the cached
        // feature buffer instead of a fresh array.
        let mut features = self.features.view_mut();
        let mut mel = features.index_axis_mut(Axis(0), 0);
        general_mat_mul(
            1.0,
            &self.mel_basis.view(),
            &self.spectrogram.view(),
            0.0,
            &mut mel,
        );
        for value in self.features.iter_mut() {
            *value = (*value + log_zero_guard).ln();
        }
        if time_steps <= 1 {
            return;
        }
        let num_frames = time_steps as f32;
        for feat_idx in 0..num_features {
            let mut feature = self.features.index_axis_mut(Axis(1), feat_idx);
            let mean: f32 = feature.iter().sum::<f32>() / num_frames;
            let variance: f32 = feature
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / (num_frames - 1.0);
            let std = variance.sqrt() + 1e-5;
            for value in feature.iter_mut() {
                *value = (*value - mean) / std;
            }
        }
    }
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    // Straightforward reimplementation of the recipe, with no cached buffers
    // and no reuse. The optimized `extract` must agree with it.
    fn reference(audio: &[f32]) -> Array3<f32> {
        let config = PreprocessorConfig::tdt_0_6b();
        let mel_basis =
            create_mel_filterbank(config.n_fft, config.feature_size, config.sampling_rate);
        let mut preemphasized: Vec<f32> = Vec::new();
        if let Some(&first) = audio.first() {
            preemphasized.push(first);
            for i in 1..audio.len() {
                preemphasized.push(audio[i] - config.preemphasis * audio[i - 1]);
            }
        }
        let pad = config.n_fft / 2;
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(&preemphasized);
        padded.resize(padded.len() + pad, 0.0);

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(config.n_fft);
        let window = hann_window(config.win_length);
        let num_frames = (padded.len() - config.n_fft) / config.hop_length + 1;
        let freq_bins = config.n_fft / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
        let mut input = vec![0.0f32; config.n_fft];
        for frame_idx in 0..num_frames {
            let start = frame_idx * config.hop_length;
            input.fill(0.0);
            for i in 0..config.win_length.min(padded.len() - start) {
                input[i] = padded[start + i] * window[i];
            }
            let mut output = plan.make_output_vec();
            let mut scratch = plan.make_scratch_vec();
            plan.process_with_scratch(&mut input, &mut output, &mut scratch)
                .unwrap();
            for k in 0..freq_bins {
                spectrogram[[k, frame_idx]] = output[k].norm_sqr();
            }
        }
        let mut log_mel = mel_basis.dot(&spectrogram);
        let guard: f32 = 2.0f32.powi(-24);
        log_mel = log_mel.mapv(|x| (x + guard).ln());
        let mut out = log_mel.t().to_owned();
        let frames = out.shape()[0];
        let features = out.shape()[1];
        if frames <= 1 {
            // Not enough frames for a per-feature mean/variance.
            let time = out.nrows();
            return Array3::from_shape_vec((1, features, time), out.into_raw_vec()).unwrap();
        }
        for feat_idx in 0..features {
            let mut column = out.column_mut(feat_idx);
            let mean: f32 = column.iter().sum::<f32>() / frames as f32;
            let variance: f32 = column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / (frames as f32 - 1.0);
            let std = variance.sqrt() + 1e-5;
            for value in column.iter_mut() {
                *value = (*value - mean) / std;
            }
        }
        let time = out.nrows();
        Array3::from_shape_vec((1, features, time), out.into_raw_vec()).unwrap()
    }

    #[test]
    fn extract_matches_the_unbuffered_reference() {
        let audio = tone(4000);
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        let got = frontend.extract(&audio).unwrap();
        let want = reference(&audio);
        assert_eq!(got.shape(), want.shape());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 1e-6,
                "feature mismatch: {g} vs {w}"
            );
        }
    }

    #[test]
    fn extract_reuses_buffers_across_calls() {
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        let first = frontend.extract(&tone(3200)).unwrap();
        let second = frontend.extract(&tone(4800)).unwrap();
        // A longer utterance needs more frames than the first one.
        assert!(second.shape()[2] > first.shape()[2]);
        // The second result must not alias the first: the frontend hands its
        // buffer out and allocates a fresh one.
        let third = frontend.extract(&tone(3200)).unwrap();
        assert_eq!(first.shape(), third.shape());
        for (a, c) in first.iter().zip(third.iter()) {
            assert!((a - c).abs() < 1e-6);
        }
    }

    // `ort` builds the encoder input tensor from this array and rejects a
    // non-contiguous layout, so the result must always be row-major.
    #[test]
    fn extract_output_is_standard_layout() {
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        for len in [1600, 4800, 8000] {
            let features = frontend.extract(&tone(len)).unwrap();
            let time = features.shape()[2];
            assert_eq!(features.shape(), &[1, 128, time]);
            // Row-major strides of [1, feature_size, time_steps].
            assert_eq!(
                features.strides(),
                &[(128 * time) as isize, time as isize, 1],
                "features must be row-major contiguous"
            );
            assert!(features.is_standard_layout());
            assert!(features.as_slice().is_some());
        }
    }

    #[test]
    fn extract_of_a_single_frame_utterance_is_not_normalized() {
        // Fewer than two frames: the NeMo recipe leaves the features as they
        // are, so the per-feature mean/var step is skipped.
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        let got = frontend.extract(&tone(64)).unwrap();
        let want = reference(&tone(64));
        assert_eq!(got.shape()[2], 1);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6);
        }
    }
}


