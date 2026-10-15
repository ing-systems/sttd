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
// The padded signal and the STFT (`crate::mel::Stft`: window, FFT plan and
// buffers, spectrogram) are built once in `new` and reused by every `extract`
// call, and the features are written into a grow-only buffer the caller gets
// a view of, so a transcription allocates nothing once the frontend has seen
// an utterance that long.
// ---------------------------------------------------------------------------

use ndarray17::linalg::general_mat_mul;
use ndarray17::{Array2, ArrayView3, ArrayViewMut2, Axis};
use std::f32::consts::PI;

use super::types::ParakeetError;
use crate::mel::{create_mel_filterbank, Stft};

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
    pub const fn tdt_0_6b() -> Self {
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
    /// Symmetric Hann analysis window (win_length) inside an n_fft STFT.
    stft: Stft,
    /// Zero-padded, preemphasized signal of the current utterance.
    padded: Vec<f32>,
    /// Feature buffer: [1, feature_size, time_steps] in encoder layout, grown
    /// on demand and never shrunk; a shorter utterance is a prefix of it.
    features: Vec<f32>,
}

impl Frontend {
    pub fn new(config: PreprocessorConfig) -> Self {
        let mel_basis =
            create_mel_filterbank(config.n_fft, config.feature_size, config.sampling_rate);
        let stft = Stft::new(config.n_fft, hann_window(config.win_length));
        Self {
            config,
            mel_basis,
            stft,
            padded: Vec::new(),
            features: Vec::new(),
        }
    }

    /// Raw 16 kHz mono samples -> `[1, 128, time_steps]` log-mel features in
    /// the layout the encoder expects. The returned view borrows the
    /// frontend's own buffer, which the next call rewrites.
    pub fn extract(&mut self, audio: &[f32]) -> Result<ArrayView3<'_, f32>, ParakeetError> {
        self.pad_and_preemphasize(audio);
        let time_steps = self.mel_features()?;
        let len = self.config.feature_size * time_steps;
        Ok(ArrayView3::from_shape(
            (1, self.config.feature_size, time_steps),
            &self.features[..len],
        )?)
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

    /// Power spectrogram of the padded signal, then mel projection,
    /// `ln(x + 2^-24)` and per-feature normalization (mean=0, std=1, Bessel's
    /// correction) into `self.features`. Returns the frame count.
    fn mel_features(&mut self) -> Result<usize, ParakeetError> {
        // Every frame has a full `n_fft` of padded signal, so the analysis
        // window (shorter than `n_fft`) always fits.
        let time_steps = (self.padded.len() - self.config.n_fft) / self.config.hop_length + 1;
        let spectrogram = self
            .stft
            .power(&self.padded, self.config.hop_length, time_steps)
            .map_err(|e| ParakeetError::Frontend(format!("FFT failed: {e}")))?;
        let num_features = self.config.feature_size;
        let len = num_features * time_steps;
        if self.features.len() < len {
            self.features.resize(len, 0.0);
        }
        // NeMo: log_zero_guard_type="add", value=2^-24.
        let log_zero_guard: f32 = 2.0f32.powi(-24);
        // C <- 1 * mel_basis * spectrogram + 0 * C: the same gemm call the
        // former `mel_basis.dot(&spectrogram)` made, writing into the cached
        // feature buffer instead of a fresh array.
        let mut mel =
            ArrayViewMut2::from_shape((num_features, time_steps), &mut self.features[..len])?;
        general_mat_mul(1.0, &self.mel_basis.view(), &spectrogram, 0.0, &mut mel);
        for value in mel.iter_mut() {
            *value = (*value + log_zero_guard).ln();
        }
        if time_steps <= 1 {
            return Ok(time_steps);
        }
        let num_frames = time_steps as f32;
        for mut feature in mel.axis_iter_mut(Axis(0)) {
            let mean: f32 = feature.iter().sum::<f32>() / num_frames;
            let variance: f32 =
                feature.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames - 1.0);
            let std = variance.sqrt() + 1e-5;
            for value in feature.iter_mut() {
                *value = (*value - mean) / std;
            }
        }
        Ok(time_steps)
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
    use ndarray17::Array3;

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
            return Array3::from_shape_vec((1, features, time), out.into_raw_vec_and_offset().0)
                .unwrap();
        }
        for feat_idx in 0..features {
            let mut column = out.column_mut(feat_idx);
            let mean: f32 = column.iter().sum::<f32>() / frames as f32;
            let variance: f32 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (frames as f32 - 1.0);
            let std = variance.sqrt() + 1e-5;
            for value in column.iter_mut() {
                *value = (*value - mean) / std;
            }
        }
        let time = out.nrows();
        Array3::from_shape_vec((1, features, time), out.into_raw_vec_and_offset().0).unwrap()
    }

    #[test]
    fn extract_matches_the_unbuffered_reference() {
        let audio = tone(4000);
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        let got = frontend.extract(&audio).unwrap();
        let want = reference(&audio);
        assert_eq!(got.shape(), want.shape());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "feature mismatch: {g} vs {w}");
        }
    }

    #[test]
    fn extract_reuses_buffers_across_calls() {
        let mut frontend = Frontend::new(PreprocessorConfig::tdt_0_6b());
        let first = frontend.extract(&tone(3200)).unwrap().to_owned();
        let second = frontend.extract(&tone(4800)).unwrap().to_owned();
        // A longer utterance needs more frames than the first one.
        assert!(second.shape()[2] > first.shape()[2]);
        // Back to the shorter utterance: a prefix of the grown buffer, fully
        // rewritten, so nothing of the longer call leaks into it.
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
