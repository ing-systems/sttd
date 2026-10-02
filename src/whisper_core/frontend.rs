// ---------------------------------------------------------------------------
// whisper_core frontend: Whisper log-mel features without ONNX.
// ---------------------------------------------------------------------------
//
// The recipe is OpenAI Whisper's `log_mel_spectrogram`, which the HF export's
// `preprocessor_config.json` describes (`feature_size` 128, `n_fft` 400, hop
// 160, `nb_max_frames` 3000). Unlike Parakeet's frontend this one is
// fully determined by the window size: the encoder graph's `input_features`
// axes are constants, so the output is always `[1, 128, 3000]` and the audio
// is padded (or truncated) to the window rather than the window growing with
// the utterance. See `crate::whisper_core::types::MelConfig`.
//
// Two details separate a correct implementation from plausible garbage, and
// both are in the reference:
//
// - the Hann window is *periodic* (`torch.hann_window`, denominator N), not
//   symmetric (N-1) as in `parakeet_core::frontend`;
// - the STFT is centred with reflect padding (`torch.stft(center=True)`), so
//   the 200 samples either side of the window mirror the signal. The window is
//   right-padded with zeros first, which makes the right-hand reflection
//   zeros, but the left-hand one still reflects real audio.
//
// The filterbank, FFT plan and scratch buffers are built once in `new` and
// reused by every `extract`, as in the Parakeet frontend.
// ---------------------------------------------------------------------------

use ndarray17::linalg::general_mat_mul;
use ndarray17::{Array2, ArrayView3, ArrayViewMut2, Axis};
use realfft::num_complex::Complex;
use realfft::RealToComplex;
use std::f32::consts::PI;
use std::sync::Arc;

use super::types::{MelConfig, WhisperError};
use crate::mel::create_mel_filterbank;

/// Floor applied before `log10`, matching `torch.clamp(mel_spec, min=1e-10)`.
const LOG_FLOOR: f32 = 1e-10;

/// How far below the loudest mel bin the log spectrum is clipped, and the
/// affine map applied afterwards: `(x + 4) / 4`. Both from
/// `log_mel_spectrogram`. The clamp is relative to the whole window's maximum,
/// so a quiet window scales itself up to the same range as a loud one.
const DYNAMIC_RANGE_DB: f32 = 8.0;
const LOG_OFFSET: f32 = 4.0;
const LOG_SCALE: f32 = 4.0;

/// Cached mel filterbank, FFT plan and scratch buffers, built once at model
/// load. `extract` takes `&mut self` because it reuses the buffers.
pub struct Frontend {
    pub config: MelConfig,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    /// Periodic Hann analysis window (win_length), precomputed.
    window: Vec<f32>,
    /// The current utterance, zero-padded to `n_samples` and reflect-padded
    /// by `n_fft / 2` on each side.
    padded: Vec<f32>,
    /// One windowed frame (n_fft samples).
    frame: Vec<f32>,
    /// Reusable FFT input/output/scratch buffers.
    fft_out: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    /// Power spectrogram scratch: `[n_fft / 2 + 1, frames]`, reallocated only
    /// when the window length changes.
    spectrogram: Array2<f32>,
    /// The encoder's `[1, n_mels, frames]` input, in a buffer sized for the
    /// full window: a shorter window is a prefix of it, so a change of window
    /// never allocates.
    features: Vec<f32>,
}

impl Frontend {
    pub fn new(config: MelConfig) -> Self {
        let mel_basis = create_mel_filterbank(config.n_fft, config.n_mels, config.sampling_rate);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(config.n_fft);
        let freq_bins = config.n_fft / 2 + 1;
        let window = periodic_hann(config.win_length);
        let frame = vec![0.0; config.n_fft];
        Self {
            config,
            mel_basis,
            fft_out: fft_plan.make_output_vec(),
            fft_scratch: fft_plan.make_scratch_vec(),
            fft_plan,
            window,
            padded: Vec::new(),
            frame,
            spectrogram: Array2::zeros((freq_bins, config.n_frames)),
            features: vec![0.0; config.n_mels * config.n_frames],
        }
    }

    /// Raw 16 kHz mono samples -> `[1, 128, n_frames]` log-mel features in the
    /// layout the encoder expects.
    ///
    /// `n_frames` is the encoder window (`types::window_frames`): 3000 for the
    /// fixed export, any value up to that for the patched one. The returned
    /// view borrows the frontend's own buffer, which is rewritten in full on
    /// every call, so nothing is allocated per utterance.
    ///
    /// Audio longer than the window is truncated to it and logged: an
    /// utterance is capped at 30 s upstream (`MAX_UTTERANCE_SAMPLES`), so this
    /// only bites when trailing dither padding pushes it over, and what is cut
    /// is pad rather than speech.
    pub fn extract(
        &mut self,
        audio: &[f32],
        n_frames: usize,
    ) -> Result<ArrayView3<'_, f32>, WhisperError> {
        if n_frames == 0 || n_frames > self.config.n_frames {
            return Err(WhisperError::Frontend(format!(
                "window of {n_frames} frames is outside 1..={}",
                self.config.n_frames
            )));
        }
        let n_samples = n_frames * self.config.hop_length;
        if audio.len() > n_samples {
            log::debug!(
                "whisper window is {n_samples} samples, truncating {} ({} samples dropped)",
                audio.len(),
                audio.len() - n_samples
            );
        }
        self.pad_and_reflect(audio, n_samples);
        self.stft(n_frames)?;
        self.log_mel(n_frames)?;
        let len = self.config.n_mels * n_frames;
        Ok(ArrayView3::from_shape(
            (1, self.config.n_mels, n_frames),
            &self.features[..len],
        )?)
    }

    /// Zero-pad (or truncate) to the window length, then reflect-pad by
    /// `n_fft / 2` on each side — `torch.stft(center=True)`'s default.
    ///
    /// Both reflections read the *window-padded* signal, not the caller's
    /// slice: for an utterance shorter than the window the mirrored samples
    /// come from the zero padding, and the right-hand reflection is zeros
    /// throughout. The reflect pad omits the edge sample, hence the `- 1`.
    fn pad_and_reflect(&mut self, audio: &[f32], n_samples: usize) {
        let n = n_samples;
        let half = self.config.n_fft / 2;
        let len = audio.len().min(n);
        // Layout: `half` reflected samples, then `n` window samples (the audio
        // right-padded with zeros), then `half` more. The trailing half is zeros
        // because the window is zero-padded before reflecting — matching the
        // reference, where the right reflection mirrors its own zero padding.
        self.padded.clear();
        self.padded.resize(n + 2 * half, 0.0);
        self.padded[half..half + len].copy_from_slice(&audio[..len]);
        // Left reflection, read from the padded window so a short utterance
        // mirrors its own zeros. The reflect pad omits the edge sample, hence
        // the `- 1`.
        for i in 0..half {
            let src = n - 1 - i;
            self.padded[i] = if src < len { audio[src] } else { 0.0 };
        }
    }

    /// Power spectrogram of the padded signal into `self.spectrogram`.
    ///
    /// The centre pad makes this `n_samples / hop` frames — 3000 for the 30 s
    /// window, which is exactly the encoder's `input_features` frame count. The
    /// scratch is reallocated only when the window length changes.
    fn stft(&mut self, n_frames: usize) -> Result<(), WhisperError> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.config.win_length;
        let freq_bins = n_fft / 2 + 1;
        let num_frames = (self.padded.len() - n_fft) / hop_length;
        debug_assert_eq!(num_frames, n_frames);
        if self.spectrogram.shape() != [freq_bins, num_frames] {
            self.spectrogram = Array2::zeros((freq_bins, num_frames));
        }
        // Whisper's analysis window fills the FFT buffer exactly (`win_length
        // == n_fft == 400`) and every frame has `n_fft` samples available, so
        // there is no unwritten tail to clear and the old
        // `win_length.min(..)` bound was always `win_length`. A shorter
        // analysis window would leave one, which is all this flag guards.
        let needs_tail_clear = win_length < n_fft;

        for (frame_idx, mut column) in self.spectrogram.axis_iter_mut(Axis(1)).enumerate() {
            let start = frame_idx * hop_length;
            if needs_tail_clear {
                self.frame.fill(0.0);
            }
            for (dst, (&sample, &taper)) in self.frame[..win_length]
                .iter_mut()
                .zip(self.padded[start..].iter().zip(&self.window))
            {
                *dst = sample * taper;
            }
            self.fft_plan
                .process_with_scratch(&mut self.frame, &mut self.fft_out, &mut self.fft_scratch)
                .map_err(|e| WhisperError::Frontend(format!("FFT failed: {e}")))?;
            for (bin, value) in column.iter_mut().zip(&self.fft_out) {
                *bin = value.norm_sqr();
            }
        }
        Ok(())
    }

    /// Mel projection, `log10` with the window-relative clamp, and the affine
    /// map into the encoder's input range, all into `self.features`.
    fn log_mel(&mut self, n_frames: usize) -> Result<(), WhisperError> {
        let len = self.config.n_mels * n_frames;
        let features = &mut self.features[..len];
        // C <- 1 * mel_basis * spectrogram + 0 * C: writes into the frontend's
        // own feature buffer instead of allocating per utterance.
        let mut mel = ArrayViewMut2::from_shape((self.config.n_mels, n_frames), &mut *features)?;
        general_mat_mul(
            1.0,
            &self.mel_basis.view(),
            &self.spectrogram.view(),
            0.0,
            &mut mel,
        );
        // `log_spec.max()` is over the whole window, so a silent utterance
        // scales its own noise floor up to full range rather than collapsing
        // to a constant — the model was trained on windows like this.
        let peak = features.iter().fold(LOG_FLOOR, |acc, &v| acc.max(v));
        let floor = (peak.max(LOG_FLOOR)).log10() - DYNAMIC_RANGE_DB;
        for value in features.iter_mut() {
            *value = ((*value).max(LOG_FLOOR).log10()).max(floor);
            *value = (*value + LOG_OFFSET) / LOG_SCALE;
        }
        Ok(())
    }
}

/// Periodic Hann window: `w[i] = 0.5 - 0.5 * cos(2*pi*i / N)`. The `N`
/// denominator is what `torch.hann_window` uses; the symmetric `N - 1` form
/// here would differ from Whisper's reference by ~1e-3 per bin, which is
/// enough to shift a decode.
fn periodic_hann(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / len as f32).cos())
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

    /// One `extract` on a fresh frontend, copied out, which is what the engine
    /// does not do (it reuses one frontend) but is what these checks want.
    fn features_in(audio: &[f32], n_frames: usize) -> Array3<f32> {
        let mut frontend = Frontend::new(MelConfig::large_v3_turbo());
        frontend.extract(audio, n_frames).unwrap().to_owned()
    }

    fn features(audio: &[f32]) -> Array3<f32> {
        features_in(audio, MelConfig::large_v3_turbo().n_frames)
    }

    /// Straightforward reimplementation of `log_mel_spectrogram` with no
    /// cached buffers and no reuse: pad to the window, reflect-pad, windowed
    /// rFFT, drop the last frame, mel, clamp, scale. The optimized `extract`
    /// must agree with it.
    fn reference(audio: &[f32], n_frames: usize) -> Array3<f32> {
        let c = MelConfig::large_v3_turbo();
        let n_samples = n_frames * c.hop_length;
        let half = c.n_fft / 2;
        let len = audio.len().min(n_samples);

        // Zero-pad to the window, then reflect-pad by n_fft/2 each side. The
        // reflections read the padded window, so a short utterance mirrors its
        // own zero padding.
        let mut signal = vec![0.0f32; half + n_samples + half];
        signal[half..half + len].copy_from_slice(&audio[..len]);
        let left = &mut signal[..half];
        for (dst, i) in left.iter_mut().zip((n_samples - half..n_samples).rev()) {
            *dst = if i < len { audio[i] } else { 0.0 };
        }
        assert_eq!(signal.len(), n_samples + 2 * half);

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(c.n_fft);
        let window = periodic_hann(c.win_length);
        // Centre pad gives n_samples/hop + 1 frames; drop the last.
        let num_frames = (signal.len() - c.n_fft) / c.hop_length;
        assert_eq!(num_frames, n_frames);
        let freq_bins = c.n_fft / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
        let mut input = vec![0.0f32; c.n_fft];
        for frame_idx in 0..num_frames {
            let start = frame_idx * c.hop_length;
            input.fill(0.0);
            for i in 0..c.win_length.min(signal.len() - start) {
                input[i] = signal[start + i] * window[i];
            }
            let mut output = plan.make_output_vec();
            let mut scratch = plan.make_scratch_vec();
            plan.process_with_scratch(&mut input, &mut output, &mut scratch)
                .unwrap();
            for k in 0..freq_bins {
                spectrogram[[k, frame_idx]] = output[k].norm_sqr();
            }
        }
        let mel_basis = create_mel_filterbank(c.n_fft, c.n_mels, c.sampling_rate);
        let mut mel = mel_basis.dot(&spectrogram);
        let peak = mel.iter().copied().fold(LOG_FLOOR, f32::max);
        let floor = peak.max(LOG_FLOOR).log10() - DYNAMIC_RANGE_DB;
        for v in mel.iter_mut() {
            *v = ((*v).max(LOG_FLOOR).log10()).max(floor);
            *v = (*v + LOG_OFFSET) / LOG_SCALE;
        }
        Array3::from_shape_vec(
            (1, c.n_mels, mel.shape()[1]),
            mel.into_raw_vec_and_offset().0,
        )
        .unwrap()
    }

    #[test]
    fn extract_matches_the_unbuffered_reference() {
        // Short enough to exercise zero padding, long enough for real content.
        for len in [8_000, 96_000, 480_000] {
            let audio = tone(len);
            let got = features(&audio);
            let want = reference(&audio, 3000);
            assert_eq!(got.shape(), want.shape(), "len={len}");
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < 1e-5,
                    "len={len} feature mismatch: {g} vs {w}"
                );
            }
        }
    }

    /// A shorter window is the same recipe over fewer samples, so it must agree
    /// with the reference at that length — in particular the left reflection
    /// reads the *end of the shorter window*, and the peak clamp is taken over
    /// the shorter window only.
    #[test]
    fn extract_matches_the_reference_at_shorter_windows() {
        for (len, n_frames) in [
            (8_000, 1000),
            (96_000, 1000),
            (96_000, 1500),
            (160_000, 1000),
        ] {
            let audio = tone(len);
            let got = features_in(&audio, n_frames);
            let want = reference(&audio, n_frames);
            assert_eq!(got.shape(), [1, 128, n_frames], "len={len}");
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < 1e-5,
                    "len={len} n_frames={n_frames} feature mismatch: {g} vs {w}"
                );
            }
        }
    }

    /// The encoder graph's `input_features` axes are constants in the fixed
    /// export, so the default window must be `[1, 128, 3000]` for *every* input
    /// length — this is the property that makes per-utterance cost independent
    /// of utterance length there.
    #[test]
    fn output_is_always_the_requested_window() {
        for len in [0, 1_600, 48_000, 480_000, 960_000] {
            let features = features(&tone(len));
            assert_eq!(features.shape(), &[1, 128, 3000], "len={len}");
            assert!(
                features.is_standard_layout(),
                "len={len}: ort rejects a non-contiguous tensor"
            );
            assert!(features.as_slice().is_some(), "len={len}");
        }
        for n_frames in [100, 1000, 1234, 2999] {
            let features = features_in(&tone(48_000), n_frames);
            assert_eq!(features.shape(), &[1, 128, n_frames]);
            assert!(features.is_standard_layout());
        }
    }

    #[test]
    fn extract_rejects_a_window_outside_the_export() {
        let mut frontend = Frontend::new(MelConfig::large_v3_turbo());
        assert!(frontend.extract(&tone(1_600), 0).is_err());
        assert!(frontend.extract(&tone(1_600), 3001).is_err());
        assert!(frontend.extract(&tone(1_600), 3000).is_ok());
    }

    /// An empty buffer must still yield a full window of finite features: a
    /// dropped utterance reaches the model (a VAD-opened buffer that turned
    /// out to hold nothing), and NaN there would poison the encoder.
    #[test]
    fn silence_yields_finite_features() {
        let features = features(&[]);
        assert_eq!(features.shape(), &[1, 128, 3000]);
        assert!(
            features.iter().all(|v| v.is_finite()),
            "silence produced a non-finite feature"
        );
        assert!(features_in(&[], 1000).iter().all(|v| v.is_finite()));
    }

    /// The log clamp is relative to the window maximum, so a quiet window is
    /// scaled up to the same range as a loud one instead of collapsing to the
    /// floor. A very quiet tone must therefore still reach near the top of the
    /// range, while pure digital silence sits at the floor.
    #[test]
    fn quiet_audio_is_scaled_relative_to_its_own_maximum() {
        let quiet: Vec<f32> = tone(96_000).iter().map(|s| s * 1e-4).collect();
        let quiet_feats = features(&quiet);
        let silence_feats = features(&[]);
        let peak = |a: &Array3<f32>| a.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            peak(&quiet_feats) > peak(&silence_feats),
            "quiet tone {} must exceed silence {}",
            peak(&quiet_feats),
            peak(&silence_feats)
        );
        // Both sit inside the affine range the encoder expects.
        for feats in [&quiet_feats, &silence_feats] {
            for v in feats.iter() {
                assert!(
                    (-1.5..=1.0).contains(v),
                    "feature {v} outside the expected range"
                );
            }
        }
    }

    /// `extract` rewrites the frontend's buffer in full and keeps every other
    /// one, so the same audio must produce the same features on the second pass
    /// — even with a different window in between. A buffer that kept stale
    /// values from the utterance in between would show up here.
    #[test]
    fn extract_refills_the_buffer_every_call_across_window_changes() {
        let mut frontend = Frontend::new(MelConfig::large_v3_turbo());
        let first = frontend.extract(&tone(96_000), 1000).unwrap().to_owned();
        frontend.extract(&tone(48_000), 3000).unwrap();
        frontend.extract(&tone(48_000), 1500).unwrap();
        let again = frontend.extract(&tone(96_000), 1000).unwrap().to_owned();
        assert_eq!(first.shape(), again.shape());
        for (a, b) in first.iter().zip(again.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "{a} vs {b}: buffer was not fully rewritten"
            );
        }
    }

    /// The window is periodic (denominator N), not symmetric (N - 1). Whisper's
    /// reference uses the periodic form, so a symmetric window here would
    /// silently mis-scale every bin.
    #[test]
    fn hann_window_is_periodic() {
        let w = periodic_hann(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6, "first sample must be 0, got {}", w[0]);
        // Periodic: w[N/2] is exactly 1 (symmetric would be slightly under).
        assert!((w[200] - 1.0).abs() < 1e-6, "midpoint {} != 1", w[200]);
        let symmetric: Vec<f32> = (0..400)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / 399.0).cos())
            .collect();
        assert!(
            (w[200] - symmetric[200]).abs() > 1e-6,
            "test would not distinguish periodic from symmetric"
        );
    }
}
