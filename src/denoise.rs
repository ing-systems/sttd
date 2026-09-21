use anyhow::{Context, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};
use nnnoiseless::DenoiseState;

use crate::config::{DENOISE_RATE, FRAME_48K};

// ---------------------------------------------------------------------------
// Denoiser: the one thing on the VAD branch that knows there's a choice of
// noise-suppression model. Everything downstream (resample -> VAD ->
// utterance buffer) only ever sees a clean 48 kHz frame; it doesn't know or
// care which of these produced it.
// ---------------------------------------------------------------------------

/// Which model cleans the VAD branch's 48 kHz frame before it's resampled
/// down to 16 kHz for `webrtc-vad`. Chosen once at startup via
/// `STTD_DENOISER` and fixed for the process's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenoiserKind {
    /// DeepFilterNet3 (tract/onnx). Default: markedly better than RNNoise on
    /// non-stationary noise, at the cost of a 20 ms lookahead and a heavier
    /// per-frame model pass. Runs at exactly `DENOISE_RATE`/`FRAME_48K`, so
    /// it drops into the pipeline's existing 48 kHz VAD-branch frame with no
    /// resampling changes.
    DeepFilter,
    /// RNNoise via `nnnoiseless`. Cheap, no lookahead, weaker on noise that
    /// isn't stationary hum/fan-type noise. Kept as a fallback and for A/B
    /// comparison against DeepFilter on the same recording (`--file`).
    RnNoise,
    /// No denoising: the VAD branch sees the raw 48 kHz frame. Useful to
    /// isolate how much a denoiser is actually helping (or hurting) VAD
    /// verdicts on a given recording.
    Bypass,
}

impl DenoiserKind {
    /// Parse an `STTD_DENOISER` value. Pure function so it's testable without
    /// touching the environment.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deepfilter" => Some(Self::DeepFilter),
            "rnnoise" => Some(Self::RnNoise),
            "none" => Some(Self::Bypass),
            _ => None,
        }
    }

    /// `STTD_DENOISER`, falling back to `DeepFilter` (with a warning) if
    /// unset or unrecognized — same override pattern as `f32_from_env`.
    pub(crate) fn from_env() -> Self {
        match std::env::var("STTD_DENOISER") {
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                eprintln!(
                    "[sttd] warning: STTD_DENOISER={v:?} is not one of \
                     deepfilter|rnnoise|none, using default deepfilter"
                );
                Self::DeepFilter
            }),
            Err(_) => Self::DeepFilter,
        }
    }
}

/// The active denoiser, built once by `FrameProcessor::new` from the
/// `DenoiserKind` chosen at startup.
pub(crate) enum Denoiser {
    DeepFilter(Box<DfTract>),
    RnNoise {
        state: Box<DenoiseState<'static>>,
        // RNNoise's frame API wants i16-scale samples; the rest of the
        // pipeline works in f32 [-1, 1], so this scaling is entirely this
        // variant's business now, not `FrameProcessor`'s.
        scaled_in: Vec<f32>,
        scaled_out: Vec<f32>,
    },
    Bypass,
}

impl Denoiser {
    pub(crate) fn new(kind: DenoiserKind) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let denoiser = match kind {
            DenoiserKind::DeepFilter => {
                let params = DfParams::default();
                let rp = RuntimeParams::default_with_ch(1);
                let tract = DfTract::new(params, &rp)
                    .context("failed to initialize DeepFilterNet (tract)")?;
                // Guard against a silent frame desync: everything downstream
                // assumes the VAD branch's frame is exactly DENOISE_RATE /
                // FRAME_48K samples.
                if tract.sr != DENOISE_RATE as usize || tract.hop_size != FRAME_48K {
                    anyhow::bail!(
                        "DeepFilterNet model rate/frame mismatch: sr={} hop_size={} \
                         (pipeline expects sr={DENOISE_RATE} hop_size={FRAME_48K})",
                        tract.sr,
                        tract.hop_size
                    );
                }
                Denoiser::DeepFilter(Box::new(tract))
            }
            DenoiserKind::RnNoise => Denoiser::RnNoise {
                state: DenoiseState::new(),
                scaled_in: vec![0.0f32; FRAME_48K],
                scaled_out: vec![0.0f32; FRAME_48K],
            },
            DenoiserKind::Bypass => Denoiser::Bypass,
        };
        eprintln!(
            "[sttd] denoiser: {} (init {:.1}ms)",
            denoiser.name(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(denoiser)
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Denoiser::DeepFilter(_) => "deepfilter",
            Denoiser::RnNoise { .. } => "rnnoise",
            Denoiser::Bypass => "none",
        }
    }

    /// Denoise one `FRAME_48K`-sample frame (f32, [-1, 1]) into `clean`.
    /// `noisy` and `clean` must both be exactly `FRAME_48K` samples.
    pub(crate) fn denoise_frame(&mut self, noisy: &[f32], clean: &mut [f32]) {
        debug_assert_eq!(noisy.len(), FRAME_48K);
        debug_assert_eq!(clean.len(), FRAME_48K);
        match self {
            Denoiser::DeepFilter(tract) => {
                let noisy_view = ArrayView2::from_shape((1, FRAME_48K), noisy)
                    .expect("noisy frame is exactly FRAME_48K samples");
                let mut enh_view = ArrayViewMut2::from_shape((1, FRAME_48K), clean)
                    .expect("clean frame is exactly FRAME_48K samples");
                if let Err(e) = tract.process(noisy_view, enh_view.view_mut()) {
                    // Hand the VAD branch the raw frame rather than lose
                    // frame-count sync with the rest of the pipeline.
                    eprintln!("[sttd] deepfilter process error: {e}; passing frame through");
                    clean.copy_from_slice(noisy);
                }
            }
            Denoiser::RnNoise {
                state,
                scaled_in,
                scaled_out,
            } => {
                for (i, s) in noisy.iter().enumerate() {
                    scaled_in[i] = (s * 32768.0).clamp(-32768.0, 32767.0);
                }
                state.process_frame(scaled_out, scaled_in);
                for (i, s) in scaled_out.iter().enumerate() {
                    clean[i] = s / 32768.0;
                }
            }
            Denoiser::Bypass => clean.copy_from_slice(noisy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoiser_kind_parses_known_values() {
        assert_eq!(DenoiserKind::parse("deepfilter"), Some(DenoiserKind::DeepFilter));
        assert_eq!(DenoiserKind::parse("DeepFilter"), Some(DenoiserKind::DeepFilter));
        assert_eq!(DenoiserKind::parse("rnnoise"), Some(DenoiserKind::RnNoise));
        assert_eq!(DenoiserKind::parse("none"), Some(DenoiserKind::Bypass));
        assert_eq!(DenoiserKind::parse("bzdura"), None);
        assert_eq!(DenoiserKind::parse(""), None);
    }

    #[test]
    fn bypass_passes_frame_through() {
        let mut denoiser = Denoiser::new(DenoiserKind::Bypass).unwrap();
        let noisy: Vec<f32> = (0..FRAME_48K).map(|i| (i as f32) * 0.001).collect();
        let mut clean = vec![0.0f32; FRAME_48K];
        denoiser.denoise_frame(&noisy, &mut clean);
        assert_eq!(noisy, clean);
    }

    #[test]
    fn rnnoise_preserves_frame_length_on_silence() {
        let mut denoiser = Denoiser::new(DenoiserKind::RnNoise).unwrap();
        let noisy = vec![0.0f32; FRAME_48K];
        let mut clean = vec![0.0f32; FRAME_48K];
        denoiser.denoise_frame(&noisy, &mut clean);
        assert_eq!(clean.len(), FRAME_48K);
    }

    #[test]
    fn deepfilter_frame_matches_pipeline_frame() {
        // Confirms the bundled DFN3 model's sr/hop_size line up with the
        // pipeline's own DENOISE_RATE/FRAME_48K constants -- the same
        // assumption `Denoiser::new` enforces at runtime.
        let params = DfParams::default();
        let rp = RuntimeParams::default_with_ch(1);
        let tract = DfTract::new(params, &rp).unwrap();
        assert_eq!(tract.sr, DENOISE_RATE as usize);
        assert_eq!(tract.hop_size, FRAME_48K);
    }

    #[test]
    fn deepfilter_preserves_frame_length_on_silence() {
        let mut denoiser = Denoiser::new(DenoiserKind::DeepFilter).unwrap();
        let noisy = vec![0.0f32; FRAME_48K];
        let mut clean = vec![0.0f32; FRAME_48K];
        denoiser.denoise_frame(&noisy, &mut clean);
        assert_eq!(clean.len(), FRAME_48K);
    }
}
