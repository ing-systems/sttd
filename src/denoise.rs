use anyhow::{Context, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};

use crate::config::{DENOISE_RATE, FRAME_48K};

// ---------------------------------------------------------------------------
// Denoiser: the one thing on the VAD branch that knows there's a choice of
// noise-suppression model. Everything downstream (resample -> VAD ->
// utterance buffer) only ever sees a clean 16 kHz frame; it doesn't know or
// care whether a model produced it.
// ---------------------------------------------------------------------------

/// Which model cleans the VAD branch's 48 kHz frame before it's resampled
/// down to 16 kHz for `earshot`. Chosen once at startup via
/// `STTD_DENOISER` and fixed for the process's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenoiserKind {
    /// DeepFilterNet3 (tract/onnx). Opt-in via `STTD_DENOISER=deepfilter`:
    /// markedly better than raw audio on non-stationary noise, at the cost
    /// of a 20 ms lookahead and a heavier per-frame model pass. Runs at
    /// exactly `DENOISE_RATE`/`FRAME_48K`, so it drops into the pipeline's
    /// 48 kHz VAD-branch frame with no resampling changes.
    DeepFilter,
    /// No denoising (default): the VAD branch resamples straight from the
    /// hw rate to 16 kHz with no 48 kHz intermediate at all (see
    /// `FrameProcessor`). Useful to isolate how much a denoiser is actually
    /// helping (or hurting) VAD verdicts on a given recording.
    Bypass,
}

impl DenoiserKind {
    /// Parse an `STTD_DENOISER` value. Pure function so it's testable without
    /// touching the environment.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deepfilter" => Some(Self::DeepFilter),
            "none" => Some(Self::Bypass),
            _ => None,
        }
    }

    /// `STTD_DENOISER`, falling back to `Bypass` (with a warning) if
    /// unrecognized — same override pattern as `f32_from_env`. Unset means
    /// `Bypass` with no warning: denoising is opt-in.
    pub(crate) fn from_env() -> Self {
        match std::env::var("STTD_DENOISER") {
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                eprintln!(
                    "[sttd] warning: STTD_DENOISER={v:?} is not one of \
                     deepfilter|none, using default none"
                );
                Self::Bypass
            }),
            Err(_) => Self::Bypass,
        }
    }
}

/// The active denoiser, built once by `FrameProcessor::new` when the
/// `DenoiserKind` chosen at startup is `DeepFilter`. There is no bypass
/// variant: `Bypass` means no denoiser object and no 48 kHz path at all.
pub(crate) enum Denoiser {
    DeepFilter(Box<DfTract>),
}

impl Denoiser {
    pub(crate) fn new(kind: DenoiserKind) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let denoiser = match kind {
            DenoiserKind::Bypass => {
                anyhow::bail!("Bypass has no denoiser object (no 48 kHz path)");
            }
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
        assert_eq!(DenoiserKind::parse("none"), Some(DenoiserKind::Bypass));
        assert_eq!(DenoiserKind::parse("rnnoise"), None);
        assert_eq!(DenoiserKind::parse("bzdura"), None);
        assert_eq!(DenoiserKind::parse(""), None);
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
