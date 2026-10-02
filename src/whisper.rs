// ---------------------------------------------------------------------------
// Whisper engine: Whisper large-v3-turbo `SpeechModel` impl.
// ---------------------------------------------------------------------------
//
// The second engine behind the `asr::SpeechModel` seam, alongside
// `parakeet::ParakeetModel`. Everything engine-agnostic (worker loop, gates,
// metrics, conditioning, dictionary correction) stays in `asr`; this module
// owns only what Whisper needs: the mel frontend and the greedy decode over
// direct `ort` sessions (see `whisper_core`).
//
// Two things about this engine shape the pipeline around it:
//
// - the encoder takes a fixed 30 s window, so an utterance's length does not
//   change the inference cost (~1.2 s per utterance on a 16-core 9950X, against
//   Parakeet's tens of milliseconds). `decoded_sample_count` therefore reports
//   the window rather than the utterance, which keeps the RTF column and the
//   `audio_sec` metrics honest;
// - Whisper needs an explicit language token, unlike Parakeet's per-utterance
//   auto-detection. `--language` chooses it (see `Language`).
//
// No phrase boosting: Whisper has no logit-level booster here, so
// `set_boost_words` stays the trait default. The post-gate dictionary
// correction in `dictionary` still applies, and the per-word confidences this
// engine does report gate it as they do for Parakeet.
//
// Runs on CPU. `--gpu` warns and falls back, as with Parakeet.
// ---------------------------------------------------------------------------

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
use crate::config::{MAX_UTTERANCE_SAMPLES, WHISPER_MAX_NEW_TOKENS};
use crate::whisper_core::WhisperModel as CoreModel;

pub(crate) struct WhisperModel {
    inner: CoreModel,
    ready_line: String,
}

impl WhisperModel {
    /// Load a Whisper model from `dir` (already guaranteed to hold the three
    /// graphs and `tokenizer.json` by `model_store::ensure_available`).
    pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
        if opts.use_gpu {
            log::warn!("--gpu is set but Whisper runs on CPU in this build; running on CPU");
        }
        let intra_threads = opts.n_threads.max(1) as usize;
        let language = opts.language;
        log::info!("loading whisper model ({})...", dir.display());
        let inner = with_loading_ticker("whisper model", || {
            CoreModel::new(dir, language, intra_threads)
                .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))
                .context("failed to load whisper model")
        })?;
        let ready_line = format!(
            "whisper ready (model={}, lang={}, threads={}, gpu=false)",
            opts.model_spec, language, opts.n_threads
        );
        log::info!("{}", ready_line);
        Ok(Self { inner, ready_line })
    }
}

impl SpeechModel for WhisperModel {
    fn transcribe(
        &mut self,
        pcm_16k: &[f32],
        running: &Arc<AtomicBool>,
    ) -> Result<Transcript> {
        // Same reasoning as the Parakeet path: `ort` has no abort callback and
        // one inference is seconds, not minutes, so a pre-call check is enough.
        if !running.load(Ordering::Relaxed) {
            anyhow::bail!("shutdown requested");
        }
        let result = self
            .inner
            .transcribe_samples(pcm_16k, WHISPER_MAX_NEW_TOKENS)
            .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;
        if result.truncated {
            // Already warned by the engine; repeat at `warn` so it lands next to
            // the utterance's own timing line rather than deep in the engine.
            log::warn!("whisper returned a truncated transcript");
        }
        Ok(Transcript {
            text: result.text,
            // Per-word confidences gate the dictionary correction, same as
            // Parakeet's.
            word_confidences: result.word_confidences,
        })
    }

    fn describe(&self) -> String {
        self.ready_line.clone()
    }

    fn describe_reload(&self) -> String {
        format!("{} (reloaded after pause)", self.ready_line)
    }

    fn engine_name(&self) -> &'static str {
        "whisper"
    }

    /// Whisper always decodes a fixed 30 s window (the encoder's
    /// `input_features` shape is constant in the export), so this reports the
    /// window for every utterance. A 2 s utterance really did cost a full
    /// window's encoder pass, and RTF should say so.
    fn decoded_sample_count(&self, _conditioned_len: usize) -> usize {
        window_samples()
    }
}

/// Samples the encoder actually decodes for any utterance: the export's fixed
/// 30 s window.
///
/// A free function rather than an inline constant so the tests can assert the
/// relationship to `MAX_UTTERANCE_SAMPLES` without a loaded session — the
/// engine is too heavy to build in a unit test.
fn window_samples() -> usize {
    MAX_UTTERANCE_SAMPLES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper_core::types::{Language, TOK_LANG_EN, TOK_LANG_PL};

    #[test]
    fn the_decoded_window_is_thirty_seconds() {
        // The encoder graph pins the mel frame count, so the window is fixed:
        // RTF must not be flattered by a short utterance reporting its own
        // length instead.
        assert_eq!(window_samples(), MAX_UTTERANCE_SAMPLES);
        assert_eq!(window_samples(), 30 * 16_000, "30 s at 16 kHz");
        // It must equal the frontend's window, or `audio_sec` would describe
        // audio the model never saw.
        assert_eq!(
            window_samples(),
            crate::whisper_core::types::MelConfig::large_v3_turbo().n_samples
        );
    }

    /// The compiled-in prompt tokens must be the exported ones: Whisper has no
    /// per-engine vocabulary lookup for these (see
    /// `whisper_core::types`), so a wrong id decodes garbage.
    #[test]
    fn language_prompt_tokens_are_the_exported_ones() {
        assert_eq!(Language::Polish.prompt_token(), TOK_LANG_PL);
        assert_eq!(Language::English.prompt_token(), TOK_LANG_EN);
        assert_eq!(Language::Auto.prompt_token(), TOK_LANG_PL, "auto falls back to pl");
        assert_eq!(Language::Polish.as_str(), "pl");
        assert_eq!(Language::English.as_str(), "en");
        assert_eq!(Language::Auto.as_str(), "auto");
        assert_eq!(Language::English.to_string(), "en");
        // The default is what `--language` resolves to when unset, and sttd is
        // a Polish speech-to-text tool, so it must be Polish.
        assert_eq!(Language::default(), Language::Polish);
    }
}
