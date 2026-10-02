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
//   Parakeet's tens of milliseconds). The encoder is patched in
//   memory so the window follows the audio (15 s at least by default, set with
//   `--audio-ctx`; see `whisper_core::types::window_frames`), which cuts a short
//   utterance to ~0.3 s. Either way `decoded_sample_count` reports the window
//   rather than the utterance, which keeps the RTF column and the `audio_sec`
//   metrics honest;
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
use crate::config::WHISPER_MAX_NEW_TOKENS;
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
            // Streaming always asks for the length-free encoder: the export is
            // patched in memory (see `dyn_encoder`) and the window follows the
            // audio. If the patch does not apply the fixed export is loaded
            // instead, with a warning, and `dynamic_window` ends up false.
            CoreModel::new(dir, language, intra_threads, true, opts.audio_ctx)
                .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))
                .context("failed to load whisper model")
        })?;
        let window = if inner.dynamic_window {
            // The floor is `--audio-ctx`; with the fixed encoder the floor does
            // not apply, so the line says so instead of naming it.
            format!("dynamic, floor={}", inner.min_window_frames)
        } else {
            "fixed 30s".to_string()
        };
        let ready_line = format!(
            "whisper ready (model={}, lang={}, threads={}, gpu=false, window={window})",
            opts.model_spec, language, opts.n_threads
        );
        log::info!("{}", ready_line);
        Ok(Self { inner, ready_line })
    }
}

impl SpeechModel for WhisperModel {
    fn transcribe(&mut self, pcm_16k: &[f32], running: &Arc<AtomicBool>) -> Result<Transcript> {
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

    /// The window the encoder ran, not the utterance. The export's encoder
    /// always decodes a fixed 30 s window (its `input_features` shape is
    /// constant), so a 2 s utterance really did cost a full window's encoder
    /// pass, and RTF should say so. The patched encoder follows
    /// the audio, and this follows it.
    fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
        self.inner.window_samples(conditioned_len)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MAX_UTTERANCE_SAMPLES;
    use crate::whisper_core::types::{
        window_frames, Language, MelConfig, DEFAULT_WINDOW_FRAMES, TOK_LANG_EN, TOK_LANG_PL,
    };

    #[test]
    fn the_fixed_window_is_thirty_seconds() {
        // The export's encoder graph pins the mel frame count, so the window is
        // fixed: RTF must not be flattered by a short utterance reporting its
        // own length instead.
        let config = MelConfig::large_v3_turbo();
        let fixed = window_frames(1_600, false, DEFAULT_WINDOW_FRAMES, &config) * config.hop_length;
        assert_eq!(fixed, MAX_UTTERANCE_SAMPLES);
        assert_eq!(fixed, 30 * 16_000, "30 s at 16 kHz");
        // It must equal the frontend's window, or `audio_sec` would describe
        // audio the model never saw.
        assert_eq!(fixed, config.n_frames * config.hop_length);
    }

    #[test]
    fn the_dynamic_window_reports_what_the_encoder_ran() {
        let config = MelConfig::large_v3_turbo();
        let ran = |samples| {
            window_frames(samples, true, DEFAULT_WINDOW_FRAMES, &config) * config.hop_length
        };
        // A short utterance runs the 15 s floor, not 30 s: that is the saving.
        assert_eq!(ran(2 * 16_000), 15 * 16_000);
        // Never more than the export's window.
        assert_eq!(ran(40 * 16_000), 30 * 16_000);
    }

    /// The compiled-in prompt tokens must be the exported ones: Whisper has no
    /// per-engine vocabulary lookup for these (see
    /// `whisper_core::types`), so a wrong id decodes garbage.
    #[test]
    fn language_prompt_tokens_are_the_exported_ones() {
        assert_eq!(Language::Polish.prompt_token(), TOK_LANG_PL);
        assert_eq!(Language::English.prompt_token(), TOK_LANG_EN);
        assert_eq!(
            Language::Auto.prompt_token(),
            TOK_LANG_PL,
            "auto falls back to pl"
        );
        assert_eq!(Language::Polish.as_str(), "pl");
        assert_eq!(Language::English.as_str(), "en");
        assert_eq!(Language::Auto.as_str(), "auto");
        assert_eq!(Language::English.to_string(), "en");
        // The default is what `--language` resolves to when unset, and sttd is
        // a Polish speech-to-text tool, so it must be Polish.
        assert_eq!(Language::default(), Language::Polish);
    }
}
