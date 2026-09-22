use std::path::PathBuf;

use clap::Parser;

use crate::config::{
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MODEL_SPEC, DEFAULT_PROMPT, DEFAULT_TARGET_RMS, DEFAULT_VAD_THRESHOLD, parse_bool,
};
use crate::denoise::DenoiserKind;

/// Real-time Polish speech-to-text.
///
/// Precedence for every tunable is: CLI flag > environment variable >
/// compiled-in default. Env names use the `STTD_*` spellings;
/// `WHISPER_MODEL_PATH` remains a silent fallback for `--model` so current
/// setups keep working.
#[derive(Parser, Debug, Clone)]
#[command(name = "sttd", version, about = "Real-time Polish speech-to-text")]
pub(crate) struct Cli {
    /// Transcribe a WAV file offline instead of the microphone.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["list_devices", "list_models"])]
    pub(crate) file: Option<PathBuf>,

    /// List available input devices and exit.
    #[arg(long, conflicts_with_all = ["file", "list_models"])]
    pub(crate) list_devices: bool,

    /// List known whisper models (with installed/default markers) and exit.
    #[arg(long, conflicts_with_all = ["file", "list_devices"])]
    pub(crate) list_models: bool,

    /// Substring match (case-insensitive) against an input device name.
    #[arg(long, env = "STTD_DEVICE")]
    pub(crate) device: Option<String>,

    /// Model spec (e.g. `small`, `large-v3-turbo-q8_0`) or path to a ggml
    /// whisper model file (auto-downloaded if missing). `STTD_MODEL` wins
    /// over the legacy `WHISPER_MODEL_PATH` fallback; CLI wins over both.
    #[arg(long, env = "STTD_MODEL", value_name = "SPEC|PATH")]
    pub(crate) model: Option<String>,

    /// VAD-branch denoiser.
    #[arg(
        long,
        env = "STTD_DENOISER",
        value_enum,
        ignore_case = true,
        default_value_t = DenoiserKind::Bypass
    )]
    pub(crate) denoiser: DenoiserKind,

    /// Override the whisper initial prompt.
    #[arg(long, env = "STTD_PROMPT", default_value = DEFAULT_PROMPT)]
    pub(crate) prompt: String,

    /// Write each utterance as WAV+TXT into this directory (TXT written
    /// for dropped utterances too, prefixed with the drop reason).
    #[arg(long, env = "STTD_DUMP_DIR")]
    pub(crate) dump_dir: Option<PathBuf>,

    /// Min speech-only RMS to keep an utterance.
    #[arg(long, env = "STTD_MIN_RMS", default_value_t = DEFAULT_MIN_SPEECH_RMS)]
    pub(crate) min_rms: f32,

    /// Min avg token confidence, weak acoustic evidence.
    #[arg(long, env = "STTD_MIN_CONF", default_value_t = DEFAULT_MIN_CONFIDENCE)]
    pub(crate) min_conf: f32,

    /// Min avg token confidence, strong acoustic evidence.
    #[arg(
        long,
        env = "STTD_MIN_CONF_STRONG",
        default_value_t = DEFAULT_MIN_CONFIDENCE_STRONG
    )]
    pub(crate) min_conf_strong: f32,

    /// Pre-inference gain target RMS, 0 disables.
    #[arg(long, env = "STTD_TARGET_RMS", default_value_t = DEFAULT_TARGET_RMS)]
    pub(crate) target_rms: f32,

    /// Minimum earshot score for a 16 ms window to count as speech.
    #[arg(long, env = "STTD_VAD_THRESHOLD", default_value_t = DEFAULT_VAD_THRESHOLD)]
    pub(crate) vad_threshold: f32,

    /// Copy each accepted transcript to the Wayland clipboard via `wl-copy`.
    /// Gates auto-paste together with `--paste`: paste only runs on fresh
    /// content, so both must be true for anything to be inserted.
    #[arg(
        long,
        env = "STTD_COPY",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) copy: bool,

    /// Paste the clipboard at the cursor via `ydotool key` (fixed Ctrl+V,
    /// Wayland only, requires `ydotoold` running). Skipped when the copy
    /// above did not run.
    #[arg(
        long,
        env = "STTD_PASTE",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) paste: bool,

    /// Dump per-token id/text/p/plog/is_timestamp to stderr.
    #[arg(
        long,
        env = "STTD_DEBUG_TOKENS",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) debug_tokens: bool,
}

impl Cli {
    /// Raw `--model` value with env precedence resolved: CLI (`STTD_MODEL`
    /// via clap) > legacy `WHISPER_MODEL_PATH` > compiled-in default spec.
    pub(crate) fn model_input(&self) -> String {
        effective_model_input(self.model.as_deref(), std::env::var("WHISPER_MODEL_PATH").ok())
    }
}

/// Pure precedence helper (see `Cli::model_input`), split out for tests so
/// they don't touch the process environment.
pub(crate) fn effective_model_input(cli_model: Option<&str>, legacy_env: Option<String>) -> String {
    if let Some(m) = cli_model {
        return m.to_string();
    }
    if let Some(legacy) = legacy_env {
        if !legacy.trim().is_empty() {
            return legacy;
        }
    }
    DEFAULT_MODEL_SPEC.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_match_compiled_constants() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert_eq!(cli.model, None);
        assert!(!cli.list_models);
        assert_eq!(
            effective_model_input(cli.model.as_deref(), None),
            DEFAULT_MODEL_SPEC
        );
        // Default spec resolves to the historical default path.
        let resolved = crate::weights::resolve_model_path(DEFAULT_MODEL_SPEC)
            .expect("default spec must resolve");
        assert_eq!(
            resolved,
            PathBuf::from("models/ggml-large-v3-turbo-q8_0.bin")
        );
        assert_eq!(cli.prompt, DEFAULT_PROMPT);
        assert_eq!(cli.min_rms, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(cli.min_conf, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(cli.min_conf_strong, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert_eq!(cli.target_rms, DEFAULT_TARGET_RMS);
        assert_eq!(cli.vad_threshold, DEFAULT_VAD_THRESHOLD);
        assert!(cli.copy);
        assert!(cli.paste);
        assert!(!cli.debug_tokens);
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
    }

    #[test]
    fn cli_flags_override_defaults() {
        let cli = Cli::try_parse_from([
            "sttd",
            "--denoiser",
            "deepfilter",
            "--min-rms",
            "0.01",
            "--copy",
            "off",
            "--debug-tokens",
            "1",
        ])
        .expect("flags must parse");
        assert_eq!(cli.denoiser, DenoiserKind::DeepFilter);
        assert!((cli.min_rms - 0.01).abs() < 1e-6);
        assert!(!cli.copy);
        assert!(cli.debug_tokens);
    }

    #[test]
    fn cli_model_accepts_spec_and_path() {
        let cli = Cli::try_parse_from(["sttd", "--model", "small"])
            .expect("spec must parse");
        assert_eq!(cli.model.as_deref(), Some("small"));
        let cli = Cli::try_parse_from(["sttd", "--model", "models/custom.bin"])
            .expect("path must parse");
        assert_eq!(cli.model.as_deref(), Some("models/custom.bin"));
    }

    #[test]
    fn cli_list_models_conflicts_with_file() {
        assert!(Cli::try_parse_from(["sttd", "--list-models", "--file", "a.wav"]).is_err());
    }

    #[test]
    fn model_precedence_is_cli_over_legacy_over_default() {
        assert_eq!(
            effective_model_input(Some("small"), Some("models/legacy.bin".into())),
            "small"
        );
        assert_eq!(
            effective_model_input(None, Some("models/legacy.bin".into())),
            "models/legacy.bin"
        );
        assert_eq!(
            effective_model_input(None, Some("  ".into())),
            DEFAULT_MODEL_SPEC
        );
        assert_eq!(effective_model_input(None, None), DEFAULT_MODEL_SPEC);
    }
}
