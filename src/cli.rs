use std::path::PathBuf;

use clap::Parser;

use crate::config::{
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MODEL_PATH, DEFAULT_PROMPT, DEFAULT_TARGET_RMS, DEFAULT_VAD_THRESHOLD, parse_bool,
};
use crate::denoise::DenoiserKind;

/// Real-time Polish speech-to-text.
///
/// Precedence for every tunable is: CLI flag > environment variable >
/// compiled-in default. Env names keep the existing `STTD_*` /
/// `WHISPER_MODEL_PATH` spellings so current setups keep working.
#[derive(Parser, Debug, Clone)]
#[command(name = "sttd", version, about = "Real-time Polish speech-to-text")]
pub(crate) struct Cli {
    /// Transcribe a WAV file offline instead of the microphone.
    #[arg(long, value_name = "PATH", conflicts_with = "list_devices")]
    pub(crate) file: Option<PathBuf>,

    /// List available input devices and exit.
    #[arg(long, conflicts_with = "file")]
    pub(crate) list_devices: bool,

    /// Substring match (case-insensitive) against an input device name.
    #[arg(long, env = "STTD_DEVICE")]
    pub(crate) device: Option<String>,

    /// Path to a ggml whisper model (auto-downloaded if missing).
    #[arg(long, env = "WHISPER_MODEL_PATH", default_value = DEFAULT_MODEL_PATH)]
    pub(crate) model: PathBuf,

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
    #[arg(
        long,
        env = "STTD_COPY",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) copy: bool,

    /// Paste the clipboard at the cursor via `wtype -M ctrl -k v -m ctrl`.
    /// Skipped when the copy above did not run.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_match_compiled_constants() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert_eq!(cli.model, PathBuf::from(DEFAULT_MODEL_PATH));
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
}
