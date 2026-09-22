use std::path::PathBuf;

use clap::Parser;

use crate::config::{
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
    DEFAULT_MIN_SPEECH_RMS, DEFAULT_MODEL_SPEC, DEFAULT_PAUSED_UNLOAD_AFTER_SECS,
    DEFAULT_PROMPT, DEFAULT_TARGET_RMS, DEFAULT_VAD_THRESHOLD, parse_bool,
};
use crate::denoise::DenoiserKind;
use crate::whisper::Language;

/// Real-time Polish-English speech-to-text.
///
/// Precedence for every tunable is: CLI flag > environment variable >
/// compiled-in default. Env names use the `STTD_*` spellings;
/// `WHISPER_MODEL_PATH` remains a silent fallback for `--model` so current
/// setups keep working.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "sttd",
    version,
    about = "Real-time Polish-English speech-to-text"
)]
pub(crate) struct Cli {
    /// Transcribe a WAV file offline instead of the microphone.
    #[arg(short, long, value_name = "PATH", help_heading = "Input", conflicts_with_all = ["list_devices", "list_models", "start", "stop", "status"])]
    pub(crate) file: Option<PathBuf>,

    /// Substring match (case-insensitive) against an input device name.
    #[arg(short, long, env = "STTD_DEVICE", help_heading = "Input")]
    pub(crate) device: Option<String>,

    /// List available input devices and exit.
    #[arg(long, help_heading = "Input", conflicts_with_all = ["file", "list_models", "start", "stop", "status"])]
    pub(crate) list_devices: bool,

    /// Model spec (e.g. `small`, `large-v3-turbo-q8_0`) or path to a ggml
    /// whisper model file (auto-downloaded if missing). `STTD_MODEL` wins
    /// over the legacy `WHISPER_MODEL_PATH` fallback; CLI wins over both.
    #[arg(short, long, env = "STTD_MODEL", value_name = "SPEC|PATH", help_heading = "Model")]
    pub(crate) model: Option<String>,

    /// List known whisper models (with installed/default markers) and exit.
    #[arg(long, help_heading = "Model", conflicts_with_all = ["file", "list_devices", "start", "stop", "status"])]
    pub(crate) list_models: bool,

    /// Whisper decoding language: `pl` forces Polish, `en` forces English,
    /// `auto` detects per utterance.
    #[arg(
        short,
        long,
        env = "STTD_LANGUAGE",
        value_enum,
        ignore_case = true,
        default_value_t = Language::Polish,
        help_heading = "Model"
    )]
    pub(crate) language: Language,

    /// Override the whisper initial prompt.
    #[arg(short, long, env = "STTD_PROMPT", default_value = DEFAULT_PROMPT, help_heading = "Model")]
    pub(crate) prompt: String,

    /// VAD-branch denoiser.
    #[arg(
        long,
        env = "STTD_DENOISER",
        value_enum,
        ignore_case = true,
        default_value_t = DenoiserKind::Bypass,
        help_heading = "Model"
    )]
    pub(crate) denoiser: DenoiserKind,

    /// Minimum earshot score for a 16 ms window to count as speech.
    #[arg(long, env = "STTD_VAD_THRESHOLD", default_value_t = DEFAULT_VAD_THRESHOLD, help_heading = "Acoustic gates")]
    pub(crate) vad_threshold: f32,

    /// Min speech-only RMS to keep an utterance.
    #[arg(long, env = "STTD_MIN_RMS", default_value_t = DEFAULT_MIN_SPEECH_RMS, help_heading = "Acoustic gates")]
    pub(crate) min_rms: f32,

    /// Min share of transmitted frames flagged as speech (long utterances exempt).
    #[arg(long, env = "STTD_MIN_DENSITY", default_value_t = DEFAULT_MIN_DENSITY, help_heading = "Acoustic gates")]
    pub(crate) min_density: f32,

    /// Pre-inference gain target RMS, 0 disables.
    #[arg(long, env = "STTD_TARGET_RMS", default_value_t = DEFAULT_TARGET_RMS, help_heading = "Acoustic gates")]
    pub(crate) target_rms: f32,

    /// Min avg token confidence, weak acoustic evidence.
    #[arg(long, env = "STTD_MIN_CONF", default_value_t = DEFAULT_MIN_CONFIDENCE, help_heading = "Transcript gates")]
    pub(crate) min_conf: f32,

    /// Min avg token confidence, strong acoustic evidence.
    #[arg(
        long,
        env = "STTD_MIN_CONF_STRONG",
        default_value_t = DEFAULT_MIN_CONFIDENCE_STRONG,
        help_heading = "Transcript gates"
    )]
    pub(crate) min_conf_strong: f32,

    /// Copy each accepted transcript to the Wayland clipboard via `wl-copy`.
    /// Gates auto-paste together with `--paste`: paste only runs on fresh
    /// content, so both must be true for anything to be inserted.
    #[arg(
        long,
        env = "STTD_COPY",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Emission"
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
        default_missing_value = "true",
        help_heading = "Emission"
    )]
    pub(crate) paste: bool,

    /// Write each utterance as WAV+TXT into this directory (TXT written
    /// for dropped utterances too, prefixed with the drop reason).
    #[arg(short = 'o', long, env = "STTD_DUMP_DIR", help_heading = "Emission")]
    pub(crate) dump_dir: Option<PathBuf>,

    /// Start microphone mode paused (no audio reaches whisper until
    /// SIGUSR2/SIGHUP). File mode ignores this.
    #[arg(
        long,
        env = "STTD_START_PAUSED",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) start_paused: bool,

    /// Dump per-token id/text/p/plog/is_timestamp to stderr.
    #[arg(
        long,
        env = "STTD_DEBUG_TOKENS",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) debug_tokens: bool,

    /// Start microphone mode as a background daemon (fork, detach, pid
    /// file, combined stdout+stderr log). Parent prints the pid/log paths
    /// and exits; the child keeps running past terminal close.
    #[arg(
        long,
        env = "STTD_START",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime",
        conflicts_with_all = ["file", "list_devices", "list_models", "stop", "status"]
    )]
    pub(crate) start: bool,

    /// PID file written by `--start` and read by `--stop`/`--status`.
    /// Defaults to `<tmp>/sttd.pid`.
    #[arg(long, env = "STTD_PID_FILE", value_name = "PATH", help_heading = "Runtime")]
    pub(crate) pid_file: Option<PathBuf>,

    /// Combined stdout+stderr log (append) for `--start`. Transcripts
    /// (`println!`) and diagnostics (`log`) both land here.
    /// Defaults to `<tmp>/sttd.log`.
    #[arg(long, env = "STTD_LOG_FILE", value_name = "PATH", help_heading = "Runtime")]
    pub(crate) log_file: Option<PathBuf>,

    /// Stop the background daemon: SIGTERM its pid (graceful flush),
    /// wait, then SIGKILL on timeout. Reads `--pid-file`.
    #[arg(
        long,
        help_heading = "Runtime",
        conflicts_with_all = ["file", "list_devices", "list_models", "start", "status"]
    )]
    pub(crate) stop: bool,

    /// Check whether the background daemon is running. Reads `--pid-file`;
    /// exit code 0 iff running.
    #[arg(
        long,
        help_heading = "Runtime",
        conflicts_with_all = ["file", "list_devices", "list_models", "start", "stop"]
    )]
    pub(crate) status: bool,

    /// While paused longer than this (seconds), unload the whisper model
    /// to free VRAM; reloaded eagerly on resume. `0` disables unloading.
    #[arg(
        long,
        env = "STTD_PAUSED_UNLOAD_AFTER",
        default_value_t = DEFAULT_PAUSED_UNLOAD_AFTER_SECS,
        value_name = "SECS",
        help_heading = "Runtime"
    )]
    pub(crate) paused_unload_after: u64,
}

impl Cli {
    /// Raw `--model` value with env precedence resolved: CLI (`STTD_MODEL`
    /// via clap) > legacy `WHISPER_MODEL_PATH` > compiled-in default spec.
    pub(crate) fn model_input(&self) -> String {
        effective_model_input(self.model.as_deref(), std::env::var("WHISPER_MODEL_PATH").ok())
    }

    /// Effective pid-file path: `--pid-file`/`STTD_PID_FILE` or `<tmp>/sttd.pid`.
    pub(crate) fn pid_file_path(&self) -> PathBuf {
        self.pid_file.clone().unwrap_or_else(crate::daemon::default_pid_file)
    }

    /// Effective log path: `--log-file`/`STTD_LOG_FILE` or `<tmp>/sttd.log`.
    pub(crate) fn log_file_path(&self) -> PathBuf {
        self.log_file.clone().unwrap_or_else(crate::daemon::default_log_file)
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
        assert_eq!(cli.language, crate::whisper::Language::Polish);
        assert_eq!(cli.min_rms, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(cli.min_density, DEFAULT_MIN_DENSITY);
        assert_eq!(cli.min_conf, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(cli.min_conf_strong, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert_eq!(cli.target_rms, DEFAULT_TARGET_RMS);
        assert_eq!(cli.vad_threshold, DEFAULT_VAD_THRESHOLD);
        assert!(cli.copy);
        assert!(cli.paste);
        assert!(!cli.debug_tokens);
        assert!(cli.start_paused);
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
        assert!(!cli.start);
        assert!(!cli.stop);
        assert!(!cli.status);
        assert_eq!(cli.pid_file, None);
        assert_eq!(cli.log_file, None);
        assert_eq!(cli.paused_unload_after, DEFAULT_PAUSED_UNLOAD_AFTER_SECS);
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
            "--start-paused",
            "off",
        ])
        .expect("flags must parse");
        assert_eq!(cli.denoiser, DenoiserKind::DeepFilter);
        assert!((cli.min_rms - 0.01).abs() < 1e-6);
        assert!(!cli.copy);
        assert!(cli.debug_tokens);
        assert!(!cli.start_paused);
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
    fn cli_start_conflicts_with_file_and_stop_and_status() {
        assert!(Cli::try_parse_from(["sttd", "--start", "--file", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--start", "--stop"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--start", "--status"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--start", "--list-devices"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--start", "--list-models"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--stop", "--status"]).is_err());
    }

    #[test]
    fn cli_daemon_paths_and_unload_timeout_parse() {
        let cli = Cli::try_parse_from([
            "sttd",
            "--pid-file",
            "/tmp/custom.pid",
            "--log-file",
            "/tmp/custom.log",
            "--paused-unload-after",
            "0",
        ])
        .expect("daemon paths must parse");
        assert_eq!(cli.pid_file.as_deref(), Some(std::path::Path::new("/tmp/custom.pid")));
        assert_eq!(cli.log_file.as_deref(), Some(std::path::Path::new("/tmp/custom.log")));
        assert_eq!(cli.paused_unload_after, 0);
        // Defaults resolve into tmp when unset.
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert!(cli.pid_file_path().ends_with("sttd.pid"));
        assert!(cli.log_file_path().ends_with("sttd.log"));
    }

    #[test]
    fn cli_language_accepts_pl_en_auto() {
        use crate::whisper::Language;
        let cli = Cli::try_parse_from(["sttd", "--language", "en"]).expect("en must parse");
        assert_eq!(cli.language, Language::English);
        let cli = Cli::try_parse_from(["sttd", "--language", "auto"]).expect("auto must parse");
        assert_eq!(cli.language, Language::Auto);
        let cli = Cli::try_parse_from(["sttd", "--language", "PL"]).expect("case-insensitive");
        assert_eq!(cli.language, Language::Polish);
        assert!(Cli::try_parse_from(["sttd", "--language", "de"]).is_err());
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
