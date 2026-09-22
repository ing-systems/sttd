use std::path::PathBuf;

use clap::Parser;

use crate::config::{
    parse_beam_size, parse_bool, parse_threads, DEFAULT_AUTO_UNLOAD, DEFAULT_AUTO_UNLOAD_AFTER_SECS,
    DEFAULT_BEAM_SIZE, DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
    DEFAULT_MIN_SPEECH_RMS, DEFAULT_MODEL_SPEC, DEFAULT_PROMPT, DEFAULT_TARGET_RMS,
    DEFAULT_THREADS, DEFAULT_USE_GPU, DEFAULT_VAD_ADAPT, DEFAULT_VAD_THRESHOLD,
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

    /// List available input devices as an aligned table (`*` = system
    /// default, `>` = matches `--device`) and exit.
    #[arg(long, help_heading = "Input", conflicts_with_all = ["file", "list_models", "start", "stop", "status"])]
    pub(crate) list_devices: bool,

    /// Model spec (e.g. `small`, `large-v3-turbo-q8_0`) or path to a ggml
    /// whisper model file (auto-downloaded if missing). `STTD_MODEL` wins
    /// over the legacy `WHISPER_MODEL_PATH` fallback; CLI wins over both.
    #[arg(
        short,
        long,
        env = "STTD_MODEL",
        value_name = "SPEC|PATH",
        help_heading = "Model"
    )]
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

    /// Beam width of whisper's beam-search decoder (`1` = greedy).
    #[arg(long, env = "STTD_BEAM_SIZE", default_value_t = DEFAULT_BEAM_SIZE, value_parser = parse_beam_size, value_name = "N", help_heading = "Model")]
    pub(crate) beam_size: i32,

    /// Decoder thread count; `0` = auto (`available_parallelism`).
    #[arg(long, env = "STTD_THREADS", default_value_t = DEFAULT_THREADS, value_parser = parse_threads, value_name = "N", help_heading = "Model")]
    pub(crate) threads: i32,

    /// Load the model onto the GPU (`off`/`0` = CPU-only).
    #[arg(
        long,
        env = "STTD_GPU",
        default_value_t = DEFAULT_USE_GPU,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Model"
    )]
    pub(crate) gpu: bool,

    /// Minimum earshot score for a 16 ms window to count as speech.
    #[arg(long, env = "STTD_VAD_THRESHOLD", default_value_t = DEFAULT_VAD_THRESHOLD, help_heading = "Acoustic gates")]
    pub(crate) vad_threshold: f32,

    /// Adapt the VAD threshold to the background noise floor with hysteresis
    /// (`off` = single fixed threshold, pre-adapt behavior).
    #[arg(long, env = "STTD_VAD_ADAPT", default_value_t = DEFAULT_VAD_ADAPT, value_parser = parse_bool, num_args = 0..=1, default_missing_value = "true", help_heading = "Acoustic gates")]
    pub(crate) vad_adapt: bool,

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

    /// Print each accepted transcript to stdout (one line per utterance).
    /// Opt-in: off by default so transcripts only go to the clipboard,
    /// auto-paste and the dump dir unless this is enabled.
    #[arg(
        long,
        env = "STTD_STDOUT",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Emission"
    )]
    pub(crate) stdout: bool,

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

    /// After each emitted transcript, pause again automatically
    /// (single-shot resume: every `SIGUSR2`/`SIGHUP` arms one sentence).
    /// Drops (low-confidence, hallucination, repetition, acoustic) never
    /// trigger it, so noise can't steal the shot. File mode ignores this.
    #[arg(
        long,
        env = "STTD_AUTOPAUSE",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) autopause: bool,

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
    #[arg(
        long,
        env = "STTD_PID_FILE",
        value_name = "PATH",
        help_heading = "Runtime"
    )]
    pub(crate) pid_file: Option<PathBuf>,

    /// Combined stdout+stderr log (append) for `--start`. Diagnostics
    /// (`log`) always land here; transcripts (`println!`) only with
    /// `--stdout`.
    /// Defaults to `<tmp>/sttd.log`.
    #[arg(
        long,
        env = "STTD_LOG_FILE",
        value_name = "PATH",
        help_heading = "Runtime"
    )]
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

    /// Unload the whisper model while paused to free VRAM; reloaded eagerly
    /// on resume. Disabled by default.
    #[arg(
        long,
        env = "STTD_AUTO_UNLOAD",
        default_value_t = DEFAULT_AUTO_UNLOAD,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) auto_unload: bool,

    /// While paused longer than this (seconds), unload the whisper model
    /// to free VRAM; reloaded eagerly on resume. Only takes effect if
    /// `--auto-unload` is enabled. `0` unloads immediately on pause.
    #[arg(
        long,
        env = "STTD_AUTO_UNLOAD_AFTER",
        default_value_t = DEFAULT_AUTO_UNLOAD_AFTER_SECS,
        value_name = "SECS",
        help_heading = "Runtime"
    )]
    pub(crate) auto_unload_after: u64,

    /// Play audible cues on pause (low beep), resume (mid blip) and when
    /// transcription is actually ready (rising two-tone) via the default
    /// output device. Best-effort: missing/busy audio only logs.
    #[arg(
        long,
        env = "STTD_SOUND",
        default_value_t = true,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) sound: bool,
}

impl Cli {
    /// Raw `--model` value with env precedence resolved: CLI (`STTD_MODEL`
    /// via clap) > legacy `WHISPER_MODEL_PATH` > compiled-in default spec.
    pub(crate) fn model_input(&self) -> String {
        effective_model_input(
            self.model.as_deref(),
            std::env::var("WHISPER_MODEL_PATH").ok(),
        )
    }

    /// Effective pid-file path: `--pid-file`/`STTD_PID_FILE` or `<tmp>/sttd.pid`.
    pub(crate) fn pid_file_path(&self) -> PathBuf {
        self.pid_file
            .clone()
            .unwrap_or_else(crate::daemon::default_pid_file)
    }

    /// Effective log path: `--log-file`/`STTD_LOG_FILE` or `<tmp>/sttd.log`.
    pub(crate) fn log_file_path(&self) -> PathBuf {
        self.log_file
            .clone()
            .unwrap_or_else(crate::daemon::default_log_file)
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
        assert_eq!(cli.beam_size, DEFAULT_BEAM_SIZE);
        assert_eq!(cli.threads, DEFAULT_THREADS);
        assert_eq!(cli.gpu, DEFAULT_USE_GPU);
        assert_eq!(cli.min_rms, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(cli.min_density, DEFAULT_MIN_DENSITY);
        assert_eq!(cli.min_conf, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(cli.min_conf_strong, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert_eq!(cli.target_rms, DEFAULT_TARGET_RMS);
        assert_eq!(cli.vad_threshold, DEFAULT_VAD_THRESHOLD);
        assert_eq!(cli.vad_adapt, DEFAULT_VAD_ADAPT);
        assert!(cli.copy);
        assert!(cli.paste);
        assert!(!cli.stdout);
        assert!(!cli.debug_tokens);
        assert!(cli.start_paused);
        assert!(!cli.autopause);
        assert!(cli.sound);
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
        assert!(!cli.start);
        assert!(!cli.stop);
        assert!(!cli.status);
        assert_eq!(cli.pid_file, None);
        assert_eq!(cli.log_file, None);
        assert!(!cli.auto_unload);
        assert_eq!(cli.auto_unload, DEFAULT_AUTO_UNLOAD);
        assert_eq!(cli.auto_unload_after, DEFAULT_AUTO_UNLOAD_AFTER_SECS);
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
            "--stdout",
            "--debug-tokens",
            "1",
            "--start-paused",
            "off",
            "--autopause",
            "on",
            "--beam-size",
            "3",
            "--threads",
            "0",
            "--gpu",
            "off",
        ])
        .expect("flags must parse");
        assert_eq!(cli.denoiser, DenoiserKind::DeepFilter);
        assert!((cli.min_rms - 0.01).abs() < 1e-6);
        assert!(!cli.copy);
        assert!(cli.stdout);
        assert!(cli.debug_tokens);
        assert!(!cli.start_paused);
        assert!(cli.autopause);
        assert_eq!(cli.beam_size, 3);
        assert_eq!(cli.threads, 0);
        assert!(!cli.gpu);
    }

    #[test]
    fn cli_model_accepts_spec_and_path() {
        let cli = Cli::try_parse_from(["sttd", "--model", "small"]).expect("spec must parse");
        assert_eq!(cli.model.as_deref(), Some("small"));
        let cli =
            Cli::try_parse_from(["sttd", "--model", "models/custom.bin"]).expect("path must parse");
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
            "--auto-unload",
            "--auto-unload-after",
            "0",
        ])
        .expect("daemon paths must parse");
        assert_eq!(
            cli.pid_file.as_deref(),
            Some(std::path::Path::new("/tmp/custom.pid"))
        );
        assert_eq!(
            cli.log_file.as_deref(),
            Some(std::path::Path::new("/tmp/custom.log"))
        );
        assert_eq!(cli.auto_unload_after, 0);
        assert!(cli.auto_unload);
        // Defaults resolve into tmp when unset.
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert!(cli.pid_file_path().ends_with("sttd.pid"));
        assert!(cli.log_file_path().ends_with("sttd.log"));
    }

    #[test]
    fn cli_vad_adapt_toggles() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert!(cli.vad_adapt);
        let cli =
            Cli::try_parse_from(["sttd", "--vad-adapt", "off"]).expect("off must parse");
        assert!(!cli.vad_adapt);
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
    fn cli_inference_tunables_reject_out_of_range() {
        assert!(Cli::try_parse_from(["sttd", "--beam-size", "0"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--beam-size", "-1"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--threads", "-1"]).is_err());
        let cli = Cli::try_parse_from(["sttd", "--threads", "0"]).expect("0 = auto");
        assert_eq!(cli.threads, 0);
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
