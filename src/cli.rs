use std::path::PathBuf;

use clap::Parser;

use crate::config::{
    parse_bool, parse_fraction, parse_hangover_ms, parse_linear_rms, parse_threads,
    DEFAULT_AUTO_UNLOAD, DEFAULT_AUTO_UNLOAD_AFTER_SECS, DEFAULT_HANGOVER_MS, DEFAULT_MIN_DENSITY,
    DEFAULT_MIN_SPEECH_RMS, DEFAULT_MODEL_SPEC, DEFAULT_TARGET_RMS, DEFAULT_USE_GPU,
    DEFAULT_VAD_ADAPT, DEFAULT_VAD_THRESHOLD,
};
use crate::denoise::DenoiserKind;
use crate::dsp::{parse_channel_sel, ChannelSel};
use crate::whisper_core::types::Language;

/// How `--paste` inserts text via `ydotool` (Wayland only).
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PasteMode {
    /// LEFTCTRL+V (default).
    #[default]
    CtrlV,
    /// LEFTCTRL+LEFTSHIFT+V (terminals).
    CtrlShiftV,
    /// `ydotool type` via stdin (no clipboard, layout-dependent).
    Type,
}

/// Real-time Polish-English speech-to-text.
///
/// Precedence for every tunable is: CLI flag > environment variable >
/// compiled-in default. Env names use the `STTD_*` spellings.
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

    /// Input channel to transcribe: `0` = first channel, `1` = second, ...;
    /// `mix` averages all channels down to mono (default, previous behavior),
    /// `auto` measures once at startup and keeps the loudest channel.
    /// Useful when one channel is a noisy webcam capsule and the other a
    /// headset mic.
    #[arg(long, env = "STTD_CHANNEL", value_name = "MIX|AUTO|N", default_value = "mix", value_parser = parse_channel_sel, help_heading = "Input")]
    pub(crate) channel: ChannelSel,

    /// Model spec (e.g. `parakeet`, `parakeet-ultra-int8`, `parakeet-redux-int8`,
    /// `whisper`, `whisper-large-v3-turbo-int8`) or path to model weights (a
    /// model directory; auto-downloaded if missing).
    #[arg(
        short,
        long,
        env = "STTD_MODEL",
        value_name = "SPEC|PATH",
        help_heading = "Model"
    )]
    pub(crate) model: Option<String>,

    /// List known models (with installed/default markers) and exit.
    #[arg(long, help_heading = "Model", conflicts_with_all = ["file", "list_devices", "start", "stop", "status"])]
    pub(crate) list_models: bool,

    /// Custom vocabulary file for post-gate transcript correction (one phrase
    /// per line, canonical casing preserved; e.g. proper nouns). Applied only
    /// to accepted transcripts, never to drops. Empty/unset = off.
    #[arg(
        long,
        env = "STTD_DICTIONARY",
        value_name = "PATH",
        help_heading = "Model"
    )]
    pub(crate) dictionary: Option<PathBuf>,

    /// VAD-branch denoiser (`deepfilter` requires the `denoise` Cargo feature).
    #[arg(
        long,
        env = "STTD_DENOISER",
        value_enum,
        ignore_case = true,
        default_value_t = DenoiserKind::Bypass,
        help_heading = "Model"
    )]
    pub(crate) denoiser: DenoiserKind,

    /// Decoder thread count; `0` = auto (physical cores).
    #[arg(long, env = "STTD_THREADS", default_value_t = 0, value_parser = parse_threads, value_name = "N", help_heading = "Model")]
    pub(crate) threads: i32,

    /// Decoding language for Whisper models; Parakeet auto-detects and ignores
    /// it. `auto` detects per utterance and costs nothing extra (the prompt run
    /// already produces the language distribution).
    #[arg(
        long,
        env = "STTD_LANGUAGE",
        value_enum,
        ignore_case = true,
        default_value_t = Language::Polish,
        help_heading = "Model"
    )]
    pub(crate) language: Language,

    /// Live-streaming mode. While you speak, sttd decodes the open utterance
    /// about once a second and logs the partial text (`info`, `partial`); the
    /// final transcript, gates, clipboard and paste are unchanged. With
    /// Whisper it also patches the encoder in memory so the window follows the
    /// audio (15 s at least) instead of always being 30 s: about 2x less
    /// time per utterance. Microphone mode only for partials; `--file` gets the
    /// short window alone.
    #[arg(
        long,
        env = "STTD_STREAM",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Model"
    )]
    pub(crate) stream: bool,

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
    #[arg(long, env = "STTD_VAD_THRESHOLD", default_value_t = DEFAULT_VAD_THRESHOLD, value_parser = parse_fraction, help_heading = "Acoustic gates")]
    pub(crate) vad_threshold: f32,

    /// Adapt the VAD threshold to the background noise floor with hysteresis
    /// (`off` = single fixed threshold, pre-adapt behavior).
    #[arg(long, env = "STTD_VAD_ADAPT", default_value_t = DEFAULT_VAD_ADAPT, value_parser = parse_bool, num_args = 0..=1, default_missing_value = "true", help_heading = "Acoustic gates")]
    pub(crate) vad_adapt: bool,

    /// Min speech-only RMS to keep an utterance.
    #[arg(long, env = "STTD_MIN_RMS", default_value_t = DEFAULT_MIN_SPEECH_RMS, value_parser = parse_linear_rms, help_heading = "Acoustic gates")]
    pub(crate) min_rms: f32,

    /// Min share of transmitted frames flagged as speech (long utterances exempt).
    #[arg(long, env = "STTD_MIN_DENSITY", default_value_t = DEFAULT_MIN_DENSITY, value_parser = parse_fraction, help_heading = "Acoustic gates")]
    pub(crate) min_density: f32,

    /// Silence hangover before an utterance closes (ms, step 10). Lower =
    /// faster emission, more sentence splits.
    #[arg(long, env = "STTD_HANGOVER_MS", default_value_t = DEFAULT_HANGOVER_MS, value_parser = parse_hangover_ms, value_name = "MS", help_heading = "Acoustic gates")]
    pub(crate) hangover_ms: u64,

    /// Frame-level Agc target RMS (VAD-gated shared gain, speech-only), 0 disables.
    #[arg(long, env = "STTD_TARGET_RMS", default_value_t = DEFAULT_TARGET_RMS, value_parser = parse_linear_rms, help_heading = "Acoustic gates")]
    pub(crate) target_rms: f32,

    /// Remove hesitation/filler words (yyy, eee, um, ...) from emitted
    /// transcripts. Off by default: the list is stripped whatever the
    /// language, so it can delete real words.
    #[arg(
        long,
        env = "STTD_REMOVE_HESITATIONS",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Transcript gates"
    )]
    pub(crate) remove_hesitations: bool,

    /// Copy each accepted transcript to the Wayland clipboard via `wl-copy`.
    /// Gates auto-paste together with `--paste`: paste only runs on fresh
    /// content, so both must be true for anything to be inserted.
    #[arg(
        long,
        env = "STTD_COPY",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Emission"
    )]
    pub(crate) copy: bool,

    /// Paste the clipboard at the cursor via `ydotool` (Wayland only,
    /// requires `ydotoold` running). Skipped when the copy above did not run.
    /// `--paste-mode` selects the key sequence (`ctrl-v` default,
    /// `ctrl-shift-v` for terminals, `type` to type via `ydotool type`).
    #[arg(
        long,
        env = "STTD_PASTE",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Emission"
    )]
    pub(crate) paste: bool,

    /// How `--paste` inserts: `ctrl-v` (LEFTCTRL+V), `ctrl-shift-v`
    /// (LEFTCTRL+LEFTSHIFT+V for terminals), or `type` (`ydotool type`
    /// via stdin, no clipboard). Wayland only.
    #[arg(
        long,
        env = "STTD_PASTE_MODE",
        value_enum,
        ignore_case = true,
        default_value_t = PasteMode::CtrlV,
        help_heading = "Emission"
    )]
    pub(crate) paste_mode: PasteMode,

    /// Re-copy and re-paste the last emitted transcription and exit.
    /// Reads the daemon's last-text file; warns when none exists yet.
    #[arg(
        long,
        help_heading = "Emission",
        conflicts_with_all = ["file", "list_devices", "list_models", "start", "stop", "status"]
    )]
    pub(crate) paste_last: bool,

    /// Print each accepted transcript to stdout (one line per utterance).
    /// Opt-in: off by default.
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
    /// Owner-only permissions (0700/0600 on unix); only the newest 100
    /// utterance pairs are kept, older ones pruned.
    #[arg(short = 'o', long, env = "STTD_DUMP_DIR", help_heading = "Emission")]
    pub(crate) dump_dir: Option<PathBuf>,

    /// Start microphone mode paused (no audio is transcribed until
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

    /// Append one JSON object per transcribed utterance to this file
    /// (tag, audio length, inference time, RTF, acoustic evidence,
    /// verdict, text). For offline tuning and regression
    /// analysis; transcription never depends on it (open errors only warn).
    #[arg(
        long,
        env = "STTD_METRICS",
        value_name = "PATH",
        help_heading = "Runtime"
    )]
    pub(crate) metrics: Option<PathBuf>,

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

    /// Unload the model while paused to free VRAM; reloaded eagerly
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

    /// While paused longer than this (seconds), unload the model
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

    /// Do not serve the session-bus interface (`org.sttd.Daemon`, object
    /// `/org/sttd/Daemon`). Signal pause/resume stays available. File mode
    /// never serves it anyway.
    #[arg(
        long,
        env = "STTD_NO_DBUS",
        default_value_t = false,
        value_parser = parse_bool,
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Runtime"
    )]
    pub(crate) no_dbus: bool,

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
    /// via clap) or compiled-in default spec.
    pub(crate) fn model_input(&self) -> String {
        effective_model_input(self.model.as_deref())
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

/// Pure precedence helper (see `Cli::model_input`), split out for tests.
pub(crate) fn effective_model_input(cli_model: Option<&str>) -> String {
    if let Some(m) = cli_model {
        return m.to_string();
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
            effective_model_input(cli.model.as_deref()),
            DEFAULT_MODEL_SPEC
        );
        // Default spec resolves to the Parakeet model directory.
        let resolved = crate::model_store::resolve_model(DEFAULT_MODEL_SPEC)
            .expect("default spec must resolve");
        assert_eq!(
            resolved,
            crate::model_store::ModelRef {
                dir: PathBuf::from("models/parakeet-ultra-int8"),
                assets: &[
                    "encoder-model.int8.onnx",
                    "decoder_joint-model.int8.onnx",
                    "vocab.txt",
                    "tokenizer.json",
                ],
                kind: crate::model_store::EngineKind::Parakeet,
            }
        );
        assert_eq!(cli.hangover_ms, DEFAULT_HANGOVER_MS);
        assert_eq!(cli.channel, ChannelSel::Mix);
        assert_eq!(cli.threads, 0);
        assert_eq!(cli.gpu, DEFAULT_USE_GPU);
        assert_eq!(cli.min_rms, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(cli.min_density, DEFAULT_MIN_DENSITY);
        assert_eq!(cli.target_rms, DEFAULT_TARGET_RMS);
        assert_eq!(cli.vad_threshold, DEFAULT_VAD_THRESHOLD);
        assert_eq!(cli.vad_adapt, DEFAULT_VAD_ADAPT);
        assert!(!cli.copy);
        assert!(!cli.stream, "streaming is opt-in");
        assert!(!cli.paste);
        assert_eq!(cli.paste_mode, PasteMode::CtrlV);
        assert!(!cli.paste_last);
        assert!(!cli.remove_hesitations);
        assert!(!cli.stdout);
        assert!(cli.start_paused);
        assert!(!cli.autopause);
        assert!(!cli.no_dbus);
        assert!(cli.sound);
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
        assert!(!cli.start);
        assert!(!cli.stop);
        assert!(!cli.status);
        assert_eq!(cli.pid_file, None);
        assert_eq!(cli.log_file, None);
        assert_eq!(cli.metrics, None);
        assert_eq!(cli.dictionary, None);
        assert!(!cli.auto_unload);
        assert_eq!(cli.auto_unload, DEFAULT_AUTO_UNLOAD);
        assert_eq!(cli.auto_unload_after, DEFAULT_AUTO_UNLOAD_AFTER_SECS);
    }

    #[test]
    fn cli_flags_override_defaults() {
        #[cfg(feature = "denoise")]
        let cli = Cli::try_parse_from([
            "sttd",
            "--denoiser",
            "deepfilter",
            "--min-rms",
            "0.01",
            "--copy",
            "off",
            "--stdout",
            "--start-paused",
            "off",
            "--autopause",
            "on",
            "--threads",
            "0",
            "--gpu",
            "off",
        ])
        .expect("flags must parse");
        #[cfg(not(feature = "denoise"))]
        let cli = Cli::try_parse_from([
            "sttd",
            "--denoiser",
            "none",
            "--min-rms",
            "0.01",
            "--copy",
            "off",
            "--stdout",
            "--start-paused",
            "off",
            "--autopause",
            "on",
            "--threads",
            "0",
            "--gpu",
            "off",
        ])
        .expect("flags must parse");
        #[cfg(feature = "denoise")]
        assert_eq!(cli.denoiser, DenoiserKind::DeepFilter);
        #[cfg(not(feature = "denoise"))]
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
        assert!((cli.min_rms - 0.01).abs() < 1e-6);
        assert!(!cli.copy);
        assert!(cli.stdout);
        assert!(!cli.start_paused);
        assert!(cli.autopause);
        assert_eq!(cli.threads, 0);
        assert!(!cli.gpu);
    }

    #[cfg(not(feature = "denoise"))]
    #[test]
    fn cli_deepfilter_rejected_without_denoise_feature() {
        assert!(Cli::try_parse_from(["sttd", "--denoiser", "deepfilter"]).is_err());
        let cli = Cli::try_parse_from(["sttd", "--denoiser", "none"]).expect("none must parse");
        assert_eq!(cli.denoiser, DenoiserKind::Bypass);
    }

    #[test]
    fn cli_channel_and_hangover_parse() {
        let cli = Cli::try_parse_from(["sttd", "--channel", "1", "--hangover-ms", "350"])
            .expect("channel/hangover must parse");
        assert_eq!(cli.channel, ChannelSel::Pick(1));
        assert_eq!(cli.hangover_ms, 350);
        let cli = Cli::try_parse_from(["sttd", "--channel", "auto"]).expect("auto must parse");
        assert_eq!(cli.channel, ChannelSel::Auto);
        let cli = Cli::try_parse_from(["sttd", "--channel", "mix"]).expect("mix must parse");
        assert_eq!(cli.channel, ChannelSel::Mix);
        assert!(Cli::try_parse_from(["sttd", "--hangover-ms", "355"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--hangover-ms", "100"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--channel", "bzdura"]).is_err());
    }

    #[test]
    fn cli_model_accepts_spec_and_path() {
        let cli = Cli::try_parse_from(["sttd", "--model", "parakeet"]).expect("spec must parse");
        assert_eq!(cli.model.as_deref(), Some("parakeet"));
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
    fn cli_paste_last_conflicts_and_parses() {
        assert!(Cli::try_parse_from(["sttd", "--paste-last", "--file", "a.wav"]).is_err());
        assert!(Cli::try_parse_from(["sttd", "--paste-last", "--start"]).is_err());
        let cli = Cli::try_parse_from(["sttd", "--paste-last"]).expect("must parse");
        assert!(cli.paste_last);
    }

    #[test]
    fn cli_paste_mode_and_hesitations_parse() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert_eq!(cli.paste_mode, PasteMode::CtrlV);
        assert!(!cli.remove_hesitations);
        let cli =
            Cli::try_parse_from(["sttd", "--paste-mode", "ctrl-shift-v"]).expect("must parse");
        assert_eq!(cli.paste_mode, PasteMode::CtrlShiftV);
        let cli = Cli::try_parse_from(["sttd", "--paste-mode", "type"]).expect("must parse");
        assert_eq!(cli.paste_mode, PasteMode::Type);
        let cli =
            Cli::try_parse_from(["sttd", "--remove-hesitations"]).expect("bare flag must parse");
        assert!(cli.remove_hesitations);
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
    fn cli_stream_parses_bare_and_with_a_value() {
        let cli = Cli::try_parse_from(["sttd", "--stream"]).expect("bare flag must parse");
        assert!(cli.stream);
        let cli = Cli::try_parse_from(["sttd", "--stream", "off"]).expect("off must parse");
        assert!(!cli.stream);
        assert!(Cli::try_parse_from(["sttd", "--stream", "sometimes"]).is_err());
    }

    #[test]
    fn cli_vad_adapt_toggles() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert!(cli.vad_adapt);
        let cli = Cli::try_parse_from(["sttd", "--vad-adapt", "off"]).expect("off must parse");
        assert!(!cli.vad_adapt);
    }

    #[test]
    fn cli_inference_tunables_reject_out_of_range() {
        assert!(Cli::try_parse_from(["sttd", "--threads", "-1"]).is_err());
        let cli = Cli::try_parse_from(["sttd", "--threads", "0"]).expect("0 = auto");
        assert_eq!(cli.threads, 0);
    }

    #[test]
    fn model_precedence_is_cli_over_default() {
        assert_eq!(effective_model_input(Some("parakeet")), "parakeet");
        assert_eq!(effective_model_input(None), DEFAULT_MODEL_SPEC);
    }
}
