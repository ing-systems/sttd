//! Real-time Speech-to-Text pipeline.
//!
//! Data flow:
//! `cpal` (hw rate, N channels) -> channel pick (`--channel`) or mix to mono
//! -> per ~10 ms raw chunk
//! (`FrameProcessor`: `ResamplerFir` behind a 10 ms framing adapter, see
//! `FirStream`). By default (`--denoiser none`) a single `resampler`
//! hop (-> 16 kHz) feeds both branches below, each completed frame serving
//! both roles; with `--denoiser deepfilter` (requires the `denoise` Cargo
//! feature) the VAD branch instead takes
//! a 48 kHz detour:
//!   - VAD branch:     `resampler` (-> 16 kHz, shared hop) -> `earshot`,
//!       or `resampler` (-> 48 kHz) -> denoise (`deepfilter`, see `denoise`)
//!       -> `resampler` (-> 16 kHz) -> `earshot`
//!   - model branch: `resampler` (-> 16 kHz, direct hop, raw) -> utterance buffer
//! An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive speech
//! frames (a single VAD false positive no longer opens one), with ~300 ms of
//! pre-roll prepended at that point so quiet word starts aren't clipped.
//! Once open, every frame is appended regardless of the VAD verdict (speech
//! or not) until the `--hangover-ms` silence hangover closes it — but only
//! `MIN_SPEECH_FRAMES` worth of actual VAD-flagged speech at a `MIN_DENSITY`
//! share of the transmitted frames (long utterances exempt), trimmed to
//! `TAIL_KEEP_SAMPLES` of trailing silence, is what actually reaches the model,
//! carried alongside its acoustic evidence (speech-frame count, density, RMS) as an
//! `Utterance` -> `rtrb` queue (raw samples bounded with counted
//! sheds; utterances shed weakest-first past `ASR_SHED_KEEP`) -> isolated
//! Parakeet inference
//! thread (CPU, model per flags):
//! conditioning (`condition_utterance`: DC removal, 5 ms fades, peak safety;
//! hum removed earlier by `Hp`, levels by VAD-gated shared-gain `Agc`)
//! -> known-hallucination-boilerplate filter (`is_hallucination`, exact plus
//! fuzzy for long patterns) -> repetition-loop filter (`is_repetition`,
//! unique-ratio plus bigram loops). A device disconnect rebuilds the input
//! path instead of capturing deaf.
//!
//! `sttd --file <wav>` runs the same pipeline offline against a WAV file
//! instead of the microphone, for comparing tuning changes against a fixed
//! recording. `sttd --list-devices` prints available input devices as an
//! aligned table with a leading marker (`*` = system default, `>` = matches
//! `--device`/`STTD_DEVICE`).
//! `sttd --list-models` prints known Parakeet models as an aligned table
//! with a leading marker (`*` = currently selected via `--model`/`STTD_MODEL`,
//! `I` = installed on disk). `sttd --list-voices [FILTER]` does the same for
//! the piper voices `--speak` can read with (`FILTER` is a substring of the
//! voice key or its language; the catalog is ~1300 rows). `sttd --start` forks
//! microphone mode into the
//! (pid file, combined log); `--stop`/`--status` manage it via the pid
//! file. With `--speak` an accepted transcript is also spoken aloud through the
//! default output device, and the microphone is held while it is, so sttd does
//! not transcribe itself. With `--auto-unload`, while paused longer than
//! `--auto-unload-after` the model is unloaded to free VRAM and
//! reloaded eagerly on resume.
//! `--autopause` re-pauses after each emitted transcript (single-shot
//! resume: every `SIGUSR2`/`SIGHUP` arms one sentence).
//! On the session bus (unless `--no-dbus`, and never in `--file` mode) one name,
//! `org.sttd.Daemon`, carries three interfaces: pause at `/org/sttd/Daemon`, the
//! live word list at `/org/sttd/Dictionary`, and the translation step at
//! `/org/sttd/Translation` — where `Enable`, `Disable` and `SetTargetLanguage`
//! turn the step on, off or into another language while the run continues.
//! Every tunable is a CLI flag with `STTD_*` env
//! fallback (see `--help`); CLI wins over env.

use anyhow::Result;
use clap::Parser;

mod agc;
mod asr;
mod capture;
mod clean;
mod cli;
mod clock;
mod config;
mod control;
mod daemon;
mod dbus;
mod decode;
mod denoise;
mod device;
mod dictionary;
mod download;
mod dsp;
mod emit;
mod frame;
mod fsutil;
mod gate;
mod hp;
mod language;
mod mel;
mod metrics;
mod model_store;
mod noise;
mod parakeet;
mod parakeet_core;
mod piper;
mod run;
mod signal_path;
mod sound;
mod speech;
mod translate;
mod translator_store;
mod utterance;
mod vad;
mod voice_store;
mod wav;
mod whisper;
mod whisper_core;

use cli::Cli;
use run::{
    print_available_models, print_available_voices, run_file_mode, run_microphone_mode,
    run_paste_last,
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();

    if cli.stop {
        return daemon::run_stop(&cli.pid_file_path());
    }
    if cli.status {
        std::process::exit(daemon::run_status(&cli.pid_file_path()));
    }
    if cli.list_devices {
        return capture::print_input_devices(cli.device.as_deref());
    }
    if cli.list_models {
        print_available_models(&cli);
        return Ok(());
    }
    if let Some(filter) = cli.list_voices.as_deref() {
        return print_available_voices(&cli, filter);
    }
    if cli.paste_last {
        return run_paste_last(&cli);
    }
    if let Some(path) = cli.file.clone() {
        return run_file_mode(&path, &cli);
    }
    if cli.start {
        daemon::daemonize_start(&cli.pid_file_path(), &cli.log_file_path())?;
    }

    run_microphone_mode(&cli)
}

/// Initialize `env_logger` with timestamps, defaulting to `info` when
/// `RUST_LOG` is unset. Must run before any `log::*` call.
fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
}
