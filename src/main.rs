//! Real-time Speech-to-Text pipeline.
//!
//! Data flow:
//! `cpal` (hw rate, stereo) -> downmix mono -> per ~10 ms raw chunk
//! (`FrameProcessor`: `ResamplerFft` behind a 10 ms framing adapter, see
//! `FftStream`). By default (`--denoiser none`) a single `resampler`
//! hop (-> 16 kHz) feeds both branches below, each completed frame serving
//! both roles; with `--denoiser deepfilter` the VAD branch instead takes
//! a 48 kHz detour:
//!   - VAD branch:     `resampler` (-> 16 kHz, shared hop) -> `earshot`,
//!       or `resampler` (-> 48 kHz) -> denoise (`deepfilter`, see `denoise`)
//!       -> `resampler` (-> 16 kHz) -> `earshot`
//!   - whisper branch: `resampler` (-> 16 kHz, direct hop, raw) -> utterance buffer
//! An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive speech
//! frames (a single VAD false positive no longer opens one), with ~300 ms of
//! pre-roll prepended at that point so quiet word starts aren't clipped.
//! Once open, every frame is appended regardless of the VAD verdict (speech
//! or not) until `SILENCE_FRAMES_THRESHOLD` closes it — but only
//! `MIN_SPEECH_FRAMES` worth of actual VAD-flagged speech at a `MIN_DENSITY`
//! share of the transmitted frames (long utterances exempt), trimmed to
//! `TAIL_KEEP_SAMPLES` of trailing silence, is what actually reaches whisper,
//! carried alongside its acoustic evidence (speech-frame count, density, RMS) as an
//! `Utterance` -> `crossbeam-channel` queue -> isolated `whisper-rs` inference
//! thread (GPU/CPU and model per flags): conditioning (`condition_utterance`:
//! DC removal, 60 Hz high-pass, 5 ms fades, peak safety) -> pre-inference
//! gain toward `--target-rms`
//! (`apply_gain`, quiet-but-real speech only, never attenuates) -> average
//! *text-token* probability gate (`average_text_confidence` excludes the
//! timestamp tokens every whisper.cpp segment carries; whisper.cpp's own
//! `no_speech_thold` being a no-op) at a floor `confidence_floor` relaxes when
//! the acoustic evidence already vouches for real speech ->
//! known-hallucination-boilerplate filter (`is_hallucination`).
//!
//! `sttd --file <wav>` runs the same pipeline offline against a WAV file
//! instead of the microphone, for comparing tuning changes against a fixed
//! recording. `sttd --list-devices` prints available input devices as an
//! aligned table with a leading marker (`*` = system default, `>` = matches
//! `--device`/`STTD_DEVICE`).
//! `sttd --list-models` prints known whisper models as an aligned table
//! with a leading marker (`*` = currently selected via `--model`/`STTD_MODEL`,
//! `I` = installed on disk). `sttd --start` forks microphone mode into the
//! (pid file, combined log); `--stop`/`--status` manage it via the pid
//! file. With `--auto-unload`, while paused longer than
//! `--auto-unload-after` the whisper model is unloaded to free VRAM and
//! reloaded eagerly on resume.
//! `--autopause` re-pauses after each emitted transcript (single-shot
//! resume: every `SIGUSR2`/`SIGHUP` arms one sentence).
//! Every tunable is a CLI flag with `STTD_*` env
//! fallback (see `--help`); CLI wins over env. `--model` additionally honors
//! the legacy `WHISPER_MODEL_PATH` when `STTD_MODEL` is unset.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, StreamTrait};
use crossbeam_channel::{unbounded, Receiver, Sender};
use signal_hook::consts::{SIGHUP, SIGTERM, SIGUSR1, SIGUSR2};
use signal_hook::iterator::Signals;

mod cli;
use cli::Cli;
mod config;
use config::{model_file_for_spec, DEFAULT_MODELS_DIR, MODEL_CATALOG};
mod capture;
use capture::*;
mod daemon;
mod denoise;
mod dsp;
mod emit;
use emit::SystemEmitter;
mod frame;
mod gate;
mod noise;
mod signal;
mod sound;
use frame::*;
mod utterance;
use utterance::*;
mod vad;
mod wav;
use wav::*;
mod weights;
use weights::*;
mod whisper;
use whisper::*;

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
        return print_input_devices(cli.device.as_deref());
    }
    if cli.list_models {
        print_available_models(&cli);
        return Ok(());
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
/// `RUST_LOG` is unset. Must run before any `log::*` call. Also routes
/// whisper.cpp/ggml output through the `log` crate instead of raw stderr.
fn init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
    whisper_rs::install_whisper_log_trampoline();
}

/// Print known whisper models (`MODEL_CATALOG`) as an aligned table with a
/// two-character leading marker: `*` = currently selected (effective
/// `--model` after CLI > `STTD_MODEL` > legacy `WHISPER_MODEL_PATH` >
/// compiled-in default), `I` = installed (`models/<filename>` exists on
/// disk). A row can carry both (`*I`).
fn print_available_models(cli: &Cli) {
    let input = cli.model_input();
    let selected = selected_spec_for_input(&input);
    let installed: Vec<bool> = MODEL_CATALOG
        .iter()
        .map(|(_, file, _)| Path::new(DEFAULT_MODELS_DIR).join(file).exists())
        .collect();
    print!("{}", format_available_models(selected, &installed));
    if selected.is_none() {
        let trimmed = input.trim();
        if !trimmed.is_empty() {
            println!("selected: {trimmed} (not in catalog)");
        }
    }
}

/// Map a `--model` input value (spec or path, see `resolve_model_path`) to
/// the catalog spec it selects, if any. Exact spec matches win; otherwise
/// the path's basename is compared against catalog filenames
/// (`models/custom.bin` matching `ggml-*.bin` selects that row, a truly
/// custom file selects nothing).
fn selected_spec_for_input(input: &str) -> Option<&'static str> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    if model_file_for_spec(trimmed).is_some() {
        return MODEL_CATALOG
            .iter()
            .find(|(s, _, _)| *s == trimmed)
            .map(|(s, _, _)| *s);
    }
    let basename = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    MODEL_CATALOG
        .iter()
        .find(|(_, f, _)| *f == basename)
        .map(|(s, _, _)| *s)
}

/// Render the `--list-models` table: header plus one row per catalog entry,
/// `SPEC`/`FILE` left-aligned, `SIZE` right-aligned, leading `*`/`I` marker.
fn format_available_models(selected: Option<&str>, installed: &[bool]) -> String {
    let spec_w = MODEL_CATALOG
        .iter()
        .map(|(s, _, _)| s.len())
        .max()
        .unwrap_or(0)
        .max("SPEC".len());
    let file_w = MODEL_CATALOG
        .iter()
        .map(|(_, f, _)| f.len())
        .max()
        .unwrap_or(0)
        .max("FILE".len());
    let size_w = MODEL_CATALOG
        .iter()
        .map(|(_, _, s)| s.len())
        .max()
        .unwrap_or(0)
        .max("SIZE".len());
    let mut out = String::new();
    out.push_str(&format!(
        "   {0:<1$}  {2:<3$}  {4:>5$}\n",
        "SPEC", spec_w, "FILE", file_w, "SIZE", size_w
    ));
    for (i, (spec, file, size)) in MODEL_CATALOG.iter().enumerate() {
        let is_selected = selected == Some(*spec);
        let is_installed = installed.get(i).copied().unwrap_or(false);
        let marker = match (is_selected, is_installed) {
            (true, true) => "*I",
            (true, false) => "* ",
            (false, true) => "I ",
            (false, false) => "  ",
        };
        out.push_str(&format!(
            "{marker} {0:<1$}  {2:<3$}  {4:>5$}\n",
            spec, spec_w, file, file_w, size, size_w
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Microphone mode
// ---------------------------------------------------------------------------

fn run_microphone_mode(cli: &Cli) -> Result<()> {
    // Cheap, fallible audio setup first (device pick, hw config) — *before*
    // the heavy model resolve/download and GPU load. Spawning the whisper
    // worker starts an async CUDA context init on its own thread; if any of
    // this setup then failed via `?`, the process would tear down mid-init
    // and race whisper.cpp's CUDA cleanup (observed: "CUDA error: driver
    // shutting down" abort on e.g. a mistyped --device). Resolving (and
    // downloading, ~800 MiB, minutes) before `play()` also keeps `raw_rx`
    // from growing unbounded while the main loop hasn't started draining
    // yet. GPU load itself still happens in the background after capture
    // starts, so audio queues only briefly in `raw_rx` while it loads.
    let host = cpal::default_host();
    let device = select_input_device(&host, cli.device.as_deref())?;
    log::info!(
        "input device: {}",
        device.name().unwrap_or_else(|_| "<unknown>".into())
    );

    let supported = device
        .default_input_config()
        .context("failed to query default input config")?;
    let hw_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    log::info!(
        "hw config: {} Hz, {} ch, {:?}",
        hw_rate,
        channels,
        sample_format
    );

    // Heavy but pre-capture: resolve/download the model before the stream
    // starts so the callback can't outrun the drain loop.
    let model_path = resolve_model_input(cli)?;
    log::info!("model: {}", model_path.display());

    let mut accumulator: Vec<f32> = Vec::with_capacity(hw_rate as usize * 2);

    // Raw mono captures (hw rate, f32 in [-1, 1]) from the cpal callback.
    // Unbounded + `send` (never blocks) keeps callback lock time minimal.
    let (raw_tx, raw_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

    // `FrameProcessor` owns all resampling (a shared hw -> 16k hop by
    // default, or hw -> 48k -> denoise -> 48k -> 16k for the VAD branch plus
    // hw -> 16k for whisper with `--denoiser deepfilter`); building it is
    // fallible setup like everything above (including DeepFilter init), so
    // it happens here too — before capture starts.
    let (whisper_tx, whisper_rx): (Sender<Utterance>, Receiver<Utterance>) = unbounded();
    let mut fp = FrameProcessor::new(
        hw_rate,
        whisper_tx,
        cli.denoiser,
        cli.vad_threshold,
        cli.vad_adapt,
        gate_config(cli),
    )?;
    let input_frames = fp.required_input_frames();

    // ---- build + start the cpal input stream (callback does NOT unwrap) ---
    let running = Arc::new(AtomicBool::new(true));
    let paused = Arc::new(AtomicBool::new(cli.start_paused));
    let stream = build_input_stream(&device, &stream_config, channels, sample_format, raw_tx)?;
    stream.play().context("failed to start input stream")?;
    if cli.start_paused {
        log::info!("starting paused (SIGUSR2/SIGHUP to resume)");
    } else {
        log::info!("listening... Ctrl-C to stop");
    }
    log::info!("pause: SIGUSR1, resume: SIGUSR2, toggle: SIGHUP");

    // Ctrl-C sets the flag; the loop below flushes trailing speech first.
    {
        let running = Arc::clone(&running);
        if let Err(e) = install_ctrlc_handler(move || running.store(false, Ordering::Release)) {
            log::warn!("{e:?} (Ctrl-C will terminate immediately)");
        }
    }

    // USR1/USR2/HUP drive the pause flag, TERM requests a graceful shutdown
    // (same flag as Ctrl-C, so `--stop`/`kill` flushes trailing speech); a
    // failed install isn't fatal, pause/resume via signals is then simply
    // unavailable.
    {
        let paused = Arc::clone(&paused);
        let running = Arc::clone(&running);
        let sound_enabled = cli.sound;
        if let Err(e) = install_pause_handlers(paused, running, sound_enabled) {
            log::warn!("{e:?} (signal pause/resume unavailable)");
        }
    }

    // ---- now the model load: onto the GPU/CPU (per `--gpu`) on its own thread.
    // Capture is already running above and the drain loop below keeps
    // `raw_rx` bounded while it loads.
    let (whisper_handle, whisper_ready) = spawn_whisper_worker(
        whisper_rx,
        model_path,
        whisper_options(cli),
        make_emitter(cli)?,
        Arc::clone(&paused),
        cli.auto_unload,
        cli.auto_unload_after,
        cli.sound,
        cli.autopause,
    );
    let mut whisper_ready_ok = false;

    // ---- main processing loop (NOT the audio callback) --------------------
    let mut was_paused = paused.load(Ordering::Acquire);
    if was_paused {
        // Start-paused: drop any callback audio that raced stream start.
        fp.discard_pending();
    }
    while running.load(Ordering::Acquire) {
        // Fail fast if the whisper worker died during init instead of
        // capturing deaf: ready Err / disconnect, or early thread exit.
        match whisper_ready.try_recv() {
            Ok(Ok(())) => whisper_ready_ok = true,
            Ok(Err(e)) => anyhow::bail!("whisper worker failed to start: {e}"),
            Err(std::sync::mpsc::TryRecvError::Disconnected) if !whisper_ready_ok => {
                anyhow::bail!("whisper worker exited before signaling ready")
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                anyhow::bail!("whisper worker died")
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if whisper_handle.is_finished() && !whisper_ready_ok {
                    anyhow::bail!("whisper worker exited before signaling ready");
                }
            }
        }
        // While paused, no utterance may open, but the background keeps
        // teaching the noise floor: drain the callback into the accumulator
        // and observe (resample + RMS -> floor, no earshot, no buffering).
        // Pre-pause speech state was already dropped (`discard_speech_keep_warm`,
        // idempotent), so resume starts clean with no pre-pause pre-roll
        // while the threshold stays adapted to the room.
        let now_paused = paused.load(Ordering::Acquire);
        if now_paused && !was_paused {
            // Pause entry edge: full clear (resamplers + detector + utterance,
            // noise floor preserved) so pre-pause partials can't leak into the
            // background estimate or the first post-resume utterance.
            fp.discard_pending();
        }
        was_paused = now_paused;
        if now_paused {
            fp.discard_speech_keep_warm();
            while let Ok(chunk) = raw_rx.try_recv() {
                accumulator.extend_from_slice(&chunk);
            }
            while accumulator.len() >= input_frames {
                fp.observe_pause(&accumulator[..input_frames]);
                accumulator.drain(..input_frames);
            }
            // Keep only a partial chunk for continuity; full chunks are always
            // consumed above, so this just caps a callback burst.
            if accumulator.len() >= input_frames {
                accumulator.clear();
            }
            thread::sleep(Duration::from_millis(50));
            continue;
        }
        // Drain everything the callback delivered since last iteration.
        let mut got_audio = false;
        while let Ok(chunk) = raw_rx.try_recv() {
            got_audio = true;
            accumulator.extend_from_slice(&chunk);
        }

        // Raw hw-rate chunks -> FrameProcessor (denoise/VAD + whisper
        // branches, all resampling included).
        while accumulator.len() >= input_frames {
            fp.process_frame(&accumulator[..input_frames]);
            accumulator.drain(..input_frames);
        }

        if !got_audio && accumulator.len() < input_frames {
            thread::sleep(Duration::from_millis(2));
        }
    }

    // Flush trailing speech, then wait for the whisper thread to actually
    // finish transcribing it (and any earlier backlog) before the process
    // exits — otherwise the last utterance can be silently dropped, and an
    // in-flight CUDA op can race process teardown the same way an early `?`
    // above used to.
    fp.flush_trailing_speech();
    drop(fp);
    whisper_handle
        .join()
        .map_err(|_| anyhow::anyhow!("whisper worker panicked"))?;
    log::info!("stopped");
    Ok(())
}

/// Install the Ctrl-C handler that stops the processing loop. A failed
/// install isn't fatal: Ctrl-C then terminates the process immediately,
/// losing only the trailing-speech flush.
fn install_ctrlc_handler(cb: impl Fn() + Send + 'static) -> Result<()> {
    ctrlc::set_handler(cb).context("failed to install Ctrl-C handler")
}

/// Install the pause/resume/shutdown signal handlers: `SIGUSR1` pauses,
/// `SIGUSR2` resumes, `SIGHUP` toggles, `SIGTERM` requests a graceful
/// shutdown (same `running` flag as Ctrl-C, so `--stop`/`kill` flushes
/// trailing speech instead of dying mid-CUDA-op). The handler thread only
/// flips shared flags; the main loop does the actual discarding, so the
/// transition is idempotent (pause-while-paused / resume-while-running are
/// no-ops, double TERM is one shutdown). Audible ack (`sound` module) fires
/// alongside the flag flip so a daemon user hears the transition; `ready`
/// itself is signaled separately by the whisper thread once transcription
/// can actually resume.
fn install_pause_handlers(
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    sound_enabled: bool,
) -> Result<()> {
    let mut signals = Signals::new([SIGUSR1, SIGUSR2, SIGHUP, SIGTERM])
        .context("failed to register pause/resume signals")?;
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGUSR1 => {
                    paused.store(true, Ordering::Release);
                    log::info!("paused (SIGUSR1)");
                    if sound_enabled {
                        crate::sound::play_paused();
                    }
                }
                SIGUSR2 => {
                    paused.store(false, Ordering::Release);
                    log::info!("resumed (SIGUSR2)");
                    if sound_enabled {
                        crate::sound::play_resumed();
                    }
                }
                SIGHUP => {
                    let was_paused = paused.fetch_xor(true, Ordering::AcqRel);
                    if was_paused {
                        log::info!("resumed (SIGHUP toggle)");
                        if sound_enabled {
                            crate::sound::play_resumed();
                        }
                    } else {
                        log::info!("paused (SIGHUP toggle)");
                        if sound_enabled {
                            crate::sound::play_paused();
                        }
                    }
                }
                SIGTERM => {
                    log::info!("shutting down (SIGTERM)");
                    running.store(false, Ordering::Release);
                }
                _ => {}
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Offline (file) mode
// ---------------------------------------------------------------------------

fn run_file_mode(path: &Path, cli: &Cli) -> Result<()> {
    // Same ordering rationale as run_microphone_mode: do the cheap, fallible
    // setup (read + validate the WAV, resolve the model path) before the
    // heavy work (denoiser init, GPU load), so a bad file/spec fails fast
    // instead of racing whisper.cpp's CUDA teardown.
    let wav = read_wav_mono_f32(path)?;
    log::info!(
        "file: {} ({} Hz, {} samples, {:.1}s)",
        path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f64 / wav.sample_rate as f64
    );

    let model_path = resolve_model_input(cli)?;
    log::info!("model: {}", model_path.display());

    let (whisper_tx, whisper_rx): (Sender<Utterance>, Receiver<Utterance>) = unbounded();
    let mut fp = FrameProcessor::new(
        wav.sample_rate,
        whisper_tx.clone(),
        cli.denoiser,
        cli.vad_threshold,
        cli.vad_adapt,
        gate_config(cli),
    )?;
    let input_frames = fp.required_input_frames();

    let (whisper_handle, whisper_ready) = spawn_whisper_worker(
        whisper_rx,
        model_path,
        whisper_options(cli),
        make_emitter(cli)?,
        // File mode has no pause signalling; a permanently-clear flag plus
        // disabled auto-unload can never trigger an unload mid-file. Sounds
        // stay off offline: no pause/resume transitions exist to cue.
        // Autopause is forced off for the same reason: file input must never
        // flip the (unread) pause flag into an unload mid-file.
        Arc::new(AtomicBool::new(false)),
        false,
        cli.auto_unload_after,
        false,
        false,
    );
    // Fail fast on model-load/dump-dir errors before processing the file:
    // no realtime capture to keep bounded here, so blocking wait is fine.
    match whisper_ready.recv_timeout(Duration::from_secs(600)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => anyhow::bail!("whisper worker failed to start: {e}"),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("timed out waiting for whisper worker to start")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("whisper worker exited before signaling ready")
        }
    }

    let mut offset = 0usize;
    while offset + input_frames <= wav.samples.len() {
        fp.process_frame(&wav.samples[offset..offset + input_frames]);
        offset += input_frames;
    }
    // A trailing partial chunk (< input_frames samples, at most ~30 ms) is
    // dropped; fine for offline comparison runs.
    fp.flush_trailing_speech();

    drop(fp);
    drop(whisper_tx);
    whisper_handle
        .join()
        .map_err(|_| anyhow::anyhow!("whisper worker panicked"))?;
    Ok(())
}

/// Resolve `--model` (spec or path) to a guaranteed-present file path:
/// spec mapping first (`resolve_model_path`), then download-if-missing
/// (`ensure_available`).
fn resolve_model_input(cli: &Cli) -> Result<std::path::PathBuf> {
    let input = cli.model_input();
    let path = resolve_model_path(&input)?;
    ensure_available(&path)
}

fn whisper_options(cli: &Cli) -> WhisperOptions {
    WhisperOptions {
        model_spec: cli.model_input(),
        prompt: cli.prompt.clone(),
        language: cli.language,
        gate: gate_config(cli),
        target_rms: cli.target_rms,
        debug_tokens: cli.debug_tokens,
        beam_size: cli.beam_size,
        n_threads: resolve_n_threads(cli.threads, auto_threads()),
        use_gpu: cli.gpu,
    }
}

/// Emission policy (dump dir, clipboard, paste, stdout), resolved once from
/// `Cli`. Runs on the main thread before the worker spawns, so a bad dump
/// dir fails fast like the model load it precedes.
fn make_emitter(cli: &Cli) -> Result<SystemEmitter> {
    SystemEmitter::new(cli.dump_dir.clone(), cli.copy, cli.paste, cli.stdout)
}

/// Thresholds for deciding-not-to-emit, resolved once from `Cli` and shared
/// by the acoustic path (`FrameProcessor`) and the transcript path
/// (`WhisperOptions`). Keeps `Utterance` free of config, only evidence.
fn gate_config(cli: &Cli) -> gate::GateConfig {
    gate::GateConfig::new(
        cli.min_rms,
        cli.min_density,
        cli.min_conf,
        cli.min_conf_strong,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_spec_matches_spec_exactly() {
        assert_eq!(selected_spec_for_input("small"), Some("small"));
        assert_eq!(
            selected_spec_for_input("large-v3-turbo-q8_0"),
            Some("large-v3-turbo-q8_0")
        );
        assert_eq!(selected_spec_for_input("  small  "), Some("small"));
    }

    #[test]
    fn selected_spec_matches_path_basename() {
        assert_eq!(
            selected_spec_for_input("models/ggml-small.bin"),
            Some("small")
        );
        assert_eq!(
            selected_spec_for_input("/abs/models/ggml-base.bin"),
            Some("base")
        );
        assert_eq!(selected_spec_for_input("ggml-tiny.bin"), Some("tiny"));
        assert_eq!(
            selected_spec_for_input(r"models\ggml-small.bin"),
            Some("small")
        );
    }

    #[test]
    fn selected_spec_rejects_custom_and_empty() {
        assert_eq!(selected_spec_for_input("models/custom.bin"), None);
        assert_eq!(selected_spec_for_input("bogus"), None);
        assert_eq!(selected_spec_for_input(""), None);
        assert_eq!(selected_spec_for_input("   "), None);
    }

    #[test]
    fn models_table_marks_selected_and_installed() {
        let installed = vec![false; MODEL_CATALOG.len()];
        let table = format_available_models(Some("small"), &installed);
        let small_row = table
            .lines()
            .find(|l| l.contains("ggml-small.bin"))
            .expect("small row must exist");
        assert!(small_row.starts_with("* "), "row={small_row:?}");
        assert!(!small_row.contains("installed"), "row={small_row:?}");
        assert!(!small_row.contains("default"), "row={small_row:?}");
        // No other row carries `*`.
        for line in table.lines().skip(1) {
            if line.contains("ggml-small.bin") {
                continue;
            }
            assert!(!line.starts_with('*'), "line={line:?}");
        }
    }

    #[test]
    fn models_table_combines_selected_and_installed() {
        let mut installed = vec![false; MODEL_CATALOG.len()];
        let idx = MODEL_CATALOG
            .iter()
            .position(|(s, _, _)| *s == "base")
            .expect("base must exist");
        installed[idx] = true;
        let table = format_available_models(Some("base"), &installed);
        let base_row = table
            .lines()
            .find(|l| l.contains("ggml-base.bin"))
            .expect("base row must exist");
        assert!(base_row.starts_with("*I"), "row={base_row:?}");
        let other_idx = MODEL_CATALOG
            .iter()
            .position(|(s, _, _)| *s == "tiny")
            .expect("tiny must exist");
        assert!(!installed[other_idx]);
        let tiny_row = table
            .lines()
            .find(|l| l.contains("ggml-tiny.bin ") || l.ends_with("ggml-tiny.bin"))
            .expect("tiny row must exist");
        assert!(tiny_row.starts_with("  "), "row={tiny_row:?}");
    }

    #[test]
    fn models_table_aligns_columns() {
        let installed = vec![false; MODEL_CATALOG.len()];
        let table = format_available_models(Some("tiny"), &installed);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), MODEL_CATALOG.len() + 1);
        assert!(lines[0].contains("SPEC"));
        assert!(lines[0].contains("FILE"));
        assert!(lines[0].contains("SIZE"));
        // Fixed-width columns: every line the same display width, and the
        // SIZE field ends at the same offset (right-aligned).
        let widths: Vec<usize> = lines.iter().map(|l| l.len()).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "widths={widths:?}");
        let size_end = lines[0].len();
        for line in &lines {
            assert_eq!(line.len(), size_end);
            assert_eq!(line[..2].chars().count(), 2);
        }
    }
}
