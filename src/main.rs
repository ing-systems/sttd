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
//! thread (CUDA, large-v3-turbo-q8_0): conditioning (`condition_utterance`:
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
//! recording. `sttd --list-devices` prints available input devices.
//! `sttd --list-models` prints known whisper models with installed/default
//! markers. Every tunable is a CLI flag with `STTD_*` env
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
use crossbeam_channel::{Receiver, Sender, unbounded};
use signal_hook::consts::{SIGHUP, SIGUSR1, SIGUSR2};
use signal_hook::iterator::Signals;

mod cli;
use cli::Cli;
mod config;
use config::{DEFAULT_MODELS_DIR, DEFAULT_MODEL_SPEC, MODEL_CATALOG};
mod capture;
use capture::*;
mod denoise;
mod dsp;
mod emit;
use emit::SystemEmitter;
mod gate;
mod signal;
mod frame;
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
    let cli = Cli::parse();

    if cli.list_devices {
        return print_input_devices();
    }
    if cli.list_models {
        print_available_models();
        return Ok(());
    }
    if let Some(path) = cli.file.clone() {
        return run_file_mode(&path, &cli);
    }

    run_microphone_mode(&cli)
}

/// Print known whisper models (`MODEL_CATALOG`) with installed/default
/// markers to stdout: `spec filename remote-size [installed] [*default*]`.
/// "installed" means `models/<filename>` exists on disk.
fn print_available_models() {
    for (spec, file, size) in MODEL_CATALOG {
        let installed =
            Path::new(DEFAULT_MODELS_DIR).join(file).exists();
        let mut markers = String::new();
        if installed {
            markers.push_str(" installed");
        }
        if *spec == DEFAULT_MODEL_SPEC {
            markers.push_str(" *default*");
        }
        println!("{spec} {file} {size}{markers}");
    }
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
    eprintln!(
        "[sttd] input device: {}",
        device.name().unwrap_or_else(|_| "<unknown>".into())
    );

    let supported = device
        .default_input_config()
        .context("failed to query default input config")?;
    let hw_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    eprintln!(
        "[sttd] hw config: {} Hz, {} ch, {:?}",
        hw_rate, channels, sample_format
    );

    // Heavy but pre-capture: resolve/download the model before the stream
    // starts so the callback can't outrun the drain loop.
    let model_path = resolve_model_input(cli)?;
    eprintln!("[sttd] model: {}", model_path.display());

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
        gate_config(cli),
    )?;
    let input_frames = fp.required_input_frames();

    // ---- build + start the cpal input stream (callback does NOT unwrap) ---
    let running = Arc::new(AtomicBool::new(true));
    let paused = Arc::new(AtomicBool::new(cli.start_paused));
    let stream = build_input_stream(&device, &stream_config, channels, sample_format, raw_tx)?;
    stream.play().context("failed to start input stream")?;
    if cli.start_paused {
        eprintln!("[sttd] starting paused (SIGUSR2/SIGHUP to resume)");
    } else {
        eprintln!("[sttd] listening... Ctrl-C to stop");
    }
    eprintln!("[sttd] pause: SIGUSR1, resume: SIGUSR2, toggle: SIGHUP");

    // Ctrl-C sets the flag; the loop below flushes trailing speech first.
    {
        let running = Arc::clone(&running);
        if let Err(e) = install_ctrlc_handler(move || running.store(false, Ordering::Release)) {
            eprintln!("[sttd] warning: {e:?} (Ctrl-C will terminate immediately)");
        }
    }

    // USR1/USR2/HUP drive the pause flag; a failed install isn't fatal,
    // pause/resume via signals is then simply unavailable.
    {
        let paused = Arc::clone(&paused);
        if let Err(e) = install_pause_handlers(paused) {
            eprintln!("[sttd] warning: {e:?} (signal pause/resume unavailable)");
        }
    }

    // ---- now the GPU part: load the model onto the GPU on its own thread.
    // Capture is already running above and the drain loop below keeps
    // `raw_rx` bounded while it loads.
    let (whisper_handle, whisper_ready) = spawn_whisper_worker(
        whisper_rx,
        model_path,
        whisper_options(cli),
        make_emitter(cli)?,
    );
    let mut whisper_ready_ok = false;

    // ---- main processing loop (NOT the audio callback) --------------------
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
        // While paused, discard everything the callback delivered: no
        // audio reaches FrameProcessor, and its in-flight state was
        // already dropped, so resume starts clean with no pre-pause
        // pre-roll. Idempotent: staying paused just keeps discarding.
        if paused.load(Ordering::Acquire) {
            while raw_rx.try_recv().is_ok() {}
            accumulator.clear();
            fp.discard_pending();
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
    eprintln!("[sttd] stopped");
    Ok(())
}

/// Install the Ctrl-C handler that stops the processing loop. A failed
/// install isn't fatal: Ctrl-C then terminates the process immediately,
/// losing only the trailing-speech flush.
fn install_ctrlc_handler(cb: impl Fn() + Send + 'static) -> Result<()> {
    ctrlc::set_handler(cb).context("failed to install Ctrl-C handler")
}

/// Install the pause/resume signal handlers: `SIGUSR1` pauses, `SIGUSR2`
/// resumes, `SIGHUP` toggles. The handler thread only flips the shared
/// flag; the main loop does the actual discarding, so the transition is
/// idempotent (pause-while-paused / resume-while-running are no-ops).
fn install_pause_handlers(paused: Arc<AtomicBool>) -> Result<()> {
    let mut signals = Signals::new([SIGUSR1, SIGUSR2, SIGHUP])
        .context("failed to register pause/resume signals")?;
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGUSR1 => {
                    paused.store(true, Ordering::Release);
                    eprintln!("[sttd] paused (SIGUSR1)");
                }
                SIGUSR2 => {
                    paused.store(false, Ordering::Release);
                    eprintln!("[sttd] resumed (SIGUSR2)");
                }
                SIGHUP => {
                    let was_paused = paused.fetch_xor(true, Ordering::AcqRel);
                    if was_paused {
                        eprintln!("[sttd] resumed (SIGHUP toggle)");
                    } else {
                        eprintln!("[sttd] paused (SIGHUP toggle)");
                    }
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
    eprintln!(
        "[sttd] file: {} ({} Hz, {} samples, {:.1}s)",
        path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f64 / wav.sample_rate as f64
    );

    let model_path = resolve_model_input(cli)?;
    eprintln!("[sttd] model: {}", model_path.display());

    let (whisper_tx, whisper_rx): (Sender<Utterance>, Receiver<Utterance>) = unbounded();
    let mut fp = FrameProcessor::new(
        wav.sample_rate,
        whisper_tx.clone(),
        cli.denoiser,
        cli.vad_threshold,
        gate_config(cli),
    )?;
    let input_frames = fp.required_input_frames();

    let (whisper_handle, whisper_ready) = spawn_whisper_worker(
        whisper_rx,
        model_path,
        whisper_options(cli),
        make_emitter(cli)?,
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
        prompt: cli.prompt.clone(),
        language: cli.language,
        gate: gate_config(cli),
        target_rms: cli.target_rms,
        debug_tokens: cli.debug_tokens,
    }
}

/// Emission policy (dump dir, clipboard, paste), resolved once from `Cli`.
/// Runs on the main thread before the worker spawns, so a bad dump dir fails
/// fast like the model load it precedes.
fn make_emitter(cli: &Cli) -> Result<SystemEmitter> {
    SystemEmitter::new(cli.dump_dir.clone(), cli.copy, cli.paste)
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
