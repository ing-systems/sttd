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

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, StreamTrait};
use rtrb::{Consumer, Producer, RingBuffer};
use signal_hook::consts::{SIGHUP, SIGTERM, SIGUSR1, SIGUSR2};
use signal_hook::iterator::Signals;

mod cli;
use cli::Cli;
mod capture;
mod config;
use capture::*;
mod clean;
mod clock;
mod control;
mod daemon;
mod dbus;
mod denoise;
mod dictionary;
mod download;
mod dsp;
mod emit;
mod metrics;
use emit::{EmissionPolicy, SpeechOut, SystemEmitter};
mod agc;
mod frame;
mod fsutil;
mod gate;
mod hp;
mod language;
mod noise;
mod signal;
mod sound;
mod speech;
use frame::*;
use speech::EngineConfig;
mod utterance;
use utterance::*;
mod vad;
mod wav;
use wav::*;
mod model_store;
use model_store::{ensure_available, resolve_model, EngineKind, ModelRef, CATALOGS};
mod asr;
use asr::{spawn_asr_worker, AsrConfig};
use device::{auto_threads, resolve_n_threads};
mod decode;
mod device;
mod mel;
mod parakeet;
mod parakeet_core;
mod piper;
mod translate;
use translate::{TranslationOut, TranslationPlan, TranslationState};
mod translator_store;
mod voice_store;
mod whisper;
mod whisper_core;
use language::Language;
use whisper_core::types::DEFAULT_WINDOW_FRAMES;

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

/// Print known models — both catalogs — as one aligned table per engine, each
/// rendered by its own catalog (columns SPEC/DIR/SIZE, since both engines ship
/// as a directory of files rather than one), with a two-character leading
/// marker: `*` = currently selected (effective `--model` after CLI >
/// `STTD_MODEL` > compiled-in default), `I` = installed (weights present on
/// disk). A row can carry both (`*I`).
fn print_available_models(cli: &Cli) {
    let input = cli.model_input();

    let mut any_selected = false;
    for catalog in CATALOGS {
        let selected = catalog.entry_for_spec(&input).map(|e| e.spec);
        println!("{} models:", catalog.engine.as_str());
        print!("{}", catalog.render_table(selected));
        any_selected |= selected.is_some();
    }

    if !any_selected {
        let trimmed = input.trim();
        if !trimmed.is_empty() {
            println!("selected: {trimmed} (not in catalog)");
        }
    }
}

/// Print piper voices as an aligned table: `SPEC` / `LANG` / `QUALITY` /
/// `SIZE` / `SPK`, with the same leading marker as `--list-models` (`*` =
/// selected via `--voice`/`STTD_VOICE`, `I` = downloaded to disk).
///
/// An optional substring filters on the voice key or its language, because the
/// catalog is ~1300 rows — unfiltered it is a reference dump, not a listing.
/// The index comes from upstream's own `voices.json`, so the first call needs
/// the network; after that it is read from the cache beside the voices.
fn print_available_voices(cli: &Cli, filter: &str) -> Result<()> {
    let index = crate::voice_store::VoiceIndex::load()?;
    let rows = index.filtered(Some(filter));
    if rows.is_empty() {
        println!("no piper voice matches {filter:?}");
        return Ok(());
    }
    // Resolving is what knows where a row's two files live, and it is the same
    // path `--voice` takes, so the `I` markers cannot disagree with it.
    let refs: Vec<crate::voice_store::VoiceRef> = rows
        .iter()
        .filter_map(|row| crate::voice_store::resolve_voice(&row.key, &index).ok())
        .collect();
    println!(
        "piper voices ({} of {}), from https://huggingface.co/{}/tree/main:",
        rows.len(),
        index.filtered(None).len(),
        crate::config::PIPER_VOICES_REPO
    );
    print!(
        "{}",
        crate::voice_store::render_table(&rows, &refs, Some(cli.voice.trim()))
    );
    if rows.iter().any(|r| r.multi_speaker()) {
        println!("* multi-speaker: listed, but --speak refuses one (it has no speaker to pick)");
    }
    if refs.iter().all(|v| !v.is_installed()) {
        println!("none downloaded yet; `sttd --speak --voice <spec>` fetches one (~60 MiB)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Microphone mode
// ---------------------------------------------------------------------------

/// Seconds of raw mono samples the callback may backlog. Past that the
/// callback drops (counted) instead of growing memory without bound while
/// the model loads or the GPU lags. Sized in `open_input_path` from the hw
/// rate (`hw_rate * RAW_RING_SECS`).
pub(crate) const RAW_RING_SECS: usize = 4;

/// Drain every available sample from the `rtrb` consumer into the accumulator.
/// Main thread only; allocation (accumulator growth) is fine here — only the
/// cpal callback is real-time.
fn drain_raw_into(raw_rx: &mut Consumer<f32>, accumulator: &mut Vec<f32>) {
    while let Ok(s) = raw_rx.pop() {
        accumulator.push(s);
    }
}

/// Drain complete frames from the utterance buffer into the VAD branch and
/// the model branch, handing each closed utterance to `push` for emission.
/// Frames are consumed through a cursor and the remainder is compacted once
/// at the end, instead of a `drain` per frame (which memmoved the whole
/// remainder every time).
///
/// A frame that closed nothing may instead leave a partial snapshot of the open
/// utterance in `partials` (latest wins), and a close drops the one waiting: its
/// utterance is now a final in the queue.
fn drain_frames(
    utterance_buffer: &mut Vec<f32>,
    input_frames: usize,
    fp: &mut FrameProcessor,
    asr_tx: &mut Producer<Utterance>,
    push: fn(&mut Producer<Utterance>, Utterance),
    partials: &crate::asr::PartialSlot,
) {
    let mut consumed = 0;
    while utterance_buffer.len() - consumed >= input_frames {
        let frame = &utterance_buffer[consumed..consumed + input_frames];
        if let Some(utterance) = fp.process_frame(frame) {
            partials.clear();
            push(asr_tx, utterance);
        } else if let Some(pcm) = fp.take_partial() {
            partials.put(pcm);
        }
        consumed += input_frames;
    }
    utterance_buffer.drain(..consumed);
}

/// Throttled backpressure report state for the live capture loop. Pairs the
/// last emission time with the last shed totals so growth in the utterance
/// buffer path shows once per interval.
struct BackpressureState<'a> {
    last_queue_warn: &'a mut Instant,
    last_drop_report: &'a mut (usize, usize),
}

/// Backpressure visibility: bounded rings shed under load, so report depths
/// plus shed totals for the VAD branch and the model branch pairing.
/// Throttled to one emission per 5 s. Preserves Ordering, timing, and log
/// text exactly as in the inline loop body.
fn report_backpressure(
    state: BackpressureState<'_>,
    raw_rx: &Consumer<f32>,
    asr_tx: &Producer<Utterance>,
    fp: &FrameProcessor,
    telemetry: &StreamTelemetry,
) {
    if state.last_queue_warn.elapsed() >= Duration::from_secs(5) {
        *state.last_queue_warn = Instant::now();
        let raw_depth = raw_rx.slots();
        let asr_depth = producer_occupied(asr_tx, crate::asr::ASR_QUEUE_CAP);
        let (signal_vad, signal_model) = fp.signal_pending_depths();
        let raw_dropped = telemetry.dropped_chunks.load(Ordering::Relaxed);
        let stream_errors = telemetry.errors.load(Ordering::Relaxed);
        let drops = (raw_dropped, stream_errors);
        let drops_grew = drops != *state.last_drop_report;
        *state.last_drop_report = drops;
        if raw_depth > 64
            || asr_depth > 4
            || signal_vad > crate::signal::SIGNAL_PENDING_WARN
            || signal_model > crate::signal::SIGNAL_PENDING_WARN
            || drops_grew
        {
            log::warn!(
                "queues (raw_rx={raw_depth}, asr={asr_depth}, signal_vad={signal_vad}, signal_model={signal_model}, dropped_chunks={raw_dropped}, stream_errors={stream_errors}): capture outruns resampling, branches diverged, or the model outruns real time"
            );
        }
    }
}

/// Discard capture around sound-cue playback so cue tones are never
/// transcribed. Full pipeline hold, drain, wait out the longest cue
/// (`sound::CUE_QUIET_MS`), drain again. Skipped when an utterance is already
/// open — in-flight speech wins over a cue, never the reverse.
///
/// Callers: resume edge + first worker-ready (both unpaused + sound only).
/// Steady-state never sleeps. Reload-while-unpaused is not covered: the
/// worker's post-reload jingle has no signal the main loop can key on, so a
/// 300 ms tone can still reach an otherwise-idle pipeline there (likely
/// gate-dropped as non-speech-shaped audio).
fn cue_guard(fp: &mut FrameProcessor, raw_rx: &mut Consumer<f32>, accumulator: &mut Vec<f32>) {
    if fp.has_open_utterance() {
        return;
    }
    fp.hold_pipeline();
    drain_raw_into(raw_rx, accumulator);
    accumulator.clear();
    thread::sleep(Duration::from_millis(crate::sound::CUE_QUIET_MS));
    drain_raw_into(raw_rx, accumulator);
    accumulator.clear();
}

/// What a full ASR ring means, i.e. the one policy that differs between the
/// two input paths.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backpressure {
    /// Live capture: never block the processing loop, drop the newest
    /// utterance with a warn. A dead worker (`is_abandoned`) is also just a
    /// warn. In practice the worker-side weakest-first shed keeps the
    /// backlog small so this rarely fires.
    Drop,
    /// Offline file mode (`shed_stale = false`) promises lossless decoding,
    /// but resampling runs at ~100x realtime while inference runs at roughly
    /// realtime, so the producer routinely outruns `ASR_QUEUE_CAP` (16) on any
    /// recording longer than a few tens of seconds. Dropping there would
    /// silently discard most of a long file, so wait for the worker to make
    /// room instead: safe here because file mode has no realtime deadline to
    /// protect.
    Wait,
}

/// Push a closed utterance into the bounded ASR ring under `policy`
/// (`Backpressure`). Returns once the utterance is queued or the worker is
/// gone (a dead worker only warns, in both policies).
fn push_utterance(asr_tx: &mut Producer<Utterance>, utterance: Utterance, policy: Backpressure) {
    if asr_tx.is_abandoned() {
        log::warn!("asr worker gone, dropping utterance");
        return;
    }
    let mut pending = utterance;
    loop {
        match asr_tx.push(pending) {
            Ok(()) => return,
            Err(rtrb::PushError::Full(back)) => match policy {
                Backpressure::Drop => {
                    log::warn!("asr queue full, dropping newest utterance");
                    return;
                }
                Backpressure::Wait => {
                    if asr_tx.is_abandoned() {
                        log::warn!("asr worker gone, dropping utterance");
                        return;
                    }
                    pending = back;
                    thread::sleep(Duration::from_millis(1));
                }
            },
        }
    }
}

/// Microphone-mode push, as a plain function pointer for `drain_frames`.
/// File mode calls `push_utterance` with `Backpressure::Wait` directly.
fn push_live_utterance(asr_tx: &mut Producer<Utterance>, utterance: Utterance) {
    push_utterance(asr_tx, utterance, Backpressure::Drop);
}

/// Occupied slots on the producer side (`Producer::slots` reports free).
fn producer_occupied<T>(tx: &Producer<T>, cap: usize) -> usize {
    cap.saturating_sub(tx.slots())
}

/// Resolve/download the model and create the bounded utterance ring: the
/// pre-capture work both input paths share, so a bad `--model` or a bad
/// `--dump-dir` fails fast before any capture or heavy setup.
///
/// The ring is bounded (`ASR_QUEUE_CAP`, each utterance seconds of audio) and
/// the ASR worker sheds weakest-first past `ASR_SHED_KEEP`, so live speech
/// stays fresh instead of lagging minutes behind. It survives a device
/// rebuild: the main loop owns the producer, the worker the consumer, and
/// `FrameProcessor` just returns closed utterances.
fn prepare_model(cli: &Cli) -> Result<(ModelRef, Producer<Utterance>, Consumer<Utterance>)> {
    let model = resolve_model_input(cli)?;
    log::info!("model: {model}");
    validate_model_flags(cli, &model)?;
    let (tx, rx) = RingBuffer::<Utterance>::new(crate::asr::ASR_QUEUE_CAP);
    Ok((model, tx, rx))
}

/// Shared shutdown tail for both input paths, in this order: drop the frame
/// processor, then the utterance producer (so the worker sees `is_abandoned`
/// and exits), then join it. Dropping the producer before the join is what
/// keeps Ctrl-C/`--stop` from hanging forever; the join is what flushes the
/// trailing utterance the worker is still transcribing.
fn finish_run(
    fp: FrameProcessor,
    asr_tx: Producer<Utterance>,
    asr_handle: thread::JoinHandle<()>,
) -> Result<()> {
    drop(fp);
    drop(asr_tx);
    asr_handle
        .join()
        .map_err(|_| anyhow::anyhow!("asr worker panicked"))
}

fn run_microphone_mode(cli: &Cli) -> Result<()> {
    // Cheap, fallible audio setup first (device pick, hw config) — *before*
    // the heavy model resolve/download and load. Resolving (and
    // downloading, ~640 MiB, minutes) before `play()` also keeps the raw ring
    // from shedding while the main loop hasn't started draining
    // yet. GPU load itself still happens in the background after capture
    // starts, so only the first seconds of audio are retained while it loads
    // (the default `--start-paused` discards even that).
    let host = cpal::default_host();

    // Heavy but pre-capture: resolve/download the model before the stream
    // starts so the callback can't outrun the drain loop.
    let (model, mut asr_tx, asr_rx) = prepare_model(cli)?;
    match model.kind {
        EngineKind::Parakeet => log::info!(
            "streaming: partial transcripts in the log while you speak (Parakeet already sizes its window to the audio)"
        ),
        EngineKind::Whisper => log::info!(
            "streaming: partial transcripts in the log while you speak, encoder window follows the audio"
        ),
    }
    let acoustic = AcousticConfig::from_cli(cli);
    let telemetry = StreamTelemetry::default();

    // Raw mono samples (hw rate, f32 in [-1, 1]) from the cpal callback via a
    // realtime-safe `rtrb` sample ring (no allocation in the callback;
    // overflow sheds counted samples instead of growing memory without
    // bound). Created inside `open_input_path` once the hw rate is known.
    //
    // First open fails fast (bad --device/--channel), like before; later
    // disconnects rebuild inside the loop (see `device_gone`). The raw ring
    // is recreated per open (capacity follows the hw rate).
    let (mut hw_rate, mut stream, mut fp, mut raw_rx) =
        open_input_path(&host, cli, &acoustic, &telemetry)?;
    let mut input_frames = fp.required_input_frames();
    let mut accumulator: Vec<f32> = Vec::with_capacity(hw_rate as usize * 2);

    // ---- shared control state (D-Bus, signals, pipeline, ASR worker) ------
    // `PauseState` and `DictionaryState` are separate, one per bus interface.
    // Each is a cheap clone, so the bus thread, the signal handlers, the main
    // loop and the worker all observe the same objects.
    // `LiveDictionary::new` attaches the ASR booster hook and hands back the
    // hooked clone for `dbus::spawn`.
    let pause_state = crate::control::PauseState::new(cli.start_paused);
    let dict_state = crate::control::DictionaryState::new(
        load_dictionary_or_warn(cli),
        cli.dictionary.clone().unwrap_or_default(),
    );
    let (hooked_dict, live_dict) = crate::asr::LiveDictionary::new(dict_state);
    let paused = pause_state.paused_flag();
    // No service with `--no-dbus`; signals and the pipeline still work.
    // (The D-Bus service itself starts further down, once the translation state
    // has loaded what `--translate` asked for.)

    // ---- capture is running (callback does NOT unwrap) ----------------------
    let running = Arc::new(AtomicBool::new(true));
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
        let shared = pause_state.clone();
        let running = Arc::clone(&running);
        let sound_enabled = cli.sound;
        if let Err(e) = install_pause_handlers(shared, running, sound_enabled) {
            log::warn!("{e:?} (signal pause/resume unavailable)");
        }
    }

    // ---- now the model load: onto the GPU/CPU (per `--gpu`) on its own thread.
    // Capture is already running above and the drain loop below sheds (counted)
    // while it loads.
    let asr_config = asr_config(cli, true, live_dict);
    // The capture loop fills this, the worker drains it (partial transcripts).
    let partials = asr_config.partials.clone();
    // The emitter owns the speak thread; its flag is how the capture loop below
    // knows to hold. Taken from the emitter rather than built here so there is
    // exactly one of each and no path can hold the pipeline for a thread that
    // does not exist.
    let emitter = make_emitter(cli)?;
    let speaking = emitter.speaking_flag();
    // The same reasoning for the Ctrl-C speech cut: taken here, while the
    // emitter is still on this thread and the `SpeechOut` has not moved onto
    // the emit worker. `None` without `--speak`, so the block below is inert.
    let speech_cut = emitter.speech_cut();
    // The translation step's shared state, resolved once from the flags. The
    // `--metrics` writer moves out of the config and into one slot both the
    // worker and the translation thread write through, because the worker writes
    // a row for every utterance it does not hand over and the thread writes the
    // rows whose `text` column is a translation.
    let metrics = metrics::MetricsSlot::new(metrics::open_metrics_log(cli.metrics.as_deref()));
    let translation_state = TranslationState::new(translation_plan(cli), translate::RealTranslator);
    // With `--translate`, the translator is loaded here, on the main thread,
    // before the worker starts: it is ~114 MiB of weights and a setup problem
    // (no file, no write access) that should fail fast rather than on the first
    // utterance. A run without the flag loads nothing until the bus asks.
    translation_state.load_startup()?;
    // No service with `--no-dbus`; signals and the pipeline still work. One
    // flag governs all three interfaces. Spawned after the startup load so the
    // served state is never behind what the run is doing.
    if !cli.no_dbus {
        crate::dbus::spawn(
            pause_state.clone(),
            hooked_dict.clone(),
            translation_state.clone(),
        );
    }
    // The lane exists even with no translator: a switch arrives on the bus thread
    // and the worker cannot be handed one after it started. It is fed only when a
    // translation will really happen, so this costs a parked thread and nothing
    // else.
    let translation = TranslationOut::spawn(
        translation_state,
        Box::new(emitter.sharing()),
        metrics.clone(),
    );
    let (asr_handle, asr_ready) = spawn_asr_worker(
        asr_rx,
        model,
        asr_config,
        emitter,
        Some(translation),
        metrics,
        Arc::clone(&paused),
        Arc::clone(&running),
    );
    let mut asr_ready_ok = false;

    // ---- main processing loop (NOT the audio callback) --------------------
    let mut last_queue_warn = Instant::now();
    let mut last_drop_report = (0usize, 0usize);
    let mut rebuild_attempts: u32 = 0;
    let mut was_paused = paused.load(Ordering::Acquire);
    // Edge state for the speaking hold, tracked beside `was_paused` rather than
    // folded into it: one flag raises the hold, the other says whose it is.
    let mut was_speaking = speaking.load(Ordering::Acquire);
    if was_paused {
        // Start-paused: drop any callback audio that raced stream start. Not
        // `flush_on_pause`: nothing has been spoken yet, so there is no open
        // utterance to save.
        fp.discard_pending();
    }
    while running.load(Ordering::Acquire) {
        // USB mics vanish mid-session: rebuild the device + stream + resamplers
        // instead of capturing deaf. Stale audio is discarded; the noise
        // floor is carried across explicitly (see `adopt_noise_floor`) since
        // the rebuild constructs a brand new `FrameProcessor`/`SpeechDetector`
        // rather than resetting the old one.
        if telemetry.device_gone.load(Ordering::Acquire) {
            rebuild_attempts += 1;
            log::warn!("input device lost, rebuild attempt #{rebuild_attempts}...");
            let warm_noise_floor = fp.noise_floor();
            thread::sleep(Duration::from_secs(2));
            match open_input_path(&host, cli, &acoustic, &telemetry) {
                Ok((new_rate, new_stream, mut new_fp, new_raw_rx)) => {
                    new_fp.adopt_noise_floor(warm_noise_floor);
                    hw_rate = new_rate;
                    stream = new_stream;
                    fp = new_fp;
                    raw_rx = new_raw_rx;
                    input_frames = fp.required_input_frames();
                    accumulator.clear();
                    accumulator.reserve(hw_rate as usize * 2);
                    fp.discard_pending();
                    partials.clear();
                    telemetry.device_gone.store(false, Ordering::Release);
                    rebuild_attempts = 0;
                    log::info!("input stream rebuilt ({} Hz)", hw_rate);
                }
                Err(e) => {
                    log::warn!("rebuild failed: {e:?}; retrying...");
                    continue;
                }
            }
        }
        // Fail fast if the ASR worker died during init instead of
        // capturing deaf: ready Err / disconnect, or early thread exit.
        match asr_ready.try_recv() {
            Ok(Ok(())) => {
                // First load done: the worker plays the ready jingle when
                // unpaused, on a detached thread whose device-open latency
                // means it goes audible just after this signal arrives.
                if !asr_ready_ok && cli.sound && !paused.load(Ordering::Acquire) {
                    cue_guard(&mut fp, &mut raw_rx, &mut accumulator);
                }
                asr_ready_ok = true;
            }
            Ok(Err(e)) => anyhow::bail!("asr worker failed to start: {e}"),
            Err(std::sync::mpsc::TryRecvError::Disconnected) if !asr_ready_ok => {
                anyhow::bail!("asr worker exited before signaling ready")
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                anyhow::bail!("asr worker died")
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if asr_handle.is_finished() && !asr_ready_ok {
                    anyhow::bail!("asr worker exited before signaling ready");
                }
            }
        }
        // Full-pipeline hold while paused: no resampling, no VAD, no
        // utterance, no floor learning, no ASR input. The raw ring is
        // drained and dropped so resume starts fresh with no stale speech;
        // the ASR worker transcribes the pause flush and drops the rest of
        // its queue (see asr).
        let now_paused = paused.load(Ordering::Acquire);
        if now_paused && !was_paused {
            // Pause entry edge. First the frames already drained: bounded by
            // one hw frame (no fresh `drain_raw_into`, which after a device
            // rebuild could hold seconds of pre-rebuild audio), so the cut
            // below sees the tail of what was said.
            drain_frames(
                &mut accumulator,
                input_frames,
                &mut fp,
                &mut asr_tx,
                push_live_utterance,
                &partials,
            );
            // Then cut and queue the open utterance before the hold discards
            // it: pausing must not lose the last sentence. `Wait` rather than
            // the live `Drop` policy — a queue near full means a slow model,
            // which is exactly when this sentence would be dropped, and the
            // loop is about to park for the whole pause anyway.
            if let Some(utterance) = fp.flush_on_pause() {
                push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
            }
            // The hold itself (resamplers + Hp + Agc + detector + pre-roll,
            // noise floor preserved) is inside `flush_on_pause`, so pre-pause
            // audio cannot leak into the first post-resume utterance.
            partials.clear();
        }
        if !now_paused && was_paused {
            // Resume edge: the signal blip already fired and the worker's
            // ready jingle follows within ~50 ms, both while the mic is
            // live — discard the window so cue tones are never transcribed
            // (Murmure sleeps past its start beep for the same reason).
            if cli.sound {
                cue_guard(&mut fp, &mut raw_rx, &mut accumulator);
            }
        }
        was_paused = now_paused;
        // Speaking hold (`--speak`): sttd's own voice must not be transcribed.
        // The same pipeline hold as a pause, on the same drain-and-drop path,
        // but a *separate* flag — the pause flag is the user's, and setting it
        // would trigger an unload or an autopause that has nothing to do with
        // this.
        //
        // The rising edge cuts the open utterance first, exactly as the pause
        // edge does: what was said just before sttd started talking is real
        // speech and gets transcribed like any other. It is not marked as a
        // pause flush, because the pause flag is not set and so the worker's
        // held-pause queue policy never sees it (`FrameProcessor::flush_on_speaking`).
        let now_speaking = speaking.load(Ordering::Acquire);
        if now_speaking && !was_speaking {
            if let Some(utterance) = fp.flush_on_speaking() {
                push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
            }
            partials.clear();
        }
        was_speaking = now_speaking;
        if now_paused || now_speaking {
            // Drain-and-drop without feeding SignalPath; park briefly so resume
            // is noticed within ~50 ms while idle CPU stays negligible.
            //
            // While speaking, the park also paces the loop: sttd talks for
            // seconds, and 50 ms of parking costs nothing to detect the end of
            // it. The `hold_pipeline` itself was already taken on the rising
            // edge, so no resampling or VAD happens in here.
            drain_raw_into(&mut raw_rx, &mut accumulator);
            accumulator.clear();
            if !raw_rx.is_abandoned() {
                thread::sleep(Duration::from_millis(50));
                drain_raw_into(&mut raw_rx, &mut accumulator);
                accumulator.clear();
            }
            continue;
        }
        // Drain everything the callback delivered since last iteration.
        drain_raw_into(&mut raw_rx, &mut accumulator);
        // A full hw frame still isn't available: park briefly for the next
        // samples instead of busy-spinning. Same ~2 ms worst-case cadence
        // for the device/ASR checks below.
        if accumulator.len() < input_frames && !raw_rx.is_abandoned() {
            thread::sleep(Duration::from_millis(2));
            drain_raw_into(&mut raw_rx, &mut accumulator);
        }

        // Raw hw-rate samples -> FrameProcessor (denoise/VAD + model
        // branches, all resampling included).
        drain_frames(
            &mut accumulator,
            input_frames,
            &mut fp,
            &mut asr_tx,
            push_live_utterance,
            &partials,
        );

        // Backpressure visibility: bounded rings shed under load (callback
        // must never block, live speech must stay fresh), so report depths
        // plus shed totals — throttled to one line per 5 s. Signal-path
        // pairing depths catch a resampler divergence the others miss.
        report_backpressure(
            BackpressureState {
                last_queue_warn: &mut last_queue_warn,
                last_drop_report: &mut last_drop_report,
            },
            &raw_rx,
            &asr_tx,
            &fp,
            &telemetry,
        );
    }

    // Stop speaking first. The user asked to stop, so the trailing utterance must
    // not be read aloud after the fact: `SpeechOut::drop` would otherwise hold
    // the join for up to `SPEAK_DRAIN_MS` while the speak thread finished the
    // queued sentence, and Ctrl-C would look like it was ignored. The abort
    // marks the drop path, so it arms no drain window.
    if let Some(cut) = &speech_cut {
        cut.abort();
    }

    // Stop capture before the trailing flush so no new audio races teardown.
    // Dropping the stream drops the raw producer; the consumer is drained
    // below only for full frames already accumulated.
    drop(stream);
    drain_raw_into(&mut raw_rx, &mut accumulator);
    drain_frames(
        &mut accumulator,
        input_frames,
        &mut fp,
        &mut asr_tx,
        push_live_utterance,
        &partials,
    );
    // Flush trailing speech, then wait for the ASR thread to finish with it (and
    // any earlier backlog) before the process exits, so an in-flight CUDA op
    // cannot race process teardown the same way an early `?` above used to.
    //
    // The flush buys the *opportunity*, not a transcript: `running` is already
    // false here, so both engines' pre-call check (`parakeet.rs`, `whisper.rs`)
    // rejects the decode and the utterance ends at an info line. That is the
    // right trade for an interrupted live run — the user asked to stop — but it
    // means the trailing utterance reaches no channel here. `--file` is the
    // path that really transcribes to the end, because it never clears `running`.
    if let Some(utterance) = fp.flush_trailing_speech() {
        push_utterance(&mut asr_tx, utterance, Backpressure::Drop);
    }
    drop(raw_rx);
    finish_run(fp, asr_tx, asr_handle)?;
    log::info!("stopped");
    Ok(())
}

/// Open the input device + stream + frame processor in one go, playing the
/// stream before returning. Used for the initial open (fails fast, so a bad
/// `--device`/`--channel` aborts before the model loads) and for rebuilds
/// after `DeviceNotAvailable` (see the main loop). A rebuild re-creates the
/// resamplers for the (possibly changed) hw rate and returns a brand new
/// `FrameProcessor`, so utterance state is lost; the caller carries the warm
/// VAD noise floor across explicitly via `adopt_noise_floor` (this function
/// itself has no memory of it). The raw `rtrb` ring is created here (capacity
/// follows the hw rate); the utterance ring lives in the caller and survives
/// rebuilds.
///
/// `--channel auto` probes the loudest channel once per open (~500 ms) and
/// uses that pick for the stream; `mix`/numeric picks never probe.
#[allow(clippy::too_many_arguments)]
fn open_input_path(
    host: &cpal::Host,
    cli: &Cli,
    acoustic: &AcousticConfig,
    telemetry: &StreamTelemetry,
) -> Result<(u32, cpal::Stream, FrameProcessor, Consumer<f32>)> {
    let device = select_input_device(host, cli.device.as_deref())?;
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
    // Resolve the effective mono channel: auto probes, the rest is pure.
    let effective_channel: Option<usize> = match cli.channel {
        crate::dsp::ChannelSel::Auto => {
            let picked =
                probe_loudest_channel(&device, &stream_config, channels, sample_format, 500)?;
            crate::dsp::resolve_channel_sel(cli.channel, channels, Some(picked))?
        }
        sel => crate::dsp::resolve_channel_sel(sel, channels, None).with_context(|| {
            format!("--channel/STTD_CHANNEL={sel} out of range for {channels}-channel input")
        })?,
    };
    log::info!(
        "hw config: {} Hz, {} ch, {:?}, channel={} (requested {})",
        hw_rate,
        channels,
        sample_format,
        effective_channel.map_or("mix".to_string(), |c| c.to_string()),
        cli.channel,
    );
    // `FrameProcessor` owns all resampling (a shared hw -> 16k hop by
    // default, or hw -> 48k -> denoise -> 48k -> 16k for the VAD branch plus
    // hw -> 16k for the model with `--denoiser deepfilter`).
    let fp = acoustic.make_frame_processor(hw_rate)?;
    let raw_cap = (hw_rate as usize).saturating_mul(RAW_RING_SECS).max(65536);
    let (raw_tx, raw_rx) = RingBuffer::<f32>::new(raw_cap);
    let stream = build_input_stream(
        &device,
        &stream_config,
        channels,
        effective_channel,
        sample_format,
        raw_tx,
        telemetry,
    )?;
    stream.play().context("failed to start input stream")?;
    Ok((hw_rate, stream, fp, raw_rx))
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
/// itself is signaled separately by the ASR thread once transcription
/// can actually resume.
fn install_pause_handlers(
    pause_state: crate::control::PauseState,
    running: Arc<AtomicBool>,
    sound_enabled: bool,
) -> Result<()> {
    let mut signals = Signals::new([SIGUSR1, SIGUSR2, SIGHUP, SIGTERM])
        .context("failed to register pause/resume signals")?;
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGUSR1 => {
                    let new_val = pause_state.set_paused(true);
                    log::info!("paused (SIGUSR1)");
                    if sound_enabled {
                        crate::sound::play_paused();
                    }
                    crate::dbus::emit_state_changed(new_val);
                }
                SIGUSR2 => {
                    let new_val = pause_state.set_paused(false);
                    log::info!("resumed (SIGUSR2)");
                    if sound_enabled {
                        crate::sound::play_resumed();
                    }
                    crate::dbus::emit_state_changed(new_val);
                }
                SIGHUP => {
                    let new_val = pause_state.toggle_paused();
                    if new_val {
                        log::info!("paused (SIGHUP toggle)");
                        if sound_enabled {
                            crate::sound::play_paused();
                        }
                    } else {
                        log::info!("resumed (SIGHUP toggle)");
                        if sound_enabled {
                            crate::sound::play_resumed();
                        }
                    }
                    crate::dbus::emit_state_changed(new_val);
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
    // instead of racing model teardown.
    let wav = read_wav_mono_f32_with_channel(path, cli.channel)?;
    log::info!(
        "file: {} ({} Hz, {} samples, {:.1}s, channel={})",
        path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f64 / wav.sample_rate as f64,
        cli.channel,
    );

    let (model, mut asr_tx, asr_rx) = prepare_model(cli)?;
    if model.kind == EngineKind::Whisper {
        // Offline decoding has no live text to show, so streaming buys the
        // short encoder window here and nothing else.
        log::info!(
            "file mode: short encoder window ({} frame floor), no partial transcripts",
            cli.audio_ctx
        );
    }
    let mut fp = AcousticConfig::from_cli(cli).make_frame_processor(wav.sample_rate)?;
    let input_frames = fp.required_input_frames();

    // File mode runs offline with no D-Bus service (see the interview), but
    // the worker still takes the live handle so `asr_options` has one shape.
    // The booster hook stays unconsumed: nothing swaps words here.
    let live_dict = crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
        load_dictionary_or_warn(cli),
        cli.dictionary.clone().unwrap_or_default(),
    ))
    .1;

    // File mode has no pause signalling, so a permanently-clear flag plus
    // disabled auto-unload can never trigger an unload mid-file. Sounds stay off
    // offline: no pause/resume transitions exist to cue. Autopause is forced off
    // for the same reason: file input must never flip the (unread) pause flag into
    // an unload mid-file.
    let config = AsrConfig {
        auto_unload: false,
        sound: false,
        autopause: false,
        ..asr_config(cli, false, live_dict)
    };
    // The emitter is built before the translation step because the translation
    // thread emits through a sharing handle onto the same queue.
    let emitter = make_emitter(cli)?;
    // File mode builds the translation step exactly as the microphone path does,
    // so an offline `--file ... --translate en` prints English. There is no bus
    // here, so nothing can switch it later: the lane exists only when a
    // translator did, and the worker emits inline when it does not.
    let metrics = metrics::MetricsSlot::new(metrics::open_metrics_log(cli.metrics.as_deref()));
    let translation_state = TranslationState::new(translation_plan(cli), translate::RealTranslator);
    translation_state.load_startup()?;
    let translation = translation_state.is_loaded().then(|| {
        TranslationOut::spawn(
            translation_state,
            Box::new(emitter.sharing()),
            metrics.clone(),
        )
    });
    let (asr_handle, asr_ready) = spawn_asr_worker(
        asr_rx,
        model,
        config,
        emitter,
        translation,
        metrics,
        Arc::new(AtomicBool::new(false)),
        // File input runs to completion; inference is never aborted.
        Arc::new(AtomicBool::new(true)),
    );
    // Fail fast on model-load/dump-dir errors before processing the file:
    // no realtime capture to keep bounded here, so blocking wait is fine.
    match asr_ready.recv_timeout(Duration::from_secs(600)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => anyhow::bail!("asr worker failed to start: {e}"),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("timed out waiting for asr worker to start")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("asr worker exited before signaling ready")
        }
    }

    let mut offset = 0usize;
    while offset + input_frames <= wav.samples.len() {
        if let Some(utterance) = fp.process_frame(&wav.samples[offset..offset + input_frames]) {
            push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
        }
        offset += input_frames;
    }
    // A trailing partial chunk (< input_frames samples, at most ~30 ms) is
    // dropped; fine for offline comparison runs.
    if let Some(utterance) = fp.flush_trailing_speech() {
        push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
    }

    finish_run(fp, asr_tx, asr_handle)?;
    Ok(())
}

/// Resolve `--model` (spec or path, either engine) to a guaranteed-present
/// `ModelRef`: engine dispatch first (`resolve_model`), then
/// download-if-missing (`ensure_available`).
fn resolve_model_input(cli: &Cli) -> Result<ModelRef> {
    let input = cli.model_input();
    let model = resolve_model(&input)?;
    ensure_available(&model)?;
    Ok(model)
}

/// Flags the selected engine ignores, so the no-effect is visible instead of
/// silent. Without the `cuda` feature both engines run on CPU, so `--gpu` is
/// accepted but has no provider behind it. With the `cuda` feature the
/// Parakeet and Whisper encoders use CUDA and the decoders stay on CPU, so
/// `--gpu` applies and is not reported here. `--language` only means something
/// to Whisper.
fn ignored_model_flags(cli: &Cli, model: &ModelRef) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if cli.gpu && !cfg!(feature = "cuda") {
        flags.push("--gpu");
    }
    if model.kind == EngineKind::Parakeet && cli.language.is_some() {
        // Parakeet auto-detects per utterance, so a fixed language cannot apply.
        // Any explicit flag counts, including one that matches the default the
        // system locale would have supplied: the user typed it, so it is worth
        // saying that it does nothing here.
        flags.push("--language");
    }
    if cli.audio_ctx != DEFAULT_WINDOW_FRAMES && model.kind == EngineKind::Parakeet {
        // Parakeet sizes its window itself; a Whisper encoder follows the audio
        // because the graph is patched when the model loads.
        flags.push("--audio-ctx");
    }
    flags
}

/// Warn about flags the selected engine ignores, then log what that engine
/// does about the spoken language.
fn validate_model_flags(cli: &Cli, model: &ModelRef) -> Result<()> {
    let ignored = ignored_model_flags(cli, model);
    if !ignored.is_empty() {
        log::warn!(
            "{} ignores flag(s): {} (no effect with this --model)",
            model.kind.as_str(),
            ignored.join(", ")
        );
    }
    match model.kind {
        EngineKind::Parakeet => {
            log::info!("Parakeet auto-detects the spoken language per utterance")
        }
        EngineKind::Whisper => match cli.language() {
            Language::Auto => {
                log::info!("Whisper detects the spoken language per utterance (--language auto)")
            }
            language => log::info!("Whisper decodes in {language} (--language)"),
        },
    }
    Ok(())
}

/// The worker's whole configuration, resolved once from `Cli` on this thread
/// before the worker starts. The `Cli` → config mapping lives here, not in
/// `asr`, so the worker module never learns what a command line is.
fn asr_config(cli: &Cli, shed_stale: bool, dictionary: crate::asr::LiveDictionary) -> AsrConfig {
    AsrConfig {
        engine: EngineConfig {
            model_spec: cli.model_input(),
            n_threads: resolve_n_threads(cli.threads, auto_threads()),
            use_gpu: cli.gpu,
            language: cli.language(),
            audio_ctx: cli.audio_ctx,
        },
        // The transcript gate shares the acoustic thresholds, so both come
        // from one resolved config.
        gate: AcousticConfig::from_cli(cli).gate,
        remove_hesitations: cli.remove_hesitations,
        // The live handle, not a `Vec` snapshot: the D-Bus thread swaps the
        // word list at runtime and the worker observes it per utterance.
        dictionary,
        shed_stale,
        partials: crate::asr::PartialSlot::default(),
        auto_unload: cli.auto_unload,
        auto_unload_after: Duration::from_secs(cli.auto_unload_after),
        sound: cli.sound,
        autopause: cli.autopause,
    }
}

/// Re-copy and re-paste the last emitted transcription. One-shot: reads the
/// daemon's last-text file written on every emission. `type` mode types
/// directly without touching the clipboard; key modes copy via `wl-copy`
/// then paste via `ydotool`. Warns (no failure) when no emission exists yet.
fn run_paste_last(cli: &Cli) -> Result<()> {
    let path = crate::emit::last_text_path();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let text = text.trim();
    if text.is_empty() {
        log::warn!(
            "paste-last: no emitted transcription yet ({})",
            path.display()
        );
        return Ok(());
    }
    if !crate::emit::is_wayland_session_env() {
        log::warn!("paste-last: not a Wayland session; copy attempted anyway");
    }
    if cli.paste_mode == crate::emit::PasteMode::Type {
        crate::emit::type_via_ydotool(text)
            .context("paste-last: ydotool type failed (is ydotoold running?)")?;
        log::info!("paste-last: typed {} chars", text.chars().count());
        return Ok(());
    }
    crate::emit::copy_to_clipboard(text).context("paste-last: wl-copy failed")?;
    log::info!("paste-last: copied {} chars", text.chars().count());
    crate::emit::paste_via_ydotool(cli.paste_mode)
        .context("paste-last: ydotool failed (is ydotoold running?)")?;
    Ok(())
}

/// The run's translation settings, read out of the flags. See
/// `translate::PlanFlags` for what each one is.
fn translation_plan(cli: &Cli) -> TranslationPlan {
    TranslationPlan::from_flags(translate::PlanFlags {
        translate: cli.translate.as_deref(),
        translate_model: cli.translate_model.as_deref(),
        language: cli.language(),
        intra_threads: resolve_n_threads(cli.threads, auto_threads()).max(1) as usize,
        voice: cli.speak.then(|| cli.voice.clone()),
        runtime_requests: !cli.no_dbus && cli.file.is_none(),
    })
}

/// Load `--dictionary` once at startup. A broken path warns and disables
/// correction (never fails transcription); success logs the entry count.
/// Large dictionaries warn: above `POSTCORR_WARN_DICT_WORDS` the false-positive
/// risk grows, at or above `POSTCORR_MAX_DICT_WORDS` fuzzy is off (exact stays).
fn load_dictionary_or_warn(cli: &Cli) -> Vec<String> {
    let Some(path) = cli.dictionary.as_deref() else {
        return Vec::new();
    };
    match crate::dictionary::load_dictionary(path) {
        Ok(words) => {
            log::info!(
                "dictionary: {} entr(y/ies) from {}",
                words.len(),
                path.display()
            );
            if words.len() > crate::dictionary::POSTCORR_MAX_DICT_WORDS {
                log::warn!(
                    "dictionary: {} entries, fuzzy correction disabled (over {}); exact casing restore stays on",
                    words.len(),
                    crate::dictionary::POSTCORR_MAX_DICT_WORDS
                );
            } else if words.len() > crate::dictionary::POSTCORR_WARN_DICT_WORDS {
                log::warn!(
                    "dictionary: {} entries, false-positive risk grows; keep the list small and targeted",
                    words.len()
                );
            }
            words
        }
        Err(e) => {
            log::warn!(
                "dictionary: failed to load {}: {e:?} (correction disabled)",
                path.display()
            );
            Vec::new()
        }
    }
}

/// Emission policy (dump dir, clipboard, paste, stdout, spoken transcripts),
/// resolved once from `Cli`. Runs on the main thread before the worker spawns,
/// so a bad dump dir, voice key or voice download fails fast like the model load
/// it precedes.
fn make_emitter(cli: &Cli) -> Result<SystemEmitter> {
    SystemEmitter::new(EmissionPolicy {
        dump_dir: cli.dump_dir.clone(),
        copy: cli.copy,
        paste: cli.paste,
        paste_mode: cli.paste_mode,
        stdout: cli.stdout,
        speech: make_speech(cli)?,
    })
}

/// The spoken-emission channel, when `--speak` is on: resolve the voice key,
/// download what is missing and load the graph, then hand it to its own thread.
///
/// Off by default, and inert without `--speak`: `--voice` is never resolved and
/// nothing is downloaded, because a flag that silently pulled 60 MiB nobody
/// asked for is worse than one that does nothing. The voice load is a few
/// hundred milliseconds, on the main thread, before the worker starts — the same
/// place and cost profile as the speech model's own load.
fn make_speech(cli: &Cli) -> Result<Option<SpeechOut>> {
    if !cli.speak {
        // Warned rather than silently ignored, and only when the user actually
        // typed one: the default is not a thing they asked for.
        if cli.voice != crate::config::DEFAULT_VOICE {
            log::warn!("--voice is ignored without --speak");
        }
        return Ok(None);
    }
    let voice = crate::voice_store::prepare_voice(&cli.voice)
        .with_context(|| format!("failed to prepare voice {}", cli.voice))?;
    log::info!("speak: reading every emitted transcript aloud (voice={voice})");
    if !cli.autopause {
        // The capture side holds itself while sttd speaks, so a second utterance
        // cannot even be recorded during playback. That is right for
        // single-shot dictation and wrong for a continuous run, where it eats
        // the user's own words; `--autopause` is the pairing that works.
        log::warn!(
            "--speak without --autopause discards everything you say while sttd is speaking \
             (use --autopause, or headphones)"
        );
    }
    // One thread for the voice, half of `--threads` at most: a sentence is a
    // few seconds of audio and the graph is small, so it must not take the
    // whole machine away from the speech model decoding the next utterance.
    let speech_threads = (resolve_n_threads(cli.threads, 1) / 2).max(1) as usize;
    let loaded = crate::piper::PiperVoice::load(&voice, speech_threads)
        .context("failed to load the piper voice")?;
    Ok(Some(SpeechOut::spawn(loaded)))
}

/// The capture-side tunables, resolved once from `Cli` and shared by both
/// input paths: the microphone opens a `FrameProcessor` per device open (and
/// per rebuild), file mode opens one, and the same thresholds reach the ASR
/// worker through `AsrConfig::gate`. Keeping them in one struct is what
/// stops the two paths from drifting apart, and it keeps `Utterance` free of
/// config (only evidence).
struct AcousticConfig {
    denoiser: crate::denoise::DenoiserKind,
    vad_threshold: f32,
    vad_adapt: bool,
    gate: gate::GateConfig,
    hangover_frames: u32,
    target_rms: f32,
}

impl AcousticConfig {
    const fn from_cli(cli: &Cli) -> Self {
        Self {
            denoiser: cli.denoiser,
            vad_threshold: cli.vad_threshold,
            vad_adapt: cli.vad_adapt,
            gate: gate::GateConfig::new(cli.min_rms, cli.min_density),
            hangover_frames: crate::config::hangover_frames(cli.hangover_ms),
            target_rms: cli.target_rms,
        }
    }

    /// Build the frame processor for a given input rate. `FrameProcessor`
    /// owns all resampling (a shared hw -> 16 kHz hop by default, or
    /// hw -> 48 kHz -> denoise -> 48 kHz -> 16 kHz for the VAD branch plus
    /// hw -> 16 kHz for the model with `--denoiser deepfilter`).
    fn make_frame_processor(&self, hw_rate: u32) -> Result<FrameProcessor> {
        FrameProcessor::new(
            hw_rate,
            self.denoiser,
            self.vad_threshold,
            self.vad_adapt,
            self.gate.clone(),
            self.hangover_frames,
            self.target_rms,
            crate::config::MAX_GAIN,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ModelRef` for flag-validation tests: `assets` is empty (a
    /// user-supplied directory), which is all the flag checks look at.
    fn model_ref(kind: EngineKind, dir: &str) -> ModelRef {
        ModelRef {
            dir: std::path::PathBuf::from(dir),
            assets: &[],
            kind,
        }
    }

    #[test]
    fn ignored_model_flags_are_empty_for_plain_cli() {
        // `--gpu` defaults to false and `--language` to the `Language`
        // default, so a plain CLI is clean for either engine.
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        for (kind, dir) in [
            (EngineKind::Parakeet, "models/parakeet-ultra-int8"),
            (EngineKind::Whisper, "models/whisper-large-v3-turbo-int8"),
        ] {
            let model = model_ref(kind, dir);
            assert!(
                ignored_model_flags(&cli, &model).is_empty(),
                "{}",
                kind.as_str()
            );
            assert!(validate_model_flags(&cli, &model).is_ok());
        }
    }

    /// `--gpu` is a no-op only in a CPU-only build. A `cuda` build runs the
    /// encoders on the CUDA EP, so the flag applies to both engines and no
    /// longer belongs in the ignored set.
    #[test]
    fn gpu_is_ignored_by_both_engines_without_cuda() {
        let cli = Cli::try_parse_from(["sttd", "--gpu"]).expect("flags must parse");
        let expected: Vec<&'static str> = if cfg!(feature = "cuda") {
            Vec::new()
        } else {
            vec!["--gpu"]
        };
        for kind in [EngineKind::Parakeet, EngineKind::Whisper] {
            let model = model_ref(kind, "models/x");
            assert_eq!(ignored_model_flags(&cli, &model), expected);
        }
    }

    /// Parakeet auto-detects per utterance, so a fixed `--language` cannot
    /// apply and must be reported rather than silently dropped.
    #[test]
    fn language_is_ignored_by_parakeet_but_honored_by_whisper() {
        let cli = Cli::try_parse_from(["sttd", "--language", "en"]).expect("flags must parse");
        let parakeet = model_ref(EngineKind::Parakeet, "models/parakeet-ultra-int8");
        let whisper = model_ref(EngineKind::Whisper, "models/whisper-large-v3-turbo-int8");
        assert_eq!(ignored_model_flags(&cli, &parakeet), vec!["--language"]);
        assert!(ignored_model_flags(&cli, &whisper).is_empty());

        // `auto` is a deviation from the default for Whisper's log line but a
        // Parakeet user who set it explicitly should still be told.
        let auto = Cli::try_parse_from(["sttd", "--language", "auto"]).expect("flags must parse");
        assert_eq!(auto.language, Some(Language::Auto));
        assert_eq!(auto.language(), Language::Auto);
        assert_eq!(
            ignored_model_flags(&auto, &parakeet),
            vec!["--language"],
            "explicit --language auto must still be reported under Parakeet"
        );
        assert!(ignored_model_flags(&auto, &whisper).is_empty());
    }

    #[test]
    fn audio_ctx_is_reported_only_where_the_window_cannot_follow_the_audio() {
        let parakeet = model_ref(EngineKind::Parakeet, "models/parakeet-ultra-int8");
        let whisper = model_ref(EngineKind::Whisper, "models/whisper-large-v3-turbo-int8");
        let cli = Cli::try_parse_from(["sttd", "--audio-ctx", "300"]).expect("must parse");

        // Parakeet sizes its own window.
        assert_eq!(ignored_model_flags(&cli, &parakeet), vec!["--audio-ctx"]);
        // Whisper's encoder is patched when the model loads, so the floor applies.
        assert!(ignored_model_flags(&cli, &whisper).is_empty());
        assert!(validate_model_flags(&cli, &parakeet).is_ok());

        // The default is not a setting, so nothing is reported for it.
        let plain = Cli::try_parse_from(["sttd"]).expect("must parse");
        assert!(ignored_model_flags(&plain, &whisper).is_empty());
        assert!(ignored_model_flags(&plain, &parakeet).is_empty());
    }

    /// Every code Whisper accepts goes through, and the unset default is the
    /// system locale's language rather than a fixed one. sttd keeps no list of
    /// languages: the model decides at load time which codes it has.
    #[test]
    fn language_flag_defaults_to_the_system_locale_and_takes_any_code() {
        let plain = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert_eq!(plain.language, None, "unset, not defaulted to a code");
        assert_eq!(
            plain.language(),
            crate::config::system_language(),
            "unset resolves through the locale"
        );

        for (value, expected) in [
            ("pl", Language::Explicit("pl".to_string())),
            ("en", Language::Explicit("en".to_string())),
            ("de", Language::Explicit("de".to_string())),
            ("yue", Language::Explicit("yue".to_string())),
            ("DE", Language::Explicit("de".to_string())),
            ("auto", Language::Auto),
            ("Auto", Language::Auto),
            (" pl ", Language::Explicit("pl".to_string())),
        ] {
            let cli = Cli::try_parse_from(["sttd", "--language", value])
                .unwrap_or_else(|e| panic!("--language {value} must parse: {e}"));
            assert_eq!(cli.language, Some(expected.clone()), "value={value:?}");
            assert_eq!(cli.language(), expected, "value={value:?}");
        }

        // A locale name, a word, a digit and an over-long code are not language
        // codes, so they are refused before a model is ever loaded. Which codes
        // the model *does* have is its own business, checked at load.
        for bad in ["x", "pl_PL", "english", "p1", "abcd", "", "en-US"] {
            assert!(
                Cli::try_parse_from(["sttd", "--language", bad]).is_err(),
                "must reject {bad:?}"
            );
        }
    }

    #[test]
    fn asr_options_carry_the_language_to_the_worker() {
        let cli = Cli::try_parse_from(["sttd", "--language", "en"]).expect("flags must parse");
        let dictionary = crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
            Vec::new(),
            Default::default(),
        ))
        .1;
        assert_eq!(
            asr_config(&cli, true, dictionary).engine.language,
            Language::Explicit("en".to_string())
        );
    }

    #[test]
    fn asr_config_starts_with_an_empty_partial_slot() {
        let dictionary = || {
            crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
                Vec::new(),
                Default::default(),
            ))
            .1
        };
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        let config = asr_config(&cli, true, dictionary());
        assert!(config.partials.take().is_none());
        // The capture side holds a clone of the same slot.
        let capture_side = config.partials.clone();
        capture_side.put(vec![0.0; 4]);
        assert_eq!(config.partials.take().map(|p| p.len()), Some(4));
    }

    #[test]
    fn acoustic_config_carries_the_cli_tunables() {
        let cli = Cli::try_parse_from([
            "sttd",
            "--vad-threshold",
            "0.4",
            "--vad-adapt",
            "off",
            "--min-rms",
            "0.01",
            "--min-density",
            "0.5",
            "--hangover-ms",
            "350",
            "--target-rms",
            "0.02",
        ])
        .expect("flags must parse");
        let acoustic = AcousticConfig::from_cli(&cli);
        assert_eq!(acoustic.vad_threshold, 0.4);
        assert!(!acoustic.vad_adapt);
        assert_eq!(acoustic.gate.min_speech_rms, 0.01);
        assert_eq!(acoustic.gate.min_density, 0.5);
        assert_eq!(acoustic.hangover_frames, 35);
        assert_eq!(acoustic.target_rms, 0.02);
        // The ASR worker gets the same thresholds, from the same builder.
        let dictionary = crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
            Vec::new(),
            Default::default(),
        ))
        .1;
        assert_eq!(asr_config(&cli, true, dictionary).gate.min_density, 0.5);
    }

    #[test]
    fn drain_frames_consumes_whole_frames_and_keeps_the_remainder() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        let mut fp = AcousticConfig::from_cli(&cli)
            .make_frame_processor(crate::config::SAMPLE_RATE)
            .expect("frame processor must build");
        let input_frames = fp.required_input_frames();
        let (mut asr_tx, _asr_rx) = RingBuffer::<Utterance>::new(crate::asr::ASR_QUEUE_CAP);
        // Two whole frames plus a partial one; silence opens no utterance, so
        // nothing is handed to the worker.
        let mut buffer = vec![0.0f32; input_frames * 2 + 7];
        drain_frames(
            &mut buffer,
            input_frames,
            &mut fp,
            &mut asr_tx,
            push_live_utterance,
            &crate::asr::PartialSlot::default(),
        );
        assert_eq!(buffer.len(), 7);
        assert!(buffer.iter().all(|&s| s == 0.0));
        assert_eq!(asr_tx.slots(), crate::asr::ASR_QUEUE_CAP);
    }
}
