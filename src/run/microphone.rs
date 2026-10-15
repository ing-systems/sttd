//! Microphone mode: the live capture loop, from the cpal callback's raw ring
//! to the ASR worker's utterance queue, with the pause and speaking holds and
//! the device rebuild in between.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use rtrb::{Consumer, Producer, RingBuffer};

use super::queue::{
    drain_frames, drain_raw_into, finish_run, producer_occupied, push_live_utterance,
    push_utterance, Backpressure,
};
use super::setup::{
    asr_config, live_dictionary, make_emitter, open_metrics, prepare_model, start_translation,
    AcousticConfig,
};
use super::unix_signals::{install_ctrlc_handler, install_pause_handlers};
use crate::asr::{spawn_asr_worker, PartialSlot};
use crate::capture::{
    build_input_stream, probe_loudest_channel, select_input_device, StreamTelemetry,
};
use crate::cli::Cli;
use crate::emit::SpeechCut;
use crate::frame::FrameProcessor;
use crate::model_store::EngineKind;
use crate::translate::TranslationOut;
use crate::utterance::Utterance;

/// Seconds of raw mono samples the callback may backlog. Past that the
/// callback drops (counted) instead of growing memory without bound while
/// the model loads or the GPU lags. Sized in `open_input_path` from the hw
/// rate (`hw_rate * RAW_RING_SECS`).
pub(crate) const RAW_RING_SECS: usize = 4;

pub(crate) fn run_microphone_mode(cli: &Cli) -> Result<()> {
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
    let (model, asr_tx, asr_rx) = prepare_model(cli)?;
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
    let input = open_input_path(&host, cli, &acoustic, &telemetry)?;
    let accumulator: Vec<f32> = Vec::with_capacity(input.hw_rate as usize * 2);

    // ---- shared control state (D-Bus, signals, pipeline, ASR worker) ------
    // `PauseState` and `DictionaryState` are separate, one per bus interface.
    // Each is a cheap clone, so the bus thread, the signal handlers, the main
    // loop and the worker all observe the same objects.
    // `LiveDictionary::new` attaches the ASR booster hook and hands back the
    // hooked clone for `dbus::spawn`.
    let pause_state = crate::control::PauseState::new(cli.start_paused);
    let (hooked_dict, live_dict) = live_dictionary(cli);
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
    let asr_config = asr_config(cli, &acoustic, true, live_dict);
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
    // the emit worker. `None` without `--speak`, so the shutdown abort is inert.
    let speech_cut = emitter.speech_cut();
    let metrics = open_metrics(cli);
    let translation_state = start_translation(cli)?;
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

    // ---- main processing loop (NOT the audio callback) --------------------
    let was_paused = paused.load(Ordering::Acquire);
    // Edge state for the speaking hold, tracked beside `was_paused` rather than
    // folded into it: one flag raises the hold, the other says whose it is.
    let was_speaking = speaking.load(Ordering::Acquire);
    MicLoop {
        cli,
        host,
        acoustic,
        telemetry,
        input,
        accumulator,
        asr_tx,
        partials,
        running,
        paused,
        speaking,
        speech_cut,
        asr_handle,
        asr_ready,
        asr_ready_ok: false,
        was_paused,
        was_speaking,
        rebuild_attempts: 0,
        last_queue_warn: Instant::now(),
        last_drop_report: (0, 0),
    }
    .run()
}

/// One open input device: the playing stream, the frame processor built for
/// its rate, and the raw ring its callback fills. Replaced whole on a rebuild.
struct InputPath {
    hw_rate: u32,
    stream: cpal::Stream,
    fp: FrameProcessor,
    raw_rx: Consumer<f32>,
    /// Raw hw-rate samples per 10 ms frame (`FrameProcessor::required_input_frames`).
    input_frames: usize,
}

/// The capture loop's state: the input path, the utterance producer that
/// survives a rebuild, the flags other threads raise, and the edge trackers for
/// the two holds. One method per step of the loop body, in loop order.
struct MicLoop<'a> {
    cli: &'a Cli,
    host: cpal::Host,
    acoustic: AcousticConfig,
    telemetry: StreamTelemetry,
    input: InputPath,
    /// Raw hw-rate samples drained from the ring but not yet framed.
    accumulator: Vec<f32>,
    asr_tx: Producer<Utterance>,
    partials: PartialSlot,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
    speech_cut: Option<Arc<SpeechCut>>,
    asr_handle: JoinHandle<()>,
    asr_ready: Receiver<Result<(), String>>,
    asr_ready_ok: bool,
    was_paused: bool,
    was_speaking: bool,
    rebuild_attempts: u32,
    last_queue_warn: Instant,
    last_drop_report: (usize, usize),
}

impl MicLoop<'_> {
    fn run(mut self) -> Result<()> {
        if self.was_paused {
            // Start-paused: drop any callback audio that raced stream start. Not
            // `flush_on_pause`: nothing has been spoken yet, so there is no open
            // utterance to save.
            self.input.fp.discard_pending();
        }
        while self.running.load(Ordering::Acquire) {
            if !self.rebuild_if_device_lost() {
                continue;
            }
            self.poll_worker_ready()?;
            let now_paused = self.on_pause_edges();
            let now_speaking = self.on_speaking_edges();
            if now_paused || now_speaking {
                self.hold();
                continue;
            }
            self.pump();
            self.report_backpressure();
        }
        self.shutdown()
    }

    /// USB mics vanish mid-session: rebuild the device + stream + resamplers
    /// instead of capturing deaf. Stale audio is discarded; the noise floor is
    /// carried across explicitly (see `adopt_noise_floor`) since the rebuild
    /// constructs a brand new `FrameProcessor`/`SpeechDetector` rather than
    /// resetting the old one.
    ///
    /// `false` when the device is gone and the rebuild failed: the loop goes
    /// round again (and retries) without touching the dead input.
    fn rebuild_if_device_lost(&mut self) -> bool {
        if !self.telemetry.device_gone.load(Ordering::Acquire) {
            return true;
        }
        self.rebuild_attempts += 1;
        log::warn!(
            "input device lost, rebuild attempt #{}...",
            self.rebuild_attempts
        );
        let warm_noise_floor = self.input.fp.noise_floor();
        thread::sleep(Duration::from_secs(2));
        match open_input_path(&self.host, self.cli, &self.acoustic, &self.telemetry) {
            Ok(mut new_input) => {
                new_input.fp.adopt_noise_floor(warm_noise_floor);
                self.input = new_input;
                self.accumulator.clear();
                self.accumulator.reserve(self.input.hw_rate as usize * 2);
                self.input.fp.discard_pending();
                self.partials.clear();
                self.telemetry.device_gone.store(false, Ordering::Release);
                self.rebuild_attempts = 0;
                log::info!("input stream rebuilt ({} Hz)", self.input.hw_rate);
                true
            }
            Err(e) => {
                log::warn!("rebuild failed: {e:?}; retrying...");
                false
            }
        }
    }

    /// Fail fast if the ASR worker died during init instead of capturing deaf:
    /// ready Err / disconnect, or early thread exit.
    fn poll_worker_ready(&mut self) -> Result<()> {
        match self.asr_ready.try_recv() {
            Ok(Ok(())) => {
                // First load done: the worker plays the ready jingle when
                // unpaused, on a detached thread whose device-open latency
                // means it goes audible just after this signal arrives.
                if !self.asr_ready_ok && self.cli.sound && !self.paused.load(Ordering::Acquire) {
                    self.cue_guard();
                }
                self.asr_ready_ok = true;
            }
            Ok(Err(e)) => anyhow::bail!("asr worker failed to start: {e}"),
            Err(TryRecvError::Disconnected) if !self.asr_ready_ok => {
                anyhow::bail!("asr worker exited before signaling ready")
            }
            Err(TryRecvError::Disconnected) => {
                anyhow::bail!("asr worker died")
            }
            Err(TryRecvError::Empty) => {
                if self.asr_handle.is_finished() && !self.asr_ready_ok {
                    anyhow::bail!("asr worker exited before signaling ready");
                }
            }
        }
        Ok(())
    }

    /// Full-pipeline hold while paused: no resampling, no VAD, no utterance, no
    /// floor learning, no ASR input. The raw ring is drained and dropped so
    /// resume starts fresh with no stale speech; the ASR worker transcribes the
    /// pause flush and drops the rest of its queue (see asr).
    ///
    /// Handles both edges and returns whether the run is paused now.
    fn on_pause_edges(&mut self) -> bool {
        let now_paused = self.paused.load(Ordering::Acquire);
        if now_paused && !self.was_paused {
            // Pause entry edge. First the frames already drained: bounded by
            // one hw frame (no fresh `drain_raw_into`, which after a device
            // rebuild could hold seconds of pre-rebuild audio), so the cut
            // below sees the tail of what was said.
            self.drain_accumulated();
            // Then cut and queue the open utterance before the hold discards
            // it: pausing must not lose the last sentence. `Wait` rather than
            // the live `Drop` policy — a queue near full means a slow model,
            // which is exactly when this sentence would be dropped, and the
            // loop is about to park for the whole pause anyway.
            if let Some(utterance) = self.input.fp.flush_on_pause() {
                push_utterance(&mut self.asr_tx, utterance, Backpressure::Wait);
            }
            // The hold itself (resamplers + Hp + Agc + detector + pre-roll,
            // noise floor preserved) is inside `flush_on_pause`, so pre-pause
            // audio cannot leak into the first post-resume utterance.
            self.partials.clear();
        }
        if !now_paused && self.was_paused {
            // Resume edge: the signal blip already fired and the worker's
            // ready jingle follows within ~50 ms, both while the mic is
            // live — discard the window so cue tones are never transcribed
            // (Murmure sleeps past its start beep for the same reason).
            if self.cli.sound {
                self.cue_guard();
            }
        }
        self.was_paused = now_paused;
        now_paused
    }

    /// Speaking hold (`--speak`): sttd's own voice must not be transcribed.
    /// The same pipeline hold as a pause, on the same drain-and-drop path,
    /// but a *separate* flag — the pause flag is the user's, and setting it
    /// would trigger an unload or an autopause that has nothing to do with
    /// this.
    ///
    /// The rising edge cuts the open utterance first, exactly as the pause
    /// edge does: what was said just before sttd started talking is real
    /// speech and gets transcribed like any other. It is not marked as a
    /// pause flush, because the pause flag is not set and so the worker's
    /// held-pause queue policy never sees it (`FrameProcessor::flush_on_speaking`).
    ///
    /// Returns whether sttd is speaking now.
    fn on_speaking_edges(&mut self) -> bool {
        let now_speaking = self.speaking.load(Ordering::Acquire);
        if now_speaking && !self.was_speaking {
            if let Some(utterance) = self.input.fp.flush_on_speaking() {
                push_utterance(&mut self.asr_tx, utterance, Backpressure::Wait);
            }
            self.partials.clear();
        }
        self.was_speaking = now_speaking;
        now_speaking
    }

    /// Drain-and-drop without feeding SignalPath; park briefly so resume is
    /// noticed within ~50 ms while idle CPU stays negligible.
    ///
    /// While speaking, the park also paces the loop: sttd talks for seconds,
    /// and 50 ms of parking costs nothing to detect the end of it. The
    /// `discard_pending` hold itself was already taken on the rising edge, so no
    /// resampling or VAD happens in here.
    fn hold(&mut self) {
        drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
        self.accumulator.clear();
        if !self.input.raw_rx.is_abandoned() {
            thread::sleep(Duration::from_millis(50));
            drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
            self.accumulator.clear();
        }
    }

    /// One pass of live capture: everything the callback delivered since the
    /// last pass, through the frame processor.
    fn pump(&mut self) {
        // Drain everything the callback delivered since last iteration.
        drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
        // A full hw frame still isn't available: park briefly for the next
        // samples instead of busy-spinning. Same ~2 ms worst-case cadence
        // for the device/ASR checks.
        if self.accumulator.len() < self.input.input_frames && !self.input.raw_rx.is_abandoned() {
            thread::sleep(Duration::from_millis(2));
            drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
        }

        // Raw hw-rate samples -> FrameProcessor (denoise/VAD + model
        // branches, all resampling included).
        self.drain_accumulated();
    }

    /// Feed every whole frame already accumulated through the frame processor,
    /// queueing what it closes (live policy) and leaving partials in the slot.
    fn drain_accumulated(&mut self) {
        drain_frames(
            &mut self.accumulator,
            self.input.input_frames,
            &mut self.input.fp,
            &mut self.asr_tx,
            push_live_utterance,
            &self.partials,
        );
    }

    /// Backpressure visibility: bounded rings shed under load (callback must
    /// never block, live speech must stay fresh), so report depths plus shed
    /// totals for the VAD branch and the model branch pairing — throttled to
    /// one line per 5 s. Signal-path pairing depths catch a resampler
    /// divergence the others miss.
    fn report_backpressure(&mut self) {
        if self.last_queue_warn.elapsed() < Duration::from_secs(5) {
            return;
        }
        self.last_queue_warn = Instant::now();
        let raw_depth = self.input.raw_rx.slots();
        let asr_depth = producer_occupied(&self.asr_tx, crate::asr::ASR_QUEUE_CAP);
        let (signal_vad, signal_model) = self.input.fp.signal_pending_depths();
        let raw_dropped = self.telemetry.dropped_chunks.load(Ordering::Relaxed);
        let stream_errors = self.telemetry.errors.load(Ordering::Relaxed);
        let drops = (raw_dropped, stream_errors);
        let drops_grew = drops != self.last_drop_report;
        self.last_drop_report = drops;
        if raw_depth > 64
            || asr_depth > 4
            || signal_vad > crate::signal_path::SIGNAL_PENDING_WARN
            || signal_model > crate::signal_path::SIGNAL_PENDING_WARN
            || drops_grew
        {
            log::warn!(
                "queues (raw_rx={raw_depth}, asr={asr_depth}, signal_vad={signal_vad}, signal_model={signal_model}, dropped_chunks={raw_dropped}, stream_errors={stream_errors}): capture outruns resampling, branches diverged, or the model outruns real time"
            );
        }
    }

    /// Discard capture around sound-cue playback so cue tones are never
    /// transcribed. Full pipeline hold, drain, wait out the longest cue
    /// (`sound::CUE_QUIET_MS`), drain again. Skipped when an utterance is
    /// already open — in-flight speech wins over a cue, never the reverse.
    ///
    /// Callers: resume edge + first worker-ready (both unpaused + sound only).
    /// Steady-state never sleeps. Reload-while-unpaused is not covered: the
    /// worker's post-reload jingle has no signal the main loop can key on, so a
    /// 300 ms tone can still reach an otherwise-idle pipeline there (likely
    /// gate-dropped as non-speech-shaped audio).
    fn cue_guard(&mut self) {
        if self.input.fp.has_open_utterance() {
            return;
        }
        self.input.fp.discard_pending();
        drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
        self.accumulator.clear();
        thread::sleep(Duration::from_millis(crate::sound::CUE_QUIET_MS));
        drain_raw_into(&mut self.input.raw_rx, &mut self.accumulator);
        self.accumulator.clear();
    }

    /// The loop has stopped: cut the spoken channel, stop capture, flush what
    /// was said, and wait for the worker to finish with it.
    fn shutdown(self) -> Result<()> {
        // Stop speaking first. The user asked to stop, so the trailing utterance
        // must not be read aloud after the fact: `SpeechOut::drop` would
        // otherwise hold the join for up to `SPEAK_DRAIN_MS` while the speak
        // thread finished the queued sentence, and Ctrl-C would look like it was
        // ignored. The abort marks the drop path, so it arms no drain window.
        if let Some(cut) = &self.speech_cut {
            cut.abort();
        }
        let MicLoop {
            input,
            mut accumulator,
            mut asr_tx,
            partials,
            asr_handle,
            ..
        } = self;
        let InputPath {
            stream,
            mut fp,
            mut raw_rx,
            input_frames,
            ..
        } = input;

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
        // Flush trailing speech, then wait for the ASR thread to finish with it
        // (and any earlier backlog) before the process exits, so an in-flight
        // CUDA op cannot race process teardown the same way an early `?` used to.
        //
        // The flush buys the *opportunity*, not a transcript: `running` is
        // already false here, so both engines' pre-call check (`parakeet.rs`,
        // `whisper.rs`) rejects the decode and the utterance ends at an info
        // line. That is the right trade for an interrupted live run — the user
        // asked to stop — but it means the trailing utterance reaches no channel
        // here. `--file` is the path that really transcribes to the end, because
        // it never clears `running`.
        if let Some(utterance) = fp.flush_trailing_speech() {
            push_utterance(&mut asr_tx, utterance, Backpressure::Drop);
        }
        drop(raw_rx);
        finish_run(fp, asr_tx, asr_handle)?;
        log::info!("stopped");
        Ok(())
    }
}

/// Open the input device + stream + frame processor in one go, playing the
/// stream before returning. Used for the initial open (fails fast, so a bad
/// `--device`/`--channel` aborts before the model loads) and for rebuilds
/// after `DeviceNotAvailable` (see `MicLoop::rebuild_if_device_lost`). A
/// rebuild re-creates the resamplers for the (possibly changed) hw rate and
/// returns a brand new `FrameProcessor`, so utterance state is lost; the caller
/// carries the warm VAD noise floor across explicitly via `adopt_noise_floor`
/// (this function itself has no memory of it). The raw `rtrb` ring is created
/// here (capacity follows the hw rate); the utterance ring lives in the caller
/// and survives rebuilds.
///
/// `--channel auto` probes the loudest channel once per open (~500 ms) and
/// uses that pick for the stream; `mix`/numeric picks never probe.
fn open_input_path(
    host: &cpal::Host,
    cli: &Cli,
    acoustic: &AcousticConfig,
    telemetry: &StreamTelemetry,
) -> Result<InputPath> {
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
    let input_frames = fp.required_input_frames();
    Ok(InputPath {
        hw_rate,
        stream,
        fp,
        raw_rx,
        input_frames,
    })
}
