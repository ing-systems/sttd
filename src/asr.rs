// ---------------------------------------------------------------------------
// ASR seam: the worker loop, gates, metrics, conditioning.
// ---------------------------------------------------------------------------
//
// One speech model exists (Parakeet-TDT-0.6B-v3 via direct `ort` sessions,
// see `parakeet_core`), selected once at startup by `--model`
// (see `model_store::ModelRef`) and fixed for the process's lifetime.
// This module owns the worker thread and its load/unload/pause/shed loop,
// the transcript gate call, `--metrics` JSONL, audio conditioning, and the
// split-seam dedup. The engine's own file (`parakeet`) owns only what's
// genuinely engine-specific — the actual inference call, and enough of a
// description to log — behind the `SpeechModel` trait.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use rtrb::Consumer;

use crate::config::{
    CONDITION_FADE_SAMPLES, DEFAULT_THREADS, INFERENCE_PAD_SAMPLES, RTF_WARN_RTF, SAMPLE_RATE,
};
use crate::emit::{Emission, Emitter};
use crate::gate::GateConfig;
use crate::utterance::Utterance;
use crate::model_store::ModelRef;

/// Result of one inference: the joined text plus per-word model confidences
/// when the engine reports them (Parakeet, for the dictionary gate).
pub(crate) struct Transcript {
    pub(crate) text: String,
    pub(crate) word_confidences: Vec<(String, f32)>,
}

/// One loaded, ready-to-run speech model. The worker loop in this module
/// only ever sees this trait.
pub(crate) trait SpeechModel {
    /// Run one inference over already-conditioned 16 kHz mono audio.
    fn transcribe(
        &mut self,
        pcm_16k: &[f32],
        running: &Arc<AtomicBool>,
    ) -> Result<Transcript>;

    /// First-load readiness log line (full config summary).
    fn describe(&self) -> String;

    /// Reload readiness log line, logged after an auto-unload/resume cycle
    /// instead of `describe()` (which would re-print the full config).
    fn describe_reload(&self) -> String;

    /// Short engine name for the shared load/unload log lines this module
    /// prints ("{name} model unloaded...", "reloading {name} model...").
    fn engine_name(&self) -> &'static str;

    /// Samples this engine actually decoded for a conditioned utterance of
    /// `conditioned_len` samples, accounting for any internal short-utterance
    /// padding, so RTF and the `--metrics` `audio_sec` column reflect what
    /// was fed to the model rather than the raw utterance length.
    fn decoded_sample_count(&self, conditioned_len: usize) -> usize;
}

/// Load the model `ModelRef` names. Called once per load/reload cycle by
/// `AsrWorker::load_once`.
pub(crate) fn load_model(model: &ModelRef, opts: &AsrOptions) -> Result<Box<dyn SpeechModel>> {
    Ok(Box::new(crate::parakeet::ParakeetModel::load(&model.dir, opts)?))
}

/// Runs `f` while ticking `"loading {label}... Ns"` once per second on a
/// side thread. Model construction can block for tens of seconds with no
/// progress callback (a Parakeet fp32 encoder), so this exists to keep that
/// hung. Joins the ticker before returning `f`'s result.
pub(crate) fn with_loading_ticker<T>(label: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let load_done = Arc::new(AtomicBool::new(false));
    let ticker = {
        let done = Arc::clone(&load_done);
        let label = label.to_string();
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut last_printed: u64 = 0;
            loop {
                // Short sleeps so joining after the load finishes costs at
                // most 100 ms; only a whole-second tick is printed.
                std::thread::sleep(Duration::from_millis(100));
                if done.load(Ordering::Relaxed) {
                    break;
                }
                let elapsed = start.elapsed().as_secs();
                if elapsed > last_printed {
                    last_printed = elapsed;
                    log::info!("loading {label}... {elapsed}s");
                }
            }
        })
    };
    let result = f();
    load_done.store(true, Ordering::Relaxed);
    let _ = ticker.join();
    result
}

/// Tunables the ASR worker needs, resolved once from `Cli`.
/// Emission policy (`dump_dir`, clipboard, paste) lives with the
/// `Emitter`, not here.
#[derive(Debug, Clone)]
pub(crate) struct AsrOptions {
    pub(crate) model_spec: String,
    pub(crate) gate: GateConfig,
    pub(crate) n_threads: i32,
    pub(crate) use_gpu: bool,
    /// Custom vocabulary for post-gate correction (see `dictionary`).
    /// Empty = off. Loaded once at startup, fixed for the process lifetime.
    pub(crate) dictionary: Vec<String>,
    /// Strip hesitation/filler words in post-gate cleaning (see `clean`).
    /// Off by default.
    pub(crate) remove_hesitations: bool,
    /// Append per-utterance JSON metrics here when `Some` (see `--metrics`).
    /// Opened by the worker at startup; open errors only warn, never fail
    /// transcription.
    pub(crate) metrics_path: Option<PathBuf>,
    /// Shed oldest-first past `ASR_SHED_KEEP` (microphone mode). File mode
    /// sets false: offline decoding must be lossless.
    pub(crate) shed_stale: bool,
}

/// Backlog past which the loaded loop sheds the weakest utterances (see
/// `shed_survivors`): live speech stays fresh instead of transcribing
/// minutes-late noise.
pub(crate) const ASR_SHED_KEEP: usize = 4;

/// Bounded `rtrb` capacity for the utterance queue. utterances are seconds of
/// audio; 16 slots is ample (shedding in the worker keeps the backlog at
/// `ASR_SHED_KEEP`), and a full ring drops the newest with a warn rather
/// than blocking the processing loop.
pub(crate) const ASR_QUEUE_CAP: usize = 16;

/// Query physical cores for `--threads 0` (auto). Falls back to
/// `available_parallelism` and then `DEFAULT_THREADS` when the query fails.
pub(crate) fn auto_threads() -> i32 {
    pick_auto_threads(
        num_cpus::get_physical(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    )
}

/// Pick an auto thread count from a physical-core count and a logical-core
/// count (`0` = query failed). Prefers physical (ORT `intra_op` guidance:
/// one thread per physical core); falls back to logical, then
/// `DEFAULT_THREADS`. Pure over its inputs so tests don't touch the
/// scheduler. Always >= 1.
pub(crate) fn pick_auto_threads(physical: usize, logical: usize) -> i32 {
    let n = if physical > 0 {
        physical
    } else if logical > 0 {
        logical
    } else {
        DEFAULT_THREADS.max(1) as usize
    };
    (n.min(i32::MAX as usize) as i32).max(1)
}

/// Resolve a `--threads` value: `0` means auto, anything else is used as-is.
/// Pure over its inputs so tests don't touch the scheduler.
pub(crate) fn resolve_n_threads(requested: i32, auto: i32) -> i32 {
    if requested == 0 {
        auto.max(1)
    } else {
        requested
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_asr_worker(
    rx: Consumer<Utterance>,
    model: ModelRef,
    opts: AsrOptions,
    emitter: impl Emitter + Send + 'static,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    auto_unload: bool,
    auto_unload_after_secs: u64,
    sound_enabled: bool,
    autopause_enabled: bool,
) -> (
    JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("asr".into())
        .spawn(move || {
            let mut emitter = emitter;
            let mut rx = rx;
            if let Err(e) = asr_loop(
                &mut rx,
                &model,
                &opts,
                ready_tx,
                &mut emitter,
                &paused,
                &running,
                auto_unload,
                auto_unload_after_secs,
                sound_enabled,
                autopause_enabled,
            ) {
                log::error!("asr worker failed: {e:?}");
            }
        })
        .expect("failed to spawn asr thread");
    (handle, ready_rx)
}

#[allow(clippy::too_many_arguments)]
fn asr_loop(
    rx: &mut Consumer<Utterance>,
    model: &ModelRef,
    opts: &AsrOptions,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &mut impl Emitter,
    paused: &Arc<AtomicBool>,
    running: &Arc<AtomicBool>,
    auto_unload: bool,
    auto_unload_after_secs: u64,
    sound_enabled: bool,
    autopause_enabled: bool,
) -> Result<()> {
    let mut worker = AsrWorker {
        rx,
        model_ref: model,
        opts,
        ready: &ready,
        emitter,
        paused,
        running,
        auto_unload,
        unload_after: Duration::from_secs(auto_unload_after_secs),
        sound_enabled,
        autopause_enabled,
        // Resolved once here, not per utterance: the gate thresholds, the
        // dictionary and the cleaning policy are fixed for the session.
        gate: opts.gate.clone(),
        dictionary: opts.dictionary.clone(),
        remove_hesitations: opts.remove_hesitations,
        // Observability only: a broken metrics path must never take down
        // transcription, so open errors warn once and disable the log.
        metrics: open_metrics_log(opts.metrics_path.as_deref()),
        first_load: true,
        utterance_no: 0,
        prev_emit_text: String::new(),
        stash: Vec::new(),
        was_paused: false,
        last_unpaused: Instant::now(),
    };
    // `AsrWorker::load_once` signals the *first* load outcome via `ready`
    // itself (later reloads only log); an `Err` return here therefore always
    // means initial init failed, and re-sending is a harmless duplicate.
    match worker.run() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            Err(e)
        }
    }
}

/// Whether the model should be unloaded right now: auto-unload enabled and
/// paused continuously for at least `unload_after` (`0` unloads immediately
/// on pause). Pure over its inputs so tests don't need a GPU or threads.
pub(crate) fn should_unload(
    enabled: bool,
    paused: bool,
    paused_for: Duration,
    unload_after: Duration,
) -> bool {
    if !enabled || !paused {
        return false;
    }
    paused_for >= unload_after
}

/// Whether a just-finished utterance should re-pause the pipeline:
/// single-shot resume fires only on an actually emitted transcript, never
/// on a drop or a failed inference, so noise can't steal the shot.
/// Pure over its inputs for tests.
pub(crate) fn should_autopause(autopause_enabled: bool, emitted: bool) -> bool {
    autopause_enabled && emitted
}

/// Whether the worker should drain the ASR queue right now: exactly on
/// the false->true edge of `paused` (`was_paused` is the pause state as of
/// the last time the caller checked it). Pure so the edge condition is
/// testable without a live `rtrb` queue or thread; the actual drain (`rx.pop`
/// in a loop) stays a side effect in the caller.
///
/// Utterances queued just before a pause is noticed must not survive it:
/// with `--copy`/`--paste` enabled, transcribing them on resume types
/// stale content into whatever window has focus, possibly minutes later.
/// Firing only on the edge (not on every paused tick) means a long pause
/// drains once, not on a busy-loop.
pub(crate) fn should_drain_on_pause(was_paused: bool, now_paused: bool) -> bool {
    now_paused && !was_paused
}

/// Drain all queued utterances, returning how many were dropped. Pause-edge
/// and paused-shutdown discards share this so stale utterances never survive
/// a pause into a later emission.
fn drain_queue(rx: &mut Consumer<Utterance>) -> usize {
    let mut n = 0;
    while rx.pop().is_ok() {
        n += 1;
    }
    n
}

/// Fire single-shot autopause after an utterance the gate emitted: set the
/// pause flag and the pause-edge state, log, and cue the paused sound.
/// Drops and failed inferences never fire, so noise cannot steal the shot.
fn fire_autopause(
    autopause_enabled: bool,
    emitted: bool,
    paused: &Arc<AtomicBool>,
    was_paused: &mut bool,
    utterance_no: u64,
    sound_enabled: bool,
) {
    if should_autopause(autopause_enabled, emitted) {
        paused.store(true, Ordering::Release);
        *was_paused = true;
        log::info!("autopaused after utt_{utterance_no:06} (--autopause, single-shot)");
        if sound_enabled {
            crate::sound::play_paused();
        }
    }
}

/// Outcome of one `load_once` attempt.
enum Loaded {
    Model(Box<dyn SpeechModel>),
    /// A reload failed and was already logged; the outer loop retries.
    Retry,
}

/// The ASR worker's own state, so the loop reads as its phases (load, replay,
/// transcribe, unload, wait) instead of one long function. Everything here is
/// worker-thread local; the only shared state is the `paused` and `running`
/// flags.
struct AsrWorker<'a> {
    rx: &'a mut Consumer<Utterance>,
    model_ref: &'a ModelRef,
    opts: &'a AsrOptions,
    ready: &'a std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &'a mut dyn Emitter,
    paused: &'a Arc<AtomicBool>,
    running: &'a Arc<AtomicBool>,
    auto_unload: bool,
    /// `--auto-unload-after` as a duration.
    unload_after: Duration,
    sound_enabled: bool,
    autopause_enabled: bool,
    /// Resolved once from `opts` at startup, then fixed.
    gate: GateConfig,
    dictionary: Vec<String>,
    remove_hesitations: bool,
    /// `--metrics` JSONL writer, disabled on any open/write error.
    metrics: Option<std::io::BufWriter<std::fs::File>>,
    first_load: bool,
    /// Tag counter, owned here so numbering stays monotonic across reloads.
    utterance_no: u64,
    /// Previous emission's final text for split-seam dedup (see
    /// `remove_split_overlap`): only emits update it, drops never do, and it
    /// survives reloads so a split prefix is not lost to an unload racing
    /// the continuation.
    prev_emit_text: String,
    /// Utterances that arrive while the model is unloaded (resume raced the
    /// reload, or shutdown raced it): replayed in order after the next load.
    /// In practice this stays empty while paused, since the main loop
    /// discards audio instead of sending utterances.
    stash: Vec<Utterance>,
    /// Pause flag as of the last check, and when the model was last loaded
    /// unpaused (the auto-unload clock). Both are per-load state, reset by
    /// `load_once`.
    was_paused: bool,
    last_unpaused: Instant,
}

impl AsrWorker<'_> {
    /// Outer loop: one loaded model per iteration. The model is created and
    /// dropped inside the iteration (no self-referential struct), so unload
    /// is a plain scope exit that frees VRAM, and reload is the next
    /// iteration. All load/unload runs on this one thread, synchronously, so
    /// fast pause/resume toggles collapse to whatever the flag reads at the
    /// poll interval — a second load can never start mid-load.
    fn run(&mut self) -> Result<()> {
        loop {
            let mut model = match self.load_once() {
                Ok(Loaded::Model(m)) => m,
                Ok(Loaded::Retry) => continue,
                Err(e) => return Err(e),
            };
            // `true` = the producer is gone, so shut down.
            if self.run_loaded(model.as_mut()) {
                return Ok(());
            }
            let engine_name = model.engine_name();
            // Scope exit frees VRAM (Parakeet: ORT sessions).
            drop(model);
            log::info!("{engine_name} model unloaded (paused, VRAM freed)");
            if self.wait_unloaded(engine_name) {
                return Ok(());
            }
        }
    }

    /// Load one model instance and announce it (full summary on the first
    /// load, the short line on every reload).
    ///
    /// A failed *first* load is fatal — captures would go into a dead queue,
    /// so the error goes out on `ready` and ends the worker. A failed reload
    /// only warns and retries, so a transient disk error does not kill a
    /// running session. Also resets the per-load state: the auto-unload
    /// clock and the pause-edge tracker.
    fn load_once(&mut self) -> Result<Loaded> {
        let model = match load_model(self.model_ref, self.opts) {
            Ok(m) => m,
            Err(e) => {
                if self.first_load {
                    // Fail fast: captures would go into a dead queue.
                    // `asr_loop` re-sends as well (harmless duplicate).
                    let _ = self.ready.send(Err(format!("{e:#}")));
                    return Err(e);
                }
                log::warn!("model reload failed: {e:?}; retrying in 5s");
                std::thread::sleep(Duration::from_secs(5));
                return Ok(Loaded::Retry);
            }
        };
        if self.first_load {
            log::info!("{}", model.describe());
            // Init succeeded: signal ready so the main thread knows captures
            // aren't going into a dead queue. Emission policy (dump dir,
            // clipboard, paste, stdout) already resolved on the main thread
            // inside `SystemEmitter::new`, before spawn.
            let _ = self.ready.send(Ok(()));
            self.first_load = false;
        } else {
            log::info!("{}", model.describe_reload());
        }
        // Ready jingle ("speak now") only when actually unpaused: a first
        // load that finishes while `--start-paused` (or a reload that races
        // a re-pause) stays silent, and the resume edge below cues later.
        if self.sound_enabled && !self.paused.load(Ordering::Relaxed) {
            crate::sound::play_ready();
        }
        // A pause that raced the (tens of seconds) load gets a full fresh
        // timeout rather than an instant unload.
        self.last_unpaused = Instant::now();
        // Tracks the pause flag across ticks so a short-pause resume (model
        // never unloaded) still gets its ready jingle: the reload path above
        // only covers the long-pause case.
        self.was_paused = self.paused.load(Ordering::Relaxed);
        Ok(Loaded::Model(model))
    }

    /// Full-pipeline hold: transcribe what was stashed while the model was
    /// unloaded, in order, before the live loop starts.
    ///
    /// A re-pause mid-replay drops the rest of the stash outright, same as
    /// the live queue's pause-edge drain (`should_drain_on_pause`): stale
    /// audio from before an arbitrary-length pause must not silently wait for
    /// a future resume and then transcribe (and with --copy/--paste enabled,
    /// be typed into whatever window has focus).
    fn replay_stash(&mut self, model: &mut dyn SpeechModel) {
        let mut stash_iter = std::mem::take(&mut self.stash).into_iter();
        for utterance in stash_iter.by_ref() {
            if self.paused.load(Ordering::Relaxed) {
                // Re-paused mid-replay: stop here: the remainder is dropped
                // below.
                break;
            }
            if utterance.pcm.is_empty() {
                continue;
            }
            self.transcribe(model, utterance);
        }
        let dropped = stash_iter.count();
        if dropped > 0 {
            log::warn!("dropping {dropped} stashed utterance(s) (re-paused during reload replay)");
        }
    }

    /// One utterance through inference, gate and emission, then the
    /// single-shot autopause. A drop or a failed inference never fires the
    /// pause, so noise cannot steal the shot.
    fn transcribe(&mut self, model: &mut dyn SpeechModel, utterance: Utterance) {
        self.utterance_no += 1;
        let emitted = transcribe_and_emit(
            model,
            &utterance,
            self.utterance_no,
            &self.gate,
            self.running,
            &mut *self.emitter,
            &mut self.metrics,
            &self.dictionary,
            self.remove_hesitations,
            &mut self.prev_emit_text,
        );
        fire_autopause(
            self.autopause_enabled,
            emitted,
            self.paused,
            &mut self.was_paused,
            self.utterance_no,
            self.sound_enabled,
        );
    }

    /// Loaded loop: replay the stash, then transcribe until the pause-idle
    /// timeout asks for an unload (`false`), or the producer is gone
    /// (`true` = shut down).
    ///
    /// `rtrb` has no blocking recv: poll with a short sleep so idle CPU
    /// stays negligible while pause/unload checks run every ~20 ms
    /// instead of an old 1 s tick.
    fn run_loaded(&mut self, model: &mut dyn SpeechModel) -> bool {
        self.replay_stash(model);
        loop {
            // Full hold: while paused, idle without popping so queued
            // utterances wait for resume instead of transcribing through pause.
            if self.paused.load(Ordering::Relaxed) {
                if should_drain_on_pause(self.was_paused, true) {
                    // Pause edge: drop whatever is already queued. Without
                    // this, utterances closed just before the pause was
                    // noticed survive it and transcribe (and, with --copy/
                    // --paste enabled, get typed into whatever window has
                    // focus) on resume — possibly minutes later. Consistent
                    // with the shutdown-while-paused drain a few lines below,
                    // and with `hold_pipeline`'s own discard semantics on the
                    // capture side of this same edge.
                    let n = drain_queue(self.rx);
                    if n > 0 {
                        log::warn!("dropping {n} queued utterance(s) (paused)");
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
                self.was_paused = true;
                if self.should_unload_now(true) {
                    self.log_unload(model);
                    return false;
                }
                if self.rx.is_abandoned() {
                    // Shutdown while paused with a held queue: drop it (paused
                    // shutdown has nothing worth flushing) and exit.
                    let n = drain_queue(self.rx);
                    if n > 0 {
                        log::warn!("dropping {n} queued utterance(s) (shut down while paused)");
                    }
                    return true;
                }
                continue;
            }
            match self.rx.pop() {
                Ok(utterance) => {
                    self.last_unpaused = Instant::now();
                    self.ack_resume();
                    if utterance.pcm.is_empty() {
                        continue;
                    }
                    // Backlogged GPU: shed the weakest utterances (lowest
                    // density, then lowest rms) so the transcripts that
                    // follow are the best evidence, not minutes-late noise.
                    for utterance in self.take_backlog(utterance, model.engine_name()) {
                        if utterance.pcm.is_empty() {
                            continue;
                        }
                        self.transcribe(model, utterance);
                    }
                }
                Err(_) => {
                    if self.rx.is_abandoned() {
                        return true;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                    let now_paused = self.paused.load(Ordering::Relaxed);
                    // Only clear `was_paused` here (mirroring the `Ok` arm
                    // above); never set it true. The top-of-loop paused
                    // branch is the sole place a false->true transition runs
                    // the pause-edge queue drain (see its comment) — setting
                    // `was_paused = now_paused` unconditionally here used to
                    // race ahead of that check (this Err arm can observe the
                    // pause flag flip before the top-of-loop branch does),
                    // silently skipping the drain for that transition.
                    if !now_paused {
                        self.last_unpaused = Instant::now();
                        self.ack_resume();
                    }
                    if self.should_unload_now(now_paused) {
                        self.log_unload(model);
                        return false;
                    }
                }
            }
        }
    }

    /// Unloaded wait: never transcribe here, just stash late arrivals and
    /// reload eagerly the moment resume is seen — before the next utterance,
    /// so the first post-resume speech does not pay load latency twice.
    /// Returns `true` once the producer is gone, so the caller exits:
    /// abandoned + empty means shutdown while unloaded, and the stash is
    /// dropped (paused shutdown has nothing worth flushing) so the join
    /// does not hang.
    fn wait_unloaded(&mut self, engine_name: &str) -> bool {
        loop {
            match self.rx.pop() {
                Ok(utterance) => self.stash.push(utterance),
                Err(_) => {
                    if self.rx.is_abandoned() {
                        if !self.stash.is_empty() {
                            log::warn!(
                                "dropping {} queued utterance(s) (shut down while model unloaded)",
                                self.stash.len()
                            );
                        }
                        return true;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if !self.paused.load(Ordering::Relaxed) {
                log::info!("resumed, reloading {engine_name} model...");
                return false;
            }
        }
    }

    /// Auto-unload decision for this tick. `should_unload` itself stays a
    /// pure function so its edge cases are testable without a worker.
    fn should_unload_now(&self, now_paused: bool) -> bool {
        should_unload(
            self.auto_unload,
            now_paused,
            self.last_unpaused.elapsed(),
            self.unload_after,
        )
    }

    fn log_unload(&self, model: &dyn SpeechModel) {
        log::info!(
            "paused {}s, unloading {} model to free VRAM...",
            self.unload_after.as_secs(),
            model.engine_name()
        );
    }

    /// Ready jingle on the resume edge: a short-pause resume (model never
    /// unloaded) has no reload path to cue it.
    fn ack_resume(&mut self) {
        if self.sound_enabled && self.was_paused {
            crate::sound::play_ready();
        }
        self.was_paused = false;
    }

    /// What to transcribe for this arrival: just it, or the whole drained
    /// queue with the weakest shed first (see `shed_survivors`). Survivors
    /// stay in arrival order.
    fn take_backlog(&mut self, first: Utterance, engine_name: &str) -> Vec<Utterance> {
        if !self.opts.shed_stale || self.rx.slots() <= ASR_SHED_KEEP {
            return vec![first];
        }
        let mut backlog = vec![first];
        while let Ok(u) = self.rx.pop() {
            backlog.push(u);
        }
        let scores: Vec<(f32, f32)> = backlog
            .iter()
            .map(|u| (u.evidence.density(), u.evidence.rms))
            .collect();
        let keep = shed_survivors(&scores, ASR_SHED_KEEP);
        let shed = backlog.len() - keep.len();
        if shed > 0 {
            log::warn!(
                "shed {shed} low-quality utterance(s): {engine_name} outruns real time \
                 (kept {keep} best of {total})",
                keep = keep.len(),
                total = backlog.len(),
            );
        }
        let mut survivors = Vec::with_capacity(keep.len());
        for (i, u) in backlog.into_iter().enumerate() {
            if keep.contains(&i) {
                survivors.push(u);
            }
        }
        survivors
    }
}

/// Open the `--metrics` JSONL log for appending (creating parent dirs).
/// Observability only: errors are returned for the caller to warn-and-disable.
fn open_metrics_log(path: Option<&std::path::Path>) -> Option<std::io::BufWriter<std::fs::File>> {
    let path = path?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("metrics: failed to create {}: {e:?}", parent.display());
                return None;
            }
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => {
            log::info!(
                "metrics: appending per-utterance JSONL to {}",
                path.display()
            );
            Some(std::io::BufWriter::new(f))
        }
        Err(e) => {
            log::warn!(
                "metrics: failed to open {}: {e:?} (metrics disabled)",
                path.display()
            );
            None
        }
    }
}

/// Machine-readable verdict for the `--metrics` log (human-readable
/// `DROPPED(...)` strings stay in the dump dir via `dump_reason`).
pub(crate) fn verdict_slug(verdict: crate::gate::TranscriptVerdict) -> &'static str {
    match verdict {
        crate::gate::TranscriptVerdict::Emit => "emit",
        crate::gate::TranscriptVerdict::DropEmpty => "drop:empty",
        crate::gate::TranscriptVerdict::Drop { kind, .. } => match kind {
            crate::gate::TranscriptDropKind::Hallucination => "drop:hallucination",
            crate::gate::TranscriptDropKind::Repetition => "drop:repetition",
        },
    }
}

/// Escape a string for embedding in JSON (quotes, backslashes, control
/// chars); non-ASCII passes through as UTF-8. Stdlib only (no serde).
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Inference seconds per audio second for one utterance: `(audio_sec,
/// rtf)`. `rtf > 1` means slower than realtime. Pure for tests; the same
/// pair feeds the `--metrics` row and the slower-than-realtime warn.
pub(crate) fn infer_rtf(samples: usize, infer_ms: f64) -> (f64, f64) {
    let audio_sec = samples as f64 / SAMPLE_RATE as f64;
    let rtf = if audio_sec > 0.0 {
        infer_ms / 1000.0 / audio_sec
    } else {
        0.0
    };
    (audio_sec, rtf)
}

/// Pick which backlog utterances survive shedding: top `keep` by
/// (density, rms), ties to newer arrivals; survivors return as
/// chronological indices. Pure for tests; the worker transcribes the
/// survivors in arrival order.
pub(crate) fn shed_survivors(scores: &[(f32, f32)], keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // Best first: higher density, then higher rms, then newer arrival.
    order.sort_by(|&a, &b| {
        let (da, ra) = scores[a];
        let (db, rb) = scores[b];
        db.total_cmp(&da)
            .then_with(|| rb.total_cmp(&ra))
            .then_with(|| b.cmp(&a))
    });
    order.truncate(keep.min(scores.len()));
    order.sort_unstable();
    order
}

/// One `--metrics` JSONL row for a transcribed utterance. Pure over its
/// inputs (formatting only) so tests don't touch the filesystem or GPU.
/// `infer_ms` covers inference alone (conditioning/gain excluded); `rtf` is
/// inference seconds per audio second (>1 = slower than realtime).
#[allow(clippy::too_many_arguments)]
pub(crate) fn format_metrics_row(
    tag: &str,
    samples: usize,
    evidence: &crate::utterance::Evidence,
    verdict: crate::gate::TranscriptVerdict,
    text: &str,
    infer_ms: f64,
) -> String {
    let (audio_sec, rtf) = infer_rtf(samples, infer_ms);
    format!(
        "{{\"tag\":\"{}\",\"samples\":{},\"audio_sec\":{:.3},\"infer_ms\":{:.1},\"rtf\":{:.3},\
         \"speech_frames\":{},\"total_frames\":{},\"density\":{:.3},\"rms\":{:.5},\
         \"verdict\":\"{}\",\"text\":\"{}\"}}",
        escape_json_string(tag),
        samples,
        audio_sec,
        infer_ms,
        rtf,
        evidence.speech_frames,
        evidence.total_frames,
        evidence.density(),
        evidence.rms,
        verdict_slug(verdict),
        escape_json_string(text),
    )
}

/// Append one row via `format_metrics_row`, flushing so `--stop`/Ctrl-C
/// keeps every transcribed utterance even on a hard kill right after.
/// Write errors disable the log with a single warning (never fail
/// transcription for observability).
#[allow(clippy::too_many_arguments)]
fn write_metrics_row(
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
    tag: &str,
    samples: usize,
    evidence: &crate::utterance::Evidence,
    verdict: crate::gate::TranscriptVerdict,
    text: &str,
    infer_ms: f64,
) {
    use std::io::Write;
    let Some(w) = metrics else { return };
    let row = format_metrics_row(tag, samples, evidence, verdict, text, infer_ms);
    if let Err(e) = writeln!(w, "{row}") {
        log::warn!("metrics: write failed: {e:?} (metrics disabled)");
        *metrics = None;
        return;
    }
    if let Err(e) = w.flush() {
        log::warn!("metrics: flush failed: {e:?} (metrics disabled)");
        *metrics = None;
    }
}

/// Post-gate transcript polish, in the one order that matters: dictionary
/// first (casing restore plus strict fuzzy repair), then `clean` (collapse
/// 3+ repeats, optionally strip fillers).
///
/// Only an `Emit` verdict reaches here, so a near-miss dictionary key cannot
/// resurrect boilerplate, and cleaning can only shorten what the gate
/// accepted. `confidences` gates the fuzzy dictionary step (see
/// `dictionary`); it is `None` when the engine reported no per-word scores.
/// Returns the polished text — empty (or all whitespace) means cleaning
/// removed everything and the caller must turn the verdict into
/// `DropEmpty` rather than emit empty text.
fn polish_transcript(
    tag: &str,
    text: &str,
    dictionary: &[String],
    confidences: Option<&std::collections::HashMap<String, f32>>,
    remove_hesitations: bool,
) -> String {
    let corrected: std::borrow::Cow<'_, str> = if dictionary.is_empty() {
        std::borrow::Cow::Borrowed(text)
    } else {
        let corrected = crate::dictionary::correct_transcription(text, dictionary, confidences);
        if corrected != text {
            log::debug!("{tag}: dictionary {text:?} -> {corrected:?}");
        }
        std::borrow::Cow::Owned(corrected)
    };
    let cleaned = crate::clean::clean_transcription(&corrected, remove_hesitations);
    if cleaned != corrected {
        log::debug!("{tag}: clean {corrected:?} -> {cleaned:?}");
    }
    cleaned
}

/// One utterance through conditioning, inference, gate and emission. Tag
/// numbering is owned by the caller so it stays monotonic across reloads.
/// Returns whether the transcript was actually emitted (vs dropped or
/// failed), so the caller can drive single-shot `--autopause` without
/// seeing the gate.
#[allow(clippy::too_many_arguments)]
fn transcribe_and_emit(
    model: &mut dyn SpeechModel,
    utterance: &Utterance,
    utterance_no: u64,
    gate: &GateConfig,
    running: &Arc<AtomicBool>,
    emitter: &mut dyn Emitter,
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
    dictionary: &[String],
    remove_hesitations: bool,
    prev_emit_text: &mut String,
) -> bool {
    let tag = format!("utt_{utterance_no:06}");

    // Levels already normalized frame-wise by `Agc` (VAD-gated, shared-gain);
    // no per-utterance static gain here — that path is deleted.
    let conditioned = condition_utterance(&utterance.pcm);
    // Trailing context past the last word so final tokens are not clipped
    // (dithered, not digital zero — see `INFERENCE_PAD_SAMPLES`). No leading
    // pad: the utterance already starts with pre-roll. The WAV
    // dump still uses raw `utterance.pcm`; metrics/RTF count what the model
    // actually decoded (padded length, plus the short-input floor via
    // `decoded_sample_count`).
    let padded = pad_trailing_with_dither(&conditioned, INFERENCE_PAD_SAMPLES);

    // Inference time only (conditioning + edge pad excluded): feeds the RTF
    // column of the `--metrics` JSONL log.
    let t0 = Instant::now();
    let result = model.transcribe(&padded, running);
    let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let decoded_samples = model.decoded_sample_count(padded.len());
    let (audio_sec, rtf) = infer_rtf(decoded_samples, infer_ms);
    log::info!(
        "{tag}: transcribe took {infer_ms:.0} ms ({audio_sec:.1} s audio, rtf={rtf:.2})"
    );
    if rtf > RTF_WARN_RTF {
        log::warn!(
            "{tag}: inference slower than realtime (rtf={rtf:.2}, {infer_ms:.0} ms for \
             {audio_sec:.1} s audio): consider a smaller --model, --threads 0, or GPU"
        );
    }
    match result {
        Ok(t) => {
            // Split-seam dedup before the gate: a continuation re-decodes the
            // boundary words, so strip the overlap vs the previous emission.
            // Borrowed unless the dedup actually changed something.
            let raw_trimmed = t.text.trim();
            let deduped: std::borrow::Cow<'_, str> =
                if utterance.split_continuation && !prev_emit_text.is_empty() {
                    let stripped = remove_split_overlap(prev_emit_text, raw_trimmed);
                    if stripped.len() != raw_trimmed.len() {
                        log::debug!(
                            "{tag}: split-seam dedup stripped {} chars",
                            raw_trimmed.len() - stripped.len()
                        );
                    }
                    std::borrow::Cow::Owned(stripped)
                } else {
                    std::borrow::Cow::Borrowed(raw_trimmed)
                };
            let deduped = deduped.trim();
            let mut verdict = gate.decide_transcript(deduped);
            let mut emitted = matches!(verdict, crate::gate::TranscriptVerdict::Emit);
            // Per-word confidences gate the fuzzy dictionary step. Built only
            // when they can be used: no dictionary, or no per-word scores from
            // the engine, means the ungated form.
            let confidences = if emitted && !dictionary.is_empty() {
                (!t.word_confidences.is_empty())
                    .then(|| crate::dictionary::confidence_map(&t.word_confidences))
            } else {
                None
            };
            let mut final_text = if emitted {
                polish_transcript(
                    &tag,
                    deduped,
                    dictionary,
                    confidences.as_ref(),
                    remove_hesitations,
                )
            } else {
                deduped.to_string()
            };
            if emitted && final_text.trim().is_empty() {
                // Cleaning removed everything (hesitations only with the flag
                // on): drop instead of emitting empty text, which would
                // clobber the clipboard.
                verdict = crate::gate::TranscriptVerdict::DropEmpty;
                emitted = false;
                final_text.clear();
            }
            let final_trimmed = final_text.trim();
            if emitted {
                *prev_emit_text = final_trimmed.to_string();
            }
            write_metrics_row(
                metrics,
                &tag,
                decoded_samples,
                &utterance.evidence,
                verdict,
                final_trimmed,
                infer_ms,
            );
            emitter.emit(Emission {
                tag: &tag,
                pcm: &utterance.pcm,
                text: final_trimmed,
                verdict,
                evidence: &utterance.evidence,
            });
            emitted
        }
        Err(e) => {
            // Shutdown skips in-flight inference via a pre-call running
            // check (see `ParakeetModel`): not a failure, just a fast exit.
            if !running.load(Ordering::Relaxed) {
                log::info!("{tag}: transcription aborted (shutdown)");
            } else {
                log::error!("{tag}: transcription failed: {e:?}");
            }
            false
        }
    }
}

/// Condition an utterance for the model: DC removal, short linear fades at
/// both ends, and peak safety to 0.99. Hum/rumble already removed upstream by
/// `Hp` (80 Hz biquad after resampling); levels already normalized by `Agc`.
///
/// Kept deliberately light: the 5 ms fades land in pre-roll/tail silence
/// rather than speech.
pub(crate) fn condition_utterance(pcm: &[f32]) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    // DC removal: subtract the mean so soundcard offset can't bias fades or
    // peak safety.
    let mean = pcm.iter().fold(0.0f64, |m, &s| m + s as f64) / pcm.len() as f64;
    let mut out: Vec<f32> = pcm.iter().map(|&s| s - mean as f32).collect();

    // Linear fades at both ends; shortened for very short utterances so the
    // ramps never overlap.
    let fade = CONDITION_FADE_SAMPLES.min(out.len() / 2);
    for i in 0..fade {
        let ramp = (i + 1) as f32 / (fade + 1) as f32;
        out[i] *= ramp;
        let j = out.len() - 1 - i;
        out[j] *= ramp;
    }

    // Peak safety: resampler overshoot can exceed 1.0; keep headroom.
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.99 {
        let scale = 0.99 / peak;
        for s in out.iter_mut() {
            *s *= scale;
        }
    }
    out
}

/// One deterministic dither stream: `count` samples at `+-1e-4` from an
/// xorshift32 generator (cheap, no extra dependency).
///
/// Both padding sites below share this stream, not their seed: the seed is
/// derived from the input length by each caller, so its padding stays stable
/// per input shape.
fn dither_samples(seed: u32, count: usize) -> impl Iterator<Item = f32> {
    let mut seed = seed;
    (0..count).map(move |_| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32 - 0.5) * 2e-4 // +-1e-4
    })
}

/// Pad sub-second audio with quiet dithered noise instead of exact digital
/// silence: a long run of true zero is a hallucination trigger for the mel
/// front-end, and a poor input for short-utterance handling in general.
/// Grows the input to `min_samples`; longer input is returned unchanged.
pub(crate) fn pad_with_dither(pcm_16k: &[f32], min_samples: usize) -> Vec<f32> {
    let missing = min_samples.saturating_sub(pcm_16k.len());
    let mut out = Vec::with_capacity(min_samples.max(pcm_16k.len()));
    out.extend_from_slice(pcm_16k);
    let seed = 0x9E37_79B9 ^ (pcm_16k.len() as u32).wrapping_add(1);
    out.extend(dither_samples(seed, missing));
    out
}

/// Words compared on each side for split-seam dedup (see
/// `remove_split_overlap`): mirrors Murmure's `MERGE_WINDOW_WORDS`.
pub(crate) const MERGE_WINDOW_WORDS: usize = 6;

/// Append `pad_samples` of quiet dither after conditioned audio
/// (see `config::INFERENCE_PAD_SAMPLES` for why dither, not zero).
/// Empty input stays empty (callers skip empty utterances before here).
pub(crate) fn pad_trailing_with_dither(pcm_16k: &[f32], pad_samples: usize) -> Vec<f32> {
    if pcm_16k.is_empty() || pad_samples == 0 {
        return pcm_16k.to_vec();
    }
    let mut out = Vec::with_capacity(pcm_16k.len() + pad_samples);
    out.extend_from_slice(pcm_16k);
    let seed = 0x9E37_79B9 ^ (pcm_16k.len() as u32).wrapping_mul(0x85EB_CA6B);
    out.extend(dither_samples(seed, pad_samples));
    out
}

/// Strip text repeated across a split seam: longest normalized word overlap
/// (up to `MERGE_WINDOW_WORDS`) between the tail of `prev` and the head of
/// `cur` is dropped from `cur`. Comparison reuses `gate::normalize_transcript`
/// (lowercase + Polish fold + punctuation spacing), so `marché/marche` and
/// case differences still match. Returns `cur` trimmed when there is no
/// overlap, empty, or either side is empty.
pub(crate) fn remove_split_overlap(prev: &str, cur: &str) -> String {
    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let cur_words: Vec<&str> = cur.split_whitespace().collect();
    if prev_words.is_empty() || cur_words.is_empty() {
        return cur.trim().to_string();
    }
    let max_overlap = MERGE_WINDOW_WORDS
        .min(prev_words.len())
        .min(cur_words.len());
    // Normalize each word once instead of inside the window loop: the loop
    // looks at up to `MERGE_WINDOW_WORDS` windows of up to
    // `MERGE_WINDOW_WORDS` words, which re-normalized the same words over and
    // over. The comparison (and therefore the result) is unchanged.
    let normalized = |words: &[&str]| -> Vec<String> {
        words
            .iter()
            .map(|w| crate::gate::normalize_transcript(w))
            .collect()
    };
    let prev_norm = normalized(&prev_words);
    let cur_norm = normalized(&cur_words);
    let mut matched = 0;
    for len in (1..=max_overlap).rev() {
        let prev_tail = &prev_norm[prev_norm.len() - len..];
        let cur_head = &cur_norm[..len];
        if prev_tail == cur_head {
            matched = len;
            break;
        }
    }
    cur_words[matched..].join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS};
    use crate::gate::{is_hallucination, is_repetition};
    use crate::utterance::Evidence;

    #[test]
    fn verdict_slug_covers_every_disposition() {
        use crate::gate::{TranscriptDropKind, TranscriptVerdict};
        assert_eq!(verdict_slug(TranscriptVerdict::Emit), "emit");
        assert_eq!(verdict_slug(TranscriptVerdict::DropEmpty), "drop:empty");
        for (kind, slug) in [
            (TranscriptDropKind::Hallucination, "drop:hallucination"),
            (TranscriptDropKind::Repetition, "drop:repetition"),
        ] {
            assert_eq!(
                verdict_slug(TranscriptVerdict::Drop { kind }),
                slug
            );
        }
    }

    #[test]
    fn metrics_row_is_valid_json_with_rtf() {
        use crate::gate::TranscriptVerdict;
        let evidence = Evidence {
            speech_frames: 100,
            total_frames: 200,
            rms: 0.02,
        };
        // 16_000 samples = 1.0 s audio, 250 ms inference -> rtf 0.25.
        let row = format_metrics_row(
            "utt_000001",
            16_000,
            &evidence,
            TranscriptVerdict::Emit,
            "Dzień dobry",
            250.0,
        );
        assert!(row.contains("\"tag\":\"utt_000001\""), "{row}");
        assert!(row.contains("\"samples\":16000"), "{row}");
        assert!(row.contains("\"audio_sec\":1.000"), "{row}");
        assert!(row.contains("\"infer_ms\":250.0"), "{row}");
        assert!(row.contains("\"rtf\":0.250"), "{row}");
        assert!(row.contains("\"verdict\":\"emit\""), "{row}");
        assert!(row.contains("\"text\":\"Dzień dobry\""), "{row}");
        assert!(row.starts_with('{') && row.ends_with('}'));
    }

    #[test]
    fn metrics_row_escapes_text_for_json() {
        use crate::gate::TranscriptVerdict;
        let evidence = Evidence {
            speech_frames: 20,
            total_frames: 100,
            rms: 0.005,
        };
        let row = format_metrics_row(
            "utt_000002",
            16_000,
            &evidence,
            TranscriptVerdict::DropEmpty,
            "quote \"here\"\nnewline",
            0.0,
        );
        assert!(
            row.contains("\"text\":\"quote \\\"here\\\"\\nnewline\""),
            "{row}"
        );
        assert_eq!(escape_json_string("zażółć"), "zażółć");
    }

    #[test]
    fn infer_rtf_divides_infer_by_audio() {
        // 16_000 samples = 1.0 s; 250 ms inference -> rtf 0.25.
        let (sec, rtf) = infer_rtf(16_000, 250.0);
        assert!((sec - 1.0).abs() < 1e-9);
        assert!((rtf - 0.25).abs() < 1e-9);
        // Empty audio never divides by zero.
        assert_eq!(infer_rtf(0, 100.0), (0.0, 0.0));
    }

    #[test]
    fn shed_survivors_keep_best_by_density_then_rms() {
        // 6 backlogged, keep 4: the two weakest densities drop, order stays
        // chronological.
        let scores = vec![
            (0.9, 0.02), // 0 keep
            (0.1, 0.05), // 1 drop (worst density)
            (0.8, 0.02), // 2 keep
            (0.2, 0.01), // 3 drop
            (0.7, 0.02), // 4 keep
            (0.6, 0.03), // 5 keep
        ];
        assert_eq!(shed_survivors(&scores, 4), vec![0, 2, 4, 5]);
        // No backlog: everything survives.
        assert_eq!(shed_survivors(&scores, 10), vec![0, 1, 2, 3, 4, 5]);
        assert!(shed_survivors(&scores, 0).is_empty());
    }

    #[test]
    fn shed_survivors_break_ties_toward_newer() {
        // Identical quality: the newest two of three survive.
        let scores = vec![(0.5, 0.02), (0.5, 0.02), (0.5, 0.02)];
        assert_eq!(shed_survivors(&scores, 2), vec![1, 2]);
        // Same density: higher rms wins.
        let scores = vec![(0.5, 0.01), (0.5, 0.03)];
        assert_eq!(shed_survivors(&scores, 1), vec![1]);
    }

    fn default_gate() -> GateConfig {
        GateConfig::new(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY)
    }

    #[test]
    fn condition_removes_dc_offset() {
        // Soundcard offset must not reach the model: a constant 0.1 offset
        // carries no speech and would bias gain.
        let pcm = vec![0.1f32; 1600];
        let out = condition_utterance(&pcm);
        assert_eq!(out.len(), pcm.len());
        let mean = out.iter().fold(0.0f64, |m, &s| m + s as f64) / out.len() as f64;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn condition_keeps_speech_band_intact() {
        // 1 kHz tone (speech band) must pass nearly untouched in the middle.
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let out = condition_utterance(&tone);
        let rms = |v: &[f32]| {
            (v.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let before = rms(&tone[2000..14_000]);
        let after = rms(&out[2000..14_000]);
        let ratio = after / before;
        assert!((0.9..=1.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn condition_fades_edges_and_caps_peak() {
        // Flat loud input: fades pull both ends to near zero, peak safety
        // caps the middle at 0.99.
        let pcm = vec![2.0f32; 1600];
        let out = condition_utterance(&pcm);
        assert_eq!(out.len(), pcm.len());
        assert!(out[0].abs() < 0.05, "first={}", out[0]);
        assert!(
            out[out.len() - 1].abs() < 0.05,
            "last={}",
            out[out.len() - 1]
        );
        assert!(out.iter().all(|&s| s.abs() <= 0.99 + 1e-6));
        // Short input: fades shorten instead of overlapping.
        let short = vec![0.5f32; 10];
        let out_short = condition_utterance(&short);
        assert_eq!(out_short.len(), short.len());
    }

    #[test]
    fn condition_empty_stays_empty() {
        assert!(condition_utterance(&[]).is_empty());
    }

    #[test]
    fn hum_rejection_lives_upstream_in_hp() {
        // 50 Hz hum is removed by `Hp` (80 Hz biquad after resampling), not by
        // `condition_utterance` (DC/fades/peak only). A hum-heavy clip keeps
        // its level here; the pipeline relies on Hp + Agc instead of a static
        // per-utterance boost.
        let hum: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let out = condition_utterance(&hum);
        assert_eq!(out.len(), hum.len());
    }

    #[test]
    fn hallucination_filter_catches_known_boilerplate() {
        assert!(is_hallucination(
            "Napisy stworzone przez społeczność Amara.org"
        ));
        assert!(is_hallucination("  napisy  "));
        assert!(is_hallucination("..."));
        assert!(is_hallucination(""));
        // Observed forced-input decodes: white noise (rms=0.008) at
        // avg_p=0.890 and 50 Hz hum at avg_p=0.993 — both clear either
        // confidence floor, so the pattern list is the only gate.
        assert!(is_hallucination("Wszelkie prawa."));
        assert!(is_hallucination("Wszelkie prawa zastrzeżone."));
    }

    #[test]
    fn hallucination_filter_handles_diacritics() {
        // "Dziękuje za uwagę." (log-observed hallucination) differs from the
        // pattern "dziękuję za uwagę" by a single ogonek; without diacritic
        // folding this exact-match filter let it straight through.
        assert!(is_hallucination("Dziękuje za uwagę."));
        assert!(is_hallucination("dziekuje za uwage"));
    }

    #[test]
    fn repetition_filter_catches_decode_loops() {
        // Observed production-shape leak: 300 ms noise burst -> 26 speech
        // frames -> "Współpraca w ..." at avg_p=0.741, clear of the 0.6
        // weak floor and absent from the boilerplate list.
        assert!(is_repetition(
            "Współpraca w Współpraca w Współpraca w Współpraca w Współpraca."
        ));
        assert!(is_repetition("tak tak tak tak tak tak"));
    }

    #[test]
    fn repetition_filter_leaves_real_speech_alone() {
        assert!(!is_repetition(
            "Zrób commit i wypchnij zmiany do repozytorium."
        ));
        assert!(!is_repetition(
            "Dziękuję, że przyszedłeś na spotkanie o dziewiątej."
        ));
        // Short emphasis is exempt (< 6 words).
        assert!(!is_repetition("nie nie nie"));
        assert!(!is_repetition("tak"));
    }

    #[test]
    fn hallucination_filter_leaves_real_sentences_alone() {
        assert!(!is_hallucination(
            "Zrób commit i wypchnij zmiany do repozytorium."
        ));
        assert!(!is_hallucination(
            "Dziękuję, że przyszedłeś na spotkanie o dziewiątej."
        ));
    }

    #[test]
    fn unload_predicate_requires_enabled_pause_and_timeout() {
        // Auto-unload disabled never unloads, even when paused past timeout.
        assert!(!should_unload(
            false,
            true,
            Duration::from_secs(3600),
            Duration::from_secs(180)
        ));
        assert!(!should_unload(
            false,
            true,
            Duration::from_secs(3600),
            Duration::ZERO
        ));
        // Unpaused silence never unloads.
        assert!(!should_unload(
            true,
            false,
            Duration::from_secs(3600),
            Duration::from_secs(180)
        ));
        // Paused but under the timeout.
        assert!(!should_unload(
            true,
            true,
            Duration::from_secs(179),
            Duration::from_secs(180)
        ));
        // Boundary unloads.
        assert!(should_unload(
            true,
            true,
            Duration::from_secs(180),
            Duration::from_secs(180)
        ));
        assert!(should_unload(
            true,
            true,
            Duration::from_secs(3600),
            Duration::from_secs(180)
        ));
        // Zero timeout with auto-unload enabled unloads immediately on pause.
        assert!(should_unload(true, true, Duration::ZERO, Duration::ZERO));
        assert!(!should_unload(true, false, Duration::ZERO, Duration::ZERO));
    }

    #[test]
    fn autopause_fires_only_on_emit_when_enabled() {
        // Disabled never fires, even on emit.
        assert!(!should_autopause(false, true));
        assert!(!should_autopause(false, false));
        // Enabled fires only when the transcript was actually emitted;
        // drops and failed inferences (false) must not steal the shot.
        assert!(!should_autopause(true, false));
        assert!(should_autopause(true, true));
    }

    #[test]
    fn drain_on_pause_fires_only_on_the_edge() {
        // The transition into pause (edge) drains once...
        assert!(should_drain_on_pause(false, true));
        // ...but a tick that's already paused (no fresh edge) must not
        // re-drain on every 50 ms poll of a long pause.
        assert!(!should_drain_on_pause(true, true));
        // Not paused: nothing to drain either way.
        assert!(!should_drain_on_pause(false, false));
        assert!(!should_drain_on_pause(true, false));
    }

    #[test]
    fn thread_count_zero_means_auto() {
        assert_eq!(DEFAULT_THREADS, 8);
        assert_eq!(resolve_n_threads(0, 16), 16);
        assert_eq!(resolve_n_threads(8, 16), 8);
        // Auto of 0/negative can never happen via `auto_threads`, but clamp
        // defensively so the model never gets 0 threads.
        assert_eq!(resolve_n_threads(0, 0), 1);
        assert!(auto_threads() >= 1);
        // Physical preferred; logical fallback; then DEFAULT_THREADS.
        assert_eq!(pick_auto_threads(6, 12), 6);
        assert_eq!(pick_auto_threads(0, 12), 12);
        assert_eq!(pick_auto_threads(0, 0), DEFAULT_THREADS);
    }

    #[test]
    fn asr_worker_signals_init_failure() {
        // Bad model dir must propagate via the ready channel instead of
        // silently capturing into a dead queue.
        let (_tx, rx) = rtrb::RingBuffer::new(ASR_QUEUE_CAP);
        let opts = AsrOptions {
            model_spec: "bogus".to_string(),
            gate: default_gate(),
            n_threads: DEFAULT_THREADS,
            use_gpu: false,
            metrics_path: None,
            shed_stale: false,
            dictionary: Vec::new(),
            remove_hesitations: false,
        };
        let bad = ModelRef {
            dir: std::path::PathBuf::from("/nonexistent/parakeet-bogus"),
            assets: &[],
        };
        let (handle, ready) = spawn_asr_worker(
            rx,
            bad,
            opts,
            crate::emit::RecordingEmitter::new(),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            false,
            0,
            false,
            false,
        );
        match ready.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(Ok(())) => panic!("bad model path must not signal ready"),
            Ok(Err(e)) => assert!(
                e.contains("failed to load parakeet model"),
                "unexpected error: {e}"
            ),
            Err(e) => panic!("ready channel must send once, got {e:?}"),
        }
        handle.join().expect("worker thread must exit");
    }

    #[test]
    fn polish_applies_dictionary_then_clean_then_empty_check() {
        let dict = vec!["Kubernetes".to_string()];
        // Casing restore first, then the 3+ repeat collapse sees the
        // canonical form (not the raw one).
        assert_eq!(
            polish_transcript(
                "utt_000001",
                "kubernetes kubernetes kubernetes",
                &dict,
                None,
                false
            ),
            "Kubernetes Kubernetes"
        );
        // Filler stripping only when asked; whitespace is normalized either way.
        assert_eq!(
            polish_transcript("utt_000002", "  yyy   dzień  dobry ", &[], None, true),
            "dzień dobry"
        );
        assert_eq!(
            polish_transcript("utt_000003", "  yyy   dzień  dobry ", &[], None, false),
            "yyy dzień dobry"
        );
        // Cleaning can empty the transcript: the caller turns that into
        // DropEmpty, so this must not fall back to the input.
        assert_eq!(polish_transcript("utt_000004", "yyy eee", &[], None, true), "");
        assert_eq!(polish_transcript("utt_000005", "", &[], None, false), "");
    }

    #[test]
    fn trailing_pad_appends_dither_and_preserves_start() {
        use crate::config::INFERENCE_PAD_SAMPLES;
        let pcm = vec![0.5f32; 1_600];
        let padded = pad_trailing_with_dither(&pcm, INFERENCE_PAD_SAMPLES);
        assert_eq!(padded.len(), pcm.len() + INFERENCE_PAD_SAMPLES);
        assert_eq!(&padded[..pcm.len()], &pcm[..]);
        assert!(padded[pcm.len()..].iter().all(|&s| s.abs() <= 1e-4));
        assert!(padded[pcm.len()..].iter().any(|&s| s != 0.0));
    }

    #[test]
    fn trailing_pad_empty_and_zero_are_noop() {
        assert!(pad_trailing_with_dither(&[], 4000).is_empty());
        let pcm = vec![0.1f32; 100];
        assert_eq!(pad_trailing_with_dither(&pcm, 0), pcm);
    }

    #[test]
    fn min_pad_grows_short_audio_only_and_stays_deterministic() {
        use crate::config::PARAKEET_MIN_SAMPLES;
        let short = vec![0.25f32; 1_600];
        let padded = pad_with_dither(&short, PARAKEET_MIN_SAMPLES);
        assert_eq!(padded.len(), PARAKEET_MIN_SAMPLES);
        assert_eq!(&padded[..short.len()], &short[..]);
        assert!(padded[short.len()..].iter().all(|&s| s.abs() <= 1e-4));
        assert!(padded[short.len()..].iter().any(|&s| s != 0.0));
        // Same input, same dither stream.
        assert_eq!(pad_with_dither(&short, PARAKEET_MIN_SAMPLES), padded);
        // Already long enough: untouched, no pad added.
        let long = vec![0.25f32; PARAKEET_MIN_SAMPLES + 10];
        assert_eq!(pad_with_dither(&long, PARAKEET_MIN_SAMPLES), long);
    }

    #[test]
    fn split_overlap_strips_longest_seam() {
        assert_eq!(
            remove_split_overlap("bonjour le monde", "monde entier ici"),
            "entier ici"
        );
        assert_eq!(
            remove_split_overlap("je vais au marché", "au marché acheter du pain"),
            "acheter du pain"
        );
        assert_eq!(
            remove_split_overlap("podaj zolc", "żółć proszę"),
            "proszę"
        );
        assert_eq!(
            remove_split_overlap("Dzień dobry świecie", "DZIEŃ DOBRY ŚWIECIE i hej"),
            "i hej"
        );
    }

    #[test]
    fn split_overlap_no_match_and_edges() {
        assert_eq!(
            remove_split_overlap("bonjour le monde", "autre chose"),
            "autre chose"
        );
        assert_eq!(remove_split_overlap("", "bonjour"), "bonjour");
        assert_eq!(remove_split_overlap("bonjour", ""), "");
        assert_eq!(
            remove_split_overlap("je vais au marché", "au marché"),
            ""
        );
    }

    #[test]
    fn split_overlap_caps_at_window_size() {
        let prev = "zero un deux trois quatre cinq six";
        assert_eq!(remove_split_overlap(prev, "un deux trois quatre cinq six sept"), "sept");
        let long_prev = "un deux trois quatre cinq six sept";
        let long_cur = "un deux trois quatre cinq six sept huit";
        assert_eq!(remove_split_overlap(long_prev, long_cur), long_cur);
    }
}
