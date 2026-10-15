// ---------------------------------------------------------------------------
// ASR worker: the loop, gates, conditioning.
// ---------------------------------------------------------------------------
//
// This module owns the worker thread and its load/unload/pause/shed loop,
// the transcript gate call, audio conditioning, and the split-seam dedup. The
// engines sit behind `speech::SpeechModel` and own only what is genuinely
// engine-specific — the actual inference call, and enough of a description to
// log. The `--metrics` rows the worker writes are formatted by `metrics`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use rtrb::Consumer;

use crate::config::{CONDITION_FADE_SAMPLES, INFERENCE_PAD_SAMPLES, RTF_WARN_RTF, SAMPLE_RATE};
use crate::emit::{Emission, Emitter};
use crate::gate::GateConfig;
use crate::metrics::{infer_rtf, write_metrics_row, MetricsSlot};
use crate::model_store::ModelRef;
use crate::speech::{pad_trailing_with_dither, EngineConfig, SpeechModel};
use crate::translate::{TranslationJob, TranslationOut};
use crate::utterance::Utterance;

/// Load the model `ModelRef` names. Called once per load/reload cycle by
/// `AsrWorker::load_once`. Dispatches on the kind `resolve_model` classified,
/// so the engine choice is settled before any session is opened.
/// `boost_words` is the live dictionary at load time, so an engine that boosts
/// starts from the current list.
fn load_model(
    model: &ModelRef,
    config: &EngineConfig,
    boost_words: &[String],
) -> Result<Box<dyn SpeechModel>> {
    Ok(match model.kind {
        crate::model_store::EngineKind::Parakeet => Box::new(crate::parakeet::ParakeetModel::load(
            &model.dir,
            config,
            boost_words,
        )?) as Box<dyn SpeechModel>,
        crate::model_store::EngineKind::Whisper => {
            Box::new(crate::whisper::WhisperModel::load(&model.dir, config)?)
                as Box<dyn SpeechModel>
        }
    })
}

/// The ASR side of the live dictionary: a `control::DictionaryState` handle for the
/// per-utterance word list, plus a one-slot queue the D-Bus hook fills so the
/// engine boost automaton is rebuilt on the *worker* thread (the engine is
/// worker-thread local and must never be touched from the bus thread).
///
/// Ownership split: the hooked `control::DictionaryState` goes to `dbus::spawn`,
/// this value goes into `AsrConfig`. Both are cheap clones of the same state,
/// so a swap on the bus is visible to the next utterance either way.
pub(crate) struct LiveDictionary {
    shared: crate::control::DictionaryState,
    /// Newest word list reported by the hook, waiting for the worker to apply
    /// it. `None` = nothing pending, so the steady state costs one atomic-free
    /// mutex lock per utterance.
    boost: Arc<Mutex<Option<Arc<Vec<String>>>>>,
}

impl LiveDictionary {
    /// Attach the booster hook to a `control::DictionaryState` clone.
    ///
    /// Returns the hooked clone (hand it to `dbus::spawn`) and this value
    /// (put it in `AsrConfig::dictionary`). A swap then reaches the worker
    /// twice over the same list: through `words()` for post-correction, and
    /// through the queue for the engine boost.
    pub(crate) fn new(
        shared: crate::control::DictionaryState,
    ) -> (crate::control::DictionaryState, Self) {
        let boost: Arc<Mutex<Option<Arc<Vec<String>>>>> = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&boost);
        let shared = shared.with_words_hook(Arc::new(move |words| {
            // A poisoned lock must not kill a dictionary swap: take the
            // poisoned value and continue.
            *sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(words);
        }));
        (shared.clone(), Self { shared, boost })
    }

    /// Snapshot of the live word list. The caller owns one `Arc` per
    /// utterance; no lock is held across inference or post-correction.
    pub(crate) fn words(&self) -> Arc<Vec<String>> {
        self.shared.words()
    }

    /// Take the pending boost list, if a swap arrived since the last call.
    fn take_boost(&self) -> Option<Arc<Vec<String>>> {
        std::mem::take(&mut *self.boost.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Clone for LiveDictionary {
    fn clone(&self) -> Self {
        // A clone shares the queue with the original, so both see the same
        // pending swap. The engine has one owner (the worker), so the list is
        // applied exactly once there.
        Self {
            shared: self.shared.clone(),
            boost: Arc::clone(&self.boost),
        }
    }
}

impl std::fmt::Debug for LiveDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveDictionary")
            .field("words", &self.words().len())
            .finish()
    }
}

/// Latest-wins hand-off of the open utterance's audio from the capture loop to
/// the ASR worker, for partial transcripts.
///
/// One slot, not a queue: a partial is stale the moment newer audio exists, so
/// a model slower than the capture step just skips steps instead of building a
/// backlog. The capture side `put`s a snapshot and `clear`s the slot when the
/// utterance closes or the pipeline pauses; the worker `take`s one only when no
/// closed utterance waits, so a final is never delayed by more than one pass.
#[derive(Clone, Default, Debug)]
pub(crate) struct PartialSlot(Arc<Mutex<Option<Vec<f32>>>>);

impl PartialSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Vec<f32>>> {
        // A poisoned lock holds plain audio: take it and go on.
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Store `pcm`, replacing any partial the worker has not taken yet.
    pub(crate) fn put(&self, pcm: Vec<f32>) {
        *self.lock() = Some(pcm);
    }

    /// Take the waiting partial, if any.
    pub(crate) fn take(&self) -> Option<Vec<f32>> {
        self.lock().take()
    }

    /// Drop the waiting partial: its utterance has closed or been discarded.
    pub(crate) fn clear(&self) {
        *self.lock() = None;
    }
}

/// Everything the ASR worker needs, resolved once from `Cli` on the main thread
/// and owned by the worker from then on.
///
/// One carrier for the worker's configuration: the engines, the gates and the
/// queue policy all read it from here, and the worker holds no second copy of
/// any of it. Emission policy (`dump_dir`, clipboard, paste) is not here — it
/// belongs to the `Emitter`, which is built and owned separately.
#[derive(Debug, Clone)]
pub(crate) struct AsrConfig {
    /// Engine settings, read by `parakeet` / `whisper` at load.
    pub(crate) engine: EngineConfig,

    // ---- gates and cleaning ----
    /// Shared with the capture side, so both paths judge the same thresholds.
    pub(crate) gate: GateConfig,
    /// Strip hesitation/filler words in post-gate cleaning (see `clean`).
    /// Off by default.
    pub(crate) remove_hesitations: bool,
    /// Live custom vocabulary: the post-gate correction list plus the queue
    /// that hands swapped words to the loaded engine (see `LiveDictionary`).
    /// Empty = off. Shared with the D-Bus thread, so it changes at runtime.
    pub(crate) dictionary: LiveDictionary,

    // ---- queue policy ----
    /// Shed oldest-first past `ASR_SHED_KEEP` (microphone mode). File mode
    /// sets false: offline decoding must be lossless.
    pub(crate) shed_stale: bool,
    /// Where the capture loop leaves the open utterance for partial decoding.
    /// Shared with the capture side.
    pub(crate) partials: PartialSlot,

    // ---- observability ----
    // The `--metrics` writer itself is not here: `spawn_asr_worker` takes an
    // already-opened `MetricsSlot`, because the translation thread writes rows
    // too and one file cannot be opened twice.

    // ---- runtime policy ----
    /// `--auto-unload`: free VRAM while paused, reload eagerly on resume.
    pub(crate) auto_unload: bool,
    /// `--auto-unload-after` as a duration.
    pub(crate) auto_unload_after: Duration,
    /// Audible cues on pause, resume and when transcription is ready.
    pub(crate) sound: bool,
    /// `--autopause`: re-pause after each emitted transcript, single-shot.
    pub(crate) autopause: bool,
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

/// Start the ASR worker thread and hand back its handle plus a channel that
/// reports the *first* model load's outcome.
///
/// Six parameters, one of them the configuration: everything the worker needs
/// beyond its two shared flags arrives in `config`, resolved once on the main
/// thread. The worker owns it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_asr_worker(
    mut rx: Consumer<Utterance>,
    model: ModelRef,
    config: AsrConfig,
    mut emitter: impl Emitter + Send + 'static,
    translation: Option<TranslationOut>,
    metrics: MetricsSlot,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
) -> (
    JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("asr".into())
        .spawn(move || {
            let mut worker = AsrWorker::new(
                &mut rx,
                model,
                config,
                &ready_tx,
                &mut emitter,
                translation,
                metrics,
                &paused,
                &running,
            );
            // `load_once` signals the first load outcome via `ready` itself
            // (later reloads only log), so an `Err` here always means initial
            // init failed and re-sending is a harmless duplicate.
            if let Err(e) = worker.run() {
                log::error!("asr worker failed: {e:?}");
                let _ = ready_tx.send(Err(e.to_string()));
            }
        })
        .expect("failed to spawn asr thread");
    (handle, ready_rx)
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
/// on a drop or a failed inference, so noise can't steal the shot. An
/// utterance transcribed while the run is already held (the pause flush)
/// never re-pauses: the pipeline is already quiet, so firing would only
/// repeat the log line and the beep. Pure over its inputs for tests.
pub(crate) const fn should_autopause(
    autopause_enabled: bool,
    emitted: bool,
    already_paused: bool,
) -> bool {
    autopause_enabled && emitted && !already_paused
}

/// How long a queued utterance may wait and still be transcribed while a
/// pause is held (see `keep_while_paused`). The pause is noticed within one
/// loop iteration (~12 ms) and a normal decode takes well under a second, so
/// anything inside this window is text the speaker just said. Past it the
/// model is behind by seconds, which is what makes the text stale.
pub(crate) const PAUSE_KEEP_MAX_WAITED: Duration = Duration::from_secs(2);

/// Whether an utterance that reached the queue before the pause may still be
/// transcribed while the run is held. A pause flush always may: the capture
/// side cut it at this very edge, so it is the freshest audio there is. An
/// ordinary close may too while it is still fresh, which covers the sentence
/// that closed milliseconds before the pause was noticed — dropping that one
/// is the loss this whole path exists to prevent.
///
/// Everything older is a real backlog (the model runs slower than realtime,
/// or the run sat unloaded), and it stays dropped: with `--copy`/`--paste`
/// enabled, transcribing it later types stale content into whatever window
/// has focus, possibly minutes after it was said. Pure for tests.
pub(crate) fn keep_while_paused(pause_flush: bool, waited: Duration) -> bool {
    pause_flush || waited < PAUSE_KEEP_MAX_WAITED
}

/// Drain all queued utterances, returning how many were dropped. The
/// paused-shutdown discard uses this so stale utterances never survive into a
/// later emission.
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
    if should_autopause(autopause_enabled, emitted, paused.load(Ordering::Acquire)) {
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

/// Hand the pending dictionary swap to the engine, and report how many words
/// the engine received (`None` when no swap was pending).
///
/// This is the last hop of the live-dictionary path: a D-Bus mutator on
/// `DictionaryState` swaps the word list, the booster hook queues it here, and
/// the worker thread drains it into `SpeechModel::set_boost_words`, which
/// rebuilds the engine's `BoostTree`. The engine is worker-thread local, so the
/// queue exists to keep the bus thread from touching it.
///
/// Split out of `AsrWorker::apply_boost_words` so a test can drive the
/// delivery without building a worker.
fn deliver_boost_words(
    dictionary: &mut LiveDictionary,
    model: &mut dyn SpeechModel,
) -> Option<usize> {
    let words = dictionary.take_boost()?;
    log::info!(
        "dictionary: {} word(s) active (booster updated)",
        words.len()
    );
    model.set_boost_words(&words);
    Some(words.len())
}

/// The ASR worker's own state, so the loop reads as its phases (load, replay,
/// transcribe, unload, wait) instead of one long function. Everything here is
/// worker-thread local; the only shared state is the `paused` and `running`
/// flags.
struct AsrWorker<'a> {
    rx: &'a mut Consumer<Utterance>,
    model_ref: ModelRef,
    /// The one copy of the worker's configuration. Everything the loop, the
    /// gates and the engines need is read from here.
    config: AsrConfig,
    ready: &'a std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &'a mut dyn Emitter,
    /// The translation step, owned by the worker like the engine. `None` (file
    /// mode without `--translate`) = the step cannot run at all and every utterance
    /// is emitted as transcribed.
    ///
    /// This is the *queue*, not the translator: the `ort` sessions live in the
    /// state the lane and the bus share, because a value that appears on the bus
    /// thread after the worker started could not reach an `Option` here. The worker
    /// only submits jobs, and only when the step will actually translate
    /// (`TranslationState::will_translate`), so a run that never translates keeps
    /// the inline path and this queue's drop-oldest policy never applies to it.
    translation: Option<TranslationOut>,
    paused: &'a Arc<AtomicBool>,
    running: &'a Arc<AtomicBool>,
    /// `--metrics` JSONL writer, shared with the translation thread. Disabled on
    /// any open/write error. Opened once before the worker starts (see
    /// `open_metrics_log`), so a broken path cannot fail transcription mid-run.
    metrics: MetricsSlot,
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

impl<'a> AsrWorker<'a> {
    /// The one construction of a worker: production and tests both come through
    /// here, so the per-load state below cannot be set up two different ways.
    #[allow(clippy::too_many_arguments)]
    fn new(
        rx: &'a mut Consumer<Utterance>,
        model_ref: ModelRef,
        config: AsrConfig,
        ready: &'a std::sync::mpsc::Sender<Result<(), String>>,
        emitter: &'a mut dyn Emitter,
        translation: Option<TranslationOut>,
        metrics: MetricsSlot,
        paused: &'a Arc<AtomicBool>,
        running: &'a Arc<AtomicBool>,
    ) -> Self {
        Self {
            rx,
            model_ref,
            config,
            ready,
            emitter,
            translation,
            paused,
            running,
            metrics,
            first_load: true,
            utterance_no: 0,
            prev_emit_text: String::new(),
            stash: Vec::new(),
            was_paused: false,
            last_unpaused: Instant::now(),
        }
    }
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
        let mut model = match load_model(
            &self.model_ref,
            &self.config.engine,
            &self.config.dictionary.words(),
        ) {
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
        // A swap that arrived while the model was unloaded (or before the
        // first load finished) is still queued: apply it now, so a reload
        // never starts with a stale boost automaton. No-op when nothing
        // changed, which is the normal case.
        self.apply_boost_words(model.as_mut());
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
        if self.config.sound && !self.paused.load(Ordering::Relaxed) {
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
    /// the live queue's stale drop (`keep_while_paused`): stale
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

    /// Apply a pending dictionary swap to the engine's boost automaton. Called
    /// before every inference and at every load, so a `ReplaceWords` /
    /// `LoadFile` takes effect on the next utterance whether the model
    /// is loaded or was just reloaded. Takes the list out of the queue first,
    /// so a swap racing a long inference is never dropped: the next
    /// `transcribe` picks it up.
    fn apply_boost_words(&mut self, model: &mut dyn SpeechModel) {
        deliver_boost_words(&mut self.config.dictionary, model);
    }

    /// One utterance through inference, gate and emission, then the
    /// single-shot autopause. A drop or a failed inference never fires the
    /// pause, so noise cannot steal the shot.
    fn transcribe(&mut self, model: &mut dyn SpeechModel, utterance: Utterance) {
        self.utterance_no += 1;
        // Live swap reaches the engine before it decodes this utterance.
        self.apply_boost_words(model);
        // One `Arc` per utterance, dropped at the end: post-correction sees a
        // consistent list, and no lock is held across inference.
        let words = self.config.dictionary.words();
        let emitted = transcribe_and_emit(
            model,
            &utterance,
            self.utterance_no,
            &self.config.gate,
            self.running,
            &mut *self.emitter,
            &mut self.translation,
            &self.metrics,
            &words,
            self.config.remove_hesitations,
            &mut self.prev_emit_text,
        );
        fire_autopause(
            self.config.autopause,
            emitted,
            self.paused,
            &mut self.was_paused,
            self.utterance_no,
            self.config.sound,
        );
    }

    /// One partial transcript of the still-open utterance: logged, nothing
    /// else. It does not take an utterance number, touch the split-seam text,
    /// reach the emitter or the metrics, or fire autopause — the final
    /// transcript of the same utterance does all of that when it closes.
    fn transcribe_partial(&mut self, model: &mut dyn SpeechModel, pcm: &[f32]) {
        self.apply_boost_words(model);
        // The number the utterance will get when it closes.
        let tag = format!("utt_{:06}", self.utterance_no + 1);
        if let Some(line) = partial_line(model, pcm, &tag, &self.config.gate, self.running) {
            log::info!("{line}");
        }
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
            // Full hold: while paused, take only what the pause itself
            // produced — the flush the capture side cut at this edge, plus
            // anything still fresh — and drop the rest.
            if self.paused.load(Ordering::Relaxed) {
                let mut fresh = Vec::new();
                let mut dropped = 0usize;
                while let Ok(utterance) = self.rx.pop() {
                    // Empty audio is skipped silently, as the unpaused arm
                    // below does: it is not a dropped sentence.
                    if utterance.pcm.is_empty() {
                        continue;
                    }
                    if keep_while_paused(utterance.pause_flush, utterance.queued_at.elapsed()) {
                        fresh.push(utterance);
                    } else {
                        dropped += 1;
                    }
                }
                if dropped > 0 {
                    // With --copy/--paste enabled, transcribing these on
                    // resume would type stale content into whatever window
                    // has focus, possibly minutes after it was said.
                    log::warn!("dropping {dropped} queued utterance(s) (paused)");
                }
                // Arrival order, and no autopause from here: `transcribe`
                // reads the pause flag and stays quiet (see
                // `should_autopause`).
                for utterance in fresh {
                    self.transcribe(model, utterance);
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
                    // Idle: decode the open utterance's newest snapshot if one
                    // is waiting, else nap. Only here, so a closed utterance in
                    // the queue always goes first.
                    match self.config.partials.take() {
                        Some(pcm) => self.transcribe_partial(model, &pcm),
                        None => std::thread::sleep(Duration::from_millis(20)),
                    }
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
            self.config.auto_unload,
            now_paused,
            self.last_unpaused.elapsed(),
            self.config.auto_unload_after,
        )
    }

    fn log_unload(&self, model: &dyn SpeechModel) {
        log::info!(
            "paused {}s, unloading {} model to free VRAM...",
            self.config.auto_unload_after.as_secs(),
            model.engine_name()
        );
    }

    /// Ready jingle on the resume edge: a short-pause resume (model never
    /// unloaded) has no reload path to cue it.
    fn ack_resume(&mut self) {
        if self.config.sound && self.was_paused {
            crate::sound::play_ready();
        }
        self.was_paused = false;
    }

    /// What to transcribe for this arrival: just it, or the whole drained
    /// queue with the weakest shed first (see `shed_survivors`). Survivors
    /// stay in arrival order.
    fn take_backlog(&mut self, first: Utterance, engine_name: &str) -> Vec<Utterance> {
        if !self.config.shed_stale || self.rx.slots() <= ASR_SHED_KEEP {
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

/// Decode one partial and return its log line, or `None` when there is nothing
/// worth showing: an empty or gated transcript (hallucination boilerplate,
/// a repetition loop) or a failed decode. Conditioned and padded exactly like a
/// final (`transcribe_and_emit`), so a partial reads the audio the same way.
fn partial_line(
    model: &mut dyn SpeechModel,
    pcm: &[f32],
    tag: &str,
    gate: &GateConfig,
    running: &Arc<AtomicBool>,
) -> Option<String> {
    let conditioned = condition_utterance(pcm);
    let padded = pad_trailing_with_dither(&conditioned, INFERENCE_PAD_SAMPLES);
    let t0 = Instant::now();
    let result = model.transcribe(&padded, running);
    let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(t) => {
            let text = t.text.trim();
            if !matches!(
                gate.decide_transcript(text),
                crate::gate::TranscriptVerdict::Emit
            ) {
                return None;
            }
            let audio_sec = pcm.len() as f64 / SAMPLE_RATE as f64;
            Some(format!(
                "{tag}: partial ({audio_sec:.1} s audio, {infer_ms:.0} ms): {text}"
            ))
        }
        Err(e) => {
            // Shutdown skips inference through the pre-call check: not a failure.
            if running.load(Ordering::Relaxed) {
                log::warn!("{tag}: partial failed: {e:?}");
            }
            None
        }
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
    translation: &mut Option<TranslationOut>,
    metrics: &MetricsSlot,
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
    log::info!("{tag}: transcribe took {infer_ms:.0} ms ({audio_sec:.1} s audio, rtf={rtf:.2})");
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
            // The split-seam dedup stays source-vs-source: it compares what was
            // said, which is what the gates judged, not what was translated.
            if emitted {
                *prev_emit_text = final_trimmed.to_string();
            }
            // The translation step runs last, after every gate, and only for an
            // utterance that is actually emitted: a drop is never translated.
            // `fallback_or_translated` owns the failure policy, so a failed or
            // empty translation emits the source text with a warn rather than
            // losing the utterance.
            //
            // With the step on, the job is handed to the translation thread and
            // this function does not emit: the thread owns the translated text,
            // so it owns the emission too. The `pcm` copy is what the queue needs,
            // since the utterance is gone by the time the thread gets to it.
            //
            // `will_translate` is what keeps a run that never translates on the
            // inline path below: the lane exists from startup (a switch arrives on
            // the bus thread, which cannot reach into this worker), but an
            // utterance is only handed over when a translation will really happen.
            // So the lane's drop-oldest queue never applies to untranslated text.
            if let Some(out) = translation.as_ref() {
                if emitted && out.will_translate() {
                    out.enqueue(TranslationJob {
                        tag: tag.clone(),
                        pcm: utterance.pcm.clone(),
                        source: final_trimmed.to_string(),
                        verdict,
                        evidence: utterance.evidence,
                        samples: decoded_samples,
                        infer_ms,
                        // Stamped by `enqueue`, the only producer: a job cannot
                        // be queued without recording which translator was live.
                        generation: 0,
                    });
                    return true;
                }
            }
            // No step, or a drop: the transcript is its own emission. A drop is
            // never translated, so nothing is spent on text the pipeline decided
            // not to emit.
            let mut writer = metrics.lock();
            write_metrics_row(
                &mut writer,
                &tag,
                decoded_samples,
                &utterance.evidence,
                verdict,
                final_trimmed,
                infer_ms,
                None,
            );
            // Released before the emission: the emit queue has its own lock, and
            // nothing should hold two at once.
            drop(writer);
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

/// Words compared on each side for split-seam dedup (see
/// `remove_split_overlap`): mirrors Murmure's `MERGE_WINDOW_WORDS`.
pub(crate) const MERGE_WINDOW_WORDS: usize = 6;

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
    use crate::config::{DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS, DEFAULT_THREADS};
    use crate::gate::{is_hallucination, is_repetition};
    use crate::speech::Transcript;
    use crate::utterance::Evidence;

    /// A live dictionary with no words, for tests that never swap one.
    fn empty_live_dictionary() -> LiveDictionary {
        LiveDictionary::new(crate::control::DictionaryState::new(
            Vec::new(),
            std::path::PathBuf::new(),
        ))
        .1
    }

    /// A live dictionary seeded with `words`, plus the `control::DictionaryState`
    /// a test would hand to `dbus::spawn`, so a test can drive a swap and see
    /// what the worker would apply.
    fn live_dictionary_with(words: &[&str]) -> (crate::control::DictionaryState, LiveDictionary) {
        LiveDictionary::new(crate::control::DictionaryState::new(
            words.iter().map(|w| w.to_string()).collect(),
            std::path::PathBuf::new(),
        ))
    }

    #[test]
    fn words_swap_reaches_the_worker_handle_and_the_boost_queue() {
        let (bus, live) = live_dictionary_with(&["AppImage"]);
        // A swap on the bus thread is visible to the worker in both forms.
        assert_eq!(*live.words(), vec!["AppImage".to_string()]);
        assert!(live.take_boost().is_none(), "no swap yet");
        bus.replace_words(vec!["Parakeet".to_string()]);
        assert_eq!(*live.words(), vec!["Parakeet".to_string()]);
        assert_eq!(
            *live
                .take_boost()
                .expect("swap must be queued for the booster"),
            vec!["Parakeet".to_string()]
        );
        // One-shot: the engine is rebuilt once, so nothing is left pending.
        assert!(live.take_boost().is_none());
    }

    /// A `SpeechModel` that records what the worker hands the booster. This
    /// stands in for `ParakeetModel`, whose `set_boost_words` is what rebuilds
    /// the `BoostTree`.
    struct RecordingModel {
        /// Every list the worker delivered, in order.
        delivered: Vec<Vec<String>>,
    }

    impl RecordingModel {
        fn new() -> Self {
            Self {
                delivered: Vec::new(),
            }
        }

        /// The list the booster was last built from, or `None` if the worker
        /// never delivered one.
        fn boost_words(&self) -> Option<&[String]> {
            self.delivered.last().map(Vec::as_slice)
        }
    }

    impl SpeechModel for RecordingModel {
        fn transcribe(
            &mut self,
            _pcm_16k: &[f32],
            _running: &Arc<AtomicBool>,
        ) -> Result<Transcript> {
            panic!("these tests never decode audio")
        }

        fn set_boost_words(&mut self, words: &[String]) {
            self.delivered.push(words.to_vec());
        }

        fn describe(&self) -> String {
            "recording model".to_string()
        }

        fn describe_reload(&self) -> String {
            "recording model reloaded".to_string()
        }

        fn engine_name(&self) -> &'static str {
            "recording"
        }

        fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
            conditioned_len
        }
    }

    /// The whole live-dictionary path, from the D-Bus-visible handle to the
    /// engine's boost list: the bus thread mutates `DictionaryState`, and the
    /// worker thread hands the result to `set_boost_words`.
    #[test]
    fn a_dbus_visible_dictionary_reaches_the_engine_booster() {
        let (bus, mut live) = live_dictionary_with(&["AppImage"]);
        let mut model = RecordingModel::new();

        // Nothing has been delivered yet: the initial list reaches the engine
        // through `ParakeetModel::load`, not through this queue.
        assert_eq!(
            deliver_boost_words(&mut live, &mut model),
            None,
            "an unswapped dictionary must not queue a rebuild"
        );

        // ReplaceWords.
        assert_eq!(bus.replace_words(vec!["Parakeet".to_string()]), 1);
        assert_eq!(deliver_boost_words(&mut live, &mut model), Some(1));
        assert_eq!(model.boost_words(), Some(&["Parakeet".to_string()][..]));

        // AddWords: the engine gets the merged list, not just the addition.
        bus.add_words(vec!["żółć".to_string()]);
        assert_eq!(deliver_boost_words(&mut live, &mut model), Some(2));
        assert_eq!(
            model.boost_words(),
            Some(&["Parakeet".to_string(), "żółć".to_string()][..])
        );

        // RemoveWords, matched on the normalized form.
        bus.remove_words(vec!["PARAKEET".to_string()]);
        assert_eq!(deliver_boost_words(&mut live, &mut model), Some(1));
        assert_eq!(model.boost_words(), Some(&["żółć".to_string()][..]));

        // The booster list and the post-correction list are the same list, so
        // a swap cannot correct one way and boost another.
        assert_eq!(*model.boost_words().expect("delivered"), *live.words());

        // A no-op mutation queues nothing, so the engine is not rebuilt.
        bus.add_words(vec!["ZOLC".to_string()]);
        assert_eq!(
            deliver_boost_words(&mut live, &mut model),
            None,
            "a plain re-case must not queue a rebuild"
        );
        assert_eq!(model.delivered.len(), 3);
    }

    /// An uppercase Polish spelling is the same word, so it must not queue a
    /// booster rebuild. `normalize_word` lowercases before folding, which is
    /// what makes this hold.
    #[test]
    fn an_uppercase_polish_recase_does_not_queue_a_rebuild() {
        let (bus, mut live) = live_dictionary_with(&["żółć"]);
        let mut model = RecordingModel::new();

        let (count, added) = bus.add_words(vec!["ŻÓŁĆ".to_string()]);
        assert_eq!((count, added), (1, 0), "the uppercase form is folded");
        assert_eq!(
            deliver_boost_words(&mut live, &mut model),
            None,
            "no rebuild for a pure re-case"
        );
        assert_eq!(model.delivered.len(), 0);
    }

    /// `LoadFile` reaches the engine the same way, and clears the dirty flag
    /// the caller can read back over the bus.
    #[test]
    fn a_loaded_file_reaches_the_engine_booster() {
        let path = std::env::temp_dir().join(format!("sttd-asr-boost-{}.txt", std::process::id()));
        std::fs::write(&path, "AppImage\nSyntocinon\n").expect("temp file must be writable");
        let (bus, mut live) = live_dictionary_with(&[]);
        let mut model = RecordingModel::new();

        bus.replace_words(vec!["stale".to_string()]);
        assert!(bus.is_dirty(), "a caller-supplied list diverges");
        assert_eq!(deliver_boost_words(&mut live, &mut model), Some(1));

        bus.load_file(&path).expect("the temp file must load");
        assert!(!bus.is_dirty(), "the list now matches the source");
        assert_eq!(deliver_boost_words(&mut live, &mut model), Some(2));
        assert_eq!(
            model.boost_words(),
            Some(&["AppImage".to_string(), "Syntocinon".to_string()][..])
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn boost_queue_survives_a_reload_after_auto_unload() {
        // Swap while the model is unloaded: the words stay queued, so the next
        // `load_once` applies them (that is the auto-unload path).
        let (bus, live) = live_dictionary_with(&[]);
        bus.replace_words(vec!["zoxide".to_string()]);
        let pending = live.take_boost().expect("queued while unloaded");
        // A worker clone sees the same queue slot, so a re-created worker
        // applying after a reload still finds the list.
        let reloaded = live.clone();
        assert!(reloaded.take_boost().is_none(), "the worker took it");
        assert_eq!(*pending, vec!["zoxide".to_string()]);
    }

    #[test]
    fn live_dictionary_is_debug_and_clone() {
        let (_bus, live) = live_dictionary_with(&["a", "b"]);
        let clone = live.clone();
        assert_eq!(*clone.words(), *live.words());
        assert!(format!("{live:?}").contains("words: 2"), "{live:?}");
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

    /// A worker's configuration with the settings tests want: no shedding, no
    /// metrics file, no dictionary, no sound, default thresholds. Declared here
    /// rather than at module level because it draws on `default_gate` and
    /// `empty_live_dictionary`, which are both in this module.
    ///
    /// This supplies defaults; it does not wire. The wiring still crosses one
    /// seam, `AsrWorker::new`.
    fn for_tests() -> AsrConfig {
        AsrConfig {
            engine: EngineConfig {
                model_spec: "scripted".to_string(),
                n_threads: DEFAULT_THREADS,
                use_gpu: false,
                language: crate::language::Language::Auto,
                audio_ctx: crate::whisper_core::types::DEFAULT_WINDOW_FRAMES,
            },
            gate: default_gate(),
            remove_hesitations: false,
            dictionary: empty_live_dictionary(),
            shed_stale: false,
            partials: PartialSlot::default(),
            auto_unload: false,
            auto_unload_after: Duration::from_secs(0),
            sound: false,
            autopause: false,
        }
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
        assert!(!should_autopause(false, true, false));
        assert!(!should_autopause(false, false, false));
        // Enabled fires only when the transcript was actually emitted;
        // drops and failed inferences (false) must not steal the shot.
        assert!(!should_autopause(true, false, false));
        assert!(should_autopause(true, true, false));
        // A pause flush is transcribed while the run is already held: firing
        // again would only repeat the log line and the beep.
        assert!(!should_autopause(true, true, true));
        assert!(!should_autopause(false, true, true));
    }

    #[test]
    fn keep_while_paused_takes_the_flush_and_only_fresh_text() {
        // The pause flush was cut at this very edge, so it is never stale.
        assert!(keep_while_paused(true, Duration::from_secs(600)));
        assert!(keep_while_paused(true, Duration::ZERO));
        // An ordinary close survives only while it is still fresh: the
        // sentence that closed milliseconds before the pause must go through,
        // a backlog the model is minutes behind on must not.
        assert!(keep_while_paused(false, Duration::ZERO));
        assert!(keep_while_paused(
            false,
            PAUSE_KEEP_MAX_WAITED - Duration::from_millis(1)
        ));
        assert!(!keep_while_paused(false, PAUSE_KEEP_MAX_WAITED));
        assert!(!keep_while_paused(false, Duration::from_secs(600)));
    }

    #[test]
    fn asr_worker_signals_init_failure() {
        // Bad model dir must propagate via the ready channel instead of
        // silently capturing into a dead queue.
        let (_tx, rx) = rtrb::RingBuffer::new(ASR_QUEUE_CAP);
        let mut config = for_tests();
        config.engine.model_spec = "bogus".to_string();
        let bad = ModelRef {
            dir: std::path::PathBuf::from("/nonexistent/parakeet-bogus"),
            assets: &[],
            kind: crate::model_store::EngineKind::Parakeet,
        };
        let (handle, ready) = spawn_asr_worker(
            rx,
            bad,
            config,
            crate::emit::RecordingEmitter::new(),
            None,
            MetricsSlot::default(),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
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

    /// A model that returns a fixed transcript and remembers how long each
    /// input was, in call order.
    struct ScriptedModel {
        text: String,
        inputs: Arc<Mutex<Vec<usize>>>,
        fail: bool,
    }

    impl ScriptedModel {
        fn new(text: &str) -> (Self, Arc<Mutex<Vec<usize>>>) {
            let inputs = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    text: text.to_string(),
                    inputs: Arc::clone(&inputs),
                    fail: false,
                },
                inputs,
            )
        }
    }

    impl SpeechModel for ScriptedModel {
        fn transcribe(
            &mut self,
            pcm_16k: &[f32],
            _running: &Arc<AtomicBool>,
        ) -> Result<Transcript> {
            self.inputs.lock().unwrap().push(pcm_16k.len());
            if self.fail {
                anyhow::bail!("scripted failure");
            }
            Ok(Transcript {
                text: self.text.clone(),
                word_confidences: Vec::new(),
            })
        }

        fn describe(&self) -> String {
            "scripted".into()
        }

        fn describe_reload(&self) -> String {
            "scripted reloaded".into()
        }

        fn engine_name(&self) -> &'static str {
            "scripted"
        }

        fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
            conditioned_len
        }
    }

    #[test]
    fn partial_slot_keeps_only_the_newest_snapshot() {
        let slot = PartialSlot::default();
        assert!(slot.take().is_none());
        slot.put(vec![1.0; 10]);
        slot.put(vec![2.0; 20]);
        assert_eq!(slot.take().map(|p| p.len()), Some(20), "latest wins");
        assert!(slot.take().is_none(), "taking empties the slot");
        // A clone shares the slot: that is how the capture side reaches the worker.
        let other = slot.clone();
        other.put(vec![3.0; 5]);
        slot.clear();
        assert!(other.take().is_none(), "clear reaches every handle");
    }

    #[test]
    fn partial_line_reports_text_audio_and_the_next_tag() {
        let (mut model, inputs) = ScriptedModel::new("  dzień dobry  ");
        let running = Arc::new(AtomicBool::new(true));
        let pcm = vec![0.1f32; 24_000];
        let line = partial_line(&mut model, &pcm, "utt_000007", &default_gate(), &running)
            .expect("a normal transcript is shown");
        assert!(
            line.starts_with("utt_000007: partial (1.5 s audio, "),
            "{line}"
        );
        assert!(line.ends_with("): dzień dobry"), "{line}");
        // Fed like a final: conditioned, plus the trailing pad.
        assert_eq!(*inputs.lock().unwrap(), [24_000 + INFERENCE_PAD_SAMPLES]);
    }

    #[test]
    fn partial_line_hides_gated_empty_and_failed_decodes() {
        let running = Arc::new(AtomicBool::new(true));
        let pcm = vec![0.1f32; 16_000];
        for text in ["", "   ", "Dziękuję za uwagę."] {
            let (mut model, _) = ScriptedModel::new(text);
            assert!(
                partial_line(&mut model, &pcm, "utt_000001", &default_gate(), &running).is_none(),
                "{text:?} must not be shown"
            );
        }
        let (mut model, _) = ScriptedModel::new("tekst");
        model.fail = true;
        assert!(partial_line(&mut model, &pcm, "utt_000001", &default_gate(), &running).is_none());
    }

    fn closed_utterance(samples: usize) -> Utterance {
        Utterance {
            pcm: vec![0.1; samples],
            evidence: Evidence {
                speech_frames: 50,
                total_frames: 60,
                rms: 0.05,
            },
            split_continuation: false,
            queued_at: Instant::now(),
            pause_flush: false,
        }
    }

    /// Same, but as the pause edge or a stale backlog would leave it: tagged
    /// as a flush, or queued long enough ago to count as stale.
    fn queued_utterance(samples: usize, pause_flush: bool, waited: Duration) -> Utterance {
        Utterance {
            queued_at: Instant::now()
                .checked_sub(waited)
                .expect("host uptime must exceed the waited window"),
            pause_flush,
            ..closed_utterance(samples)
        }
    }

    /// Run `AsrWorker::run_loaded` on a thread against `model` until it has made
    /// `calls` decodes (or 10 s pass), then close the queue and join it.
    /// `paused` starts the worker with the run already held. Returns the
    /// worker's final `utterance_no` and its `prev_emit_text`.
    fn run_worker_until(
        queued: Vec<Utterance>,
        partial: Option<Vec<f32>>,
        mut model: ScriptedModel,
        inputs: &Arc<Mutex<Vec<usize>>>,
        calls: usize,
        settle: Duration,
        paused: bool,
    ) -> (u64, String) {
        let (mut tx, mut rx) = rtrb::RingBuffer::new(ASR_QUEUE_CAP);
        let model_ref = ModelRef {
            dir: std::path::PathBuf::new(),
            assets: &[],
            kind: crate::model_store::EngineKind::Parakeet,
        };
        let (ready_tx, _ready_rx) = std::sync::mpsc::channel();
        let mut emitter = crate::emit::RecordingEmitter::new();
        let paused = Arc::new(AtomicBool::new(paused));
        let running = Arc::new(AtomicBool::new(true));
        // The same constructor production uses, so the per-load state under
        // test is the state the worker really starts in.
        let mut worker = AsrWorker::new(
            &mut rx,
            model_ref,
            for_tests(),
            &ready_tx,
            &mut emitter,
            // No translation step: these tests drive the source-text pipeline.
            None,
            MetricsSlot::default(),
            &paused,
            &running,
        );
        // `run_loaded` is entered directly, so the first-load bookkeeping
        // `new` performed has already happened and must not repeat.
        worker.first_load = false;
        for u in queued {
            assert!(tx.push(u).is_ok());
        }
        if let Some(p) = partial {
            worker.config.partials.put(p);
        }
        // The worker holds a non-`Send` emitter, so it runs here; a helper
        // thread waits for the expected decodes and then closes the queue.
        let producer_gone = std::thread::scope(|scope| {
            scope.spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(10);
                while inputs.lock().unwrap().len() < calls && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                std::thread::sleep(settle);
                drop(tx);
            });
            worker.run_loaded(&mut model)
        });
        assert!(producer_gone, "the loop must end because the producer left");
        (worker.utterance_no, worker.prev_emit_text.clone())
    }

    #[test]
    fn a_closed_utterance_goes_before_a_waiting_partial() {
        let (model, inputs) = ScriptedModel::new("dzień dobry");
        let (utterance_no, prev) = run_worker_until(
            vec![closed_utterance(8_000)],
            Some(vec![0.1; 20_000]),
            model,
            &inputs,
            2,
            Duration::from_millis(50),
            false,
        );
        let seen = inputs.lock().unwrap().clone();
        assert_eq!(
            seen,
            [
                8_000 + INFERENCE_PAD_SAMPLES,
                20_000 + INFERENCE_PAD_SAMPLES
            ],
            "final first, then the partial"
        );
        // The partial left no trace in the final's bookkeeping.
        assert_eq!(utterance_no, 1, "only the final took a number");
        assert_eq!(prev, "dzień dobry");
    }

    #[test]
    fn a_partial_alone_is_decoded_but_takes_no_number() {
        let (model, inputs) = ScriptedModel::new("dzień dobry");
        let (utterance_no, prev) = run_worker_until(
            vec![],
            Some(vec![0.1; 16_000]),
            model,
            &inputs,
            1,
            Duration::from_millis(50),
            false,
        );
        assert_eq!(inputs.lock().unwrap().len(), 1);
        assert_eq!(utterance_no, 0);
        assert!(prev.is_empty(), "a partial must not feed split-seam dedup");
    }

    #[test]
    fn a_held_run_transcribes_the_pause_flush_and_only_fresh_text() {
        let (model, inputs) = ScriptedModel::new("dzień dobry");
        // Arrival order: a backlog the model is minutes behind on, a sentence
        // that closed just before the pause, then the flush the pause edge cut.
        let (utterance_no, prev) = run_worker_until(
            vec![
                queued_utterance(8_000, false, Duration::from_secs(5)),
                queued_utterance(9_000, false, Duration::ZERO),
                queued_utterance(10_000, true, Duration::from_secs(5)),
            ],
            None,
            model,
            &inputs,
            2,
            Duration::from_millis(50),
            true,
        );
        assert_eq!(
            *inputs.lock().unwrap(),
            [
                9_000 + INFERENCE_PAD_SAMPLES,
                10_000 + INFERENCE_PAD_SAMPLES
            ],
            "the fresh close and the flush go through while held; the stale backlog does not"
        );
        assert_eq!(
            utterance_no, 2,
            "only the two kept utterances took a number"
        );
        assert_eq!(prev, "dzień dobry");
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
        assert_eq!(
            polish_transcript("utt_000004", "yyy eee", &[], None, true),
            ""
        );
        assert_eq!(polish_transcript("utt_000005", "", &[], None, false), "");
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
        assert_eq!(remove_split_overlap("podaj zolc", "żółć proszę"), "proszę");
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
        assert_eq!(remove_split_overlap("je vais au marché", "au marché"), "");
    }

    #[test]
    fn split_overlap_caps_at_window_size() {
        let prev = "zero un deux trois quatre cinq six";
        assert_eq!(
            remove_split_overlap(prev, "un deux trois quatre cinq six sept"),
            "sept"
        );
        let long_prev = "un deux trois quatre cinq six sept";
        let long_cur = "un deux trois quatre cinq six sept huit";
        assert_eq!(remove_split_overlap(long_prev, long_cur), long_cur);
    }

    /// One utterance through `transcribe_and_emit` with a scripted decode, a
    /// recording emitter and no metrics file. Returns what was emitted and the
    /// caller's `prev_emit_text` afterwards, which is the pair the two tests
    /// below need to pin.
    fn run_pass(
        text: &str,
        remove_hesitations: bool,
        prev_emit_text: &mut String,
    ) -> (bool, crate::emit::RecordedEmission) {
        let (mut model, _inputs) = ScriptedModel::new(text);
        let running = Arc::new(AtomicBool::new(true));
        let mut emitter = crate::emit::RecordingEmitter::new();
        let emitted = transcribe_and_emit(
            &mut model,
            &closed_utterance(16_000),
            1,
            &default_gate(),
            &running,
            &mut emitter,
            // No translation step: these tests pin the source-text pipeline.
            &mut None,
            &MetricsSlot::default(),
            &[],
            remove_hesitations,
            prev_emit_text,
        );
        let record = emitter
            .records
            .pop()
            .expect("every pass reaches the emitter exactly once");
        (emitted, record)
    }

    /// Cleaning can remove everything (`clean.rs`), and emitting the empty
    /// result would clobber the clipboard. The verdict must be downgraded to
    /// `DropEmpty` and the split-seam text left alone, so the next utterance's
    /// dedup still compares against what was actually said.
    #[test]
    fn cleaning_to_nothing_downgrades_the_verdict_instead_of_emitting_empty_text() {
        let mut prev = String::from("coś wcześniej");
        // The gate accepts it (not empty, not boilerplate, not a loop);
        // `clean` then strips every filler.
        let (emitted, record) = run_pass("eee um", true, &mut prev);

        assert!(!emitted, "nothing was emitted");
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::DropEmpty,
            "the verdict must say the text was cleaned away"
        );
        assert!(record.text.trim().is_empty(), "text={:?}", record.text);
        assert!(!record.emitted, "an empty Emit must not count as emitted");
        assert_eq!(record.drop_kind, None, "DropEmpty is not a drop kind");
        assert!(
            !record.copied && !record.pasted && !record.printed,
            "no side effect may run for an empty transcript"
        );
        assert_eq!(
            prev, "coś wcześniej",
            "a drop must not advance the split-seam text"
        );
    }

    /// Split-seam dedup compares against the last thing actually said, so
    /// `prev_emit_text` advances on an emit and is left alone by a drop.
    /// Without that, one dropped utterance would let the next continuation
    /// strip words the speaker did say twice.
    #[test]
    fn prev_emit_text_advances_on_an_emit_and_never_on_a_drop() {
        let mut prev = String::new();
        let (emitted, record) = run_pass("dzień dobry", false, &mut prev);
        assert!(emitted);
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::Emit,
            "verdict={:?}",
            record.verdict
        );
        assert_eq!(prev, "dzień dobry", "an emit advances the dedup baseline");

        // `napisy` is exact-match hallucination boilerplate (`config.rs`).
        let (emitted, record) = run_pass("napisy", false, &mut prev);
        assert!(!emitted, "boilerplate must not be emitted");
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::Drop {
                kind: crate::gate::TranscriptDropKind::Hallucination
            },
            "verdict={:?}",
            record.verdict
        );
        assert_eq!(
            prev, "dzień dobry",
            "a drop must not become the dedup baseline"
        );
    }

    /// The text the emitter receives when the translation step is on: the
    /// translation, or the source text when the translation fails or comes back
    /// empty.
    ///
    /// `Translator` holds two `ort` sessions, so no unit test can build one.
    /// What is under test here is the step's failure policy, which
    /// `fallback_or_translated` owns outright, and the emitter's text is that
    /// function's return value. The model-run half needs weights and is out of
    /// scope for a unit test.
    ///
    /// The call is now on the translation thread rather than inline here, which
    /// does not change the policy: `emit_one` in `translate` applies this same
    /// function to the job's `source` before it touches the emitter.
    #[test]
    fn the_emitted_text_is_the_translation_or_the_source_fallback() {
        use crate::translate::fallback_or_translated;
        // A translation that arrives: that is what is emitted.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok("Good morning".to_string()), "utt_000001"),
            "Good morning"
        );
        // A translation that failed: the source, not an empty emission.
        assert_eq!(
            fallback_or_translated(
                "Dzień dobry",
                Err(anyhow::anyhow!("the decoder ran out of cache slots")),
                "utt_000002"
            ),
            "Dzień dobry"
        );
        // A translation that came back empty: also the source, so an empty
        // string never reaches the clipboard.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok(String::new()), "utt_000003"),
            "Dzień dobry"
        );
        // A whitespace-only translation is empty for this purpose too.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok("   ".to_string()), "utt_000004"),
            "Dzień dobry"
        );
    }
}
