//! The worker thread: load, unload, pause, shed, and the loop that drives one
//! loaded model.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use rtrb::Consumer;

use crate::emit::Emitter;
use crate::metrics::MetricsSlot;
use crate::model_store::ModelRef;
use crate::speech::{EngineConfig, SpeechModel};
use crate::translate::TranslationOut;
use crate::utterance::Utterance;

use super::live::LiveDictionary;
use super::pipeline::{partial_line, transcribe_and_emit};
use super::{AsrConfig, ASR_SHED_KEEP};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::test_support::*;
    use crate::asr::ASR_QUEUE_CAP;
    use crate::config::INFERENCE_PAD_SAMPLES;
    use std::sync::Mutex;

    use crate::speech::Transcript;

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
}
