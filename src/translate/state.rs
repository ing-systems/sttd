//! The translation switch: which translator is live, and swapping it while
//! the run continues.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use anyhow::Result;

use super::engine::Translator;
use super::plan::{load_for, TranslationPlan};

// ---------------------------------------------------------------------------
// Runtime control
// ---------------------------------------------------------------------------
//
// `engine` and `plan` load a translator once; everything here decides *which*
// one is live, and swaps it while the run continues.
//
// The state lives in `translate` rather than in `dbus` because it owns the
// weights and the loader threads, and `translator_store` plus three `ort`
// sessions are not a bus concern: `dbus` holds this handle and forwards calls,
// the way `asr` holds `control::DictionaryState`. The dependency therefore
// points one way, `dbus -> translate`: what this module announces goes through
// a hook `dbus::spawn` installs (`TranslationState::set_change_hook`).
//
// Two rules hold the whole thing together:
//
// * **One copy of the weights, behind one lock.** The translation thread looks
//   the translator up per job instead of caching it, because a second copy is
//   exactly what would make `Disable(true)` unable to free the weights at once.
//   The cost is that a mutating call waits for a translation already running
//   (60 to 130 ms for a pair, ~700 ms for a multilingual export) and never for a
//   load.
// * **The generation rides in that same lock.** A load that finished after its
//   request was superseded must not install itself, and a `Disable` that promised
//   to free the weights must not be raced by one still arriving, so the check and
//   the write are one critical section rather than two steps.

/// How a state of `L` produces one: from a plan and a target language.
///
/// A trait so the pipeline that loads, installs and announces can be driven in a
/// test without weights, and so production's one implementation is named in the
/// type rather than picked inside the body.
pub(crate) trait Loads<L>: Clone + Send + Sync + 'static {
    /// Resolve, download and load. The real one is [`load_for`].
    fn load(&self, plan: &TranslationPlan, target: &str) -> Result<L>;
}

/// The production loader: the real translator, built from the run's own flags.
#[derive(Clone, Copy)]
pub(crate) struct RealTranslator;

impl Loads<Translator> for RealTranslator {
    fn load(&self, plan: &TranslationPlan, target: &str) -> Result<Translator> {
        load_for(plan, target)
    }
}

/// What the state needs from a loaded translator: translate one string, and name
/// itself in a log line.
///
/// A trait rather than `Translator` directly so the state is testable. A
/// `Translator` is two or three `ort` sessions and a tokenizer, so no unit test
/// can build one, and the bookkeeping around a load — refusals, the generation
/// that cancels one, which field each switch clears — is exactly what needs
/// testing. Production uses the one implementor below.
pub(crate) trait Loaded: Send + 'static {
    /// Translate one utterance. A failure is not fatal: the caller falls back to
    /// the source text.
    fn translate(&mut self, source: &str) -> Result<String>;

    /// The upstream spec, for log lines and the `Model` property.
    fn spec(&self) -> &str;

    /// The direction, as a log should read it.
    fn direction(&self) -> &str;
}

impl Loaded for Translator {
    fn translate(&mut self, source: &str) -> Result<String> {
        Translator::translate(self, source)
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn direction(&self) -> &str {
        &self.direction
    }
}

/// The one copy of the loaded weights, and the generation that names it.
///
/// `generation` moves with the translator on purpose: comparing a request's
/// generation against this one *while holding the lock that guards the
/// translator* is what makes "a superseded load must not install" and "`Disable`
/// frees the weights at once" true rather than nearly true.
pub(crate) struct TranslatorSlot<T> {
    pub(crate) translator: Option<T>,
    pub(crate) generation: u64,
}

/// What the bus reads. A snapshot of the truth, written by whoever holds the
/// slot, because a property read must not queue behind a running translation.
#[derive(Default, Clone)]
struct Cached {
    enabled: bool,
    target: String,
    model: String,
}

/// The translation step as the rest of the program and the bus see it: on or
/// off, writing which language, with which translator, and loading one now.
///
/// A cheap `Clone`, so the bus thread, the ASR worker and the translation thread
/// all hold the same state.
pub(crate) struct TranslationState<T = Translator, L = RealTranslator> {
    slot: Arc<Mutex<TranslatorSlot<T>>>,
    /// What the user asked for. `Enabled` is this **and** a loaded translator, so
    /// intent and capability never disagree in a report.
    wanted: Arc<AtomicBool>,
    /// The language of the load in flight, empty when nothing is loading.
    requested: Arc<RwLock<String>>,
    /// The language to write, remembered across a disable and an unload so
    /// `Enable` knows what to reload. The wanted language, which is not always
    /// the live one: a switch keeps the old translator serving until the new one
    /// is ready, and an unload leaves this behind on purpose.
    target: Arc<RwLock<String>>,
    /// The language the **loaded** translator writes, empty when none is loaded.
    /// Kept apart from `target` because that gap is what makes a switch atomic
    /// from the reader's side: a client asking mid-switch sees the old language
    /// still being written, which is true.
    live: Arc<RwLock<String>>,
    /// The spec of the loaded translator. Reported beside `live`, and cleared
    /// with it.
    model_spec: Arc<RwLock<String>>,
    cached: Arc<RwLock<Cached>>,
    plan: Arc<TranslationPlan>,
    /// How a request becomes a translator. `Copy`, so every clone can start a load
    /// without sharing anything.
    loader: L,
    /// One load at a time: a superseded load cannot be cancelled, so this is what
    /// keeps two sets of weights from being built side by side beyond the pair
    /// already held and the one arriving.
    loading: Arc<Mutex<()>>,
    /// Who hears about a change no bus call is waiting on. Set once, after
    /// construction, and shared by every clone.
    on_change: Arc<OnceLock<ChangeHook>>,
}

/// The listener for [`TranslationState`] changes: `(enabled, target, model, ok,
/// message)`, the arguments of the bus's `TranslationChanged` signal. Injected
/// rather than called, so the dependency points one way, `dbus -> translate`.
pub(crate) type ChangeHook = Box<dyn Fn(bool, &str, &str, bool, &str) + Send + Sync>;

/// A poisoned lock must not kill transcription: take the value and continue.
fn lock_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn lock_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

impl<T: Loaded, L: Loads<T>> TranslationState<T, L> {
    /// A step that is off, with nothing loaded and nothing to load until a
    /// target is named.
    pub(crate) fn new(plan: TranslationPlan, loader: L) -> Self {
        Self {
            slot: Arc::new(Mutex::new(TranslatorSlot {
                translator: None::<T>,
                generation: 0,
            })),
            wanted: Arc::new(AtomicBool::new(false)),
            requested: Arc::new(RwLock::new(String::new())),
            target: Arc::new(RwLock::new(String::new())),
            live: Arc::new(RwLock::new(String::new())),
            model_spec: Arc::new(RwLock::new(String::new())),
            cached: Arc::new(RwLock::new(Cached::default())),
            plan: Arc::new(plan),
            loader,
            loading: Arc::new(Mutex::new(())),
            on_change: Arc::new(OnceLock::new()),
        }
    }

    /// Install the listener `announce` reports to. Only the first call takes
    /// effect; a run has one bus.
    pub(crate) fn set_change_hook(&self, hook: ChangeHook) {
        if self.on_change.set(hook).is_err() {
            log::warn!("translate: a change hook is already installed");
        }
    }

    /// The slot guard. Poison-tolerant, because a poisoned step must still emit
    /// the source text rather than stop the pipeline.
    pub(crate) fn lock_slot(&self) -> std::sync::MutexGuard<'_, TranslatorSlot<T>> {
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---- reads, all of which the bus serves without waiting on a translation --

    /// Whether the step is translating right now: asked for **and** loaded.
    pub(crate) fn is_enabled(&self) -> bool {
        lock_read(&self.cached).enabled
    }

    /// The language the live translator writes, empty when none is loaded. It
    /// survives a disable, because the translator does, and it does **not** follow
    /// a switch until the new translator lands.
    pub(crate) fn target_language(&self) -> String {
        lock_read(&self.cached).target.clone()
    }

    /// The spec of the live translator, empty when none is loaded.
    pub(crate) fn model(&self) -> String {
        lock_read(&self.cached).model.clone()
    }

    /// The language of the load in flight, empty when nothing is loading. Its
    /// being non-empty is the whole "please wait" signal, so there is no separate
    /// `Loading` property to fall out of step with it.
    pub(crate) fn requested_language(&self) -> String {
        lock_read(&self.requested).clone()
    }

    /// `(enabled, target, requested, model)` in one round trip.
    #[allow(clippy::type_complexity)]
    pub(crate) fn state(&self) -> (bool, String, String, String) {
        let cached = lock_read(&self.cached);
        (
            cached.enabled,
            cached.target.clone(),
            lock_read(&self.requested).clone(),
            cached.model.clone(),
        )
    }

    /// Whether the *next* utterance will be translated, which is what the ASR
    /// worker asks before handing a job over. Cheap on purpose: an `AtomicBool`
    /// and a lock that is never held across a decode.
    pub(crate) fn will_translate(&self) -> bool {
        self.wanted.load(Ordering::Acquire) && self.lock_slot().translator.is_some()
    }

    /// The intent flag alone. The lane reads it per job while it holds the slot,
    /// so it must not need a lock of its own.
    pub(crate) fn wanted_now(&self) -> bool {
        self.wanted.load(Ordering::Acquire)
    }

    /// Whether a translator is loaded, whatever the switch says.
    pub(crate) fn is_loaded(&self) -> bool {
        self.lock_slot().translator.is_some()
    }

    /// The panic a `catch_unwind` caught, as text.
    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_string())
    }

    // ---- writes ----

    /// Recompute what the bus reads. Idempotent, and called by every path that
    /// changes the truth, so the snapshot cannot drift from the slot.
    fn refresh_cached(&self) {
        let loaded = self.lock_slot().translator.is_some();
        let snapshot = Cached {
            enabled: loaded && self.wanted.load(Ordering::Acquire),
            target: if loaded {
                lock_read(&self.live).clone()
            } else {
                String::new()
            },
            model: if loaded {
                lock_read(&self.model_spec).clone()
            } else {
                String::new()
            },
        };
        *lock_write(&self.cached) = snapshot;
    }

    /// Turn the step on, returning `(enabled, message)`.
    ///
    /// With a translator loaded this is a flag. Without one it asks for the
    /// remembered target, which is `--translate` on a fresh run and whatever was
    /// live before an unload since. With nothing remembered it refuses, because
    /// there is no language to write and inventing one would be a lie — hence the
    /// message, which is empty on every other path.
    ///
    /// The announcement is the caller's: a bus method has a `SignalEmitter` to hand
    /// and the loader thread has not, so this only reports.
    pub(crate) fn enable(&self) -> (bool, String) {
        self.wanted.store(true, Ordering::Release);
        if self.is_loaded() {
            self.refresh_cached();
            return (true, String::new());
        }
        let remembered = lock_read(&self.target).clone();
        if remembered.is_empty() {
            self.wanted.store(false, Ordering::Release);
            self.refresh_cached();
            return (
                false,
                "no target language is set; call SetTargetLanguage first \
                 (or start with --translate LANG)"
                    .to_string(),
            );
        }
        self.start_load(&remembered);
        (self.is_enabled(), String::new())
    }

    /// Turn the step off. `unload` also frees the weights, before this returns.
    ///
    /// Either way a load in flight is cancelled: a run that was told to stop
    /// translating must not end up downloading a translator afterwards. The
    /// remembered target stays either way, so `Enable` is instant without an
    /// unload and a reload from the local cache with one.
    pub(crate) fn disable(&self, unload: bool) -> bool {
        self.cancel_pending();
        self.wanted.store(false, Ordering::Release);
        if unload {
            let mut slot = self.lock_slot();
            slot.translator = None;
            *lock_write(&self.live) = String::new();
            *lock_write(&self.model_spec) = String::new();
        }
        self.refresh_cached();
        self.is_enabled()
    }

    /// `Disable(false)` when on, `Enable` when off. Never unloads: the point of a
    /// toggle is the state, and dropping a gigabyte of weights is a decision of
    /// its own, which `Disable(true)` is for.
    pub(crate) fn toggle(&self) -> bool {
        if self.is_enabled() {
            self.disable(false)
        } else {
            self.enable().0
        }
    }

    /// Write `lang` from now on, loading whatever that needs.
    ///
    /// Returns `(accepted, target, message)`. A direction the catalog cannot
    /// serve is refused here, before any thread starts: the caller gets the same
    /// message the startup path prints, immediately. An accepted target is not
    /// yet a loaded one — the previous translator keeps serving until the new one
    /// is ready, and `TranslationChanged` says when the swap happened.
    ///
    /// No announcement here: the caller holds the bus emitter. What this returns is
    /// the whole report.
    #[allow(clippy::type_complexity)]
    pub(crate) fn set_target_language(&self, lang: &str) -> (bool, String, String) {
        let target = lang.trim().to_ascii_lowercase();
        if target.is_empty() {
            let message = "empty target language".to_string();
            return (false, self.target_language(), message);
        }
        let direction = self.plan.direction(&target);
        if let Err(e) =
            crate::translator_store::reference_for(direction, self.plan.model.as_deref())
        {
            let message = format!("{e:#}");
            return (false, self.target_language(), message);
        }
        // The same language, already loaded: this is a switch on, not a load.
        if lock_read(&self.live).eq(target.as_str()) && self.is_loaded() {
            self.wanted.store(true, Ordering::Release);
            self.refresh_cached();
            return (true, target, String::new());
        }
        *lock_write(&self.target) = target.clone();
        self.wanted.store(true, Ordering::Release);
        self.refresh_cached();
        self.start_load(&target);
        (true, target, String::new())
    }

    // ---- the load pipeline ----

    /// Load the language `--translate` named, on the calling thread, failing the
    /// run when it cannot: startup is where a missing model should stop everything
    /// rather than wait for the first utterance.
    ///
    /// Naming the language at startup is also the intent to use it, so the step is
    /// wanted from the first utterance: the worker asks `will_translate` before
    /// every hand-off, and a translator loaded but not wanted is never used.
    pub(crate) fn load_startup(&self) -> Result<()> {
        let Some(target) = self.plan.configured.clone() else {
            return Ok(());
        };
        *lock_write(&self.target) = target.clone();
        self.wanted.store(true, Ordering::Release);
        let generation = self.begin_request(&target);
        let translator = self.loader.load(&self.plan, &target)?;
        self.install(generation, &target, translator);
        Ok(())
    }

    /// Register a request and hand back the generation that will name it.
    fn begin_request(&self, target: &str) -> u64 {
        let generation = {
            let mut slot = self.lock_slot();
            slot.generation = slot.generation.wrapping_add(1);
            slot.generation
        };
        *lock_write(&self.requested) = target.to_string();
        generation
    }

    /// Abandon whatever was loading. The generation bump is what makes an arriving
    /// load stale rather than installed; clearing the request is what stops a
    /// client from waiting out a load that will now do nothing.
    fn cancel_pending(&self) {
        {
            let mut slot = self.lock_slot();
            slot.generation = slot.generation.wrapping_add(1);
        }
        *lock_write(&self.requested) = String::new();
    }

    /// Put the translator in place, if this request is still the current one.
    ///
    /// The generation is compared and the translator written under one lock, so
    /// a `Disable` cannot slip between the two and be followed by weights it
    /// promised to free.
    fn install(&self, generation: u64, target: &str, translator: T) -> bool {
        let mut slot = self.lock_slot();
        if slot.generation != generation {
            log::debug!("translate: the {target} load was superseded before it landed; dropped");
            return false;
        }
        let spec = translator.spec().to_string();
        slot.translator = Some(translator);
        slot.generation = generation;
        // Still under the slot, so a client cannot read a loaded translator whose
        // language and spec have not been written yet.
        *lock_write(&self.live) = target.to_string();
        *lock_write(&self.model_spec) = spec.clone();
        *lock_write(&self.requested) = String::new();
        drop(slot);
        self.refresh_cached();
        log::info!("translate: writing {target} (translator {spec})");
        self.announce(true, "");
        true
    }

    /// Report a load that failed. The previous translator keeps serving: a
    /// refused switch must not cost the language the run already had.
    fn fail_request(&self, generation: u64, message: &str) {
        {
            let slot = self.lock_slot();
            if slot.generation != generation {
                log::debug!("translate: a superseded load failed; its error is stale");
                return;
            }
            if slot.translator.is_none() {
                // Nothing to fall back on, so the step is off rather than on with
                // no translator: `Enabled` must never claim work that cannot
                // happen.
                self.wanted.store(false, Ordering::Release);
            }
        }
        *lock_write(&self.requested) = String::new();
        self.refresh_cached();
        self.announce(false, message);
    }

    /// Load `target` on a helper thread, through `load`.
    ///
    /// Off the bus thread on purpose: a first use downloads 114 MiB to 1.3 GiB and
    /// then builds three `ort` sessions. `Pause` and the dictionary have to keep
    /// working while that happens, and the blocking zbus API serves one thread.
    ///
    /// `load` is a parameter rather than a direct [`load_for`] call so a test can
    /// drive the whole pipeline — request, serialize, install or fail, announce —
    /// without weights and without a network.
    fn start_load(&self, target: &str) {
        let generation = self.begin_request(target);
        let state = self.clone();
        let target = target.to_string();
        let spawned = std::thread::Builder::new()
            .name("translate-load".into())
            .spawn(move || {
                // Held across the load, so a superseded request waits its turn
                // instead of building a second set of weights beside the first.
                let _serialized = state.loading.lock().unwrap_or_else(|e| e.into_inner());
                match state.loader.load(&state.plan, &target) {
                    Ok(translator) => {
                        state.install(generation, &target, translator);
                    }
                    Err(e) => {
                        let message = format!("{e:#}");
                        log::warn!("translate: cannot write {target}: {message}");
                        state.fail_request(generation, &message);
                    }
                }
            });
        if let Err(e) = spawned {
            let message = format!("failed to start the translator load: {e}");
            log::error!("translate: {message}");
            self.fail_request(generation, &message);
        }
    }

    /// A load that was superseded before it landed, reported as a debug line
    /// rather than a failure: it did its work, and a newer request wants the slot.
    #[cfg(test)]
    pub(crate) fn report_stale_install(&self, generation: u64, target: &str) {
        if self.lock_slot().generation != generation {
            log::debug!("translate: the {target} load was superseded; dropped");
        }
    }

    /// Report a change from outside an interface method — the loader thread,
    /// and every refusal, which have no emitter to hand — through the hook
    /// [`Self::set_change_hook`] installed. Silent when none is installed.
    fn announce(&self, ok: bool, message: &str) {
        let Some(hook) = self.on_change.get() else {
            return;
        };
        let cached = lock_read(&self.cached);
        hook(cached.enabled, &cached.target, &cached.model, ok, message);
    }

    /// The translation thread's half: run one decode under the slot lock, so a
    /// disable cannot free a translator mid-decode, and report a panicking decode
    /// as a failed translation instead of a dead thread.
    ///
    /// A panic drops the translator rather than keeping it: the sessions take
    /// `&mut self` and their state after a panic is not something to keep
    /// decoding with. The step then reports itself off, the ASR worker goes back
    /// to emitting source text, and `Enable` loads a fresh one.
    pub(crate) fn translate_locked(&self, slot: &mut TranslatorSlot<T>, source: &str) -> Attempt {
        let Some(translator) = slot.translator.as_mut() else {
            return Attempt::Failed("no translator is loaded".to_string());
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            translator.translate(source)
        })) {
            Ok(Ok(text)) => Attempt::Translated(text),
            Ok(Err(e)) => Attempt::Failed(format!("{e:#}")),
            Err(payload) => Attempt::Panicked(Self::panic_message(payload)),
        }
    }

    /// Free the weights of a translator that panicked, taking the slot itself.
    ///
    /// Not a `&mut TranslatorSlot` argument: the caller holds that lock while it
    /// decodes, and a lock that tries to take itself again deadlocks the thread
    /// that has to be rescued. So the lane releases it first and calls this.
    pub(crate) fn drop_panicked_translator(&self) {
        {
            let mut slot = self.lock_slot();
            slot.translator = None;
            *lock_write(&self.live) = String::new();
            *lock_write(&self.model_spec) = String::new();
        }
        self.refresh_cached();
        self.announce(
            false,
            "the translator panicked and was dropped; call Enable to load it again",
        );
    }
}

/// The clone is what lets the bus thread, the ASR worker, the translation thread
/// and every loader hold one state; `T` sits behind an `Arc` and is never copied.
impl<T, L: Clone> Clone for TranslationState<T, L> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
            wanted: Arc::clone(&self.wanted),
            requested: Arc::clone(&self.requested),
            target: Arc::clone(&self.target),
            live: Arc::clone(&self.live),
            model_spec: Arc::clone(&self.model_spec),
            cached: Arc::clone(&self.cached),
            plan: Arc::clone(&self.plan),
            loader: self.loader.clone(),
            loading: Arc::clone(&self.loading),
            on_change: Arc::clone(&self.on_change),
        }
    }
}

/// What one decode attempt produced.
///
/// `Panicked` is separate from `Failed` because the two need different
/// afterwards: a failed decode leaves a translator worth keeping, while a panicked
/// one leaves sessions that took `&mut self` and are not worth decoding with again.
pub(crate) enum Attempt {
    Translated(String),
    Failed(String),
    Panicked(String),
}

impl Attempt {
    /// The shape [`fallback_or_translated`] takes: every failure is the same to it,
    /// because every failure emits the source text.
    pub(super) fn into_result(self) -> Result<String> {
        match self {
            Self::Translated(text) => Ok(text),
            Self::Failed(what) | Self::Panicked(what) => Err(anyhow::anyhow!(what)),
        }
    }
}

/// Stand-ins for the state machine's loader and translator, shared by this
/// module's tests and the lane's.
#[cfg(test)]
pub(super) mod fakes {
    use super::*;
    use std::sync::Condvar;
    use std::time::Duration;

    /// A translator that needs no weights: it records what it was asked and returns
    /// a fixed string. Enough for the state machine, since none of it touches an
    /// `ort` session.
    #[derive(Clone)]
    pub(in crate::translate) struct FakeTranslator {
        spec: String,
        direction: String,
        answer: String,
    }

    impl Loaded for FakeTranslator {
        fn translate(&mut self, source: &str) -> Result<String> {
            Ok(format!("{}|{source}", self.answer))
        }
        fn spec(&self) -> &str {
            &self.spec
        }
        fn direction(&self) -> &str {
            &self.direction
        }
    }

    pub(in crate::translate) fn fake(spec: &str) -> FakeTranslator {
        FakeTranslator {
            spec: spec.to_string(),
            direction: "pl -> en".to_string(),
            answer: "translated".to_string(),
        }
    }

    /// A load that blocks until the test releases it, so "waiting for a load" is a
    /// state a test can observe rather than a race. `None` means load at once.
    pub(in crate::translate) type Gate = Option<Arc<Gate2>>;

    /// An open/closed flag with a wakeup: a load waits on it, a test opens it.
    pub(in crate::translate) struct Gate2 {
        open: Mutex<bool>,
        wakeup: Condvar,
    }

    #[derive(Clone)]
    pub(in crate::translate) struct FakeLoad {
        pub(in crate::translate) gate: Gate,
    }

    impl FakeLoad {
        /// Wait for the test to open the gate, if there is one. The load runs on
        /// its own thread, so this is the seam that makes "a load in flight" a state
        /// a test can observe instead of a race.
        fn wait(&self) {
            let Some(gate) = &self.gate else {
                return;
            };
            let mut open = gate.open.lock().unwrap_or_else(|e| e.into_inner());
            while !*open {
                open = gate.wakeup.wait(open).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Release every load blocked on this gate.
    pub(in crate::translate) fn release(gate: &Gate) {
        let Some(gate) = gate else {
            return;
        };
        *gate.open.lock().unwrap_or_else(|e| e.into_inner()) = true;
        gate.wakeup.notify_all();
    }

    impl Loads<FakeTranslator> for FakeLoad {
        fn load(&self, plan: &TranslationPlan, target: &str) -> Result<FakeTranslator> {
            self.wait();
            if target == "unobtainable" {
                anyhow::bail!("no translator for {} -> {target}", plan.source_label());
            }
            Ok(FakeTranslator {
                spec: format!("opus-mt-{}-{target}", plan.source_label()),
                direction: format!("{} -> {target}", plan.source_label()),
                answer: "translated".to_string(),
            })
        }
    }

    /// The state these tests drive: a stand-in translator behind the real
    /// bookkeeping.
    pub(in crate::translate) type TestState = TranslationState<FakeTranslator, FakeLoad>;

    /// A state over a Polish source, so `en` resolves and `de` does not, with loads
    /// that land at once.
    pub(in crate::translate) fn state(source: &str) -> TestState {
        gated_state(source, None)
    }

    /// The same, with every load blocked until [`release`] is called: what a client
    /// sees while a first-time download runs.
    pub(in crate::translate) fn gated_state(source: &str, gate: Gate) -> TestState {
        TranslationState::new(TranslationPlan::for_tests(source), FakeLoad { gate })
    }

    /// A state that accepts any target, because the load half is under test rather
    /// than the catalog: the translator is named outright, which is the one path
    /// `reference_for` resolves without consulting the pair table.
    pub(in crate::translate) fn any_target_state(gate: Gate) -> TestState {
        TranslationState::new(
            TranslationPlan::for_tests_with_model("pl", "nllb-200-distilled-600M"),
            FakeLoad { gate },
        )
    }

    /// A fresh gate to hand to [`gated_state`].
    pub(in crate::translate) fn new_gate() -> Gate {
        Some(Arc::new(Gate2 {
            open: Mutex::new(false),
            wakeup: Condvar::new(),
        }))
    }

    /// Wait until `what` reads `expected`, so a test observes the state instead of
    /// racing the loader thread.
    pub(in crate::translate) fn wait_until<T: PartialEq + std::fmt::Debug>(
        what: &str,
        mut read: impl FnMut() -> T,
        expected: T,
    ) {
        for _ in 0..400 {
            if read() == expected {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("{what} never became {expected:?}");
    }

    /// Put a translator in the slot the way a successful load would, so every
    /// read below turns on a real occupant without any weights.
    pub(in crate::translate) fn with_fake_translator(s: &TestState, target: &str, spec: &str) {
        *lock_write(&s.target) = target.to_string();
        seat(s, target, spec, || fake(spec));
    }

    /// Seat a translator built by `make` as the live one for `target`, the way an
    /// installed load leaves the state. Generic, so a test can seat a translator
    /// of its own (one that panics, say) without reaching into the fields.
    pub(in crate::translate) fn seat<T: Loaded, L: Loads<T>>(
        s: &TranslationState<T, L>,
        target: &str,
        spec: &str,
        make: impl Fn() -> T,
    ) {
        let mut slot = s.lock_slot();
        slot.generation += 1;
        let generation = slot.generation;
        slot.translator = Some(make());
        drop(slot);
        *lock_write(&s.live) = target.to_string();
        *lock_write(&s.model_spec) = spec.to_string();
        s.wanted.store(true, Ordering::Release);
        assert!(
            s.install(generation, target, make()),
            "a current generation must install"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::*;
    use super::*;

    // -- runtime state ------------------------------------------------------
    //
    // No weights anywhere below: a `Translator` is two or three `ort` sessions, so
    // a unit test cannot build one. What is under test is the bookkeeping around
    // the load — refusals, the generation that cancels one, and which fields each
    // switch clears — none of which needs a model. Every path that would fetch one
    // is asserted to refuse *before* it starts, so no test here touches the
    // network.

    #[test]
    fn the_startup_load_is_wanted_from_the_first_utterance() {
        // `--translate en` at startup: the translator is loaded *and* the step is
        // wanted, because the worker asks `will_translate` before every hand-off. A
        // translator loaded but not wanted would silently never run.
        let s: TestState = TranslationState::new(
            TranslationPlan::for_tests_configured("pl", "en"),
            FakeLoad { gate: None },
        );
        s.load_startup().expect("the startup load must succeed");
        assert!(s.is_enabled(), "a run that asked for translation is on");
        assert!(
            s.will_translate(),
            "and the worker must hand utterances over from the first one"
        );
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.requested_language(), "", "nothing is loading any more");
    }

    #[test]
    fn a_startup_load_that_cannot_be_produced_fails_the_run() {
        // Startup is the one place a missing translator is fatal, rather than a warn
        // on a later utterance.
        let s: TestState = TranslationState::new(
            TranslationPlan::for_tests_configured("pl", "unobtainable"),
            FakeLoad { gate: None },
        );
        assert!(
            s.load_startup().is_err(),
            "the caller must see the failure and stop"
        );
        assert!(!s.is_loaded(), "and nothing half-loaded is left behind");
    }

    #[test]
    fn no_translate_flag_loads_nothing_at_all() {
        let s = state("pl");
        s.load_startup().expect("no flag is not a failure");
        assert!(
            !s.is_loaded(),
            "nothing is downloaded for a run that did not ask"
        );
        assert!(!s.is_enabled());
        assert_eq!(s.target_language(), "");
    }

    #[test]
    fn a_fresh_state_is_off_with_nothing_loaded() {
        let s = state("pl");
        assert!(!s.is_enabled(), "nothing asked and nothing loaded");
        assert!(!s.is_loaded());
        assert_eq!(s.target_language(), "");
        assert_eq!(s.model(), "");
        assert_eq!(
            s.requested_language(),
            "",
            "an empty request is what says nothing is loading"
        );
        assert!(
            !s.will_translate(),
            "so nothing reaches the translation thread"
        );
    }

    #[test]
    fn an_empty_target_language_is_refused_without_a_load() {
        let s = state("pl");
        let (accepted, target, message) = s.set_target_language("   ");
        assert!(!accepted);
        assert_eq!(message, "empty target language");
        assert_eq!(target, "");
        assert_eq!(
            s.requested_language(),
            "",
            "a refusal must not register a request, or a loader would start"
        );
    }

    #[test]
    fn a_direction_the_catalog_cannot_serve_is_refused_with_the_startup_message() {
        let s = state("pl");
        let (accepted, target, message) = s.set_target_language("de");
        assert!(!accepted, "Polish has no exported pair into German");
        assert!(
            message.contains("no translator for pl -> de"),
            "the caller gets the message the startup path prints: {message}"
        );
        // It names what Polish *can* reach, which is the useful half.
        assert!(message.contains("en"), "{message}");
        assert_eq!(target, "", "the live target is unchanged by a refusal");
        assert_eq!(s.requested_language(), "");
    }

    #[test]
    fn an_unknown_source_language_is_refused_too() {
        let s = state("qq");
        let (accepted, _, message) = s.set_target_language("en");
        assert!(!accepted);
        assert!(message.contains("qq"), "{message}");
    }

    #[test]
    fn a_refusal_leaves_a_loaded_translator_serving() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.is_enabled(), "the fake load enables the step");
        let (accepted, target, _) = s.set_target_language("de");
        assert!(!accepted);
        assert_eq!(target, "en", "a refused switch costs the language it had");
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert!(s.will_translate());
    }

    #[test]
    fn enable_with_no_translator_and_no_target_refuses() {
        let s = state("pl");
        let (enabled, message) = s.enable();
        assert!(!enabled);
        assert!(
            message.contains("no target language is set"),
            "the caller is told why nothing happened: {message}"
        );
        assert!(!s.will_translate());
        assert_eq!(s.requested_language(), "", "and no load was started");
    }

    #[test]
    fn enable_after_an_unload_reports_off_until_the_load_lands() {
        // `Enabled` is false while the reload runs: it claims work that is not
        // happening yet, and the client waits on `RequestedLanguage` instead.
        let gate = new_gate();
        let s = gated_state("pl", gate.clone());
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(true);

        let (enabled, message) = s.enable();
        assert!(!enabled, "the reload has not finished");
        assert!(message.is_empty(), "this is not a refusal, only a load");
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "en".to_string(),
        );
        assert!(!s.will_translate(), "still nothing to translate with");

        release(&gate);
        wait_until("Enabled", || s.is_enabled(), true);
        assert_eq!(
            s.model(),
            "opus-mt-pl-en",
            "a fresh translator, reported by spec"
        );
        assert_eq!(s.requested_language(), "", "and the request is cleared");
    }

    #[test]
    fn a_first_switch_from_nothing_loads_before_it_reports_enabled() {
        // The one case with no translator to fall back on, so this is the window in
        // which utterances come out untranslated.
        let gate = new_gate();
        let s = any_target_state(gate.clone());
        let (accepted, target, message) = s.set_target_language("en");
        assert!(accepted);
        assert_eq!(target, "en");
        assert!(message.is_empty());
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "en".to_string(),
        );
        assert!(!s.is_enabled(), "not yet: nothing is loaded");
        assert_eq!(s.target_language(), "", "so no language is claimed");
        assert!(!s.will_translate(), "and the worker stays inline");

        release(&gate);
        wait_until("Enabled", || s.is_enabled(), true);
        assert_eq!(s.target_language(), "en");
    }

    #[test]
    fn a_switch_keeps_the_old_translator_serving_until_the_new_one_lands() {
        // The property that makes a switch atomic from the reader's side.
        let gate = new_gate();
        let s = any_target_state(gate.clone());
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.is_enabled());

        assert!(s.set_target_language("de").0);
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "de".to_string(),
        );
        assert_eq!(
            s.target_language(),
            "en",
            "the old translator still serves, and that is what is reported"
        );
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert!(s.will_translate(), "so nothing comes out untranslated");
    }

    #[test]
    fn a_load_that_cannot_be_produced_leaves_the_step_off() {
        let s = any_target_state(None);
        assert!(s.set_target_language("unobtainable").0);
        // The load runs on its own thread and fails there.
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "".to_string(),
        );
        assert!(
            !s.is_enabled(),
            "`Enabled` never claims work that cannot happen"
        );
        assert!(!s.is_loaded());
        assert!(!s.will_translate());
    }

    #[test]
    fn a_load_that_cannot_be_produced_keeps_the_language_the_run_had() {
        let s = any_target_state(None);
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.set_target_language("unobtainable").0);
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "".to_string(),
        );
        assert!(
            s.is_enabled(),
            "the old translator is untouched by a failed switch"
        );
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.model(), "opus-mt-pl-en");
    }

    #[test]
    fn a_disable_cancels_the_load_in_flight() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let generation = s.lock_slot().generation;
        // Stand in for a request: the generation the loader would carry.
        let stale = s.begin_request("en");
        assert!(stale > generation);

        s.disable(false);
        // The load is stale now, so its result cannot install.
        assert_ne!(
            s.lock_slot().generation,
            stale,
            "a disable must bump the generation, or an arriving load would still land"
        );
        assert_eq!(s.requested_language(), "", "and clear the request");
    }

    #[test]
    fn a_superseded_load_never_installs() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let stale = s.begin_request("de");
        s.disable(false);
        // `install` with a stale generation returns without touching the slot.
        let before = s.model();
        s.report_stale_install(stale, "de");
        assert_eq!(s.model(), before, "the live translator is untouched");
        assert_eq!(s.target_language(), "en");
    }

    #[test]
    fn disable_without_unload_keeps_the_weights_for_an_instant_enable() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.disable(false), "off, and it returns the new state");
        assert!(!s.is_enabled(), "off");
        assert!(s.is_loaded(), "but still loaded, so Enable costs nothing");
        assert_eq!(
            s.target_language(),
            "en",
            "and it still says what it would write"
        );
        assert_eq!(s.model(), "opus-mt-pl-en");

        let (enabled, message) = s.enable();
        assert!(enabled, "no reload: the translator is right there");
        assert!(message.is_empty());
        assert!(s.will_translate());
    }

    #[test]
    fn disable_with_unload_frees_the_weights_and_the_reported_state() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.disable(true));
        assert!(!s.is_loaded(), "the weights are gone");
        assert_eq!(s.target_language(), "", "nothing is being written");
        assert_eq!(s.model(), "", "so there is no translator to name");
        assert!(
            !s.will_translate(),
            "and the worker stays on the inline path"
        );
    }

    #[test]
    fn an_unload_keeps_the_remembered_target_so_enable_can_reload_it() {
        // Gated, so the reload is still in flight when it is observed: an
        // ungated fake loader can install before `enable` returns.
        let gate = new_gate();
        let s = gated_state("pl", gate.clone());
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(true);
        // `Enable` finds the remembered `en`, so it asks for a load rather than
        // refusing the way a run that never named a language does.
        let (enabled, message) = s.enable();
        assert!(!enabled, "the reload is in flight");
        assert!(message.is_empty(), "not a refusal");
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "en".to_string(),
        );
        release(&gate);
    }

    #[test]
    fn toggle_flips_the_step_without_unloading() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.toggle(), "on -> off");
        assert!(
            s.is_loaded(),
            "a toggle is a state change, not a memory decision"
        );
        assert!(s.toggle(), "off -> on, with no reload");
    }

    #[test]
    fn the_same_language_again_is_a_switch_on_not_a_load() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(false);
        let (accepted, target, message) = s.set_target_language("en");
        assert!(accepted);
        assert_eq!(target, "en");
        assert!(message.is_empty());
        assert!(s.is_enabled(), "the same language turns the step back on");
        assert_eq!(
            s.requested_language(),
            "",
            "and nothing is loaded, because the translator was already there"
        );
    }

    #[test]
    fn the_target_language_is_normalized_the_way_the_catalog_compares_it() {
        let s = state("pl");
        // Trimmed and lowercased, so ` EN ` and `en` are one language and the
        // reported value matches what the catalog resolved.
        let (accepted, target, _) = s.set_target_language(" EN ");
        assert!(accepted);
        assert_eq!(target, "en");
    }

    #[test]
    fn a_failed_first_load_leaves_the_step_off() {
        let s = state("pl");
        let generation = s.begin_request("en");
        s.wanted.store(true, Ordering::Release);
        s.fail_request(generation, "the decoder ran out of cache slots");
        assert!(
            !s.is_enabled(),
            "`Enabled` must never claim work that cannot happen"
        );
        assert_eq!(s.requested_language(), "");
        assert!(!s.will_translate());
    }

    #[test]
    fn a_failed_switch_keeps_the_language_the_run_had() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let generation = s.begin_request("de");
        s.fail_request(generation, "the directory is missing a graph");
        assert!(s.is_enabled(), "the old translator still serves");
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert_eq!(s.requested_language(), "", "the failed request is cleared");
    }

    #[test]
    fn a_superseded_failure_is_not_reported() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let stale = s.begin_request("de");
        let live = s.begin_request("en");
        assert!(live > stale);
        // The first load fails after the second was asked for: its message is
        // about a language nobody wants any more, so it is dropped.
        s.fail_request(stale, "stale failure");
        assert_eq!(
            s.requested_language(),
            "en",
            "the live request is untouched"
        );
    }

    #[test]
    fn get_state_reports_the_snapshot_the_properties_read() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (enabled, target, requested, model) = s.state();
        assert!(enabled);
        assert_eq!(target, "en");
        assert_eq!(requested, "");
        assert_eq!(model, "opus-mt-pl-en");
        // The four reads must agree, or a client stitching them together sees a
        // state no single moment had.
        assert_eq!(enabled, s.is_enabled());
        assert_eq!(target, s.target_language());
        assert_eq!(requested, s.requested_language());
        assert_eq!(model, s.model());
    }
}
