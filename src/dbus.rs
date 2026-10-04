// ---------------------------------------------------------------------------
// Session-bus control surface: pause control and live dictionary control
// ---------------------------------------------------------------------------
//
// One well-known name, one connection, one thread, two object paths:
//
//   org.sttd.Daemon     /org/sttd/Daemon      Pause/Resume/Toggle + Paused
//   org.sttd.Dictionary /org/sttd/Dictionary  LoadFile/ReplaceWords/AddWords/
//                                            RemoveWords/GetWords/Contains
//                                            + Count, Dirty
//
// The service runs the zbus blocking API on one dedicated thread; dropping the
// connection unregisters the name, so the connection is parked in a `OnceLock`
// and kept for the process life.
//
// A missing session bus is not fatal: the thread retries with a backoff
// (2 s, 5 s, 10 s, then 30 s) and claims the name as soon as a bus appears.
// Signal pause/resume never depends on the bus.
//
// File IO in `LoadFile` is a plain `std::fs` read on purpose: a dictionary
// file is a few kB, and no async filesystem feature is pulled in.
//
// The D-Bus interface names, the zbus method signatures and the mutators
// behind them are reached only through `spawn`, and the ones below it through
// `asr`/`main`; the compiler cannot see across those dynamic boundaries.
#![allow(dead_code)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::Duration;

use zbus::blocking::connection::Builder;
use zbus::blocking::Connection;
use zbus::object_server::SignalEmitter;

/// Well-known bus name. Both interfaces are served on it.
pub(crate) const SERVICE_NAME: &str = "org.sttd.Daemon";

/// Pause control.
pub(crate) const DAEMON_PATH: &str = "/org/sttd/Daemon";
pub(crate) const DAEMON_INTERFACE: &str = "org.sttd.Daemon";

/// Live dictionary control.
pub(crate) const DICT_PATH: &str = "/org/sttd/Dictionary";
pub(crate) const DICT_INTERFACE: &str = "org.sttd.Dictionary";

/// The connection, kept alive for the process life. `None` until the service
/// thread claims the name (or forever with `--no-dbus`).
static CONNECTION: OnceLock<Connection> = OnceLock::new();

/// A poisoned lock must not kill transcription: take the value and continue.
fn lock_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn lock_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Pause state
// ---------------------------------------------------------------------------

/// The pause flag shared by D-Bus, the signal handlers, the main loop and the
/// ASR worker.
///
/// The main loop polls the `AtomicBool`, so this type records the state and
/// performs no pipeline edge itself.
#[derive(Clone)]
pub(crate) struct PauseState {
    paused: Arc<AtomicBool>,
}

impl PauseState {
    /// Build the pause state with its initial value (`--start-paused`).
    pub(crate) fn new(start_paused: bool) -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(start_paused)),
        }
    }

    /// Current pause state.
    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// The pause flag itself, for the main loop and the ASR worker, which poll
    /// an `Arc<AtomicBool>` instead of this handle. One flag serves D-Bus, the
    /// signal handlers and the pipeline.
    pub(crate) fn paused_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.paused)
    }

    /// Set the pause state and return the new value.
    pub(crate) fn set_paused(&self, paused: bool) -> bool {
        self.paused.store(paused, Ordering::Release);
        paused
    }

    /// Flip the pause state and return the new value.
    pub(crate) fn toggle_paused(&self) -> bool {
        !self.paused.fetch_xor(true, Ordering::AcqRel)
    }
}

// ---------------------------------------------------------------------------
// Dictionary state
// ---------------------------------------------------------------------------

/// The live word list shared by the bus thread, the main loop and the ASR
/// worker.
///
/// `words` is double-wrapped in `Arc` so a reader clones one pointer per
/// utterance instead of copying the list, and no lock is ever held across
/// inference. `dirty` records that the live list diverged from `source`, so a
/// caller knows a `LoadFile` would discard its edits.
#[derive(Clone)]
pub(crate) struct DictionaryState {
    words: Arc<RwLock<Arc<Vec<String>>>>,
    source: Arc<RwLock<PathBuf>>,
    dirty: Arc<AtomicBool>,
    /// Called after every successful word swap so the ASR lane can update the
    /// ONNX boost automaton. Attachable after construction, so the wiring in
    /// `main` never has to change when the booster hook is added.
    on_words_swapped: Arc<dyn Fn(Arc<Vec<String>>) + Send + Sync>,
}

impl DictionaryState {
    /// Build the dictionary state. `source` is the dictionary file the words
    /// came from; an empty path means "no dictionary configured". The list
    /// starts clean, because a startup load matches its own source.
    pub(crate) fn new(words: Vec<String>, source: PathBuf) -> Self {
        Self {
            words: Arc::new(RwLock::new(Arc::new(words))),
            source: Arc::new(RwLock::new(source)),
            dirty: Arc::new(AtomicBool::new(false)),
            on_words_swapped: Arc::new(|words| {
                log::debug!(
                    "dictionary: {} word(s) swapped (no booster attached)",
                    words.len()
                );
            }),
        }
    }

    /// Attach the booster update callback (ASR lane).
    pub(crate) fn with_words_hook(
        mut self,
        hook: Arc<dyn Fn(Arc<Vec<String>>) + Send + Sync>,
    ) -> Self {
        self.on_words_swapped = hook;
        self
    }

    /// Snapshot of the live word list.
    pub(crate) fn words(&self) -> Arc<Vec<String>> {
        Arc::clone(&lock_read(&self.words))
    }

    /// Number of entries in the live list. Uncapped: the `--dictionary` caps
    /// apply in post-correction, not at swap time, so this can exceed
    /// `POSTCORR_MAX_DICT_WORDS`.
    pub(crate) fn count(&self) -> u32 {
        lock_read(&self.words).len() as u32
    }

    /// Whether the live list diverged from the source file.
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Current dictionary source path (empty when none is configured).
    pub(crate) fn source(&self) -> PathBuf {
        lock_read(&self.source).clone()
    }

    /// Replace the whole list from a caller-provided one. The source path stays
    /// unchanged, so the result counts as dirty: the server cannot know that a
    /// caller-supplied list matches the file. Returns the new count.
    pub(crate) fn replace_words(&self, words: Vec<String>) -> u32 {
        self.swap(crate::dictionary::sanitize_entries(
            words.iter().map(String::as_str),
        ));
        self.mark_dirty(true);
        self.count()
    }

    /// Merge entries into the live list.
    ///
    /// Sanitizing matches the file path: entries with digits, entries of more
    /// than two words, and duplicates within the input are dropped. A
    /// normalized collision with a live entry keeps the live entry, so an add
    /// never re-cases an existing word. Returns `(count, added)`, where
    /// `added` is 0 when every entry was already present.
    pub(crate) fn add_words(&self, words: Vec<String>) -> (u32, u32) {
        let clean = crate::dictionary::sanitize_entries(words.iter().map(String::as_str));
        let mut merged = lock_read(&self.words).as_ref().clone();
        let before = merged.len();
        let mut seen: HashSet<String> = merged
            .iter()
            .map(|w| crate::dictionary::normalize_word(w))
            .collect();
        for entry in clean {
            if seen.insert(crate::dictionary::normalize_word(&entry)) {
                merged.push(entry);
            }
        }
        let added = merged.len() - before;
        if added > 0 {
            self.swap(merged);
            self.mark_dirty(true);
        }
        (self.count(), added as u32)
    }

    /// Remove entries by normalized form, so `APPIMAGE` drops `AppImage` and
    /// `ZOLC` drops `żółć`. Returns `(count, removed)`, where `removed` is 0
    /// when nothing matched.
    pub(crate) fn remove_words(&self, words: Vec<String>) -> (u32, u32) {
        let drop: HashSet<String> = words
            .iter()
            .map(|w| crate::dictionary::normalize_word(w))
            .collect();
        let existing = lock_read(&self.words).as_ref().clone();
        let kept: Vec<String> = existing
            .iter()
            .filter(|w| !drop.contains(&crate::dictionary::normalize_word(w)))
            .cloned()
            .collect();
        let removed = existing.len() - kept.len();
        if removed > 0 {
            self.swap(kept);
            self.mark_dirty(true);
        }
        (self.count(), removed as u32)
    }

    /// Membership by normalized form, so `ZOLC` matches `żółć`.
    pub(crate) fn contains(&self, word: &str) -> bool {
        let key = crate::dictionary::normalize_word(word);
        lock_read(&self.words)
            .iter()
            .any(|w| crate::dictionary::normalize_word(w) == key)
    }

    /// Load a dictionary file and swap the live list.
    ///
    /// On success the path becomes the new source and the list is clean again.
    /// On failure the old words, the old source and the dirty flag all stay as
    /// they were.
    pub(crate) fn load_file(&self, path: &Path) -> anyhow::Result<u32> {
        let words = crate::dictionary::load_dictionary(path)?;
        self.swap(words);
        *lock_write(&self.source) = path.to_path_buf();
        self.mark_dirty(false);
        Ok(self.count())
    }

    /// Swap in an already-sanitized list and run the booster hook after it, so
    /// a reader can never see the new list with a stale automaton.
    fn swap(&self, words: Vec<String>) {
        let swapped = Arc::new(words);
        *lock_write(&self.words) = Arc::clone(&swapped);
        (self.on_words_swapped)(swapped);
    }

    fn mark_dirty(&self, dirty: bool) {
        self.dirty.store(dirty, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Pause interface
// ---------------------------------------------------------------------------

/// The pause control interface. Every mutating method emits `StateChanged`, so
/// signal and D-Bus clients observe the same transitions.
struct Daemon {
    state: PauseState,
}

#[zbus::interface(name = "org.sttd.Daemon")]
impl Daemon {
    /// Pause transcription. Idempotent.
    async fn pause(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.state.set_paused(true);
        emitter
            .state_changed(true)
            .await
            .map_err(zbus::fdo::Error::ZBus)
    }

    /// Resume transcription. Idempotent. With `--autopause` the ASR lane arms
    /// its one-shot on the resume edge it observes in the main loop.
    async fn resume(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.state.set_paused(false);
        emitter
            .state_changed(false)
            .await
            .map_err(zbus::fdo::Error::ZBus)
    }

    /// Flip the pause state and return the new value.
    async fn toggle(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let paused = self.state.toggle_paused();
        emitter.state_changed(paused).await?;
        Ok(paused)
    }

    /// Current pause state.
    #[zbus(property)]
    fn paused(&self) -> bool {
        self.state.is_paused()
    }

    /// Pause state transition, from D-Bus calls and from signal handlers.
    #[zbus(signal)]
    async fn state_changed(signal_emitter: &SignalEmitter<'_>, paused: bool) -> zbus::Result<()>;
}

// ---------------------------------------------------------------------------
// Dictionary interface
// ---------------------------------------------------------------------------

/// The live dictionary control interface.
///
/// `Count` and `Dirty` are point reads: zbus does not emit `PropertiesChanged`
/// for a getter-only property, so a client that tracks changes subscribes to
/// `DictionaryChanged`, whose first argument is the count after every
/// mutation.
struct DictionaryIface {
    state: DictionaryState,
}

#[zbus::interface(name = "org.sttd.Dictionary")]
impl DictionaryIface {
    /// Load a dictionary file and swap the live list.
    ///
    /// On success the path becomes the new source and `Dirty` clears. An empty
    /// or whitespace-only path is rejected. On failure the old list stays
    /// active and the error is reported through the return value and
    /// `DictionaryChanged`.
    #[zbus(out_args("ok", "count", "message"))]
    async fn load_file(
        &self,
        path: String,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<(bool, u32, String)> {
        if path.trim().is_empty() {
            let old_count = self.state.count();
            let message = "empty path".to_string();
            emitter
                .dictionary_changed(old_count, false, message.as_str())
                .await?;
            return Ok((false, old_count, message));
        }
        // The untrimmed path is the real one: a filename may end in a space.
        match self.state.load_file(&PathBuf::from(&path)) {
            Ok(count) => {
                emitter.dictionary_changed(count, true, "").await?;
                Ok((true, count, String::new()))
            }
            Err(e) => {
                // Keep the old words; report the failure only.
                let old_count = self.state.count();
                let message = format!("{e:#}");
                emitter
                    .dictionary_changed(old_count, false, message.as_str())
                    .await?;
                Ok((false, old_count, message))
            }
        }
    }

    /// Replace the whole live list from a caller-provided one.
    ///
    /// The source path stays unchanged, so `Dirty` becomes true: a reload of
    /// that source discards this list. Returns the new count.
    async fn replace_words(
        &self,
        words: Vec<String>,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<u32> {
        let count = self.state.replace_words(words);
        emitter.dictionary_changed(count, true, "").await?;
        Ok(count)
    }

    /// Merge entries into the live list. Returns `(count, added)`.
    ///
    /// A normalized collision with a live entry keeps the live entry, so
    /// `added` is 0 for a pure re-case such as `appimage` against a loaded
    /// `AppImage`.
    #[zbus(out_args("count", "added"))]
    async fn add_words(
        &self,
        words: Vec<String>,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<(u32, u32)> {
        let (count, added) = self.state.add_words(words);
        emitter.dictionary_changed(count, true, "").await?;
        Ok((count, added))
    }

    /// Remove entries by normalized form. Returns `(count, removed)`.
    #[zbus(out_args("count", "removed"))]
    async fn remove_words(
        &self,
        words: Vec<String>,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<(u32, u32)> {
        let (count, removed) = self.state.remove_words(words);
        emitter.dictionary_changed(count, true, "").await?;
        Ok((count, removed))
    }

    /// The live word list, its source path, its length and the dirty flag.
    #[zbus(out_args("words", "path", "count", "dirty"))]
    fn get_words(&self) -> (Vec<String>, String, u32, bool) {
        let words = self.state.words();
        (
            words.as_ref().clone(),
            self.state.source().to_string_lossy().into_owned(),
            self.state.count(),
            self.state.is_dirty(),
        )
    }

    /// Whether the live list holds `word`, compared case- and
    /// accent-insensitively: `ZOLC` matches `żółć`.
    fn contains(&self, word: String) -> bool {
        self.state.contains(&word)
    }

    /// Number of entries in the live list. Uncapped, so it can exceed
    /// `POSTCORR_MAX_DICT_WORDS`.
    #[zbus(property)]
    fn count(&self) -> u32 {
        self.state.count()
    }

    /// Whether the live list diverged from the source file.
    #[zbus(property)]
    fn dirty(&self) -> bool {
        self.state.is_dirty()
    }

    /// Dictionary list changed, including a failed load (old words kept).
    #[zbus(signal)]
    async fn dictionary_changed(
        signal_emitter: &SignalEmitter<'_>,
        count: u32,
        ok: bool,
        message: &str,
    ) -> zbus::Result<()>;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Start the service thread.
///
/// Returns immediately. A missing session bus only logs a warning; the thread
/// retries until it claims the name.
pub(crate) fn spawn(paused: PauseState, dictionary: DictionaryState) {
    thread::spawn(move || {
        const BACKOFF: [Duration; 4] = [
            Duration::from_secs(2),
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(30),
        ];
        let mut attempt = 0usize;
        loop {
            match build(paused.clone(), dictionary.clone()) {
                Ok(conn) => {
                    // The name stays claimed for the process life: a drop
                    // would unregister it.
                    if CONNECTION.set(conn).is_err() {
                        log::warn!("dbus: service already running, nothing to do");
                    } else {
                        log::info!(
                            "dbus: serving {DAEMON_INTERFACE} on {DAEMON_PATH} \
                             and {DICT_INTERFACE} on {DICT_PATH}"
                        );
                    }
                    return;
                }
                Err(e) => {
                    let wait = BACKOFF[attempt.min(BACKOFF.len() - 1)];
                    attempt += 1;
                    log::warn!(
                        "dbus: {e}; retrying in {}s (signals unaffected)",
                        wait.as_secs()
                    );
                    thread::sleep(wait);
                }
            }
        }
    });
}

/// Connect to the session bus, register both interfaces and claim the name.
fn build(paused: PauseState, dictionary: DictionaryState) -> zbus::Result<Connection> {
    Builder::session()?
        .name(SERVICE_NAME)?
        .serve_at(DAEMON_PATH, Daemon { state: paused })?
        .serve_at(DICT_PATH, DictionaryIface { state: dictionary })?
        .build()
}

/// Emit `StateChanged` from outside an interface method (signal handlers).
///
/// Silent no-op when the bus is not available: the caller is a signal
/// handler, and a pause must never fail because the bus is missing.
pub(crate) fn emit_state_changed(paused: bool) {
    let Some(conn) = CONNECTION.get() else {
        return;
    };
    let emitter = match SignalEmitter::new(conn.inner(), DAEMON_PATH) {
        Ok(emitter) => emitter,
        Err(e) => {
            log::debug!("dbus: no signal emitter ({e})");
            return;
        }
    };
    if let Err(e) = zbus::block_on(emitter.state_changed(paused)) {
        log::warn!("dbus: StateChanged({paused}) failed: {e}");
    }
}

/// Emit `DictionaryChanged` from outside an interface method. Silent no-op
/// without a connection, like `emit_state_changed`.
pub(crate) fn emit_dictionary_changed(count: u32, ok: bool, message: &str) {
    let Some(conn) = CONNECTION.get() else {
        return;
    };
    let emitter = match SignalEmitter::new(conn.inner(), DICT_PATH) {
        Ok(emitter) => emitter,
        Err(e) => {
            log::debug!("dbus: no signal emitter ({e})");
            return;
        }
    };
    if let Err(e) = zbus::block_on(emitter.dictionary_changed(count, ok, message)) {
        log::warn!("dbus: DictionaryChanged({count}, {ok}) failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pause_state() -> PauseState {
        PauseState::new(false)
    }

    fn dictionary_with(words: &[&str]) -> DictionaryState {
        DictionaryState::new(
            words.iter().map(|w| w.to_string()).collect(),
            PathBuf::new(),
        )
    }

    /// A dictionary file in the test temp dir, per the `wav.rs` fixture
    /// pattern. The caller removes it.
    fn write_temp_dictionary(name: &str, body: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("sttd-dbus-test-{}-{name}", std::process::id()));
        std::fs::write(&path, body).expect("temp dictionary must be writable");
        path
    }

    #[test]
    fn toggle_returns_the_new_state() {
        let state = pause_state();
        assert!(state.toggle_paused());
        assert!(state.is_paused());
        assert!(!state.toggle_paused());
        assert!(!state.is_paused());
        assert!(state.set_paused(true));
        assert!(state.is_paused());
    }

    #[test]
    fn pause_state_starts_from_the_initial_value() {
        assert!(!pause_state().is_paused());
        assert!(PauseState::new(true).is_paused());
    }

    #[test]
    fn pause_flag_is_shared_with_the_polling_lanes() {
        let state = pause_state();
        let flag = state.paused_flag();
        state.set_paused(true);
        assert!(
            flag.load(Ordering::Acquire),
            "the poller must see the change"
        );
    }

    #[test]
    fn replace_words_sanitizes_like_the_file_path() {
        let state = dictionary_with(&["old"]);
        let count = state.replace_words(vec![
            "AppImage".to_string(),
            "appimage".to_string(), // normalized duplicate
            "123".to_string(),      // digits rejected
            "a b c".to_string(),    // more than two words rejected
            "żółć".to_string(),
        ]);
        assert_eq!(count, 2);
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "żółć".to_string()]
        );
        assert!(state.is_dirty(), "a caller-supplied list diverges");
    }

    #[test]
    fn words_hook_runs_after_the_swap() {
        let seen: Arc<RwLock<Option<Arc<Vec<String>>>>> = Arc::new(RwLock::new(None));
        let sink = Arc::clone(&seen);
        let state = dictionary_with(&[])
            .with_words_hook(Arc::new(move |words| *lock_write(&sink) = Some(words)));
        state.replace_words(vec!["Parakeet".to_string()]);
        // The hook observes the already-swapped list.
        assert_eq!(
            *lock_read(&seen),
            Some(Arc::new(vec!["Parakeet".to_string()]))
        );
    }

    #[test]
    fn add_words_merges_and_counts() {
        let state = dictionary_with(&["AppImage"]);
        let (count, added) = state.add_words(vec![
            "Syntocinon".to_string(),
            "1Password".to_string(), // digits rejected
        ]);
        assert_eq!(added, 1);
        assert_eq!(count, 2);
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "Syntocinon".to_string()]
        );
        assert!(state.is_dirty());
    }

    #[test]
    fn add_words_keeps_the_existing_casing() {
        let state = dictionary_with(&["AppImage"]);
        let (count, added) = state.add_words(vec!["appimage".to_string()]);
        assert_eq!(added, 0, "a re-case is not a new entry");
        assert_eq!(count, 1);
        assert_eq!(*state.words(), vec!["AppImage".to_string()]);
    }

    #[test]
    fn add_words_of_nothing_leaves_the_dirty_flag_alone() {
        let state = dictionary_with(&["AppImage"]);
        let (_, added) = state.add_words(vec!["appimage".to_string()]);
        assert_eq!(added, 0);
        assert!(!state.is_dirty(), "nothing changed, so nothing diverged");
    }

    #[test]
    fn remove_words_matches_case_insensitively() {
        let state = dictionary_with(&["AppImage", "żółć", "Syntocinon"]);
        let (count, removed) = state.remove_words(vec!["APPIMAGE".to_string(), "ZOLC".to_string()]);
        assert_eq!(removed, 2);
        assert_eq!(count, 1);
        assert_eq!(*state.words(), vec!["Syntocinon".to_string()]);
        assert!(state.is_dirty());
    }

    #[test]
    fn remove_words_reports_zero_when_absent() {
        let state = dictionary_with(&["AppImage"]);
        let (count, removed) = state.remove_words(vec!["nothing-like-this".to_string()]);
        assert_eq!(removed, 0);
        assert_eq!(count, 1);
        assert!(!state.is_dirty(), "nothing changed, so nothing diverged");
    }

    #[test]
    fn contains_folds_case_and_accents() {
        let state = dictionary_with(&["AppImage", "żółć"]);
        assert!(state.contains("appimage"));
        assert!(state.contains("ZOLC"));
        assert!(state.contains("żółć"));
        assert!(state.contains("ŻÓŁĆ"), "uppercase diacritics fold too");
        assert!(!state.contains("syntocinon"));
    }

    #[test]
    fn count_reflects_every_mutation() {
        let state = dictionary_with(&["a", "b"]);
        assert_eq!(state.count(), 2);
        state.replace_words(vec!["only".to_string()]);
        assert_eq!(state.count(), 1);
        assert_eq!(state.add_words(vec!["two".to_string()]).0, 2);
        assert_eq!(state.remove_words(vec!["only".to_string()]).0, 1);
    }

    #[test]
    fn load_file_sets_the_source_and_clears_dirty() {
        let path = write_temp_dictionary("ok.txt", "AppImage\nSyntocinon\n");
        let state = dictionary_with(&["stale"]);
        state.replace_words(vec!["edited".to_string()]);
        assert!(state.is_dirty());

        let count = state.load_file(&path).expect("the temp file must load");
        assert_eq!(count, 2);
        assert_eq!(state.source(), path);
        assert!(!state.is_dirty(), "the list now matches the source");
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "Syntocinon".to_string()]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_file_failure_keeps_the_old_state() {
        let path = std::env::temp_dir().join(format!("sttd-dbus-missing-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let state = dictionary_with(&["kept"]);
        state.replace_words(vec!["kept".to_string()]);
        assert!(state.is_dirty());

        assert!(state.load_file(&path).is_err());
        assert_eq!(*state.words(), vec!["kept".to_string()]);
        assert!(state.source().as_os_str().is_empty(), "source stays unset");
        assert!(state.is_dirty(), "a failed load changes nothing");
    }

    #[test]
    fn emit_without_a_connection_is_a_noop() {
        // No connection was ever set in this test binary run: both emitters
        // must return quietly instead of panicking.
        emit_state_changed(true);
        emit_dictionary_changed(0, false, "test");
    }

    #[test]
    fn identity_strings_are_the_documented_ones() {
        assert_eq!(SERVICE_NAME, "org.sttd.Daemon");
        assert_eq!(DAEMON_PATH, "/org/sttd/Daemon");
        assert_eq!(DAEMON_INTERFACE, "org.sttd.Daemon");
        assert_eq!(DICT_PATH, "/org/sttd/Dictionary");
        assert_eq!(DICT_INTERFACE, "org.sttd.Dictionary");
    }

    // -- real-bus round trip ------------------------------------------------
    //
    // The macro owns the wire names, so unit tests cannot see them. This check
    // drives the served objects over a real session bus. It is `#[ignore]`d
    // because it needs a bus; run it with:
    //
    //   dbus-run-session -- cargo test -- --ignored real_bus
    //
    // One test, not several: a process holds one bus name and one
    // `CONNECTION`, so parallel tests would race for both.

    /// Wait for the service thread to claim the name.
    fn connect_to_service() -> Connection {
        for _ in 0..200 {
            if let Ok(conn) = Connection::session() {
                let claimed = conn
                    .call_method(
                        Some("org.freedesktop.DBus"),
                        "/org/freedesktop/DBus",
                        Some("org.freedesktop.DBus"),
                        "NameHasOwner",
                        &(SERVICE_NAME,),
                    )
                    .and_then(|reply| reply.body().deserialize::<bool>())
                    .unwrap_or(false);
                if claimed {
                    return conn;
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!("{SERVICE_NAME} never appeared; is DBUS_SESSION_BUS_ADDRESS set?");
    }

    fn introspect(conn: &Connection, path: &str) -> String {
        conn.call_method(
            Some(SERVICE_NAME),
            path,
            Some("org.freedesktop.DBus.Introspectable"),
            "Introspect",
            &(),
        )
        .and_then(|reply| reply.body().deserialize::<String>())
        .unwrap_or_else(|e| panic!("Introspect {path} must succeed: {e}"))
    }

    fn call<B>(
        conn: &Connection,
        path: &str,
        iface: &str,
        member: &str,
        body: &B,
    ) -> zbus::Result<zbus::Message>
    where
        B: zbus::zvariant::export::serde::ser::Serialize + zbus::zvariant::DynamicType,
    {
        conn.call_method(Some(SERVICE_NAME), path, Some(iface), member, body)
    }

    /// Read a property and deserialize its value out of the returned variant.
    fn get_property<T>(conn: &Connection, path: &str, name: &str) -> T
    where
        T: for<'a> TryFrom<&'a zbus::zvariant::Value<'a>, Error = zbus::zvariant::Error>,
    {
        let reply = conn
            .call_method(
                Some(SERVICE_NAME),
                path,
                Some("org.freedesktop.DBus.Properties"),
                "Get",
                &(DICT_INTERFACE, name),
            )
            .unwrap_or_else(|e| panic!("Get {name} must succeed: {e}"));
        let body = reply.body();
        let value = body
            .deserialize::<zbus::zvariant::Value<'_>>()
            .unwrap_or_else(|e| panic!("{name} must be a variant: {e}"));
        value
            .downcast_ref::<T>()
            .unwrap_or_else(|e| panic!("{name} must hold the documented type: {e}"))
    }

    /// The whole served surface, driven over a real bus: both interfaces at
    /// their documented paths with the documented members, the old dictionary
    /// members gone from the pause interface, and the mutators reaching the
    /// shared state.
    ///
    /// One test, not two: a process holds one bus name and one `CONNECTION`,
    /// so two parallel tests would race for it.
    #[test]
    #[ignore = "needs a session bus"]
    fn real_bus_round_trip() {
        spawn(PauseState::new(false), dictionary_with(&["AppImage"]));
        let conn = connect_to_service();

        // -- introspection: the wire names the macro generated ------------
        let daemon_xml = introspect(&conn, DAEMON_PATH);
        assert!(daemon_xml.contains(DAEMON_INTERFACE), "{daemon_xml}");
        for member in ["Pause", "Resume", "Toggle", "Paused", "StateChanged"] {
            assert!(
                daemon_xml.contains(member),
                "pause interface missing {member}"
            );
        }
        // The dictionary members moved off this interface.
        for gone in ["ReloadDictionary", "SetWords", "GetDictionary"] {
            assert!(
                !daemon_xml.contains(gone),
                "{gone} must not be on {DAEMON_PATH}"
            );
        }

        let dict_xml = introspect(&conn, DICT_PATH);
        assert!(dict_xml.contains(DICT_INTERFACE), "{dict_xml}");
        for member in [
            "LoadFile",
            "ReplaceWords",
            "AddWords",
            "RemoveWords",
            "GetWords",
            "Contains",
            "Count",
            "Dirty",
            "DictionaryChanged",
        ] {
            assert!(
                dict_xml.contains(member),
                "dictionary interface missing {member}"
            );
        }

        // -- pause: the bus call flips the flag the lane polls -------------
        let reply =
            call(&conn, DAEMON_PATH, DAEMON_INTERFACE, "Toggle", &()).expect("Toggle must succeed");
        assert!(reply
            .body()
            .deserialize::<bool>()
            .expect("Toggle returns b"));
        let paused_reply = conn
            .call_method(
                Some(SERVICE_NAME),
                DAEMON_PATH,
                Some("org.freedesktop.DBus.Properties"),
                "Get",
                &(DAEMON_INTERFACE, "Paused"),
            )
            .expect("Paused must be readable");
        let paused_body = paused_reply.body();
        let paused = paused_body
            .deserialize::<zbus::zvariant::Value<'_>>()
            .expect("variant");
        assert!(paused.downcast_ref::<bool>().expect("Paused is b"));

        // -- dictionary: merge, membership, properties -------------------
        // AddWords merges and reports both totals.
        let reply = call(
            &conn,
            DICT_PATH,
            DICT_INTERFACE,
            "AddWords",
            &(vec!["Syntocinon".to_string(), "appimage".to_string()],),
        )
        .expect("AddWords must succeed");
        let (count, added) = reply
            .body()
            .deserialize::<(u32, u32)>()
            .expect("AddWords returns (u, u)");
        assert_eq!((count, added), (2, 1), "a re-case must not count as new");

        // Contains folds case and accents.
        let reply = call(&conn, DICT_PATH, DICT_INTERFACE, "Contains", &("appimage",))
            .expect("Contains must succeed");
        assert!(
            reply
                .body()
                .deserialize::<bool>()
                .expect("Contains returns b"),
            "Contains must match case-insensitively"
        );
        let reply = call(&conn, DICT_PATH, DICT_INTERFACE, "Contains", &("absent",))
            .expect("Contains must succeed");
        assert!(
            !reply
                .body()
                .deserialize::<bool>()
                .expect("Contains returns b"),
            "Contains must reject an absent word"
        );

        // The properties report the same state.
        assert_eq!(get_property::<u32>(&conn, DICT_PATH, "Count"), 2);
        assert!(get_property::<bool>(&conn, DICT_PATH, "Dirty"));

        // An empty path is rejected instead of reloading the source.
        let reply = call(&conn, DICT_PATH, DICT_INTERFACE, "LoadFile", &("   ",))
            .expect("LoadFile must answer, not fail");
        let (ok, count, message) = reply
            .body()
            .deserialize::<(bool, u32, String)>()
            .expect("LoadFile returns (b, u, s)");
        assert!(!ok);
        assert_eq!(count, 2, "a rejected load keeps the old list");
        assert_eq!(message, "empty path");

        // GetWords reports the four documented fields.
        let reply =
            call(&conn, DICT_PATH, DICT_INTERFACE, "GetWords", &()).expect("GetWords must succeed");
        let (words, path, count, dirty) = reply
            .body()
            .deserialize::<(Vec<String>, String, u32, bool)>()
            .expect("GetWords returns (as, s, u, b)");
        assert_eq!(
            words,
            vec!["AppImage".to_string(), "Syntocinon".to_string()]
        );
        assert_eq!(path, "", "no source was configured");
        assert_eq!(count, 2);
        assert!(dirty);

        // RemoveWords drops by normalized form.
        let reply = call(
            &conn,
            DICT_PATH,
            DICT_INTERFACE,
            "RemoveWords",
            &(vec!["APPIMAGE".to_string()],),
        )
        .expect("RemoveWords must succeed");
        let (count, removed) = reply
            .body()
            .deserialize::<(u32, u32)>()
            .expect("RemoveWords returns (u, u)");
        assert_eq!((count, removed), (1, 1));
        assert_eq!(get_property::<u32>(&conn, DICT_PATH, "Count"), 1);

        // ReplaceWords swaps the whole list in one step.
        let reply = call(
            &conn,
            DICT_PATH,
            DICT_INTERFACE,
            "ReplaceWords",
            &(vec!["zoxide".to_string(), "7zip".to_string()],),
        )
        .expect("ReplaceWords must succeed");
        assert_eq!(
            reply.body().deserialize::<u32>().expect("returns u"),
            1,
            "an entry with a digit is rejected"
        );
        assert!(
            !conn
                .call_method(
                    Some(SERVICE_NAME),
                    DICT_PATH,
                    Some(DICT_INTERFACE),
                    "Contains",
                    &("Syntocinon",),
                )
                .map(|reply| reply.body().deserialize::<bool>().unwrap_or(false))
                .unwrap_or(false),
            "ReplaceWords must drop the old list"
        );

        // The moved member is gone from the pause interface at runtime.
        assert!(
            call(
                &conn,
                DAEMON_PATH,
                DAEMON_INTERFACE,
                "SetWords",
                &(vec!["AppImage".to_string()],)
            )
            .is_err(),
            "SetWords must be UnknownMethod on {DAEMON_INTERFACE}"
        );
    }
}
