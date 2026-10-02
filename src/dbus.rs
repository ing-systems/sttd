// ---------------------------------------------------------------------------
// Session-bus control surface: Pause/Resume/Toggle plus live dictionary control
// ---------------------------------------------------------------------------
//
// Bus name, object path and interface name are all `org.sttd.Daemon` (v1 has
// no custom-name flag). The service runs the zbus blocking API on one
// dedicated thread; dropping the connection unregisters the name, so the
// connection is parked in a `OnceLock` and kept for the process life.
//
// A missing session bus is not fatal: the thread retries with a backoff
// (2 s, 5 s, 10 s, then 30 s) and claims the name as soon as a bus appears.
// Signal pause/resume never depends on the bus.
//
// File IO in the interface methods is a plain `std::fs` read on purpose: a
// dictionary file is a few kB, and no async filesystem feature is pulled in.

// The public surface is wired by the main lane (startup, signal handlers, ASR
// booster hook); until then the unused items must not warn.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::Duration;

use zbus::blocking::connection::Builder;
use zbus::blocking::Connection;
use zbus::object_server::SignalEmitter;

/// Well-known bus name, object path and interface name (all the same string).
pub(crate) const SERVICE_NAME: &str = "org.sttd.Daemon";
pub(crate) const OBJECT_PATH: &str = "/org/sttd/Daemon";
pub(crate) const INTERFACE_NAME: &str = "org.sttd.Daemon";

/// The connection, kept alive for the process life. `None` until the service
/// thread claims the name (or forever with `--no-dbus`).
static CONNECTION: OnceLock<Connection> = OnceLock::new();

/// State shared with the main loop and the ASR worker.
///
/// `words` is double-wrapped in `Arc` so a reader clones one pointer per
/// utterance instead of copying the list, and no lock is ever held across
/// inference.
#[derive(Clone)]
pub(crate) struct Shared {
    paused: Arc<AtomicBool>,
    words: Arc<RwLock<Arc<Vec<String>>>>,
    source: Arc<RwLock<PathBuf>>,
    /// Called after every successful word swap so the ASR lane can update the
    /// ONNX boost automaton. Attachable after construction, so the wiring in
    /// `main` never has to change when the booster hook is added.
    on_words_swapped: Arc<dyn Fn(Arc<Vec<String>>) + Send + Sync>,
}

impl Shared {
    /// Build the shared state. `source` is the dictionary file the words came
    /// from; an empty path means "no dictionary configured".
    pub(crate) fn new(words: Vec<String>, source: PathBuf) -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
            words: Arc::new(RwLock::new(Arc::new(words))),
            source: Arc::new(RwLock::new(source)),
            on_words_swapped: Arc::new(|words| {
                log::debug!("dictionary: {} word(s) swapped (no booster attached)", words.len());
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

    /// Current pause state.
    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// The pause flag itself, for the main loop and the ASR worker, which
    /// poll an `Arc<AtomicBool>` instead of this handle. One flag serves
    /// D-Bus, the signal handlers and the pipeline.
    pub(crate) fn paused_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.paused)
    }

    /// Set the pause state and return the new value. The main loop polls the
    /// flag, so it performs the pipeline hold/resume edges itself.
    pub(crate) fn set_paused(&self, paused: bool) -> bool {
        self.paused.store(paused, Ordering::Release);
        paused
    }

    /// Flip the pause state and return the new value.
    pub(crate) fn toggle_paused(&self) -> bool {
        !self.paused.fetch_xor(true, Ordering::AcqRel)
    }

    /// Snapshot of the live word list.
    pub(crate) fn words(&self) -> Arc<Vec<String>> {
        Arc::clone(&lock_read(&self.words))
    }

    /// Current dictionary source path (empty when none is configured).
    pub(crate) fn source(&self) -> PathBuf {
        lock_read(&self.source).clone()
    }

    /// Point the reload target at another file. Only called after a
    /// successful load, so a bad path never becomes the reload source.
    pub(crate) fn set_source(&self, path: PathBuf) {
        *lock_write(&self.source) = path;
    }

    /// Sanitize, swap in and report the new word count. The booster hook runs
    /// after the swap, so a reader can never see the new list with a stale
    /// automaton.
    pub(crate) fn set_words(&self, words: Vec<String>) -> u32 {
        let clean = crate::dictionary::sanitize_entries(words.iter().map(String::as_str));
        let swapped = Arc::new(clean);
        let count = swapped.len() as u32;
        *lock_write(&self.words) = Arc::clone(&swapped);
        (self.on_words_swapped)(swapped);
        count
    }
}

/// A poisoned lock must not kill transcription: take the value and continue.
fn lock_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn lock_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// The exported interface. Every mutating method emits its signal so signal
/// and D-Bus clients observe the same transitions.
struct Daemon {
    shared: Shared,
}

#[zbus::interface(name = "org.sttd.Daemon")]
impl Daemon {
    /// Pause transcription. Idempotent.
    async fn pause(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.shared.set_paused(true);
        emitter.state_changed(true).await.map_err(zbus::fdo::Error::ZBus)
    }

    /// Resume transcription. Idempotent. With `--autopause` the ASR lane arms
    /// its one-shot on the resume edge it observes in the main loop.
    async fn resume(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.shared.set_paused(false);
        emitter.state_changed(false).await.map_err(zbus::fdo::Error::ZBus)
    }

    /// Flip the pause state and return the new value.
    async fn toggle(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let paused = self.shared.toggle_paused();
        emitter.state_changed(paused).await?;
        Ok(paused)
    }

    /// Current pause state.
    #[zbus(property)]
    fn paused(&self) -> bool {
        self.shared.is_paused()
    }

    /// Load a dictionary file and swap the live list on success. An empty
    /// `path` reloads the current source. On failure the old list stays
    /// active and the error is reported through the return value and
    /// `DictionaryChanged`.
    #[zbus(out_args("ok", "count", "message"))]
    async fn reload_dictionary(
        &self,
        path: String,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<(bool, u32, String)> {
        let target = if path.trim().is_empty() {
            let current = self.shared.source();
            if current.as_os_str().is_empty() {
                let old_count = self.shared.words().len() as u32;
                let message = "no dictionary path configured".to_string();
                emitter.dictionary_changed(old_count, false, message.as_str()).await?;
                return Ok((false, old_count, message));
            }
            current
        } else {
            PathBuf::from(&path)
        };

        match crate::dictionary::load_dictionary(&target) {
            Ok(words) => {
                let count = self.shared.set_words(words);
                if !path.trim().is_empty() {
                    self.shared.set_source(target);
                }
                emitter.dictionary_changed(count, true, "").await?;
                Ok((true, count, String::new()))
            }
            Err(e) => {
                // Keep the old words; report the failure only.
                let old_count = self.shared.words().len() as u32;
                let message = format!("{e:#}");
                emitter.dictionary_changed(old_count, false, message.as_str()).await?;
                Ok((false, old_count, message))
            }
        }
    }

    /// Replace the live word list from a caller-provided list. The source path
    /// stays unchanged.
    async fn set_words(
        &self,
        words: Vec<String>,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<u32> {
        let count = self.shared.set_words(words);
        emitter.dictionary_changed(count, true, "").await?;
        Ok(count)
    }

    /// The live word list, its source path and its length.
    #[zbus(out_args("words", "path", "count"))]
    fn get_dictionary(&self) -> (Vec<String>, String, u32) {
        let words = self.shared.words();
        let path = self.shared.source();
        let count = words.len() as u32;
        let words = words.as_ref().clone();
        (words, path.to_string_lossy().into_owned(), count)
    }

    /// Pause state transition, from D-Bus calls and from signal handlers.
    #[zbus(signal)]
    async fn state_changed(
        signal_emitter: &SignalEmitter<'_>,
        paused: bool,
    ) -> zbus::Result<()>;

    /// Dictionary list changed, including a failed reload (old words kept).
    #[zbus(signal)]
    async fn dictionary_changed(
        signal_emitter: &SignalEmitter<'_>,
        count: u32,
        ok: bool,
        message: &str,
    ) -> zbus::Result<()>;
}

/// Start the service thread.
///
/// Returns immediately. A missing session bus only logs a warning; the thread
/// retries until it claims the name.
pub(crate) fn spawn(shared: Shared) {
    thread::spawn(move || {
        const BACKOFF: [Duration; 4] = [
            Duration::from_secs(2),
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(30),
        ];
        let mut attempt = 0usize;
        loop {
            match build(shared.clone()) {
                Ok(conn) => {
                    // The name stays claimed for the process life: a drop
                    // would unregister it.
                    if CONNECTION.set(conn).is_err() {
                        log::warn!("dbus: service already running, nothing to do");
                    } else {
                        log::info!("dbus: serving {INTERFACE_NAME} on {OBJECT_PATH}");
                    }
                    return;
                }
                Err(e) => {
                    let wait = BACKOFF[attempt.min(BACKOFF.len() - 1)];
                    attempt += 1;
                    log::warn!("dbus: {e}; retrying in {}s (signals unaffected)", wait.as_secs());
                    thread::sleep(wait);
                }
            }
        }
    });
}

/// Connect to the session bus, register the interface and claim the name.
fn build(shared: Shared) -> zbus::Result<Connection> {
    Builder::session()?
        .name(SERVICE_NAME)?
        .serve_at(OBJECT_PATH, Daemon { shared })?
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
    let emitter = match SignalEmitter::new(conn.inner(), OBJECT_PATH) {
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
    let emitter = match SignalEmitter::new(conn.inner(), OBJECT_PATH) {
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

    fn shared_with(words: &[&str]) -> Shared {
        Shared::new(
            words.iter().map(|w| w.to_string()).collect(),
            PathBuf::new(),
        )
    }

    #[test]
    fn toggle_returns_the_new_state() {
        let shared = shared_with(&[]);
        assert!(shared.toggle_paused());
        assert!(shared.is_paused());
        assert!(!shared.toggle_paused());
        assert!(!shared.is_paused());
        assert!(shared.set_paused(true));
        assert!(shared.is_paused());
    }

    #[test]
    fn set_words_sanitizes_like_the_file_path() {
        let shared = shared_with(&["old"]);
        let count = shared.set_words(vec![
            "AppImage".to_string(),
            "appimage".to_string(), // normalized duplicate
            "123".to_string(),      // digits rejected
            "a b c".to_string(),    // more than two words rejected
            "żółć".to_string(),
        ]);
        assert_eq!(count, 2);
        assert_eq!(
            *shared.words(),
            vec!["AppImage".to_string(), "żółć".to_string()]
        );
    }

    #[test]
    fn words_hook_runs_after_the_swap() {
        let seen: Arc<RwLock<Option<Arc<Vec<String>>>>> = Arc::new(RwLock::new(None));
        let sink = Arc::clone(&seen);
        let shared =
            shared_with(&[]).with_words_hook(Arc::new(move |words| *lock_write(&sink) = Some(words)));
        shared.set_words(vec!["Parakeet".to_string()]);
        // The hook observes the already-swapped list.
        assert_eq!(*lock_read(&seen), Some(Arc::new(vec!["Parakeet".to_string()])));
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
        assert_eq!(OBJECT_PATH, "/org/sttd/Daemon");
        assert_eq!(INTERFACE_NAME, "org.sttd.Daemon");
    }
}
