// ---------------------------------------------------------------------------
// Session-bus control surface: pause control, live dictionary control and
// translation control
// ---------------------------------------------------------------------------
//
// One well-known name, one connection, one thread, three object paths:
//
//   org.sttd.Daemon     /org/sttd/Daemon      Pause/Resume/Toggle + Paused
//   org.sttd.Dictionary /org/sttd/Dictionary  LoadFile/ReplaceWords/AddWords/
//                                            RemoveWords/GetWords/Contains
//                                            + Count, Dirty
//   org.sttd.Translation /org/sttd/Translation Enable/Disable/Toggle/
//                                            SetTargetLanguage/GetState
//                                            + Enabled, TargetLanguage,
//                                            RequestedLanguage, Model
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
// **The translation interface never blocks on a model.** `SetTargetLanguage`
// answers in milliseconds, refusing an impossible direction with the message the
// startup path prints, and the load itself runs on `translate`'s own thread: this
// one thread also serves `Pause` and the dictionary, and a first use downloads
// 114 MiB to 1.3 GiB. What a client waits on instead is `RequestedLanguage`
// becoming empty, or the `TranslationChanged` signal.
//
// The state the interfaces mutate is not owned here: pause and dictionary
// state live in `control`, the translation switch in `translate`. This module
// only adapts them to the bus.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use zbus::blocking::connection::Builder;
use zbus::blocking::Connection;
use zbus::object_server::SignalEmitter;

use crate::control::{DictionaryState, PauseState};

/// Well-known bus name. Both interfaces are served on it.
pub(crate) const SERVICE_NAME: &str = "org.sttd.Daemon";

/// Pause control.
pub(crate) const DAEMON_PATH: &str = "/org/sttd/Daemon";
pub(crate) const DAEMON_INTERFACE: &str = "org.sttd.Daemon";

/// Live dictionary control.
pub(crate) const DICT_PATH: &str = "/org/sttd/Dictionary";
pub(crate) const DICT_INTERFACE: &str = "org.sttd.Dictionary";

/// Translation control.
pub(crate) const TRANSLATION_PATH: &str = "/org/sttd/Translation";
pub(crate) const TRANSLATION_INTERFACE: &str = "org.sttd.Translation";

/// The connection, kept alive for the process life. `None` until the service
/// thread claims the name (or forever with `--no-dbus`).
static CONNECTION: OnceLock<Connection> = OnceLock::new();

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
// Translation interface
// ---------------------------------------------------------------------------

/// The translation step control interface.
///
/// Four properties, four mutators and one signal, and the split between them is
/// the whole design: `Enabled`, `TargetLanguage` and `Model` describe what is
/// being written **now**, `RequestedLanguage` describes what is on its way, and
/// `TranslationChanged` says when one became the other. Every mutator emits the
/// signal, refusals included, so a client never has to poll to find out.
///
/// `Enabled` is true only when a translator is loaded *and* wanted, so it can
/// never report work that is not happening: during the very first load, or after
/// a failed one, it reads false however the run was started.
struct TranslationIface {
    state: crate::translate::TranslationState,
}

#[zbus::interface(name = "org.sttd.Translation")]
impl TranslationIface {
    /// Translate every accepted transcript from now on. With nothing loaded this
    /// reloads the language the run was started with, or whatever was live before
    /// an unload; with no language ever named it refuses, because there is nothing
    /// to write.
    async fn enable(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let (enabled, message) = self.state.enable();
        announce_translation_changed(&emitter, &self.state, message.is_empty(), &message).await?;
        Ok(enabled)
    }

    /// Stop translating. `unload` also frees the weights, before this returns.
    ///
    /// Either way a load in flight is cancelled: a run told to stop translating
    /// must not end up downloading a translator afterwards. Without `unload` the
    /// translator stays loaded, so `Enable` right after is instant.
    ///
    /// The wait this can take is one running translation (60 to 130 ms for a
    /// pair, ~700 ms for a multilingual export), because the weights are freed
    /// under the lock the translation thread holds. Never a load.
    async fn disable(
        &self,
        unload: bool,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let enabled = self.state.disable(unload);
        announce_translation_changed(&emitter, &self.state, true, "").await?;
        Ok(enabled)
    }

    /// Flip the step. Turning it off keeps the weights, like `Disable(false)`:
    /// freeing memory is its own decision, not part of a toggle.
    async fn toggle(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let enabled = self.state.toggle();
        announce_translation_changed(&emitter, &self.state, true, "").await?;
        Ok(enabled)
    }

    /// Write `language` from now on, loading whatever that needs. Also turns the
    /// step on.
    ///
    /// Returns `(accepted, target, message)`:
    ///
    /// * An empty or whitespace-only language is rejected, like `LoadFile`'s.
    /// * A direction the catalog cannot serve is rejected here, before anything
    ///   is fetched, with the message the startup path prints.
    /// * Otherwise `accepted` is true and the load runs on its own thread. The
    ///   translator that was live keeps serving until the new one is ready, so a
    ///   switch has no window where text comes out untranslated — except for the
    ///   very first one, when nothing was loaded to keep. `RequestedLanguage` is
    ///   non-empty meanwhile, and `TranslationChanged` reports the outcome,
    ///   `ok=false` if the load failed.
    #[zbus(out_args("ok", "target", "message"))]
    async fn set_target_language(
        &self,
        language: String,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<(bool, String, String)> {
        let (accepted, target, message) = self.state.set_target_language(&language);
        announce_translation_changed(&emitter, &self.state, accepted, &message).await?;
        Ok((accepted, target, message))
    }

    /// `(enabled, target, requested, model)` in one round trip.
    #[zbus(out_args("enabled", "target", "requested", "model"))]
    fn get_state(&self) -> (bool, String, String, String) {
        self.state.state()
    }

    /// Whether the step is translating right now: wanted **and** loaded.
    #[zbus(property)]
    fn enabled(&self) -> bool {
        self.state.is_enabled()
    }

    /// The language the loaded translator writes. Empty when none is loaded, and
    /// unchanged by a disable that kept the weights, so it also answers "what
    /// would `Enable` write".
    #[zbus(property)]
    fn target_language(&self) -> String {
        self.state.target_language()
    }

    /// The language of the load in flight, empty when nothing is loading. Its
    /// being non-empty is the "please wait" signal, so there is no separate
    /// `Loading` property that could fall out of step with it.
    #[zbus(property)]
    fn requested_language(&self) -> String {
        self.state.requested_language()
    }

    /// The spec of the loaded translator (`opus-mt-pl-en`), empty when none is
    /// loaded. Worth reading under a multilingual override, where the spec names
    /// no direction at all.
    #[zbus(property)]
    fn model(&self) -> String {
        self.state.model()
    }

    /// Translation state changed: after every mutator, including a refused
    /// language and a failed load. `ok` says whether the call that caused it
    /// worked, and `message` carries the refusal or the failure.
    #[zbus(signal)]
    async fn translation_changed(
        signal_emitter: &SignalEmitter<'_>,
        enabled: bool,
        target: String,
        model: String,
        ok: bool,
        message: &str,
    ) -> zbus::Result<()>;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Emit `TranslationChanged` from an interface method, reporting the state as it
/// is now. The signal is the consistency point: a client that waits for it sees
/// `Enabled`, `TargetLanguage` and `Model` agreeing, where a property read during
/// a switch may still describe the previous translator.
async fn announce_translation_changed(
    emitter: &SignalEmitter<'_>,
    state: &crate::translate::TranslationState,
    ok: bool,
    message: &str,
) -> zbus::fdo::Result<()> {
    let (enabled, target, _requested, model) = state.state();
    emitter
        .translation_changed(enabled, target, model, ok, message)
        .await
        .map_err(zbus::fdo::Error::ZBus)
}

/// Start the service thread.
///
/// Returns immediately. A missing session bus only logs a warning; the thread
/// retries until it claims the name.
pub(crate) fn spawn(
    paused: PauseState,
    dictionary: DictionaryState,
    translation: crate::translate::TranslationState,
) {
    thread::spawn(move || {
        const BACKOFF: [Duration; 4] = [
            Duration::from_secs(2),
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(30),
        ];
        let mut attempt = 0usize;
        loop {
            match build(paused.clone(), dictionary.clone(), translation.clone()) {
                Ok(conn) => {
                    // The name stays claimed for the process life: a drop
                    // would unregister it.
                    if CONNECTION.set(conn).is_err() {
                        log::warn!("dbus: service already running, nothing to do");
                    } else {
                        log::info!(
                            "dbus: serving {DAEMON_INTERFACE} on {DAEMON_PATH}, \
                             {DICT_INTERFACE} on {DICT_PATH} and \
                             {TRANSLATION_INTERFACE} on {TRANSLATION_PATH}"
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

/// Connect to the session bus, register all three interfaces and claim the name.
fn build(
    paused: PauseState,
    dictionary: DictionaryState,
    translation: crate::translate::TranslationState,
) -> zbus::Result<Connection> {
    Builder::session()?
        .name(SERVICE_NAME)?
        .serve_at(DAEMON_PATH, Daemon { state: paused })?
        .serve_at(DICT_PATH, DictionaryIface { state: dictionary })?
        .serve_at(TRANSLATION_PATH, TranslationIface { state: translation })?
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

/// Emit `TranslationChanged` from outside an interface method: the loader thread,
/// which has no emitter and no business holding one. Silent no-op without a
/// connection, like `emit_state_changed`.
///
/// The loader is the one writer that cannot be reached from a bus call, which is
/// exactly why the state announces nothing itself on the paths a bus method owns.
pub(crate) fn emit_translation_changed(
    enabled: bool,
    target: &str,
    model: &str,
    ok: bool,
    message: &str,
) {
    let Some(conn) = CONNECTION.get() else {
        return;
    };
    let emitter = match SignalEmitter::new(conn.inner(), TRANSLATION_PATH) {
        Ok(emitter) => emitter,
        Err(e) => {
            log::debug!("dbus: no signal emitter ({e})");
            return;
        }
    };
    if let Err(e) = zbus::block_on(emitter.translation_changed(
        enabled,
        target.to_string(),
        model.to_string(),
        ok,
        message,
    )) {
        log::warn!("dbus: TranslationChanged({enabled}, {target}) failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dictionary_with(words: &[&str]) -> DictionaryState {
        DictionaryState::new(
            words.iter().map(|w| w.to_string()).collect(),
            PathBuf::new(),
        )
    }

    #[test]
    fn emit_without_a_connection_is_a_noop() {
        // No connection was ever set in this test binary run: both emitters
        // must return quietly instead of panicking.
        emit_state_changed(true);
        emit_translation_changed(false, "", "", false, "test");
    }

    #[test]
    fn identity_strings_are_the_documented_ones() {
        assert_eq!(SERVICE_NAME, "org.sttd.Daemon");
        assert_eq!(DAEMON_PATH, "/org/sttd/Daemon");
        assert_eq!(DAEMON_INTERFACE, "org.sttd.Daemon");
        assert_eq!(DICT_PATH, "/org/sttd/Dictionary");
        assert_eq!(DICT_INTERFACE, "org.sttd.Dictionary");
        assert_eq!(TRANSLATION_PATH, "/org/sttd/Translation");
        assert_eq!(TRANSLATION_INTERFACE, "org.sttd.Translation");
    }

    // -- real-bus round trip ------------------------------------------------
    //
    // The macro owns the wire names, so unit tests cannot see them. These checks
    // drive the served objects over a real session bus. They are `#[ignore]`d
    // because they need a bus, and **one at a time**, because a process holds one
    // bus name and one `CONNECTION`:
    //
    //   dbus-run-session -- cargo test -- --ignored real_bus_round_trip
    //   dbus-run-session -- cargo test -- --ignored real_weights_switch

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

    /// Read a property of `iface` and deserialize its value out of the returned
    /// variant.
    fn get_property<T>(conn: &Connection, path: &str, iface: &str, name: &str) -> T
    where
        T: for<'a> TryFrom<&'a zbus::zvariant::Value<'a>, Error = zbus::zvariant::Error>,
    {
        let reply = conn
            .call_method(
                Some(SERVICE_NAME),
                path,
                Some("org.freedesktop.DBus.Properties"),
                "Get",
                &(iface, name),
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

    /// The whole served surface, driven over a real bus: all three interfaces at
    /// their documented paths with the documented members, the old dictionary
    /// members gone from the pause interface, and the mutators reaching the
    /// shared state.
    ///
    /// The translation part covers what needs no weights: the wire names, the two
    /// refusals (`SetTargetLanguage` with an empty and an impossible language), and
    /// the state transitions around them. What needs weights — a real load landing
    /// — is not here, because it would download on a test machine; that half is the
    /// ignored `real_weights_switch` test below, run separately on purpose.
    #[test]
    #[ignore = "needs a session bus"]
    fn real_bus_round_trip() {
        spawn(
            PauseState::new(false),
            dictionary_with(&["AppImage"]),
            crate::translate::TranslationState::new(
                crate::translate::TranslationPlan::for_tests("pl"),
                crate::translate::RealTranslator,
            ),
        );
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

        // -- translation: the wire names, and the refusals that need no weights
        let translation_xml = introspect(&conn, TRANSLATION_PATH);
        assert!(
            translation_xml.contains(TRANSLATION_INTERFACE),
            "{translation_xml}"
        );
        for member in [
            "Enable",
            "Disable",
            "Toggle",
            "SetTargetLanguage",
            "GetState",
            "Enabled",
            "TargetLanguage",
            "RequestedLanguage",
            "Model",
            "TranslationChanged",
        ] {
            assert!(
                translation_xml.contains(member),
                "translation interface missing {member}"
            );
        }

        // An empty language is refused, exactly like `LoadFile`'s empty path: it
        // never reloads anything.
        let reply = call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "SetTargetLanguage",
            &("   ",),
        )
        .expect("SetTargetLanguage must answer, not fail");
        let (ok, target, message) = reply
            .body()
            .deserialize::<(bool, String, String)>()
            .expect("SetTargetLanguage returns (b, s, s)");
        assert!(!ok);
        assert_eq!(target, "");
        assert_eq!(message, "empty target language");

        // A direction the catalog cannot serve is refused before anything is
        // fetched, with the message the startup path prints.
        let reply = call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "SetTargetLanguage",
            &("de",),
        )
        .expect("SetTargetLanguage must answer, not fail");
        let (ok, _target, message) = reply
            .body()
            .deserialize::<(bool, String, String)>()
            .expect("SetTargetLanguage returns (b, s, s)");
        assert!(!ok, "Polish has no exported pair into German");
        assert!(
            message.contains("no translator for pl -> de"),
            "the refusal names the direction: {message}"
        );

        // `Enable` with nothing loaded and nothing remembered refuses, and says
        // why: there is no language to write.
        let reply = call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "Enable",
            &(),
        )
        .expect("Enable must answer, not fail");
        assert!(
            !reply
                .body()
                .deserialize::<bool>()
                .expect("Enable returns b"),
            "no translator and no target, so nothing can be enabled"
        );

        // `GetState` reports the four documented fields, and they agree with the
        // properties: a client stitching them together sees one real state.
        let reply = call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "GetState",
            &(),
        )
        .expect("GetState must succeed");
        let (enabled, target, requested, model) = reply
            .body()
            .deserialize::<(bool, String, String, String)>()
            .expect("GetState returns (b, s, s, s)");
        assert!(!enabled);
        assert_eq!(target, "");
        assert_eq!(requested, "", "no load was started by a refusal");
        assert_eq!(model, "", "nothing was loaded");
        assert_eq!(
            get_property::<bool>(&conn, TRANSLATION_PATH, TRANSLATION_INTERFACE, "Enabled"),
            enabled
        );
        assert_eq!(
            get_property::<String>(
                &conn,
                TRANSLATION_PATH,
                TRANSLATION_INTERFACE,
                "TargetLanguage"
            ),
            target
        );
        assert_eq!(
            get_property::<String>(
                &conn,
                TRANSLATION_PATH,
                TRANSLATION_INTERFACE,
                "RequestedLanguage"
            ),
            requested
        );
        assert_eq!(
            get_property::<String>(&conn, TRANSLATION_PATH, TRANSLATION_INTERFACE, "Model"),
            model
        );

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
        assert_eq!(
            get_property::<u32>(&conn, DICT_PATH, DICT_INTERFACE, "Count"),
            2
        );
        assert!(get_property::<bool>(
            &conn,
            DICT_PATH,
            DICT_INTERFACE,
            "Dirty"
        ));

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
        assert_eq!(
            get_property::<u32>(&conn, DICT_PATH, DICT_INTERFACE, "Count"),
            1
        );

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

        // The translation members are not on the other two interfaces either.
        for (path, iface) in [(DAEMON_PATH, DAEMON_INTERFACE), (DICT_PATH, DICT_INTERFACE)] {
            assert!(
                conn.call_method(
                    Some(SERVICE_NAME),
                    path,
                    Some(iface),
                    "SetTargetLanguage",
                    &("en",),
                )
                .is_err(),
                "SetTargetLanguage belongs to {TRANSLATION_INTERFACE} only"
            );
        }
    }

    /// The load half, over a real bus and with real weights: `SetTargetLanguage`
    /// starts a load on its own thread, the state reports it landing, and
    /// `Disable true` then frees the weights while `Disable false` keeps them.
    ///
    /// Separate from `real_bus_round_trip` because it needs
    /// `models/translators/opus-mt-pl-en` on disk; skipped when the weights are
    /// absent, so the command below is safe to run anywhere:
    ///
    ///   dbus-run-session -- cargo test -- --ignored real_weights_switch
    #[test]
    #[ignore = "needs a session bus and opus-mt-pl-en weights"]
    fn real_weights_switch() {
        if !crate::translator_store::translator_is_present(crate::translator_store::PL_EN_SPEC) {
            eprintln!(
                "skipped: {} is not downloaded",
                crate::translator_store::PL_EN_SPEC
            );
            return;
        }
        let state = crate::translate::TranslationState::new(
            crate::translate::TranslationPlan::for_tests("pl"),
            crate::translate::RealTranslator,
        );
        spawn(PauseState::new(false), dictionary_with(&[]), state.clone());
        let conn = connect_to_service();

        let reply = call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "SetTargetLanguage",
            &("en",),
        )
        .expect("SetTargetLanguage must answer");
        let (accepted, target, message) = reply
            .body()
            .deserialize::<(bool, String, String)>()
            .expect("returns (b, s, s)");
        assert!(accepted, "pl -> en is a published pair: {message}");
        assert_eq!(target, "en");

        // The load runs off the bus thread, so wait for it rather than assume.
        for _ in 0..600 {
            if state.is_enabled() {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(state.is_enabled(), "the translator never finished loading");
        assert_eq!(state.target_language(), "en");
        assert_eq!(state.model(), crate::translator_store::PL_EN_SPEC);
        assert_eq!(state.requested_language(), "");

        // `Disable(false)` keeps the weights, so `Enable` is instant.
        assert!(!call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "Disable",
            &(false,),
        )
        .expect("Disable must answer")
        .body()
        .deserialize::<bool>()
        .expect("Disable returns b"));
        assert_eq!(
            state.model(),
            crate::translator_store::PL_EN_SPEC,
            "a disable without unload keeps the weights"
        );
        assert!(call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "Enable",
            &()
        )
        .expect("Enable must answer")
        .body()
        .deserialize::<bool>()
        .expect("Enable returns b"));

        // `Disable(true)` frees them, and the reported state empties with them.
        assert!(!call(
            &conn,
            TRANSLATION_PATH,
            TRANSLATION_INTERFACE,
            "Disable",
            &(true,),
        )
        .expect("Disable must answer")
        .body()
        .deserialize::<bool>()
        .expect("Disable returns b"));
        assert!(!state.is_loaded(), "the weights must be gone");
        assert_eq!(state.model(), "");
        assert_eq!(state.target_language(), "", "and no language is claimed");
    }
}
