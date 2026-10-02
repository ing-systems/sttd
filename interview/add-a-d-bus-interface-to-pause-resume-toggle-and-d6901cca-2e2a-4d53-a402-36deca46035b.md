# D-Bus interface: pause/resume/toggle + reload dictionary

## 1. Goal
Add a session-bus D-Bus interface to `sttd` for pause control and live dictionary control. Keep existing signal control. No change to audio path.

## 2. Facts (repo today)
- Crate: single Rust 2021 binary `sttd` (`Cargo.toml:1-10`). No tokio. No `zbus` dep.
- Pause flag: `Arc<AtomicBool> paused` (`src/main.rs:459`). Main loop edges at `src/main.rs:575-604`.
- Signals: `SIGUSR1` pause, `SIGUSR2` resume, `SIGHUP` toggle (`src/main.rs:753-800`). File mode disables pause (`src/main.rs:830-834`).
- Dictionary: `--dictionary` / `STTD_DICTIONARY` (`src/cli.rs:74-78`). Load once via `load_dictionary_or_warn` (`src/main.rs:956`) -> `dictionary::load_dictionary` (`src/dictionary.rs:76`). Cloned into `AsrOptions.dictionary` (`src/asr.rs:116,248`). Boost applied only at `ParakeetModel::load` via `set_boost_words` (`src/parakeet.rs:56`, `src/parakeet_core/engine.rs:244`).
- Config: CLI > `STTD_*` env > default (`src/cli.rs:26-35`, `README.md:160-198`).
- Service: systemd user unit (`packaging/sttd.service`).
- Dictionary rules: `sanitize_entries` (`src/dictionary.rs:84`), caps `POSTCORR_MAX_DICT_WORDS=100` (`src/dictionary.rs:40`).

## 3. Decisions (interview)
- Bus: session bus, well-known name.
- Identity: name `org.sttd.Daemon`, path `/org/sttd/Daemon`, interface `org.sttd.Daemon`.
- Pause API: stateful + signal (`Pause`, `Resume`, `Toggle`, readable `Paused`, `StateChanged` signal).
- Signals: keep `SIGUSR1`/`SIGUSR2`/`SIGHUP`. Both write same `AtomicBool`.
- Dictionary API: full manage + `SetWords`. Methods: `ReloadDictionary`, `GetDictionary`/`GetDictionaryPath`, `SetWords`. Signal: `DictionaryChanged`.
- Reload error: async signal. Load to temp, swap only on success, keep old words on failure.
- Boost scope: both paths. Update post-correction list and ONNX boost automaton together.
- No bus: retry loop in background. Log warning. Signals keep working.
- CLI: add `--no-dbus` (`STTD_NO_DBUS`). No custom name flag.
- File mode (`--file`): D-Bus disabled. No service, no retry.
- Acceptance: `busctl` checks + `cargo test`.

## 4. Interface (zbus 5, `async-io`)
```toml
zbus = { version = "5", default-features = false, features = ["async-io", "async-lock", "proxy", "service"] }
```
Do not add `tokio`. Use `zbus::blocking` builder on its own thread. Keep `Connection` in `OnceLock`. Drop = unregister, so hold for process life.

Methods (interface `org.sttd.Daemon`):
- `Pause()` -> no return. Sets `paused=true`. Idempotent.
- `Resume()` -> no return. Sets `paused=false`. Arms one shot if `--autopause`.
- `Toggle() -> bool`. Flips flag, returns new `paused`.
- `ReloadDictionary(path: s = "") -> (ok: b, count: u, message: s)`. Empty path = reload startup path. Non-empty = load file then switch source on success.
- `SetWords(words: as) -> (count: u)`. Sanitizes via `sanitize_entries`, swaps live list, updates booster, emits signal.
- `GetDictionary() -> (words: as, path: s, count: u)`.

Properties:
- `Paused: b` (readable). Reads same `AtomicBool`.

Signals:
- `StateChanged(paused: b)`. Emitted by D-Bus calls and by signal-handler transitions.
- `DictionaryChanged(count: u, ok: b, message: s)`. Emitted after `ReloadDictionary`/`SetWords`, including async completion and failure (old words kept).

Notes:
- Interface name reuses bus basename per user choice. If `zbus` objects, use `org.sttd.Daemon1` as interface and keep bus name `org.sttd.Daemon`.
- All pause edges must call existing actions: pause entry `fp.hold_pipeline()`, resume entry `cue_guard()`.

## 5. State sharing
- Pause: keep `Arc<AtomicBool> paused`. D-Bus handlers write it. Main loop already polls it. Emit `StateChanged` after each change. Signal handlers also emit via shared emitter handle.
- Dictionary: replace `Vec<String>` clones with `Arc<RwLock<Arc<Vec<String>>>>` (or `arc-swap`) shared between main, ASR worker (`src/asr.rs:248`), and D-Bus thread. Add `Arc<RwLock<PathBuf>>` for source path.
- ASR hot path: clone `Arc` per utterance, then `correct_transcription(text, &words, ...)` (`src/dictionary.rs:161`). No lock held across inference.
- Booster: after swap, call `engine.set_boost_words(&words)` (`src/parakeet_core/engine.rs:244`) on worker thread. If model unloaded (`--auto-unload`), store words and apply at next `load_once` (`src/asr.rs:426`).

## 6. Reload flow
1. Resolve path: arg if non-empty, else current source path.
2. `load_dictionary(path)` (`src/dictionary.rs:76`). On `Err`, keep old list, return `(false, old_count, err)` and emit `DictionaryChanged(old_count, false, err)`.
3. On `Ok`, `sanitize_entries` already applied. Swap `Arc`, update source path only if arg non-empty and load ok, update booster, emit `DictionaryChanged(new_count, true, "")`, return `(true, new_count, "")`.
4. `SetWords`: same swap path without file IO. Validate with `sanitize_entries`. Source path unchanged.

## 7. Startup / no-bus / file mode
- Mic mode default: start D-Bus thread unless `--no-dbus` / `STTD_NO_DBUS=1`.
- If session bus missing: log warning, keep signals, retry in background with backoff (e.g. 2 s, 5 s, 10 s, then 30 s). Do not exit non-zero. Claim name when bus appears.
- `--file` mode: skip D-Bus init entirely. No retry thread.
- `--start-paused` (`src/cli.rs:218-229`) still sets initial `paused`. D-Bus `Paused` reflects it.

## 8. CLI
- Add `--no-dbus` (`STTD_NO_DBUS`), `help_heading = "Runtime"`, default false. Conflicts with none. File mode ignores it (already off).
- No `--dbus-name` in v1.

## 9. Security
- Session bus needs no policy. Do not add system bus or polkit. Peer is the user session.
- `SetWords`/`ReloadDictionary` trust the session user. Document that any local process as same user can call them.

## 10. Verification
- Manual:
  ```sh
  busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Pause
  busctl --user get-property org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Paused
  busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Toggle
  busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon ReloadDictionary s ""
  busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon SetWords as 2 AppImage Syntocinon
  busctl --user monitor org.sttd.Daemon
  ```
- Tests: `cargo test`. Add unit tests for swap-keep-old on `Err`, `sanitize_entries` via `SetWords`, `Toggle` return value. Add integration test with private session bus if available, else gated.
- Signals still work: `systemctl --user kill -s SIGHUP sttd` flips `Paused` and emits `StateChanged`.

## 11. Docs / rollout
- Update `README.md` Runtime control + Configuration table (`--no-dbus`), and `packaging/sttd.service` comments with `busctl` examples.
- Keep `CONTEXT.md` vocabulary unchanged unless new terms appear.
- v1 scope excludes: system bus, custom bus name flag, properties-only control, post-corr-only or boost-only reload.
