// ---------------------------------------------------------------------------
// Control state: what the run can be told to change while it continues
// ---------------------------------------------------------------------------
//
// The pause flag and the live dictionary are shared by every lane that can
// change or read them — the session bus, the Unix signal handlers, the main
// loop and the ASR worker — so they live here rather than in any one of them.
// `dbus` is one adapter over this state, not its owner: a `--file` run or a
// `--no-dbus` run uses the same types with no bus at all.

mod dictionary;
mod pause;

pub(crate) use dictionary::DictionaryState;
pub(crate) use pause::PauseState;

use std::sync::RwLock;

/// A poisoned lock must not kill transcription: take the value and continue.
fn lock_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn lock_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}
