//! The worker's live inputs that other threads write: the dictionary handle
//! and the partial-transcript slot.

use std::sync::{Arc, Mutex};

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
    pub(super) fn take_boost(&self) -> Option<Arc<Vec<String>>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::test_support::*;

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

    #[test]
    fn live_dictionary_is_debug_and_clone() {
        let (_bus, live) = live_dictionary_with(&["a", "b"]);
        let clone = live.clone();
        assert_eq!(*clone.words(), *live.words());
        assert!(format!("{live:?}").contains("words: 2"), "{live:?}");
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
}
