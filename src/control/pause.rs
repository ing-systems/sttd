//! The pause flag.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn pause_state() -> PauseState {
        PauseState::new(false)
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
}
