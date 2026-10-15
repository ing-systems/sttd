//! The process-wide monotonic millisecond clock the speak and cue threads
//! share their finish deadlines through.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Milliseconds since an arbitrary fixed instant, for the finish deadline.
///
/// `Instant` cannot cross the thread boundary as a shared value, and
/// `SystemTime` can jump. A monotonic counter derived from a process-start
/// `Instant` is the cheap monotonic clock this needs.
pub(crate) fn now_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Whether `finish_by_ms` holds a deadline that has already passed. Zero means
/// no deadline, so a caller that never set one is never cut off.
///
/// Takes the clock as an argument so the wait itself is testable without a
/// device: `now_ms` is a process-wide monotonic counter, not wall time.
pub(crate) fn past_deadline(finish_by_ms: &AtomicU64, now_ms: u64) -> bool {
    let deadline = finish_by_ms.load(Ordering::Acquire);
    deadline != 0 && now_ms >= deadline
}
