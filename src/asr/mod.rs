// ---------------------------------------------------------------------------
// ASR worker: the loop, gates, conditioning.
// ---------------------------------------------------------------------------
//
// This module owns the worker thread and its load/unload/pause/shed loop,
// the transcript gate call, audio conditioning, and the split-seam dedup. The
// engines sit behind `speech::SpeechModel` and own only what is genuinely
// engine-specific — the actual inference call, and enough of a description to
// log. The `--metrics` rows the worker writes are formatted by `metrics`.
//
// `worker` runs the thread and its loop; `pipeline` takes one utterance
// through the model; `condition` holds the pure audio and text helpers around
// a decode; `live` the state other threads hand the worker at runtime.

mod condition;
mod live;
mod pipeline;
#[cfg(test)]
mod test_support;
mod worker;

use std::time::Duration;

use crate::gate::GateConfig;
use crate::speech::EngineConfig;

pub(crate) use live::{LiveDictionary, PartialSlot};
pub(crate) use worker::spawn_asr_worker;

/// Everything the ASR worker needs, resolved once from `Cli` on the main thread
/// and owned by the worker from then on.
///
/// One carrier for the worker's configuration: the engines, the gates and the
/// queue policy all read it from here, and the worker holds no second copy of
/// any of it. Emission policy (`dump_dir`, clipboard, paste) is not here — it
/// belongs to the `Emitter`, which is built and owned separately.
#[derive(Debug, Clone)]
pub(crate) struct AsrConfig {
    /// Engine settings, read by `parakeet` / `whisper` at load.
    pub(crate) engine: EngineConfig,

    // ---- gates and cleaning ----
    /// Shared with the capture side, so both paths judge the same thresholds.
    pub(crate) gate: GateConfig,
    /// Strip hesitation/filler words in post-gate cleaning (see `clean`).
    /// Off by default.
    pub(crate) remove_hesitations: bool,
    /// Live custom vocabulary: the post-gate correction list plus the queue
    /// that hands swapped words to the loaded engine (see `LiveDictionary`).
    /// Empty = off. Shared with the D-Bus thread, so it changes at runtime.
    pub(crate) dictionary: LiveDictionary,

    // ---- queue policy ----
    /// Shed oldest-first past `ASR_SHED_KEEP` (microphone mode). File mode
    /// sets false: offline decoding must be lossless.
    pub(crate) shed_stale: bool,
    /// Where the capture loop leaves the open utterance for partial decoding.
    /// Shared with the capture side.
    pub(crate) partials: PartialSlot,

    // ---- observability ----
    // The `--metrics` writer itself is not here: `spawn_asr_worker` takes an
    // already-opened `MetricsSlot`, because the translation thread writes rows
    // too and one file cannot be opened twice.

    // ---- runtime policy ----
    /// `--auto-unload`: free VRAM while paused, reload eagerly on resume.
    pub(crate) auto_unload: bool,
    /// `--auto-unload-after` as a duration.
    pub(crate) auto_unload_after: Duration,
    /// Audible cues on pause, resume and when transcription is ready.
    pub(crate) sound: bool,
    /// `--autopause`: re-pause after each emitted transcript, single-shot.
    pub(crate) autopause: bool,
}

/// Backlog past which the loaded loop sheds the weakest utterances (see
/// `shed_survivors`): live speech stays fresh instead of transcribing
/// minutes-late noise.
pub(crate) const ASR_SHED_KEEP: usize = 4;

/// Bounded `rtrb` capacity for the utterance queue. utterances are seconds of
/// audio; 16 slots is ample (shedding in the worker keeps the backlog at
/// `ASR_SHED_KEEP`), and a full ring drops the newest with a warn rather
/// than blocking the processing loop.
pub(crate) const ASR_QUEUE_CAP: usize = 16;
