// ---------------------------------------------------------------------------
// Emission: what happens after the transcript gate emits, behind one seam.
// ---------------------------------------------------------------------------
//
// One step — handing a decided transcript to the user — used to live inside
// the ASR worker loop alongside inference, conditioning and the gate call:
// `println!`, metrics `log::info!`, clipboard copy, auto-paste and WAV/TXT
// dumps shared one loop with the decoder. To answer "was this emitted, and
// where did it go" you had to run the model.
//
// This module owns verdict → action mapping plus every emission side effect,
// so the ASR loop hands it one `Emission` per utterance and gets back
// nothing. `SystemEmitter` is the prod adapter (opt-in stdout, `wl-copy`,
// `ydotool`, spoken transcripts, dump dir); `RecordingEmitter` is the in-memory
// adapter tests drive the gate → emission composition through. Inference never
// sees either.
//
// `system` is the production adapter and its worker; the side effects it
// drives sit beside it, one per submodule: `speech` (the speak thread),
// `clipboard` (Wayland copy, paste, last-text file) and `dump` (`--dump-dir`).

mod clipboard;
mod dump;
mod speech;
mod system;
#[cfg(test)]
mod test_support;

use crate::gate::TranscriptVerdict;
use crate::utterance::Evidence;

pub(crate) use clipboard::{
    copy_to_clipboard, is_wayland_session_env, last_text_path, paste_via_ydotool, type_via_ydotool,
    PasteMode,
};
pub(crate) use speech::{SpeechCut, SpeechOut};
pub(crate) use system::{EmissionPolicy, SystemEmitter};
#[cfg(test)]
pub(crate) use test_support::{RecordedEmission, RecordingEmitter};

/// One utterance's outcome, ready to be handed to the user.
pub(crate) struct Emission<'a> {
    /// `utt_NNNNNN` tag, shared by the dump names and the log lines.
    pub(crate) tag: &'a str,
    /// Raw utterance audio for the WAV dump (pre-conditioning, as before).
    pub(crate) pcm: &'a [f32],
    /// Trimmed transcript text.
    pub(crate) text: &'a str,
    /// Transcript gate disposition, decided by the caller.
    pub(crate) verdict: TranscriptVerdict,
    /// Acoustic evidence, for the metrics line's density.
    pub(crate) evidence: &'a Evidence,
}

/// Owned handoff across the async emission queue: the ASR thread builds
/// this (one clone of tag/text/pcm) and sends it without blocking; the emit
/// worker thread owns all side effects in FIFO order.
pub(crate) struct OwnedEmission {
    pub(crate) tag: String,
    pub(crate) pcm: Vec<f32>,
    pub(crate) text: String,
    pub(crate) verdict: TranscriptVerdict,
    pub(crate) evidence: Evidence,
}

impl OwnedEmission {
    /// The `pcm` copy is what keeps `emit()` non-blocking: the borrow comes
    /// from the utterance the ASR thread still owns, so the samples have to
    /// be duplicated for the queue. Seconds of audio per utterance, once,
    /// against a disk write and a process spawn on the other side.
    fn from_borrowed(e: Emission<'_>) -> Self {
        Self {
            tag: e.tag.to_string(),
            pcm: e.pcm.to_vec(),
            text: e.text.to_string(),
            verdict: e.verdict,
            evidence: *e.evidence,
        }
    }
}

/// The emission seam: one verdict in, side effects out, nothing back.
pub(crate) trait Emitter {
    fn emit(&mut self, emission: Emission<'_>);
}
