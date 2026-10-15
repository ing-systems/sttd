//! Test adapters: an emitter that records instead of acting, the policy with
//! every side effect off, and the fixtures this module's own tests share.

use super::*;
use crate::config::{DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS};
use crate::gate::{GateConfig, TranscriptDropKind, TranscriptVerdict};
use crate::utterance::Evidence;

/// Emission policy with every channel off — the shape most tests want.
///
/// Spoken emission is absent rather than disabled: it has no in-memory stand-in
/// (it would need a loaded voice and an output device), so `RecordingEmitter`
/// mirrors it with a flag and the real thread is asserted elsewhere.
#[cfg(test)]
pub(crate) fn policy_off() -> EmissionPolicy {
    EmissionPolicy {
        dump_dir: None,
        copy: false,
        paste: false,
        paste_mode: PasteMode::CtrlV,
        stdout: false,
        speech: None,
    }
}

/// In-memory adapter: records what would reach the user instead of printing,
/// copying, pasting, or touching the filesystem. Tests drive the gate →
/// emission composition through this.
#[cfg(test)]
pub(crate) struct RecordingEmitter {
    pub records: Vec<RecordedEmission>,
    /// Simulate a failed clipboard copy: nothing is pasted, but the
    /// transcript still counts as emitted.
    pub fail_copy: bool,
    /// Mirror of `--stdout`: when false (the default, like prod),
    /// an emitted transcript is recorded without printing.
    pub stdout_enabled: bool,
    /// Mirror of `--speak`: when false (the default, like prod), nothing is
    /// handed to the speak thread.
    pub speak_enabled: bool,
}

/// What one `emit` call would have done.
#[cfg(test)]
pub(crate) struct RecordedEmission {
    pub tag: String,
    pub text: String,
    /// The disposition as the worker decided it, verbatim. Kept because
    /// `emitted` + `drop_kind` cannot express it: `DropEmpty` is neither an
    /// emit nor a `Drop { kind }`, so an `Emit` of empty text and a
    /// `DropEmpty` would otherwise be indistinguishable here.
    pub verdict: TranscriptVerdict,
    pub emitted: bool,
    pub drop_kind: Option<TranscriptDropKind>,
    pub copied: bool,
    pub pasted: bool,
    pub printed: bool,
    /// Mirror of `--speak`: whether this utterance was handed to the speak
    /// thread. Recorded rather than performed — speech needs a loaded voice and
    /// an output device.
    pub spoken: bool,
    pub wav_samples: usize,
}

#[cfg(test)]
impl RecordingEmitter {
    pub(crate) fn new() -> Self {
        Self {
            records: Vec::new(),
            fail_copy: false,
            stdout_enabled: false,
            speak_enabled: false,
        }
    }
}

#[cfg(test)]
impl Emitter for RecordingEmitter {
    fn emit(&mut self, emission: Emission<'_>) {
        let wav_samples = emission.pcm.len();
        let mut record = RecordedEmission {
            tag: emission.tag.to_string(),
            text: emission.text.to_string(),
            verdict: emission.verdict,
            emitted: false,
            drop_kind: None,
            copied: false,
            pasted: false,
            printed: false,
            spoken: false,
            wav_samples,
        };
        if let TranscriptVerdict::Emit = emission.verdict {
            // Mirror prod guard: empty Emit never touches clipboard.
            if emission.text.trim().is_empty() {
                self.records.push(record);
                return;
            }
            record.emitted = true;
            record.copied = !self.fail_copy;
            // Same AND semantics as the prod adapter: paste only runs on
            // fresh clipboard content, never after a failed copy.
            record.pasted = record.copied;
            record.printed = self.stdout_enabled;
            // Prod enqueues speech past the empty-text guard too, so an emitted
            // utterance is spoken regardless of whether the clipboard ran.
            record.spoken = self.speak_enabled;
        } else if let TranscriptVerdict::Drop { kind, .. } = emission.verdict {
            record.drop_kind = Some(kind);
        }
        self.records.push(record);
    }
}

pub(super) fn gate() -> GateConfig {
    GateConfig::new(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY)
}

pub(super) fn evidence() -> Evidence {
    Evidence {
        speech_frames: 137,
        total_frames: 200,
        rms: 0.01178,
    }
}

pub(super) fn strong_evidence() -> Evidence {
    evidence()
}

pub(super) fn weak_evidence() -> Evidence {
    evidence()
}

pub(super) fn emit<'a>(
    emitter: &mut RecordingEmitter,
    tag: &'a str,
    pcm: &'a [f32],
    text: &'a str,
    verdict: TranscriptVerdict,
    evidence: &'a Evidence,
) {
    emitter.emit(Emission {
        tag,
        pcm,
        text,
        verdict,
        evidence,
    });
}
