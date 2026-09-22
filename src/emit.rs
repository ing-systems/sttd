// ---------------------------------------------------------------------------
// Emission: what happens after the transcript gate emits, behind one seam.
// ---------------------------------------------------------------------------
//
// One step — handing a decided transcript to the user — used to live inside
// `whisper_init_and_run` alongside inference, conditioning and the gate call:
// `println!`, metrics `log::info!`, clipboard copy, auto-paste and WAV/TXT
// dumps shared one loop with the decoder. To answer "was this emitted, and
// where did it go" you had to run the model.
//
// This module owns verdict → action mapping plus every emission side effect,
// so the whisper loop hands it one `Emission` per utterance and gets back
// nothing. `SystemEmitter` is the prod adapter (opt-in stdout, `wl-copy`,
// `ydotool`, dump dir); `RecordingEmitter` is the in-memory adapter tests drive the
// gate → emission composition through. Inference never sees either.

use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::config::WHISPER_RATE;
use crate::gate::{TranscriptDropKind, TranscriptVerdict};
use crate::utterance::Evidence;
use crate::wav::write_wav_mono16;

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
    /// Text-token average probability and log-probability, for metrics lines.
    pub(crate) prob: f32,
    pub(crate) logprob: f32,
    /// Acoustic evidence, for the metrics line's density.
    pub(crate) evidence: &'a Evidence,
}

/// Owned handoff across the async emission queue: the whisper thread builds
/// this (one clone of tag/text/pcm) and sends it without blocking; the emit
/// worker thread owns all side effects in FIFO order.
pub(crate) struct OwnedEmission {
    pub(crate) tag: String,
    pub(crate) pcm: Vec<f32>,
    pub(crate) text: String,
    pub(crate) verdict: TranscriptVerdict,
    pub(crate) prob: f32,
    pub(crate) logprob: f32,
    pub(crate) evidence: Evidence,
}

impl OwnedEmission {
    fn from_borrowed(e: Emission<'_>) -> Self {
        Self {
            tag: e.tag.to_string(),
            pcm: e.pcm.to_vec(),
            text: e.text.to_string(),
            verdict: e.verdict,
            prob: e.prob,
            logprob: e.logprob,
            evidence: *e.evidence,
        }
    }
}

/// The emission seam: one verdict in, side effects out, nothing back.
pub(crate) trait Emitter {
    fn emit(&mut self, emission: Emission<'_>);
}

/// Prod adapter: opt-in stdout, Wayland clipboard, auto-paste, dump dir.
///
/// Async by design: `emit()` only clones into an `OwnedEmission` and does a
/// non-blocking `rtrb` push, so a stuck `wl-copy`/`ydotool`/disk never
/// stalls whisper inference. A single `emit` worker thread owns all side
/// effects in FIFO order; `Drop` closes the channel and joins it, so shutdown
/// (`whisper_handle.join()`) flushes the queue before the process exits.
/// Untested glue by design — the logic worth testing (verdict → action) is
/// exercised through `RecordingEmitter`; this only shells out to binaries
/// and files.
pub(crate) struct SystemEmitter {
    tx: Option<Producer<OwnedEmission>>,
    worker: Option<JoinHandle<()>>,
}

/// Bounded `rtrb` capacity for the emission queue. Emissions are seconds
/// apart; 512 slots never fills in practice (the 200-rapid-emits test fits),
/// and a full ring drops the newest with a warn rather than blocking
/// inference.
pub(crate) const EMIT_QUEUE_CAP: usize = 512;

/// Mutable worker state: lives on the emit thread only. `clipboard_available`
/// / `paste_available` flip off on `NotFound`, same as before, but without
/// touching the inference thread.
struct EmitterWorker {
    dump_dir: Option<PathBuf>,
    clipboard_available: bool,
    paste_available: bool,
    stdout_enabled: bool,
}

impl SystemEmitter {
    /// Resolve emission policy once: create the dump dir (fatal, like the
    /// model load it precedes on the main thread) and settle clipboard/paste
    /// availability with one warning each instead of one per utterance.
    /// Then spawn the worker thread that owns the side effects.
    pub(crate) fn new(
        dump_dir: Option<PathBuf>,
        copy_enabled: bool,
        paste_enabled: bool,
        stdout_enabled: bool,
    ) -> Result<Self> {
        if let Some(dir) = &dump_dir {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
            log::info!("dumping utterances to {}", dir.display());
        }
        let mut worker = EmitterWorker {
            dump_dir,
            clipboard_available: copy_enabled,
            paste_available: paste_enabled,
            stdout_enabled,
        };
        if !copy_enabled {
            log::info!("clipboard copy disabled (--copy=false)");
        }
        if !paste_enabled {
            log::info!("auto-paste disabled (--paste=false)");
        } else if !copy_enabled {
            log::warn!(
                "--paste is enabled but --copy=false, paste will be skipped (nothing fresh to paste)"
            );
        }
        // Both `wl-copy` and the paste target only make sense on Wayland; on
        // X.org (or headless/`--file` without a Wayland socket) disable once
        // with a warning rather than failing every utterance.
        if (worker.clipboard_available || worker.paste_available) && !is_wayland_session_env() {
            log::warn!(
                "auto-paste is enabled but this is not a Wayland session (no WAYLAND_DISPLAY, XDG_SESSION_TYPE!=wayland); disabling clipboard + ydotool paste"
            );
            worker.clipboard_available = false;
            worker.paste_available = false;
        }
        let (tx, mut rx): (Producer<OwnedEmission>, Consumer<OwnedEmission>) =
            RingBuffer::new(EMIT_QUEUE_CAP);
        let handle = std::thread::Builder::new()
            .name("emit".into())
            .spawn(move || {
                loop {
                    match rx.pop() {
                        Ok(owned) => worker.handle(owned),
                        Err(_) => {
                            if rx.is_abandoned() {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn emit thread: {e}"))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(handle),
        })
    }
}

impl Drop for SystemEmitter {
    fn drop(&mut self) {
        // Close the channel first so the worker drains and exits, then join.
        // Flushing here is what guarantees `--stop`/Ctrl-C keeps the trailing
        // utterance instead of dropping it at process teardown.
        self.tx.take();
        if let Some(h) = self.worker.take() {
            if h.join().is_err() {
                log::warn!("emit worker panicked");
            }
        }
    }
}

impl Emitter for SystemEmitter {
    fn emit(&mut self, emission: Emission<'_>) {
        let owned = OwnedEmission::from_borrowed(emission);
        if let Some(tx) = self.tx.as_mut() {
            // Bounded: never blocks inference. Only fails when full or when
            // the worker died, neither of which must take transcription down.
            if let Err(rtrb::PushError::Full(back)) = tx.push(owned) {
                if tx.is_abandoned() {
                    log::warn!("emit worker gone, dropping {}", back.tag);
                } else {
                    log::warn!("emit queue full, dropping {}", back.tag);
                }
            }
        }
    }
}

impl EmitterWorker {
    fn handle(&mut self, owned: OwnedEmission) {
        let OwnedEmission {
            tag,
            pcm,
            text,
            verdict,
            prob,
            logprob,
            evidence,
        } = &owned;
        let tag: &str = tag.as_str();
        let pcm: &[f32] = pcm.as_slice();
        let text: &str = text.as_str();
        // WAV dump first, for every utterance (including drops), as before.
        if let Some(dir) = &self.dump_dir {
            let wav_path = dir.join(format!("{tag}.wav"));
            if let Err(e) = write_wav_mono16(&wav_path, pcm, WHISPER_RATE) {
                log::warn!("failed to dump {}: {e:?}", wav_path.display());
            }
        }
        match verdict {
            TranscriptVerdict::DropEmpty => {
                dump_drop_reason(&self.dump_dir, tag, "DROPPED(empty)");
                return;
            }
            TranscriptVerdict::Emit { floor, strength } => {
                log::info!(
                    "{tag}: avg_p={prob:.3} avg_logprob={logprob:.3} threshold={floor:.3} \
                     ({strength} evidence, density={:.3})",
                    evidence.density()
                );
                if self.stdout_enabled {
                    println!("{text}");
                }
            }
            TranscriptVerdict::Drop {
                kind,
                floor,
                strength,
            } => {
                log::info!(
                    "{tag}: avg_p={prob:.3} avg_logprob={logprob:.3} threshold={floor:.3} \
                     ({strength} evidence, density={:.3})",
                    evidence.density()
                );
                match kind {
                    TranscriptDropKind::LowConfidence => {
                        log::info!("{tag}: dropped low-confidence transcript: {text:?}");
                    }
                    TranscriptDropKind::Hallucination => {
                        log::info!("{tag}: dropped likely hallucination: {text:?}");
                    }
                    TranscriptDropKind::Repetition => {
                        log::info!("{tag}: dropped repetition-loop decode: {text:?}");
                    }
                }
                let reason = TranscriptVerdict::Drop {
                    kind: *kind,
                    floor: *floor,
                    strength: *strength,
                }
                .dump_reason(text, *prob);
                dump_drop_reason(&self.dump_dir, tag, &reason);
                return;
            }
        }
        // Track whether this utterance actually refreshed the
        // clipboard: paste synthesizes Ctrl+V, so it must only run
        // on fresh content, never on stale clipboard or a failed
        // copy.
        let mut copied = false;
        if self.clipboard_available {
            match copy_to_clipboard(text) {
                Ok(()) => copied = true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    log::warn!(
                        "--copy is enabled but `wl-copy` was not found on PATH; disabling clipboard copy"
                    );
                    self.clipboard_available = false;
                }
                Err(e) => log::warn!("{tag}: clipboard copy failed: {e:?}"),
            }
        }
        if self.paste_available && copied {
            match paste_via_ydotool() {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    log::warn!(
                        "--paste is enabled but `ydotool` was not found on PATH; disabling auto-paste"
                    );
                    self.paste_available = false;
                }
                Err(e) => log::warn!("{tag}: auto-paste failed (is ydotoold running?): {e:?}"),
            }
        }
        if let Some(dir) = &self.dump_dir {
            let txt_path = dir.join(format!("{tag}.txt"));
            if let Err(e) = std::fs::write(&txt_path, text) {
                log::warn!("failed to write {}: {e:?}", txt_path.display());
            }
        }
    }
}

/// Write `reason` as `<dump_dir>/<tag>.txt` for an utterance that was
/// dropped rather than printed, so `STTD_DUMP_DIR` output stays correlatable
/// with the stderr log: previously a dropped utterance left only a WAV with
/// no indication of what became of it or why.
fn dump_drop_reason(dump_dir: &Option<PathBuf>, tag: &str, reason: &str) {
    if let Some(dir) = dump_dir {
        let txt_path = dir.join(format!("{tag}.txt"));
        if let Err(e) = std::fs::write(&txt_path, reason) {
            log::warn!("failed to write {}: {e:?}", txt_path.display());
        }
    }
}

/// Whether the current session is Wayland, where clipboard (`wl-copy`) + paste
/// (`ydotool key` Ctrl+V) is supported. Pure over its inputs so tests don't touch the process
/// environment (see `is_wayland_session_env` for the env-reading wrapper).
/// `WAYLAND_DISPLAY` set wins over `DISPLAY` (XWayland sets both), so an
/// X.org session is `XDG_SESSION_TYPE=x11` with `WAYLAND_DISPLAY` empty.
pub(crate) fn is_wayland_session(
    xdg_session_type: Option<&str>,
    wayland_display: Option<&str>,
) -> bool {
    if wayland_display.is_some_and(|v| !v.trim().is_empty()) {
        return true;
    }
    xdg_session_type.is_some_and(|v| v.trim().eq_ignore_ascii_case("wayland"))
}

/// Env-reading wrapper around [`is_wayland_session`].
pub(crate) fn is_wayland_session_env() -> bool {
    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
    let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();
    is_wayland_session(session_type.as_deref(), wayland_display.as_deref())
}

/// Copy `text` to the Wayland clipboard via `wl-copy` (from wl-clipboard).
/// Pipes the trimmed transcript over `wl-copy`'s stdin with no trailing
/// newline, so it is pasted verbatim (UTF-8, including Polish diacritics) by
/// the subsequent paste step. Passes `--sensitive` (`x-kde-passwordManagerHint=secret`)
/// so Klipper skips clipboard history while paste still works.
/// Returns the `io::Error` unwrapped (not `anyhow`)
/// so the caller can match `NotFound` and disable itself with a single warning
/// on machines without wl-clipboard.
pub(crate) fn copy_to_clipboard(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("wl-copy")
        .arg("--sensitive")
        .stdin(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("wl-copy stdin unavailable"))?;
    stdin.write_all(text.as_bytes())?;
    drop(stdin);
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "wl-copy exited with {status}"
        )))
    }
}

/// Paste the clipboard at the current cursor position via `ydotool`
/// (requires the `ydotoold` daemon). Synthesizes a single Ctrl+V
/// (`29:1 47:1 47:0 29:0`, i.e. LEFTCTRL down, V down/up, LEFTCTRL up) instead
/// of typing the transcript key-by-key: a fixed 4-event sequence, so it is
/// fast and preserves non-ASCII text already on the clipboard. Returns
/// `io::Error` unwrapped so the caller can match `NotFound` and disable
/// itself with a single warning on machines without `ydotool`.
pub(crate) fn paste_via_ydotool() -> std::io::Result<()> {
    use std::process::Command;
    let status = Command::new("ydotool")
        .args(["key", "29:1", "47:1", "47:0", "29:0"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ydotool exited with {status}"
        )))
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
}

/// What one `emit` call would have done.
#[cfg(test)]
pub(crate) struct RecordedEmission {
    pub tag: String,
    pub text: String,
    pub emitted: bool,
    pub drop_kind: Option<TranscriptDropKind>,
    pub copied: bool,
    pub pasted: bool,
    pub printed: bool,
    pub wav_samples: usize,
}

#[cfg(test)]
impl RecordingEmitter {
    pub(crate) fn new() -> Self {
        Self {
            records: Vec::new(),
            fail_copy: false,
            stdout_enabled: false,
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
            emitted: false,
            drop_kind: None,
            copied: false,
            pasted: false,
            printed: false,
            wav_samples,
        };
        if let TranscriptVerdict::Emit { .. } = emission.verdict {
            record.emitted = true;
            record.copied = !self.fail_copy;
            // Same AND semantics as the prod adapter: paste only runs on
            // fresh clipboard content, never after a failed copy.
            record.pasted = record.copied;
            record.printed = self.stdout_enabled;
        } else if let TranscriptVerdict::Drop { kind, .. } = emission.verdict {
            record.drop_kind = Some(kind);
        }
        self.records.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
        DEFAULT_MIN_SPEECH_RMS, MIN_SPEECH_FRAMES,
    };
    use crate::gate::GateConfig;

    fn gate() -> GateConfig {
        GateConfig::new(
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_DENSITY,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        )
    }

    fn strong_evidence() -> Evidence {
        Evidence {
            speech_frames: 137,
            total_frames: 200,
            rms: 0.01178,
        }
    }

    fn weak_evidence() -> Evidence {
        Evidence {
            speech_frames: MIN_SPEECH_FRAMES,
            total_frames: 131,
            rms: DEFAULT_MIN_SPEECH_RMS + 0.0005,
        }
    }

    fn emit<'a>(
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
            prob: 0.9,
            logprob: -0.1,
            evidence,
        });
    }

    #[test]
    fn confident_transcript_is_emitted_with_copy_and_paste() {
        let evidence = strong_evidence();
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", 0.9);
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emit(
            &mut emitter,
            "utt_000001",
            &pcm,
            "Dzień dobry",
            verdict,
            &evidence,
        );
        assert_eq!(emitter.records.len(), 1);
        let r = &emitter.records[0];
        assert_eq!(r.tag, "utt_000001");
        assert!(r.emitted);
        assert_eq!(r.text, "Dzień dobry");
        assert!(r.copied);
        assert!(r.pasted);
        assert!(!r.printed);
        assert_eq!(r.wav_samples, 1600);
    }

    #[test]
    fn stdout_print_is_opt_in() {
        let evidence = strong_evidence();
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", 0.9);
        let pcm = vec![0.01f32; 1600];
        let mut emitter = RecordingEmitter::new();
        emit(
            &mut emitter,
            "utt_000001",
            &pcm,
            "Dzień dobry",
            verdict,
            &evidence,
        );
        assert!(emitter.records[0].emitted);
        assert!(!emitter.records[0].printed);

        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", 0.9);
        let mut emitter = RecordingEmitter::new();
        emitter.stdout_enabled = true;
        emit(
            &mut emitter,
            "utt_000002",
            &pcm,
            "Dzień dobry",
            verdict,
            &evidence,
        );
        assert!(emitter.records[0].emitted);
        assert!(emitter.records[0].printed);

        // Drops never print, even when stdout is enabled.
        let verdict = gate().decide_transcript(
            evidence.speech_frames,
            evidence.rms,
            "tak tak tak tak tak tak",
            0.9,
        );
        emit(
            &mut emitter,
            "utt_000003",
            &pcm,
            "tak tak tak tak tak tak",
            verdict,
            &evidence,
        );
        let r = &emitter.records[1];
        assert!(!r.emitted);
        assert!(!r.printed);
    }

    #[test]
    fn paste_only_runs_on_fresh_copy() {
        let evidence = strong_evidence();
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", 0.9);
        let mut emitter = RecordingEmitter::new();
        emitter.fail_copy = true;
        let pcm = vec![0.01f32; 1600];
        emit(
            &mut emitter,
            "utt_000001",
            &pcm,
            "Dzień dobry",
            verdict,
            &evidence,
        );
        let r = &emitter.records[0];
        assert!(r.emitted);
        assert!(!r.copied);
        assert!(!r.pasted);
        assert!(!r.printed);
    }

    #[test]
    fn low_confidence_drop_emits_nothing() {
        let evidence = weak_evidence();
        let verdict = gate().decide_transcript(
            evidence.speech_frames,
            evidence.rms,
            "To jest prawdziwe zdanie testowe",
            0.1,
        );
        assert!(matches!(
            verdict,
            TranscriptVerdict::Drop {
                kind: TranscriptDropKind::LowConfidence,
                ..
            }
        ));
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emit(
            &mut emitter,
            "utt_000002",
            &pcm,
            "To jest prawdziwe zdanie testowe",
            verdict,
            &evidence,
        );
        let r = &emitter.records[0];
        assert!(!r.emitted);
        assert_eq!(r.drop_kind, Some(TranscriptDropKind::LowConfidence));
        assert!(!r.copied);
        assert!(!r.pasted);
        assert!(!r.printed);
        assert_eq!(r.wav_samples, 1600);
    }

    #[test]
    fn hallucination_and_repetition_drops_record_kind() {
        let evidence = strong_evidence();
        for (text, kind) in [
            (
                "Napisy stworzone przez społeczność Amara.org",
                TranscriptDropKind::Hallucination,
            ),
            ("tak tak tak tak tak tak", TranscriptDropKind::Repetition),
        ] {
            let verdict = gate().decide_transcript(evidence.speech_frames, evidence.rms, text, 0.9);
            assert!(matches!(verdict, TranscriptVerdict::Drop { kind: k, .. } if k == kind));
            let mut emitter = RecordingEmitter::new();
            let pcm = vec![0.01f32; 1600];
            emit(&mut emitter, "utt_000003", &pcm, text, verdict, &evidence);
            let r = &emitter.records[0];
            assert!(!r.emitted);
            assert_eq!(r.drop_kind, Some(kind));
            assert!(!r.copied);
        }
    }

    #[test]
    fn empty_transcript_drops_but_wav_still_recorded() {
        let evidence = weak_evidence();
        let verdict = gate().decide_transcript(evidence.speech_frames, evidence.rms, "   ", 0.0);
        assert_eq!(verdict, TranscriptVerdict::DropEmpty);
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emit(&mut emitter, "utt_000004", &pcm, "   ", verdict, &evidence);
        let r = &emitter.records[0];
        assert!(!r.emitted);
        assert_eq!(r.drop_kind, None);
        assert_eq!(r.wav_samples, 1600);
    }

    #[test]
    fn strong_evidence_relaxes_floor_through_the_seam() {
        // Same production numbers as the whisper-side regression test, now
        // asserted through the emission seam: avg_p=0.478 over strong
        // evidence must emit (and copy), not drop.
        let evidence = strong_evidence();
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", 0.478);
        assert!(matches!(verdict, TranscriptVerdict::Emit { .. }));
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emit(
            &mut emitter,
            "utt_000005",
            &pcm,
            "Dzień dobry",
            verdict,
            &evidence,
        );
        let r = &emitter.records[0];
        assert!(r.emitted);
        assert!(r.copied);
    }

    #[test]
    fn system_emitter_drop_flushes_queue() {
        // Async regression: `emit()` must not lose utterances at shutdown.
        // `Drop` closes the channel and joins the worker, so WAV+TXT for both
        // an emit and a drop must exist immediately after drop returns.
        let dir = std::env::temp_dir().join(format!("sttd-emit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let evidence = strong_evidence();
        {
            let mut emitter = SystemEmitter::new(Some(dir.clone()), false, false, false).unwrap();
            let pcm = vec![0.01f32; 1600];
            emitter.emit(Emission {
                tag: "utt_000001",
                pcm: &pcm,
                text: "Dzień dobry",
                verdict: gate().decide_transcript(
                    evidence.speech_frames,
                    evidence.rms,
                    "Dzień dobry",
                    0.9,
                ),
                prob: 0.9,
                logprob: -0.1,
                evidence: &evidence,
            });
            emitter.emit(Emission {
                tag: "utt_000002",
                pcm: &pcm,
                text: "   ",
                verdict: TranscriptVerdict::DropEmpty,
                prob: 0.0,
                logprob: 0.0,
                evidence: &evidence,
            });
            // No join yet: Drop at scope end must flush.
        }
        assert!(dir.join("utt_000001.wav").exists());
        assert!(dir.join("utt_000001.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("utt_000001.txt")).unwrap(),
            "Dzień dobry"
        );
        assert!(dir.join("utt_000002.wav").exists());
        assert!(dir.join("utt_000002.txt").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn system_emitter_emit_never_blocks_inference() {
        // `emit()` is a single clone + bounded push: 200 rapid emits must
        // finish well under a whisper inference (seconds), even with disk I/O
        // still draining on the worker.
        let dir = std::env::temp_dir().join(format!("sttd-emit-nb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let evidence = strong_evidence();
        let mut emitter = SystemEmitter::new(Some(dir.clone()), false, false, false).unwrap();
        let pcm = vec![0.01f32; 1600];
        let t0 = std::time::Instant::now();
        for i in 0..200 {
            let tag = format!("utt_{i:06}");
            // Borrow tag for the call; OwnedEmission clones inside.
            emitter.emit(Emission {
                tag: &tag,
                pcm: &pcm,
                text: "Dzień dobry",
                verdict: gate().decide_transcript(
                    evidence.speech_frames,
                    evidence.rms,
                    "Dzień dobry",
                    0.9,
                ),
                prob: 0.9,
                logprob: -0.1,
                evidence: &evidence,
            });
        }
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "200 emits took {:?}, worker is blocking inference",
            t0.elapsed()
        );
        drop(emitter);
        assert!(dir.join("utt_000199.wav").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wayland_session_detection_prefers_wayland_display() {
        // Plain Wayland, X11-only, XWayland (both DISPLAY and
        // WAYLAND_DISPLAY set counts as Wayland), and headless.
        assert!(is_wayland_session(Some("wayland"), Some("wayland-0")));
        assert!(is_wayland_session(Some("wayland"), None));
        assert!(is_wayland_session(None, Some("wayland-0")));
        assert!(is_wayland_session(Some("WAYLAND"), None));
        assert!(!is_wayland_session(Some("x11"), None));
        assert!(!is_wayland_session(Some("x11"), Some("")));
        assert!(!is_wayland_session(None, None));
        assert!(!is_wayland_session(Some(""), Some("  ")));
    }
}
