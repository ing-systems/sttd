//! The production emitter: a bounded queue in front of a worker thread that
//! maps each verdict to stdout, clipboard, paste, speech and dumps.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};

use super::{Emission, Emitter, OwnedEmission};
use crate::config::{DUMP_KEEP_FILES, SAMPLE_RATE};
use crate::gate::{TranscriptDropKind, TranscriptVerdict};
use crate::wav::write_wav_mono16;

use super::clipboard::{
    copy_to_clipboard, is_wayland_session_env, paste_via_ydotool, type_via_ydotool,
    write_last_text, PasteMode,
};
use super::dump::{dump_drop_reason, ensure_owner_only_dir, prune_dump_dir};
use super::speech::{SpeechCut, SpeechOut};
use crate::fsutil::write_owner_only_text;

/// Prod adapter: opt-in stdout, Wayland clipboard, auto-paste, spoken
/// transcripts, dump dir.
///
/// Async by design: `emit()` only clones into an `OwnedEmission` and does a
/// non-blocking send on a bounded channel, so a stuck `wl-copy`/`ydotool`/disk never
/// stalls inference. A single `emit` worker thread owns all side
/// effects in FIFO order; `Drop` closes the channel and joins it, so shutdown
/// (`asr_handle.join()`) flushes the queue before the process exits.
/// Untested glue by design — the logic worth testing (verdict → action) is
/// exercised through `RecordingEmitter`; this only shells out to binaries
/// and files.
pub(crate) struct SystemEmitter {
    /// The emission queue's sending end; every `sharing()` handle holds a
    /// clone. The worker exits once the last one is dropped.
    tx: Option<SyncSender<OwnedEmission>>,
    worker: Option<JoinHandle<()>>,
    /// True on the handle that owns the queue and the worker thread.
    ///
    /// This flag, not `worker: None`, is what tells `Drop` whether closing the
    /// queue is its job: the owner's `Drop` joins the worker, which only exits
    /// once every sender (the sharing handles' too) is gone.
    owns_queue: bool,
    /// Kept for the final prune in `Drop` (the worker thread owns the only
    /// other copy).
    dump_dir: Option<PathBuf>,
    /// The speak thread's "currently talking" flag, kept here so the capture
    /// loop can read it. The `SpeechOut` itself has moved onto the emit worker
    /// thread (it owns the channel end), so this is the only handle left on
    /// this side — and the flag is an `Arc`, so both observe the same bit.
    worker_speaking: Option<Arc<AtomicBool>>,
    /// The same deal for the cut handle: the `SpeechOut` that owns it has moved
    /// onto the emit worker thread, so this is the only handle through which
    /// the capture loop can stop speech on Ctrl-C. `None` when `--speak` is off.
    worker_cut: Option<Arc<SpeechCut>>,
}

/// Bounded capacity of the emission queue. Emissions are seconds
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
    paste_mode: PasteMode,
    stdout_enabled: bool,
    /// Utterances handled since the last `prune_dump_dir` scan.
    since_prune: usize,
    //// The spoken channel, when `--speak` is on.
    ///
    /// It is dropped on the emit worker thread, which is what ends the speak
    /// thread. Its trailing-utterance drain is real only on the paths that
    /// transcribe to the end (`--file`): in a live run the engine's shutdown
    /// check rejects the trailing decode, so what the drain waits for is the
    /// queue that is already there — and Ctrl-C cuts that instead of waiting.
    speech: Option<SpeechOut>,
}

/// One `SystemEmitter::new` argument per emission channel. A struct because
/// the bare form had grown to six positional parameters, which is exactly where
/// a caller starts transposing two of them.
pub(crate) struct EmissionPolicy {
    pub(crate) dump_dir: Option<PathBuf>,
    pub(crate) copy: bool,
    pub(crate) paste: bool,
    pub(crate) paste_mode: PasteMode,
    pub(crate) stdout: bool,
    /// The loaded voice plus its thread, from `SpeechOut::spawn`.
    pub(crate) speech: Option<SpeechOut>,
}

impl SystemEmitter {
    /// Resolve emission policy once: create the dump dir (fatal, like the
    /// model load it precedes on the main thread) and settle clipboard/paste
    /// availability with one warning each instead of one per utterance.
    /// Then spawn the worker thread that owns the side effects.
    pub(crate) fn new(policy: EmissionPolicy) -> Result<Self> {
        let EmissionPolicy {
            dump_dir,
            copy: copy_enabled,
            paste: paste_enabled,
            paste_mode,
            stdout: stdout_enabled,
            speech: policy_speech,
        } = policy;
        if let Some(dir) = &dump_dir {
            ensure_owner_only_dir(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
            log::info!(
                "dumping utterances to {} (owner-only, newest {} kept)",
                dir.display(),
                DUMP_KEEP_FILES
            );
        }
        let mut worker = EmitterWorker {
            dump_dir: dump_dir.clone(),
            clipboard_available: copy_enabled,
            paste_available: paste_enabled,
            paste_mode,
            stdout_enabled,
            since_prune: 0,
            speech: policy_speech,
        };
        // The capture-side flag is read out before the `SpeechOut` moves onto
        // the worker thread; it is an `Arc`, so both sides share one bit.
        let worker_speaking = worker_speech_flag(worker.speech.as_ref());
        // The cut handle is read out for the same reason, so Ctrl-C reaches the
        // deadline even though the `SpeechOut` lives on the worker thread.
        let worker_cut = worker_speech_cut(worker.speech.as_ref());
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
        let (tx, rx) = sync_channel::<OwnedEmission>(EMIT_QUEUE_CAP);
        // Blocks while the queue is empty and ends when every sender is gone,
        // so an idle daemon has no thread polling the queue.
        let handle = std::thread::Builder::new()
            .name("emit".into())
            .spawn(move || {
                for owned in rx {
                    worker.handle(owned);
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn emit thread: {e}"))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(handle),
            owns_queue: true,
            dump_dir,
            worker_speaking,
            worker_cut,
        })
    }

    /// A second handle onto the same emission queue, for a thread that must emit
    /// off the ASR worker.
    ///
    /// The translation step is the reason: it runs on its own thread so a
    /// translation never stalls the next decode, and it emits through this so the
    /// clipboard, the paste, the dump and the speech all stay in one order on one
    /// worker. A second queue would mean a second worker racing the first for the
    /// clipboard.
    ///
    /// Its `Drop` is a no-op for the queue (`owns_queue` is `false`), so the
    /// owning emitter is still the one that closes and joins, and it may be
    /// dropped in either order.
    pub(crate) fn sharing(&self) -> SystemEmitter {
        SystemEmitter {
            tx: self.tx.clone(),
            worker: None,
            owns_queue: false,
            // The worker owns the only other copy of this; a sharing handle has
            // no business pruning it, and `Drop` returns before that anyway.
            dump_dir: None,
            worker_speaking: None,
            // No cut handle: a sharing handle does not own the speech channel,
            // and the capture loop always takes the handle from the owner
            // before it moves onto the worker thread.
            worker_cut: None,
        }
    }
}

/// The speak thread's flag, taken while the `SpeechOut` is still owned here.
/// `None` when `--speak` is off, which the caller turns into a permanently-clear
/// flag so the capture loop needs no separate "is speech on" case.
fn worker_speech_flag(speech: Option<&SpeechOut>) -> Option<Arc<AtomicBool>> {
    speech.map(SpeechOut::speaking_flag)
}

/// The cut handle, taken while the `SpeechOut` is still owned here. `None` when
/// `--speak` is off, so a caller without `--speak` needs no "is speech on" case.
fn worker_speech_cut(speech: Option<&SpeechOut>) -> Option<Arc<SpeechCut>> {
    speech.map(SpeechOut::cut_handle)
}

impl SystemEmitter {
    /// The flag the capture side reads to hold the pipeline while sttd is
    /// speaking. A permanently-clear flag when `--speak` is off, so the main
    /// loop needs no separate "is speech on" case.
    pub(crate) fn speaking_flag(&self) -> Arc<AtomicBool> {
        self.worker_speaking
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)))
    }

    /// The handle that stops speech when a run is interrupted, or `None` without
    /// `--speak`. Taken before the emitter moves into the ASR worker, because the
    /// `SpeechOut` that owns it ends up on the emit worker thread.
    pub(crate) fn speech_cut(&self) -> Option<Arc<SpeechCut>> {
        self.worker_cut.clone()
    }
}

impl Drop for SystemEmitter {
    fn drop(&mut self) {
        if !self.owns_queue {
            // A sharing handle owns nothing: the queue, the worker thread and the
            // dump directory all belong to the emitter that spawned them, and it
            // is that handle's `Drop` that closes and joins. Dropping the last
            // sharing handle before the owner would otherwise leave the queue
            // open with nothing reading it.
            return;
        }
        // Close the channel first so the worker drains and exits, then join.
        // Flushing here is what guarantees `--stop`/Ctrl-C keeps the trailing
        // utterance instead of dropping it at process teardown.
        self.tx.take();
        if let Some(h) = self.worker.take() {
            if h.join().is_err() {
                log::warn!("emit worker panicked");
            }
        }
        // The worker's own scans stop at the last whole interval, so a clean
        // shutdown is where the last few pairs past `DUMP_KEEP_FILES` go.
        if let Some(dir) = &self.dump_dir {
            prune_dump_dir(dir);
        }
        // The speak thread dies with the emit worker that owns it, so a
        // trailing utterance is spoken (or cut short by `SpeechOut::drop`)
        // before this returns.
    }
}

impl Emitter for SystemEmitter {
    fn emit(&mut self, emission: Emission<'_>) {
        let owned = OwnedEmission::from_borrowed(emission);
        let Some(tx) = self.tx.as_ref() else { return };
        // Bounded: never blocks inference. Only fails when full or when
        // the worker died, neither of which must take transcription down.
        match tx.try_send(owned) {
            Ok(()) => {}
            Err(TrySendError::Full(back)) => {
                log::warn!("emit queue full, dropping {}", back.tag);
            }
            Err(TrySendError::Disconnected(back)) => {
                log::warn!("emit worker gone, dropping {}", back.tag);
            }
        }
    }
}

/// Utterances between two `prune_dump_dir` scans. Each scan is a `read_dir`
/// plus a sort, which the emit thread has no reason to pay per utterance:
/// between scans the directory can overshoot `DUMP_KEEP_FILES` by at most
/// this many pairs, and `SystemEmitter::drop` runs a final scan, so a clean
/// shutdown always leaves exactly the newest `DUMP_KEEP_FILES`.
const DUMP_PRUNE_EVERY: usize = 20;

/// Warn used for both ydotool arms (`key` and `type`): one line, then the
/// auto-paste tool is disabled for the rest of the session.
const YDOTOOL_MISSING_WARN: &str =
    "--paste is enabled but `ydotool` was not found on PATH; disabling auto-paste";

impl EmitterWorker {
    fn handle(&mut self, owned: OwnedEmission) {
        self.handle_inner(&owned);
        // Bound the directory: newest pairs win, oldest are deleted. Runs
        // for every utterance (emits and drops alike), but only scans every
        // `DUMP_PRUNE_EVERY` of them.
        if self.dump_dir.is_none() {
            return;
        }
        self.since_prune += 1;
        if self.since_prune < DUMP_PRUNE_EVERY {
            return;
        }
        self.since_prune = 0;
        if let Some(dir) = &self.dump_dir {
            prune_dump_dir(dir);
        }
    }

    /// Emission helper: disable one emission tool on `NotFound`.
    /// Warn once with the caller text, flip the flag off, report handled.
    fn disable_on_not_found(available: &mut bool, e: &std::io::Error, warn_msg: &str) -> bool {
        if e.kind() == std::io::ErrorKind::NotFound {
            log::warn!("{warn_msg}");
            *available = false;
            true
        } else {
            false
        }
    }

    /// Clipboard + paste step of an emitted transcript. Pasted keys must only
    /// run on content this emission actually copied, except in `type` mode,
    /// which needs no clipboard at all.
    fn copy_paste_and_remember(&mut self, tag: &str, text: &str) {
        // Track whether this utterance actually refreshed the clipboard: paste
        // synthesizes keys, so it must only run on fresh content, never on
        // stale clipboard or a failed copy.
        let mut copied = false;
        if self.clipboard_available {
            match copy_to_clipboard(text) {
                Ok(()) => copied = true,
                Err(e)
                    if Self::disable_on_not_found(
                        &mut self.clipboard_available,
                        &e,
                        "--copy is enabled but `wl-copy` was not found on PATH; disabling clipboard copy",
                    ) => {}
                Err(e) => log::warn!("{tag}: clipboard copy failed: {e:?}"),
            }
        }
        // Remember the last emitted text for `--paste-last`, independent of
        // `--copy`: one-shot re-paste must work even when copy was off.
        write_last_text(text);
        let mode = self.paste_mode;
        if !self.paste_available {
            return;
        }
        // `paste_via_ydotool` for CtrlV/CtrlShiftV (the clipboard holds the
        // fresh text), `type_via_ydotool` for `type` mode (no clipboard).
        let result = if copied {
            paste_via_ydotool(mode)
        } else if mode == PasteMode::Type {
            type_via_ydotool(text)
        } else {
            return;
        };
        match result {
            Ok(()) => {}
            Err(e)
                if Self::disable_on_not_found(
                    &mut self.paste_available,
                    &e,
                    YDOTOOL_MISSING_WARN,
                ) => {}
            Err(e) => log::warn!("{tag}: auto-paste failed (is ydotoold running?): {e:?}"),
        }
    }

    /// Verdict -> action: the WAV dump for every utterance, then either the
    /// drop bookkeeping or the emission itself.
    fn handle_inner(&mut self, owned: &OwnedEmission) {
        let OwnedEmission {
            tag,
            pcm,
            text,
            verdict,
            evidence,
        } = &owned;
        let tag: &str = tag.as_str();
        let pcm: &[f32] = pcm.as_slice();
        let text: &str = text.as_str();
        // WAV dump first, for every utterance (including drops), as before.
        if let Some(dir) = &self.dump_dir {
            let wav_path = dir.join(format!("{tag}.wav"));
            if let Err(e) = write_wav_mono16(&wav_path, pcm, SAMPLE_RATE) {
                log::warn!("failed to dump {}: {e:?}", wav_path.display());
            }
        }
        match verdict {
            TranscriptVerdict::DropEmpty => {
                dump_drop_reason(&self.dump_dir, tag, "DROPPED(empty)");
                return;
            }
            TranscriptVerdict::Emit => {
                // Never clobber the clipboard with empty text (e.g. cleaning
                // removed everything, or an image is currently copied): keep
                // any image/content the user has there instead.
                if text.trim().is_empty() {
                    dump_drop_reason(&self.dump_dir, tag, "DROPPED(empty-cleaned)");
                    return;
                }
                log::info!("{tag}: emitted (density={:.3})", evidence.density());
                if self.stdout_enabled {
                    println!("{text}");
                }
            }
            TranscriptVerdict::Drop { kind } => {
                log::info!("{tag}: dropped (density={:.3})", evidence.density());
                match kind {
                    TranscriptDropKind::Hallucination => {
                        log::info!("{tag}: dropped likely hallucination: {text:?}");
                    }
                    TranscriptDropKind::Repetition => {
                        log::info!("{tag}: dropped repetition-loop decode: {text:?}");
                    }
                }
                let reason = TranscriptVerdict::Drop { kind: *kind }.dump_reason(text);
                dump_drop_reason(&self.dump_dir, tag, &reason);
                return;
            }
        }
        self.copy_paste_and_remember(tag, text);
        if let Some(dir) = &self.dump_dir {
            let txt_path = dir.join(format!("{tag}.txt"));
            if let Err(e) = write_owner_only_text(&txt_path, text) {
                log::warn!("failed to write {}: {e:?}", txt_path.display());
            }
        }
        // Last, because it is the only channel that takes seconds: the
        // clipboard, the paste and the dump of this utterance and of everything
        // behind it must not wait on a voice.
        if let Some(speech) = &self.speech {
            speech.enqueue(tag, text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::test_support::*;

    #[test]
    fn confident_transcript_is_emitted_with_copy_and_paste() {
        let evidence = strong_evidence();
        let verdict = gate().decide_transcript("Dzień dobry");
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
        let verdict = gate().decide_transcript("Dzień dobry");
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

        let verdict = gate().decide_transcript("Dzień dobry");
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
        let verdict = gate().decide_transcript("tak tak tak tak tak tak");
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
        let verdict = gate().decide_transcript("Dzień dobry");
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
    fn hallucination_and_repetition_drops_record_kind() {
        let evidence = strong_evidence();
        for (text, kind) in [
            (
                "Napisy stworzone przez społeczność Amara.org",
                TranscriptDropKind::Hallucination,
            ),
            ("tak tak tak tak tak tak", TranscriptDropKind::Repetition),
        ] {
            let verdict = gate().decide_transcript(text);
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
        let verdict = gate().decide_transcript("   ");
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
    fn system_emitter_drop_flushes_queue() {
        // Async regression: `emit()` must not lose utterances at shutdown.
        // `Drop` closes the channel and joins the worker, so WAV+TXT for both
        // an emit and a drop must exist immediately after drop returns.
        let dir = std::env::temp_dir().join(format!("sttd-emit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let evidence = strong_evidence();
        {
            let mut emitter = SystemEmitter::new(EmissionPolicy {
                dump_dir: Some(dir.clone()),
                ..policy_off()
            })
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            emitter.emit(Emission {
                tag: "utt_000001",
                pcm: &pcm,
                text: "Dzień dobry",
                verdict: gate().decide_transcript("Dzień dobry"),
                evidence: &evidence,
            });
            emitter.emit(Emission {
                tag: "utt_000002",
                pcm: &pcm,
                text: "   ",
                verdict: TranscriptVerdict::DropEmpty,
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
    fn emission_still_copies() {
        // The emission path must treat Emit exactly like any other Emit verdict.
        let evidence = strong_evidence();
        let verdict = gate().decide_transcript("Dzień dobry");
        assert!(matches!(verdict, TranscriptVerdict::Emit));
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emitter.emit(Emission {
            tag: "utt_000001",
            pcm: &pcm,
            text: "Dzień dobry",
            verdict,
            evidence: &evidence,
        });
        let r = &emitter.records[0];
        assert!(r.emitted);
        assert!(r.copied);
    }

    #[test]
    fn system_emitter_emit_never_blocks_inference() {
        // `emit()` is a single clone + bounded push: 200 rapid emits must
        // finish well under an inference (seconds), even with disk I/O
        // still draining on the worker.
        let dir = std::env::temp_dir().join(format!("sttd-emit-nb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let evidence = strong_evidence();
        let mut emitter = SystemEmitter::new(EmissionPolicy {
            dump_dir: Some(dir.clone()),
            ..policy_off()
        })
        .unwrap();
        let pcm = vec![0.01f32; 1600];
        let t0 = std::time::Instant::now();
        for i in 0..200 {
            let tag = format!("utt_{i:06}");
            // Borrow tag for the call; OwnedEmission clones inside.
            emitter.emit(Emission {
                tag: &tag,
                pcm: &pcm,
                text: "Dzień dobry",
                verdict: gate().decide_transcript("Dzień dobry"),
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
    fn empty_emit_never_copies() {
        let evidence = strong_evidence();
        let verdict = gate().decide_transcript("Dzień dobry");
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emit(&mut emitter, "utt_000001", &pcm, "   ", verdict, &evidence);
        let r = &emitter.records[0];
        assert!(!r.emitted);
        assert!(!r.copied);
        assert!(!r.pasted);
    }

    #[test]
    fn empty_emit_dumped_as_cleaned_drop() {
        let dir = std::env::temp_dir().join(format!("sttd-emit-empty-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let evidence = strong_evidence();
        {
            let mut emitter = SystemEmitter::new(EmissionPolicy {
                dump_dir: Some(dir.clone()),
                ..policy_off()
            })
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            emitter.emit(Emission {
                tag: "utt_000001",
                pcm: &pcm,
                text: "   ",
                verdict: gate().decide_transcript("Dzień dobry"),
                evidence: &evidence,
            });
        }
        assert_eq!(
            std::fs::read_to_string(dir.join("utt_000001.txt")).unwrap(),
            "DROPPED(empty-cleaned)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
