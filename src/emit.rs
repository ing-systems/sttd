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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::config::{DUMP_KEEP_FILES, SAMPLE_RATE, SPEAK_TAIL_MS};
use crate::gate::{TranscriptDropKind, TranscriptVerdict};
use crate::piper::PiperVoice;
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

/// One utterance's text, on its way to the speakers.
struct SpeechJob {
    tag: String,
    text: String,
}

/// The spoken emission: hand it text, a dedicated thread speaks it.
///
/// Off the emit worker on purpose. Speaking is seconds of wall clock (the
/// synthesis plus the audio playing out), and the emit worker still has the
/// clipboard, the paste and the dump of every later utterance to do — speaking
/// in line would make a transcript arrive minutes late.
///
/// The queue holds **one** utterance and the newest replaces what is waiting.
/// That is the same rule the partial transcript follows (only the latest is
/// kept, see `asr::PartialSlot`): with `--autopause` there is never a second
/// utterance while one is being spoken, and without it a backlog of speech is
/// already stale by the time it would be read — reading it aloud would put words
/// in the user's mouth that they have already moved on from.
///
/// `speaking` is the flag the capture side watches: the whole point of speaking
/// through the microphone's own room is that sttd must not transcribe itself,
/// so it is raised for the lead window, the playback and the tail.
pub(crate) struct SpeechOut {
    /// `(slot, wakeup)`: the pending job and the signal to look at it.
    slot: Arc<(Mutex<Option<SpeechJob>>, Condvar)>,
    /// Raised from just before playback starts until the tail has played out.
    speaking: Arc<AtomicBool>,
    /// Deadline for the final utterance, set by `drop`, in milliseconds since
    /// the thread started — a monotonic `u64` rather than an `Instant` so it
    /// can cross the thread boundary. Zero means "no deadline".
    finish_by_ms: Arc<AtomicU64>,
    /// Set when espeak goes away, so the emit side stops queueing into a thread
    /// that will never speak again.
    disabled: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

/// How long `SpeechOut::drop` waits for the utterance already queued to be
/// spoken before cutting it off.
///
/// The trailing utterance has to be spoken, not dropped: `--file --speak` with
/// the last transcript swallowed would be the channel silently losing work, and
/// it is the same reasoning `asr`'s shutdown flush uses for the transcript
/// itself. Bounded because a device that never drains, or a synthesis that
/// never returns, must not hang the process — past this the playback is cut and
/// the utterance is lost, which a log line says.
const SPEAK_DRAIN_MS: u64 = 15_000;

impl SpeechOut {
    /// Spawn the speak thread around a loaded voice.
    pub(crate) fn spawn(voice: PiperVoice) -> Self {
        let slot: Arc<(Mutex<Option<SpeechJob>>, Condvar)> =
            Arc::new((Mutex::new(None), Condvar::new()));
        let speaking = Arc::new(AtomicBool::new(false));
        let finish_by_ms = Arc::new(AtomicU64::new(0));
        let disabled = Arc::new(AtomicBool::new(false));

        let handle = {
            let (slot, speaking, finish_by_ms, disabled) = (
                Arc::clone(&slot),
                Arc::clone(&speaking),
                Arc::clone(&finish_by_ms),
                Arc::clone(&disabled),
            );
            std::thread::Builder::new()
                .name("speak".into())
                .spawn(move || speak_loop(&slot, &speaking, &finish_by_ms, &disabled, voice))
                .ok()
        };
        Self {
            slot,
            speaking,
            finish_by_ms,
            disabled,
            handle,
        }
    }

    /// The flag the capture side reads to hold the pipeline while sttd talks.
    pub(crate) fn speaking_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Queue `text` to be spoken, replacing anything still waiting. Never
    /// blocks and never fails: a dead or disabled speak thread costs one
    /// utterance of audio, never the transcript.
    fn enqueue(&self, tag: &str, text: &str) {
        if self.disabled.load(Ordering::Relaxed) {
            return;
        }
        let (lock, wakeup) = &*self.slot;
        if let Ok(mut pending) = lock.lock() {
            *pending = Some(SpeechJob {
                tag: tag.to_string(),
                text: text.to_string(),
            });
            wakeup.notify_one();
        }
    }
}

impl Drop for SpeechOut {
    fn drop(&mut self) {
        // Let whatever is queued be spoken, within a bound: the pending slot is
        // left alone and the thread is woken, so the trailing utterance is
        // heard. `finish_by_ms` is the bound it must respect, so this join
        // cannot hang the process on a device that never drains.
        self.finish_by_ms
            .store(now_ms() + SPEAK_DRAIN_MS, Ordering::Release);
        let (_lock, wakeup) = &*self.slot;
        wakeup.notify_all();
        if let Some(h) = self.handle.take() {
            if h.join().is_err() {
                log::warn!("speak thread panicked");
            }
        }
    }
}

/// Milliseconds since an arbitrary fixed instant, for the finish deadline.
///
/// `Instant` cannot cross the thread boundary as a shared value, and
/// `SystemTime` can jump. A monotonic counter derived from a process-start
/// `Instant` is the cheap monotonic clock this needs.
pub(crate) fn now_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Whether the finish deadline has passed. Zero means no deadline, so a thread
/// that was never asked to finish never sees one. Delegates to the same rule
/// `sound` applies while it waits.
fn past_deadline(finish_by_ms: &AtomicU64) -> bool {
    crate::sound::past_deadline(finish_by_ms, now_ms())
}

/// Sleep `total`, giving up early once the finish deadline passes. Returns
/// whether the sleep was cut short, so the caller can skip the work it was
/// about to start.
fn sleep_until_deadline(total: Duration, finish_by_ms: &AtomicU64) -> bool {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline {
        if past_deadline(finish_by_ms) {
            return true;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(Duration::from_millis(20).min(left));
    }
    past_deadline(finish_by_ms)
}

/// The speak thread: take the pending utterance, synthesize it, play it.
///
/// Never propagates anything. A voice that cannot speak (espeak gone, a graph
/// that rejects its input) disables the channel with one warning and the
/// thread idles, because a transcription tool that stops transcribing because
/// its speaker broke is worse than a silent one.
fn speak_loop(
    slot: &Arc<(Mutex<Option<SpeechJob>>, Condvar)>,
    speaking: &AtomicBool,
    finish_by_ms: &AtomicU64,
    disabled: &AtomicBool,
    mut voice: PiperVoice,
) {
    let rate = voice.sample_rate();
    loop {
        let job = {
            let (lock, wakeup) = &**slot;
            let Ok(mut pending) = lock.lock() else {
                return;
            };
            loop {
                if let Some(job) = pending.take() {
                    break job;
                }
                // A deadline that has passed with nothing queued means the drop
                // is over: nothing left to speak, so the thread ends here.
                if past_deadline(finish_by_ms) {
                    return;
                }
                let (again, _) = wakeup
                    .wait_timeout(pending, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner());
                pending = again;
            }
        };

        // Raise the flag first, then wait out the lead window, then synthesize:
        // the capture side reacts to the flag within its ~2 ms loop cadence and
        // takes its hold immediately, so by the time the first sample reaches
        // the speakers the pipeline is already discarding. The wait doubles as
        // the last chance for a deadline to cut this utterance off.
        speaking.store(true, Ordering::Release);
        if sleep_until_deadline(Duration::from_millis(crate::config::SPEAK_LEAD_MS), finish_by_ms) {
            speaking.store(false, Ordering::Release);
            continue;
        }
        let t0 = Instant::now();
        let synth = voice.synthesize(&job.text);
        let audio = match synth {
            Ok(a) => a,
            Err(e) => {
                let missing = e.kind() == std::io::ErrorKind::NotFound;
                disabled.store(true, Ordering::Release);
                log::warn!(
                    "{}: spoken emission disabled ({e:?}); transcripts still reach the other \
                     channels",
                    job.tag
                );
                if missing {
                    log::warn!("espeak-ng was not found on PATH, so text cannot be phonemized");
                }
                // Clear the flag before idling: the capture side must not stay
                // held for a channel that is now off.
                speaking.store(false, Ordering::Release);
                if missing {
                    return;
                }
                continue;
            }
        };
        let synth_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if !audio.is_empty() {
            let audio_sec = audio.len() as f64 / rate as f64;
            let played = crate::sound::play_mono(&audio, rate, finish_by_ms);
            if played {
                log::info!(
                    "{}: spoke {audio_sec:.1} s in {synth_ms:.0} ms (synthesis rtf={:.2})",
                    job.tag,
                    synth_ms / 1000.0 / audio_sec.max(0.001)
                );
            } else {
                log::warn!("{}: cut off at shutdown before it could be spoken", job.tag);
            }
        }
        // The tail is inside the raised flag too: the device still holds the
        // last samples, and the capture side must stay held until it has let
        // them out.
        let _ = sleep_until_deadline(Duration::from_millis(SPEAK_TAIL_MS), finish_by_ms);
        speaking.store(false, Ordering::Release);
    }
}

/// Prod adapter: opt-in stdout, Wayland clipboard, auto-paste, spoken
/// transcripts, dump dir.
///
/// Async by design: `emit()` only clones into an `OwnedEmission` and does a
/// non-blocking `rtrb` push, so a stuck `wl-copy`/`ydotool`/disk never
/// stalls inference. A single `emit` worker thread owns all side
/// effects in FIFO order; `Drop` closes the channel and joins it, so shutdown
/// (`asr_handle.join()`) flushes the queue before the process exits.
/// Untested glue by design — the logic worth testing (verdict → action) is
/// exercised through `RecordingEmitter`; this only shells out to binaries
/// and files.
pub(crate) struct SystemEmitter {
    /// The emission queue, shared with any `sharing()` handle.
    ///
    /// `Producer` is `Send` but not `Sync`, so two emitters writing the same queue
    /// need the mutex. One uncontended lock per utterance against a disk write and
    /// a process spawn on the other side.
    tx: Option<Arc<Mutex<Producer<OwnedEmission>>>>,
    worker: Option<JoinHandle<()>>,
    /// True on the handle that owns the queue and the worker thread.
    ///
    /// This flag, not `worker: None`, is what tells `Drop` whether closing the
    /// queue is its job: `tx` is an `Arc`, so `take()` on a sharing handle would
    /// close the queue for the owner too and lose the trailing utterance.
    owns_queue: bool,
    /// Kept for the final prune in `Drop` (the worker thread owns the only
    /// other copy).
    dump_dir: Option<PathBuf>,
    /// The speak thread's "currently talking" flag, kept here so the capture
    /// loop can read it. The `SpeechOut` itself has moved onto the emit worker
    /// thread (it owns the channel end), so this is the only handle left on
    /// this side — and the flag is an `Arc`, so both observe the same bit.
    worker_speaking: Option<Arc<AtomicBool>>,
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
    paste_mode: crate::cli::PasteMode,
    stdout_enabled: bool,
    /// Utterances handled since the last `prune_dump_dir` scan.
    since_prune: usize,
    /// The spoken channel, when `--speak` is on.
    speech: Option<SpeechOut>,
}

/// One `SystemEmitter::new` argument per emission channel. A struct because
/// the bare form had grown to six positional parameters, which is exactly where
/// a caller starts transposing two of them.
pub(crate) struct EmissionPolicy {
    pub(crate) dump_dir: Option<PathBuf>,
    pub(crate) copy: bool,
    pub(crate) paste: bool,
    pub(crate) paste_mode: crate::cli::PasteMode,
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
            .spawn(move || loop {
                match rx.pop() {
                    Ok(owned) => worker.handle(owned),
                    Err(_) => {
                        if rx.is_abandoned() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn emit thread: {e}"))?;
        Ok(Self {
            tx: Some(Arc::new(Mutex::new(tx))),
            worker: Some(handle),
            owns_queue: true,
            dump_dir,
            worker_speaking,
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
        }
    }
}

/// The speak thread's flag, taken while the `SpeechOut` is still owned here.
/// `None` when `--speak` is off, which the caller turns into a permanently-clear
/// flag so the capture loop needs no separate "is speech on" case.
fn worker_speech_flag(speech: Option<&SpeechOut>) -> Option<Arc<AtomicBool>> {
    speech.map(SpeechOut::speaking_flag)
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
        let Some(cell) = self.tx.as_ref() else { return };
        // Locked rather than `get_mut`: with a translation step in play this
        // emitter is one of two writing the same queue.
        let mut tx = cell.lock().unwrap_or_else(|e| e.into_inner());
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
        } else if mode == crate::cli::PasteMode::Type {
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

/// Write `reason` as `<dump_dir>/<tag>.txt` for an utterance that was
/// dropped rather than printed, so `STTD_DUMP_DIR` output stays correlatable
/// with the stderr log: previously a dropped utterance left only a WAV with
/// no indication of what became of it or why.
fn dump_drop_reason(dump_dir: &Option<PathBuf>, tag: &str, reason: &str) {
    if let Some(dir) = dump_dir {
        let txt_path = dir.join(format!("{tag}.txt"));
        if let Err(e) = write_owner_only_text(&txt_path, reason) {
            log::warn!("failed to write {}: {e:?}", txt_path.display());
        }
    }
}

/// Create/truncate `path` with owner-only permissions (0600 on unix). Mic
/// audio and transcripts are sensitive; `File::create` honors umask (often
/// 0644) and keeps a pre-existing path's mode — neither is acceptable here.
pub(crate) fn owner_only_file(path: &Path) -> Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

/// Write `contents` to `path` with owner-only permissions (see
/// `owner_only_file`).
fn write_owner_only_text(path: &Path, contents: &str) -> Result<()> {
    use std::io::Write;
    let mut file = owner_only_file(path)?;
    file.write_all(contents.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

/// Ensure `dir` exists with owner-only permissions (0700 on unix), like
/// Murmure's recordings dir. Creation errors propagate (fatal at startup);
/// a chmod failure only warns — transcription never depends on it.
fn ensure_owner_only_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir)
            .with_context(|| format!("failed to stat {}", dir.display()))?
            .permissions()
            .mode()
            & 0o777;
        if mode != 0o700 {
            if let Err(e) = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)) {
                log::warn!("failed to chmod 0700 {}: {e:?}", dir.display());
            }
        }
    }
    Ok(())
}

/// Delete utterance pairs past `DUMP_KEEP_FILES`, newest-first by tag name
/// (`utt_%06d` sorts chronologically). Observability hygiene only: I/O
/// errors warn and keep going, never fail transcription.
fn prune_dump_dir(dir: &Path) {
    let stems = match dump_stems(dir) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("dump prune: failed to list {}: {e:?}", dir.display());
            return;
        }
    };
    for stem in stems_to_prune(stems, DUMP_KEEP_FILES) {
        for ext in ["wav", "txt"] {
            let path = dir.join(format!("{stem}.{ext}"));
            // `remove_file` already reports a missing file as `NotFound`; the
            // `exists()` probe this replaces only doubled the syscalls.
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("dump prune: failed to remove {}: {e:?}", path.display()),
            }
        }
    }
}

/// Sorted `utt_NNNNNN` stems of the pairs in `dir` (the `.wav` names; the
/// matching `.txt` is implied).
fn dump_stems(dir: &Path) -> Result<Vec<String>> {
    let mut stems = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("utt_") && name.ends_with(".wav") {
            stems.push(name.trim_end_matches(".wav").to_string());
        }
    }
    stems.sort();
    Ok(stems)
}

/// Pure selection: which stems to delete keeping the newest `keep`. Takes
/// the names already sorted by the caller (`dump_stems`), oldest first.
fn stems_to_prune(mut stems: Vec<String>, keep: usize) -> Vec<String> {
    if stems.len() > keep {
        stems.truncate(stems.len() - keep);
        stems
    } else {
        Vec::new()
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
/// (requires the `ydotoold` daemon). `CtrlV` synthesizes LEFTCTRL+V
/// (`29:1 47:1 47:0 29:0`); `CtrlShiftV` adds LEFTSHIFT for terminals;
/// `Type` types via `ydotool type` on stdin (no clipboard, layout-dependent).
/// Returns `io::Error` unwrapped so the caller can match `NotFound` and
/// disable itself with a single warning on machines without `ydotool`.
pub(crate) fn paste_via_ydotool(mode: crate::cli::PasteMode) -> std::io::Result<()> {
    use crate::cli::PasteMode;
    match mode {
        PasteMode::CtrlV => run_ydotool_key(&["29:1", "47:1", "47:0", "29:0"]),
        PasteMode::CtrlShiftV => run_ydotool_key(&["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]),
        PasteMode::Type => {
            // No clipboard text available here (one-shot path passes text
            // explicitly via `type_via_ydotool`); without text there is
            // nothing to type.
            Err(std::io::Error::other(
                "paste-mode=type needs text; nothing to paste",
            ))
        }
    }
}

/// Type `text` via `ydotool type --file -` (stdin). Used by `Type` paste mode
/// and never touches the clipboard.
pub(crate) fn type_via_ydotool(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("ydotool")
        .args(["type", "--file", "-"])
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("ydotool type stdin unavailable"))?
        .write_all(text.as_bytes())?;
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ydotool type exited with {status}"
        )))
    }
}

fn run_ydotool_key(events: &[&str]) -> std::io::Result<()> {
    use std::process::Command;
    let status = Command::new("ydotool").arg("key").args(events).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ydotool exited with {status}"
        )))
    }
}

/// Path of the last-emitted-transcription file for `--paste-last`:
/// `<tmp>/sttd-last.txt` (owner-only, overwritten per emission).
pub(crate) fn last_text_path() -> PathBuf {
    std::env::temp_dir().join("sttd-last.txt")
}

/// Remember the last emitted text for `--paste-last`. Best-effort only:
/// failures warn, never fail transcription.
fn write_last_text(text: &str) {
    if text.trim().is_empty() {
        return;
    }
    let path = last_text_path();
    if let Err(e) = write_owner_only_text(&path, text) {
        log::warn!("failed to write {}: {e:?}", path.display());
    }
}

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
        paste_mode: crate::cli::PasteMode::CtrlV,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS};
    use crate::gate::GateConfig;

    fn gate() -> GateConfig {
        GateConfig::new(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY)
    }

    fn evidence() -> Evidence {
        Evidence {
            speech_frames: 137,
            total_frames: 200,
            rms: 0.01178,
        }
    }

    fn strong_evidence() -> Evidence {
        evidence()
    }

    fn weak_evidence() -> Evidence {
        evidence()
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
            evidence,
        });
    }

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
    fn a_disabled_speak_channel_stops_accepting_and_clears_the_flag() {
        // The espeak-gone path: the thread warns once, raises `disabled`, and
        // clears `speaking` so the capture side is not left held for a channel
        // that is now off.
        let speaking = Arc::new(AtomicBool::new(true));
        let disabled = Arc::new(AtomicBool::new(false));
        // Model the loop's own bookkeeping on the failure branch.
        speaking.store(false, Ordering::Release);
        disabled.store(true, Ordering::Release);
        assert!(!speaking.load(Ordering::Acquire));
        assert!(disabled.load(Ordering::Acquire));
        // An empty voice name is refused at load with the reason, which is what
        // stops this state being reached silently.
        let json = r#"{"audio": {"sample_rate": 22050}, "phoneme_type": "espeak",
                        "phoneme_id_map": {"_": [0], "^": [1], "$": [2]}, "num_symbols": 256}"#;
        let config: crate::piper::VoiceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.espeak.voice, "");
    }

    #[test]
    fn speech_is_opt_in_and_follows_the_same_emit_rules() {
        // The channel mirrors stdout: off by default, and only ever handed an
        // utterance the gate actually emitted.
        let evidence = strong_evidence();
        let pcm = vec![0.01f32; 1600];
        let mut emitter = RecordingEmitter::new();
        emit(
            &mut emitter,
            "utt_000001",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[0].emitted);
        assert!(!emitter.records[0].spoken, "--speak is off by default");

        let mut emitter = RecordingEmitter::new();
        emitter.speak_enabled = true;
        emit(
            &mut emitter,
            "utt_000002",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[0].spoken);
        // A failed clipboard copy does not stop the voice: prod enqueues speech
        // after the copy step and neither depends on the other.
        emitter.fail_copy = true;
        emit(
            &mut emitter,
            "utt_000003",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[1].emitted);
        assert!(!emitter.records[1].copied);
        assert!(emitter.records[1].spoken);

        // Drops never speak: a hallucination and an empty-cleaned utterance
        // both reach no channel at all.
        for (text, expected) in [
            ("Napisy stworzone przez społeczność Amara.org", false),
            ("   ", false),
        ] {
            emit(
                &mut emitter,
                "utt_000004",
                &pcm,
                text,
                gate().decide_transcript(text),
                &evidence,
            );
            let r = emitter.records.last().unwrap();
            assert!(!r.emitted, "text={text:?}");
            assert_eq!(r.spoken, expected, "text={text:?}");
        }
    }

    #[test]
    fn the_speaking_flag_is_clear_without_the_channel() {
        // The capture loop reads this flag unconditionally, so with `--speak`
        // off it must exist and stay false rather than the loop needing a
        // separate "is speech on" case.
        let emitter = SystemEmitter::new(policy_off()).unwrap();
        let flag = emitter.speaking_flag();
        assert!(!flag.load(Ordering::Acquire));
        // Every call returns a flag that can be read; two calls must not share
        // state, or a stale one would leak between runs in one process (tests).
        let other = emitter.speaking_flag();
        assert!(!Arc::ptr_eq(&flag, &other) || !flag.load(Ordering::Acquire));
    }

    #[test]
    fn a_past_deadline_cuts_the_wait_short() {
        // `SpeechOut::drop` must not block for the length of a playback: the
        // wait polls the deadline every 20 ms. Asserted on the helpers rather
        // than a real playback, since there is no voice or device here.
        let deadline = AtomicU64::new(1);
        assert!(
            crate::sound::past_deadline(&deadline, 1),
            "a deadline at or before now is past"
        );
        assert!(crate::sound::past_deadline(&deadline, 2));
        assert!(
            !crate::sound::past_deadline(&deadline, 0),
            "not yet, a millisecond before"
        );
        let t0 = Instant::now();
        assert!(sleep_until_deadline(Duration::from_secs(30), &deadline));
        assert!(
            t0.elapsed() < Duration::from_millis(200),
            "waited {:?} on an already-past deadline",
            t0.elapsed()
        );
    }

    #[test]
    fn no_deadline_means_never_cut_off() {
        // Zero is "no deadline", so a thread that was never asked to finish
        // waits its whole course out. (Otherwise every speak would cut itself
        // off at once, since the clock is always past zero.)
        let none = AtomicU64::new(0);
        assert!(!crate::sound::past_deadline(&none, u64::MAX / 2));
        assert!(!crate::sound::past_deadline(&none, 0));
        let t0 = Instant::now();
        assert!(!sleep_until_deadline(Duration::from_millis(50), &none));
        assert!(
            t0.elapsed() >= Duration::from_millis(50),
            "no deadline must let the sleep run: {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn the_finish_deadline_is_what_drop_sets_and_it_is_bounded() {
        // The drop path's whole contract: the trailing utterance is spoken
        // within a bound, so a device that never drains cannot hang the
        // process on the way out.
        assert!(SPEAK_DRAIN_MS > 0, "a zero deadline would cut at once");
        assert!(
            SPEAK_DRAIN_MS < 60_000,
            "a drain this long is a hang in disguise: {SPEAK_DRAIN_MS} ms"
        );
        // The deadline arithmetic `drop` uses: now + bound, which must land in
        // the future rather than in the past (an overflow would).
        let deadline = now_ms() + SPEAK_DRAIN_MS;
        assert!(!crate::sound::past_deadline(&AtomicU64::new(deadline), now_ms()));
        assert!(crate::sound::past_deadline(&AtomicU64::new(deadline), deadline));
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

    #[test]
    fn prune_keeps_newest_and_drops_oldest() {
        let stems: Vec<String> = (0..5).map(|i| format!("utt_{i:06}")).collect();
        assert!(stems_to_prune(stems.clone(), 5).is_empty());
        assert!(stems_to_prune(stems.clone(), 10).is_empty());
        assert_eq!(
            stems_to_prune(stems.clone(), 2),
            vec![
                "utt_000000".to_string(),
                "utt_000001".to_string(),
                "utt_000002".to_string()
            ]
        );
        assert!(stems_to_prune(Vec::new(), 2).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn dump_dir_and_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("sttd-emit-perms-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        ensure_owner_only_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "dir mode={mode:o}");
        let txt = dir.join("utt_000001.txt");
        write_owner_only_text(&txt, "Dzień dobry").unwrap();
        let mode = std::fs::metadata(&txt).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "file mode={mode:o}");
        assert_eq!(std::fs::read_to_string(&txt).unwrap(), "Dzień dobry");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dump_retention_prunes_oldest_pairs() {
        use crate::config::DUMP_KEEP_FILES;
        let dir = std::env::temp_dir().join(format!("sttd-emit-retain-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let evidence = strong_evidence();
        {
            let mut emitter = SystemEmitter::new(EmissionPolicy {
                dump_dir: Some(dir.clone()),
                ..policy_off()
            })
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            for i in 0..(DUMP_KEEP_FILES + 5) {
                let tag = format!("utt_{i:06}");
                emitter.emit(Emission {
                    tag: &tag,
                    pcm: &pcm,
                    text: "Dzień dobry",
                    verdict: gate().decide_transcript("Dzień dobry"),
                    evidence: &evidence,
                });
            }
            // Drop at scope end flushes the async queue.
        }
        let wavs: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .collect();
        assert_eq!(wavs.len(), DUMP_KEEP_FILES, "kept={}", wavs.len());
        assert!(!dir.join("utt_000000.wav").exists());
        assert!(!dir.join("utt_000000.txt").exists());
        let newest = format!("utt_{:06}.wav", DUMP_KEEP_FILES + 4);
        assert!(dir.join(newest).exists());
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

    #[test]
    fn last_text_path_is_tmp_file() {
        assert!(last_text_path().ends_with("sttd-last.txt"));
    }
}
