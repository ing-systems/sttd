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

use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::config::{DUMP_KEEP_FILES, WHISPER_RATE};
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
    /// Text-token average probability and log-probability, for metrics
    /// lines. `None` for models that don't report per-token probabilities
    /// (Parakeet) — see `GateConfig::decide_transcript`.
    pub(crate) prob: Option<f32>,
    pub(crate) logprob: Option<f32>,
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
    pub(crate) prob: Option<f32>,
    pub(crate) logprob: Option<f32>,
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
    paste_mode: crate::cli::PasteMode,
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
        paste_mode: crate::cli::PasteMode,
        stdout_enabled: bool,
    ) -> Result<Self> {
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
            dump_dir,
            clipboard_available: copy_enabled,
            paste_available: paste_enabled,
            paste_mode,
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
        self.handle_inner(&owned);
        // Bound the directory: newest pairs win, oldest are deleted. Runs
        // for every utterance (emits and drops alike).
        if let Some(dir) = &self.dump_dir {
            prune_dump_dir(dir);
        }
    }

    fn handle_inner(&mut self, owned: &OwnedEmission) {
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
                // Never clobber the clipboard with empty text (e.g. cleaning
                // removed everything, or an image is currently copied): keep
                // any image/content the user has there instead.
                if text.trim().is_empty() {
                    dump_drop_reason(&self.dump_dir, tag, "DROPPED(empty-cleaned)");
                    return;
                }
                log::info!(
                    "{tag}: avg_p={} avg_logprob={} threshold={floor:.3} \
                     ({strength} evidence, density={:.3})",
                    fmt_prob(*prob),
                    fmt_prob(*logprob),
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
                    "{tag}: avg_p={} avg_logprob={} threshold={floor:.3} \
                     ({strength} evidence, density={:.3})",
                    fmt_prob(*prob),
                    fmt_prob(*logprob),
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
        // clipboard: paste synthesizes keys, so it must only run
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
        // Remember the last emitted text for `--paste-last`, independent of
        // `--copy`: one-shot re-paste must work even when copy was off.
        write_last_text(text);
        if self.paste_available && copied {
            match paste_via_ydotool(self.paste_mode) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    log::warn!(
                        "--paste is enabled but `ydotool` was not found on PATH; disabling auto-paste"
                    );
                    self.paste_available = false;
                }
                Err(e) => log::warn!("{tag}: auto-paste failed (is ydotoold running?): {e:?}"),
            }
        } else if self.paste_available && !copied && self.paste_mode == crate::cli::PasteMode::Type
        {
            // `type` mode needs no clipboard: paste directly even when copy
            // is off or failed.
            match type_via_ydotool(text) {
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
            if let Err(e) = write_owner_only_text(&txt_path, text) {
                log::warn!("failed to write {}: {e:?}", txt_path.display());
            }
        }
    }
}

/// Render an optional probability/log-probability for a log line: `n/a` when
/// the model reports none (Parakeet), 3 decimals otherwise.
fn fmt_prob(p: Option<f32>) -> String {
    match p {
        Some(p) => format!("{p:.3}"),
        None => "n/a".to_string(),
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
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
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
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    log::warn!("dump prune: failed to remove {}: {e:?}", path.display());
                }
            }
        }
    }
}

fn dump_stems(dir: &Path) -> Result<Vec<String>> {
    let mut stems = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))? {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("utt_") && name.ends_with(".wav") {
            stems.push(name.trim_end_matches(".wav").to_string());
        }
    }
    stems.sort();
    Ok(stems)
}

/// Pure selection: which stems to delete keeping the newest `keep`.
/// Newest-first ordering comes from the caller-sorted names.
fn stems_to_prune(mut stems: Vec<String>, keep: usize) -> Vec<String> {
    stems.sort();
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
            Err(std::io::Error::other("paste-mode=type needs text; nothing to paste"))
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
        Err(std::io::Error::other(format!("ydotool type exited with {status}")))
    }
}

fn run_ydotool_key(events: &[&str]) -> std::io::Result<()> {
    use std::process::Command;
    let status = Command::new("ydotool")
        .arg("key")
        .args(events)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!("ydotool exited with {status}")))
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
            prob: Some(0.9),
            logprob: Some(-0.1),
            evidence,
        });
    }

    #[test]
    fn confident_transcript_is_emitted_with_copy_and_paste() {
        let evidence = strong_evidence();
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.9));
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
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.9));
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
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.9));
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
            Some(0.9),
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
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.9));
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
            Some(0.1),
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
            let verdict = gate().decide_transcript(evidence.speech_frames, evidence.rms, text, Some(0.9));
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
        let verdict = gate().decide_transcript(evidence.speech_frames, evidence.rms, "   ", Some(0.0));
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
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.478));
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
            let mut emitter = SystemEmitter::new(
                Some(dir.clone()),
                false,
                false,
                crate::cli::PasteMode::CtrlV,
                false,
            )
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            emitter.emit(Emission {
                tag: "utt_000001",
                pcm: &pcm,
                text: "Dzień dobry",
                verdict: gate().decide_transcript(
                    evidence.speech_frames,
                    evidence.rms,
                    "Dzień dobry",
                    Some(0.9),
                ),
                prob: Some(0.9),
                logprob: Some(-0.1),
                evidence: &evidence,
            });
            emitter.emit(Emission {
                tag: "utt_000002",
                pcm: &pcm,
                text: "   ",
                verdict: TranscriptVerdict::DropEmpty,
                prob: None,
                logprob: None,
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
    fn no_confidence_signal_still_emits_and_copies() {
        // Parakeet-shaped emission: `prob`/`logprob` are `None`, the gate
        // never scored a confidence floor (see gate.rs), yet the emission
        // path must treat this exactly like any other Emit verdict.
        let evidence = strong_evidence();
        let verdict = gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", None);
        assert!(matches!(verdict, TranscriptVerdict::Emit { .. }));
        let mut emitter = RecordingEmitter::new();
        let pcm = vec![0.01f32; 1600];
        emitter.emit(Emission {
            tag: "utt_000001",
            pcm: &pcm,
            text: "Dzień dobry",
            verdict,
            prob: None,
            logprob: None,
            evidence: &evidence,
        });
        let r = &emitter.records[0];
        assert!(r.emitted);
        assert!(r.copied);
    }

    #[test]
    fn fmt_prob_renders_none_as_na() {
        assert_eq!(fmt_prob(Some(0.9)), "0.900");
        assert_eq!(fmt_prob(None), "n/a");
    }

    #[test]
    fn system_emitter_emit_never_blocks_inference() {
        // `emit()` is a single clone + bounded push: 200 rapid emits must
        // finish well under a whisper inference (seconds), even with disk I/O
        // still draining on the worker.
        let dir = std::env::temp_dir().join(format!("sttd-emit-nb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let evidence = strong_evidence();
        let mut emitter = SystemEmitter::new(
            Some(dir.clone()),
            false,
            false,
            crate::cli::PasteMode::CtrlV,
            false,
        )
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
                verdict: gate().decide_transcript(
                    evidence.speech_frames,
                    evidence.rms,
                    "Dzień dobry",
                    Some(0.9),
                ),
                prob: Some(0.9),
                logprob: Some(-0.1),
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
            vec!["utt_000000".to_string(), "utt_000001".to_string(), "utt_000002".to_string()]
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
            let mut emitter = SystemEmitter::new(
                Some(dir.clone()),
                false,
                false,
                crate::cli::PasteMode::CtrlV,
                false,
            )
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            for i in 0..(DUMP_KEEP_FILES + 5) {
                let tag = format!("utt_{i:06}");
                emitter.emit(Emission {
                    tag: &tag,
                    pcm: &pcm,
                    text: "Dzień dobry",
                    verdict: gate().decide_transcript(
                        evidence.speech_frames,
                        evidence.rms,
                        "Dzień dobry",
                        Some(0.9),
                    ),
                    prob: Some(0.9),
                    logprob: Some(-0.1),
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
        let verdict =
            gate().decide_transcript(evidence.speech_frames, evidence.rms, "Dzień dobry", Some(0.9));
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
            let mut emitter = SystemEmitter::new(
                Some(dir.clone()),
                false,
                false,
                crate::cli::PasteMode::CtrlV,
                false,
            )
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            emitter.emit(Emission {
                tag: "utt_000001",
                pcm: &pcm,
                text: "   ",
                verdict: gate().decide_transcript(
                    evidence.speech_frames,
                    evidence.rms,
                    "Dzień dobry",
                    Some(0.9),
                ),
                prob: Some(0.9),
                logprob: Some(-0.1),
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
