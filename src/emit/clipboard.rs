//! The Wayland side effects: clipboard copy, `ydotool` paste or typing, and
//! the last-transcript file `--paste-last` reads.

use std::path::PathBuf;

use crate::fsutil::write_owner_only_text;

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

/// How `--paste` inserts text via `ydotool` (Wayland only).
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PasteMode {
    /// LEFTCTRL+V (default).
    #[default]
    CtrlV,
    /// LEFTCTRL+LEFTSHIFT+V (terminals).
    CtrlShiftV,
    /// `ydotool type` via stdin (no clipboard, layout-dependent).
    Type,
}

/// Paste the clipboard at the current cursor position via `ydotool`
/// (requires the `ydotoold` daemon). `CtrlV` synthesizes LEFTCTRL+V
/// (`29:1 47:1 47:0 29:0`); `CtrlShiftV` adds LEFTSHIFT for terminals;
/// `Type` types via `ydotool type` on stdin (no clipboard, layout-dependent).
/// Returns `io::Error` unwrapped so the caller can match `NotFound` and
/// disable itself with a single warning on machines without `ydotool`.
pub(crate) fn paste_via_ydotool(mode: PasteMode) -> std::io::Result<()> {
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
pub(super) fn write_last_text(text: &str) {
    if text.trim().is_empty() {
        return;
    }
    let path = last_text_path();
    if let Err(e) = write_owner_only_text(&path, text) {
        log::warn!("failed to write {}: {e:?}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn last_text_path_is_tmp_file() {
        assert!(last_text_path().ends_with("sttd-last.txt"));
    }
}
