// ---------------------------------------------------------------------------
// Background daemon: `--start`/`--stop`/`--status` behind one seam.
// ---------------------------------------------------------------------------
//
// Mic mode is foreground by default. `--start` forks into the background
// (double-fork + `setsid` via `daemonize`): the parent prints the pid/log
// paths and exits, the child redirects combined stdout+stderr to the log
// file and keeps capturing past terminal close. `--stop` sends `SIGTERM`
// (graceful flush, same as Ctrl-C) then `SIGKILL` on timeout; `--status`
// reports liveness via `kill(pid, 0)`. All three locate the daemon through
// the pid file only, so `--pid-file` must match across invocations.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Default pid-file path: `<tmp>/sttd.pid`.
pub(crate) fn default_pid_file() -> PathBuf {
    std::env::temp_dir().join("sttd.pid")
}

/// Default combined log path: `<tmp>/sttd.log`.
pub(crate) fn default_log_file() -> PathBuf {
    std::env::temp_dir().join("sttd.log")
}

/// Fork into the background. Parent prints pid/log and exits 0; child
/// (stdout+stderr appended to `log`) returns `Ok` to keep running.
/// Relative paths (an explicit `--model models/...`, `--dump-dir`) keep
/// working because the daemon keeps the current working directory instead of
/// `chdir("/")`. Catalog specs don't need that: they resolve into
/// `DEFAULT_MODELS_DIR`, baked absolute at build time.
pub(crate) fn daemonize_start(pid_file: &Path, log: &Path) -> Result<()> {
    if let Some(parent) = log.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let stdout_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("failed to open {}", log.display()))?;
    let stderr_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("failed to open {}", log.display()))?;
    let cwd = std::env::current_dir().context("failed to read working directory")?;

    match daemonize::Daemonize::new()
        .pid_file(pid_file)
        .working_directory(cwd)
        .stdout(stdout_log)
        .stderr(stderr_log)
        .execute()
    {
        daemonize::Outcome::Parent(Ok(_)) => {
            // Pid file is written by the daemon side; read it back for the
            // startup line (fall back to paths when it isn't there yet).
            match read_pid_file(pid_file) {
                Ok(pid) => println!(
                    "[sttd] started (pid {pid}, log {}, pid-file {})",
                    log.display(),
                    pid_file.display()
                ),
                Err(_) => println!(
                    "[sttd] started (log {}, pid-file {})",
                    log.display(),
                    pid_file.display()
                ),
            }
            std::process::exit(0);
        }
        daemonize::Outcome::Parent(Err(e)) => {
            anyhow::bail!("failed to daemonize (is another instance running?): {e}");
        }
        daemonize::Outcome::Child(Ok(_)) => Ok(()),
        daemonize::Outcome::Child(Err(e)) => {
            anyhow::bail!("failed to daemonize: {e}");
        }
    }
}

/// Parse a pid file to a pid. Errors on missing/unparseable content so
/// callers can report stale state instead of signaling pid 0.
pub(crate) fn read_pid_file(path: &Path) -> Result<i32> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("no pid file at {} (is the daemon running?)", path.display()))?;
    parse_pid(&text).with_context(|| format!("unparseable pid file at {}", path.display()))
}

/// Pure pid parse, split out for tests (no filesystem).
pub(crate) fn parse_pid(text: &str) -> Result<i32> {
    let pid: i32 = text
        .trim()
        .parse()
        .with_context(|| format!("expected a pid, got {text:?}"))?;
    if pid <= 0 {
        anyhow::bail!("expected a positive pid, got {pid}");
    }
    Ok(pid)
}

/// Liveness via `kill(pid, 0)`: no signal sent. `EPERM` (exists, no
/// permission) counts as alive; anything else non-zero counts as dead.
pub(crate) fn process_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: sig 0 performs error checking only, no signal is delivered.
    let r = unsafe { libc::kill(pid, 0) };
    if r == 0 {
        return true;
    }
    matches!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(e) if e == libc::EPERM
    )
}

fn send_signal(pid: i32, sig: libc::c_int, name: &str) -> Result<()> {
    // SAFETY: kill with a valid pid/sig is async-signal-safe and has no
    // invariants beyond its arguments.
    let r = unsafe { libc::kill(pid, sig) };
    if r == 0 {
        return Ok(());
    }
    Err(std::io::Error::last_os_error()).with_context(|| format!("failed to send {name} to pid {pid}"))
}

fn wait_for_exit(pid: i32, timeout: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if !process_alive(pid) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    !process_alive(pid)
}

/// Stop the daemon at `pid_file`: `SIGTERM` (graceful flush on the daemon
/// side), wait ~10 s, then `SIGKILL` on timeout. A dead pid with a
/// leftover file is reported as stale and the file removed.
pub(crate) fn run_stop(pid_file: &Path) -> Result<()> {
    let pid = read_pid_file(pid_file)?;
    if !process_alive(pid) {
        std::fs::remove_file(pid_file).ok();
        anyhow::bail!("[sttd] not running (stale pid file, pid {pid})");
    }
    send_signal(pid, libc::SIGTERM, "SIGTERM")?;
    println!("[sttd] stopping (pid {pid})...");
    if wait_for_exit(pid, std::time::Duration::from_secs(10)) {
        println!("[sttd] stopped (pid {pid})");
        if let Ok(cur) = read_pid_file(pid_file) {
            if cur == pid {
                std::fs::remove_file(pid_file).ok();
            }
        }
        return Ok(());
    }
    log::warn!("SIGTERM timed out, sending SIGKILL (pid {pid})...");
    send_signal(pid, libc::SIGKILL, "SIGKILL")?;
    if wait_for_exit(pid, std::time::Duration::from_secs(2)) {
        println!("[sttd] killed (pid {pid})");
        Ok(())
    } else {
        anyhow::bail!("[sttd] still running after SIGKILL (pid {pid})");
    }
}

/// Report daemon liveness. Returns the process exit code (0 iff running);
/// never fails, so `--status` composes with `&&`/`||`.
pub(crate) fn run_status(pid_file: &Path) -> i32 {
    match read_pid_file(pid_file) {
        Ok(pid) if process_alive(pid) => {
            println!("[sttd] running (pid {pid})");
            0
        }
        Ok(pid) => {
            println!("[sttd] not running (stale pid file, pid {pid})");
            1
        }
        Err(e) => {
            println!("[sttd] not running ({e:#})");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sttd-test-{}-{}", std::process::id(), name))
    }

    #[test]
    fn pid_parse_accepts_plain_pid_with_whitespace() {
        assert_eq!(parse_pid("1234\n").unwrap(), 1234);
        assert_eq!(parse_pid("  42  ").unwrap(), 42);
        assert!(parse_pid("").is_err());
        assert!(parse_pid("abc").is_err());
        assert!(parse_pid("0").is_err());
        assert!(parse_pid("-7").is_err());
    }

    #[test]
    fn pid_file_roundtrip_and_missing() {
        let path = tmp_path("pid");
        std::fs::write(&path, "4242\n").unwrap();
        assert_eq!(read_pid_file(&path).unwrap(), 4242);
        std::fs::remove_file(&path).ok();
        assert!(read_pid_file(&path).is_err());
        // Unparseable content errors instead of signaling pid 0.
        std::fs::write(&path, "bogus\n").unwrap();
        assert!(read_pid_file(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn liveness_reports_self_alive_and_absurd_pid_dead() {
        let me = std::process::id() as i32;
        assert!(process_alive(me));
        assert!(!process_alive(0));
        assert!(!process_alive(-1));
        assert!(!process_alive(i32::MAX));
    }

    #[test]
    fn default_paths_live_in_tmp() {
        assert!(default_pid_file().ends_with("sttd.pid"));
        assert!(default_log_file().ends_with("sttd.log"));
    }
}
