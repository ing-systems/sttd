//! Process signals: Ctrl-C and `SIGTERM` stop the run, `SIGUSR1`/`SIGUSR2`/
//! `SIGHUP` pause, resume and toggle it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};
use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM, SIGUSR1, SIGUSR2};
use signal_hook::iterator::Signals;

/// Install the stop and pause/resume signal handlers: Ctrl-C (`SIGINT`) and
/// `SIGTERM` request a graceful shutdown (the `running` flag, so Ctrl-C,
/// `--stop` and `kill` all flush trailing speech instead of dying
/// mid-CUDA-op), `SIGUSR1` pauses, `SIGUSR2` resumes, `SIGHUP` toggles. The
/// handler thread only flips shared flags; the main loop does the actual
/// discarding, so the transition is idempotent (pause-while-paused /
/// resume-while-running are no-ops, double TERM is one shutdown). Audible ack
/// (`sound` module) fires alongside the flag flip so a daemon user hears the
/// transition; `ready` itself is signaled separately by the ASR thread once
/// transcription can actually resume.
pub(super) fn install_signal_handlers(
    pause_state: crate::control::PauseState,
    running: Arc<AtomicBool>,
    sound_enabled: bool,
) -> Result<()> {
    let mut signals = Signals::new([SIGINT, SIGTERM, SIGUSR1, SIGUSR2, SIGHUP])
        .context("failed to register stop/pause/resume signals")?;
    thread::spawn(move || {
        for sig in signals.forever() {
            let (paused, why) = match sig {
                SIGINT => {
                    running.store(false, Ordering::Release);
                    continue;
                }
                SIGTERM => {
                    log::info!("shutting down (SIGTERM)");
                    running.store(false, Ordering::Release);
                    continue;
                }
                SIGUSR1 => (pause_state.set_paused(true), "SIGUSR1"),
                SIGUSR2 => (pause_state.set_paused(false), "SIGUSR2"),
                SIGHUP => (pause_state.toggle_paused(), "SIGHUP toggle"),
                _ => continue,
            };
            if paused {
                log::info!("paused ({why})");
                if sound_enabled {
                    crate::sound::play_paused();
                }
            } else {
                log::info!("resumed ({why})");
                if sound_enabled {
                    crate::sound::play_resumed();
                }
            }
            crate::dbus::emit_state_changed(paused);
        }
    });
    Ok(())
}
