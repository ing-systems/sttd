// ---------------------------------------------------------------------------
// Running sttd: the orchestration layer between the command line and the
// pipeline
// ---------------------------------------------------------------------------
//
// `main` parses the flags and picks one of these; everything below this layer
// is built from resolved values, never from `Cli`.
//
// * `microphone` is the live run: capture, the pause and speaking holds, the
//   device rebuild, and the ordered shutdown.
// * `file` runs the same pipeline offline over a WAV.
// * `setup` is what both resolve from the flags before the first frame;
//   `queue` the capture side of the two queues into the ASR worker.
// * `unix_signals` wires Ctrl-C and the pause signals; `commands` holds the
//   one-shot commands that run instead of a session.

mod commands;
mod file;
mod microphone;
mod queue;
mod setup;
mod unix_signals;

pub(crate) use commands::{print_available_models, print_available_voices, run_paste_last};
pub(crate) use file::run_file_mode;
pub(crate) use microphone::run_microphone_mode;
