use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

use crate::config::{
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MODEL_PATH, DEFAULT_TARGET_RMS,
};

pub(crate) fn print_usage() {
    eprintln!(
        "sttd - real-time Polish speech-to-text\n\n\
         USAGE:\n\
         \x20   sttd                  listen on the default (or STTD_DEVICE) mic, print transcripts to stdout\n\
         \x20   sttd --file <path>    transcribe a WAV file offline (for tuning/comparison)\n\
         \x20   sttd --list-devices   list available input devices and exit\n\n\
         ENV:\n\
         \x20   WHISPER_MODEL_PATH    path to a ggml model (default: {DEFAULT_MODEL_PATH})\n\
         \x20   STTD_DEVICE           substring match (case-insensitive) against an input device name\n\
         \x20   STTD_PROMPT           override the whisper initial_prompt\n\
         \x20   STTD_DUMP_DIR         write each utterance as WAV+TXT into this directory (TXT written\n\
         \x20                         for dropped utterances too, prefixed with the drop reason)\n\
         \x20   STTD_MIN_RMS          min speech-only RMS to keep an utterance (default: {DEFAULT_MIN_SPEECH_RMS})\n\
         \x20   STTD_MIN_CONF         min avg token confidence, weak acoustic evidence (default: {DEFAULT_MIN_CONFIDENCE})\n\
         \x20   STTD_MIN_CONF_STRONG  min avg token confidence, strong acoustic evidence (default: {DEFAULT_MIN_CONFIDENCE_STRONG})\n\
         \x20   STTD_TARGET_RMS       pre-inference gain target RMS, 0 disables (default: {DEFAULT_TARGET_RMS})\n\
         \x20   STTD_DEBUG_TOKENS     1 = dump per-token id/text/p/plog/is_timestamp to stderr\n"
    );
}

pub(crate) fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());
        match device.default_input_config() {
            Ok(cfg) => println!("{name}  ({} Hz, {} ch)", cfg.sample_rate().0, cfg.channels()),
            Err(_) => println!("{name}  (config unavailable)"),
        }
    }
    Ok(())
}

/// Pick the input device: `STTD_DEVICE` (substring match, case-insensitive)
/// if set, otherwise the system default. Without this the tool is stuck on
/// whatever the OS considers the default mic (e.g. a webcam capsule).
pub(crate) fn select_input_device(host: &cpal::Host) -> Result<cpal::Device> {
    if let Ok(wanted) = std::env::var("STTD_DEVICE") {
        let wanted_lower = wanted.to_lowercase();
        let mut devices = host
            .input_devices()
            .context("failed to enumerate input devices")?;
        let found = devices.find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&wanted_lower))
                .unwrap_or(false)
        });
        return found
            .with_context(|| format!("STTD_DEVICE={wanted:?} matched no input device (see --list-devices)"));
    }
    host.default_input_device()
        .context("no default input device available")
}
