// ---------------------------------------------------------------------------
// Speech-model seam: what an engine is, and how one is loaded
// ---------------------------------------------------------------------------
//
// Two speech models exist (Parakeet-TDT-0.6B-v3 and Whisper large-v3-turbo, via
// direct `ort` sessions, see `parakeet_core` / `whisper_core`), selected once at
// startup by `--model` (see `model_store::ModelRef`) and fixed for the process's
// lifetime. The engines' own files (`parakeet`, `whisper`) implement
// `SpeechModel` against this module only, and the worker that drives them
// (`asr`, which also picks the engine in `load_model`) depends on it too: this
// module depends on neither.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

/// Result of one inference: the joined text plus per-word model confidences
/// when the engine reports them (Parakeet, for the dictionary gate).
pub(crate) struct Transcript {
    pub(crate) text: String,
    pub(crate) word_confidences: Vec<(String, f32)>,
}

/// What an engine reads when it loads, resolved once from the flags on the main
/// thread. The rest of the worker's configuration (`asr::AsrConfig`) is none of
/// the engines' business.
#[derive(Debug, Clone)]
pub(crate) struct EngineConfig {
    /// `--model` as typed, for the engine's ready line.
    pub(crate) model_spec: String,
    pub(crate) n_threads: i32,
    pub(crate) use_gpu: bool,
    /// Decoding language for the Whisper engine (fixed once at startup). Parakeet
    /// auto-detects per utterance and ignores this.
    pub(crate) language: crate::language::Language,
    /// `--audio-ctx`: floor of the dynamic Whisper window, in mel frames. The
    /// encoder graph is patched at load, so the window follows the audio;
    /// Parakeet sizes its own window and ignores this.
    pub(crate) audio_ctx: usize,
}

/// One loaded, ready-to-run speech model. The worker loop in this module
/// only ever sees this trait.
pub(crate) trait SpeechModel {
    /// Run one inference over already-conditioned 16 kHz mono audio.
    fn transcribe(&mut self, pcm_16k: &[f32], running: &Arc<AtomicBool>) -> Result<Transcript>;

    /// Rebuild the engine's phrase-boosting automaton from a new dictionary
    /// list. Default is a no-op: an engine without boosting ignores it.
    /// Called on the worker thread, never from the D-Bus thread.
    fn set_boost_words(&mut self, _words: &[String]) {}

    /// First-load readiness log line (full config summary).
    fn describe(&self) -> String;

    /// Reload readiness log line, logged after an auto-unload/resume cycle
    /// instead of `describe()` (which would re-print the full config).
    fn describe_reload(&self) -> String;

    /// Short engine name for the shared load/unload log lines this module
    /// prints ("{name} model unloaded...", "reloading {name} model...").
    fn engine_name(&self) -> &'static str;

    /// Samples this engine actually decoded for a conditioned utterance of
    /// `conditioned_len` samples, accounting for any internal short-utterance
    /// padding, so RTF and the `--metrics` `audio_sec` column reflect what
    /// was fed to the model rather than the raw utterance length.
    fn decoded_sample_count(&self, conditioned_len: usize) -> usize;
}

/// Runs `f` while ticking `"loading {label}... Ns"` once per second on a
/// side thread. Model construction can block for tens of seconds with no
/// progress callback (a Parakeet fp32 encoder), so this exists to keep that
/// hung. Joins the ticker before returning `f`'s result.
pub(crate) fn with_loading_ticker<T>(label: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let load_done = Arc::new(AtomicBool::new(false));
    let ticker = {
        let done = Arc::clone(&load_done);
        let label = label.to_string();
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut last_printed: u64 = 0;
            loop {
                // Short sleeps so joining after the load finishes costs at
                // most 100 ms; only a whole-second tick is printed.
                std::thread::sleep(Duration::from_millis(100));
                if done.load(Ordering::Relaxed) {
                    break;
                }
                let elapsed = start.elapsed().as_secs();
                if elapsed > last_printed {
                    last_printed = elapsed;
                    log::info!("loading {label}... {elapsed}s");
                }
            }
        })
    };
    let result = f();
    load_done.store(true, Ordering::Relaxed);
    let _ = ticker.join();
    result
}

/// One deterministic dither stream: `count` samples at `+-1e-4` from an
/// xorshift32 generator (cheap, no extra dependency).
///
/// Both padding sites below share this stream, not their seed: the seed is
/// derived from the input length by each caller, so its padding stays stable
/// per input shape.
fn dither_samples(seed: u32, count: usize) -> impl Iterator<Item = f32> {
    let mut seed = seed;
    (0..count).map(move |_| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32 - 0.5) * 2e-4 // +-1e-4
    })
}

/// Pad sub-second audio with quiet dithered noise instead of exact digital
/// silence: a long run of true zero is a hallucination trigger for the mel
/// front-end, and a poor input for short-utterance handling in general.
/// Grows the input to `min_samples`; longer input is returned unchanged.
pub(crate) fn pad_with_dither(pcm_16k: &[f32], min_samples: usize) -> Vec<f32> {
    let missing = min_samples.saturating_sub(pcm_16k.len());
    let mut out = Vec::with_capacity(min_samples.max(pcm_16k.len()));
    out.extend_from_slice(pcm_16k);
    let seed = 0x9E37_79B9 ^ (pcm_16k.len() as u32).wrapping_add(1);
    out.extend(dither_samples(seed, missing));
    out
}

/// Append `pad_samples` of quiet dither after conditioned audio
/// (see `config::INFERENCE_PAD_SAMPLES` for why dither, not zero).
/// Empty input stays empty (callers skip empty utterances before here).
pub(crate) fn pad_trailing_with_dither(pcm_16k: &[f32], pad_samples: usize) -> Vec<f32> {
    if pcm_16k.is_empty() || pad_samples == 0 {
        return pcm_16k.to_vec();
    }
    let mut out = Vec::with_capacity(pcm_16k.len() + pad_samples);
    out.extend_from_slice(pcm_16k);
    let seed = 0x9E37_79B9 ^ (pcm_16k.len() as u32).wrapping_mul(0x85EB_CA6B);
    out.extend(dither_samples(seed, pad_samples));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_pad_appends_dither_and_preserves_start() {
        use crate::config::INFERENCE_PAD_SAMPLES;
        let pcm = vec![0.5f32; 1_600];
        let padded = pad_trailing_with_dither(&pcm, INFERENCE_PAD_SAMPLES);
        assert_eq!(padded.len(), pcm.len() + INFERENCE_PAD_SAMPLES);
        assert_eq!(&padded[..pcm.len()], &pcm[..]);
        assert!(padded[pcm.len()..].iter().all(|&s| s.abs() <= 1e-4));
        assert!(padded[pcm.len()..].iter().any(|&s| s != 0.0));
    }

    #[test]
    fn trailing_pad_empty_and_zero_are_noop() {
        assert!(pad_trailing_with_dither(&[], 4000).is_empty());
        let pcm = vec![0.1f32; 100];
        assert_eq!(pad_trailing_with_dither(&pcm, 0), pcm);
    }

    #[test]
    fn min_pad_grows_short_audio_only_and_stays_deterministic() {
        use crate::config::PARAKEET_MIN_SAMPLES;
        let short = vec![0.25f32; 1_600];
        let padded = pad_with_dither(&short, PARAKEET_MIN_SAMPLES);
        assert_eq!(padded.len(), PARAKEET_MIN_SAMPLES);
        assert_eq!(&padded[..short.len()], &short[..]);
        assert!(padded[short.len()..].iter().all(|&s| s.abs() <= 1e-4));
        assert!(padded[short.len()..].iter().any(|&s| s != 0.0));
        // Same input, same dither stream.
        assert_eq!(pad_with_dither(&short, PARAKEET_MIN_SAMPLES), padded);
        // Already long enough: untouched, no pad added.
        let long = vec![0.25f32; PARAKEET_MIN_SAMPLES + 10];
        assert_eq!(pad_with_dither(&long, PARAKEET_MIN_SAMPLES), long);
    }
}
