//! Defaults and stand-ins the worker, pipeline and live-state tests share.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use super::*;
use crate::config::{DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS, DEFAULT_THREADS};
use crate::gate::GateConfig;
use crate::speech::{EngineConfig, SpeechModel, Transcript};
use crate::utterance::{Evidence, Utterance};

/// A live dictionary with no words, for tests that never swap one.
pub(super) fn empty_live_dictionary() -> LiveDictionary {
    LiveDictionary::new(crate::control::DictionaryState::new(
        Vec::new(),
        std::path::PathBuf::new(),
    ))
    .1
}

/// A live dictionary seeded with `words`, plus the `control::DictionaryState`
/// a test would hand to `dbus::spawn`, so a test can drive a swap and see
/// what the worker would apply.
pub(super) fn live_dictionary_with(
    words: &[&str],
) -> (crate::control::DictionaryState, LiveDictionary) {
    LiveDictionary::new(crate::control::DictionaryState::new(
        words.iter().map(|w| w.to_string()).collect(),
        std::path::PathBuf::new(),
    ))
}

pub(super) fn default_gate() -> GateConfig {
    GateConfig::new(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY)
}

/// A worker's configuration with the settings tests want: no shedding, no
/// metrics file, no dictionary, no sound, default thresholds. Declared here
/// rather than at module level because it draws on `default_gate` and
/// `empty_live_dictionary`, which are both in this module.
///
/// This supplies defaults; it does not wire. The wiring still crosses one
/// seam, `AsrWorker::new`.
pub(super) fn for_tests() -> AsrConfig {
    AsrConfig {
        engine: EngineConfig {
            model_spec: "scripted".to_string(),
            n_threads: DEFAULT_THREADS,
            use_gpu: false,
            language: crate::language::Language::Auto,
            audio_ctx: crate::whisper_core::types::DEFAULT_WINDOW_FRAMES,
        },
        gate: default_gate(),
        remove_hesitations: false,
        dictionary: empty_live_dictionary(),
        shed_stale: false,
        partials: PartialSlot::default(),
        auto_unload: false,
        auto_unload_after: Duration::from_secs(0),
        sound: false,
        autopause: false,
    }
}

/// A model that returns a fixed transcript and remembers how long each
/// input was, in call order.
pub(super) struct ScriptedModel {
    pub(super) text: String,
    pub(super) inputs: Arc<Mutex<Vec<usize>>>,
    pub(super) fail: bool,
}

impl ScriptedModel {
    pub(super) fn new(text: &str) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let inputs = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                text: text.to_string(),
                inputs: Arc::clone(&inputs),
                fail: false,
            },
            inputs,
        )
    }
}

impl SpeechModel for ScriptedModel {
    fn transcribe(&mut self, pcm_16k: &[f32], _running: &Arc<AtomicBool>) -> Result<Transcript> {
        self.inputs.lock().unwrap().push(pcm_16k.len());
        if self.fail {
            anyhow::bail!("scripted failure");
        }
        Ok(Transcript {
            text: self.text.clone(),
            word_confidences: Vec::new(),
        })
    }

    fn describe(&self) -> String {
        "scripted".into()
    }

    fn describe_reload(&self) -> String {
        "scripted reloaded".into()
    }

    fn engine_name(&self) -> &'static str {
        "scripted"
    }

    fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
        conditioned_len
    }
}

pub(super) fn closed_utterance(samples: usize) -> Utterance {
    Utterance {
        pcm: vec![0.1; samples],
        evidence: Evidence {
            speech_frames: 50,
            total_frames: 60,
            rms: 0.05,
        },
        split_continuation: false,
        queued_at: Instant::now(),
        pause_flush: false,
    }
}

/// Same, but as the pause edge or a stale backlog would leave it: tagged
/// as a flush, or queued long enough ago to count as stale.
pub(super) fn queued_utterance(samples: usize, pause_flush: bool, waited: Duration) -> Utterance {
    Utterance {
        queued_at: Instant::now()
            .checked_sub(waited)
            .expect("host uptime must exceed the waited window"),
        pause_flush,
        ..closed_utterance(samples)
    }
}
