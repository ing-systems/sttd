// ---------------------------------------------------------------------------
// Parakeet engine: NVIDIA Parakeet-TDT-0.6B-v3 `SpeechModel` impl.
//
// Wraps `parakeet_core::ParakeetModel` (FastConformer-TDT, 0.6B params, 25
// European languages with per-utterance auto-detection): direct `ort`
// sessions with dictionary phrase-boosting on the greedy TDT decode (see
// `parakeet_core`). Unlike `parakeet-rs` (removed), this engine reports
// per-word confidences, so the dictionary fuzzy correction runs gated
// (Phase B) instead of ungated.
//
// With `--gpu` (and a build that has CUDA support) the encoder session runs on
// the CUDA execution provider. The decoder_joint always stays on CPU. When
// CUDA is unavailable, or the CUDA load fails, the encoder falls back to CPU and
// the fallback is logged.
// ---------------------------------------------------------------------------

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
use crate::config::PARAKEET_MIN_SAMPLES;
use crate::parakeet_core::ParakeetModel as CoreModel;

pub(crate) struct ParakeetModel {
    inner: CoreModel,
    ready_line: String,
}

impl ParakeetModel {
    /// Load a Parakeet model from `dir` (already guaranteed to hold an
    /// encoder, a decoder_joint and `vocab.txt` by
    /// `model_store::ensure_available`). `tokenizer.json` in the same directory
    /// enables dictionary phrase-boosting; without it transcription works
    /// but boosting stays off (warned once). The live dictionary is synced
    /// here (see `SpeechModel::set_boost_words`), so a swap that landed while
    /// a previous instance was unloaded is already in place.
    pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
        let device = crate::device::requested(opts.use_gpu);
        if opts.use_gpu && matches!(device, crate::device::Device::Cpu) {
            log::warn!("--gpu is set but this build has no CUDA support (build with --features cuda) or CUDA is unavailable; running on CPU");
        }
        let intra_threads = opts.n_threads.max(1) as usize;
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_opt = tokenizer_path.exists().then_some(tokenizer_path);
        if tokenizer_opt.is_none() {
            log::warn!(
                "tokenizer.json not found in {}; phrase boosting disabled (transcription unaffected)",
                dir.display()
            );
        }
        log::info!("loading parakeet model ({})...", dir.display());
        let mut inner = with_loading_ticker("parakeet model", || {
            CoreModel::new(dir, true, tokenizer_opt.as_deref(), intra_threads, device)
                .map_err(|e| anyhow::anyhow!("failed to load parakeet model: {e}"))
                .context("failed to load parakeet model")
        })?;
        // Snapshot, not a held lock: `opts.dictionary` is the live handle.
        let startup_words = opts.dictionary.words();
        inner.set_boost_words(&startup_words);
        Ok(Self {
            inner,
            ready_line: format!(
                "parakeet ready (model={}, threads={}, gpu={})",
                opts.model_spec,
                opts.n_threads,
                matches!(device, crate::device::Device::Cuda)
            ),
        })
    }
}

impl SpeechModel for ParakeetModel {
    fn transcribe(&mut self, pcm_16k: &[f32], running: &Arc<AtomicBool>) -> Result<Transcript> {
        // `ort` has no abort callback: a single utterance's inference is
        // hundreds of ms, not minutes, so a pre-call check (rather than
        // aborting mid-inference) is enough to avoid running needless work
        // after shutdown was requested.
        if !running.load(Ordering::Relaxed) {
            anyhow::bail!("shutdown requested");
        }
        // Encoder does 8x subsampling; a very short clip can yield too few
        // frames for the TDT greedy decoder to emit anything (see
        // PARAKEET_MIN_SAMPLES's doc comment).
        let padded;
        let pcm = if pcm_16k.len() < PARAKEET_MIN_SAMPLES {
            padded = crate::asr::pad_with_dither(pcm_16k, PARAKEET_MIN_SAMPLES);
            &padded[..]
        } else {
            pcm_16k
        };
        let result = self
            .inner
            .transcribe_samples(pcm)
            .map_err(|e| anyhow::anyhow!("parakeet inference failed: {e}"))?;
        Ok(Transcript {
            text: result.text,
            // Per-word confidences gate the dictionary correction.
            word_confidences: result.word_confidences,
        })
    }

    /// Rebuild the phrase-boosting automaton after a live dictionary swap
    /// (`ReplaceWords` / `LoadFile`). Called on the worker thread only,
    /// from `AsrWorker::apply_boost_words`. A no-op when the word set is
    /// unchanged, so the common case costs nothing.
    fn set_boost_words(&mut self, words: &[String]) {
        self.inner.set_boost_words(words);
    }

    fn describe(&self) -> String {
        self.ready_line.clone()
    }

    fn describe_reload(&self) -> String {
        format!("{} (reloaded after pause)", self.ready_line)
    }

    fn engine_name(&self) -> &'static str {
        "parakeet"
    }

    fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
        conditioned_len.max(PARAKEET_MIN_SAMPLES)
    }
}
