// ---------------------------------------------------------------------------
// Parakeet engine: NVIDIA Parakeet-TDT-0.6B-v3 `SpeechModel` impl.
//
// Wraps `ort_engine::ParakeetModel` (FastConformer-TDT, 0.6B params, 25
// European languages with per-utterance auto-detection): direct `ort`
// sessions with dictionary phrase-boosting on the greedy TDT decode (see
// `ort_engine`). Unlike `parakeet-rs` (removed), this engine reports
// per-word confidences, so the dictionary fuzzy correction runs gated
// (Phase B) instead of ungated.
//
// Runs on CPU. `--gpu` warns and falls back to CPU: the prebuilt ORT CUDA
// EP targets older CUDA/cuDNN than the CUDA 13.4/sm_120 toolchain
// whisper.cpp is built against here.
// ---------------------------------------------------------------------------

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
use crate::config::PARAKEET_MIN_SAMPLES;
use crate::ort_engine::ParakeetModel as OrtModel;

pub(crate) struct ParakeetModel {
    inner: OrtModel,
    ready_line: String,
}

impl ParakeetModel {
    /// Load a Parakeet model from `dir` (already guaranteed to hold an
    /// encoder, a decoder_joint and `vocab.txt` by
    /// `weights::ensure_available`). `tokenizer.json` in the same directory
    /// enables dictionary phrase-boosting; without it transcription works
    /// but boosting stays off (warned once). The startup dictionary is
    /// synced here — it is fixed for the process lifetime, and
    /// `set_boost_words` is a no-op when unchanged.
    pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
        if opts.use_gpu {
            log::warn!("--gpu is set but Parakeet runs on CPU in this build; running on CPU");
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
            OrtModel::new(dir, true, tokenizer_opt.as_deref(), intra_threads)
                .map_err(|e| anyhow::anyhow!("failed to load parakeet model: {e}"))
                .context("failed to load parakeet model")
        })?;
        inner.set_boost_words(&opts.dictionary);
        Ok(Self {
            inner,
            ready_line: format!(
                "parakeet ready (model={}, threads={}, gpu=false)",
                opts.model_spec, opts.n_threads
            ),
        })
    }
}

impl SpeechModel for ParakeetModel {
    fn transcribe(
        &mut self,
        pcm_16k: &[f32],
        _prompt: &str,
        running: &Arc<AtomicBool>,
    ) -> Result<Transcript> {
        // `ort` has no abort callback, unlike whisper.cpp: a single
        // utterance's inference is hundreds of ms, not minutes, so a
        // pre-call check (rather than aborting mid-inference) is enough to
        // avoid running needless work after shutdown was requested.
        if !running.load(Ordering::Relaxed) {
            anyhow::bail!("shutdown requested");
        }
        // Encoder does 8x subsampling; a very short clip can yield too few
        // frames for the TDT greedy decoder to emit anything (see
        // PARAKEET_MIN_SAMPLES's doc comment). Reuses the shared dither-pad
        // helper — the technique isn't whisper-specific.
        let padded;
        let pcm = if pcm_16k.len() < PARAKEET_MIN_SAMPLES {
            padded = crate::asr::pad_with_dither(pcm_16k, PARAKEET_MIN_SAMPLES);
            &padded[..]
        } else {
            pcm_16k
        };
        let result = self
            .inner
            .transcribe_samples(pcm.to_vec())
            .map_err(|e| anyhow::anyhow!("parakeet inference failed: {e}"))?;
        Ok(Transcript {
            text: result.text,
            // Greedy TDT decode reports no utterance-level average
            // probability, so the transcript gate's confidence-floor arm
            // stays a no-op (as with `parakeet-rs` before); per-word
            // confidences below still gate the dictionary correction.
            confidence: None,
            word_confidences: result.word_confidences,
        })
    }

    fn uses_prompt(&self) -> bool {
        false
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
