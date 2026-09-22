// ---------------------------------------------------------------------------
// Parakeet engine: the NVIDIA Parakeet-TDT-0.6B-v3 `SpeechModel` impl.
// ---------------------------------------------------------------------------
//
// Wraps `parakeet_rs::ParakeetTDT` (FastConformer-TDT, 0.6B params, 25
// European languages with per-utterance auto-detection). Unlike whisper.cpp,
// decoding is greedy and reports no per-token probability, so `transcribe`
// always returns `confidence: None` (see `asr::Transcript`) — the transcript
// gate's confidence-floor arm is a no-op for this engine, and its whisper-only
// tunables (`--beam-size`, `--temperature*`, `--prompt`, `--translate`, ...)
// are rejected or warned about in `main::validate_parakeet_model_flags`
// before the worker ever loads this.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber as _};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
use crate::config::PARAKEET_MIN_SAMPLES;

#[cfg(feature = "parakeet-cuda")]
use parakeet_rs::ExecutionProvider;

pub(crate) struct ParakeetModel {
    inner: ParakeetTDT,
    ready_line: String,
}

impl ParakeetModel {
    /// Load a Parakeet model from `dir` (already guaranteed to hold an
    /// encoder, a decoder_joint and `vocab.txt` by
    /// `weights::ensure_available`). `--gpu` selects CUDA only when this
    /// binary was built with the `parakeet-cuda` feature; otherwise it warns
    /// once and runs on CPU (see `main::unsupported_parakeet_flags`, which
    /// warns about this same case before the worker gets here).
    pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
        let intra_threads = opts.n_threads.max(1) as usize;

        #[cfg(feature = "parakeet-cuda")]
        if opts.use_gpu {
            let cuda_config = ExecutionConfig::new()
                .with_execution_provider(ExecutionProvider::Cuda)
                .with_intra_threads(intra_threads)
                .with_inter_threads(1);
            match Self::load_with_config(dir, cuda_config) {
                Ok(inner) => {
                    return Ok(Self {
                        inner,
                        ready_line: format!(
                            "parakeet ready (model={}, threads={}, gpu=true)",
                            opts.model_spec, opts.n_threads
                        ),
                    });
                }
                Err(e) => {
                    log::warn!("parakeet CUDA load failed ({e:#}), falling back to CPU");
                }
            }
        }
        #[cfg(not(feature = "parakeet-cuda"))]
        if opts.use_gpu {
            log::warn!(
                "--gpu is set but this build has no `parakeet-cuda` feature; Parakeet runs \
                 on CPU (rebuild with `--features parakeet-cuda` for GPU)"
            );
        }

        let cpu_config = ExecutionConfig::new()
            .with_intra_threads(intra_threads)
            .with_inter_threads(1);
        let inner = Self::load_with_config(dir, cpu_config)?;
        Ok(Self {
            inner,
            ready_line: format!(
                "parakeet ready (model={}, threads={}, gpu=false)",
                opts.model_spec, opts.n_threads
            ),
        })
    }

    fn load_with_config(dir: &Path, exec: ExecutionConfig) -> Result<ParakeetTDT> {
        log::info!("loading parakeet model ({})...", dir.display());
        with_loading_ticker("parakeet model", || {
            ParakeetTDT::from_pretrained(dir, Some(exec))
                .map_err(|e| anyhow::anyhow!("failed to load parakeet model: {e}"))
                .context("failed to load parakeet model")
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
        // `ort`/parakeet-rs has no abort callback, unlike whisper.cpp: a
        // single utterance's inference is hundreds of ms, not minutes, so a
        // pre-call check (rather than aborting mid-inference) is enough to
        // avoid running needless work after shutdown was requested.
        if !running.load(Ordering::Relaxed) {
            anyhow::bail!("shutdown requested");
        }
        // Encoder does 8x subsampling; a very short clip can yield too few
        // frames for the TDT greedy decoder to emit anything (see
        // PARAKEET_MIN_SAMPLES's doc comment). Reuses whisper's dither-pad
        // helper — the technique isn't whisper-specific, only its usual home
        // is.
        let padded;
        let pcm = if pcm_16k.len() < PARAKEET_MIN_SAMPLES {
            padded = crate::whisper::pad_with_dither(pcm_16k, PARAKEET_MIN_SAMPLES);
            &padded[..]
        } else {
            pcm_16k
        };
        let result = self
            .inner
            .transcribe_samples(pcm.to_vec(), 16_000, 1, Some(TimestampMode::Sentences))
            .map_err(|e| anyhow::anyhow!("parakeet inference failed: {e}"))?;
        Ok(Transcript {
            text: result.text,
            confidence: None,
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
