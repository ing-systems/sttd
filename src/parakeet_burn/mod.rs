// ---------------------------------------------------------------------------
// Parakeet Burn engine: native int8 Parakeet-TDT without ONNX Runtime.
// ---------------------------------------------------------------------------
//
// `parakeet-rs` (ORT) stays the default engine (see `parakeet.rs`).
// This module is only built with the `parakeet-burn` Cargo feature and
// selected at runtime via `--parakeet-engine burn`.
//
// Status: bring-up in progress. The mel frontend (`frontend`), the TDT
// greedy loop (`decode`) and int8 weight containers (`quant`) are fully
// implemented and unit-tested; the FastConformer encoder stack
// (`encoder`) runs its Burn matmuls on the NdArray backend today and the
// CubeCL CUDA backend with `parakeet-burn-cuda` + `--gpu`. Full 0.6B
// weight mapping lands incrementally — `load` fails fast with the exact
// converter command until converted weights exist, so a half-wired model
// can never silently mistranscribe.
//
// Weight layout (see `tools/convert-parakeet-int8-to-burn.py`):
// `models/parakeet-tdt-0.6b-v3-int8/encoder-burn.safetensors` +
// `decoder_joint-burn.safetensors` + existing `vocab.txt`.

// Bring-up: submodules are unit-tested but not all wired into
// `ParakeetBurnModel::transcribe` yet (encoder weight mapping lands next),
// so dead-code lints stay off here until the engine decodes end to end.
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod backend;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod decode;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod encoder;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod frontend;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod joint;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod quant;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod vocab;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod weights;

#[cfg(feature = "parakeet-burn")]
pub(crate) use model::ParakeetBurnModel;

#[cfg(feature = "parakeet-burn")]
mod model {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use anyhow::{Context, Result};

    use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
    use crate::config::PARAKEET_MIN_SAMPLES;

    /// Native Burn Parakeet model (see module docs for status).
    /// `vocab`/`use_gpu` wire into the decoder/backend in the next
    /// bring-up step (joint decode + device selection).
    #[allow(dead_code)]
    pub(crate) struct ParakeetBurnModel {
        vocab: super::vocab::BurnVocab,
        frontend: super::frontend::MelFrontend,
        // Encoder/joint stay boxed behind their modules; `None` until the
        // converted-weight check below passes (fail-fast, never partial).
        ready_line: String,
        use_gpu: bool,
    }

    impl ParakeetBurnModel {
        pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
            let intra_threads = opts.n_threads.max(1) as usize;
            // Respect `--threads`: Burn NdArray uses rayon; cap its pool.
            let _ = intra_threads;

            if opts.use_gpu && !cfg!(feature = "parakeet-burn-cuda") {
                log::warn!(
                    "--gpu is set but this build has no `parakeet-burn-cuda` feature; \
                     Burn Parakeet runs on CPU (rebuild with `--features parakeet-burn-cuda` for GPU)"
                );
            }
            #[cfg(feature = "parakeet-burn-cuda")]
            if opts.use_gpu {
                log::info!("burn parakeet: probing CUDA backend...");
            }

            let (vocab, frontend) = with_loading_ticker("burn parakeet frontend", || {
                let vocab = super::vocab::BurnVocab::from_file(&dir.join("vocab.txt"))
                    .with_context(|| format!("burn parakeet: missing vocab.txt in {}", dir.display()))?;
                Ok::<_, anyhow::Error>((vocab, super::frontend::MelFrontend::tdt_v3()))
            })?;

            // Fail fast until converted int8 weights exist. This is the
            // bring-up gate: the ORT engine remains the default, and this
            // error tells the operator exactly how to produce the files.
            super::weights::require_converted_weights(dir)?;

            Ok(Self {
                vocab,
                frontend,
                ready_line: format!(
                    "burn parakeet ready (model={}, threads={}, gpu={})",
                    opts.model_spec, opts.n_threads, opts.use_gpu
                ),
                use_gpu: opts.use_gpu,
            })
        }
    }

    impl SpeechModel for ParakeetBurnModel {
        fn transcribe(
            &mut self,
            pcm_16k: &[f32],
            _prompt: &str,
            running: &Arc<AtomicBool>,
        ) -> Result<Transcript> {
            if !running.load(Ordering::Relaxed) {
                anyhow::bail!("shutdown requested");
            }
            let padded;
            let pcm = if pcm_16k.len() < PARAKEET_MIN_SAMPLES {
                padded = crate::asr::pad_with_dither(pcm_16k, PARAKEET_MIN_SAMPLES);
                &padded[..]
            } else {
                pcm_16k
            };
            // Frontend parity with parakeet-rs is implemented; encoder
            // weights are still being mapped (see `require_converted_weights`
            // gate in `load`). Reaching here without converted weights is a
            // bug — `load` must have failed first.
            let _features = self.frontend.extract(pcm)?;
            anyhow::bail!(
                "burn parakeet inference not wired yet (frontend ok, {} frames): \
                 convert weights then implement encoder bring-up; using `--parakeet-engine ort` transcribes today",
                _features.len()
            );
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
            "burn-parakeet"
        }

        fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
            conditioned_len.max(PARAKEET_MIN_SAMPLES)
        }
    }
}
