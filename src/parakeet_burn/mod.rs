// ---------------------------------------------------------------------------
// Parakeet Burn engine: native int8 Parakeet-TDT without ONNX Runtime.
// ---------------------------------------------------------------------------
//
// `parakeet-rs` (ORT) stays the default engine (see `parakeet.rs`).
// This module is only built with the `parakeet-burn` Cargo feature and
// selected at runtime via `--parakeet-engine burn`.
//
// Status: end-to-end transcription runs (frontend -> `BurnEncoder` ->
// `BurnJoint::decode` -> `BurnVocab::rebuild`), ORT-verified per module
// in isolation (see `docs/parakeet-burn.md`). The forward mirrors the
// int8 graph's dynamic activation quantization in integer arithmetic
// (`quant::dynamic_quantize_u8` + i64 MAC + one f32 scale-up, bit-exact
// on a 1-op ORT probe to 1.5e-6), so chained encoder drift is f32
// rounding only. Word-level parity against `--parakeet-engine ort` is
// demonstrated file-by-file (step 6); the port stays opt-in.
// `parakeet-rs` (ORT) stays the default engine.
//
// Weight layout (see `tools/convert-parakeet-int8-to-burn.py`):
// `models/parakeet-tdt-0.6b-v3-int8/encoder-burn.safetensors` +
// `decoder_joint-burn.safetensors` + existing `vocab.txt`.

// Bring-up: all submodules are wired into `ParakeetBurnModel` now;
// dead-code lints stay off until word-level parity against ORT (step 6)
// decides whether the port can become anything but opt-in.
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod backend;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod block;
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
mod pre_encode;
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
    pub(crate) struct ParakeetBurnModel {
        vocab: super::vocab::BurnVocab,
        frontend: super::frontend::MelFrontend,
        encoder: super::encoder::BurnEncoder,
        joint: super::joint::BurnJoint,
        ready_line: String,
    }

    impl ParakeetBurnModel {
        pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
            let intra_threads = opts.n_threads.max(1) as usize;
            // Honor `--threads` (resolved: 0 = physical cores, min 1):
            // size rayon's process-wide pool once. See
            // `backend::cap_thread_pool` for why this is a verified
            // no-op today (nothing in the Burn path uses the pool yet)
            // that pins the size for the tensor-ops swap.
            super::backend::cap_thread_pool(intra_threads);
            let device = super::backend::select_device(opts.use_gpu);
            #[cfg(not(feature = "parakeet-burn-cuda"))]
            if opts.use_gpu {
                log::warn!(
                    "--gpu is set but this build has no `parakeet-burn-cuda` feature; \
                     Burn Parakeet runs on CPU (rebuild with `--features parakeet-burn-cuda` for GPU)"
                );
            }
            #[cfg(feature = "parakeet-burn-cuda")]
            if device == super::backend::BurnDevice::Cuda {
                log::info!("burn parakeet: CUDA backend selected (threads={intra_threads})");
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

            let (encoder, joint) = with_loading_ticker("burn parakeet weights", || {
                let manifest_json = std::fs::read_to_string(dir.join("encoder-manifest.json"))
                    .with_context(|| {
                        format!(
                            "burn parakeet: missing encoder-manifest.json in {}",
                            dir.display()
                        )
                    })?;
                let manifest = super::encoder::load_manifest(&manifest_json)?;
                let enc_want = super::encoder::encoder_want(&manifest)?;
                let enc_tensors = super::weights::load_selected(
                    &dir.join(super::weights::ENCODER_BURN_FILE),
                    &enc_want,
                )?;
                let encoder = super::encoder::BurnEncoder::load(&enc_tensors, &manifest)?;
                let joint_tensors = super::weights::load_selected(
                    &dir.join(super::weights::JOINT_BURN_FILE),
                    &super::joint::joint_want(),
                )?;
                let joint = super::joint::BurnJoint::load(&joint_tensors, vocab.size())?;
                Ok::<_, anyhow::Error>((encoder, joint))
            })?;

            Ok(Self {
                vocab,
                frontend,
                encoder,
                joint,
                ready_line: format!(
                    "burn parakeet ready (model={}, threads={intra_threads}, device={device:?})",
                    opts.model_spec
                ),
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
            // Full native path: mel -> encoder states -> TDT greedy
            // decode -> text rebuild. Like the ORT engine, greedy decoding
            // reports no per-token probability (`confidence: None`).
            let mel = self.frontend.extract(pcm)?;
            let enc = self.encoder.forward(&mel)?;
            let (tokens, _frames, _durations) = self.joint.decode(&enc);
            Ok(Transcript {
                text: self.vocab.rebuild(&tokens),
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
            "burn-parakeet"
        }

        fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
            conditioned_len.max(PARAKEET_MIN_SAMPLES)
        }
    }
}
