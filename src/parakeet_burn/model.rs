// Native Burn Parakeet model wiring: vocab + frontend + encoder + joint.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, Transcript};
use crate::config::PARAKEET_MIN_SAMPLES;

/// Native Burn Parakeet model (see module docs for status).
pub(crate) struct ParakeetBurnModel {
    vocab: super::vocab::BurnVocab,
    frontend: super::mel_frontend::MelFrontend,
    encoder: super::encoder::BurnEncoder,
    joint: super::joint::BurnJoint,
    ready_line: String,
    tensor: bool,
    device: super::runtime::CpuDevice,
}

impl ParakeetBurnModel {
    pub(crate) fn load(dir: &Path, opts: &AsrOptions) -> Result<Self> {
        let intra_threads = opts.n_threads.max(1) as usize;
        // Honor `--threads` (resolved: 0 = physical cores, min 1):
        // size rayon's process-wide pool once. See
        // `runtime::size_thread_pool` for why this is a verified
        // no-op today (nothing in the Burn path uses the pool yet)
        // that pins the size for the tensor-ops swap.
        super::runtime::size_thread_pool(intra_threads);
        let device = super::runtime::select_device(opts.use_gpu);
        #[cfg(not(feature = "parakeet-burn-cuda"))]
        if opts.use_gpu {
            log::warn!(
                "--gpu is set but this build has no `parakeet-burn-cuda` feature; \
                 Burn Parakeet runs on CPU (rebuild with `--features parakeet-burn-cuda` for GPU)"
            );
        }
        #[cfg(feature = "parakeet-burn-cuda")]
        if device == super::runtime::BurnDevice::Cuda {
            log::info!("burn parakeet: CUDA backend selected (threads={intra_threads})");
        }

        let (vocab, frontend) = with_loading_ticker("burn parakeet frontend", || {
            let vocab = super::vocab::BurnVocab::from_file(&dir.join("vocab.txt"))
                .with_context(|| format!("burn parakeet: missing vocab.txt in {}", dir.display()))?;
            Ok::<_, anyhow::Error>((vocab, super::mel_frontend::MelFrontend::tdt_v3()))
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
            let manifest = super::manifest::load_manifest(&manifest_json)?;
            let enc_want = super::encoder::required_tensors(&manifest)?;
            let enc_tensors = super::weights::load_selected(
                &dir.join(super::weights::ENCODER_BURN_FILE),
                &enc_want,
            )?;
            let encoder = super::encoder::BurnEncoder::load(&enc_tensors, &manifest)?;
            let joint_tensors = super::weights::load_selected(
                &dir.join(super::weights::JOINT_BURN_FILE),
                &super::joint::required_tensors(),
            )?;
            let joint = super::joint::BurnJoint::load(&joint_tensors, vocab.size())?;
            // Warm the tensor-path caches now (dequant + `from_data`
            // once per weight, in parallel): without this the first
            // utterance pays the full build (~4 s stall). Inside the
            // loading ticker so mic/file startup shows progress.
            let cpu = super::runtime::cpu_device();
            encoder.prime_cache(&cpu);
            joint.prime_cache(&cpu);
            Ok::<_, anyhow::Error>((encoder, joint))
        })?;

        // Tensor-path selector, read once (never in tests: env vars
        // are process-global). Tensor is the default since Phase 7
        // (goldens + file-mode proven); STTD_BURN_TENSOR=0 keeps the
        // scalar fallback for debugging regressions.
        let tensor = super::runtime::tensor_path_enabled();
        let cpu = super::runtime::cpu_device();
        if !tensor {
            log::info!("burn parakeet: scalar path forced (STTD_BURN_TENSOR=0)");
        }
        Ok(Self {
            vocab,
            frontend,
            encoder,
            joint,
            ready_line: format!(
                "burn parakeet ready (model={}, threads={intra_threads}, device={device:?}, tensor={tensor})",
                opts.model_spec
            ),
            tensor,
            device: cpu,
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
        // decode -> text. Like the ORT engine, greedy decoding
        // reports no per-token probability (`confidence: None`).
        // `ctx` flips every linear between its scalar twin (exact
        // integer path) and the Flex matmul twin at once.
        let ctx = super::tensor_ctx::TensorCtx {
            device: self.device,
            enabled: self.tensor,
        };
        let mel = self.frontend.extract(pcm)?;
        let enc = self.encoder.forward(&mel, &ctx)?;
        let (tokens, _frames, _durations) = self.joint.decode(&enc, &ctx);
        Ok(Transcript {
            text: self.vocab.decode_to_text(&tokens),
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
