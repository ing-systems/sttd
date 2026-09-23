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
// (`quant_linear::dynamic_quantize_u8` + i64 MAC + one f32 scale-up,
// bit-exact on a 1-op ORT probe to 1.5e-6), so chained encoder drift is
// f32 rounding only. Word-level parity against `--parakeet-engine ort`
// is demonstrated file-by-file (step 6); the port stays opt-in.
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
mod runtime;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod conformer_block;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod tdt_greedy;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod encoder;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod mel_frontend;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod int_gemm;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod joint;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod subsampler;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod quant_linear;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod tensor_ctx;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod vocab;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod weights;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod manifest;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod model;
#[cfg(feature = "parakeet-burn")]
#[allow(dead_code)]
mod ops;

#[cfg(feature = "parakeet-burn")]
pub(crate) use model::ParakeetBurnModel;
