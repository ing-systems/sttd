// ---------------------------------------------------------------------------
// parakeet_core: Parakeet-TDT-0.6B-v3 inference over direct `ort` sessions.
//
// Vendored from Murmure (`src-tauri/src/engine/`, AGPL-3.0-or-later,
// Copyright al1x-ai.com contributors) with two changes:
// - the `nemo128.onnx` preprocessor session is gone; mel features come from
//   `frontend` (Rust STFT + Slaney mel, ported from `parakeet-rs` 0.3.7,
//   MIT OR Apache-2.0, Copyright altunenes) instead of ONNX;
// - `helpers::fold_accents` also folds Polish diacritics (see `gate`).
//
// Why not `parakeet-rs`: it hides the decoder behind `transcribe_samples`
// with no logit access, so neither dictionary phrase-boosting nor per-word
// confidences are possible. This module keeps Murmure's greedy TDT decode
// loop with its `BoostTree` logit bias and softmax confidences.
// ---------------------------------------------------------------------------

pub mod boost_tree;
pub mod engine;
pub mod frontend;
pub mod helpers;
pub mod transcription_engine;
pub mod types;

pub use types::ParakeetModel;
