// ---------------------------------------------------------------------------
// whisper_core: Whisper large-v3-turbo inference over direct `ort` sessions.
// ---------------------------------------------------------------------------
//
// The encoder-decoder counterpart to `parakeet_core`, with the same shape:
// Rust mel frontend (no ONNX preprocessor session), direct `ort` sessions,
// greedy decode. Where it differs, the ONNX export dictates it:
//
// - three graphs, not two: `decoder_model` runs the prompt and produces the
//   cross-attention KV cache, then `decoder_with_past_model` runs every
//   following token off that cache. The single-graph `decoder_model_merged`
//   would re-run cross-attention over all 1500 encoder frames per token.
// - the encoder takes a *fixed* 30 s window (`input_features` axes 1 and 2
//   are constants in the graph, not symbolic), so an utterance's length never
//   changes the encoder cost.
// - no phrase boosting. Parakeet's logit-level dictionary boost tree has no
//   counterpart here; the post-gate dictionary correction in `dictionary`
//   still applies, and per-word confidences still gate it.
// ---------------------------------------------------------------------------

pub mod engine;
pub mod frontend;
pub mod types;

pub use types::WhisperModel;
