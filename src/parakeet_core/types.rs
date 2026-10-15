// ---------------------------------------------------------------------------
// parakeet_core types: session handles, vocab, boost state, results.
// Ported from Murmure `src-tauri/src/engine/types.rs` (AGPL-3.0-or-later).
// Dropped: the `nemo128` preprocessor `Session` (see `frontend`), the
// token/word/segment timestamp hierarchy and `TimestampGranularity`
// (sttd needs text + confidences only), serde derives.
// ---------------------------------------------------------------------------

use ndarray17::Array3;
use ort::session::Session;
use tokenizers::Tokenizer;

use super::boost_tree::BoostTree;

pub type DecoderState = (Array3<f32>, Array3<f32>);

/// One decoded sequence: the joined text, and the vocab pieces and their
/// probabilities that per-word confidences are grouped from.
#[derive(Debug, Clone)]
pub struct DecodedTokens {
    pub text: String,
    pub tokens: Vec<String>,
    /// Softmax probability of each emitted token over the raw (unboosted)
    /// vocab logits, aligned with `tokens`.
    pub probs: Vec<f32>,
}

#[derive(thiserror::Error, Debug)]
pub enum ParakeetError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray17::ShapeError),
    #[error("audio frontend error: {0}")]
    Frontend(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
}

// `ort` 2.x types builder errors as `Error<SessionBuilder>`; the session
// methods used here (`run`, tensor conversions) report plain `Error`.
// Both funnel here so `?` works on either.
impl From<ort::Error<ort::session::builder::SessionBuilder>> for ParakeetError {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        Self::Ort(e.into())
    }
}

pub struct ParakeetModel {
    pub encoder: Session,
    pub decoder_joint: Session,
    pub vocab: Vec<String>,
    pub blank_idx: i32,
    pub tokenizer: Option<Tokenizer>,
    pub boost_tree: Option<BoostTree>,
    pub boost_alpha: f32,
    /// Sorted word list the current boost tree was built from, to skip
    /// rebuilding when the dictionary has not changed between transcriptions.
    pub boost_words: Vec<String>,
    pub frontend: super::frontend::Frontend,
}
