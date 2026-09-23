// ---------------------------------------------------------------------------
// ort_engine transcription result: text plus per-word confidences.
// Adapted from Murmure `src-tauri/src/engine/transcription_engine.rs`
// (AGPL-3.0-or-later): segments dropped, confidences kept — they gate
// sttd's dictionary fuzzy correction (Phase B).
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TranscriptionResult {
    pub text: String,
    /// Per-word model confidence (min softmax prob over the word's tokens),
    /// in text order. Used to gate the dictionary fuzzy post-correction.
    pub word_confidences: Vec<(String, f32)>,
}
