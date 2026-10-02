// ---------------------------------------------------------------------------
// parakeet_core helpers: accent fold, boost word variants, BPE tokenization,
// per-word confidences. Ported from Murmure
// `src-tauri/src/engine/helpers.rs` (AGPL-3.0-or-later); timestamp builders
// dropped (sttd needs text + confidences only).
//
// Difference: `fold_accents` also folds Polish diacritics via
// `gate::fold_polish_diacritic`, so dictionary variants cover Polish
// spellings (`zolc` for `żółć`), not just French.
// ---------------------------------------------------------------------------

use std::borrow::Cow;

use tokenizers::Tokenizer;

use super::types::TimestampedResult;

/// Replace accented Latin letters with their ASCII base. Murmure's
/// French/Latin coverage plus Polish (ą ć ę ł ń ó ś ź ż) via `gate`, so
/// dictionary variants cover Polish spellings (`zolc` for `żółć`).
pub fn fold_accents(text: &str) -> String {
    text.chars()
        .map(|c| {
            // Polish fold first (identity for non-Polish chars).
            let c = crate::gate::fold_polish_diacritic(c);
            match c {
                'à' | 'â' | 'ä' => "a",
                'é' | 'è' | 'ê' | 'ë' => "e",
                'î' | 'ï' => "i",
                'ô' | 'ö' => "o",
                'ù' | 'û' | 'ü' => "u",
                'ÿ' => "y",
                'ç' => "c",
                'À' | 'Â' | 'Ä' => "A",
                'É' | 'È' | 'Ê' | 'Ë' => "E",
                'Î' | 'Ï' => "I",
                'Ô' | 'Ö' => "O",
                'Ù' | 'Û' | 'Ü' => "U",
                'Ÿ' => "Y",
                'Ç' => "C",
                'œ' => "oe",
                'æ' => "ae",
                'Œ' => "Oe",
                'Æ' => "Ae",
                other => return other.to_string(),
            }
            .to_string()
        })
        .collect()
}

fn title_case(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Spelling variants of a dictionary word for phrase boosting: the product of
/// {as-is, accent-folded} × {as-is, lowercase, Title-case}, deduplicated. The
/// model may capitalize or drop accents, so each variant is boosted
/// independently; the original spelling is kept so mixed-case words
/// ("GitHub") are boosted under their exact form too.
pub fn word_variants(word: &str) -> Vec<String> {
    let bases = [word.to_string(), fold_accents(word)];
    let mut variants: Vec<String> = Vec::new();
    for base in bases {
        let lower = base.to_lowercase();
        for variant in [base.clone(), title_case(&lower), lower] {
            if !variant.is_empty() && !variants.contains(&variant) {
                variants.push(variant);
            }
        }
    }
    variants
}

/// BPE-tokenize a single word with the model's own tokenizer. The ids returned
/// are the model's vocab ids (verified to align 1:1 with vocab.txt), so they can
/// feed the boost tree directly. The Metaspace pre-tokenizer prepends the word
/// boundary marker itself; the word must not be space-prefixed.
/// Returns `None` if encoding fails or yields the unknown token (id 0).
pub fn tokenize_word_to_ids(tokenizer: &Tokenizer, word: &str) -> Option<Vec<i32>> {
    let encoding = tokenizer.encode(word, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() || ids.contains(&UNK_ID) {
        return None;
    }
    Some(ids.iter().map(|&id| id as i32).collect())
}

const UNK_ID: u32 = 0;

/// Load the model's BPE tokenizer from an explicitly resolved `tokenizer.json`
/// path. Missing path/file or parse error degrades to `None` so transcription
/// keeps working without phrase boosting.
pub fn load_tokenizer(tokenizer_path: Option<&std::path::Path>) -> Option<Tokenizer> {
    let path = tokenizer_path?;
    match Tokenizer::from_file(path) {
        Ok(tokenizer) => Some(tokenizer),
        Err(err) => {
            log::warn!(
                "Tokenizer unavailable at {}, phrase boosting disabled: {}",
                path.display(),
                err
            );
            None
        }
    }
}

/// Per-word confidence over `result`'s decoded tokens: the minimum token
/// probability (weakest link), with punctuation-only tokens excluded from the
/// minimum. The grouping itself lives in `crate::decode::word_confidences`,
/// shared with Whisper; the pieces here are Parakeet's vocab strings, where
/// Whisper's are byte-level BPE pieces.
pub fn word_confidences(result: &TimestampedResult) -> Vec<(String, f32)> {
    crate::decode::word_confidences(
        result
            .tokens
            .iter()
            .zip(result.probs.iter())
            .map(|(token, &prob)| (Cow::Borrowed(token.as_str()), prob)),
    )
}
