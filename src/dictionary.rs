// ---------------------------------------------------------------------------
// Dictionary: opt-in custom-vocabulary correction, post-gate only.
// ---------------------------------------------------------------------------
//
// Ported from Murmure's `dictionary/dictionary.rs`, adapted to Polish
// (`gate::fold_polish_diacritic`, not Murmure's French-only fold).
// Corrects only transcripts the gate already accepted (`Emit`); drops
// (hallucination, repetition, low-confidence, empty) never reach here, so a
// near-miss dictionary key cannot resurrect boilerplate.
//
// Phase A (this spike): exact normalized restore + strict fuzzy, ungated.
// `parakeet-rs 0.3` exposes no per-token probabilities (only
// `text + TimedToken{start,end}`), so Murmure's confidence gate
// (`POSTCORR_CONF_THRESHOLD`) has nothing to read. The signature already
// takes `confidences: Option<...>` so Phase B is a fill-in once token probs
// exist; callers pass `None` today.
//
// Strictness caps (same values as Murmure):
// - words < 5 chars never fuzzy-corrected;
// - < 8 chars allow 1 edit, >= 8 chars allow 2;
// - > 100 dictionary words disables fuzzy entirely (exact restore stays);
// - distance ties are left untouched;
// - bigram join repairs `app image -> AppImage` splits (TDT segmentation).

use std::collections::HashMap;
use std::path::Path;

/// Words shorter than this (normalized chars) are never fuzzy-corrected.
pub(crate) const POSTCORR_MIN_LEN: usize = 5;
/// Words this long and above may absorb 2 edits; shorter ones only 1.
pub(crate) const POSTCORR_LONG_LEN: usize = 8;
/// Above this many dictionary words fuzzy is disabled entirely; exact
/// normalized restore (casing/accents) stays on.
pub(crate) const POSTCORR_MAX_DICT_WORDS: usize = 100;
/// Model-confidence gate for Phase B (per-word min token prob). Unused while
/// callers pass `None`; kept so the threshold is not re-tuned later.
pub(crate) const POSTCORR_CONF_THRESHOLD: f32 = 0.45;
/// Joined length at which the bigram pass may absorb 3 edits.
const POSTCORR_BIGRAM_LONG_LEN: usize = 12;

fn normalize_word(word: &str) -> String {
    word.chars()
        .map(crate::gate::fold_polish_diacritic)
        .collect::<String>()
        .to_lowercase()
}

/// Load a `--dictionary` file: UTF-8, one phrase per line, trimmed, empties
/// dropped, canonical casing preserved as written. Missing path returns an
/// error for the caller to warn-and-disable (never fail transcription).
pub(crate) fn load_dictionary(path: &Path) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Production path: correct `text` against `dictionary`.
/// `confidences` maps normalized word -> model confidence; `None` (all
/// current engines) behaves ungated.
pub(crate) fn correct_transcription(
    text: &str,
    dictionary: &[String],
    confidences: Option<&HashMap<String, f32>>,
) -> String {
    restore_dictionary_casing_gated(text, dictionary, confidences)
}

pub(crate) fn restore_dictionary_casing_gated(
    text: &str,
    dictionary: &[String],
    confidences: Option<&HashMap<String, f32>>,
) -> String {
    if dictionary.is_empty() {
        return text.to_string();
    }
    let fuzzy_enabled = dictionary.len() <= POSTCORR_MAX_DICT_WORDS;
    let normalized: HashMap<String, &String> = dictionary
        .iter()
        .map(|key| (normalize_word(key), key))
        .collect();

    let segments: Vec<(&str, &str)> = text
        .split_inclusive(|c: char| !c.is_alphanumeric())
        .map(|segment| {
            let boundary_len = segment
                .chars()
                .rev()
                .take_while(|c| !c.is_alphanumeric())
                .map(char::len_utf8)
                .sum::<usize>();
            segment.split_at(segment.len() - boundary_len)
        })
        .collect();

    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < segments.len() {
        let (word, trailing) = segments[i];
        if let Some(canonical) =
            best_dictionary_match(word, &normalized, confidences, fuzzy_enabled)
        {
            out.push_str(canonical);
            out.push_str(trailing);
        } else if let Some(canonical) =
            bigram_match(&segments, i, &normalized, confidences, fuzzy_enabled)
        {
            out.push_str(canonical);
            out.push_str(segments[i + 1].1);
            i += 2;
            continue;
        } else {
            out.push_str(word);
            out.push_str(trailing);
        }
        i += 1;
    }
    out
}

fn max_distance_for(len: usize) -> usize {
    if len >= POSTCORR_LONG_LEN {
        2
    } else {
        1
    }
}

fn best_dictionary_match<'a>(
    word: &str,
    normalized: &HashMap<String, &'a String>,
    confidences: Option<&HashMap<String, f32>>,
    fuzzy_enabled: bool,
) -> Option<&'a String> {
    let target = normalize_word(word);
    if target.is_empty() {
        return None;
    }
    if let Some(canonical) = normalized.get(&target) {
        return Some(canonical);
    }
    if !fuzzy_enabled {
        return None;
    }
    let target_len = target.chars().count();
    if target_len < POSTCORR_MIN_LEN {
        return None;
    }
    if let Some(map) = confidences {
        if let Some(&conf) = map.get(&target) {
            if conf >= POSTCORR_CONF_THRESHOLD {
                return None;
            }
        }
    }
    fuzzy_match(&target, normalized, max_distance_for(target_len))
}

fn bigram_match<'a>(
    segments: &[(&str, &str)],
    i: usize,
    normalized: &HashMap<String, &'a String>,
    confidences: Option<&HashMap<String, f32>>,
    fuzzy_enabled: bool,
) -> Option<&'a String> {
    let (first, separator) = segments[i];
    let &(second, _) = segments.get(i + 1)?;
    if first.is_empty() || second.is_empty() || !matches!(separator, " " | "-") {
        return None;
    }
    let joined = format!("{}{}", normalize_word(first), normalize_word(second));
    if let Some(canonical) = normalized.get(&joined) {
        return Some(canonical);
    }
    if !fuzzy_enabled
        || joined.chars().count() < POSTCORR_LONG_LEN
        || best_dictionary_match(second, normalized, confidences, fuzzy_enabled).is_some()
    {
        return None;
    }
    if let Some(map) = confidences {
        let fragments_conf = [normalize_word(first), normalize_word(second)]
            .iter()
            .filter_map(|key| map.get(key))
            .fold(0.0f32, |acc, &conf| acc.max(conf));
        if fragments_conf >= POSTCORR_CONF_THRESHOLD {
            return None;
        }
    }
    let max_distance = if joined.chars().count() >= POSTCORR_BIGRAM_LONG_LEN {
        3
    } else {
        max_distance_for(joined.chars().count())
    };
    fuzzy_match(&joined, normalized, max_distance)
}

fn fuzzy_match<'a>(
    target: &str,
    normalized: &HashMap<String, &'a String>,
    max_distance: usize,
) -> Option<&'a String> {
    let mut best: Option<(usize, &String)> = None;
    let mut tied = false;
    for (key, canonical) in normalized {
        let dist = crate::gate::levenshtein(target, key);
        match best {
            Some((best_dist, _)) if dist > best_dist => {}
            Some((best_dist, _)) if dist == best_dist => tied = true,
            _ => {
                best = Some((dist, canonical));
                tied = false;
            }
        }
    }
    match best {
        Some((dist, canonical)) if !tied && dist <= max_distance => Some(canonical),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn restore_casing_rewrites_whole_word_match() {
        let d = dict(&["Syntocinon"]);
        assert_eq!(
            correct_transcription("on injecte syntocinon teraz", &d, None),
            "on injecte Syntocinon teraz"
        );
    }

    #[test]
    fn preserves_punctuation_and_ignores_substrings() {
        let d = dict(&["Syntocinon"]);
        assert_eq!(
            correct_transcription("dawka: syntocinon.", &d, None),
            "dawka: Syntocinon."
        );
        assert_eq!(
            correct_transcription("syntocinon", &dict(&["cin"]), None),
            "syntocinon"
        );
    }

    #[test]
    fn polish_diacritics_restored() {
        let d = dict(&["żółć"]);
        assert_eq!(correct_transcription("zolc", &d, None), "żółć");
        assert_eq!(correct_transcription("Zolc", &d, None), "żółć");
    }

    #[test]
    fn fuzzy_corrects_one_edit_on_long_word() {
        let d = dict(&["paracetamol"]);
        assert_eq!(
            correct_transcription("paracetamoll", &d, None),
            "paracetamol"
        );
    }

    #[test]
    fn fuzzy_never_touches_short_words() {
        let d = dict(&["les", "des"]);
        assert_eq!(correct_transcription("les amis", &d, None), "les amis");
    }

    #[test]
    fn fuzzy_requires_long_word_for_two_edits() {
        let d = dict(&["persil"]);
        assert_eq!(correct_transcription("parcil", &d, None), "parcil");
    }

    #[test]
    fn fuzzy_skips_on_distance_tie() {
        let d = dict(&["barrer", "barret"]);
        assert_eq!(correct_transcription("barres", &d, None), "barres");
    }

    #[test]
    fn bigram_exact_join_repairs_segmentation() {
        let d = dict(&["AppImage"]);
        assert_eq!(
            correct_transcription("dostepne jako app image.", &d, None),
            "dostepne jako AppImage."
        );
    }

    #[test]
    fn fuzzy_disabled_above_cap_but_exact_stays() {
        let mut d: Vec<String> = (0..POSTCORR_MAX_DICT_WORDS)
            .map(|i| format!("slownik{i:03}"))
            .collect();
        d.push("Syntocinon".to_string());
        assert_eq!(
            correct_transcription("dawka sintocinon", &d, None),
            "dawka sintocinon"
        );
        assert_eq!(
            correct_transcription("dawka syntocinon", &d, None),
            "dawka Syntocinon"
        );
    }

    #[test]
    fn empty_dictionary_is_noop() {
        assert_eq!(correct_transcription("Dzien dobry", &[], None), "Dzien dobry");
    }

    #[test]
    fn load_dictionary_trims_and_skips_blanks() {
        let dir = std::env::temp_dir().join(format!(
            "sttd-dict-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.txt");
        std::fs::write(&path, "  Syntocinon\n\nżółć  \n").unwrap();
        let words = load_dictionary(&path).expect("must load");
        assert_eq!(words, vec!["Syntocinon".to_string(), "żółć".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
        assert!(load_dictionary(&dir.join("missing.txt")).is_err());
    }
}
