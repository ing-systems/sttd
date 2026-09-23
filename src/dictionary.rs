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
// `parakeet-rs 0.3` is gone: `parakeet_core` reports per-word confidences, so
// the production path (`asr::transcribe_and_emit`) passes a gated map and
// Phase B is live — confident words are never fuzzy-corrected. Callers
// without confidences (whisper, tests) pass `None` for the ungated form.
//
// Strictness caps (same values as Murmure):
// - words < 5 chars never fuzzy-corrected;
// - < 8 chars allow 1 edit, >= 8 chars allow 2;
// - > 100 dictionary words disables fuzzy entirely (exact restore stays);
// - distance ties are left untouched;
// - bigram join repairs `app image -> AppImage` splits (TDT segmentation).
//
// Entry safeguards (mirroring Murmure's `validate_dictionary_format`):
// - entries with ASCII digits are rejected (use formatting rules for numbers);
// - entries with more than one space (> 2 words) are rejected (matcher only
//   handles unigrams + bigrams);
// - duplicates by normalized form (case/accent-insensitive) keep the first;
// - UTF-8 BOM stripped, surrounding whitespace trimmed, empties dropped.

use std::collections::HashMap;
use std::path::Path;

/// Words shorter than this (normalized chars) are never fuzzy-corrected.
pub(crate) const POSTCORR_MIN_LEN: usize = 5;
/// Words this long and above may absorb 2 edits; shorter ones only 1.
pub(crate) const POSTCORR_LONG_LEN: usize = 8;
/// Above this many dictionary words fuzzy is disabled entirely; exact
/// normalized restore (casing/accents) stays on.
pub(crate) const POSTCORR_MAX_DICT_WORDS: usize = 100;
/// Above this many words log a false-positive warning (yellow zone); at or
/// above `POSTCORR_MAX_DICT_WORDS` fuzzy is off entirely (red zone).
pub(crate) const POSTCORR_WARN_DICT_WORDS: usize = 50;
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
///
/// Invalid entries are skipped with a warning (never fail the whole file):
/// entries with ASCII digits, entries with more than one space (> 2 words),
/// and normalized duplicates (case/accent-insensitive, first wins).
pub(crate) fn load_dictionary(path: &Path) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(sanitize_entries(text.lines().map(|l| l.trim())))
}

/// Pure entry filter shared by `load_dictionary` and tests: strips a leading
/// BOM, trims, drops empties, rejects digits / > 1 space, dedups by
/// normalized form keeping the first occurrence.
pub(crate) fn sanitize_entries<'a, I>(lines: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen: HashMap<String, ()> = HashMap::new();
    let mut kept = Vec::new();
    let mut skipped_digits = 0usize;
    let mut skipped_spaces = 0usize;
    let mut skipped_dupes = 0usize;
    for raw in lines {
        let line = raw.strip_prefix('\u{feff}').unwrap_or(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.chars().any(|c| c.is_ascii_digit()) {
            skipped_digits += 1;
            continue;
        }
        if line.chars().filter(|&c| c == ' ').count() > 1 {
            skipped_spaces += 1;
            continue;
        }
        let key = normalize_word(line);
        if seen.contains_key(&key) {
            skipped_dupes += 1;
            continue;
        }
        seen.insert(key, ());
        kept.push(line.to_string());
    }
    if skipped_digits > 0 {
        log::warn!(
            "dictionary: skipped {skipped_digits} entr(y/ies) with digits (numbers are not supported)"
        );
    }
    if skipped_spaces > 0 {
        log::warn!(
            "dictionary: skipped {skipped_spaces} entr(y/ies) with more than two words (at most one space allowed)"
        );
    }
    if skipped_dupes > 0 {
        log::warn!(
            "dictionary: skipped {skipped_dupes} duplicate entr(y/ies) (case/accent-insensitive)"
        );
    }
    kept
}

/// A word appearing several times keeps its max confidence: protecting a
/// confident occurrence beats correcting a mumbled duplicate. Engine words
/// are split on non-alphanumeric characters so the keys match the lookup
/// segmentation of the restore pass.
pub(crate) fn confidence_map(word_confidences: &[(String, f32)]) -> HashMap<String, f32> {
    let mut map: HashMap<String, f32> = HashMap::new();
    for (word, conf) in word_confidences {
        for fragment in word.split(|c: char| !c.is_alphanumeric()) {
            let key = normalize_word(fragment);
            if key.is_empty() {
                continue;
            }
            map.entry(key)
                .and_modify(|c| *c = c.max(*conf))
                .or_insert(*conf);
        }
    }
    map
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

    #[test]
    fn sanitize_rejects_digits_and_long_entries() {
        assert!(sanitize_entries(["abc123"]).is_empty());
        assert!(sanitize_entries(["a b c"]).is_empty());
        assert_eq!(
            sanitize_entries(["New York", "AppImage", "a b c", "paracetamol2"]),
            vec!["New York".to_string(), "AppImage".to_string()]
        );
    }

    #[test]
    fn sanitize_dedups_case_and_accent_insensitive() {
        assert_eq!(
            sanitize_entries(["Syntocinon", "syntocinon", "SYNTOCINON"]),
            vec!["Syntocinon".to_string()]
        );
        assert_eq!(
            sanitize_entries(["żółć", "zolc"]),
            vec!["żółć".to_string()]
        );
    }

    #[test]
    fn sanitize_strips_bom_and_trims() {
        assert_eq!(
            sanitize_entries(["\u{feff}Kubernetes", "  Parakeet  ", "", "   "]),
            vec!["Kubernetes".to_string(), "Parakeet".to_string()]
        );
    }
}
