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
// without confidences (tests) pass `None` for the ungated form.
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

/// Fold Polish diacritics and lowercase (the dictionary's lookup key form).
///
/// Lowercasing runs before folding, so an uppercase spelling reaches the same
/// key as its lowercase form: `żółć`, `ŻÓŁĆ` and `ZOLC` all key to `zolc`.
///
/// `pub(crate)` so the bus thread reuses the exact key form that
/// `sanitize_entries` and `confidence_map` use: a duplicate in `AddWords` and a
/// miss in `Contains` must agree with the post-correction lookup.
pub(crate) fn normalize_word(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    normalize_word_into(word, &mut out);
    out
}

/// Same normalization as `normalize_word`, written into a caller-owned
/// buffer so per-word passes reuse one allocation instead of one per word.
fn normalize_word_into(word: &str, out: &mut String) {
    out.clear();
    for c in word.chars() {
        // Lowercase first, then fold. `fold_polish_diacritic` matches only
        // lowercase letters, so folding before lowercasing let uppercase Polish
        // diacritics through: `Ż` survived the fold and became `ż` only
        // afterwards, so `ŻÓŁĆ` and `żółć` produced different keys.
        //
        // Per-char lowercasing keeps the Greek word-final sigma case that
        // `str::to_lowercase` alone would get wrong, and iterating the result
        // handles the chars that expand to more than one (`İ`).
        for lower in c.to_lowercase() {
            out.push(crate::gate::fold_polish_diacritic(lower));
        }
    }
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
    // One reusable key buffer: a word already in the map (the common case for
    // a repeated term) is then looked up without allocating a key.
    let mut key = String::new();
    for (word, conf) in word_confidences {
        for fragment in word.split(|c: char| !c.is_alphanumeric()) {
            normalize_word_into(fragment, &mut key);
            if key.is_empty() {
                continue;
            }
            match map.get_mut(key.as_str()) {
                Some(slot) => *slot = slot.max(*conf),
                None => {
                    map.insert(key.clone(), *conf);
                }
            }
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
    let mut post = Post::new(dictionary, confidences);

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
        if let Some(canonical) = post.best_match(word) {
            out.push_str(canonical);
            out.push_str(trailing);
        } else if let Some(canonical) = post.bigram_match(word, trailing, segments.get(i + 1)) {
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

const fn max_distance_for(len: usize) -> usize {
    if len >= POSTCORR_LONG_LEN {
        2
    } else {
        1
    }
}

/// Everything one correction pass needs, built once per utterance: the
/// normalized dictionary index, the optional confidence gate, and whether
/// this dictionary size still gets fuzzy correction. The per-word passes
/// below share it instead of each rebuilding the index.
struct Post<'a> {
    /// Normalized entry -> canonical entry as written in the file (the
    /// exact restore arm).
    index: HashMap<String, &'a String>,
    confidences: Option<&'a HashMap<String, f32>>,
    fuzzy_enabled: bool,
    /// Reusable normalized-key buffers, so matching a word never allocates
    /// (only the emitted output does).
    first: String,
    second: String,
    joined: String,
}

impl<'a> Post<'a> {
    fn new(dictionary: &'a [String], confidences: Option<&'a HashMap<String, f32>>) -> Self {
        Self {
            index: dictionary
                .iter()
                .map(|key| (normalize_word(key), key))
                .collect(),
            confidences,
            fuzzy_enabled: dictionary.len() <= POSTCORR_MAX_DICT_WORDS,
            first: String::new(),
            second: String::new(),
            joined: String::new(),
        }
    }

    /// Best canonical entry for one raw word: an exact normalized hit first,
    /// then a strictly-best fuzzy hit (short words and confident words never
    /// reach the fuzzy arm).
    fn best_match(&mut self, word: &str) -> Option<&'a String> {
        normalize_word_into(word, &mut self.first);
        let target = self.first.as_str();
        if target.is_empty() {
            return None;
        }
        if let Some(canonical) = self.index.get(target) {
            return Some(canonical);
        }
        if !self.fuzzy_enabled {
            return None;
        }
        let target_len = target.chars().count();
        if target_len < POSTCORR_MIN_LEN || self.is_confident(target) {
            return None;
        }
        self.fuzzy_match(target, max_distance_for(target_len))
    }

    /// Bigram join repair `app image` -> `AppImage` (TDT segmentation), with
    /// `separator` the text between the two raw fragments (only a space or a
    /// hyphen joins). Returns `None` when the pair cannot be repaired; the
    /// caller then handles each fragment on its own.
    fn bigram_match(
        &mut self,
        first: &str,
        separator: &str,
        second: Option<&(&str, &str)>,
    ) -> Option<&'a String> {
        let &(second, _) = second?;
        if first.is_empty() || second.is_empty() || !matches!(separator, " " | "-") {
            return None;
        }
        normalize_word_into(first, &mut self.first);
        normalize_word_into(second, &mut self.second);
        self.joined.clear();
        self.joined.push_str(&self.first);
        self.joined.push_str(&self.second);
        if let Some(canonical) = self.index.get(self.joined.as_str()) {
            return Some(canonical);
        }
        // Pure predicates below, so they are checked in the order that keeps
        // the shared buffers alive (`best_match` reuses `first`).
        let joined_len = self.joined.chars().count();
        if !self.fuzzy_enabled || joined_len < POSTCORR_LONG_LEN {
            return None;
        }
        if self.fragments_confident() {
            return None;
        }
        let max_distance = if joined_len >= POSTCORR_BIGRAM_LONG_LEN {
            3
        } else {
            max_distance_for(joined_len)
        };
        if self.best_match(second).is_some() {
            return None;
        }
        self.fuzzy_match(self.joined.as_str(), max_distance)
    }

    /// Whether the model was confident about this word (Phase B: a confident
    /// occurrence is never fuzzy-corrected).
    fn is_confident(&self, word: &str) -> bool {
        self.confidences
            .and_then(|map| map.get(word))
            .is_some_and(|&conf| conf >= POSTCORR_CONF_THRESHOLD)
    }

    /// Whether either fragment of the current bigram is confident: the
    /// highest confidence the engine reported for the two words.
    fn fragments_confident(&self) -> bool {
        let conf = [self.first.as_str(), self.second.as_str()]
            .iter()
            .filter_map(|key| self.confidences.and_then(|map| map.get(*key)))
            .fold(0.0f32, |acc, &conf| acc.max(conf));
        conf >= POSTCORR_CONF_THRESHOLD
    }

    /// Closest entry within `max_distance` edits, or `None`. A tie leaves the
    /// word untouched.
    ///
    /// The length gap lower-bounds an edit distance, so the gap check throws
    /// out most of the dictionary before the DP runs, and the capped DP
    /// returns `max_distance + 1` for whatever is left without finishing it.
    /// Both keep the ranking exact: every entry within range survives, and
    /// the over-range ones can neither win nor tie.
    fn fuzzy_match(&self, target: &str, max_distance: usize) -> Option<&'a String> {
        let target_len = target.chars().count();
        let mut best: Option<(usize, &'a String)> = None;
        let mut tied = false;
        for (key, canonical) in &self.index {
            if key.chars().count().abs_diff(target_len) > max_distance {
                continue;
            }
            let dist = crate::gate::levenshtein_capped(target, key, max_distance);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every casing of a Polish word reaches one key, which is what makes
    /// `sanitize_entries` dedup and the `confidence_map` lookup agree.
    #[test]
    fn uppercase_polish_diacritics_fold() {
        assert_eq!(normalize_word("żółć"), "zolc");
        assert_eq!(normalize_word("ZOLC"), "zolc");
        assert_eq!(normalize_word("ŻÓŁĆ"), "zolc");
    }

    /// A char that lowercases to more than one char must not be dropped.
    #[test]
    fn a_multi_char_lowercase_expansion_survives() {
        // `İ` (U+0130) lowercases to `i` plus a combining dot above.
        assert_eq!(normalize_word("İ"), "i\u{0307}");
    }

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
    fn bigram_fuzzy_join_repairs_one_edit() {
        let d = dict(&["Application"]);
        // Joined form "applicatio" is 10 chars: inside the fuzzy window (8+
        // chars, two edits) and one edit from the entry.
        assert_eq!(
            correct_transcription("uzywam ap plicatio", &d, None),
            "uzywam Application"
        );
        // Two edits are inside the window for a 10-char join.
        assert_eq!(
            correct_transcription("uzywam ap plicati", &d, None),
            "uzywam Application"
        );
        // Three edits are not.
        assert_eq!(
            correct_transcription("uzywam ap plicat", &d, None),
            "uzywam ap plicat"
        );
        // A join under 8 chars never fuzzy-corrects.
        let d = dict(&["AppImage"]);
        assert_eq!(correct_transcription("ap imag", &d, None), "ap imag");
    }

    #[test]
    fn bigram_join_respects_the_confidence_gate() {
        let d = dict(&["Application"]);
        // One confident fragment blocks the fuzzy join.
        let mut confidences: HashMap<String, f32> = HashMap::new();
        confidences.insert("plicatio".to_string(), POSTCORR_CONF_THRESHOLD);
        assert_eq!(
            correct_transcription("uzywam ap plicatio", &d, Some(&confidences)),
            "uzywam ap plicatio"
        );
        // The other fragment confident: same outcome.
        let mut confidences: HashMap<String, f32> = HashMap::new();
        confidences.insert("ap".to_string(), 0.9);
        assert_eq!(
            correct_transcription("uzywam ap plicatio", &d, Some(&confidences)),
            "uzywam ap plicatio"
        );
        // Neither confident: repair runs.
        let confidences: HashMap<String, f32> = HashMap::new();
        assert_eq!(
            correct_transcription("uzywam ap plicatio", &d, Some(&confidences)),
            "uzywam Application"
        );
    }

    #[test]
    fn confident_word_is_never_fuzzy_corrected() {
        let d = dict(&["paracetamol"]);
        let mut confidences: HashMap<String, f32> = HashMap::new();
        confidences.insert("paracetamoll".to_string(), POSTCORR_CONF_THRESHOLD);
        assert_eq!(
            correct_transcription("paracetamoll", &d, Some(&confidences)),
            "paracetamoll"
        );
        // Just under the threshold still corrects.
        let mut confidences: HashMap<String, f32> = HashMap::new();
        confidences.insert("paracetamoll".to_string(), POSTCORR_CONF_THRESHOLD - 0.01);
        assert_eq!(
            correct_transcription("paracetamoll", &d, Some(&confidences)),
            "paracetamol"
        );
    }

    #[test]
    fn confidence_map_keeps_the_max_per_word() {
        let map = confidence_map(&[
            ("Dzień".to_string(), 0.4),
            ("dzień, dobry".to_string(), 0.9),
            ("  ".to_string(), 0.99),
        ]);
        assert_eq!(map.get("dzien"), Some(&0.9));
        assert_eq!(map.get("dobry"), Some(&0.9));
        assert_eq!(map.len(), 2);
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
