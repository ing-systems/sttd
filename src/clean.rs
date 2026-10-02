// ---------------------------------------------------------------------------
// Clean: post-gate transcript polish, before emission.
// ---------------------------------------------------------------------------
//
// Ported from Murmure's `audio/clean_recording.rs::clean_transcription`:
// collapse runs of 3+ identical consecutive words down to 2
// (case-insensitive), and optionally strip hesitation/filler words.
//
// Runs only on transcripts the gate already accepted (`Emit`); drops stay
// drops, so cleaning can never resurrect boilerplate or repetition loops.
// If cleaning removes everything (e.g. hesitations-only with the flag on),
// the caller converts the verdict to `DropEmpty` instead of emitting empty
// text (which would clobber the clipboard).

/// Hesitation/filler tokens stripped when `remove_hesitations` is on.
/// Conservative exact-match list (lowercased, punctuation-trimmed): English
/// base from Murmure plus Polish `yyy`/`eee` variants. Opt-in and off by
/// default: the list is stripped whatever the language, so it can delete
/// real words (Murmure hit German `um` the same way).
static FILLER_WORDS: &[&str] = &[
    "yyy", "yyyy", "eee", "euh", "hmm", "hm", "mmm", "uh", "um", "uhm", "umm", "uhh", "uhhh", "eh",
    "ehh", "ehm", "ah", "ha", "mm", "mh", "mm-hmm",
];

fn is_filler(word: &str) -> bool {
    let normalized = word.trim_matches(|c: char| !c.is_alphanumeric());
    if normalized.is_empty() {
        return false;
    }
    FILLER_WORDS.contains(&normalized.to_lowercase().as_str())
}

/// Polish `clean_transcription`: dedup repeats always, filler strip only
/// when `remove_hesitations` is true. Pure over inputs for tests.
pub(crate) fn clean_transcription(text: &str, remove_hesitations: bool) -> String {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| !remove_hesitations || !is_filler(w))
        .collect();
    if words.is_empty() {
        return String::new();
    }
    let mut result: Vec<&str> = Vec::with_capacity(words.len());
    // One lowercase form per word, reused by the run comparison below: the
    // inner loop used to allocate a fresh copy of the candidate word on
    // every step.
    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let mut i = 0;
    while i < words.len() {
        let mut count = 1;
        while i + count < words.len() && lower[i + count] == lower[i] {
            count += 1;
        }
        if count >= 3 {
            result.push(words[i]);
            result.push(words[i + 1]);
        } else {
            for j in 0..count {
                result.push(words[i + j]);
            }
        }
        i += count;
    }
    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_four_to_two() {
        assert_eq!(clean_transcription("je je je je vais", false), "je je vais");
    }

    #[test]
    fn dedup_two_kept() {
        assert_eq!(clean_transcription("oui oui", false), "oui oui");
    }

    #[test]
    fn dedup_case_insensitive() {
        assert_eq!(
            clean_transcription("Hello HELLO hello world", false),
            "Hello HELLO world"
        );
    }

    #[test]
    fn filler_removed_when_enabled() {
        assert_eq!(clean_transcription("je yyy vais", true), "je vais");
        assert_eq!(clean_transcription("eee bonjour", true), "bonjour");
    }

    #[test]
    fn filler_kept_when_disabled() {
        assert_eq!(clean_transcription("je yyy vais", false), "je yyy vais");
    }

    #[test]
    fn filler_only_becomes_empty() {
        assert_eq!(clean_transcription("yyy eee", true), "");
    }

    #[test]
    fn filler_substring_kept() {
        assert_eq!(
            clean_transcription("aujourd'hui hammer", true),
            "aujourd'hui hammer"
        );
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(clean_transcription("", true), "");
        assert_eq!(clean_transcription("   ", false), "");
    }
}
