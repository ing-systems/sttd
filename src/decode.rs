// ---------------------------------------------------------------------------
// Greedy-decode helpers shared by both speech engines.
// ---------------------------------------------------------------------------
//
// Extracted the way `mel` was: Parakeet-TDT and Whisper large-v3-turbo decode
// the same way underneath — greedy argmax over one logit row, the softmax share
// of the winning logit as that token's confidence, weakest-link confidences
// grouped into words — and each engine had its own copy of the three functions.
// One owner, so the two engines cannot drift apart.
//
// What is deliberately *not* shared is `parakeet_core::engine::argmax_token`:
// it falls back to a blank id on an empty row and lets the last maximum win a
// tie, which is what TDT's blank-padded joint needs. `argmax_index` has no
// fallback and breaks ties toward the lowest id.
// ---------------------------------------------------------------------------

use std::borrow::Cow;

/// Index of the highest logit, with ties going to the lowest id. Returns 0 on
/// an empty slice.
///
/// `#[inline]`: one call per generated token in both engines.
#[inline]
pub(crate) fn argmax_index(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Softmax probability of one token over `logits`, via the max-subtraction
/// trick. Out-of-range ids score 0 rather than panicking: a wrong id here
/// means a broken export, and the confidence then reads as "no confidence"
/// for the dictionary gate instead of taking the process down.
///
/// `#[inline]`: one call per generated token in both engines.
#[inline]
pub(crate) fn softmax_prob(logits: &[f32], token: usize) -> f32 {
    let Some(&target) = logits.get(token) else {
        return 0.0;
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum > 0.0 {
        ((target - max).exp() / sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Per-word confidences: a word opens on a piece that starts with a space,
/// continues on one that does not, and its confidence is the weakest piece
/// probability among the ones carrying alphanumerics. Entries left with no
/// alphanumerics are dropped, so the dictionary gate never looks up an empty
/// key.
///
/// The caller supplies the pieces because the two engines spell one token
/// differently: Parakeet has a vocab table, Whisper a byte-level BPE
/// tokenizer. `Cow` lets the table path borrow while the tokenizer path owns.
pub(crate) fn word_confidences<'a>(
    pieces: impl Iterator<Item = (Cow<'a, str>, f32)>,
) -> Vec<(String, f32)> {
    let mut words: Vec<(String, f32)> = Vec::new();
    for (piece, prob) in pieces {
        let opens_word = piece.starts_with(' ');
        let has_content = piece.chars().any(|c| c.is_alphanumeric());
        if opens_word || words.is_empty() {
            words.push((
                piece.trim_start().to_string(),
                if has_content { prob } else { 1.0 },
            ));
        } else if let Some((text, conf)) = words.last_mut() {
            text.push_str(&piece);
            if has_content {
                *conf = conf.min(prob);
            }
        }
    }
    words.retain(|(text, _)| text.chars().any(|c| c.is_alphanumeric()));
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_highest_logit_and_the_lowest_id_on_a_tie() {
        assert_eq!(argmax_index(&[0.0, 1.0, 2.0]), 2);
        assert_eq!(argmax_index(&[2.0, 1.0, 2.0]), 0, "ties go to the lowest id");
        assert_eq!(argmax_index(&[f32::NEG_INFINITY; 4]), 0);
        assert_eq!(argmax_index(&[]), 0, "empty logits must not panic");
    }

    #[test]
    fn softmax_prob_is_the_softmax_share() {
        // Two equal logits: each is half the mass.
        let p = softmax_prob(&[0.0, 0.0], 0);
        assert!((p - 0.5).abs() < 1e-6, "p={p}");
        // A logit 20 above the rest is essentially all the mass.
        assert!(softmax_prob(&[20.0, 0.0], 0) > 0.999);
        assert!(softmax_prob(&[20.0, 0.0], 1) < 1e-6);
        // Out of range and empty are 0, not a panic.
        assert_eq!(softmax_prob(&[1.0], 5), 0.0);
        assert_eq!(softmax_prob(&[], 0), 0.0);
        // Sum to 1 across the vocab.
        let logits = [1.0, 2.0, 3.0, 0.5];
        let total: f32 = (0..logits.len())
            .map(|i| softmax_prob(&logits, i))
            .sum();
        assert!((total - 1.0).abs() < 1e-5, "total={total}");
    }

    fn pieces<'a>(pairs: &'a [(&'a str, f32)]) -> Vec<(Cow<'a, str>, f32)> {
        pairs
            .iter()
            .map(|&(p, prob)| (Cow::Borrowed(p), prob))
            .collect()
    }

    /// The grouping rule both engines rely on: a leading space opens a word,
    /// anything else continues the previous one, and the word's confidence is
    /// the weakest piece that carried an alphanumeric. Trailing punctuation
    /// rides along on the word it follows and does not lower its confidence.
    #[test]
    fn words_group_pieces_and_take_the_weakest_link() {
        let got = word_confidences(
            pieces(&[
                (" Dzie", 0.9),
                ("ń", 0.8),
                (" dob", 0.7),
                ("ry", 0.6),
                (".", 0.95),
                (" a", 0.5),
            ])
            .into_iter(),
        );
        assert_eq!(
            got,
            vec![
                ("Dzień".to_string(), 0.8),
                ("dobry.".to_string(), 0.6),
                ("a".to_string(), 0.5),
            ]
        );
    }

    /// Punctuation carries no confidence of its own, so an utterance made only
    /// of it yields no words at all — the dictionary gate looks words up by
    /// name, and an empty or whitespace entry would never match anything.
    #[test]
    fn punctuation_only_pieces_yield_no_words() {
        assert!(
            word_confidences(pieces(&[(" -", 0.9), (" .", 0.9), (" ,", 0.9)]).into_iter())
                .is_empty()
        );
    }

    /// A leading piece with no space still opens the first word: a decode that
    /// starts on a word-initial token must not drop it.
    #[test]
    fn the_first_word_opens_without_a_leading_space() {
        let got = word_confidences(pieces(&[("Dzień", 0.9), (" dobry", 0.4)]).into_iter());
        assert_eq!(
            got,
            vec![
                ("Dzień".to_string(), 0.9),
                ("dobry".to_string(), 0.4),
            ]
        );
    }
}
