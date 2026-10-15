//! The pure text half: sentence splitting, and keeping the source when a
//! translation comes back empty or failed.

use anyhow::Result;

/// Split text into the pieces that go through the model one at a time.
///
/// The terminators are `.` `!` `?` `;` and a newline. A colon is not one: it
/// separates a label from its value (`cena: 12.30 zł`), and cutting there sends
/// half a number to the model. A `.` between two digits is not one either, for
/// the same reason.
///
/// Every delimiter stays with the piece it ends, so no non-space character is
/// ever lost, and no empty piece is emitted. Text with no terminator stays one
/// piece.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        let ends_piece = match ch {
            '\n' | '!' | '?' | ';' => true,
            // `12.30` is one number, not two sentences.
            '.' => !matches!(chars.peek(), Some(next) if next.is_ascii_digit()),
            _ => false,
        };
        if ends_piece {
            push_piece(&mut pieces, &mut current);
        }
    }
    push_piece(&mut pieces, &mut current);
    merge_terminator_pieces(&mut pieces);
    pieces
}

/// Move a finished piece into the list, dropping it when it holds only spaces.
fn push_piece(pieces: &mut Vec<String>, current: &mut String) {
    let piece = current.trim();
    if !piece.is_empty() {
        pieces.push(piece.to_string());
    }
    current.clear();
}

/// Fold a run of terminators onto the piece before it, so `Wow!!` stays one
/// piece instead of `Wow!` plus a piece holding only `!`.
fn merge_terminator_pieces(pieces: &mut Vec<String>) {
    let mut merged: Vec<String> = Vec::with_capacity(pieces.len());
    for piece in pieces.drain(..) {
        let only_terminators = piece
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '.' | '!' | '?' | ';' | '\n'));
        match merged.last_mut() {
            Some(last) if only_terminators => last.push_str(&piece),
            _ => merged.push(piece),
        }
    }
    *pieces = merged;
}

/// The text to emit for one utterance: the translation, or the source text.
///
/// A translation that fails or comes back empty warns and falls back to
/// `source`, because an utterance the user heard in the wrong language is
/// worse than one they heard in the right one. `tag` names the utterance in the
/// log line.
pub(crate) fn fallback_or_translated(source: &str, r: Result<String>, tag: &str) -> String {
    match r {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        Ok(_) => {
            log::warn!("{tag}: translation came back empty; emitting the source text");
            source.to_string()
        }
        Err(e) => {
            log::warn!("{tag}: translation failed ({e:#}); emitting the source text");
            source.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn split_cuts_after_a_terminator() {
        assert_eq!(
            split_sentences("Cześć. Jak się masz?"),
            vec!["Cześć.", "Jak się masz?"]
        );
    }

    #[test]
    fn split_keeps_a_colon_piece_whole() {
        // The `.` in `12.30` is a decimal point and the `:` is a label, so this
        // is one piece, not three.
        assert_eq!(split_sentences("cena: 12.30 zł"), vec!["cena: 12.30 zł"]);
    }

    #[test]
    fn split_keeps_text_without_a_terminator_whole() {
        assert_eq!(split_sentences("cena 12 zł"), vec!["cena 12 zł"]);
    }

    #[test]
    fn split_never_emits_an_empty_piece() {
        // The leading `.;` has no piece to fold onto, so it stays a piece of
        // its own: it holds characters, and nothing may be dropped.
        let pieces = split_sentences("  \n .  ;  Dobry wieczór!  \n\n");
        assert_eq!(pieces, vec![".;", "Dobry wieczór!"]);
        assert!(pieces.iter().all(|p| !p.trim().is_empty()));
    }

    #[test]
    fn split_keeps_every_delimiter_with_its_piece() {
        assert_eq!(split_sentences("A; B\nC!"), vec!["A;", "B", "C!"]);
    }

    #[test]
    fn split_of_empty_text_is_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("  \n ").is_empty());
    }

    #[test]
    fn fallback_keeps_a_translation() {
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok("Good evening".to_string()), "t"),
            "Good evening"
        );
    }

    #[test]
    fn fallback_keeps_the_source_on_an_empty_result() {
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok(String::new()), "t"),
            "Dobry wieczór"
        );
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok("   ".to_string()), "t"),
            "Dobry wieczór"
        );
    }

    #[test]
    fn fallback_keeps_the_source_on_an_error() {
        assert_eq!(
            fallback_or_translated(
                "Dobry wieczór",
                Err(anyhow!("the decoder ran out of cache slots")),
                "t"
            ),
            "Dobry wieczór"
        );
    }
}
