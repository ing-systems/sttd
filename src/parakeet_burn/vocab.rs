// Vocabulary for the Burn port: loads the same `vocab.txt` as the ORT
// path (one SentencePiece token per line) and mirrors the TDT decoder's
// text rebuild (`▁` -> space, digit spacing, `<...>` special skip).

use anyhow::{Context, Result};
use std::path::Path;

pub(crate) struct BurnVocab {
    id_to_token: Vec<String>,
}

impl BurnVocab {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let id_to_token: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        if id_to_token.is_empty() {
            anyhow::bail!("vocab is empty: {}", path.display());
        }
        Ok(Self { id_to_token })
    }

    pub(crate) fn size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Blank id: last token (matches `parakeet-rs` TDT convention).
    pub(crate) fn blank_id(&self) -> usize {
        self.id_to_token.len().saturating_sub(1)
    }

    pub(crate) fn id_to_text(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Rebuild text from greedy token ids (mirrors `decoder_tdt.rs`).
    pub(crate) fn rebuild(&self, tokens: &[usize]) -> String {
        let mut full = String::new();
        for &id in tokens {
            let Some(tok) = self.id_to_text(id) else { continue };
            if tok.starts_with('<') && tok.ends_with('>') && tok != "<unk>" {
                continue;
            }
            let mut display = tok.replace('▁', " ");
            if !full.is_empty()
                && !display.starts_with(' ')
                && display.chars().all(|c| c.is_ascii_digit())
            {
                let trailing: usize =
                    full.chars().rev().take_while(|c| c.is_alphabetic()).count();
                let last = full.chars().last();
                let is_article_a = trailing == 1 && last == Some('a');
                if trailing > 1 || is_article_a {
                    display.insert(0, ' ');
                }
            }
            full.push_str(&display);
        }
        full.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_handles_sentencepiece_and_digits() {
        let v = BurnVocab {
            id_to_token: vec!["▁like".into(), "1".into(), "0".into(), "0".into()],
        };
        assert_eq!(v.rebuild(&[0, 1, 2, 3]), "like 100");
        assert_eq!(v.blank_id(), 3);
    }

    #[test]
    fn blank_and_special_tokens_skipped() {
        let v = BurnVocab {
            id_to_token: vec!["▁a".into(), "<blank>".into()],
        };
        assert_eq!(v.rebuild(&[0, 1]), "a");
    }

    #[test]
    fn real_vocab_loads() {
        let path = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8/vocab.txt");
        if !path.exists() {
            return;
        }
        let v = BurnVocab::from_file(path).expect("vocab loads");
        // v3 multilingual vocab: 8193 SentencePiece tokens incl. blank.
        assert!(v.size() > 8000, "size={}", v.size());
    }
}
