// Vocabulary for the Burn port: loads the same `vocab.txt` as the ORT
// path (one SentencePiece token per line) and mirrors the TDT decoder's
// text decode_to_text (`▁` -> space, digit spacing, `<...>` special skip).

use anyhow::{Context, Result};
use std::path::Path;

pub(crate) struct BurnVocab {
    id_to_token: Vec<String>,
}

impl BurnVocab {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        // Same `{token} {id}` format `parakeet-rs` parses (`splitn(2,
        // ' ')`, token at its id): the trailing id is NOT part of the
        // token text. Taking whole lines leaks ids into transcripts
        // ("▁ma 483" rebuilds as " ma 483").
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut id_to_token: Vec<String> = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                continue;
            }
            let Ok(id) = parts[1].parse::<usize>() else {
                continue;
            };
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = parts[0].to_string();
        }
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
    pub(crate) fn decode_to_text(&self, tokens: &[usize]) -> String {
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
        assert_eq!(v.decode_to_text(&[0, 1, 2, 3]), "like 100");
        assert_eq!(v.blank_id(), 3);
    }

    #[test]
    fn blank_and_special_tokens_skipped() {
        let v = BurnVocab {
            id_to_token: vec!["▁a".into(), "<blank>".into()],
        };
        assert_eq!(v.decode_to_text(&[0, 1]), "a");
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

    #[test]
    fn file_format_strips_trailing_ids() {
        // On-disk lines are `{token} {id}` (parakeet-rs `splitn(2,
        // ' ')`); the id must not leak into token text or decode_to_text
        // emits phantom digits ("▁ma 483" -> " ma 483").
        let dir = std::env::temp_dir().join(format!(
            "sttd-burn-vocab-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.txt");
        std::fs::write(&path, "▁like 0\n100 1\n<blk> 2\n").unwrap();
        let v = BurnVocab::from_file(&path).expect("parses");
        assert_eq!(v.size(), 3);
        assert_eq!(v.id_to_text(0), Some("▁like"));
        assert_eq!(v.id_to_text(1), Some("100"));
        assert_eq!(v.decode_to_text(&[0, 1, 2]), "like 100");
        assert_eq!(v.blank_id(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
