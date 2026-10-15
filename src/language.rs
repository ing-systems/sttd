//! The language a run asks to transcribe in, as the user or the system locale
//! named it. What it *resolves* to depends on the speech engine (Whisper turns
//! it into a prompt token, see `whisper_core::types`); this module only holds
//! the request, so `cli`, `config` and `translate` can name it without
//! depending on an engine.

/// Parse `--language`/`STTD_LANGUAGE`: a Whisper language code (`pl`, `en`,
/// `de`, `yue`) or `auto`.
///
/// The code is not checked against a list of languages here, because sttd holds
/// none: whether the model has a token for it is decided at load time against
/// the model's own `tokenizer.json` (`Language::resolve`), the only place that
/// knows. What this does reject is input that cannot be a language code at all
/// — a locale name, a word, a digit — so that mistake becomes a message instead
/// of a load that fails later.
pub fn parse_language(s: &str) -> Result<Language, String> {
    if s.trim().eq_ignore_ascii_case("auto") {
        return Ok(Language::Auto);
    }
    let code = s.trim().to_ascii_lowercase();
    let malformed =
        || format!("expected a Whisper language code (pl, en, de, yue, ...) or `auto`, got {s:?}");
    if !(2..=3).contains(&code.len()) || !code.bytes().all(|b| b.is_ascii_alphabetic()) {
        return Err(malformed());
    }
    Ok(Language::Explicit(code))
}

/// The language Whisper is told to decode in, as requested. Chosen once at
/// startup and fixed for the process's lifetime.
///
/// The origin is kept beside the code, not just the code, because the two are
/// treated differently when the model turns out not to have a token for them:
/// an explicit `--language` fails the load, and the system locale's language
/// falls back to `Auto` (see `resolve`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Language {
    /// An explicit code from `--language` / `STTD_LANGUAGE`. The model must
    /// have a token for it.
    Explicit(String),
    /// The system locale's language, used when `--language` is unset. Soft: a
    /// model with no token for it falls back to `Auto` with a warning, because
    /// the user did not ask for this code.
    System(String),
    /// Detect per utterance from the prompt's first-token logits.
    Auto,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit(code) | Self::System(code) => f.write_str(code),
            Self::Auto => f.write_str("auto"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parser takes a code and lowercases it; it must not decide which codes
    /// exist, because sttd holds no list of them.
    #[test]
    fn the_language_parser_accepts_codes_and_auto_only() {
        for code in ["pl", "en", "de", "yue", "haw", "DE", " pl "] {
            assert_eq!(
                parse_language(code),
                Ok(Language::Explicit(code.trim().to_ascii_lowercase())),
                "code={code:?}"
            );
        }
        assert_eq!(parse_language("auto"), Ok(Language::Auto));
        assert_eq!(parse_language("Auto"), Ok(Language::Auto));
        for bad in ["", "x", "abcd", "pl_PL", "en-US", "p1", "english", "12"] {
            assert!(parse_language(bad).is_err(), "must reject {bad:?}");
        }
        // The message must not list the languages either: it cannot, because it
        // does not know them.
        let err = parse_language("pl_PL").expect_err("a locale is not a code");
        assert!(!err.contains("polski"), "must not enumerate languages");
    }
}
