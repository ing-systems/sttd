//! sttd's ISO-639-1 codes, and the FLORES spellings a multilingual export
//! needs instead.

use anyhow::Result;

use super::catalog::NLLB_SPEC;
use super::Direction;

/// A language code as the catalog spells it: trimmed, lowercased, with `_` for
/// `-` so `pt_BR`-shaped input and `pt-br` reach the same comparison.
pub(super) fn normalize(code: &str) -> String {
    code.trim().to_ascii_lowercase().replace('_', "-")
}

/// A language code as NLLB spells it: trimmed and lowercased, and **nothing
/// else**.
///
/// Not [`normalize`], which folds `_` into `-` for a pair code and would turn
/// `pol_Latn` into `pol-latn` — not a language the model has.
fn normalize_nllb(code: &str) -> String {
    code.trim().to_ascii_lowercase()
}

/// Whether `code` is spelled the way NLLB spells a language: a three-letter
/// ISO-639-3 code, an underscore, a four-letter script. `pol_Latn` is; `pl` is
/// not.
///
/// This gate is load-bearing, not a shape check for tidiness. NLLB's vocabulary is
/// a 256k-entry BPE, and it holds **subwords that look like language codes**:
/// `token_to_id("pl")` returns `Some(2678)`, a plain `pl` piece of text. Passing an
/// unrecognised ISO code straight through to the vocabulary would therefore
/// resolve, translate nothing in particular, and fail silently — the exact failure
/// mode `translate`'s start-token story exists to prevent.
fn is_flores_code(code: &str) -> bool {
    let Some((iso3, script)) = code.split_once('_') else {
        return false;
    };
    // Both halves have a fixed case, and both matter: the language part is
    // lowercase and the script part is capitalised in every one of the 202 codes
    // (`pol_Latn`, `zho_Hans`, `yue_Hant`). Requiring that exact shape is what
    // refuses `POL_LATN` — which `token_to_id` would not resolve anyway, so letting
    // it through would trade a clear error for a bare "no such token".
    let lower = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_lowercase());
    let capitalised = |s: &str| {
        s.len() == 4
            && s.bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_lowercase())
            && s.starts_with(|c: char| c.is_ascii_uppercase())
    };
    lower(iso3, 3) && capitalised(script)
}

/// The FLORES-200 code NLLB spells each ISO-639-1 code as.
///
/// **This table cannot be derived**, and the reason is worth recording, because
/// every other language lookup in sttd reads the model's own tokenizer (see
/// `whisper_core::types::language_token`). NLLB needs three things sttd does not
/// have: an ISO-639-3 code where sttd speaks ISO-639-1, a script suffix, and the
/// FLORES spelling — which for five languages is *neither* ISO-639-1 nor
/// ISO-639-3: Arabic is `arb`, Persian `pes`, Latvian `lvs`, Norwegian `nob` and
/// Malay `zsm`, where ISO-639-3 says `ara`/`fas`/`lav`/`nor`/`msa`. A mapping
/// spelled out by rule would be wrong for exactly the languages a mapping is
/// wanted for.
///
/// So the rows are data, generated from `iso_639-2.xml` (the terminological
/// `/T` codes, which are what FLORES follows) and then checked against the
/// tokenizer's own added tokens — `every_row_names_a_code_the_model_has`
/// does that against the downloaded export, so a typo fails here rather than in
/// front of a user. Widening it is adding a row.
///
/// Every ISO-639-1 code sttd names anywhere — the pair catalog's sources and
/// targets — is here, so no `--translate` code that works today stops working.
const NLLB_CODES: &[(&str, &str)] = &[
    ("af", "afr_Latn"),
    ("ar", "arb_Arab"),
    ("bg", "bul_Cyrl"),
    ("ca", "cat_Latn"),
    ("cs", "ces_Latn"),
    ("da", "dan_Latn"),
    ("de", "deu_Latn"),
    ("el", "ell_Grek"),
    ("en", "eng_Latn"),
    ("es", "spa_Latn"),
    ("et", "est_Latn"),
    ("eu", "eus_Latn"),
    ("fa", "pes_Arab"),
    ("fi", "fin_Latn"),
    ("fr", "fra_Latn"),
    ("ga", "gle_Latn"),
    ("he", "heb_Hebr"),
    ("hi", "hin_Deva"),
    ("hr", "hrv_Latn"),
    ("hu", "hun_Latn"),
    ("id", "ind_Latn"),
    ("it", "ita_Latn"),
    ("ja", "jpn_Jpan"),
    ("ko", "kor_Hang"),
    ("lt", "lit_Latn"),
    ("lv", "lvs_Latn"),
    ("mk", "mkd_Cyrl"),
    ("ms", "zsm_Latn"),
    ("nl", "nld_Latn"),
    ("no", "nob_Latn"),
    ("pl", "pol_Latn"),
    ("ro", "ron_Latn"),
    ("ru", "rus_Cyrl"),
    ("sk", "slk_Latn"),
    ("sl", "slv_Latn"),
    ("sr", "srp_Cyrl"),
    ("sv", "swe_Latn"),
    ("th", "tha_Thai"),
    ("tr", "tur_Latn"),
    ("uk", "ukr_Cyrl"),
    ("vi", "vie_Latn"),
    // Not ISO-639-1 at all: Cantonese is already a three-letter code, and it is
    // the case that proves the verbatim path works rather than only the table.
    ("yue", "yue_Hant"),
    ("zh", "zho_Hans"),
];

/// The FLORES code for a language sttd was asked for, or `None`.
///
/// The table first, then a verbatim FLORES code passes through untouched, so
/// `--translate pol_Latn` and `--translate zho_Hant` work for any of the model's
/// 202 languages without a row here. Anything else is refused — see
/// [`is_flores_code`] for why an unknown code must not reach the vocabulary.
fn nllb_code(code: &str) -> Option<String> {
    let trimmed = code.trim();
    // Normalized once, not once per row: the table is 44 rows and the
    // normalization allocates.
    let lower = normalize_nllb(trimmed);
    if let Some((_, flores)) = NLLB_CODES.iter().find(|(iso, _)| *iso == lower) {
        return Some((*flores).to_string());
    }
    // The verbatim path keeps its **case**: FLORES spells the script in capitals
    // (`pol_Latn`) and `token_to_id` is case-sensitive, so lowercasing here would
    // resolve nothing and look like a refusal. `is_flores_code` is likewise
    // case-preserving, which is why `POL_LATN` is refused rather than folded.
    is_flores_code(trimmed).then(|| trimmed.to_string())
}

/// The two FLORES codes a run translates between, or the reason it cannot.
///
/// Every failure here is a user mistake with a fix, so every message names it: a
/// missing source is `--language`, an unknown target is `--translate`, and the
/// multilingual export cannot guess either.
pub(super) fn nllb_codes(direction: Direction<'_>) -> Result<(String, String)> {
    let Some(src) = direction.src else {
        anyhow::bail!(
            "{NLLB_SPEC} translates a named source into a named target, and \
             --language auto leaves the source unknown; pass --language CODE \
             (e.g. --language pl --translate en), or drop --translate-model to use \
             a pair export, which can cover an unknown source"
        )
    };
    let src = nllb_code(src).ok_or_else(|| unknown_nllb(src, "--language"))?;
    let tgt = nllb_code(direction.tgt).ok_or_else(|| unknown_nllb(direction.tgt, "--translate"))?;
    Ok((src, tgt))
}

/// Why a language code reached [`nllb_codes`] and was refused.
///
/// The examples are drawn from the table on purpose: they show both accepted
/// spellings, so the fix is visible from the message rather than looked up.
fn unknown_nllb(code: &str, flag: &str) -> anyhow::Error {
    let examples: Vec<String> = NLLB_CODES
        .iter()
        .take(3)
        .map(|(iso, flores)| format!("{iso} -> {flores}"))
        .collect();
    anyhow::anyhow!(
        "{NLLB_SPEC} does not know the language {code:?}; it spells languages as \
         FLORES codes (three-letter code, underscore, script). {flag} accepts the \
         ISO-639-1 codes sttd uses elsewhere ({}) or a FLORES code verbatim \
         (pol_Latn, eng_Latn, zho_Hans). An unknown code is refused rather than \
         guessed at: NLLB's vocabulary holds ordinary BPE pieces that look like \
         language codes, so a typo would mistranslate instead of failing.",
        examples.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use crate::translator_store::catalog::CATALOG;
    use crate::translator_store::fetch::{translator_dir, TOKENIZER_FILE};

    use super::*;

    /// The table is the one piece of language data sttd compiles in, so it is
    /// held to what the model actually publishes. This runs against the
    /// downloaded NLLB tokenizer: a row naming a code the export does not have is
    /// a mistranslation waiting to happen, and no weight-free test can see it.
    ///
    /// Ignored because it needs the ~16 MiB `tokenizer.json` at
    /// `models/translators/nllb-200-distilled-600M/`; it also skips itself when
    /// they are not there, so `--ignored` is safe on a fresh checkout.
    #[test]
    #[ignore = "needs the downloaded nllb-200-distilled-600M tokenizer"]
    fn every_row_names_a_code_the_model_has() {
        let path = translator_dir(NLLB_SPEC).join(TOKENIZER_FILE);
        if !path.is_file() {
            eprintln!("skipping: no tokenizer at {}", path.display());
            return;
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("tokenizer must load");
        for (iso, flores) in NLLB_CODES {
            assert_eq!(
                tokenizer.token_to_id(flores),
                Some(
                    tokenizers::Tokenizer::from_file(&path)
                        .unwrap()
                        .token_to_id(flores)
                        .unwrap()
                ),
                "{iso} maps to {flores}, which the model must have"
            );
            assert!(
                tokenizer.token_to_id(flores).is_some(),
                "{iso} maps to {flores}, which this export does not have: the row is wrong"
            );
            assert!(is_flores_code(flores), "{flores} is not FLORES-shaped");
        }
        // And the premise that makes the shape gate load-bearing.
        assert_eq!(
            tokenizer.token_to_id("pl"),
            Some(2678),
            "the NLLB vocabulary holds a plain `pl` BPE piece; if that ever changes, \
             the shape gate in is_flores_code can be relaxed"
        );
    }

    /// `nllb_code` maps what sttd names and passes through what the model spells,
    /// and refuses everything else. The refusal is the load-bearing part: the
    /// vocabulary holds subwords that look like language codes, so an unknown code
    /// reaching it resolves instead of failing.
    #[test]
    fn a_language_code_maps_to_a_flores_code_or_is_refused() {
        // Every ISO-639-1 code sttd names anywhere is mapped, so no
        // `--translate` code that works today stops working.
        for code in CATALOG.iter().flat_map(|p| [p.src, p.tgt]) {
            if code == "mul" {
                continue; // not a language: a pair export's many-source marker
            }
            assert!(
                nllb_code(code).is_some(),
                "{code} is named by the catalog and must reach {NLLB_SPEC}"
            );
        }
        assert_eq!(nllb_code("pl").as_deref(), Some("pol_Latn"));
        assert_eq!(nllb_code("en").as_deref(), Some("eng_Latn"));
        assert_eq!(nllb_code("de").as_deref(), Some("deu_Latn"));
        // Case and surrounding whitespace are the user's.
        assert_eq!(nllb_code("  PL  ").as_deref(), Some("pol_Latn"));
        assert_eq!(nllb_code("PL").as_deref(), Some("pol_Latn"));
        // NLLB's five deliberate departures from ISO-639-3, which are the rows
        // that prove the table cannot be generated by a rule.
        for (iso, flores) in [
            ("ar", "arb_Arab"),
            ("fa", "pes_Arab"),
            ("lv", "lvs_Latn"),
            ("no", "nob_Latn"),
            ("ms", "zsm_Latn"),
        ] {
            assert_eq!(nllb_code(iso).as_deref(), Some(flores));
            assert_ne!(
                flores.split('_').next(),
                Some(iso),
                "{iso}: this row is only interesting if FLORES disagrees with ISO-639-1"
            );
        }
        // A FLORES code verbatim reaches all 202 languages, not just the 44 above.
        for code in ["pol_Latn", "eng_Latn", "zho_Hant", "cat_Latn", "kmr_Latn"] {
            assert_eq!(nllb_code(code).as_deref(), Some(code));
        }
        // And anything else is refused rather than handed to the vocabulary.
        for bad in [
            "pl_PL",
            "aut",
            "klingon",
            "pol",
            "pol_Latin",
            "POL_LATN",
            "12",
            "",
        ] {
            assert_eq!(nllb_code(bad), None, "{bad:?} must not resolve");
        }
        // `normalize` would fold the separator away; the NLLB path must not.
        assert_eq!(normalize("pol_Latn"), "pol-latn", "premise about normalize");
        assert_eq!(nllb_code("pol_Latn").as_deref(), Some("pol_Latn"));
    }

    /// The FLORES shape is `xxx_Ssss`: three letters, an underscore, four. The
    /// gate exists because the vocabulary would otherwise resolve `pl` to a BPE
    /// piece, so this is asserted directly rather than only through `nllb_code`.
    #[test]
    fn the_flores_shape_is_three_letters_an_underscore_and_four() {
        for good in ["pol_Latn", "eng_Latn", "zho_Hans", "yue_Hant", "arb_Arab"] {
            assert!(is_flores_code(good), "{good} must be accepted");
        }
        for bad in [
            "pl",
            "pol",
            "pol_latn2",
            "pol_Latn2",
            "POL_LATN",
            "pol-Latn",
            "pol_Latin",
            "pol_Lat",
            "pol_Latnn",
            "polx_Latn",
            "12_Latn",
            "_Latn",
            "pol_",
        ] {
            assert!(!is_flores_code(bad), "{bad} must not be accepted");
        }
    }

    /// Every row is FLORES-shaped, unique, and sorted: the file is read by people,
    /// and a duplicate would mean one of the two rows silently wins.
    #[test]
    fn the_code_table_is_well_formed() {
        let mut keys: Vec<&str> = NLLB_CODES.iter().map(|(iso, _)| *iso).collect();
        let sorted = {
            let mut s = keys.clone();
            s.sort_unstable();
            s
        };
        assert_eq!(keys, sorted, "the table must be sorted by ISO code");
        keys.dedup();
        assert_eq!(keys.len(), NLLB_CODES.len(), "no duplicate ISO code");
        let mut values: Vec<&str> = NLLB_CODES.iter().map(|(_, flores)| *flores).collect();
        values.sort_unstable();
        values.dedup();
        assert_eq!(
            values.len(),
            NLLB_CODES.len(),
            "no duplicate FLORES code: two ISO codes sharing one would make one of \
             them unspellable by its own name"
        );
        for (iso, flores) in NLLB_CODES {
            assert!(
                is_flores_code(flores),
                "{iso} -> {flores} is not FLORES-shaped"
            );
            // Two letters for an ISO-639-1 code, three for the one row that is not
            // one: Cantonese is already three letters everywhere. The *key* is
            // shorter than a FLORES code's language part on purpose — `af` maps to
            // `afr_Latn` — so only the mapped side is shape-checked.
            assert!(
                matches!(iso.len(), 2 | 3),
                "{iso} is neither an ISO-639-1 code nor the three-letter case"
            );
        }
        // The FLORES half keeps its capitals and the ISO half does not, so the two
        // halves are checked against their own conventions. `is_flores_code` is
        // case-preserving precisely because `token_to_id` is.
        assert!(
            NLLB_CODES
                .iter()
                .all(|(_, flores)| flores != &flores.to_ascii_lowercase()),
            "a FLORES code must keep its script capitals or token_to_id finds nothing"
        );
        assert!(
            NLLB_CODES
                .iter()
                .all(|(iso, _)| iso == &iso.to_ascii_lowercase()),
            "the ISO half is spelled lowercase throughout"
        );
    }

    /// A run's direction, resolved into the two codes a multilingual export is
    /// told. Every refusal names the fix.
    #[test]
    fn a_direction_becomes_two_codes_or_the_reason_it_cannot() {
        let pol_en = |src: Option<&str>, tgt: &str| nllb_codes(Direction { src, tgt });
        // `anyhow::Error` is not `PartialEq`, so a success is compared by value and
        // a failure by its message.
        let codes = |src: Option<&str>, tgt: &str| pol_en(src, tgt).map_err(|e| e.to_string());
        assert_eq!(
            codes(Some("pl"), "en"),
            Ok(("pol_Latn".to_string(), "eng_Latn".to_string()))
        );
        assert_eq!(
            codes(Some("pl"), "pol_Latn"),
            Ok(("pol_Latn".to_string(), "pol_Latn".to_string())),
            "a FLORES target passes through"
        );
        // `--language auto` is the one case with no answer: the source is unknown
        // per utterance and this model is told which way to go.
        let err = pol_en(None, "en")
            .expect_err("auto has no source")
            .to_string();
        assert!(err.contains("--language"), "err={err}");
        assert!(
            err.contains("--translate-model"),
            "the way out is named: {err}"
        );
        // An unknown target is `--translate`'s to fix, and the message says so.
        let err = pol_en(Some("pl"), "klingon")
            .expect_err("an unknown target must be refused")
            .to_string();
        assert!(err.contains("--translate"), "err={err}");
        assert!(err.contains("klingon"), "err={err}");
        // An unknown source is `--language`'s.
        let err = pol_en(Some("klingon"), "en")
            .expect_err("an unknown source must be refused")
            .to_string();
        assert!(err.contains("--language"), "err={err}");
        // The message shows both accepted spellings, so it is self-sufficient: a
        // FLORES code verbatim, and the ISO-639-1 code that maps to one.
        assert!(err.contains("pol_Latn"), "err={err}");
        assert!(err.contains(" -> "), "a mapped example is shown: {err}");
        // And it says *why* it refuses rather than passing the code through — which
        // would have resolved, and been wrong.
        assert!(err.contains("BPE"), "err={err}");
    }
}
