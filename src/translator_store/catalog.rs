//! The pair catalog: which MarianMT exports exist, by source and target.

use super::lang_codes::normalize;

/// One language pair the Marian catalog can translate.
///
/// Only meaningful for a pair export. A multilingual export names no pair —
/// see [`NLLB_SPEC`] and [`Direction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TranslatorPair {
    pub(crate) src: &'static str,
    pub(crate) tgt: &'static str,
}

/// The pairs verified to exist as ONNX exports under [`OPUS_MT_REPO`].
///
/// Coverage is lopsided upstream and the catalog says so rather than implying
/// more: of the 76 published exports, 25 are sources into English and 20 are
/// targets from it. **`pl` is the only Polish source with an export at all** —
/// `Helsinki-NLP` publishes ten `pl-*` models (`pl-ar`, `pl-de`, `pl-en`,
/// `pl-eo`, `pl-es`, `pl-fr`, `pl-lt`, `pl-no`, `pl-sv`, `pl-uk`) and none of the
/// other nine has been converted, and there is no `X-pl` pair in either direction.
///
/// `mul-en` is upstream's many-sources-into-English model and is what
/// `--language auto` resolves to. `en-mul` is deliberately absent: `mul` is not a
/// language a user can name through `--translate`, and the model picks its target
/// per sentence, so it cannot honour the flag.
///
/// Pairs upstream does not publish (`en-pl`, `pt-en`, `en-pt`, `ro-en`, `el-en`)
/// are absent on purpose, so `--translate <code>` fails fast instead of pulling a
/// model that cannot translate. The one way out is [`NLLB_SPEC`], which is not a
/// pair export and has every direction.
pub(crate) const CATALOG: &[TranslatorPair] = &[
    // Polish has exactly one ONNX-exported pair upstream: `Helsinki-NLP`
    // publishes ten `pl-*` models, but only this one has been converted. This is
    // the primary path, and `--translate` into any other language from Polish
    // fails with the valid targets named rather than silently mistranslating --
    // unless the run names `nllb-200-distilled-600M`, which can.
    TranslatorPair {
        src: "pl",
        tgt: "en",
    },
    // The `--language auto` fallback: many supported sources into English.
    TranslatorPair {
        src: "mul",
        tgt: "en",
    },
    TranslatorPair {
        src: "de",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "de",
    },
    TranslatorPair {
        src: "fr",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "fr",
    },
    TranslatorPair {
        src: "es",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "es",
    },
    TranslatorPair {
        src: "it",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "it",
    },
    TranslatorPair {
        src: "uk",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "uk",
    },
    TranslatorPair {
        src: "ru",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "ru",
    },
    TranslatorPair {
        src: "cs",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "cs",
    },
    TranslatorPair {
        src: "nl",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "nl",
    },
    TranslatorPair {
        src: "sv",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "sv",
    },
    TranslatorPair {
        src: "da",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "da",
    },
    TranslatorPair {
        src: "tr",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "hu",
    },
    TranslatorPair {
        src: "en",
        tgt: "fi",
    },
    TranslatorPair {
        src: "et",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "zh",
    },
    TranslatorPair {
        src: "zh",
        tgt: "en",
    },
    TranslatorPair {
        src: "ja",
        tgt: "en",
    },
    TranslatorPair {
        src: "ko",
        tgt: "en",
    },
    TranslatorPair {
        src: "ar",
        tgt: "en",
    },
    TranslatorPair {
        src: "en",
        tgt: "ar",
    },
    TranslatorPair {
        src: "hi",
        tgt: "en",
    },
    TranslatorPair {
        src: "th",
        tgt: "en",
    },
    TranslatorPair {
        src: "vi",
        tgt: "en",
    },
    TranslatorPair {
        src: "af",
        tgt: "en",
    },
    TranslatorPair {
        src: "id",
        tgt: "en",
    },
];

/// The multilingual export: one repository, every direction.
///
/// `Xenova/nllb-200-distilled-600M`, the 600M distilled NLLB-200 checkpoint.
/// Unlike a Marian pair it is *not* a direction: it is told which way to
/// translate, so the run's own `--language` and `--translate` are required
/// rather than inferred. ~870 MiB against a Marian pair's ~114 MiB, and a
/// heavier decode (12 layers of `d_model` 1024 against 6 of 512) — see
/// `translate::engine::Translator`.
pub(crate) const NLLB_SPEC: &str = "nllb-200-distilled-600M";

/// The one pair export the tests that need real weights can rely on being present
/// on a machine that has them: the Polish-to-English path `--language pl
/// --translate en` resolves to, and the one pair the README calls primary.
#[cfg(test)]
pub(crate) const PL_EN_SPEC: &str = "opus-mt-pl-en";

/// Every spec a run may name, multilingual exports included, sorted. The one
/// place the set of downloadable translators is written down, so `--help`,
/// a refusal message and [`resolve`] cannot disagree about it.
pub(super) fn valid_specs() -> String {
    let mut specs: Vec<String> = CATALOG.iter().map(spec_for).collect();
    specs.push(NLLB_SPEC.to_string());
    specs.sort();
    specs.join(", ")
}

/// The catalog row for `src` into `tgt`, or `None` when the pair is not listed.
pub(super) fn pair_for(src: &str, tgt: &str) -> Option<&'static TranslatorPair> {
    CATALOG.iter().find(|p| p.src == src && p.tgt == tgt)
}

/// Whether the catalog holds a `src` into `tgt` model, without downloading one.
///
/// The `--language auto` path needs this to ask a question before it commits to a
/// download: `opus-mt-mul-en` exists and `opus-mt-mul-de` does not, and the second
/// case has to be an error rather than a 404.
pub(super) fn has_pair(src: &str, tgt: &str) -> bool {
    let (src, tgt) = (normalize(src), normalize(tgt));
    pair_for(&src, &tgt).is_some()
}

/// The sources the catalog can translate into `tgt`, sorted.
///
/// The counterpart to [`valid_targets`], for the case where the *target* is what
/// the user typed and the source is unknown.
pub(super) fn sources_for(tgt: &str) -> Vec<&'static str> {
    let tgt = normalize(tgt);
    let mut sources: Vec<&'static str> = CATALOG
        .iter()
        .filter(|p| p.tgt == tgt)
        .map(|p| p.src)
        .collect();
    sources.sort_unstable();
    sources
}

/// The languages the catalog can translate `src` into, sorted and comma
/// separated.
///
/// The user typed `--translate <tgt>`, so the useful part of a refusal is what
/// they *could* have typed for that source. Naming whole pairs instead would
/// answer a question they did not ask: `opus-mt-de-en` is not a fix for
/// `--language pl --translate de`.
pub(super) fn valid_targets(src: &str) -> String {
    let src = normalize(src);
    let mut targets: Vec<&'static str> = CATALOG
        .iter()
        .filter(|p| p.src == src)
        .map(|p| p.tgt)
        .collect();
    targets.sort_unstable();
    if targets.is_empty() {
        return String::new();
    }
    targets.join(", ")
}

/// The spec for a pair: the upstream repository name, unchanged. Keeping the
/// two the same string is what lets a log line, a directory name and a
/// `--translate-model` value all be read interchangeably.
pub(crate) fn spec_for(pair: &TranslatorPair) -> String {
    format!("opus-mt-{}-{}", pair.src, pair.tgt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator_store::reference::resolve_pair;

    #[test]
    fn every_catalog_source_can_reach_at_least_one_target() {
        // A row whose source is unreachable from any flag would be dead weight,
        // and `valid_targets` returning empty is the shape of that bug.
        for pair in CATALOG {
            assert!(
                !valid_targets(pair.src).is_empty(),
                "{} is listed but {} reaches nothing",
                spec_for(pair),
                pair.src
            );
        }
        // Polish is the case the coverage limit is about: exactly one target.
        assert_eq!(valid_targets("pl"), "en");
        assert_eq!(valid_targets("PL"), "en", "the same list after normalizing");
        assert!(valid_targets("klingon").is_empty());
    }

    #[test]
    fn mul_is_reachable_as_a_source_and_never_as_a_target() {
        // `--language auto --translate en` resolves to `mul-en`, so `mul` must be
        // listed as a source.
        assert!(pair_for("mul", "en").is_some());
        // `en-mul` is not in the catalog: `mul` is not a language a user can name
        // through `--translate`, and that model picks its own target per sentence,
        // so it could not honour the flag.
        assert!(pair_for("en", "mul").is_none());
        assert!(resolve_pair("en", "mul").is_err());
    }
}
