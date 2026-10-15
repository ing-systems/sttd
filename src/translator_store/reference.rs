//! From a `--translate-model` value or a direction to a prepared translator:
//! the references, the resolution rules and the entry points.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::TRANSLATE_EXAMPLE_PAIR;

use super::catalog::{
    has_pair, pair_for, sources_for, spec_for, valid_specs, valid_targets, TranslatorPair,
    NLLB_SPEC,
};
use super::facts::{read_config, Family};
use super::fetch::{
    ensure_available, missing_in, translator_dir, ASSETS, CONFIG_FILE, DECODER_FILE,
    DECODER_WITH_PAST_FILE, ENCODER_FILE, TOKENIZER_FILE,
};
use super::lang_codes::{nllb_codes, normalize};
use super::{Direction, Prepared, TranslatorRef};

/// Build the reference for a catalog pair: the pair names the direction, so
/// `src`/`tgt` are the catalog's own codes.
fn ref_for(pair: &TranslatorPair) -> TranslatorRef {
    let spec = spec_for(pair);
    TranslatorRef::in_dir(translator_dir(&spec), spec, pair.src, pair.tgt, ASSETS)
}

/// Build the reference for a multilingual spec.
///
/// No `src`/`tgt`: the direction is not a property of these weights, and filling
/// them in would put a lie in the log line that prints them.
fn multilingual_ref(spec: &str) -> TranslatorRef {
    TranslatorRef::in_dir(translator_dir(spec), spec.to_string(), "", "", ASSETS)
}

/// Resolve a source and a target language into a pair translator. A pair the
/// catalog does not hold is refused here, before any file is fetched.
pub(super) fn resolve_pair(src: &str, tgt: &str) -> Result<TranslatorRef> {
    let (src, tgt) = (normalize(src), normalize(tgt));
    if let Some(pair) = pair_for(&src, &tgt) {
        return Ok(ref_for(pair));
    }
    // Say what this source *can* go to. For `pl` that is exactly `en`, which is
    // the honest answer: no Polish-to-German pair export exists, and pretending
    // otherwise would send a user looking through the whole catalog. A source the
    // catalog never lists at all gets the whole list instead, because there is
    // nothing specific to point at.
    let targets = valid_targets(&src);
    if targets.is_empty() {
        anyhow::bail!(
            "no translator for {src} -> {tgt}; {src} is not a listed source language, \
             and no {src}-into-anything model is published. Available: {}",
            valid_specs()
        )
    }
    anyhow::bail!(
        "no translator for {src} -> {tgt}; {src} can be translated into: {targets}. \
         {NLLB_SPEC} translates every direction in one model -- \
         --translate-model {NLLB_SPEC}"
    )
}

/// The catalog row a pair spec names, or `None`. A pair spec is
/// `opus-mt-<src>-<tgt>` and must name a listed pair: the same string is both the
/// directory name and the repository name, so a spec that is not in the catalog
/// cannot be fetched either.
pub(super) fn pair_for_spec(spec: &str) -> Option<&'static TranslatorPair> {
    let rest = spec.strip_prefix("opus-mt-")?;
    let (src, tgt) = rest.split_once('-')?;
    pair_for(src, tgt)
}

/// Resolve a `--translate-model` value: a directory path (contains a path
/// separator) passes through untouched and is only validated; anything else is
/// an explicit spec, which must name a catalog pair or the multilingual export.
pub(crate) fn resolve(input: &str) -> Result<TranslatorRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!(
            "empty --translate-model value; expected a translator spec \
             (e.g. {TRANSLATE_EXAMPLE_PAIR}, {NLLB_SPEC}) or a directory holding \
             {ENCODER_FILE}, {DECODER_FILE}, {DECODER_WITH_PAST_FILE}, \
             {TOKENIZER_FILE} and {CONFIG_FILE}"
        );
    }
    if crate::fsutil::looks_like_path(trimmed) {
        let dir = PathBuf::from(trimmed);
        if let Some(missing) = missing_in(&dir) {
            anyhow::bail!(
                "translator path {trimmed:?} is not a translator directory: {} is missing",
                missing.display()
            );
        }
        // A user directory names no catalog pair, so the codes are empty: the
        // direction is a property of the weights, and sttd does not read a
        // direction out of a directory name. Its own name is the identity.
        return Ok(TranslatorRef::in_dir(
            dir.clone(),
            dir.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| trimmed.to_string()),
            "",
            "",
            &[],
        ));
    }
    let spec = normalize(trimmed);
    // `nllb-200-distilled-600M` is matched case-insensitively so a spec typed in
    // any case still resolves, but the *canonical* spelling is what names the
    // repository and the directory: the comparison value is not reused for either.
    if spec == NLLB_SPEC.to_ascii_lowercase() {
        return Ok(multilingual_ref(NLLB_SPEC));
    }
    let pair = pair_for_spec(&spec).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown translator {trimmed:?}; expected opus-mt-<src>-<tgt> or \
             {NLLB_SPEC}, one of: {}",
            valid_specs()
        )
    })?;
    Ok(ref_for(pair))
}

/// Download whatever is missing and read the config, in one step. Shared by both
/// entry points below, so a directory and a catalog spec cannot diverge.
fn finish(reference: TranslatorRef, direction: Direction<'_>) -> Result<Prepared> {
    ensure_available(&reference)?;
    let facts = read_config(&reference)?;
    // Resolved *after* the read, because only the read knows the family: a
    // `--translate-model` pointing at a directory names neither, and the same
    // code has to serve a catalog spec and a user-supplied export.
    let codes = match facts.family {
        Family::Marian => None,
        Family::Nllb => Some(nllb_codes(direction)?),
    };
    Ok(Prepared {
        reference,
        facts,
        codes,
    })
}

/// Resolve `--translate-model` and download whatever is missing, in one call: the
/// startup-time path, where an unknown spec or an unwritable directory has to
/// fail fast like the speech-model load it sits beside.
///
/// `direction` is only read for a multilingual export; a pair export takes its
/// direction from which weights were loaded and ignores it.
pub(crate) fn prepare(input: &str, direction: Direction<'_>) -> Result<Prepared> {
    finish(resolve(input)?, direction)
}

/// Which translator files a direction names, resolved **without fetching
/// anything**.
///
/// Split out of [`prepare_from_direction`] so a caller that only needs to know
/// whether a direction is possible can get the very same refusal the startup
/// path prints, without touching the network: the bus asks, and a load takes
/// seconds to minutes. Pure apart from the existence check a
/// `--translate-model` **directory** needs, so a run pointing at one still hears
/// about a missing file here rather than after a download.
pub(crate) fn reference_for(
    direction: Direction<'_>,
    model: Option<&str>,
) -> Result<TranslatorRef> {
    match model.map(str::trim).filter(|spec| !spec.is_empty()) {
        Some(spec) => resolve(spec),
        None => default_reference_for(direction),
    }
}

/// The translator a run that named no spec resolves to: the pair `--language`
/// and `--translate` resolve to.
///
/// `--language auto` is the one case that needs a rule rather than a lookup: the
/// source is unknown per utterance, and a pair export is a fixed direction, so
/// there is no pair to name. Into English the answer is `opus-mt-mul-en`,
/// upstream's many-sources-into-English model. Into anything else there is no
/// honest answer — `opus-mt-<src>-<tgt>` would have to guess the source — so it
/// refuses and says which targets have a many-source model.
///
/// Coverage is thin in one direction and the message says so: Polish has exactly
/// one exported pair upstream, so `--language pl --translate de` is an error naming
/// `en`, not a mistranslation.
fn default_reference_for(direction: Direction<'_>) -> Result<TranslatorRef> {
    match direction.src {
        Some(src) => resolve_pair(src, direction.tgt),
        None => {
            if has_pair("mul", direction.tgt) {
                log::info!(
                    "--language auto leaves the source unknown; using the many-source \
                     model (opus-mt-mul-{}). Pass --language CODE for a pair specific \
                     to your language, or --translate-model {NLLB_SPEC} for any direction",
                    direction.tgt
                );
                resolve_pair("mul", direction.tgt)
            } else {
                // No many-source model into this target: the direction is the
                // model's identity, so there is nothing to fall back to. Name the
                // sources that do work, since one of them is probably the answer.
                let sources = sources_for(direction.tgt);
                let named = if sources.is_empty() {
                    String::new()
                } else {
                    format!(
                        " Languages that can be translated into {}: {}.",
                        direction.tgt,
                        sources.join(", ")
                    )
                };
                anyhow::bail!(
                    "--language auto cannot translate into {}: the only pair model covering \
                     an unknown source is opus-mt-mul-en.{named} Or pass --language CODE to \
                     name the source (e.g. --language pl --translate en), or \
                     --translate-model {NLLB_SPEC}, which needs a named source and \
                     translates any direction",
                    direction.tgt
                )
            }
        }
    }
}

/// The translator for a run that named no spec, downloaded if it is not on disk
/// yet: the startup-time entry point beside [`prepare`].
pub(crate) fn prepare_from_direction(direction: Direction<'_>) -> Result<Prepared> {
    finish(default_reference_for(direction)?, direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator_store::catalog::CATALOG;
    use crate::translator_store::test_support::*;

    #[test]
    fn every_catalog_row_resolves_and_names_its_own_repository() {
        // The catalog is the whole availability check: a row upstream does not
        // publish is a 404 on first use, so each one is asserted here rather
        // than discovered by a user.
        assert!(!CATALOG.is_empty(), "an empty catalog translates nothing");
        for pair in CATALOG {
            let t = resolve_pair(pair.src, pair.tgt)
                .unwrap_or_else(|e| panic!("{} -> {} must resolve: {e}", pair.src, pair.tgt));
            assert_eq!(t.spec, spec_for(pair), "spec is the repo name");
            assert_eq!(t.src, pair.src);
            assert_eq!(t.tgt, pair.tgt);
            assert_eq!(t.dir, translator_dir(&t.spec));
            assert_eq!(t.encoder, translator_dir(&t.spec).join(ENCODER_FILE));
            assert_eq!(t.decoder, translator_dir(&t.spec).join(DECODER_FILE));
            assert_eq!(
                t.decoder_with_past,
                translator_dir(&t.spec).join(DECODER_WITH_PAST_FILE),
                "the cached decoder is fetched like the other two graphs"
            );
            assert_eq!(t.tokenizer, translator_dir(&t.spec).join(TOKENIZER_FILE));
            assert_eq!(t.assets, ASSETS.to_vec(), "one file set per pair");
            // The spec round-trips back to the same row, so a directory name
            // and a `--translate-model` value are interchangeable.
            let back = pair_for_spec(&t.spec).expect("a catalog spec must parse back");
            assert_eq!(back, pair);
        }
    }

    #[test]
    fn an_unlisted_pair_is_refused_naming_what_the_source_can_reach() {
        // The message names targets, not pairs: the user typed `--translate de`,
        // so the fix is a target code, not a whole spec in the other direction.
        let err = resolve_pair("pl", "de")
            .expect_err("pl-de is not published")
            .to_string();
        assert!(err.contains("pl -> de"), "err={err}");
        assert!(
            err.contains("pl can be translated into: en"),
            "the only Polish target must be named: {err}"
        );
        assert!(
            !err.contains("opus-mt-de-en"),
            "the wrong direction is not a fix for this: {err}"
        );
        // And the way out is named, because there is one.
        assert!(err.contains(NLLB_SPEC), "err={err}");

        // A source with several targets lists them all, sorted. `en-pt` is not
        // published upstream, unlike most `en-*` pairs.
        let err = resolve_pair("en", "pt")
            .expect_err("en-pt is not published")
            .to_string();
        assert!(err.contains("en -> pt"), "err={err}");
        let targets = valid_targets("en");
        assert!(
            targets.contains("de") && targets.contains("zh"),
            "{targets}"
        );
        let listed: Vec<&str> = targets.split(", ").collect();
        let mut sorted = listed.clone();
        sorted.sort_unstable();
        assert_eq!(listed, sorted, "the target list must be sorted: {targets}");

        // A source the catalog never lists gets the whole catalog instead,
        // because there is nothing specific to point at.
        for src in ["pt", "ro", "yue"] {
            let err = resolve_pair(src, "en")
                .expect_err("an unlisted source must not resolve")
                .to_string();
            assert!(err.contains(&format!("{src} -> en")), "err={err}");
            assert!(err.contains("opus-mt-pl-en"), "err={err}");
            assert!(err.contains("not a listed source"), "err={err}");
            assert!(err.contains(NLLB_SPEC), "err={err}");
        }
        // `pl` into a target it does have resolves, so the refusal above is about
        // the pair and not about Polish being unsupportable.
        assert!(resolve_pair("pl", "en").is_ok());
        // And `en-pl` is refused in the same way, from the other side.
        let err = resolve_pair("en", "pl")
            .expect_err("en-pl is not published")
            .to_string();
        assert!(err.contains("en -> pl"), "err={err}");
        assert!(err.contains("en can be translated into"), "err={err}");
    }

    #[test]
    fn a_spec_is_refused_unless_it_names_a_listed_pair_or_the_multilingual_export() {
        // The spec form of the same check: a directory name and a
        // `--translate-model` value are the same string, so an unlisted spec is
        // both unresolvable and unfetchable.
        for bad in ["opus-mt-en-pl", "opus-mt-pl-de", "opus-mt-pt-en"] {
            let err = resolve(bad)
                .expect_err("an unlisted spec must not resolve")
                .to_string();
            assert!(err.contains("opus-mt-pl-en"), "{bad}: {err}");
            assert!(err.contains(NLLB_SPEC), "{bad}: {err}");
        }
    }

    #[test]
    fn codes_normalize_before_the_catalog_is_asked() {
        // Case, surrounding whitespace and `_` are the user's, not the
        // catalog's: each of these is the same pair as `pl-en`.
        let pair = pair_for("pl", "en").unwrap();
        for (src, tgt) in [("pl", "en"), ("PL", "EN"), ("  pl  ", "en"), ("pl", "EN")] {
            let t = resolve_pair(src, tgt).unwrap_or_else(|e| panic!("{src}->{tgt}: {e}"));
            assert_eq!(t.spec, spec_for(pair));
        }
        assert_eq!(normalize("pt_BR"), "pt-br");
    }

    /// The multilingual spec resolves, and **names no direction**. `src`/`tgt`
    /// stay empty on purpose: the direction is not a property of these weights,
    /// and filling them in would put a lie in the log line that prints them.
    #[test]
    fn the_multilingual_spec_resolves_with_no_direction_of_its_own() {
        let t = resolve(NLLB_SPEC).expect("the multilingual spec must resolve");
        assert_eq!(t.spec, NLLB_SPEC);
        assert_eq!(t.dir, translator_dir(NLLB_SPEC));
        assert_eq!(t.encoder, translator_dir(NLLB_SPEC).join(ENCODER_FILE));
        assert_eq!(t.assets, ASSETS.to_vec());
        assert!(t.src.is_empty() && t.tgt.is_empty());
        // Matched case-insensitively, but the *canonical* spelling is what names
        // the repository and the directory: the lowercased match value is not
        // reused for either.
        for typed in [
            "nllb-200-distilled-600M",
            "NLLB-200-DISTILLED-600M",
            "  nllb-200-distilled-600m ",
        ] {
            let t = resolve(typed).unwrap_or_else(|e| panic!("{typed:?}: {e}"));
            assert_eq!(
                t.spec, NLLB_SPEC,
                "{typed:?} must normalize to the canonical spec"
            );
            assert_eq!(t.dir, translator_dir(NLLB_SPEC), "{typed:?}");
        }
    }

    #[test]
    fn a_directory_path_passes_through_and_is_never_downloaded_into() {
        let dir = temp_dir("path");
        write_dir(&dir, &ALL_FILES, MARIAN_CONFIG);
        let t = resolve(dir.to_str().unwrap()).expect("must resolve");
        assert_eq!(
            t.spec,
            dir.file_name().unwrap().to_string_lossy(),
            "the directory's own name is the identity on a path run"
        );
        assert_eq!(t.encoder, dir.join(ENCODER_FILE));
        assert_eq!(t.decoder, dir.join(DECODER_FILE));
        assert_eq!(t.decoder_with_past, dir.join(DECODER_WITH_PAST_FILE));
        assert_eq!(t.tokenizer, dir.join(TOKENIZER_FILE));
        assert!(t.assets.is_empty(), "a path is validated, never fetched");
        assert!(
            t.src.is_empty() && t.tgt.is_empty(),
            "a directory names no catalog pair"
        );
        // Nothing to download, so validation passes and the facts come out of the
        // config on disk.
        let prepared = prepare(
            dir.to_str().unwrap(),
            Direction {
                src: None,
                tgt: "en",
            },
        )
        .expect("a complete directory must validate");
        assert_eq!(prepared.facts.family, Family::Marian);
        assert_eq!(prepared.facts.start_token, 63429);
        assert_eq!(prepared.facts.eos_token, 0);
        assert_eq!(prepared.facts.pad_token, Some(63429));
        assert!(
            prepared.codes.is_none(),
            "a pair export has no codes to pass"
        );

        // A directory missing its decoder is not a translator: refused at
        // resolution, before anything is opened, with the file named.
        std::fs::remove_file(dir.join(DECODER_FILE)).unwrap();
        let err = resolve(dir.to_str().unwrap())
            .expect_err("a directory without a decoder graph is not a translator")
            .to_string();
        assert!(err.contains(DECODER_FILE), "err={err}");

        // The cached decoder is required too, and that is a real cost of running a
        // decode through two graphs: a directory holding the other four files is no
        // longer a translator. Refused by name, the same way, rather than loading
        // and failing on a missing session.
        std::fs::write(dir.join(DECODER_FILE), b"x").unwrap();
        std::fs::remove_file(dir.join(DECODER_WITH_PAST_FILE)).unwrap();
        let err = resolve(dir.to_str().unwrap())
            .expect_err("a directory without the cached decoder is not a translator")
            .to_string();
        assert!(err.contains(DECODER_WITH_PAST_FILE), "err={err}");

        // Neither is a directory with an empty one in it, nor one that does not
        // exist.
        let empty = temp_dir("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve(empty.to_str().unwrap()).is_err());
        assert!(resolve("/nonexistent/translator").is_err());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    /// A directory holding an NLLB export is served by the same path as the
    /// catalog spec, and is told which direction to run — because the family comes
    /// from the file, not from a name. This is the case a spec-string lookup would
    /// get wrong, and the reason [`Family`] is read rather than matched.
    #[test]
    fn a_directory_is_told_its_family_by_its_own_config() {
        let dir = temp_dir("dir-nllb");
        write_dir(&dir, &ALL_FILES, NLLB_CONFIG);
        let path = dir.to_str().unwrap();

        // With a source: read as NLLB, and the direction becomes two codes.
        let prepared = prepare(
            path,
            Direction {
                src: Some("pl"),
                tgt: "de",
            },
        )
        .expect("an NLLB directory must prepare");
        assert_eq!(prepared.facts.family, Family::Nllb);
        assert_eq!(prepared.facts.shape.cache_slots(), 48);
        assert_eq!(
            prepared.codes,
            Some(("pol_Latn".to_string(), "deu_Latn".to_string())),
            "the direction is taken from the run, not from the weights"
        );

        // Without one: refused, because this family has no many-source fallback.
        let err = prepare(
            path,
            Direction {
                src: None,
                tgt: "de",
            },
        )
        .expect_err("an NLLB directory cannot run without a source")
        .to_string();
        assert!(err.contains("--language"), "err={err}");

        // The same directory with a Marian config is a different family entirely,
        // with no codes to pass and the start-token rule back in force.
        write_dir(&dir, &ALL_FILES, MARIAN_CONFIG);
        let prepared = prepare(
            path,
            Direction {
                src: Some("pl"),
                tgt: "de",
            },
        )
        .expect("the same directory must prepare as Marian");
        assert_eq!(prepared.facts.family, Family::Marian);
        assert_eq!(prepared.facts.shape.cache_slots(), 24);
        assert!(prepared.codes.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_and_unknown_values_are_refused_with_the_expected_form_named() {
        let err = resolve("   ")
            .expect_err("an empty value must not resolve")
            .to_string();
        assert!(err.contains("--translate-model"), "err={err}");
        assert!(err.contains(NLLB_SPEC), "err={err}");
        // A name that is not a spec at all, and one that is a spec shape for a
        // pair that does not exist.
        for bad in ["flan-t5-base", "opus-mt-xx-yy", "opus-mt-pl", "pl-en"] {
            let err = resolve(bad).expect_err("must not resolve").to_string();
            assert!(err.contains("opus-mt-pl-en"), "{bad}: err={err}");
        }
        // An empty pair code is refused the same way, not silently defaulted.
        assert!(resolve_pair("pl", "").is_err());
        assert!(resolve_pair("", "en").is_err());
        // And an NLLB direction needs both halves.
        assert!(nllb_codes(Direction {
            src: Some("pl"),
            tgt: ""
        })
        .is_err());
    }
}
