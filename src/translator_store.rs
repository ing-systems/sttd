// ---------------------------------------------------------------------------
// Translators: `--translate-model` resolution and download of the ONNX export.
// ---------------------------------------------------------------------------
//
// One flag, `--translate <LANG>`, turns transcripts into another language; the
// weights behind it come from here. The models are MarianMT (`opus-mt`)
// exports, published as one repository per language pair, so the catalog here
// is a list of pairs and a spec is the upstream repository name
// (`opus-mt-<src>-<tgt>`). Only pairs verified to exist are listed; anything
// else is refused at resolve, before a download starts, with the valid list in
// the message.
//
// Layout mirrors `voice_store` deliberately: a spec resolves into
// `<models>/translators/<spec>/` with the three files sttd reads flat on disk,
// whatever nesting the repository uses upstream (`onnx/encoder_model_quantized.onnx`).
// The translate step never sees the repository layout, and no renaming happens
// after download.
//
// Deliberately *not* part of `model_store`: a translator is not a speech
// engine, so it must not be reachable through `--model`, and `model_store`'s
// engine detection (`EngineKind::marker_file`) must never classify one.
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{
    DEFAULT_MODELS_DIR, OPUS_MT_REPO, TRANSLATE_DEFAULT_PAIR, TRANSLATORS_SUBDIR,
};

/// Encoder graph, inside a translator directory and inside the repository
/// (`onnx/`).
const ENCODER_FILE: &str = "encoder_model_quantized.onnx";
/// Merged decoder graph (prompt step and later steps in one graph), same two
/// locations. The `decoder_with_past` variant is deliberately not used.
const DECODER_FILE: &str = "decoder_model_merged_quantized.onnx";
/// SentencePiece tokenizer, at the repository root.
const TOKENIZER_FILE: &str = "tokenizer.json";
/// The export's `config.json`, also at the repository root. ~2 KiB, and the only
/// place the decoder's opening and stopping token ids are published: they are
/// per-pair (63429 for `pl-en`, 64171 for `mul-en`) and compiling either in means
/// every other pair decodes garbage.
const CONFIG_FILE: &str = "config.json";

/// The four files every pair needs, as `(repo path, file name)`. The repository
/// nests the graphs under `onnx/` and keeps the tokenizer and the config at the
/// root; on disk all are flat. One file set for the whole catalog, so a constant
/// rather than a per-pair list.
const ASSETS: &[(&str, &str)] = &[
    ("onnx/encoder_model_quantized.onnx", ENCODER_FILE),
    (
        "onnx/decoder_model_merged_quantized.onnx",
        DECODER_FILE,
    ),
    ("tokenizer.json", TOKENIZER_FILE),
    ("config.json", CONFIG_FILE),
];

/// One language pair the catalog can translate.
///
/// Upstream publishes one repository per pair, so this is the whole identity of
/// a translator: there is no size or digest row to keep, unlike the voices
/// index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TranslatorPair {
    pub(crate) src: &'static str,
    pub(crate) tgt: &'static str,
}

/// The pairs verified to exist as ONNX exports under `OPUS_MT_REPO`.
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
/// model that cannot translate.
pub(crate) const CATALOG: &[TranslatorPair] = &[
    // Polish has exactly one ONNX-exported pair upstream: `Helsinki-NLP`
    // publishes ten `pl-*` models, but only this one has been converted. This is
    // the primary path, and `--translate` into any other language from Polish
    // fails with the valid targets named rather than silently mistranslating.
    TranslatorPair { src: "pl", tgt: "en" },
    // The `--language auto` fallback: many supported sources into English.
    TranslatorPair { src: "mul", tgt: "en" },
    TranslatorPair { src: "de", tgt: "en" },
    TranslatorPair { src: "en", tgt: "de" },
    TranslatorPair { src: "fr", tgt: "en" },
    TranslatorPair { src: "en", tgt: "fr" },
    TranslatorPair { src: "es", tgt: "en" },
    TranslatorPair { src: "en", tgt: "es" },
    TranslatorPair { src: "it", tgt: "en" },
    TranslatorPair { src: "en", tgt: "it" },
    TranslatorPair { src: "uk", tgt: "en" },
    TranslatorPair { src: "en", tgt: "uk" },
    TranslatorPair { src: "ru", tgt: "en" },
    TranslatorPair { src: "en", tgt: "ru" },
    TranslatorPair { src: "cs", tgt: "en" },
    TranslatorPair { src: "en", tgt: "cs" },
    TranslatorPair { src: "nl", tgt: "en" },
    TranslatorPair { src: "en", tgt: "nl" },
    TranslatorPair { src: "sv", tgt: "en" },
    TranslatorPair { src: "en", tgt: "sv" },
    TranslatorPair { src: "da", tgt: "en" },
    TranslatorPair { src: "en", tgt: "da" },
    TranslatorPair { src: "tr", tgt: "en" },
    TranslatorPair { src: "en", tgt: "hu" },
    TranslatorPair { src: "en", tgt: "fi" },
    TranslatorPair { src: "et", tgt: "en" },
    TranslatorPair { src: "en", tgt: "zh" },
    TranslatorPair { src: "zh", tgt: "en" },
    TranslatorPair { src: "ja", tgt: "en" },
    TranslatorPair { src: "ko", tgt: "en" },
    TranslatorPair { src: "ar", tgt: "en" },
    TranslatorPair { src: "en", tgt: "ar" },
    TranslatorPair { src: "hi", tgt: "en" },
    TranslatorPair { src: "th", tgt: "en" },
    TranslatorPair { src: "vi", tgt: "en" },
    TranslatorPair { src: "af", tgt: "en" },
    TranslatorPair { src: "id", tgt: "en" },
];

/// The catalog row for `src` into `tgt`, or `None` when the pair is not listed.
fn pair_for(src: &str, tgt: &str) -> Option<&'static TranslatorPair> {
    CATALOG.iter().find(|p| p.src == src && p.tgt == tgt)
}

/// Whether the catalog holds a `src` into `tgt` model, without downloading one.
///
/// The `--language auto` path needs this to ask a question before it commits to a
/// download: `opus-mt-mul-en` exists and `opus-mt-mul-de` does not, and the second
/// case has to be an error rather than a 404.
pub(crate) fn has_pair(src: &str, tgt: &str) -> bool {
    let (src, tgt) = (normalize(src), normalize(tgt));
    pair_for(&src, &tgt).is_some()
}

/// The sources the catalog can translate into `tgt`, sorted.
///
/// The counterpart to [`valid_targets`], for the case where the *target* is what
/// the user typed and the source is unknown.
pub(crate) fn sources_for(tgt: &str) -> Vec<&'static str> {
    let tgt = normalize(tgt);
    let mut sources: Vec<&'static str> = CATALOG
        .iter()
        .filter(|p| p.tgt == tgt)
        .map(|p| p.src)
        .collect();
    sources.sort_unstable();
    sources
}

/// The catalog specs, sorted and comma separated, for a refusal message. The
/// user asked for a pair that cannot exist, so the fix is in the list, not in
/// the wording.
fn valid_pairs() -> String {
    let mut specs: Vec<String> = CATALOG.iter().map(spec_for).collect();
    specs.sort();
    specs.join(", ")
}

/// The languages the catalog can translate `src` into, sorted and comma
/// separated.
///
/// The user typed `--translate <tgt>`, so the useful part of a refusal is what
/// they *could* have typed for that source. Naming whole pairs instead would
/// answer a question they did not ask: `opus-mt-de-en` is not a fix for
/// `--language pl --translate de`.
fn valid_targets(src: &str) -> String {
    let src = normalize(src);
    let mut targets: Vec<&str> = CATALOG
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

/// Directory every downloaded translator resolves into.
pub(crate) fn translators_dir() -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{TRANSLATORS_SUBDIR}"))
}

/// Directory one translator spec resolves into.
pub(crate) fn translator_dir(spec: &str) -> PathBuf {
    translators_dir().join(spec)
}

/// The spec for a pair: the upstream repository name, unchanged. Keeping the
/// two the same string is what lets a log line, a directory name and a
/// `--translate-model` value all be read interchangeably.
pub(crate) fn spec_for(pair: &TranslatorPair) -> String {
    format!("opus-mt-{}-{}", pair.src, pair.tgt)
}

/// Where one repository is fetched from. `resolve/main` rather than `main` so a
/// spec pinned to a revision upstream resolves.
fn repo_base(spec: &str) -> String {
    format!("https://huggingface.co/{OPUS_MT_REPO}/{spec}/resolve/main/")
}

/// The repository one pair lives in: the spec is the repository name, so this is
/// the same string `repo_base` builds.
pub(crate) fn repo_for(pair: &TranslatorPair) -> String {
    repo_base(&spec_for(pair))
}

/// Where a published file of `pair` is fetched from.
///
/// The one place a download URL is built, so a token in [`ASSETS`] and the URL it
/// resolves to cannot drift apart.
pub(crate) fn asset_url(pair: &TranslatorPair, repo_path: &str) -> String {
    format!("{}{repo_path}", repo_for(pair))
}

/// A resolved `--translate-model`: which pair it is, where its three files
/// live, and what still has to be fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslatorRef {
    /// Upstream spec (`opus-mt-pl-en`).
    pub(crate) spec: String,
    /// Source language code, as the catalog spells it.
    pub(crate) src: String,
    /// Target language code, as the catalog spells it.
    pub(crate) tgt: String,
    pub(crate) dir: PathBuf,
    /// Encoder graph inside `dir`.
    pub(crate) encoder: PathBuf,
    /// Merged decoder graph inside `dir`.
    pub(crate) decoder: PathBuf,
    /// Tokenizer inside `dir`.
    pub(crate) tokenizer: PathBuf,
    /// The export's `config.json` inside `dir`.
    pub(crate) config: PathBuf,
    /// `(repo path, file name)` pairs to fetch when missing. Empty for a
    /// user-supplied directory, which is validated but never downloaded into.
    pub(crate) assets: Vec<(&'static str, &'static str)>,
    /// `decoder_start_token_id`: the token the decoder opens its sequence with.
    ///
    /// **Per pair, not a constant.** `pl-en` pads at 63429 and `mul-en` at 64171,
    /// and both are also `pad_token_id`. Feeding the wrong one is not an error —
    /// the graph runs and returns punctuation (`"..... .."`), which is why the id
    /// is read from the model's own config at load rather than written down here.
    pub(crate) decoder_start_token_id: i64,
    /// `eos_token_id`: where a decode stops. 0 (`</s>`) in both pairs checked,
    /// but read from the same place as the start token rather than assumed.
    pub(crate) eos_token_id: i64,
    /// Ids a decode may never emit: `pad_token_id`, plus the start token.
    ///
    /// The pad id must be suppressed, and this is not a precaution. The int8
    /// `opus-mt-mul-en` export pins the pad logit at exactly `0.000` while every
    /// real token sits around `-40`, so a plain argmax picks `<pad>` on every step
    /// and the decode runs to the token cap emitting nothing. Upstream's
    /// `generation_config` lists the same id under `bad_words_ids`, which is how
    /// the reference implementation avoids it. Pad is never part of an answer.
    pub(crate) suppressed_ids: Vec<i64>,
}

/// A language code as the catalog spells it: trimmed, lowercased, with `_` for
/// `-` so `pt_BR`-shaped input and `pt-br` reach the same comparison.
fn normalize(code: &str) -> String {
    code.trim().to_ascii_lowercase().replace('_', "-")
}

/// Build the reference for a catalog pair.
fn ref_for(pair: &TranslatorPair) -> TranslatorRef {
    let spec = spec_for(pair);
    let dir = translator_dir(&spec);
    TranslatorRef {
        encoder: dir.join(ENCODER_FILE),
        decoder: dir.join(DECODER_FILE),
        tokenizer: dir.join(TOKENIZER_FILE),
        config: dir.join(CONFIG_FILE),
        dir,
        assets: ASSETS.to_vec(),
        spec,
        src: pair.src.to_string(),
        tgt: pair.tgt.to_string(),
        // Placeholders: `prepare` overwrites these from the downloaded config
        // before anything is loaded, and `Translator::load` refuses a
        // `TranslatorRef` that still holds them.
        decoder_start_token_id: UNREAD_TOKEN_ID,
        eos_token_id: UNREAD_TOKEN_ID,
        suppressed_ids: vec![UNREAD_TOKEN_ID],
    }
}

/// The sentinel a `TranslatorRef` carries before its `config.json` has been read.
/// Chosen to be an id no vocabulary has, so a forgotten read cannot look valid.
const UNREAD_TOKEN_ID: i64 = -1;

/// Resolve a source and a target language into a translator. A pair the
/// catalog does not hold is refused here, before any file is fetched.
pub(crate) fn resolve_pair(src: &str, tgt: &str) -> Result<TranslatorRef> {
    let (src, tgt) = (normalize(src), normalize(tgt));
    if let Some(pair) = pair_for(&src, &tgt) {
        return Ok(ref_for(pair));
    }
    // Say what this source *can* go to. For `pl` that is exactly `en`, which is
    // the honest answer: no Polish-to-German export exists, and pretending
    // otherwise would send a user looking through the whole catalog. A source the
    // catalog never lists at all gets the whole list instead, because there is
    // nothing specific to point at.
    let targets = valid_targets(&src);
    if targets.is_empty() {
        anyhow::bail!(
            "no translator for {src} -> {tgt}; {src} is not a listed source language, \
             and no {src}-into-anything model is published. Available: {}",
            valid_pairs()
        );
    }
    anyhow::bail!("no translator for {src} -> {tgt}; {src} can be translated into: {targets}")
}

/// The catalog row a spec names, or `None`. A spec is
/// `opus-mt-<src>-<tgt>` and must name a listed pair: the same string is both
/// the directory name and the repository name, so a spec that is not in the
/// catalog cannot be fetched either.
fn pair_for_spec(spec: &str) -> Option<&'static TranslatorPair> {
    let rest = spec.strip_prefix("opus-mt-")?;
    let (src, tgt) = rest.split_once('-')?;
    CATALOG
        .iter()
        .find(|p| p.src == src && p.tgt == tgt)
}

/// Resolve a `--translate-model` value: a directory path (contains a path
/// separator) passes through untouched and is only validated; anything else is
/// an explicit spec, which must name a catalog pair.
pub(crate) fn resolve(input: &str) -> Result<TranslatorRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!(
            "empty --translate-model value; expected a translator spec \
             (e.g. {TRANSLATE_DEFAULT_PAIR}) or a directory holding \
             {ENCODER_FILE}, {DECODER_FILE}, {TOKENIZER_FILE} and {CONFIG_FILE}"
        );
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        let dir = PathBuf::from(trimmed);
        if let Some(missing) = missing_in(&dir) {
            anyhow::bail!(
                "translator path {trimmed:?} is not a translator directory: {} is missing",
                missing.display()
            );
        }
        // A user directory names no catalog pair, so the codes are empty: the
        // pair is a property of the directory, and sttd does not read a pair
        // out of a directory name.
        return Ok(TranslatorRef {
            spec: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| trimmed.to_string()),
            src: String::new(),
            tgt: String::new(),
            encoder: dir.join(ENCODER_FILE),
            decoder: dir.join(DECODER_FILE),
            tokenizer: dir.join(TOKENIZER_FILE),
            config: dir.join(CONFIG_FILE),
            dir,
            assets: Vec::new(),
            decoder_start_token_id: UNREAD_TOKEN_ID,
            eos_token_id: UNREAD_TOKEN_ID,
            suppressed_ids: vec![UNREAD_TOKEN_ID],
        });
    }
    let pair = pair_for_spec(&normalize(trimmed)).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown translator {trimmed:?}; expected opus-mt-<src>-<tgt>, one of: {}",
            valid_pairs()
        )
    })?;
    Ok(ref_for(pair))
}

/// Every file a translator directory must hold.
const REQUIRED_FILES: [&str; 4] = [ENCODER_FILE, DECODER_FILE, TOKENIZER_FILE, CONFIG_FILE];

/// The first required file that is not in `dir`, or `None` when the directory is
/// a complete translator.
fn missing_in(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return Some(dir.to_path_buf());
    }
    REQUIRED_FILES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| !path.is_file())
}

/// A translator, guaranteed to exist **and to know its own token ids** on return:
/// downloads whatever of its files is missing, then reads `config.json`.
///
/// The ids are read here rather than compiled in because they differ per pair —
/// `pl-en` pads at 63429, `mul-en` at 64171 — and a wrong start token is not an
/// error. The graph runs and returns punctuation, which is exactly what
/// `opus-mt-mul-en` did before this read existed.
pub(crate) fn ensure_available(t: &mut TranslatorRef) -> Result<()> {
    if t.assets.is_empty() {
        // Nothing to fetch, so this is the "did the user point at a real
        // translator directory" check, already done at resolve; repeat it so the
        // function stands on its own.
        if let Some(missing) = missing_in(&t.dir) {
            anyhow::bail!(
                "{} is not a translator directory: {} is missing",
                t.dir.display(),
                missing.display()
            );
        }
    } else {
        std::fs::create_dir_all(&t.dir).with_context(|| install_hint(&t.dir))?;
        // The spec was resolved through the catalog, so it names a row; going back
        // through it means the URL is built in exactly one place.
        let pair = pair_for_spec(&t.spec)
            .ok_or_else(|| anyhow::anyhow!("{} is not a catalog translator spec", t.spec))?;
        for (repo_path, file_name) in &t.assets {
            let dest = t.dir.join(file_name);
            if dest.exists() {
                continue;
            }
            let url = asset_url(pair, repo_path);
            log::info!(
                "translator file not found at {}, downloading {url}...",
                dest.display()
            );
            crate::model_store::download(&url, &dest)
                .with_context(|| format!("{file_name}: {}", install_hint(&t.dir)))?;
        }
    }
    read_token_ids(t)
}

/// Read `decoder_start_token_id`, `eos_token_id` and `pad_token_id` out of the
/// export's `config.json` into the reference.
///
/// Separate from the download so a user-supplied directory is read the same way a
/// downloaded one is. A missing or malformed config is an error: proceeding would
/// mean decoding from a placeholder id, which fails silently.
fn read_token_ids(t: &mut TranslatorRef) -> Result<()> {
    let raw = std::fs::read_to_string(&t.config)
        .with_context(|| format!("failed to read {}", t.config.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", t.config.display()))?;
    let field = |name: &str| -> Result<i64> {
        config
            .get(name)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} declares no integer {name}; sttd needs it to open and close a decode",
                    t.config.display()
                )
            })
    };
    t.decoder_start_token_id = field("decoder_start_token_id")?;
    t.eos_token_id = field("eos_token_id")?;
    if t.decoder_start_token_id == t.eos_token_id {
        anyhow::bail!(
            "{} opens and stops on the same id ({}); the decode would never end",
            t.config.display(),
            t.decoder_start_token_id
        );
    }
    // `pad_token_id` is optional upstream, and where it is absent the start token
    // is the pad anyway, so either way at least one id ends up suppressed.
    let mut suppressed = vec![t.decoder_start_token_id];
    if let Some(pad) = config.get("pad_token_id").and_then(serde_json::Value::as_i64) {
        if !suppressed.contains(&pad) {
            suppressed.push(pad);
        }
    }
    t.suppressed_ids = suppressed;
    Ok(())
}

/// What to say when a translator has to be fetched into a directory that is
/// not writable — the usual case being a packaged install's root-owned models
/// directory, where only the bundled speech model is present. Same wording as
/// the voice path, because the cause and the fix are identical.
fn install_hint(dir: &Path) -> String {
    format!(
        "cannot prepare translator directory {} (packaged model dirs are \
         root-owned: install the sttd package, run once with sudo, or point \
         --translate-model at a writable directory)",
        dir.display()
    )
}

/// Resolve `--translate-model` and download whatever is missing, in one call:
/// the startup-time path, where an unknown spec or an unwritable directory has
/// to fail fast like the speech-model load it sits beside.
pub(crate) fn prepare(input: &str) -> Result<TranslatorRef> {
    let mut t = resolve(input)?;
    ensure_available(&mut t)?;
    Ok(t)
}

/// The same, for a language pair rather than a spec: the path a run takes,
/// since the pair comes from the decoding language plus `--translate`.
pub(crate) fn prepare_pair(src: &str, tgt: &str) -> Result<TranslatorRef> {
    let mut t = resolve_pair(src, tgt)?;
    ensure_available(&mut t)?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in export directory: the four files, zero bytes each, plus a
    /// `config.json` carrying real token ids so `ensure_available` can read them.
    fn write_dir(dir: &Path, names: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for name in names {
            if *name == CONFIG_FILE {
                std::fs::write(
                    dir.join(name),
                    br#"{"decoder_start_token_id": 63429, "eos_token_id": 0}"#,
                )
                .unwrap();
                continue;
            }
            std::fs::write(dir.join(name), b"x").unwrap();
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sttd-translator-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        dir
    }

    const ALL_FILES: [&str; 4] = REQUIRED_FILES;

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
            assert_eq!(t.tokenizer, translator_dir(&t.spec).join(TOKENIZER_FILE));
            assert_eq!(t.assets, ASSETS.to_vec(), "one file set per pair");
            // The spec round-trips back to the same row, so a directory name
            // and a `--translate-model` value are interchangeable.
            let back = pair_for_spec(&t.spec).expect("a catalog spec must parse back");
            assert_eq!(back, pair);
        }
    }

    #[test]
    fn spec_and_repository_naming_follow_upstream() {
        let pair = pair_for("pl", "en").expect("pl-en is in the catalog");
        assert_eq!(spec_for(pair), "opus-mt-pl-en");
        assert_eq!(
            repo_for(pair),
            "https://huggingface.co/Xenova/opus-mt-pl-en/resolve/main/"
        );
        // Each pair is its own repository, so two pairs never share one.
        assert_ne!(
            repo_for(pair),
            repo_for(pair_for("de", "en").expect("de-en is in the catalog"))
        );
        // The multilingual pair is the `--language auto` fallback, and it is
        // listed as a source only.
        assert!(pair_for("mul", "en").is_some());
        assert!(pair_for("en", "mul").is_none());
    }

    #[test]
    fn repository_urls_split_by_folder_and_the_tokenizer_stays_at_the_root() {
        // The graphs are nested under `onnx/` upstream; the tokenizer is not.
        // Getting this wrong is a 404 at first use, so it is asserted directly.
        let pair = pair_for("pl", "en").expect("pl-en is in the catalog");
        let base = "https://huggingface.co/Xenova/opus-mt-pl-en/resolve/main/";
        assert_eq!(
            asset_url(pair, "onnx/encoder_model_quantized.onnx"),
            format!("{base}onnx/encoder_model_quantized.onnx")
        );
        assert_eq!(
            asset_url(pair, "onnx/decoder_model_merged_quantized.onnx"),
            format!("{base}onnx/decoder_model_merged_quantized.onnx")
        );
        assert_eq!(
            asset_url(pair, "tokenizer.json"),
            format!("{base}tokenizer.json")
        );
        // Every catalog asset resolves to a real repository path.
        for (repo_path, _) in ASSETS {
            assert!(
                asset_url(pair, repo_path).starts_with(base),
                "{repo_path} must sit under the pair's own repository"
            );
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

        // A source with several targets lists them all, sorted. `en-pt` is not
        // published upstream, unlike most `en-*` pairs.
        let err = resolve_pair("en", "pt")
            .expect_err("en-pt is not published")
            .to_string();
        assert!(err.contains("en -> pt"), "err={err}");
        let targets = valid_targets("en");
        assert!(targets.contains("de") && targets.contains("zh"), "{targets}");
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

    #[test]
    fn a_spec_is_refused_unless_it_names_a_listed_pair() {
        // The spec form of the same check: a directory name and a
        // `--translate-model` value are the same string, so an unlisted spec is
        // both unresolvable and unfetchable.
        for bad in ["opus-mt-en-pl", "opus-mt-pl-de", "opus-mt-pt-en"] {
            let err = resolve(bad)
                .expect_err("an unlisted spec must not resolve")
                .to_string();
            assert!(err.contains("opus-mt-pl-en"), "{bad}: {err}");
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
        // The default pair is one a run can actually reach.
        let (src, tgt) = TRANSLATE_DEFAULT_PAIR.split_once('-').expect("pair is src-tgt");
        assert!(pair_for(src, tgt).is_some(), "the default pair must be listed");
    }

    #[test]
    fn translators_live_beside_the_models_never_inside_one() {
        // The directory name is what keeps `model_store`'s engine detection
        // from ever being pointed at a translator.
        assert_eq!(
            translators_dir(),
            PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{TRANSLATORS_SUBDIR}"))
        );
        assert_ne!(
            translator_dir("opus-mt-pl-en"),
            translator_dir("parakeet-ultra-int8")
        );
    }

    #[test]
    fn a_directory_path_passes_through_and_is_never_downloaded_into() {
        let dir = temp_dir("path");
        write_dir(&dir, &ALL_FILES);
        let t = resolve(dir.to_str().unwrap()).expect("must resolve");
        assert_eq!(
            t.spec,
            dir.file_name().unwrap().to_string_lossy(),
            "the directory's own name is the identity on a path run"
        );
        assert_eq!(t.encoder, dir.join(ENCODER_FILE));
        assert_eq!(t.decoder, dir.join(DECODER_FILE));
        assert_eq!(t.tokenizer, dir.join(TOKENIZER_FILE));
        assert!(t.assets.is_empty(), "a path is validated, never fetched");
        assert!(
            t.src.is_empty() && t.tgt.is_empty(),
            "a directory names no catalog pair"
        );
        // Nothing to download, so validation passes and the token ids come out of
        // the config on disk.
        let mut t = t;
        ensure_available(&mut t).expect("a complete directory must validate");
        assert_eq!(t.decoder_start_token_id, 63429);
        assert_eq!(t.eos_token_id, 0);

        // A directory missing its decoder is not a translator: refused at
        // resolution, before anything is opened, with the file named.
        std::fs::remove_file(dir.join(DECODER_FILE)).unwrap();
        let err = resolve(dir.to_str().unwrap())
            .expect_err("a directory without a decoder graph is not a translator")
            .to_string();
        assert!(err.contains(DECODER_FILE), "err={err}");

        // Neither is a directory with an empty one in it, nor one that does not
        // exist.
        let empty = temp_dir("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve(empty.to_str().unwrap()).is_err());
        assert!(resolve("/nonexistent/translator").is_err());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn empty_and_unknown_values_are_refused_with_the_expected_form_named() {
        let err = resolve("   ")
            .expect_err("an empty value must not resolve")
            .to_string();
        assert!(err.contains("--translate-model"), "err={err}");
        assert!(err.contains(TRANSLATE_DEFAULT_PAIR), "err={err}");
        // A name that is not a spec at all, and one that is a spec shape for a
        // pair that does not exist.
        for bad in ["flan-t5-base", "opus-mt-xx-yy", "opus-mt-pl", "pl-en"] {
            let err = resolve(bad).expect_err("must not resolve").to_string();
            assert!(err.contains("opus-mt-pl-en"), "{bad}: err={err}");
        }
        // An empty pair code is refused the same way, not silently defaulted.
        assert!(resolve_pair("pl", "").is_err());
        assert!(resolve_pair("", "en").is_err());
    }
}