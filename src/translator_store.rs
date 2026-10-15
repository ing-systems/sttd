// ---------------------------------------------------------------------------
// Translators: `--translate-model` resolution, download and reading of an ONNX
// export.
// ---------------------------------------------------------------------------
//
// One flag, `--translate <LANG>`, turns transcripts into another language; the
// weights behind it come from here. Two model families sit behind it:
//
// * **MarianMT** (`model_type: "marian"`), published as one repository per
//   language pair. The pair is the repository name, the catalog here is a list
//   of pairs, and the *weights* fix the direction: there is nothing to tell the
//   model, it already only does one thing.
// * **NLLB-200** (`model_type: "m2m_100"`), one repository for every direction.
//   The direction is an *input*: a source code in front of the encoder's input
//   ids, a target code as the decoder's opening token. So a spec no longer
//   names a pair, and `--language` becomes required rather than optional.
//
// Which family an export belongs to is **read out of its own `config.json`**,
// not inferred from the spec string, for two reasons: a `--translate-model`
// that points at a directory names no repository to infer from, and the shapes
// (`decoder_layers`, `decoder_attention_heads`, `d_model`) come from the same
// read, so a re-exported or re-quantized model is described by its file rather
// than by constants compiled here.
//
// **Both decoder graphs are fetched, and both are needed.** The merged one runs
// the prompt step and is the only graph that computes the cross-attention keys;
// the `with_past` one runs every step after it and is the only one that reads a
// cache. A decode through the merged graph alone re-runs the whole prefix per
// token, which is O(n²) in output length — see `translate::engine::Translator`.
//
// Layout mirrors `voice_store` deliberately: a spec resolves into
// `<models>/translators/<spec>/` with the five files sttd reads flat on disk,
// whatever nesting the repository uses upstream (`onnx/encoder_model_quantized.onnx`).
// The translate step never sees the repository layout, and no renaming happens
// after download. Both families publish the *same five filenames at the same
// repository paths*, so the asset list is one list, not one list per family.
//
// Deliberately *not* part of `model_store`: a translator is not a speech
// engine, so it must not be reachable through `--model`, and `model_store`'s
// engine detection (`EngineKind::marker_file`) must never classify one.
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{
    DEFAULT_MODELS_DIR, NLLB_REPO, OPUS_MT_REPO, TRANSLATE_EXAMPLE_PAIR, TRANSLATORS_SUBDIR,
};

/// Encoder graph, inside a translator directory and inside the repository
/// (`onnx/`). Same name for both families.
const ENCODER_FILE: &str = "encoder_model_quantized.onnx";
/// Merged decoder graph: the prompt step, and every later step as well, because
/// one graph takes both. Same two locations.
const DECODER_FILE: &str = "decoder_model_merged_quantized.onnx";
/// Decoder graph for the steps *after* the prompt, the one that reads a
/// key/value cache instead of recomputing it. Same two locations.
///
/// The second half of the decode. The merged graph above re-runs the whole
/// sequence on every step, which is **O(n²) in output length**; this graph takes
/// one token per step against the cache the merged graph returned, which is
/// O(n). Both are published for both families, and the split is the same one
/// `whisper_core` runs — see `translate::engine::Translator::run_cached_step`.
const DECODER_WITH_PAST_FILE: &str = "decoder_with_past_model_quantized.onnx";
/// Tokenizer, at the repository root.
const TOKENIZER_FILE: &str = "tokenizer.json";
/// The export's `config.json`, also at the repository root. Under a KiB, and the
/// only place the model describes *itself*: its family, its decoder geometry
/// and its opening and stopping token ids.
const CONFIG_FILE: &str = "config.json";

/// The five files every translator needs, as `(repo path, file name)`. The
/// repository nests the graphs under `onnx/` and keeps the tokenizer and the
/// config at the root; on disk all are flat. One file set for **both** families,
/// which is why it is a constant and not a per-family list: `Xenova` publishes
/// `onnx/encoder_model_quantized.onnx`,
/// `onnx/decoder_model_merged_quantized.onnx` and
/// `onnx/decoder_with_past_model_quantized.onnx` under `opus-mt-pl-en` and under
/// `nllb-200-distilled-600M` alike.
const ASSETS: &[(&str, &str)] = &[
    ("onnx/encoder_model_quantized.onnx", ENCODER_FILE),
    ("onnx/decoder_model_merged_quantized.onnx", DECODER_FILE),
    (
        "onnx/decoder_with_past_model_quantized.onnx",
        DECODER_WITH_PAST_FILE,
    ),
    ("tokenizer.json", TOKENIZER_FILE),
    ("config.json", CONFIG_FILE),
];

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
fn valid_specs() -> String {
    let mut specs: Vec<String> = CATALOG.iter().map(spec_for).collect();
    specs.push(NLLB_SPEC.to_string());
    specs.sort();
    specs.join(", ")
}

/// The catalog row for `src` into `tgt`, or `None` when the pair is not listed.
fn pair_for(src: &str, tgt: &str) -> Option<&'static TranslatorPair> {
    CATALOG.iter().find(|p| p.src == src && p.tgt == tgt)
}

/// Whether the catalog holds a `src` into `tgt` model, without downloading one.
///
/// The `--language auto` path needs this to ask a question before it commits to a
/// download: `opus-mt-mul-en` exists and `opus-mt-mul-de` does not, and the second
/// case has to be an error rather than a 404.
fn has_pair(src: &str, tgt: &str) -> bool {
    let (src, tgt) = (normalize(src), normalize(tgt));
    pair_for(&src, &tgt).is_some()
}

/// The sources the catalog can translate into `tgt`, sorted.
///
/// The counterpart to [`valid_targets`], for the case where the *target* is what
/// the user typed and the source is unknown.
fn sources_for(tgt: &str) -> Vec<&'static str> {
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
fn valid_targets(src: &str) -> String {
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

/// Directory every downloaded translator resolves into.
fn translators_dir() -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{TRANSLATORS_SUBDIR}"))
}

/// Directory one translator spec resolves into.
fn translator_dir(spec: &str) -> PathBuf {
    translators_dir().join(spec)
}

/// The spec for a pair: the upstream repository name, unchanged. Keeping the
/// two the same string is what lets a log line, a directory name and a
/// `--translate-model` value all be read interchangeably.
pub(crate) fn spec_for(pair: &TranslatorPair) -> String {
    format!("opus-mt-{}-{}", pair.src, pair.tgt)
}

/// Which HF organization an export is published under. Both families are
/// converted by the same publisher, which is why one `ASSETS` list serves both.
fn repo_of(spec: &str) -> &'static str {
    if spec == NLLB_SPEC {
        NLLB_REPO
    } else {
        OPUS_MT_REPO
    }
}

/// Where one repository is fetched from. `resolve/main` rather than `main` so a
/// spec pinned to a revision upstream resolves.
fn repo_base(spec: &str) -> String {
    format!(
        "https://huggingface.co/{}/{spec}/resolve/main/",
        repo_of(spec)
    )
}

/// Where a published file of `spec` is fetched from.
///
/// `repo_base` and the asset's repository path, in one place, so a token in
/// [`ASSETS`] and the URL it resolves to cannot drift apart.
fn asset_url(spec: &str, repo_path: &str) -> String {
    format!("{}{repo_path}", repo_base(spec))
}

/// Which model family an export belongs to, read out of its `config.json`.
///
/// Read rather than assumed because `--translate-model` also accepts a directory,
/// which names no repository and therefore no family: a user who converted their
/// own Marian export, or dropped in an NLLB one, is served by the same path as a
/// catalog spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    /// `model_type: "marian"`. One repository per language pair; the weights fix
    /// the direction, so the decoder opens on `decoder_start_token_id` and the
    /// encoder gets the raw text ids.
    Marian,
    /// `model_type: "m2m_100"`. One repository for every direction, which is taken
    /// as an input: the encoder's ids are prefixed with the source code and closed
    /// with `eos`, and the decoder opens on the *target* code.
    Nllb,
}

impl Family {
    /// The `model_type` this family is published under.
    const fn model_type(self) -> &'static str {
        match self {
            Self::Marian => "marian",
            Self::Nllb => "m2m_100",
        }
    }

    /// Every family this build understands, named in a refusal. A `config.json`
    /// naming a third architecture fails at load rather than half-working.
    pub(crate) const KNOWN: &'static [Family] = &[Self::Marian, Self::Nllb];

    /// Longest run of tokens a translation decode may emit before it is cut off.
    ///
    /// A repetition-loop guard, not a quality limit and not a cost limit — a
    /// translated sentence is a sentence or two. It is small because a repetition
    /// loop is the one way a greedy decode runs away, and the decode is otherwise
    /// **O(n) in output length**: the prompt step fills a key/value cache and every
    /// step after it adds one token (see `translate::engine::Translator::run_cached_step`).
    /// So this is a bound on how much work one utterance can ask for, not the thing
    /// keeping a quadratic tail in check — the single-graph form that needed a
    /// per-family cap for that is gone.
    ///
    /// It stays per family anyway, because the two decoders cost different amounts
    /// per step: Marian is 6 layers of `d_model` 512, NLLB 12 of 1024. Measured on
    /// four threads with the cache, one sentence is 79 ms for a pair and 677 ms for
    /// NLLB, so 96 covers a runaway in about 2 s and 48 in about 11 s. Marian's 96
    /// also still covers a sentence with no terminator in it, which is what
    /// `translate::text::split_sentences` hands over.
    pub(crate) const fn max_new_tokens(self) -> usize {
        match self {
            Self::Marian => 96,
            Self::Nllb => 48,
        }
    }
}

/// The family a `model_type` names, or `None` when it is not one sttd runs.
fn family_of(model_type: &str) -> Option<Family> {
    Family::KNOWN
        .iter()
        .copied()
        .find(|f| f.model_type() == model_type)
}

/// The decoder geometry a `config.json` publishes.
///
/// Everything the cache contract needs, read from the model rather than compiled
/// in. The numbers differ between the families by enough to matter — Marian is 6
/// layers of 8 heads, NLLB 12 of 16 — and a constant that disagreed with the graph
/// would not be an error, just a decode that quietly falls apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphShape {
    /// Decoder layers, each contributing a self- and a cross-attention key and
    /// value.
    pub(crate) layers: usize,
    /// Attention heads per layer.
    pub(crate) heads: usize,
    /// `d_model / heads`. Never read from the file: it is the quotient, and every
    /// export declares `d_model` and the head count rather than the head width.
    head_dim: usize,
}

impl GraphShape {
    /// The `past_key_values.*` inputs one export declares: four per layer (a
    /// self- and a cross-attention key and value).
    pub(crate) fn cache_slots(&self) -> usize {
        self.layers * 4
    }

    /// Width of one head, the last axis of a cache slot.
    pub(crate) const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The shape the shape-check is anchored on, for a test that pins the two
    /// families against the exports they came from.
    #[cfg(test)]
    pub(crate) const fn new(layers: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            layers,
            heads,
            head_dim,
        }
    }
}

/// What an export's own `config.json` says about itself.
///
/// Returned beside the [`TranslatorRef`] rather than written into it, so there
/// is no "not read yet" placeholder for a load to trip over: a [`Prepared`]
/// translator has either been read or does not exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelFacts {
    pub(crate) family: Family,
    pub(crate) shape: GraphShape,
    /// `decoder_start_token_id`: the id a Marian decoder opens its sequence with,
    /// and also that export's `pad_token_id`. For NLLB it is the `</s>` prefix the
    /// decoder opens on *before* the target code, so it is used there too.
    pub(crate) start_token: i64,
    /// `eos_token_id`: where a decode stops. 0 in both Marian pairs checked, 2 in
    /// NLLB — read from the same place as the start token rather than assumed.
    pub(crate) eos_token: i64,
    /// `pad_token_id`, when the export publishes one. Must be suppressed from a
    /// decode; see `translate::engine::Suppressed`.
    pub(crate) pad_token: Option<i64>,
}

/// Where a translator's files are and what is left to fetch.
///
/// Deliberately says nothing about what the model *is*: that is [`ModelFacts`],
/// read from the same download. A reference is a location; facts are a claim the
/// model makes about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslatorRef {
    /// Upstream spec: `opus-mt-pl-en` or `nllb-200-distilled-600M`.
    pub(crate) spec: String,
    /// Source language code, as the catalog spells it. Empty for a directory,
    /// which names no pair.
    pub(crate) src: String,
    /// Target language code, as the catalog spells it. Empty for a directory.
    pub(crate) tgt: String,
    pub(crate) dir: PathBuf,
    /// Encoder graph inside `dir`.
    pub(crate) encoder: PathBuf,
    /// Merged decoder graph inside `dir`: the prompt step.
    pub(crate) decoder: PathBuf,
    /// Cached decoder graph inside `dir`: every step after the prompt.
    pub(crate) decoder_with_past: PathBuf,
    /// Tokenizer inside `dir`.
    pub(crate) tokenizer: PathBuf,
    /// The export's `config.json` inside `dir`.
    pub(crate) config: PathBuf,
    /// `(repo path, file name)` pairs to fetch when missing. Empty for a
    /// user-supplied directory, which is validated but never downloaded into.
    pub(crate) assets: Vec<(&'static str, &'static str)>,
}

impl TranslatorRef {
    /// The five files of a translator, named inside `dir`.
    ///
    /// One constructor for all three ways a reference is built — a catalog pair,
    /// the multilingual spec, and a directory the user pointed at — because the
    /// five file names and the four path joins are the same in every one of them
    /// and only `spec`, the direction and the asset list differ. Written out
    /// three times, a fifth file would have reached two of them.
    ///
    /// `src`/`tgt` are empty for a spec or directory that names no pair: the
    /// direction is not a property of those weights, and filling them in would
    /// put a lie in the log line that prints them.
    fn in_dir(
        dir: PathBuf,
        spec: String,
        src: &str,
        tgt: &str,
        assets: &[(&'static str, &'static str)],
    ) -> Self {
        Self {
            encoder: dir.join(ENCODER_FILE),
            decoder: dir.join(DECODER_FILE),
            decoder_with_past: dir.join(DECODER_WITH_PAST_FILE),
            tokenizer: dir.join(TOKENIZER_FILE),
            config: dir.join(CONFIG_FILE),
            dir,
            assets: assets.to_vec(),
            spec,
            src: src.to_string(),
            tgt: tgt.to_string(),
        }
    }
}

/// A translator, resolved, downloaded **and read**: where its files are, and what
/// its own config says about it. Nothing further has to be guessed at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Prepared {
    pub(crate) reference: TranslatorRef,
    pub(crate) facts: ModelFacts,
    /// NLLB only: the two FLORES codes this run translates between, resolved
    /// from the run's own `--language` and `--translate`. `None` for a pair
    /// export, which takes its direction from which weights were loaded.
    pub(crate) codes: Option<(String, String)>,
}

/// The direction a run translates in, as the run knows it.
///
/// `src` is `None` under `--language auto`: the source is then unknown per
/// utterance. A pair export can answer that on its own (`mul-en`), and a
/// multilingual one cannot — which is the whole difference between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Direction<'a> {
    /// `--language`, or `None` when it is `auto`.
    pub(crate) src: Option<&'a str>,
    /// `--translate`.
    pub(crate) tgt: &'a str,
}

/// A language code as the catalog spells it: trimmed, lowercased, with `_` for
/// `-` so `pt_BR`-shaped input and `pt-br` reach the same comparison.
fn normalize(code: &str) -> String {
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
fn nllb_codes(direction: Direction<'_>) -> Result<(String, String)> {
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
fn resolve_pair(src: &str, tgt: &str) -> Result<TranslatorRef> {
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
fn pair_for_spec(spec: &str) -> Option<&'static TranslatorPair> {
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
    if trimmed.contains('/') || trimmed.contains('\\') {
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

/// Every file a translator directory must hold.
const REQUIRED_FILES: [&str; 5] = [
    ENCODER_FILE,
    DECODER_FILE,
    DECODER_WITH_PAST_FILE,
    TOKENIZER_FILE,
    CONFIG_FILE,
];

/// Whether `spec`'s directory is on disk and complete, without touching the
/// network. What a weights-gated test asks before it would download anything.
#[cfg(test)]
pub(crate) fn translator_is_present(spec: &str) -> bool {
    missing_in(&translator_dir(spec)).is_none()
}

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

/// Download whatever of a translator's files is missing, then validate that the
/// directory is complete.
fn ensure_available(t: &TranslatorRef) -> Result<()> {
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
        return Ok(());
    }
    std::fs::create_dir_all(&t.dir).with_context(|| install_hint(&t.dir))?;
    for (repo_path, file_name) in &t.assets {
        let dest = t.dir.join(file_name);
        if dest.exists() {
            continue;
        }
        let url = asset_url(&t.spec, repo_path);
        log::info!(
            "translator file not found at {}, downloading {url}...",
            dest.display()
        );
        crate::model_store::download(&url, &dest)
            .with_context(|| format!("{file_name}: {}", install_hint(&t.dir)))?;
    }
    Ok(())
}

/// Read an export's own `config.json`: which family it is, how its decoder is
/// shaped, and which ids a decode opens and stops on.
///
/// Separate from the download so a user-supplied directory is read the same way a
/// downloaded one is. A missing or malformed config is an error: proceeding would
/// mean feeding the graphs input they do not declare.
///
/// Nothing here is compiled in. The ids differ per export (`pl-en` opens on
/// 63429, `mul-en` on 64171) and the shapes differ per family (Marian 6 layers of
/// 8 heads, NLLB 12 of 16); a wrong value is not an error, it is a graph that runs
/// and returns punctuation.
fn read_config(t: &TranslatorRef) -> Result<ModelFacts> {
    let raw = std::fs::read_to_string(&t.config)
        .with_context(|| format!("failed to read {}", t.config.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", t.config.display()))?;
    let bad = |what: &str| {
        anyhow::anyhow!(
            "{} declares no {what}; sttd needs it to run this export",
            t.config.display()
        )
    };
    let field = |name: &str| -> Result<i64> {
        config
            .get(name)
            .and_then(serde_json::Value::as_i64)
            .filter(|v| *v >= 0)
            .ok_or_else(|| bad(&format!("non-negative integer {name}")))
    };

    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| bad("string model_type"))?;
    let family = family_of(model_type).ok_or_else(|| {
        anyhow::anyhow!(
            "{} is model_type {model_type:?}, which sttd cannot run; it runs {}",
            t.config.display(),
            Family::KNOWN
                .iter()
                .map(|f| format!("{f:?} (model_type {:?})", f.model_type()))
                .collect::<Vec<_>>()
                .join(" and ")
        )
    })?;

    let shape = GraphShape {
        layers: usize::try_from(field("decoder_layers")?)
            .map_err(|_| bad("decoder_layers that fits a usize"))?,
        heads: usize::try_from(field("decoder_attention_heads")?)
            .map_err(|_| bad("decoder_attention_heads that fits a usize"))?,
        head_dim: 0,
    };
    if shape.layers == 0 || shape.heads == 0 {
        anyhow::bail!(
            "{} declares a {} of {} layers and {} heads; that is not a decoder",
            t.config.display(),
            family.model_type(),
            shape.layers,
            shape.heads
        );
    }
    let d_model =
        usize::try_from(field("d_model")?).map_err(|_| bad("d_model that fits a usize"))?;
    if d_model % shape.heads != 0 {
        anyhow::bail!(
            "{} declares d_model {d_model} over {} heads; the head width would not be a whole number",
            t.config.display(),
            shape.heads
        );
    }
    let shape = GraphShape {
        head_dim: d_model / shape.heads,
        ..shape
    };

    let start_token = field("decoder_start_token_id")?;
    let eos_token = field("eos_token_id")?;
    // Opening and stopping on one id means a decode that can never end. Marian
    // cannot: `pl-en` opens on 63429 and stops on 0, `mul-en` on 64171 and 0. NLLB
    // declares **both as 2** and is perfectly runnable, because it does not open
    // on `decoder_start_token_id` at all -- it opens on the target language code
    // (see `translate::engine::Feed`). So the rule belongs to the family that is bound by
    // it, and applying it to NLLB would refuse a working model.
    if family == Family::Marian && start_token == eos_token {
        anyhow::bail!(
            "{} opens and stops on the same id ({start_token}); the decode would never end",
            t.config.display()
        );
    }
    Ok(ModelFacts {
        family,
        shape,
        start_token,
        eos_token,
        pad_token: config
            .get("pad_token_id")
            .and_then(serde_json::Value::as_i64)
            .filter(|v| *v >= 0),
    })
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

    /// A stand-in export directory: the four files, zero bytes each, plus a
    /// `config.json` describing the export the way its own would.
    fn write_dir(dir: &Path, names: &[&str], config: &str) {
        std::fs::create_dir_all(dir).unwrap();
        for name in names {
            if *name == CONFIG_FILE {
                std::fs::write(dir.join(name), config).unwrap();
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

    const ALL_FILES: [&str; 5] = REQUIRED_FILES;

    /// `opus-mt-pl-en`'s own numbers, as literals. What sttd compiles in is only
    /// the *absence* of these; the values arrive from the file.
    const MARIAN_CONFIG: &str = r#"{
        "model_type": "marian",
        "decoder_layers": 6, "decoder_attention_heads": 8, "d_model": 512,
        "decoder_start_token_id": 63429, "eos_token_id": 0, "pad_token_id": 63429
    }"#;
    /// `nllb-200-distilled-600M`'s own numbers. Note `decoder_start_token_id` and
    /// `eos_token_id` are **both 2**: the config sttd must accept, and the one a
    /// family-blind check would refuse.
    const NLLB_CONFIG: &str = r#"{
        "model_type": "m2m_100",
        "decoder_layers": 12, "decoder_attention_heads": 16, "d_model": 1024,
        "decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1
    }"#;

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
    fn spec_and_repository_naming_follow_upstream() {
        let pair = pair_for("pl", "en").expect("pl-en is in the catalog");
        assert_eq!(spec_for(pair), "opus-mt-pl-en");
        assert_eq!(
            repo_base(&spec_for(pair)),
            "https://huggingface.co/Xenova/opus-mt-pl-en/resolve/main/"
        );
        // Each pair is its own repository, so two pairs never share one.
        assert_ne!(
            repo_base(&spec_for(pair)),
            repo_base(&spec_for(
                pair_for("de", "en").expect("de-en is in the catalog")
            ))
        );
        // The multilingual export lives in the same publisher, so it resolves too,
        // and it is not a pair: no `opus-mt-` prefix, and nothing to split.
        assert_eq!(
            repo_base(NLLB_SPEC),
            "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/"
        );
        assert!(pair_for_spec(NLLB_SPEC).is_none());
        // The `--language auto` fallback is listed as a source only.
        assert!(pair_for("mul", "en").is_some());
        assert!(pair_for("en", "mul").is_none());
    }

    #[test]
    fn repository_urls_split_by_folder_and_the_tokenizer_stays_at_the_root() {
        // The graphs are nested under `onnx/` upstream; the tokenizer is not.
        // Getting this wrong is a 404 at first use, so it is asserted directly --
        // for **both** families, because one asset list serves both and that only
        // holds while they agree.
        for spec in ["opus-mt-pl-en", NLLB_SPEC] {
            let base = format!("https://huggingface.co/Xenova/{spec}/resolve/main/");
            assert_eq!(
                asset_url(spec, "onnx/encoder_model_quantized.onnx"),
                format!("{base}onnx/encoder_model_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "onnx/decoder_model_merged_quantized.onnx"),
                format!("{base}onnx/decoder_model_merged_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "onnx/decoder_with_past_model_quantized.onnx"),
                format!("{base}onnx/decoder_with_past_model_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "tokenizer.json"),
                format!("{base}tokenizer.json")
            );
            // Every catalog asset resolves to a real repository path.
            for (repo_path, _) in ASSETS {
                assert!(
                    asset_url(spec, repo_path).starts_with(&base),
                    "{repo_path} must sit under {spec}'s own repository"
                );
            }
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
        assert_ne!(
            translator_dir(NLLB_SPEC),
            translator_dir("parakeet-ultra-int8")
        );
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

    /// Both configs are read out of their own files and come back as what those
    /// files say. The numbers are checked against the exports they came from, so
    /// a config that stops publishing them fails here rather than as a decode
    /// that quietly returns punctuation.
    #[test]
    fn a_config_says_which_family_it_is_and_how_its_decoder_is_shaped() {
        for (tag, config, family, layers, heads, head_dim, slots) in [
            ("marian", MARIAN_CONFIG, Family::Marian, 6, 8, 64, 24),
            ("nllb", NLLB_CONFIG, Family::Nllb, 12, 16, 64, 48),
        ] {
            let dir = temp_dir(tag);
            write_dir(&dir, &ALL_FILES, config);
            let t = resolve(dir.to_str().unwrap()).expect("must resolve");
            let facts = read_config(&t).unwrap_or_else(|e| panic!("{tag}: {e}"));
            assert_eq!(facts.family, family, "{tag}");
            assert_eq!(facts.shape.layers, layers, "{tag}");
            assert_eq!(facts.shape.heads, heads, "{tag}");
            assert_eq!(facts.shape.head_dim(), head_dim, "{tag}");
            assert_eq!(facts.shape.cache_slots(), slots, "{tag}");
            // The head width is the quotient, never read from the file: neither
            // config declares it.
            assert!(!config.contains("head_dim"), "{tag} must not declare one");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// The bug this whole family split exists to prevent, in one assertion: NLLB's
    /// `config.json` declares `decoder_start_token_id` and `eos_token_id` as the
    /// **same** id, 2. A check that refuses a model opening and stopping on one
    /// token is right for Marian and wrong here, and applying it to both refuses a
    /// model that runs perfectly well.
    #[test]
    fn a_family_that_does_not_decode_from_its_start_token_may_reuse_the_id() {
        let dir = temp_dir("same-id");
        write_dir(&dir, &ALL_FILES, NLLB_CONFIG);
        let t = resolve(dir.to_str().unwrap()).unwrap();
        let facts = read_config(&t).expect("an NLLB config must be accepted");
        assert_eq!(facts.start_token, facts.eos_token);
        assert_eq!(facts.family, Family::Nllb);
        std::fs::remove_dir_all(&dir).ok();

        // Marian with the same shape is a real fault: nothing would stop it.
        let dir = temp_dir("same-id-marian");
        let same_id = MARIAN_CONFIG.replace("\"eos_token_id\": 0", "\"eos_token_id\": 63429");
        write_dir(&dir, &ALL_FILES, &same_id);
        let t = resolve(dir.to_str().unwrap()).unwrap();
        let err = read_config(&t)
            .expect_err("a Marian export that opens and stops on one id must be refused")
            .to_string();
        assert!(err.contains("would never end"), "err={err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A config sttd cannot run, or cannot run *as itself*, fails at load.
    #[test]
    fn a_config_sttd_cannot_run_is_refused_by_name() {
        let cases = [
            (
                r#"{"model_type": "t5", "decoder_layers": 6, "decoder_attention_heads": 8,
                    "d_model": 512, "decoder_start_token_id": 0, "eos_token_id": 1}"#,
                "t5",
            ),
            (
                r#"{"decoder_layers": 6, "decoder_attention_heads": 8, "d_model": 512,
                    "decoder_start_token_id": 0, "eos_token_id": 1}"#,
                "model_type",
            ),
            (
                r#"{"model_type": "marian", "decoder_layers": 6, "decoder_attention_heads": 8,
                    "d_model": 513, "decoder_start_token_id": 0, "eos_token_id": 1}"#,
                "whole number",
            ),
            (
                r#"{"model_type": "marian", "decoder_layers": 0, "decoder_attention_heads": 8,
                    "d_model": 512, "decoder_start_token_id": 0, "eos_token_id": 1}"#,
                "not a decoder",
            ),
            (
                r#"{"model_type": "marian", "decoder_layers": 6, "decoder_attention_heads": 8,
                    "d_model": 512, "eos_token_id": 1}"#,
                "decoder_start_token_id",
            ),
        ];
        for (config, expected) in cases {
            let dir = temp_dir("bad-config");
            write_dir(&dir, &ALL_FILES, config);
            let t = resolve(dir.to_str().unwrap()).unwrap();
            let err = read_config(&t)
                .expect_err("an unusable config must be refused")
                .to_string();
            assert!(err.contains(expected), "expected {expected:?} in: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

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
