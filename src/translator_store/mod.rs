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
//
// `reference` resolves a value or a direction to a translator and prepares it;
// what it consults sits beside it: `catalog` (the pairs), `lang_codes` (the
// FLORES spellings), `fetch` (paths, URLs, download) and `facts` (what an
// export's `config.json` says about it).
// ---------------------------------------------------------------------------

mod catalog;
mod facts;
mod fetch;
mod lang_codes;
mod reference;

use std::path::PathBuf;

use facts::ModelFacts;
use fetch::{CONFIG_FILE, DECODER_FILE, DECODER_WITH_PAST_FILE, ENCODER_FILE, TOKENIZER_FILE};

pub(crate) use catalog::NLLB_SPEC;
#[cfg(test)]
pub(crate) use catalog::{CATALOG, PL_EN_SPEC};
pub(crate) use facts::{Family, GraphShape};
#[cfg(test)]
pub(crate) use fetch::translator_is_present;
#[cfg(test)]
pub(crate) use reference::resolve;
pub(crate) use reference::{prepare, prepare_from_direction, reference_for};

#[cfg(test)]
mod test_support;

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
