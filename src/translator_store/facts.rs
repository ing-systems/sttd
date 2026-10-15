//! What an export says about itself in its `config.json`: its family and the
//! shape of its decoder cache.

use anyhow::{Context, Result};

use super::TranslatorRef;

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
pub(super) fn read_config(t: &TranslatorRef) -> Result<ModelFacts> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator_store::reference::resolve;
    use crate::translator_store::test_support::*;

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
}
