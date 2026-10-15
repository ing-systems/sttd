// ---------------------------------------------------------------------------
// Translation step: text in, text out, with an ONNX seq2seq model as the
// translator.
// ---------------------------------------------------------------------------
//
// The step runs after the gates and before the emitter, so everything that
// decides *whether* to speak still reads the source text. Only what the user
// hears changes. A failure here never drops an utterance: the translation thread
// runs the result through `fallback_or_translated`, which warns and returns the
// source.
//
// Four parts live in this file. The pure half (`split_sentences`,
// `fallback_or_translated`, `patched_value`, `next_token`) needs no weights, so
// it is unit-tested without a model. `Translator` owns the three `ort` sessions
// and the tokenizer. `TranslationOut` is the thread the translator runs on, with a
// bounded FIFO in front of it so a translation never stalls the next decode.
//
// **The decode is two graphs, and why is the one thing here worth reading.** The
// export ships a merged decoder and a `decoder_with_past` one. A single graph
// cannot do both halves of a seq2seq decode, because the graph that computes the
// cross-attention keys is not the graph that reads them back:
//
// * The **prompt step** runs the merged graph. It takes `encoder_hidden_states`,
//   opens the decoder on the family's opening ids, and — this is the part that is
//   measured rather than assumed — returns the key/value cache under `present.`
//   even with `use_cache_branch` false. `true` cannot: that branch fails on an
//   empty past with `encoder_attn/Reshape_4 ... The dimension with value zero
//   exceeds the dimension size of the input tensor`.
// * Every **later step** runs the cached graph, one token at a time against the
//   cache. It takes no `encoder_hidden_states` and no cache flag, because the
//   cross-attention half travels inside the cache, and it returns the self half
//   only.
//
// So the decode is **O(n) in output length** instead of the O(n²) a single graph
// re-running its own prefix costs, and the cross-attention keys and values are
// computed once per utterance rather than once per token.
//
// Two model families run through those same graphs, and they differ in exactly
// two places, both of them **inputs**:
//
// * A MarianMT pair export takes raw text and opens its decoder on
//   `decoder_start_token_id`, because the weights already only do one direction.
// * A multilingual NLLB-200 export is told which way to go: the source code goes
//   in front of the encoder's ids and closes them with `</s>`, and the decoder
//   opens on the *target* code. `decoder_start_token_id` is never used, which is
//   why that export declaring it equal to `eos_token_id` is fine.
//
// [`Feed`] is that difference, as a value the loaded translator holds; everything
// after it is shared. The decoder's *shape* also differs, but that is read out of
// each export's own `config.json` rather than compiled in — see
// `translator_store::GraphShape` — so a re-exported model describes itself.
// ---------------------------------------------------------------------------

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::Instant;

use anyhow::{Context, Result};
use ndarray17::{ArrayD, ArrayViewD};
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::{Tensor, TensorRef, ValueType};
use tokenizers::Tokenizer;

use crate::config::TRANSLATE_MAX_INPUT_TOKENS;
use crate::emit::{Emission, Emitter};
use crate::translator_store::{Family, GraphShape, Prepared};

/// The name prefix of the decoder's cache inputs. The matching outputs are the
/// same strings under `present.`, which [`present_name`] derives from this.
///
/// A `past_key_values.N.{decoder,encoder}.{key,value}` shape, which is what
/// names a slot: `N` walks layers and the two halves are self- and
/// cross-attention. How many there are is the export's own
/// [`GraphShape::cache_slots`], not a constant here.
const PAST_PREFIX: &str = "past_key_values.";

/// The two halves of a cache slot, as the export spells them inside a slot name.
///
/// Which attention a slot belongs to is not something sttd decides: the export
/// names it, and reading the name is what keeps a cross-attention tensor out of a
/// self-attention slot. Both halves are the same shape, so nothing else would.
const SELF_SLOT: &str = ".decoder.";
const CROSS_SLOT: &str = ".encoder.";

/// Split text into the pieces that go through the model one at a time.
///
/// The terminators are `.` `!` `?` `;` and a newline. A colon is not one: it
/// separates a label from its value (`cena: 12.30 zł`), and cutting there sends
/// half a number to the model. A `.` between two digits is not one either, for
/// the same reason.
///
/// Every delimiter stays with the piece it ends, so no non-space character is
/// ever lost, and no empty piece is emitted. Text with no terminator stays one
/// piece.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        let ends_piece = match ch {
            '\n' | '!' | '?' | ';' => true,
            // `12.30` is one number, not two sentences.
            '.' => !matches!(chars.peek(), Some(next) if next.is_ascii_digit()),
            _ => false,
        };
        if ends_piece {
            push_piece(&mut pieces, &mut current);
        }
    }
    push_piece(&mut pieces, &mut current);
    merge_terminator_pieces(&mut pieces);
    pieces
}

/// Move a finished piece into the list, dropping it when it holds only spaces.
fn push_piece(pieces: &mut Vec<String>, current: &mut String) {
    let piece = current.trim();
    if !piece.is_empty() {
        pieces.push(piece.to_string());
    }
    current.clear();
}

/// Fold a run of terminators onto the piece before it, so `Wow!!` stays one
/// piece instead of `Wow!` plus a piece holding only `!`.
fn merge_terminator_pieces(pieces: &mut Vec<String>) {
    let mut merged: Vec<String> = Vec::with_capacity(pieces.len());
    for piece in pieces.drain(..) {
        let only_terminators = piece
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '.' | '!' | '?' | ';' | '\n'));
        match merged.last_mut() {
            Some(last) if only_terminators => last.push_str(&piece),
            _ => merged.push(piece),
        }
    }
    *pieces = merged;
}

/// The text to emit for one utterance: the translation, or the source text.
///
/// A translation that fails or comes back empty warns and falls back to
/// `source`, because an utterance the user heard in the wrong language is
/// worse than one they heard in the right one. `tag` names the utterance in the
/// log line.
pub(crate) fn fallback_or_translated(source: &str, r: Result<String>, tag: &str) -> String {
    match r {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        Ok(_) => {
            log::warn!("{tag}: translation came back empty; emitting the source text");
            source.to_string()
        }
        Err(e) => {
            log::warn!("{tag}: translation failed ({e:#}); emitting the source text");
            source.to_string()
        }
    }
}

/// Whether a normalizer carries the null charsmap `tokenizers` 0.23 panics on.
///
/// Walked rather than pattern-matched, because where the `Precompiled` sits inside
/// the normalizer is the converter's choice: Xenova writes a bare `Precompiled`
/// for Marian and wraps the same node in a `Sequence` (with a `Replace` beside it)
/// for NLLB. A `Parallel` would be a third shape and would need the same walk.
fn has_null_charsmap(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, child)| {
            (key == "precompiled_charsmap" && child.is_null()) || has_null_charsmap(child)
        }),
        serde_json::Value::Array(items) => items.iter().any(has_null_charsmap),
        _ => false,
    }
}

/// Drop the normalizer **only** when it is the one `tokenizers` cannot load, and
/// report whether it did.
///
/// Xenova's Marian exports carry
/// `"normalizer":{"type":"Precompiled","precompiled_charsmap":null}`, and
/// `tokenizers` 0.23 panics on the null while deserializing it:
///
/// ```text
/// tokenizers-0.23.2/src/normalizers/mod.rs:143:26:
/// Precompiled: Error("invalid type: null, expected a borrowed string", line: 1, column: 28)
/// ```
///
/// The charsmap is what `transformers.js` writes as `null` because it does not
/// use it, and the working reference produces byte-identical ids with the
/// normalizer dropped (`Dzisiaj jest poniedzialek.` →
/// `[8261,45,54,95,12080,187,2,0]`), so the field is removed rather than
/// substituted. `source.spm` would be the real charsmap, but only Xenova's
/// converted `tokenizer.json` is downloaded, and that is where the null is.
///
/// **The test is on the null, not on the field.** NLLB's export publishes a real
/// 316 KB charsmap, and dropping a normalizer that works would change its
/// tokenization — measured against the real file, it loads in 979 ms with the
/// normalizer intact. Blanket-dropping was right for the one export it was written
/// against and wrong for the next one; asking whether the value is the panic is
/// right for both, which is why this is a value-level function a test can pin
/// rather than a side effect hidden in the loader.
fn patched_value(value: &mut serde_json::Value) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    let null_charsmap = object.get("normalizer").is_some_and(has_null_charsmap);
    if null_charsmap {
        object.remove("normalizer");
    }
    null_charsmap
}

/// Load a SentencePiece or BPE `tokenizer.json`, dropping a null charsmap first.
///
/// The only patch is [`patched_value`]'s. This is the graph contract's first
/// dependency, so a tokenizer the build cannot read fails here rather than
/// mid-utterance.
fn patched_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let raw = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
    if !value.is_object() {
        anyhow::bail!("{} is not a tokenizer object", path.display());
    }
    let patched = patched_value(&mut value);
    // Only a patched document is written back out. An unpatched one is already the
    // bytes on disk, and a multilingual export's `tokenizer.json` is 16.5 MiB, so
    // re-serializing it would move the whole file to arrive at the same place.
    let bytes = if patched {
        serde_json::to_vec(&value)?
    } else {
        raw
    };
    let tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", path.display()))?;
    if patched {
        log::debug!(
            "{}: dropped a normalizer with a null charsmap, which tokenizers 0.23 cannot load",
            path.display()
        );
    }
    Ok(tokenizer)
}

/// How one export is fed, which is the whole difference between the two families
/// once the sessions are open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feed {
    /// A pair export: the encoder gets the text ids as they come out of the
    /// tokenizer, and the decoder opens on the export's own
    /// `decoder_start_token_id`.
    Fixed {
        /// The id the decoder opens its sequence with. Per export, not a constant:
        /// `pl-en` opens on 63429 and `mul-en` on 64171, and both are also the pad
        /// id. Feeding the wrong one is not an error — the graph runs and returns
        /// punctuation (`"..... .."`) — which is why it is read out of the model's
        /// own config rather than written down here.
        start: i64,
    },
    /// A multilingual export: the source code goes in front of the encoder's ids
    /// and closes them with `</s>`, and the decoder opens on
    /// `[start, target code]`.
    ///
    /// Both ends are needed and neither is the other's job: the source code tells
    /// the encoder which language it is reading, and the target code tells the
    /// decoder which language to write.
    LangTagged {
        /// The source language code's id, from the run's `--language`. Prefixed to
        /// the encoder's input ids, and **not** among the ids the decoder may emit.
        src: i64,
        /// The target language code's id, from `--translate`. Opens the decoder,
        /// and is never emitted mid-sentence.
        tgt: i64,
        /// `decoder_start_token_id`, which opens the decoder *before* the target
        /// code. It is equal to `eos_token_id` in this family, which is why the
        /// start==eos refusal in `translator_store` is scoped to pair exports.
        start: i64,
    },
}

impl Feed {
    /// Whether the tokenizer's own post-processor should run.
    ///
    /// `true` for a pair export, whose post-processor is what it has always been.
    /// `false` for a multilingual one, whose `tokenizer.json` ships a
    /// `TemplateProcessing` that hard-codes **`eng_Latn`** as the source prefix and
    /// appends `</s>` — right for English, silently wrong for every other source.
    /// Both ends are built here instead, out of the resolved codes.
    const fn adds_special_tokens(self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    /// Ids the framing spends around the text: none for a pair export, two for a
    /// multilingual one (the source code and the closing `</s>`). Counted against
    /// the input budget so the two cannot silently overrun it.
    const fn framing(self) -> usize {
        match self {
            Self::Fixed { .. } => 0,
            Self::LangTagged { .. } => 2,
        }
    }

    /// The ids a decode opens its sequence with.
    ///
    /// A pair export opens on one id: its own `decoder_start_token_id`.
    ///
    /// A multilingual export opens on **two** — `[start, target code]`. This is the
    /// one framing fact that cost the most to find, and it is measured, not
    /// inferred. NLLB's `forced_bos_token_id` mechanism is *generation*, not a
    /// different prefix: `generate` starts the decoder at
    /// `decoder_start_token_id` and then forces the first **output** to be the
    /// target code, so the sequence the decoder actually sees is
    /// `[start, target]`. Opening on the target code alone gives `"is"` for
    /// `"Kot siedzi na stole."`, and opening on `[start, target, start]` gives
    /// `""` with a language code winning the first step — all measured on the
    /// downloaded weights, all three with the same greedy decode.
    ///
    /// Two ids is also why `decoder_start_token_id == eos_token_id == 2` is legal
    /// for this family: the start token *is* the `</s>` decoder prefix, so a decode
    /// that opened and stopped on the same id is not the pathology
    /// `translator_store::read_config` refuses it for in a pair export.
    fn opening(self) -> Vec<i64> {
        match self {
            Self::Fixed { start } => vec![start],
            Self::LangTagged { start, tgt, .. } => vec![start, tgt],
        }
    }
}

/// The ids a decode may never emit.
///
/// A lookup table rather than a `Vec<i64>` scanned with `contains`, because a
/// multilingual export suppresses every language code it holds — 202 of them, over
/// a 256,206-wide vocabulary — so the scan would be 51 million comparisons on
/// *every* step. The table is sized to the highest suppressed id, and anything past
/// it is allowed, which is what a suppressed id outside the table should do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Suppressed(Vec<bool>);

impl Suppressed {
    /// A table blocking every id in `ids`. Negative and duplicate ids are dropped,
    /// and an empty set produces a table that blocks nothing.
    fn build(ids: impl IntoIterator<Item = i64>) -> Self {
        let ids: Vec<i64> = ids.into_iter().filter(|id| *id >= 0).collect();
        let width = ids.iter().copied().max().map_or(0, |max| max as usize + 1);
        let mut table = vec![false; width];
        for id in ids {
            table[id as usize] = true;
        }
        Self(table)
    }

    /// Whether `id` may not be emitted. Anything past the table is allowed.
    fn blocks(&self, id: usize) -> bool {
        self.0.get(id).copied().unwrap_or(false)
    }
}

/// A loaded translator: all three graphs, the tokenizer, and the family's
/// framing.
///
/// One `Translator` is one direction, and the direction is fixed for the life of
/// the process, so nothing here is rebuilt per utterance. The sessions live on
/// the translation thread, which is the only thread that touches them.
pub(crate) struct Translator {
    /// Upstream spec, for log lines.
    spec: String,
    /// The direction this translator runs, as a log should read it:
    /// `pol_Latn -> eng_Latn` for a multilingual export, `pl -> en` for a pair,
    /// and the bare spec for a directory, which names no direction at all.
    direction: String,
    encoder: Session,
    /// The prompt step, on the merged decoder: it takes `encoder_hidden_states`,
    /// opens the decoder on the family's opening ids, and hands back the cache
    /// every later step reads.
    decoder: Session,
    /// Every step after the prompt, on the cached decoder: one token against the
    /// cache, so the cost of a step no longer grows with the output.
    ///
    /// A separate graph because the merged one cannot be asked for its own cache:
    /// `use_cache_branch` is `true` only on a path that fails with an empty past
    /// (see [`PAST_PREFIX`]), and `false` returns the cache but recomputes
    /// everything in front of it. This graph takes the past as an input, has no
    /// `use_cache_branch` and no `encoder_hidden_states` — the cross-attention
    /// half travels inside the cache — and returns the self-attention half only.
    decoder_cached: Session,
    tokenizer: Tokenizer,
    /// How this family is fed: open on the export's own start token, or wrap the
    /// source in a language code and open on the target one.
    feed: Feed,
    /// `eos_token_id` from the export's config. Where a decode stops.
    eos_token: i64,
    /// Ids the decode may never emit. See [`Suppressed`].
    suppressed: Suppressed,
    /// Longest run of tokens one decode may emit, per family. See
    /// `translator_store::Family::max_new_tokens`.
    max_new_tokens: usize,
    /// The prompt step's cache inputs, empty: the merged graph's own
    /// `past_key_values.*` names with the shapes the prompt sends them in. Read
    /// at load because the graph's metadata does not change, and a step that
    /// re-walked it would rebuild 24 or 48 names per token.
    prompt_slots: CacheSlots,
    /// The same for the cached graph, which is what a decode's cache is keyed by.
    cached_slots: CacheSlots,
}

impl Translator {
    /// Load the encoder, both decoder graphs and the tokenizer.
    ///
    /// `prepared` comes from `translator_store::prepare`, so the five files are
    /// already on disk **and the export's own `config.json` has been read**: the
    /// family, the decoder shape and the token ids all arrive as facts rather than
    /// as placeholders a load has to second-guess. All three sessions run on the
    /// CPU with the settings the piper voice uses: one intra-op thread count, no
    /// parallel execution, no memory pattern.
    pub(crate) fn load(prepared: &Prepared, intra_threads: usize) -> Result<Self> {
        let r = &prepared.reference;
        let facts = &prepared.facts;
        let tokenizer = patched_tokenizer(&r.tokenizer)?;
        let encoder = cpu_session(&r.encoder, intra_threads, "translation encoder")?;
        let decoder = cpu_session(&r.decoder, intra_threads, "translation decoder")?;
        let decoder_cached = cpu_session(
            &r.decoder_with_past,
            intra_threads,
            "cached translation decoder",
        )?;
        require_input(&encoder, "input_ids", &r.encoder)?;
        require_input(&encoder, "attention_mask", &r.encoder)?;
        require_input(&decoder, "input_ids", &r.decoder)?;
        require_input(&decoder, "encoder_hidden_states", &r.decoder)?;
        require_input(&decoder, "encoder_attention_mask", &r.decoder)?;
        require_input(&decoder, "use_cache_branch", &r.decoder)?;
        // The cached graph opens one token at a time and reads the cross-attention
        // half out of the cache, so these three are the whole of its input list.
        require_input(&decoder_cached, "input_ids", &r.decoder_with_past)?;
        require_input(
            &decoder_cached,
            "encoder_attention_mask",
            &r.decoder_with_past,
        )?;
        let prompt_slots = cache_slots(&decoder, &r.decoder, &facts.shape)?;
        let cached_slots = cache_slots(&decoder_cached, &r.decoder_with_past, &facts.shape)?;
        // Both graphs have to name their slots the same way, or the cache the
        // prompt fills would be fed back under names the cached graph does not
        // declare -- which is not an error, it is a decode that reads nothing.
        if prompt_slots.names() != cached_slots.names() {
            anyhow::bail!(
                "{} and {} name their cache inputs differently; the two decoders \
                 cannot share one cache",
                r.decoder.display(),
                r.decoder_with_past.display(),
            );
        }

        let (feed, suppressed) = match (facts.family, &prepared.codes) {
            // A pair export: open on the id its own config publishes, and block
            // that id plus the pad. Both are in every export checked, and the pad
            // has to be blocked -- see `next_token`.
            (Family::Marian, _) => {
                let mut ids = vec![facts.start_token];
                ids.extend(facts.pad_token);
                (
                    Feed::Fixed {
                        start: facts.start_token,
                    },
                    Suppressed::build(ids),
                )
            }
            // A multilingual export: open on the target code, and block every
            // added token but `</s>`. That is the pad, the unknown, every one of
            // the 202 language codes and `<mask>` -- all of which the model can
            // otherwise emit mid-sentence, and a language code in the middle of a
            // translation is the model's decision to switch language, which is
            // never what the user asked for. Read off the tokenizer rather than
            // written down, so a re-converted vocabulary cannot go stale.
            (Family::Nllb, Some((src_code, tgt_code))) => {
                let token_of = |code: &str| -> Result<i64> {
                    tokenizer
                        .token_to_id(code)
                        .map(|id| id as i64)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                            "{} has no {code:?} token; this export cannot translate that language",
                            r.spec
                        )
                        })
                };
                let added: Vec<i64> = tokenizer
                    .get_added_tokens_decoder()
                    .keys()
                    .map(|id| *id as i64)
                    .filter(|id| *id != facts.eos_token)
                    .collect();
                (
                    Feed::LangTagged {
                        src: token_of(src_code)?,
                        tgt: token_of(tgt_code)?,
                        start: facts.start_token,
                    },
                    Suppressed::build(added),
                )
            }
            // `prepare` refuses a multilingual export with no codes before it gets
            // here, so this arm is unreachable. It is stated rather than collapsed
            // into the arm above because the two are genuinely different rules and
            // a silent fall-through would pick one of them by accident.
            (Family::Nllb, None) => anyhow::bail!(
                "{} was prepared without a direction; go through translator_store::prepare",
                r.spec
            ),
        };

        let direction = match &prepared.codes {
            Some((src, tgt)) => format!("{src} -> {tgt}"),
            None if !r.src.is_empty() => format!("{} -> {}", r.src, r.tgt),
            // A user directory names no direction, and saying " -> " would be a
            // lie about what sttd knows.
            None => r.spec.clone(),
        };
        log::info!(
            "translator: {spec} ({direction}, CPU, eos={eos}, {slots} cache slots, \
             cap {cap} tokens, cached decode)",
            spec = r.spec,
            eos = facts.eos_token,
            slots = facts.shape.cache_slots(),
            cap = facts.family.max_new_tokens(),
        );
        Ok(Self {
            spec: r.spec.clone(),
            direction,
            encoder,
            decoder,
            decoder_cached,
            tokenizer,
            feed,
            eos_token: facts.eos_token,
            suppressed,
            max_new_tokens: facts.family.max_new_tokens(),
            prompt_slots,
            cached_slots,
        })
    }

    /// Translate `text`, one sentence piece at a time, and join the results with
    /// a single space.
    ///
    /// Long text is split because the token budget is per piece: one short piece
    /// fits the 512-token window the export was converted for, while a whole
    /// paragraph does not.
    pub(crate) fn translate(&mut self, text: &str) -> Result<String> {
        let pieces = split_sentences(text);
        if pieces.is_empty() {
            return Ok(String::new());
        }
        let mut translated: Vec<String> = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            let piece = self.translate_piece(piece)?;
            if piece.is_empty() {
                log::debug!("translator {}: piece produced no text", self.spec);
                continue;
            }
            translated.push(piece);
        }
        Ok(translated.join(" "))
    }

    /// One piece through the encoder, the prompt step and the cached decode.
    ///
    /// The framing is the family's, and it is the only place [`Feed`] is read:
    /// a pair export encodes the text as it stands, a multilingual one encodes it
    /// bare and wraps it in `[src] + ids + [eos]`.
    fn translate_piece(&mut self, piece: &str) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(piece, self.feed.adds_special_tokens())
            .map_err(|e| anyhow::anyhow!("failed to tokenize the source text: {e}"))?;
        let text: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
        // The framing counts against the budget, so the two cannot overrun it
        // between them: a 512-token piece plus a code and an `</s>` is 514 ids.
        let budget = TRANSLATE_MAX_INPUT_TOKENS.saturating_sub(self.feed.framing());
        let text: Vec<i64> = if text.len() > budget {
            log::warn!(
                "translator {}: piece of {} tokens cut to {budget}",
                self.spec,
                text.len()
            );
            text[..budget].to_vec()
        } else {
            text
        };
        // One `match` builds the whole frame, because which ids it holds is the
        // family's rule rather than two independent pushes.
        let mut ids: Vec<i64> = Vec::with_capacity(text.len() + self.feed.framing());
        match self.feed {
            Feed::Fixed { .. } => ids.extend_from_slice(&text),
            Feed::LangTagged { src, .. } => {
                ids.push(src);
                ids.extend_from_slice(&text);
                // The closing `</s>` is what tells the encoder the source sentence
                // has ended. Without it the source code in front is just another
                // token.
                ids.push(self.eos_token);
            }
        }
        let hidden = self.run_encoder(&ids)?;
        let enc_len = ids.len();
        // The one attention mask both steps take, built once: the prompt step
        // masks the encoder for the whole decode, and so does every step after it.
        let encoder_mask = vec![1i64; enc_len];

        // The decoder opens on the family's opening ids and nothing else: the source
        // text reaches the model through `encoder_hidden_states` and nowhere else.
        //
        // For a pair export that is `decoder_start_token_id` from `config.json`,
        // which is the `<pad>` id and **not** 2. Opening on 2 (which is `.` in that
        // vocabulary) is what produced `"i i i i ..."` on every input: the same
        // graphs, the same ids and the same weights give correct English from
        // 63429. For a multilingual export it is `[start, target code]` — see
        // `Feed::opening`, where the two-id shape is measured against the weights.
        //
        // That run also fills the cache: it is the only step that can, and it is
        // why the prompt runs on the merged graph at all.
        let opening = self.feed.opening();
        let (mut cache, first) = self.run_prompt_step(&opening, &hidden, &encoder_mask)?;

        let mut tokens: Vec<u32> = Vec::new();
        let mut next = first;
        let mut produced = 0usize;
        loop {
            if next as i64 == self.eos_token {
                break;
            }
            tokens.push(next as u32);
            produced += 1;
            if produced >= self.max_new_tokens {
                log::warn!(
                    "translator {}: decode hit the {}-token cap; \
                     the translation of this piece is cut short",
                    self.spec,
                    self.max_new_tokens
                );
                break;
            }
            next = self.run_cached_step(next as i64, &mut cache, &encoder_mask)?;
        }
        self.tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("failed to decode the translated tokens: {e}"))
    }

    /// The encoder over one piece's token ids, returning its hidden states.
    fn run_encoder(&mut self, ids: &[i64]) -> Result<ArrayD<f32>> {
        let len = ids.len();
        let named: Vec<(&str, SessionInputValue<'_>)> = vec![
            (
                "input_ids",
                Tensor::from_array((vec![1usize, len], ids.to_vec()))?.into(),
            ),
            (
                "attention_mask",
                Tensor::from_array((vec![1usize, len], vec![1i64; len]))?.into(),
            ),
        ];
        let outputs = self
            .encoder
            .run(named)
            .context("failed to run the translation encoder")?;
        let hidden = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("the encoder returned no last_hidden_state"))?
            .try_extract_array::<f32>()
            .context("the encoder hidden state is not an f32 array")?;
        let shape = hidden.shape().to_vec();
        match shape.as_slice() {
            [1, seq, _model_dim] if *seq == len => {}
            other => {
                return Err(anyhow::anyhow!(
                    "expected encoder states [1, {len}, model_dim], got {other:?}"
                ))
            }
        }
        Ok(hidden.to_owned())
    }

    /// The prompt step: the opening ids through the merged decoder, returning the
    /// cache it filled and the first token to emit.
    ///
    /// **This is the only step that runs the merged graph, and the only one that
    /// can fill the cache.** The merged decoder is the graph that computes the
    /// cross-attention keys and values in the first place, and it is the graph
    /// that takes `encoder_hidden_states`; the cached one has neither. So the
    /// prompt runs here, and every token after it runs on `decoder_cached`.
    ///
    /// `use_cache_branch` is `false`, and that is what makes the run succeed *and*
    /// hand back a cache. `true` is the branch that fails on an empty past:
    ///
    /// ```text
    /// If node 'optimum::if' ... encoder_attn/Reshape_4 ...
    /// The dimension with value zero exceeds the dimension size of the input tensor.
    /// ```
    ///
    /// The empty past is still sent because omitting the inputs fails that same
    /// node with `No operator matching`. The `false` branch returns the current
    /// keys and values under `present.` — measured on both exports, and the
    /// reason the whole split works: the cache this step fills is the one every
    /// later step reads, so nothing is recomputed and no step grows with the
    /// output.
    fn run_prompt_step(
        &mut self,
        opening: &[i64],
        hidden: &ArrayD<f32>,
        encoder_mask: &[i64],
    ) -> Result<(KvCache, usize)> {
        let slots = self.prompt_slots.empty_inputs();
        let mut named: Vec<(&str, SessionInputValue<'_>)> =
            Vec::with_capacity(self.prompt_slots.empty.len() + 4);
        named.push((
            "input_ids",
            Tensor::from_array((vec![1usize, opening.len()], opening.to_vec()))?.into(),
        ));
        named.push((
            "encoder_hidden_states",
            TensorRef::from_array_view(hidden.view())?.into(),
        ));
        named.push((
            "encoder_attention_mask",
            Tensor::from_array((vec![1usize, encoder_mask.len()], encoder_mask.to_vec()))?.into(),
        ));
        named.push(("use_cache_branch", use_cache_branch_tensor(false)?.into()));
        named.extend(slots);
        let outputs = self
            .decoder
            .run(named)
            .context("failed to run the translation decoder's prompt step")?;
        let first = pick_token(&outputs, &self.suppressed, "the prompt step")?;
        let cache = KvCache::capture(&self.cached_slots, &outputs)
            .context("failed to capture the decoder's key/value cache")?;
        Ok((cache, first))
    }

    /// One generated token through the cached decoder, against the cache the
    /// prompt step filled, returning the next token to emit.
    ///
    /// One id in, one position of work: the tokens already emitted are in the
    /// cache, and the cross-attention keys and values are in it too, which is why
    /// this graph takes neither `encoder_hidden_states` nor a cache flag. The
    /// whole decode is therefore **O(n) in output length** rather than the O(n²) a
    /// single graph re-running its own prefix costs.
    fn run_cached_step(
        &mut self,
        token: i64,
        cache: &mut KvCache,
        encoder_mask: &[i64],
    ) -> Result<usize> {
        let mut named: Vec<(&str, SessionInputValue<'_>)> = Vec::with_capacity(cache.len() + 2);
        named.push((
            "input_ids",
            Tensor::from_array((vec![1usize, 1], vec![token]))?.into(),
        ));
        named.push((
            "encoder_attention_mask",
            Tensor::from_array((vec![1usize, encoder_mask.len()], encoder_mask.to_vec()))?.into(),
        ));
        named.extend(cache.named_inputs()?);
        let outputs = self
            .decoder_cached
            .run(named)
            .context("failed to run the translation decoder's cached step")?;
        let next = pick_token(&outputs, &self.suppressed, "a cached step")?;
        cache
            .update_self(&outputs)
            .context("failed to read the decoder's next key/value cache")?;
        Ok(next)
    }
}

/// The token a decoder run's last logits row points at.
///
/// The row is read in place: `try_extract_array` hands back a view into the
/// session's own output, so copying the row out would move the whole vocabulary —
/// 63,430 floats for a pair, 256,206 for a multilingual one — once per token to
/// look at it a single time.
///
/// A free function rather than a method because a run's outputs borrow the
/// session, and the suppressed table is a different field of the translator; the
/// two borrows are only disjoint if neither reaches for all of `self`.
fn pick_token(
    outputs: &ort::session::SessionOutputs<'_>,
    suppressed: &Suppressed,
    what: &str,
) -> Result<usize> {
    let logits = outputs
        .get("logits")
        .ok_or_else(|| anyhow::anyhow!("{what} returned no logits"))?
        .try_extract_array::<f32>()
        .with_context(|| format!("{what} logits are not an f32 array"))?;
    let (row, _vocab) =
        last_logits_row(&logits).with_context(|| format!("{what} returned unusable logits"))?;
    Ok(next_token(row, suppressed))
}

/// The rank of `use_cache_branch`: one axis holding one flag.
///
/// The merged decoder declares this input as a BOOL tensor of shape `[1]`, not
/// as a rank-0 scalar. A rank-0 tensor is rejected before the graph runs
/// (`Invalid rank for input: use_cache_branch Got: 0 Expected: 1`). Read out of
/// both exports' protobuf rather than assumed: element type 9 (BOOL) and a single
/// fixed dimension of 1, identically in a Marian and an NLLB merged decoder.
const USE_CACHE_BRANCH_SHAPE: [i64; 1] = [1];

/// The next token id for one logits row: greedy argmax over the ids a decode may
/// emit.
///
/// Marian has no `<extra_id_*>` sentinel block (that is a T5 artifact), but the
/// **pad id does have to be suppressed**, and on the int8 exports it is not
/// optional. `opus-mt-mul-en` pins the pad logit at exactly `0.000` while every
/// real token sits near `-40`, so a plain argmax picks `<pad>` on every step and
/// the decode runs to the cap emitting nothing — observed, with `"..... .."` as
/// the visible symptom. Upstream's `generation_config` lists the same id under
/// `bad_words_ids`, which is how the reference implementation avoids it, and
/// `pl-en` is unaffected only by luck of its quantization.
///
/// A multilingual export suppresses a far bigger set — every added token but
/// `</s>`, so all 202 language codes among them — which is why the membership test
/// is a table lookup and not a scan of the row.
///
/// The winning id is not skipped outright: the best id that is allowed wins
/// instead, which costs one pass over the row rather than a sort of it. A tie
/// keeps the lower id, the rule `argmax_index` uses. A row with nothing allowed in
/// it falls back to the plain argmax, in a second pass — which neither family
/// ever reaches, since neither suppresses a whole row, and which is why the
/// common case compares each logit once rather than twice.
fn next_token(logits: &[f32], suppressed: &Suppressed) -> usize {
    let mut best: Option<(usize, f32)> = None;
    for (id, &logit) in logits.iter().enumerate() {
        if suppressed.blocks(id) {
            continue;
        }
        if best.is_none_or(|(_, top)| logit > top) {
            best = Some((id, logit));
        }
    }
    if best.is_some() {
        return best.map_or(0, |(id, _)| id);
    }
    argmax(logits)
}

/// The largest logit's id, the lower one on a tie. The fallback for a row that
/// suppresses every id in it: the decode still has to produce a token it can act
/// on rather than nothing.
fn argmax(logits: &[f32]) -> usize {
    let mut best: Option<(usize, f32)> = None;
    for (id, &logit) in logits.iter().enumerate() {
        if best.is_none_or(|(_, top)| logit > top) {
            best = Some((id, logit));
        }
    }
    best.map_or(0, |(id, _)| id)
}

/// The last logits row of a `[1, positions, vocab]` decoder output, and the row
/// stride.
///
/// The final position is the one that predicts the next token; the ones before it
/// predict tokens already chosen. `ort` hands back a dense tensor, so the row is
/// one contiguous `vocab`-long run of the flat data. The rank is checked rather
/// than assumed: an unexpected shape means the graph is not the one this decode
/// was written against, and every row read off it would be silently wrong.
fn last_logits_row<'a>(logits: &'a ArrayViewD<'a, f32>) -> Result<(&'a [f32], usize)> {
    let shape = logits.shape();
    let vocab = match shape {
        [_batch, positions, vocab] if *positions > 0 && *vocab > 0 => *vocab,
        other => {
            anyhow::bail!("expected decoder logits [1, positions, vocab], got {other:?}")
        }
    };
    let flat = logits
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("the decoder logits are not contiguous"))?;
    Ok((&flat[flat.len() - vocab..], vocab))
}

/// The `use_cache_branch` flag tensor for one decoder run.
///
/// Built here rather than inline at the call site because the rank is the whole
/// point: a scalar of the right value is still the wrong tensor, and only the run
/// itself says so. The value is always `false` (see `run_prompt_step`); the shape
/// is what a test can check without weights.
fn use_cache_branch_tensor(use_cache: bool) -> ort::Result<Tensor<bool>> {
    Tensor::from_array((USE_CACHE_BRANCH_SHAPE, vec![use_cache]))
}

/// The shape of one empty cache tensor, built from the graph's declared axes and
/// the export's own config.
///
/// A cache slot is `[batch, heads, sequence, head_dim]`, so axis 2 is the one that
/// goes to zero. Every other axis keeps the value the metadata declares, and an
/// axis the metadata leaves dynamic falls back to `shape` — read from
/// `config.json`, never compiled in, because a zero-length tensor cannot carry a
/// dynamic axis and the two families disagree on both numbers that matter (8 heads
/// of 64 against 16 of 64).
fn empty_cache_shape(declared: &[Option<usize>], shape: GraphShape) -> Vec<usize> {
    let default = [1usize, shape.heads, 0, shape.head_dim()];
    if declared.len() != default.len() {
        return default.to_vec();
    }
    std::array::from_fn::<usize, 4, _>(|axis| {
        if axis == 2 {
            0
        } else {
            declared[axis].unwrap_or(default[axis])
        }
    })
    .to_vec()
}

/// A CPU session with the settings the piper voice uses.
///
/// `ort`'s builder errors are its own type rather than `anyhow`-friendly, so
/// each step is mapped with a message instead of `.context()`.
fn cpu_session(
    path: &std::path::Path,
    intra_threads: usize,
    what: &'static str,
) -> Result<Session> {
    Session::builder()
        .map_err(ort_err(what, "build the session builder"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort_err(what, "set the optimization level"))?
        .with_intra_threads(intra_threads.max(1))
        .map_err(ort_err(what, "set the thread count"))?
        .with_inter_threads(1)
        .map_err(ort_err(what, "set the inter-op thread count"))?
        .with_memory_pattern(false)
        .map_err(ort_err(what, "set the memory pattern"))?
        .with_parallel_execution(false)
        .map_err(ort_err(what, "set parallel execution"))?
        .commit_from_file(path)
        .with_context(|| format!("failed to load {}", path.display()))
}

/// Attach a message to an `ort` builder error.
///
/// `ort::Error<T>` is parameterized by the value the failed call would have
/// returned, and each step of the builder chain fails with a different one, so
/// this is generic over `T`.
fn ort_err<T>(what: &'static str, step: &'static str) -> impl Fn(ort::Error<T>) -> anyhow::Error {
    move |e| anyhow::anyhow!("failed to {step} for the {what}: {e:?}")
}

/// Fail at load, with the path, when the graph does not take `name`.
fn require_input(session: &Session, name: &str, path: &std::path::Path) -> Result<()> {
    if session.inputs().iter().any(|i| i.name() == name) {
        return Ok(());
    }
    let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    anyhow::bail!(
        "{} takes no {name} input; its inputs are {names:?}",
        path.display()
    )
}

/// What one decoder session declares about its key/value cache, read once at load.
///
/// Both halves of the answer, because a decode needs both: the empty tensors the
/// prompt step sends, and which of them a cached step replaces.
struct CacheSlots {
    /// `(input name, empty tensor)` in the order the graph declares its inputs.
    /// Built at load because the graph's metadata does not change and a step that
    /// re-walked it would rebuild 24 or 48 names per generated token.
    empty: Vec<(String, ArrayD<f32>)>,
    /// `(index into empty, output name)` for the self-attention slots only — the
    /// ones a step replaces. The output name is the input name without its
    /// `past_key_values.` prefix, so a name and the tensor read into it cannot
    /// drift apart.
    self_attention: Vec<(usize, String)>,
}

impl CacheSlots {
    /// The slot names, sorted, for comparing two graphs' cache contracts.
    fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.empty.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// The empty slots as named inputs, borrowed from the tensors built at load.
    ///
    /// Views rather than fresh tensors: a zero-length slot holds no elements, so
    /// the only per-step cost is naming it.
    fn empty_inputs(&self) -> impl Iterator<Item = (&str, SessionInputValue<'_>)> {
        self.empty.iter().map(|(name, tensor)| {
            (
                name.as_str(),
                TensorRef::from_array_view(tensor.view())
                    .expect("a cache slot built here must be convertible")
                    .into(),
            )
        })
    }
}

/// The key/value cache one utterance's decode runs against: what the prompt step
/// produced, fed back to every step after it.
///
/// **Two halves, and only one of them moves.** A slot is
/// `[batch, heads, sequence, head_dim]`, and the export's own name says which
/// attention a slot belongs to: `past_key_values.N.decoder.*` is self-attention
/// and grows by one token per step, `past_key_values.N.encoder.*` is
/// cross-attention and is a function of the encoder output alone. The cached
/// decoder returns **only the self half** under `present.` — measured on both
/// exports — because the cross half is an input it never rewrites. So the cross
/// tensors the prompt filled are handed back unchanged for the whole decode, and
/// reading them out of every step's output would copy the encoder's keys and
/// values once per token for no change at all.
///
/// Slot order is the cached graph's own, read at load and shared with the
/// [`CacheSlots`] that names them, so a cross-attention tensor cannot end up in a
/// self-attention slot. That mistake would be no error: both halves are
/// `[1, heads, n, head_dim]`, and only the transcript would show it.
struct KvCache {
    /// `(input name, tensor)` in slot order: per layer, the self-attention key
    /// and value, then the cross-attention key and value.
    slots: Vec<(String, ArrayD<f32>)>,
    /// `(index into slots, output name)` for the self-attention slots. See
    /// [`CacheSlots::self_attention`]; the cross half is not in it because a step
    /// never rewrites it.
    self_attention: Vec<(usize, String)>,
}

impl KvCache {
    /// Both halves, read out of a prompt run's outputs by the names
    /// [`CacheSlots`] derived from the cached graph.
    ///
    /// Owned rather than borrowed: the tensors belong to the session's output
    /// borrow, and the decode outlives it. The copy is the point of caching — one
    /// copy of the encoder's keys and values, against re-running the whole prefix
    /// on every step.
    fn capture(slots: &CacheSlots, outputs: &ort::session::SessionOutputs<'_>) -> Result<Self> {
        let read = |name: &str| -> Result<ArrayD<f32>> {
            outputs
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("the prompt step returned no {name}"))?
                .try_extract_array::<f32>()
                .map(|view| view.to_owned())
                .with_context(|| format!("{name} is not an f32 array"))
        };
        let mut cache = KvCache {
            slots: Vec::with_capacity(slots.empty.len()),
            self_attention: slots.self_attention.clone(),
        };
        for (name, _) in &slots.empty {
            cache.slots.push((name.clone(), read(&present_name(name))?));
        }
        Ok(cache)
    }

    /// How many slots this cache feeds, for sizing the step's input list.
    fn len(&self) -> usize {
        self.slots.len()
    }

    /// Every slot as a named input, in slot order, borrowed rather than rebuilt.
    fn named_inputs(&self) -> Result<Vec<(&str, SessionInputValue<'_>)>> {
        self.slots
            .iter()
            .map(|(name, tensor)| {
                Ok((
                    name.as_str(),
                    TensorRef::from_array_view(tensor.view())
                        .with_context(|| format!("{name} must be convertible"))?
                        .into(),
                ))
            })
            .collect()
    }

    /// Replace the self-attention half from one cached run, leaving the
    /// cross-attention half as the prompt left it.
    ///
    /// Only the self half is read out, and that is the whole reason this is not
    /// `cache.slots = capture(...)`: the cross tensors are megabytes that would
    /// be copied out of the output and back on every single token, unchanged.
    fn update_self(&mut self, outputs: &ort::session::SessionOutputs<'_>) -> Result<()> {
        for (index, name) in &self.self_attention {
            self.slots[*index].1 = outputs
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("a cached step returned no {name}"))?
                .try_extract_array::<f32>()
                .with_context(|| format!("{name} is not an f32 array"))?
                .to_owned();
        }
        Ok(())
    }
}

/// The name a `past_key_values.N.*` cache input is returned under: the same
/// string without its prefix (`present.N.decoder.key`).
///
/// Derived from the input name rather than read off the graph's output list, which
/// is what makes a name and the tensor stored under it one fact instead of two.
fn present_name(input_name: &str) -> String {
    input_name.replacen(PAST_PREFIX, "present.", 1)
}

/// Whether a cache slot is the self-attention half — the only one a cached step
/// rewrites, and the only one a slot's name has to be read to tell apart.
///
/// The cross half is named too, and that is the whole point of asking: both are
/// `[1, heads, n, head_dim]`, so a decoder key fed to an encoder slot is not a
/// shape error, it is a translation that quietly falls apart.
fn is_self_slot(name: &str) -> bool {
    name.contains(SELF_SLOT)
}

/// Read one decoder session's cache contract, and refuse a graph that does not
/// have the one this decode needs.
///
/// The export must declare exactly as many slots as its own `config.json` says it
/// has layers (four per layer: a self- and a cross-attention key and value), with
/// no duplicates, all of them f32, and it must name both halves — a session
/// declaring only one would decode against a cache silently missing half its
/// attention.
///
/// Every one of those is worth a load-time error rather than a per-step failure,
/// because they are properties of the export rather than of the text. Nor is the
/// count a constant: a Marian pair declares 24 and NLLB 48, so it is compared
/// against the geometry read out of the model rather than a number written here.
fn cache_slots(
    session: &Session,
    path: &std::path::Path,
    shape: &GraphShape,
) -> Result<CacheSlots> {
    let mut empty: Vec<(String, ArrayD<f32>)> = Vec::new();
    let mut self_attention: Vec<(usize, String)> = Vec::new();
    for input in session.inputs() {
        if !input.name().starts_with(PAST_PREFIX) {
            continue;
        }
        let name = input.name().to_string();
        let declared = match input.dtype() {
            ValueType::Tensor { shape, ty, .. }
                if *ty == ort::value::TensorElementType::Float32 =>
            {
                shape
                    .as_ref()
                    .iter()
                    .map(|d| if *d < 0 { None } else { Some(*d as usize) })
                    .collect::<Vec<_>>()
            }
            other => anyhow::bail!(
                "{} holds cache input {name} of a type this decode does not build: \
                 {other:?}",
                path.display()
            ),
        };
        // Which half a slot belongs to is read off the export's own name, so the
        // self slots are the ones a step replaces and the cross slots the ones it
        // leaves alone. A slot named neither is refused rather than guessed at:
        // guessing would put an encoder key where a decoder key belongs, and both
        // are `[1, heads, n, head_dim]`, so nothing would say so.
        if is_self_slot(&name) {
            self_attention.push((empty.len(), present_name(&name)));
        } else if !name.contains(CROSS_SLOT) {
            anyhow::bail!(
                "{} names cache input {name} as neither self- nor cross-attention; \
                 this decode cannot tell which half of the cache it belongs to",
                path.display()
            )
        }
        empty.push((
            name,
            ArrayD::from_shape_vec(empty_cache_shape(&declared, *shape), vec![])?,
        ));
    }
    let expected = shape.cache_slots();
    if empty.len() != expected {
        anyhow::bail!(
            "{} declares {} cache inputs, expected {expected} ({} layers)",
            path.display(),
            empty.len(),
            shape.layers,
        );
    }
    if self_attention.is_empty() {
        anyhow::bail!(
            "{} declares no self-attention cache input; a cached step would have \
             nothing to write its keys into",
            path.display()
        );
    }
    let mut names: Vec<&str> = empty.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    if names.len() != expected {
        anyhow::bail!("{} declares duplicate cache input names", path.display());
    }
    Ok(CacheSlots {
        empty,
        self_attention,
    })
}

// ---------------------------------------------------------------------------
// Runtime control
// ---------------------------------------------------------------------------
//
// Everything above loads a translator once; everything here decides *which* one
// is live, and swaps it while the run continues.
//
// The state lives in this module rather than in `dbus` because it owns the
// weights and the loader threads, and `translator_store` plus three `ort`
// sessions are not a bus concern: `dbus` holds this handle and forwards calls,
// the way `asr` holds `control::DictionaryState`. The dependency therefore
// points one way, `dbus -> translate`: what this module announces goes through
// a hook `dbus::spawn` installs (`TranslationState::set_change_hook`).
//
// Two rules hold the whole thing together:
//
// * **One copy of the weights, behind one lock.** The translation thread looks
//   the translator up per job instead of caching it, because a second copy is
//   exactly what would make `Disable(true)` unable to free the weights at once.
//   The cost is that a mutating call waits for a translation already running
//   (60 to 130 ms for a pair, ~700 ms for a multilingual export) and never for a
//   load.
// * **The generation rides in that same lock.** A load that finished after its
//   request was superseded must not install itself, and a `Disable` that promised
//   to free the weights must not be raced by one still arriving, so the check and
//   the write are one critical section rather than two steps.

/// The run's own translation settings, resolved once from the CLI.
///
/// A load needs all four, and at runtime only the bus is left to ask with — so
/// they are captured here rather than re-read from `Cli` on the bus thread. The
/// source language is the one Whisper was told to decode in, because that is what
/// the transcript is; it is fixed for the run, so a runtime switch changes the
/// target only.
pub(crate) struct TranslationPlan {
    /// `--language`, or `None` under `--language auto`: the source is then
    /// unknown per utterance, which a pair export can answer for and a
    /// multilingual one cannot.
    source: Option<String>,
    /// `--translate-model`, overriding which translator a direction resolves to.
    model: Option<String>,
    /// The language `--translate` asked for, if any. What `Enable` reloads when
    /// nothing is loaded.
    configured: Option<String>,
    intra_threads: usize,
    /// `--voice`, kept only under `--speak`, so a runtime switch can warn about
    /// the same voice/language mismatch the startup path warns about.
    voice: Option<String>,
}

impl TranslationPlan {
    /// The warnings are the startup ones: a `--translate` with no code cannot name
    /// a direction, and a `--translate-model` with nothing asking for a
    /// translation downloads nothing.
    ///
    /// The second is only *unused* when no one can ask at runtime, which is why the
    /// session bus counts as an asker: the override now applies to a
    /// `SetTargetLanguage` too, not only to a `--translate` named at startup. So
    /// `--translate-model` on its own warns under `--no-dbus` and in file mode, and
    /// stays quiet in microphone mode with the bus, where it does something.
    /// A plan for a test: one named source, no override, nothing configured and no
    /// load. Every direction it can resolve is a catalog lookup, so a bus test
    /// exercises the refusals without weights and without a network.
    #[cfg(test)]
    pub(crate) fn for_tests(source: &str) -> Self {
        Self {
            source: Some(source.to_string()),
            model: None,
            configured: None,
            intra_threads: 1,
            voice: None,
        }
    }

    /// A plan whose translator is named outright, so the catalog's pair table is
    /// bypassed: `reference_for` then validates the spec alone and any target is
    /// accepted. That is what lets a test drive the *load* half — a switch, a
    /// failure — against targets the real catalog would refuse before a thread ever
    /// started.
    #[cfg(test)]
    pub(crate) fn for_tests_with_model(source: &str, model: &str) -> Self {
        Self {
            model: Some(model.to_string()),
            ..Self::for_tests(source)
        }
    }

    /// A plan as `--translate <target>` makes it, so the startup path is drivable
    /// without weights.
    #[cfg(test)]
    pub(crate) fn for_tests_configured(source: &str, target: &str) -> Self {
        Self {
            configured: Some(target.to_string()),
            ..Self::for_tests_with_model(source, "nllb-200-distilled-600M")
        }
    }

    /// Read the plan out of the flags, warning about the combinations that
    /// cannot work. See the type for what each field is.
    pub(crate) fn from_cli(cli: &crate::cli::Cli) -> Self {
        let configured = cli
            .translate
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase());
        let model = cli
            .translate_model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if configured.is_none() {
            if cli.translate.is_some() {
                log::warn!("--translate needs a language code (e.g. en); the step is off");
            }
            if model.is_some() && !can_ask_at_runtime(cli) {
                log::warn!("--translate-model is ignored without --translate");
            }
        }
        let source = match cli.language() {
            crate::whisper_core::types::Language::Explicit(code)
            | crate::whisper_core::types::Language::System(code) => Some(code),
            crate::whisper_core::types::Language::Auto => None,
        };
        Self {
            source,
            model,
            configured,
            intra_threads: crate::asr::resolve_n_threads(cli.threads, crate::asr::auto_threads())
                .max(1) as usize,
            voice: cli.speak.then(|| cli.voice.clone()),
        }
    }

    /// The direction this run translates `target` in.
    fn direction<'a>(&'a self, target: &'a str) -> crate::translator_store::Direction<'a> {
        crate::translator_store::Direction {
            src: self.source.as_deref(),
            tgt: target,
        }
    }

    /// The source language as a log or a spec reads it.
    #[cfg(test)]
    fn source_label(&self) -> &str {
        self.source.as_deref().unwrap_or("auto")
    }
}

/// Whether this run has any way to ask for a translation other than `--translate`:
/// the session bus in microphone mode.
fn can_ask_at_runtime(cli: &crate::cli::Cli) -> bool {
    !cli.no_dbus && cli.file.is_none()
}

/// Resolve, download and load the translator for `target`, on the calling thread.
///
/// The whole startup path for one language, and the whole runtime path too: the
/// direction comes from the plan, which the flags fixed. Whether a direction is
/// even possible is answered before anything is fetched — see
/// [`crate::translator_store::reference_for`] — so a refusal costs nothing.
pub(crate) fn load_for(plan: &TranslationPlan, target: &str) -> Result<Translator> {
    let direction = plan.direction(target);
    let prepared = match plan.model.as_deref() {
        Some(spec) => crate::translator_store::prepare(spec, direction)
            .with_context(|| format!("failed to prepare translator {spec}"))?,
        None => crate::translator_store::prepare_from_direction(direction)?,
    };
    log_translator_direction(&prepared, target);
    if let Some(voice) = plan.voice.as_deref() {
        warn_on_voice_language_mismatch(voice, target);
    }
    Translator::load(&prepared, plan.intra_threads).context("failed to load the translator")
}

/// Log the direction a run translates in, and warn when the engine does not decode
/// in the language that direction assumes.
///
/// A **multilingual** export is the only case with an assumption to state: it is
/// told which language the source is, so under an engine that decides for itself
/// the flag names what the *translator* should read, not what the engine should
/// hear. Parakeet detects the spoken language per utterance and ignores
/// `--language` for transcription, so without this line `--language pl` alongside
/// a multilingual model reads as "transcribe Polish" — and a Polish sentence
/// spoken in German would be translated as Polish.
///
/// A **pair** export has no such gap: its weights only do one direction, so
/// `--language auto` picks a many-source model and a named code picks the matching
/// pair, and either way the direction was already resolved rather than assumed.
/// Nothing is warned about, which is the point of keeping the check on `codes`
/// rather than on the translator existing.
fn log_translator_direction(prepared: &Prepared, target: &str) {
    log::info!(
        "translate: emitting every accepted transcript into {target} (translator {})",
        prepared.reference.spec
    );
    let Some((src, _)) = &prepared.codes else {
        return;
    };
    log::warn!(
        "the selected engine detects the spoken language per utterance and ignores \
         --language for transcription, but {NLLB_SPEC_IN_LOG} is told to read every \
         transcript as {src}; speech in any other language will be mistranslated"
    );
}

/// The spec name, spelled once here rather than interpolated into the format
/// string above: `translator_store` owns the constant and a renamed spec must not
/// leave a stale copy in a warning.
const NLLB_SPEC_IN_LOG: &str = crate::translator_store::NLLB_SPEC;

/// The language part of a piper voice key (`pl_PL-...`, `en_US-...`), which is
/// the only place a piper voice states its language. Split out so the rule is
/// testable without a voice or a translator.
pub(crate) fn voice_language(key: &str) -> Option<&str> {
    let key = key.trim();
    if key.contains('/') || key.contains('\\') {
        return None;
    }
    let lang = key.split(['_', '-']).next()?;
    (!lang.is_empty()).then_some(lang)
}

/// Warn when the `--speak` voice speaks a different language than the one being
/// written to the clipboard. Not an error: a translation can still be read aloud
/// in the wrong language, which is bad but not fatal, and the user may know
/// their voice speaks it.
fn warn_on_voice_language_mismatch(voice: &str, target: &str) {
    let Some(voice_lang) = voice_language(voice) else {
        return;
    };
    if voice_lang.eq_ignore_ascii_case(target) {
        return;
    }
    log::warn!(
        "--voice {voice} is a {voice_lang} voice but translation writes {target}: \
         sttd will read the translation aloud in {voice_lang} \
         (pick a {target} voice with --voice)"
    );
}

/// How a state of `L` produces one: from a plan and a target language.
///
/// A trait so the pipeline that loads, installs and announces can be driven in a
/// test without weights, and so production's one implementation is named in the
/// type rather than picked inside the body.
pub(crate) trait Loads<L>: Clone + Send + Sync + 'static {
    /// Resolve, download and load. The real one is [`load_for`].
    fn load(&self, plan: &TranslationPlan, target: &str) -> Result<L>;
}

/// The production loader: the real translator, built from the run's own flags.
#[derive(Clone, Copy)]
pub(crate) struct RealTranslator;

impl Loads<Translator> for RealTranslator {
    fn load(&self, plan: &TranslationPlan, target: &str) -> Result<Translator> {
        load_for(plan, target)
    }
}

/// What the state needs from a loaded translator: translate one string, and name
/// itself in a log line.
///
/// A trait rather than `Translator` directly so the state is testable. A
/// `Translator` is two or three `ort` sessions and a tokenizer, so no unit test
/// can build one, and the bookkeeping around a load — refusals, the generation
/// that cancels one, which field each switch clears — is exactly what needs
/// testing. Production uses the one implementor below.
pub(crate) trait Loaded: Send + 'static {
    /// Translate one utterance. A failure is not fatal: the caller falls back to
    /// the source text.
    fn translate(&mut self, source: &str) -> Result<String>;

    /// The upstream spec, for log lines and the `Model` property.
    fn spec(&self) -> &str;

    /// The direction, as a log should read it.
    fn direction(&self) -> &str;
}

impl Loaded for Translator {
    fn translate(&mut self, source: &str) -> Result<String> {
        Translator::translate(self, source)
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn direction(&self) -> &str {
        &self.direction
    }
}

/// The one copy of the loaded weights, and the generation that names it.
///
/// `generation` moves with the translator on purpose: comparing a request's
/// generation against this one *while holding the lock that guards the
/// translator* is what makes "a superseded load must not install" and "`Disable`
/// frees the weights at once" true rather than nearly true.
pub(crate) struct TranslatorSlot<T> {
    pub(crate) translator: Option<T>,
    pub(crate) generation: u64,
}

/// What the bus reads. A snapshot of the truth, written by whoever holds the
/// slot, because a property read must not queue behind a running translation.
#[derive(Default, Clone)]
struct Cached {
    enabled: bool,
    target: String,
    model: String,
}

/// The translation step as the rest of the program and the bus see it: on or
/// off, writing which language, with which translator, and loading one now.
///
/// A cheap `Clone`, so the bus thread, the ASR worker and the translation thread
/// all hold the same state.
pub(crate) struct TranslationState<T = Translator, L = RealTranslator> {
    slot: Arc<Mutex<TranslatorSlot<T>>>,
    /// What the user asked for. `Enabled` is this **and** a loaded translator, so
    /// intent and capability never disagree in a report.
    wanted: Arc<AtomicBool>,
    /// The language of the load in flight, empty when nothing is loading.
    requested: Arc<RwLock<String>>,
    /// The language to write, remembered across a disable and an unload so
    /// `Enable` knows what to reload. The wanted language, which is not always
    /// the live one: a switch keeps the old translator serving until the new one
    /// is ready, and an unload leaves this behind on purpose.
    target: Arc<RwLock<String>>,
    /// The language the **loaded** translator writes, empty when none is loaded.
    /// Kept apart from `target` because that gap is what makes a switch atomic
    /// from the reader's side: a client asking mid-switch sees the old language
    /// still being written, which is true.
    live: Arc<RwLock<String>>,
    /// The spec of the loaded translator. Reported beside `live`, and cleared
    /// with it.
    model_spec: Arc<RwLock<String>>,
    cached: Arc<RwLock<Cached>>,
    plan: Arc<TranslationPlan>,
    /// How a request becomes a translator. `Copy`, so every clone can start a load
    /// without sharing anything.
    loader: L,
    /// One load at a time: a superseded load cannot be cancelled, so this is what
    /// keeps two sets of weights from being built side by side beyond the pair
    /// already held and the one arriving.
    loading: Arc<Mutex<()>>,
    /// Who hears about a change no bus call is waiting on. Set once, after
    /// construction, and shared by every clone.
    on_change: Arc<OnceLock<ChangeHook>>,
}

/// The listener for [`TranslationState`] changes: `(enabled, target, model, ok,
/// message)`, the arguments of the bus's `TranslationChanged` signal. Injected
/// rather than called, so the dependency points one way, `dbus -> translate`.
pub(crate) type ChangeHook = Box<dyn Fn(bool, &str, &str, bool, &str) + Send + Sync>;

/// A poisoned lock must not kill transcription: take the value and continue.
fn lock_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn lock_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

impl<T: Loaded, L: Loads<T>> TranslationState<T, L> {
    /// A step that is off, with nothing loaded and nothing to load until a
    /// target is named.
    pub(crate) fn new(plan: TranslationPlan, loader: L) -> Self {
        Self {
            slot: Arc::new(Mutex::new(TranslatorSlot {
                translator: None::<T>,
                generation: 0,
            })),
            wanted: Arc::new(AtomicBool::new(false)),
            requested: Arc::new(RwLock::new(String::new())),
            target: Arc::new(RwLock::new(String::new())),
            live: Arc::new(RwLock::new(String::new())),
            model_spec: Arc::new(RwLock::new(String::new())),
            cached: Arc::new(RwLock::new(Cached::default())),
            plan: Arc::new(plan),
            loader,
            loading: Arc::new(Mutex::new(())),
            on_change: Arc::new(OnceLock::new()),
        }
    }

    /// Install the listener `announce` reports to. Only the first call takes
    /// effect; a run has one bus.
    pub(crate) fn set_change_hook(&self, hook: ChangeHook) {
        if self.on_change.set(hook).is_err() {
            log::warn!("translate: a change hook is already installed");
        }
    }

    /// The slot guard. Poison-tolerant, because a poisoned step must still emit
    /// the source text rather than stop the pipeline.
    pub(crate) fn lock_slot(&self) -> std::sync::MutexGuard<'_, TranslatorSlot<T>> {
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---- reads, all of which the bus serves without waiting on a translation --

    /// Whether the step is translating right now: asked for **and** loaded.
    pub(crate) fn is_enabled(&self) -> bool {
        lock_read(&self.cached).enabled
    }

    /// The language the live translator writes, empty when none is loaded. It
    /// survives a disable, because the translator does, and it does **not** follow
    /// a switch until the new translator lands.
    pub(crate) fn target_language(&self) -> String {
        lock_read(&self.cached).target.clone()
    }

    /// The spec of the live translator, empty when none is loaded.
    pub(crate) fn model(&self) -> String {
        lock_read(&self.cached).model.clone()
    }

    /// The language of the load in flight, empty when nothing is loading. Its
    /// being non-empty is the whole "please wait" signal, so there is no separate
    /// `Loading` property to fall out of step with it.
    pub(crate) fn requested_language(&self) -> String {
        lock_read(&self.requested).clone()
    }

    /// `(enabled, target, requested, model)` in one round trip.
    #[allow(clippy::type_complexity)]
    pub(crate) fn state(&self) -> (bool, String, String, String) {
        let cached = lock_read(&self.cached);
        (
            cached.enabled,
            cached.target.clone(),
            lock_read(&self.requested).clone(),
            cached.model.clone(),
        )
    }

    /// Whether the *next* utterance will be translated, which is what the ASR
    /// worker asks before handing a job over. Cheap on purpose: an `AtomicBool`
    /// and a lock that is never held across a decode.
    pub(crate) fn will_translate(&self) -> bool {
        self.wanted.load(Ordering::Acquire) && self.lock_slot().translator.is_some()
    }

    /// The intent flag alone. The lane reads it per job while it holds the slot,
    /// so it must not need a lock of its own.
    pub(crate) fn wanted_now(&self) -> bool {
        self.wanted.load(Ordering::Acquire)
    }

    /// Whether a translator is loaded, whatever the switch says.
    pub(crate) fn is_loaded(&self) -> bool {
        self.lock_slot().translator.is_some()
    }

    /// The panic a `catch_unwind` caught, as text.
    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_string())
    }

    // ---- writes ----

    /// Recompute what the bus reads. Idempotent, and called by every path that
    /// changes the truth, so the snapshot cannot drift from the slot.
    fn refresh_cached(&self) {
        let loaded = self.lock_slot().translator.is_some();
        let snapshot = Cached {
            enabled: loaded && self.wanted.load(Ordering::Acquire),
            target: if loaded {
                lock_read(&self.live).clone()
            } else {
                String::new()
            },
            model: if loaded {
                lock_read(&self.model_spec).clone()
            } else {
                String::new()
            },
        };
        *lock_write(&self.cached) = snapshot;
    }

    /// Turn the step on, returning `(enabled, message)`.
    ///
    /// With a translator loaded this is a flag. Without one it asks for the
    /// remembered target, which is `--translate` on a fresh run and whatever was
    /// live before an unload since. With nothing remembered it refuses, because
    /// there is no language to write and inventing one would be a lie — hence the
    /// message, which is empty on every other path.
    ///
    /// The announcement is the caller's: a bus method has a `SignalEmitter` to hand
    /// and the loader thread has not, so this only reports.
    pub(crate) fn enable(&self) -> (bool, String) {
        self.wanted.store(true, Ordering::Release);
        if self.is_loaded() {
            self.refresh_cached();
            return (true, String::new());
        }
        let remembered = lock_read(&self.target).clone();
        if remembered.is_empty() {
            self.wanted.store(false, Ordering::Release);
            self.refresh_cached();
            return (
                false,
                "no target language is set; call SetTargetLanguage first \
                 (or start with --translate LANG)"
                    .to_string(),
            );
        }
        self.start_load(&remembered);
        (self.is_enabled(), String::new())
    }

    /// Turn the step off. `unload` also frees the weights, before this returns.
    ///
    /// Either way a load in flight is cancelled: a run that was told to stop
    /// translating must not end up downloading a translator afterwards. The
    /// remembered target stays either way, so `Enable` is instant without an
    /// unload and a reload from the local cache with one.
    pub(crate) fn disable(&self, unload: bool) -> bool {
        self.cancel_pending();
        self.wanted.store(false, Ordering::Release);
        if unload {
            let mut slot = self.lock_slot();
            slot.translator = None;
            *lock_write(&self.live) = String::new();
            *lock_write(&self.model_spec) = String::new();
        }
        self.refresh_cached();
        self.is_enabled()
    }

    /// `Disable(false)` when on, `Enable` when off. Never unloads: the point of a
    /// toggle is the state, and dropping a gigabyte of weights is a decision of
    /// its own, which `Disable(true)` is for.
    pub(crate) fn toggle(&self) -> bool {
        if self.is_enabled() {
            self.disable(false)
        } else {
            self.enable().0
        }
    }

    /// Write `lang` from now on, loading whatever that needs.
    ///
    /// Returns `(accepted, target, message)`. A direction the catalog cannot
    /// serve is refused here, before any thread starts: the caller gets the same
    /// message the startup path prints, immediately. An accepted target is not
    /// yet a loaded one — the previous translator keeps serving until the new one
    /// is ready, and `TranslationChanged` says when the swap happened.
    ///
    /// No announcement here: the caller holds the bus emitter. What this returns is
    /// the whole report.
    #[allow(clippy::type_complexity)]
    pub(crate) fn set_target_language(&self, lang: &str) -> (bool, String, String) {
        let target = lang.trim().to_ascii_lowercase();
        if target.is_empty() {
            let message = "empty target language".to_string();
            return (false, self.target_language(), message);
        }
        let direction = self.plan.direction(&target);
        if let Err(e) =
            crate::translator_store::reference_for(direction, self.plan.model.as_deref())
        {
            let message = format!("{e:#}");
            return (false, self.target_language(), message);
        }
        // The same language, already loaded: this is a switch on, not a load.
        if lock_read(&self.live).eq(target.as_str()) && self.is_loaded() {
            self.wanted.store(true, Ordering::Release);
            self.refresh_cached();
            return (true, target, String::new());
        }
        *lock_write(&self.target) = target.clone();
        self.wanted.store(true, Ordering::Release);
        self.refresh_cached();
        self.start_load(&target);
        (true, target, String::new())
    }

    // ---- the load pipeline ----

    /// Load the language `--translate` named, on the calling thread, failing the
    /// run when it cannot: startup is where a missing model should stop everything
    /// rather than wait for the first utterance.
    ///
    /// Naming the language at startup is also the intent to use it, so the step is
    /// wanted from the first utterance: the worker asks `will_translate` before
    /// every hand-off, and a translator loaded but not wanted is never used.
    pub(crate) fn load_startup(&self) -> Result<()> {
        let Some(target) = self.plan.configured.clone() else {
            return Ok(());
        };
        *lock_write(&self.target) = target.clone();
        self.wanted.store(true, Ordering::Release);
        let generation = self.begin_request(&target);
        let translator = self.loader.load(&self.plan, &target)?;
        self.install(generation, &target, translator);
        Ok(())
    }

    /// Register a request and hand back the generation that will name it.
    fn begin_request(&self, target: &str) -> u64 {
        let generation = {
            let mut slot = self.lock_slot();
            slot.generation = slot.generation.wrapping_add(1);
            slot.generation
        };
        *lock_write(&self.requested) = target.to_string();
        generation
    }

    /// Abandon whatever was loading. The generation bump is what makes an arriving
    /// load stale rather than installed; clearing the request is what stops a
    /// client from waiting out a load that will now do nothing.
    fn cancel_pending(&self) {
        {
            let mut slot = self.lock_slot();
            slot.generation = slot.generation.wrapping_add(1);
        }
        *lock_write(&self.requested) = String::new();
    }

    /// Put the translator in place, if this request is still the current one.
    ///
    /// The generation is compared and the translator written under one lock, so
    /// a `Disable` cannot slip between the two and be followed by weights it
    /// promised to free.
    fn install(&self, generation: u64, target: &str, translator: T) -> bool {
        let mut slot = self.lock_slot();
        if slot.generation != generation {
            log::debug!("translate: the {target} load was superseded before it landed; dropped");
            return false;
        }
        let spec = translator.spec().to_string();
        slot.translator = Some(translator);
        slot.generation = generation;
        // Still under the slot, so a client cannot read a loaded translator whose
        // language and spec have not been written yet.
        *lock_write(&self.live) = target.to_string();
        *lock_write(&self.model_spec) = spec.clone();
        *lock_write(&self.requested) = String::new();
        drop(slot);
        self.refresh_cached();
        log::info!("translate: writing {target} (translator {spec})");
        self.announce(true, "");
        true
    }

    /// Report a load that failed. The previous translator keeps serving: a
    /// refused switch must not cost the language the run already had.
    fn fail_request(&self, generation: u64, message: &str) {
        {
            let slot = self.lock_slot();
            if slot.generation != generation {
                log::debug!("translate: a superseded load failed; its error is stale");
                return;
            }
            if slot.translator.is_none() {
                // Nothing to fall back on, so the step is off rather than on with
                // no translator: `Enabled` must never claim work that cannot
                // happen.
                self.wanted.store(false, Ordering::Release);
            }
        }
        *lock_write(&self.requested) = String::new();
        self.refresh_cached();
        self.announce(false, message);
    }

    /// Load `target` on a helper thread, through `load`.
    ///
    /// Off the bus thread on purpose: a first use downloads 114 MiB to 1.3 GiB and
    /// then builds three `ort` sessions. `Pause` and the dictionary have to keep
    /// working while that happens, and the blocking zbus API serves one thread.
    ///
    /// `load` is a parameter rather than a direct [`load_for`] call so a test can
    /// drive the whole pipeline — request, serialize, install or fail, announce —
    /// without weights and without a network.
    fn start_load(&self, target: &str) {
        let generation = self.begin_request(target);
        let state = self.clone();
        let target = target.to_string();
        let spawned = std::thread::Builder::new()
            .name("translate-load".into())
            .spawn(move || {
                // Held across the load, so a superseded request waits its turn
                // instead of building a second set of weights beside the first.
                let _serialized = state.loading.lock().unwrap_or_else(|e| e.into_inner());
                match state.loader.load(&state.plan, &target) {
                    Ok(translator) => {
                        state.install(generation, &target, translator);
                    }
                    Err(e) => {
                        let message = format!("{e:#}");
                        log::warn!("translate: cannot write {target}: {message}");
                        state.fail_request(generation, &message);
                    }
                }
            });
        if let Err(e) = spawned {
            let message = format!("failed to start the translator load: {e}");
            log::error!("translate: {message}");
            self.fail_request(generation, &message);
        }
    }

    /// A load that was superseded before it landed, reported as a debug line
    /// rather than a failure: it did its work, and a newer request wants the slot.
    #[cfg(test)]
    pub(crate) fn report_stale_install(&self, generation: u64, target: &str) {
        if self.lock_slot().generation != generation {
            log::debug!("translate: the {target} load was superseded; dropped");
        }
    }

    /// Report a change from outside an interface method — the loader thread,
    /// and every refusal, which have no emitter to hand — through the hook
    /// [`Self::set_change_hook`] installed. Silent when none is installed.
    fn announce(&self, ok: bool, message: &str) {
        let Some(hook) = self.on_change.get() else {
            return;
        };
        let cached = lock_read(&self.cached);
        hook(cached.enabled, &cached.target, &cached.model, ok, message);
    }

    /// The translation thread's half: run one decode under the slot lock, so a
    /// disable cannot free a translator mid-decode, and report a panicking decode
    /// as a failed translation instead of a dead thread.
    ///
    /// A panic drops the translator rather than keeping it: the sessions take
    /// `&mut self` and their state after a panic is not something to keep
    /// decoding with. The step then reports itself off, the ASR worker goes back
    /// to emitting source text, and `Enable` loads a fresh one.
    pub(crate) fn translate_locked(&self, slot: &mut TranslatorSlot<T>, source: &str) -> Attempt {
        let Some(translator) = slot.translator.as_mut() else {
            return Attempt::Failed("no translator is loaded".to_string());
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            translator.translate(source)
        })) {
            Ok(Ok(text)) => Attempt::Translated(text),
            Ok(Err(e)) => Attempt::Failed(format!("{e:#}")),
            Err(payload) => Attempt::Panicked(Self::panic_message(payload)),
        }
    }

    /// Free the weights of a translator that panicked, taking the slot itself.
    ///
    /// Not a `&mut TranslatorSlot` argument: the caller holds that lock while it
    /// decodes, and a lock that tries to take itself again deadlocks the thread
    /// that has to be rescued. So the lane releases it first and calls this.
    pub(crate) fn drop_panicked_translator(&self) {
        {
            let mut slot = self.lock_slot();
            slot.translator = None;
            *lock_write(&self.live) = String::new();
            *lock_write(&self.model_spec) = String::new();
        }
        self.refresh_cached();
        self.announce(
            false,
            "the translator panicked and was dropped; call Enable to load it again",
        );
    }
}

/// One accepted transcript on its way to the translation thread.
///
/// The `samples` and `infer_ms` travel with the job because the metrics row is
/// written *after* translation: the `text` column has to hold what was actually
/// emitted, and after this step that is the translation, not the source.
pub(crate) struct TranslationJob {
    pub(crate) tag: String,
    pub(crate) pcm: Vec<f32>,
    /// The transcript as the gates left it, before translation.
    pub(crate) source: String,
    pub(crate) verdict: crate::gate::TranscriptVerdict,
    pub(crate) evidence: crate::utterance::Evidence,
    pub(crate) samples: usize,
    pub(crate) infer_ms: f64,
    /// Which translator was live when this job was queued, so a language switch
    /// mid-queue is visible in the log rather than a silent change of output
    /// language. Filled by [`TranslationOut::enqueue`], which is the only
    /// producer, so it cannot be forgotten at a call site.
    pub(crate) generation: u64,
}

/// The clone is what lets the bus thread, the ASR worker, the translation thread
/// and every loader hold one state; `T` sits behind an `Arc` and is never copied.
impl<T, L: Clone> Clone for TranslationState<T, L> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
            wanted: Arc::clone(&self.wanted),
            requested: Arc::clone(&self.requested),
            target: Arc::clone(&self.target),
            live: Arc::clone(&self.live),
            model_spec: Arc::clone(&self.model_spec),
            cached: Arc::clone(&self.cached),
            plan: Arc::clone(&self.plan),
            loader: self.loader.clone(),
            loading: Arc::clone(&self.loading),
            on_change: Arc::clone(&self.on_change),
        }
    }
}

/// Bounded depth of the translation queue.
///
/// Emissions are seconds apart and a translation takes 80 to 700 ms, so this never
/// fills in practice. It is here so that a stalled or slow translator costs queued
/// work rather than the inference thread's time.
pub(crate) const TRANSLATION_QUEUE_DEPTH: usize = 4;

/// The translated emission: hand it a transcript, a dedicated thread translates
/// and emits it.
///
/// Off the ASR worker on purpose. Translation is 60 to 130 ms of CPU per
/// utterance against the worker's 50 ms of inference, and inline it would add
/// that to every gap between utterances.
///
/// The queue is a **bounded FIFO**, oldest dropped first. Not newest-wins like
/// `SpeechOut`: that rule exists because stale *audio* puts words in the user's
/// mouth, and a backlog of speech is worthless by the time it would be read. A
/// backlog of translations is not stale — it is work in order, and the user is
/// waiting for all of it. Dropping the oldest is the honest choice when the
/// translator cannot keep up: the newest transcript is the one the user just
/// said and still cares about.
///
/// The thread exists from startup in microphone mode even with nothing loaded,
/// because the switch that turns the step on arrives on the bus thread and the
/// ASR worker cannot be handed a lane after it started. It is fed **only** when a
/// translation will actually happen (see [`TranslationState::will_translate`]),
/// so a run that never translates keeps today's path: the worker emits inline and
/// this queue never fills.
pub(crate) struct TranslationOut<T = Translator, L = RealTranslator> {
    /// `(queue, wakeup)`: the pending jobs in order, and the signal to look.
    queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)>,
    /// Set by `drop`, so the thread finishes the queue and exits.
    closed: Arc<AtomicBool>,
    state: TranslationState<T, L>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl<T: Loaded, L: Loads<T>> TranslationOut<T, L> {
    /// Spawn the translation thread around the shared state, an emitter handle
    /// onto the shared emission queue, and the `--metrics` writer.
    ///
    /// The translator is not an argument: it arrives in the state's slot, loaded
    /// at startup by `--translate` or later over the bus.
    pub(crate) fn spawn(
        state: TranslationState<T, L>,
        mut emitter: Box<dyn Emitter + Send>,
        metrics: crate::asr::MetricsSlot,
    ) -> Self {
        let queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let closed = Arc::new(AtomicBool::new(false));

        let handle = {
            let (queue, closed) = (Arc::clone(&queue), Arc::clone(&closed));
            let thread_state = state.clone();
            std::thread::Builder::new()
                .name("translate".into())
                .spawn(move || {
                    translate_loop(&thread_state, &queue, &closed, emitter.as_mut(), &metrics)
                })
                .ok()
        };
        Self {
            queue,
            closed,
            state,
            handle,
        }
    }

    /// Whether the next job would be translated, which is what the ASR worker asks
    /// before handing one over: false means the utterance stays on the worker's
    /// inline path, so a run with the step off never enters this queue.
    pub(crate) fn will_translate(&self) -> bool {
        self.state.will_translate()
    }

    /// Queue one transcript for translation. Never blocks and never fails: a
    /// dead thread or a full queue costs one utterance, never the transcript.
    ///
    /// The job is stamped with the translator that is live now, so a language
    /// switch while it waits is visible in the log rather than a silent change of
    /// output language.
    ///
    /// A full queue drops the **oldest** waiting job and warns, naming the tag,
    /// so the log says which transcript was lost rather than only that one was.
    pub(crate) fn enqueue(&self, mut job: TranslationJob) {
        job.generation = self.state.lock_slot().generation;
        let dropped = {
            let (lock, wakeup) = &*self.queue;
            let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut dropped = None;
            while pending.len() >= TRANSLATION_QUEUE_DEPTH {
                match pending.pop_front() {
                    Some(old) => dropped = Some(old.tag),
                    // The queue is non-empty whenever `len >= depth >= 1`.
                    None => break,
                }
            }
            pending.push_back(job);
            // Notified with the lock held, which is what lets the thread wait
            // without a timeout: a notification sent after the waiter has checked
            // the queue and before it sleeps is the one a bare `notify_all` loses.
            wakeup.notify_all();
            dropped
        };
        if let Some(tag) = dropped {
            log::warn!(
                "translation queue full ({TRANSLATION_QUEUE_DEPTH} deep): dropped the \
                 oldest waiting transcript {tag} to make room"
            );
        }
    }
}

impl<T, L> Drop for TranslationOut<T, L> {
    fn drop(&mut self) {
        // Close the queue, then join: the thread finishes what is waiting and
        // exits, and the join is what waits for that. No more jobs can arrive,
        // since the only producer is the ASR worker holding this value.
        self.closed.store(true, Ordering::Release);
        {
            let (lock, wakeup) = &*self.queue;
            // The lock is taken and released only to be sure the flag is visible
            // before the wakeup, which is what `enqueue` does too: a notification
            // sent between the thread's check of the queue and its sleep is the one
            // a bare `notify_all` loses, and holding the lock across both is what
            // rules that out.
            drop(lock.lock().unwrap_or_else(|e| e.into_inner()));
            wakeup.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                log::warn!("translation thread panicked");
            }
        }
    }
}

/// One thread's whole job: take the oldest pending transcript, translate it,
/// write its metrics row, emit it, repeat until closed and empty.
///
/// The trailing utterance is translated, not dropped: `--file --translate en` with
/// the last transcript swallowed would be the channel silently losing work, and it
/// is the same reasoning the ASR shutdown flush and `SpeechOut::drop` use. So
/// `closed` ends the loop only once the queue is empty.
///
/// The wait is therefore unbounded on purpose, and what bounds the shutdown is
/// the work itself: a piece's decode stops at
/// [`translator_store::Family::max_new_tokens`] tokens, so the drain is bounded by
/// the queue depth and the transcript's length. A wall-clock deadline would have
/// to interrupt a translation already in flight to be any tighter, which is not
/// something a flag can do.
fn translate_loop<T: Loaded, L: Loads<T>>(
    state: &TranslationState<T, L>,
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
    emitter: &mut dyn Emitter,
    metrics: &crate::asr::MetricsSlot,
) {
    while let Some(job) = next_job(queue, closed) {
        emit_one(state, emitter, metrics, job);
    }
}

/// The next job to translate, or `None` once the queue is closed and empty.
fn next_job(
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
) -> Option<TranslationJob> {
    let (lock, wakeup) = queue;
    let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(job) = pending.pop_front() {
            return Some(job);
        }
        if closed.load(Ordering::Acquire) {
            return None;
        }
        // No timeout: both notifiers send under the lock, so a wakeup cannot be
        // lost between the check above and this sleep.
        pending = wakeup.wait(pending).unwrap_or_else(|e| e.into_inner());
    }
}

/// Translate one job, log what it cost, write its metrics row and emit it.
///
/// The metrics row is written here rather than by the ASR worker because its
/// `text` column has to be what the emitter received. A dropped queue entry never
/// reaches this function, so it never reaches the metrics log either — which is
/// why the queue drop warns with the tag.
///
/// One lock is held across the decode, which is what lets `Disable(true)` promise
/// the weights are gone when it returns: it waits for a translation already
/// running instead of pulling the translator out from under one. Every other
/// utterance still takes the inline path on the worker, so nothing waits behind
/// this lock except the switch that asked for it.
fn emit_one<T: Loaded, L: Loads<T>>(
    state: &TranslationState<T, L>,
    emitter: &mut dyn Emitter,
    metrics: &crate::asr::MetricsSlot,
    job: TranslationJob,
) {
    let mut panicked = false;
    let attempt = {
        let mut slot = state.lock_slot();
        if job.generation != slot.generation {
            log::info!(
                "{}: the target language changed while this waited; translating with the \
                 live one",
                job.tag
            );
        }
        let t0 = Instant::now();
        let attempt = if state.wanted_now() {
            state.translate_locked(&mut slot, &job.source)
        } else {
            // Disabled between the enqueue and here. The source text is what the
            // step would have emitted anyway, and saying so is the failure policy
            // rather than losing the utterance.
            Attempt::Failed("the translation step was disabled".to_string())
        };
        let took = t0.elapsed().as_secs_f64() * 1000.0;
        let direction = slot
            .translator
            .as_ref()
            .map(|t| t.direction().to_string())
            .unwrap_or_else(|| "no translator".to_string());
        if let Attempt::Panicked(what) = &attempt {
            log::error!("{}: the decoder panicked: {what}", job.tag);
            panicked = true;
        }
        (attempt, direction, took)
    };
    let (attempt, direction, translate_ms) = attempt;
    // After the guard is released: dropping the weights takes the same lock.
    if panicked {
        state.drop_panicked_translator();
    }
    let ran = matches!(attempt, Attempt::Translated(_));
    let text = fallback_or_translated(&job.source, attempt.into_result(), &job.tag);
    log::info!(
        "{}: translate ({direction}) took {translate_ms:.0} ms ({:?} -> {text:?})",
        job.tag,
        job.source
    );
    // `src_text` only when a translation ran for *this* utterance, so a log from a
    // run with the step off stays comparable with one from before it existed.
    let mut writer = metrics.lock();
    crate::asr::write_metrics_row(
        &mut writer,
        &job.tag,
        job.samples,
        &job.evidence,
        job.verdict,
        &text,
        job.infer_ms,
        ran.then_some(job.source.as_str()),
    );
    // The writer is released before the emission: the emit queue has its own lock,
    // and nothing should hold two at once.
    drop(writer);
    emitter.emit(Emission {
        tag: &job.tag,
        pcm: &job.pcm,
        text: &text,
        verdict: job.verdict,
        evidence: &job.evidence,
    });
}

/// What one decode attempt produced.
///
/// `Panicked` is separate from `Failed` because the two need different
/// afterwards: a failed decode leaves a translator worth keeping, while a panicked
/// one leaves sessions that took `&mut self` and are not worth decoding with again.
pub(crate) enum Attempt {
    Translated(String),
    Failed(String),
    Panicked(String),
}

impl Attempt {
    /// The shape [`fallback_or_translated`] takes: every failure is the same to it,
    /// because every failure emits the source text.
    fn into_result(self) -> Result<String> {
        match self {
            Self::Translated(text) => Ok(text),
            Self::Failed(what) | Self::Panicked(what) => Err(anyhow::anyhow!(what)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::MetricsSlot;
    use crate::emit::{Emission, Emitter};
    use crate::gate::TranscriptVerdict;
    use anyhow::anyhow;
    use std::time::Duration;

    // -- the queue, with a stand-in translator ------------------------------
    //
    // `TranslationOut` takes a `Box<dyn Emitter>`, so a lane can be driven with the
    // same recording emitter the `asr` tests use, and with a translator that needs
    // no weights. What is under test is the queue contract and the two policies
    // that hang off it: the source text is what an untranslated or failed utterance
    // emits, and the drain at shutdown keeps the trailing utterance.

    /// What a lane test reads back: `(tag, text)` per emission, in order.
    type Sink = Arc<Mutex<Vec<(String, String)>>>;

    /// An emitter that records into a shared sink, because the lane owns its
    /// emitter as a `'static` box (production hands it to a thread) and a test still
    /// has to read the result afterwards.
    struct Recorder(Sink);

    impl Emitter for Recorder {
        fn emit(&mut self, emission: Emission<'_>) {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((emission.tag.to_string(), emission.text.to_string()));
        }
    }

    /// A lane over `state`, recording into a fresh sink.
    fn recording_lane<T: Loaded, L: Loads<T>>(
        state: TranslationState<T, L>,
    ) -> (TranslationOut<T, L>, Sink) {
        let sink: Sink = Arc::new(Mutex::new(Vec::new()));
        let lane = TranslationOut::spawn(
            state,
            Box::new(Recorder(Arc::clone(&sink))),
            MetricsSlot::default(),
        );
        (lane, sink)
    }

    /// The `(tag, text)` pairs a sink collected, once the lane has been dropped.
    fn recorded(sink: &Sink) -> Vec<(String, String)> {
        sink.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn job(tag: &str, source: &str) -> TranslationJob {
        TranslationJob {
            tag: tag.to_string(),
            pcm: vec![0.0; 160],
            source: source.to_string(),
            verdict: TranscriptVerdict::Emit,
            evidence: crate::utterance::Evidence {
                speech_frames: 1,
                total_frames: 2,
                rms: 0.01,
            },
            samples: 16_000,
            infer_ms: 12.0,
            generation: 0,
        }
    }

    #[test]
    fn the_lane_translates_a_job_through_the_shared_state() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane); // the drain is what waits for it
        assert_eq!(
            recorded(&sink),
            vec![(
                "utt_000001".to_string(),
                "translated|Dzień dobry".to_string()
            )],
            "the emitter receives the translation, under the job's own tag"
        );
    }

    #[test]
    fn a_disabled_step_emits_the_source_text_rather_than_losing_it() {
        // The job was queued while the step was on and disabled before the lane
        // reached it. Losing the utterance would be the failure; the source text is
        // what the step emits anyway.
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s.clone());
        s.disable(false);
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane);
        assert_eq!(
            recorded(&sink),
            vec![("utt_000001".to_string(), "Dzień dobry".to_string())],
            "the utterance must still arrive, untranslated"
        );
    }

    #[test]
    fn the_drain_keeps_the_trailing_utterance() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        // Fewer than the queue depth, so the drain is what is under test and the
        // drop-oldest policy is not.
        for i in 1..=TRANSLATION_QUEUE_DEPTH - 1 {
            lane.enqueue(job(&format!("utt_{i:06}"), "tekst"));
        }
        drop(lane);
        let tags: Vec<String> = recorded(&sink).into_iter().map(|(tag, _)| tag).collect();
        assert_eq!(
            tags,
            vec!["utt_000001", "utt_000002", "utt_000003"],
            "shutdown drains the queue in order rather than dropping the tail"
        );
    }

    #[test]
    fn a_full_queue_drops_the_oldest_and_keeps_the_newest() {
        // The documented policy for a translator that cannot keep up: the newest
        // transcript is the one the user just said.
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        for i in 1..=TRANSLATION_QUEUE_DEPTH + 2 {
            lane.enqueue(job(&format!("utt_{i:06}"), "tekst"));
        }
        drop(lane);
        let tags: Vec<String> = recorded(&sink).into_iter().map(|(tag, _)| tag).collect();
        assert!(
            tags.len() < TRANSLATION_QUEUE_DEPTH + 3,
            "a full queue sheds rather than growing without bound: {tags:?}"
        );
        assert_eq!(
            tags.last().map(String::as_str),
            Some(format!("utt_{:06}", TRANSLATION_QUEUE_DEPTH + 2).as_str()),
            "the newest survives"
        );
    }

    #[test]
    fn a_panicking_decode_emits_the_source_text_and_drops_the_translator() {
        // A dead lane would lose every later utterance, because the worker had
        // already handed them over. So a panic becomes a failed translation: the
        // source text is emitted, and the translator is dropped so `Enable` loads
        // a fresh one.
        struct Panicking;
        impl Loaded for Panicking {
            fn translate(&mut self, _source: &str) -> Result<String> {
                panic!("the decoder exploded");
            }
            fn spec(&self) -> &str {
                "panicking"
            }
            fn direction(&self) -> &str {
                "pl -> en"
            }
        }
        #[derive(Clone)]
        struct PanicLoad;
        impl Loads<Panicking> for PanicLoad {
            fn load(&self, _plan: &TranslationPlan, _target: &str) -> Result<Panicking> {
                Ok(Panicking)
            }
        }
        let s: TranslationState<Panicking, PanicLoad> = TranslationState::new(
            TranslationPlan::for_tests_with_model("pl", "nllb-200-distilled-600M"),
            PanicLoad,
        );
        let mut slot = s.lock_slot();
        slot.generation += 1;
        let generation = slot.generation;
        slot.translator = Some(Panicking);
        drop(slot);
        *lock_write(&s.live) = "en".to_string();
        *lock_write(&s.model_spec) = "panicking".to_string();
        s.wanted.store(true, Ordering::Release);
        assert!(s.install(generation, "en", Panicking));

        let (lane, sink) = recording_lane(s.clone());
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane);
        assert_eq!(
            recorded(&sink),
            vec![("utt_000001".to_string(), "Dzień dobry".to_string())],
            "a panicking decode must not cost the utterance"
        );
        assert!(!s.is_loaded(), "and the translator is dropped");
        assert!(
            !s.will_translate(),
            "so the worker goes back to the inline path"
        );
    }

    // -- runtime state ------------------------------------------------------
    //
    // No weights anywhere below: a `Translator` is two or three `ort` sessions, so
    // a unit test cannot build one. What is under test is the bookkeeping around
    // the load — refusals, the generation that cancels one, and which fields each
    // switch clears — none of which needs a model. Every path that would fetch one
    // is asserted to refuse *before* it starts, so no test here touches the
    // network.

    /// A translator that needs no weights: it records what it was asked and returns
    /// a fixed string. Enough for the state machine, since none of it touches an
    /// `ort` session.
    #[derive(Clone)]
    struct FakeTranslator {
        spec: String,
        direction: String,
        answer: String,
    }

    impl Loaded for FakeTranslator {
        fn translate(&mut self, source: &str) -> Result<String> {
            Ok(format!("{}|{source}", self.answer))
        }
        fn spec(&self) -> &str {
            &self.spec
        }
        fn direction(&self) -> &str {
            &self.direction
        }
    }

    fn fake(spec: &str) -> FakeTranslator {
        FakeTranslator {
            spec: spec.to_string(),
            direction: "pl -> en".to_string(),
            answer: "translated".to_string(),
        }
    }

    /// A load that blocks until the test releases it, so "waiting for a load" is a
    /// state a test can observe rather than a race. `None` means load at once.
    type Gate = Option<Arc<Gate2>>;

    /// An open/closed flag with a wakeup: a load waits on it, a test opens it.
    struct Gate2 {
        open: Mutex<bool>,
        wakeup: Condvar,
    }

    #[derive(Clone)]
    struct FakeLoad {
        gate: Gate,
    }

    impl FakeLoad {
        /// Wait for the test to open the gate, if there is one. The load runs on
        /// its own thread, so this is the seam that makes "a load in flight" a state
        /// a test can observe instead of a race.
        fn wait(&self) {
            let Some(gate) = &self.gate else {
                return;
            };
            let mut open = gate.open.lock().unwrap_or_else(|e| e.into_inner());
            while !*open {
                open = gate.wakeup.wait(open).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Release every load blocked on this gate.
    fn release(gate: &Gate) {
        let Some(gate) = gate else {
            return;
        };
        *gate.open.lock().unwrap_or_else(|e| e.into_inner()) = true;
        gate.wakeup.notify_all();
    }

    impl Loads<FakeTranslator> for FakeLoad {
        fn load(&self, plan: &TranslationPlan, target: &str) -> Result<FakeTranslator> {
            self.wait();
            if target == "unobtainable" {
                anyhow::bail!("no translator for {} -> {target}", plan.source_label());
            }
            Ok(FakeTranslator {
                spec: format!("opus-mt-{}-{target}", plan.source_label()),
                direction: format!("{} -> {target}", plan.source_label()),
                answer: "translated".to_string(),
            })
        }
    }

    /// The state these tests drive: a stand-in translator behind the real
    /// bookkeeping.
    type TestState = TranslationState<FakeTranslator, FakeLoad>;

    /// A state over a Polish source, so `en` resolves and `de` does not, with loads
    /// that land at once.
    fn state(source: &str) -> TestState {
        gated_state(source, None)
    }

    /// The same, with every load blocked until [`release`] is called: what a client
    /// sees while a first-time download runs.
    fn gated_state(source: &str, gate: Gate) -> TestState {
        TranslationState::new(TranslationPlan::for_tests(source), FakeLoad { gate })
    }

    /// A state that accepts any target, because the load half is under test rather
    /// than the catalog: the translator is named outright, which is the one path
    /// `reference_for` resolves without consulting the pair table.
    fn any_target_state(gate: Gate) -> TestState {
        TranslationState::new(
            TranslationPlan::for_tests_with_model("pl", "nllb-200-distilled-600M"),
            FakeLoad { gate },
        )
    }

    /// A fresh gate to hand to [`gated_state`].
    fn new_gate() -> Gate {
        Some(Arc::new(Gate2 {
            open: Mutex::new(false),
            wakeup: Condvar::new(),
        }))
    }

    /// Wait until `what` reads `expected`, so a test observes the state instead of
    /// racing the loader thread.
    fn wait_until<T: PartialEq + std::fmt::Debug>(
        what: &str,
        mut read: impl FnMut() -> T,
        expected: T,
    ) {
        for _ in 0..400 {
            if read() == expected {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("{what} never became {expected:?}");
    }

    /// Put a translator in the slot the way a successful load would, so every
    /// read below turns on a real occupant without any weights.
    fn with_fake_translator(s: &TestState, target: &str, spec: &str) {
        let mut slot = s.lock_slot();
        slot.generation += 1;
        let generation = slot.generation;
        slot.translator = Some(fake(spec));
        drop(slot);
        *lock_write(&s.live) = target.to_string();
        *lock_write(&s.model_spec) = spec.to_string();
        *lock_write(&s.target) = target.to_string();
        s.wanted.store(true, Ordering::Release);
        assert!(
            s.install(generation, target, fake(spec)),
            "a current generation must install"
        );
    }

    #[test]
    fn the_startup_load_is_wanted_from_the_first_utterance() {
        // `--translate en` at startup: the translator is loaded *and* the step is
        // wanted, because the worker asks `will_translate` before every hand-off. A
        // translator loaded but not wanted would silently never run.
        let s: TestState = TranslationState::new(
            TranslationPlan::for_tests_configured("pl", "en"),
            FakeLoad { gate: None },
        );
        s.load_startup().expect("the startup load must succeed");
        assert!(s.is_enabled(), "a run that asked for translation is on");
        assert!(
            s.will_translate(),
            "and the worker must hand utterances over from the first one"
        );
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.requested_language(), "", "nothing is loading any more");
    }

    #[test]
    fn a_startup_load_that_cannot_be_produced_fails_the_run() {
        // Startup is the one place a missing translator is fatal, rather than a warn
        // on a later utterance.
        let s: TestState = TranslationState::new(
            TranslationPlan::for_tests_configured("pl", "unobtainable"),
            FakeLoad { gate: None },
        );
        assert!(
            s.load_startup().is_err(),
            "the caller must see the failure and stop"
        );
        assert!(!s.is_loaded(), "and nothing half-loaded is left behind");
    }

    #[test]
    fn no_translate_flag_loads_nothing_at_all() {
        let s = state("pl");
        s.load_startup().expect("no flag is not a failure");
        assert!(
            !s.is_loaded(),
            "nothing is downloaded for a run that did not ask"
        );
        assert!(!s.is_enabled());
        assert_eq!(s.target_language(), "");
    }

    #[test]
    fn a_fresh_state_is_off_with_nothing_loaded() {
        let s = state("pl");
        assert!(!s.is_enabled(), "nothing asked and nothing loaded");
        assert!(!s.is_loaded());
        assert_eq!(s.target_language(), "");
        assert_eq!(s.model(), "");
        assert_eq!(
            s.requested_language(),
            "",
            "an empty request is what says nothing is loading"
        );
        assert!(
            !s.will_translate(),
            "so nothing reaches the translation thread"
        );
    }

    #[test]
    fn an_empty_target_language_is_refused_without_a_load() {
        let s = state("pl");
        let (accepted, target, message) = s.set_target_language("   ");
        assert!(!accepted);
        assert_eq!(message, "empty target language");
        assert_eq!(target, "");
        assert_eq!(
            s.requested_language(),
            "",
            "a refusal must not register a request, or a loader would start"
        );
    }

    #[test]
    fn a_direction_the_catalog_cannot_serve_is_refused_with_the_startup_message() {
        let s = state("pl");
        let (accepted, target, message) = s.set_target_language("de");
        assert!(!accepted, "Polish has no exported pair into German");
        assert!(
            message.contains("no translator for pl -> de"),
            "the caller gets the message the startup path prints: {message}"
        );
        // It names what Polish *can* reach, which is the useful half.
        assert!(message.contains("en"), "{message}");
        assert_eq!(target, "", "the live target is unchanged by a refusal");
        assert_eq!(s.requested_language(), "");
    }

    #[test]
    fn an_unknown_source_language_is_refused_too() {
        let s = state("qq");
        let (accepted, _, message) = s.set_target_language("en");
        assert!(!accepted);
        assert!(message.contains("qq"), "{message}");
    }

    #[test]
    fn a_refusal_leaves_a_loaded_translator_serving() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.is_enabled(), "the fake load enables the step");
        let (accepted, target, _) = s.set_target_language("de");
        assert!(!accepted);
        assert_eq!(target, "en", "a refused switch costs the language it had");
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert!(s.will_translate());
    }

    #[test]
    fn enable_with_no_translator_and_no_target_refuses() {
        let s = state("pl");
        let (enabled, message) = s.enable();
        assert!(!enabled);
        assert!(
            message.contains("no target language is set"),
            "the caller is told why nothing happened: {message}"
        );
        assert!(!s.will_translate());
        assert_eq!(s.requested_language(), "", "and no load was started");
    }

    #[test]
    fn enable_after_an_unload_reports_off_until_the_load_lands() {
        // `Enabled` is false while the reload runs: it claims work that is not
        // happening yet, and the client waits on `RequestedLanguage` instead.
        let gate = new_gate();
        let s = gated_state("pl", gate.clone());
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(true);

        let (enabled, message) = s.enable();
        assert!(!enabled, "the reload has not finished");
        assert!(message.is_empty(), "this is not a refusal, only a load");
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "en".to_string(),
        );
        assert!(!s.will_translate(), "still nothing to translate with");

        release(&gate);
        wait_until("Enabled", || s.is_enabled(), true);
        assert_eq!(
            s.model(),
            "opus-mt-pl-en",
            "a fresh translator, reported by spec"
        );
        assert_eq!(s.requested_language(), "", "and the request is cleared");
    }

    #[test]
    fn a_first_switch_from_nothing_loads_before_it_reports_enabled() {
        // The one case with no translator to fall back on, so this is the window in
        // which utterances come out untranslated.
        let gate = new_gate();
        let s = any_target_state(gate.clone());
        let (accepted, target, message) = s.set_target_language("en");
        assert!(accepted);
        assert_eq!(target, "en");
        assert!(message.is_empty());
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "en".to_string(),
        );
        assert!(!s.is_enabled(), "not yet: nothing is loaded");
        assert_eq!(s.target_language(), "", "so no language is claimed");
        assert!(!s.will_translate(), "and the worker stays inline");

        release(&gate);
        wait_until("Enabled", || s.is_enabled(), true);
        assert_eq!(s.target_language(), "en");
    }

    #[test]
    fn a_switch_keeps_the_old_translator_serving_until_the_new_one_lands() {
        // The property that makes a switch atomic from the reader's side.
        let gate = new_gate();
        let s = any_target_state(gate.clone());
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.is_enabled());

        assert!(s.set_target_language("de").0);
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "de".to_string(),
        );
        assert_eq!(
            s.target_language(),
            "en",
            "the old translator still serves, and that is what is reported"
        );
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert!(s.will_translate(), "so nothing comes out untranslated");
    }

    #[test]
    fn a_load_that_cannot_be_produced_leaves_the_step_off() {
        let s = any_target_state(None);
        assert!(s.set_target_language("unobtainable").0);
        // The load runs on its own thread and fails there.
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "".to_string(),
        );
        assert!(
            !s.is_enabled(),
            "`Enabled` never claims work that cannot happen"
        );
        assert!(!s.is_loaded());
        assert!(!s.will_translate());
    }

    #[test]
    fn a_load_that_cannot_be_produced_keeps_the_language_the_run_had() {
        let s = any_target_state(None);
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(s.set_target_language("unobtainable").0);
        wait_until(
            "RequestedLanguage",
            || s.requested_language(),
            "".to_string(),
        );
        assert!(
            s.is_enabled(),
            "the old translator is untouched by a failed switch"
        );
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.model(), "opus-mt-pl-en");
    }

    #[test]
    fn a_disable_cancels_the_load_in_flight() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let generation = s.lock_slot().generation;
        // Stand in for a request: the generation the loader would carry.
        let stale = s.begin_request("en");
        assert!(stale > generation);

        s.disable(false);
        // The load is stale now, so its result cannot install.
        assert_ne!(
            s.lock_slot().generation,
            stale,
            "a disable must bump the generation, or an arriving load would still land"
        );
        assert_eq!(s.requested_language(), "", "and clear the request");
    }

    #[test]
    fn a_superseded_load_never_installs() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let stale = s.begin_request("de");
        s.disable(false);
        // `install` with a stale generation returns without touching the slot.
        let before = s.model();
        s.report_stale_install(stale, "de");
        assert_eq!(s.model(), before, "the live translator is untouched");
        assert_eq!(s.target_language(), "en");
    }

    #[test]
    fn disable_without_unload_keeps_the_weights_for_an_instant_enable() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.disable(false), "off, and it returns the new state");
        assert!(!s.is_enabled(), "off");
        assert!(s.is_loaded(), "but still loaded, so Enable costs nothing");
        assert_eq!(
            s.target_language(),
            "en",
            "and it still says what it would write"
        );
        assert_eq!(s.model(), "opus-mt-pl-en");

        let (enabled, message) = s.enable();
        assert!(enabled, "no reload: the translator is right there");
        assert!(message.is_empty());
        assert!(s.will_translate());
    }

    #[test]
    fn disable_with_unload_frees_the_weights_and_the_reported_state() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.disable(true));
        assert!(!s.is_loaded(), "the weights are gone");
        assert_eq!(s.target_language(), "", "nothing is being written");
        assert_eq!(s.model(), "", "so there is no translator to name");
        assert!(
            !s.will_translate(),
            "and the worker stays on the inline path"
        );
    }

    #[test]
    fn an_unload_keeps_the_remembered_target_so_enable_can_reload_it() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(true);
        // `Enable` finds the remembered `en`, so it asks for a load rather than
        // refusing the way a run that never named a language does.
        let (enabled, message) = s.enable();
        assert!(!enabled, "the reload is in flight");
        assert!(message.is_empty(), "not a refusal");
        assert_eq!(
            s.requested_language(),
            "en",
            "and it asks for the right one"
        );
    }

    #[test]
    fn toggle_flips_the_step_without_unloading() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        assert!(!s.toggle(), "on -> off");
        assert!(
            s.is_loaded(),
            "a toggle is a state change, not a memory decision"
        );
        assert!(s.toggle(), "off -> on, with no reload");
    }

    #[test]
    fn the_same_language_again_is_a_switch_on_not_a_load() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        s.disable(false);
        let (accepted, target, message) = s.set_target_language("en");
        assert!(accepted);
        assert_eq!(target, "en");
        assert!(message.is_empty());
        assert!(s.is_enabled(), "the same language turns the step back on");
        assert_eq!(
            s.requested_language(),
            "",
            "and nothing is loaded, because the translator was already there"
        );
    }

    #[test]
    fn the_target_language_is_normalized_the_way_the_catalog_compares_it() {
        let s = state("pl");
        // Trimmed and lowercased, so ` EN ` and `en` are one language and the
        // reported value matches what the catalog resolved.
        let (accepted, target, _) = s.set_target_language(" EN ");
        assert!(accepted);
        assert_eq!(target, "en");
    }

    #[test]
    fn a_failed_first_load_leaves_the_step_off() {
        let s = state("pl");
        let generation = s.begin_request("en");
        s.wanted.store(true, Ordering::Release);
        s.fail_request(generation, "the decoder ran out of cache slots");
        assert!(
            !s.is_enabled(),
            "`Enabled` must never claim work that cannot happen"
        );
        assert_eq!(s.requested_language(), "");
        assert!(!s.will_translate());
    }

    #[test]
    fn a_failed_switch_keeps_the_language_the_run_had() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let generation = s.begin_request("de");
        s.fail_request(generation, "the directory is missing a graph");
        assert!(s.is_enabled(), "the old translator still serves");
        assert_eq!(s.target_language(), "en");
        assert_eq!(s.model(), "opus-mt-pl-en");
        assert_eq!(s.requested_language(), "", "the failed request is cleared");
    }

    #[test]
    fn a_superseded_failure_is_not_reported() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let stale = s.begin_request("de");
        let live = s.begin_request("en");
        assert!(live > stale);
        // The first load fails after the second was asked for: its message is
        // about a language nobody wants any more, so it is dropped.
        s.fail_request(stale, "stale failure");
        assert_eq!(
            s.requested_language(),
            "en",
            "the live request is untouched"
        );
    }

    #[test]
    fn get_state_reports_the_snapshot_the_properties_read() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (enabled, target, requested, model) = s.state();
        assert!(enabled);
        assert_eq!(target, "en");
        assert_eq!(requested, "");
        assert_eq!(model, "opus-mt-pl-en");
        // The four reads must agree, or a client stitching them together sees a
        // state no single moment had.
        assert_eq!(enabled, s.is_enabled());
        assert_eq!(target, s.target_language());
        assert_eq!(requested, s.requested_language());
        assert_eq!(model, s.model());
    }

    #[test]
    fn a_voice_key_states_its_language_and_a_path_states_nothing() {
        assert_eq!(voice_language("pl_PL-gosia-medium"), Some("pl"));
        assert_eq!(voice_language("en_US-amy-medium"), Some("en"));
        assert_eq!(voice_language("de_DE-thorsten"), Some("de"));
        // A path is a directory, not a key: no language to compare against.
        assert_eq!(voice_language("/opt/voices/amy"), None);
        assert_eq!(voice_language("C:\\voices\\amy"), None);
        // Nothing before the first separator.
        assert_eq!(voice_language("-medium"), None);
    }

    #[test]
    fn split_cuts_after_a_terminator() {
        assert_eq!(
            split_sentences("Cześć. Jak się masz?"),
            vec!["Cześć.", "Jak się masz?"]
        );
    }

    #[test]
    fn split_keeps_a_colon_piece_whole() {
        // The `.` in `12.30` is a decimal point and the `:` is a label, so this
        // is one piece, not three.
        assert_eq!(split_sentences("cena: 12.30 zł"), vec!["cena: 12.30 zł"]);
    }

    #[test]
    fn split_keeps_text_without_a_terminator_whole() {
        assert_eq!(split_sentences("cena 12 zł"), vec!["cena 12 zł"]);
    }

    #[test]
    fn split_never_emits_an_empty_piece() {
        // The leading `.;` has no piece to fold onto, so it stays a piece of
        // its own: it holds characters, and nothing may be dropped.
        let pieces = split_sentences("  \n .  ;  Dobry wieczór!  \n\n");
        assert_eq!(pieces, vec![".;", "Dobry wieczór!"]);
        assert!(pieces.iter().all(|p| !p.trim().is_empty()));
    }

    #[test]
    fn split_keeps_every_delimiter_with_its_piece() {
        assert_eq!(split_sentences("A; B\nC!"), vec!["A;", "B", "C!"]);
    }

    #[test]
    fn split_of_empty_text_is_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("  \n ").is_empty());
    }

    #[test]
    fn fallback_keeps_a_translation() {
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok("Good evening".to_string()), "t"),
            "Good evening"
        );
    }

    #[test]
    fn fallback_keeps_the_source_on_an_empty_result() {
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok(String::new()), "t"),
            "Dobry wieczór"
        );
        assert_eq!(
            fallback_or_translated("Dobry wieczór", Ok("   ".to_string()), "t"),
            "Dobry wieczór"
        );
    }

    #[test]
    fn fallback_keeps_the_source_on_an_error() {
        assert_eq!(
            fallback_or_translated(
                "Dobry wieczór",
                Err(anyhow!("the decoder ran out of cache slots")),
                "t"
            ),
            "Dobry wieczór"
        );
    }

    /// The `tokenizers` 0.23 panic, reproduced from a literal so the workaround
    /// is pinned without downloading weights.
    ///
    /// Built from a trimmed stand-in: only the `normalizer` key and the two
    /// fields that must survive matter here.
    #[test]
    fn the_null_precompiled_normalizer_is_dropped_before_loading() {
        let raw = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "normalizer": {"type": "Precompiled", "precompiled_charsmap": null},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "WhitespaceSplit"},
                    {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
                ]
            },
            "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
            "model": {"type": "Unigram", "unk_id": 1, "vocab": [["▁the", -1.0], ["</s>", 0.0]]}
        }"#;
        let mut value: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert!(
            value.get("normalizer").is_some(),
            "the fixture must reproduce the null charsmap"
        );
        assert!(patched_value(&mut value), "a null charsmap must be dropped");
        assert!(
            !String::from_utf8_lossy(&serde_json::to_vec(&value).unwrap())
                .contains("precompiled_charsmap"),
            "the null charsmap must not survive the patch"
        );
        // The rest of the tokenizer is untouched.
        assert!(value.get("pre_tokenizer").is_some());
        assert!(value.get("decoder").is_some());
        assert!(value.get("model").is_some());
        // And the patched form is what `tokenizers` accepts.
        let patched = serde_json::to_vec(&value).unwrap();
        Tokenizer::from_bytes(&patched).expect("a normalizer-free tokenizer must load");
        // The unpatched form is what it *panics* on, which is why this is a patch
        // and not a fallback: `from_bytes` never gets to return an error, so the
        // failure has to be caught rather than asserted on a `Result`.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let unpatched_panics =
            std::panic::catch_unwind(|| Tokenizer::from_bytes(raw.as_bytes())).is_err();
        std::panic::set_hook(previous_hook);
        assert!(
            unpatched_panics,
            "if the null charsmap now loads, the patch is no longer needed"
        );
    }

    /// A working normalizer is **kept**, which is the case the patch was wrong
    /// about. NLLB's export publishes a real 316 KB charsmap wrapped in a
    /// `Sequence` beside a `Replace`, and dropping it would change the
    /// tokenization — so the test is on the value being null, never on the field
    /// being present.
    ///
    /// The charsmap is a stand-in (real `precompiled_charsmap` values are a long
    /// base64 blob, and this is about the branch taken, not about what
    /// `tokenizers` makes of the bytes); the `#[ignore]`d test below loads the real
    /// file.
    #[test]
    fn a_normalizer_with_a_real_charsmap_survives_the_patch() {
        let build = |normalizer: &str| {
            format!(
                r#"{{
                    "version": "1.0",
                    "normalizer": {normalizer},
                    "decoder": {{"type": "Metaspace", "replacement": "▁"}},
                    "model": {{"type": "Unigram", "unk_id": 1,
                              "vocab": [["▁the", -1.0], ["</s>", 0.0]]}}
                }}"#
            )
        };
        // The NLLB shape: a `Sequence` whose `Precompiled` carries a real charsmap.
        let mut nllb: serde_json::Value = serde_json::from_str(&build(
            r#"{"type": "Sequence", "normalizers": [
                   {"type": "Precompiled", "precompiled_charsmap": "AAAAAA=="},
                   {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
               ]}"#,
        ))
        .unwrap();
        assert!(
            !patched_value(&mut nllb),
            "a real charsmap must not be dropped"
        );
        assert!(
            nllb.get("normalizer").is_some(),
            "the normalizer must survive a patch that had nothing to fix"
        );

        // The same shape with a null inside it is still dropped: the walk does not
        // care how deep the null is buried, which is what a `Parallel` would need.
        let mut nested: serde_json::Value = serde_json::from_str(&build(
            r#"{"type": "Sequence", "normalizers": [
                   {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "},
                   {"type": "Precompiled", "precompiled_charsmap": null}
               ]}"#,
        ))
        .unwrap();
        assert!(
            patched_value(&mut nested),
            "a null charsmap buried in a Sequence must still be found"
        );
        assert!(nested.get("normalizer").is_none());

        // And a normalizer with no charsmap at all is not ours to remove.
        let mut other: serde_json::Value =
            serde_json::from_str(&build(r#"{"type": "NFC"}"#)).unwrap();
        assert!(!patched_value(&mut other));
        assert!(other.get("normalizer").is_some());

        // A tokenizer with no normalizer is a valid shape, not an error.
        let mut none: serde_json::Value = serde_json::from_str(
            r#"{"version": "1.0",
                "decoder": {"type": "Metaspace", "replacement": "▁"},
                "model": {"type": "Unigram", "unk_id": 1,
                          "vocab": [["▁the", -1.0], ["</s>", 0.0]]}}"#,
        )
        .unwrap();
        assert!(!patched_value(&mut none));
    }

    /// The feed is the whole difference between the two families, so each half is
    /// pinned: what the tokenizer is asked for, what the framing costs, and which
    /// ids the decoder opens on.
    #[test]
    fn a_pair_export_takes_raw_text_and_a_multilingual_one_is_wrapped() {
        let fixed = Feed::Fixed { start: 63429 };
        assert!(fixed.adds_special_tokens(), "the post-processor still runs");
        assert_eq!(fixed.framing(), 0, "a pair export spends no ids on framing");
        assert_eq!(fixed.opening(), vec![63429], "one id, alone");

        let nllb = Feed::LangTagged {
            src: 256140,
            tgt: 256047,
            start: 2,
        };
        assert!(
            !nllb.adds_special_tokens(),
            "the post-processor hard-codes eng_Latn, so it must not run"
        );
        assert_eq!(nllb.framing(), 2, "the source code and the closing </s>");
        // **Two** ids, start first. Measured against the weights: opening on the
        // target alone decodes `"is"`, and `[start, target, start]` decodes `""`.
        assert_eq!(nllb.opening(), vec![2, 256047]);
        // The source code never appears in the decoder's opening: the source
        // reaches the model through the encoder, and opening on it would translate
        // *towards* Polish whatever the user asked for.
        assert!(!nllb.opening().contains(&256140));

        // Which is what makes NLLB's config legal: `decoder_start_token_id` and
        // `eos_token_id` are both 2, because the start token *is* the `</s>` prefix.
        assert_eq!(nllb.opening()[0], 2, "on the `</s>` prefix, first");
        assert_eq!(nllb.opening()[1], 256047, "the target code comes second");
    }

    /// `Suppressed` is a lookup table because a multilingual export blocks 202
    /// language codes over a 256,206-wide row. Both halves of that matter: the
    /// blocked ids must block, and an id past the table must be allowed rather
    /// than panic or be silently blocked.
    #[test]
    fn suppression_is_a_table_sized_to_its_highest_id() {
        let empty = Suppressed::build([]);
        assert!(!empty.blocks(0));
        assert!(!empty.blocks(256_140), "an empty table blocks nothing");

        let some = Suppressed::build([3, 102, 64171]);
        for id in [3, 102, 64171] {
            assert!(some.blocks(id), "{id} must be blocked");
        }
        for id in [0usize, 2, 4, 101, 103, 64170, 64172] {
            assert!(!some.blocks(id), "{id} must not be blocked");
        }
        // Past the table: allowed, not an error. This is the behaviour a
        // suppressed id outside the vocabulary has to have, or the row's real peak
        // would be unreachable.
        assert!(!some.blocks(64172));
        assert!(!some.blocks(256_206));
        assert_eq!(some.0.len(), 64172, "the table stops at its highest id");
        assert_eq!(Suppressed::build([]).0.len(), 0);

        // Negative ids cannot be table slots and are dropped rather than
        // panicking, so a malformed set still yields a usable table.
        let odd = Suppressed::build([-1, 5]);
        assert!(odd.blocks(5));
        assert!(!odd.blocks(0));
        assert_eq!(odd.0.len(), 6);
        // Duplicates collapse.
        assert_eq!(Suppressed::build([7, 7, 7]).0.len(), 8);
    }

    #[test]
    fn greedy_argmax_takes_the_row_peak() {
        let none = Suppressed::build([]);
        // Marian has no `<extra_id_*>` sentinel block, so with nothing suppressed
        // this is a plain argmax.
        let mut logits = vec![0f32; 200];
        logits[150] = 9.0;
        logits[160] = 8.0;
        assert_eq!(next_token(&logits, &none), 150);
        // Id 2 is `.` in these vocabularies, and it wins outright here: nothing
        // suppresses it. That is why a wrong start token yields punctuation rather
        // than an error.
        let mut logits = vec![0f32; 200];
        logits[2] = 10.0;
        assert_eq!(next_token(&logits, &none), 2);
        // A tie keeps the lower id.
        let mut logits = vec![0f32; 200];
        logits[103] = 2.0;
        logits[107] = 2.0;
        assert_eq!(next_token(&logits, &none), 103);
        // An empty row must not panic.
        assert_eq!(next_token(&[], &none), 0);
    }

    /// The suppression the int8 `mul-en` export forces on us: its pad logit is
    /// pinned at `0.000` and every real token sits near `-40`.
    #[test]
    fn the_pad_id_loses_to_the_best_real_token() {
        // Reproduced from the real logits of `opus-mt-mul-en` on its first step.
        let mut logits = vec![-45f32; 64172];
        logits[64171] = 0.000; // <pad>, pinned
        logits[4016] = -40.218; // "Like", the best real token
        let suppressed = Suppressed::build([64171]);
        assert_eq!(
            next_token(&logits, &suppressed),
            4016,
            "the pinned pad must not win"
        );
        // Without the suppression it does win, which is the whole bug: the decode
        // then emits `<pad>` forever and returns nothing.
        assert_eq!(next_token(&logits, &Suppressed::build([])), 64171);

        // A suppressed id that is not in range changes nothing: the table is sized
        // to its own contents, so 9999 with a 150-wide row does not block 150.
        let mut logits = vec![0f32; 200];
        logits[150] = 9.0;
        assert_eq!(next_token(&logits, &Suppressed::build([9999])), 150);
        // Several suppressed ids at once, the shape `pl-en` also uses (start and
        // pad are the same id there, so the list holds one entry).
        let mut logits = vec![0f32; 200];
        logits[3] = 10.0;
        logits[102] = 9.0;
        logits[150] = 5.0;
        assert_eq!(next_token(&logits, &Suppressed::build([3, 102])), 150);
        // A row whose every id is suppressed falls back to the plain argmax
        // rather than returning nothing the decoder cannot use.
        let mut logits = vec![0f32; 200];
        logits[3] = 10.0;
        logits[102] = 9.0;
        assert_eq!(next_token(&logits, &Suppressed::build(0..200)), 3);
    }

    /// A multilingual export blocks every added token but `</s>`, which over its
    /// 256,206-wide vocabulary means 202 language codes plus the pad and `<mask>`
    /// must all lose to a real token. Reproduced at that scale because the smaller
    /// rows above cannot show why the membership test is a lookup and not a scan.
    #[test]
    fn a_language_code_never_wins_mid_sentence() {
        let mut logits = vec![-45f32; 256_206];
        // `eng_Latn` at its real id, the one the decode *opened* on — a model that
        // repeats it is switching language, not translating.
        logits[256047] = 10.0;
        logits[1000] = -40.0;
        let suppressed = Suppressed::build((256_001..=256_203).chain([1, 0, 3]));
        assert_eq!(
            next_token(&logits, &suppressed),
            1000,
            "a language code must not win over a real token"
        );
        // Unsuppressed it wins outright, which is the failure the table prevents.
        assert_eq!(next_token(&logits, &Suppressed::build([1])), 256047);
        // `</s>` (id 2) is deliberately *not* blocked: a decode has to be able to
        // reach it, or it never stops.
        assert!(!suppressed.blocks(2));
        let mut logits = vec![-45f32; 8];
        logits[2] = 10.0;
        logits[6] = -1.0;
        assert_eq!(
            next_token(&logits, &suppressed),
            2,
            "eos must stay reachable"
        );
    }

    /// The ids come from the model, so the load path has to insist on having read
    /// them. This is the check that would have caught the `mul-en` bug: its start
    /// token is 64171, and decoding `pl-en`'s 63429 from it returns punctuation
    /// instead of failing.
    ///
    /// There is no sentinel any more. A [`Prepared`] translator has been read or
    /// does not exist, so the ids cannot be stale at load — which is a stronger
    /// guarantee than the placeholder this replaced, and the reason the load-time
    /// refusal could go.
    #[test]
    fn the_token_ids_are_read_from_the_model_not_compiled_in() {
        // Two real configs, fetched from the pairs that differ.
        for (pair, start, eos) in [("pl-en", 63429, 0), ("mul-en", 64171, 0)] {
            let raw = format!(
                r#"{{"decoder_start_token_id": {start}, "eos_token_id": {eos},
                    "pad_token_id": {start}, "bos_token_id": {eos}}}"#
            );
            let config: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(
                config
                    .get("decoder_start_token_id")
                    .and_then(serde_json::Value::as_i64),
                Some(start),
                "{pair} start token"
            );
            assert_eq!(
                config
                    .get("eos_token_id")
                    .and_then(serde_json::Value::as_i64),
                Some(eos),
                "{pair} eos token"
            );
        }
        // The two differ, which is the whole reason they cannot be constants.
        assert_ne!(63429, 64171);
    }

    #[test]
    fn use_cache_branch_is_rank_one() {
        // The flag the run path builds, read back off the tensor itself: this is
        // the assertion that keeps the rank from regressing to a scalar, which
        // the graph rejects before it runs ("Invalid rank for input:
        // use_cache_branch Got: 0 Expected: 1") and which no weight-free run
        // could otherwise catch.
        let tensor = use_cache_branch_tensor(false)
            .unwrap_or_else(|e| panic!("use_cache_branch tensor must build: {e}"));
        let dtype = tensor.dtype();
        let ValueType::Tensor { ty, shape, .. } = dtype else {
            panic!("use_cache_branch must be a tensor, got {dtype:?}");
        };
        assert_eq!(
            *ty,
            ort::value::TensorElementType::Bool,
            "the graph declares BOOL, not a number"
        );
        assert_eq!(
            shape.as_ref(),
            USE_CACHE_BRANCH_SHAPE.as_slice(),
            "use_cache_branch must be rank 1, not a rank-0 scalar"
        );
        assert_eq!(shape.num_elements(), 1, "one element in total");
        // The rank is not a coincidence of the value.
        let other = use_cache_branch_tensor(true).expect("both values build");
        assert_eq!(other.dtype(), dtype);
        assert_eq!(USE_CACHE_BRANCH_SHAPE.to_vec(), vec![1]);
    }

    /// The two exports' own geometry, as their `config.json` files publish it. Read
    /// out of the ONNX protobuf for the cache slot count and confirmed against each
    /// config for the rest: a Marian merged decoder declares
    /// `past_key_values.0..5.{decoder,encoder}.{key,value}` and NLLB the same to
    /// `..11`.
    fn marian_shape() -> GraphShape {
        GraphShape::new(6, 8, 64)
    }
    fn nllb_shape() -> GraphShape {
        GraphShape::new(12, 16, 64)
    }

    #[test]
    fn an_empty_cache_shape_keeps_the_declared_axes_and_zeroes_the_sequence() {
        // The metadata of a Marian export: `[batch, heads, sequence, head_dim]`
        // with the sequence dynamic. `ort` reports a dynamic axis as -1.
        assert_eq!(
            empty_cache_shape(&[None, Some(8), None, Some(64)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        // A fully declared sequence axis still goes to zero: that is the whole
        // point of the empty past this decode sends every step.
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8), Some(20), Some(64)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        // A dynamic head count falls back to the export's own config, because a
        // zero-length tensor cannot carry a dynamic axis. This is the assertion
        // that the fallback is *per family*: the same dynamic metadata must yield
        // 8 heads for a pair and 16 for NLLB.
        assert_eq!(
            empty_cache_shape(&[None, None, None, None], marian_shape()),
            vec![1, 8, 0, 64]
        );
        assert_eq!(
            empty_cache_shape(&[None, None, None, None], nllb_shape()),
            vec![1, 16, 0, 64]
        );
        // A rank this decode does not know falls back to the whole shape.
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8)], nllb_shape()),
            vec![1, 16, 0, 64]
        );
    }

    /// The prompt step is the only one that sends an empty cache, and a cached step
    /// is the only one that sends a filled one. Pinned as a shape assertion because
    /// the failure it prevents is a load-time shape error, not a wrong answer.
    #[test]
    fn a_past_slot_is_fed_empty_by_the_prompt_step() {
        for (shape, heads) in [(marian_shape(), 8), (nllb_shape(), 16)] {
            let slot = empty_cache_shape(&[None, Some(heads), None, Some(64)], shape);
            assert_eq!(slot, vec![1, heads, 0, 64]);
            assert_eq!(
                slot.iter().product::<usize>(),
                0,
                "an empty past holds no elements at all"
            );
            assert_eq!(slot[2], 0, "only the sequence axis is zeroed");
        }
    }

    /// Which slots a cached step replaces, and what it reads them out of.
    ///
    /// The one thing nothing else can check: `cache_slots` only ever runs against a
    /// loaded session, and getting the split wrong feeds each layer's
    /// cross-attention tensors to its self-attention slot — the same
    /// `[1, heads, n, head_dim]` either way, so nothing errors and only the
    /// transcript shows it. So the naming, the split and the output name are pinned
    /// here against the export's own convention, over both families' layer counts.
    #[test]
    fn a_cached_step_replaces_the_self_attention_slots_only() {
        for shape in [marian_shape(), nllb_shape()] {
            // What a real export calls its slots, in its own order: per layer, the
            // self key and value, then the cross key and value.
            let slots: Vec<String> = (0..shape.layers)
                .flat_map(|layer| {
                    [SELF_SLOT, CROSS_SLOT].into_iter().flat_map(move |half| {
                        ["key", "value"]
                            .into_iter()
                            .map(move |kv| format!("{PAST_PREFIX}{layer}{half}{kv}"))
                    })
                })
                .collect();
            assert_eq!(slots.len(), shape.cache_slots(), "four slots per layer");

            let self_slots: Vec<&String> = slots.iter().filter(|n| is_self_slot(n)).collect();
            assert_eq!(
                self_slots.len(),
                shape.layers * 2,
                "one self key and a value per layer"
            );
            for name in &self_slots {
                assert!(
                    !name.contains(CROSS_SLOT),
                    "{name} is one half or the other"
                );
                // And each is read back out under its own name with the prefix
                // swapped, which is the only derivation that keeps a name and the
                // tensor stored under it from drifting apart.
                let present = present_name(name);
                assert_eq!(present, name.replacen(PAST_PREFIX, "present.", 1));
                assert!(
                    present.starts_with("present.") && !present.contains(PAST_PREFIX),
                    "{present} is not a present name"
                );
            }
            // The cross half is captured once, at the prompt, and never rewritten:
            // a cached step that replaced it would copy the encoder's keys and
            // values out of the output and back on every token, unchanged.
            for name in slots.iter().filter(|n| n.contains(CROSS_SLOT)) {
                assert!(!is_self_slot(name), "{name} must stay out of the self half");
            }
        }
        // A name the export did not use is not a slot this decode can place, which
        // is why `cache_slots` refuses one rather than guessing which half it is.
        assert!(!is_self_slot("past_key_values.0.encoder.key"));
        assert!(!is_self_slot("past_key_values.0.something.key"));
    }

    /// The cache contract is read from the model, so what is pinned here is the
    /// *arithmetic* — four slots per layer — and the two shapes it produces. The
    /// slot count is not written down in either family: a Marian export declares
    /// 24 and NLLB 48, and `check_cache_inputs` compares the graph against whatever
    /// the config said rather than against a constant.
    #[test]
    fn the_cache_slot_count_follows_the_layer_count() {
        assert_eq!(
            marian_shape().cache_slots(),
            24,
            "6 x self+cross x key+value"
        );
        assert_eq!(
            nllb_shape().cache_slots(),
            48,
            "12 x self+cross x key+value"
        );
        // Both agree with `d_model / head_dim == heads`, which is where the head
        // width comes from.
        for (shape, d_model) in [(marian_shape(), 512), (nllb_shape(), 1024)] {
            assert_eq!(d_model / shape.head_dim(), shape.heads);
            assert_eq!(shape.cache_slots(), shape.layers * 4);
        }
        // The decode caps are per family for the same reason the shapes are: the
        // same O(n^2) decode over a decoder four times the width.
        assert!(Family::Nllb.max_new_tokens() < Family::Marian.max_new_tokens());
    }

    /// End-to-end against the downloaded export, which is the only thing that can
    /// catch a wrong start token, a wrong cache shape or a wrong graph contract —
    /// every one of which produced garbage while unit tests were green.
    ///
    /// Ignored by default because it needs the ~110 MiB of weights at
    /// `models/translators/opus-mt-pl-en/`. It skips itself when they are not
    /// there rather than fetching them: `--ignored` on a fresh checkout is a
    /// question about what is on disk, not a 1.3 GiB download.
    ///
    /// Run it with:
    /// `cargo test -- --ignored real_export_translates_polish_to_english --nocapture`
    #[test]
    #[ignore = "needs the downloaded opus-mt-pl-en weights in models/translators/opus-mt-pl-en/"]
    fn real_export_translates_polish_to_english() {
        if !weights_are_present("opus-mt-pl-en") {
            return;
        }
        let prepared = crate::translator_store::prepare(
            "opus-mt-pl-en",
            crate::translator_store::Direction {
                src: Some("pl"),
                tgt: "en",
            },
        )
        .expect("the pl-en spec must resolve");
        assert_eq!(prepared.facts.family, Family::Marian);
        assert_eq!(prepared.facts.start_token, 63429);
        assert_eq!(prepared.facts.shape.cache_slots(), 24);
        let mut translator = Translator::load(&prepared, 4).expect("the export must load");
        // The raw source sentence, which is what the pipeline hands the step: the
        // gates' output, never a prompt. Marian takes the text as it is.
        let (out, ms) = timed_translate(&mut translator, "Kot siedzi na stole.");
        eprintln!("marian pl->en output: {out:?} ({ms:.0} ms)");
        assert_translation_is_english_prose(&out, "cat");
        // A cached decode is O(n) in output length, so one sentence is a handful of
        // single-token steps: 79 ms measured on four threads. The bound is loose on
        // purpose, and it is about the decode still being O(n) — a regression to the
        // single-graph form costs a factor, not a rounding error.
        assert!(ms < 1_000.0, "one sentence took {ms:.0} ms");
    }

    /// End-to-end against the downloaded **multilingual** export, which is the only
    /// thing that can catch a wrong language code in either position: a decoder
    /// opened on the *source* code translates towards Polish whatever was asked
    /// for, and an encoder missing its trailing `</s>` reads the source sentence as
    /// unfinished. Both produce fluent text, so neither is caught by an assertion
    /// on emptiness.
    ///
    /// Ignored by default because it needs the ~1.3 GiB of weights at
    /// `models/translators/nllb-200-distilled-600M/`; it skips itself when they are
    /// not there, so `--ignored` is safe on a fresh checkout.
    ///
    /// Run it with:
    /// `cargo test -- --ignored real_nllb_translates_polish_to_english --nocapture`
    #[test]
    #[ignore = "needs the downloaded nllb-200-distilled-600M weights"]
    fn real_nllb_translates_polish_to_english() {
        if !weights_are_present(crate::translator_store::NLLB_SPEC) {
            return;
        }
        let prepared = crate::translator_store::prepare(
            crate::translator_store::NLLB_SPEC,
            crate::translator_store::Direction {
                src: Some("pl"),
                tgt: "en",
            },
        )
        .expect("the NLLB spec must resolve");
        // The family and the geometry come out of the file, and they are the two
        // things a hard-coded shape would get wrong.
        assert_eq!(prepared.facts.family, Family::Nllb);
        assert_eq!(prepared.facts.shape.cache_slots(), 48);
        assert_eq!(prepared.facts.shape.heads, 16);
        assert_eq!(
            prepared.codes,
            Some(("pol_Latn".to_string(), "eng_Latn".to_string())),
            "the run's own languages become the codes the model is told"
        );
        let mut translator = Translator::load(&prepared, 4).expect("the export must load");
        let (english, ms) = timed_translate(&mut translator, "Kot siedzi na stole.");
        eprintln!("nllb pl->en output: {english:?} ({ms:.0} ms)");
        assert_translation_is_english_prose(&english, "cat");
        // The cap is what bounds a runaway, so a normal sentence has to be far
        // under it: 48 single-token steps over a decoder four times the width is
        // 677 ms measured on four threads.
        assert!(ms < 8_000.0, "one sentence took {ms:.0} ms");
        // Both directions' sessions are live at once from here, which is ~1.8 GiB of
        // weights plus their arenas, so the first one goes before the second loads.
        drop(translator);

        // A direction a Marian pair export cannot reach at all: this is the whole
        // reason the multilingual export exists.
        let german = crate::translator_store::prepare(
            crate::translator_store::NLLB_SPEC,
            crate::translator_store::Direction {
                src: Some("pl"),
                tgt: "de",
            },
        )
        .expect("pl->de must resolve even though opus-mt-pl-de is not published");
        assert_eq!(
            german.codes,
            Some(("pol_Latn".to_string(), "deu_Latn".to_string()))
        );
        let mut translator = Translator::load(&german, 4).expect("the export must load");
        let (german_out, ms) = timed_translate(&mut translator, "Kot siedzi na stole.");
        eprintln!("nllb pl->de output: {german_out:?} ({ms:.0} ms)");
        assert!(
            !german_out.trim().is_empty(),
            "translation must not be empty"
        );
        assert!(
            !german_out.contains('\u{2581}'),
            "metaspace leaked: {german_out:?}"
        );
        // It must be German, not the English the run above produced: the target
        // code is what decides the direction, and a decode that ignored it would
        // return the same text for both.
        assert_ne!(
            german_out.to_lowercase(),
            english.to_lowercase(),
            "pl->de returned what pl->en returned: the target code was not honoured"
        );
        assert!(ms < 8_000.0, "one sentence took {ms:.0} ms");
    }

    /// Whether a spec's weights are already on disk, saying so when they are not.
    ///
    /// Checked **before** `prepare`, which is the point: `prepare` downloads
    /// whatever is missing, so a helper that ran after it would find the files it
    /// was checking for and never skip. An `--ignored` test that quietly fetches
    /// 1.3 GiB is not a test anyone runs on a fresh checkout.
    fn weights_are_present(spec: &str) -> bool {
        let r = crate::translator_store::resolve(spec).expect("the spec must resolve");
        let present = [
            &r.encoder,
            &r.decoder,
            &r.decoder_with_past,
            &r.tokenizer,
            &r.config,
        ]
        .into_iter()
        .all(|path| path.is_file());
        if !present {
            eprintln!(
                "skipping: no weights in {} (run sttd with --translate-model {spec} once)",
                r.dir.display()
            );
        }
        present
    }

    fn timed_translate(translator: &mut Translator, text: &str) -> (String, f64) {
        let t0 = Instant::now();
        let out = translator.translate(text).expect("the decode must run");
        (out, t0.elapsed().as_secs_f64() * 1000.0)
    }

    /// The checks a translated sentence has to pass whatever produced it: not
    /// empty, no leaked metaspace marker, mostly letters rather than punctuation
    /// soup, and containing the word the reference translation has.
    fn assert_translation_is_english_prose(out: &str, expected: &str) {
        assert!(!out.trim().is_empty(), "translation must not be empty");
        assert!(
            !out.contains('\u{2581}'),
            "a metaspace marker leaked into the output: {out:?}"
        );
        assert!(
            out.to_lowercase().contains(expected),
            "expected a translation containing {expected:?}, got {out:?}"
        );
        let letters = out.chars().filter(char::is_ascii_alphabetic).count();
        let visible = out.chars().filter(|c| !c.is_whitespace()).count();
        assert!(
            visible > 0,
            "translation has no visible characters: {out:?}"
        );
        assert!(
            letters * 10 >= visible * 6,
            "output is not mostly letters ({letters} of {visible}): {out:?}"
        );
    }
}
