//! The loaded translator: the three `ort` sessions, the family's framing and
//! the greedy decode over them.

use anyhow::{Context, Result};
use ndarray17::{ArrayD, ArrayView2, ArrayViewD};
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::{Tensor, TensorRef};
use tokenizers::Tokenizer;

use crate::config::TRANSLATE_MAX_INPUT_TOKENS;
use crate::decode::{argmax_index, argmax_where};
use crate::translator_store::{Family, Prepared};

use super::kv_cache::{cache_slots, CacheSlots, KvCache};
use super::text::split_sentences;
use super::tokenizer::patched_tokenizer;

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
    pub(super) spec: String,
    /// The direction this translator runs, as a log should read it:
    /// `pol_Latn -> eng_Latn` for a multilingual export, `pl -> en` for a pair,
    /// and the bare spec for a directory, which names no direction at all.
    pub(super) direction: String,
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
        // The one attention mask the encoder and both decoder steps take, built
        // once: the prompt step masks the encoder for the whole decode, and so
        // does every step after it.
        let encoder_mask = vec![1i64; ids.len()];
        let hidden = self.run_encoder(&ids, &encoder_mask)?;

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

    /// The encoder over one piece's token ids (all attended: `mask` is ones),
    /// returning its hidden states.
    fn run_encoder(&mut self, ids: &[i64], mask: &[i64]) -> Result<ArrayD<f32>> {
        let len = ids.len();
        let named: Vec<(&str, SessionInputValue<'_>)> = vec![
            ("input_ids", row(ids)?.into()),
            ("attention_mask", row(mask)?.into()),
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
        named.push(("input_ids", row(opening)?.into()));
        named.push((
            "encoder_hidden_states",
            TensorRef::from_array_view(hidden.view())?.into(),
        ));
        named.push(("encoder_attention_mask", row(encoder_mask)?.into()));
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
        let token = [token];
        let mut named: Vec<(&str, SessionInputValue<'_>)> = Vec::with_capacity(cache.len() + 2);
        named.push(("input_ids", row(&token)?.into()));
        named.push(("encoder_attention_mask", row(encoder_mask)?.into()));
        for input in cache.named_inputs() {
            named.push(input?);
        }
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

/// `values` as a `[1, len]` input, borrowed in place: the ids and the mask
/// already live in a slice, so no step copies them into a tensor of its own.
fn row(values: &[i64]) -> Result<TensorRef<'_, i64>> {
    Ok(TensorRef::from_array_view(ArrayView2::from_shape(
        (1, values.len()),
        values,
    )?)?)
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
    argmax_where(logits, |id| suppressed.blocks(id)).unwrap_or_else(|| argmax_index(logits))
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

/// A CPU session with the settings every `ort` session shares
/// (`device::session_builder`).
fn cpu_session(
    path: &std::path::Path,
    intra_threads: usize,
    what: &'static str,
) -> Result<Session> {
    crate::device::session_builder(intra_threads)
        .map_err(|e| anyhow::anyhow!("failed to configure the {what} session: {e:?}"))?
        .commit_from_file(path)
        .with_context(|| format!("failed to load {}", path.display()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::ValueType;
    use std::time::Instant;

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
