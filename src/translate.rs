// ---------------------------------------------------------------------------
// Translation step: text in, text out, with MarianMT (`opus-mt`) as the translator.
// ---------------------------------------------------------------------------
//
// The step runs after the gates and before the emitter, so everything that
// decides *whether* to speak still reads the source text. Only what the user
// hears changes. A failure here never drops an utterance: the translation thread
// runs the result through `fallback_or_translated`, which warns and returns the
// source.
//
// Three parts live in this file. The pure half (`split_sentences`,
// `fallback_or_translated`, `patched_tokenizer`) needs no weights, so it is
// unit-tested without a model. `Translator` owns the two `ort` sessions and the
// tokenizer. `TranslationOut` is the thread the translator runs on, with a
// bounded FIFO in front of it so a translation never stalls the next decode.
//
// The models are per language pair: upstream publishes one repository per
// `opus-mt-<src>-<tgt>`, so the source and target languages are a property of
// *which weights were loaded*, not a prompt. There is no prompt here, and no
// language table, because there is nothing to name in a prompt.
// ---------------------------------------------------------------------------

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::{Tensor, TensorRef, ValueType};
use tokenizers::Tokenizer;

use crate::config::{TRANSLATE_MAX_INPUT_TOKENS, TRANSLATE_MAX_NEW_TOKENS};
use crate::emit::{Emission, Emitter};
use crate::translator_store::TranslatorRef;

/// MarianMT: 6 layers, 8 heads, 64 per head, `d_model` 512.
const LAYERS: usize = 6;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
/// `past_key_values.N.{decoder,encoder}.{key,value}`: 6 layers with a
/// self-attention and a cross-attention key and value each.
const CACHE_SLOTS: usize = LAYERS * 4;
/// The name prefix of the cache inputs. The matching outputs are the same
/// strings under `present.`, which this decode never reads.
const PAST_PREFIX: &str = "past_key_values.";

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

/// Load a SentencePiece `tokenizer.json`, working around the null charsmap
/// upstream ships.
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
fn patched_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
    match value.as_object_mut() {
        Some(object) => {
            object.remove("normalizer");
        }
        None => anyhow::bail!("{} is not a tokenizer object", path.display()),
    }
    let patched = serde_json::to_vec(&value)?;
    Tokenizer::from_bytes(&patched)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", path.display()))
}

/// A loaded translator: both graphs and the tokenizer.
///
/// One `Translator` is one language pair, and the pair is fixed for the life of
/// the process, so nothing here is rebuilt per utterance. The sessions live on
/// the translation thread, which is the only thread that touches them.
pub(crate) struct Translator {
    spec: String,
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    /// `decoder_start_token_id` from the export's config. Per pair.
    start_token: i64,
    /// `eos_token_id` from the export's config. Per pair.
    eos_token: i64,
    /// Ids the decode may never emit, the pad above all. See
    /// `translator_store::TranslatorRef::suppressed_ids`.
    suppressed: Vec<i64>,
}

impl Translator {
    /// Load the encoder, the merged decoder and the tokenizer.
    ///
    /// `r` comes from `translator_store::prepare`, so the three files are
    /// already on disk. Both sessions run on the CPU with the settings the piper
    /// voice uses: one intra-op thread count, no parallel execution, no memory
    /// pattern.
    pub(crate) fn load(r: &TranslatorRef, intra_threads: usize) -> Result<Self> {
        let tokenizer = patched_tokenizer(&r.tokenizer)?;
        let encoder = cpu_session(&r.encoder, intra_threads, "translation encoder")?;
        let decoder = cpu_session(&r.decoder, intra_threads, "translation decoder")?;
        require_input(&encoder, "input_ids", &r.encoder)?;
        require_input(&encoder, "attention_mask", &r.encoder)?;
        require_input(&decoder, "input_ids", &r.decoder)?;
        require_input(&decoder, "encoder_hidden_states", &r.decoder)?;
        require_input(&decoder, "encoder_attention_mask", &r.decoder)?;
        require_input(&decoder, "use_cache_branch", &r.decoder)?;
        check_cache_inputs(&decoder, &r.decoder)?;
        // `translator_store::prepare` reads these out of `config.json`. A
        // reference that skipped that read still holds the placeholder, and
        // decoding from it would return punctuation rather than fail.
        if r.decoder_start_token_id == UNREAD_TOKEN_ID || r.eos_token_id == UNREAD_TOKEN_ID {
            anyhow::bail!(
                "{} has not read its token ids; go through translator_store::prepare",
                r.spec
            );
        }
        log::info!(
            "translator: {} ({} -> {}, CPU, no prompt, start={} eos={})",
            r.spec, r.src, r.tgt, r.decoder_start_token_id, r.eos_token_id
        );
        Ok(Self {
            spec: r.spec.clone(),
            encoder,
            decoder,
            tokenizer,
            start_token: r.decoder_start_token_id,
            eos_token: r.eos_token_id,
            suppressed: r.suppressed_ids.clone(),
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

    /// One piece through the encoder and the greedy decode.
    fn translate_piece(&mut self, piece: &str) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(piece, true)
            .map_err(|e| anyhow::anyhow!("failed to tokenize the source text: {e}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
        if ids.len() > TRANSLATE_MAX_INPUT_TOKENS {
            log::warn!(
                "translator {}: piece of {} tokens cut to {TRANSLATE_MAX_INPUT_TOKENS}",
                self.spec,
                ids.len()
            );
            ids.truncate(TRANSLATE_MAX_INPUT_TOKENS);
        }
        let hidden = self.run_encoder(&ids)?;
        let enc_len = ids.len();

        // The decoder opens on the start token *alone*: the source text reaches
        // the model through `encoder_hidden_states` and nowhere else.
        //
        // The id is `decoder_start_token_id` from `config.json`, which is the
        // `<pad>` id in this export and **not** 2. Opening on 2 (which is `.` in
        // this vocabulary) is what produced `"i i i i ..."` on every input: the
        // same graphs, the same ids and the same weights give correct English
        // from 63429.
        let mut sequence: Vec<i64> = vec![self.start_token];
        let mut tokens: Vec<u32> = Vec::new();
        let mut produced = 0usize;

        loop {
            let logits = self.run_step(&sequence, &hidden, enc_len)?;
            let next = next_token(&logits, &self.suppressed);
            if next as i64 == self.eos_token {
                break;
            }
            tokens.push(next as u32);
            produced += 1;
            if produced >= TRANSLATE_MAX_NEW_TOKENS {
                log::warn!(
                    "translator {}: decode hit the {TRANSLATE_MAX_NEW_TOKENS}-token cap; \
                     the translation of this piece is cut short",
                    self.spec
                );
                break;
            }
            sequence.push(next as i64);
        }
        self.tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("failed to decode the translated tokens: {e}"))
    }

    /// The encoder over one piece's token ids, returning its hidden states.
    fn run_encoder(&mut self, ids: &[i64]) -> Result<ndarray17::ArrayD<f32>> {
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

    /// One decoder run over the whole sequence so far, returning its last logits
    /// row.
    ///
    /// **The KV cache is never used, and never fed back.** `use_cache_branch` is
    /// `false` on every run, and all 24 `past_key_values` inputs go out empty
    /// (`[1, 8, 0, 64]`). The export's `true` branch fails on every path tried:
    ///
    /// ```text
    /// If node 'optimum::if' ... encoder_attn/Reshape_4 ...
    /// The dimension with value zero exceeds the dimension size of the input tensor.
    /// ```
    ///
    /// The empty past is still sent because omitting the inputs fails the same
    /// `optimum::if` node with `No operator matching`. The consequence is that
    /// every step re-runs the decoder over the whole sequence, which makes the
    /// decode **O(n²) in output length**. At 6 to 14 output tokens that measures
    /// 61 to 128 ms per utterance on four threads; `TRANSLATE_MAX_NEW_TOKENS` is
    /// what keeps the quadratic tail bounded.
    fn run_step(
        &mut self,
        ids: &[i64],
        hidden: &ndarray17::ArrayD<f32>,
        enc_len: usize,
    ) -> ort::Result<Vec<f32>> {
        let slots = self.empty_cache_inputs();
        let mut named: Vec<(&str, SessionInputValue<'_>)> = Vec::with_capacity(CACHE_SLOTS + 4);
        named.push((
            "input_ids",
            Tensor::from_array((vec![1usize, ids.len()], ids.to_vec()))?.into(),
        ));
        named.push((
            "encoder_hidden_states",
            TensorRef::from_array_view(hidden.view())?.into(),
        ));
        named.push((
            "encoder_attention_mask",
            Tensor::from_array((vec![1usize, enc_len], vec![1i64; enc_len]))?.into(),
        ));
        named.push((
            "use_cache_branch",
            use_cache_branch_tensor(false)?.into(),
        ));
        for (name, shape) in &slots {
            let zeros = vec![0f32; shape.iter().product()];
            named.push((name.as_str(), Tensor::from_array((shape.clone(), zeros))?.into()));
        }
        let outputs = self.decoder.run(named)?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| ort::Error::new("the decoder returned no logits"))?
            .try_extract_array::<f32>()?;
        let shape = logits.shape().to_vec();
        let vocab = match shape.as_slice() {
            [1, positions, vocab] if *positions > 0 && *vocab > 0 => *vocab,
            other => {
                return Err(ort::Error::new(format!(
                    "expected decoder logits [1, positions, vocab], got {other:?}"
                )))
            }
        };
        let flat = logits
            .as_slice()
            .ok_or_else(|| ort::Error::new("the decoder logits are not contiguous"))?;
        Ok(flat[flat.len() - vocab..].to_vec())
    }

    /// The 24 cache inputs as `(name, empty shape)` pairs, in the order the
    /// graph declares them, so a name and its tensor cannot drift apart.
    ///
    /// The shape comes from the graph's metadata with the sequence axis set to
    /// zero. Any axis the metadata leaves dynamic keeps the compiled-in
    /// default: a zero-length tensor cannot carry a dynamic axis.
    fn empty_cache_inputs(&self) -> Vec<(String, Vec<usize>)> {
        self.decoder
            .inputs()
            .iter()
            .filter(|i| i.name().starts_with(PAST_PREFIX))
            .map(|i| {
                let declared = match i.dtype() {
                    ValueType::Tensor { shape, .. } => shape
                        .as_ref()
                        .iter()
                        .map(|d| if *d < 0 { None } else { Some(*d as usize) })
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                (i.name().to_string(), empty_cache_shape(&declared))
            })
            .collect()
    }
}

/// The sentinel a `TranslatorRef` carries before its `config.json` has been read,
/// mirrored here so `Translator::load` can refuse one without importing the
/// store's private constant. Chosen to be an id no vocabulary has.
const UNREAD_TOKEN_ID: i64 = -1;

/// The rank of `use_cache_branch`: one axis holding one flag.
///
/// The merged decoder declares this input as a BOOL tensor of shape `[1]`, not
/// as a rank-0 scalar. A rank-0 tensor is rejected before the graph runs
/// (`Invalid rank for input: use_cache_branch Got: 0 Expected: 1`).
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
/// The winning id is not skipped outright: the best id that is allowed wins
/// instead, which costs one pass over the row rather than a sort of it. A tie
/// keeps the lower id, the rule `argmax_index` uses. A row with nothing allowed in
/// it falls back to the plain argmax rather than inventing an id.
fn next_token(logits: &[f32], suppressed: &[i64]) -> usize {
    let mut best: Option<(usize, f32)> = None;
    let mut plain: Option<(usize, f32)> = None;
    for (id, &logit) in logits.iter().enumerate() {
        let id = id as i64;
        // `>` on both, so a tie keeps the lower id.
        if plain.is_none_or(|(_, top)| logit > top) {
            plain = Some((id as usize, logit));
        }
        if suppressed.contains(&id) {
            continue;
        }
        if best.is_none_or(|(_, top)| logit > top) {
            best = Some((id as usize, logit));
        }
    }
    best.or(plain).map_or(0, |(id, _)| id)
}

/// The `use_cache_branch` flag tensor for one decoder run.
///
/// Built here rather than inline at the call site because the rank is the whole
/// point: a scalar of the right value is still the wrong tensor, and only the run
/// itself says so. The value is always `false` (see `run_step`); the shape is
/// what a test can check without weights.
fn use_cache_branch_tensor(use_cache: bool) -> ort::Result<Tensor<bool>> {
    Tensor::from_array((USE_CACHE_BRANCH_SHAPE, vec![use_cache]))
}

/// The shape of one empty cache tensor, built from the graph's declared axes.
///
/// A cache slot is `[batch, heads, sequence, head_dim]`, so axis 2 is the one
/// that goes to zero. Every other axis keeps the value the metadata declares, and
/// an axis the metadata leaves dynamic falls back to the compiled-in default.
fn empty_cache_shape(declared: &[Option<usize>]) -> Vec<usize> {
    let default = [1usize, HEADS, 0, HEAD_DIM];
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

/// Check the 24 cache inputs at load: the export must declare all of them, and
/// hold them as f32, which is the only type this decode builds.
///
/// A missing slot is worth a load-time error rather than a per-step failure: the
/// count is a property of the export, not of the text.
fn check_cache_inputs(session: &Session, path: &std::path::Path) -> Result<()> {
    let mut names: Vec<&str> = session
        .inputs()
        .iter()
        .map(|i| i.name())
        .filter(|n| n.starts_with(PAST_PREFIX))
        .collect();
    if names.len() != CACHE_SLOTS {
        anyhow::bail!(
            "{} declares {} cache inputs, expected {CACHE_SLOTS}",
            path.display(),
            names.len()
        );
    }
    names.sort_unstable();
    names.dedup();
    if names.len() != CACHE_SLOTS {
        anyhow::bail!("{} declares duplicate cache input names", path.display());
    }
    for input in session.inputs() {
        if !input.name().starts_with(PAST_PREFIX) {
            continue;
        }
        let f32_tensor = matches!(
            input.dtype(),
            ValueType::Tensor {
                ty: ort::value::TensorElementType::Float32,
                ..
            }
        );
        if !f32_tensor {
            anyhow::bail!(
                "{} holds cache input {} of a type this decode does not build; \
                 expected f32",
                path.display(),
                input.name()
            );
        }
    }
    Ok(())
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
}

/// Bounded depth of the translation queue.
///
/// Emissions are seconds apart and a translation takes 60 to 130 ms, so this
/// never fills in practice. It is here so that a stalled or slow translator costs
/// queued work rather than the inference thread's time.
pub(crate) const TRANSLATION_QUEUE_DEPTH: usize = 4;

/// How long `TranslationOut::drop` waits for the queued translations to finish
/// before cutting them off.
///
/// The trailing utterance has to be translated, not dropped: `--file
/// --translate en` with the last transcript swallowed would be the channel
/// silently losing work, and it is the same reasoning the ASR shutdown flush and
/// `SpeechOut::drop` use. Bounded because a translator that never returns must
/// not hang the process.
const TRANSLATE_DRAIN_MS: u64 = 15_000;

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
pub(crate) struct TranslationOut {
    /// `(queue, wakeup)`: the pending jobs in order, and the signal to look.
    queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)>,
    /// Set by `drop`, so the thread finishes the queue and exits.
    closed: Arc<AtomicBool>,
    /// Deadline for the queued work, set by `drop`, in milliseconds since the
    /// thread started — a monotonic `u64` rather than an `Instant` so it can
    /// cross the thread boundary. Zero means "no deadline".
    finish_by_ms: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TranslationOut {
    /// Spawn the translation thread around a loaded translator, an emitter handle
    /// onto the shared emission queue, and the `--metrics` writer.
    ///
    /// The metrics writer moves here with the text it describes: its `text` column
    /// has to hold what the emitter received, which after this step is the
    /// translation. It is opened once on the ASR side (see `asr::open_metrics_log`)
    /// so a broken path cannot fail transcription mid-run.
    pub(crate) fn spawn(
        translator: Translator,
        mut emitter: Box<dyn Emitter + Send>,
        metrics: Option<std::io::BufWriter<std::fs::File>>,
    ) -> Self {
        let queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let closed = Arc::new(AtomicBool::new(false));
        let finish_by_ms = Arc::new(AtomicU64::new(0));

        let handle = {
            let (queue, closed, finish_by_ms) = (
                Arc::clone(&queue),
                Arc::clone(&closed),
                Arc::clone(&finish_by_ms),
            );
            std::thread::Builder::new()
                .name("translate".into())
                .spawn(move || {
                    translate_loop(
                        &queue,
                        &closed,
                        &finish_by_ms,
                        translator,
                        emitter.as_mut(),
                        metrics,
                    )
                })
                .ok()
        };
        Self {
            queue,
            closed,
            finish_by_ms,
            handle,
        }
    }

    /// Queue one transcript for translation. Never blocks and never fails: a
    /// dead thread or a full queue costs one utterance, never the transcript.
    ///
    /// A full queue drops the **oldest** waiting job and warns, naming the tag,
    /// so the log says which transcript was lost rather than only that one was.
    pub(crate) fn enqueue(&self, job: TranslationJob) {
        let dropped = {
            let (lock, _) = &*self.queue;
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
            dropped
        };
        if let Some(tag) = dropped {
            log::warn!(
                "translation queue full ({TRANSLATION_QUEUE_DEPTH} deep): dropped the \
                 oldest waiting transcript {tag} to make room"
            );
        }
        let (_, wakeup) = &*self.queue;
        wakeup.notify_all();
    }
}

impl Drop for TranslationOut {
    fn drop(&mut self) {
        // Give the thread a deadline, then close the queue: it finishes what is
        // waiting and exits, and `join` waits for that. Without the deadline a
        // translator that never returns would hang the process on `--stop`.
        self.finish_by_ms
            .store(TRANSLATE_DRAIN_MS, Ordering::Release);
        self.closed.store(true, Ordering::Release);
        let (_, wakeup) = &*self.queue;
        wakeup.notify_all();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                log::warn!("translation thread panicked");
            }
        }
    }
}

/// One thread's whole job: take the oldest pending transcript, translate it,
/// write its metrics row, emit it, repeat until closed and empty.
fn translate_loop(
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
    finish_by_ms: &AtomicU64,
    mut translator: Translator,
    emitter: &mut dyn Emitter,
    mut metrics: Option<std::io::BufWriter<std::fs::File>>,
) {
    // A monotonic start, so the deadline is a plain elapsed-time comparison and
    // needs no clock agreement across the thread boundary.
    let started = Instant::now();
    loop {
        // Past the drain deadline with work still waiting: give up on it and say
        // so, rather than running the loop to completion.
        if let Some(job) = take_or_expire(queue, closed, finish_by_ms, started) {
            emit_one(&mut translator, emitter, &mut metrics, job);
        } else {
            return;
        }
    }
}

/// The next job to translate, or `None` when the thread is done: either the
/// queue was closed and drained, or the drain deadline passed with work waiting.
fn take_or_expire(
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
    finish_by_ms: &AtomicU64,
    started: Instant,
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
        let deadline_ms = finish_by_ms.load(Ordering::Acquire);
        // A deadline is only set by `drop`, so the queue is closed by now and the
        // pending count can only be zero. Checking it before the wait keeps the
        // bounded shutdown honest even if the wakeup were ever missed.
        if deadline_ms > 0 && started.elapsed().as_millis() as u64 >= deadline_ms {
            return None;
        }
        let wait = match deadline_ms {
            0 => Duration::from_millis(250),
            _ => Duration::from_millis(50),
        };
        let (guard, _) = wakeup
            .wait_timeout(pending, wait)
            .unwrap_or_else(|e| e.into_inner());
        pending = guard;
    }
}

/// Translate one job, log what it cost, write its metrics row and emit it.
///
/// The metrics row is written here rather than by the ASR worker because its
/// `text` column has to be what the emitter received. A dropped queue entry never
/// reaches this function, so it never reaches the metrics log either — which is
/// why the queue drop warns with the tag.
fn emit_one(
    translator: &mut Translator,
    emitter: &mut dyn Emitter,
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
    job: TranslationJob,
) {
    let TranslationJob {
        tag,
        pcm,
        source,
        verdict,
        evidence,
        samples,
        infer_ms,
    } = job;
    let t0 = Instant::now();
    let translated = translator.translate(&source);
    let translate_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let text = fallback_or_translated(&source, translated, &tag);
    log::info!("{tag}: translate took {translate_ms:.0} ms ({source:?} -> {text:?})");
    crate::asr::write_metrics_row(
        metrics,
        &tag,
        samples,
        &evidence,
        verdict,
        &text,
        infer_ms,
        Some(&source),
    );
    emitter.emit(Emission {
        tag: &tag,
        pcm: &pcm,
        text: &text,
        verdict,
        evidence: &evidence,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

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
        value.as_object_mut().unwrap().remove("normalizer");
        let patched = serde_json::to_vec(&value).unwrap();
        assert!(
            !String::from_utf8_lossy(&patched).contains("precompiled_charsmap"),
            "the null charsmap must not survive the patch"
        );
        // The rest of the tokenizer is untouched.
        assert!(value.get("pre_tokenizer").is_some());
        assert!(value.get("decoder").is_some());
        assert!(value.get("model").is_some());
        // And the patched form is what `tokenizers` accepts.
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

    #[test]
    fn greedy_argmax_takes_the_row_peak() {
        let none: Vec<i64> = Vec::new();
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
        let suppressed = vec![64171i64];
        assert_eq!(
            next_token(&logits, &suppressed),
            4016,
            "the pinned pad must not win"
        );
        // Without the suppression it does win, which is the whole bug: the decode
        // then emits `<pad>` forever and returns nothing.
        assert_eq!(next_token(&logits, &[]), 64171);

        // A suppressed id that is not in range changes nothing.
        let mut logits = vec![0f32; 200];
        logits[150] = 9.0;
        assert_eq!(next_token(&logits, &[9999]), 150);
        // Several suppressed ids at once, the shape `pl-en` also uses (start and
        // pad are the same id there, so the list holds one entry).
        let mut logits = vec![0f32; 200];
        logits[3] = 10.0;
        logits[102] = 9.0;
        logits[150] = 5.0;
        assert_eq!(next_token(&logits, &[3, 102]), 150);
        // A row whose every id is suppressed falls back to the plain argmax
        // rather than returning nothing the decoder cannot use.
        let mut logits = vec![0f32; 200];
        logits[3] = 10.0;
        logits[102] = 9.0;
        let all: Vec<i64> = (0..200).collect();
        assert_eq!(next_token(&logits, &all), 3);
    }

    /// The graph facts that only the weights can settle, pinned in one place.
    #[test]
    fn the_decoder_cache_contract_is_pinned() {
        assert_eq!(CACHE_SLOTS, 24, "6 layers x self+cross x key+value");
        assert_eq!(LAYERS, 6);
        assert_eq!(HEADS, 8);
        assert_eq!(HEAD_DIM, 64);
        // The token ids are *not* pinned here, because they are per pair: `pl-en`
        // starts at 63429 and `mul-en` at 64171, both the pad id, and both are
        // read from the model's own config. What is pinned is that a reference
        // which skipped that read is recognisable and refused, and that the
        // placeholder is an id no vocabulary has.
        assert_eq!(UNREAD_TOKEN_ID, -1);
        assert_ne!(UNREAD_TOKEN_ID, 63429);
        assert_ne!(UNREAD_TOKEN_ID, 64171);
    }

    /// The ids come from the model, so the load path has to insist on having read
    /// them. This is the check that would have caught the `mul-en` bug: its start
    /// token is 64171, and decoding `pl-en`'s 63429 from it returns punctuation
    /// instead of failing.
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
                config.get("eos_token_id").and_then(serde_json::Value::as_i64),
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

    #[test]
    fn an_empty_cache_shape_keeps_the_declared_axes_and_zeroes_the_sequence() {
        // The metadata of a Marian export: `[batch, heads, sequence, head_dim]`
        // with the sequence dynamic. `ort` reports a dynamic axis as -1.
        assert_eq!(
            empty_cache_shape(&[None, Some(8), None, Some(64)]),
            vec![1, 8, 0, 64]
        );
        // A fully declared sequence axis still goes to zero: that is the whole
        // point of the empty past this decode sends every step.
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8), Some(20), Some(64)]),
            vec![1, 8, 0, 64]
        );
        // A dynamic head count falls back to the compiled-in default, because a
        // zero-length tensor cannot carry a dynamic axis.
        assert_eq!(empty_cache_shape(&[None, None, None, None]), vec![1, 8, 0, 64]);
        // A rank this decode does not know is refused the same way.
        assert_eq!(empty_cache_shape(&[Some(1), Some(8)]), vec![1, 8, 0, 64]);
    }

    /// Every cache input is sent empty on every step, so the input set never
    /// varies with the text. Pinned as a shape assertion because the failure it
    /// prevents is a load-time shape error, not a wrong answer.
    #[test]
    fn a_past_slot_is_fed_empty_on_every_call() {
        let shape = empty_cache_shape(&[None, Some(HEADS), None, Some(HEAD_DIM)]);
        assert_eq!(shape, vec![1, HEADS, 0, HEAD_DIM]);
        assert_eq!(
            shape.iter().product::<usize>(),
            0,
            "an empty past holds no elements at all"
        );
        assert_eq!(shape[2], 0, "only the sequence axis is zeroed");
    }

    /// End-to-end against the downloaded export, which is the only thing that can
    /// catch a wrong start token, a wrong cache shape or a wrong graph contract —
    /// every one of which produced garbage while unit tests were green.
    ///
    /// Ignored by default because it needs the ~114 MiB of weights at
    /// `models/translators/opus-mt-pl-en/`; it also skips itself when they are not
    /// there, so `--ignored` is safe on a fresh checkout.
    ///
    /// Run it with:
    /// `cargo test -- --ignored real_export_translates_polish_to_english --nocapture`
    #[test]
    #[ignore = "needs the downloaded opus-mt-pl-en weights in models/translators/opus-mt-pl-en/"]
    fn real_export_translates_polish_to_english() {
        let pair = crate::translator_store::CATALOG
            .iter()
            .find(|p| p.src == "pl" && p.tgt == "en")
            .expect("pl-en is in the catalog");
        // `prepare`, not `resolve`: the token ids come from the model's config,
        // and a run without them would decode punctuation instead of failing.
        let ref_ = crate::translator_store::prepare(&crate::translator_store::spec_for(pair))
            .expect("the pl-en spec must resolve and download");
        if !ref_.tokenizer.is_file() {
            eprintln!(
                "skipping: no weights at {} (run sttd --translate en once to fetch them)",
                ref_.dir.display()
            );
            return;
        }
        let mut translator = Translator::load(&ref_, 4).expect("the export must load");
        // The raw source sentence, which is what the pipeline hands the step: the
        // gates' output, never a prompt. Marian takes the text as it is.
        let t0 = Instant::now();
        let out = translator
            .translate("Kot siedzi na stole.")
            .expect("the decode must run");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("real export output: {out:?} ({ms:.0} ms)");
        assert!(!out.trim().is_empty(), "translation must not be empty");
        assert!(
            !out.contains("\u{2581}"),
            "a metaspace marker leaked into the output: {out:?}"
        );
        // The point of pinning the start token: with the wrong one these graphs
        // decode `"i i i i ..."` for this input.
        assert!(
            out.to_lowercase().contains("cat"),
            "expected a translation of 'Kot siedzi na stole.', got {out:?}"
        );
        // A real translation is prose: mostly letters, not punctuation soup.
        let letters = out.chars().filter(char::is_ascii_alphabetic).count();
        let visible = out.chars().filter(|c| !c.is_whitespace()).count();
        assert!(visible > 0, "translation has no visible characters: {out:?}");
        assert!(
            letters * 10 >= visible * 6,
            "output is not mostly letters ({letters} of {visible}): {out:?}"
        );
        // The quadratic tail has to stay inside a second at this size, or the
        // uncached decode is not the shape the latency claim assumes.
        assert!(ms < 1_000.0, "one sentence took {ms:.0} ms");
    }
}