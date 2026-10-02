// ---------------------------------------------------------------------------
// whisper_core inference: encoder + prompt decoder + cached-token decoder.
// ---------------------------------------------------------------------------
//
// Written against the ONNX export's own input and output names (read out of
// the `onnx-community` graphs), not against Whisper's Python API:
//
//   encoder_model_int8          input_features      -> last_hidden_state
//   decoder_model_int8          input_ids, encoder_hidden_states
//                                -> logits, present.N.{decoder,encoder}.{key,value}
//   decoder_with_past_model_int8 input_ids, past_key_values.N.{decoder,encoder}.{key,value}
//                                -> logits, present.N.decoder.{key,value}
//
// Two facts about those names drive the whole loop:
//
// - `decoder_with_past_model` takes no `encoder_hidden_states`. Its
//   cross-attention cache comes in through the past inputs, which is why the
//   encoder half of `KvCache` is captured once and re-supplied every step.
// - it also does not *return* the encoder half. So the cache is updated in
//   place, replacing only the self-attention tensors.
//
// The prompt is `[<|startoftranscript|>, <|lang|>, <|transcribe|>,
// <|notimestamps|>]`, which the whole prompt goes through in one `decoder_model`
// call; the token it predicts is the first generated token. Every later token
// is a `decoder_with_past_model` call over a single id.
//
// Decoding is greedy (argmax). No beam search, no temperature fallback: a
// live utterance is a few seconds of speech and the pipeline's transcript gate
// already drops runaway output, so the extra machinery would buy nothing here.
// ---------------------------------------------------------------------------

use std::borrow::Cow;
use std::path::Path;

use ndarray17::{Array2, ArrayD};
use ort::inputs;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use super::types::{
    window_frames, DecodingLanguage, KvCache, Language, MelConfig, WhisperError, WhisperModel,
    BEGIN_SUPPRESS_TOKENS, LANG_TOKEN_COUNT, LANG_TOKEN_FIRST, TOK_EOT, TOK_SOT, TOK_TRANSCRIBE,
};
use crate::decode::{argmax_index, softmax_prob, word_confidences as group_words};

/// Tokens in one prompt: start-of-transcript, language, task, no-timestamps.
const PROMPT_LEN: usize = 4;

/// Token budget of a decode over `audio_samples` with the dynamic window:
/// 20 tokens plus 15 per second of audio, never above `cap`.
///
/// Speech needs ~10 tokens a second at the fast end, so this has room to spare
/// for real text and cuts a decode that loops on a short window (measured: a
/// looping 2.5 s clip ran to the 448-token cap, 1.3 s of decoder time). The
/// fixed window keeps the plain cap, as before.
pub(crate) fn token_budget(audio_samples: usize, cap: usize) -> usize {
    let seconds = audio_samples.div_ceil(16_000);
    (20 + 15 * seconds).min(cap)
}

/// `last_hidden_state` is `[1, encoder_seq, d_model]`, and the export pins the
/// encoder length (1500 = 3000 mel frames / 2) but leaves `d_model` symbolic.
/// Logits come back `[1, seq, vocab]`.
fn encoder_hidden_states(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<ArrayD<f32>, WhisperError> {
    Ok(outputs
        .get("last_hidden_state")
        .ok_or_else(|| WhisperError::OutputNotFound("last_hidden_state".to_string()))?
        .try_extract_array::<f32>()?
        .to_owned())
}

/// The prompt for one inference: start-of-transcript, a language token, the
/// transcribe task, and no timestamps.
///
/// `language_token` is a token id, not a `Language`: for `Auto` the caller
/// passes the fallback and, once the first prompt run has reported what it
/// detected, the id that was actually detected (see `transcribe_samples`).
pub(crate) const fn build_prompt(language_token: i64) -> [i64; PROMPT_LEN] {
    [
        TOK_SOT,
        language_token,
        TOK_TRANSCRIBE,
        super::types::TOK_NOTIMESTAMPS,
    ]
}

/// Language id chosen by `Auto`, from the first position's logits over the
/// language-token range. `language_logits` is that range only — the caller
/// slices it out of the prompt's position-0 row rather than handing over all
/// 51866 vocabulary logits to have 100 of them read.
///
/// `None` only when there is no usable slice (too few logits, or a non-finite
/// value in range), in which case the caller's fallback language stands. A
/// *flat* slice is not such a case: it is a tie, and ties break toward the
/// lowest id, which for a silent or noise-only window spends one extra prompt
/// pass to land on `<|en|>`. Rejecting ties outright would leave the fallback
/// (`<|pl|>`) in charge, and neither answer is better — the transcript gate
/// decides whether anything is emitted.
pub(crate) fn detect_language(language_logits: &[f32]) -> Option<i64> {
    let slice = language_logits.get(..LANG_TOKEN_COUNT)?;
    if !slice.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((LANG_TOKEN_FIRST + argmax_index(slice)) as i64)
}

/// Mask `token` out of the winning set on the first generated step
/// (`begin_suppress_tokens`). The decode does not skip the id outright: it
/// takes the best logit that is *not* suppressed, so a step whose argmax is
/// exactly a suppressed token still lands on the runner-up.
fn is_suppressed(token: usize) -> bool {
    BEGIN_SUPPRESS_TOKENS.contains(&token)
}

/// The highest-scoring id that is not suppressed, ties going to the lowest id.
/// `None` when every id is suppressed, in which case the caller keeps its
/// argmax: there is nothing better to decode.
fn best_unsuppressed(logits: &[f32]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &v) in logits.iter().enumerate() {
        if !is_suppressed(i) && best.is_none_or(|b| v > logits[b]) {
            best = Some(i);
        }
    }
    best
}

impl WhisperModel {
    /// Load the three graphs and the tokenizer from `model_dir` (already
    /// guaranteed to hold them by `model_store::ensure_available`).
    ///
    /// `dynamic_window` asks for the length-free encoder (streaming): the
    /// export is patched in memory (see `dyn_encoder`) and the window follows
    /// the audio. If the patch does not apply the fixed export is loaded
    /// instead, with a warning, and `dynamic_window` ends up false.
    ///
    /// `min_window_frames` is the floor that follow stops at (`--audio-ctx`);
    /// it is stored but unused unless `dynamic_window` ends up true.
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        language: Language,
        intra_threads: usize,
        dynamic_window: bool,
        min_window_frames: usize,
    ) -> Result<Self, WhisperError> {
        let dir = model_dir.as_ref();
        let threads = intra_threads.max(1);
        let (encoder, dynamic_window) = init_encoder(dir, threads, dynamic_window)?;
        let decoder = init_session(dir, "decoder_model_int8.onnx", threads)?;
        let decoder_with_past = init_session(dir, "decoder_with_past_model_int8.onnx", threads)?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| WhisperError::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;
        verify_token_ids(&tokenizer)?;
        // A code the model has no token for fails here rather than at the first
        // utterance: an explicit `--language` is a promise about the language of
        // the output, and decoding in some other one would not be it.
        let language = language.resolve(&tokenizer)?;

        let frontend = super::frontend::Frontend::new(MelConfig::large_v3_turbo());
        Ok(Self {
            encoder,
            decoder,
            decoder_with_past,
            tokenizer,
            language,
            frontend,
            dynamic_window,
            min_window_frames,
        })
    }

    /// Samples the encoder decodes for `audio_samples` of input: the window,
    /// not the audio. 30 s with the fixed export; with the dynamic one it
    /// follows the audio (`window_frames`), which is what keeps RTF honest.
    pub fn window_samples(&self, audio_samples: usize) -> usize {
        let config = &self.frontend.config;
        window_frames(
            audio_samples,
            self.dynamic_window,
            self.min_window_frames,
            config,
        ) * config.hop_length
    }

    /// One utterance through the encoder and the greedy decode.
    ///
    /// `max_tokens` caps the generated run (the pipeline passes
    /// `WHISPER_MAX_NEW_TOKENS`). Whisper emits `<|endoftext|>` when it is done,
    /// so the cap is a backstop against a decode that never stops; a run that
    /// hits it leaves a truncated transcript, so the caller is told via the log
    /// line and the partial text.
    pub fn transcribe_samples(
        &mut self,
        pcm: &[f32],
        max_tokens: usize,
    ) -> Result<TranscriptionResult, WhisperError> {
        let n_frames = window_frames(
            pcm.len(),
            self.dynamic_window,
            self.min_window_frames,
            &self.frontend.config,
        );
        let max_tokens = if self.dynamic_window {
            token_budget(pcm.len(), max_tokens)
        } else {
            max_tokens
        };
        let hidden = self.run_encoder(pcm, n_frames)?;

        // Prompt run. For `--language auto` the language distribution is the
        // first position's logits, which this very call produces, so detection
        // is free unless the fallback guess was wrong — then the prompt is
        // re-run once with what was detected.
        //
        // That fallback is the first language token, `<|en|>`, which is also
        // what `detect_language` returns for a flat distribution. So the no-
        // signal case costs nothing extra, and a wrong guess only ever happens
        // when detection had real signal to disagree with.
        let mut lang_token = match self.language {
            DecodingLanguage::Fixed { token, .. } => token,
            DecodingLanguage::Auto => LANG_TOKEN_FIRST as i64,
        };
        let mut prompt = self.run_prompt(&hidden, lang_token)?;
        if self.language == DecodingLanguage::Auto {
            if let Some(detected) = detect_language(&prompt.language) {
                if detected != lang_token {
                    log::debug!("whisper detected language id {detected}, re-running the prompt");
                    lang_token = detected;
                    prompt = self.run_prompt(&hidden, lang_token)?;
                }
            }
            log::debug!("whisper decoded with language token id {lang_token}");
        }
        // The prompt's first position has served its purpose (language
        // detection); the decode starts from its last position.
        let PromptRun {
            mut cache, next, ..
        } = prompt;

        let mut logits = next;
        let mut tokens: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut truncated = false;

        // The loop runs at most `max_tokens` steps but issues one *fewer*
        // `decoder_with_past_model` call than that: the last token needs no
        // successor logits, since there is no step left to use them in.
        for step in 0..max_tokens {
            let mut next = argmax_index(&logits);
            if step == 0 && is_suppressed(next) {
                // The winning id is suppressed on the opening step: take the
                // best one that is not. One pass over the vocabulary, not a
                // sort of it — this is 51866 ids on the hottest path there is.
                next = best_unsuppressed(&logits).unwrap_or(next);
            }
            if next as i64 == TOK_EOT {
                break;
            }
            tokens.push(next as u32);
            probs.push(softmax_prob(&logits, next));
            if step + 1 == max_tokens {
                // Out of budget with the model still talking.
                truncated = true;
                break;
            }
            logits = self.run_cached_token(next as i64, &mut cache)?;
        }
        if truncated {
            log::warn!("whisper decode hit the {max_tokens}-token cap; transcript truncated");
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| WhisperError::Tokenizer(format!("decode failed: {e}")))?;
        Ok(TranscriptionResult {
            text,
            word_confidences: word_confidences(&self.tokenizer, &tokens, &probs),
            truncated,
        })
    }

    /// Log-mel features for the utterance (`n_frames` of window), then the
    /// encoder over them.
    ///
    /// The feature buffer belongs to the frontend, so it is not allocated per
    /// utterance.
    fn run_encoder(&mut self, pcm: &[f32], n_frames: usize) -> Result<ArrayD<f32>, WhisperError> {
        let features = self.frontend.extract(pcm, n_frames)?;
        debug_assert!(
            features.as_slice().is_some(),
            "encoder input must be contiguous"
        );
        log::trace!("running whisper encoder over {n_frames} frames...");
        let outputs = self.encoder.run(inputs![
            "input_features" => TensorRef::from_array_view(features)?,
        ])?;
        encoder_hidden_states(&outputs)
    }

    /// The whole prompt through `decoder_model`, returning the KV cache it
    /// produced plus the logits rows the decode starts from.
    ///
    /// Both halves are copied out here so nothing borrows the session's output
    /// past this call — the caller goes on to run the *other* session, and
    /// holding a `SessionOutputs` across that would tie both to one borrow.
    fn run_prompt(
        &mut self,
        hidden: &ArrayD<f32>,
        language_token: i64,
    ) -> Result<PromptRun, WhisperError> {
        log::trace!("running whisper prompt decoder...");
        let prompt = build_prompt(language_token);
        let ids = Array2::from_shape_vec((1, PROMPT_LEN), prompt.to_vec())?;
        let outputs = self.decoder.run(inputs![
            "input_ids" => TensorRef::from_array_view(ids.view())?,
            "encoder_hidden_states" => TensorRef::from_array_view(hidden.view())?,
        ])?;
        let cache = KvCache::capture(&outputs)?;
        // `[1, prompt_len, vocab]`: position 0 predicts the language token
        // (what `--language auto` reads) and the last position predicts the
        // first generated token.
        let logits = outputs
            .get("logits")
            .ok_or_else(|| WhisperError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;
        let (flat, vocab) = logits_rows(&logits)?;
        // A vocabulary too small to hold the language range leaves `language`
        // empty, and `detect_language` reads that as "no answer" — the
        // caller's fallback language stands.
        let language = flat
            .get(LANG_TOKEN_FIRST..LANG_TOKEN_FIRST + LANG_TOKEN_COUNT)
            .map(<[f32]>::to_vec)
            .unwrap_or_default();
        Ok(PromptRun {
            cache,
            language,
            next: flat[flat.len() - vocab..].to_vec(),
        })
    }

    /// One generated token through `decoder_with_past_model`, growing the
    /// self-attention half of `cache` in place.
    fn run_cached_token(
        &mut self,
        token: i64,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, WhisperError> {
        log::trace!("running whisper cached decoder step...");
        let ids = Array2::from_shape_vec((1, 1), vec![token])?;
        // The graph takes 17 named inputs and 16 of the names are generated
        // (`past_key_values.3.encoder.value` and friends), so `inputs!` — which
        // needs literal names — cannot build this. A named map is the same
        // thing `inputs!` produces.
        let mut named: Vec<(&str, ort::session::SessionInputValue<'_>)> = Vec::with_capacity(17);
        named.push(("input_ids", TensorRef::from_array_view(ids.view())?.into()));
        named.extend(cache.named_inputs());
        let outputs = self.decoder_with_past.run(named)?;
        cache.update_decoder(&outputs)?;
        // `[1, 1, vocab]` for a single token, so the flat data *is* the row.
        let logits = outputs
            .get("logits")
            .ok_or_else(|| WhisperError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;
        let (flat, _) = logits_rows(&logits)?;
        Ok(flat.to_vec())
    }
}

/// The contiguous data of a `[1, positions, vocab]` logits output, with the
/// row stride.
///
/// `ort` hands back a dense ONNX tensor, so each position's logits are one
/// contiguous `vocab`-long run and callers can slice rows straight off the flat
/// data. Both halves are checked rather than assumed: an unexpected rank means
/// the graph is not the one this engine was written against, and every row
/// slice would be silently wrong.
fn logits_rows<'a>(
    logits: &'a ndarray17::ArrayViewD<'a, f32>,
) -> Result<(&'a [f32], usize), WhisperError> {
    let shape = logits.shape();
    let (_, vocab) = match shape {
        [1, positions, vocab] if *positions > 0 && *vocab > 0 => (*positions, *vocab),
        other => {
            return Err(WhisperError::Logits(format!(
                "expected [1, positions, vocab], got {other:?}"
            )))
        }
    };
    let flat = logits
        .as_slice()
        .ok_or_else(|| WhisperError::Logits("output is not contiguous".to_string()))?;
    Ok((flat, vocab))
}

/// What the prompt run leaves behind: the KV cache to feed the cached steps,
/// and the two logits the decode reads — the language range of position 0 and
/// the whole of the last position.
///
/// Fully owned so the caller can keep using the sessions afterwards.
struct PromptRun {
    cache: KvCache,
    /// Position 0's logits over the language tokens only, which is all
    /// `detect_language` reads.
    language: Vec<f32>,
    /// The last position's full row: the first generated token.
    next: Vec<f32>,
}

/// One utterance's transcription. `word_confidences` is index-aligned with the
/// decoded words (not tokens) for the dictionary gate; `truncated` says the
/// token cap cut the decode short, so `text` is a prefix of what the model
/// wanted to say.
pub struct TranscriptionResult {
    pub text: String,
    pub word_confidences: Vec<(String, f32)>,
    pub truncated: bool,
}

/// Per-word confidences from the generated token run, in the same shape the
/// dictionary gate expects from Parakeet (see
/// `parakeet_core::helpers::word_confidences`).
///
/// Whisper's tokens are byte-level BPE, so "starts with a space" is the
/// decoded piece starting with `' '` — the same test as the Parakeet path,
/// reached through the tokenizer instead of a vocab table. The grouping itself
/// is `crate::decode::word_confidences`.
pub fn word_confidences(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    probs: &[f32],
) -> Vec<(String, f32)> {
    group_words(tokens.iter().zip(probs).map(|(&id, &prob)| {
        let piece = tokenizer.decode(&[id], false).unwrap_or_default();
        (Cow::Owned(piece), prob)
    }))
}

/// Cross-check the compiled-in prompt token ids against `tokenizer.json`.
///
/// The ids are model-specific constants, and a different export (or a
/// re-tokenized checkpoint) moves them. Reading them back out of the tokenizer
/// that will decode the output turns a silent garbage-in-garbage-out failure
/// into a load-time error naming both sets.
fn verify_token_ids(tokenizer: &Tokenizer) -> Result<(), WhisperError> {
    let lookup =
        |content: &str| -> Option<i64> { tokenizer.token_to_id(content).map(|id| id as i64) };
    let t_sot = lookup("<|startoftranscript|>");
    let t_transcribe = lookup("<|transcribe|>");
    let t_notimestamps = lookup("<|notimestamps|>");
    let mismatch = |t_sot: Option<i64>, t_transcribe: Option<i64>, t_notimestamps: Option<i64>| {
        WhisperError::TokenIdMismatch {
            sot: TOK_SOT,
            transcribe: TOK_TRANSCRIBE,
            notimestamps: super::types::TOK_NOTIMESTAMPS,
            t_sot: t_sot.unwrap_or(-1),
            t_transcribe: t_transcribe.unwrap_or(-1),
            t_notimestamps: t_notimestamps.unwrap_or(-1),
        }
    };
    match (t_sot, t_transcribe, t_notimestamps) {
        (Some(sot), Some(transcribe), Some(notimestamps))
            if sot == TOK_SOT
                && transcribe == TOK_TRANSCRIBE
                && notimestamps == super::types::TOK_NOTIMESTAMPS =>
        {
            Ok(())
        }
        (sot, transcribe, notimestamps) => Err(mismatch(sot, transcribe, notimestamps)),
    }
}

/// One `ort` session with sttd's fixed CPU settings.
///
/// Deliberately the same configuration `parakeet_core::engine::init_session`
/// uses (Level3 graph optimization, fixed intra/inter thread counts, no memory
/// pattern, no parallel execution): the two engines then get the same
/// treatment, and `--threads` means the same thing for both.
///
/// Every knob here was measured, and none of them is worth turning. Against a
/// 30-token utterance on a 16-core 9950X (median of three, `--threads 16`,
/// transcripts identical in every case), the session settings moved nothing:
///
/// | variant                                   | 30-token | 1-token |
/// |-------------------------------------------|---------:|--------:|
/// | this configuration                        |   1306 ms | 1077 ms |
/// | `with_memory_pattern(true)`               |   1305 ms | 1079 ms |
/// | `set_denormal_as_zero`                    |   1318 ms | 1078 ms |
/// | `decoder_with_past` at 1 / 2 / 4 threads  | 1323 / 1305 / 1306 ms | — |
///
/// That is what the profile says too: 97 % of inference is inside MLAS kernels
/// (fp32 GEMM ~50 %, int8 GEMM ~18 %, attention softmax ~9 %), so there is no
/// allocator, no arena and no sync overhead left to remove. Thread count is the
/// one knob that does move anything, and it is `--threads`: 4.6 s at 1, 2.5 s at
/// 2, 1.7 s at 4, 1.38 s at 8, 1.30 s at 16 — near-linear to 8, then flat, so
/// the default of one thread per physical core is already the right pick.
///
/// The one real lever left is the encoder's fixed 30 s window (~1.06 s of the
/// total, whatever the utterance holds), and the export pins it at
/// `input_features = [batch, 128, 3000]` — no shorter window can be fed as it
/// stands, and `with_dimension_override` would only lie about a shape the
/// graph's own reshapes depend on. Streaming pulls that lever by patching the
/// graph itself (`dyn_encoder`), not by overriding its declared shape.
fn init_session(dir: &Path, file: &str, intra_threads: usize) -> Result<Session, WhisperError> {
    let path = dir.join(file);
    if !path.exists() {
        return Err(WhisperError::MissingAsset(path.display().to_string()));
    }
    log::trace!("loading whisper session from {}...", path.display());
    Ok(session_builder(intra_threads)?.commit_from_file(path)?)
}

/// The settings every Whisper session shares (see `init_session`).
pub(super) fn session_builder(intra_threads: usize) -> Result<SessionBuilder, WhisperError> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(1)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?)
}

const ENCODER_FILE: &str = "encoder_model_int8.onnx";

/// The encoder session, and whether it is the length-free one.
///
/// With `dynamic` the exported graph is read, patched in memory
/// (`dyn_encoder::patch_encoder`) and loaded from the patched bytes. A graph the
/// patch does not recognise is not an error: the fixed export loads as before
/// and the caller learns it from the returned flag.
fn init_encoder(
    dir: &Path,
    intra_threads: usize,
    dynamic: bool,
) -> Result<(Session, bool), WhisperError> {
    if dynamic {
        let path = dir.join(ENCODER_FILE);
        if !path.exists() {
            return Err(WhisperError::MissingAsset(path.display().to_string()));
        }
        let bytes = std::fs::read(&path)
            .map_err(|e| WhisperError::MissingAsset(format!("{}: {e}", path.display())))?;
        match super::dyn_encoder::patch_encoder(bytes) {
            Ok(patched) => {
                log::info!(
                    "whisper encoder patched for variable-length windows ({} MiB)",
                    patched.len() >> 20
                );
                let session = session_builder(intra_threads)?.commit_from_memory(&patched)?;
                return Ok((session, true));
            }
            Err(e) => log::warn!(
                "cannot make the whisper encoder length-free ({e}); using the fixed 30 s window"
            ),
        }
    }
    Ok((init_session(dir, ENCODER_FILE, intra_threads)?, false))
}

/// Sanity check that the constants in `types` describe this export's token
/// layout: 100 language tokens between `<|en|>` and `<|yue|>`, the special
/// tokens above the base vocabulary, and a language range that does not run
/// into `<|transcribe|>`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_constants_match_the_export_layout() {
        assert_eq!(TOK_SOT, 50258, "<|startoftranscript|>");
        assert_eq!(TOK_EOT, 50257, "<|endoftext|>");
        assert_eq!(TOK_TRANSCRIBE, 50360, "<|transcribe|>");
        assert_eq!(
            super::super::types::TOK_NOTIMESTAMPS,
            50364,
            "<|notimestamps|>"
        );
        // Languages run <|en|>..<|yue|> inclusive, and `<|translate|>` (50359)
        // must sit outside the range or auto-detection could pick a task token.
        // No language id inside the range is compiled in: `--language` reads it
        // out of `tokenizer.json` instead.
        assert_eq!(LANG_TOKEN_FIRST, 50259);
        assert_eq!(super::super::types::LANG_TOKEN_LAST, 50358);
        assert_eq!(LANG_TOKEN_COUNT, 100);
        assert!((super::super::types::LANG_TOKEN_LAST as i64) < TOK_TRANSCRIBE);
        assert!(BEGIN_SUPPRESS_TOKENS == [220, 50256]);
    }

    #[test]
    fn token_budget_scales_with_audio_and_respects_the_cap() {
        assert_eq!(token_budget(0, 448), 20);
        assert_eq!(token_budget(16_000, 448), 35);
        assert_eq!(
            token_budget(16_001, 448),
            50,
            "a partial second counts whole"
        );
        assert_eq!(token_budget(10 * 16_000, 448), 170);
        assert_eq!(token_budget(30 * 16_000, 448), 448, "capped");
        assert_eq!(
            token_budget(2 * 16_000, 10),
            10,
            "never above the caller's cap"
        );
    }

    #[test]
    fn prompt_is_sot_lang_task_notimestamps() {
        // The token id is whatever the model resolved `--language` to, so a
        // prompt is built from a bare id: 50269 is `<|pl|>` in this export and
        // 50259 is `<|en|>`, but neither is compiled in anywhere.
        assert_eq!(
            build_prompt(50_269),
            [TOK_SOT, 50_269, TOK_TRANSCRIBE, 50_364],
            "a resolved language token goes in unchanged"
        );
        // A detected language id goes in the same way.
        assert_eq!(
            build_prompt(50_259),
            [TOK_SOT, 50_259, TOK_TRANSCRIBE, 50_364]
        );
    }

    /// `Auto` builds its prompt on the first language token, which is also what
    /// `detect_language` returns for a flat distribution — so the no-signal case
    /// does not pay for a second prompt run, and a re-run happens only when
    /// detection had real signal to disagree with.
    #[test]
    fn auto_falls_back_to_the_first_language_token() {
        assert_eq!(
            build_prompt(LANG_TOKEN_FIRST as i64),
            build_prompt(50_259),
            "LANG_TOKEN_FIRST is <|en|>"
        );
    }

    /// `detect_language` is handed the language range only — 100 ids, not the
    /// 51866-strong row it is cut from — so the test feeds it exactly that.
    #[test]
    fn language_detection_reads_the_language_token_range_only() {
        let mut range = vec![0.0f32; LANG_TOKEN_COUNT];
        range[50_269 - LANG_TOKEN_FIRST] = 10.0;
        assert_eq!(detect_language(&range), Some(50_269));
        // English wins instead.
        range[50_269 - LANG_TOKEN_FIRST] = 0.0;
        range[50_259 - LANG_TOKEN_FIRST] = 10.0;
        assert_eq!(detect_language(&range), Some(50_259));
        // A flat distribution is a tie, and `argmax_index` breaks ties toward
        // the lowest id: the first language token wins, and the prompt is not
        // re-run. That is a real outcome for a silent or noise-only window, and
        // falling back to `<|en|>` is the right way to spend one extra pass.
        assert_eq!(
            detect_language(&vec![0.0f32; LANG_TOKEN_COUNT]),
            Some(LANG_TOKEN_FIRST as i64)
        );
        // A range too short to be the language block (a vocabulary that does
        // not hold 100 of them) or a non-finite value inside it must not panic
        // or invent a language.
        assert_eq!(detect_language(&[]), None);
        assert_eq!(detect_language(&vec![0.0f32; 99]), None);
        let mut nan = vec![0.0f32; LANG_TOKEN_COUNT];
        nan[50_300 - LANG_TOKEN_FIRST] = f32::NAN;
        assert_eq!(detect_language(&nan), None);
    }

    #[test]
    fn suppression_covers_the_exported_begin_tokens() {
        assert!(is_suppressed(220));
        assert!(is_suppressed(50_256));
        assert!(!is_suppressed(0));
        assert!(!is_suppressed(50_257), "EOT must never be suppressed");
    }

    /// The opening step must not open on a suppressed id, and the fallback is
    /// the best *allowed* one, not merely the second-best overall — so a row
    /// where both suppressed ids outrank everything must still land on the
    /// best ordinary token.
    #[test]
    fn best_unsuppressed_skips_every_suppressed_id() {
        let mut vocab = vec![0.0f32; 51_866];
        vocab[220] = 9.0;
        vocab[50_256] = 8.0;
        vocab[4242] = 1.0;
        assert_eq!(argmax_index(&vocab), 220, "premise: a suppressed id wins");
        assert_eq!(best_unsuppressed(&vocab), Some(4242));
        // A flat row breaks the tie toward the lowest unsuppressed id.
        assert_eq!(best_unsuppressed(&[0.0; 4]), Some(0));
        // A lone suppressed id leaves the rest of the row as the candidates.
        assert_eq!(best_unsuppressed(&[0.0, 5.0]), Some(1));
        // An empty row has nothing to offer; the caller keeps its own argmax.
        assert_eq!(best_unsuppressed(&[]), None);
    }

    /// The downloaded model's own `tokenizer.json`, when present. The word
    /// aggregation above is the one piece of the engine that cannot be checked
    /// offline: it reads the byte-level BPE vocabulary to decide where a word
    /// starts, and no hand-written token list can stand in for the real 51866
    /// entries. Skipped when the weights are not downloaded, so `cargo test`
    /// stays runnable on a clean checkout.
    fn downloaded_tokenizer() -> Option<Tokenizer> {
        let path = crate::model_store::catalog_dir(crate::config::WHISPER_DEFAULT_SPEC)
            .join("tokenizer.json");
        if !path.exists() {
            eprintln!("skipping: {} not downloaded", path.display());
            return None;
        }
        Tokenizer::from_file(&path).ok()
    }

    /// The compiled-in prompt ids must match the tokenizer that will decode the
    /// output. This is the same check `WhisperModel::new` makes at load time,
    /// run against the real export rather than a stand-in.
    #[test]
    fn compiled_token_ids_match_the_downloaded_tokenizer() {
        let Some(tokenizer) = downloaded_tokenizer() else {
            return;
        };
        verify_token_ids(&tokenizer).expect("compiled token ids must match the export");
    }

    /// Word confidences must come out indexed by word, not by token, and the
    /// confidence of a word is its weakest token. Encodes a real phrase with
    /// the real tokenizer, then checks the pieces were grouped and not emitted
    /// one-per-token.
    #[test]
    fn word_confidences_group_subword_pieces_into_words() {
        let Some(tokenizer) = downloaded_tokenizer() else {
            return;
        };
        // A phrase long enough that whisper's BPE must split it.
        let phrase = "Dzień dobry";
        let ids = tokenizer
            .encode(phrase, false)
            .expect("phrase must encode")
            .get_ids()
            .to_vec();
        assert!(
            ids.len() > 2,
            "expected {phrase:?} to need several tokens, got {ids:?}"
        );
        // Descending probabilities, so the first token is the strongest and the
        // weakest link is a later one.
        let n = ids.len();
        let probs: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 / (2 * n) as f32).collect();
        let words = word_confidences(&tokenizer, &ids, &probs);
        assert_eq!(words.len(), 2, "two words in {phrase:?}, got {words:?}");
        let joined: String = words.iter().map(|(w, _)| w.as_str()).collect();
        assert!(
            joined.eq_ignore_ascii_case(phrase.replace(' ', "").as_str())
                || joined.to_lowercase().contains("dzie")
                || joined.to_lowercase().contains("dob"),
            "words {words:?} must reassemble into {phrase:?}"
        );
        // Each word's confidence is the minimum over its own tokens, so it is at
        // most the first token's probability and strictly inside (0, 1].
        for (word, conf) in &words {
            assert!(
                (0.0..=1.0).contains(conf),
                "word {word:?} confidence {conf} out of range"
            );
            assert!(
                *conf <= probs[0] + 1e-6,
                "word {word:?} cannot exceed its first token"
            );
        }
        // At least one word is below the strongest token: the minimum really is
        // being taken over a multi-token word.
        assert!(
            words.iter().any(|(_, c)| *c < probs[0]),
            "expected at least one word weaker than its first token"
        );
    }

    /// End-to-end for the dictionary gate: a confident word the dictionary
    /// holds a near-miss spelling of must NOT be fuzzy-corrected, and the same
    /// word reported at a low confidence must be.
    ///
    /// This is the behaviour `word_confidences` exists for, and it is the only
    /// place the two halves meet, so it is tested through
    /// `dictionary::correct_transcription` rather than by re-deriving the gate
    /// here.
    #[test]
    fn word_confidences_drive_the_dictionary_confidence_gate() {
        let Some(tokenizer) = downloaded_tokenizer() else {
            return;
        };
        let dictionary = vec!["transkrypcyi".to_string()];
        // Encode the correctly-spelled word the way a decode would: as tokens
        // carrying confidences.
        let ids = tokenizer
            .encode("transkrypcji", false)
            .expect("must encode")
            .get_ids()
            .to_vec();
        let run = |prob: f32| {
            let words = word_confidences(&tokenizer, &ids, &vec![prob; ids.len()]);
            let map = crate::dictionary::confidence_map(&words);
            crate::dictionary::correct_transcription(
                "test transkrypcji mowy",
                &dictionary,
                Some(&map),
            )
        };
        let threshold = crate::dictionary::POSTCORR_CONF_THRESHOLD;
        let confident = run(threshold);
        assert!(
            confident.contains("transkrypcji"),
            "a confident word must survive, got {confident:?}"
        );
        assert!(
            !confident.contains("transkrypcyi"),
            "the dictionary near-miss was applied to a confident word: {confident:?}"
        );
        let unsure = run(threshold - 0.01);
        assert!(
            unsure.contains("transkrypcyi"),
            "an unsure word must be corrected to the dictionary entry, got {unsure:?}"
        );
    }

    /// A decode that emits only punctuation or spaces must produce no words:
    /// the dictionary gate looks words up by name, and an empty or whitespace
    /// entry would never match anything.
    #[test]
    fn word_confidences_drop_punctuation_only_entries() {
        let Some(tokenizer) = downloaded_tokenizer() else {
            return;
        };
        for phrase in [" - ", " .", " ,"] {
            let ids = tokenizer
                .encode(phrase, false)
                .expect("must encode")
                .get_ids()
                .to_vec();
            let probs = vec![0.9; ids.len()];
            let words = word_confidences(&tokenizer, &ids, &probs);
            assert!(words.is_empty(), "phrase {phrase:?} yielded {words:?}");
        }
    }

    /// The confidences must be keyed the way the dictionary gate looks them up:
    /// normalized (lowercased, Polish diacritics folded) fragments. That
    /// mapping lives in `dictionary::confidence_map`, so this test only has to
    /// supply words in the engine's own shape — including the split that happens
    /// where BPE attached punctuation to a subword piece.
    #[test]
    fn word_confidences_feed_the_dictionary_gate_with_usable_keys() {
        let Some(tokenizer) = downloaded_tokenizer() else {
            return;
        };
        let phrase = "Dzień dobry.";
        let ids = tokenizer
            .encode(phrase, false)
            .expect("must encode")
            .get_ids()
            .to_vec();
        let words = word_confidences(&tokenizer, &ids, &vec![0.8; ids.len()]);
        let map = crate::dictionary::confidence_map(&words);
        // Both content words are present under their normalized keys: the
        // dictionary gate looks entries up that way, and an engine word that
        // does not normalize to the same key never gates a correction.
        assert_eq!(map.get("dzien").copied(), Some(0.8), "map={map:?}");
        assert_eq!(map.get("dobry").copied(), Some(0.8), "map={map:?}");
        // Exactly two keys: the trailing full stop must not become a word, and
        // an empty key would match every dictionary entry.
        assert_eq!(map.len(), 2, "only the two content words: {map:?}");
    }
}
