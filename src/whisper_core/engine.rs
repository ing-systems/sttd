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

use std::path::Path;

use ndarray17::{Array2, ArrayD};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use super::types::{
    Language, KvCache, MelConfig, WhisperError, WhisperModel, BEGIN_SUPPRESS_TOKENS,
    LANG_TOKEN_FIRST, LANG_TOKEN_LAST, TOK_EOT, TOK_SOT, TOK_TRANSCRIBE,
};

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

/// Argmax over `logits`, with ties going to the lowest id. Returns the
/// position in `logits` of the winning token.
pub(crate) fn argmax_index(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Softmax probability of one token over `logits`, via the max-subtraction
/// trick. Out-of-range ids score 0 rather than panicking: a wrong id here
/// means a broken export, and the confidence then reads as "no confidence"
/// for the dictionary gate instead of taking the process down.
pub(crate) fn token_probability(logits: &[f32], token: usize) -> f32 {
    let Some(&target) = logits.get(token) else {
        return 0.0;
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum > 0.0 {
        ((target - max).exp() / sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The prompt for one inference: start-of-transcript, a language token, the
/// transcribe task, and no timestamps.
///
/// `language_token` is a token id, not a `Language`: for `Auto` the caller
/// passes the fallback and, once the first prompt run has reported what it
/// detected, the id that was actually detected (see `transcribe_samples`).
pub(crate) fn build_prompt(language_token: i64) -> Vec<i64> {
    vec![
        TOK_SOT,
        language_token,
        TOK_TRANSCRIBE,
        super::types::TOK_NOTIMESTAMPS,
    ]
}

/// Language id chosen by `Auto`, from the first position's logits over the
/// language-token range.
///
/// `None` only when there is no usable slice (too few logits, or a non-finite
/// value in range), in which case the caller's fallback language stands. A
/// *flat* slice is not such a case: it is a tie, and ties break toward the
/// lowest id, which for a silent or noise-only window spends one extra prompt
/// pass to land on `<|en|>`. Rejecting ties outright would leave the fallback
/// (`<|pl|>`) in charge, and neither answer is better — the transcript gate
/// decides whether anything is emitted.
pub(crate) fn detect_language(logits: &[f32]) -> Option<i64> {
    let slice = logits.get(LANG_TOKEN_FIRST..=LANG_TOKEN_LAST)?;
    if !slice.iter().all(|v| v.is_finite()) {
        return None;
    }
    let win = LANG_TOKEN_FIRST + argmax_index(slice);
    Some(win as i64)
}

/// Mask `token` out of the winning set on the first generated step
/// (`begin_suppress_tokens`). Applied by setting the logit below every
/// competing logit rather than by skipping the id, so a step whose argmax is
/// exactly the suppressed token still lands on the runner-up.
fn is_suppressed(token: usize) -> bool {
    BEGIN_SUPPRESS_TOKENS.contains(&token)
}

impl WhisperModel {
    /// Load the three graphs and the tokenizer from `model_dir` (already
    /// guaranteed to hold them by `model_store::ensure_available`).
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        language: Language,
        intra_threads: usize,
    ) -> Result<Self, WhisperError> {
        let dir = model_dir.as_ref();
        let threads = intra_threads.max(1);
        let encoder = init_session(dir, "encoder_model_int8.onnx", threads)?;
        let decoder = init_session(dir, "decoder_model_int8.onnx", threads)?;
        let decoder_with_past = init_session(dir, "decoder_with_past_model_int8.onnx", threads)?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            WhisperError::Tokenizer(format!("{}: {e}", tokenizer_path.display()))
        })?;
        verify_token_ids(&tokenizer)?;

        Ok(Self {
            encoder,
            decoder,
            decoder_with_past,
            tokenizer,
            language,
            frontend: super::frontend::Frontend::new(MelConfig::large_v3_turbo()),
        })
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
        let mel = self.frontend.extract(pcm)?;
        let hidden = self.run_encoder(&mel)?;

        // Prompt run. For `--language auto` the language distribution is the
        // first position's logits, which this very call produces, so detection
        // is free unless the fallback guess was wrong — then the prompt is
        // re-run once with what was detected.
        let mut lang_token = self.language.prompt_token();
        let mut first = self.run_prompt(&hidden, lang_token)?;
        if self.language == Language::Auto {
            if let Some(detected) = detect_language(&first.first) {
                if detected != lang_token {
                    log::debug!("whisper detected language id {detected}, re-running the prompt");
                    lang_token = detected;
                    first = self.run_prompt(&hidden, lang_token)?;
                }
            }
            log::debug!("whisper decoded with language token id {lang_token}");
        }
        // `first` has served its purpose (language detection); the decode
        // starts from the last prompt position.
        let PromptRun { mut cache, next, .. } = first;

        let mut logits = next;
        let mut tokens: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut truncated = false;

        // The loop runs at most `max_tokens` steps but issues one *fewer*
        // `decoder_with_past_model` call than that: the last token needs no
        // successor logits, since there is no step left to use them in.
        let mut step = 0;
        while step < max_tokens {
            let mut next = argmax_index(&logits);
            if step == 0 && is_suppressed(next) {
                // The winning id is suppressed on the opening step: take the
                // best one that is not. Sorting the whole vocab is wasteful
                // here, but it happens at most once per utterance.
                let mut ranked: Vec<usize> = (0..logits.len()).collect();
                ranked.sort_unstable_by(|&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                next = ranked
                    .into_iter()
                    .find(|&i| !is_suppressed(i) && i != next)
                    .unwrap_or(next);
            }
            if next as i64 == TOK_EOT {
                break;
            }
            tokens.push(next as u32);
            probs.push(token_probability(&logits, next));
            step += 1;
            if step >= max_tokens {
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

    fn run_encoder(&mut self, mel: &ndarray17::Array3<f32>) -> Result<ArrayD<f32>, WhisperError> {
        debug_assert!(
            mel.as_slice().is_some(),
            "encoder input must be contiguous"
        );
        log::trace!("running whisper encoder...");
        let outputs = self.encoder.run(inputs![
            "input_features" => TensorRef::from_array_view(mel.view())?,
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
        let ids = Array2::from_shape_vec((1, prompt.len()), prompt)?;
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
            .ok_or(WhisperError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;
        let positions = logits.shape()[1];
        let row = |i: usize| -> Vec<f32> {
            logits
                .index_axis(ndarray17::Axis(0), 0)
                .index_axis(ndarray17::Axis(0), i)
                .iter()
                .copied()
                .collect()
        };
        Ok(PromptRun {
            cache,
            first: row(0),
            next: row(positions - 1),
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
        let mut named: Vec<(String, ort::session::SessionInputValue<'_>)> = Vec::with_capacity(17);
        named.push((
            "input_ids".to_string(),
            TensorRef::from_array_view(ids.view())?.into(),
        ));
        named.extend(cache.named_inputs());
        let outputs = self.decoder_with_past.run(named)?;
        cache.update_decoder(&outputs)?;
        let logits = outputs
            .get("logits")
            .ok_or(WhisperError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;
        // `[1, 1, vocab]` for a single token.
        Ok(logits
            .index_axis(ndarray17::Axis(0), 0)
            .index_axis(ndarray17::Axis(0), 0)
            .iter()
            .copied()
            .collect())
    }
}

/// What the prompt run leaves behind: the KV cache to feed the cached steps,
/// and the two logits rows the decode reads — position 0 (the language
/// distribution) and the last position (the first generated token).
///
/// Fully owned so the caller can keep using the sessions afterwards.
struct PromptRun {
    cache: KvCache,
    first: Vec<f32>,
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
/// `parakeet_core::helpers::word_confidences`): a word opens on a token that
/// starts with a space, continues on one that does not, and its confidence is
/// the weakest token probability among the ones carrying alphanumerics.
///
/// Whisper's tokens are byte-level BPE, so "starts with a space" is the
/// decoded piece starting with `' '` — the same test as the Parakeet path,
/// reached through the tokenizer instead of a vocab table.
pub fn word_confidences(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    probs: &[f32],
) -> Vec<(String, f32)> {
    let mut words: Vec<(String, f32)> = Vec::new();
    for (&id, &prob) in tokens.iter().zip(probs) {
        let piece = tokenizer.decode(&[id], false).unwrap_or_default();
        let opens_word = piece.starts_with(' ');
        let has_content = piece.chars().any(|c| c.is_alphanumeric());
        if opens_word || words.is_empty() {
            words.push((
                piece.trim_start().to_string(),
                if has_content { prob } else { 1.0 },
            ));
        } else if let Some((text, conf)) = words.last_mut() {
            text.push_str(&piece);
            if has_content {
                *conf = conf.min(prob);
            }
        }
    }
    words.retain(|(text, _)| text.chars().any(|c| c.is_alphanumeric()));
    words
}

/// Cross-check the compiled-in prompt token ids against `tokenizer.json`.
///
/// The ids are model-specific constants, and a different export (or a
/// re-tokenized checkpoint) moves them. Reading them back out of the tokenizer
/// that will decode the output turns a silent garbage-in-garbage-out failure
/// into a load-time error naming both sets.
fn verify_token_ids(tokenizer: &Tokenizer) -> Result<(), WhisperError> {
    let lookup = |content: &str| -> Option<i64> {
        tokenizer
            .token_to_id(content)
            .map(|id| id as i64)
    };
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
fn init_session(dir: &Path, file: &str, intra_threads: usize) -> Result<Session, WhisperError> {
    let path = dir.join(file);
    if !path.exists() {
        return Err(WhisperError::MissingAsset(path.display().to_string()));
    }
    log::trace!("loading whisper session from {}...", path.display());
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(1)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .commit_from_file(path)?)
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
        assert_eq!(super::super::types::TOK_NOTIMESTAMPS, 50364, "<|notimestamps|>");
        assert_eq!(super::super::types::TOK_LANG_PL, 50269, "<|pl|>");
        assert_eq!(super::super::types::TOK_LANG_EN, 50259, "<|en|>");
        // Languages run <|en|>..<|yue|> inclusive, and `<|translate|>` (50359)
        // must sit outside the range or auto-detection could pick a task token.
        assert_eq!(LANG_TOKEN_FIRST, 50259);
        assert_eq!(LANG_TOKEN_LAST, 50358);
        assert_eq!(LANG_TOKEN_LAST - LANG_TOKEN_FIRST + 1, 100);
        assert!((LANG_TOKEN_LAST as i64) < TOK_TRANSCRIBE);
        assert!(BEGIN_SUPPRESS_TOKENS == [220, 50256]);
    }

    #[test]
    fn prompt_is_sot_lang_task_notimestamps() {
        assert_eq!(
            build_prompt(Language::Polish.prompt_token()),
            vec![TOK_SOT, 50269, TOK_TRANSCRIBE, 50364]
        );
        assert_eq!(
            build_prompt(Language::English.prompt_token()),
            vec![TOK_SOT, 50259, TOK_TRANSCRIBE, 50364]
        );
        // Auto builds the Polish prompt as a fallback; detection supplies the
        // token id for the re-run.
        assert_eq!(
            build_prompt(Language::Auto.prompt_token()),
            build_prompt(Language::Polish.prompt_token())
        );
        assert_eq!(
            build_prompt(50_259),
            vec![TOK_SOT, 50_259, TOK_TRANSCRIBE, 50364],
            "a detected language id goes in unchanged"
        );
    }

    #[test]
    fn argmax_picks_the_highest_logit_and_the_lowest_id_on_a_tie() {
        assert_eq!(argmax_index(&[0.0, 1.0, 2.0]), 2);
        assert_eq!(argmax_index(&[2.0, 1.0, 2.0]), 0, "ties go to the lowest id");
        assert_eq!(argmax_index(&[f32::NEG_INFINITY; 4]), 0);
        assert_eq!(argmax_index(&[]), 0, "empty logits must not panic");
    }

    #[test]
    fn token_probability_is_the_softmax_share() {
        // Two equal logits: each is half the mass.
        let p = token_probability(&[0.0, 0.0], 0);
        assert!((p - 0.5).abs() < 1e-6, "p={p}");
        // A logit 20 above the rest is essentially all the mass.
        assert!(token_probability(&[20.0, 0.0], 0) > 0.999);
        assert!(token_probability(&[20.0, 0.0], 1) < 1e-6);
        // Out of range and empty are 0, not a panic.
        assert_eq!(token_probability(&[1.0], 5), 0.0);
        assert_eq!(token_probability(&[], 0), 0.0);
        // Sum to 1 across the vocab.
        let logits = [1.0, 2.0, 3.0, 0.5];
        let total: f32 = (0..logits.len())
            .map(|i| token_probability(&logits, i))
            .sum();
        assert!((total - 1.0).abs() < 1e-5, "total={total}");
    }

    #[test]
    fn language_detection_reads_the_language_token_range_only() {
        // 51866 logits, with Polish (50269) winning.
        let mut logits = vec![0.0f32; 51_866];
        logits[50_269] = 10.0;
        // Something huge outside the language range must be ignored.
        logits[40_000] = 100.0;
        logits[50_360] = 50.0;
        assert_eq!(detect_language(&logits), Some(50_269));
        // English wins instead.
        logits[50_269] = 0.0;
        logits[50_259] = 10.0;
        assert_eq!(detect_language(&logits), Some(50_259));
        // A flat distribution is a tie, and `argmax_index` breaks ties toward
        // the lowest id: the first language token wins, and the prompt is not
        // re-run. That is a real outcome for a silent or noise-only window, and
        // falling back to `<|en|>` is the right way to spend one extra pass.
        let flat = vec![0.0f32; 51_866];
        assert_eq!(detect_language(&flat), Some(LANG_TOKEN_FIRST as i64));
        // Short or non-finite logits must not panic or invent a language.
        assert_eq!(detect_language(&[]), None);
        assert_eq!(detect_language(&[0.0; 100]), None);
        let mut nan = vec![0.0f32; 51_866];
        nan[50_300] = f32::NAN;
        assert_eq!(detect_language(&nan), None);
    }

    #[test]
    fn suppression_covers_the_exported_begin_tokens() {
        assert!(is_suppressed(220));
        assert!(is_suppressed(50_256));
        assert!(!is_suppressed(0));
        assert!(!is_suppressed(50_257), "EOT must never be suppressed");
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
            let corrected = crate::dictionary::correct_transcription(
                "test transkrypcji mowy",
                &dictionary,
                Some(&map),
            );
            corrected
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
