// ---------------------------------------------------------------------------
// ort_engine inference: encoder + TDT decoder_joint over direct `ort`
// sessions, with dictionary phrase-boosting on the greedy decode.
//
// Ported from Murmure `src-tauri/src/engine/engine.rs`
// (AGPL-3.0-or-later). Differences:
// - no `nemo128` preprocessor session: `transcribe_samples` runs the Rust
//   `frontend` (mel) and feeds the encoder directly;
// - `ParakeetModel::new` takes ORT intra/inter thread counts (sttd
//   `--threads`);
// - the `ParakeetEngine` wrapper and timestamp-granularity plumbing are
//   gone; sttd calls `ParakeetModel` directly and only needs
//   `TranscriptionResult { text, word_confidences }`.
// ---------------------------------------------------------------------------

use ndarray17::{Array, Array1, Array2, ArrayD, ArrayViewD, IxDyn};
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use regex::Regex;

use std::fs;
use std::path::Path;

use super::boost_tree::{BiasCandidate, BoostTree};
use super::frontend::{Frontend, PreprocessorConfig};
use super::helpers::{load_tokenizer, tokenize_word_to_ids, word_confidences, word_variants};
use super::transcription_engine::TranscriptionResult;
use super::types::{DecoderState, ParakeetError, ParakeetModel, TimestampedResult};

/// Tokens, frame timestamps and raw-logit probabilities of one decoded
/// sequence, kept index-aligned by the decode loops.
type DecodedSequence = (Vec<i32>, Vec<usize>, Vec<f32>);

const SUBSAMPLING_FACTOR: usize = 8;
const WINDOW_SIZE: f32 = 0.01;
const MAX_TOKENS_PER_STEP: usize = 10;

// Fusion weight bounds applied to boost scores before argmax. Alpha decays
// with dictionary size: more words means more first-tokens armed at the root,
// so we lower the volume to keep false positives down on large dictionaries.
const BOOST_ALPHA_MAX: f32 = 3.5;
const BOOST_ALPHA_MIN: f32 = 1.0;
// A dictionary token is only boosted when it already ranks within the top-K of
// the raw logits. In greedy decoding only near-misses are recoverable, so the
// gate stays tight at phrase start to avoid spurious dictionary insertions,
// then relaxes once the match is engaged (BOOST_DEEP_DEPTH tokens in): the
// prefix is strong evidence, and a continuation falling out of the tight gate
// would leave a broken boosted prefix in the output.
const BOOST_TOP_K: usize = 5;
const BOOST_TOP_K_DEEP: usize = 20;
const BOOST_DEEP_DEPTH: usize = 3;

// When boosting flips a long utterance into another language, its tokens
// diverge sharply from the unboosted decode. Above these bounds, fall back to
// the unboosted decode; short utterances are never guarded.
const GUARD_MIN_TOKENS: usize = 24;
const GUARD_DIVERGENCE: f32 = 0.35;

/// Normalized token-level edit distance between two decoded sequences, in
/// `[0, 1]`. 0 means identical, 1 means fully different.
fn token_divergence(a: &[i32], b: &[i32]) -> f32 {
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut curr = vec![i + 1];
        for (j, &y) in b.iter().enumerate() {
            curr.push(
                (prev[j + 1] + 1)
                    .min(curr[j] + 1)
                    .min(prev[j] + usize::from(x != y)),
            );
        }
        prev = curr;
    }
    prev[b.len()] as f32 / n as f32
}

fn degressive_alpha(word_count: usize) -> f32 {
    (BOOST_ALPHA_MAX - (word_count as f32 / 5.0).log10()).clamp(BOOST_ALPHA_MIN, BOOST_ALPHA_MAX)
}

fn top_k_for_depth(depth: usize) -> usize {
    if depth >= BOOST_DEEP_DEPTH {
        BOOST_TOP_K_DEEP
    } else {
        BOOST_TOP_K
    }
}

fn in_top_k(logits: &[f32], token: usize, k: usize) -> bool {
    logits
        .get(token)
        .is_some_and(|&l| l >= top_k_threshold(logits, k))
}

// Value of the k-th largest logit. A token is within the top-K of the raw
// logits iff its logit is >= this threshold.
fn top_k_threshold(logits: &[f32], k: usize) -> f32 {
    if logits.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut buf = logits.to_vec();
    let k = k.min(buf.len());
    let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    *kth
}

static DECODE_SPACE_RE: Lazy<Result<Regex, regex::Error>> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

fn softmax_prob(logits: &[f32], token: usize) -> f32 {
    let target = match logits.get(token) {
        Some(&l) => l,
        None => return 0.0,
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum > 0.0 {
        ((target - max).exp() / sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn argmax_token(logits: &[f32], blank_idx: i32) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i32)
        .unwrap_or(blank_idx)
}

/// Index of the maximum value (first on ties). Returns 0 on an empty slice.
fn argmax_index(logits: &[f32]) -> usize {
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

impl Drop for ParakeetModel {
    fn drop(&mut self) {
        log::debug!(
            "Dropping ParakeetModel with {} vocab tokens",
            self.vocab.len()
        );
    }
}

impl ParakeetModel {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        quantized: bool,
        tokenizer_path: Option<&Path>,
        intra_threads: usize,
    ) -> Result<Self, ParakeetError> {
        let threads = intra_threads.max(1);
        let encoder = Self::init_session(&model_dir, "encoder-model", threads, quantized)?;
        let decoder_joint =
            Self::init_session(&model_dir, "decoder_joint-model", threads, quantized)?;

        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;
        let vocab_size = vocab.len();

        log::trace!(
            "Loaded vocabulary with {} tokens, blank_idx={}",
            vocab_size,
            blank_idx
        );

        let tokenizer = load_tokenizer(tokenizer_path);

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
            blank_idx,
            vocab_size,
            tokenizer,
            boost_tree: None,
            boost_alpha: BOOST_ALPHA_MAX,
            boost_words: Vec::new(),
            frontend: Frontend::new(PreprocessorConfig::tdt_0_6b()),
        })
    }

    /// Rebuild the phrase-boosting automaton from the user dictionary words.
    /// Each word is expanded into casing/accent variants (see `word_variants`),
    /// every tokenizable variant becoming an independent boosted phrase.
    /// Variants that cannot be tokenized are skipped; an empty result clears it.
    /// No-op when the word set is unchanged (sync runs before every
    /// transcription).
    pub fn set_boost_words(&mut self, words: &[String]) {
        let mut sorted = words.to_vec();
        sorted.sort();
        if sorted == self.boost_words {
            return;
        }
        self.boost_words = sorted;

        let tokenizer = match self.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer,
            None => {
                self.boost_tree = None;
                return;
            }
        };
        let mut phrases: Vec<Vec<i32>> = Vec::new();
        for word in words {
            let variants = word_variants(word);
            let mut tokenized: Vec<(String, Vec<i32>)> = Vec::new();
            for variant in variants {
                if let Some(ids) = tokenize_word_to_ids(tokenizer, &variant) {
                    tokenized.push((variant, ids));
                }
            }
            if log::log_enabled!(log::Level::Debug) {
                let detail: Vec<(&str, Vec<&str>)> = tokenized
                    .iter()
                    .map(|(v, ids)| {
                        (
                            v.as_str(),
                            ids.iter().map(|&id| self.token_str(id)).collect(),
                        )
                    })
                    .collect();
                log::debug!("Boost word {:?} -> variants {:?}", word, detail);
            }
            phrases.extend(tokenized.into_iter().map(|(_, ids)| ids));
        }

        // Alpha is driven by the dictionary size, not the variant count: more
        // variants per word must not lower the boost volume.
        self.boost_tree = if phrases.is_empty() {
            None
        } else {
            self.boost_alpha = degressive_alpha(words.len());
            log::debug!(
                "Phrase boosting active for {} word(s), {} phrase(s), alpha={}",
                words.len(),
                phrases.len(),
                self.boost_alpha
            );
            Some(BoostTree::new(&phrases))
        };
    }

    fn token_str(&self, id: i32) -> &str {
        self.vocab
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("?")
    }

    // Diagnostic only: one compact line per emitted token listing the 5
    // best-ranked dictionary candidates and their raw rank (* = boosted, i.e.
    // within the top-K gate), to see where a word's tokens really stand.
    fn log_boost_step(&self, logits: &[f32], candidates: &[BiasCandidate], emitted: i32) {
        let rank_of = |tok: i32| {
            let raw = logits
                .get(tok as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            logits.iter().filter(|&&l| l > raw).count()
        };
        let mut ranked: Vec<(i32, usize, usize)> = candidates
            .iter()
            .map(|c| (c.token, rank_of(c.token), c.depth))
            .collect();
        ranked.sort_by_key(|&(_, rank, _)| rank);
        let shown: Vec<String> = ranked
            .iter()
            .take(5)
            .map(|&(tok, rank, depth)| {
                let mark = if in_top_k(logits, tok as usize, top_k_for_depth(depth)) {
                    "*"
                } else {
                    ""
                };
                format!("{:?}#{}{}", self.token_str(tok), rank, mark)
            })
            .collect();
        log::trace!(
            "boost: out={:?} | cand {}",
            self.token_str(emitted),
            shown.join(" ")
        );
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        intra_threads: usize,
        try_quantized: bool,
    ) -> Result<Session, ParakeetError> {
        // CPU is the default execution provider: no `with_execution_providers`
        // call needed (mirrors `parakeet-rs` CPU path).

        // Try quantized version first if requested, fallback to regular version
        let model_filename = if try_quantized {
            let quantized_name = format!("{}.int8.onnx", model_name);
            let quantized_path = model_dir.as_ref().join(&quantized_name);
            if quantized_path.exists() {
                log::trace!("Loading quantized model from {}...", quantized_name);
                quantized_name
            } else {
                let regular_name = format!("{}.onnx", model_name);
                log::trace!(
                    "Quantized model not found, loading regular model from {}...",
                    regular_name
                );
                regular_name
            }
        } else {
            let regular_name = format!("{}.onnx", model_name);
            log::trace!("Loading model from {}...", regular_name);
            regular_name
        };

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?
            .commit_from_file(model_dir.as_ref().join(&model_filename))?;

        Ok(session)
    }

    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, i32), ParakeetError> {
        let vocab_path = model_dir.as_ref().join("vocab.txt");
        let content = fs::read_to_string(vocab_path)?;

        let mut max_id = 0;
        let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
        let mut blank_idx: Option<usize> = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.trim_end().split(' ').collect();
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<usize>() {
                    if token == "<blk>" {
                        blank_idx = Some(id);
                    }
                    tokens_with_ids.push((token, id));
                    max_id = max_id.max(id);
                }
            }
        }

        // Create vocab vector with \u2581 replaced with space
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in tokens_with_ids {
            vocab[id] = token.replace('\u{2581}', " ");
        }

        let blank_idx = blank_idx.ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing <blk> token in vocabulary",
            ))
        })? as i32;

        Ok((vocab, blank_idx))
    }

    pub fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        log::trace!("Running encoder inference...");
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| ParakeetError::OutputNotFound("encoded_lengths".to_string()))?
            .try_extract_array()?;

        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));

        Ok((encoder_output.to_owned(), encoded_lengths.to_owned()))
    }

    pub fn create_decoder_state(&self) -> Result<DecoderState, ParakeetError> {
        // LSTM state layout is fixed for this model: 2 layers x batch 1 x
        // 640 hidden (mirrors `parakeet-rs`; no session introspection).
        Ok((Array::zeros((2, 1, 640)), Array::zeros((2, 1, 640))))
    }

    pub fn decode_step(
        &mut self,
        prev_tokens: &[i32],
        prev_state: &DecoderState,
        encoder_out: &ArrayViewD<f32>, // [time_steps, 1024]
    ) -> Result<(ArrayD<f32>, DecoderState), ParakeetError> {
        log::trace!("Running decoder inference...");

        // Get last token or blank_idx if empty
        let target_token = prev_tokens.last().copied().unwrap_or(self.blank_idx);

        // Prepare inputs matching Python: encoder_out[None, :, None] -> [1, time_steps, 1]
        let encoder_outputs = encoder_out
            .to_owned()
            .insert_axis(ndarray17::Axis(0))
            .insert_axis(ndarray17::Axis(2));
        let targets = Array2::from_shape_vec((1, 1), vec![target_token])?;
        let target_length = Array1::from_vec(vec![1]);

        let inputs = inputs![
            "encoder_outputs" => TensorRef::from_array_view(encoder_outputs.view())?,
            "targets" => TensorRef::from_array_view(targets.view())?,
            "target_length" => TensorRef::from_array_view(target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(prev_state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(prev_state.1.view())?,
        ];

        let outputs = self.decoder_joint.run(inputs)?;

        let logits = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        log::trace!(
            "Logits shape: {:?}, vocab_size: {}",
            logits.shape(),
            self.vocab_size
        );
        let state1 = outputs
            .get("output_states_1")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_1".to_string()))?
            .try_extract_array()?;
        let state2 = outputs
            .get("output_states_2")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_2".to_string()))?
            .try_extract_array()?;

        // Squeeze outputs like Python (remove batch dimension)
        let logits = logits.remove_axis(ndarray17::Axis(0));

        // Convert ArrayD back to Array3 to match expected return type
        let state1_3d = state1.to_owned().into_dimensionality::<ndarray17::Ix3>()?;
        let state2_3d = state2.to_owned().into_dimensionality::<ndarray17::Ix3>()?;

        Ok((logits.to_owned(), (state1_3d, state2_3d)))
    }

    /// Full pipeline over raw 16 kHz mono samples: Rust mel frontend,
    /// encoder, greedy (optionally boosted) TDT decode.
    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<TranscriptionResult, ParakeetError> {
        let mel = self.frontend.extract(&samples)?;
        let time_steps = mel.shape()[0];
        let feature_size = mel.shape()[1];
        if time_steps == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                word_confidences: Vec::new(),
            });
        }
        // Encoder layout is [batch, features, time].
        let audio_signal = mel
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(ParakeetError::Shape)?
            .to_owned()
            .into_dyn();
        let length = Array1::from_vec(vec![time_steps as i64]).into_dyn();

        let (encoder_out, encoder_out_lens) = self.encode(&audio_signal.view(), &length.view())?;
        let encodings_len = encoder_out_lens.iter().next().copied().unwrap_or(0).max(0) as usize;
        let mut results = Vec::new();
        for encodings in encoder_out.outer_iter() {
            let (tokens, timestamps, probs) =
                self.decode_sequence(&encodings.view(), encodings_len)?;
            results.push(self.decode_tokens(tokens, timestamps, probs));
        }
        let timestamped = results.into_iter().next().ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No transcription result returned",
            ))
        })?;
        Ok(TranscriptionResult {
            word_confidences: word_confidences(&timestamped),
            text: timestamped.text,
        })
    }

    fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>, // [time_steps, 1024]
        encodings_len: usize,
    ) -> Result<DecodedSequence, ParakeetError> {
        // Logit boosting only runs when a boost tree is active. With no boost
        // tree the greedy path must stay bit-exact (default path, streaming),
        // so it lives in its own function with no boost code on it.
        if self.boost_tree.is_some() {
            let boosted = self.decode_sequence_boosted(encodings, encodings_len)?;
            // Encoder already ran: replay the light greedy decode and use it
            // when the boost diverged too far (flipped language). A genuine
            // other-language utterance decodes the same both ways, so it stays.
            if boosted.0.len() >= GUARD_MIN_TOKENS {
                let greedy = self.decode_sequence_greedy(encodings, encodings_len)?;
                let divergence = token_divergence(&boosted.0, &greedy.0);
                if divergence > GUARD_DIVERGENCE {
                    log::debug!(
                        "boost guard: token divergence {:.2} over {} tokens, using unboosted decode",
                        divergence,
                        boosted.0.len()
                    );
                    return Ok(greedy);
                }
            }
            Ok(boosted)
        } else {
            self.decode_sequence_greedy(encodings, encodings_len)
        }
    }

    // Pure greedy TDT decode, no boosting. Bit-exact reference path.
    fn decode_sequence_greedy(
        &mut self,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
    ) -> Result<DecodedSequence, ParakeetError> {
        let mut prev_state = self.create_decoder_state()?;
        let mut tokens = Vec::new();
        let mut timestamps = Vec::new();
        let mut token_probs = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray17::s![t, ..]);
            let encoder_step_dyn = encoder_step.to_owned().into_dyn();
            let (probs, new_state) =
                self.decode_step(&tokens, &prev_state, &encoder_step_dyn.view())?;

            let (vocab_logits, dur_logits) = Self::tdt_logits(&probs, self.vocab_size)?;
            let token = argmax_token(vocab_logits, self.blank_idx);

            if token != self.blank_idx {
                token_probs.push(softmax_prob(vocab_logits, token as usize));
                prev_state = new_state;
                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
            }

            let duration = argmax_index(dur_logits);
            if duration > 0 {
                t += duration;
                emitted_tokens = 0;
            } else if token == self.blank_idx || emitted_tokens == MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok((tokens, timestamps, token_probs))
    }

    // Split a (possibly TDT) decoder output into (vocab_logits, duration_logits);
    // the duration slice is empty for a plain RNN-T joint.
    fn tdt_logits(
        probs: &ArrayD<f32>,
        vocab_size: usize,
    ) -> Result<(&[f32], &[f32]), ParakeetError> {
        let slice = probs.as_slice().ok_or_else(|| {
            ParakeetError::Shape(ndarray17::ShapeError::from_kind(
                ndarray17::ErrorKind::IncompatibleShape,
            ))
        })?;
        Ok(if slice.len() > vocab_size {
            (&slice[..vocab_size], &slice[vocab_size..])
        } else {
            (slice, &[][..])
        })
    }

    // Greedy decode with dictionary logit boosting. At each frame the boost
    // tree biases the expected dictionary tokens on the non-blank logits (gated
    // by the top-K guard) before the argmax; the boost state advances along the
    // trie with every emitted token. One decode_step per frame, no parallel
    // hypotheses.
    fn decode_sequence_boosted(
        &mut self,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
    ) -> Result<DecodedSequence, ParakeetError> {
        let tree = match self.boost_tree.take() {
            Some(tree) => tree,
            None => return self.decode_sequence_greedy(encodings, encodings_len),
        };

        let result = self.run_boosted_decode(encodings, encodings_len, &tree);
        self.boost_tree = Some(tree);
        result
    }

    fn run_boosted_decode(
        &mut self,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
        tree: &BoostTree,
    ) -> Result<DecodedSequence, ParakeetError> {
        let mut prev_state = self.create_decoder_state()?;
        let mut tokens: Vec<i32> = Vec::new();
        let mut timestamps: Vec<usize> = Vec::new();
        let mut token_probs: Vec<f32> = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut boost_state = tree.root();

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray17::s![t, ..]);
            let encoder_step_dyn = encoder_step.to_owned().into_dyn();

            let (probs, new_state) =
                self.decode_step(&tokens, &prev_state, &encoder_step_dyn.view())?;
            let (vocab_logits, dur_logits) = Self::tdt_logits(&probs, self.vocab_size)?;

            let candidates = tree.bias(boost_state);
            let mut boosted = vocab_logits.to_vec();
            if !candidates.is_empty() {
                let tight = top_k_threshold(vocab_logits, BOOST_TOP_K);
                let deep = top_k_threshold(vocab_logits, BOOST_TOP_K_DEEP);
                for cand in &candidates {
                    let idx = cand.token as usize;
                    let threshold = if top_k_for_depth(cand.depth) == BOOST_TOP_K_DEEP {
                        deep
                    } else {
                        tight
                    };
                    if cand.token != self.blank_idx
                        && idx < boosted.len()
                        && vocab_logits[idx] >= threshold
                    {
                        boosted[idx] += self.boost_alpha * cand.score;
                    }
                }
            }
            let token = argmax_token(&boosted, self.blank_idx);
            if log::log_enabled!(log::Level::Trace)
                && token != self.blank_idx
                && !candidates.is_empty()
            {
                self.log_boost_step(vocab_logits, &candidates, token);
            }

            if token != self.blank_idx {
                // Confidence from the raw logits, not the boosted ones, so
                // boosted tokens stay correctable downstream.
                token_probs.push(softmax_prob(vocab_logits, token as usize));
                prev_state = new_state;
                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
                boost_state = tree.advance(boost_state, token);
            }

            let duration = argmax_index(dur_logits);
            if duration > 0 {
                t += duration;
                emitted_tokens = 0;
            } else if token == self.blank_idx || emitted_tokens == MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok((tokens, timestamps, token_probs))
    }

    fn decode_tokens(
        &self,
        ids: Vec<i32>,
        timestamps: Vec<usize>,
        probs: Vec<f32>,
    ) -> TimestampedResult {
        // tokens and kept_probs go through the same filter to stay aligned.
        let mut tokens: Vec<String> = Vec::with_capacity(ids.len());
        let mut kept_probs: Vec<f32> = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            let idx = id as usize;
            if idx < self.vocab.len() {
                tokens.push(self.vocab[idx].clone());
                kept_probs.push(probs.get(i).copied().unwrap_or(1.0));
            }
        }

        let text = match &*DECODE_SPACE_RE {
            Ok(regex) => regex
                .replace_all(&tokens.join(""), |caps: &regex::Captures| {
                    if caps.get(1).is_some() {
                        " "
                    } else {
                        ""
                    }
                })
                .to_string(),
            Err(_) => tokens.join(""), // Fallback if regex failed to compile
        };

        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| WINDOW_SIZE * SUBSAMPLING_FACTOR as f32 * t as f32)
            .collect();

        TimestampedResult {
            text,
            timestamps: float_timestamps,
            tokens,
            probs: kept_probs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degressive_alpha_decays_with_word_count() {
        assert_eq!(degressive_alpha(5), 3.5);
        assert_eq!(degressive_alpha(50), 2.5);
        assert_eq!(degressive_alpha(500), 1.5);
        assert_eq!(degressive_alpha(50000), 1.0);
    }

    #[test]
    fn degressive_alpha_is_clamped_at_both_ends() {
        assert_eq!(degressive_alpha(1), 3.5);
        assert_eq!(degressive_alpha(5000), 1.0);
    }

    #[test]
    fn in_top_k_accepts_ranked_token_and_rejects_outranked() {
        let logits = [0.1, 5.0, 2.0, 3.0];
        assert!(in_top_k(&logits, 1, 1));
        assert!(in_top_k(&logits, 3, 2));
        assert!(!in_top_k(&logits, 0, 3));
    }

    #[test]
    fn top_k_relaxes_once_match_is_engaged() {
        assert_eq!(top_k_for_depth(1), BOOST_TOP_K);
        assert_eq!(top_k_for_depth(2), BOOST_TOP_K);
        assert_eq!(top_k_for_depth(BOOST_DEEP_DEPTH), BOOST_TOP_K_DEEP);
        assert_eq!(top_k_for_depth(10), BOOST_TOP_K_DEEP);
    }
}
