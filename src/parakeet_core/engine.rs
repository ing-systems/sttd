// ---------------------------------------------------------------------------
// parakeet_core inference: encoder + TDT decoder_joint over direct `ort`
// sessions, with dictionary phrase-boosting on the greedy decode.
//
// Ported from Murmure `src-tauri/src/engine/engine.rs`
// (AGPL-3.0-or-later). Differences:
// - no `nemo128` preprocessor session: `transcribe_samples` runs the Rust
//   `frontend` (mel) and feeds the encoder directly;
// - `ParakeetModel::new` takes ORT intra/inter thread counts (sttd
//   `--threads`);
// - the `ParakeetEngine` wrapper and the timestamp plumbing are gone; sttd
//   calls `ParakeetModel` directly and only needs the text and per-word
//   confidences (`speech::Transcript`).
// ---------------------------------------------------------------------------

use ndarray17::{Array, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, IxDyn};
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use regex::Regex;

use std::fs;
use std::path::Path;

use super::boost_tree::{BiasCandidate, BoostTree};
use super::frontend::{Frontend, PreprocessorConfig};
use super::helpers::{load_tokenizer, tokenize_word_to_ids, word_confidences, word_variants};
use super::types::{DecodedTokens, DecoderState, ParakeetError, ParakeetModel};
use crate::decode::{argmax_index, softmax_prob};
use crate::device::Device;
use crate::speech::Transcript;

/// Tokens and raw-logit probabilities of one decoded sequence, kept
/// index-aligned by the decode loops.
type DecodedSequence = (Vec<i32>, Vec<f32>);

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

const fn top_k_for_depth(depth: usize) -> usize {
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
    let kth = select_kth_largest(&mut buf, k);
    *kth
}

// The boost gates the candidates need, BOOST_TOP_K (`tight`, for depths
// below BOOST_DEEP_DEPTH) and BOOST_TOP_K_DEEP (`deep`), from a single copy of
// the logits into `scratch`. A gate no candidate needs is not computed and
// reads NEG_INFINITY: at the root every candidate is a first token, so only
// `tight` is. The second selection runs on the permuted scratch, which does not
// change the k-th largest value.
fn top_k_thresholds_with_scratch(
    logits: &[f32],
    candidates: &[BiasCandidate],
    scratch: &mut Vec<f32>,
) -> (f32, f32) {
    let needs_tight = candidates.iter().any(|c| c.depth < BOOST_DEEP_DEPTH);
    let needs_deep = candidates.iter().any(|c| c.depth >= BOOST_DEEP_DEPTH);
    if logits.is_empty() || !(needs_tight || needs_deep) {
        return (f32::NEG_INFINITY, f32::NEG_INFINITY);
    }
    scratch.clear();
    scratch.extend_from_slice(logits);
    let tight = if needs_tight {
        *select_kth_largest(scratch, BOOST_TOP_K)
    } else {
        f32::NEG_INFINITY
    };
    let deep = if needs_deep {
        *select_kth_largest(scratch, BOOST_TOP_K_DEEP)
    } else {
        f32::NEG_INFINITY
    };
    (tight, deep)
}

// Partition `buf` so that the k-th largest element is at index k - 1, and
// return a reference to it.
fn select_kth_largest(buf: &mut [f32], k: usize) -> &f32 {
    let k = k.min(buf.len());
    let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    kth
}

static DECODE_SPACE_RE: Lazy<Result<Regex, regex::Error>> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

/// Encoder output `[1, time, features]` -> decoder layout `[1, features, time]`,
/// owned and in standard (row-major) layout.
///
/// `as_standard_layout` is required, not cosmetic. The permuted view is
/// contiguous in time-major order, and `to_owned` keeps the memory order of its
/// input, so a plain `to_owned` yields strides `[1, T, features]`. Every
/// per-frame row view `s![t, ..]` would then be strided by `T`, and `ort`
/// rejects a non-contiguous tensor.
fn to_decoder_layout(encoder_output: &ArrayViewD<f32>) -> ArrayD<f32> {
    encoder_output
        .view()
        .permuted_axes(IxDyn(&[0, 2, 1]))
        .as_standard_layout()
        .into_owned()
}

/// The vocabulary piece for `id`, `?` when it is out of range (diagnostics).
fn token_str(vocab: &[String], id: i32) -> &str {
    vocab.get(id as usize).map(String::as_str).unwrap_or("?")
}

// Diagnostic only: one compact line per emitted token listing the 5
// best-ranked dictionary candidates and their raw rank (* = boosted, i.e.
// within the top-K gate), to see where a word's tokens really stand.
fn log_boost_step(vocab: &[String], logits: &[f32], candidates: &[BiasCandidate], emitted: i32) {
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
            format!("{:?}#{}{}", token_str(vocab, tok), rank, mark)
        })
        .collect();
    log::trace!(
        "boost: out={:?} | cand {}",
        token_str(vocab, emitted),
        shown.join(" ")
    );
}

/// Index of the maximum value, taking the blank id when the row is empty.
/// Unlike `crate::decode::argmax_index` this lets the *last* maximum win a
/// tie: TDT's joint pads its vocabulary with the blank id, so the highest
/// index is as valid a winner as the lowest.
fn argmax_token(logits: &[f32], blank_idx: i32) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i32)
        .unwrap_or(blank_idx)
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
        tokenizer_path: Option<&Path>,
        intra_threads: usize,
        device: Device,
    ) -> Result<Self, ParakeetError> {
        let threads = intra_threads.max(1);
        let encoder = Self::init_session(&model_dir, "encoder-model", threads, device)?;
        // The TDT joint runs once per emitted frame on tiny tensors, so it
        // always stays on CPU: CUDA transfer latency costs more than the
        // compute it saves.
        let decoder_joint =
            Self::init_session(&model_dir, "decoder_joint-model", threads, Device::Cpu)?;

        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;

        log::trace!(
            "Loaded vocabulary with {} tokens, blank_idx={}",
            vocab.len(),
            blank_idx
        );

        let tokenizer = load_tokenizer(tokenizer_path);

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
            blank_idx,
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
                            ids.iter().map(|&id| token_str(&self.vocab, id)).collect(),
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

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        intra_threads: usize,
        device: Device,
    ) -> Result<Session, ParakeetError> {
        // The int8 export when the directory has one, the fp32 one otherwise.
        let quantized_name = format!("{}.int8.onnx", model_name);
        let model_filename = if model_dir.as_ref().join(&quantized_name).exists() {
            log::trace!("Loading quantized model from {}...", quantized_name);
            quantized_name
        } else {
            let regular_name = format!("{}.onnx", model_name);
            log::trace!(
                "Quantized model not found, loading regular model from {}...",
                regular_name
            );
            regular_name
        };

        let model_path = model_dir.as_ref().join(&model_filename);
        // `device` owns the downgrade and the CPU retry, so all this has to
        // describe is where a session goes and which file it loads.
        let build = |device| -> Result<Session, ParakeetError> {
            Ok(
                crate::device::apply(crate::device::session_builder(intra_threads)?, device)
                    .commit_from_file(&model_path)?,
            )
        };
        crate::device::commit_with_cpu_fallback(device, &format!("parakeet {model_name}"), build)
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

    /// Encoder pass over `[1, features, time]` mel features. A free function
    /// over the session so the features can stay borrowed from the frontend.
    fn encode(
        encoder: &mut Session,
        audio_signal: ArrayView3<f32>,
    ) -> Result<(ArrayD<f32>, usize), ParakeetError> {
        log::trace!("Running encoder inference...");
        let length = [audio_signal.shape()[2] as i64];
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal)?,
            "length" => TensorRef::from_array_view(ArrayView1::from(&length))?,
        ];
        let outputs = encoder.run(inputs)?;

        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| ParakeetError::OutputNotFound("encoded_lengths".to_string()))?
            .try_extract_array::<i64>()?;
        let encodings_len = encoded_lengths.iter().next().copied().unwrap_or(0).max(0) as usize;

        Ok((to_decoder_layout(&encoder_output), encodings_len))
    }

    /// Full pipeline over raw 16 kHz mono samples: Rust mel frontend,
    /// encoder, greedy (optionally boosted) TDT decode.
    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<Transcript, ParakeetError> {
        // The frontend already returns the encoder layout [1, features, time].
        let audio_signal = self.frontend.extract(samples)?;
        if audio_signal.shape()[2] == 0 {
            return Ok(Transcript {
                text: String::new(),
                word_confidences: Vec::new(),
            });
        }
        let (encoder_out, encodings_len) = Self::encode(&mut self.encoder, audio_signal)?;
        let mut results = Vec::new();
        for encodings in encoder_out.outer_iter() {
            let (tokens, probs) = self.decode_sequence(&encodings.view(), encodings_len)?;
            results.push(self.decode_tokens(tokens, probs));
        }
        let decoded = results.into_iter().next().ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No transcription result returned",
            ))
        })?;
        Ok(Transcript {
            word_confidences: word_confidences(&decoded),
            text: decoded.text,
        })
    }

    fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>, // [time_steps, 1024]
        encodings_len: usize,
    ) -> Result<DecodedSequence, ParakeetError> {
        let mut joint = Joint {
            session: &mut self.decoder_joint,
            vocab: &self.vocab,
            blank_idx: self.blank_idx,
        };
        // Logit boosting only runs when a boost tree is active. With no boost
        // tree the decode is plain greedy and bit-exact (default path,
        // streaming): `Joint::decode` runs no boost code without one.
        let Some(tree) = &self.boost_tree else {
            return joint.decode(encodings, encodings_len, None);
        };
        let boosted = joint.decode(encodings, encodings_len, Some((tree, self.boost_alpha)))?;
        // Encoder already ran: replay the light greedy decode and use it when
        // the boost diverged too far (flipped language). A genuine
        // other-language utterance decodes the same both ways, so it stays.
        if boosted.0.len() >= GUARD_MIN_TOKENS {
            let greedy = joint.decode(encodings, encodings_len, None)?;
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
    }

    fn decode_tokens(&self, ids: Vec<i32>, probs: Vec<f32>) -> DecodedTokens {
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

        DecodedTokens {
            text,
            tokens,
            probs: kept_probs,
        }
    }
}

/// The TDT decoder_joint session with what its decode loop reads: borrowed
/// field by field from `ParakeetModel`, so the boost tree (another field) can
/// stay borrowed alongside it.
struct Joint<'a> {
    session: &'a mut Session,
    vocab: &'a [String],
    blank_idx: i32,
}

impl Joint<'_> {
    /// Greedy TDT decode over the first `encodings_len` encoder frames,
    /// optionally with dictionary logit boosting (`boost`: the tree and its
    /// fusion weight).
    ///
    /// With a boost tree, at each frame the tree biases the expected
    /// dictionary tokens on the non-blank logits (gated by the top-K guard)
    /// before the argmax, and the boost state advances along the trie with
    /// every emitted token. One joint run per frame, no parallel hypotheses.
    /// Without one, the argmax reads the raw logits directly.
    fn decode(
        &mut self,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
        boost: Option<(&BoostTree, f32)>,
    ) -> Result<DecodedSequence, ParakeetError> {
        // LSTM state layout is fixed for this model: 2 layers x batch 1 x
        // 640 hidden (mirrors `parakeet-rs`; no session introspection).
        let mut state: DecoderState = (Array::zeros((2, 1, 640)), Array::zeros((2, 1, 640)));
        let mut tokens: Vec<i32> = Vec::new();
        let mut token_probs: Vec<f32> = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        // Unused without a boost tree.
        let mut boost_node = boost.map_or(0, |(tree, _)| tree.root());

        // Reused across steps so the decode loop does not allocate per frame.
        let mut candidates: Vec<BiasCandidate> = Vec::new();
        let mut boosted: Vec<f32> = Vec::new();
        let mut topk_scratch: Vec<f32> = Vec::new();

        while t < encodings_len {
            // A row view of the encoder output: contiguous, so the decoder
            // reads it without a per-step copy. encoder_out[None, :, None]
            // -> [1, 1024, 1] is a stride-only reshape of it.
            let encoder_step = encodings.slice(ndarray17::s![t, ..]);
            debug_assert!(
                encoder_step.as_slice().is_some(),
                "the joint needs a contiguous encoder frame"
            );
            let encoder_outputs = encoder_step
                .insert_axis(ndarray17::Axis(0))
                .insert_axis(ndarray17::Axis(2));
            let target = [tokens.last().copied().unwrap_or(self.blank_idx)];
            let target_length = [1i32];
            let outputs = self.session.run(inputs![
                "encoder_outputs" => TensorRef::from_array_view(encoder_outputs)?,
                "targets" => TensorRef::from_array_view(ArrayView2::from_shape((1, 1), &target)?)?,
                "target_length" => TensorRef::from_array_view(ArrayView1::from(&target_length))?,
                "input_states_1" => TensorRef::from_array_view(state.0.view())?,
                "input_states_2" => TensorRef::from_array_view(state.1.view())?,
            ])?;
            let logits = output(&outputs, "outputs")?;
            let (vocab_logits, dur_logits) = tdt_logits(&logits, self.vocab.len())?;

            let token = match boost {
                None => argmax_token(vocab_logits, self.blank_idx),
                Some((tree, alpha)) => {
                    tree.bias_into(boost_node, &mut candidates);
                    if boosted.len() != vocab_logits.len() {
                        boosted.resize(vocab_logits.len(), 0.0);
                    }
                    boosted.copy_from_slice(vocab_logits);
                    if !candidates.is_empty() {
                        let (tight, deep) = top_k_thresholds_with_scratch(
                            vocab_logits,
                            &candidates,
                            &mut topk_scratch,
                        );
                        for cand in &candidates {
                            let idx = cand.token as usize;
                            let threshold = if cand.depth >= BOOST_DEEP_DEPTH {
                                deep
                            } else {
                                tight
                            };
                            if cand.token != self.blank_idx
                                && idx < boosted.len()
                                && vocab_logits[idx] >= threshold
                            {
                                boosted[idx] += alpha * cand.score;
                            }
                        }
                    }
                    let token = argmax_token(&boosted, self.blank_idx);
                    if log::log_enabled!(log::Level::Trace)
                        && token != self.blank_idx
                        && !candidates.is_empty()
                    {
                        log_boost_step(self.vocab, vocab_logits, &candidates, token);
                    }
                    token
                }
            };

            if token != self.blank_idx {
                // Confidence from the raw logits, not the boosted ones, so
                // boosted tokens stay correctable downstream.
                token_probs.push(softmax_prob(vocab_logits, token as usize));
                // The LSTM state only advances on an emitted token, so it is
                // copied out of the outputs only then (blank frames are the
                // majority), and into the arrays already held.
                state.0.assign(
                    &output(&outputs, "output_states_1")?
                        .into_dimensionality::<ndarray17::Ix3>()?,
                );
                state.1.assign(
                    &output(&outputs, "output_states_2")?
                        .into_dimensionality::<ndarray17::Ix3>()?,
                );
                tokens.push(token);
                emitted_tokens += 1;
                if let Some((tree, _)) = boost {
                    boost_node = tree.advance(boost_node, token);
                }
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

        Ok((tokens, token_probs))
    }
}

/// One named f32 output of a session run, viewed in place.
fn output<'a>(
    outputs: &'a ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<ArrayViewD<'a, f32>, ParakeetError> {
    Ok(outputs
        .get(name)
        .ok_or_else(|| ParakeetError::OutputNotFound(name.to_string()))?
        .try_extract_array()?)
}

// Split a (possibly TDT) joint output into (vocab_logits, duration_logits);
// the duration slice is empty for a plain RNN-T joint.
fn tdt_logits<'a>(
    logits: &'a ArrayViewD<'_, f32>,
    vocab_size: usize,
) -> Result<(&'a [f32], &'a [f32]), ParakeetError> {
    let slice = logits.as_slice().ok_or_else(|| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_thresholds_compute_only_the_gates_candidates_need() {
        let logits: Vec<f32> = (0..40).map(|i| ((i * 7) % 40) as f32).collect();
        let cand = |depth| BiasCandidate {
            token: 0,
            score: 1.0,
            depth,
        };
        let mut scratch = Vec::new();
        // The 5th and 20th largest of 0..40.
        let (tight, deep) = (35.0, 20.0);
        assert_eq!(
            top_k_thresholds_with_scratch(
                &logits,
                &[cand(1), cand(BOOST_DEEP_DEPTH)],
                &mut scratch
            ),
            (tight, deep)
        );
        assert_eq!(
            top_k_thresholds_with_scratch(&logits, &[cand(1)], &mut scratch),
            (tight, f32::NEG_INFINITY)
        );
        assert_eq!(
            top_k_thresholds_with_scratch(&logits, &[cand(BOOST_DEEP_DEPTH)], &mut scratch),
            (f32::NEG_INFINITY, deep)
        );
    }

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

    // Regression: `to_owned` keeps the memory order of its input, so the
    // permuted encoder output came out time-major with strides [1, T, F].
    // Every `s![t, ..]` frame view was then strided by `T`, and `ort` rejected
    // the `[1, F, 1]` decoder input as non-contiguous.
    #[test]
    fn decoder_layout_is_standard_so_every_frame_is_contiguous() {
        let time = 5;
        let features = 8;
        // [1, time, features], the shape the encoder session returns.
        let encoder_output = Array::from_shape_vec(
            (1, time, features),
            (0..time * features).map(|v| v as f32).collect(),
        )
        .unwrap();
        let decoder = to_decoder_layout(&encoder_output.view().into_dyn());
        assert_eq!(decoder.shape(), &[1, features, time]);
        assert!(
            decoder.is_standard_layout(),
            "decoder layout must be row-major, got strides {:?}",
            decoder.strides()
        );
        // Follow the real decode path: `transcribe_samples` iterates the batch
        // axis, then each loop slices one time step out of the [time, features]
        // view. Every such frame must be contiguous.
        for encodings in decoder.outer_iter() {
            for t in 0..time {
                let frame = encodings.slice(ndarray17::s![t, ..]).into_dyn();
                assert!(
                    frame.as_slice().is_some(),
                    "frame {t} must be contiguous, got strides {:?}",
                    frame.strides()
                );
            }
        }
    }

    #[test]
    fn decoder_layout_keeps_the_encoder_values() {
        let encoder_output =
            Array::from_shape_vec((1, 3, 4), (0..12).map(|v| v as f32).collect()).unwrap();
        let decoder = to_decoder_layout(&encoder_output.view().into_dyn());
        for t in 0..3 {
            for f in 0..4 {
                assert_eq!(decoder[[0, f, t]], encoder_output[[0, t, f]]);
            }
        }
    }

    // The decoder feeds `encoder_outputs` straight from the frame view, so the
    // `[1, F, 1]` shape built with `insert_axis` must stay standard layout.
    #[test]
    fn decoder_step_input_stays_standard_layout() {
        let features = 1024;
        let frame = ndarray17::Array1::<f32>::zeros(features);
        let encoder_outputs = frame
            .view()
            .into_dyn()
            .insert_axis(ndarray17::Axis(0))
            .insert_axis(ndarray17::Axis(2));
        assert_eq!(encoder_outputs.shape(), &[1, features, 1]);
        assert!(
            encoder_outputs.is_standard_layout(),
            "insert_axis must not break contiguity, got strides {:?}",
            encoder_outputs.strides()
        );
    }
}
