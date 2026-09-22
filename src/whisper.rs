// ---------------------------------------------------------------------------
// Whisper engine: the whisper.cpp `SpeechModel` implementation.
// ---------------------------------------------------------------------------
//
// Everything engine-agnostic (the worker loop, gates, metrics, conditioning,
// prompt-context tail) lives in `asr`; this module owns only what whisper.cpp
// itself needs: `Language`/decode parameters, `FullParams` construction,
// token-confidence scoring, and the `WhisperModel` struct that implements
// `asr::SpeechModel`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::asr::{with_loading_ticker, AsrOptions, SpeechModel, TokenConfidence, Transcript};
use crate::config::WHISPER_MIN_SAMPLES;

/// Whisper decoding language, chosen once at startup via `STTD_LANGUAGE`
/// and fixed for the process's lifetime. One inference decodes in exactly
/// one language — but different utterances may use different languages when
/// `Auto` lets whisper.cpp detect it per utterance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Language {
    /// Force Polish decoding (`set_language(Some("pl"))`).
    #[value(name = "pl")]
    Polish,
    /// Force English decoding (`set_language(Some("en"))`).
    #[value(name = "en")]
    English,
    /// Per-utterance auto-detection (`set_language(None)`).
    #[value(name = "auto")]
    Auto,
}

impl Language {
    /// Whisper language code for `FullParams::set_language`, or `None` for
    /// auto-detection.
    pub(crate) fn as_whisper_code(self) -> Option<&'static str> {
        match self {
            Self::Polish => Some("pl"),
            Self::English => Some("en"),
            Self::Auto => None,
        }
    }

    /// Stable short spelling for log lines.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Polish => "pl",
            Self::English => "en",
            Self::Auto => "auto",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A loaded whisper.cpp model, implementing `asr::SpeechModel`. Decode
/// parameters are captured from `AsrOptions` once at load time (the trait's
/// `transcribe` signature is engine-agnostic, so there's nowhere else to
/// carry them).
pub(crate) struct WhisperModel {
    /// Kept alive alongside `state` so both drop together on unload,
    /// mirroring the two-local scope (`ctx`, `state`) this struct replaces.
    /// `WhisperState` holds its own `Arc<WhisperInnerContext>` clone
    /// internally, so nothing actually depends on drop order — but keeping
    /// both together is the least surprising shape.
    #[allow(dead_code)]
    ctx: WhisperContext,
    state: whisper_rs::WhisperState,
    token_beg: whisper_rs::WhisperToken,
    language: Language,
    translate: bool,
    debug_tokens: bool,
    beam_size: i32,
    temperature: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    n_threads: i32,
    model_spec: String,
    ready_line: String,
}

impl WhisperModel {
    /// Load a whisper.cpp model from `model_path` (a `ggml-*.bin` file,
    /// already guaranteed present by `weights::ensure_available`) and every
    /// decode parameter `AsrOptions` carries.
    pub(crate) fn load(model_path: &Path, opts: &AsrOptions) -> Result<Self> {
        // Belt-and-braces: `main` already installs this in `init_logging`,
        // but this is the actual whisper.cpp user, so ensure the redirect
        // even for entry points/tests that skip `init_logging`.
        // `Once`-guarded, no-op after first.
        whisper_rs::install_whisper_log_trampoline();
        let path_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let ctx = load_whisper_context(path_str, model_path, opts.use_gpu)?;
        let state = ctx.create_state().context("failed to create whisper state")?;
        let token_beg = ctx.token_beg();
        let ready_line = format!(
            "whisper ready (model={}, lang={}, translate={}, beam={}, temp={}/{}, \
             entropy_thold={}, logprob_thold={}, threads={}, gpu={})",
            opts.model_spec,
            opts.language.as_str(),
            opts.translate,
            opts.beam_size,
            opts.temperature,
            opts.temperature_inc,
            opts.entropy_thold,
            opts.logprob_thold,
            opts.n_threads,
            opts.use_gpu,
        );
        Ok(Self {
            ctx,
            state,
            token_beg,
            language: opts.language,
            translate: opts.translate,
            debug_tokens: opts.debug_tokens,
            beam_size: opts.beam_size,
            temperature: opts.temperature,
            temperature_inc: opts.temperature_inc,
            entropy_thold: opts.entropy_thold,
            logprob_thold: opts.logprob_thold,
            n_threads: opts.n_threads,
            model_spec: opts.model_spec.clone(),
            ready_line,
        })
    }
}

impl SpeechModel for WhisperModel {
    fn transcribe(
        &mut self,
        pcm_16k: &[f32],
        prompt: &str,
        running: &Arc<AtomicBool>,
    ) -> Result<Transcript> {
        let t = transcribe(
            &mut self.state,
            self.token_beg,
            pcm_16k,
            prompt,
            self.language,
            self.translate,
            self.debug_tokens,
            self.beam_size,
            self.temperature,
            self.temperature_inc,
            self.entropy_thold,
            self.logprob_thold,
            self.n_threads,
            running,
        )?;
        Ok(Transcript {
            text: t.text,
            confidence: Some(t.confidence),
        })
    }

    fn uses_prompt(&self) -> bool {
        // Translate mode decodes standalone: feeding back prior outputs as
        // `initial_prompt` context would bias the next translation
        // (translated English, or a source-language leak, as prior), so
        // only transcribe mode grows the conversational tail.
        !self.translate
    }

    fn describe(&self) -> String {
        self.ready_line.clone()
    }

    fn describe_reload(&self) -> String {
        format!("whisper ready (model={}, reloaded after pause)", self.model_spec)
    }

    fn engine_name(&self) -> &'static str {
        "whisper"
    }

    fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
        decoded_sample_count(conditioned_len)
    }
}

/// Blocking model load with a per-second progress tick (large models take
/// tens of seconds with no callback). Returns only the context; the caller
/// creates its state separately.
fn load_whisper_context(
    path_str: &str,
    model_path: &Path,
    use_gpu: bool,
) -> Result<WhisperContext> {
    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu = use_gpu;
    let backend = if use_gpu { "gpu requested" } else { "cpu" };
    match std::fs::metadata(model_path) {
        Ok(meta) => log::info!(
            "loading whisper context ({:.1} MiB, {backend})...",
            meta.len() as f64 / 1_048_576.0
        ),
        Err(_) => log::info!("loading whisper context ({backend})..."),
    }
    with_loading_ticker("whisper context", || {
        WhisperContext::new_with_params(path_str, ctx_params)
            .context("failed to load whisper model")
    })
}

/// Samples whisper actually decoded for a conditioned utterance of
/// `conditioned_len` samples: `transcribe` pads anything under
/// `WHISPER_MIN_SAMPLES` (see `pad_with_dither`) before inference, so the
/// un-padded length alone understates a short utterance's actual decoded
/// audio — inflating `rtf` and the `--metrics` `audio_sec` column by up to
/// ~1.7x on exactly the sub-second utterances confidence-floor tuning cares
/// about most. Pure for tests.
pub(crate) fn decoded_sample_count(conditioned_len: usize) -> usize {
    conditioned_len.max(WHISPER_MIN_SAMPLES)
}

/// Result of one whisper inference: the joined segment text plus its
/// `TokenConfidence` (see `asr::TokenConfidence`). Distinct from
/// `asr::Transcript` (whose `confidence` is `Option<TokenConfidence>`,
/// `None` for engines without per-token probabilities) — this one always
/// has a confidence, so `WhisperModel::transcribe` wraps it in `Some` at
/// the trait boundary.
struct WhisperTranscript {
    text: String,
    confidence: TokenConfidence,
}

/// One token decoded by whisper, scored for confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoredToken {
    pub(crate) id: whisper_rs::WhisperToken,
    pub(crate) p: f32,
    pub(crate) plog: f32,
    /// Whether this token contains alphanumeric characters (spoken text).
    /// Pure punctuation tokens (e.g. trailing '.', ',', '?') are emitted by
    /// whisper's LM and carry low LM probabilities on isolated words, dragging
    /// short-utterance confidence down; excluding them scores the spoken words themselves.
    pub(crate) is_lexical: bool,
}

/// Average token probability/log-probability over *text* tokens only, given
/// each token's `ScoredToken`. Timestamp tokens (`id >= token_beg`) sit in
/// every whisper.cpp segment — a leading `<|0.00|>` and a closing `<|t|>`
/// (confirmed in the vendored whisper.cpp 1.7.1 source,
/// `whisper.cpp:6191-6193`: `result_all.back().tokens` is filled from
/// `tokens_cur[i0..=i]`, where token `i` is the timestamp token that closed
/// the segment and `tokens_cur[i0]` the timestamp that opened it) — and
/// their probability says nothing about transcript quality. Averaging them
/// in dilutes the score, worst on short utterances, which is exactly where
/// keeping the gate accurate matters most.
///
/// Furthermore, pure punctuation tokens (like trailing periods on 1-word
/// utterances) are excluded when lexical tokens are present: whisper's LM
/// attaches low probability to sentence terminators on headless single words,
/// which would otherwise cut confidence of genuine speech below the gate.
pub(crate) fn average_text_confidence(
    tokens: &[ScoredToken],
    token_beg: whisper_rs::WhisperToken,
) -> TokenConfidence {
    let mut prob_sum = 0f64;
    let mut logprob_sum = 0f64;
    let mut count = 0usize;

    // First pass: average over lexical text tokens (words/syllables).
    for token in tokens {
        if token.id >= token_beg || !token.is_lexical {
            continue;
        }
        prob_sum += token.p as f64;
        logprob_sum += token.plog as f64;
        count += 1;
    }

    // Fallback: if no tokens were lexical (e.g. all punctuation), average all text tokens.
    if count == 0 {
        for token in tokens {
            if token.id >= token_beg {
                continue;
            }
            prob_sum += token.p as f64;
            logprob_sum += token.plog as f64;
            count += 1;
        }
    }

    if count == 0 {
        TokenConfidence {
            prob: 0.0,
            logprob: 0.0,
        }
    } else {
        TokenConfidence {
            prob: (prob_sum / count as f64) as f32,
            logprob: (logprob_sum / count as f64) as f32,
        }
    }
}

/// Pad sub-second audio with quiet dithered noise instead of exact digital
/// silence: Whisper's mel front-end treats long runs of true zero as a
/// hallucination trigger. Also used by `parakeet` for its own short-utterance
/// floor (`PARAKEET_MIN_SAMPLES`) — the technique isn't whisper-specific,
/// only its home is (kept alongside `WHISPER_MIN_SAMPLES`, its first user).
pub(crate) fn pad_with_dither(pcm_16k: &[f32], min_samples: usize) -> Vec<f32> {
    crate::asr::pad_with_dither(pcm_16k, min_samples)
}

/// Runs one whisper inference. `token_beg` (from `ctx.token_beg()`) is
/// needed to tell text tokens from timestamp tokens when scoring confidence
/// (see `average_text_confidence`). `debug_tokens` dumps every token's
/// id/text/p/plog/is-timestamp to stderr, for confirming that scoring on a
/// real utterance (`STTD_DEBUG_TOKENS=1`). `running` aborts in-flight
/// inference on shutdown (Ctrl-C/`--stop`): whisper.cpp polls the callback,
/// so a long CPU decode exits promptly instead of delaying the trailing
/// flush. The per-call closure box leaks a few dozen bytes, same class as the
/// existing `initial_prompt` leak.
#[allow(clippy::too_many_arguments)]
fn transcribe(
    state: &mut whisper_rs::WhisperState,
    token_beg: whisper_rs::WhisperToken,
    pcm_16k: &[f32],
    prompt: &str,
    language: Language,
    translate: bool,
    debug_tokens: bool,
    beam_size: i32,
    temperature: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    n_threads: i32,
    running: &Arc<AtomicBool>,
) -> Result<WhisperTranscript> {
    // Pad sub-second utterances (e.g. 300-999 ms, which the acoustic gates are
    // happy to pass) so whisper processes them instead of warning "input is too
    // short" and returning zero segments.
    let padded;
    let pcm_16k = if pcm_16k.len() < WHISPER_MIN_SAMPLES {
        padded = pad_with_dither(pcm_16k, WHISPER_MIN_SAMPLES);
        &padded[..]
    } else {
        pcm_16k
    };

    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size,
        patience: -1.0,
    });
    params.set_n_threads(n_threads);
    params.set_language(language.as_whisper_code());
    params.set_translate(translate);
    // Leaked per call: whisper.cpp reads a fresh initial_prompt C-string on
    // every whisper_full and whisper-rs's FullParams has no Drop to free it
    // (crate-level quirk). A few hundred bytes per utterance; not worth
    // engineering around.
    params.set_initial_prompt(prompt);
    // This module manages conversational context explicitly through
    // `initial_prompt` + `push_prompt_context` (drops never feed the tail —
    // see its doc comment), which only holds if whisper.cpp clears its own
    // `prompt_past` between calls. The vendored whisper.cpp defaults
    // `no_context` to `true` already, but pin it explicitly rather than rely
    // on an upstream default that could flip silently and double-feed
    // context.
    params.set_no_context(true);
    params.set_suppress_blank(true);
    params.set_suppress_non_speech_tokens(true);
    params.set_temperature(temperature);
    // Fallback ladder [temperature, +inc, .., 1.0]: rescues genuinely hard
    // speech from a repetition loop. `temperature_inc <= 0.0` disables the
    // ladder (single-temp decode). It is *not* a rejection mechanism —
    // whisper.cpp only re-tries at a non-final temperature when
    // `avg_logprobs < logprob_thold` or entropy exceeds `entropy_thold`;
    // at the final temperature the result is accepted unconditionally
    // (`whisper.cpp`: `if (it != temperatures.size() - 1) { ... success =
    // false ... }`). Confidence rejection is done by the caller instead,
    // via `full_get_token_prob` (see this function's doc comment).
    // NOTE on `no_speech`: whisper.cpp declares `no_speech_thold` but the
    // decoder never reads it (`whisper.h`: `float no_speech_thold; // TODO:
    // not implemented`), and whisper-rs 0.13 exposes no per-segment
    // no-speech probability getter — so there is no no-speech gate to tune
    // here. Silence rejection stays with the acoustic gates
    // (`MIN_SPEECH_FRAMES`, density, `min_rms`) plus the transcript gates
    // (confidence, hallucination, repetition).
    params.set_temperature_inc(temperature_inc);
    params.set_entropy_thold(entropy_thold);
    params.set_logprob_thold(logprob_thold);
    // No `set_no_speech_thold` call: whisper.cpp declares the field but the
    // decoder never reads it (`whisper.h`: `float no_speech_thold; // TODO:
    // not implemented`) — setting it is a no-op, not a safety net.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Abort in-flight inference on shutdown. Raw C callback: the closure-box
    // helper (`set_abort_callback_safe`) can't infer its type here, and the
    // box would leak per utterance anyway. `abort_flag` outlives `full()`
    // (both local to this call; whisper.cpp only polls during `full()`), so
    // the pointer never dangles.
    // SAFETY: `abort_cb` only reads the `Arc<AtomicBool>` behind `user_data`;
    // no mutation, no whisper state touched.
    unsafe extern "C" fn abort_cb(user_data: *mut std::ffi::c_void) -> bool {
        let flag = &*(user_data as *const Arc<AtomicBool>);
        !flag.load(Ordering::Relaxed)
    }
    let abort_flag = Arc::clone(running);
    unsafe {
        params.set_abort_callback(Some(abort_cb));
        params.set_abort_callback_user_data(
            &abort_flag as *const Arc<AtomicBool> as *mut std::ffi::c_void,
        );
    }

    state.full(params, pcm_16k).context("whisper full failed")?;
    let n = state.full_n_segments().context("no segments")?;
    let mut out = String::new();
    let mut tokens: Vec<ScoredToken> = Vec::new();
    for i in 0..n {
        match state.full_get_segment_text_lossy(i) {
            Ok(s) => {
                let t = s.trim();
                if !t.is_empty() {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(t);
                }
            }
            Err(e) => log::warn!("segment {i} read failed: {e:?}"),
        }
        let n_tokens = state.full_n_tokens(i).unwrap_or(0);
        for t in 0..n_tokens {
            if let Ok(data) = state.full_get_token_data(i, t) {
                let text = state
                    .full_get_token_text_lossy(i, t)
                    .unwrap_or_else(|_| "?".to_string());
                if debug_tokens {
                    log::debug!(
                        "  token seg={i} idx={t} id={} p={:.3} plog={:.3} \
                         timestamp={} {text:?}",
                        data.id,
                        data.p,
                        data.plog,
                        data.id >= token_beg
                    );
                }
                let is_lexical = text.chars().any(|c| c.is_alphanumeric());
                tokens.push(ScoredToken {
                    id: data.id,
                    p: data.p,
                    plog: data.plog,
                    is_lexical,
                });
            }
        }
    }
    let confidence = average_text_confidence(&tokens, token_beg);
    Ok(WhisperTranscript {
        text: out,
        confidence,
    })
}

// Text-gate helpers (`fold_polish_diacritic`, `normalize_transcript`,
// `is_hallucination`, `is_repetition`) live in `crate::gate`; the transcript
// decision owns their ordering.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_BEAM_SIZE, DEFAULT_ENTROPY_THOLD, DEFAULT_LOGPROB_THOLD, DEFAULT_TEMPERATURE,
        DEFAULT_TEMPERATURE_INC, DEFAULT_USE_GPU,
    };

    #[test]
    fn language_maps_to_whisper_code() {
        assert_eq!(Language::Polish.as_whisper_code(), Some("pl"));
        assert_eq!(Language::English.as_whisper_code(), Some("en"));
        assert_eq!(Language::Auto.as_whisper_code(), None);
        assert_eq!(Language::Polish.as_str(), "pl");
        assert_eq!(Language::English.as_str(), "en");
        assert_eq!(Language::Auto.as_str(), "auto");
    }

    #[test]
    fn confidence_ignores_timestamp_tokens() {
        // Regression: naively averaging over every token (as the code used
        // to) pulls in the two timestamp tokens whisper.cpp emits per
        // segment (whisper.cpp:6191-6193: `tokens_cur[i0..=i]`, where i0 and
        // i bracket the segment with its opening/closing timestamp tokens),
        // diluting the score. Worst on short utterances, which carry
        // relatively more timestamp weight. The expected means below are a
        // 3-token average, so including the two 0.99 tokens would miss them.
        let token_beg: whisper_rs::WhisperToken = 50364; // whisper.cpp's typical <|0.00|> id
        let tokens = vec![
            ScoredToken {
                id: token_beg,
                p: 0.99,
                plog: -0.01,
                is_lexical: false,
            },
            ScoredToken {
                id: 100,
                p: 0.9,
                plog: -0.1,
                is_lexical: true,
            },
            ScoredToken {
                id: 101,
                p: 0.5,
                plog: -0.7,
                is_lexical: true,
            },
            ScoredToken {
                id: 102,
                p: 0.4,
                plog: -0.9,
                is_lexical: true,
            },
            ScoredToken {
                id: token_beg + 50,
                p: 0.99,
                plog: -0.01,
                is_lexical: false,
            },
        ];
        let confidence = average_text_confidence(&tokens, token_beg);
        assert!((confidence.prob - (0.9 + 0.5 + 0.4) / 3.0).abs() < 1e-6);
        assert!((confidence.logprob - (-0.1 - 0.7 - 0.9) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn confidence_ignores_trailing_punctuation() {
        // Regression: trailing period/punctuation on single-word utterances
        // carries low LM probability (e.g. 0.272), dragging down average
        // confidence of genuine speech. Lexical tokens only must be scored.
        let token_beg: whisper_rs::WhisperToken = 50364;
        let tokens = vec![
            ScoredToken {
                id: token_beg,
                p: 0.88,
                plog: -0.12,
                is_lexical: false,
            },
            ScoredToken {
                id: 9366,
                p: 0.71,
                plog: -0.34,
                is_lexical: true,
            }, // " Ale"
            ScoredToken {
                id: 13,
                p: 0.27,
                plog: -1.30,
                is_lexical: false,
            }, // "."
            ScoredToken {
                id: token_beg + 10,
                p: 0.23,
                plog: -1.46,
                is_lexical: false,
            },
        ];
        let confidence = average_text_confidence(&tokens, token_beg);
        assert!((confidence.prob - 0.71).abs() < 1e-6);
        assert!((confidence.logprob - (-0.34)).abs() < 1e-6);
    }

    #[test]
    fn pad_extends_short_utterance_with_dither() {
        // Regression for: "input is too short - 860 ms < 1000 ms".
        // 860 ms = 13_760 samples must be padded to 16_000.
        let short = vec![0.5f32; 13_760];
        let padded = pad_with_dither(&short, WHISPER_MIN_SAMPLES);
        assert_eq!(padded.len(), WHISPER_MIN_SAMPLES);
        assert_eq!(&padded[..13_760], &short[..]);
        assert!(padded[13_760..].iter().all(|&s| s.abs() <= 1e-4));
    }

    #[test]
    fn pad_leaves_long_utterance_untouched() {
        let long = vec![0.5f32; WHISPER_MIN_SAMPLES + 100];
        let padded = pad_with_dither(&long, WHISPER_MIN_SAMPLES);
        assert_eq!(padded.len(), long.len());
        assert_eq!(padded, long);
    }

    #[test]
    fn decoded_sample_count_reflects_the_pad_floor() {
        // Regression: RTF/metrics used to be computed on the un-padded
        // utterance length, understating a short utterance's actual decoded
        // audio (transcribe pads up to WHISPER_MIN_SAMPLES) and inflating
        // rtf accordingly.
        assert_eq!(decoded_sample_count(0), WHISPER_MIN_SAMPLES);
        assert_eq!(decoded_sample_count(9_600), WHISPER_MIN_SAMPLES); // 600 ms padded up
        assert_eq!(decoded_sample_count(WHISPER_MIN_SAMPLES), WHISPER_MIN_SAMPLES);
        // Already long enough: passes through untouched (no padding happens).
        assert_eq!(
            decoded_sample_count(WHISPER_MIN_SAMPLES + 5_000),
            WHISPER_MIN_SAMPLES + 5_000
        );
    }

    #[test]
    fn inference_defaults_match_compiled_constants() {
        assert_eq!(DEFAULT_BEAM_SIZE, 5);
        assert_eq!(DEFAULT_TEMPERATURE, 0.0);
        assert_eq!(DEFAULT_TEMPERATURE_INC, 0.2);
        assert_eq!(DEFAULT_ENTROPY_THOLD, 2.4);
        assert_eq!(DEFAULT_LOGPROB_THOLD, -1.0);
        assert_eq!(crate::config::DEFAULT_THREADS, 8);
        assert!(DEFAULT_USE_GPU);
    }
}
