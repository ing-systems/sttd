use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::{
    CONDITION_FADE_SAMPLES, CONDITION_HP_CUTOFF_HZ, MAX_GAIN, WHISPER_MIN_SAMPLES, WHISPER_RATE,
};
use crate::emit::{Emission, Emitter};
use crate::gate::GateConfig;
use crate::utterance::Utterance;

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

/// Tunables the whisper worker needs, resolved once from `Cli`. Emission
/// policy (`dump_dir`, clipboard, paste) lives with the `Emitter`, not here.
#[derive(Debug, Clone)]
pub(crate) struct WhisperOptions {
    pub(crate) prompt: String,
    pub(crate) language: Language,
    pub(crate) gate: GateConfig,
    pub(crate) target_rms: f32,
    pub(crate) debug_tokens: bool,
}

pub(crate) fn spawn_whisper_worker(
    rx: Receiver<Utterance>,
    model_path: PathBuf,
    opts: WhisperOptions,
    emitter: impl Emitter + Send + 'static,
) -> (
    JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("whisper".into())
        .spawn(move || {
            let mut emitter = emitter;
            if let Err(e) = whisper_loop(&rx, &model_path, &opts, ready_tx, &mut emitter) {
                eprintln!("[sttd] whisper worker failed: {e:?}");
            }
        })
        .expect("failed to spawn whisper thread");
    (handle, ready_rx)
}

/// Write `reason` as `<dump_dir>/<tag>.txt` for an utterance that was
/// dropped rather than printed, so `STTD_DUMP_DIR` output stays correlatable
/// with the stderr log: previously a dropped utterance left only a WAV with
/// no indication of what became of it or why.
fn whisper_loop(
    rx: &Receiver<Utterance>,
    model_path: &Path,
    opts: &WhisperOptions,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &mut impl Emitter,
) -> Result<()> {
    // Run init; report the outcome once via `ready` so the main thread can
    // fail fast instead of capturing into a dead queue.
    match whisper_init_and_run(rx, model_path, opts, &ready, emitter) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            Err(e)
        }
    }
}

fn whisper_init_and_run(
    rx: &Receiver<Utterance>,
    model_path: &Path,
    opts: &WhisperOptions,
    ready: &std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &mut impl Emitter,
) -> Result<()> {
    let path_str = model_path
        .to_str()
        .context("model path is not valid UTF-8")?;

    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu = true;
    eprintln!("[sttd] loading whisper context (gpu requested)...");
    let ctx = WhisperContext::new_with_params(path_str, ctx_params)
        .context("failed to load whisper model")?;
    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;
    let token_beg = ctx.token_beg();

    let model_name = ctx
        .model_type_readable()
        .unwrap_or_else(|_| "unknown".to_string());
    eprintln!(
        "[sttd] whisper ready (model={model_name}, lang={}, beam=5, threads=8, gpu=true)",
        opts.language.as_str()
    );

    let prompt = opts.prompt.clone();
    let language = opts.language;
    let gate = opts.gate.clone();
    let target_rms = opts.target_rms;
    let debug_tokens = opts.debug_tokens;
    // Init succeeded (model + state): signal ready so the main thread knows
    // captures aren't going into a dead queue. Emission policy (dump dir,
    // clipboard, paste) already resolved on the main thread inside
    // `SystemEmitter::new`, before the worker spawned.
    let _ = ready.send(Ok(()));

    let mut utterance_no: u64 = 0;
    for utterance in rx.iter() {
        if utterance.pcm.is_empty() {
            continue;
        }
        utterance_no += 1;
        let tag = format!("utt_{utterance_no:06}");

        let conditioned = condition_utterance(&utterance.pcm);
        let gained = apply_gain(&conditioned, utterance.evidence.rms, target_rms, MAX_GAIN);

        match transcribe(
            &mut state,
            token_beg,
            gained.as_ref(),
            &prompt,
            language,
            debug_tokens,
        ) {
            Ok(t) => {
                let trimmed = t.text.trim();
                let verdict = gate.decide_transcript(
                    utterance.evidence.speech_frames,
                    utterance.evidence.rms,
                    trimmed,
                    t.confidence.prob,
                );
                emitter.emit(Emission {
                    tag: &tag,
                    pcm: &utterance.pcm,
                    text: trimmed,
                    verdict,
                    prob: t.confidence.prob,
                    logprob: t.confidence.logprob,
                    evidence: &utterance.evidence,
                });
            }
            Err(e) => eprintln!("[sttd] {tag}: transcription failed: {e:?}"),
        }
    }
    Ok(())
}

/// Condition an utterance for whisper: DC removal, gentle one-pole
/// high-pass at `CONDITION_HP_CUTOFF_HZ`, short linear fades at both ends,
/// and peak safety to 0.99. Runs before `apply_gain` so the gain stage sees
/// hum-free, click-free audio.
///
/// Kept deliberately light because the whisper branch is raw by design (see
/// `Whisper branch` in `CONTEXT.md`): 60 Hz / 6 dB-oct costs ~0.5 dB at
/// 150 Hz and nothing in the 4-8 kHz sibilant band, and the 5 ms fades land
/// in pre-roll/tail silence rather than speech.
pub(crate) fn condition_utterance(pcm: &[f32]) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    // DC removal: subtract the mean so soundcard offset can't bias the
    // high-pass startup transient or the gain stage.
    let mean = pcm.iter().fold(0.0f64, |m, &s| m + s as f64) / pcm.len() as f64;
    let mut out: Vec<f32> = pcm.iter().map(|&s| s - mean as f32).collect();

    // One-pole high-pass: y[n] = x[n] - x[n-1] + a * y[n-1],
    // a = exp(-2*pi*fc/fs). Zero-state start is fine post-DC-removal: the
    // first samples are pre-roll silence near zero.
    let a = (-2.0 * std::f32::consts::PI * CONDITION_HP_CUTOFF_HZ / WHISPER_RATE as f32).exp();
    let mut x_prev = out[0];
    let mut y_prev = 0.0f32;
    for s in out.iter_mut() {
        let x = *s;
        let y = x - x_prev + a * y_prev;
        x_prev = x;
        y_prev = y;
        *s = y;
    }

    // Linear fades at both ends; shortened for very short utterances so the
    // ramps never overlap.
    let fade = CONDITION_FADE_SAMPLES.min(out.len() / 2);
    for i in 0..fade {
        let ramp = (i + 1) as f32 / (fade + 1) as f32;
        out[i] *= ramp;
        let j = out.len() - 1 - i;
        out[j] *= ramp;
    }

    // Peak safety: resampler overshoot can exceed 1.0; keep headroom so
    // `apply_gain` isn't blocked by a single hot sample.
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.99 {
        let scale = 0.99 / peak;
        for s in out.iter_mut() {
            *s *= scale;
        }
    }
    out
}

/// Apply gain toward `target_rms`, computed from the speech-only `rms`
/// `UtteranceBuffer::close` already measured (not recomputed over `pcm`, which
/// includes the padded silence tail). Only ever amplifies (gain >= 1.0) and
/// never attenuates: a clip reaching this point already cleared the acoustic
/// gates, so there's no silence here to accidentally boost. Caps at
/// `max_gain`, and backs off further if that gain would clip the loudest
/// sample past 0.99. `target_rms <= 0.0` or `rms <= 0.0` disables gain.
/// Returns borrowed when no gain is needed so the loud common case avoids a copy.
pub(crate) fn apply_gain<'a>(
    pcm: &'a [f32],
    rms: f32,
    target_rms: f32,
    max_gain: f32,
) -> std::borrow::Cow<'a, [f32]> {
    if target_rms <= 0.0 || rms <= 0.0 {
        return std::borrow::Cow::Borrowed(pcm);
    }
    let mut gain = (target_rms / rms).clamp(1.0, max_gain);
    if gain <= 1.0 {
        return std::borrow::Cow::Borrowed(pcm);
    }
    let peak = pcm.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.0 {
        gain = gain.min(0.99 / peak);
    }
    std::borrow::Cow::Owned(pcm.iter().map(|&s| s * gain).collect())
}

/// Pad sub-second audio with quiet dithered noise instead of exact digital
/// silence: Whisper's mel front-end treats long runs of true zero as a
/// hallucination trigger.
pub(crate) fn pad_with_dither(pcm_16k: &[f32], min_samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(min_samples.max(pcm_16k.len()));
    out.extend_from_slice(pcm_16k);
    // xorshift32: cheap, deterministic, no extra dependency.
    let mut seed: u32 = 0x9E37_79B9 ^ (pcm_16k.len() as u32).wrapping_add(1);
    while out.len() < min_samples {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let n = (seed as f32 / u32::MAX as f32 - 0.5) * 2e-4; // +-1e-4
        out.push(n);
    }
    out
}

/// Confidence of one inference, averaged over *text* tokens only (see
/// `average_text_confidence`).
pub(crate) struct TokenConfidence {
    /// `sttd`'s stand-in for whisper.cpp's `no_speech_thold`, which is
    /// declared in `whisper_full_params` but never actually read by the
    /// decoder (see `DEFAULT_MIN_CONFIDENCE`'s doc comment) — so whisper.cpp
    /// itself never rejects a low-confidence segment, and the caller must
    /// do it based on this value.
    pub(crate) prob: f32,
    /// Same average, in log space — the statistic whisper.cpp's own
    /// fallback ladder uses internally (`whisper.cpp`: `avg_logprobs`),
    /// logged alongside `prob` for comparable-scale tuning.
    pub(crate) logprob: f32,
}

/// Result of one whisper inference: the joined segment text plus its
/// `TokenConfidence`.
struct Transcript {
    text: String,
    confidence: TokenConfidence,
}

/// Average token probability/log-probability over *text* tokens only, given
/// each token's `(id, p, plog)`. Timestamp tokens (`id >= token_beg`) sit in
/// every whisper.cpp segment — a leading `<|0.00|>` and a closing `<|t|>`
/// (confirmed in the vendored whisper.cpp 1.7.1 source,
/// `whisper.cpp:6191-6193`: `result_all.back().tokens` is filled from
/// `tokens_cur[i0..=i]`, where token `i` is the timestamp token that closed
/// the segment and `tokens_cur[i0]` the timestamp that opened it) — and
/// their probability says nothing about transcript quality. Averaging them
/// in dilutes the score, worst on short utterances, which is exactly where
/// keeping the gate accurate matters most.
pub(crate) fn average_text_confidence(
    tokens: &[(whisper_rs::WhisperToken, f32, f32)],
    token_beg: whisper_rs::WhisperToken,
) -> TokenConfidence {
    let mut prob_sum = 0f64;
    let mut logprob_sum = 0f64;
    let mut count = 0usize;
    for &(id, p, plog) in tokens {
        if id >= token_beg {
            continue;
        }
        prob_sum += p as f64;
        logprob_sum += plog as f64;
        count += 1;
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

/// Runs one whisper inference. `token_beg` (from `ctx.token_beg()`) is
/// needed to tell text tokens from timestamp tokens when scoring confidence
/// (see `average_text_confidence`). `debug_tokens` dumps every token's
/// id/text/p/plog/is-timestamp to stderr, for confirming that scoring on a
/// real utterance (`STTD_DEBUG_TOKENS=1`).
fn transcribe(
    state: &mut whisper_rs::WhisperState,
    token_beg: whisper_rs::WhisperToken,
    pcm_16k: &[f32],
    prompt: &str,
    language: Language,
    debug_tokens: bool,
) -> Result<Transcript> {
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
        beam_size: 5,
        patience: -1.0,
    });
    params.set_n_threads(8);
    params.set_language(language.as_whisper_code());
    params.set_translate(false);
    // Leaked per call: whisper.cpp reads a fresh initial_prompt C-string on
    // every whisper_full and whisper-rs's FullParams has no Drop to free it
    // (crate-level quirk). A few hundred bytes per utterance; not worth
    // engineering around.
    params.set_initial_prompt(prompt);
    params.set_suppress_blank(true);
    params.set_suppress_non_speech_tokens(true);
    params.set_temperature(0.0);
    // Fallback ladder [0.0, 0.2, .., 1.0]: rescues genuinely hard speech
    // from a repetition loop. It is *not* a rejection mechanism — whisper.cpp
    // only re-tries at a non-final temperature when `avg_logprobs <
    // logprob_thold`; at the final temperature (here, 1.0) the result is
    // accepted unconditionally (`whisper.cpp`: `if (it != temperatures.size()
    // - 1) { ... success = false ... }`). Confidence rejection is done by
    // the caller instead, via `full_get_token_prob` (see this function's
    // doc comment).
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    // No `set_no_speech_thold` call: whisper.cpp declares the field but the
    // decoder never reads it (`whisper.h`: `float no_speech_thold; // TODO:
    // not implemented`) — setting it is a no-op, not a safety net.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.full(params, pcm_16k).context("whisper full failed")?;
    let n = state.full_n_segments().context("no segments")?;
    let mut out = String::new();
    let mut tokens: Vec<(whisper_rs::WhisperToken, f32, f32)> = Vec::new();
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
            Err(e) => eprintln!("[sttd] segment {i} read failed: {e:?}"),
        }
        let n_tokens = state.full_n_tokens(i).unwrap_or(0);
        for t in 0..n_tokens {
            if let Ok(data) = state.full_get_token_data(i, t) {
                if debug_tokens {
                    let text = state
                        .full_get_token_text_lossy(i, t)
                        .unwrap_or_else(|_| "?".to_string());
                    eprintln!(
                        "[sttd]   token seg={i} idx={t} id={} p={:.3} plog={:.3} \
                         timestamp={} {text:?}",
                        data.id,
                        data.p,
                        data.plog,
                        data.id >= token_beg
                    );
                }
                tokens.push((data.id, data.p, data.plog));
            }
        }
    }
    let confidence = average_text_confidence(&tokens, token_beg);
    Ok(Transcript { text: out, confidence })
}

// Text-gate helpers (`fold_polish_diacritic`, `normalize_transcript`,
// `is_hallucination`, `is_repetition`) live in `crate::gate` and are
// re-exported above; the transcript decision owns their ordering.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
        DEFAULT_MIN_SPEECH_RMS, DEFAULT_TARGET_RMS, MAX_GAIN, MIN_SPEECH_FRAMES,
        WHISPER_MIN_SAMPLES,
    };
    use crate::gate::{EvidenceStrength, GateConfig, is_hallucination, is_repetition};
    use crate::utterance::Evidence;

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
            (token_beg, 0.99, -0.01),      // leading timestamp: must be excluded
            (100, 0.9, -0.1),              // text
            (101, 0.5, -0.7),              // text
            (102, 0.4, -0.9),              // text
            (token_beg + 50, 0.99, -0.01), // closing timestamp: must be excluded
        ];
        let confidence = average_text_confidence(&tokens, token_beg);
        assert!((confidence.prob - (0.9 + 0.5 + 0.4) / 3.0).abs() < 1e-6);
        assert!((confidence.logprob - (-0.1 - 0.7 - 0.9) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn strong_acoustic_evidence_relaxes_confidence_gate() {
        // Exact numbers from the production log this fix addresses:
        // "[sttd] utterance: 34720 samples (137 speech frames, rms=0.01178)
        // -> whisper" followed by "dropped low-confidence transcript
        // (avg_p=0.478 < 0.600)" for a transcript the user confirmed was
        // close to correct (one misheard word), not a hallucination.
        let evidence = Evidence {
            speech_frames: 137,
            total_frames: 200,
            rms: 0.01178,
        };
        let gate = GateConfig::new(
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_DENSITY,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        let (floor, strength) = gate.floor_for(evidence.speech_frames, evidence.rms);
        assert_eq!(strength, EvidenceStrength::Strong);
        assert_eq!(floor, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert!(
            0.478 >= floor,
            "avg_p=0.478 must clear the relaxed threshold {floor}"
        );
    }

    #[test]
    fn weak_evidence_keeps_strict_gate() {
        // Regression for the hole 71c05b0 closed: an utterance with barely
        // enough VAD-flagged speech to survive `close`'s own gates, and RMS
        // just above the noise floor, must still face the strict confidence
        // floor — even at the same avg_p (0.478) that the strong-evidence
        // case above lets through.
        let evidence = Evidence {
            speech_frames: MIN_SPEECH_FRAMES,
            total_frames: 131,
            rms: DEFAULT_MIN_SPEECH_RMS + 0.0005,
        };
        let gate = GateConfig::new(
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_DENSITY,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        let (floor, strength) = gate.floor_for(evidence.speech_frames, evidence.rms);
        assert_eq!(strength, EvidenceStrength::Weak);
        assert_eq!(floor, DEFAULT_MIN_CONFIDENCE);
        assert!(
            0.478 < floor,
            "avg_p=0.478 must NOT clear the strict threshold {floor}"
        );
    }

    #[test]
    fn apply_gain_is_capped() {
        // A near-silent clip (rms=0.0001) must not get amplified by the raw
        // target/rms ratio (500x here) — that would blow quantization noise
        // up into something whisper mistakes for speech. Capped at
        // MAX_GAIN, and backed off further if it would clip.
        let pcm = vec![0.0001f32; 100];
        let out = apply_gain(&pcm, 0.0001, DEFAULT_TARGET_RMS, MAX_GAIN);
        let applied_gain = out[0] / pcm[0];
        assert!(
            applied_gain <= MAX_GAIN + 1e-3,
            "gain={applied_gain} exceeds cap {MAX_GAIN}"
        );
        assert!(out.iter().all(|&s| s.abs() <= 0.99 + 1e-6));
    }

    #[test]
    fn apply_gain_disabled_by_zero_target() {
        let pcm = vec![0.01f32; 100];
        let out = apply_gain(&pcm, 0.01, 0.0, MAX_GAIN);
        assert_eq!(out.as_ref(), pcm.as_slice());
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn apply_gain_never_attenuates() {
        // A clip already louder than the target must be left alone, not
        // turned down toward it — this gate only ever helps quiet-but-real
        // speech, never touches an already-strong signal.
        let pcm = vec![0.2f32; 100];
        let out = apply_gain(&pcm, 0.2, DEFAULT_TARGET_RMS, MAX_GAIN);
        assert_eq!(out.as_ref(), pcm.as_slice());
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn apply_gain_borrows_when_no_gain_needed() {
        // Loud common case must avoid a copy: gain <= 1.0 returns borrowed.
        let pcm = vec![0.2f32; 100];
        let out = apply_gain(&pcm, 0.2, DEFAULT_TARGET_RMS, MAX_GAIN);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), pcm.as_slice());
        // Quiet case that needs gain returns owned.
        let quiet = vec![0.001f32; 100];
        let out_owned = apply_gain(&quiet, 0.001, DEFAULT_TARGET_RMS, MAX_GAIN);
        assert!(matches!(out_owned, std::borrow::Cow::Owned(_)));
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
    fn condition_removes_dc_offset() {
        // Soundcard offset must not reach whisper: a constant 0.1 offset
        // carries no speech and would bias gain.
        let pcm = vec![0.1f32; 1600];
        let out = condition_utterance(&pcm);
        assert_eq!(out.len(), pcm.len());
        let mean = out.iter().fold(0.0f64, |m, &s| m + s as f64) / out.len() as f64;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn condition_attenuates_50hz_hum() {
        // 50 Hz hum at 16 kHz: compare RMS before/after over the steady
        // middle (past the HP startup transient and fades).
        let hum: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let out = condition_utterance(&hum);
        let rms = |v: &[f32]| (v.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / v.len() as f64).sqrt();
        let before = rms(&hum[2000..14_000]);
        let after = rms(&out[2000..14_000]);
        assert!(after < before * 0.7, "before={before} after={after}");
    }

    #[test]
    fn condition_keeps_speech_band_intact() {
        // 1 kHz tone (speech band) must pass nearly untouched in the middle.
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.2)
            .collect();
        let out = condition_utterance(&tone);
        let rms = |v: &[f32]| (v.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / v.len() as f64).sqrt();
        let before = rms(&tone[2000..14_000]);
        let after = rms(&out[2000..14_000]);
        let ratio = after / before;
        assert!((0.9..=1.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn condition_fades_edges_and_caps_peak() {
        // Flat loud input: fades pull both ends to near zero, peak safety
        // caps the middle at 0.99.
        let pcm = vec![2.0f32; 1600];
        let out = condition_utterance(&pcm);
        assert_eq!(out.len(), pcm.len());
        assert!(out[0].abs() < 0.05, "first={}", out[0]);
        assert!(out[out.len() - 1].abs() < 0.05, "last={}", out[out.len() - 1]);
        assert!(out.iter().all(|&s| s.abs() <= 0.99 + 1e-6));
        // Short input: fades shorten instead of overlapping.
        let short = vec![0.5f32; 10];
        let out_short = condition_utterance(&short);
        assert_eq!(out_short.len(), short.len());
    }

    #[test]
    fn condition_empty_stays_empty() {
        assert!(condition_utterance(&[]).is_empty());
    }

    #[test]
    fn hallucination_filter_catches_known_boilerplate() {
        assert!(is_hallucination("Napisy stworzone przez społeczność Amara.org"));
        assert!(is_hallucination("  napisy  "));
        assert!(is_hallucination("..."));
        assert!(is_hallucination(""));
        // Observed forced-input decodes: white noise (rms=0.008) at
        // avg_p=0.890 and 50 Hz hum at avg_p=0.993 — both clear either
        // confidence floor, so the pattern list is the only gate.
        assert!(is_hallucination("Wszelkie prawa."));
        assert!(is_hallucination("Wszelkie prawa zastrzeżone."));
    }

    #[test]
    fn hallucination_filter_handles_diacritics() {
        // "Dziękuje za uwagę." (log-observed hallucination) differs from the
        // pattern "dziękuję za uwagę" by a single ogonek; without diacritic
        // folding this exact-match filter let it straight through.
        assert!(is_hallucination("Dziękuje za uwagę."));
        assert!(is_hallucination("dziekuje za uwage"));
    }

    #[test]
    fn repetition_filter_catches_decode_loops() {
        // Observed production-shape leak: 300 ms noise burst -> 26 speech
        // frames -> "Współpraca w ..." at avg_p=0.741, clear of the 0.6
        // weak floor and absent from the boilerplate list.
        assert!(is_repetition(
            "Współpraca w Współpraca w Współpraca w Współpraca w Współpraca."
        ));
        assert!(is_repetition("tak tak tak tak tak tak"));
    }

    #[test]
    fn repetition_filter_leaves_real_speech_alone() {
        assert!(!is_repetition("Zrób commit i wypchnij zmiany do repozytorium."));
        assert!(!is_repetition(
            "Dziękuję, że przyszedłeś na spotkanie o dziewiątej."
        ));
        // Short emphasis is exempt (< 6 words).
        assert!(!is_repetition("nie nie nie"));
        assert!(!is_repetition("tak"));
    }

    #[test]
    fn hallucination_filter_leaves_real_sentences_alone() {
        assert!(!is_hallucination("Zrób commit i wypchnij zmiany do repozytorium."));
        assert!(!is_hallucination(
            "Dziękuję, że przyszedłeś na spotkanie o dziewiątej."
        ));
    }

    #[test]
    fn whisper_worker_signals_init_failure() {
        // Bad model path must propagate via the ready channel instead of
        // silently capturing into a dead queue.
        let (_tx, rx) = crossbeam_channel::unbounded();
        let opts = WhisperOptions {
            prompt: String::new(),
            language: Language::Polish,
            gate: GateConfig::new(
                DEFAULT_MIN_SPEECH_RMS,
                DEFAULT_MIN_DENSITY,
                DEFAULT_MIN_CONFIDENCE,
                DEFAULT_MIN_CONFIDENCE_STRONG,
            ),
            target_rms: DEFAULT_TARGET_RMS,
            debug_tokens: false,
        };
        let bad = std::path::PathBuf::from("/nonexistent/ggml-bogus.bin");
        let (handle, ready) =
            spawn_whisper_worker(rx, bad, opts, crate::emit::RecordingEmitter::new());
        match ready.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(Ok(())) => panic!("bad model path must not signal ready"),
            Ok(Err(e)) => assert!(
                e.contains("failed to load whisper model"),
                "unexpected error: {e}"
            ),
            Err(e) => panic!("ready channel must send once, got {e:?}"),
        }
        handle.join().expect("worker thread must exit");
    }
}
