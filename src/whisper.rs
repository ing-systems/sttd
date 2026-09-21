use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::{
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_PROMPT, DEFAULT_TARGET_RMS, HALLUCINATION_PATTERNS, MAX_GAIN, WHISPER_MIN_SAMPLES,
    WHISPER_RATE, env_f32,
};
use crate::utterance::{Clip, confidence_threshold};
use crate::wav::write_wav_mono16;

pub(crate) fn spawn_whisper_worker(rx: Receiver<Clip>, model_path: PathBuf) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("whisper".into())
        .spawn(move || {
            if let Err(e) = whisper_loop(&rx, &model_path) {
                eprintln!("[sttd] whisper worker failed: {e:?}");
            }
        })
        .expect("failed to spawn whisper thread")
}

/// Write `reason` as `<dump_dir>/<tag>.txt` for an utterance that was
/// dropped rather than printed, so `STTD_DUMP_DIR` output stays correlatable
/// with the stderr log: previously a dropped utterance left only a WAV with
/// no indication of what became of it or why.
fn dump_drop_reason(dump_dir: &Option<PathBuf>, tag: &str, reason: &str) {
    if let Some(dir) = dump_dir {
        let txt_path = dir.join(format!("{tag}.txt"));
        if let Err(e) = std::fs::write(&txt_path, reason) {
            eprintln!("[sttd] failed to write {}: {e:?}", txt_path.display());
        }
    }
}

fn whisper_loop(rx: &Receiver<Clip>, model_path: &Path) -> Result<()> {
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
    eprintln!("[sttd] whisper ready (model={model_name}, pl, beam=5, threads=8, gpu=true)");

    let prompt = std::env::var("STTD_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let min_confidence = env_f32("STTD_MIN_CONF", DEFAULT_MIN_CONFIDENCE);
    let min_confidence_strong = env_f32("STTD_MIN_CONF_STRONG", DEFAULT_MIN_CONFIDENCE_STRONG);
    // Same override `FrameProcessor::new` reads for the acoustic gate,
    // re-read here rather than threaded through the channel: it's one env
    // lookup and keeps `Clip` free of config, only evidence.
    let min_speech_rms = env_f32("STTD_MIN_RMS", DEFAULT_MIN_SPEECH_RMS);
    let target_rms = env_f32("STTD_TARGET_RMS", DEFAULT_TARGET_RMS);
    let debug_tokens = std::env::var("STTD_DEBUG_TOKENS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let dump_dir = std::env::var_os("STTD_DUMP_DIR").map(PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        eprintln!("[sttd] dumping utterances to {}", dir.display());
    }

    let mut utterance_no: u64 = 0;
    for clip in rx.iter() {
        if clip.pcm.is_empty() {
            continue;
        }
        utterance_no += 1;
        let tag = format!("utt_{utterance_no:06}");

        if let Some(dir) = &dump_dir {
            let wav_path = dir.join(format!("{tag}.wav"));
            if let Err(e) = write_wav_mono16(&wav_path, &clip.pcm, WHISPER_RATE) {
                eprintln!("[sttd] failed to dump {}: {e:?}", wav_path.display());
            }
        }

        let gained = apply_gain(&clip.pcm, clip.rms, target_rms, MAX_GAIN);

        match transcribe(&mut state, token_beg, &gained, &prompt, debug_tokens) {
            Ok(t) => {
                let trimmed = t.text.trim();
                if trimmed.is_empty() {
                    dump_drop_reason(&dump_dir, &tag, "DROPPED(empty)");
                    continue;
                }
                let (threshold, gate) = confidence_threshold(
                    &clip,
                    min_speech_rms,
                    min_confidence,
                    min_confidence_strong,
                );
                eprintln!(
                    "[sttd] {tag}: avg_p={:.3} avg_logprob={:.3} threshold={threshold:.3} \
                     ({gate} evidence)",
                    t.avg_prob, t.avg_logprob
                );
                if t.avg_prob < threshold {
                    eprintln!("[sttd] {tag}: dropped low-confidence transcript: {trimmed:?}");
                    dump_drop_reason(
                        &dump_dir,
                        &tag,
                        &format!(
                            "DROPPED(low-conf avg_p={:.3} < {threshold:.3}, {gate}): {trimmed}",
                            t.avg_prob
                        ),
                    );
                    continue;
                }
                if is_hallucination(trimmed) {
                    eprintln!("[sttd] {tag}: dropped likely hallucination: {trimmed:?}");
                    dump_drop_reason(
                        &dump_dir,
                        &tag,
                        &format!("DROPPED(hallucination): {trimmed}"),
                    );
                    continue;
                }
                println!("{trimmed}");
                if let Some(dir) = &dump_dir {
                    let txt_path = dir.join(format!("{tag}.txt"));
                    if let Err(e) = std::fs::write(&txt_path, trimmed) {
                        eprintln!("[sttd] failed to write {}: {e:?}", txt_path.display());
                    }
                }
            }
            Err(e) => eprintln!("[sttd] {tag}: transcription failed: {e:?}"),
        }
    }
    Ok(())
}

/// Apply gain toward `target_rms`, computed from the speech-only `rms`
/// `flush_utterance` already measured (not recomputed over `pcm`, which
/// includes the padded silence tail). Only ever amplifies (gain >= 1.0) and
/// never attenuates: a clip reaching this point already cleared the acoustic
/// gates, so there's no silence here to accidentally boost. Caps at
/// `max_gain`, and backs off further if that gain would clip the loudest
/// sample past 0.99. `target_rms <= 0.0` or `rms <= 0.0` disables gain.
pub(crate) fn apply_gain(pcm: &[f32], rms: f32, target_rms: f32, max_gain: f32) -> Vec<f32> {
    if target_rms <= 0.0 || rms <= 0.0 {
        return pcm.to_vec();
    }
    let mut gain = (target_rms / rms).clamp(1.0, max_gain);
    if gain <= 1.0 {
        return pcm.to_vec();
    }
    let peak = pcm.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.0 {
        gain = gain.min(0.99 / peak);
    }
    pcm.iter().map(|&s| s * gain).collect()
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

/// Result of one whisper inference: the joined segment text plus confidence
/// averaged over *text* tokens only (see `average_text_confidence`).
struct Transcript {
    text: String,
    /// `sttd`'s stand-in for whisper.cpp's `no_speech_thold`, which is
    /// declared in `whisper_full_params` but never actually read by the
    /// decoder (see `DEFAULT_MIN_CONFIDENCE`'s doc comment) — so whisper.cpp
    /// itself never rejects a low-confidence segment, and the caller must
    /// do it based on this value.
    avg_prob: f32,
    /// Same average, in log space — the statistic whisper.cpp's own
    /// fallback ladder uses internally (`whisper.cpp`: `avg_logprobs`),
    /// logged alongside `avg_prob` for comparable-scale tuning.
    avg_logprob: f32,
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
) -> (f32, f32, usize) {
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
        (0.0, 0.0, 0)
    } else {
        (
            (prob_sum / count as f64) as f32,
            (logprob_sum / count as f64) as f32,
            count,
        )
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
    debug_tokens: bool,
) -> Result<Transcript> {
    // Pad sub-second utterances (e.g. 300-999 ms that pass MIN_UTTERANCE_SAMPLES)
    // so whisper processes them instead of warning "input is too short" and
    // returning zero segments.
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
    params.set_language(Some("pl"));
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
    let (avg_prob, avg_logprob, _text_token_count) = average_text_confidence(&tokens, token_beg);
    Ok(Transcript {
        text: out,
        avg_prob,
        avg_logprob,
    })
}

/// Fold a Polish diacritic to its ASCII base letter; every other char is
/// returned unchanged. Applied before comparing against
/// `HALLUCINATION_PATTERNS` (themselves written without diacritics) so a
/// transcript like "Dziękuje za uwagę" — a plausible ASR misspelling of
/// "Dziękuję za uwagę" — still matches instead of sliding past on a single
/// ogonek.
pub(crate) fn fold_diacritics(c: char) -> char {
    match c {
        'ą' => 'a',
        'ć' => 'c',
        'ę' => 'e',
        'ł' => 'l',
        'ń' => 'n',
        'ó' => 'o',
        'ś' => 's',
        'ź' | 'ż' => 'z',
        _ => c,
    }
}

/// Whether `text` (a full transcript) is a known Whisper hallucination on
/// silence/noise, checked as an exact match after normalization so genuine
/// sentences that merely contain e.g. "dziękuję" are never caught.
pub(crate) fn is_hallucination(text: &str) -> bool {
    // Punctuation becomes a space (not dropped): "Amara.org" must normalize
    // to "amara org" to match the pattern list, not "amaraorg". Diacritics
    // are folded (see `fold_diacritics`) before the alphanumeric check, not
    // after, since they're themselves alphanumeric and would otherwise
    // survive untouched.
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(fold_diacritics)
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return true; // pure punctuation, e.g. "..."
    }
    HALLUCINATION_PATTERNS.contains(&normalized.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_TARGET_RMS, MAX_GAIN, WHISPER_MIN_SAMPLES};

    #[test]
    fn confidence_ignores_timestamp_tokens() {
        // Regression: naively averaging over every token (as the code used
        // to) pulls in the two timestamp tokens whisper.cpp emits per
        // segment (whisper.cpp:6191-6193: `tokens_cur[i0..=i]`, where i0 and
        // i bracket the segment with its opening/closing timestamp tokens),
        // diluting the score. Worst on short utterances, which carry
        // relatively more timestamp weight.
        let token_beg: whisper_rs::WhisperToken = 50364; // whisper.cpp's typical <|0.00|> id
        let tokens = vec![
            (token_beg, 0.99, -0.01),      // leading timestamp: must be excluded
            (100, 0.9, -0.1),              // text
            (101, 0.5, -0.7),              // text
            (102, 0.4, -0.9),              // text
            (token_beg + 50, 0.99, -0.01), // closing timestamp: must be excluded
        ];
        let (avg_prob, avg_logprob, count) = average_text_confidence(&tokens, token_beg);
        assert_eq!(count, 3);
        assert!((avg_prob - (0.9 + 0.5 + 0.4) / 3.0).abs() < 1e-6);
        assert!((avg_logprob - (-0.1 - 0.7 - 0.9) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn normalization_gain_is_capped() {
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
    fn normalization_gain_disabled_by_zero_target() {
        let pcm = vec![0.01f32; 100];
        let out = apply_gain(&pcm, 0.01, 0.0, MAX_GAIN);
        assert_eq!(out, pcm);
    }

    #[test]
    fn normalization_gain_never_attenuates() {
        // A clip already louder than the target must be left alone, not
        // turned down toward it — this gate only ever helps quiet-but-real
        // speech, never touches an already-strong signal.
        let pcm = vec![0.2f32; 100];
        let out = apply_gain(&pcm, 0.2, DEFAULT_TARGET_RMS, MAX_GAIN);
        assert_eq!(out, pcm);
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
    fn hallucination_filter_catches_known_boilerplate() {
        assert!(is_hallucination("Napisy stworzone przez społeczność Amara.org"));
        assert!(is_hallucination("  napisy  "));
        assert!(is_hallucination("..."));
        assert!(is_hallucination(""));
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
    fn hallucination_filter_leaves_real_sentences_alone() {
        assert!(!is_hallucination("Zrób commit i wypchnij zmiany do repozytorium."));
        assert!(!is_hallucination(
            "Dziękuję, że przyszedłeś na spotkanie o dziewiątej."
        ));
    }
}
