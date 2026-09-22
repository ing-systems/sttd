use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::{
    HALLUCINATION_PATTERNS, MAX_GAIN, STRONG_RMS_FACTOR, STRONG_SPEECH_FRAMES,
    WHISPER_MIN_SAMPLES, WHISPER_RATE,
};
use crate::utterance::{Evidence, Utterance};
use crate::wav::write_wav_mono16;

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

/// Tunables the whisper worker needs, resolved once from `Cli`.
#[derive(Debug, Clone)]
pub(crate) struct WhisperOptions {
    pub(crate) prompt: String,
    pub(crate) language: Language,
    pub(crate) min_confidence: f32,
    pub(crate) min_confidence_strong: f32,
    pub(crate) min_speech_rms: f32,
    pub(crate) target_rms: f32,
    pub(crate) debug_tokens: bool,
    pub(crate) dump_dir: Option<PathBuf>,
    pub(crate) copy_enabled: bool,
    pub(crate) paste_enabled: bool,
}

pub(crate) fn spawn_whisper_worker(
    rx: Receiver<Utterance>,
    model_path: PathBuf,
    opts: WhisperOptions,
) -> (
    JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("whisper".into())
        .spawn(move || {
            if let Err(e) = whisper_loop(&rx, &model_path, &opts, ready_tx) {
                eprintln!("[sttd] whisper worker failed: {e:?}");
            }
        })
        .expect("failed to spawn whisper thread");
    (handle, ready_rx)
}

/// How much trust an utterance's acoustics have already earned, which selects
/// the confidence floor its transcript then has to clear.
#[derive(PartialEq, Eq, Debug)]
pub(crate) enum EvidenceStrength {
    Weak,
    Strong,
}

impl std::fmt::Display for EvidenceStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // These spellings go straight into the log lines below, so they are
        // fixed.
        f.write_str(match self {
            Self::Weak => "weak",
            Self::Strong => "strong",
        })
    }
}

/// Confidence floor for this utterance's transcript, paired with the
/// `EvidenceStrength` that selected it. The confidence gate exists to catch
/// hallucinations on silence/noise; where the acoustics already testify to
/// real speech, it has nothing to add, and applying the strict floor anyway
/// discards the only transcript we have. It doesn't disappear entirely: the
/// relaxed floor still catches a repetition-loop decode (avg_p ~0.1), which
/// isn't a transcript of anything.
pub(crate) fn confidence_floor(
    evidence: &Evidence,
    min_rms: f32,
    weak_floor: f32,
    strong_floor: f32,
) -> (f32, EvidenceStrength) {
    if evidence.speech_frames >= STRONG_SPEECH_FRAMES && evidence.rms >= min_rms * STRONG_RMS_FACTOR
    {
        (strong_floor, EvidenceStrength::Strong)
    } else {
        (weak_floor, EvidenceStrength::Weak)
    }
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

/// Whether the current session is Wayland, where clipboard (`wl-copy`) + paste
/// (`ydotool key` Ctrl+V) is supported. Pure over its inputs so tests don't touch the process
/// environment (see `is_wayland_session_env` for the env-reading wrapper).
/// `WAYLAND_DISPLAY` set wins over `DISPLAY` (XWayland sets both), so an
/// X.org session is `XDG_SESSION_TYPE=x11` with `WAYLAND_DISPLAY` empty.
pub(crate) fn is_wayland_session(
    xdg_session_type: Option<&str>,
    wayland_display: Option<&str>,
) -> bool {
    if wayland_display.is_some_and(|v| !v.trim().is_empty()) {
        return true;
    }
    xdg_session_type.is_some_and(|v| v.trim().eq_ignore_ascii_case("wayland"))
}

/// Env-reading wrapper around [`is_wayland_session`].
pub(crate) fn is_wayland_session_env() -> bool {
    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
    let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();
    is_wayland_session(session_type.as_deref(), wayland_display.as_deref())
}

/// Copy `text` to the Wayland clipboard via `wl-copy` (from wl-clipboard).
/// Pipes the trimmed transcript over `wl-copy`'s stdin with no trailing
/// newline, so it is pasted verbatim (UTF-8, including Polish diacritics) by
/// the subsequent paste step. Passes `--sensitive` (`x-kde-passwordManagerHint=secret`)
/// so Klipper skips clipboard history while paste still works.
/// Returns the `io::Error` unwrapped (not `anyhow`)
/// so the caller can match `NotFound` and disable itself with a single warning
/// on machines without wl-clipboard.
pub(crate) fn copy_to_clipboard(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("wl-copy")
        .arg("--sensitive")
        .stdin(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("wl-copy stdin unavailable"))?;
    stdin.write_all(text.as_bytes())?;
    drop(stdin);
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "wl-copy exited with {status}"
        )))
    }
}

/// Paste the clipboard at the current cursor position via `ydotool`
/// (requires the `ydotoold` daemon). Synthesizes a single Ctrl+V
/// (`29:1 47:1 47:0 29:0`, i.e. LEFTCTRL down, V down/up, LEFTCTRL up) instead
/// of typing the transcript key-by-key: a fixed 4-event sequence, so it is
/// fast and preserves non-ASCII text already on the clipboard. Returns
/// `io::Error` unwrapped so the caller can match `NotFound` and disable
/// itself with a single warning on machines without `ydotool`.
pub(crate) fn paste_via_ydotool() -> std::io::Result<()> {
    use std::process::Command;
    let status = Command::new("ydotool")
        .args(["key", "29:1", "47:1", "47:0", "29:0"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ydotool exited with {status}"
        )))
    }
}

fn whisper_loop(
    rx: &Receiver<Utterance>,
    model_path: &Path,
    opts: &WhisperOptions,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
) -> Result<()> {
    // Run init; report the outcome once via `ready` so the main thread can
    // fail fast instead of capturing into a dead queue.
    match whisper_init_and_run(rx, model_path, opts, &ready) {
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
    let min_confidence = opts.min_confidence;
    let min_confidence_strong = opts.min_confidence_strong;
    // Same value `FrameProcessor::new` receives for the acoustic gate,
    // passed via `WhisperOptions` rather than threaded through the channel:
    // keeps `Utterance` free of config, only evidence.
    let min_speech_rms = opts.min_speech_rms;
    let target_rms = opts.target_rms;
    let debug_tokens = opts.debug_tokens;
    let dump_dir = opts.dump_dir.clone();
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        eprintln!("[sttd] dumping utterances to {}", dir.display());
    }
    // Init succeeded (model + state + dump dir): signal ready so the main
    // thread knows captures aren't going into a dead queue.
    let _ = ready.send(Ok(()));
    let copy_enabled = opts.copy_enabled;
    let paste_enabled = opts.paste_enabled;
    // Clipboard (`wl-copy`) carries the transcript verbatim (UTF-8, so Polish
    // diacritics survive), then a single Ctrl+V via `ydotool key` pastes it.
    // Both flags still gate the action (AND semantics preserve the old
    // behavior where `--copy=false` disabled everything and `--paste=false`
    // disabled insertion). Each becomes `false` permanently after "binary not
    // found" so a machine without the binary logs one warning instead of one
    // per utterance; daemon-down failures keep retrying since `ydotoold` may
    // start later.
    let mut clipboard_available = copy_enabled;
    let mut paste_available = paste_enabled;
    if !copy_enabled {
        eprintln!("[sttd] clipboard copy disabled (--copy=false)");
    }
    if !paste_enabled {
        eprintln!("[sttd] auto-paste disabled (--paste=false)");
    } else if !copy_enabled {
        eprintln!(
            "[sttd] warning: --paste is enabled but --copy=false, paste will be skipped (nothing fresh to paste)"
        );
    }
    // Both `wl-copy` and the paste target only make sense on Wayland; on X.org
    // (or headless/`--file` without a Wayland socket) disable once with a
    // warning rather than failing every utterance.
    if (clipboard_available || paste_available) && !is_wayland_session_env() {
        eprintln!(
            "[sttd] warning: auto-paste is enabled but this is not a Wayland session (no WAYLAND_DISPLAY, XDG_SESSION_TYPE!=wayland); disabling clipboard + ydotool paste"
        );
        clipboard_available = false;
        paste_available = false;
    }

    let mut utterance_no: u64 = 0;
    for utterance in rx.iter() {
        if utterance.pcm.is_empty() {
            continue;
        }
        utterance_no += 1;
        let tag = format!("utt_{utterance_no:06}");

        if let Some(dir) = &dump_dir {
            let wav_path = dir.join(format!("{tag}.wav"));
            if let Err(e) = write_wav_mono16(&wav_path, &utterance.pcm, WHISPER_RATE) {
                eprintln!("[sttd] failed to dump {}: {e:?}", wav_path.display());
            }
        }

        let gained = apply_gain(
            &utterance.pcm,
            utterance.evidence.rms,
            target_rms,
            MAX_GAIN,
        );

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
                if trimmed.is_empty() {
                    dump_drop_reason(&dump_dir, &tag, "DROPPED(empty)");
                    continue;
                }
                let (floor, strength) = confidence_floor(
                    &utterance.evidence,
                    min_speech_rms,
                    min_confidence,
                    min_confidence_strong,
                );
                eprintln!(
                    "[sttd] {tag}: avg_p={:.3} avg_logprob={:.3} threshold={floor:.3} \
                     ({strength} evidence, density={:.3})",
                    t.confidence.prob,
                    t.confidence.logprob,
                    utterance.evidence.density()
                );
                if t.confidence.prob < floor {
                    eprintln!("[sttd] {tag}: dropped low-confidence transcript: {trimmed:?}");
                    dump_drop_reason(
                        &dump_dir,
                        &tag,
                        &format!(
                            "DROPPED(low-conf avg_p={:.3} < {floor:.3}, {strength}): {trimmed}",
                            t.confidence.prob
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
                if is_repetition(trimmed) {
                    eprintln!("[sttd] {tag}: dropped repetition-loop decode: {trimmed:?}");
                    dump_drop_reason(
                        &dump_dir,
                        &tag,
                        &format!("DROPPED(repetition): {trimmed}"),
                    );
                    continue;
                }
                println!("{trimmed}");
                // Track whether this utterance actually refreshed the
                // clipboard: paste synthesizes Ctrl+V, so it must only run
                // on fresh content, never on stale clipboard or a failed
                // copy.
                let mut copied = false;
                if clipboard_available {
                    match copy_to_clipboard(trimmed) {
                        Ok(()) => copied = true,
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            eprintln!(
                                "[sttd] warning: --copy is enabled but `wl-copy` was not found on PATH; disabling clipboard copy"
                            );
                            clipboard_available = false;
                        }
                        Err(e) => eprintln!("[sttd] {tag}: clipboard copy failed: {e:?}"),
                    }
                }
                if paste_available && copied {
                    match paste_via_ydotool() {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            eprintln!(
                                "[sttd] warning: --paste is enabled but `ydotool` was not found on PATH; disabling auto-paste"
                            );
                            paste_available = false;
                        }
                        Err(e) => eprintln!("[sttd] {tag}: auto-paste failed (is ydotoold running?): {e:?}"),
                    }
                }
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

/// Fold a single Polish diacritic to its ASCII base letter; every other char is
/// returned unchanged. Applied before comparing against
/// `HALLUCINATION_PATTERNS` (themselves written without diacritics) so a
/// transcript like "Dziękuje za uwagę" — a plausible ASR misspelling of
/// "Dziękuję za uwagę" — still matches instead of sliding past on a single
/// ogonek.
pub(crate) fn fold_polish_diacritic(c: char) -> char {
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

/// Lowercase, fold Polish diacritics, turn punctuation into spaces and
/// collapse whitespace. Shared by [`is_hallucination`] and
/// [`is_repetition`] so both gates judge the same spelling.
pub(crate) fn normalize_transcript(text: &str) -> String {
    // Punctuation becomes a space (not dropped): "Amara.org" must normalize
    // to "amara org" to match the pattern list, not "amaraorg". Diacritics
    // are folded (see `fold_polish_diacritic`) before the alphanumeric check,
    // not after, since they're themselves alphanumeric and would otherwise
    // survive untouched.
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(fold_polish_diacritic)
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `text` (a full transcript) is a Whisper repetition-loop decode:
/// the same 1-2 words on repeat (e.g. the observed
/// "Współpraca w Współpraca w ..." emitted for a 300 ms noise burst with
/// `avg_p=0.741`, well clear of either confidence floor). Judged on
/// [`normalize_transcript`] words: 6+ words with at most half of them
/// unique. Short emphases ("nie nie nie", 3 words) stay exempt; genuine
/// 6-word sentences never approach a 0.5 uniqueness ratio.
pub(crate) fn is_repetition(text: &str) -> bool {
    let normalized = normalize_transcript(text);
    let words: Vec<&str> = normalized.split(' ').filter(|w| !w.is_empty()).collect();
    if words.len() < 6 {
        return false;
    }
    let mut unique = words.to_vec();
    unique.sort_unstable();
    unique.dedup();
    unique.len() * 2 <= words.len()
}

/// Whether `text` (a full transcript) is a known Whisper hallucination on
/// silence/noise, checked as an exact match after normalization so genuine
/// sentences that merely contain e.g. "dziękuję" are never caught.
pub(crate) fn is_hallucination(text: &str) -> bool {
    let normalized = normalize_transcript(text);
    if normalized.is_empty() {
        return true; // pure punctuation, e.g. "..."
    }
    HALLUCINATION_PATTERNS.contains(&normalized.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
        DEFAULT_TARGET_RMS, MAX_GAIN, MIN_SPEECH_FRAMES, WHISPER_MIN_SAMPLES,
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
        let (floor, strength) = confidence_floor(
            &evidence,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
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
        let (floor, strength) = confidence_floor(
            &evidence,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
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
    fn wayland_session_detection_prefers_wayland_display() {
        // Plain Wayland, X11-only, XWayland (both DISPLAY and
        // WAYLAND_DISPLAY set counts as Wayland), and headless.
        assert!(is_wayland_session(Some("wayland"), Some("wayland-0")));
        assert!(is_wayland_session(Some("wayland"), None));
        assert!(is_wayland_session(None, Some("wayland-0")));
        assert!(is_wayland_session(Some("WAYLAND"), None));
        assert!(!is_wayland_session(Some("x11"), None));
        assert!(!is_wayland_session(Some("x11"), Some("")));
        assert!(!is_wayland_session(None, None));
        assert!(!is_wayland_session(Some(""), Some("  ")));
    }

    #[test]
    fn whisper_worker_signals_init_failure() {
        // Bad model path must propagate via the ready channel instead of
        // silently capturing into a dead queue.
        let (_tx, rx) = crossbeam_channel::unbounded();
        let opts = WhisperOptions {
            prompt: String::new(),
            language: Language::Polish,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            min_confidence_strong: DEFAULT_MIN_CONFIDENCE_STRONG,
            min_speech_rms: DEFAULT_MIN_SPEECH_RMS,
            target_rms: DEFAULT_TARGET_RMS,
            debug_tokens: false,
            dump_dir: None,
            copy_enabled: false,
            paste_enabled: false,
        };
        let bad = std::path::PathBuf::from("/nonexistent/ggml-bogus.bin");
        let (handle, ready) = spawn_whisper_worker(rx, bad, opts);
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
