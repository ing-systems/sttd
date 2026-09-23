// ---------------------------------------------------------------------------
// Sherpa-onnx engine family: offline Nemo-transducer / Canary / Whisper behind
// `asr::SpeechModel` (Canary serves `sherpa-canary-1b-v2-int8`, the community
// Sarphix int8 export of `nvidia/canary-1b-v2`; Polish needs a native lib
// newer than 1.13.8, see `SherpaKind`), plus VAD-driven simulated streaming
// where sherpa's Silero VAD owns the cuts.
// ---------------------------------------------------------------------------
//
// Two modes share the weights (`weights::ModelRef::Sherpa`) but not the
// worker:
//
// * offline (`SherpaMode::Offline`): `SherpaOfflineModel` implements
//   `asr::SpeechModel`, so the utterance pipeline (`FrameProcessor` ->
//   `Utterance` queue -> `asr` worker -> gates -> emitter) is byte-identical
//   to the parakeet-rs path. Only load + inference are sherpa-specific.
// * streaming (`SherpaMode::Streaming`): `spawn_sherpa_streaming_worker`
//   feeds 16 kHz samples straight into sherpa's `VoiceActivityDetector`;
//   each VAD segment decodes through the same `OfflineRecognizer` (this is
//   sherpa's endorsed streaming path for NeMo models — they ship no true
//   online encoder, and their own APKs do exactly VAD + offline decode).
//   Finals run the same transcript gates + emitter as offline; interim
//   partials only log (never clipboard/paste).
//
// Like parakeet-rs, decoding is greedy with no per-token probability, so
// both modes report `confidence: None` — the confidence-floor arm stays a
// no-op and hallucination/repetition/empty gates do the work.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rtrb::Consumer;

use crate::asr::{
    condition_utterance, format_metrics_row, pad_with_dither, should_autopause, should_drain_on_pause,
    should_unload, with_loading_ticker, AsrOptions, SpeechModel, Transcript,
};
use crate::config::{
    PARAKEET_MIN_SAMPLES, RTF_WARN_RTF, SHERPA_PARTIAL_EVERY_SECS, SHERPA_PARTIAL_MAX_SECS,
    SHERPA_VAD_MAX_SPEECH, SHERPA_VAD_WINDOW, WHISPER_RATE, SherpaKind,
};
use crate::emit::{Emission, Emitter};
use crate::gate::GateConfig;
use crate::utterance::Evidence;
use crate::weights::ModelRef;

// ---------------------------------------------------------------------------
// Mode + offline model
// ---------------------------------------------------------------------------

/// Sherpa decode mode, chosen once at startup via `--sherpa-mode` and fixed
/// for the process's lifetime. Only meaningful with a sherpa `--model`;
/// other engines ignore it (see `validate_sherpa_model_flags`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum SherpaMode {
    /// Utterance pipeline (`FrameProcessor` cuts, hangover closes).
    #[value(name = "offline")]
    Offline,
    /// Sherpa Silero VAD cuts (`spawn_sherpa_streaming_worker`).
    #[value(name = "streaming")]
    Streaming,
}

impl SherpaMode {
    /// Stable short spelling for log lines.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Offline => "offline",
            Self::Streaming => "streaming",
        }
    }
}

impl std::fmt::Display for SherpaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A loaded sherpa `OfflineRecognizer`, implementing `asr::SpeechModel` for
/// the offline mode. `kind` selects the sub-config (transducer vs Canary vs
/// Whisper); `lang` is Canary's src == tgt and sherpa-whisper's language
/// (transducer auto-detects and ignores it).
pub(crate) struct SherpaOfflineModel {
    inner: sherpa_onnx::OfflineRecognizer,
    /// Retried for decode-path selection later (e.g. per-family padding);
    /// today both families share the transducer floor.
    #[allow(dead_code)]
    kind: SherpaKind,
    /// Canary src == tgt; kept for the reload log symmetry with whisper.
    #[allow(dead_code)]
    lang: String,
    ready_line: String,
}

impl SherpaOfflineModel {
    /// Load from `dir` (already guaranteed to hold the catalog assets by
    /// `weights::ensure_available`). Always CPU: the crate's default static
    /// prebuilt ships no CUDA EP (sherpa itself logs `Fallback to cpu` when
    /// asked), so `--gpu` is reported in `unsupported_sherpa_flags` instead
    /// of attempted.
    pub(crate) fn load(dir: &Path, kind: SherpaKind, opts: &AsrOptions) -> Result<Self> {
        let intra_threads = opts.n_threads.max(1);
        let config = offline_config(dir, kind, &opts.sherpa_lang, intra_threads, "cpu");
        let name = match kind {
            SherpaKind::NemoTransducer => "sherpa-parakeet",
            SherpaKind::Canary => "sherpa-canary",
            SherpaKind::Whisper => "sherpa-whisper",
        };
        log::info!("loading {name} model ({}, provider=cpu)...", dir.display());
        match with_loading_ticker("sherpa model", || {
            sherpa_onnx::OfflineRecognizer::create(&config)
                .context("failed to create sherpa recognizer (cpu)")
        }) {
            Ok(inner) => Ok(Self {
                inner,
                kind,
                lang: opts.sherpa_lang.clone(),
                ready_line: format!(
                    "sherpa ready (model={}, kind={kind:?}, lang={}, threads={}, provider=cpu)",
                    opts.model_spec, opts.sherpa_lang, opts.n_threads,
                ),
            }),
            Err(e) => {
                // Canary-1B-v2 covers 25 languages, but the 1.13.8 native lib
                // rejects anything outside en/de/es/fr in `Validate()` — Polish
                // needs upstream PR #3962 (no crates.io release carries it yet).
                // Surface the native version so the failure is actionable.
                if kind == SherpaKind::Canary {
                    anyhow::bail!(
                        "failed to load sherpa model (native v{}): {e:#} \
                         (Canary note: sherpa-onnx <= 1.13.8 only accepts \
                         en/de/es/fr; '{}' needs a newer native lib)",
                        sherpa_onnx::version(),
                        opts.sherpa_lang,
                    )
                }
                anyhow::bail!("failed to load sherpa model: {e:#}")
            }
        }
    }

    /// Decode already-conditioned 16 kHz mono audio through a fresh stream.
    fn decode(&self, pcm_16k: &[f32]) -> Result<String> {
        // Same floor as `parakeet-rs`: the TDT encoder's 8x subsampling needs
        // frames to chew on; Canary and Whisper get the same pad harmlessly.
        let padded;
        let pcm = if pcm_16k.len() < PARAKEET_MIN_SAMPLES {
            padded = pad_with_dither(pcm_16k, PARAKEET_MIN_SAMPLES);
            &padded[..]
        } else {
            pcm_16k
        };
        let stream = self.inner.create_stream();
        stream.accept_waveform(WHISPER_RATE as i32, pcm);
        self.inner.decode(&stream);
        let result = stream
            .get_result()
            .context("sherpa decode produced no result")?;
        Ok(result.text)
    }
}

impl SpeechModel for SherpaOfflineModel {
    fn transcribe(
        &mut self,
        pcm_16k: &[f32],
        _prompt: &str,
        running: &Arc<AtomicBool>,
    ) -> Result<Transcript> {
        // No abort callback (same class as `parakeet-rs`): a pre-call check
        // avoids needless work after shutdown was requested.
        if !running.load(Ordering::Relaxed) {
            anyhow::bail!("shutdown requested");
        }
        Ok(Transcript {
            text: self.decode(pcm_16k)?,
            confidence: None,
        })
    }

    fn uses_prompt(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        self.ready_line.clone()
    }

    fn describe_reload(&self) -> String {
        format!("{} (reloaded after pause)", self.ready_line)
    }

    fn engine_name(&self) -> &'static str {
        "sherpa"
    }

    fn decoded_sample_count(&self, conditioned_len: usize) -> usize {
        conditioned_len.max(PARAKEET_MIN_SAMPLES)
    }
}

/// Pure `OfflineRecognizerConfig` builder (no recognizer created), so tests
/// don't touch the native lib. Transducer fills encoder/decoder/joiner with
/// `model_type = "nemo_transducer"`; Canary fills encoder/decoder + src/tgt
/// lang with `model_type` empty; Whisper fills encoder/decoder + language/
/// task with `model_type` empty (sherpa dispatches on the filled sub-config —
/// see its `canary.html` / `large-v3.html` CLI dumps, `model_type=""`).
pub(crate) fn offline_config(
    dir: &Path,
    kind: SherpaKind,
    lang: &str,
    num_threads: i32,
    provider: &str,
) -> sherpa_onnx::OfflineRecognizerConfig {
    use sherpa_onnx::{
        OfflineCanaryModelConfig, OfflineRecognizerConfig, OfflineTransducerModelConfig,
        OfflineWhisperModelConfig,
    };
    let mut config = OfflineRecognizerConfig::default();
    let s = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
    match kind {
        SherpaKind::NemoTransducer => {
            config.model_config.transducer = OfflineTransducerModelConfig {
                encoder: s("encoder.int8.onnx"),
                decoder: s("decoder.int8.onnx"),
                joiner: s("joiner.int8.onnx"),
            };
            config.model_config.model_type = Some("nemo_transducer".into());
            config.model_config.tokens = s("tokens.txt");
        }
        SherpaKind::Canary => {
            config.model_config.canary = OfflineCanaryModelConfig {
                encoder: s("encoder.int8.onnx"),
                decoder: s("decoder.int8.onnx"),
                src_lang: Some(lang.to_string()),
                tgt_lang: Some(lang.to_string()),
                use_pnc: true,
            };
            config.model_config.tokens = s("tokens.txt");
        }
        SherpaKind::Whisper => {
            config.model_config.whisper = OfflineWhisperModelConfig {
                encoder: s("turbo-encoder.int8.onnx"),
                decoder: s("turbo-decoder.int8.onnx"),
                language: Some(lang.to_string()),
                task: Some("transcribe".into()),
                tail_paddings: -1,
                enable_token_timestamps: false,
                enable_segment_timestamps: false,
            };
            config.model_config.tokens = s("turbo-tokens.txt");
        }
    }
    config.model_config.num_threads = num_threads;
    config.model_config.provider = Some(provider.to_string());
    config
}

// ---------------------------------------------------------------------------
// Streaming worker (VAD owns the cuts)
// ---------------------------------------------------------------------------

/// Bounded sample queue for the streaming worker: 30 s of 16 kHz mono.
/// Decode runs far below realtime (RTF 0.03-0.3), so this only fills when
/// the worker is paused/unloaded — and the paused main loop drops raw before
/// it ever gets here.
pub(crate) const SHERPA_SAMPLE_QUEUE_CAP: usize = 16_000 * 30;

/// Samples stashed while the model is unloaded (resume raced the reload):
/// ~60 s cap, newest dropped past it with a warn. In practice stays empty —
/// the paused main loop discards instead of feeding.
const SHERPA_STASH_CAP: usize = 16_000 * 60;

/// Tunables the streaming worker needs, resolved once from `Cli`.
/// Mirrors the `AsrOptions` subset the offline path uses, plus the VAD
/// cut parameters and the partials flag.
#[derive(Debug, Clone)]
pub(crate) struct SherpaStreamOptions {
    pub(crate) model_spec: String,
    pub(crate) lang: String,
    pub(crate) gate: GateConfig,
    pub(crate) n_threads: i32,
    pub(crate) metrics_path: Option<PathBuf>,
    pub(crate) vad_threshold: f32,
    pub(crate) vad_min_silence: f32,
    pub(crate) vad_min_speech: f32,
    /// Log interim partials (`log::info`, never emission). Off by default:
    /// each partial is a full re-decode of the open segment.
    pub(crate) partials: bool,
}

/// Pure `VadModelConfig` builder (no native object created) for tests.
pub(crate) fn vad_config(
    vad_model: &Path,
    threshold: f32,
    min_silence: f32,
    min_speech: f32,
) -> sherpa_onnx::VadModelConfig {
    let mut config = sherpa_onnx::VadModelConfig::default();
    config.silero_vad.model = Some(vad_model.to_string_lossy().into_owned());
    config.silero_vad.threshold = threshold;
    config.silero_vad.min_silence_duration = min_silence;
    config.silero_vad.min_speech_duration = min_speech;
    config.silero_vad.max_speech_duration = SHERPA_VAD_MAX_SPEECH;
    config.silero_vad.window_size = SHERPA_VAD_WINDOW;
    config.sample_rate = WHISPER_RATE as i32;
    config
}

/// One loaded streaming session: VAD (owns the cuts) + recognizer. Created
/// and dropped inside the outer load/unload iteration, so unload is a plain
/// scope exit that frees ORT sessions.
struct StreamingSession {
    vad: sherpa_onnx::VoiceActivityDetector,
    recognizer: sherpa_onnx::OfflineRecognizer,
    kind: SherpaKind,
}

fn load_streaming_session(
    model: &ModelRef,
    vad_model: &Path,
    opts: &SherpaStreamOptions,
    provider: &str,
) -> Result<StreamingSession> {    let (dir, kind) = match model {
        ModelRef::Sherpa { dir, kind, .. } => (dir, *kind),
        _ => anyhow::bail!("sherpa streaming worker got a non-sherpa model"),
    };
    let vad = {
        let config = vad_config(
            vad_model,
            opts.vad_threshold,
            opts.vad_min_silence,
            opts.vad_min_speech,
        );
        with_loading_ticker("sherpa vad", || {
            sherpa_onnx::VoiceActivityDetector::create(&config, 30.0)
                .context("failed to create sherpa VAD")
        })?
    };
    let config = offline_config(dir, kind, &opts.lang, opts.n_threads.max(1), provider);
    let recognizer = with_loading_ticker("sherpa model", || {
        sherpa_onnx::OfflineRecognizer::create(&config)
            .context(format!("failed to create sherpa recognizer ({provider})"))
    })?;
    Ok(StreamingSession {
        vad,
        recognizer,
        kind,
    })
}

/// Acoustic evidence synthesized for a VAD-cut segment: the VAD already
/// removed the silence, so every transmitted frame counts as speech
/// (density 1.0) and `speech_frames == total_frames`. Only feeds
/// `decide_transcript`'s weak/strong split + logs — the acoustic gates
/// (`check_acoustic`) never run here; `min_speech_duration` is their
/// replacement. Pure for tests.
pub(crate) fn evidence_for_segment(pcm_16k: &[f32]) -> Evidence {
    const FRAME: usize = 160;
    let frames = (pcm_16k.len() / FRAME).max(1) as u32;
    let rms = if pcm_16k.is_empty() {
        0.0
    } else {
        let sum_sq: f64 = pcm_16k.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / pcm_16k.len() as f64).sqrt() as f32
    };
    Evidence {
        speech_frames: frames,
        total_frames: frames,
        rms,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_sherpa_streaming_worker(
    rx: Consumer<f32>,
    model: ModelRef,
    vad_model: PathBuf,
    opts: SherpaStreamOptions,
    emitter: impl Emitter + Send + 'static,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    auto_unload: bool,
    auto_unload_after_secs: u64,
    sound_enabled: bool,
    autopause_enabled: bool,
) -> (
    JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("sherpa-stream".into())
        .spawn(move || {
            let mut emitter = emitter;
            let mut rx = rx;
            if let Err(e) = sherpa_stream_init_and_run(
                &mut rx,
                &model,
                &vad_model,
                &opts,
                &ready_tx,
                &mut emitter,
                &paused,
                &running,
                auto_unload,
                auto_unload_after_secs,
                sound_enabled,
                autopause_enabled,
            ) {
                log::error!("sherpa streaming worker failed: {e:?}");
            }
        })
        .expect("failed to spawn sherpa streaming thread");
    (handle, ready_rx)
}

#[allow(clippy::too_many_arguments)]
fn sherpa_stream_init_and_run(
    rx: &mut Consumer<f32>,
    model: &ModelRef,
    vad_model: &Path,
    opts: &SherpaStreamOptions,
    ready: &std::sync::mpsc::Sender<Result<(), String>>,
    emitter: &mut impl Emitter,
    paused: &Arc<AtomicBool>,
    running: &Arc<AtomicBool>,
    auto_unload: bool,
    auto_unload_after_secs: u64,
    sound_enabled: bool,
    autopause_enabled: bool,
) -> Result<()> {
    let unload_after = Duration::from_secs(auto_unload_after_secs);
    let mut metrics = open_metrics_log(opts.metrics_path.as_deref());
    let mut first_load = true;
    let mut utterance_no: u64 = 0;
    // Samples that arrived while unloaded (resume raced the reload).
    let mut stash: Vec<f32> = Vec::new();
    let mut stash_warned = false;

    loop {
        // CPU-only (see `SherpaOfflineModel::load`): a single attempt, no
        // provider loop.
        let session = load_streaming_session(model, vad_model, opts, "cpu");
        let session: StreamingSession = match session {
            Ok(s) => s,
            Err(e) => {
                let last_err = format!("{e:#}");
                if first_load {
                    let _ = ready.send(Err(last_err.clone()));
                    anyhow::bail!("sherpa streaming load failed: {last_err}");
                }
                log::warn!("sherpa streaming reload failed; retrying in 5s");
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }
        };

        if first_load {
            log::info!(
                "sherpa streaming ready (model={}, kind={:?}, mode=streaming, threads={})",
                opts.model_spec,
                session.kind,
                opts.n_threads,
            );
            let _ = ready.send(Ok(()));
            first_load = false;
        } else {
            log::info!("sherpa streaming ready (reloaded after pause)");
        }
        if sound_enabled && !paused.load(Ordering::Relaxed) {
            crate::sound::play_ready();
        }
        let mut last_unpaused = Instant::now();
        let mut was_paused = paused.load(Ordering::Relaxed);

        // Replay stashed samples through the fresh VAD first; any cuts
        // they form drain in the loaded loop below.
        if !stash.is_empty() {
            let stashed = std::mem::take(&mut stash);
            stash_warned = false;
            feed_vad_windows(&session.vad, &stashed);
        }

        // In-progress audio since the last final (for interim partials) +
        // VAD window remainder (< 512 samples).
        let mut pending: Vec<f32> = Vec::new();
        let mut window_tail: Vec<f32> = Vec::new();
        let mut last_partial = Instant::now();

        let disconnected = loop {
            if paused.load(Ordering::Relaxed) {
                if should_drain_on_pause(was_paused, true) {
                    let mut n = 0;
                    while rx.pop().is_ok() {
                        n += 1;
                    }
                    if n > 0 {
                        log::warn!("dropping {n} queued sample(s) (paused)");
                    }
                    session.vad.reset();
                    pending.clear();
                    window_tail.clear();
                }
                std::thread::sleep(Duration::from_millis(50));
                was_paused = true;
                if should_unload(auto_unload, true, last_unpaused.elapsed(), unload_after) {
                    log::info!(
                        "paused {}s, unloading sherpa streaming session to free VRAM...",
                        unload_after.as_secs()
                    );
                    break false;
                }
                if rx.is_abandoned() {
                    let mut n = 0;
                    while rx.pop().is_ok() {
                        n += 1;
                    }
                    if n > 0 {
                        log::warn!("dropping {n} queued sample(s) (shut down while paused)");
                    }
                    break true;
                }
                continue;
            }
            // Drain all available samples (non-blocking); park 20 ms when
            // idle, same cadence as the utterance worker.
            let mut got = 0;
            while let Ok(s) = rx.pop() {
                pending.push(s);
                window_tail.push(s);
                got += 1;
                // Feed the VAD in exact windows; keep the remainder.
                while window_tail.len() >= SHERPA_VAD_WINDOW as usize {
                    let end = SHERPA_VAD_WINDOW as usize;
                    let window: Vec<f32> = window_tail.drain(..end).collect();
                    session.vad.accept_waveform(&window);
                }
            }
            if got > 0 {
                last_unpaused = Instant::now();
                if sound_enabled && was_paused {
                    crate::sound::play_ready();
                }
                was_paused = false;
            } else if !rx.is_abandoned() {
                std::thread::sleep(Duration::from_millis(20));
                let now_paused = paused.load(Ordering::Relaxed);
                if !now_paused {
                    last_unpaused = Instant::now();
                    if sound_enabled && was_paused {
                        crate::sound::play_ready();
                    }
                    was_paused = false;
                }
                if should_unload(auto_unload, now_paused, last_unpaused.elapsed(), unload_after) {
                    log::info!(
                        "paused {}s, unloading sherpa streaming session to free VRAM...",
                        unload_after.as_secs()
                    );
                    break false;
                }
            }
            // End of input (file drained / shutdown): flush trailing speech
            // into a final cut on every pass, so the last stretch closes
            // even without `min_silence_duration` of trailing audio.
            // End of input (file drained / shutdown): flush trailing speech
            // into a final cut on every pass, so the last stretch closes
            // even without `min_silence_duration` of trailing audio.
            if rx.is_abandoned() {
                session.vad.flush();
            }
            // Bound pure-silence growth: with no speech ever detected,
            // `pending` would grow forever — keep a 320 ms run-up (same job
            // as pre-roll) and drop the rest.
            if !session.vad.detected() && pending.len() > 5120 && session.vad.is_empty() {
                let drop = pending.len() - 5120;
                pending.drain(..drop);
            }
            // Interim partial: full re-decode of the open segment, throttled
            // and bounded. Display only (`log::info`), never emission.
            if opts.partials
                && session.vad.detected()
                && last_partial.elapsed().as_secs_f32() >= SHERPA_PARTIAL_EVERY_SECS
                && pending.len() < (SHERPA_PARTIAL_MAX_SECS * WHISPER_RATE as f32) as usize
                && running.load(Ordering::Relaxed)
            {
                last_partial = Instant::now();
                let conditioned = condition_utterance(&pending);
                if let Ok(text) = decode_samples(&session, &conditioned) {
                    let text = text.trim();
                    if !text.is_empty() {
                        log::info!("partial: {text}");
                    }
                }
            }
            // Finals: every VAD segment decodes through gates + emitter.
            let mut closed_any = false;
            while let Some(segment) = session.vad.front() {
                let pcm: Vec<f32> = segment.samples().to_vec();
                session.vad.pop();
                closed_any = true;
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                utterance_no += 1;
                let emitted = emit_stream_final(
                    &session,
                    &pcm,
                    utterance_no,
                    &opts.gate,
                    running,
                    emitter,
                    &mut metrics,
                );
                if should_autopause(autopause_enabled, emitted) {
                    paused.store(true, Ordering::Release);
                    was_paused = true;
                    log::info!(
                        "autopaused after utt_{utterance_no:06} (--autopause, single-shot)"
                    );
                    if sound_enabled {
                        crate::sound::play_paused();
                    }
                }
            }
            if closed_any {
                pending.clear();
                last_partial = Instant::now();
            }
            // Exit only once the sample queue is dry *and* no cuts remain:
            // `vad.is_empty` alone is not enough — this pass may have popped
            // samples that are still forming a cut, and abandonment may race
            // a partially-drained queue (exiting then would strand unpopped
            // samples, including the trailing silence that closes the cut).
            // A sub-`min_speech_duration` blip may sit in `pending`
            // undecoded — same call as the offline acoustic gate
            // (`MIN_SPEECH_FRAMES` ≈ 200 ms vs 250 ms here): skip it.
            if rx.is_abandoned() && got == 0 && session.vad.is_empty() {
                break true;
            }
        };
        if disconnected {
            flush_metrics(&mut metrics);
            return Ok(());
        }
        drop(session);
        log::info!("sherpa streaming session unloaded (paused, VRAM freed)");

        // Unloaded wait: stash late arrivals (capped, oldest dropped past
        // the cap so replay favors the most recent speech), reload on resume.
        loop {
            match rx.pop() {
                Ok(s) => {
                    if stash.len() >= SHERPA_STASH_CAP {
                        if !stash_warned {
                            log::warn!("sherpa stash full, dropping oldest samples");
                            stash_warned = true;
                        }
                        stash.remove(0);
                    }
                    stash.push(s);
                }
                Err(_) => {
                    if rx.is_abandoned() {
                        if !stash.is_empty() {
                            log::warn!(
                                "dropping {} queued sample(s) (shut down while unloaded)",
                                stash.len()
                            );
                        }
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if !paused.load(Ordering::Relaxed) {
                log::info!("resumed, reloading sherpa streaming session...");
                break;
            }
        }
    }
}

/// Feed whole windows from a slice, dropping the sub-window remainder.
/// Used for stash replay (remainder rejoins the live stream's tail).
fn feed_vad_windows(vad: &sherpa_onnx::VoiceActivityDetector, samples: &[f32]) {
    let mut off = 0;
    while off + SHERPA_VAD_WINDOW as usize <= samples.len() {
        vad.accept_waveform(&samples[off..off + SHERPA_VAD_WINDOW as usize]);
        off += SHERPA_VAD_WINDOW as usize;
    }
}

/// Decode conditioned samples through the session's recognizer (fresh
/// stream per call — `OfflineRecognizer` streams are single-use).
fn decode_samples(session: &StreamingSession, conditioned: &[f32]) -> Result<String> {
    let padded;
    let pcm = if conditioned.len() < PARAKEET_MIN_SAMPLES {
        padded = pad_with_dither(conditioned, PARAKEET_MIN_SAMPLES);
        &padded[..]
    } else {
        conditioned
    };
    let stream = session.recognizer.create_stream();
    stream.accept_waveform(WHISPER_RATE as i32, pcm);
    session.recognizer.decode(&stream);
    let result = stream
        .get_result()
        .context("sherpa decode produced no result")?;
    Ok(result.text)
}

/// One VAD-cut segment through conditioning, inference, gate and emission.
/// Returns whether the transcript was actually emitted (drives
/// `--autopause` without seeing the gate).
fn emit_stream_final(
    session: &StreamingSession,
    pcm_16k: &[f32],
    utterance_no: u64,
    gate: &GateConfig,
    running: &Arc<AtomicBool>,
    emitter: &mut impl Emitter,
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> bool {
    let tag = format!("utt_{utterance_no:06}");
    let conditioned = condition_utterance(pcm_16k);
    let t0 = Instant::now();
    let result = if !running.load(Ordering::Relaxed) {
        Err(anyhow::anyhow!("shutdown requested"))
    } else {
        decode_samples(session, &conditioned)
    };
    let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // What was actually decoded (post-pad), so RTF reflects the model input.
    let decoded_samples = conditioned.len().max(PARAKEET_MIN_SAMPLES);
    let (audio_sec, rtf) = crate::asr::infer_rtf(decoded_samples, infer_ms);
    log::info!(
        "{tag}: transcribe took {infer_ms:.0} ms ({audio_sec:.1} s audio, rtf={rtf:.2})"
    );
    if rtf > RTF_WARN_RTF {
        log::warn!(
            "{tag}: inference slower than realtime (rtf={rtf:.2}, {infer_ms:.0} ms for \
             {audio_sec:.1} s audio): consider --threads 0, or CPU/GPU provider"
        );
    }
    match result {
        Ok(text) => {
            let trimmed = text.trim();
            let evidence = evidence_for_segment(&conditioned);
            let verdict = gate.decide_transcript(
                evidence.speech_frames,
                evidence.rms,
                trimmed,
                None,
            );
            let emitted = matches!(verdict, crate::gate::TranscriptVerdict::Emit { .. });
            let row = format_metrics_row(
                &tag,
                decoded_samples,
                &evidence,
                None,
                None,
                verdict,
                trimmed,
                infer_ms,
            );
            write_metrics_row(metrics, &row);
            emitter.emit(Emission {
                tag: &tag,
                pcm: pcm_16k,
                text: trimmed,
                verdict,
                prob: None,
                logprob: None,
                evidence: &evidence,
            });
            emitted
        }
        Err(e) => {
            if !running.load(Ordering::Relaxed) {
                log::info!("{tag}: transcription aborted (shutdown)");
            } else {
                log::error!("{tag}: transcription failed: {e:?}");
            }
            false
        }
    }
}

/// Open the `--metrics` JSONL log for appending (creating parent dirs).
/// Warn-and-disable on error — observability never fails transcription.
/// (Mirrors `asr::open_metrics_log`, which is private to that module.)
fn open_metrics_log(path: Option<&std::path::Path>) -> Option<std::io::BufWriter<std::fs::File>> {
    let path = path?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("metrics: failed to create {}: {e:?}", parent.display());
                return None;
            }
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => {
            log::info!("metrics: appending per-utterance JSONL to {}", path.display());
            Some(std::io::BufWriter::new(f))
        }
        Err(e) => {
            log::warn!("metrics: failed to open {}: {e:?} (metrics disabled)", path.display());
            None
        }
    }
}

/// Append one pre-rendered row + flush (same durability contract as the
/// utterance worker: every final survives a hard kill right after).
fn write_metrics_row(
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
    row: &str,
) {
    use std::io::Write as _;
    let Some(w) = metrics else { return };
    if let Err(e) = writeln!(w, "{row}") {
        log::warn!("metrics: write failed: {e:?} (metrics disabled)");
        *metrics = None;
        return;
    }
    if let Err(e) = w.flush() {
        log::warn!("metrics: flush failed: {e:?} (metrics disabled)");
        *metrics = None;
    }
}

fn flush_metrics(metrics: &mut Option<std::io::BufWriter<std::fs::File>>) {
    use std::io::Write as _;
    if let Some(w) = metrics {
        let _ = w.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_SHERPA_MIN_SILENCE, DEFAULT_SHERPA_MIN_SPEECH, DEFAULT_SHERPA_VAD_THRESHOLD,
    };

    #[test]
    fn mode_spellings_are_stable() {
        assert_eq!(SherpaMode::Offline.as_str(), "offline");
        assert_eq!(SherpaMode::Streaming.as_str(), "streaming");
        assert_eq!(SherpaMode::Offline.to_string(), "offline");
        assert_eq!(SherpaMode::Streaming.to_string(), "streaming");
    }

    #[test]
    fn offline_config_selects_transducer_sub_config() {
        let dir = Path::new("models/sherpa-parakeet-tdt-0.6b-v3-int8");
        let config = offline_config(dir, SherpaKind::NemoTransducer, "pl", 8, "cpu");
        assert_eq!(
            config.model_config.transducer.encoder,
            Some("models/sherpa-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx".to_string())
        );
        assert_eq!(
            config.model_config.transducer.joiner,
            Some("models/sherpa-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx".to_string())
        );
        assert_eq!(config.model_config.model_type.as_deref(), Some("nemo_transducer"));
        assert_eq!(
            config.model_config.tokens,
            Some("models/sherpa-parakeet-tdt-0.6b-v3-int8/tokens.txt".to_string())
        );
        assert_eq!(config.model_config.num_threads, 8);
        assert_eq!(config.model_config.provider.as_deref(), Some("cpu"));
        // Canary sub-config stays empty so sherpa dispatches to transducer.
        assert_eq!(config.model_config.canary.encoder, None);
    }

    #[test]
    fn offline_config_selects_canary_sub_config() {
        let dir = Path::new("models/sherpa-canary-1b-v2-int8");
        let config = offline_config(dir, SherpaKind::Canary, "pl", 4, "cuda");
        assert_eq!(
            config.model_config.canary.encoder,
            Some("models/sherpa-canary-1b-v2-int8/encoder.int8.onnx".to_string())
        );
        assert!(config.model_config.canary.decoder.is_some());
        assert_eq!(config.model_config.canary.src_lang.as_deref(), Some("pl"));
        assert_eq!(config.model_config.canary.tgt_lang.as_deref(), Some("pl"));
        assert!(config.model_config.canary.use_pnc);
        assert_eq!(config.model_config.transducer.encoder, None);
        assert_eq!(config.model_config.provider.as_deref(), Some("cuda"));
    }

    #[test]
    fn offline_config_selects_whisper_sub_config() {
        let dir = Path::new("models/sherpa-whisper-large-v3-turbo-int8");
        let config = offline_config(dir, SherpaKind::Whisper, "pl", 8, "cpu");
        assert_eq!(
            config.model_config.whisper.encoder,
            Some("models/sherpa-whisper-large-v3-turbo-int8/turbo-encoder.int8.onnx".to_string())
        );
        assert_eq!(
            config.model_config.whisper.decoder,
            Some("models/sherpa-whisper-large-v3-turbo-int8/turbo-decoder.int8.onnx".to_string())
        );
        assert_eq!(config.model_config.whisper.language.as_deref(), Some("pl"));
        assert_eq!(config.model_config.whisper.task.as_deref(), Some("transcribe"));
        assert_eq!(
            config.model_config.tokens,
            Some("models/sherpa-whisper-large-v3-turbo-int8/turbo-tokens.txt".to_string())
        );
        assert_eq!(config.model_config.num_threads, 8);
        assert_eq!(config.model_config.provider.as_deref(), Some("cpu"));
        // Transducer/Canary sub-configs stay empty so sherpa dispatches to whisper.
        assert_eq!(config.model_config.transducer.encoder, None);
        assert_eq!(config.model_config.canary.encoder, None);
        assert!(config.model_config.model_type.is_none());
    }

    #[test]
    fn vad_config_maps_cut_parameters() {
        let vad = Path::new("models/silero_vad.onnx");
        let config = vad_config(
            vad,
            DEFAULT_SHERPA_VAD_THRESHOLD,
            DEFAULT_SHERPA_MIN_SILENCE,
            DEFAULT_SHERPA_MIN_SPEECH,
        );
        assert_eq!(config.silero_vad.model.as_deref(), Some("models/silero_vad.onnx"));
        assert!((config.silero_vad.threshold - 0.5).abs() < 1e-6);
        assert!((config.silero_vad.min_silence_duration - 0.5).abs() < 1e-6);
        assert!((config.silero_vad.min_speech_duration - 0.25).abs() < 1e-6);
        assert!((config.silero_vad.max_speech_duration - SHERPA_VAD_MAX_SPEECH).abs() < 1e-6);
        assert_eq!(config.silero_vad.window_size, SHERPA_VAD_WINDOW);
        assert_eq!(config.sample_rate, WHISPER_RATE as i32);
    }

    #[test]
    fn segment_evidence_counts_every_frame_as_speech() {
        // 1 s exact: 100 frames, density 1.0 by construction.
        let pcm = vec![0.01f32; 16_000];
        let ev = evidence_for_segment(&pcm);
        assert_eq!(ev.speech_frames, 100);
        assert_eq!(ev.total_frames, 100);
        assert!((ev.rms - 0.01).abs() < 1e-6);
        assert!((ev.density() - 1.0).abs() < 1e-6);
        // Empty decodes to the 1-frame floor, never div-by-zero.
        let ev = evidence_for_segment(&[]);
        assert_eq!(ev.speech_frames, 1);
        assert_eq!(ev.rms, 0.0);
    }

    #[test]
    fn sample_queue_cap_is_thirty_seconds() {
        assert_eq!(SHERPA_SAMPLE_QUEUE_CAP, 16_000 * 30);
    }

    #[test]
    fn stream_defaults_match_compiled_constants() {
        assert_eq!(DEFAULT_SHERPA_VAD_THRESHOLD, 0.5);
        assert!((SHERPA_PARTIAL_EVERY_SECS - 1.0).abs() < 1e-6);
        assert!((SHERPA_PARTIAL_MAX_SECS - 20.0).abs() < 1e-6);
    }
}
