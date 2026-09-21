//! Real-time Speech-to-Text pipeline.
//!
//! Data flow:
//! `cpal` (hw rate, stereo) -> downmix mono -> per ~10 ms raw chunk, two
//! independent `rubato` resamplings (`FrameProcessor`, see `WhisperResample`
//! for when the second collapses to one hop instead of two):
//!   - VAD branch:     `rubato` (-> 48 kHz) -> `nnnoiseless` denoise -> `rubato` (-> 16 kHz) -> `webrtc-vad`
//!   - whisper branch: `rubato` (-> 16 kHz, direct hop when possible) -> utterance buffer
//! An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive speech
//! frames (a single VAD false positive no longer opens one), with ~300 ms of
//! pre-roll prepended at that point so quiet word starts aren't clipped.
//! Once open, every frame is appended regardless of the VAD verdict (speech
//! or not) until `SILENCE_FRAMES_THRESHOLD` closes it — but only
//! `MIN_SPEECH_FRAMES` worth of actual VAD-flagged speech, trimmed to
//! `TAIL_KEEP_SAMPLES` of trailing silence, is what actually reaches whisper,
//! carried alongside its acoustic evidence (speech-frame count, RMS) as a
//! `Clip` -> `crossbeam-channel` queue -> isolated `whisper-rs` inference
//! thread (CUDA, large-v3-turbo): pre-inference gain toward `STTD_TARGET_RMS`
//! (`apply_gain`, quiet-but-real speech only, never attenuates) -> average
//! *text-token* probability gate (`average_text_confidence` excludes the
//! timestamp tokens every whisper.cpp segment carries; whisper.cpp's own
//! `no_speech_thold` being a no-op) at a threshold `confidence_threshold`
//! relaxes when the acoustic evidence already vouches for real speech ->
//! known-hallucination-boilerplate filter (`is_hallucination`).
//!
//! `sttd --file <wav>` runs the same pipeline offline against a WAV file
//! instead of the microphone, for comparing tuning changes against a fixed
//! recording. `sttd --list-devices` prints available input devices.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{unbounded, Receiver, Sender};
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};
use webrtc_vad::{Vad, VadMode};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sample rate required by `nnnoiseless` (RNNoise).
const DENOISE_RATE: u32 = 48_000;
/// Samples per 10 ms frame at 48 kHz (`DenoiseState::FRAME_SIZE`).
const FRAME_48K: usize = 480;
/// Sample rate required by whisper + VAD.
const WHISPER_RATE: u32 = 16_000;
/// Samples per 10 ms frame at 16 kHz.
const FRAME_16K: usize = 160;

/// Pre-roll ring buffer length: 300 ms = 30 x 10 ms frames. Prepended to
/// every utterance at onset so a quiet/unvoiced word start isn't clipped by
/// VAD's inherent onset latency.
const RING_FRAMES: usize = 30;

/// Silence hangover before cutting an utterance: 800 ms = 80 x 10 ms frames.
/// This only decides *when* to close an utterance; how much of that silence
/// actually reaches whisper is capped separately by `TAIL_KEEP_SAMPLES`.
const SILENCE_FRAMES_THRESHOLD: u32 = 80;
/// Consecutive speech frames required before an utterance is opened at all.
/// A single VAD false positive (a click, a breath, mains hum) used to be
/// enough to open a full utterance — padded out to ~1.2-1.9s by pre-roll and
/// silence hangover, which whisper then dutifully hallucinated a sentence
/// for. The onset frames themselves aren't lost: `preroll` already holds
/// them by the time the utterance opens, since it's updated every frame
/// regardless of the VAD verdict.
const ONSET_SPEECH_FRAMES: u32 = 3; // 30 ms
/// Total VAD-flagged speech frames (over the utterance's whole life, not
/// just at onset) required to actually send it to whisper. This is the gate
/// that replaces the old, dead `MIN_UTTERANCE_SAMPLES`: that constant
/// compared against the *buffer* length, which is dominated by pre-roll and
/// silence hangover and so was satisfied by every utterance, always.
const MIN_SPEECH_FRAMES: u32 = 20; // 200 ms
/// How much trailing silence (past the last speech frame) is kept when an
/// utterance is sent to whisper. `SILENCE_FRAMES_THRESHOLD` still decides
/// when to *close* the utterance; this trims how much of that hangover is
/// actually *transmitted*, so a one-word utterance isn't ~70% silence by the
/// time it reaches the model.
const TAIL_KEEP_SAMPLES: usize = 3_200; // 200 ms @ 16 kHz
/// Whisper refuses input whose *computed mel length* is under 1000 ms (see
/// whisper.cpp `whisper_full`: `if (seek_end < seek_start + 100)`). That
/// length isn't raw sample count / 16: mel framing pads and windows the
/// signal (`log_mel_spectrogram`: frame_size=400, hop=160, stage_2_pad=200),
/// so `mel.n_len_org = 1 + (n_samples + 200 - 400) / 160`. Exactly
/// `WHISPER_RATE` (16_000) samples yields only 990 ms this way and still
/// gets rejected — confirmed via `sttd --file` on a sub-second clip. The
/// true minimum is 16_040; padded well past it for margin.
const WHISPER_MIN_SAMPLES: usize = 16_400;
/// Force-flush long speech even without silence (whisper is ~30 s max).
const MAX_UTTERANCE_SAMPLES: usize = WHISPER_RATE as usize * 30;
/// Minimum RMS amplitude across an utterance's VAD-flagged speech frames
/// (silence padding excluded). Catches transient VAD false positives that
/// pass `ONSET_SPEECH_FRAMES`/`MIN_SPEECH_FRAMES` but carry essentially no
/// signal. ~-50 dBFS. Override with `STTD_MIN_RMS` while tuning against real
/// mic noise.
const DEFAULT_MIN_SPEECH_RMS: f32 = 0.003;
/// Target speech-only RMS for the pre-inference gain applied to a clip that
/// already cleared the acoustic gates (see `apply_gain`). Only ever
/// amplifies, never attenuates — quiet-but-real speech (e.g. the
/// rms=0.01178 case that prompted this) gets a boost toward whisper's
/// comfort zone. `0` disables gain entirely. Override with `STTD_TARGET_RMS`.
const DEFAULT_TARGET_RMS: f32 = 0.05;
/// Ceiling on the gain `apply_gain` will apply (8x = +18 dB), so a near-
/// silent clip that slipped past `STTD_MIN_RMS` doesn't get blown up into
/// clipped noise.
const MAX_GAIN: f32 = 8.0;
/// Minimum average whisper token probability for a transcript to be kept,
/// used when the utterance's *acoustic* evidence (see `Clip`) is weak.
/// whisper.cpp declares `no_speech_thold` but never reads it (`whisper.h`:
/// `float no_speech_thold; // TODO: not implemented` — confirmed against the
/// vendored whisper.cpp 1.7.1 source, `whisper_full_default_params` is its
/// only reference in the whole file), so whisper.cpp itself never rejects a
/// low-confidence segment; this is the actual gate. Override with
/// `STTD_MIN_CONF`.
const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;
/// Relaxed confidence floor used when the utterance already carries strong
/// acoustic evidence of real speech (see `confidence_threshold`). Whisper's
/// average token probability on genuine-but-hard speech regularly lands
/// around 0.4-0.5 — the same range a silence hallucination can land in — so
/// a single scalar threshold can't separate the two cases; VAD/RMS evidence
/// can. Override with `STTD_MIN_CONF_STRONG`.
const DEFAULT_MIN_CONFIDENCE_STRONG: f32 = 0.35;
/// `speech_frames` (see `Utterance`) required to count as "strong" acoustic
/// evidence: 500 ms of VAD-flagged speech, well past `MIN_SPEECH_FRAMES`.
const STRONG_SPEECH_FRAMES: u32 = 50;
/// Multiple of `min_speech_rms` required to count as "strong" acoustic
/// evidence.
const STRONG_RMS_FACTOR: f32 = 3.0;

/// Whisper model auto-download. The URL is derived from the filename (see
/// `ensure_model_available`), so `WHISPER_MODEL_PATH` can point at any
/// ggml model published in this repo, not just the compiled-in default.
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
const DEFAULT_MODEL_PATH: &str = "models/ggml-large-v3-turbo.bin";

/// Default `initial_prompt`: primes Polish orthography/diacritics only.
/// Used to carry a long list of IT vocabulary (Kubernetes, Docker, ...), but
/// whisper.cpp feeds `initial_prompt` into the decoder as prior context
/// (`prompt_past`), and on low/no-signal audio the model falls back on that
/// prior instead of the (absent) acoustic evidence — i.e. it would recite
/// from the prompt itself. Override with `STTD_PROMPT` to opt back into a
/// domain vocabulary at the cost of that risk.
const DEFAULT_PROMPT: &str = "Rozmowa techniczna po polsku.";

/// Transcripts that are near-exact matches for common Whisper hallucinations
/// on silence/low-energy audio (subtitle-credit boilerplate picked up from
/// training data), compared after lowercasing, stripping punctuation, and
/// folding Polish diacritics to their ASCII base letter (see
/// `fold_diacritics`) — so patterns are written here without diacritics too.
const HALLUCINATION_PATTERNS: &[&str] = &[
    "napisy stworzone przez spolecznosc amara org",
    "napisy",
    "napisy pl",
    "subtitles by",
    "dziekuje za obejrzenie",
    "dziekuje za uwage",
    "zapraszam do subskrypcji",
    "zapraszam do subskrypcji kanalu",
];

/// Read an f32 override from an env var, falling back to `default` if the
/// var is unset or fails to parse (logging a warning in the latter case, so
/// a typo'd override doesn't silently revert to the default).
fn env_f32(key: &str, default: f32) -> f32 {
    match std::env::var(key) {
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            eprintln!(
                "[sttd] warning: {key}={v:?} is not a valid number, using default {default}"
            );
            default
        }),
        Err(_) => default,
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-devices") {
        return list_devices();
    }
    if let Some(pos) = args.iter().position(|a| a == "--file") {
        let path = args
            .get(pos + 1)
            .context("--file requires a path argument")?;
        return run_file_mode(Path::new(path));
    }

    run_live_mode()
}

fn print_usage() {
    eprintln!(
        "sttd - real-time Polish speech-to-text\n\n\
         USAGE:\n\
         \x20   sttd                  listen on the default (or STTD_DEVICE) mic, print transcripts to stdout\n\
         \x20   sttd --file <path>    transcribe a WAV file offline (for tuning/comparison)\n\
         \x20   sttd --list-devices   list available input devices and exit\n\n\
         ENV:\n\
         \x20   WHISPER_MODEL_PATH    path to a ggml model (default: {DEFAULT_MODEL_PATH})\n\
         \x20   STTD_DEVICE           substring match (case-insensitive) against an input device name\n\
         \x20   STTD_PROMPT           override the whisper initial_prompt\n\
         \x20   STTD_DUMP_DIR         write each utterance as WAV+TXT into this directory (TXT written\n\
         \x20                         for dropped utterances too, prefixed with the drop reason)\n\
         \x20   STTD_MIN_RMS          min speech-only RMS to keep an utterance (default: {DEFAULT_MIN_SPEECH_RMS})\n\
         \x20   STTD_MIN_CONF         min avg token confidence, weak acoustic evidence (default: {DEFAULT_MIN_CONFIDENCE})\n\
         \x20   STTD_MIN_CONF_STRONG  min avg token confidence, strong acoustic evidence (default: {DEFAULT_MIN_CONFIDENCE_STRONG})\n\
         \x20   STTD_TARGET_RMS       pre-inference gain target RMS, 0 disables (default: {DEFAULT_TARGET_RMS})\n\
         \x20   STTD_DEBUG_TOKENS     1 = dump per-token id/text/p/plog/is_timestamp to stderr\n"
    );
}

fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());
        match device.default_input_config() {
            Ok(cfg) => println!("{name}  ({} Hz, {} ch)", cfg.sample_rate().0, cfg.channels()),
            Err(_) => println!("{name}  (config unavailable)"),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Live (microphone) mode
// ---------------------------------------------------------------------------

fn run_live_mode() -> Result<()> {
    // Cheap, fallible audio setup first (device pick, hw config, resampler,
    // stream) — *before* loading the model onto the GPU. Spawning the whisper
    // worker starts an async CUDA context init on its own thread; if any of
    // this setup then failed via `?`, the process would tear down mid-init
    // and race whisper.cpp's CUDA cleanup (observed: "CUDA error: driver
    // shutting down" abort on e.g. a mistyped STTD_DEVICE). Loading the model
    // only once we know the mic actually works avoids that race entirely,
    // and doesn't cost anything: capture starts immediately after and audio
    // simply queues up in `raw_rx` while the model loads in the background.
    let host = cpal::default_host();
    let device = select_input_device(&host)?;
    eprintln!(
        "[sttd] input device: {}",
        device.name().unwrap_or_else(|_| "<unknown>".into())
    );

    let supported = device
        .default_input_config()
        .context("failed to query default input config")?;
    let hw_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    eprintln!(
        "[sttd] hw config: {} Hz, {} ch, {:?}",
        hw_rate, channels, sample_format
    );

    let mut accumulator: Vec<f32> = Vec::with_capacity(hw_rate as usize * 2);

    // Raw mono captures (hw rate, f32 in [-1, 1]) from the cpal callback.
    // Unbounded + `send` (never blocks) keeps callback lock time minimal.
    let (raw_tx, raw_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

    // ---- build + start the cpal input stream (callback does NOT unwrap) ---
    let running = Arc::new(AtomicBool::new(true));
    let stream = build_input_stream(&device, &stream_config, channels, sample_format, raw_tx)?;
    stream.play().context("failed to start input stream")?;
    eprintln!("[sttd] listening... Ctrl-C to stop");

    // Ctrl-C sets the flag; the loop below flushes trailing speech first.
    {
        let running = Arc::clone(&running);
        if let Err(e) = ctrlc_handler(move || running.store(false, Ordering::SeqCst)) {
            eprintln!("[sttd] warning: {e:?} (Ctrl-C will terminate immediately)");
        }
    }

    // `FrameProcessor` owns all resampling (hw -> 48k for the VAD branch,
    // hw -> 16k for the whisper branch); building it is cheap, fallible
    // setup like everything above, so it happens here too — before the
    // model load below.
    let (whisper_tx, whisper_rx): (Sender<Clip>, Receiver<Clip>) = unbounded();
    let mut fp = FrameProcessor::new(hw_rate, whisper_tx)?;
    let input_frames = fp.input_frames();

    // ---- now the heavy part: resolve/download the model and load it onto
    // the GPU on its own thread. Capture is already running above.
    let model_path = ensure_model_available()?;
    eprintln!("[sttd] model: {}", model_path.display());
    let whisper_handle = spawn_whisper_worker(whisper_rx, model_path);

    // ---- main processing loop (NOT the audio callback) --------------------
    while running.load(Ordering::SeqCst) {
        // Drain everything the callback delivered since last iteration.
        let mut got_audio = false;
        while let Ok(chunk) = raw_rx.try_recv() {
            got_audio = true;
            accumulator.extend_from_slice(&chunk);
        }

        // Raw hw-rate chunks -> FrameProcessor (denoise/VAD + whisper
        // branches, all resampling included).
        while accumulator.len() >= input_frames {
            fp.process_frame(&accumulator[..input_frames]);
            accumulator.drain(..input_frames);
        }

        if !got_audio && accumulator.len() < input_frames {
            thread::sleep(Duration::from_millis(2));
        }
    }

    // Flush trailing speech, then wait for the whisper thread to actually
    // finish transcribing it (and any earlier backlog) before the process
    // exits — otherwise the last utterance can be silently dropped, and an
    // in-flight CUDA op can race process teardown the same way an early `?`
    // above used to.
    fp.finish();
    drop(fp);
    whisper_handle.join().ok();
    eprintln!("[sttd] stopped");
    Ok(())
}

/// Pick the input device: `STTD_DEVICE` (substring match, case-insensitive)
/// if set, otherwise the system default. Without this the tool is stuck on
/// whatever the OS considers the default mic (e.g. a webcam capsule).
fn select_input_device(host: &cpal::Host) -> Result<cpal::Device> {
    if let Ok(wanted) = std::env::var("STTD_DEVICE") {
        let wanted_lower = wanted.to_lowercase();
        let mut devices = host
            .input_devices()
            .context("failed to enumerate input devices")?;
        let found = devices.find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&wanted_lower))
                .unwrap_or(false)
        });
        return found
            .with_context(|| format!("STTD_DEVICE={wanted:?} matched no input device (see --list-devices)"));
    }
    host.default_input_device()
        .context("no default input device available")
}

/// Build an hw_rate -> `target_rate` resampler with a fixed ~10 ms input
/// chunk size, so its output lands on a round 10 ms frame count at
/// `target_rate` (480 for 48 kHz, 160 for 16 kHz). Shared by both
/// `FrameProcessor` resamplers (`resampler_hw48`, and the `Direct` variant
/// of `whisper_resample`) — one formula, so their input chunk sizes agree
/// whenever that's mathematically possible (see `WhisperResample`).
fn make_resampler(hw_rate: u32, target_rate: u32) -> Result<(FftFixedInOut<f32>, usize)> {
    let desired_in = ((hw_rate as f64 / 100.0).round() as usize).max(1);
    let resampler = FftFixedInOut::<f32>::new(hw_rate as usize, target_rate as usize, desired_in, 1)
        .with_context(|| format!("failed to create rubato resampler (hw -> {target_rate})"))?;
    let input_frames = resampler.input_frames_next();
    Ok((resampler, input_frames))
}

// ---------------------------------------------------------------------------
// Offline (file) mode
// ---------------------------------------------------------------------------

fn run_file_mode(path: &Path) -> Result<()> {
    // Same ordering rationale as run_live_mode: do the cheap, fallible setup
    // (read + validate the WAV, build the resampler) before loading the
    // model onto the GPU, so a bad file fails fast instead of racing
    // whisper.cpp's CUDA teardown.
    let wav = read_wav_mono_f32(path)?;
    eprintln!(
        "[sttd] file: {} ({} Hz, {} samples, {:.1}s)",
        path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f64 / wav.sample_rate as f64
    );

    let (whisper_tx, whisper_rx): (Sender<Clip>, Receiver<Clip>) = unbounded();
    let mut fp = FrameProcessor::new(wav.sample_rate, whisper_tx.clone())?;
    let input_frames = fp.input_frames();

    let model_path = ensure_model_available()?;
    eprintln!("[sttd] model: {}", model_path.display());
    let whisper_handle = spawn_whisper_worker(whisper_rx, model_path);

    let mut offset = 0usize;
    while offset + input_frames <= wav.samples.len() {
        fp.process_frame(&wav.samples[offset..offset + input_frames]);
        offset += input_frames;
    }
    // A trailing partial chunk (< input_frames samples, at most ~30 ms) is
    // dropped; fine for offline comparison runs.
    fp.finish();

    drop(fp);
    drop(whisper_tx);
    whisper_handle.join().ok();
    Ok(())
}

// ---------------------------------------------------------------------------
// Frame processing: denoise/VAD branch + whisper branch + utterance buffer
// ---------------------------------------------------------------------------

/// Owns all per-frame DSP/VAD/buffering state, including the hw-rate ->
/// 48 kHz / hw-rate -> 16 kHz resamplers (previously built and driven by the
/// caller). Fed one raw hw-rate chunk at a time (`input_frames()` samples)
/// by either the live capture loop or file mode, so both share identical
/// resampling and utterance-detection behavior.
struct FrameProcessor {
    // hw_rate -> 48 kHz (RNNoise's required rate). Always used to feed the
    // VAD branch, since denoise needs 48 kHz; its input chunk size
    // (`hw_input_frames`) is the raw hw-rate frame size every caller must
    // feed `process_frame` with.
    resampler_hw48: FftFixedInOut<f32>,
    resample_hw48_in: Vec<Vec<f32>>,
    resample_hw48_out: Vec<Vec<f32>>,
    hw_input_frames: usize,

    // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
    resampler_vad: FftFixedInOut<f32>,
    resample_vad_in: Vec<Vec<f32>>,
    resample_vad_out: Vec<Vec<f32>>,

    /// Whisper branch resampling: a direct hw_rate -> 16 kHz hop when the
    /// hw rate allows it (see `WhisperResample`), avoiding the redundant
    /// 48 kHz intermediate whisper's (non-denoised) signal never needed.
    whisper_resample: WhisperResample,

    denoise: Box<DenoiseState<'static>>,
    denoise_in: Vec<f32>,
    denoise_out: Vec<f32>,
    clean48_vad: Vec<f32>,

    vad: Vad,
    vad_i16: Vec<i16>,

    /// Rolling ~300 ms of whisper-branch audio, prepended to an utterance
    /// the instant it opens so onsets aren't clipped by VAD/onset latency.
    preroll: VecDeque<f32>,
    utterance: Utterance,
    /// `STTD_MIN_RMS` override (or `DEFAULT_MIN_SPEECH_RMS`), read once at
    /// startup and reused for every utterance.
    min_speech_rms: f32,

    whisper_tx: Sender<Clip>,
}

/// How the whisper branch gets from hw_rate down to 16 kHz.
///
/// `Direct` is one resampling hop instead of two (hw -> 48k -> 16k): the
/// whisper branch never uses the 48 kHz intermediate for anything (it
/// exists only so RNNoise, which is fixed at 48 kHz, can run for the VAD
/// branch), so paying for it was pure waste. It's only valid when a
/// hw_rate -> 16 kHz resampler consumes the *same* input chunk size as the
/// hw_rate -> 48 kHz one (verified for 16k/44.1k/48k/96k hw rates in
/// `dual_resamplers_agree_on_input_frames`) — otherwise the two branches
/// would desync in time. `Chained` is the original hw->48k->16k path,
/// reusing the VAD branch's already-resampled 48 kHz frame, kept as a
/// fallback for any hw rate where that assumption doesn't hold.
enum WhisperResample {
    Direct {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
    Chained {
        resampler: FftFixedInOut<f32>,
        in_buf: Vec<Vec<f32>>,
        out_buf: Vec<Vec<f32>>,
    },
}

/// Per-utterance buffering state, reset every time an utterance is flushed
/// (whether accepted or dropped). Kept as one struct rather than loose
/// `&mut` parameters so `flush_utterance` can reset everything atomically
/// and so new gates (see `flush_utterance`) can add fields without touching
/// every call site.
#[derive(Default)]
struct Utterance {
    /// Whisper-branch audio (16 kHz), from the moment the utterance opens.
    buf: Vec<f32>,
    /// Consecutive non-speech frames since the last speech frame, once the
    /// utterance is open. Drives the close decision (`SILENCE_FRAMES_THRESHOLD`).
    silence_frames: u32,
    /// Consecutive speech frames seen before the utterance has opened.
    /// Drives the open decision (`ONSET_SPEECH_FRAMES`); reset to 0 on any
    /// non-speech frame.
    onset_run: u32,
    /// Total VAD-flagged speech frames over the utterance's whole life
    /// (including pre-open onset frames). Drives `MIN_SPEECH_FRAMES`.
    speech_frames: u32,
    /// `buf.len()` as of the last speech frame appended. Used to trim
    /// trailing silence down to `TAIL_KEEP_SAMPLES` at flush time.
    samples_at_last_speech: usize,
    /// Sum of squared samples from VAD-flagged speech frames only (silence
    /// padding excluded), paired with `speech_sample_count` so
    /// `flush_utterance` can compute an RMS over just the speech content.
    speech_energy_sum_sq: f64,
    speech_sample_count: usize,
}

/// One utterance plus the acoustic evidence `flush_utterance` gathered for
/// it while buffering. The whisper worker needs this to decide how much
/// trust the acoustics already earned, before deciding how much trust to
/// demand from the model's own confidence (see `confidence_threshold`) —
/// evidence that used to be computed and then thrown away at the
/// `Sender<Vec<f32>>` boundary.
struct Clip {
    pcm: Vec<f32>,
    /// Total VAD-flagged speech frames over the utterance's life (see
    /// `Utterance::speech_frames`).
    speech_frames: u32,
    /// RMS over VAD-flagged speech frames only (see `Utterance::speech_energy_sum_sq`).
    rms: f32,
}

/// Confidence floor for this utterance's transcript. The confidence gate
/// exists to catch hallucinations on silence/noise; where the acoustics
/// already testify to real speech, it has nothing to add, and applying the
/// strict floor anyway discards the only transcript we have. It doesn't
/// disappear entirely: the relaxed floor still catches a repetition-loop
/// decode (avg_p ~0.1), which isn't a transcript of anything.
fn confidence_threshold(clip: &Clip, min_rms: f32, weak: f32, strong: f32) -> (f32, &'static str) {
    if clip.speech_frames >= STRONG_SPEECH_FRAMES && clip.rms >= min_rms * STRONG_RMS_FACTOR {
        (strong, "strong")
    } else {
        (weak, "weak")
    }
}

impl FrameProcessor {
    fn new(hw_rate: u32, whisper_tx: Sender<Clip>) -> Result<Self> {
        let (resampler_hw48, hw_input_frames) = make_resampler(hw_rate, DENOISE_RATE)
            .context("failed to create rubato resampler (hw -> 48k)")?;
        eprintln!(
            "[sttd] rubato: {hw_rate} -> {DENOISE_RATE}, in={hw_input_frames} out={}",
            resampler_hw48.output_frames_next()
        );
        let resample_hw48_in = resampler_hw48.input_buffer_allocate(true);
        let resample_hw48_out = resampler_hw48.output_buffer_allocate(true);

        let resampler_vad = FftFixedInOut::<f32>::new(
            DENOISE_RATE as usize,
            WHISPER_RATE as usize,
            FRAME_48K,
            1,
        )
        .context("failed to create rubato resampler (48k -> 16k, vad branch)")?;
        debug_assert_eq!(resampler_vad.input_frames_next(), FRAME_48K);
        debug_assert_eq!(resampler_vad.output_frames_next(), FRAME_16K);
        let resample_vad_in = resampler_vad.input_buffer_allocate(true);
        let resample_vad_out = resampler_vad.output_buffer_allocate(true);

        // Try a direct hw -> 16k hop for the whisper branch; only usable
        // when it consumes the same-sized raw hw chunk as the hw -> 48k
        // resampler above, since both are fed from the same raw stream in
        // lockstep (see `WhisperResample`).
        let whisper_resample = match make_resampler(hw_rate, WHISPER_RATE) {
            Ok((resampler, hw16_input_frames)) if hw16_input_frames == hw_input_frames => {
                debug_assert_eq!(resampler.output_frames_next(), FRAME_16K);
                let in_buf = resampler.input_buffer_allocate(true);
                let out_buf = resampler.output_buffer_allocate(true);
                WhisperResample::Direct {
                    resampler,
                    in_buf,
                    out_buf,
                }
            }
            other => {
                match &other {
                    Ok((_, hw16_input_frames)) => eprintln!(
                        "[sttd] hw->16k and hw->48k resamplers disagree on input chunk size \
                         ({hw16_input_frames} vs {hw_input_frames} samples) at {hw_rate} Hz; \
                         falling back to hw->48k->16k for the whisper branch"
                    ),
                    Err(e) => eprintln!(
                        "[sttd] failed to build a direct hw->16k resampler ({e}); falling back \
                         to hw->48k->16k for the whisper branch"
                    ),
                }
                let resampler = FftFixedInOut::<f32>::new(
                    DENOISE_RATE as usize,
                    WHISPER_RATE as usize,
                    FRAME_48K,
                    1,
                )
                .context("failed to create rubato resampler (48k -> 16k, whisper branch)")?;
                debug_assert_eq!(resampler.input_frames_next(), FRAME_48K);
                debug_assert_eq!(resampler.output_frames_next(), FRAME_16K);
                let in_buf = resampler.input_buffer_allocate(true);
                let out_buf = resampler.output_buffer_allocate(true);
                WhisperResample::Chained {
                    resampler,
                    in_buf,
                    out_buf,
                }
            }
        };

        let mut vad =
            Vad::new(WHISPER_RATE as i32).map_err(|_| anyhow::anyhow!("invalid VAD rate"))?;
        // Aggressive (not VeryAggressive): with pre-roll (1.4) and the fix
        // that stops discarding mid-utterance non-speech (1.1), the cost of
        // a false positive is a few extra ms of silence, while a false
        // negative here used to mean lost phonemes.
        vad.fvad_set_mode(VadMode::Aggressive)
            .map_err(|_| anyhow::anyhow!("failed to set VAD mode"))?;

        Ok(Self {
            resampler_hw48,
            resample_hw48_in,
            resample_hw48_out,
            hw_input_frames,
            resampler_vad,
            resample_vad_in,
            resample_vad_out,
            whisper_resample,
            denoise: DenoiseState::new(),
            denoise_in: vec![0.0f32; FRAME_48K],
            denoise_out: vec![0.0f32; FRAME_48K],
            clean48_vad: vec![0.0f32; FRAME_48K],
            vad,
            vad_i16: vec![0i16; FRAME_16K],
            preroll: VecDeque::with_capacity(RING_FRAMES * FRAME_16K),
            utterance: Utterance::default(),
            min_speech_rms: env_f32("STTD_MIN_RMS", DEFAULT_MIN_SPEECH_RMS),
            whisper_tx,
        })
    }

    /// The raw hw-rate chunk size every call to `process_frame` must be fed.
    fn input_frames(&self) -> usize {
        self.hw_input_frames
    }

    /// Process one raw hw-rate chunk (`input_frames()` samples).
    fn process_frame(&mut self, frame_hw_raw: &[f32]) {
        debug_assert_eq!(frame_hw_raw.len(), self.hw_input_frames);

        // ---- hw -> 48 kHz: shared basis for the VAD branch (denoise needs
        // 48 kHz), and for the whisper branch too when using the `Chained`
        // fallback -----------------------------------------------------------
        self.resample_hw48_in[0][..frame_hw_raw.len()].copy_from_slice(frame_hw_raw);
        let frame48_produced = match self.resampler_hw48.process_into_buffer(
            &self.resample_hw48_in,
            &mut self.resample_hw48_out,
            None,
        ) {
            Ok((_, produced)) => produced,
            Err(e) => {
                eprintln!("[sttd] hw->48k resample error: {e}");
                return;
            }
        };
        if frame48_produced != FRAME_48K {
            // Denoise's frame API is fixed-size; an unexpected chunk count
            // (resampler startup/edge case) would desync it. Skip this
            // frame rather than feed denoise a mismatched buffer.
            eprintln!(
                "[sttd] hw->48k produced {frame48_produced} samples, expected {FRAME_48K}; \
                 skipping frame"
            );
            return;
        }
        // Owned: needed below both for denoise and, in the `Chained` case,
        // for the whisper branch's own resample call, which would otherwise
        // alias this borrow of `self`.
        let frame_48k_raw = self.resample_hw48_out[0][..FRAME_48K].to_vec();

        // ---- VAD branch: denoise, then resample to 16 kHz -----------------
        for (i, s) in frame_48k_raw.iter().enumerate() {
            self.denoise_in[i] = (s * 32768.0).clamp(-32768.0, 32767.0);
        }
        self.denoise
            .process_frame(&mut self.denoise_out, &self.denoise_in);
        for (i, s) in self.denoise_out.iter().enumerate() {
            self.clean48_vad[i] = s / 32768.0;
        }
        self.resample_vad_in[0][..FRAME_48K].copy_from_slice(&self.clean48_vad);
        let vad_produced = match self.resampler_vad.process_into_buffer(
            &self.resample_vad_in,
            &mut self.resample_vad_out,
            None,
        ) {
            Ok((_, produced)) => produced,
            Err(e) => {
                eprintln!("[sttd] vad resample error: {e}");
                return;
            }
        };

        // ---- whisper branch: raw (no denoise) -------------------------------
        let whisper16 = match &mut self.whisper_resample {
            WhisperResample::Direct {
                resampler,
                in_buf,
                out_buf,
            } => {
                in_buf[0][..frame_hw_raw.len()].copy_from_slice(frame_hw_raw);
                match resampler.process_into_buffer(in_buf, out_buf, None) {
                    Ok((_, produced)) => out_buf[0][..produced].to_vec(),
                    Err(e) => {
                        eprintln!("[sttd] whisper resample error: {e}");
                        return;
                    }
                }
            }
            WhisperResample::Chained {
                resampler,
                in_buf,
                out_buf,
            } => {
                in_buf[0][..FRAME_48K].copy_from_slice(&frame_48k_raw);
                match resampler.process_into_buffer(in_buf, out_buf, None) {
                    Ok((_, produced)) => out_buf[0][..produced].to_vec(),
                    Err(e) => {
                        eprintln!("[sttd] whisper resample error: {e}");
                        return;
                    }
                }
            }
        };

        // ---- VAD decision --------------------------------------------------
        // `vad_i16` is pre-sized for the steady-state FRAME_16K case; guard
        // against a resampler startup/edge call producing a different count
        // instead of panicking on an out-of-bounds write.
        if self.vad_i16.len() < vad_produced {
            self.vad_i16.resize(vad_produced, 0);
        }
        for (i, s) in self.resample_vad_out[0][..vad_produced].iter().enumerate() {
            self.vad_i16[i] = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        }
        let is_speech = match self.vad.is_voice_segment(&self.vad_i16[..vad_produced]) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("[sttd] VAD rejected frame length");
                false
            }
        };

        // ---- utterance buffering --------------------------------------------
        let should_flush = buffer_frame(&mut self.utterance, &self.preroll, &whisper16, is_speech);
        if should_flush {
            self.flush();
        }

        // ---- pre-roll ring update -------------------------------------------
        self.preroll.extend(whisper16.iter().copied());
        while self.preroll.len() > RING_FRAMES * FRAME_16K {
            self.preroll.pop_front();
        }
    }

    fn flush(&mut self) {
        flush_utterance(&mut self.utterance, &self.whisper_tx, self.min_speech_rms);
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    fn finish(&mut self) {
        if !self.utterance.buf.is_empty() {
            self.flush();
        }
    }
}

/// Pure utterance-buffering decision: given the current VAD verdict, decide
/// whether to append `frame` and whether the utterance should be flushed now.
/// Extracted out of `FrameProcessor::process_frame` so it's unit-testable
/// without a live VAD signal.
///
/// Two behaviors, both load-bearing for hallucination avoidance:
/// - An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive
///   speech frames, not the first one, so a single VAD false positive no
///   longer opens (and thus sends) a full utterance. The onset frames
///   themselves aren't lost: `preroll` already contains them by the time
///   the utterance opens, since the caller updates it every frame
///   regardless of `is_speech`.
/// - Once open, every frame is appended regardless of `is_speech` (non-
///   speech frames become natural in-utterance padding instead of being
///   spliced out) — but `speech_frames`/`samples_at_last_speech`/energy are
///   tracked only from VAD-flagged frames, so `flush_utterance` can later
///   gate and trim on actual speech content instead of total buffer length.
fn buffer_frame(utt: &mut Utterance, preroll: &VecDeque<f32>, frame: &[f32], is_speech: bool) -> bool {
    if is_speech {
        utt.onset_run += 1;
        utt.speech_frames += 1;
        for &s in frame {
            utt.speech_energy_sum_sq += (s as f64) * (s as f64);
        }
        utt.speech_sample_count += frame.len();

        if utt.buf.is_empty() {
            if utt.onset_run < ONSET_SPEECH_FRAMES {
                return false;
            }
            utt.buf.extend(preroll.iter().copied());
        }
        utt.buf.extend_from_slice(frame);
        utt.samples_at_last_speech = utt.buf.len();
        utt.silence_frames = 0;
        utt.buf.len() >= MAX_UTTERANCE_SAMPLES
    } else {
        utt.onset_run = 0;
        if !utt.buf.is_empty() {
            utt.buf.extend_from_slice(frame);
            utt.silence_frames += 1;
            utt.silence_frames > SILENCE_FRAMES_THRESHOLD
        } else {
            false
        }
    }
}

/// Decide whether the accumulated utterance is real speech and, if so, trim
/// its trailing silence and move it into the whisper queue. Resets `utt` to
/// its default state either way; `send` failure (worker gone) is logged,
/// never panics, so the audio loop survives.
///
/// Two gates, both defending against the same failure mode: a VAD false
/// positive (a click, a breath, mains hum) opening an utterance that
/// pre-roll and silence hangover then pad out to ~1-2s of near-silence —
/// which whisper, lacking any working confidence gate of its own (see
/// `DEFAULT_MIN_CONFIDENCE`'s doc comment), would otherwise happily
/// hallucinate a sentence for:
/// - `MIN_SPEECH_FRAMES`: not enough VAD-flagged speech overall.
/// - `min_rms`: VAD-flagged frames present, but too quiet to be real speech.
fn flush_utterance(utt: &mut Utterance, whisper_tx: &Sender<Clip>, min_rms: f32) {
    let speech_frames = utt.speech_frames;
    if speech_frames < MIN_SPEECH_FRAMES {
        eprintln!(
            "[sttd] dropping utterance ({speech_frames} speech frames, {} samples)",
            utt.buf.len()
        );
        *utt = Utterance::default();
        return;
    }

    let rms = if utt.speech_sample_count > 0 {
        (utt.speech_energy_sum_sq / utt.speech_sample_count as f64).sqrt() as f32
    } else {
        0.0
    };
    if rms < min_rms {
        eprintln!(
            "[sttd] dropping low-energy utterance ({speech_frames} speech frames, \
             rms={rms:.5} < {min_rms:.5})"
        );
        *utt = Utterance::default();
        return;
    }

    // Trim the tail: `SILENCE_FRAMES_THRESHOLD` decided *when* to close this
    // utterance, but everything past `samples_at_last_speech +
    // TAIL_KEEP_SAMPLES` is silence hangover that doesn't need to reach
    // whisper.
    let tail_end = (utt.samples_at_last_speech + TAIL_KEEP_SAMPLES).min(utt.buf.len());
    utt.buf.truncate(tail_end);

    let pcm = std::mem::take(&mut utt.buf);
    eprintln!(
        "[sttd] utterance: {} samples ({speech_frames} speech frames, rms={rms:.5}) -> whisper",
        pcm.len()
    );
    *utt = Utterance::default();
    if let Err(e) = whisper_tx.send(Clip {
        pcm,
        speech_frames,
        rms,
    }) {
        eprintln!("[sttd] whisper queue closed: {e}");
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Downmix an interleaved multi-channel buffer to mono by averaging.
fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Build the cpal input stream for the device's native sample format.
///
/// The callback only downmixes to mono and forwards to `raw_tx`. It never
/// calls `unwrap`/`expect`, never blocks, and does no heavy work.
fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    sample_format: cpal::SampleFormat,
    raw_tx: Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let err_fn = |err| eprintln!("[sttd] stream error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono(data, channels);
                    if let Err(e) = tx.send(mono) {
                        eprintln!("[sttd] raw queue closed: {e}");
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    let mut tmp = Vec::with_capacity(data.len());
                    for s in data {
                        tmp.push(*s as f32 / 32768.0);
                    }
                    let mono = downmix_to_mono(&tmp, channels);
                    if let Err(e) = tx.send(mono) {
                        eprintln!("[sttd] raw queue closed: {e}");
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            let tx = raw_tx.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    let mut tmp = Vec::with_capacity(data.len());
                    for s in data {
                        tmp.push(*s as f32 / 65535.0 * 2.0 - 1.0);
                    }
                    let mono = downmix_to_mono(&tmp, channels);
                    if let Err(e) = tx.send(mono) {
                        eprintln!("[sttd] raw queue closed: {e}");
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        fmt => anyhow::bail!("unsupported sample format: {fmt:?}"),
    }
}

/// Install a Ctrl-C handler that flips `running` to false.
///
/// Uses the `ctrlc` crate so the main loop can flush trailing speech before
/// exiting. Failures are logged, never fatal.
fn ctrlc_handler(cb: impl Fn() + Send + 'static) -> Result<()> {
    ctrlc::set_handler(cb).context("failed to install Ctrl-C handler")
}

// ---------------------------------------------------------------------------
// Model auto-download
// ---------------------------------------------------------------------------

fn resolve_model_path() -> PathBuf {
    std::env::var_os("WHISPER_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn ensure_model_available() -> Result<PathBuf> {
    let path = resolve_model_path();
    if path.exists() {
        return Ok(path);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    // Derive the download URL from the filename so `WHISPER_MODEL_PATH` can
    // point at any ggml model in this HF repo, not just the compiled-in
    // default. Previously the URL was hardcoded to `ggml-base.bin`, so
    // pointing the env var at a missing `ggml-large-v3.bin` would silently
    // download `base` under that filename.
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("model path has no filename: {}", path.display()))?;
    if !file_name.starts_with("ggml-") || !file_name.ends_with(".bin") {
        anyhow::bail!(
            "model not found at {} and its filename doesn't look like a whisper.cpp \
             model (expected ggml-<name>.bin); download it manually",
            path.display()
        );
    }
    let url = format!("{MODEL_BASE_URL}{file_name}");
    eprintln!(
        "[sttd] model not found at {}, downloading {url}...",
        path.display()
    );
    download_model(&url, &path)?;
    Ok(path)
}

fn download_model(url: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("model download request failed: {url}"))?;

    let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    if let Some(n) = total {
        eprintln!("[sttd] downloading {:.1} MiB...", n as f64 / 1_048_576.0);
    } else {
        eprintln!("[sttd] downloading (size unknown)...");
    }

    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&tmp)
        .with_context(|| format!("failed to create {}", tmp.display()))?;
    let bytes = std::io::copy(&mut reader, &mut out).context("download I/O failed")?;
    out.flush().ok();
    drop(out);
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("failed to move {} -> {}", tmp.display(), dest.display()))?;
    eprintln!(
        "[sttd] downloaded {} ({:.1} MiB)",
        dest.display(),
        bytes as f64 / 1_048_576.0
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Whisper worker
// ---------------------------------------------------------------------------

fn spawn_whisper_worker(rx: Receiver<Clip>, model_path: PathBuf) -> JoinHandle<()> {
    thread::Builder::new()
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
fn apply_gain(pcm: &[f32], rms: f32, target_rms: f32, max_gain: f32) -> Vec<f32> {
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
fn pad_with_dither(pcm_16k: &[f32], min_samples: usize) -> Vec<f32> {
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
fn average_text_confidence(
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
fn fold_diacritics(c: char) -> char {
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
fn is_hallucination(text: &str) -> bool {
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

// ---------------------------------------------------------------------------
// WAV I/O (utterance dumps + `--file` mode); no extra dependency.
// ---------------------------------------------------------------------------

fn write_wav_mono16(path: &Path, pcm_f32: &[f32], sample_rate: u32) -> Result<()> {
    let mut file =
        std::fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;

    let num_samples = pcm_f32.len() as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    let data_bytes = num_samples * 2;

    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_bytes).to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for &s in pcm_f32 {
        let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        file.write_all(&clamped.to_le_bytes())?;
    }
    Ok(())
}

struct WavAudio {
    sample_rate: u32,
    /// Mono, f32 in [-1, 1].
    samples: Vec<f32>,
}

/// Minimal RIFF/WAVE reader: PCM16 or IEEE-float32, any sample rate/channel
/// count. Anything else (e.g. compressed WAV) is rejected with a message
/// suggesting an `ffmpeg` conversion.
fn read_wav_mono_f32(path: &Path) -> Result<WavAudio> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        anyhow::bail!("{}: not a RIFF/WAVE file", path.display());
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut samples: Option<Vec<f32>> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_size).min(data.len());
        if body_start > data.len() {
            break;
        }

        if chunk_id == b"fmt " {
            let body = &data[body_start..body_end];
            if body.len() < 16 {
                anyhow::bail!("{}: truncated fmt chunk", path.display());
            }
            audio_format = u16::from_le_bytes(body[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if chunk_id == b"data" {
            let body = &data[body_start..body_end];
            samples = Some(decode_pcm(body, audio_format, bits_per_sample, channels)?);
        }

        // Chunks are word-aligned.
        pos = body_start + chunk_size + (chunk_size % 2);
    }

    let samples = samples.with_context(|| format!("{}: no data chunk found", path.display()))?;
    if sample_rate == 0 {
        anyhow::bail!("{}: no fmt chunk found", path.display());
    }
    Ok(WavAudio { sample_rate, samples })
}

fn decode_pcm(body: &[u8], audio_format: u16, bits_per_sample: u16, channels: u16) -> Result<Vec<f32>> {
    let channels = channels.max(1) as usize;
    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => body
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => body
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        (fmt, bits) => anyhow::bail!(
            "unsupported WAV format (audioFormat={fmt}, bitsPerSample={bits}); convert with e.g. \
             `ffmpeg -i in.wav -ac 1 -sample_fmt s16 out.wav`"
        ),
    };
    Ok(downmix_to_mono(&interleaved, channels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_chunk_for(hw_rate: u32) -> usize {
        ((hw_rate as f64 * FRAME_48K as f64 / DENOISE_RATE as f64).round() as usize).max(1)
    }

    #[test]
    fn chunk_sizes_yield_480_out() {
        assert_eq!(input_chunk_for(44_100), 441);
        assert_eq!(input_chunk_for(48_000), 480);
        assert_eq!(input_chunk_for(16_000), 160);
        assert_eq!(input_chunk_for(96_000), 960);
    }

    #[test]
    fn denoise_frame_is_480() {
        assert_eq!(DenoiseState::FRAME_SIZE, FRAME_48K);
    }

    #[test]
    fn rubato_produces_480_for_common_rates() {
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let chunk = input_chunk_for(hw);
            let resampler = FftFixedInOut::<f32>::new(hw as usize, 48_000, chunk, 1).unwrap();
            assert_eq!(resampler.output_frames_next(), 480, "hw={hw}");
        }
    }

    #[test]
    fn rubato_48k_to_16k_produces_160() {
        let resampler = FftFixedInOut::<f32>::new(48_000, 16_000, FRAME_48K, 1).unwrap();
        assert_eq!(resampler.input_frames_next(), FRAME_48K);
        assert_eq!(resampler.output_frames_next(), FRAME_16K);
    }

    #[test]
    fn dual_resamplers_agree_on_input_frames() {
        // `FrameProcessor::new` only takes the `WhisperResample::Direct`
        // (single-hop hw->16k) path when a hw->16k resampler consumes the
        // same input chunk size as the hw->48k one, since both are fed from
        // the same raw stream in lockstep. Confirms that holds for common
        // rates, and that both branches still land on a round 10 ms frame
        // count (480 @ 48k, 160 @ 16k) — i.e. that `make_resampler`'s single
        // formula (`hw_rate / 100`) is the right one for both targets.
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let (r48, in48) = make_resampler(hw, DENOISE_RATE).unwrap();
            let (r16, in16) = make_resampler(hw, WHISPER_RATE).unwrap();
            assert_eq!(in48, in16, "hw={hw}: input chunk sizes must match");
            assert_eq!(r48.output_frames_next(), FRAME_48K, "hw={hw}");
            assert_eq!(r16.output_frames_next(), FRAME_16K, "hw={hw}");
        }
    }

    #[test]
    fn vad_accepts_10ms_16k_frame() {
        let mut vad = Vad::new(16_000).unwrap();
        vad.fvad_set_mode(VadMode::Aggressive).unwrap();
        let silence = vec![0i16; FRAME_16K];
        assert!(vad.is_voice_segment(&silence).is_ok());
    }

    #[test]
    fn buffer_frame_keeps_non_speech_inside_open_utterance() {
        // Regression for the original bug: non-speech frames in the middle
        // of an utterance used to be silently dropped instead of appended.
        let mut utt = Utterance {
            buf: vec![0.5f32; 10],
            ..Default::default()
        };
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(!flush);
        assert_eq!(utt.buf.len(), 10 + FRAME_16K);
        assert_eq!(utt.silence_frames, 1);
    }

    #[test]
    fn buffer_frame_ignores_non_speech_before_utterance_starts() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(!flush);
        assert!(utt.buf.is_empty());
    }

    #[test]
    fn single_vad_hit_does_not_open_utterance() {
        // Regression for the production bug: one VAD false positive (a
        // click, a breath) used to open a full utterance immediately.
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let frame = vec![0.5f32; FRAME_16K];

        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            let flush = buffer_frame(&mut utt, &preroll, &frame, true);
            assert!(!flush);
            assert!(utt.buf.is_empty(), "must not open before ONSET_SPEECH_FRAMES");
        }
        // The frame that reaches ONSET_SPEECH_FRAMES opens it.
        buffer_frame(&mut utt, &preroll, &frame, true);
        assert!(!utt.buf.is_empty(), "must open at ONSET_SPEECH_FRAMES");
    }

    #[test]
    fn buffer_frame_prepends_preroll_on_utterance_start() {
        let mut utt = Utterance::default();
        let mut preroll = VecDeque::new();
        preroll.extend(vec![0.25f32; FRAME_16K * 2]);
        let frame = vec![0.5f32; FRAME_16K];

        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            buffer_frame(&mut utt, &preroll, &frame, true);
        }
        assert!(utt.buf.is_empty());
        buffer_frame(&mut utt, &preroll, &frame, true);
        assert_eq!(utt.buf.len(), FRAME_16K * 3);
        assert!(utt.buf[..FRAME_16K * 2].iter().all(|&s| s == 0.25));
        assert!(utt.buf[FRAME_16K * 2..].iter().all(|&s| s == 0.5));
    }

    #[test]
    fn buffer_frame_flushes_after_silence_threshold() {
        let mut utt = Utterance {
            buf: vec![0.5f32; 10],
            silence_frames: SILENCE_FRAMES_THRESHOLD,
            ..Default::default()
        };
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(flush);
    }

    #[test]
    fn utterance_from_log_is_rejected() {
        // Regression for the production log: a 7-speech-frame VAD blip,
        // padded by pre-roll and a full silence hangover, used to produce
        // an 18_880-sample utterance that whisper then hallucinated
        // "Dziękuje za uwagę." for.
        let mut utt = Utterance::default();
        let mut preroll = VecDeque::new();
        preroll.extend(vec![0.0f32; RING_FRAMES * FRAME_16K]);
        let speech = vec![0.3f32; FRAME_16K];
        let silence = vec![0.0f32; FRAME_16K];

        for _ in 0..7 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let mut flush = false;
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            flush = buffer_frame(&mut utt, &preroll, &silence, false);
            if flush {
                break;
            }
        }
        assert!(flush, "silence hangover should eventually close the utterance");

        let (tx, rx) = unbounded();
        flush_utterance(&mut utt, &tx, 0.0); // rms gate disabled: isolate MIN_SPEECH_FRAMES
        assert!(
            rx.try_recv().is_err(),
            "an utterance with too little VAD-flagged speech must be dropped"
        );
    }

    #[test]
    fn real_utterance_survives_gates() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer_frame(&mut utt, &preroll, &silence, false);
        }

        let (tx, rx) = unbounded();
        flush_utterance(&mut utt, &tx, DEFAULT_MIN_SPEECH_RMS);
        let got = rx.try_recv().expect("real speech must be sent to whisper");
        assert!(!got.pcm.is_empty());
        assert_eq!(got.speech_frames, 25);
        assert!(got.rms > 0.0);
    }

    #[test]
    fn tail_is_trimmed_to_200ms() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let samples_at_last_speech = utt.samples_at_last_speech;
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer_frame(&mut utt, &preroll, &silence, false);
        }

        let (tx, rx) = unbounded();
        flush_utterance(&mut utt, &tx, 0.0);
        let got = rx.try_recv().unwrap();
        assert_eq!(got.pcm.len(), samples_at_last_speech + TAIL_KEEP_SAMPLES);
    }

    #[test]
    fn flush_drops_short_and_sends_long() {
        let (tx, rx) = unbounded();

        let mut short = Utterance {
            buf: vec![0.0f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES - 1,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0, // high RMS: isolate the frame-count gate
            ..Default::default()
        };
        flush_utterance(&mut short, &tx, 0.0);
        assert!(rx.try_recv().is_err());

        let mut long = Utterance {
            buf: vec![0.5f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES,
            samples_at_last_speech: 5_000,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0,
            ..Default::default()
        };
        flush_utterance(&mut long, &tx, 0.0);
        let got = rx.try_recv().unwrap();
        assert_eq!(got.pcm.len(), 5_000);
    }

    #[test]
    fn strong_acoustic_evidence_relaxes_confidence_gate() {
        // Exact numbers from the production log this fix addresses:
        // "[sttd] utterance: 34720 samples (137 speech frames, rms=0.01178)
        // -> whisper" followed by "dropped low-confidence transcript
        // (avg_p=0.478 < 0.600)" for a transcript the user confirmed was
        // close to correct (one misheard word), not a hallucination.
        let clip = Clip {
            pcm: Vec::new(),
            speech_frames: 137,
            rms: 0.01178,
        };
        let (threshold, gate) = confidence_threshold(
            &clip,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        assert_eq!(gate, "strong");
        assert_eq!(threshold, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert!(
            0.478 >= threshold,
            "avg_p=0.478 must clear the relaxed threshold {threshold}"
        );
    }

    #[test]
    fn weak_evidence_keeps_strict_gate() {
        // Regression for the hole 71c05b0 closed: an utterance with barely
        // enough VAD-flagged speech to survive `flush_utterance`'s own
        // gates, and RMS just above the noise floor, must still face the
        // strict confidence floor — even at the same avg_p (0.478) that the
        // strong-evidence case above lets through.
        let clip = Clip {
            pcm: Vec::new(),
            speech_frames: MIN_SPEECH_FRAMES,
            rms: DEFAULT_MIN_SPEECH_RMS + 0.0005,
        };
        let (threshold, gate) = confidence_threshold(
            &clip,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        assert_eq!(gate, "weak");
        assert_eq!(threshold, DEFAULT_MIN_CONFIDENCE);
        assert!(
            0.478 < threshold,
            "avg_p=0.478 must NOT clear the strict threshold {threshold}"
        );
    }

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
    fn silence_threshold_is_800ms() {
        assert_eq!(SILENCE_FRAMES_THRESHOLD, 80);
        assert_eq!(SILENCE_FRAMES_THRESHOLD * 10, 800);
    }

    #[test]
    fn whisper_min_clears_whisper_cpp_mel_length_check() {
        // Regression: exactly WHISPER_RATE (16_000) samples passes our own
        // "is it 1000ms" arithmetic but whisper.cpp's mel-framed length
        // check computes only 990ms for it and rejects anyway (see comment
        // on WHISPER_MIN_SAMPLES). Assert the real formula clears 100 frames.
        let n_len_org = 1 + (WHISPER_MIN_SAMPLES as i64 + 200 - 400) / 160;
        assert!(
            n_len_org >= 100,
            "WHISPER_MIN_SAMPLES={WHISPER_MIN_SAMPLES} only yields {n_len_org} 10ms mel frames, whisper.cpp requires >= 100"
        );
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

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(downmix_to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn wav_roundtrip_mono16() {
        let dir = std::env::temp_dir().join(format!("sttd_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.wav");
        let pcm: Vec<f32> = (0..1600).map(|i| ((i % 100) as f32 / 100.0) - 0.5).collect();
        write_wav_mono16(&path, &pcm, WHISPER_RATE).unwrap();
        let wav = read_wav_mono_f32(&path).unwrap();
        assert_eq!(wav.sample_rate, WHISPER_RATE);
        assert_eq!(wav.samples.len(), pcm.len());
        for (a, b) in wav.samples.iter().zip(pcm.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
