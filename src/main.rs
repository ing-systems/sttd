//! Real-time Speech-to-Text pipeline.
//!
//! Data flow:
//! `cpal` (hw rate, stereo) -> downmix mono -> `rubato` (-> 48 kHz, 480-sample
//! frames) -> per 10 ms frame, two branches:
//!   - VAD branch:     `nnnoiseless` denoise -> `rubato` (-> 16 kHz) -> `webrtc-vad`
//!   - whisper branch: (unprocessed)         -> `rubato` (-> 16 kHz) -> utterance buffer
//! The VAD decides only *when* to cut an utterance, never what to keep: every
//! frame while an utterance is open is appended (speech or not), and ~300 ms
//! of pre-roll is prepended at onset so quiet word starts aren't clipped.
//! -> `crossbeam-channel` queue -> isolated `whisper-rs` inference thread
//! (CUDA, large-v3-turbo).
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
const SILENCE_FRAMES_THRESHOLD: u32 = 80;
/// Drop utterances shorter than this (avoids whispering on blips). Short
/// enough to keep one-word Polish utterances (tak/nie/co/już) that the VAD
/// has already trimmed down.
const MIN_UTTERANCE_SAMPLES: usize = 4_800; // 300 ms @ 16 kHz
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

/// Whisper model auto-download. The URL is derived from the filename (see
/// `ensure_model_available`), so `WHISPER_MODEL_PATH` can point at any
/// ggml model published in this repo, not just the compiled-in default.
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
const DEFAULT_MODEL_PATH: &str = "models/ggml-large-v3-turbo.bin";

/// Default `initial_prompt`: primes Polish orthography/diacritics and the
/// technical vocabulary this tool is normally dictated into. Override with
/// `STTD_PROMPT`.
const DEFAULT_PROMPT: &str = "Rozmowa techniczna po polsku. Terminy: Kubernetes, Docker, \
Rust, cargo, commit, merge request, repozytorium, deployment, kontener, backend, frontend, \
API, endpoint, debugowanie, refaktoryzacja, Linux, systemd, Wayland, Neovim.";

/// Transcripts that are near-exact matches for common Whisper hallucinations
/// on silence/low-energy audio (subtitle-credit boilerplate picked up from
/// training data), compared after lowercasing and stripping punctuation.
const HALLUCINATION_PATTERNS: &[&str] = &[
    "napisy stworzone przez społeczność amara org",
    "napisy",
    "napisy pl",
    "subtitles by",
    "dziękuję za obejrzenie",
    "dziękuję za uwagę",
    "zapraszam do subskrypcji",
    "zapraszam do subskrypcji kanału",
];

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
         \x20   STTD_DUMP_DIR         write each utterance as WAV+TXT into this directory\n"
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

    let (mut resampler, input_frames) = make_hw_to_48k_resampler(hw_rate)?;
    eprintln!(
        "[sttd] rubato: {hw_rate} -> {DENOISE_RATE}, in={input_frames} out={}",
        resampler.output_frames_next()
    );
    let mut resample_in = resampler.input_buffer_allocate(true);
    let mut resample_out = resampler.output_buffer_allocate(true);

    let mut pending_48k: Vec<f32> = Vec::with_capacity(FRAME_48K * 4);
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

    // ---- now the heavy part: resolve/download the model and load it onto
    // the GPU on its own thread. Capture is already running above.
    let model_path = ensure_model_available()?;
    eprintln!("[sttd] model: {}", model_path.display());
    let (whisper_tx, whisper_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let whisper_handle = spawn_whisper_worker(whisper_rx, model_path);
    let mut fp = FrameProcessor::new(whisper_tx)?;

    // ---- main processing loop (NOT the audio callback) --------------------
    while running.load(Ordering::SeqCst) {
        // Drain everything the callback delivered since last iteration.
        let mut got_audio = false;
        while let Ok(chunk) = raw_rx.try_recv() {
            got_audio = true;
            accumulator.extend_from_slice(&chunk);
        }

        // Resample full input chunks -> pending 48 kHz stream.
        while accumulator.len() >= input_frames {
            resample_in[0][..input_frames].copy_from_slice(&accumulator[..input_frames]);
            accumulator.drain(..input_frames);

            let (used_in, produced) =
                match resampler.process_into_buffer(&resample_in, &mut resample_out, None) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[sttd] resample error: {e}");
                        break;
                    }
                };
            debug_assert_eq!(used_in, input_frames);
            pending_48k.extend_from_slice(&resample_out[0][..produced]);
        }

        // 480-sample 48 kHz frames -> denoise/VAD + whisper branches.
        while pending_48k.len() >= FRAME_48K {
            fp.process_frame(&pending_48k[..FRAME_48K]);
            pending_48k.drain(..FRAME_48K);
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

/// Build the hw_rate -> 48 kHz resampler with a fixed input chunk size
/// chosen so its output is exactly `FRAME_48K` samples.
fn make_hw_to_48k_resampler(hw_rate: u32) -> Result<(FftFixedInOut<f32>, usize)> {
    let desired_in =
        ((hw_rate as f64 * FRAME_48K as f64 / DENOISE_RATE as f64).round() as usize).max(1);
    let resampler = FftFixedInOut::<f32>::new(hw_rate as usize, DENOISE_RATE as usize, desired_in, 1)
        .context("failed to create rubato resampler (hw -> 48k)")?;
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

    let (mut resampler, input_frames) = make_hw_to_48k_resampler(wav.sample_rate)?;
    let mut resample_in = resampler.input_buffer_allocate(true);
    let mut resample_out = resampler.output_buffer_allocate(true);

    let model_path = ensure_model_available()?;
    eprintln!("[sttd] model: {}", model_path.display());
    let (whisper_tx, whisper_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let whisper_handle = spawn_whisper_worker(whisper_rx, model_path);
    let mut fp = FrameProcessor::new(whisper_tx.clone())?;

    let mut pending_48k: Vec<f32> = Vec::with_capacity(FRAME_48K * 4);
    let mut offset = 0usize;
    while offset + input_frames <= wav.samples.len() {
        resample_in[0][..input_frames].copy_from_slice(&wav.samples[offset..offset + input_frames]);
        offset += input_frames;

        let (_, produced) = resampler
            .process_into_buffer(&resample_in, &mut resample_out, None)
            .context("resample error")?;
        pending_48k.extend_from_slice(&resample_out[0][..produced]);

        while pending_48k.len() >= FRAME_48K {
            fp.process_frame(&pending_48k[..FRAME_48K]);
            pending_48k.drain(..FRAME_48K);
        }
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

/// Owns all per-frame DSP/VAD/buffering state. Fed one 480-sample 48 kHz
/// frame at a time by either the live capture loop or file mode, so both
/// share identical utterance-detection behavior.
struct FrameProcessor {
    // 48 kHz -> 16 kHz resamplers. Two independent instances (not shared)
    // since each carries its own internal filter state and consumes a
    // different signal (denoised vs. raw).
    resampler_vad: FftFixedInOut<f32>,
    resampler_whisper: FftFixedInOut<f32>,
    resample_vad_in: Vec<Vec<f32>>,
    resample_vad_out: Vec<Vec<f32>>,
    resample_whisper_in: Vec<Vec<f32>>,
    resample_whisper_out: Vec<Vec<f32>>,

    denoise: Box<DenoiseState<'static>>,
    denoise_in: Vec<f32>,
    denoise_out: Vec<f32>,
    clean48_vad: Vec<f32>,

    vad: Vad,
    vad_i16: Vec<i16>,

    /// Rolling ~300 ms of whisper-branch audio, prepended to an utterance
    /// the instant it starts so onsets aren't clipped by VAD latency.
    preroll: VecDeque<f32>,
    speech_buf: Vec<f32>,
    silence_frames: u32,

    whisper_tx: Sender<Vec<f32>>,
}

impl FrameProcessor {
    fn new(whisper_tx: Sender<Vec<f32>>) -> Result<Self> {
        let resampler_vad = FftFixedInOut::<f32>::new(
            DENOISE_RATE as usize,
            WHISPER_RATE as usize,
            FRAME_48K,
            1,
        )
        .context("failed to create rubato resampler (48k -> 16k, vad branch)")?;
        let resampler_whisper = FftFixedInOut::<f32>::new(
            DENOISE_RATE as usize,
            WHISPER_RATE as usize,
            FRAME_48K,
            1,
        )
        .context("failed to create rubato resampler (48k -> 16k, whisper branch)")?;
        debug_assert_eq!(resampler_vad.input_frames_next(), FRAME_48K);
        debug_assert_eq!(resampler_vad.output_frames_next(), FRAME_16K);
        debug_assert_eq!(resampler_whisper.input_frames_next(), FRAME_48K);
        debug_assert_eq!(resampler_whisper.output_frames_next(), FRAME_16K);

        let resample_vad_in = resampler_vad.input_buffer_allocate(true);
        let resample_vad_out = resampler_vad.output_buffer_allocate(true);
        let resample_whisper_in = resampler_whisper.input_buffer_allocate(true);
        let resample_whisper_out = resampler_whisper.output_buffer_allocate(true);

        let mut vad =
            Vad::new(WHISPER_RATE as i32).map_err(|_| anyhow::anyhow!("invalid VAD rate"))?;
        // Aggressive (not VeryAggressive): with pre-roll (1.4) and the fix
        // that stops discarding mid-utterance non-speech (1.1), the cost of
        // a false positive is a few extra ms of silence, while a false
        // negative here used to mean lost phonemes.
        vad.fvad_set_mode(VadMode::Aggressive)
            .map_err(|_| anyhow::anyhow!("failed to set VAD mode"))?;

        Ok(Self {
            resampler_vad,
            resampler_whisper,
            resample_vad_in,
            resample_vad_out,
            resample_whisper_in,
            resample_whisper_out,
            denoise: DenoiseState::new(),
            denoise_in: vec![0.0f32; FRAME_48K],
            denoise_out: vec![0.0f32; FRAME_48K],
            clean48_vad: vec![0.0f32; FRAME_48K],
            vad,
            vad_i16: vec![0i16; FRAME_16K],
            preroll: VecDeque::with_capacity(RING_FRAMES * FRAME_16K),
            speech_buf: Vec::with_capacity(MAX_UTTERANCE_SAMPLES),
            silence_frames: 0,
            whisper_tx,
        })
    }

    /// Process one 480-sample 48 kHz frame.
    fn process_frame(&mut self, frame_48k_raw: &[f32]) {
        debug_assert_eq!(frame_48k_raw.len(), FRAME_48K);

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

        // ---- whisper branch: raw (no denoise), resample to 16 kHz --------
        self.resample_whisper_in[0][..FRAME_48K].copy_from_slice(frame_48k_raw);
        let whisper_produced = match self.resampler_whisper.process_into_buffer(
            &self.resample_whisper_in,
            &mut self.resample_whisper_out,
            None,
        ) {
            Ok((_, produced)) => produced,
            Err(e) => {
                eprintln!("[sttd] whisper resample error: {e}");
                return;
            }
        };
        // Owned (not borrowed from `self`): `self.flush()` below needs
        // exclusive access to `*self`, which an outstanding borrow of
        // `self.resample_whisper_out` would conflict with. 160 floats.
        let whisper16 = self.resample_whisper_out[0][..whisper_produced].to_vec();

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
        let should_flush = buffer_frame(
            &mut self.speech_buf,
            &mut self.silence_frames,
            &self.preroll,
            &whisper16,
            is_speech,
        );
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
        flush_utterance(&mut self.speech_buf, &self.whisper_tx);
        self.silence_frames = 0;
    }

    /// Flush any trailing speech. Call once at shutdown / end of input.
    fn finish(&mut self) {
        if !self.speech_buf.is_empty() {
            self.flush();
        }
    }
}

/// Pure utterance-buffering decision: given the current VAD verdict, decide
/// whether to append `frame` and whether the utterance should be flushed now.
/// Extracted out of `FrameProcessor::process_frame` so it's unit-testable
/// without a live VAD signal.
///
/// The VAD decides only *when* to cut an utterance, never what to keep: once
/// `speech_buf` is non-empty, every frame is appended regardless of `is_speech`
/// (non-speech frames become natural in-utterance padding instead of being
/// spliced out). `preroll` is prepended the instant an utterance starts.
fn buffer_frame(
    speech_buf: &mut Vec<f32>,
    silence_frames: &mut u32,
    preroll: &VecDeque<f32>,
    frame: &[f32],
    is_speech: bool,
) -> bool {
    if is_speech {
        if speech_buf.is_empty() {
            speech_buf.extend(preroll.iter().copied());
        }
        speech_buf.extend_from_slice(frame);
        *silence_frames = 0;
        speech_buf.len() >= MAX_UTTERANCE_SAMPLES
    } else if !speech_buf.is_empty() {
        speech_buf.extend_from_slice(frame);
        *silence_frames += 1;
        *silence_frames > SILENCE_FRAMES_THRESHOLD
    } else {
        false
    }
}

/// Move the accumulated utterance into the whisper queue.
///
/// Short blips are discarded; `send` failure (worker gone) is logged, never
/// panics, so the audio loop survives.
fn flush_utterance(speech_buf: &mut Vec<f32>, whisper_tx: &Sender<Vec<f32>>) {
    if speech_buf.len() < MIN_UTTERANCE_SAMPLES {
        eprintln!(
            "[sttd] dropping short utterance ({} samples)",
            speech_buf.len()
        );
        speech_buf.clear();
        return;
    }
    let utterance = std::mem::take(speech_buf);
    eprintln!("[sttd] utterance: {} samples -> whisper", utterance.len());
    if let Err(e) = whisper_tx.send(utterance) {
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

fn spawn_whisper_worker(rx: Receiver<Vec<f32>>, model_path: PathBuf) -> JoinHandle<()> {
    thread::Builder::new()
        .name("whisper".into())
        .spawn(move || {
            if let Err(e) = whisper_loop(&rx, &model_path) {
                eprintln!("[sttd] whisper worker failed: {e:?}");
            }
        })
        .expect("failed to spawn whisper thread")
}

fn whisper_loop(rx: &Receiver<Vec<f32>>, model_path: &Path) -> Result<()> {
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

    let model_name = ctx
        .model_type_readable()
        .unwrap_or_else(|_| "unknown".to_string());
    eprintln!("[sttd] whisper ready (model={model_name}, pl, beam=5, threads=8, gpu=true)");

    let prompt = std::env::var("STTD_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let dump_dir = std::env::var_os("STTD_DUMP_DIR").map(PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        eprintln!("[sttd] dumping utterances to {}", dir.display());
    }

    let mut utterance_no: u64 = 0;
    for pcm in rx.iter() {
        if pcm.is_empty() {
            continue;
        }
        utterance_no += 1;

        if let Some(dir) = &dump_dir {
            let wav_path = dir.join(format!("utt_{utterance_no:06}.wav"));
            if let Err(e) = write_wav_mono16(&wav_path, &pcm, WHISPER_RATE) {
                eprintln!("[sttd] failed to dump {}: {e:?}", wav_path.display());
            }
        }

        match transcribe(&mut state, &pcm, &prompt) {
            Ok(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if is_hallucination(trimmed) {
                    eprintln!("[sttd] dropped likely hallucination: {trimmed:?}");
                    continue;
                }
                println!("{trimmed}");
                if let Some(dir) = &dump_dir {
                    let txt_path = dir.join(format!("utt_{utterance_no:06}.txt"));
                    if let Err(e) = std::fs::write(&txt_path, trimmed) {
                        eprintln!("[sttd] failed to write {}: {e:?}", txt_path.display());
                    }
                }
            }
            Err(e) => eprintln!("[sttd] transcription failed: {e:?}"),
        }
    }
    Ok(())
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

fn transcribe(
    state: &mut whisper_rs::WhisperState,
    pcm_16k: &[f32],
    prompt: &str,
) -> Result<String> {
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
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.full(params, pcm_16k).context("whisper full failed")?;
    let n = state.full_n_segments().context("no segments")?;
    let mut out = String::new();
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
    }
    Ok(out)
}

/// Whether `text` (a full transcript) is a known Whisper hallucination on
/// silence/noise, checked as an exact match after normalization so genuine
/// sentences that merely contain e.g. "dziękuję" are never caught.
fn is_hallucination(text: &str) -> bool {
    // Punctuation becomes a space (not dropped): "Amara.org" must normalize
    // to "amara org" to match the pattern list, not "amaraorg".
    let normalized: String = text
        .to_lowercase()
        .chars()
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
        let mut speech_buf = vec![0.5f32; 10];
        let mut silence_frames = 0;
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut speech_buf, &mut silence_frames, &preroll, &frame, false);
        assert!(!flush);
        assert_eq!(speech_buf.len(), 10 + FRAME_16K);
        assert_eq!(silence_frames, 1);
    }

    #[test]
    fn buffer_frame_ignores_non_speech_before_utterance_starts() {
        let mut speech_buf: Vec<f32> = Vec::new();
        let mut silence_frames = 0;
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut speech_buf, &mut silence_frames, &preroll, &frame, false);
        assert!(!flush);
        assert!(speech_buf.is_empty());
    }

    #[test]
    fn buffer_frame_prepends_preroll_on_utterance_start() {
        let mut speech_buf: Vec<f32> = Vec::new();
        let mut silence_frames = 0;
        let mut preroll = VecDeque::new();
        preroll.extend(vec![0.25f32; FRAME_16K * 2]);
        let frame = vec![0.5f32; FRAME_16K];
        buffer_frame(&mut speech_buf, &mut silence_frames, &preroll, &frame, true);
        assert_eq!(speech_buf.len(), FRAME_16K * 3);
        assert!(speech_buf[..FRAME_16K * 2].iter().all(|&s| s == 0.25));
        assert!(speech_buf[FRAME_16K * 2..].iter().all(|&s| s == 0.5));
    }

    #[test]
    fn buffer_frame_flushes_after_silence_threshold() {
        let mut speech_buf = vec![0.5f32; 10];
        let mut silence_frames = SILENCE_FRAMES_THRESHOLD;
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut speech_buf, &mut silence_frames, &preroll, &frame, false);
        assert!(flush);
    }

    #[test]
    fn flush_drops_short_and_sends_long() {
        let (tx, rx) = unbounded();
        let mut short = vec![0.0f32; MIN_UTTERANCE_SAMPLES - 1];
        flush_utterance(&mut short, &tx);
        assert!(short.is_empty());
        assert!(rx.try_recv().is_err());

        let mut long = vec![0.5f32; MIN_UTTERANCE_SAMPLES + 10];
        flush_utterance(&mut long, &tx);
        assert!(long.is_empty());
        let got = rx.try_recv().unwrap();
        assert_eq!(got.len(), MIN_UTTERANCE_SAMPLES + 10);
    }

    #[test]
    fn silence_threshold_is_800ms() {
        assert_eq!(SILENCE_FRAMES_THRESHOLD, 80);
        assert_eq!(SILENCE_FRAMES_THRESHOLD * 10, 800);
    }

    #[test]
    fn min_utterance_is_300ms() {
        assert_eq!(MIN_UTTERANCE_SAMPLES, 4_800);
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
