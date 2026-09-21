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

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, unbounded};

mod config;
mod audio;
use audio::*;
mod wav;
use wav::*;
mod model;
use model::*;
mod utterance;
use utterance::*;
mod pipeline;
use pipeline::*;
mod whisper;
use whisper::*;
mod cli;
use cli::*;

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
