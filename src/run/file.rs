//! File mode (`--file`): the same pipeline run offline over a WAV file, to the
//! end and losslessly, for comparing tuning changes against a fixed recording.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use super::queue::{finish_run, push_utterance, Backpressure};
use super::setup::{
    asr_config, live_dictionary, make_emitter, open_metrics, prepare_model, start_translation,
    AcousticConfig,
};
use crate::asr::{spawn_asr_worker, AsrConfig};
use crate::cli::Cli;
use crate::model_store::EngineKind;
use crate::translate::TranslationOut;
use crate::wav::read_wav_mono_f32_with_channel;

pub(crate) fn run_file_mode(path: &Path, cli: &Cli) -> Result<()> {
    // Same ordering rationale as run_microphone_mode: do the cheap, fallible
    // setup (read + validate the WAV, resolve the model path) before the
    // heavy work (denoiser init, GPU load), so a bad file/spec fails fast
    // instead of racing model teardown.
    let wav = read_wav_mono_f32_with_channel(path, cli.channel)?;
    log::info!(
        "file: {} ({} Hz, {} samples, {:.1}s, channel={})",
        path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f64 / wav.sample_rate as f64,
        cli.channel,
    );

    let (model, mut asr_tx, asr_rx) = prepare_model(cli)?;
    if model.kind == EngineKind::Whisper {
        // Offline decoding has no live text to show, so streaming buys the
        // short encoder window here and nothing else.
        log::info!(
            "file mode: short encoder window ({} frame floor), no partial transcripts",
            cli.audio_ctx
        );
    }
    let acoustic = AcousticConfig::from_cli(cli);
    let mut fp = acoustic.make_frame_processor(wav.sample_rate)?;
    let input_frames = fp.required_input_frames();

    // File mode runs offline with no D-Bus service (see the interview), but
    // the worker still takes the live handle so `AsrConfig` has one shape.
    // The booster hook stays unconsumed: nothing swaps words here.
    let (_, live_dict) = live_dictionary(cli);

    // File mode has no pause signalling, so a permanently-clear flag plus
    // disabled auto-unload can never trigger an unload mid-file. Sounds stay off
    // offline: no pause/resume transitions exist to cue. Autopause is forced off
    // for the same reason: file input must never flip the (unread) pause flag into
    // an unload mid-file.
    let config = AsrConfig {
        auto_unload: false,
        sound: false,
        autopause: false,
        ..asr_config(cli, &acoustic, false, live_dict)
    };
    // The emitter is built before the translation step because the translation
    // thread emits through a sharing handle onto the same queue.
    let emitter = make_emitter(cli)?;
    // File mode builds the translation step exactly as the microphone path does,
    // so an offline `--file ... --translate en` prints English. There is no bus
    // here, so nothing can switch it later: the lane exists only when a
    // translator did, and the worker emits inline when it does not.
    let metrics = open_metrics(cli);
    let translation_state = start_translation(cli)?;
    let translation = translation_state.is_loaded().then(|| {
        TranslationOut::spawn(
            translation_state,
            Box::new(emitter.sharing()),
            metrics.clone(),
        )
    });
    let (asr_handle, asr_ready) = spawn_asr_worker(
        asr_rx,
        model,
        config,
        emitter,
        translation,
        metrics,
        Arc::new(AtomicBool::new(false)),
        // File input runs to completion; inference is never aborted.
        Arc::new(AtomicBool::new(true)),
    );
    // Fail fast on model-load/dump-dir errors before processing the file:
    // no realtime capture to keep bounded here, so blocking wait is fine.
    match asr_ready.recv_timeout(Duration::from_secs(600)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => anyhow::bail!("asr worker failed to start: {e}"),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("timed out waiting for asr worker to start")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("asr worker exited before signaling ready")
        }
    }

    for frame in wav.samples.chunks_exact(input_frames) {
        if let Some(utterance) = fp.process_frame(frame) {
            push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
        }
    }
    // A trailing partial chunk (< input_frames samples, at most ~30 ms) is
    // dropped; fine for offline comparison runs.
    if let Some(utterance) = fp.flush_trailing_speech() {
        push_utterance(&mut asr_tx, utterance, Backpressure::Wait);
    }

    finish_run(fp, asr_tx, asr_handle)?;
    Ok(())
}
