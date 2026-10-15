//! The two queues between capture and the ASR worker, from the capture side:
//! draining the cpal callback's raw ring, framing it through the
//! `FrameProcessor`, and pushing closed utterances under each input path's
//! backpressure policy.

use std::thread;
use std::time::Duration;

use anyhow::Result;
use rtrb::{Consumer, Producer};

use crate::asr::PartialSlot;
use crate::frame::FrameProcessor;
use crate::utterance::Utterance;

/// Drain every available sample from the `rtrb` consumer into the accumulator,
/// as at most two slice copies (the ring's contiguous runs) instead of one
/// atomic pop per sample. Samples the callback pushes meanwhile wait for the
/// next drain. Main thread only; allocation (accumulator growth) is fine here
/// — only the cpal callback is real-time.
pub(super) fn drain_raw_into(raw_rx: &mut Consumer<f32>, accumulator: &mut Vec<f32>) {
    let Ok(chunk) = raw_rx.read_chunk(raw_rx.slots()) else {
        return;
    };
    let (first, second) = chunk.as_slices();
    accumulator.extend_from_slice(first);
    accumulator.extend_from_slice(second);
    chunk.commit_all();
}

/// Drain complete frames from the utterance buffer into the VAD branch and
/// the model branch, handing each closed utterance to `push` for emission.
/// Frames are consumed through a cursor and the remainder is compacted once
/// at the end, instead of a `drain` per frame (which memmoved the whole
/// remainder every time).
///
/// A frame that closed nothing may instead leave a partial snapshot of the open
/// utterance in `partials` (latest wins), and a close drops the one waiting: its
/// utterance is now a final in the queue.
pub(super) fn drain_frames(
    utterance_buffer: &mut Vec<f32>,
    input_frames: usize,
    fp: &mut FrameProcessor,
    asr_tx: &mut Producer<Utterance>,
    push: fn(&mut Producer<Utterance>, Utterance),
    partials: &PartialSlot,
) {
    let mut consumed = 0;
    while utterance_buffer.len() - consumed >= input_frames {
        let frame = &utterance_buffer[consumed..consumed + input_frames];
        if let Some(utterance) = fp.process_frame(frame) {
            partials.clear();
            push(asr_tx, utterance);
        } else if let Some(pcm) = fp.take_partial() {
            partials.put(pcm);
        }
        consumed += input_frames;
    }
    utterance_buffer.drain(..consumed);
}

/// What a full ASR ring means, i.e. the one policy that differs between the
/// two input paths.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Backpressure {
    /// Live capture: never block the processing loop, drop the newest
    /// utterance with a warn. A dead worker (`is_abandoned`) is also just a
    /// warn. In practice the worker-side weakest-first shed keeps the
    /// backlog small so this rarely fires.
    Drop,
    /// Offline file mode (`shed_stale = false`) promises lossless decoding,
    /// but resampling runs at ~100x realtime while inference runs at roughly
    /// realtime, so the producer routinely outruns `ASR_QUEUE_CAP` (16) on any
    /// recording longer than a few tens of seconds. Dropping there would
    /// silently discard most of a long file, so wait for the worker to make
    /// room instead: safe here because file mode has no realtime deadline to
    /// protect.
    Wait,
}

/// Push a closed utterance into the bounded ASR ring under `policy`
/// (`Backpressure`). Returns once the utterance is queued or the worker is
/// gone (a dead worker only warns, in both policies).
pub(super) fn push_utterance(
    asr_tx: &mut Producer<Utterance>,
    utterance: Utterance,
    policy: Backpressure,
) {
    if asr_tx.is_abandoned() {
        log::warn!("asr worker gone, dropping utterance");
        return;
    }
    let mut pending = utterance;
    loop {
        match asr_tx.push(pending) {
            Ok(()) => return,
            Err(rtrb::PushError::Full(back)) => match policy {
                Backpressure::Drop => {
                    log::warn!("asr queue full, dropping newest utterance");
                    return;
                }
                Backpressure::Wait => {
                    if asr_tx.is_abandoned() {
                        log::warn!("asr worker gone, dropping utterance");
                        return;
                    }
                    pending = back;
                    thread::sleep(Duration::from_millis(1));
                }
            },
        }
    }
}

/// Microphone-mode push, as a plain function pointer for `drain_frames`.
/// File mode calls `push_utterance` with `Backpressure::Wait` directly.
pub(super) fn push_live_utterance(asr_tx: &mut Producer<Utterance>, utterance: Utterance) {
    push_utterance(asr_tx, utterance, Backpressure::Drop);
}

/// Occupied slots on the producer side (`Producer::slots` reports free).
pub(super) fn producer_occupied<T>(tx: &Producer<T>, cap: usize) -> usize {
    cap.saturating_sub(tx.slots())
}

/// Shared shutdown tail for both input paths, in this order: drop the frame
/// processor, then the utterance producer (so the worker sees `is_abandoned`
/// and exits), then join it. Dropping the producer before the join is what
/// keeps Ctrl-C/`--stop` from hanging forever; the join is what flushes the
/// trailing utterance the worker is still transcribing.
pub(super) fn finish_run(
    fp: FrameProcessor,
    asr_tx: Producer<Utterance>,
    asr_handle: thread::JoinHandle<()>,
) -> Result<()> {
    drop(fp);
    drop(asr_tx);
    asr_handle
        .join()
        .map_err(|_| anyhow::anyhow!("asr worker panicked"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use crate::run::setup::AcousticConfig;
    use clap::Parser;
    use rtrb::RingBuffer;

    #[test]
    fn drain_raw_into_keeps_order_across_the_ring_wrap() {
        let (mut tx, mut rx) = RingBuffer::<f32>::new(8);
        let mut acc = Vec::new();
        for s in 0..6 {
            tx.push(s as f32).unwrap();
        }
        drain_raw_into(&mut rx, &mut acc);
        // The next six wrap past the end of the 8-slot ring.
        for s in 6..12 {
            tx.push(s as f32).unwrap();
        }
        drain_raw_into(&mut rx, &mut acc);
        drain_raw_into(&mut rx, &mut acc);
        assert_eq!(acc, (0..12).map(|s| s as f32).collect::<Vec<_>>());
        assert!(rx.is_empty());
    }

    #[test]
    fn drain_frames_consumes_whole_frames_and_keeps_the_remainder() {
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        let mut fp = AcousticConfig::from_cli(&cli)
            .make_frame_processor(crate::config::SAMPLE_RATE)
            .expect("frame processor must build");
        let input_frames = fp.required_input_frames();
        let (mut asr_tx, _asr_rx) = RingBuffer::<Utterance>::new(crate::asr::ASR_QUEUE_CAP);
        // Two whole frames plus a partial one; silence opens no utterance, so
        // nothing is handed to the worker.
        let mut buffer = vec![0.0f32; input_frames * 2 + 7];
        drain_frames(
            &mut buffer,
            input_frames,
            &mut fp,
            &mut asr_tx,
            push_live_utterance,
            &crate::asr::PartialSlot::default(),
        );
        assert_eq!(buffer.len(), 7);
        assert!(buffer.iter().all(|&s| s == 0.0));
        assert_eq!(asr_tx.slots(), crate::asr::ASR_QUEUE_CAP);
    }
}
