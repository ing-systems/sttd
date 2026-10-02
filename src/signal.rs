// ---------------------------------------------------------------------------
// Signal path: VAD branch + model branch, behind one seam.
// ---------------------------------------------------------------------------
//
// One concept — a paired VAD verdict describing the model-branch audio it
// is attached to — used to require bouncing between `frame`, `dsp`,
// `denoise` and `vad`. `FrameProcessor` managed three `FirStream` adapters,
// paired queues, `Arc` sharing and `clean48` scratch, triplicating
// Direct vs Denoised logic across feed / flush / discard.
//
// This module owns branch topology, resampling cadence, denoising, high-pass
// filtering and verdict pairing internally. Callers feed hw-rate chunks and
// receive ready `PairedFrames` (already 80 Hz HP'd); they never see
// `FirStream`, the denoiser's 48 kHz intermediate, or how pairing stays in
// lockstep. `feed`/`flush`/`clear` each have one body per topology; only the
// resampler call differs between feeding a chunk and flushing the tail.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::{FRAME_16K, SAMPLE_RATE};
#[cfg(feature = "denoise")]
use crate::config::{DENOISE_RATE, FRAME_48K};
use crate::denoise::DenoiserKind;
#[cfg(feature = "denoise")]
use crate::denoise::Denoiser;
use crate::dsp::FirStream;
use crate::hp::Hp;

/// A queued 16 kHz frame, shared cheaply between the two branch queues. The
/// default (`Direct`) path decodes one frame and queues it for both the VAD
/// verdict and the utterance buffer; `Arc::new(Vec)` moves the buffer without
/// copying elements and the second queue entry costs one atomic increment.
pub(crate) type SharedFrame = Arc<Vec<f32>>;

/// One completed 10 ms frame pair, handed to the caller as owned buffers
/// (always `FRAME_16K` samples each).
///
/// `Shared` is the default (`Direct`) topology: the two branch queues held
/// `Arc` clones of one allocation, so the caller gets that single buffer back
/// — the Agc applies the shared gain once and both roles read the same
/// samples. `Split` is the opt-in `Denoised` topology, where the VAD branch
/// (denoised) and the model branch (raw) come from independent resamplers and
/// own independent buffers.
pub(crate) enum PairedFrames {
    Shared(Vec<f32>),
    Split(Vec<f32>, Vec<f32>),
}

/// One pass over the branch topology: `Feed` consumes a raw hw-rate chunk,
/// `Flush` zero-pads each resampler's trailing partial chunk so tail audio
/// still reaches the utterance. Both end in the same queue-then-pair step, so
/// `feed`/`flush` share one body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cadence {
    Feed,
    Flush,
}

impl Cadence {
    /// Run `resampler` for this cadence into the caller's reused `scratch`.
    /// `input` is unused on `Flush`: the resampler drains its own FIR state.
    fn drain(self, resampler: &mut FirStream, input: &[f32], scratch: &mut Vec<Vec<f32>>) {
        match self {
            Cadence::Feed => resampler.feed_into(input, scratch),
            Cadence::Flush => resampler.flush_into(scratch),
        }
    }
}

/// Warn threshold for unpaired frames waiting in `SignalPath` queues (see
/// `pending_depths`). Both branches average 100 frames/s, so steady state is
/// 0-2; anything above means resamplers diverged or the consumer stalled.
pub(crate) const SIGNAL_PENDING_WARN: usize = 8;

/// Owns all resampling / denoising state. Fed one raw hw-rate chunk at a
/// time; returns the completed VAD/model pairs in order, so the VAD verdict
/// always describes the model audio it is attached to.
pub(crate) struct SignalPath {
    vad_path: VadPath,
    pending_vad: VecDeque<SharedFrame>,
    pending_model: VecDeque<SharedFrame>,
    fed_chunks: u64,
    paired_frames: u64,
    max_pending: usize,
    /// Reused outer container for `FirStream::feed_into` / `flush_into`:
    /// cleared (capacity kept) on every resampler call so the 100 fps hot
    /// path doesn't reallocate the `Vec<Vec<_>>` wrapper per hw chunk. The
    /// inner per-frame `Vec`s are moved out (`drain(..)`) into `Arc`s and
    /// become queue entries — no element copy.
    scratch: Vec<Vec<f32>>,
}

/// The VAD branch's signal path, chosen once at startup from `--denoiser`
/// and fixed for the process's lifetime.
enum VadPath {
    /// Default (`none`): a single hw_rate -> 16 kHz hop feeds both the VAD
    /// verdict and the utterance buffer — no 48 kHz intermediate, no
    /// denoiser object. Each completed 16 kHz frame serves both roles, so
    /// VAD/model pairing is trivially 1:1.
    Direct { resampler: FirStream, hp: Hp },
    /// Opt-in (`deepfilter`, `denoise` feature): hw_rate -> 48 kHz -> denoise
    /// -> 48 kHz -> 16 kHz for VAD, plus an independent direct hw_rate ->
    /// 16 kHz hop for the model branch (which stays raw — the model was
    /// trained on raw audio, and the 48 kHz intermediate exists only so the
    /// denoiser, which is fixed at 48 kHz, can run for the VAD branch).
    #[cfg(feature = "denoise")]
    Denoised {
        // hw_rate -> 48 kHz (the denoiser's required rate). Its resampler's
        // fixed FFT chunks complete on their own cadence, so drained
        // 480-frames are denoised one by one as they arrive rather than one
        // per input chunk.
        resampler_hw48: FirStream,
        // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
        resampler_vad: FirStream,
        resampler_model: FirStream,
        denoiser: Denoiser,
        clean48_vad: Vec<f32>,
        hp_vad: Hp,
        hp_model: Hp,
    },
}

impl SignalPath {
    pub(crate) fn new(hw_rate: u32, denoiser_kind: DenoiserKind) -> Result<Self> {
        let vad_path = match denoiser_kind {
            DenoiserKind::Bypass => {
                let resampler = FirStream::new(hw_rate, SAMPLE_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {SAMPLE_RATE})"))?;
                log::info!("denoiser: none");
                VadPath::Direct { resampler, hp: Hp::new() }
            }
            #[cfg(feature = "denoise")]
            DenoiserKind::DeepFilter => {
                let resampler_hw48 = FirStream::new(hw_rate, DENOISE_RATE, FRAME_48K)
                    .with_context(|| format!("failed to create resampler (hw -> {DENOISE_RATE})"))?;
                log::info!(
                    "resampler: {hw_rate} -> {DENOISE_RATE} (fir, frame_out={FRAME_48K})"
                );

                let resampler_vad = FirStream::new(DENOISE_RATE, SAMPLE_RATE, FRAME_16K)
                    .context("failed to create resampler (48k -> 16k, vad branch)")?;

                let resampler_model = FirStream::new(hw_rate, SAMPLE_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {SAMPLE_RATE})"))?;

                let denoiser = Denoiser::new(DenoiserKind::DeepFilter)?;
                VadPath::Denoised {
                    resampler_hw48,
                    resampler_vad,
                    resampler_model,
                    denoiser,
                    clean48_vad: vec![0.0f32; FRAME_48K],
                    hp_vad: Hp::new(),
                    hp_model: Hp::new(),
                }
            }
        };
        Ok(Self {
            vad_path,
            pending_vad: VecDeque::new(),
            pending_model: VecDeque::new(),
            fed_chunks: 0,
            paired_frames: 0,
            max_pending: 0,
            scratch: Vec::new(),
        })
    }

    /// Frames currently waiting unpaired per branch: `(vad, model)`. After
    /// every `feed`/`flush` one side is empty (see `drain_paired`); a growing
    /// non-zero side means the branches diverged.
    pub(crate) fn pending_depths(&self) -> (usize, usize) {
        (self.pending_vad.len(), self.pending_model.len())
    }

    /// Totals for lockstep monitoring: `(fed_chunks, paired_frames,
    /// max_pending)`. `max_pending` is the largest pre-drain queue seen so
    /// far (either branch). Test-only reader; prod uses `pending_depths`.
    #[allow(dead_code)]
    pub(crate) const fn stats(&self) -> (u64, u64, usize) {
        (self.fed_chunks, self.paired_frames, self.max_pending)
    }

    /// Feed one raw hw-rate chunk; returns every completed pair in order. Each
    /// branch's `FirStream` may complete zero, one, or (on catch-up) several
    /// 10 ms frames per call; a branch running ahead waits in its queue. Both
    /// average 100 frames/s, so the queues stay bounded at a frame or two.
    pub(crate) fn feed(&mut self, frame_hw_raw: &[f32]) -> Vec<PairedFrames> {
        self.fed_chunks += 1;
        self.run(frame_hw_raw, Cadence::Feed)
    }

    /// Zero-pad each resampler's trailing partial chunk so tail audio stuck
    /// in an incomplete FFT chunk still reaches the utterance instead of
    /// being lost at shutdown, then return every paired frame left buffered.
    pub(crate) fn flush(&mut self) -> Vec<PairedFrames> {
        self.run(&[], Cadence::Flush)
    }

    /// One pass over the branch topology — resample (and denoise, opt-in),
    /// high-pass, queue — then hand every completed pair to the caller.
    /// `feed` and `flush` differ only in `cadence`.
    fn run(&mut self, frame_hw_raw: &[f32], cadence: Cadence) -> Vec<PairedFrames> {
        match &mut self.vad_path {
            VadPath::Direct { resampler, hp } => {
                self.scratch.clear();
                cadence.drain(resampler, frame_hw_raw, &mut self.scratch);
                for mut frame in self.scratch.drain(..) {
                    hp.process(&mut frame);
                    // One allocation for both branches: the queue entries are
                    // `Arc` clones, and `drain_paired` recognizes the pair by
                    // pointer and hands the buffer back untouched (no 640 B
                    // memcpy, no second alloc per 10 ms frame).
                    let shared: SharedFrame = Arc::new(frame);
                    self.pending_model.push_back(Arc::clone(&shared));
                    self.pending_vad.push_back(shared);
                }
            }
            #[cfg(feature = "denoise")]
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_model,
                denoiser,
                clean48_vad,
                hp_vad,
                hp_model,
            } => {
                self.scratch.clear();
                cadence.drain(resampler_hw48, frame_hw_raw, &mut self.scratch);
                for frame48 in self.scratch.drain(..) {
                    debug_assert_eq!(frame48.len(), FRAME_48K);
                    denoiser.denoise_frame(&frame48, clean48_vad);
                    // Opt-in path only: outer `Vec` alloc per denoised frame
                    // is acceptable here; the default `Direct` path above is
                    // the 100 fps hotspot and uses `feed_into` + sharing.
                    for mut vad_frame in resampler_vad.feed(&clean48_vad[..]) {
                        hp_vad.process(&mut vad_frame);
                        self.pending_vad.push_back(Arc::new(vad_frame));
                    }
                }
                if cadence == Cadence::Flush {
                    // The two 16 kHz branches still hold their own partial
                    // chunks; zero-pad them before pairing.
                    for mut vad_frame in resampler_vad.flush() {
                        hp_vad.process(&mut vad_frame);
                        self.pending_vad.push_back(Arc::new(vad_frame));
                    }
                    for mut model_frame in resampler_model.flush() {
                        hp_model.process(&mut model_frame);
                        self.pending_model.push_back(Arc::new(model_frame));
                    }
                } else {
                    self.scratch.clear();
                    cadence.drain(resampler_model, frame_hw_raw, &mut self.scratch);
                    for mut frame in self.scratch.drain(..) {
                        hp_model.process(&mut frame);
                        self.pending_model.push_back(Arc::new(frame));
                    }
                }
            }
        }
        self.note_pending();
        self.drain_paired()
    }

    /// Drop in-flight audio: paired-frame queues + resampler partial-chunk
    /// state, so the first frames after resume aren't blended from pre-pause
    /// audio. Idempotent. Also resets pairing counters so post-resume
    /// `stats()` start clean.
    pub(crate) fn clear(&mut self) {
        self.pending_vad.clear();
        self.pending_model.clear();
        self.scratch.clear();
        self.fed_chunks = 0;
        self.paired_frames = 0;
        self.max_pending = 0;
        match &mut self.vad_path {
            VadPath::Direct { resampler, hp } => {
                resampler.clear();
                hp.reset();
            }
            #[cfg(feature = "denoise")]
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_model,
                denoiser,
                hp_vad,
                hp_model,
                ..
            } => {
                resampler_hw48.clear();
                resampler_vad.clear();
                resampler_model.clear();
                hp_vad.reset();
                hp_model.reset();
                denoiser.reset();
            }
        }
    }

    fn note_pending(&mut self) {
        let pending = self.pending_vad.len().max(self.pending_model.len());
        self.max_pending = self.max_pending.max(pending);
    }

    fn drain_paired(&mut self) -> Vec<PairedFrames> {
        let mut out = Vec::new();
        while let (Some(vad16), Some(model16)) =
            (self.pending_vad.pop_front(), self.pending_model.pop_front())
        {
            debug_assert_eq!(vad16.len(), FRAME_16K);
            debug_assert_eq!(model16.len(), FRAME_16K);
            if Arc::ptr_eq(&vad16, &model16) {
                // `Direct` topology: both queue entries are `Arc` clones of one
                // allocation. Drop the twin handle, then take the buffer.
                drop(model16);
                out.push(PairedFrames::Shared(into_buffer(vad16)));
            } else {
                out.push(PairedFrames::Split(
                    into_buffer(vad16),
                    into_buffer(model16),
                ));
            }
        }
        self.paired_frames += out.len() as u64;
        out
    }
}

/// Take the owned buffer out of a queued frame.
///
/// `drain_paired` drops every other handle to the allocation first (in the
/// `Direct` topology the twin queue entry is an `Arc` clone of this same
/// buffer), so the count is 1 here and the move costs nothing. The `Err` arm
/// only keeps a broken invariant from panicking the audio loop; no code path
/// reaches it.
fn into_buffer(shared: SharedFrame) -> Vec<f32> {
    Arc::try_unwrap(shared).unwrap_or_else(|shared| (*shared).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hw_input_frames;

    fn feed_silence(signal: &mut SignalPath, hw_rate: u32, chunks: usize) -> usize {
        let hw_chunk = hw_input_frames(hw_rate);
        let silence = vec![0.0f32; hw_chunk];
        let mut pairs = 0;
        for _ in 0..chunks {
            pairs += signal.feed(&silence).len();
        }
        pairs += signal.flush().len();
        pairs
    }

    /// Both roles of one pair without copying: the `Shared` topology serves
    /// them from the same allocation.
    fn roles(pair: &PairedFrames) -> (&[f32], &[f32]) {
        match pair {
            PairedFrames::Shared(frame) => (frame, frame),
            PairedFrames::Split(vad, model) => (vad, model),
        }
    }

    #[test]
    fn direct_pairs_one_to_one_with_exact_frames() {
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let silence = vec![0.0f32; hw_chunk];
            for _ in 0..50 {
                for pair in signal.feed(&silence) {
                    let (vad, model) = roles(&pair);
                    assert_eq!(vad.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(model.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(vad, model, "hw={hw}: direct hop shares one hop");
                }
            }
        }
    }

    #[test]
    fn direct_shares_one_allocation_between_branches() {
        // Regression for the per-frame `Vec::clone()` (640 B memcpy + second
        // alloc at 100 fps): the pair must come back as the one allocation
        // both branch queues shared.
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        let input = vec![0.1f32; hw_chunk];
        let mut saw_pair = false;
        for _ in 0..50 {
            for pair in signal.feed(&input) {
                saw_pair = true;
                let (vad, model) = roles(&pair);
                assert!(
                    std::ptr::eq(vad.as_ptr(), model.as_ptr()),
                    "direct pair must share one allocation"
                );
            }
        }
        assert!(saw_pair, "expected at least one pair in 50 chunks");
    }

    #[test]
    fn branches_stay_in_lockstep() {
        // Both branches fed the same hw chunks must yield equal pair counts
        // within burst tolerance; pairing by order relies on it.
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let n = feed_silence(&mut signal, hw, 200);
            assert!(
                (193..=210).contains(&n),
                "hw={hw}: expected ~200 pairs, got {n}"
            );
        }
    }

    #[test]
    fn flush_recovers_trailing_audio() {
        // FIR buffers delay + lookahead: fed frames plus zero-drain flush must
        // recover the tail instead of losing it at shutdown.
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        assert_eq!(hw_chunk, 480);
        let fed: usize = (0..2)
            .map(|_| signal.feed(&vec![0.1f32; hw_chunk]).len())
            .sum();
        let flushed = signal.flush();
        assert!(!flushed.is_empty(), "flush must recover the tail");
        assert!(
            (2..=6).contains(&(fed + flushed.len())),
            "fed={fed} flushed={} out of range",
            flushed.len()
        );
    }

    #[test]
    fn clear_drops_in_flight_audio() {
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        // One feed may or may not complete a pair; either way, after clear
        // flush must find nothing.
        let _ = signal.feed(&vec![0.1f32; hw_chunk]);
        signal.clear();
        assert!(signal.flush().is_empty());
        // And the path still works after clear.
        let silence = vec![0.0f32; hw_chunk];
        let _ = signal.feed(&silence);
    }

    #[test]
    fn counters_track_feed_and_pair_in_lockstep() {
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let silence = vec![0.0f32; hw_chunk];
            for _ in 0..200 {
                let _ = signal.feed(&silence);
            }
            let (fed, paired, max_pending) = signal.stats();
            assert_eq!(fed, 200, "hw={hw}");
            // FIR delay shifts the count by a few frames, same
            // tolerance as `branches_stay_in_lockstep`.
            assert!(
                (193..=210).contains(&paired),
                "hw={hw}: paired={paired} for 200 fed"
            );
            assert!(
                max_pending <= 4,
                "hw={hw}: steady state bursts must stay small, got {max_pending}"
            );
            let (vad, model) = signal.pending_depths();
            assert!(
                vad <= 2 && model <= 2,
                "hw={hw}: pending depths must stay bounded, got ({vad}, {model})"
            );
        }
    }

    #[test]
    fn clear_resets_counters() {
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        let _ = signal.feed(&vec![0.1f32; hw_chunk]);
        signal.clear();
        assert_eq!(signal.stats(), (0, 0, 0));
        assert_eq!(signal.pending_depths(), (0, 0));
        assert_eq!(SIGNAL_PENDING_WARN, 8);
    }
}
