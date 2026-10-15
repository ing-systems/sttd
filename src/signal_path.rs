// ---------------------------------------------------------------------------
// Signal path: VAD branch + model branch, behind one seam.
// ---------------------------------------------------------------------------
//
// One concept — a paired VAD verdict describing the model-branch audio it
// is attached to — used to require bouncing between `frame`, `dsp`,
// `denoise` and `vad`. `FrameProcessor` managed three `FirStream` adapters,
// paired queues and `clean48` scratch, triplicating Direct vs Denoised logic
// across feed / flush / discard.
//
// This module owns branch topology, resampling cadence, denoising, high-pass
// filtering and verdict pairing internally. Callers feed hw-rate chunks and
// are lent each ready `PairedFrames` (already 80 Hz HP'd) in place, from one
// contiguous buffer per branch; they never see `FirStream`, the denoiser's
// 48 kHz intermediate, or how pairing stays in lockstep. `feed`/`flush` share one body per topology; only the resampler
// call differs between feeding a chunk and flushing the tail.

use anyhow::{Context, Result};

#[cfg(feature = "denoise")]
use crate::config::{DENOISE_RATE, FRAME_48K};
use crate::config::{FRAME_16K, SAMPLE_RATE};
#[cfg(feature = "denoise")]
use crate::denoise::Denoiser;
use crate::denoise::DenoiserKind;
use crate::dsp::FirStream;
use crate::hp::Hp;

/// One completed 10 ms frame pair, lent to the caller's callback (always
/// `FRAME_16K` samples each, already 80 Hz high-passed). The frames are
/// mutable so the Agc can gain them in place; they live in `SignalPath`'s own
/// buffers, so no frame is allocated or copied on the way out.
///
/// `Shared` is the default (`Direct`) topology: one frame serves both roles,
/// so the Agc applies the shared gain once and both roles read the same
/// samples. `Split` is the opt-in `Denoised` topology, where the VAD branch
/// (denoised) and the model branch (raw) come from independent resamplers.
pub(crate) enum PairedFrames<'a> {
    Shared(&'a mut [f32]),
    #[cfg_attr(not(feature = "denoise"), allow(dead_code))]
    Split(&'a mut [f32], &'a mut [f32]),
}

/// Warn threshold for unpaired frames waiting in `SignalPath` (see
/// `pending_depths`). Both branches average 100 frames/s, so steady state is
/// 0-2; anything above means resamplers diverged or the consumer stalled.
pub(crate) const SIGNAL_PENDING_WARN: usize = 8;

/// Owns all resampling / denoising state. Fed one raw hw-rate chunk at a
/// time; hands the completed VAD/model pairs to a callback in order, so the
/// VAD verdict always describes the model audio it is attached to.
pub(crate) struct SignalPath {
    vad_path: VadPath,
    /// Model-branch 16 kHz output not yet handed out (in `Direct`, also the
    /// VAD branch). Whole frames are cut from its front, so between calls it
    /// holds less than one frame in `Direct`, and in `Denoised` whatever the
    /// model branch ran ahead of the VAD branch.
    model16: Vec<f32>,
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
        // hw_rate -> 48 kHz (the denoiser's required rate). Its output
        // completes on its own cadence, so whole 480-frames are denoised one
        // by one as they arrive rather than one per input chunk.
        resampler_hw48: FirStream,
        // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
        resampler_vad: FirStream,
        resampler_model: FirStream,
        denoiser: Denoiser,
        /// hw -> 48 kHz output not yet denoised (less than one `FRAME_48K`
        /// between calls).
        buf48: Vec<f32>,
        clean48_vad: Vec<f32>,
        /// Denoised 16 kHz VAD-branch output not yet paired.
        vad16: Vec<f32>,
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
                VadPath::Direct {
                    resampler,
                    hp: Hp::new(),
                }
            }
            #[cfg(feature = "denoise")]
            DenoiserKind::DeepFilter => {
                let resampler_hw48 = FirStream::new(hw_rate, DENOISE_RATE, FRAME_48K)
                    .with_context(|| {
                        format!("failed to create resampler (hw -> {DENOISE_RATE})")
                    })?;
                log::info!("resampler: {hw_rate} -> {DENOISE_RATE} (fir, frame_out={FRAME_48K})");

                let resampler_vad = FirStream::new(DENOISE_RATE, SAMPLE_RATE, FRAME_16K)
                    .context("failed to create resampler (48k -> 16k, vad branch)")?;

                let resampler_model = FirStream::new(hw_rate, SAMPLE_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {SAMPLE_RATE})"))?;

                let denoiser = Denoiser::new()?;
                VadPath::Denoised {
                    resampler_hw48,
                    resampler_vad,
                    resampler_model,
                    denoiser,
                    buf48: Vec::new(),
                    clean48_vad: vec![0.0f32; FRAME_48K],
                    vad16: Vec::new(),
                    hp_vad: Hp::new(),
                    hp_model: Hp::new(),
                }
            }
        };
        Ok(Self {
            vad_path,
            model16: Vec::new(),
        })
    }

    /// Whole frames currently waiting unpaired per branch: `(vad, model)`.
    /// After every `feed`/`flush` one side is zero; a growing non-zero side
    /// means the branches diverged.
    pub(crate) fn pending_depths(&self) -> (usize, usize) {
        let model = self.model16.len() / FRAME_16K;
        match &self.vad_path {
            VadPath::Direct { .. } => (model, model),
            #[cfg(feature = "denoise")]
            VadPath::Denoised { vad16, .. } => (vad16.len() / FRAME_16K, model),
        }
    }

    /// Feed one raw hw-rate chunk and hand every completed pair to `on_pair`,
    /// in order. Each branch's `FirStream` may complete zero, one, or (on
    /// catch-up) several 10 ms frames per call; a branch running ahead waits
    /// in its buffer. Both average 100 frames/s, so that stays at a frame or
    /// two.
    pub(crate) fn feed(&mut self, frame_hw_raw: &[f32], on_pair: impl FnMut(PairedFrames<'_>)) {
        self.run(Some(frame_hw_raw), on_pair);
    }

    /// Zero-pad each resampler's trailing partial chunk so tail audio stuck
    /// in the FIR still reaches the utterance instead of being lost at
    /// shutdown, then hand every pair left buffered to `on_pair`.
    pub(crate) fn flush(&mut self, on_pair: impl FnMut(PairedFrames<'_>)) {
        self.run(None, on_pair);
    }

    /// One pass over the branch topology — resample (and denoise, opt-in),
    /// then high-pass and hand out every completed pair. `input` is `None`
    /// for `flush`.
    fn run(&mut self, input: Option<&[f32]>, mut on_pair: impl FnMut(PairedFrames<'_>)) {
        let Self { vad_path, model16 } = self;
        match vad_path {
            VadPath::Direct { resampler, hp } => {
                resample(resampler, input, model16);
                let whole = model16.len() / FRAME_16K * FRAME_16K;
                for frame in model16[..whole].chunks_exact_mut(FRAME_16K) {
                    hp.process(frame);
                    on_pair(PairedFrames::Shared(frame));
                }
                model16.drain(..whole);
            }
            #[cfg(feature = "denoise")]
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_model,
                denoiser,
                buf48,
                clean48_vad,
                vad16,
                hp_vad,
                hp_model,
            } => {
                resample(resampler_hw48, input, buf48);
                let whole48 = buf48.len() / FRAME_48K * FRAME_48K;
                for frame48 in buf48[..whole48].chunks_exact(FRAME_48K) {
                    denoiser.denoise_frame(frame48, clean48_vad);
                    resampler_vad.feed(clean48_vad, vad16);
                }
                buf48.drain(..whole48);
                if input.is_none() {
                    // The VAD branch's own FIR still holds a partial chunk;
                    // zero-pad it before pairing.
                    resampler_vad.flush(vad16);
                }
                resample(resampler_model, input, model16);
                let frames = (vad16.len() / FRAME_16K).min(model16.len() / FRAME_16K);
                let whole = frames * FRAME_16K;
                for (vad, model) in vad16[..whole]
                    .chunks_exact_mut(FRAME_16K)
                    .zip(model16[..whole].chunks_exact_mut(FRAME_16K))
                {
                    hp_vad.process(vad);
                    hp_model.process(model);
                    on_pair(PairedFrames::Split(vad, model));
                }
                vad16.drain(..whole);
                model16.drain(..whole);
            }
        }
    }

    /// Drop in-flight audio: unpaired output + resampler partial-chunk
    /// state, so the first frames after resume aren't blended from pre-pause
    /// audio. Idempotent.
    pub(crate) fn clear(&mut self) {
        self.model16.clear();
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
                buf48,
                vad16,
                hp_vad,
                hp_model,
                ..
            } => {
                resampler_hw48.clear();
                resampler_vad.clear();
                resampler_model.clear();
                buf48.clear();
                vad16.clear();
                hp_vad.reset();
                hp_model.reset();
                denoiser.reset();
            }
        }
    }
}

/// Run `resampler` on `input`, or zero-drain it when `input` is `None`
/// (flush), appending the output to `out`.
fn resample(resampler: &mut FirStream, input: Option<&[f32]>, out: &mut Vec<f32>) {
    match input {
        Some(input) => resampler.feed(input, out),
        None => resampler.flush(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hw_input_frames;

    /// One pair copied out of the callback, so tests can collect and inspect.
    struct OwnedPair {
        vad: Vec<f32>,
        model: Vec<f32>,
        shared: bool,
    }

    fn own(pairs: &mut Vec<OwnedPair>) -> impl FnMut(PairedFrames<'_>) + '_ {
        |pair| {
            pairs.push(match pair {
                PairedFrames::Shared(frame) => OwnedPair {
                    vad: frame.to_vec(),
                    model: frame.to_vec(),
                    shared: true,
                },
                PairedFrames::Split(vad, model) => OwnedPair {
                    vad: vad.to_vec(),
                    model: model.to_vec(),
                    shared: false,
                },
            })
        }
    }

    fn feed(signal: &mut SignalPath, input: &[f32]) -> Vec<OwnedPair> {
        let mut pairs = Vec::new();
        signal.feed(input, own(&mut pairs));
        pairs
    }

    fn flush(signal: &mut SignalPath) -> Vec<OwnedPair> {
        let mut pairs = Vec::new();
        signal.flush(own(&mut pairs));
        pairs
    }

    fn feed_silence(signal: &mut SignalPath, hw_rate: u32, chunks: usize) -> usize {
        let hw_chunk = hw_input_frames(hw_rate);
        let silence = vec![0.0f32; hw_chunk];
        let mut pairs = 0;
        for _ in 0..chunks {
            pairs += feed(signal, &silence).len();
        }
        pairs += flush(signal).len();
        pairs
    }

    fn roles(pair: &OwnedPair) -> (&[f32], &[f32]) {
        (&pair.vad, &pair.model)
    }

    #[test]
    fn direct_pairs_one_to_one_with_exact_frames() {
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let silence = vec![0.0f32; hw_chunk];
            for _ in 0..50 {
                for pair in feed(&mut signal, &silence) {
                    let (vad, model) = roles(&pair);
                    assert_eq!(vad.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(model.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(vad, model, "hw={hw}: direct hop shares one hop");
                }
            }
        }
    }

    #[test]
    fn direct_serves_both_roles_from_one_frame() {
        // The default topology must hand out one frame for both roles, so the
        // Agc gains it once (`Agc::process_shared`), not twice.
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        let input = vec![0.1f32; hw_chunk];
        let mut saw_pair = false;
        for _ in 0..50 {
            for pair in feed(&mut signal, &input) {
                saw_pair = true;
                assert!(pair.shared, "direct pair must be `Shared`");
            }
        }
        assert!(saw_pair, "expected at least one pair in 50 chunks");
    }

    #[cfg(feature = "denoise")]
    #[test]
    fn denoised_model_branch_loses_no_frame() {
        // Regression: the old pairing loop popped both queues before checking
        // either, so whenever the VAD branch had no frame ready the model
        // branch's oldest frame was discarded — 10 ms gaps in the audio sent
        // to the model at any rate whose 48 kHz hop lags. The paired model
        // frames must be exactly the raw hw -> 16 kHz stream, high-passed.
        for hw in [16_000, 22_050, 44_100] {
            let mut signal = SignalPath::new(hw, DenoiserKind::DeepFilter).unwrap();
            let mut reference = FirStream::new(hw, SAMPLE_RATE, FRAME_16K).unwrap();
            let mut expected = Vec::new();
            let mut got = Vec::new();
            let hw_chunk = hw_input_frames(hw);
            for k in 0..300 {
                let chunk: Vec<f32> = (0..hw_chunk)
                    .map(|i| (((k * hw_chunk + i) as f32) * 0.01).sin() * 0.1)
                    .collect();
                reference.feed(&chunk, &mut expected);
                for pair in feed(&mut signal, &chunk) {
                    assert!(!pair.shared, "hw={hw}: denoised pairs are `Split`");
                    got.extend_from_slice(&pair.model);
                }
            }
            Hp::new().process(&mut expected);
            assert!(got.len() >= 290 * FRAME_16K, "hw={hw}: got {}", got.len());
            assert_eq!(got[..], expected[..got.len()], "hw={hw}");
        }
    }

    #[test]
    fn buffers_stay_bounded_on_a_long_unpaused_run() {
        // Regression: the old staging ring in `FirStream` compacted only on
        // flush, so an unpaused run grew it by a frame per 10 ms chunk
        // (~64 KB/s at 48 kHz). 100 s of input must leave less than one frame
        // buffered, with no growth in capacity.
        for hw in [16_000, 22_050, 32_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let chunk = vec![0.1f32; hw_input_frames(hw)];
            for _ in 0..10_000 {
                signal.feed(&chunk, |_| {});
                assert!(signal.model16.len() < FRAME_16K, "hw={hw}");
            }
            assert!(
                signal.model16.capacity() <= 8 * FRAME_16K,
                "hw={hw}: capacity {}",
                signal.model16.capacity()
            );
        }
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
            .map(|_| feed(&mut signal, &vec![0.1f32; hw_chunk]).len())
            .sum();
        let flushed = flush(&mut signal);
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
        let _ = feed(&mut signal, &vec![0.1f32; hw_chunk]);
        signal.clear();
        assert!(flush(&mut signal).is_empty());
        // And the path still works after clear.
        let silence = vec![0.0f32; hw_chunk];
        let _ = feed(&mut signal, &silence);
    }

    #[test]
    fn feed_and_pair_stay_in_lockstep() {
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let silence = vec![0.0f32; hw_chunk];
            let (mut paired, mut max_pending) = (0, 0);
            for _ in 0..200 {
                // In `Direct` every whole frame a feed completes is paired
                // in that same call, so its pair count is the burst size.
                let burst = feed(&mut signal, &silence).len();
                paired += burst;
                max_pending = max_pending.max(burst);
            }
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
    fn clear_empties_pending_frames() {
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        let _ = feed(&mut signal, &vec![0.1f32; hw_chunk]);
        signal.clear();
        assert_eq!(signal.pending_depths(), (0, 0));
        assert_eq!(SIGNAL_PENDING_WARN, 8);
    }
}
