// ---------------------------------------------------------------------------
// Signal path: VAD branch + whisper branch, behind one seam.
// ---------------------------------------------------------------------------
//
// One concept — a paired VAD verdict describing the whisper-branch audio it
// is attached to — used to require bouncing between `frame`, `dsp`,
// `denoise` and `vad`. `FrameProcessor` managed three `FftStream` adapters,
// paired queues, `Arc` sharing and `clean48` scratch, triplicating
// Direct vs Denoised logic across feed / flush / discard.
//
// This module owns branch topology, resampling cadence, denoising and
// verdict pairing internally. Callers feed hw-rate chunks and receive ready
// `(vad16, whisper16)` pairs; they never see `FftStream`, the denoiser's
// 48 kHz intermediate, or how pairing stays in lockstep.

use std::collections::VecDeque;

use anyhow::{Context, Result};

use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, WHISPER_RATE};
use crate::denoise::{Denoiser, DenoiserKind};
use crate::dsp::FftStream;

/// Paired 16 kHz frames: VAD-branch audio for the verdict + whisper-branch
/// audio for the utterance buffer. Always 160 samples each.
pub(crate) type PairedFrames = (Vec<f32>, Vec<f32>);

/// Owns all resampling / denoising state. Fed one raw hw-rate chunk at a
/// time; returns the completed VAD/whisper pairs in order, so the VAD verdict
/// always describes the whisper audio it is attached to.
pub(crate) struct SignalPath {
    vad_path: VadPath,
    pending_vad: VecDeque<Vec<f32>>,
    pending_whisper: VecDeque<Vec<f32>>,
}

/// The VAD branch's signal path, chosen once at startup from `--denoiser`
/// and fixed for the process's lifetime.
enum VadPath {
    /// Default (`none`): a single hw_rate -> 16 kHz hop feeds both the VAD
    /// verdict and the utterance buffer — no 48 kHz intermediate, no
    /// denoiser object. Each completed 16 kHz frame serves both roles, so
    /// VAD/whisper pairing is trivially 1:1.
    Direct { resampler: FftStream },
    /// Opt-in (`deepfilter`): hw_rate -> 48 kHz -> denoise -> 48 kHz ->
    /// 16 kHz for VAD, plus an independent direct hw_rate -> 16 kHz hop for
    /// the whisper branch (which stays raw — the model was trained on raw
    /// audio, and the 48 kHz intermediate exists only so the denoiser, which
    /// is fixed at 48 kHz, can run for the VAD branch).
    Denoised {
        // hw_rate -> 48 kHz (the denoiser's required rate). Its resampler's
        // fixed FFT chunks complete on their own cadence, so drained
        // 480-frames are denoised one by one as they arrive rather than one
        // per input chunk.
        resampler_hw48: FftStream,
        // 48 kHz -> 16 kHz, VAD branch only (post-denoise).
        resampler_vad: FftStream,
        resampler_whisper: FftStream,
        denoiser: Denoiser,
        clean48_vad: Vec<f32>,
    },
}

impl SignalPath {
    pub(crate) fn new(hw_rate: u32, denoiser_kind: DenoiserKind) -> Result<Self> {
        let vad_path = match denoiser_kind {
            DenoiserKind::Bypass => {
                let resampler = FftStream::new(hw_rate, WHISPER_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {WHISPER_RATE})"))?;
                eprintln!("[sttd] denoiser: none");
                VadPath::Direct { resampler }
            }
            DenoiserKind::DeepFilter => {
                let resampler_hw48 = FftStream::new(hw_rate, DENOISE_RATE, FRAME_48K)
                    .with_context(|| format!("failed to create resampler (hw -> {DENOISE_RATE})"))?;
                let (chunk_in, chunk_out) = resampler_hw48.chunk_sizes();
                eprintln!(
                    "[sttd] resampler: {hw_rate} -> {DENOISE_RATE}, \
                     fft_chunk_in={chunk_in} fft_chunk_out={chunk_out} frame_out={FRAME_48K}"
                );

                let resampler_vad = FftStream::new(DENOISE_RATE, WHISPER_RATE, FRAME_16K)
                    .context("failed to create resampler (48k -> 16k, vad branch)")?;

                let resampler_whisper = FftStream::new(hw_rate, WHISPER_RATE, FRAME_16K)
                    .with_context(|| format!("failed to create resampler (hw -> {WHISPER_RATE})"))?;

                let denoiser = Denoiser::new(DenoiserKind::DeepFilter)?;
                VadPath::Denoised {
                    resampler_hw48,
                    resampler_vad,
                    resampler_whisper,
                    denoiser,
                    clean48_vad: vec![0.0f32; FRAME_48K],
                }
            }
        };
        Ok(Self {
            vad_path,
            pending_vad: VecDeque::new(),
            pending_whisper: VecDeque::new(),
        })
    }

    /// Feed one raw hw-rate chunk; returns every completed `(vad16,
    /// whisper16)` pair in order. Each branch's `FftStream` may complete
    /// zero, one, or (on catch-up) several 10 ms frames per call; a branch
    /// running ahead waits in its queue. Both average 100 frames/s, so the
    /// queues stay bounded at a frame or two.
    pub(crate) fn feed(&mut self, frame_hw_raw: &[f32]) -> Vec<PairedFrames> {
        match &mut self.vad_path {
            VadPath::Direct { resampler } => {
                for frame16 in resampler.feed(frame_hw_raw) {
                    self.pending_whisper.push_back(frame16.clone());
                    self.pending_vad.push_back(frame16);
                }
            }
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_whisper,
                denoiser,
                clean48_vad,
            } => {
                for frame48 in resampler_hw48.feed(frame_hw_raw) {
                    denoiser.denoise_frame(&frame48, clean48_vad);
                    self.pending_vad.extend(resampler_vad.feed(clean48_vad));
                }
                self.pending_whisper
                    .extend(resampler_whisper.feed(frame_hw_raw));
            }
        }
        self.drain_paired()
    }

    /// Zero-pad each resampler's trailing partial chunk so tail audio stuck
    /// in an incomplete FFT chunk still reaches the utterance instead of
    /// being lost at shutdown, then return every paired frame left buffered.
    pub(crate) fn flush(&mut self) -> Vec<PairedFrames> {
        match &mut self.vad_path {
            VadPath::Direct { resampler } => {
                for frame16 in resampler.flush() {
                    self.pending_whisper.push_back(frame16.clone());
                    self.pending_vad.push_back(frame16);
                }
            }
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_whisper,
                denoiser,
                clean48_vad,
            } => {
                for frame48 in resampler_hw48.flush() {
                    denoiser.denoise_frame(&frame48, clean48_vad);
                    self.pending_vad.extend(resampler_vad.feed(clean48_vad));
                }
                self.pending_vad.extend(resampler_vad.flush());
                self.pending_whisper.extend(resampler_whisper.flush());
            }
        }
        self.drain_paired()
    }

    /// Drop in-flight audio: paired-frame queues + resampler partial-chunk
    /// state, so the first frames after resume aren't blended from pre-pause
    /// audio. Idempotent.
    pub(crate) fn clear(&mut self) {
        self.pending_vad.clear();
        self.pending_whisper.clear();
        match &mut self.vad_path {
            VadPath::Direct { resampler } => resampler.clear(),
            VadPath::Denoised {
                resampler_hw48,
                resampler_vad,
                resampler_whisper,
                ..
            } => {
                resampler_hw48.clear();
                resampler_vad.clear();
                resampler_whisper.clear();
            }
        }
    }

    fn drain_paired(&mut self) -> Vec<PairedFrames> {
        let mut out = Vec::new();
        while let (Some(vad16), Some(whisper16)) =
            (self.pending_vad.pop_front(), self.pending_whisper.pop_front())
        {
            debug_assert_eq!(vad16.len(), FRAME_16K);
            debug_assert_eq!(whisper16.len(), FRAME_16K);
            out.push((vad16, whisper16));
        }
        out
    }
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

    #[test]
    fn direct_pairs_one_to_one_with_exact_frames() {
        for hw in [16_000, 44_100, 48_000] {
            let mut signal = SignalPath::new(hw, DenoiserKind::Bypass).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let silence = vec![0.0f32; hw_chunk];
            for _ in 0..50 {
                for (vad, whisper) in signal.feed(&silence) {
                    assert_eq!(vad.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(whisper.len(), FRAME_16K, "hw={hw}");
                    assert_eq!(vad, whisper, "hw={hw}: direct hop shares one hop");
                }
            }
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
                (198..=203).contains(&n),
                "hw={hw}: expected ~200 pairs, got {n}"
            );
        }
    }

    #[test]
    fn flush_recovers_trailing_audio() {
        // Two 480-frame feeds at 48k->16k complete one FFT chunk; without a
        // flush the tail would be lost at shutdown.
        let mut signal = SignalPath::new(48_000, DenoiserKind::Bypass).unwrap();
        let hw_chunk = hw_input_frames(48_000);
        assert_eq!(hw_chunk, 480);
        let fed: usize = (0..2)
            .map(|_| signal.feed(&vec![0.1f32; hw_chunk]).len())
            .sum();
        assert_eq!(fed, 1);
        let flushed = signal.flush();
        assert!(!flushed.is_empty(), "flush must recover the tail");
        assert_eq!(fed + flushed.len(), 3);
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
}
