use std::collections::VecDeque;

use crate::config::{
    DENSITY_LONG_EXEMPT_SAMPLES, FRAME_16K, MAX_UTTERANCE_SAMPLES, MIN_SPEECH_FRAMES,
    ONSET_SPEECH_FRAMES, RING_FRAMES, SILENCE_FRAMES_THRESHOLD, TAIL_KEEP_SAMPLES,
};

/// What the acoustics observed about an utterance, independent of what whisper
/// made of it: how many frames the VAD judged to be speech, and how loud that
/// speech was. A downstream gate needs this to tell "the model is unsure about
/// real speech" apart from "the model invented something over silence" — the
/// two cases a single confidence threshold can't separate (see
/// `confidence_floor`).
pub(crate) struct Evidence {
    /// Total VAD-flagged speech frames over the utterance's life (see
    /// `UtteranceBuffer::speech_frames`).
    pub(crate) speech_frames: u32,
    /// Transmitted frames whisper actually receives (post-trim, i.e.
    /// `samples_at_last_speech + TAIL_KEEP_SAMPLES` in frames) — the
    /// denominator for the speech-density gate.
    pub(crate) total_frames: u32,
    /// RMS over VAD-flagged speech frames only (see
    /// `UtteranceBuffer::speech_energy_sum_sq`).
    pub(crate) rms: f32,
}

impl Evidence {
    /// Share of transmitted frames that were VAD-flagged as speech.
    pub(crate) fn density(&self) -> f32 {
        self.speech_frames as f32 / self.total_frames.max(1) as f32
    }
}

/// One utterance that cleared `UtteranceBuffer::close` and is ready for
/// whisper: the trimmed 16 kHz audio plus the `Evidence` gathered while
/// buffering it. Evidence that used to be computed and then thrown away at the
/// `Sender<Vec<f32>>` boundary.
pub(crate) struct Utterance {
    pub(crate) pcm: Vec<f32>,
    pub(crate) evidence: Evidence,
}

/// Whether `UtteranceBuffer::push_frame` wants the caller to close the
/// utterance and hand it on.
#[derive(PartialEq, Eq, Debug)]
pub(crate) enum FrameAction {
    Continue,
    Flush,
}

/// Per-utterance buffering state, reset every time an utterance is closed
/// (whether accepted or dropped). Kept as one struct rather than loose
/// `&mut` parameters so `close` can reset everything atomically and so new
/// gates (see `close`) can add fields without touching every call site.
///
/// Also owns the pre-roll ring, because it has to be read on the frame that
/// opens an utterance and written on *every* frame regardless of the VAD
/// verdict — an invariant the caller could only uphold by remembering to. Only
/// `reset` clears it: the ring deliberately spans utterances, holding the
/// run-up to the *next* onset.
#[derive(Default)]
pub(crate) struct UtteranceBuffer {
    /// Rolling ~300 ms of recent whisper-branch audio (see the struct doc).
    preroll: VecDeque<f32>,
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
    /// trailing silence down to `TAIL_KEEP_SAMPLES` at close time.
    samples_at_last_speech: usize,
    /// Sum of squared samples from VAD-flagged speech frames only (silence
    /// padding excluded), paired with `speech_sample_count` so
    /// `close` can compute an RMS over just the speech content.
    speech_energy_sum_sq: f64,
    speech_sample_count: usize,
}

impl UtteranceBuffer {
    pub(crate) fn new() -> Self {
        Self {
            preroll: VecDeque::with_capacity(RING_FRAMES * FRAME_16K),
            ..Default::default()
        }
    }

    /// Whether an utterance has opened and is still accumulating.
    pub(crate) fn is_open(&self) -> bool {
        !self.buf.is_empty()
    }

    /// Buffer one 16 kHz frame with the VAD's verdict on it, and report whether
    /// the utterance now wants closing. Both behaviors below are load-bearing
    /// for hallucination avoidance:
    /// - An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive
    ///   speech frames, not the first one, so a single VAD false positive no
    ///   longer opens (and thus sends) a full utterance. The onset frames
    ///   themselves aren't lost: `record_preroll` holds them by the time the
    ///   utterance opens, since it runs on every frame regardless of
    ///   `is_speech`.
    /// - Once open, every frame is appended regardless of `is_speech` (non-
    ///   speech frames become natural in-utterance padding instead of being
    ///   spliced out) — but `speech_frames`/`samples_at_last_speech`/energy are
    ///   tracked only from VAD-flagged frames, so `close` can later gate and
    ///   trim on actual speech content instead of total buffer length.
    ///
    /// Split from `absorb` only so the pre-roll ring can't be skipped by
    /// `absorb`'s early returns. Unit-testable without a live VAD signal.
    pub(crate) fn push_frame(&mut self, frame: &[f32], is_speech: bool) -> FrameAction {
        let action = self.absorb(frame, is_speech);
        self.record_preroll(frame);
        action
    }

    fn absorb(&mut self, frame: &[f32], is_speech: bool) -> FrameAction {
        if is_speech {
            self.onset_run += 1;
            self.speech_frames += 1;
            for &s in frame {
                self.speech_energy_sum_sq += (s as f64) * (s as f64);
            }
            self.speech_sample_count += frame.len();

            if self.buf.is_empty() {
                if self.onset_run < ONSET_SPEECH_FRAMES {
                    return FrameAction::Continue;
                }
                self.buf.extend(self.preroll.iter().copied());
            }
            self.buf.extend_from_slice(frame);
            self.samples_at_last_speech = self.buf.len();
            self.silence_frames = 0;
            if self.buf.len() >= MAX_UTTERANCE_SAMPLES {
                FrameAction::Flush
            } else {
                FrameAction::Continue
            }
        } else {
            self.onset_run = 0;
            if self.buf.is_empty() {
                return FrameAction::Continue;
            }
            self.buf.extend_from_slice(frame);
            self.silence_frames += 1;
            if self.silence_frames > SILENCE_FRAMES_THRESHOLD {
                FrameAction::Flush
            } else {
                FrameAction::Continue
            }
        }
    }

    /// Record `frame` as recent audio, so it can be prepended as pre-roll if an
    /// utterance opens in the next `RING_FRAMES` frames.
    fn record_preroll(&mut self, frame: &[f32]) {
        self.preroll.extend(frame.iter().copied());
        while self.preroll.len() > RING_FRAMES * FRAME_16K {
            self.preroll.pop_front();
        }
    }

    /// Decide whether the accumulated utterance is real speech and, if so, trim
    /// its trailing silence and return it. Resets `self` to its (empty) default
    /// state either way, leaving the pre-roll ring intact.
    ///
    /// Three gates, all defending against the same failure mode: a VAD false
    /// positive (a click, a breath, mains hum) opening an utterance that
    /// pre-roll and silence hangover then pad out to ~1-2s of near-silence —
    /// which whisper, lacking any working confidence gate of its own (see
    /// `DEFAULT_MIN_CONFIDENCE`'s doc comment), would otherwise happily
    /// hallucinate a sentence for:
    /// - `MIN_SPEECH_FRAMES`: not enough VAD-flagged speech overall.
    /// - `min_density`: enough speech frames, but scattered across so much
    ///   silence that the transmitted audio is mostly not-speech (long
    ///   utterances exempt, see `DENSITY_LONG_EXEMPT_SAMPLES`).
    /// - `min_rms`: VAD-flagged frames present, but too quiet to be real speech.
    pub(crate) fn close(&mut self, min_rms: f32, min_density: f32) -> Option<Utterance> {
        let speech_frames = self.speech_frames;
        if speech_frames < MIN_SPEECH_FRAMES {
            eprintln!(
                "[sttd] dropping utterance ({speech_frames} speech frames, {} samples)",
                self.buf.len()
            );
            self.reset();
            return None;
        }

        // What whisper would actually receive (see the trim below): judging
        // density post-trim keeps a short real word with a full silence
        // hangover above the floor, while scattered VAD flicker spread over a
        // long span still scores low.
        let tail_end = (self.samples_at_last_speech + TAIL_KEEP_SAMPLES).min(self.buf.len());
        let total_frames = (tail_end / FRAME_16K) as u32;
        let density = speech_frames as f32 / total_frames.max(1) as f32;
        // Exemption judged on the same post-trim length as density: the raw
        // buffer still holds the full silence hangover at this point, which
        // would otherwise exempt utterances ~600 ms shorter than intended.
        if tail_end <= DENSITY_LONG_EXEMPT_SAMPLES
            && min_density > 0.0
            && density < min_density
        {
            eprintln!(
                "[sttd] dropping low-density utterance ({speech_frames} speech frames \
                 in {total_frames} frames, density={density:.3} < {min_density:.3})"
            );
            self.reset();
            return None;
        }

        let rms = if self.speech_sample_count > 0 {
            (self.speech_energy_sum_sq / self.speech_sample_count as f64).sqrt() as f32
        } else {
            0.0
        };
        if rms < min_rms {
            eprintln!(
                "[sttd] dropping low-energy utterance ({speech_frames} speech frames, \
                 rms={rms:.5} < {min_rms:.5})"
            );
            self.reset();
            return None;
        }

        // Trim the tail: `SILENCE_FRAMES_THRESHOLD` decided *when* to close this
        // utterance, but everything past `samples_at_last_speech +
        // TAIL_KEEP_SAMPLES` is silence hangover that doesn't need to reach
        // whisper.
        self.buf.truncate(tail_end);

        let pcm = std::mem::take(&mut self.buf);
        eprintln!(
            "[sttd] utterance: {} samples ({speech_frames} speech frames, \
             density={density:.3}, rms={rms:.5}) -> whisper",
            pcm.len()
        );
        let utterance = Utterance {
            pcm,
            evidence: Evidence {
                speech_frames,
                total_frames,
                rms,
            },
        };
        self.reset();
        Some(utterance)
    }

    /// Clear the utterance state for the next one, keeping `preroll` (see the
    /// struct doc). Every field other than `preroll` must be reset here.
    fn reset(&mut self) {
        self.buf.clear();
        self.silence_frames = 0;
        self.onset_run = 0;
        self.speech_frames = 0;
        self.samples_at_last_speech = 0;
        self.speech_energy_sum_sq = 0.0;
        self.speech_sample_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS, DENSITY_LONG_EXEMPT_SAMPLES, FRAME_16K,
        MIN_SPEECH_FRAMES, ONSET_SPEECH_FRAMES, SILENCE_FRAMES_THRESHOLD, TAIL_KEEP_SAMPLES,
    };

    #[test]
    fn push_frame_keeps_non_speech_inside_open_utterance() {
        // Regression for the original bug: non-speech frames in the middle
        // of an utterance used to be silently dropped instead of appended.
        let mut buffer = UtteranceBuffer {
            buf: vec![0.5f32; 10],
            ..Default::default()
        };
        let frame = vec![0.0f32; FRAME_16K];
        let action = buffer.push_frame(&frame, false);
        assert_eq!(action, FrameAction::Continue);
        assert_eq!(buffer.buf.len(), 10 + FRAME_16K);
        assert_eq!(buffer.silence_frames, 1);
    }

    #[test]
    fn push_frame_ignores_non_speech_before_utterance_starts() {
        let mut buffer = UtteranceBuffer::default();
        let frame = vec![0.0f32; FRAME_16K];
        assert_eq!(buffer.push_frame(&frame, false), FrameAction::Continue);
        assert!(buffer.buf.is_empty());
    }

    #[test]
    fn single_vad_hit_does_not_open_utterance() {
        // Regression for the production bug: one VAD false positive (a
        // click, a breath) used to open a full utterance immediately.
        let mut buffer = UtteranceBuffer::default();
        let frame = vec![0.5f32; FRAME_16K];

        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            assert_eq!(buffer.push_frame(&frame, true), FrameAction::Continue);
            assert!(
                buffer.buf.is_empty(),
                "must not open before ONSET_SPEECH_FRAMES"
            );
        }
        // The frame that reaches ONSET_SPEECH_FRAMES opens it.
        buffer.push_frame(&frame, true);
        assert!(!buffer.buf.is_empty(), "must open at ONSET_SPEECH_FRAMES");
    }

    #[test]
    fn push_frame_prepends_preroll_on_utterance_start() {
        // The ring accumulates while the utterance is still closed, so by the
        // time it opens the run-up is already there to be prepended. Driving
        // frames through `push_frame` rather than pre-filling the ring is the
        // point: the ring is an implementation detail of the buffer now.
        let mut buffer = UtteranceBuffer::new();
        let silence = vec![0.25f32; FRAME_16K];
        for _ in 0..2 {
            buffer.push_frame(&silence, false);
        }

        let speech = vec![0.5f32; FRAME_16K];
        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.buf.is_empty());

        buffer.push_frame(&speech, true);
        // The 2 pre-roll frames, then the ONSET_SPEECH_FRAMES frames that opened it.
        let preroll_samples = FRAME_16K * 2;
        assert_eq!(
            buffer.buf.len(),
            preroll_samples + FRAME_16K * ONSET_SPEECH_FRAMES as usize
        );
        assert!(buffer.buf[..preroll_samples].iter().all(|&s| s == 0.25));
        assert!(buffer.buf[preroll_samples..].iter().all(|&s| s == 0.5));
    }

    #[test]
    fn push_frame_reports_flush_after_silence_threshold() {
        let mut buffer = UtteranceBuffer {
            buf: vec![0.5f32; 10],
            silence_frames: SILENCE_FRAMES_THRESHOLD,
            ..Default::default()
        };
        let frame = vec![0.0f32; FRAME_16K];
        assert_eq!(buffer.push_frame(&frame, false), FrameAction::Flush);
    }

    #[test]
    fn utterance_from_log_is_rejected() {
        // Regression for the production log: a 7-speech-frame VAD blip,
        // padded by pre-roll and a full silence hangover, used to produce
        // an utterance that whisper then hallucinated "Dziękuje za uwagę."
        // for.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        let silence = vec![0.0f32; FRAME_16K];

        for _ in 0..7 {
            buffer.push_frame(&speech, true);
        }
        let mut action = FrameAction::Continue;
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            action = buffer.push_frame(&silence, false);
            if action == FrameAction::Flush {
                break;
            }
        }
        assert_eq!(
            action,
            FrameAction::Flush,
            "silence hangover should eventually close the utterance"
        );

        // rms gate disabled: isolate MIN_SPEECH_FRAMES
        assert!(
            buffer.close(0.0, 0.0).is_none(),
            "an utterance with too little VAD-flagged speech must be dropped"
        );
    }

    #[test]
    fn real_utterance_survives_gates() {
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame(&silence, false);
        }

        let utterance = buffer
            .close(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY)
            .expect("real speech must be sent to whisper");
        assert!(!utterance.pcm.is_empty());
        assert_eq!(utterance.evidence.speech_frames, 25);
        assert!(utterance.evidence.total_frames > 0);
        assert!(utterance.evidence.density() >= DEFAULT_MIN_DENSITY);
        assert!(utterance.evidence.rms > 0.0);
    }

    #[test]
    fn scattered_flicker_fails_density_gate() {
        // 25 speech frames — past MIN_SPEECH_FRAMES — but scattered across a
        // long silence span (VAD flicker on noise, never 81 silent frames in
        // a row so the utterance stays open). Post-trim the transmitted audio
        // is ~200 frames with only 25 of them speech: mostly silence that
        // whisper would hallucinate boilerplate for.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..5 {
            for _ in 0..5 {
                buffer.push_frame(&speech, true);
            }
            for _ in 0..40 {
                buffer.push_frame(&silence, false);
            }
        }
        let mut action = FrameAction::Continue;
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            action = buffer.push_frame(&silence, false);
            if action == FrameAction::Flush {
                break;
            }
        }
        assert_eq!(action, FrameAction::Flush);
        // rms gate disabled: isolate the density gate.
        assert!(
            buffer.close(0.0, DEFAULT_MIN_DENSITY).is_none(),
            "scattered speech across mostly silence must be dropped"
        );
    }

    #[test]
    fn dense_short_utterance_passes_density_gate() {
        // Consecutive speech: post-trim density ~0.5, well clear of the floor.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame(&silence, false);
        }
        let utterance = buffer
            .close(0.0, DEFAULT_MIN_DENSITY)
            .expect("dense speech must be sent to whisper");
        assert!(utterance.evidence.density() >= DEFAULT_MIN_DENSITY);
    }

    #[test]
    fn long_utterance_is_exempt_from_density_gate() {
        // Same low density (~0.08) on both sides of the exemption boundary:
        // the long one skips the gate even at a floor of 1.0, the short one
        // is dropped. rms gate disabled on both (rms 0.0 == min 0.0 passes).
        let mut long = UtteranceBuffer {
            buf: vec![0.0f32; DENSITY_LONG_EXEMPT_SAMPLES + 2_000],
            speech_frames: 25,
            samples_at_last_speech: DENSITY_LONG_EXEMPT_SAMPLES + 2_000,
            ..Default::default()
        };
        assert!(long.close(0.0, 1.0).is_some());

        let mut short = UtteranceBuffer {
            buf: vec![0.0f32; 20_000],
            speech_frames: 25,
            samples_at_last_speech: 20_000,
            ..Default::default()
        };
        assert!(short.close(0.0, 1.0).is_none());
    }

    #[test]
    fn tail_is_trimmed_to_200ms() {
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        let samples_at_last_speech = buffer.samples_at_last_speech;
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame(&silence, false);
        }

        let utterance = buffer.close(0.0, 0.0).unwrap();
        assert_eq!(
            utterance.pcm.len(),
            samples_at_last_speech + TAIL_KEEP_SAMPLES
        );
    }

    #[test]
    fn close_rejects_short_and_returns_long() {
        let mut short = UtteranceBuffer {
            buf: vec![0.0f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES - 1,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0, // high RMS: isolate the frame-count gate
            ..Default::default()
        };
        assert!(short.close(0.0, 0.0).is_none());

        let mut long = UtteranceBuffer {
            buf: vec![0.5f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES,
            samples_at_last_speech: 5_000,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0,
            ..Default::default()
        };
        let utterance = long.close(0.0, 0.0).unwrap();
        assert_eq!(utterance.pcm.len(), 5_000);
    }
}
