use std::collections::VecDeque;

use crate::config::{
    EARLY_SPLIT_QUIET_FRAMES, EARLY_SPLIT_QUIET_RMS, EARLY_SPLIT_SAMPLES,
    EARLY_SPLIT_SILENCE_FRAMES, FRAME_16K, INTERIM_SPLIT_SAMPLES, INTERIM_SPLIT_SILENCE_FRAMES,
    MAX_UTTERANCE_SAMPLES, MIC_CLIP_RATIO, MIC_CLIP_SAMPLE, MIC_LOW_PEAK, ONSET_SPEECH_FRAMES,
    PREROLL_SAFETY_FRAMES, PREROLL_TRIM_FACTOR, PREROLL_TRIM_FLOOR, RING_FRAMES,
    SILENCE_FRAMES_THRESHOLD, STRONG_RMS_FACTOR, TAIL_MAX_SAMPLES, TAIL_MIN_SAMPLES,
};
use crate::gate::GateConfig;

/// What the acoustics observed about an utterance, independent of what whisper
/// made of it: how many frames the VAD judged to be speech, and how loud that
/// speech was. A downstream gate needs this to tell "the model is unsure about
/// real speech" apart from "the model invented something over silence" — the
/// two cases a single confidence threshold can't separate (see
/// `confidence_floor`).
#[derive(Debug, Clone, Copy)]
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
/// queue boundary.
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
///
/// `hangover_frames` is config, not per-utterance state: `reset`/`discard`
/// never touch it, and `Default` carries the compiled-in default so existing
/// struct literals keep behaving.
pub(crate) struct UtteranceBuffer {
    /// Rolling ~300 ms of recent whisper-branch audio (see the struct doc).
    preroll: VecDeque<f32>,
    /// Whisper-branch audio (16 kHz), from the moment the utterance opens.
    buf: Vec<f32>,
    /// Consecutive non-speech frames since the last speech frame, once the
    /// utterance is open. Drives the close decision (`hangover_frames`).
    silence_frames: u32,
    /// Silence hangover before cutting an utterance, in 10 ms frames
    /// (`--hangover-ms / 10`). Config, never reset.
    hangover_frames: u32,
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
    /// Per-speech-frame RMS (one entry per `speech_frames`, including
    /// pre-open onset frames). Drives the energy-weighted density in
    /// `close`: quiet frames count less than loud ones, so breath/hum
    /// flicker scores below scattered loud speech at the same frame
    /// count. Bounded by the 30 s cap (~3000 entries).
    speech_frame_rms: Vec<f32>,
    /// Consecutive quiet frames (RMS below `EARLY_SPLIT_QUIET_RMS`) while
    /// open. Drives the low-energy early split past `EARLY_SPLIT_SAMPLES`
    /// — a word-boundary dip the VAD latch may ride through.
    quiet_run: u32,
}

impl Default for UtteranceBuffer {
    fn default() -> Self {
        Self {
            preroll: VecDeque::new(),
            buf: Vec::new(),
            silence_frames: 0,
            hangover_frames: SILENCE_FRAMES_THRESHOLD,
            onset_run: 0,
            speech_frames: 0,
            samples_at_last_speech: 0,
            speech_energy_sum_sq: 0.0,
            speech_sample_count: 0,
            speech_frame_rms: Vec::new(),
            quiet_run: 0,
        }
    }
}

impl UtteranceBuffer {
    pub(crate) fn new() -> Self {
        Self {
            preroll: VecDeque::with_capacity(RING_FRAMES * FRAME_16K),
            ..Default::default()
        }
    }

    /// Override the silence hangover (in 10 ms frames, see
    /// `hangover_frames`). Builder-style so `FrameProcessor::new` can apply
    /// `--hangover-ms` once while tests keep the default.
    pub(crate) fn with_hangover_frames(mut self, frames: u32) -> Self {
        self.hangover_frames = frames.max(1);
        self
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
        // Frame loudness drives both the weighted density (speech frames)
        // and the low-energy early split (any frame while open).
        let frame_rms = frame_rms(frame);
        if is_speech {
            self.onset_run += 1;
            self.speech_frames += 1;
            for &s in frame {
                self.speech_energy_sum_sq += (s as f64) * (s as f64);
            }
            self.speech_sample_count += frame.len();
            self.speech_frame_rms.push(frame_rms);

            if self.buf.is_empty() {
                if self.onset_run < ONSET_SPEECH_FRAMES {
                    return FrameAction::Continue;
                }
                // Adaptive pre-roll: prepend the run-up, but trim leading
                // silence down to a safety margin when the onset is loud
                // (see `preroll_trim_skip`). A quiet onset keeps the full
                // ring, as before.
                let skip = self.preroll_trim_skip(frame);
                if skip > 0 {
                    log::debug!(
                        "preroll trim: {} -> {} samples",
                        self.preroll.len(),
                        self.preroll.len() - skip
                    );
                }
                self.buf.extend(self.preroll.iter().skip(skip).copied());
            }
            self.buf.extend_from_slice(frame);
            self.samples_at_last_speech = self.buf.len();
            self.silence_frames = 0;
            // A speech frame is never a quiet run.
            self.quiet_run = 0;
            if self.buf.len() >= MAX_UTTERANCE_SAMPLES {
                FrameAction::Flush
            } else if self.buf.len() >= EARLY_SPLIT_SAMPLES && frame_rms < EARLY_SPLIT_QUIET_RMS {
                // Word-boundary dip inside a long utterance: count it even
                // though the VAD latch still says speech (see the
                // non-speech arm below for the shared flush).
                self.quiet_run += 1;
                if self.quiet_run >= EARLY_SPLIT_QUIET_FRAMES {
                    FrameAction::Flush
                } else {
                    FrameAction::Continue
                }
            } else {
                FrameAction::Continue
            }
        } else {
            self.onset_run = 0;
            if self.buf.is_empty() {
                self.speech_frames = 0;
                self.speech_energy_sum_sq = 0.0;
                self.speech_sample_count = 0;
                self.speech_frame_rms.clear();
                return FrameAction::Continue;
            }
            self.buf.extend_from_slice(frame);
            self.silence_frames += 1;
            if frame_rms < EARLY_SPLIT_QUIET_RMS {
                self.quiet_run += 1;
            } else {
                self.quiet_run = 0;
            }
            if self.silence_frames > self.hangover_frames {
                FrameAction::Flush
            } else if self.buf.len() >= EARLY_SPLIT_SAMPLES
                && (self.silence_frames >= EARLY_SPLIT_SILENCE_FRAMES
                    || self.quiet_run >= EARLY_SPLIT_QUIET_FRAMES)
            {
                // Near whisper's 30 s cap with a real pause — or a 50 ms
                // energy dip — in progress: cut here (in quiet) instead of
                // riding into the hard cap and cutting mid-word. `close`
                // still trims to last speech + adaptive tail.
                FrameAction::Flush
            } else if self.buf.len() >= INTERIM_SPLIT_SAMPLES
                && self.silence_frames >= INTERIM_SPLIT_SILENCE_FRAMES
            {
                // Long monologue with a 300 ms pause in progress: emit the
                // prefix now instead of growing one 30 s block. The
                // continuation starts fresh (pre-roll spans the boundary).
                FrameAction::Flush
            } else {
                FrameAction::Continue
            }
        }
    }

    /// Leading samples of `preroll` to skip when an utterance opens with
    /// `current` (the `ONSET_SPEECH_FRAMES`-th consecutive speech frame).
    ///
    /// The ring always holds the run-up (up to `RING_FRAMES` of audio), but
    /// a loud onset doesn't need all 300 ms of it: scan the ring in 10 ms
    /// frames for the first one at least as loud as the onset burst itself
    /// (`PREROLL_TRIM_FACTOR` of its RMS, floored at `PREROLL_TRIM_FLOOR`),
    /// then keep `PREROLL_SAFETY_FRAMES` of run-up in front of it. A quiet
    /// onset lowers the threshold to the floor instead, so faint run-up
    /// survives; a short ring (startup) is kept whole.
    ///
    /// Only whole leading frames are skipped; a trailing partial frame (if
    /// the ring ever isn't frame-aligned) is always kept, as is the onset
    /// tail itself — it scores far above its own fractional threshold.
    fn preroll_trim_skip(&self, current: &[f32]) -> usize {
        let n = self.preroll.len();
        if n <= PREROLL_SAFETY_FRAMES * FRAME_16K {
            return 0;
        }
        // Onset-burst RMS: the tail of the ring (the earlier onset frames,
        // already recorded) plus the current frame that opens the utterance.
        let tail_samples = ((ONSET_SPEECH_FRAMES as usize).saturating_sub(1) * FRAME_16K).min(n);
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for i in (n - tail_samples)..n {
            let s = self.preroll[i] as f64;
            sum_sq += s * s;
            count += 1;
        }
        for &s in current {
            let s = s as f64;
            sum_sq += s * s;
            count += 1;
        }
        if count == 0 {
            return 0;
        }
        let onset_rms = (sum_sq / count as f64).sqrt() as f32;
        let threshold = (onset_rms * PREROLL_TRIM_FACTOR).max(PREROLL_TRIM_FLOOR);
        let n_frames = n / FRAME_16K;
        let mut first_hit: Option<usize> = None;
        for f in 0..n_frames {
            let mut frame_sum_sq = 0.0f64;
            for i in (f * FRAME_16K)..((f + 1) * FRAME_16K) {
                let s = self.preroll[i] as f64;
                frame_sum_sq += s * s;
            }
            if (frame_sum_sq / FRAME_16K as f64).sqrt() as f32 >= threshold {
                first_hit = Some(f);
                break;
            }
        }
        match first_hit {
            Some(first) => (first.saturating_sub(PREROLL_SAFETY_FRAMES) * FRAME_16K).min(n),
            None => 0,
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
    /// - `min_density`: energy-weighted speech share below the floor (long
    ///   utterances exempt, see `DENSITY_LONG_EXEMPT_SAMPLES`). Quiet frames
    ///   count less than loud ones, so breath/hum flicker at the same frame
    ///   count scores below real speech.
    /// - `min_rms`: VAD-flagged frames present, but too quiet to be real speech.
    pub(crate) fn close(&mut self, gate: &GateConfig) -> Option<Utterance> {
        let speech_frames = self.speech_frames;
        // Adaptive tail: at least TAIL_MIN, at most TAIL_MAX past the last
        // speech frame, extending while the tail stays energetic. Judging
        // density post-trim keeps a short real word with a full silence
        // hangover above the floor, while scattered VAD flicker spread over a
        // long span still scores low.
        let tail_end =
            adaptive_tail_end(&self.buf, self.samples_at_last_speech, gate.min_speech_rms);
        let total_frames = (tail_end / FRAME_16K) as u32;
        let density = speech_frames as f32 / total_frames.max(1) as f32;
        let weighted = weighted_density(&self.speech_frame_rms, gate.min_speech_rms, total_frames);

        let rms = if self.speech_sample_count > 0 {
            (self.speech_energy_sum_sq / self.speech_sample_count as f64).sqrt() as f32
        } else {
            0.0
        };
        // When raw frame density is clearly concentrated (short dense words where
        // pre-roll and tail silence dominate total_frames), use density so quiet-but-dense
        // speech isn't starved by the weighted-density denominator.
        let effective_density = if density >= 0.35 {
            density.max(weighted)
        } else {
            weighted
        };

        // Exemption judged on the same post-trim length as density: the raw
        // buffer still holds the full silence hangover at this point, which
        // would otherwise exempt utterances ~600 ms shorter than intended.
        if let Err(drop) = gate.check_acoustic(
            speech_frames,
            effective_density,
            rms,
            tail_end,
            total_frames,
            self.buf.len(),
        ) {
            log::info!("{}", drop.message());
            self.reset();
            return None;
        }

        // Trim the tail: the hangover decided *when* to close this
        // utterance, but everything past the adaptive tail is silence
        // hangover that doesn't need to reach whisper.
        self.buf.truncate(tail_end);

        let pcm = std::mem::take(&mut self.buf);
        let (peak, clip_ratio) = mic_level(&pcm);
        log::info!(
            "utterance: {} samples ({speech_frames} speech frames, \
             density={density:.3} weighted={weighted:.3}, rms={rms:.5}, peak={peak:.3}) -> whisper",
            pcm.len()
        );
        if peak < MIC_LOW_PEAK {
            log::warn!(
                "mic level low (peak={peak:.3}, rms={rms:.5}): consider raising input gain \
                 (or lowering --min-rms if this was real speech)"
            );
        } else if clip_ratio > MIC_CLIP_RATIO {
            log::warn!(
                "mic clipping ({:.1}% samples >= {:.2}): consider lowering input gain",
                clip_ratio * 100.0,
                MIC_CLIP_SAMPLE
            );
        }
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
        self.speech_frame_rms.clear();
        self.quiet_run = 0;
    }

    /// Discard everything for a pause transition: like `reset`, but also
    /// drops `preroll` so the first utterance after resume can't prepend
    /// pre-pause audio.
    pub(crate) fn discard(&mut self) {
        self.reset();
        self.preroll.clear();
    }
}

/// RMS of one 10 ms frame (linear, not dB). Empty frame is silence.
fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / frame.len() as f64).sqrt() as f32
}

/// Energy-weighted speech share: each VAD-flagged frame contributes
/// `clamp(rms_frame / ref_rms, 0..1)` instead of 1, where `ref_rms` is
/// strong-speech loudness (`min_rms * STRONG_RMS_FACTOR`). Loud frames
/// count fully (identical to frame density for loud speech, so existing
/// tuning holds); quiet frames count fractionally, so breath/hum flicker
/// at the same frame count scores below real speech. `min_rms <= 0`
/// disables weighting (rms gate off means density behaves as before).
fn weighted_density(speech_frame_rms: &[f32], min_rms: f32, total_frames: u32) -> f32 {
    let total = total_frames.max(1) as f32;
    if min_rms <= 0.0 {
        return speech_frame_rms.len() as f32 / total;
    }
    let reference = (min_rms * STRONG_RMS_FACTOR).max(1e-6);
    let weighted: f32 = speech_frame_rms
        .iter()
        .map(|&r| (r / reference).clamp(0.0, 1.0))
        .sum();
    weighted / total
}

/// Adaptive tail end (sample index into `buf`): at least `TAIL_MIN` past
/// the last speech frame, at most `TAIL_MAX`, extending while 10 ms tail
/// frames stay at/above `min_rms`. Quiet tails cut at the minimum (less
/// silence to whisper, higher density); reverberant tails keep what is
/// still energetic. `min_rms <= 0` keeps the legacy fixed tail for tests
/// that disable the rms gate.
fn adaptive_tail_end(buf: &[f32], samples_at_last_speech: usize, min_rms: f32) -> usize {
    use crate::config::TAIL_KEEP_SAMPLES;
    if min_rms <= 0.0 {
        return (samples_at_last_speech + TAIL_KEEP_SAMPLES).min(buf.len());
    }
    let min_end = samples_at_last_speech
        .saturating_add(TAIL_MIN_SAMPLES)
        .min(buf.len());
    let max_end = samples_at_last_speech
        .saturating_add(TAIL_MAX_SAMPLES)
        .min(buf.len());
    let mut end = min_end;
    while end + FRAME_16K <= max_end {
        let rms = frame_rms(&buf[end..end + FRAME_16K]);
        if rms < min_rms {
            break;
        }
        end += FRAME_16K;
    }
    end
}

/// Mic-level advisor measurements over the transmitted utterance:
/// `(peak, clip_ratio)` where `clip_ratio` is the share of samples at or
/// above `MIC_CLIP_SAMPLE`. Pure for tests; warn-lines live in `close`.
fn mic_level(pcm: &[f32]) -> (f32, f32) {
    if pcm.is_empty() {
        return (0.0, 0.0);
    }
    let mut peak = 0.0f32;
    let mut clipped = 0usize;
    for &s in pcm {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
        if a >= MIC_CLIP_SAMPLE {
            clipped += 1;
        }
    }
    (peak, clipped as f32 / pcm.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
        DEFAULT_MIN_SPEECH_RMS, DENSITY_LONG_EXEMPT_SAMPLES, EARLY_SPLIT_QUIET_FRAMES,
        EARLY_SPLIT_SAMPLES, EARLY_SPLIT_SILENCE_FRAMES, FRAME_16K, INTERIM_SPLIT_SAMPLES,
        INTERIM_SPLIT_SILENCE_FRAMES, MAX_UTTERANCE_SAMPLES, MIN_SPEECH_FRAMES,
        ONSET_SPEECH_FRAMES, PREROLL_SAFETY_FRAMES, RING_FRAMES, SILENCE_FRAMES_THRESHOLD,
        STRONG_RMS_FACTOR, TAIL_KEEP_SAMPLES, TAIL_MAX_SAMPLES, TAIL_MIN_SAMPLES,
    };

    fn gate_for(min_rms: f32, min_density: f32) -> GateConfig {
        GateConfig::new(
            min_rms,
            min_density,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        )
    }

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
    fn hangover_default_matches_compiled_constant() {
        assert_eq!(
            UtteranceBuffer::default().hangover_frames,
            SILENCE_FRAMES_THRESHOLD
        );
        assert_eq!(
            UtteranceBuffer::new().hangover_frames,
            SILENCE_FRAMES_THRESHOLD
        );
    }

    #[test]
    fn custom_hangover_closes_earlier_and_later() {
        // Open an utterance the same way twice, then count silence frames to
        // Flush: 300 ms must close well before the 800 ms default, 1200 ms
        // must stay open past it.
        fn frames_to_flush(hangover_frames: u32) -> u32 {
            let mut buffer = UtteranceBuffer::new().with_hangover_frames(hangover_frames);
            let speech = vec![0.5f32; FRAME_16K];
            for _ in 0..ONSET_SPEECH_FRAMES {
                buffer.push_frame(&speech, true);
            }
            assert!(buffer.is_open());
            let silence = vec![0.0f32; FRAME_16K];
            for n in 1..=200u32 {
                if buffer.push_frame(&silence, false) == FrameAction::Flush {
                    return n;
                }
            }
            panic!("hangover {hangover_frames} never flushed");
        }
        // Flush fires when silence_frames *exceeds* the threshold.
        assert_eq!(frames_to_flush(30), 31);
        assert_eq!(frames_to_flush(SILENCE_FRAMES_THRESHOLD), 81);
        assert_eq!(frames_to_flush(120), 121);
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
    fn loud_onset_trims_silence_to_safety_margin() {
        // Full 300 ms ring of true silence, then a loud onset: only the
        // safety margin of silence plus the onset itself may be prepended,
        // not the whole ring.
        let mut buffer = UtteranceBuffer::new();
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..RING_FRAMES {
            buffer.push_frame(&silence, false);
        }
        let speech = vec![0.5f32; FRAME_16K];
        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.buf.is_empty());
        buffer.push_frame(&speech, true);
        assert!(buffer.is_open());
        let expected_frames = PREROLL_SAFETY_FRAMES + ONSET_SPEECH_FRAMES as usize;
        assert_eq!(buffer.buf.len(), expected_frames * FRAME_16K);
        let safety_samples = PREROLL_SAFETY_FRAMES * FRAME_16K;
        assert!(buffer.buf[..safety_samples].iter().all(|&s| s == 0.0));
        assert!(buffer.buf[safety_samples..].iter().all(|&s| s == 0.5));
    }

    #[test]
    fn trim_adapts_to_onset_energy() {
        // Same noisy background (0.01) on both sides: a loud onset treats it
        // as silence and trims to the safety margin, a quiet onset treats it
        // as run-up worth keeping and takes the whole ring.
        fn open_len(background: f32, onset: f32) -> usize {
            let mut buffer = UtteranceBuffer::new();
            let bg = vec![background; FRAME_16K];
            for _ in 0..RING_FRAMES {
                buffer.push_frame(&bg, false);
            }
            let speech = vec![onset; FRAME_16K];
            for _ in 0..ONSET_SPEECH_FRAMES {
                buffer.push_frame(&speech, true);
            }
            assert!(buffer.is_open());
            buffer.buf.len()
        }
        assert_eq!(
            open_len(0.01, 0.5),
            (PREROLL_SAFETY_FRAMES + ONSET_SPEECH_FRAMES as usize) * FRAME_16K
        );
        assert_eq!(open_len(0.01, 0.02), (RING_FRAMES + 1) * FRAME_16K);
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
            buffer.close(&gate_for(0.0, 0.0)).is_none(),
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
            .close(&gate_for(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY))
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
            buffer.close(&gate_for(0.0, DEFAULT_MIN_DENSITY)).is_none(),
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
            .close(&gate_for(0.0, DEFAULT_MIN_DENSITY))
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
        assert!(long.close(&gate_for(0.0, 1.0)).is_some());

        let mut short = UtteranceBuffer {
            buf: vec![0.0f32; 20_000],
            speech_frames: 25,
            samples_at_last_speech: 20_000,
            ..Default::default()
        };
        assert!(short.close(&gate_for(0.0, 1.0)).is_none());
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

        let utterance = buffer.close(&gate_for(0.0, 0.0)).unwrap();
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
        assert!(short.close(&gate_for(0.0, 0.0)).is_none());

        let mut long = UtteranceBuffer {
            buf: vec![0.5f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES,
            samples_at_last_speech: 5_000,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0,
            ..Default::default()
        };
        let utterance = long.close(&gate_for(0.0, 0.0)).unwrap();
        assert_eq!(utterance.pcm.len(), 5_000);
    }

    #[test]
    fn near_cap_pause_splits_early_in_silence() {
        // Past 28 s, a quiet dip already closes: the cut lands in silence
        // instead of riding into the 30 s hard cap mid-word. True silence
        // (0.0) is both VAD non-speech and low-energy — the 50 ms energy
        // path fires first (5 frames), the 100 ms VAD path covers loud
        // non-speech pauses where energy stays high.
        assert!(EARLY_SPLIT_SAMPLES < MAX_UTTERANCE_SAMPLES);
        assert_eq!(EARLY_SPLIT_SILENCE_FRAMES * 10, 100);
        assert_eq!(EARLY_SPLIT_QUIET_FRAMES * 10, 50);
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_SAMPLES / FRAME_16K + 5) {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.buf.len() >= EARLY_SPLIT_SAMPLES);
        assert!(buffer.buf.len() < MAX_UTTERANCE_SAMPLES);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_QUIET_FRAMES - 1) {
            assert_eq!(buffer.push_frame(&silence, false), FrameAction::Continue);
        }
        assert_eq!(
            buffer.push_frame(&silence, false),
            FrameAction::Flush,
            "50 ms quiet dip past 28 s must close early"
        );
        let utterance = buffer.close(&gate_for(0.0, 0.0)).unwrap();
        assert!(utterance.pcm.len() <= MAX_UTTERANCE_SAMPLES);
        assert!(!buffer.is_open());
    }

    #[test]
    fn near_cap_loud_nonspeech_needs_vad_pause() {
        // Loud room noise past 28 s: energy stays high so the quiet path
        // never fires — the 100 ms VAD-silence path still cuts in pause.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_SAMPLES / FRAME_16K + 5) {
            buffer.push_frame(&speech, true);
        }
        // Loud non-speech: VAD says silence, RMS stays above the quiet floor.
        let loud_silence = vec![0.02f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_SILENCE_FRAMES - 1) {
            assert_eq!(
                buffer.push_frame(&loud_silence, false),
                FrameAction::Continue
            );
        }
        assert_eq!(
            buffer.push_frame(&loud_silence, false),
            FrameAction::Flush,
            "100 ms VAD pause past 28 s must close even when loud"
        );
    }

    #[test]
    fn short_pause_below_window_does_not_split() {
        // Same 90 ms pause at small sizes is just in-utterance padding.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..100 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..9 {
            assert_eq!(buffer.push_frame(&silence, false), FrameAction::Continue);
        }
        assert!(buffer.is_open());
    }

    #[test]
    fn discard_drops_open_utterance_and_preroll() {
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.is_open());
        buffer.discard();
        assert!(!buffer.is_open());
        // After discard the next onset must not prepend pre-pause audio:
        // feed a fresh onset and check the buffer holds only the new
        // frames, with no 0.3-valued pre-roll from before the discard.
        let fresh = vec![0.7f32; FRAME_16K];
        for _ in 0..ONSET_SPEECH_FRAMES {
            buffer.push_frame(&fresh, true);
        }
        assert!(buffer.is_open());
        assert!(buffer.buf.iter().all(|&s| s == 0.7));
    }

    #[test]
    fn weighted_density_matches_frame_for_loud() {
        // Loud speech counts fully: identical to frame density, so existing
        // tuning holds.
        let rms = vec![0.3f32; 25];
        assert!((weighted_density(&rms, DEFAULT_MIN_SPEECH_RMS, 50) - 0.5).abs() < 1e-6);
        // Quiet frames count fractionally: same 25 frames at min_rms score
        // 1/STRONG below loud speech.
        let quiet = vec![DEFAULT_MIN_SPEECH_RMS; 25];
        let w = weighted_density(&quiet, DEFAULT_MIN_SPEECH_RMS, 50);
        let expected = 25.0 * (1.0 / STRONG_RMS_FACTOR) / 50.0;
        assert!((w - expected).abs() < 1e-6, "w={w}");
        // rms gate off disables weighting.
        assert!((weighted_density(&quiet, 0.0, 50) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adaptive_tail_cuts_quiet_extends_loud() {
        // Quiet tail: minimum survives.
        let mut buf = vec![0.3f32; 5_000];
        buf.extend(vec![0.0f32; 10_000]);
        let end = adaptive_tail_end(&buf, 5_000, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(end, 5_000 + TAIL_MIN_SAMPLES);
        // Loud tail: extends while energetic, capped at max.
        let mut buf = vec![0.3f32; 5_000];
        buf.extend(vec![0.05f32; 10_000]);
        let end = adaptive_tail_end(&buf, 5_000, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(end, 5_000 + TAIL_MAX_SAMPLES);
        // Legacy path for tests with the rms gate off.
        let end = adaptive_tail_end(&buf, 5_000, 0.0);
        assert_eq!(end, 5_000 + TAIL_KEEP_SAMPLES);
    }

    #[test]
    fn mic_level_measures_peak_and_clipping() {
        assert_eq!(mic_level(&[]), (0.0, 0.0));
        let (peak, clip) = mic_level(&[0.01, -0.02, 0.015]);
        assert!((peak - 0.02).abs() < 1e-6);
        assert_eq!(clip, 0.0);
        let loud = vec![1.0f32; 100];
        let (peak, clip) = mic_level(&loud);
        assert!((peak - 1.0).abs() < 1e-6);
        assert!((clip - 1.0).abs() < 1e-6);
        assert!(MIC_LOW_PEAK > 0.0);
        assert_eq!(EARLY_SPLIT_QUIET_FRAMES, 5);
    }

    #[test]
    fn interim_split_emits_prefix_past_8s() {
        // Past 8 s, a 300 ms VAD pause emits the prefix instead of growing
        // one 30 s block; the same pause below 8 s is just padding.
        assert!(INTERIM_SPLIT_SAMPLES < EARLY_SPLIT_SAMPLES);
        assert_eq!(INTERIM_SPLIT_SILENCE_FRAMES * 10, 300);
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..(INTERIM_SPLIT_SAMPLES / FRAME_16K + 5) {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.buf.len() >= INTERIM_SPLIT_SAMPLES);
        assert!(buffer.buf.len() < EARLY_SPLIT_SAMPLES);
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(INTERIM_SPLIT_SILENCE_FRAMES - 1) {
            assert_eq!(buffer.push_frame(&silence, false), FrameAction::Continue);
        }
        assert_eq!(
            buffer.push_frame(&silence, false),
            FrameAction::Flush,
            "300 ms pause past 8 s must emit the prefix early"
        );
        let utterance = buffer.close(&gate_for(0.0, 0.0)).unwrap();
        assert!(utterance.pcm.len() < MAX_UTTERANCE_SAMPLES);
        assert!(!buffer.is_open());
    }

    #[test]
    fn interim_pause_below_window_does_not_split() {
        // Same 300 ms pause at small sizes stays in-utterance padding.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..100 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..INTERIM_SPLIT_SILENCE_FRAMES {
            assert_eq!(buffer.push_frame(&silence, false), FrameAction::Continue);
        }
        assert!(buffer.is_open());
    }
}
