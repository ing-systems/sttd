use std::time::Instant;

use crate::config::{
    EARLY_SPLIT_QUIET_FRAMES, EARLY_SPLIT_QUIET_RMS, EARLY_SPLIT_SAMPLES,
    EARLY_SPLIT_SILENCE_FRAMES, FRAME_16K, INTERIM_SPLIT_SAMPLES, INTERIM_SPLIT_SILENCE_FRAMES,
    MAX_UTTERANCE_SAMPLES, MIC_CLIP_RATIO, MIC_CLIP_SAMPLE, MIC_LOW_PEAK, MIN_SPEECH_FRAMES,
    ONSET_SPEECH_FRAMES, PREROLL_SAFETY_FRAMES, PREROLL_TRIM_FACTOR, PREROLL_TRIM_FLOOR,
    RING_FRAMES, SILENCE_FRAMES_THRESHOLD, STREAM_MIN_SAMPLES, STREAM_STEP_SAMPLES,
    STRONG_RMS_FACTOR, TAIL_MAX_SAMPLES, TAIL_MIN_SAMPLES,
};
use crate::gate::GateConfig;
use crate::noise::frame_rms;

/// Pre-roll ring capacity in samples: `RING_FRAMES` of 16 kHz audio (300 ms).
/// A fixed array + head index, not a growable `VecDeque`: the ring never
/// changes size, so the hot path (`record_preroll`, every frame) writes into
/// preallocated storage instead of a deque that shifts its whole live tail on
/// every `pop_front`.
const PREROLL_CAPACITY: usize = RING_FRAMES * FRAME_16K;

/// What the acoustics observed about an utterance, independent of what the model
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
    /// Transmitted frames the model actually receives (post-trim, i.e.
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
/// Model input: the trimmed 16 kHz audio plus the `Evidence` gathered while
/// buffering it. Evidence that used to be computed and then thrown away at the
/// queue boundary.
pub(crate) struct Utterance {
    pub(crate) pcm: Vec<f32>,
    pub(crate) evidence: Evidence,
    /// True when this utterance continues a split prefix (interim 8s, early
    /// 28s, or 30s hard-cap cut) rather than ending on a hangover pause.
    /// The ASR worker uses it to strip text repeated across the split seam
    /// (see `asr::condition::remove_split_overlap`); hangover closes leave it false so
    /// distinct sentences are never merged.
    pub(crate) split_continuation: bool,
    /// True when this utterance was cut by the pause edge rather than by a
    /// silence hangover: the last thing said before the run was held, which no
    /// hangover will ever get to close. The worker transcribes it while the
    /// pause holds (see `asr::worker::keep_while_paused`) instead of dropping it as
    /// stale backlog.
    pub(crate) pause_flush: bool,
    /// When the buffer closed this utterance, i.e. when its text started
    /// waiting for the model. Stamped by `close` and read by the worker to
    /// tell a sentence that closed just before a pause (still what the
    /// speaker just said) from a backlog the model is minutes behind on.
    pub(crate) queued_at: Instant,
}

impl Utterance {
    /// Mark this as the utterance the pause edge flushed (see the
    /// `pause_flush` field). Called by `FrameProcessor::flush_on_pause`
    /// after the cut, so it covers an utterance that the trailing flush
    /// closed rather than the open buffer.
    pub(crate) fn mark_pause_flush(&mut self) {
        self.pause_flush = true;
    }
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
    /// Rolling ~300 ms of recent model-branch audio (see the struct doc): a
    /// fixed ring where `preroll[..preroll_len]` are live, oldest first, and
    /// the oldest sample sits at `preroll_head`.
    preroll: [f32; PREROLL_CAPACITY],
    /// Live samples in `preroll` (`<= PREROLL_CAPACITY`).
    preroll_len: usize,
    /// Index of the oldest live sample; wraps once the ring is full.
    preroll_head: usize,
    /// Model-branch audio (16 kHz), from the moment the utterance opens.
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
    /// Set when the pending `Flush` came from a split (interim/early/hard
    /// cap) rather than a hangover close. Consumed by `close` into
    /// `Utterance::split_continuation`, then cleared by `reset`.
    pending_split: bool,
    /// Length of the audio handed out by the last `partial_snapshot` of this
    /// utterance (`0` = none yet). The next one waits for `STREAM_STEP_SAMPLES`
    /// more. Cleared by `reset`.
    partial_end: usize,
}

impl Default for UtteranceBuffer {
    fn default() -> Self {
        Self {
            preroll: [0.0; PREROLL_CAPACITY],
            preroll_len: 0,
            preroll_head: 0,
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
            pending_split: false,
            partial_end: 0,
        }
    }
}

impl UtteranceBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Override the silence hangover (in 10 ms frames, see
    /// `hangover_frames`). Builder-style so `FrameProcessor::new` can apply
    /// `--hangover-ms` once while tests keep the default.
    pub(crate) fn with_hangover_frames(mut self, frames: u32) -> Self {
        self.hangover_frames = frames.max(1);
        self
    }

    /// Whether an utterance has opened and is still accumulating.
    pub(crate) const fn is_open(&self) -> bool {
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
    /// Test/default entry point: acoustic evidence is scored from `frame`'s
    /// own RMS (see `push_frame_evidenced` for the production path, which
    /// scores evidence from a separately-supplied pre-AGC RMS instead).
    /// Test-only in production (`FrameProcessor` calls `push_frame_evidenced`
    /// directly), kept as the ergonomic entry point for the many tests below
    /// that don't care about the AGC distinction.
    #[allow(dead_code)]
    pub(crate) fn push_frame(&mut self, frame: &[f32], is_speech: bool) -> FrameAction {
        let rms = frame_rms(frame);
        self.push_frame_evidenced(frame, is_speech, rms, rms)
    }

    /// Same as `push_frame`, but scores acoustic evidence
    /// (`speech_energy_sum_sq`/`speech_frame_rms`, hence `Evidence.rms`)
    /// from `evidence_rms` instead of `frame`'s own RMS, and takes the
    /// quiet-run/early-split loudness as `frame_rms` instead of measuring
    /// `frame` itself. `frame` is still what gets appended to the buffered
    /// utterance (and sent to the model) — only the two loudness
    /// measurements are redirected.
    ///
    /// `FrameProcessor` passes the *pre-AGC* model-frame RMS (the first value
    /// `Agc::process_pair` returns) as `evidence_rms`, so evidence
    /// describes what the microphone actually heard rather than what the
    /// shared-gain AGC did to it afterward: `Agc` can amplify quiet-but-real
    /// speech up to 8x toward `--target-rms`, which used to amplify the
    /// evidence RMS regardless of how quiet the speech actually was. It passes
    /// the *post-AGC* RMS (the second value) as `frame_rms`, because the quiet
    /// run and the early-split gate judge the audio that actually reaches the
    /// model.
    pub(crate) fn push_frame_evidenced(
        &mut self,
        frame: &[f32],
        is_speech: bool,
        evidence_rms: f32,
        frame_rms: f32,
    ) -> FrameAction {
        let action = self.absorb(frame, is_speech, evidence_rms, frame_rms);
        self.record_preroll(frame);
        action
    }

    /// Record one open-utterance frame toward the quiet run that drives the
    /// low-energy early-split gate. A quiet frame extends the run; a louder
    /// frame resets it. Split checks stay with the caller: the speech arm
    /// counts only past the early-split length while the VAD verdict still
    /// says speech, the non-speech arm counts every frame.
    fn note_quiet(&mut self, frame_rms: f32) {
        if frame_rms < EARLY_SPLIT_QUIET_RMS {
            self.quiet_run += 1;
        } else {
            self.quiet_run = 0;
        }
    }

    /// Buffer one frame against the current VAD verdict. Both arms share the
    /// same three-step shape — account for the frame, append it, then decide
    /// whether the accumulated utterance has reached a split point — but not
    /// the same rules, so they stay in separate functions.
    fn absorb(
        &mut self,
        frame: &[f32],
        is_speech: bool,
        evidence_rms: f32,
        frame_rms: f32,
    ) -> FrameAction {
        if is_speech {
            self.absorb_speech(frame, evidence_rms, frame_rms)
        } else {
            self.absorb_silence(frame, frame_rms)
        }
    }

    /// Speech frame: extend the onset run, accumulate acoustic evidence from
    /// `evidence_rms` (pre-AGC), prepend the trimmed pre-roll when this frame
    /// opens the utterance, then test the hard cap and the quiet dip.
    fn absorb_speech(&mut self, frame: &[f32], evidence_rms: f32, frame_rms: f32) -> FrameAction {
        self.onset_run += 1;
        self.speech_frames += 1;
        // `evidence_rms^2 * len` is exactly the sum of squared samples a
        // frame at that RMS would have (RMS is defined as
        // sqrt(sum_sq/len)), so this stays an exact accumulator (not an
        // approximation) whenever `evidence_rms == frame_rms`,
        // i.e. for every `push_frame` (test/default) call.
        self.speech_energy_sum_sq += (evidence_rms as f64).powi(2) * frame.len() as f64;
        self.speech_sample_count += frame.len();
        self.speech_frame_rms.push(evidence_rms);

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
                    self.preroll_len,
                    self.preroll_len - skip
                );
            }
            self.extend_from_preroll(skip);
        }
        self.buf.extend_from_slice(frame);
        self.samples_at_last_speech = self.buf.len();
        self.silence_frames = 0;
        if self.buf.len() >= MAX_UTTERANCE_SAMPLES {
            self.quiet_run = 0;
            self.pending_split = true;
            FrameAction::Flush
        } else if self.buf.len() >= EARLY_SPLIT_SAMPLES && frame_rms < EARLY_SPLIT_QUIET_RMS {
            // Word-boundary dip inside a long utterance: count it even
            // though the VAD latch still says speech (see the
            // non-speech arm below for the shared flush). Must not reset
            // to 0 first, or a run of quiet-but-VAD-flagged frames could
            // never accumulate past 1.
            self.note_quiet(frame_rms);
            if self.quiet_run >= EARLY_SPLIT_QUIET_FRAMES {
                self.pending_split = true;
                FrameAction::Flush
            } else {
                FrameAction::Continue
            }
        } else {
            // Speech frame loud enough that it isn't part of a quiet run.
            self.quiet_run = 0;
            FrameAction::Continue
        }
    }

    /// Non-speech frame: reset the onset run, append in-utterance padding
    /// (never spliced out), then test the hangover close and the two
    /// pause-driven splits.
    fn absorb_silence(&mut self, frame: &[f32], frame_rms: f32) -> FrameAction {
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
        self.note_quiet(frame_rms);
        if self.silence_frames > self.hangover_frames {
            self.pending_split = false;
            FrameAction::Flush
        } else if self.buf.len() >= EARLY_SPLIT_SAMPLES
            && (self.silence_frames >= EARLY_SPLIT_SILENCE_FRAMES
                || self.quiet_run >= EARLY_SPLIT_QUIET_FRAMES)
        {
            // Near the 30 s cap with a real pause — or a 50 ms
            // energy dip — in progress: cut here (in quiet) instead of
            // riding into the hard cap and cutting mid-word. `close`
            // still trims to last speech + adaptive tail.
            self.pending_split = true;
            FrameAction::Flush
        } else if self.buf.len() >= INTERIM_SPLIT_SAMPLES
            && self.silence_frames >= INTERIM_SPLIT_SILENCE_FRAMES
        {
            // Long monologue with a 300 ms pause in progress: emit the
            // prefix now instead of growing one 30 s block. The
            // continuation starts fresh (pre-roll spans the boundary).
            self.pending_split = true;
            FrameAction::Flush
        } else {
            FrameAction::Continue
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
        let n = self.preroll_len;
        if n <= PREROLL_SAFETY_FRAMES * FRAME_16K {
            return 0;
        }
        // Onset-burst RMS: the tail of the ring (the earlier onset frames,
        // already recorded) plus the current frame that opens the utterance.
        let tail_samples = ((ONSET_SPEECH_FRAMES as usize).saturating_sub(1) * FRAME_16K).min(n);
        let mut sum_sq = self.preroll_sum_sq(n - tail_samples, tail_samples);
        for &s in current {
            let s = s as f64;
            sum_sq += s * s;
        }
        let count = tail_samples + current.len();
        if count == 0 {
            return 0;
        }
        let onset_rms = (sum_sq / count as f64).sqrt() as f32;
        let threshold = (onset_rms * PREROLL_TRIM_FACTOR).max(PREROLL_TRIM_FLOOR);
        for f in 0..(n / FRAME_16K) {
            let frame_sum_sq = self.preroll_sum_sq(f * FRAME_16K, FRAME_16K);
            if (frame_sum_sq / FRAME_16K as f64).sqrt() as f32 >= threshold {
                return (f.saturating_sub(PREROLL_SAFETY_FRAMES) * FRAME_16K).min(n);
            }
        }
        0
    }

    /// Sum of squared samples over `count` ring samples starting at the
    /// `start`-th oldest, in `f64` like `frame_rms`. Walks the ring with a
    /// wrapping index instead of a modulo per sample.
    fn preroll_sum_sq(&self, start: usize, count: usize) -> f64 {
        let mut idx = self.preroll_head + start;
        let mut sum_sq = 0.0f64;
        for _ in 0..count {
            if idx >= PREROLL_CAPACITY {
                idx -= PREROLL_CAPACITY;
            }
            let s = self.preroll[idx] as f64;
            sum_sq += s * s;
            idx += 1;
        }
        sum_sq
    }

    /// Append the ring's live samples to `buf`, dropping its leading `skip`
    /// samples (see `preroll_trim_skip`). The ring is contiguous apart from at
    /// most one wrap, so this copies at most two runs.
    fn extend_from_preroll(&mut self, skip: usize) {
        let live = self.preroll_len - skip;
        let start = (self.preroll_head + skip) % PREROLL_CAPACITY;
        let first = (PREROLL_CAPACITY - start).min(live);
        self.buf
            .extend_from_slice(&self.preroll[start..start + first]);
        if live > first {
            self.buf.extend_from_slice(&self.preroll[..live - first]);
        }
    }

    /// Record `frame` as recent audio, so it can be prepended as pre-roll if an
    /// utterance opens in the next `RING_FRAMES` frames. Once the ring is
    /// full each new sample overwrites the oldest one, so the live window
    /// always spans the last `PREROLL_CAPACITY` samples.
    fn record_preroll(&mut self, frame: &[f32]) {
        for &s in frame {
            self.preroll[(self.preroll_head + self.preroll_len) % PREROLL_CAPACITY] = s;
            if self.preroll_len == PREROLL_CAPACITY {
                self.preroll_head = (self.preroll_head + 1) % PREROLL_CAPACITY;
            } else {
                self.preroll_len += 1;
            }
        }
    }

    /// The open utterance so far, for a partial transcript, or `None` when it is
    /// not time for one.
    ///
    /// Cut like `close` cuts: up to the last speech frame plus the minimum tail,
    /// so a pause inside the utterance does not feed silence to the model, and
    /// nothing new is returned while the speaker is quiet. The first snapshot
    /// needs `STREAM_MIN_SAMPLES` of audio, each later one `STREAM_STEP_SAMPLES`
    /// more than the last. The same cheap guards as `close` keep a VAD blip
    /// from costing a decode: `MIN_SPEECH_FRAMES` of speech, at `min_rms`.
    ///
    /// Takes `&mut self` only to remember where the last one ended. It leaves
    /// the utterance itself untouched: the final transcript is still cut by
    /// `close`.
    pub(crate) fn partial_snapshot(&mut self, min_rms: f32) -> Option<Vec<f32>> {
        if self.buf.is_empty() || self.speech_frames < MIN_SPEECH_FRAMES {
            return None;
        }
        let end = (self.samples_at_last_speech + TAIL_MIN_SAMPLES).min(self.buf.len());
        let due = if self.partial_end == 0 {
            end >= STREAM_MIN_SAMPLES
        } else {
            end >= self.partial_end + STREAM_STEP_SAMPLES
        };
        if !due {
            return None;
        }
        if self.speech_sample_count > 0 {
            let rms = (self.speech_energy_sum_sq / self.speech_sample_count as f64).sqrt() as f32;
            if rms < min_rms {
                return None;
            }
        }
        self.partial_end = end;
        Some(self.buf[..end].to_vec())
    }

    /// Decide whether the accumulated utterance is real speech and, if so, trim
    /// its trailing silence and return it. Resets `self` to its (empty) default
    /// state either way, leaving the pre-roll ring intact.
    ///
    /// Three gates, all defending against the same failure mode: a VAD false
    /// positive (a click, a breath, mains hum) opening an utterance that
    /// pre-roll and silence hangover then pad out to ~1-2s of near-silence —
    /// which the model would otherwise happily hallucinate a sentence for:
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
        // hangover that doesn't need to reach the model.
        self.buf.truncate(tail_end);

        let pcm = std::mem::take(&mut self.buf);
        let (peak, clip_ratio) = mic_level(&pcm);
        log::info!(
            "utterance: {} samples ({speech_frames} speech frames, \
             density={density:.3} weighted={weighted:.3}, rms={rms:.5}, peak={peak:.3}) -> model",
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
            split_continuation: self.pending_split,
            // Stamped here, not at the push: the wait starts when the
            // buffer cuts the text (see `Utterance::queued_at`).
            queued_at: Instant::now(),
            pause_flush: false,
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
        self.pending_split = false;
        self.partial_end = 0;
    }

    /// Discard everything for a pause transition: like `reset`, but also
    /// drops `preroll` so the first utterance after resume can't prepend
    /// pre-pause audio. Rewinding the ring's length and head is enough: the
    /// samples stay in the array but are no longer reachable.
    pub(crate) fn discard(&mut self) {
        self.reset();
        self.preroll_len = 0;
        self.preroll_head = 0;
    }
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
/// silence to the model, higher density); reverberant tails keep what is
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
        DEFAULT_MIN_DENSITY, DEFAULT_MIN_SPEECH_RMS, DENSITY_LONG_EXEMPT_SAMPLES,
        EARLY_SPLIT_QUIET_FRAMES, EARLY_SPLIT_QUIET_RMS, EARLY_SPLIT_SAMPLES,
        EARLY_SPLIT_SILENCE_FRAMES, FRAME_16K, INTERIM_SPLIT_SAMPLES, INTERIM_SPLIT_SILENCE_FRAMES,
        MAX_UTTERANCE_SAMPLES, MIN_SPEECH_FRAMES, ONSET_SPEECH_FRAMES, PREROLL_SAFETY_FRAMES,
        RING_FRAMES, SILENCE_FRAMES_THRESHOLD, STRONG_RMS_FACTOR, TAIL_KEEP_SAMPLES,
        TAIL_MAX_SAMPLES, TAIL_MIN_SAMPLES,
    };

    fn gate_for(min_rms: f32, min_density: f32) -> GateConfig {
        GateConfig::new(min_rms, min_density)
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
        // an utterance that the model then hallucinated "Dziękuje za uwagę."
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
            .expect("real speech must be sent to the model");
        assert!(!utterance.pcm.is_empty());
        assert_eq!(utterance.evidence.speech_frames, 25);
        assert!(utterance.evidence.total_frames > 0);
        assert!(utterance.evidence.density() >= DEFAULT_MIN_DENSITY);
        assert!(utterance.evidence.rms > 0.0);
    }

    #[test]
    fn close_leaves_the_pause_flush_unset_and_stamps_the_queue_time() {
        // A hangover close is not a pause flush: only the pause edge marks
        // one, and the worker relies on the difference to tell the sentence
        // the speaker just finished from a backlog.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame(&silence, false);
        }
        let before = Instant::now();
        let mut utterance = buffer
            .close(&gate_for(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY))
            .expect("real speech must be sent to the model");
        assert!(!utterance.pause_flush, "a hangover close is not a flush");
        assert!(
            utterance.queued_at >= before,
            "queued_at is stamped at the close, so the wait starts there"
        );

        utterance.mark_pause_flush();
        assert!(utterance.pause_flush);
    }

    #[test]
    fn agc_gained_buffer_does_not_inflate_evidence_rms() {
        // Regression (M1): the buffered/transmitted frame can be far louder
        // than the evidence_rms passed alongside it (Agc gains the former
        // toward --target-rms before FrameProcessor calls
        // push_frame_evidenced, but evidence_rms is the pre-gain value —
        // see Agc::process_pair). Evidence.rms must reflect the quiet
        // microphone signal, not the loud buffered audio, so a genuinely
        // quiet utterance still reads as Weak acoustic evidence even after
        // AGC pushed it toward the target.
        let mut buffer = UtteranceBuffer::new();
        let quiet_evidence_rms = 0.004; // just above DEFAULT_MIN_SPEECH_RMS
        let agc_gained_frame = vec![0.04f32; FRAME_16K]; // AGC's --target-rms
        let gained_rms = frame_rms(&agc_gained_frame);
        for _ in 0..25 {
            buffer.push_frame_evidenced(&agc_gained_frame, true, quiet_evidence_rms, gained_rms);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame_evidenced(&silence, false, 0.0, 0.0);
        }
        let utterance = buffer
            .close(&gate_for(DEFAULT_MIN_SPEECH_RMS, DEFAULT_MIN_DENSITY))
            .expect("quiet-but-real speech must still clear the acoustic gates");
        assert!(
            (utterance.evidence.rms - quiet_evidence_rms).abs() < 1e-4,
            "Evidence.rms must be the pre-AGC value, got {} (buffered frame was 0.04)",
            utterance.evidence.rms
        );
        // Evidence RMS stays at the quiet mic level even though the
        // buffered audio was gained up.
        assert!(
            utterance.evidence.rms < 0.01,
            "evidence must stay quiet, got {}",
            utterance.evidence.rms
        );
    }

    #[test]
    fn scattered_flicker_fails_density_gate() {
        // 25 speech frames — past MIN_SPEECH_FRAMES — but scattered across a
        // long silence span (VAD flicker on noise, never 81 silent frames in
        // a row so the utterance stays open). Post-trim the transmitted audio
        // is ~200 frames with only 25 of them speech: mostly silence that
        // the model would hallucinate boilerplate for.
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
            .expect("dense speech must be sent to the model");
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
    fn near_cap_vad_latched_quiet_dip_still_splits_early() {
        // Regression: the speech arm used to reset `quiet_run` to 0 on every
        // speech frame *before* incrementing it, so a run of quiet-but-still-
        // VAD-flagged frames (earshot's latch riding through a word-boundary
        // dip) could never reach EARLY_SPLIT_QUIET_FRAMES and this path was
        // dead. Past 28 s, VAD still says speech but energy drops below
        // EARLY_SPLIT_QUIET_RMS for 5 consecutive frames: must still flush,
        // exactly like the VAD-false case in
        // `near_cap_pause_splits_early_in_silence`.
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_SAMPLES / FRAME_16K + 5) {
            buffer.push_frame(&speech, true);
        }
        assert!(buffer.buf.len() >= EARLY_SPLIT_SAMPLES);
        let quiet_but_speech = vec![EARLY_SPLIT_QUIET_RMS / 2.0; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_QUIET_FRAMES - 1) {
            assert_eq!(
                buffer.push_frame(&quiet_but_speech, true),
                FrameAction::Continue
            );
        }
        assert_eq!(
            buffer.push_frame(&quiet_but_speech, true),
            FrameAction::Flush,
            "50 ms VAD-latched quiet dip past 28 s must close early"
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
        assert!(
            utterance.split_continuation,
            "interim split prefix must mark the continuation"
        );
    }

    #[test]
    fn hangover_close_is_not_a_continuation() {
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..SILENCE_FRAMES_THRESHOLD {
            buffer.push_frame(&silence, false);
        }
        assert_eq!(buffer.push_frame(&silence, false), FrameAction::Flush);
        let utterance = buffer.close(&gate_for(0.0, 0.0)).unwrap();
        assert!(!utterance.split_continuation);
    }

    #[test]
    fn early_split_marks_continuation() {
        let mut buffer = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_SAMPLES / FRAME_16K + 5) {
            buffer.push_frame(&speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(EARLY_SPLIT_QUIET_FRAMES - 1) {
            assert_eq!(buffer.push_frame(&silence, false), FrameAction::Continue);
        }
        assert_eq!(buffer.push_frame(&silence, false), FrameAction::Flush);
        let utterance = buffer.close(&gate_for(0.0, 0.0)).unwrap();
        assert!(utterance.split_continuation);
    }

    /// Feed `n` speech frames of 0.3 and return how many snapshots came back.
    fn snapshots_over(buffer: &mut UtteranceBuffer, n: usize, min_rms: f32) -> Vec<usize> {
        let speech = vec![0.3f32; FRAME_16K];
        let mut lens = Vec::new();
        for _ in 0..n {
            buffer.push_frame(&speech, true);
            if let Some(pcm) = buffer.partial_snapshot(min_rms) {
                lens.push(pcm.len());
            }
        }
        lens
    }

    #[test]
    fn partial_snapshot_waits_for_the_minimum_then_steps() {
        let mut buffer = UtteranceBuffer::new();
        // 350 frames = 3.5 s of speech. Onset opens at frame 3 (no pre-roll
        // here), so audio runs one frame ahead of the count; the first
        // snapshot is the first moment 1 s is buffered, then one per second.
        let lens = snapshots_over(&mut buffer, 350, DEFAULT_MIN_SPEECH_RMS);
        assert_eq!(lens.len(), 3, "snapshots at ~1 s, ~2 s, ~3 s: {lens:?}");
        assert!(lens[0] >= STREAM_MIN_SAMPLES);
        for pair in lens.windows(2) {
            assert!(
                pair[1] >= pair[0] + STREAM_STEP_SAMPLES,
                "each snapshot adds at least one step: {lens:?}"
            );
        }
    }

    #[test]
    fn partial_snapshot_is_cut_at_the_last_speech_plus_the_minimum_tail() {
        let mut buffer = UtteranceBuffer::new();
        snapshots_over(&mut buffer, 150, DEFAULT_MIN_SPEECH_RMS);
        let at_speech = buffer.samples_at_last_speech;
        // A pause: the buffer grows by silence, the snapshot must not.
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..50 {
            buffer.push_frame(&silence, false);
        }
        assert!(buffer.buf.len() > at_speech + TAIL_MIN_SAMPLES);
        // More speech so a snapshot is due; it must not contain the pause tail
        // beyond what `close` would keep.
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..120 {
            buffer.push_frame(&speech, true);
        }
        let end = buffer.samples_at_last_speech;
        let pcm = buffer
            .partial_snapshot(DEFAULT_MIN_SPEECH_RMS)
            .expect("due");
        assert_eq!(pcm.len(), (end + TAIL_MIN_SAMPLES).min(buffer.buf.len()));
        // And quiet again: nothing new is due while the speaker is silent.
        for _ in 0..100 {
            buffer.push_frame(&silence, false);
            assert!(buffer.partial_snapshot(DEFAULT_MIN_SPEECH_RMS).is_none());
        }
    }

    #[test]
    fn partial_snapshot_ignores_a_vad_blip_and_quiet_noise() {
        // Too little speech: a blip of 10 frames never gets a decode.
        let mut blip = UtteranceBuffer::new();
        assert!(snapshots_over(&mut blip, 10, DEFAULT_MIN_SPEECH_RMS).is_empty());
        // Enough speech but quieter than the rms gate: same guard `close` has.
        let mut quiet = UtteranceBuffer::new();
        assert!(snapshots_over(&mut quiet, 300, 0.5).is_empty());
        // Nothing open, nothing to snapshot.
        assert!(UtteranceBuffer::new()
            .partial_snapshot(DEFAULT_MIN_SPEECH_RMS)
            .is_none());
    }

    #[test]
    fn partial_snapshot_does_not_disturb_the_final_utterance() {
        let mut with = UtteranceBuffer::new();
        let mut without = UtteranceBuffer::new();
        let speech = vec![0.3f32; FRAME_16K];
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..250 {
            with.push_frame(&speech, true);
            with.partial_snapshot(DEFAULT_MIN_SPEECH_RMS);
            without.push_frame(&speech, true);
        }
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            with.push_frame(&silence, false);
            without.push_frame(&silence, false);
        }
        let a = with.close(&gate_for(0.0, 0.0)).unwrap();
        let b = without.close(&gate_for(0.0, 0.0)).unwrap();
        assert_eq!(a.pcm, b.pcm);
        assert_eq!(a.evidence.speech_frames, b.evidence.speech_frames);
    }

    #[test]
    fn partial_steps_restart_with_the_next_utterance_and_after_discard() {
        let mut buffer = UtteranceBuffer::new();
        let first = snapshots_over(&mut buffer, 250, DEFAULT_MIN_SPEECH_RMS);
        assert!(!first.is_empty());
        // Close it (a long hangover), then the next utterance starts over: its
        // first snapshot is the same size as the first utterance's first.
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer.push_frame(&silence, false);
        }
        buffer.close(&gate_for(0.0, 0.0)).unwrap();
        let second = snapshots_over(&mut buffer, 250, DEFAULT_MIN_SPEECH_RMS);
        let second_first = *second.first().expect("the next utterance snapshots again");
        assert!(
            (STREAM_MIN_SAMPLES..STREAM_MIN_SAMPLES + STREAM_STEP_SAMPLES).contains(&second_first),
            "a new utterance starts at the minimum, not at the old step: {second_first}"
        );
        // A pause discards the state the same way.
        buffer.discard();
        assert_eq!(buffer.partial_end, 0);
        assert!(buffer.partial_snapshot(DEFAULT_MIN_SPEECH_RMS).is_none());
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
