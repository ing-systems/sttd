use std::collections::VecDeque;

use crossbeam_channel::Sender;

use crate::config::{
    MAX_UTTERANCE_SAMPLES, MIN_SPEECH_FRAMES, ONSET_SPEECH_FRAMES, SILENCE_FRAMES_THRESHOLD,
    STRONG_RMS_FACTOR, STRONG_SPEECH_FRAMES, TAIL_KEEP_SAMPLES,
};

/// Per-utterance buffering state, reset every time an utterance is flushed
/// (whether accepted or dropped). Kept as one struct rather than loose
/// `&mut` parameters so `flush_utterance` can reset everything atomically
/// and so new gates (see `flush_utterance`) can add fields without touching
/// every call site.
#[derive(Default)]
pub(crate) struct Utterance {
    /// Whisper-branch audio (16 kHz), from the moment the utterance opens.
    pub(crate) buf: Vec<f32>,
    /// Consecutive non-speech frames since the last speech frame, once the
    /// utterance is open. Drives the close decision (`SILENCE_FRAMES_THRESHOLD`).
    pub(crate) silence_frames: u32,
    /// Consecutive speech frames seen before the utterance has opened.
    /// Drives the open decision (`ONSET_SPEECH_FRAMES`); reset to 0 on any
    /// non-speech frame.
    pub(crate) onset_run: u32,
    /// Total VAD-flagged speech frames over the utterance's whole life
    /// (including pre-open onset frames). Drives `MIN_SPEECH_FRAMES`.
    pub(crate) speech_frames: u32,
    /// `buf.len()` as of the last speech frame appended. Used to trim
    /// trailing silence down to `TAIL_KEEP_SAMPLES` at flush time.
    pub(crate) samples_at_last_speech: usize,
    /// Sum of squared samples from VAD-flagged speech frames only (silence
    /// padding excluded), paired with `speech_sample_count` so
    /// `flush_utterance` can compute an RMS over just the speech content.
    pub(crate) speech_energy_sum_sq: f64,
    pub(crate) speech_sample_count: usize,
}

/// One utterance plus the acoustic evidence `flush_utterance` gathered for
/// it while buffering. The whisper worker needs this to decide how much
/// trust the acoustics already earned, before deciding how much trust to
/// demand from the model's own confidence (see `confidence_threshold`) —
/// evidence that used to be computed and then thrown away at the
/// `Sender<Vec<f32>>` boundary.
pub(crate) struct Clip {
    pub(crate) pcm: Vec<f32>,
    /// Total VAD-flagged speech frames over the utterance's life (see
    /// `Utterance::speech_frames`).
    pub(crate) speech_frames: u32,
    /// RMS over VAD-flagged speech frames only (see `Utterance::speech_energy_sum_sq`).
    pub(crate) rms: f32,
}

/// Confidence floor for this utterance's transcript. The confidence gate
/// exists to catch hallucinations on silence/noise; where the acoustics
/// already testify to real speech, it has nothing to add, and applying the
/// strict floor anyway discards the only transcript we have. It doesn't
/// disappear entirely: the relaxed floor still catches a repetition-loop
/// decode (avg_p ~0.1), which isn't a transcript of anything.
pub(crate) fn confidence_threshold(
    clip: &Clip,
    min_rms: f32,
    weak: f32,
    strong: f32,
) -> (f32, &'static str) {
    if clip.speech_frames >= STRONG_SPEECH_FRAMES && clip.rms >= min_rms * STRONG_RMS_FACTOR {
        (strong, "strong")
    } else {
        (weak, "weak")
    }
}

/// Pure utterance-buffering decision: given the current VAD verdict, decide
/// whether to append `frame` and whether the utterance should be flushed now.
/// Extracted out of `FrameProcessor::process_frame` so it's unit-testable
/// without a live VAD signal.
///
/// Two behaviors, both load-bearing for hallucination avoidance:
/// - An utterance only *opens* after `ONSET_SPEECH_FRAMES` consecutive
///   speech frames, not the first one, so a single VAD false positive no
///   longer opens (and thus sends) a full utterance. The onset frames
///   themselves aren't lost: `preroll` already contains them by the time
///   the utterance opens, since the caller updates it every frame
///   regardless of `is_speech`.
/// - Once open, every frame is appended regardless of `is_speech` (non-
///   speech frames become natural in-utterance padding instead of being
///   spliced out) — but `speech_frames`/`samples_at_last_speech`/energy are
///   tracked only from VAD-flagged frames, so `flush_utterance` can later
///   gate and trim on actual speech content instead of total buffer length.
pub(crate) fn buffer_frame(
    utt: &mut Utterance,
    preroll: &VecDeque<f32>,
    frame: &[f32],
    is_speech: bool,
) -> bool {
    if is_speech {
        utt.onset_run += 1;
        utt.speech_frames += 1;
        for &s in frame {
            utt.speech_energy_sum_sq += (s as f64) * (s as f64);
        }
        utt.speech_sample_count += frame.len();

        if utt.buf.is_empty() {
            if utt.onset_run < ONSET_SPEECH_FRAMES {
                return false;
            }
            utt.buf.extend(preroll.iter().copied());
        }
        utt.buf.extend_from_slice(frame);
        utt.samples_at_last_speech = utt.buf.len();
        utt.silence_frames = 0;
        utt.buf.len() >= MAX_UTTERANCE_SAMPLES
    } else {
        utt.onset_run = 0;
        if !utt.buf.is_empty() {
            utt.buf.extend_from_slice(frame);
            utt.silence_frames += 1;
            utt.silence_frames > SILENCE_FRAMES_THRESHOLD
        } else {
            false
        }
    }
}

/// Decide whether the accumulated utterance is real speech and, if so, trim
/// its trailing silence and move it into the whisper queue. Resets `utt` to
/// its default state either way; `send` failure (worker gone) is logged,
/// never panics, so the audio loop survives.
///
/// Two gates, both defending against the same failure mode: a VAD false
/// positive (a click, a breath, mains hum) opening an utterance that
/// pre-roll and silence hangover then pad out to ~1-2s of near-silence —
/// which whisper, lacking any working confidence gate of its own (see
/// `DEFAULT_MIN_CONFIDENCE`'s doc comment), would otherwise happily
/// hallucinate a sentence for:
/// - `MIN_SPEECH_FRAMES`: not enough VAD-flagged speech overall.
/// - `min_rms`: VAD-flagged frames present, but too quiet to be real speech.
pub(crate) fn flush_utterance(utt: &mut Utterance, whisper_tx: &Sender<Clip>, min_rms: f32) {
    let speech_frames = utt.speech_frames;
    if speech_frames < MIN_SPEECH_FRAMES {
        eprintln!(
            "[sttd] dropping utterance ({speech_frames} speech frames, {} samples)",
            utt.buf.len()
        );
        *utt = Utterance::default();
        return;
    }

    let rms = if utt.speech_sample_count > 0 {
        (utt.speech_energy_sum_sq / utt.speech_sample_count as f64).sqrt() as f32
    } else {
        0.0
    };
    if rms < min_rms {
        eprintln!(
            "[sttd] dropping low-energy utterance ({speech_frames} speech frames, \
             rms={rms:.5} < {min_rms:.5})"
        );
        *utt = Utterance::default();
        return;
    }

    // Trim the tail: `SILENCE_FRAMES_THRESHOLD` decided *when* to close this
    // utterance, but everything past `samples_at_last_speech +
    // TAIL_KEEP_SAMPLES` is silence hangover that doesn't need to reach
    // whisper.
    let tail_end = (utt.samples_at_last_speech + TAIL_KEEP_SAMPLES).min(utt.buf.len());
    utt.buf.truncate(tail_end);

    let pcm = std::mem::take(&mut utt.buf);
    eprintln!(
        "[sttd] utterance: {} samples ({speech_frames} speech frames, rms={rms:.5}) -> whisper",
        pcm.len()
    );
    *utt = Utterance::default();
    if let Err(e) = whisper_tx.send(Clip {
        pcm,
        speech_frames,
        rms,
    }) {
        eprintln!("[sttd] whisper queue closed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_SPEECH_RMS,
        FRAME_16K, MIN_SPEECH_FRAMES, ONSET_SPEECH_FRAMES, RING_FRAMES,
        SILENCE_FRAMES_THRESHOLD, TAIL_KEEP_SAMPLES,
    };

    #[test]
    fn buffer_frame_keeps_non_speech_inside_open_utterance() {
        // Regression for the original bug: non-speech frames in the middle
        // of an utterance used to be silently dropped instead of appended.
        let mut utt = Utterance {
            buf: vec![0.5f32; 10],
            ..Default::default()
        };
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(!flush);
        assert_eq!(utt.buf.len(), 10 + FRAME_16K);
        assert_eq!(utt.silence_frames, 1);
    }

    #[test]
    fn buffer_frame_ignores_non_speech_before_utterance_starts() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(!flush);
        assert!(utt.buf.is_empty());
    }

    #[test]
    fn single_vad_hit_does_not_open_utterance() {
        // Regression for the production bug: one VAD false positive (a
        // click, a breath) used to open a full utterance immediately.
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let frame = vec![0.5f32; FRAME_16K];

        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            let flush = buffer_frame(&mut utt, &preroll, &frame, true);
            assert!(!flush);
            assert!(utt.buf.is_empty(), "must not open before ONSET_SPEECH_FRAMES");
        }
        // The frame that reaches ONSET_SPEECH_FRAMES opens it.
        buffer_frame(&mut utt, &preroll, &frame, true);
        assert!(!utt.buf.is_empty(), "must open at ONSET_SPEECH_FRAMES");
    }

    #[test]
    fn buffer_frame_prepends_preroll_on_utterance_start() {
        let mut utt = Utterance::default();
        let mut preroll = VecDeque::new();
        preroll.extend(vec![0.25f32; FRAME_16K * 2]);
        let frame = vec![0.5f32; FRAME_16K];

        for _ in 0..(ONSET_SPEECH_FRAMES - 1) {
            buffer_frame(&mut utt, &preroll, &frame, true);
        }
        assert!(utt.buf.is_empty());
        buffer_frame(&mut utt, &preroll, &frame, true);
        assert_eq!(utt.buf.len(), FRAME_16K * 3);
        assert!(utt.buf[..FRAME_16K * 2].iter().all(|&s| s == 0.25));
        assert!(utt.buf[FRAME_16K * 2..].iter().all(|&s| s == 0.5));
    }

    #[test]
    fn buffer_frame_flushes_after_silence_threshold() {
        let mut utt = Utterance {
            buf: vec![0.5f32; 10],
            silence_frames: SILENCE_FRAMES_THRESHOLD,
            ..Default::default()
        };
        let preroll = VecDeque::new();
        let frame = vec![0.0f32; FRAME_16K];
        let flush = buffer_frame(&mut utt, &preroll, &frame, false);
        assert!(flush);
    }

    #[test]
    fn utterance_from_log_is_rejected() {
        // Regression for the production log: a 7-speech-frame VAD blip,
        // padded by pre-roll and a full silence hangover, used to produce
        // an 18_880-sample utterance that whisper then hallucinated
        // "Dziękuje za uwagę." for.
        let mut utt = Utterance::default();
        let mut preroll = VecDeque::new();
        preroll.extend(vec![0.0f32; RING_FRAMES * FRAME_16K]);
        let speech = vec![0.3f32; FRAME_16K];
        let silence = vec![0.0f32; FRAME_16K];

        for _ in 0..7 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let mut flush = false;
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            flush = buffer_frame(&mut utt, &preroll, &silence, false);
            if flush {
                break;
            }
        }
        assert!(flush, "silence hangover should eventually close the utterance");

        let (tx, rx) = crossbeam_channel::unbounded();
        flush_utterance(&mut utt, &tx, 0.0); // rms gate disabled: isolate MIN_SPEECH_FRAMES
        assert!(
            rx.try_recv().is_err(),
            "an utterance with too little VAD-flagged speech must be dropped"
        );
    }

    #[test]
    fn real_utterance_survives_gates() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer_frame(&mut utt, &preroll, &silence, false);
        }

        let (tx, rx) = crossbeam_channel::unbounded();
        flush_utterance(&mut utt, &tx, DEFAULT_MIN_SPEECH_RMS);
        let got = rx.try_recv().expect("real speech must be sent to whisper");
        assert!(!got.pcm.is_empty());
        assert_eq!(got.speech_frames, 25);
        assert!(got.rms > 0.0);
    }

    #[test]
    fn tail_is_trimmed_to_200ms() {
        let mut utt = Utterance::default();
        let preroll = VecDeque::new();
        let speech = vec![0.3f32; FRAME_16K];
        for _ in 0..25 {
            buffer_frame(&mut utt, &preroll, &speech, true);
        }
        let samples_at_last_speech = utt.samples_at_last_speech;
        let silence = vec![0.0f32; FRAME_16K];
        for _ in 0..(SILENCE_FRAMES_THRESHOLD + 1) {
            buffer_frame(&mut utt, &preroll, &silence, false);
        }

        let (tx, rx) = crossbeam_channel::unbounded();
        flush_utterance(&mut utt, &tx, 0.0);
        let got = rx.try_recv().unwrap();
        assert_eq!(got.pcm.len(), samples_at_last_speech + TAIL_KEEP_SAMPLES);
    }

    #[test]
    fn flush_drops_short_and_sends_long() {
        let (tx, rx) = crossbeam_channel::unbounded();

        let mut short = Utterance {
            buf: vec![0.0f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES - 1,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0, // high RMS: isolate the frame-count gate
            ..Default::default()
        };
        flush_utterance(&mut short, &tx, 0.0);
        assert!(rx.try_recv().is_err());

        let mut long = Utterance {
            buf: vec![0.5f32; 5_000],
            speech_frames: MIN_SPEECH_FRAMES,
            samples_at_last_speech: 5_000,
            speech_sample_count: 1,
            speech_energy_sum_sq: 1.0,
            ..Default::default()
        };
        flush_utterance(&mut long, &tx, 0.0);
        let got = rx.try_recv().unwrap();
        assert_eq!(got.pcm.len(), 5_000);
    }

    #[test]
    fn strong_acoustic_evidence_relaxes_confidence_gate() {
        // Exact numbers from the production log this fix addresses:
        // "[sttd] utterance: 34720 samples (137 speech frames, rms=0.01178)
        // -> whisper" followed by "dropped low-confidence transcript
        // (avg_p=0.478 < 0.600)" for a transcript the user confirmed was
        // close to correct (one misheard word), not a hallucination.
        let clip = Clip {
            pcm: Vec::new(),
            speech_frames: 137,
            rms: 0.01178,
        };
        let (threshold, gate) = confidence_threshold(
            &clip,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        assert_eq!(gate, "strong");
        assert_eq!(threshold, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert!(
            0.478 >= threshold,
            "avg_p=0.478 must clear the relaxed threshold {threshold}"
        );
    }

    #[test]
    fn weak_evidence_keeps_strict_gate() {
        // Regression for the hole 71c05b0 closed: an utterance with barely
        // enough VAD-flagged speech to survive `flush_utterance`'s own
        // gates, and RMS just above the noise floor, must still face the
        // strict confidence floor — even at the same avg_p (0.478) that the
        // strong-evidence case above lets through.
        let clip = Clip {
            pcm: Vec::new(),
            speech_frames: MIN_SPEECH_FRAMES,
            rms: DEFAULT_MIN_SPEECH_RMS + 0.0005,
        };
        let (threshold, gate) = confidence_threshold(
            &clip,
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        );
        assert_eq!(gate, "weak");
        assert_eq!(threshold, DEFAULT_MIN_CONFIDENCE);
        assert!(
            0.478 < threshold,
            "avg_p=0.478 must NOT clear the strict threshold {threshold}"
        );
    }
}
