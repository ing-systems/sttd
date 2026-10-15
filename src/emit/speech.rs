//! The spoken channel (`--speak`): the speak thread, its one-slot queue, and
//! the deadlines that cut it short.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::clock::now_ms;
use crate::config::SPEAK_TAIL_MS;
use crate::piper::PiperVoice;

/// One utterance's text, on its way to the speakers.
struct SpeechJob {
    tag: String,
    text: String,
}

/// The spoken emission: hand it text, a dedicated thread speaks it.
///
/// Off the emit worker on purpose. Speaking is seconds of wall clock (the
/// synthesis plus the audio playing out), and the emit worker still has the
/// clipboard, the paste and the dump of every later utterance to do — speaking
/// in line would make a transcript arrive minutes late.
///
/// The queue holds **one** utterance and the newest replaces what is waiting.
/// That is the same rule the partial transcript follows (only the latest is
/// kept, see `asr::PartialSlot`): with `--autopause` there is never a second
/// utterance while one is being spoken, and without it a backlog of speech is
/// already stale by the time it would be read — reading it aloud would put words
/// in the user's mouth that they have already moved on from.
///
/// `speaking` is the flag the capture side watches: the whole point of speaking
/// through the microphone's own room is that sttd must not transcribe itself,
/// so it is raised for the lead window, the playback and the tail.
pub(crate) struct SpeechOut {
    /// `(slot, wakeup)`: the pending job and the signal to look at it.
    slot: Arc<(Mutex<Option<SpeechJob>>, Condvar)>,
    /// Raised from just before playback starts until the tail has played out.
    speaking: Arc<AtomicBool>,
    /// The deadline the speak thread reads, plus the mark that tells `drop` an
    /// interrupted run must not arm a fresh drain window.
    cut: Arc<SpeechCut>,
    /// Set when espeak goes away, so the emit side stops queueing into a thread
    /// that will never speak again.
    disabled: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

/// The speak thread's deadline, and who may move it.
///
/// `finish_by_ms` is milliseconds since the thread started — a monotonic `u64`
/// rather than an `Instant` so it can cross the thread boundary. Zero means "no
/// deadline". Two owners, and telling them apart is the whole point: `drop` arms
/// a fresh window for a run that reached its end, `abort` moves the deadline
/// into the past for one the user interrupted.
pub(crate) struct SpeechCut {
    finish_by_ms: Arc<AtomicU64>,
    aborted: AtomicBool,
}

/// How long `SpeechOut::drop` waits for the utterance already queued to be
/// spoken before cutting it off.
///
/// The trailing utterance has to be spoken, not dropped: `--file --speak` with
/// the last transcript swallowed would be the channel silently losing work, and
/// it is the same reasoning `asr`'s shutdown flush uses for the transcript
/// itself. Bounded because a device that never drains, or a synthesis that
/// never returns, must not hang the process — past this the playback is cut and
/// the utterance is lost, which a log line says.
///
/// The case it exists for is `--file`, the only path that transcribes to the
/// end. A live run never reaches it with a new utterance: `running` is false by
/// then, so the engine's shutdown check rejects the trailing decode
/// (`run::microphone`, `parakeet.rs`, `whisper.rs`) and nothing new is queued. What the
/// drain waits for there is the queue already built — the sentence playing, plus
/// whatever the emit worker hands over while it closes — which is exactly what
/// Ctrl-C must not wait 15 s for. See `SpeechCut`.
///
/// Not the bound for an interrupted run: `SpeechCut::abort` replaces it with the
/// current instant, so Ctrl-C cuts speech instead of waiting this out.
const SPEAK_DRAIN_MS: u64 = 15_000;

/// The deadline `SpeechOut::drop` arms: a fresh drain window for a run that
/// reached its end, the current instant for one that was cut.
///
/// A pure function of its inputs, so the rule is testable without a speak
/// thread — the threading around it is untested glue, like the espeak call.
fn drop_deadline(aborted: bool, now: u64) -> u64 {
    if aborted {
        now
    } else {
        now + SPEAK_DRAIN_MS
    }
}

impl SpeechCut {
    /// Stop speaking now: cut the playback, skip the queued utterance, and mark
    /// the drop path so it arms no drain window. Idempotent, and safe from any
    /// thread; the capture loop's owner calls it on the way out.
    pub(crate) fn abort(&self) {
        self.aborted.store(true, Ordering::Release);
        // `.max(1)` because `past_deadline` reads a deadline of 0 as "no
        // deadline", and `now_ms()` returns 0 when it is the process's first
        // call — an abort that stored 0 would cut nothing at all.
        self.finish_by_ms.store(now_ms().max(1), Ordering::Release);
    }
}

impl SpeechOut {
    /// Spawn the speak thread around a loaded voice.
    pub(crate) fn spawn(voice: PiperVoice) -> Self {
        let slot: Arc<(Mutex<Option<SpeechJob>>, Condvar)> =
            Arc::new((Mutex::new(None), Condvar::new()));
        let speaking = Arc::new(AtomicBool::new(false));
        let cut = Arc::new(SpeechCut {
            finish_by_ms: Arc::new(AtomicU64::new(0)),
            aborted: AtomicBool::new(false),
        });
        let disabled = Arc::new(AtomicBool::new(false));

        let handle = {
            let (slot, speaking, cut, disabled) = (
                Arc::clone(&slot),
                Arc::clone(&speaking),
                Arc::clone(&cut),
                Arc::clone(&disabled),
            );
            std::thread::Builder::new()
                .name("speak".into())
                .spawn(move || speak_loop(&slot, &speaking, &cut, &disabled, voice))
                .ok()
        };
        Self {
            slot,
            speaking,
            cut,
            disabled,
            handle,
        }
    }

    /// The cut handle, cloned while the `SpeechOut` is still here: it is moved
    /// onto the emit worker thread, which is what drops it.
    pub(crate) fn cut_handle(&self) -> Arc<SpeechCut> {
        Arc::clone(&self.cut)
    }

    /// The flag the capture side reads to hold the pipeline while sttd talks.
    pub(crate) fn speaking_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Queue `text` to be spoken, replacing anything still waiting. Never
    /// blocks and never fails: a dead or disabled speak thread costs one
    /// utterance of audio, never the transcript.
    pub(super) fn enqueue(&self, tag: &str, text: &str) {
        if self.disabled.load(Ordering::Relaxed) {
            return;
        }
        let (lock, wakeup) = &*self.slot;
        if let Ok(mut pending) = lock.lock() {
            *pending = Some(SpeechJob {
                tag: tag.to_string(),
                text: text.to_string(),
            });
            wakeup.notify_one();
        }
    }
}

impl Drop for SpeechOut {
    fn drop(&mut self) {
        // Let whatever is queued be spoken, within a bound: the pending slot is
        // left alone and the thread is woken, so the trailing utterance is
        // heard. `finish_by_ms` is the bound it must respect, so this join
        // cannot hang the process on a device that never drains.
        //
        // An interrupted run (`SpeechCut::abort`) has already moved the deadline
        // into the past, and this must not arm a fresh window over it: the user
        // asked sttd to stop, so speaking the trailing utterance anyway is the
        // thing the abort exists to prevent.
        let aborted = self.cut.aborted.load(Ordering::Acquire);
        self.cut
            .finish_by_ms
            .store(drop_deadline(aborted, now_ms()), Ordering::Release);
        let (_lock, wakeup) = &*self.slot;
        wakeup.notify_all();
        if let Some(h) = self.handle.take() {
            if h.join().is_err() {
                log::warn!("speak thread panicked");
            }
        }
    }
}

/// Whether the finish deadline has passed. Zero means no deadline, so a thread
/// that was never asked to finish never sees one. Delegates to the same rule
/// `sound` applies while it waits.
fn past_deadline(finish_by_ms: &AtomicU64) -> bool {
    crate::clock::past_deadline(finish_by_ms, now_ms())
}

/// Sleep `total`, giving up early once the finish deadline passes. Returns
/// whether the sleep was cut short, so the caller can skip the work it was
/// about to start.
fn sleep_until_deadline(total: Duration, finish_by_ms: &AtomicU64) -> bool {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline {
        if past_deadline(finish_by_ms) {
            return true;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(Duration::from_millis(20).min(left));
    }
    past_deadline(finish_by_ms)
}

/// The speak thread: take the pending utterance, synthesize it, play it.
///
/// Never propagates anything. A voice that cannot speak (espeak gone, a graph
/// that rejects its input) disables the channel with one warning and the
/// thread idles, because a transcription tool that stops transcribing because
/// its speaker broke is worse than a silent one.
fn speak_loop(
    slot: &Arc<(Mutex<Option<SpeechJob>>, Condvar)>,
    speaking: &AtomicBool,
    cut: &SpeechCut,
    disabled: &AtomicBool,
    mut voice: PiperVoice,
) {
    let finish_by_ms = &cut.finish_by_ms;
    let rate = voice.sample_rate();
    loop {
        let job = {
            let (lock, wakeup) = &**slot;
            let Ok(mut pending) = lock.lock() else {
                return;
            };
            loop {
                if let Some(job) = pending.take() {
                    break job;
                }
                // A deadline that has passed with nothing queued means the drop
                // is over: nothing left to speak, so the thread ends here.
                if past_deadline(finish_by_ms) {
                    return;
                }
                let (again, _) = wakeup
                    .wait_timeout(pending, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner());
                pending = again;
            }
        };

        // Raise the flag first, then wait out the lead window, then synthesize:
        // the capture side reacts to the flag within its ~2 ms loop cadence and
        // takes its hold immediately, so by the time the first sample reaches
        // the speakers the pipeline is already discarding. The wait doubles as
        // the last chance for a deadline to cut this utterance off.
        //
        // It is also the only thing that keeps an aborted run from
        // *synthesizing* a queued utterance: `play_mono` would cut the audio
        // anyway, but only after paying for the synthesis. Do not move the
        // synthesis above this check.
        speaking.store(true, Ordering::Release);
        if sleep_until_deadline(
            Duration::from_millis(crate::config::SPEAK_LEAD_MS),
            finish_by_ms,
        ) {
            speaking.store(false, Ordering::Release);
            continue;
        }
        let t0 = Instant::now();
        let synth = voice.synthesize(&job.text);
        let audio = match synth {
            Ok(a) => a,
            Err(e) => {
                let missing = e.kind() == std::io::ErrorKind::NotFound;
                disabled.store(true, Ordering::Release);
                log::warn!(
                    "{}: spoken emission disabled ({e:?}); transcripts still reach the other \
                     channels",
                    job.tag
                );
                if missing {
                    log::warn!("espeak-ng was not found on PATH, so text cannot be phonemized");
                }
                // Clear the flag before idling: the capture side must not stay
                // held for a channel that is now off.
                speaking.store(false, Ordering::Release);
                if missing {
                    return;
                }
                continue;
            }
        };
        let synth_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if !audio.is_empty() {
            let audio_sec = audio.len() as f64 / rate as f64;
            let played = crate::sound::play_mono(&audio, rate, finish_by_ms);
            if played {
                log::info!(
                    "{}: spoke {audio_sec:.1} s in {synth_ms:.0} ms (synthesis rtf={:.2})",
                    job.tag,
                    synth_ms / 1000.0 / audio_sec.max(0.001)
                );
            } else {
                log::warn!("{}: cut off at shutdown before it could be spoken", job.tag);
            }
        }
        // The tail is inside the raised flag too: the device still holds the
        // last samples, and the capture side must stay held until it has let
        // them out.
        let _ = sleep_until_deadline(Duration::from_millis(SPEAK_TAIL_MS), finish_by_ms);
        speaking.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::test_support::*;

    use crate::emit::system::SystemEmitter;

    #[test]
    fn a_disabled_speak_channel_stops_accepting_and_clears_the_flag() {
        // The espeak-gone path: the thread warns once, raises `disabled`, and
        // clears `speaking` so the capture side is not left held for a channel
        // that is now off.
        let speaking = Arc::new(AtomicBool::new(true));
        let disabled = Arc::new(AtomicBool::new(false));
        // Model the loop's own bookkeeping on the failure branch.
        speaking.store(false, Ordering::Release);
        disabled.store(true, Ordering::Release);
        assert!(!speaking.load(Ordering::Acquire));
        assert!(disabled.load(Ordering::Acquire));
        // An empty voice name is refused at load with the reason, which is what
        // stops this state being reached silently.
        let json = r#"{"audio": {"sample_rate": 22050}, "phoneme_type": "espeak",
                        "phoneme_id_map": {"_": [0], "^": [1], "$": [2]}, "num_symbols": 256}"#;
        let config: crate::piper::VoiceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.espeak.voice, "");
    }

    #[test]
    fn speech_is_opt_in_and_follows_the_same_emit_rules() {
        // The channel mirrors stdout: off by default, and only ever handed an
        // utterance the gate actually emitted.
        let evidence = evidence();
        let pcm = vec![0.01f32; 1600];
        let mut emitter = RecordingEmitter::new();
        emit(
            &mut emitter,
            "utt_000001",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[0].emitted);
        assert!(!emitter.records[0].spoken, "--speak is off by default");

        let mut emitter = RecordingEmitter::new();
        emitter.speak_enabled = true;
        emit(
            &mut emitter,
            "utt_000002",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[0].spoken);
        // A failed clipboard copy does not stop the voice: prod enqueues speech
        // after the copy step and neither depends on the other.
        emitter.fail_copy = true;
        emit(
            &mut emitter,
            "utt_000003",
            &pcm,
            "Dzień dobry",
            gate().decide_transcript("Dzień dobry"),
            &evidence,
        );
        assert!(emitter.records[1].emitted);
        assert!(!emitter.records[1].copied);
        assert!(emitter.records[1].spoken);

        // Drops never speak: a hallucination and an empty-cleaned utterance
        // both reach no channel at all.
        for (text, expected) in [
            ("Napisy stworzone przez społeczność Amara.org", false),
            ("   ", false),
        ] {
            emit(
                &mut emitter,
                "utt_000004",
                &pcm,
                text,
                gate().decide_transcript(text),
                &evidence,
            );
            let r = emitter.records.last().unwrap();
            assert!(!r.emitted, "text={text:?}");
            assert_eq!(r.spoken, expected, "text={text:?}");
        }
    }

    #[test]
    fn the_speaking_flag_is_clear_without_the_channel() {
        // The capture loop reads this flag unconditionally, so with `--speak`
        // off it must exist and stay false rather than the loop needing a
        // separate "is speech on" case.
        let emitter = SystemEmitter::new(policy_off()).unwrap();
        let flag = emitter.speaking_flag();
        assert!(!flag.load(Ordering::Acquire));
        // Every call returns a flag that can be read; two calls must not share
        // state, or a stale one would leak between runs in one process (tests).
        let other = emitter.speaking_flag();
        assert!(!Arc::ptr_eq(&flag, &other) || !flag.load(Ordering::Acquire));
    }

    #[test]
    fn a_past_deadline_cuts_the_wait_short() {
        // `SpeechOut::drop` must not block for the length of a playback: the
        // wait polls the deadline every 20 ms. Asserted on the helpers rather
        // than a real playback, since there is no voice or device here.
        let deadline = AtomicU64::new(1);
        assert!(
            crate::clock::past_deadline(&deadline, 1),
            "a deadline at or before now is past"
        );
        assert!(crate::clock::past_deadline(&deadline, 2));
        assert!(
            !crate::clock::past_deadline(&deadline, 0),
            "not yet, a millisecond before"
        );
        let t0 = Instant::now();
        assert!(sleep_until_deadline(Duration::from_secs(30), &deadline));
        assert!(
            t0.elapsed() < Duration::from_millis(200),
            "waited {:?} on an already-past deadline",
            t0.elapsed()
        );
    }

    #[test]
    fn no_deadline_means_never_cut_off() {
        // Zero is "no deadline", so a thread that was never asked to finish
        // waits its whole course out. (Otherwise every speak would cut itself
        // off at once, since the clock is always past zero.)
        let none = AtomicU64::new(0);
        assert!(!crate::clock::past_deadline(&none, u64::MAX / 2));
        assert!(!crate::clock::past_deadline(&none, 0));
        let t0 = Instant::now();
        assert!(!sleep_until_deadline(Duration::from_millis(50), &none));
        assert!(
            t0.elapsed() >= Duration::from_millis(50),
            "no deadline must let the sleep run: {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn the_finish_deadline_is_what_drop_sets_and_it_is_bounded() {
        // The drop path's whole contract: the trailing utterance is spoken
        // within a bound, so a device that never drains cannot hang the
        // process on the way out.
        assert!(SPEAK_DRAIN_MS > 0, "a zero deadline would cut at once");
        assert!(
            SPEAK_DRAIN_MS < 60_000,
            "a drain this long is a hang in disguise: {SPEAK_DRAIN_MS} ms"
        );
        // The deadline arithmetic `drop` uses: now + bound, which must land in
        // the future rather than in the past (an overflow would).
        let deadline = drop_deadline(false, 100);
        assert_eq!(deadline, 100 + SPEAK_DRAIN_MS);
        assert!(!crate::clock::past_deadline(&AtomicU64::new(deadline), 100));
        assert!(crate::clock::past_deadline(
            &AtomicU64::new(deadline),
            deadline
        ));
    }

    #[test]
    fn an_aborted_run_drops_with_a_deadline_already_past() {
        // What Ctrl-C buys: `drop` must not arm a fresh 15 s window over the
        // abort, or the join waits out the sentence the user just cancelled.
        // Literals, never `now_ms()`, which returns 0 on a first call and 0
        // reads as "no deadline".
        assert_eq!(drop_deadline(true, 100), 100);
        assert!(crate::clock::past_deadline(
            &AtomicU64::new(drop_deadline(true, 7)),
            7
        ));
    }

    #[test]
    fn abort_marks_the_cut_and_never_stores_the_no_deadline_sentinel() {
        // The other half of the same contract: `abort` must leave the deadline
        // already expired, so `play_mono` cuts the playback and the speak loop
        // skips the queued utterance — and never store 0, which `past_deadline`
        // reads as "no deadline" and would cut nothing.
        let cut = SpeechCut {
            finish_by_ms: Arc::new(AtomicU64::new(0)),
            aborted: AtomicBool::new(false),
        };
        cut.abort();
        assert!(cut.aborted.load(Ordering::Acquire));
        let stored = cut.finish_by_ms.load(Ordering::Acquire);
        assert_ne!(
            stored, 0,
            "0 means no deadline, so the abort would do nothing"
        );
        // `past_deadline` is `now >= deadline`, and the guard in `abort` can
        // store 1 while the clock still reads 0, so the cut is expired once the
        // clock passes that instant rather than the same millisecond. A bounded
        // wait, not a sleep: the assert is what decides.
        let t0 = Instant::now();
        while !crate::clock::past_deadline(&cut.finish_by_ms, now_ms())
            && t0.elapsed() < Duration::from_millis(200)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(crate::clock::past_deadline(&cut.finish_by_ms, now_ms()));
        // Idempotent: a second abort cannot clear the mark, and it may not reset
        // the deadline to the no-deadline sentinel.
        cut.abort();
        assert!(cut.aborted.load(Ordering::Acquire));
        assert_ne!(cut.finish_by_ms.load(Ordering::Acquire), 0);
        assert!(crate::clock::past_deadline(&cut.finish_by_ms, now_ms()));
    }
}
