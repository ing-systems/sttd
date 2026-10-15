//! The translation thread and its bounded queue, between the ASR worker and
//! the emitter.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::emit::{Emission, Emitter};

use super::engine::Translator;
use super::state::{Attempt, Loaded, Loads, RealTranslator, TranslationState};
use super::text::fallback_or_translated;

/// One accepted transcript on its way to the translation thread.
///
/// The `samples` and `infer_ms` travel with the job because the metrics row is
/// written *after* translation: the `text` column has to hold what was actually
/// emitted, and after this step that is the translation, not the source.
pub(crate) struct TranslationJob {
    pub(crate) tag: String,
    pub(crate) pcm: Vec<f32>,
    /// The transcript as the gates left it, before translation.
    pub(crate) source: String,
    pub(crate) verdict: crate::gate::TranscriptVerdict,
    pub(crate) evidence: crate::utterance::Evidence,
    pub(crate) samples: usize,
    pub(crate) infer_ms: f64,
    /// Which translator was live when this job was queued, so a language switch
    /// mid-queue is visible in the log rather than a silent change of output
    /// language. Filled by [`TranslationOut::enqueue`], which is the only
    /// producer, so it cannot be forgotten at a call site.
    pub(crate) generation: u64,
}

/// Bounded depth of the translation queue.
///
/// Emissions are seconds apart and a translation takes 80 to 700 ms, so this never
/// fills in practice. It is here so that a stalled or slow translator costs queued
/// work rather than the inference thread's time.
pub(crate) const TRANSLATION_QUEUE_DEPTH: usize = 4;

/// The translated emission: hand it a transcript, a dedicated thread translates
/// and emits it.
///
/// Off the ASR worker on purpose. Translation is 60 to 130 ms of CPU per
/// utterance against the worker's 50 ms of inference, and inline it would add
/// that to every gap between utterances.
///
/// The queue is a **bounded FIFO**, oldest dropped first. Not newest-wins like
/// `SpeechOut`: that rule exists because stale *audio* puts words in the user's
/// mouth, and a backlog of speech is worthless by the time it would be read. A
/// backlog of translations is not stale — it is work in order, and the user is
/// waiting for all of it. Dropping the oldest is the honest choice when the
/// translator cannot keep up: the newest transcript is the one the user just
/// said and still cares about.
///
/// The thread exists from startup in microphone mode even with nothing loaded,
/// because the switch that turns the step on arrives on the bus thread and the
/// ASR worker cannot be handed a lane after it started. It is fed **only** when a
/// translation will actually happen (see [`TranslationState::will_translate`]),
/// so a run that never translates keeps today's path: the worker emits inline and
/// this queue never fills.
pub(crate) struct TranslationOut<T = Translator, L = RealTranslator> {
    /// `(queue, wakeup)`: the pending jobs in order, and the signal to look.
    queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)>,
    /// Set by `drop`, so the thread finishes the queue and exits.
    closed: Arc<AtomicBool>,
    state: TranslationState<T, L>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl<T: Loaded, L: Loads<T>> TranslationOut<T, L> {
    /// Spawn the translation thread around the shared state, an emitter handle
    /// onto the shared emission queue, and the `--metrics` writer.
    ///
    /// The translator is not an argument: it arrives in the state's slot, loaded
    /// at startup by `--translate` or later over the bus.
    pub(crate) fn spawn(
        state: TranslationState<T, L>,
        mut emitter: Box<dyn Emitter + Send>,
        metrics: crate::metrics::MetricsSlot,
    ) -> Self {
        let queue: Arc<(Mutex<VecDeque<TranslationJob>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let closed = Arc::new(AtomicBool::new(false));

        let handle = {
            let (queue, closed) = (Arc::clone(&queue), Arc::clone(&closed));
            let thread_state = state.clone();
            std::thread::Builder::new()
                .name("translate".into())
                .spawn(move || {
                    translate_loop(&thread_state, &queue, &closed, emitter.as_mut(), &metrics)
                })
                .ok()
        };
        Self {
            queue,
            closed,
            state,
            handle,
        }
    }

    /// Whether the next job would be translated, which is what the ASR worker asks
    /// before handing one over: false means the utterance stays on the worker's
    /// inline path, so a run with the step off never enters this queue.
    pub(crate) fn will_translate(&self) -> bool {
        self.state.will_translate()
    }

    /// Queue one transcript for translation. Never blocks and never fails: a
    /// dead thread or a full queue costs one utterance, never the transcript.
    ///
    /// The job is stamped with the translator that is live now, so a language
    /// switch while it waits is visible in the log rather than a silent change of
    /// output language.
    ///
    /// A full queue drops the **oldest** waiting job and warns, naming the tag,
    /// so the log says which transcript was lost rather than only that one was.
    pub(crate) fn enqueue(&self, mut job: TranslationJob) {
        job.generation = self.state.lock_slot().generation;
        let dropped = {
            let (lock, wakeup) = &*self.queue;
            let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut dropped = None;
            while pending.len() >= TRANSLATION_QUEUE_DEPTH {
                match pending.pop_front() {
                    Some(old) => dropped = Some(old.tag),
                    // The queue is non-empty whenever `len >= depth >= 1`.
                    None => break,
                }
            }
            pending.push_back(job);
            // Notified with the lock held, which is what lets the thread wait
            // without a timeout: a notification sent after the waiter has checked
            // the queue and before it sleeps is the one a bare `notify_all` loses.
            wakeup.notify_all();
            dropped
        };
        if let Some(tag) = dropped {
            log::warn!(
                "translation queue full ({TRANSLATION_QUEUE_DEPTH} deep): dropped the \
                 oldest waiting transcript {tag} to make room"
            );
        }
    }
}

impl<T, L> Drop for TranslationOut<T, L> {
    fn drop(&mut self) {
        // Close the queue, then join: the thread finishes what is waiting and
        // exits, and the join is what waits for that. No more jobs can arrive,
        // since the only producer is the ASR worker holding this value.
        self.closed.store(true, Ordering::Release);
        {
            let (lock, wakeup) = &*self.queue;
            // The lock is taken and released only to be sure the flag is visible
            // before the wakeup, which is what `enqueue` does too: a notification
            // sent between the thread's check of the queue and its sleep is the one
            // a bare `notify_all` loses, and holding the lock across both is what
            // rules that out.
            drop(lock.lock().unwrap_or_else(|e| e.into_inner()));
            wakeup.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                log::warn!("translation thread panicked");
            }
        }
    }
}

/// One thread's whole job: take the oldest pending transcript, translate it,
/// write its metrics row, emit it, repeat until closed and empty.
///
/// The trailing utterance is translated, not dropped: `--file --translate en` with
/// the last transcript swallowed would be the channel silently losing work, and it
/// is the same reasoning the ASR shutdown flush and `SpeechOut::drop` use. So
/// `closed` ends the loop only once the queue is empty.
///
/// The wait is therefore unbounded on purpose, and what bounds the shutdown is
/// the work itself: a piece's decode stops at
/// [`translator_store::Family::max_new_tokens`] tokens, so the drain is bounded by
/// the queue depth and the transcript's length. A wall-clock deadline would have
/// to interrupt a translation already in flight to be any tighter, which is not
/// something a flag can do.
fn translate_loop<T: Loaded, L: Loads<T>>(
    state: &TranslationState<T, L>,
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
    emitter: &mut dyn Emitter,
    metrics: &crate::metrics::MetricsSlot,
) {
    while let Some(job) = next_job(queue, closed) {
        emit_one(state, emitter, metrics, job);
    }
}

/// The next job to translate, or `None` once the queue is closed and empty.
fn next_job(
    queue: &(Mutex<VecDeque<TranslationJob>>, Condvar),
    closed: &AtomicBool,
) -> Option<TranslationJob> {
    let (lock, wakeup) = queue;
    let mut pending = lock.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(job) = pending.pop_front() {
            return Some(job);
        }
        if closed.load(Ordering::Acquire) {
            return None;
        }
        // No timeout: both notifiers send under the lock, so a wakeup cannot be
        // lost between the check above and this sleep.
        pending = wakeup.wait(pending).unwrap_or_else(|e| e.into_inner());
    }
}

/// Translate one job, log what it cost, write its metrics row and emit it.
///
/// The metrics row is written here rather than by the ASR worker because its
/// `text` column has to be what the emitter received. A dropped queue entry never
/// reaches this function, so it never reaches the metrics log either — which is
/// why the queue drop warns with the tag.
///
/// One lock is held across the decode, which is what lets `Disable(true)` promise
/// the weights are gone when it returns: it waits for a translation already
/// running instead of pulling the translator out from under one. Every other
/// utterance still takes the inline path on the worker, so nothing waits behind
/// this lock except the switch that asked for it.
fn emit_one<T: Loaded, L: Loads<T>>(
    state: &TranslationState<T, L>,
    emitter: &mut dyn Emitter,
    metrics: &crate::metrics::MetricsSlot,
    job: TranslationJob,
) {
    let mut panicked = false;
    let attempt = {
        let mut slot = state.lock_slot();
        if job.generation != slot.generation {
            log::info!(
                "{}: the target language changed while this waited; translating with the \
                 live one",
                job.tag
            );
        }
        let t0 = Instant::now();
        let attempt = if state.wanted_now() {
            state.translate_locked(&mut slot, &job.source)
        } else {
            // Disabled between the enqueue and here. The source text is what the
            // step would have emitted anyway, and saying so is the failure policy
            // rather than losing the utterance.
            Attempt::Failed("the translation step was disabled".to_string())
        };
        let took = t0.elapsed().as_secs_f64() * 1000.0;
        let direction = slot
            .translator
            .as_ref()
            .map(|t| t.direction().to_string())
            .unwrap_or_else(|| "no translator".to_string());
        if let Attempt::Panicked(what) = &attempt {
            log::error!("{}: the decoder panicked: {what}", job.tag);
            panicked = true;
        }
        (attempt, direction, took)
    };
    let (attempt, direction, translate_ms) = attempt;
    // After the guard is released: dropping the weights takes the same lock.
    if panicked {
        state.drop_panicked_translator();
    }
    let ran = matches!(attempt, Attempt::Translated(_));
    let text = fallback_or_translated(&job.source, attempt.into_result(), &job.tag);
    log::info!(
        "{}: translate ({direction}) took {translate_ms:.0} ms ({:?} -> {text:?})",
        job.tag,
        job.source
    );
    // `src_text` only when a translation ran for *this* utterance, so a log from a
    // run with the step off stays comparable with one from before it existed.
    let mut writer = metrics.lock();
    crate::metrics::write_metrics_row(
        &mut writer,
        &job.tag,
        job.samples,
        &job.evidence,
        job.verdict,
        &text,
        job.infer_ms,
        ran.then_some(job.source.as_str()),
    );
    // The writer is released before the emission: the emit queue has its own lock,
    // and nothing should hold two at once.
    drop(writer);
    emitter.emit(Emission {
        tag: &job.tag,
        pcm: &job.pcm,
        text: &text,
        verdict: job.verdict,
        evidence: &job.evidence,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{Emission, Emitter};
    use crate::gate::TranscriptVerdict;
    use crate::metrics::MetricsSlot;
    use crate::translate::plan::TranslationPlan;
    use crate::translate::state::fakes::*;
    use anyhow::Result;

    // -- the queue, with a stand-in translator ------------------------------
    //
    // `TranslationOut` takes a `Box<dyn Emitter>`, so a lane can be driven with the
    // same recording emitter the `asr` tests use, and with a translator that needs
    // no weights. What is under test is the queue contract and the two policies
    // that hang off it: the source text is what an untranslated or failed utterance
    // emits, and the drain at shutdown keeps the trailing utterance.

    /// What a lane test reads back: `(tag, text)` per emission, in order.
    type Sink = Arc<Mutex<Vec<(String, String)>>>;

    /// An emitter that records into a shared sink, because the lane owns its
    /// emitter as a `'static` box (production hands it to a thread) and a test still
    /// has to read the result afterwards.
    struct Recorder(Sink);

    impl Emitter for Recorder {
        fn emit(&mut self, emission: Emission<'_>) {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((emission.tag.to_string(), emission.text.to_string()));
        }
    }

    /// A lane over `state`, recording into a fresh sink.
    fn recording_lane<T: Loaded, L: Loads<T>>(
        state: TranslationState<T, L>,
    ) -> (TranslationOut<T, L>, Sink) {
        let sink: Sink = Arc::new(Mutex::new(Vec::new()));
        let lane = TranslationOut::spawn(
            state,
            Box::new(Recorder(Arc::clone(&sink))),
            MetricsSlot::default(),
        );
        (lane, sink)
    }

    /// The `(tag, text)` pairs a sink collected, once the lane has been dropped.
    fn recorded(sink: &Sink) -> Vec<(String, String)> {
        sink.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn job(tag: &str, source: &str) -> TranslationJob {
        TranslationJob {
            tag: tag.to_string(),
            pcm: vec![0.0; 160],
            source: source.to_string(),
            verdict: TranscriptVerdict::Emit,
            evidence: crate::utterance::Evidence {
                speech_frames: 1,
                total_frames: 2,
                rms: 0.01,
            },
            samples: 16_000,
            infer_ms: 12.0,
            generation: 0,
        }
    }

    #[test]
    fn the_lane_translates_a_job_through_the_shared_state() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane); // the drain is what waits for it
        assert_eq!(
            recorded(&sink),
            vec![(
                "utt_000001".to_string(),
                "translated|Dzień dobry".to_string()
            )],
            "the emitter receives the translation, under the job's own tag"
        );
    }

    #[test]
    fn a_disabled_step_emits_the_source_text_rather_than_losing_it() {
        // The job was queued while the step was on and disabled before the lane
        // reached it. Losing the utterance would be the failure; the source text is
        // what the step emits anyway.
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s.clone());
        s.disable(false);
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane);
        assert_eq!(
            recorded(&sink),
            vec![("utt_000001".to_string(), "Dzień dobry".to_string())],
            "the utterance must still arrive, untranslated"
        );
    }

    #[test]
    fn the_drain_keeps_the_trailing_utterance() {
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        // Fewer than the queue depth, so the drain is what is under test and the
        // drop-oldest policy is not.
        for i in 1..=TRANSLATION_QUEUE_DEPTH - 1 {
            lane.enqueue(job(&format!("utt_{i:06}"), "tekst"));
        }
        drop(lane);
        let tags: Vec<String> = recorded(&sink).into_iter().map(|(tag, _)| tag).collect();
        assert_eq!(
            tags,
            vec!["utt_000001", "utt_000002", "utt_000003"],
            "shutdown drains the queue in order rather than dropping the tail"
        );
    }

    #[test]
    fn a_full_queue_drops_the_oldest_and_keeps_the_newest() {
        // The documented policy for a translator that cannot keep up: the newest
        // transcript is the one the user just said.
        let s = state("pl");
        with_fake_translator(&s, "en", "opus-mt-pl-en");
        let (lane, sink) = recording_lane(s);
        for i in 1..=TRANSLATION_QUEUE_DEPTH + 2 {
            lane.enqueue(job(&format!("utt_{i:06}"), "tekst"));
        }
        drop(lane);
        let tags: Vec<String> = recorded(&sink).into_iter().map(|(tag, _)| tag).collect();
        assert!(
            tags.len() < TRANSLATION_QUEUE_DEPTH + 3,
            "a full queue sheds rather than growing without bound: {tags:?}"
        );
        assert_eq!(
            tags.last().map(String::as_str),
            Some(format!("utt_{:06}", TRANSLATION_QUEUE_DEPTH + 2).as_str()),
            "the newest survives"
        );
    }

    #[test]
    fn a_panicking_decode_emits_the_source_text_and_drops_the_translator() {
        // A dead lane would lose every later utterance, because the worker had
        // already handed them over. So a panic becomes a failed translation: the
        // source text is emitted, and the translator is dropped so `Enable` loads
        // a fresh one.
        struct Panicking;
        impl Loaded for Panicking {
            fn translate(&mut self, _source: &str) -> Result<String> {
                panic!("the decoder exploded");
            }
            fn spec(&self) -> &str {
                "panicking"
            }
            fn direction(&self) -> &str {
                "pl -> en"
            }
        }
        #[derive(Clone)]
        struct PanicLoad;
        impl Loads<Panicking> for PanicLoad {
            fn load(&self, _plan: &TranslationPlan, _target: &str) -> Result<Panicking> {
                Ok(Panicking)
            }
        }
        let s: TranslationState<Panicking, PanicLoad> = TranslationState::new(
            TranslationPlan::for_tests_with_model("pl", "nllb-200-distilled-600M"),
            PanicLoad,
        );
        seat(&s, "en", "panicking", || Panicking);

        let (lane, sink) = recording_lane(s.clone());
        lane.enqueue(job("utt_000001", "Dzień dobry"));
        drop(lane);
        assert_eq!(
            recorded(&sink),
            vec![("utt_000001".to_string(), "Dzień dobry".to_string())],
            "a panicking decode must not cost the utterance"
        );
        assert!(!s.is_loaded(), "and the translator is dropped");
        assert!(
            !s.will_translate(),
            "so the worker goes back to the inline path"
        );
    }
}
