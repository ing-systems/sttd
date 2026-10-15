//! One utterance through the model: the partial preview, and the closed
//! utterance's decode, gates, cleaning and hand-off to emission.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::{INFERENCE_PAD_SAMPLES, RTF_WARN_RTF, SAMPLE_RATE};
use crate::emit::{Emission, Emitter};
use crate::gate::GateConfig;
use crate::metrics::{infer_rtf, write_metrics_row, MetricsSlot};
use crate::speech::{pad_trailing_with_dither, SpeechModel};
use crate::translate::{TranslationJob, TranslationOut};
use crate::utterance::Utterance;

use super::condition::{condition_utterance, remove_split_overlap};

/// Decode one partial and return its log line, or `None` when there is nothing
/// worth showing: an empty or gated transcript (hallucination boilerplate,
/// a repetition loop) or a failed decode. Conditioned and padded exactly like a
/// final (`transcribe_and_emit`), so a partial reads the audio the same way.
pub(super) fn partial_line(
    model: &mut dyn SpeechModel,
    pcm: &[f32],
    tag: &str,
    gate: &GateConfig,
    running: &Arc<AtomicBool>,
) -> Option<String> {
    let conditioned = condition_utterance(pcm);
    let padded = pad_trailing_with_dither(&conditioned, INFERENCE_PAD_SAMPLES);
    let t0 = Instant::now();
    let result = model.transcribe(&padded, running);
    let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(t) => {
            let text = t.text.trim();
            if !matches!(
                gate.decide_transcript(text),
                crate::gate::TranscriptVerdict::Emit
            ) {
                return None;
            }
            let audio_sec = pcm.len() as f64 / SAMPLE_RATE as f64;
            Some(format!(
                "{tag}: partial ({audio_sec:.1} s audio, {infer_ms:.0} ms): {text}"
            ))
        }
        Err(e) => {
            // Shutdown skips inference through the pre-call check: not a failure.
            if running.load(Ordering::Relaxed) {
                log::warn!("{tag}: partial failed: {e:?}");
            }
            None
        }
    }
}

/// Post-gate transcript polish, in the one order that matters: dictionary
/// first (casing restore plus strict fuzzy repair), then `clean` (collapse
/// 3+ repeats, optionally strip fillers).
///
/// Only an `Emit` verdict reaches here, so a near-miss dictionary key cannot
/// resurrect boilerplate, and cleaning can only shorten what the gate
/// accepted. `confidences` gates the fuzzy dictionary step (see
/// `dictionary`); it is `None` when the engine reported no per-word scores.
/// Returns the polished text — empty (or all whitespace) means cleaning
/// removed everything and the caller must turn the verdict into
/// `DropEmpty` rather than emit empty text.
fn polish_transcript(
    tag: &str,
    text: &str,
    dictionary: &[String],
    confidences: Option<&std::collections::HashMap<String, f32>>,
    remove_hesitations: bool,
) -> String {
    let corrected: std::borrow::Cow<'_, str> = if dictionary.is_empty() {
        std::borrow::Cow::Borrowed(text)
    } else {
        let corrected = crate::dictionary::correct_transcription(text, dictionary, confidences);
        if corrected != text {
            log::debug!("{tag}: dictionary {text:?} -> {corrected:?}");
        }
        std::borrow::Cow::Owned(corrected)
    };
    let cleaned = crate::clean::clean_transcription(&corrected, remove_hesitations);
    if cleaned != corrected {
        log::debug!("{tag}: clean {corrected:?} -> {cleaned:?}");
    }
    cleaned
}

/// One utterance through conditioning, inference, gate and emission. Tag
/// numbering is owned by the caller so it stays monotonic across reloads.
/// Returns whether the transcript was actually emitted (vs dropped or
/// failed), so the caller can drive single-shot `--autopause` without
/// seeing the gate.
#[allow(clippy::too_many_arguments)]
pub(super) fn transcribe_and_emit(
    model: &mut dyn SpeechModel,
    utterance: &Utterance,
    utterance_no: u64,
    gate: &GateConfig,
    running: &Arc<AtomicBool>,
    emitter: &mut dyn Emitter,
    translation: &mut Option<TranslationOut>,
    metrics: &MetricsSlot,
    dictionary: &[String],
    remove_hesitations: bool,
    prev_emit_text: &mut String,
) -> bool {
    let tag = format!("utt_{utterance_no:06}");

    // Levels already normalized frame-wise by `Agc` (VAD-gated, shared-gain);
    // no per-utterance static gain here — that path is deleted.
    let conditioned = condition_utterance(&utterance.pcm);
    // Trailing context past the last word so final tokens are not clipped
    // (dithered, not digital zero — see `INFERENCE_PAD_SAMPLES`). No leading
    // pad: the utterance already starts with pre-roll. The WAV
    // dump still uses raw `utterance.pcm`; metrics/RTF count what the model
    // actually decoded (padded length, plus the short-input floor via
    // `decoded_sample_count`).
    let padded = pad_trailing_with_dither(&conditioned, INFERENCE_PAD_SAMPLES);

    // Inference time only (conditioning + edge pad excluded): feeds the RTF
    // column of the `--metrics` JSONL log.
    let t0 = Instant::now();
    let result = model.transcribe(&padded, running);
    let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let decoded_samples = model.decoded_sample_count(padded.len());
    let (audio_sec, rtf) = infer_rtf(decoded_samples, infer_ms);
    log::info!("{tag}: transcribe took {infer_ms:.0} ms ({audio_sec:.1} s audio, rtf={rtf:.2})");
    if rtf > RTF_WARN_RTF {
        log::warn!(
            "{tag}: inference slower than realtime (rtf={rtf:.2}, {infer_ms:.0} ms for \
             {audio_sec:.1} s audio): consider a smaller --model, --threads 0, or GPU"
        );
    }
    match result {
        Ok(t) => {
            // Split-seam dedup before the gate: a continuation re-decodes the
            // boundary words, so strip the overlap vs the previous emission.
            // Borrowed unless the dedup actually changed something.
            let raw_trimmed = t.text.trim();
            let deduped: std::borrow::Cow<'_, str> =
                if utterance.split_continuation && !prev_emit_text.is_empty() {
                    let stripped = remove_split_overlap(prev_emit_text, raw_trimmed);
                    if stripped.len() != raw_trimmed.len() {
                        log::debug!(
                            "{tag}: split-seam dedup stripped {} chars",
                            raw_trimmed.len() - stripped.len()
                        );
                    }
                    std::borrow::Cow::Owned(stripped)
                } else {
                    std::borrow::Cow::Borrowed(raw_trimmed)
                };
            let deduped = deduped.trim();
            let mut verdict = gate.decide_transcript(deduped);
            let mut emitted = matches!(verdict, crate::gate::TranscriptVerdict::Emit);
            // Per-word confidences gate the fuzzy dictionary step. Built only
            // when they can be used: no dictionary, or no per-word scores from
            // the engine, means the ungated form.
            let confidences = if emitted && !dictionary.is_empty() {
                (!t.word_confidences.is_empty())
                    .then(|| crate::dictionary::confidence_map(&t.word_confidences))
            } else {
                None
            };
            let mut final_text = if emitted {
                polish_transcript(
                    &tag,
                    deduped,
                    dictionary,
                    confidences.as_ref(),
                    remove_hesitations,
                )
            } else {
                deduped.to_string()
            };
            if emitted && final_text.trim().is_empty() {
                // Cleaning removed everything (hesitations only with the flag
                // on): drop instead of emitting empty text, which would
                // clobber the clipboard.
                verdict = crate::gate::TranscriptVerdict::DropEmpty;
                emitted = false;
                final_text.clear();
            }
            let final_trimmed = final_text.trim();
            // The split-seam dedup stays source-vs-source: it compares what was
            // said, which is what the gates judged, not what was translated.
            if emitted {
                *prev_emit_text = final_trimmed.to_string();
            }
            // The translation step runs last, after every gate, and only for an
            // utterance that is actually emitted: a drop is never translated.
            // `fallback_or_translated` owns the failure policy, so a failed or
            // empty translation emits the source text with a warn rather than
            // losing the utterance.
            //
            // With the step on, the job is handed to the translation thread and
            // this function does not emit: the thread owns the translated text,
            // so it owns the emission too. The `pcm` copy is what the queue needs,
            // since the utterance is gone by the time the thread gets to it.
            //
            // `will_translate` is what keeps a run that never translates on the
            // inline path below: the lane exists from startup (a switch arrives on
            // the bus thread, which cannot reach into this worker), but an
            // utterance is only handed over when a translation will really happen.
            // So the lane's drop-oldest queue never applies to untranslated text.
            if let Some(out) = translation.as_ref() {
                if emitted && out.will_translate() {
                    out.enqueue(TranslationJob {
                        tag: tag.clone(),
                        pcm: utterance.pcm.clone(),
                        source: final_trimmed.to_string(),
                        verdict,
                        evidence: utterance.evidence,
                        samples: decoded_samples,
                        infer_ms,
                        // Stamped by `enqueue`, the only producer: a job cannot
                        // be queued without recording which translator was live.
                        generation: 0,
                    });
                    return true;
                }
            }
            // No step, or a drop: the transcript is its own emission. A drop is
            // never translated, so nothing is spent on text the pipeline decided
            // not to emit.
            let mut writer = metrics.lock();
            write_metrics_row(
                &mut writer,
                &tag,
                decoded_samples,
                &utterance.evidence,
                verdict,
                final_trimmed,
                infer_ms,
                None,
            );
            // Released before the emission: the emit queue has its own lock, and
            // nothing should hold two at once.
            drop(writer);
            emitter.emit(Emission {
                tag: &tag,
                pcm: &utterance.pcm,
                text: final_trimmed,
                verdict,
                evidence: &utterance.evidence,
            });
            emitted
        }
        Err(e) => {
            // Shutdown skips in-flight inference via a pre-call running
            // check (see `ParakeetModel`): not a failure, just a fast exit.
            if !running.load(Ordering::Relaxed) {
                log::info!("{tag}: transcription aborted (shutdown)");
            } else {
                log::error!("{tag}: transcription failed: {e:?}");
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::test_support::*;

    #[test]
    fn partial_line_reports_text_audio_and_the_next_tag() {
        let (mut model, inputs) = ScriptedModel::new("  dzień dobry  ");
        let running = Arc::new(AtomicBool::new(true));
        let pcm = vec![0.1f32; 24_000];
        let line = partial_line(&mut model, &pcm, "utt_000007", &default_gate(), &running)
            .expect("a normal transcript is shown");
        assert!(
            line.starts_with("utt_000007: partial (1.5 s audio, "),
            "{line}"
        );
        assert!(line.ends_with("): dzień dobry"), "{line}");
        // Fed like a final: conditioned, plus the trailing pad.
        assert_eq!(*inputs.lock().unwrap(), [24_000 + INFERENCE_PAD_SAMPLES]);
    }

    #[test]
    fn partial_line_hides_gated_empty_and_failed_decodes() {
        let running = Arc::new(AtomicBool::new(true));
        let pcm = vec![0.1f32; 16_000];
        for text in ["", "   ", "Dziękuję za uwagę."] {
            let (mut model, _) = ScriptedModel::new(text);
            assert!(
                partial_line(&mut model, &pcm, "utt_000001", &default_gate(), &running).is_none(),
                "{text:?} must not be shown"
            );
        }
        let (mut model, _) = ScriptedModel::new("tekst");
        model.fail = true;
        assert!(partial_line(&mut model, &pcm, "utt_000001", &default_gate(), &running).is_none());
    }

    #[test]
    fn polish_applies_dictionary_then_clean_then_empty_check() {
        let dict = vec!["Kubernetes".to_string()];
        // Casing restore first, then the 3+ repeat collapse sees the
        // canonical form (not the raw one).
        assert_eq!(
            polish_transcript(
                "utt_000001",
                "kubernetes kubernetes kubernetes",
                &dict,
                None,
                false
            ),
            "Kubernetes Kubernetes"
        );
        // Filler stripping only when asked; whitespace is normalized either way.
        assert_eq!(
            polish_transcript("utt_000002", "  yyy   dzień  dobry ", &[], None, true),
            "dzień dobry"
        );
        assert_eq!(
            polish_transcript("utt_000003", "  yyy   dzień  dobry ", &[], None, false),
            "yyy dzień dobry"
        );
        // Cleaning can empty the transcript: the caller turns that into
        // DropEmpty, so this must not fall back to the input.
        assert_eq!(
            polish_transcript("utt_000004", "yyy eee", &[], None, true),
            ""
        );
        assert_eq!(polish_transcript("utt_000005", "", &[], None, false), "");
    }

    /// One utterance through `transcribe_and_emit` with a scripted decode, a
    /// recording emitter and no metrics file. Returns what was emitted and the
    /// caller's `prev_emit_text` afterwards, which is the pair the two tests
    /// below need to pin.
    fn run_pass(
        text: &str,
        remove_hesitations: bool,
        prev_emit_text: &mut String,
    ) -> (bool, crate::emit::RecordedEmission) {
        let (mut model, _inputs) = ScriptedModel::new(text);
        let running = Arc::new(AtomicBool::new(true));
        let mut emitter = crate::emit::RecordingEmitter::new();
        let emitted = transcribe_and_emit(
            &mut model,
            &closed_utterance(16_000),
            1,
            &default_gate(),
            &running,
            &mut emitter,
            // No translation step: these tests pin the source-text pipeline.
            &mut None,
            &MetricsSlot::default(),
            &[],
            remove_hesitations,
            prev_emit_text,
        );
        let record = emitter
            .records
            .pop()
            .expect("every pass reaches the emitter exactly once");
        (emitted, record)
    }

    /// Cleaning can remove everything (`clean.rs`), and emitting the empty
    /// result would clobber the clipboard. The verdict must be downgraded to
    /// `DropEmpty` and the split-seam text left alone, so the next utterance's
    /// dedup still compares against what was actually said.
    #[test]
    fn cleaning_to_nothing_downgrades_the_verdict_instead_of_emitting_empty_text() {
        let mut prev = String::from("coś wcześniej");
        // The gate accepts it (not empty, not boilerplate, not a loop);
        // `clean` then strips every filler.
        let (emitted, record) = run_pass("eee um", true, &mut prev);

        assert!(!emitted, "nothing was emitted");
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::DropEmpty,
            "the verdict must say the text was cleaned away"
        );
        assert!(record.text.trim().is_empty(), "text={:?}", record.text);
        assert!(!record.emitted, "an empty Emit must not count as emitted");
        assert_eq!(record.drop_kind, None, "DropEmpty is not a drop kind");
        assert!(
            !record.copied && !record.pasted && !record.printed,
            "no side effect may run for an empty transcript"
        );
        assert_eq!(
            prev, "coś wcześniej",
            "a drop must not advance the split-seam text"
        );
    }

    /// Split-seam dedup compares against the last thing actually said, so
    /// `prev_emit_text` advances on an emit and is left alone by a drop.
    /// Without that, one dropped utterance would let the next continuation
    /// strip words the speaker did say twice.
    #[test]
    fn prev_emit_text_advances_on_an_emit_and_never_on_a_drop() {
        let mut prev = String::new();
        let (emitted, record) = run_pass("dzień dobry", false, &mut prev);
        assert!(emitted);
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::Emit,
            "verdict={:?}",
            record.verdict
        );
        assert_eq!(prev, "dzień dobry", "an emit advances the dedup baseline");

        // `napisy` is exact-match hallucination boilerplate (`config.rs`).
        let (emitted, record) = run_pass("napisy", false, &mut prev);
        assert!(!emitted, "boilerplate must not be emitted");
        assert_eq!(
            record.verdict,
            crate::gate::TranscriptVerdict::Drop {
                kind: crate::gate::TranscriptDropKind::Hallucination
            },
            "verdict={:?}",
            record.verdict
        );
        assert_eq!(
            prev, "dzień dobry",
            "a drop must not become the dedup baseline"
        );
    }

    /// The text the emitter receives when the translation step is on: the
    /// translation, or the source text when the translation fails or comes back
    /// empty.
    ///
    /// `Translator` holds two `ort` sessions, so no unit test can build one.
    /// What is under test here is the step's failure policy, which
    /// `fallback_or_translated` owns outright, and the emitter's text is that
    /// function's return value. The model-run half needs weights and is out of
    /// scope for a unit test.
    ///
    /// The call is now on the translation thread rather than inline here, which
    /// does not change the policy: `emit_one` in `translate` applies this same
    /// function to the job's `source` before it touches the emitter.
    #[test]
    fn the_emitted_text_is_the_translation_or_the_source_fallback() {
        use crate::translate::fallback_or_translated;
        // A translation that arrives: that is what is emitted.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok("Good morning".to_string()), "utt_000001"),
            "Good morning"
        );
        // A translation that failed: the source, not an empty emission.
        assert_eq!(
            fallback_or_translated(
                "Dzień dobry",
                Err(anyhow::anyhow!("the decoder ran out of cache slots")),
                "utt_000002"
            ),
            "Dzień dobry"
        );
        // A translation that came back empty: also the source, so an empty
        // string never reaches the clipboard.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok(String::new()), "utt_000003"),
            "Dzień dobry"
        );
        // A whitespace-only translation is empty for this purpose too.
        assert_eq!(
            fallback_or_translated("Dzień dobry", Ok("   ".to_string()), "utt_000004"),
            "Dzień dobry"
        );
    }
}
