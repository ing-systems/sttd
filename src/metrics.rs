// ---------------------------------------------------------------------------
// The `--metrics` log: one JSONL row per utterance
// ---------------------------------------------------------------------------
//
// Written by the ASR worker (every utterance it does not hand to translation)
// and by the translation thread (the rows whose `text` is a translation), so it
// sits beside both rather than inside either.

use std::sync::{Arc, Mutex};

use crate::config::SAMPLE_RATE;

/// Open the `--metrics` JSONL log for appending (creating parent dirs).
/// Observability only: errors are returned for the caller to warn-and-disable.
pub(crate) fn open_metrics_log(
    path: Option<&std::path::Path>,
) -> Option<std::io::BufWriter<std::fs::File>> {
    let path = path?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("metrics: failed to create {}: {e:?}", parent.display());
                return None;
            }
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => {
            log::info!(
                "metrics: appending per-utterance JSONL to {}",
                path.display()
            );
            Some(std::io::BufWriter::new(f))
        }
        Err(e) => {
            log::warn!(
                "metrics: failed to open {}: {e:?} (metrics disabled)",
                path.display()
            );
            None
        }
    }
}

/// The one `--metrics` writer, shared by the ASR worker and the translation
/// thread.
///
/// Shared rather than moved, and that is a correction of an earlier shape: the
/// worker writes a row for every utterance it does **not** hand over (each drop,
/// and every utterance while the translation step is off), so a writer that lived
/// only on the translation thread would silently lose exactly those rows. One
/// value in one place, guarded, keeps every row and keeps the lane from needing a
/// second file handle.
///
/// The lock is held across one row and its flush, which is what keeps two threads
/// from interleaving bytes in one `BufWriter`. A poisoned lock is taken anyway: a
/// metrics problem must never take transcription down with it.
#[derive(Clone, Default)]
pub(crate) struct MetricsSlot(Arc<Mutex<Option<std::io::BufWriter<std::fs::File>>>>);

impl MetricsSlot {
    /// Wrap an already-opened writer (`None` when `--metrics` is off).
    pub(crate) fn new(writer: Option<std::io::BufWriter<std::fs::File>>) -> Self {
        Self(Arc::new(Mutex::new(writer)))
    }

    /// The writer, for one row. `None` when `--metrics` is off or the open failed.
    pub(crate) fn lock(
        &self,
    ) -> std::sync::MutexGuard<'_, Option<std::io::BufWriter<std::fs::File>>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Machine-readable verdict for the `--metrics` log (human-readable
/// `DROPPED(...)` strings stay in the dump dir via `dump_reason`).
pub(crate) const fn verdict_slug(verdict: crate::gate::TranscriptVerdict) -> &'static str {
    match verdict {
        crate::gate::TranscriptVerdict::Emit => "emit",
        crate::gate::TranscriptVerdict::DropEmpty => "drop:empty",
        crate::gate::TranscriptVerdict::Drop { kind, .. } => match kind {
            crate::gate::TranscriptDropKind::Hallucination => "drop:hallucination",
            crate::gate::TranscriptDropKind::Repetition => "drop:repetition",
        },
    }
}

/// Escape a string for embedding in JSON (quotes, backslashes, control
/// chars); non-ASCII passes through as UTF-8. Stdlib only (no serde).
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Inference seconds per audio second for one utterance: `(audio_sec,
/// rtf)`. `rtf > 1` means slower than realtime. Pure for tests; the same
/// pair feeds the `--metrics` row and the slower-than-realtime warn.
pub(crate) fn infer_rtf(samples: usize, infer_ms: f64) -> (f64, f64) {
    let audio_sec = samples as f64 / SAMPLE_RATE as f64;
    let rtf = if audio_sec > 0.0 {
        infer_ms / 1000.0 / audio_sec
    } else {
        0.0
    };
    (audio_sec, rtf)
}

/// One `--metrics` JSONL row for a transcribed utterance. Pure over its
/// inputs (formatting only) so tests don't touch the filesystem or GPU.
/// `infer_ms` covers inference alone (conditioning/gain excluded); `rtf` is
/// inference seconds per audio second (>1 = slower than realtime).
///
/// `source` is the pre-translation text, written as a `src_text` column only
/// when the translation step ran. `None` writes the row exactly as it was
/// before that step existed, so a run without `--translate` and an old log stay
/// comparable.
pub(crate) fn format_metrics_row(
    tag: &str,
    samples: usize,
    evidence: &crate::utterance::Evidence,
    verdict: crate::gate::TranscriptVerdict,
    text: &str,
    infer_ms: f64,
    source: Option<&str>,
) -> String {
    let (audio_sec, rtf) = infer_rtf(samples, infer_ms);
    let src_text = match source {
        Some(src) => format!(",\"src_text\":\"{}\"", escape_json_string(src)),
        None => String::new(),
    };
    format!(
        "{{\"tag\":\"{}\",\"samples\":{},\"audio_sec\":{:.3},\"infer_ms\":{:.1},\"rtf\":{:.3},\
         \"speech_frames\":{},\"total_frames\":{},\"density\":{:.3},\"rms\":{:.5},\
         \"verdict\":\"{}\",\"text\":\"{}\"{}}}",
        escape_json_string(tag),
        samples,
        audio_sec,
        infer_ms,
        rtf,
        evidence.speech_frames,
        evidence.total_frames,
        evidence.density(),
        evidence.rms,
        verdict_slug(verdict),
        escape_json_string(text),
        src_text,
    )
}

/// Append one row via `format_metrics_row`, flushing so `--stop`/Ctrl-C
/// keeps every transcribed utterance even on a hard kill right after.
/// Write errors disable the log with a single warning (never fail
/// transcription for observability).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_metrics_row(
    metrics: &mut Option<std::io::BufWriter<std::fs::File>>,
    tag: &str,
    samples: usize,
    evidence: &crate::utterance::Evidence,
    verdict: crate::gate::TranscriptVerdict,
    text: &str,
    infer_ms: f64,
    source: Option<&str>,
) {
    use std::io::Write;
    let Some(w) = metrics else { return };
    let row = format_metrics_row(tag, samples, evidence, verdict, text, infer_ms, source);
    if let Err(e) = writeln!(w, "{row}") {
        log::warn!("metrics: write failed: {e:?} (metrics disabled)");
        *metrics = None;
        return;
    }
    if let Err(e) = w.flush() {
        log::warn!("metrics: flush failed: {e:?} (metrics disabled)");
        *metrics = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utterance::Evidence;

    #[test]
    fn verdict_slug_covers_every_disposition() {
        use crate::gate::{TranscriptDropKind, TranscriptVerdict};
        assert_eq!(verdict_slug(TranscriptVerdict::Emit), "emit");
        assert_eq!(verdict_slug(TranscriptVerdict::DropEmpty), "drop:empty");
        for (kind, slug) in [
            (TranscriptDropKind::Hallucination, "drop:hallucination"),
            (TranscriptDropKind::Repetition, "drop:repetition"),
        ] {
            assert_eq!(verdict_slug(TranscriptVerdict::Drop { kind }), slug);
        }
    }

    #[test]
    fn metrics_row_is_valid_json_with_rtf() {
        use crate::gate::TranscriptVerdict;
        let evidence = Evidence {
            speech_frames: 100,
            total_frames: 200,
            rms: 0.02,
        };
        // 16_000 samples = 1.0 s audio, 250 ms inference -> rtf 0.25.
        let row = format_metrics_row(
            "utt_000001",
            16_000,
            &evidence,
            TranscriptVerdict::Emit,
            "Dzień dobry",
            250.0,
            None,
        );
        assert!(row.contains("\"tag\":\"utt_000001\""), "{row}");
        assert!(row.contains("\"samples\":16000"), "{row}");
        assert!(row.contains("\"audio_sec\":1.000"), "{row}");
        assert!(row.contains("\"infer_ms\":250.0"), "{row}");
        assert!(row.contains("\"rtf\":0.250"), "{row}");
        assert!(row.contains("\"verdict\":\"emit\""), "{row}");
        assert!(row.contains("\"text\":\"Dzień dobry\""), "{row}");
        // No translation step ran, so no `src_text` column: the row is exactly
        // what it was before that column existed.
        assert!(!row.contains("src_text"), "{row}");
        assert!(row.starts_with('{') && row.ends_with('}'));
    }

    #[test]
    fn metrics_row_carries_the_source_text_only_when_translation_ran() {
        use crate::gate::TranscriptVerdict;
        let evidence = Evidence {
            speech_frames: 100,
            total_frames: 200,
            rms: 0.02,
        };
        let row = format_metrics_row(
            "utt_000003",
            16_000,
            &evidence,
            TranscriptVerdict::Emit,
            "Good morning",
            250.0,
            Some("Dzień dobry"),
        );
        assert!(row.contains("\"text\":\"Good morning\""), "{row}");
        assert!(row.contains("\"src_text\":\"Dzień dobry\""), "{row}");
        // `text` is the translation and comes first, so `src_text` can never be
        // mistaken for it by a reader or a `text` field lookup.
        assert!(
            row.find("\"text\":").unwrap() < row.find("\"src_text\":").unwrap(),
            "{row}"
        );
        assert!(row.ends_with("\"src_text\":\"Dzień dobry\"}"), "{row}");
        // The source is escaped like every other string.
        let row = format_metrics_row(
            "utt_000004",
            16_000,
            &evidence,
            TranscriptVerdict::Emit,
            "x",
            0.0,
            Some("quote \"here\""),
        );
        assert!(row.contains("\"src_text\":\"quote \\\"here\\\"\""), "{row}");
    }

    #[test]
    fn metrics_row_escapes_text_for_json() {
        use crate::gate::TranscriptVerdict;
        let evidence = Evidence {
            speech_frames: 20,
            total_frames: 100,
            rms: 0.005,
        };
        let row = format_metrics_row(
            "utt_000002",
            16_000,
            &evidence,
            TranscriptVerdict::DropEmpty,
            "quote \"here\"\nnewline",
            0.0,
            None,
        );
        assert!(
            row.contains("\"text\":\"quote \\\"here\\\"\\nnewline\""),
            "{row}"
        );
        assert_eq!(escape_json_string("zażółć"), "zażółć");
    }

    #[test]
    fn infer_rtf_divides_infer_by_audio() {
        // 16_000 samples = 1.0 s; 250 ms inference -> rtf 0.25.
        let (sec, rtf) = infer_rtf(16_000, 250.0);
        assert!((sec - 1.0).abs() < 1e-9);
        assert!((rtf - 0.25).abs() < 1e-9);
        // Empty audio never divides by zero.
        assert_eq!(infer_rtf(0, 100.0), (0.0, 0.0));
    }
}
