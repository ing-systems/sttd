// ---------------------------------------------------------------------------
// Gates: deciding not to emit, in one place.
// ---------------------------------------------------------------------------
//
// A single domain idea — `gate` — used to live in two modules with a thin
// `Evidence` carrier across the channel seam: `UtteranceBuffer::close` owned
// the acoustic gates, `whisper_loop` owned the transcript gates. To answer
// "why was this utterance dropped" you had to read both plus the stderr logs.
// The pure helpers were tested in isolation; composition through call order
// was not.
//
// This module owns both halves with leverage over thresholds + ordering, so
// callers pass measurements + text and get a disposition. `Utterance` stays
// data; conditioning / gain / transcription stay outside (see `whisper`).

use crate::config::{
    DENSITY_LONG_EXEMPT_SAMPLES, HALLUCINATION_FUZZ_MAX_DIST, HALLUCINATION_FUZZ_MIN_LEN,
    HALLUCINATION_PATTERNS, MIN_SPEECH_FRAMES, REPETITION_BIGRAM_COUNT, STRONG_RMS_FACTOR,
    STRONG_SPEECH_FRAMES,
};

/// How much trust an utterance's acoustics have already earned, which selects
/// the confidence floor its transcript then has to clear.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum EvidenceStrength {
    Weak,
    Strong,
}

impl std::fmt::Display for EvidenceStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Weak => "weak",
            Self::Strong => "strong",
        })
    }
}

/// Thresholds for deciding-not-to-emit, resolved once at startup from `Cli`
/// and shared by the acoustic path (`UtteranceBuffer::close` via
/// `FrameProcessor`) and the transcript path (`whisper_loop` via
/// `WhisperOptions`). Previously threaded as separate `f32`s through both.
#[derive(Debug, Clone)]
pub(crate) struct GateConfig {
    pub(crate) min_speech_rms: f32,
    pub(crate) min_density: f32,
    pub(crate) min_confidence: f32,
    pub(crate) min_confidence_strong: f32,
}

impl GateConfig {
    pub(crate) fn new(
        min_speech_rms: f32,
        min_density: f32,
        min_confidence: f32,
        min_confidence_strong: f32,
    ) -> Self {
        Self {
            min_speech_rms,
            min_density,
            min_confidence,
            min_confidence_strong,
        }
    }

    /// Acoustic gate: takes measurements `UtteranceBuffer` already computed
    /// (post-trim density, speech-only RMS, transmitted length for the
    /// long-utterance exemption). Returns the drop reason or `Ok(())`.
    pub(crate) fn check_acoustic(
        &self,
        speech_frames: u32,
        density: f32,
        rms: f32,
        transmitted_len: usize,
        total_frames: u32,
        buffered_samples: usize,
    ) -> Result<(), AcousticDrop> {
        if speech_frames < MIN_SPEECH_FRAMES {
            return Err(AcousticDrop::TooFewFrames {
                speech_frames,
                samples: buffered_samples,
            });
        }
        if transmitted_len <= DENSITY_LONG_EXEMPT_SAMPLES
            && self.min_density > 0.0
            && density < self.min_density
        {
            return Err(AcousticDrop::LowDensity {
                speech_frames,
                total_frames,
                density,
                min_density: self.min_density,
            });
        }
        if rms < self.min_speech_rms {
            return Err(AcousticDrop::LowEnergy {
                speech_frames,
                rms,
                min_rms: self.min_speech_rms,
            });
        }
        Ok(())
    }

    /// Confidence floor for a transcript, paired with the `EvidenceStrength`
    /// that selected it. Same rule as before: strong acoustic evidence relaxes
    /// the floor; weak keeps it strict.
    pub(crate) fn floor_for(&self, speech_frames: u32, rms: f32) -> (f32, EvidenceStrength) {
        if speech_frames >= STRONG_SPEECH_FRAMES && rms >= self.min_speech_rms * STRONG_RMS_FACTOR
        {
            (self.min_confidence_strong, EvidenceStrength::Strong)
        } else {
            (self.min_confidence, EvidenceStrength::Weak)
        }
    }

    /// Transcript gate: owns the order empty -> confidence floor ->
    /// hallucination -> repetition. Callers pass the already-trimmed text and
    /// its text-token average probability.
    pub(crate) fn decide_transcript(
        &self,
        speech_frames: u32,
        rms: f32,
        trimmed: &str,
        prob: f32,
    ) -> TranscriptVerdict {
        if trimmed.trim().is_empty() {
            return TranscriptVerdict::DropEmpty;
        }
        let (floor, strength) = self.floor_for(speech_frames, rms);
        if prob < floor {
            return TranscriptVerdict::Drop {
                kind: TranscriptDropKind::LowConfidence,
                floor,
                strength,
            };
        }
        if is_hallucination(trimmed) {
            return TranscriptVerdict::Drop {
                kind: TranscriptDropKind::Hallucination,
                floor,
                strength,
            };
        }
        if is_repetition(trimmed) {
            return TranscriptVerdict::Drop {
                kind: TranscriptDropKind::Repetition,
                floor,
                strength,
            };
        }
        TranscriptVerdict::Emit { floor, strength }
    }
}

/// Why an utterance never reached whisper.
#[derive(PartialEq, Debug)]
pub(crate) enum AcousticDrop {
    TooFewFrames { speech_frames: u32, samples: usize },
    LowDensity {
        speech_frames: u32,
        total_frames: u32,
        density: f32,
        min_density: f32,
    },
    LowEnergy {
        speech_frames: u32,
        rms: f32,
        min_rms: f32,
    },
}

impl AcousticDrop {
    /// stderr line, matching the historical `close` messages.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::TooFewFrames {
                speech_frames,
                samples,
            } => format!("[sttd] dropping utterance ({speech_frames} speech frames, {samples} samples)"),
            Self::LowDensity {
                speech_frames,
                total_frames,
                density,
                min_density,
            } => format!(
                "[sttd] dropping low-density utterance ({speech_frames} speech frames \
                 in {total_frames} frames, density={density:.3} < {min_density:.3})"
            ),
            Self::LowEnergy {
                speech_frames,
                rms,
                min_rms,
            } => format!(
                "[sttd] dropping low-energy utterance ({speech_frames} speech frames, \
                 rms={rms:.5} < {min_rms:.5})"
            ),
        }
    }
}

/// Transcript disposition for non-empty text (plus the empty case).
#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum TranscriptVerdict {
    Emit { floor: f32, strength: EvidenceStrength },
    DropEmpty,
    Drop {
        kind: TranscriptDropKind,
        floor: f32,
        strength: EvidenceStrength,
    },
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum TranscriptDropKind {
    LowConfidence,
    Hallucination,
    Repetition,
}

impl TranscriptVerdict {
    /// `<dump_dir>/<tag>.txt` body, matching historical `DROPPED(...)` strings.
    pub(crate) fn dump_reason(&self, trimmed: &str, prob: f32) -> String {
        match self {
            Self::DropEmpty => "DROPPED(empty)".to_string(),
            Self::Drop { kind, floor, strength } => match kind {
                TranscriptDropKind::LowConfidence => format!(
                    "DROPPED(low-conf avg_p={prob:.3} < {floor:.3}, {strength}): {trimmed}"
                ),
                TranscriptDropKind::Hallucination => format!("DROPPED(hallucination): {trimmed}"),
                TranscriptDropKind::Repetition => format!("DROPPED(repetition): {trimmed}"),
            },
            Self::Emit { .. } => format!("EMIT: {trimmed}"),
        }
    }
}

/// Fold a single Polish diacritic to its ASCII base letter; every other char is
/// returned unchanged.
pub(crate) fn fold_polish_diacritic(c: char) -> char {
    match c {
        'ą' => 'a',
        'ć' => 'c',
        'ę' => 'e',
        'ł' => 'l',
        'ń' => 'n',
        'ó' => 'o',
        'ś' => 's',
        'ź' | 'ż' => 'z',
        _ => c,
    }
}

/// Lowercase, fold Polish diacritics, turn punctuation into spaces and
/// collapse whitespace.
pub(crate) fn normalize_transcript(text: &str) -> String {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(fold_polish_diacritic)
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `text` is a Whisper repetition-loop decode: 6+ words with at most
/// half unique, or any word bigram repeated `REPETITION_BIGRAM_COUNT`+ times
/// (tight loops the ratio rule can miss on barely-6-word texts).
pub(crate) fn is_repetition(text: &str) -> bool {
    let normalized = normalize_transcript(text);
    let words: Vec<&str> = normalized.split(' ').filter(|w| !w.is_empty()).collect();
    if words.len() < 6 {
        return false;
    }
    let mut unique = words.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() * 2 <= words.len() {
        return true;
    }
    // Tight bigram loop: "pada deszcz" x4 needs only 8 words and can stay
    // above the ratio floor when padded with filler.
    use std::collections::HashMap;
    let mut counts: HashMap<(&str, &str), usize> = HashMap::new();
    for pair in words.windows(2) {
        let entry = counts.entry((pair[0], pair[1])).or_insert(0);
        *entry += 1;
        if *entry >= REPETITION_BIGRAM_COUNT {
            return true;
        }
    }
    false
}

/// Edit distance over chars (plain DP; inputs are short normalized lines).
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Whether `text` is a known Whisper hallucination on silence/noise: exact
/// match after normalization, or (for lines of at least
/// `HALLUCINATION_FUZZ_MIN_LEN` chars) within `HALLUCINATION_FUZZ_MAX_DIST`
/// edits of a long pattern. The fuzzy arm catches single-letter variants the
/// diacritic fold can't (doubled letters, dropped endings); whole-string
/// distance keeps real sentences containing boilerplate plus more words far
/// away from any pattern.
pub(crate) fn is_hallucination(text: &str) -> bool {
    let normalized = normalize_transcript(text);
    if normalized.is_empty() {
        return true;
    }
    if HALLUCINATION_PATTERNS.contains(&normalized.as_str()) {
        return true;
    }
    if normalized.chars().count() < HALLUCINATION_FUZZ_MIN_LEN {
        return false;
    }
    HALLUCINATION_PATTERNS
        .iter()
        .filter(|p| p.chars().count() >= HALLUCINATION_FUZZ_MIN_LEN)
        .any(|p| levenshtein(&normalized, p) <= HALLUCINATION_FUZZ_MAX_DIST)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE_STRONG, DEFAULT_MIN_DENSITY,
        DEFAULT_MIN_SPEECH_RMS, MIN_SPEECH_FRAMES,
    };

    fn default_gate() -> GateConfig {
        GateConfig::new(
            DEFAULT_MIN_SPEECH_RMS,
            DEFAULT_MIN_DENSITY,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_CONFIDENCE_STRONG,
        )
    }

    #[test]
    fn acoustic_rejects_too_few_frames() {
        let gate = default_gate();
        let drop = gate
            .check_acoustic(MIN_SPEECH_FRAMES - 1, 1.0, 1.0, 5_000, 31, 5_000)
            .expect_err("must drop");
        assert!(matches!(drop, AcousticDrop::TooFewFrames { .. }));
    }

    #[test]
    fn acoustic_rejects_scattered_flicker() {
        let gate = default_gate();
        // 25 speech frames over ~200 transmitted frames: density ~0.125 < 0.25.
        let drop = gate
            .check_acoustic(25, 0.125, 0.3, 20_000, 200, 30_000)
            .expect_err("must drop");
        assert!(matches!(drop, AcousticDrop::LowDensity { .. }));
    }

    #[test]
    fn acoustic_exempts_long_utterances_from_density() {
        let gate = GateConfig::new(DEFAULT_MIN_SPEECH_RMS, 1.0, 0.6, 0.35);
        assert!(gate
            .check_acoustic(25, 0.08, 0.3, DENSITY_LONG_EXEMPT_SAMPLES + 2_000, 300, 60_000)
            .is_ok());
        assert!(gate.check_acoustic(25, 0.08, 0.3, 20_000, 125, 20_000).is_err());
    }

    #[test]
    fn strong_evidence_relaxes_floor() {
        let gate = default_gate();
        let (floor, strength) = gate.floor_for(137, 0.01178);
        assert_eq!(strength, EvidenceStrength::Strong);
        assert_eq!(floor, DEFAULT_MIN_CONFIDENCE_STRONG);
        assert!(0.478 >= floor);
    }

    #[test]
    fn weak_evidence_keeps_strict_floor() {
        let gate = default_gate();
        let (floor, strength) = gate.floor_for(MIN_SPEECH_FRAMES, DEFAULT_MIN_SPEECH_RMS + 0.0005);
        assert_eq!(strength, EvidenceStrength::Weak);
        assert_eq!(floor, DEFAULT_MIN_CONFIDENCE);
        assert!(0.478 < floor);
    }

    #[test]
    fn transcript_composition_weak_low_conf_drops_before_patterns() {
        let gate = default_gate();
        // avg_p 0.478 < strict 0.6: low-conf, even though text is clean.
        let v = gate.decide_transcript(
            MIN_SPEECH_FRAMES,
            DEFAULT_MIN_SPEECH_RMS + 0.0005,
            "Zrób commit i wypchnij zmiany do repozytorium.",
            0.478,
        );
        assert!(matches!(
            v,
            TranscriptVerdict::Drop { kind: TranscriptDropKind::LowConfidence, .. }
        ));
    }

    #[test]
    fn transcript_composition_strong_passes_same_confidence() {
        let gate = default_gate();
        let v = gate.decide_transcript(
            137,
            0.01178,
            "Zrób commit i wypchnij zmiany do repozytorium.",
            0.478,
        );
        assert!(matches!(v, TranscriptVerdict::Emit { .. }));
    }

    #[test]
    fn transcript_composition_high_conf_boilerplate_hits_hallucination() {
        // White-noise decode at avg_p 0.89 clears either floor; pattern list
        // is the only gate. Must not be reported as low-conf.
        let gate = default_gate();
        let v = gate.decide_transcript(137, 0.01178, "Wszelkie prawa zastrzeżone.", 0.89);
        assert!(matches!(
            v,
            TranscriptVerdict::Drop { kind: TranscriptDropKind::Hallucination, .. }
        ));
    }

    #[test]
    fn transcript_composition_repetition_loop_caught() {
        let gate = default_gate();
        let v = gate.decide_transcript(
            137,
            0.01178,
            "Współpraca w Współpraca w Współpraca w Współpraca w Współpraca.",
            0.741,
        );
        assert!(matches!(
            v,
            TranscriptVerdict::Drop { kind: TranscriptDropKind::Repetition, .. }
        ));
    }

    #[test]
    fn transcript_empty_drops_without_floor() {
        let gate = default_gate();
        assert_eq!(
            gate.decide_transcript(137, 0.01178, "   ", 0.99),
            TranscriptVerdict::DropEmpty
        );
        assert_eq!(
            gate.decide_transcript(137, 0.01178, "", 0.99),
            TranscriptVerdict::DropEmpty
        );
    }

    #[test]
    fn hallucination_fuzzy_catches_letter_variants() {
        // Exact arm still works.
        assert!(is_hallucination("Wszelkie prawa zastrzeżone."));
        // One-edit variants the fold can't cover: doubled letter, dropped
        // ending, swapped ending.
        assert!(is_hallucination("Wszelkie prawa zastrzezonee."));
        assert!(is_hallucination("Wszelkie prawa zastrzezon."));
        assert!(is_hallucination("Dziekuje za uwagee."));
        assert!(is_hallucination("Zapraszam do subskrypcji kanałuu."));
    }

    #[test]
    fn hallucination_fuzzy_leaves_real_speech_alone() {
        // Boilerplate plus real words is dozens of edits from any pattern.
        assert!(!is_hallucination("Dziękuję za uwagę, do widzenia jutro."));
        assert!(!is_hallucination(
            "Zrób commit i wypchnij zmiany do repozytorium."
        ));
        // Short lines stay exact-only: "napis" is not "napisy".
        assert!(!is_hallucination("napis"));
        assert!(!is_hallucination("subtitles"));
    }

    #[test]
    fn repetition_bigram_loop_caught() {
        // Tight loop the ratio arm already sees.
        assert!(is_repetition(
            "pada deszcz pada deszcz pada deszcz pada deszcz"
        ));
        // Loop padded with distinct filler so the unique-ratio arm passes
        // (7 unique over 13 words: 7*2 > 13) while the bigram arm fires.
        assert!(is_repetition(
            "pada deszcz pada deszcz pada deszcz pada deszcz rano wczoraj jutro szybko wolno"
        ));
        // Real speech with a repeated pair stays clean.
        assert!(!is_repetition(
            "Zrób commit i wypchnij zmiany do repozytorium."
        ));
        assert!(!is_repetition("nie nie nie"));
    }
}
