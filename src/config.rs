// ---------------------------------------------------------------------------
// Constants / config
// ---------------------------------------------------------------------------

/// Sample rate required by the VAD branch's denoiser. Shared by both
/// `STTD_DENOISER` variants: `nnnoiseless` (RNNoise) needs 48 kHz, and the
/// bundled DeepFilterNet3 model (see `denoise`) happens to be trained at the
/// same rate, so switching denoisers never touches the resampling chain.
pub(crate) const DENOISE_RATE: u32 = 48_000;
/// Samples per 10 ms frame at 48 kHz (`DenoiseState::FRAME_SIZE`, and
/// DeepFilterNet3's `hop_size` — both denoisers agree on this frame size).
pub(crate) const FRAME_48K: usize = 480;
/// Sample rate required by whisper + VAD.
pub(crate) const WHISPER_RATE: u32 = 16_000;
/// Samples per 10 ms frame at 16 kHz.
pub(crate) const FRAME_16K: usize = 160;

/// Pre-roll ring buffer length: 300 ms = 30 x 10 ms frames. Prepended to
/// every utterance at onset so a quiet/unvoiced word start isn't clipped by
/// VAD's inherent onset latency.
pub(crate) const RING_FRAMES: usize = 30;

/// Silence hangover before cutting an utterance: 800 ms = 80 x 10 ms frames.
/// This only decides *when* to close an utterance; how much of that silence
/// actually reaches whisper is capped separately by `TAIL_KEEP_SAMPLES`.
pub(crate) const SILENCE_FRAMES_THRESHOLD: u32 = 80;
/// Consecutive speech frames required before an utterance is opened at all.
/// A single VAD false positive (a click, a breath, mains hum) used to be
/// enough to open a full utterance — padded out to ~1.2-1.9s by pre-roll and
/// silence hangover, which whisper then dutifully hallucinated a sentence
/// for. The onset frames themselves aren't lost: `preroll` already holds
/// them by the time the utterance opens, since it's updated every frame
/// regardless of the VAD verdict.
pub(crate) const ONSET_SPEECH_FRAMES: u32 = 3; // 30 ms
/// Total VAD-flagged speech frames (over the utterance's whole life, not
/// just at onset) required to actually send it to whisper. This is the gate
/// that replaces the old, dead `MIN_UTTERANCE_SAMPLES`: that constant
/// compared against the *buffer* length, which is dominated by pre-roll and
/// silence hangover and so was satisfied by every utterance, always.
pub(crate) const MIN_SPEECH_FRAMES: u32 = 20; // 200 ms
/// How much trailing silence (past the last speech frame) is kept when an
/// utterance is sent to whisper. `SILENCE_FRAMES_THRESHOLD` still decides
/// when to *close* the utterance; this trims how much of that hangover is
/// actually *transmitted*, so a one-word utterance isn't ~70% silence by the
/// time it reaches the model.
pub(crate) const TAIL_KEEP_SAMPLES: usize = 3_200; // 200 ms @ 16 kHz
/// Whisper refuses input whose *computed mel length* is under 1000 ms (see
/// whisper.cpp `whisper_full`: `if (seek_end < seek_start + 100)`). That
/// length isn't raw sample count / 16: mel framing pads and windows the
/// signal (`log_mel_spectrogram`: frame_size=400, hop=160, stage_2_pad=200),
/// so `mel.n_len_org = 1 + (n_samples + 200 - 400) / 160`. Exactly
/// `WHISPER_RATE` (16_000) samples yields only 990 ms this way and still
/// gets rejected — confirmed via `sttd --file` on a sub-second clip. The
/// true minimum is 16_040; padded well past it for margin.
pub(crate) const WHISPER_MIN_SAMPLES: usize = 16_400;
/// Force-flush long speech even without silence (whisper is ~30 s max).
pub(crate) const MAX_UTTERANCE_SAMPLES: usize = WHISPER_RATE as usize * 30;
/// Minimum RMS amplitude across an utterance's VAD-flagged speech frames
/// (silence padding excluded). Catches transient VAD false positives that
/// pass `ONSET_SPEECH_FRAMES`/`MIN_SPEECH_FRAMES` but carry essentially no
/// signal. ~-50 dBFS. Override with `STTD_MIN_RMS` while tuning against real
/// mic noise.
pub(crate) const DEFAULT_MIN_SPEECH_RMS: f32 = 0.003;
/// Target speech-only RMS for the pre-inference gain applied to an utterance
/// that already cleared the acoustic gates (see `apply_gain`). Only ever
/// amplifies, never attenuates — quiet-but-real speech (e.g. the
/// rms=0.01178 case that prompted this) gets a boost toward whisper's
/// comfort zone. `0` disables gain entirely. Override with `STTD_TARGET_RMS`.
pub(crate) const DEFAULT_TARGET_RMS: f32 = 0.05;
/// Ceiling on the gain `apply_gain` will apply (8x = +18 dB), so a near-
/// silent utterance that slipped past `STTD_MIN_RMS` doesn't get blown up
/// into clipped noise.
pub(crate) const MAX_GAIN: f32 = 8.0;
/// Minimum average whisper token probability for a transcript to be kept,
/// used when the utterance's *acoustic* evidence (see `Evidence`) is weak.
/// whisper.cpp declares `no_speech_thold` but never reads it (`whisper.h`:
/// `float no_speech_thold; // TODO: not implemented` — confirmed against the
/// vendored whisper.cpp 1.7.1 source, `whisper_full_default_params` is its
/// only reference in the whole file), so whisper.cpp itself never rejects a
/// low-confidence segment; this is the actual gate. Override with
/// `STTD_MIN_CONF`.
pub(crate) const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;
/// Relaxed confidence floor used when the utterance already carries strong
/// acoustic evidence of real speech (see `confidence_floor`). Whisper's
/// average token probability on genuine-but-hard speech regularly lands
/// around 0.4-0.5 — the same range a silence hallucination can land in — so
/// a single scalar threshold can't separate the two cases; VAD/RMS evidence
/// can. Override with `STTD_MIN_CONF_STRONG`.
pub(crate) const DEFAULT_MIN_CONFIDENCE_STRONG: f32 = 0.35;
/// Minimum earshot score (see `vad`) for a 16 ms window to count as speech.
/// earshot returns a score in `[0, 1]`; upstream suggests 0.5 as the general
/// voice threshold. Override with `STTD_VAD_THRESHOLD` while tuning against
/// real mic noise.
pub(crate) const DEFAULT_VAD_THRESHOLD: f32 = 0.5;
/// `speech_frames` (see `Evidence`) required to count as "strong" acoustic
/// evidence: 500 ms of VAD-flagged speech, well past `MIN_SPEECH_FRAMES`.
pub(crate) const STRONG_SPEECH_FRAMES: u32 = 50;
/// Multiple of `min_speech_rms` required to count as "strong" acoustic
/// evidence.
pub(crate) const STRONG_RMS_FACTOR: f32 = 3.0;

/// Whisper model auto-download. The URL is derived from the filename (see
/// `ensure_available`), so `WHISPER_MODEL_PATH` can point at any
/// ggml model published in this repo, not just the compiled-in default.
pub(crate) const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
pub(crate) const DEFAULT_MODEL_PATH: &str = "models/ggml-large-v3-turbo.bin";

/// Default `initial_prompt`: primes Polish orthography/diacritics only.
/// Used to carry a long list of IT vocabulary (Kubernetes, Docker, ...), but
/// whisper.cpp feeds `initial_prompt` into the decoder as prior context
/// (`prompt_past`), and on low/no-signal audio the model falls back on that
/// prior instead of the (absent) acoustic evidence — i.e. it would recite
/// from the prompt itself. Override with `STTD_PROMPT` to opt back into a
/// domain vocabulary at the cost of that risk.
pub(crate) const DEFAULT_PROMPT: &str = "Rozmowa techniczna po polsku.";

/// Transcripts that are near-exact matches for common Whisper hallucinations
/// on silence/low-energy audio (subtitle-credit boilerplate picked up from
/// training data), compared after lowercasing, stripping punctuation, and
/// folding Polish diacritics to their ASCII base letter (see
/// `fold_polish_diacritic`) — so patterns are written here without diacritics
/// too.
pub(crate) const HALLUCINATION_PATTERNS: &[&str] = &[
    "napisy stworzone przez spolecznosc amara org",
    "napisy",
    "napisy pl",
    "subtitles by",
    "dziekuje za obejrzenie",
    "dziekuje za uwage",
    "zapraszam do subskrypcji",
    "zapraszam do subskrypcji kanalu",
];

/// Read an f32 override from an env var, falling back to `default` if the
/// var is unset or fails to parse (logging a warning in the latter case, so
/// a typo'd override doesn't silently revert to the default).
pub(crate) fn f32_from_env(key: &str, default: f32) -> f32 {
    match std::env::var(key) {
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            eprintln!(
                "[sttd] warning: {key}={v:?} is not a valid number, using default {default}"
            );
            default
        }),
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_threshold_is_800ms() {
        assert_eq!(SILENCE_FRAMES_THRESHOLD, 80);
        assert_eq!(SILENCE_FRAMES_THRESHOLD * 10, 800);
    }

    #[test]
    fn whisper_min_clears_whisper_cpp_mel_length_check() {
        // Regression: exactly WHISPER_RATE (16_000) samples passes our own
        // "is it 1000ms" arithmetic but whisper.cpp's mel-framed length
        // check computes only 990ms for it and rejects anyway (see comment
        // on WHISPER_MIN_SAMPLES). Assert the real formula clears 100 frames.
        let n_len_org = 1 + (WHISPER_MIN_SAMPLES as i64 + 200 - 400) / 160;
        assert!(
            n_len_org >= 100,
            "WHISPER_MIN_SAMPLES={WHISPER_MIN_SAMPLES} only yields {n_len_org} 10ms mel frames, whisper.cpp requires >= 100"
        );
    }
}
