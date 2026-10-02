// ---------------------------------------------------------------------------
// Constants / config
// ---------------------------------------------------------------------------

/// Sample rate required by the VAD branch's denoiser (DeepFilterNet3,
/// opt-in via `STTD_DENOISER=deepfilter` with the `denoise` Cargo feature;
/// see `denoise`). Only the denoised VAD path uses this rate — the default
/// bypass path resamples straight from the hw rate to 16 kHz with no 48 kHz
/// intermediate.
#[cfg_attr(not(feature = "denoise"), allow(dead_code))]
pub(crate) const DENOISE_RATE: u32 = 48_000;
/// Samples per 10 ms frame at 48 kHz (DeepFilterNet3's `hop_size`).
#[cfg_attr(not(feature = "denoise"), allow(dead_code))]
pub(crate) const FRAME_48K: usize = 480;
/// Sample rate required by the model + VAD.
pub(crate) const SAMPLE_RATE: u32 = 16_000;
/// Samples per 10 ms frame at 16 kHz.
pub(crate) const FRAME_16K: usize = 160;

/// Pre-roll ring buffer length: 300 ms = 30 x 10 ms frames. Caps how much
/// run-up *may* be prepended at onset so a quiet/unvoiced word start isn't
/// clipped by VAD's inherent onset latency. How much of it is *actually*
/// prepended is adaptive (see `PREROLL_SAFETY_FRAMES`): a loud onset trims
/// the leading silence down to a safety margin instead of dragging the full
/// 300 ms of quiet into every utterance.
pub(crate) const RING_FRAMES: usize = 30;
/// Safety margin kept in front of the first energetic pre-roll frame when
/// trimming (see `UtteranceBuffer::preroll_trim_skip`): 100 ms of run-up
/// survives even a loud onset, covering VAD/earshot onset latency plus
/// plosive pre-voicing. The 5 ms conditioning fades land inside it.
pub(crate) const PREROLL_SAFETY_FRAMES: usize = 10;
/// Fraction of the onset burst's RMS used as the pre-roll trim threshold:
/// a pre-roll frame at least this loud (relative to the onset that opened
/// the utterance) counts as energetic and stops the trim. -20 dB below the
/// onset: silence and room noise fall below it, speech doesn't.
pub(crate) const PREROLL_TRIM_FACTOR: f32 = 0.1;
/// Absolute floor for the pre-roll trim threshold (linear RMS, ~-54 dBFS):
/// keeps the relative rule above from collapsing to zero on a very quiet
/// onset, where it would otherwise trim real (if faint) run-up.
pub(crate) const PREROLL_TRIM_FLOOR: f32 = 0.002;

/// Silence hangover before cutting an utterance: 800 ms = 80 x 10 ms frames.
/// This only decides *when* to close an utterance; how much of that silence
/// actually reaches the model is capped separately by `TAIL_KEEP_SAMPLES`.
/// Default for `--hangover-ms` (see `DEFAULT_HANGOVER_MS`); the constant stays
/// as the code default so `UtteranceBuffer::default()` behaves identically.
pub(crate) const SILENCE_FRAMES_THRESHOLD: u32 = 80;
/// Default silence hangover (ms) before an utterance closes. Lower = faster
/// emission, more sentence splits; higher = fewer splits, slower.
/// Override with `STTD_HANGOVER_MS` (200..=2000, step 10).
pub(crate) const DEFAULT_HANGOVER_MS: u64 = 800;
/// Bounds for `--hangover-ms`: below 200 ms word-final stops split
/// utterances; above 2000 ms the pipeline feels laggy and buffers grow.
pub(crate) const MIN_HANGOVER_MS: u64 = 200;
pub(crate) const MAX_HANGOVER_MS: u64 = 2000;
/// Consecutive speech frames required before an utterance is opened at all.
/// A single VAD false positive (a click, a breath, mains hum) used to be
/// enough to open a full utterance — padded out to ~1.2-1.9s by pre-roll and
/// silence hangover, which the model then dutifully hallucinated a sentence
/// for. The onset frames themselves aren't lost: `preroll` already holds
/// them by the time the utterance opens, since it's updated every frame
/// regardless of the VAD verdict.
pub(crate) const ONSET_SPEECH_FRAMES: u32 = 3; // 30 ms
/// Total VAD-flagged speech frames (over the utterance's whole life, not
/// just at onset) required to actually send it to the model. This is the gate
/// that replaces the old, dead `MIN_UTTERANCE_SAMPLES`: that constant
/// compared against the *buffer* length, which is dominated by pre-roll and
/// silence hangover and so was satisfied by every utterance, always.
pub(crate) const MIN_SPEECH_FRAMES: u32 = 20; // 200 ms
/// How much trailing silence (past the last speech frame) is kept when an
/// utterance is sent to the model. `SILENCE_FRAMES_THRESHOLD` still decides
/// when to *close* the utterance; this trims how much of that hangover is
/// actually *transmitted*, so a one-word utterance isn't ~70% silence by the
/// time it reaches the model.
pub(crate) const TAIL_KEEP_SAMPLES: usize = 3_200; // 200 ms @ 16 kHz
/// Adaptive tail bounds around `TAIL_KEEP_SAMPLES`: at least 120 ms always
/// survives (plosive release / reverb), at most 300 ms even when the
/// hangover tail stays energetic. The exact length is picked per utterance
/// by scanning the tail for energy (see `UtteranceBuffer::close`): quiet
/// tails cut at the minimum, loud tails extend toward the maximum.
pub(crate) const TAIL_MIN_SAMPLES: usize = 1_920; // 120 ms @ 16 kHz
pub(crate) const TAIL_MAX_SAMPLES: usize = 4_800; // 300 ms @ 16 kHz
/// Mic-level advisor thresholds (see `UtteranceBuffer::close`): a transmitted
/// utterance peaking below this is "mic too quiet" (~-34 dBFS), and a
/// transmitted utterance with more than this share of samples at >= 0.99 is
/// "mic clipping". Warn-only, never drops.
pub(crate) const MIC_LOW_PEAK: f32 = 0.02;
pub(crate) const MIC_CLIP_SAMPLE: f32 = 0.99;
pub(crate) const MIC_CLIP_RATIO: f32 = 0.005;
/// Parakeet's encoder does 8x subsampling; a very short clip can yield too
/// few encoder frames for the TDT greedy decoder to emit anything. Padded to
/// 0.5 s for margin.
pub(crate) const PARAKEET_MIN_SAMPLES: usize = SAMPLE_RATE as usize / 2;
/// Force-flush long speech even without silence (30 s max).
pub(crate) const MAX_UTTERANCE_SAMPLES: usize = SAMPLE_RATE as usize * 30;
/// Soft-split window: past 28 s, a 100 ms+ pause already closes the utterance
/// (see `UtteranceBuffer`), so the cut lands in silence instead of riding
/// into the hard 30 s cap mid-word. Must stay below `MAX_UTTERANCE_SAMPLES`
/// with enough margin for the pause itself.
pub(crate) const EARLY_SPLIT_SAMPLES: usize = SAMPLE_RATE as usize * 28;
/// Pause length that triggers the early split above: 10 x 10 ms frames.
pub(crate) const EARLY_SPLIT_SILENCE_FRAMES: u32 = 10;
/// Low-energy early split: past `EARLY_SPLIT_SAMPLES`, this many consecutive
/// quiet frames (RMS below `EARLY_SPLIT_QUIET_RMS`, 50 ms) already close the
/// utterance even when the VAD still says speech. Catches word boundaries
/// where earshot holds the latch through a dip — the cut lands at the
/// energy minimum instead of riding into the 30 s hard cap mid-word.
pub(crate) const EARLY_SPLIT_QUIET_FRAMES: u32 = 5;
/// RMS below which a 10 ms frame counts as quiet for the low-energy early
/// split above (~-50 dBFS, same neighborhood as `DEFAULT_MIN_SPEECH_RMS`).
pub(crate) const EARLY_SPLIT_QUIET_RMS: f32 = 0.003;
/// Interim split: past 8 s of open speech, a 300 ms VAD pause already emits
/// the prefix as its own utterance instead of waiting for the full
/// `--hangover-ms`. The speaker's continuation starts a fresh utterance
/// (pre-roll spans the boundary), so long monologues stream out in pieces
/// while the model works through the backlog instead of one 30 s block.
pub(crate) const INTERIM_SPLIT_SAMPLES: usize = SAMPLE_RATE as usize * 8;
/// Pause length that triggers the interim split above: 30 x 10 ms frames.
pub(crate) const INTERIM_SPLIT_SILENCE_FRAMES: u32 = 30;
/// Streaming: audio an open utterance must grow by before the next partial
/// transcript is cut from it (1 s). The ASR worker keeps only the newest
/// partial, so a model slower than this simply skips steps.
pub(crate) const STREAM_STEP_SAMPLES: usize = SAMPLE_RATE as usize;
/// Streaming: audio an open utterance needs before its first partial (1 s).
/// Less than this is a word fragment the model cannot read.
pub(crate) const STREAM_MIN_SAMPLES: usize = SAMPLE_RATE as usize;
/// Inference seconds per audio second above which `transcribe_and_emit`
/// warns: the model is slower than realtime and the backlog (plus shedding)
/// will grow. See the `--metrics` `rtf` column for per-utterance data.
pub(crate) const RTF_WARN_RTF: f64 = 1.0;
/// Minimum RMS amplitude across an utterance's VAD-flagged speech frames
/// (silence padding excluded). Catches transient VAD false positives that
/// pass `ONSET_SPEECH_FRAMES`/`MIN_SPEECH_FRAMES` but carry essentially no
/// signal. ~-50 dBFS. Override with `STTD_MIN_RMS` while tuning against real
/// mic noise.
pub(crate) const DEFAULT_MIN_SPEECH_RMS: f32 = 0.003;
/// Target speech-only RMS for the frame-level `Agc` (see `agc`): quiet-but-real
/// speech gets lifted toward the model's comfort zone. Only ever amplifies,
/// never attenuates; frozen on silence. `0` disables gain entirely. Override
/// with `STTD_TARGET_RMS`.
pub(crate) const DEFAULT_TARGET_RMS: f32 = 0.04;
/// Ceiling on the gain `Agc` will apply (8x = +18 dB), so a near-silent frame
/// that slips past the activation threshold doesn't get blown into noise.
pub(crate) const MAX_GAIN: f32 = 8.0;
/// AGC attack: how fast gain falls for loud sound (anti-clip). Per-sample
/// envelope coefficient derived from this at 16 kHz.
pub(crate) const AGC_ATTACK_MS: f32 = 10.0;
/// AGC release: how slowly gain rises for quiet sound (anti-pumping, ~10 dB/s
/// compliant with ITU-T G.169).
pub(crate) const AGC_RELEASE_MS: f32 = 200.0;
/// Linear fade applied to each end of an utterance by `condition_utterance`:
/// 5 ms at 16 kHz. Lands in pre-roll/tail silence, killing trim/resampler
/// clicks that the model can decode as a token.
pub(crate) const CONDITION_FADE_SAMPLES: usize = 80;
/// Dithered trailing pad added after each conditioned utterance before inference
/// (see `asr::pad_trailing_with_dither`): 250 ms at 16 kHz, mirroring
/// Murmure's `INFERENCE_SILENCE_PAD_SECS`. Gives the encoder context past the
/// last word so final tokens are not clipped. Content is quiet dither
/// (`+-1e-4`, same level as `pad_with_dither`), not digital zero — long
/// true-zero runs are a hallucination trigger for the mel front-end.
pub(crate) const INFERENCE_PAD_SAMPLES: usize = 4_000;
/// How many newest utterance pairs (`utt_NNNNNN.wav` + `.txt`) `--dump-dir`
/// keeps: each handled utterance prunes older pairs past this count, so a
/// long session cannot grow the directory without bound. Worst case ~100 MB
/// (100 × 30 s × 32 KB/s); typically far less.
pub(crate) const DUMP_KEEP_FILES: usize = 100;
/// Minimum share of an utterance's transmitted frames that must have been
/// VAD-flagged as speech (speech frames / post-trim transmitted frames, see
/// `UtteranceBuffer::close`). Catches utterances that clear
/// `MIN_SPEECH_FRAMES` on scattered VAD flicker yet are mostly silence by the
/// time they would reach the model — the shape a boilerplate hallucination is
/// made of. Density is judged post-trim (what the model would actually receive,
/// i.e. up to `samples_at_last_speech + TAIL_KEEP_SAMPLES`), so a short real
/// word with a full silence hangover still passes; utterances longer than
/// `DENSITY_LONG_EXEMPT_SAMPLES` skip the gate since genuine pauses lower
/// their density. `0` disables the gate. Override with `STTD_MIN_DENSITY`.
pub(crate) const DEFAULT_MIN_DENSITY: f32 = 0.25;
/// Transmitted audio longer than this skips the speech-density gate: past ~3 s real
/// pauses and turn-taking drive density down without implying silence, and
/// the downstream transcript gates (`is_hallucination`,
/// `is_repetition`) still apply.
pub(crate) const DENSITY_LONG_EXEMPT_SAMPLES: usize = SAMPLE_RATE as usize * 3;
/// Minimum earshot score (see `vad`) for a 16 ms window to count as speech.
/// earshot returns a score in `[0, 1]`; upstream suggests 0.5 as the general
/// voice threshold. Override with `STTD_VAD_THRESHOLD` while tuning against
/// real mic noise.
pub(crate) const DEFAULT_VAD_THRESHOLD: f32 = 0.5;
/// Whether the VAD uses the warm noise floor + hysteresis (see `noise`) or a
/// single fixed threshold (pre-adapt behavior). Kill-switch for the field:
/// `STTD_VAD_ADAPT=0` restores the old single-threshold verdict while keeping
/// the pause-state reset fix.
pub(crate) const DEFAULT_VAD_ADAPT: bool = true;
/// Reference multiple of `min_speech_rms` for energy-weighted density
/// (see `utterance::energy_weighted_density`): frames at this loudness count
/// fully, quieter frames fractionally. `rms` must be measured **before**
/// `Agc`'s shared-gain runs — see `UtteranceBuffer::push_frame_evidenced`.
pub(crate) const STRONG_RMS_FACTOR: f32 = 3.0;

/// Whether the ASR worker unloads the model while paused. Default false
/// (no unloading). Override with `STTD_AUTO_UNLOAD`.
pub(crate) const DEFAULT_AUTO_UNLOAD: bool = false;

/// While paused longer than this (seconds), the ASR worker unloads the
/// model to free VRAM and reloads it on resume. Only takes
/// effect if auto-unload is enabled. Default 180 s. Override with
/// `STTD_AUTO_UNLOAD_AFTER`; `0` unloads immediately on pause.
pub(crate) const DEFAULT_AUTO_UNLOAD_AFTER_SECS: u64 = 180;

/// Decoder thread-count fallback (ORT `intra_op`) when auto
/// detection fails. `0` (`--threads 0`, the CLI default) means auto =
/// physical cores at startup. Override with `STTD_THREADS`.
pub(crate) const DEFAULT_THREADS: i32 = 8;
/// Whether to load the model onto the GPU (`use_gpu`). Default false.
/// Override with `STTD_GPU=1` / `--gpu` for GPU machines.
pub(crate) const DEFAULT_USE_GPU: bool = false;

/// Directory every catalog spec resolves into — one fixed value per binary,
/// chosen at **compile time** from `STTD_MODELS_DIR` (see `build.rs`), default
/// `models` (CWD-relative, the dev checkout). The Arch package builds with
/// `STTD_MODELS_DIR=/usr/share/sttd/models` (see `packaging/PKGBUILD`), which
/// is also where it installs the bundled model.
///
/// Never probed and no fallback list: resolution is a pure join, so it can't
/// flip depending on the working directory. Setting the variable when *running*
/// `sttd` does nothing — an explicit `--model <path>` is the escape hatch, and
/// a missing asset is downloaded into this directory (root-owned in a packaged
/// install, see `model_store::ensure_available`).
pub(crate) const DEFAULT_MODELS_DIR: &str = env!("STTD_MODELS_DIR");
pub(crate) const DEFAULT_MODEL_SPEC: &str = "parakeet";

/// Comma-separated valid specs for `--model` error messages (both catalogs,
/// Parakeet first).
pub(crate) fn valid_model_specs() -> String {
    PARAKEET_CATALOG
        .iter()
        .map(|(s, ..)| *s)
        .chain(std::iter::once("parakeet"))
        .chain(WHISPER_CATALOG.iter().map(|(s, ..)| *s))
        .chain(std::iter::once("whisper"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parakeet-TDT-0.6B-v3 weights, published as loose ONNX files (no archive)
/// rather than a single `.bin`: the engine loads from a directory containing
/// an encoder, a decoder_joint, `vocab.txt` and `tokenizer.json`
/// (phrase-boosting; without it transcription works but boosting stays off).
/// Mel features are computed in Rust (`parakeet_core::frontend`), so no
/// `nemo128.onnx` and no `config.json`/`preprocessor_config.json` are needed.
///
/// Two exports exist, each in its own Hugging Face repository:
/// `parakeet-ultra-int8` (the default) and `parakeet-redux-int8`, Moondream's
/// 1.58-bit ternary post-train of `parakeet-tdt-0.6b-v3`. Both keep the base
/// architecture and tokenizer (128 mel features, subsampling 8, a `vocab.txt`
/// byte-identical to the base checkpoint), so the Rust frontend and the
/// `PreprocessorConfig::tdt_0_6b` recipe still apply unchanged.
///
/// `(spec, repo resolve root, weights folder)`. Each export lives in its own
/// Hugging Face repository, which is not one flat file list: the ONNX weights
/// sit in its `int8/` folder, `vocab.txt` at its root. `tokenizer.json` is
/// absent from both exports, so it comes from the upstream NVIDIA checkpoint
/// (`PARAKEET_TOKENIZER_URL`).
pub(crate) const PARAKEET_SOURCES: &[(&str, &str, &str)] = &[
    (
        "parakeet-ultra-int8",
        "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/main/",
        "int8/",
    ),
    (
        "parakeet-redux-int8",
        "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-redux-onnx/resolve/main/",
        "int8/",
    ),
];
/// Download source for a catalog spec: `(repo resolve root, weights folder)`.
pub(crate) fn parakeet_source_for_spec(spec: &str) -> Option<(&'static str, &'static str)> {
    PARAKEET_SOURCES
        .iter()
        .find(|(s, ..)| *s == spec)
        .map(|(_, root, folder)| (*root, *folder))
}

/// Upstream NVIDIA checkpoint, source of `tokenizer.json` (absent from both
/// exports, which drop the SentencePiece files).
pub(crate) const PARAKEET_TOKENIZER_URL: &str =
    "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json";

/// `(spec, dir under DEFAULT_MODELS_DIR, files to download, upstream size)`.
/// The ultra int8 export is the default variant (`PARAKEET_DEFAULT_SPEC`);
/// `parakeet-redux-int8` is the ternary post-train. Every spec must have a
/// matching row in `PARAKEET_SOURCES` (checked by the tests). Filenames match
/// what the engine looks for, so no renaming after download, even though the
/// repositories store the ONNX weights under an `int8/` folder (see
/// `model_store::parakeet_asset_url`).
pub(crate) const PARAKEET_CATALOG: &[(&str, &str, &[&str], &str)] = &[
    // `parakeet-ultra-int8` is the default (`PARAKEET_DEFAULT_SPEC`).
    (
        "parakeet-ultra-int8",
        "parakeet-ultra-int8",
        &[
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "vocab.txt",
            "tokenizer.json",
        ],
        "638 MiB",
    ),
    (
        "parakeet-redux-int8",
        "parakeet-redux-int8",
        &[
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "vocab.txt",
            "tokenizer.json",
        ],
        "640 MiB",
    ),
];

/// Short alias for `--model parakeet`, resolving to the default int8 variant,
/// `parakeet-ultra-int8` (see `PARAKEET_CATALOG`).
pub(crate) const PARAKEET_DEFAULT_SPEC: &str = "parakeet-ultra-int8";

/// Look up a `PARAKEET_CATALOG` entry by spec, accepting the `parakeet`
/// alias for the default (int8) variant.
pub(crate) fn parakeet_entry_for_spec(
    spec: &str,
) -> Option<&'static (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static str,
)> {
    let spec = if spec == "parakeet" {
        PARAKEET_DEFAULT_SPEC
    } else {
        spec
    };
    PARAKEET_CATALOG.iter().find(|(s, ..)| *s == spec)
}

// ---------------------------------------------------------------------------
// Whisper catalog
// ---------------------------------------------------------------------------

/// Whisper large-v3-turbo weights, published as loose ONNX files: the engine
/// loads from a directory containing three graphs and `tokenizer.json`.
/// Mel features are computed in Rust (`whisper_core::frontend`), so no
/// `nemo128.onnx`-style preprocessor session and no `config.json` /
/// `preprocessor_config.json` are needed.
///
/// Three graphs, not one: `decoder_model` runs the prompt (and produces the
/// cross-attention KV cache), then `decoder_with_past_model` runs every
/// following token off that cache. The single-graph `decoder_model_merged`
/// alternative is deliberately not used — it re-runs cross-attention over all
/// 1500 encoder frames per token, which costs ~40 GFLOP per token here.
///
/// The ONNX layout is not one flat file list: the graphs sit in the
/// repository's `onnx/` folder, `tokenizer.json` at its root.
///
/// `(spec, repo resolve root, weights folder)`.
pub(crate) const WHISPER_SOURCES: &[(&str, &str, &str)] = &[(
    "whisper-large-v3-turbo-int8",
    "https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/",
    "onnx/",
)];

/// Download source for a catalog spec: `(repo resolve root, weights folder)`.
pub(crate) fn whisper_source_for_spec(spec: &str) -> Option<(&'static str, &'static str)> {
    WHISPER_SOURCES
        .iter()
        .find(|(s, ..)| *s == spec)
        .map(|(_, root, folder)| (*root, *folder))
}

/// `(spec, dir under DEFAULT_MODELS_DIR, files to download, upstream size)`.
/// The directory name equals the spec, which selects the download source (see
/// `WHISPER_SOURCES`). Filenames match what the engine looks for, so no
/// renaming after download even though the repository stores the graphs under
/// its `onnx/` folder.
pub(crate) const WHISPER_CATALOG: &[(&str, &str, &[&str], &str)] = &[(
    "whisper-large-v3-turbo-int8",
    "whisper-large-v3-turbo-int8",
    &[
        "encoder_model_int8.onnx",
        "decoder_model_int8.onnx",
        "decoder_with_past_model_int8.onnx",
        "tokenizer.json",
    ],
    "1443 MiB",
)];

/// Short alias for `--model whisper`, resolving to the int8 variant,
/// `whisper-large-v3-turbo-int8` (see `WHISPER_CATALOG`).
pub(crate) const WHISPER_DEFAULT_SPEC: &str = "whisper-large-v3-turbo-int8";

/// Look up a `WHISPER_CATALOG` entry by spec, accepting the `whisper` alias.
pub(crate) fn whisper_entry_for_spec(
    spec: &str,
) -> Option<&'static (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static str,
)> {
    let spec = if spec == "whisper" {
        WHISPER_DEFAULT_SPEC
    } else {
        spec
    };
    WHISPER_CATALOG.iter().find(|(s, ..)| *s == spec)
}

/// Longest run of tokens a Whisper decode may emit before it is cut off,
/// mirroring the export's `max_target_positions` (448) from `config.json`.
/// Whisper emits `<|endoftext|>` at the end of an utterance; a run that hits
/// this cap instead is looping, and the transcript is truncated with a warning.
pub(crate) const WHISPER_MAX_NEW_TOKENS: usize = 448;

/// Transcripts that are near-exact matches for common hallucinations
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
    "wszelkie prawa",
    "wszelkie prawa zastrzezone",
    "zapraszam do subskrypcji",
    "zapraszam do subskrypcji kanalu",
];

/// Fuzzy hallucination matching: normalized transcripts of at least this
/// many chars (post-normalization) are also compared by Levenshtein distance
/// (see `is_hallucination`). Shorter ones stay exact-only — a 1-2 edit
/// allowance on "napisy" would eat real single words.
pub(crate) const HALLUCINATION_FUZZ_MIN_LEN: usize = 10;
/// Max edit distance for the fuzzy match above: covers a missed diacritic
/// fold, a doubled letter, a dropped ending — not a different sentence (a
/// real sentence with extra words is dozens of edits away).
pub(crate) const HALLUCINATION_FUZZ_MAX_DIST: usize = 2;
/// Word-bigram repetitions that mark a decode loop (see `is_repetition`):
/// any adjacent word pair occurring this many times. Catches tight loops
/// ("pada deszcz" x4) the unique-ratio rule can miss on short texts.
pub(crate) const REPETITION_BIGRAM_COUNT: usize = 4;

/// Parse a bool CLI/env value. Accepts `1/0`, `true/false`, `yes/no`,
/// `on/off` (case-insensitive, surrounding whitespace ignored). Used as the
/// clap `value_parser` for every bool flag so `--copy no`, `STTD_COPY=0`
/// and `STTD_COPY=off` all agree.
pub(crate) fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "expected 1/0, true/false, yes/no or on/off, got {s:?}"
        )),
    }
}

/// Parse `--threads`/`STTD_THREADS`: must be >= 0, where `0` means auto
/// (physical cores at startup).
pub(crate) fn parse_threads(s: &str) -> Result<i32, String> {
    let v: i32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected an integer >= 0 (0 = auto), got {s:?}"))?;
    if v >= 0 {
        Ok(v)
    } else {
        Err(format!("expected an integer >= 0 (0 = auto), got {s:?}"))
    }
}

/// Parse `--hangover-ms`/`STTD_HANGOVER_MS`: `MIN_HANGOVER_MS..=MAX_HANGOVER_MS`
/// in 10 ms steps (the pipeline's frame cadence — no false precision).
pub(crate) fn parse_hangover_ms(s: &str) -> Result<u64, String> {
    let v: u64 = s.trim().parse().map_err(|_| {
        format!("expected a multiple of 10 in {MIN_HANGOVER_MS}..={MAX_HANGOVER_MS} ms, got {s:?}")
    })?;
    if !(MIN_HANGOVER_MS..=MAX_HANGOVER_MS).contains(&v) || !v.is_multiple_of(10) {
        return Err(format!(
            "expected a multiple of 10 in {MIN_HANGOVER_MS}..={MAX_HANGOVER_MS} ms, got {s:?}"
        ));
    }
    Ok(v)
}

/// Whole 10 ms frames in `ms` of hangover. Exact: the parser only accepts
/// multiples of 10.
pub(crate) const fn hangover_frames(ms: u64) -> u32 {
    (ms / 10) as u32
}

/// Upper bound for `--min-rms`/`--target-rms`: both are linear RMS on
/// `[-1, 1]` samples, and 0.5 is already deep clipping territory — anything
/// higher is certainly a typo (e.g. a dBFS value pasted in by mistake), not a
/// tuning choice. `0` stays valid (disables the corresponding gate/AGC).
pub(crate) const MAX_LINEAR_RMS: f32 = 0.5;

/// Parse a `[0.0, 1.0]` fraction: shared by `--vad-threshold` and `--min-density`,
/// or a share of frames. Out-of-range values (negative, or `> 1.0`) used to
/// parse silently and misbehave downstream instead of failing at startup.
/// Note this alone doesn't close the `--vad-threshold` wedge: any in-range
/// base below `noise::VAD_HYSTERESIS` (0.1) still clamps
/// `NoiseTracker::exit_threshold` toward 0.0 — that's fixed separately by
/// flooring `exit_threshold` itself (see `noise.rs`).
pub(crate) fn parse_fraction(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected a number 0.0..=1.0, got {s:?}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("expected a number 0.0..=1.0, got {s:?}"))
    }
}

/// Parse a linear RMS amplitude: shared by `--min-rms` and `--target-rms`.
/// `0.0` is valid (disables the RMS gate / the AGC respectively); see
/// `MAX_LINEAR_RMS` for the upper bound.
pub(crate) fn parse_linear_rms(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected a number 0.0..={MAX_LINEAR_RMS}, got {s:?}"))?;
    if (0.0..=MAX_LINEAR_RMS).contains(&v) {
        Ok(v)
    } else {
        Err(format!(
            "expected a number 0.0..={MAX_LINEAR_RMS}, got {s:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_accepts_common_spellings() {
        for (val, expected) in [
            ("1", true),
            ("0", false),
            ("true", true),
            ("false", false),
            ("TRUE", true),
            ("False", false),
            ("yes", true),
            ("no", false),
            ("on", true),
            ("off", false),
            ("  Yes  ", true),
        ] {
            assert_eq!(parse_bool(val), Ok(expected), "val={val:?}");
        }
        assert!(parse_bool("bogus").is_err());
        assert!(parse_bool("").is_err());
    }

    #[test]
    fn parse_inference_tunables_validate_range() {
        assert_eq!(parse_threads("0"), Ok(0));
        assert_eq!(parse_threads("8"), Ok(8));
        assert!(parse_threads("-1").is_err());
        assert!(parse_threads("bogus").is_err());
        assert_eq!(DEFAULT_THREADS, 8);
        assert!(!DEFAULT_USE_GPU);
    }

    #[test]
    fn parse_fraction_accepts_0_to_1() {
        assert_eq!(parse_fraction("0"), Ok(0.0));
        assert_eq!(parse_fraction("0.5"), Ok(0.5));
        assert_eq!(parse_fraction("1"), Ok(1.0));
        assert_eq!(parse_fraction("1.0"), Ok(1.0));
        assert!(parse_fraction("-0.01").is_err());
        assert!(parse_fraction("1.01").is_err());
        assert!(parse_fraction("bogus").is_err());
    }

    #[test]
    fn parse_linear_rms_accepts_0_to_max() {
        assert_eq!(parse_linear_rms("0"), Ok(0.0));
        assert_eq!(parse_linear_rms("0.003"), Ok(0.003));
        assert_eq!(parse_linear_rms("0.5"), Ok(MAX_LINEAR_RMS));
        assert!(parse_linear_rms("-0.001").is_err());
        assert!(parse_linear_rms("0.51").is_err());
        assert!(parse_linear_rms("bogus").is_err());
    }

    #[test]
    fn parse_hangover_ms_needs_step_10_in_range() {
        assert_eq!(parse_hangover_ms("800"), Ok(800));
        assert_eq!(parse_hangover_ms("200"), Ok(MIN_HANGOVER_MS));
        assert_eq!(parse_hangover_ms("2000"), Ok(MAX_HANGOVER_MS));
        assert_eq!(parse_hangover_ms("350"), Ok(350));
        assert_eq!(DEFAULT_HANGOVER_MS, 800);
        assert_eq!(hangover_frames(800), SILENCE_FRAMES_THRESHOLD);
        assert_eq!(hangover_frames(350), 35);
        assert!(parse_hangover_ms("199").is_err());
        assert!(parse_hangover_ms("355").is_err());
        assert!(parse_hangover_ms("100").is_err());
        assert!(parse_hangover_ms("2100").is_err());
        assert!(parse_hangover_ms("bogus").is_err());
    }

    #[test]
    fn silence_threshold_is_800ms() {
        assert_eq!(SILENCE_FRAMES_THRESHOLD, 80);
        assert_eq!(SILENCE_FRAMES_THRESHOLD * 10, 800);
    }

    #[test]
    fn interim_split_is_8s_with_300ms_pause() {
        assert_eq!(INTERIM_SPLIT_SAMPLES, SAMPLE_RATE as usize * 8);
        assert_eq!(INTERIM_SPLIT_SILENCE_FRAMES, 30);
        assert_eq!(INTERIM_SPLIT_SILENCE_FRAMES * 10, 300);
        assert!(INTERIM_SPLIT_SAMPLES < EARLY_SPLIT_SAMPLES);
        assert!(INTERIM_SPLIT_SAMPLES < MAX_UTTERANCE_SAMPLES);
        assert_eq!(RTF_WARN_RTF, 1.0);
    }

    #[test]
    fn default_model_spec_is_parakeet() {
        assert_eq!(DEFAULT_MODEL_SPEC, "parakeet");
        assert_eq!(DEFAULT_MODELS_DIR, "models");
        // The `parakeet` alias must resolve via the Parakeet catalog.
        assert!(parakeet_entry_for_spec(DEFAULT_MODEL_SPEC).is_some());
    }

    #[test]
    fn parakeet_catalog_entries_are_self_consistent() {
        assert_eq!(PARAKEET_CATALOG.len(), 2);
        assert_eq!(PARAKEET_SOURCES.len(), PARAKEET_CATALOG.len());
        for (spec, dir, files, _size) in PARAKEET_CATALOG {
            assert_eq!(*dir, *spec, "dir must equal spec for {spec}");
            assert!(!files.is_empty(), "{spec} must list at least one file");
            assert!(
                files.contains(&"vocab.txt"),
                "{spec} must download vocab.txt"
            );
            assert_eq!(parakeet_entry_for_spec(spec).map(|e| e.0), Some(*spec));
            assert!(
                parakeet_source_for_spec(spec).is_some(),
                "{spec} must have a PARAKEET_SOURCES row"
            );
        }
        // Both exports keep the same asset set and tokenizer recipe.
        let files = |spec: &str| parakeet_entry_for_spec(spec).map(|e| e.2).unwrap();
        assert_eq!(files("parakeet-ultra-int8"), files("parakeet-redux-int8"));
        assert!(parakeet_source_for_spec("parakeet-bogus").is_none());
    }

    #[test]
    fn parakeet_alias_resolves_to_int8_default() {
        assert_eq!(PARAKEET_DEFAULT_SPEC, "parakeet-ultra-int8");
        let default_entry =
            parakeet_entry_for_spec(PARAKEET_DEFAULT_SPEC).expect("default spec must resolve");
        let alias_entry = parakeet_entry_for_spec("parakeet").expect("alias must resolve");
        assert_eq!(default_entry.0, alias_entry.0);
        assert!(
            default_entry.2.iter().any(|f| f.contains("int8")),
            "default variant must be int8"
        );
        assert!(parakeet_entry_for_spec("parakeet-bogus").is_none());
    }

    #[test]
    fn valid_model_specs_lists_parakeet_catalog() {
        let specs = valid_model_specs();
        assert!(specs.contains(DEFAULT_MODEL_SPEC));
        assert!(specs.contains(PARAKEET_DEFAULT_SPEC));
        assert!(specs.contains("parakeet"));
    }

    #[test]
    fn whisper_catalog_entries_are_self_consistent() {
        assert_eq!(WHISPER_CATALOG.len(), 1);
        assert_eq!(WHISPER_SOURCES.len(), WHISPER_CATALOG.len());
        for (spec, dir, files, size) in WHISPER_CATALOG {
            assert_eq!(*dir, *spec, "dir must equal spec for {spec}");
            assert!(!files.is_empty(), "{spec} must list at least one file");
            assert!(
                files.contains(&"tokenizer.json"),
                "{spec} must download the tokenizer"
            );
            assert_eq!(whisper_entry_for_spec(spec).map(|e| e.0), Some(*spec));
            assert!(
                whisper_source_for_spec(spec).is_some(),
                "{spec} must have a WHISPER_SOURCES row"
            );
            // The decode loop needs all three graphs: the prompt, the
            // cached steps, and the encoder that feeds both.
            for graph in [
                "encoder_model_int8.onnx",
                "decoder_model_int8.onnx",
                "decoder_with_past_model_int8.onnx",
            ] {
                assert!(files.contains(&graph), "{spec} must download {graph}");
            }
            // Stated size must at least cover the four files it lists.
            let stated: f64 = size
                .trim_end_matches(" MiB")
                .parse()
                .expect("size must be a number of MiB");
            assert!(stated > 1000.0, "{spec} size {size} looks wrong for int8");
        }
        assert!(whisper_source_for_spec("whisper-bogus").is_none());
    }

    #[test]
    fn whisper_alias_resolves_to_int8_default() {
        assert_eq!(WHISPER_DEFAULT_SPEC, "whisper-large-v3-turbo-int8");
        let default_entry =
            whisper_entry_for_spec(WHISPER_DEFAULT_SPEC).expect("default spec must resolve");
        let alias_entry = whisper_entry_for_spec("whisper").expect("alias must resolve");
        assert_eq!(default_entry.0, alias_entry.0);
        assert!(
            alias_entry.2.iter().any(|f| f.contains("int8")),
            "default variant must be int8"
        );
        assert!(whisper_entry_for_spec("whisper-bogus").is_none());
    }

    #[test]
    fn whisper_token_cap_matches_the_export_position_limit() {
        // `max_target_positions` in the export's config.json. If the export
        // ever moves, this constant is the one to re-check.
        assert_eq!(WHISPER_MAX_NEW_TOKENS, 448);
        assert!(WHISPER_MAX_NEW_TOKENS < MAX_UTTERANCE_SAMPLES / FRAME_16K);
    }

    #[test]
    fn valid_model_specs_lists_both_catalogs() {
        let specs = valid_model_specs();
        assert!(specs.contains(PARAKEET_DEFAULT_SPEC));
        assert!(specs.contains("parakeet"));
        assert!(specs.contains(WHISPER_DEFAULT_SPEC));
        assert!(specs.contains("whisper"));
    }
}
