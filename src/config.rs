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
/// Sample rate required by whisper + VAD.
pub(crate) const WHISPER_RATE: u32 = 16_000;
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
/// actually reaches whisper is capped separately by `TAIL_KEEP_SAMPLES`.
/// Default for `--hangover-ms` (see `DEFAULT_HANGOVER_MS`); the constant stays
/// as the code default so `UtteranceBuffer::default()` behaves identically.
pub(crate) const SILENCE_FRAMES_THRESHOLD: u32 = 80;
/// Default silence hangover (ms) before an utterance closes. Lower = faster
/// emission, more sentence splits; higher = fewer splits, slower. Short splits
/// are cheap now that the prompt context (see `prompt_context`) keeps the
/// next decode coherent. Override with `STTD_HANGOVER_MS` (200..=2000, step 10).
pub(crate) const DEFAULT_HANGOVER_MS: u64 = 800;
/// Bounds for `--hangover-ms`: below 200 ms word-final stops split
/// utterances; above 2000 ms the pipeline feels laggy and buffers grow.
pub(crate) const MIN_HANGOVER_MS: u64 = 200;
pub(crate) const MAX_HANGOVER_MS: u64 = 2000;
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
/// Whisper refuses input whose *computed mel length* is under 1000 ms (see
/// whisper.cpp `whisper_full`: `if (seek_end < seek_start + 100)`). That
/// length isn't raw sample count / 16: mel framing pads and windows the
/// signal (`log_mel_spectrogram`: frame_size=400, hop=160, stage_2_pad=200),
/// so `mel.n_len_org = 1 + (n_samples + 200 - 400) / 160`. Exactly
/// `WHISPER_RATE` (16_000) samples yields only 990 ms this way and still
/// gets rejected — confirmed via `sttd --file` on a sub-second clip. The
/// true minimum is 16_040; padded well past it for margin.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const WHISPER_MIN_SAMPLES: usize = 16_400;
/// Parakeet's encoder does 8x subsampling; a very short clip can yield too
/// few encoder frames for the TDT greedy decoder to emit anything. Padded to
/// 0.5 s for margin (Parakeet has no documented hard minimum like whisper's
/// mel-length check, so this is a conservative floor rather than a measured
/// one).
pub(crate) const PARAKEET_MIN_SAMPLES: usize = WHISPER_RATE as usize / 2;
/// Force-flush long speech even without silence (whisper is ~30 s max).
pub(crate) const MAX_UTTERANCE_SAMPLES: usize = WHISPER_RATE as usize * 30;
/// Soft-split window: past 28 s, a 100 ms+ pause already closes the utterance
/// (see `UtteranceBuffer`), so the cut lands in silence instead of riding
/// into the hard 30 s cap mid-word. Must stay below `MAX_UTTERANCE_SAMPLES`
/// with enough margin for the pause itself.
pub(crate) const EARLY_SPLIT_SAMPLES: usize = WHISPER_RATE as usize * 28;
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
pub(crate) const INTERIM_SPLIT_SAMPLES: usize = WHISPER_RATE as usize * 8;
/// Pause length that triggers the interim split above: 30 x 10 ms frames.
pub(crate) const INTERIM_SPLIT_SILENCE_FRAMES: u32 = 30;
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
/// speech gets lifted toward whisper's comfort zone. Only ever amplifies,
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
/// clicks that whisper can decode as a token.
pub(crate) const CONDITION_FADE_SAMPLES: usize = 80;
/// Dithered edge pad added around each conditioned utterance before inference
/// (see `asr::pad_edges_with_dither`): 250 ms per side at 16 kHz, mirroring
/// Murmure's `INFERENCE_SILENCE_PAD_SECS`. Gives the encoder context past the
/// first/last word so edge tokens are not clipped. Content is quiet dither
/// (`+-1e-4`, same level as `pad_with_dither`), not digital zero — long
/// true-zero runs are a hallucination trigger for the mel front-end.
pub(crate) const INFERENCE_PAD_SAMPLES: usize = 4_000;
/// How many newest utterance pairs (`utt_NNNNNN.wav` + `.txt`) `--dump-dir`
/// keeps: each handled utterance prunes older pairs past this count, so a
/// long session cannot grow the directory without bound. Worst case ~100 MB
/// (100 × 30 s × 32 KB/s); typically far less.
pub(crate) const DUMP_KEEP_FILES: usize = 100;
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
/// Minimum share of an utterance's transmitted frames that must have been
/// VAD-flagged as speech (speech frames / post-trim transmitted frames, see
/// `UtteranceBuffer::close`). Catches utterances that clear
/// `MIN_SPEECH_FRAMES` on scattered VAD flicker yet are mostly silence by the
/// time they would reach the model — the shape a boilerplate hallucination is
/// made of. Density is judged post-trim (what whisper would actually receive,
/// i.e. up to `samples_at_last_speech + TAIL_KEEP_SAMPLES`), so a short real
/// word with a full silence hangover still passes; utterances longer than
/// `DENSITY_LONG_EXEMPT_SAMPLES` skip the gate since genuine pauses lower
/// their density. `0` disables the gate. Override with `STTD_MIN_DENSITY`.
pub(crate) const DEFAULT_MIN_DENSITY: f32 = 0.25;
/// Transmitted audio longer than this skips the speech-density gate: past ~3 s real
/// pauses and turn-taking drive density down without implying silence, and
/// the downstream transcript gates (`confidence_floor`, `is_hallucination`,
/// `is_repetition`) still apply.
pub(crate) const DENSITY_LONG_EXEMPT_SAMPLES: usize = WHISPER_RATE as usize * 3;
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
/// `speech_frames` (see `Evidence`) required to count as "strong" acoustic
/// evidence when paired with `STRONG_RMS_FACTOR` energy: 200 ms (20 frames),
/// deliberately equal to `MIN_SPEECH_FRAMES` so an energetic short word
/// ("Ale", "Nie", ~200-250 ms) can still earn strong evidence — see the
/// `strong floor` change in the "Fix short word drops" commit, which aligned
/// these two constants on purpose.
///
/// Since `check_acoustic` already rejects anything below `MIN_SPEECH_FRAMES`,
/// this equality means the frame-count half of `floor_for`'s test is always
/// true by the time it runs: strong-vs-weak reduces to the RMS half alone
/// (`rms >= min_speech_rms * STRONG_RMS_FACTOR`). That's fine *as long as*
/// `rms` genuinely describes the microphone: it must be measured **before**
/// `Agc`'s shared-gain runs, or the AGC's own amplification toward
/// `--target-rms` (up to 8x, well above the strong cutoff) makes nearly
/// every utterance read as strong regardless of how quiet the original
/// speech was — see `Agc::process_pair`'s return value and
/// `UtteranceBuffer::push_frame_evidenced`, which is what actually keeps
/// this constant meaningful.
pub(crate) const STRONG_SPEECH_FRAMES: u32 = 20;
/// Multiple of `min_speech_rms` required to count as "strong" acoustic
/// evidence.
pub(crate) const STRONG_RMS_FACTOR: f32 = 3.0;

/// Whether the whisper worker unloads the model while paused (see `whisper`).
/// Default false (no unloading). Override with `STTD_AUTO_UNLOAD`.
pub(crate) const DEFAULT_AUTO_UNLOAD: bool = false;

/// While paused longer than this (seconds), the whisper worker unloads the
/// model to free VRAM and reloads it on resume (see `whisper`). Only takes
/// effect if auto-unload is enabled. Default 180 s. Override with
/// `STTD_AUTO_UNLOAD_AFTER`; `0` unloads immediately on pause.
pub(crate) const DEFAULT_AUTO_UNLOAD_AFTER_SECS: u64 = 180;

/// Beam width of whisper's beam-search decoder (`beam_size` in
/// `SamplingStrategy::BeamSearch`). `1` is greedy (fastest,
/// repetition-prone); larger explores more hypotheses (slower, diminishing
/// returns). Default 5 (whisper.cpp default). Override with
/// `STTD_BEAM_SIZE`. Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_BEAM_SIZE: i32 = 5;
/// Initial decoding temperature (`set_temperature`). `0.0` is greedy
/// (deterministic, fastest); higher samples more diversely. Default 0.0
/// (whisper.cpp default). Override with `STTD_TEMPERATURE`.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Temperature increment for whisper.cpp's fallback ladder
/// (`set_temperature_inc`): on `entropy/logprob` fallback the decoder
/// re-tries at `temperature + temperature_inc`, `+ 2*inc`, ... up to 1.0.
/// `0.0` disables the ladder (single-temp decode). Default 0.2, i.e. the
/// historical `[0.0, 0.2, .., 1.0]` ladder. Override with
/// `STTD_TEMPERATURE_INC`. Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_TEMPERATURE_INC: f32 = 0.2;
/// Entropy threshold for the fallback ladder (`set_entropy_thold`,
/// whisper.cpp default 2.4): a decode with entropy above this re-tries at
/// the next ladder temperature. Higher = fewer re-tries (faster, stickier);
/// lower = more re-tries (slower, rescues repetition loops). Override with
/// `STTD_ENTROPY_THOLD`. Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_ENTROPY_THOLD: f32 = 2.4;
/// Log-probability threshold for the fallback ladder (`set_logprob_thold`,
/// whisper.cpp default -1.0): a decode with average logprob below this
/// re-tries at the next ladder temperature. Less negative (e.g. -0.5) =
/// more re-tries; `-10.0` effectively disables the logprob arm. Override
/// with `STTD_LOGPROB_THOLD`. Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_LOGPROB_THOLD: f32 = -1.0;
/// Decoder thread-count fallback (`set_n_threads` / ORT `intra_op`) when auto
/// detection fails. `0` (`--threads 0`, the CLI default) means auto =
/// physical cores at startup. Override with `STTD_THREADS`.
pub(crate) const DEFAULT_THREADS: i32 = 8;
/// Whether to load the model onto the GPU (`use_gpu`). Default false.
/// Override with `STTD_GPU=1` / `--gpu` for GPU machines.
pub(crate) const DEFAULT_USE_GPU: bool = false;
/// Whether whisper translates non-English speech to English (`set_translate`)
/// instead of transcribing it. Default false. Override with
/// `STTD_TRANSLATE=1` / `--translate`. Whisper-only (see the `whisper`
/// Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_TRANSLATE: bool = false;

/// Whisper model auto-download. The URL is derived from the filename (see
/// `ensure_available`), so `--model` as an explicit path can point at any
/// ggml model published in this repo, not just the compiled-in default.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg(feature = "whisper")]
pub(crate) const MODEL_BASE_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
#[cfg(not(feature = "whisper"))]
#[allow(dead_code)]
pub(crate) const MODEL_BASE_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
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
/// install, see `weights::ensure_available`).
pub(crate) const DEFAULT_MODELS_DIR: &str = env!("STTD_MODELS_DIR");
pub(crate) const DEFAULT_MODEL_SPEC: &str = "parakeet";

/// All `--model` symbolic specs (Polish-relevant, no `.en` variants),
/// with their `ggml-*.bin` filename and upstream download size.
/// `large-v1` has no quantized build upstream; `tiny`/`base`/`small` use
/// `q5_1` while `medium`/`large-v2`/`large-v3`/`large-v3-turbo` use `q5_0`.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg(feature = "whisper")]
pub(crate) const MODEL_CATALOG: &[(&str, &str, &str)] = &[
    ("tiny", "ggml-tiny.bin", "75 MiB"),
    ("tiny-q5_1", "ggml-tiny-q5_1.bin", "31 MiB"),
    ("tiny-q8_0", "ggml-tiny-q8_0.bin", "42 MiB"),
    ("base", "ggml-base.bin", "142 MiB"),
    ("base-q5_1", "ggml-base-q5_1.bin", "57 MiB"),
    ("base-q8_0", "ggml-base-q8_0.bin", "78 MiB"),
    ("small", "ggml-small.bin", "466 MiB"),
    ("small-q5_1", "ggml-small-q5_1.bin", "181 MiB"),
    ("small-q8_0", "ggml-small-q8_0.bin", "252 MiB"),
    ("medium", "ggml-medium.bin", "1.5 GiB"),
    ("medium-q5_0", "ggml-medium-q5_0.bin", "514 MiB"),
    ("medium-q8_0", "ggml-medium-q8_0.bin", "785 MiB"),
    ("large-v1", "ggml-large-v1.bin", "2.9 GiB"),
    ("large-v2", "ggml-large-v2.bin", "2.9 GiB"),
    ("large-v2-q5_0", "ggml-large-v2-q5_0.bin", "1.1 GiB"),
    ("large-v2-q8_0", "ggml-large-v2-q8_0.bin", "1.5 GiB"),
    ("large-v3", "ggml-large-v3.bin", "2.9 GiB"),
    ("large-v3-q5_0", "ggml-large-v3-q5_0.bin", "1.1 GiB"),
    ("large-v3-turbo", "ggml-large-v3-turbo.bin", "1.5 GiB"),
    (
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        "547 MiB",
    ),
    (
        "large-v3-turbo-q8_0",
        "ggml-large-v3-turbo-q8_0.bin",
        "834 MiB",
    ),
];

/// Filename for a catalog `spec` (e.g. `base-q5_1` ->
/// `ggml-base-q5_1.bin`), or `None` when the spec is unknown.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg(feature = "whisper")]
pub(crate) fn model_file_for_spec(spec: &str) -> Option<&'static str> {
    MODEL_CATALOG
        .iter()
        .find(|(s, _, _)| *s == spec)
        .map(|(_, f, _)| *f)
}

/// Comma-separated valid specs for `--model` error messages (whisper
/// catalog plus Parakeet catalog when the `whisper` feature is enabled,
/// Parakeet catalog alone otherwise).
pub(crate) fn valid_model_specs() -> String {
    #[cfg(feature = "whisper")]
    {
        MODEL_CATALOG
            .iter()
            .map(|(s, _, _)| *s)
            .chain(PARAKEET_CATALOG.iter().map(|(s, ..)| *s))
            .chain(std::iter::once("parakeet"))
            .collect::<Vec<_>>()
            .join(", ")
    }
    #[cfg(not(feature = "whisper"))]
    {
        PARAKEET_CATALOG
            .iter()
            .map(|(s, ..)| *s)
            .chain(std::iter::once("parakeet"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Parakeet-TDT-0.6B-v3 weights, published as loose ONNX files (no archive)
/// rather than a single `.bin`: the engine loads from a directory containing
/// an encoder, a decoder_joint, `vocab.txt` and `tokenizer.json`
/// (phrase-boosting; without it transcription works but boosting stays off).
/// The ONNX weights come from the istupakov export; `tokenizer.json` comes
/// from the upstream NVIDIA checkpoint (the export has none). Mel features
/// are computed in Rust (`ort_engine::frontend`), so no `nemo128.onnx` and
/// no `config.json`/`preprocessor_config.json` are needed.
pub(crate) const PARAKEET_BASE_URL: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/";
/// Upstream NVIDIA checkpoint, source of `tokenizer.json` (absent from the
/// ONNX export above).
pub(crate) const PARAKEET_TOKENIZER_URL: &str =
    "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json";

/// `(spec, dir under DEFAULT_MODELS_DIR, files to download, upstream size)`.
/// int8 is the default variant (`PARAKEET_DEFAULT_SPEC`); fp32 is offered
/// for accuracy-sensitive use. Filenames match what the engine looks for,
/// so no renaming after download.
pub(crate) const PARAKEET_CATALOG: &[(&str, &str, &[&str], &str)] = &[
    (
        "parakeet-tdt-0.6b-v3-int8",
        "parakeet-tdt-0.6b-v3-int8",
        &[
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "vocab.txt",
            "tokenizer.json",
        ],
        "640 MiB",
    ),
    (
        "parakeet-tdt-0.6b-v3-fp32",
        "parakeet-tdt-0.6b-v3-fp32",
        &[
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
            "vocab.txt",
            "tokenizer.json",
        ],
        "2.4 GiB",
    ),
];

/// Short alias for `--model parakeet`, resolving to the int8 variant (see
/// `PARAKEET_CATALOG`).
pub(crate) const PARAKEET_DEFAULT_SPEC: &str = "parakeet-tdt-0.6b-v3-int8";

/// Look up a `PARAKEET_CATALOG` entry by spec, accepting the `parakeet`
/// alias for the default (int8) variant.
pub(crate) fn parakeet_entry_for_spec(
    spec: &str,
) -> Option<&'static (&'static str, &'static str, &'static [&'static str], &'static str)> {
    let spec = if spec == "parakeet" {
        PARAKEET_DEFAULT_SPEC
    } else {
        spec
    };
    PARAKEET_CATALOG.iter().find(|(s, ..)| *s == spec)
}


/// Default `initial_prompt`: primes Polish orthography/diacritics and English
/// IT vocabulary, without a long keyword list. It used to carry many domain
/// words (Kubernetes, Docker, ...), but whisper.cpp feeds `initial_prompt`
/// into the decoder as prior context (`prompt_past`), and on low/no-signal
/// audio the model falls back on that prior instead of the (absent) acoustic
/// evidence — i.e. it would recite from the prompt itself. Override with
/// `STTD_PROMPT` to opt back into a domain vocabulary at the cost of that risk.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_PROMPT: &str =
    "Technical conversation in Polish and English. Rozmowa techniczna po polsku i angielsku.";

/// `initial_prompt` used when `--translate` is on and the user did not
/// override `--prompt`/`STTD_PROMPT` (see `resolve_prompt`): English-only,
/// because whisper.cpp feeds it as prior decoder context (`prompt_past`) and
/// a Polish prior drags translate-mode output back to Polish (or makes the
/// model recite the prompt instead of translating). This exact string gave
/// the best translation in local `--file` tests.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_TRANSLATE_PROMPT: &str = "Technical conversation in English.";

/// How many trailing chars of accepted transcripts are appended to the
/// whisper `initial_prompt` as conversational context (see `prompt_context`
/// in `whisper`). `0` disables the carryover. Override with
/// `STTD_PROMPT_CONTEXT_CHARS`. Whisper-only (see the `whisper` Cargo
/// feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const DEFAULT_PROMPT_CONTEXT_CHARS: usize = 150;
/// Hard cap for `--prompt-context-chars`: whisper.cpp's `initial_prompt`
/// is prior context, not storage — past a few hundred chars it starts
/// reciting the prompt on low-signal audio (see `DEFAULT_PROMPT`).
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) const MAX_PROMPT_CONTEXT_CHARS: usize = 2000;

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

/// Parse `--beam-size`/`STTD_BEAM_SIZE`: must be >= 1.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_beam_size(s: &str) -> Result<i32, String> {
    let v: i32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected an integer >= 1, got {s:?}"))?;
    if v >= 1 {
        Ok(v)
    } else {
        Err(format!("expected an integer >= 1, got {s:?}"))
    }
}

/// Parse `--temperature`/`STTD_TEMPERATURE`: 0.0..=2.0 (`0.0` = greedy).
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_temperature(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected a number 0.0..=2.0, got {s:?}"))?;
    if (0.0..=2.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("expected a number 0.0..=2.0, got {s:?}"))
    }
}

/// Parse `--temperature-inc`/`STTD_TEMPERATURE_INC`: 0.0..=1.0
/// (`0.0` = single-temp decode, no fallback ladder).
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_temperature_inc(s: &str) -> Result<f32, String> {
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

/// Parse `--entropy-thold`/`STTD_ENTROPY_THOLD`: 0.0..=10.0.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_entropy_thold(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected a number 0.0..=10.0, got {s:?}"))?;
    if (0.0..=10.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("expected a number 0.0..=10.0, got {s:?}"))
    }
}

/// Parse `--logprob-thold`/`STTD_LOGPROB_THOLD`: -10.0..=0.0.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_logprob_thold(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("expected a number -10.0..=0.0, got {s:?}"))?;
    if (-10.0..=0.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("expected a number -10.0..=0.0, got {s:?}"))
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

/// Parse `--prompt-context-chars`/`STTD_PROMPT_CONTEXT_CHARS`: 0 disables,
/// max `MAX_PROMPT_CONTEXT_CHARS`. Whisper-only (see the `whisper` Cargo
/// feature).
#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
pub(crate) fn parse_context_chars(s: &str) -> Result<usize, String> {
    let v: usize = s.trim().parse().map_err(|_| {
        format!(
            "expected an integer 0..={} (0 = off), got {s:?}",
            MAX_PROMPT_CONTEXT_CHARS
        )
    })?;
    if v <= MAX_PROMPT_CONTEXT_CHARS {
        Ok(v)
    } else {
        Err(format!(
            "expected an integer 0..={} (0 = off), got {s:?}",
            MAX_PROMPT_CONTEXT_CHARS
        ))
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
pub(crate) fn hangover_frames(ms: u64) -> u32 {
    (ms / 10) as u32
}

/// Upper bound for `--min-rms`/`--target-rms`: both are linear RMS on
/// `[-1, 1]` samples, and 0.5 is already deep clipping territory — anything
/// higher is certainly a typo (e.g. a dBFS value pasted in by mistake), not a
/// tuning choice. `0` stays valid (disables the corresponding gate/AGC).
pub(crate) const MAX_LINEAR_RMS: f32 = 0.5;

/// Parse a `[0.0, 1.0]` fraction: shared by `--vad-threshold`, `--min-conf`,
/// `--min-conf-strong` and `--min-density`, every one of them a probability
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
        Err(format!("expected a number 0.0..={MAX_LINEAR_RMS}, got {s:?}"))
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
        assert_eq!(parse_beam_size("1"), Ok(1));
        assert_eq!(parse_beam_size("5"), Ok(5));
        assert!(parse_beam_size("0").is_err());
        assert!(parse_beam_size("-2").is_err());
        assert!(parse_beam_size("bogus").is_err());
        assert_eq!(parse_threads("0"), Ok(0));
        assert_eq!(parse_threads("8"), Ok(8));
        assert!(parse_threads("-1").is_err());
        assert!(parse_threads("bogus").is_err());
        assert_eq!(DEFAULT_BEAM_SIZE, 5);
        assert_eq!(DEFAULT_THREADS, 8);
        assert!(!DEFAULT_USE_GPU);
    }

    #[test]
    fn parse_decoder_tunables_validate_range() {
        assert_eq!(parse_temperature("0.0"), Ok(0.0));
        assert_eq!(parse_temperature("0.2"), Ok(0.2));
        assert_eq!(parse_temperature("2.0"), Ok(2.0));
        assert!(parse_temperature("-0.1").is_err());
        assert!(parse_temperature("2.1").is_err());
        assert!(parse_temperature("bogus").is_err());
        assert_eq!(parse_temperature_inc("0.0"), Ok(0.0));
        assert_eq!(parse_temperature_inc("0.2"), Ok(0.2));
        assert!(parse_temperature_inc("-0.1").is_err());
        assert!(parse_temperature_inc("1.1").is_err());
        assert_eq!(parse_entropy_thold("2.4"), Ok(2.4));
        assert!(parse_entropy_thold("-0.1").is_err());
        assert!(parse_entropy_thold("10.1").is_err());
        assert_eq!(parse_logprob_thold("-1.0"), Ok(-1.0));
        assert!(parse_logprob_thold("0.1").is_err());
        assert!(parse_logprob_thold("-10.1").is_err());
        assert!(parse_logprob_thold("bogus").is_err());
        // Defaults preserve the historical hardcoded ladder.
        assert_eq!(DEFAULT_TEMPERATURE, 0.0);
        assert_eq!(DEFAULT_TEMPERATURE_INC, 0.2);
        assert_eq!(DEFAULT_ENTROPY_THOLD, 2.4);
        assert_eq!(DEFAULT_LOGPROB_THOLD, -1.0);
    }

    #[test]
    fn parse_context_chars_accepts_zero_to_max() {
        assert_eq!(parse_context_chars("0"), Ok(0));
        assert_eq!(parse_context_chars("150"), Ok(150));
        assert_eq!(parse_context_chars("2000"), Ok(MAX_PROMPT_CONTEXT_CHARS));
        assert_eq!(DEFAULT_PROMPT_CONTEXT_CHARS, 150);
        assert!(parse_context_chars("2001").is_err());
        assert!(parse_context_chars("-1").is_err());
        assert!(parse_context_chars("bogus").is_err());
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
        assert_eq!(INTERIM_SPLIT_SAMPLES, WHISPER_RATE as usize * 8);
        assert_eq!(INTERIM_SPLIT_SILENCE_FRAMES, 30);
        assert_eq!(INTERIM_SPLIT_SILENCE_FRAMES * 10, 300);
        assert!(INTERIM_SPLIT_SAMPLES < EARLY_SPLIT_SAMPLES);
        assert!(INTERIM_SPLIT_SAMPLES < MAX_UTTERANCE_SAMPLES);
        assert_eq!(RTF_WARN_RTF, 1.0);
    }

    #[cfg(feature = "whisper")]
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

    #[cfg(feature = "whisper")]
    #[test]
    fn model_catalog_has_default_and_no_english_variants() {
        assert_eq!(DEFAULT_MODELS_DIR, "models");
        assert_eq!(MODEL_CATALOG.len(), 21);
        for (spec, file, _) in MODEL_CATALOG {
            assert!(
                !spec.contains(".en"),
                "English-only variant must not be listed: {spec}"
            );
            assert_eq!(*file, format!("ggml-{spec}.bin"));
            assert_eq!(model_file_for_spec(spec), Some(*file));
        }
        assert_eq!(
            model_file_for_spec("large-v3-turbo-q8_0"),
            Some("ggml-large-v3-turbo-q8_0.bin")
        );
        assert!(model_file_for_spec("tiny-q5_0").is_none());
        assert!(model_file_for_spec("medium-q5_1").is_none());
        assert!(model_file_for_spec("bogus").is_none());
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
        for (spec, dir, files, _size) in PARAKEET_CATALOG {
            assert_eq!(*dir, *spec, "dir must equal spec for {spec}");
            assert!(!files.is_empty(), "{spec} must list at least one file");
            assert!(
                files.contains(&"vocab.txt"),
                "{spec} must download vocab.txt"
            );
            assert_eq!(parakeet_entry_for_spec(spec).map(|e| e.0), Some(*spec));
        }
    }

    #[test]
    fn parakeet_alias_resolves_to_int8_default() {
        assert_eq!(PARAKEET_DEFAULT_SPEC, "parakeet-tdt-0.6b-v3-int8");
        let default_entry =
            parakeet_entry_for_spec(PARAKEET_DEFAULT_SPEC).expect("default spec must resolve");
        let alias_entry = parakeet_entry_for_spec("parakeet").expect("alias must resolve");
        assert_eq!(default_entry.0, alias_entry.0);
        assert!(default_entry
            .2
            .iter()
            .any(|f| f.contains("int8")), "default variant must be int8");
        assert!(parakeet_entry_for_spec("parakeet-bogus").is_none());
    }

    #[test]
    fn valid_model_specs_lists_both_catalogs() {
        let specs = valid_model_specs();
        assert!(specs.contains(DEFAULT_MODEL_SPEC));
        assert!(specs.contains(PARAKEET_DEFAULT_SPEC));
        assert!(specs.contains("parakeet"));
        #[cfg(feature = "whisper")]
        assert!(specs.contains("large-v3-turbo-q8_0"));
        #[cfg(not(feature = "whisper"))]
        assert!(!specs.contains("large-v3-turbo-q8_0"));
    }
}
