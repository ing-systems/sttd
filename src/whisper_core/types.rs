// ---------------------------------------------------------------------------
// whisper_core types: session handles, mel recipe, prompt tokens, KV cache.
// ---------------------------------------------------------------------------
//
// No vendored upstream to port from (unlike `parakeet_core`, which came from
// Murmure), so this is written against the ONNX export's own names: every
// input and output string below was read out of the `onnx-community`
// graphs rather than assumed.
// ---------------------------------------------------------------------------

use ndarray17::ArrayD;
use ort::session::Session;
use tokenizers::Tokenizer;

use super::frontend::Frontend;

/// Whisper large-v3-turbo mel recipe, from the export's
/// `preprocessor_config.json`: 128 mel bins, `n_fft` 400, hop 160, 16 kHz,
/// 3000 frames (30 s).
///
/// `n_fft` 400 is Whisper's, not Parakeet's 512 — same mel scale
/// (`crate::mel`), different analysis window, so the two frontends share the
/// filterbank and nothing else.
#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub sampling_rate: usize,
    /// Mel frames of the full window the encoder was exported for: 3000, i.e.
    /// 30 s of 16 kHz audio (`n_frames * hop_length` samples). A shorter window
    /// is allowed only with the patched encoder (`dyn_encoder`).
    pub n_frames: usize,
}

impl MelConfig {
    pub const fn large_v3_turbo() -> Self {
        Self {
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            sampling_rate: 16_000,
            n_frames: 3000,
        }
    }
}

/// Shortest encoder window the dynamic encoder is given by default, in mel
/// frames (15 s). `--audio-ctx` moves this floor; see `parse_audio_ctx`.
/// Measured on short clips: a window cut close to the audio makes the
/// decode loop or return garbage under ~3 s of speech, and the shorter the
/// window the more words drift. On three espeak Polish clips (1.4 s, 6 s, 10 s)
/// a 10 s floor changed one word in two of them; 15 s, 20 s and 30 s gave the
/// 30 s transcript for all three. 15 s is the cheapest of those (see the README).
pub const DEFAULT_WINDOW_FRAMES: usize = 1500;

/// Smallest floor `--audio-ctx` accepts, in mel frames (1 s). Below this the
/// window carries too little context for the decode to be worth running.
pub const MIN_WINDOW_FRAMES: usize = 100;

/// Largest floor `--audio-ctx` accepts, in mel frames: the export's own 30 s
/// window. Nothing above it can reach the encoder, whose patched position
/// table holds 1500 rows (`dyn_encoder`).
pub const MAX_WINDOW_FRAMES: usize = 3000;

/// Parse `--audio-ctx`/`STTD_AUDIO_CTX`: an even mel-frame count in
/// `MIN_WINDOW_FRAMES..=MAX_WINDOW_FRAMES`.
///
/// Even, because the encoder's stride-2 convolution halves the window and
/// `window_frames` promises an even result. Both bounds are whole seconds (100
/// frames = 1 s at 16 kHz, `hop_length` 160), so an accepted value is always a
/// whole number of seconds.
pub fn parse_audio_ctx(s: &str) -> Result<usize, String> {
    let malformed = || {
        format!(
            "expected an even number of mel frames (100 = 1 s) in \
             {MIN_WINDOW_FRAMES}..={MAX_WINDOW_FRAMES}, got {s:?}"
        )
    };
    let v: usize = s.trim().parse().map_err(|_| malformed())?;
    if !(MIN_WINDOW_FRAMES..=MAX_WINDOW_FRAMES).contains(&v) || !v.is_multiple_of(2) {
        return Err(malformed());
    }
    Ok(v)
}

/// Encoder window for `audio_samples` of input, in mel frames.
///
/// The fixed export always takes the full window. The dynamic encoder follows
/// the audio: its length rounded up to a whole second, plus one more second of
/// margin (a window with no margin repeated itself on a 6 s clip), never below
/// `floor` (the `--audio-ctx` value) and never above the export's 30 s. The
/// result is a whole number of seconds, so it is even, as the encoder's
/// stride-2 convolution needs.
pub fn window_frames(
    audio_samples: usize,
    dynamic: bool,
    floor: usize,
    config: &MelConfig,
) -> usize {
    if !dynamic {
        return config.n_frames;
    }
    let per_second = config.sampling_rate / config.hop_length;
    let audio_frames = audio_samples.div_ceil(config.hop_length);
    let wanted = audio_frames.div_ceil(per_second) * per_second + per_second;
    wanted.clamp(floor, config.n_frames)
}

/// Decoder layers in large-v3-turbo (4, not the 32 of large-v3). Every KV
/// cache tensor is named after its layer index, and the export spells them
/// `present.N.decoder.key` out to `present.3.decoder.value`.
pub const DECODER_LAYERS: usize = 4;

/// Prompt and stop token ids, read from the export's `added_tokens.json`.
///
/// These are model-specific, not generic Whisper: an export re-tokenized with
/// a different vocabulary moves them all. The tests in `engine` assert they
/// agree with the ids the engine actually feeds, and `verify_token_ids`
/// cross-checks them against `tokenizer.json` at load time, so a swapped
/// export fails loudly instead of decoding garbage.
pub const TOK_SOT: i64 = 50258;
pub const TOK_EOT: i64 = 50257;
pub const TOK_TRANSCRIBE: i64 = 50360;
/// Suppresses timestamp tokens in the generated text. sttd has no use for
/// them (emission is text only), and leaving them in produces transcripts
/// full of `<|3.20|>` markers.
pub const TOK_NOTIMESTAMPS: i64 = 50364;
/// `<|pl|>`, the Polish decoding language.
pub const TOK_LANG_PL: i64 = 50269;
/// `<|en|>`, the English decoding language.
pub const TOK_LANG_EN: i64 = 50259;

/// `<|yue|>`, the last language token before `<|translate|>` (50359).
const TOK_LANG_YUE: i64 = 50358;

/// Inclusive id range of the 100 language tokens, `<|en|>` (50259) through
/// `<|yue|>` (50358). `--language auto` takes its argmax over exactly this
/// slice of the first position's logits.
pub const LANG_TOKEN_FIRST: usize = TOK_LANG_EN as usize;
pub const LANG_TOKEN_LAST: usize = TOK_LANG_YUE as usize;
/// How many ids that range holds, i.e. its width. `--language auto` reads
/// exactly this many logits out of the prompt's first position, so the rest of
/// the vocabulary never has to be copied out of the ONNX output.
pub const LANG_TOKEN_COUNT: usize = LANG_TOKEN_LAST - LANG_TOKEN_FIRST + 1;

/// `begin_suppress_tokens` from the export's `config.json`: ids 220 (`'Ġ'`,
/// a lone space) and 50256 (`''`, the empty token). Applied to the first
/// generated token only, as HF does — this is what stops a decode from
/// opening on a stray space.
pub const BEGIN_SUPPRESS_TOKENS: [usize; 2] = [220, 50256];

#[derive(thiserror::Error, Debug)]
pub enum WhisperError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray17::ShapeError),
    #[error("audio frontend error: {0}")]
    Frontend(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("whisper logits output is unusable: {0}")]
    Logits(String),
    #[error("model directory is missing a Whisper file: {0}")]
    MissingAsset(String),
    #[error("tokenizer unavailable: {0}")]
    Tokenizer(String),
    #[error(
        "token ids do not match tokenizer.json (expected \
         sot={sot} transcribe={transcribe} notimestamps={notimestamps}, \
         tokenizer has sot={t_sot} transcribe={t_transcribe} \
         notimestamps={t_notimestamps}): this export's vocabulary differs"
    )]
    TokenIdMismatch {
        sot: i64,
        transcribe: i64,
        notimestamps: i64,
        t_sot: i64,
        t_transcribe: i64,
        t_notimestamps: i64,
    },
}

// `ort` 2.x types builder errors as `Error<SessionBuilder>`; the session
// methods used here (`run`, tensor conversions) report plain `Error`.
// Both funnel here so `?` works on either.
impl From<ort::Error<ort::session::builder::SessionBuilder>> for WhisperError {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        Self::Ort(e.into())
    }
}

/// The output name a `past_key_values.N.*` input is returned under: the same
/// string without the `past_` prefix (`present.N.decoder.key`).
fn present_name(input_name: &str) -> String {
    input_name.replacen("past_key_values.", "present.", 1)
}

/// Which half of the cache a slot reads: `Decoder` is self-attention (grows
/// every step), `Encoder` is cross-attention (fixed after the prompt). The
/// names are the export's own, so they format straight into an input or output
/// name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Half {
    Decoder,
    Encoder,
}

impl Half {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Decoder => "decoder",
            Self::Encoder => "encoder",
        }
    }
}

/// The 16 cache slots as `(layer, half, key_or_value)`, in the order the export
/// declares them: per layer, decoder key/value then encoder key/value.
///
/// The input names and the tensors under them are both derived from this one
/// list, so a name and its tensor cannot drift apart — the failure mode being a
/// cross-attention cache fed to a self-attention slot, which the shapes allow
/// and only the transcript reveals.
fn input_slots() -> impl Iterator<Item = (usize, Half, &'static str)> {
    (0..DECODER_LAYERS).flat_map(|layer| {
        [
            (Half::Decoder, "key"),
            (Half::Decoder, "value"),
            (Half::Encoder, "key"),
            (Half::Encoder, "value"),
        ]
        .into_iter()
        .map(move |(half, kv)| (layer, half, kv))
    })
}

/// Self-attention (grows every step) and cross-attention (fixed after the
/// prompt) key/value tensors, one pair per decoder layer.
///
/// The split is the reason this type exists. The export's
/// `decoder_with_past_model` returns only `present.N.decoder.*`; its
/// cross-attention cache is an *input* it never rewrites, because it is a
/// function of the encoder output alone. So the encoder half is captured once
/// from `decoder_model` and then handed back unchanged on every step — drop it
/// and the cached steps have no encoder keys to attend to.
pub struct KvCache {
    /// The 16 cached tensors in `input_slots` order: per layer, decoder
    /// key/value (self-attention, `[1, heads, seq, head_dim]`) then encoder
    /// key/value (cross-attention, `[1, heads, encoder_seq, head_dim]`).
    ///
    /// One flat list rather than two, because the slot order is what both the
    /// input names and the tensor lookup are derived from; splitting them into
    /// a decoder and an encoder vector is what let the two orders drift.
    tensors: Vec<ArrayD<f32>>,
    /// The `past_key_values.N.{decoder,encoder}.{key,value}` input names, in
    /// the same order as `tensors`. Built once at capture: they never change,
    /// and the cached steps feed them once per token.
    input_names: Vec<String>,
}

impl KvCache {
    /// Capture both halves from a `decoder_model` run's outputs, keyed by the
    /// slots `input_slots` enumerates.
    pub fn capture(outputs: &ort::session::SessionOutputs<'_>) -> Result<Self, WhisperError> {
        let mut tensors = Vec::with_capacity(DECODER_LAYERS * 4);
        let mut input_names = Vec::with_capacity(DECODER_LAYERS * 4);
        for (layer, half, kv) in input_slots() {
            input_names.push(format!("past_key_values.{layer}.{}.{kv}", half.as_str()));
            let name = present_name(input_names.last().expect("just pushed"));
            tensors.push(
                outputs
                    .get(name.as_str())
                    .ok_or(WhisperError::OutputNotFound(name.clone()))?
                    // The output tensor is owned by the session's output borrow,
                    // so the cache needs its own copy to outlive this call. The
                    // self-attn half is 4 x 20 x seq x 64 floats; the encoder
                    // half (4 x 20 x 1500 x 64) is copied once per utterance and
                    // then reused as read-only input on every step.
                    .try_extract_array::<f32>()?
                    .to_owned(),
            );
        }
        debug_assert_eq!(tensors.len(), input_names.len());
        Ok(Self {
            tensors,
            input_names,
        })
    }

    /// The 16 `past_key_values.N.{decoder,encoder}.{key,value}` inputs, named
    /// and ordered the way the export declares them: decoder key/value then
    /// encoder key/value, layer by layer.
    ///
    /// Borrowed `(name, view)` pairs over one cache, so the 16 tensors are
    /// views with no per-step copy, and the names are the ones built once in
    /// `capture` rather than re-formatted per step. Returned as `&str` pairs
    /// because `ort` takes `Cow<str>`: `inputs!` cannot build these (it needs
    /// literal names) but a named map is the same thing `inputs!` produces,
    /// and `&str` costs no allocation.
    pub fn named_inputs<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&'a str, ort::session::SessionInputValue<'a>)> {
        self.input_names
            .iter()
            .zip(self.views())
            .map(|(name, view)| {
                (
                    name.as_str(),
                    // The cache holds tensors ORT itself produced, so a failure here
                    // would be an ORT bug rather than bad caller data; `expect`
                    // keeps the hot path free of an error arm.
                    ort::value::TensorRef::from_array_view(view)
                        .expect("KV cache tensor must be convertible")
                        .into(),
                )
            })
    }

    /// Every cached tensor as a view, in slot order.
    fn views(&self) -> impl Iterator<Item = ndarray17::ArrayViewD<'_, f32>> {
        self.tensors.iter().map(|t| t.view())
    }

    /// Replace the self-attention slots from a `decoder_with_past_model` run,
    /// leaving the cross-attention slots alone (see the type comment).
    pub fn update_decoder(
        &mut self,
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<(), WhisperError> {
        for (index, (_, half, _)) in input_slots().enumerate() {
            if half != Half::Decoder {
                continue;
            }
            let name = present_name(&self.input_names[index]);
            let value = outputs
                .get(name.as_str())
                .ok_or(WhisperError::OutputNotFound(name.clone()))?
                .try_extract_array::<f32>()?
                .to_owned();
            self.tensors[index] = value;
        }
        Ok(())
    }
}

/// Decoding language for one inference. Chosen once at startup (see
/// `cli::Language`), not per utterance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Language {
    /// Force Polish decoding (`<|pl|>` in the prompt). The default, matching
    /// the project: sttd transcribes Polish.
    #[default]
    #[value(name = "pl")]
    Polish,
    /// Force English decoding (`<|en|>` in the prompt).
    #[value(name = "en")]
    English,
    /// Detect per utterance from the prompt's first-token logits.
    #[value(name = "auto")]
    Auto,
}

impl Language {
    /// The language token that goes into the prompt. For `Auto` this is the
    /// fallback the prompt is built with; the detection pass below overrides
    /// it (see `resolve_language`), so a wrong guess costs one re-run of the
    /// prompt and nothing more.
    pub const fn prompt_token(self) -> i64 {
        match self {
            Self::Polish => TOK_LANG_PL,
            Self::English => TOK_LANG_EN,
            Self::Auto => TOK_LANG_PL,
        }
    }

    /// Stable short spelling for log lines.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Polish => "pl",
            Self::English => "en",
            Self::Auto => "auto",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub struct WhisperModel {
    pub encoder: Session,
    /// Runs the prompt and emits the initial KV cache.
    pub decoder: Session,
    /// Runs each following token off the cache.
    pub decoder_with_past: Session,
    pub tokenizer: Tokenizer,
    /// What `--language` selected. Fixed for the process's lifetime; `Auto`
    /// resolves per utterance from the prompt's first-position logits.
    pub language: Language,
    /// Owns the encoder's input buffer (see `Frontend::extract`).
    pub frontend: Frontend,
    /// Whether the encoder is the patched, length-free one (`--stream`, and the
    /// patch applied), so the window follows the audio. False = the export as
    /// shipped, always the full 30 s window.
    pub dynamic_window: bool,
    /// Floor the dynamic window never goes below (`--audio-ctx`), in mel
    /// frames. Unused when `dynamic_window` is false: the export's encoder
    /// decodes its full 30 s whatever this says.
    pub min_window_frames: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_encoder_always_gets_the_full_window() {
        let c = MelConfig::large_v3_turbo();
        for samples in [0, 1, 16_000, 480_000, 960_000] {
            assert_eq!(
                window_frames(samples, false, DEFAULT_WINDOW_FRAMES, &c),
                3000,
                "samples={samples}"
            );
        }
    }

    #[test]
    fn dynamic_window_has_a_floor_a_margin_and_a_cap() {
        let c = MelConfig::large_v3_turbo();
        let w = |samples| window_frames(samples, true, DEFAULT_WINDOW_FRAMES, &c);
        // Floor: anything short gets 15 s, even empty input.
        assert_eq!(w(0), 1500);
        assert_eq!(w(16_000), 1500);
        assert_eq!(
            w(13 * 16_000),
            1500,
            "13 s + 1 s margin is still under the floor"
        );
        // Margin: whole seconds of audio plus one more.
        assert_eq!(w(14 * 16_000), 1500);
        assert_eq!(
            w(14 * 16_000 + 1),
            1600,
            "a partial second rounds up, then +1 s"
        );
        assert_eq!(w(20 * 16_000), 2100);
        // Cap: the export's 30 s, including longer input.
        assert_eq!(w(29 * 16_000), 3000);
        assert_eq!(w(30 * 16_000), 3000);
        assert_eq!(w(45 * 16_000), 3000);
    }

    /// `--audio-ctx` moves the floor and nothing else: the margin and the 30 s
    /// cap stay where they were.
    #[test]
    fn a_configured_floor_replaces_only_the_floor() {
        let c = MelConfig::large_v3_turbo();
        let w = |samples, floor| window_frames(samples, true, floor, &c);
        // 3 s floor: short audio now gets 3 s, not 15 s.
        assert_eq!(w(0, 300), 300);
        assert_eq!(w(16_000, 300), 300);
        assert_eq!(
            w(2 * 16_000, 300),
            300,
            "2 s + 1 s margin is under the floor"
        );
        // Margin unchanged above the floor.
        assert_eq!(w(4 * 16_000, 300), 500, "4 s rounded up, then +1 s");
        // Cap unchanged, whatever the floor.
        assert_eq!(w(20 * 16_000, 300), 2100);
        assert_eq!(w(45 * 16_000, 300), 3000);
        // The highest accepted floor behaves like the fixed export's window.
        assert_eq!(w(0, MAX_WINDOW_FRAMES), 3000);
        // A floor of 1 s: the smallest the parser takes.
        assert_eq!(w(0, MIN_WINDOW_FRAMES), 100);
        assert_eq!(w(1, MIN_WINDOW_FRAMES), 200, "1 s of audio + 1 s margin");
    }

    #[test]
    fn audio_ctx_parser_takes_even_seconds_only() {
        assert_eq!(parse_audio_ctx("1500"), Ok(DEFAULT_WINDOW_FRAMES));
        assert_eq!(parse_audio_ctx(" 300 "), Ok(300));
        assert_eq!(parse_audio_ctx(&MIN_WINDOW_FRAMES.to_string()), Ok(100));
        assert_eq!(parse_audio_ctx(&MAX_WINDOW_FRAMES.to_string()), Ok(3000));
        // Below the 1 s floor, above the export's 30 s, odd, and not a number.
        for bad in ["0", "99", "3001", "1501", "1500.0", "-2", "bogus", ""] {
            assert!(parse_audio_ctx(bad).is_err(), "must reject {bad:?}");
        }
        // The bounds are whole seconds, so every accepted value is even.
        assert_eq!(MIN_WINDOW_FRAMES % 100, 0);
        assert_eq!(MAX_WINDOW_FRAMES % 100, 0);
    }

    #[test]
    fn dynamic_window_is_always_even_and_never_shrinks_with_more_audio() {
        let c = MelConfig::large_v3_turbo();
        let mut last = 0;
        for samples in (0..=31 * 16_000).step_by(1_234) {
            let frames = window_frames(samples, true, DEFAULT_WINDOW_FRAMES, &c);
            assert_eq!(frames % 2, 0, "samples={samples}");
            assert!((DEFAULT_WINDOW_FRAMES..=c.n_frames).contains(&frames));
            assert!(frames >= last, "window shrank at samples={samples}");
            // It always holds the audio (up to the cap).
            assert!(frames * c.hop_length >= samples.min(c.n_frames * c.hop_length));
            last = frames;
        }
    }

    /// The 16 cache input names, in the order `decoder_with_past_model`
    /// declares them, and the output name each is captured from.
    ///
    /// Pinned here because nothing else can: `capture` and `named_inputs` are
    /// only exercised by a loaded session, and getting the order wrong feeds
    /// each layer's cross-attention tensors to its self-attention slot — same
    /// shapes, so no error, just a transcript that quietly falls apart.
    #[test]
    fn cache_slots_are_named_and_captured_in_the_export_order() {
        let slots: Vec<(usize, Half, &str)> = input_slots().collect();
        assert_eq!(slots.len(), DECODER_LAYERS * 4);
        let names: Vec<String> = slots
            .iter()
            .map(|&(layer, half, kv)| format!("past_key_values.{layer}.{}.{kv}", half.as_str()))
            .collect();
        assert_eq!(names[0], "past_key_values.0.decoder.key");
        assert_eq!(names[1], "past_key_values.0.decoder.value");
        assert_eq!(names[2], "past_key_values.0.encoder.key");
        assert_eq!(names[3], "past_key_values.0.encoder.value");
        assert_eq!(names[15], "past_key_values.3.encoder.value");
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "no slot may repeat: {unique:?}");
        // Each input is returned under the same name without `past_`, which is
        // what `capture` reads the prompt's outputs by.
        for (name, &(layer, half, kv)) in names.iter().zip(&slots) {
            assert_eq!(
                present_name(name),
                format!("present.{layer}.{}.{kv}", half.as_str())
            );
        }
        // Self-attention is the first two slots of every layer, which is what
        // `update_decoder` walks when it replaces them.
        assert_eq!(DECODER_LAYERS * 4, slots.len());
        for layer in 0..DECODER_LAYERS {
            let base = layer * 4;
            let halves: Vec<Half> = slots[base..base + 4].iter().map(|s| s.1).collect();
            assert_eq!(
                halves,
                [Half::Decoder, Half::Decoder, Half::Encoder, Half::Encoder],
                "layer {layer}"
            );
        }
    }
}
