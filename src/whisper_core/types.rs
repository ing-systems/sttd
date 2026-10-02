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
    /// 30 s of 16 kHz audio: the window the encoder was exported for.
    pub n_samples: usize,
    /// Mel frames the encoder expects (3000 = `n_samples / hop_length`).
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
            n_samples: 480_000,
            n_frames: 3000,
        }
    }
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

/// `begin_suppress_tokens` from the export's `config.json`: ids 220 (`'Ġ'`,
/// a lone space) and 50256 (`''`, the empty token). Applied to the first
/// generated token only, as HF does — this is what stops a decode from
/// opening on a stray space.
pub const BEGIN_SUPPRESS_TOKENS: [usize; 2] = [220, 50256];

#[derive(thiserror::Error, Debug)]
#[allow(dead_code)]
pub enum WhisperError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray17::ShapeError),
    #[error("audio frontend error: {0}")]
    Frontend(String),
    #[error("Model input not found: {0}")]
    InputNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
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
    /// `[1, heads, seq, head_dim]` per layer, self-attention.
    decoder: Vec<ArrayD<f32>>,
    /// `[1, heads, encoder_seq, head_dim]` per layer, cross-attention.
    encoder: Vec<ArrayD<f32>>,
}

impl KvCache {
    /// Capture both halves from a `decoder_model` run's outputs.
    ///
    /// Within each half, layers are stored key-then-value: index `2 * layer`
    /// is the key and `2 * layer + 1` the value, which is the order the export
    /// declares the inputs in.
    pub fn capture(outputs: &ort::session::SessionOutputs<'_>) -> Result<Self, WhisperError> {
        let take = |name: String| -> Result<ArrayD<f32>, WhisperError> {
            Ok(outputs
                .get(name.as_str())
                .ok_or(WhisperError::OutputNotFound(name.clone()))?
                // The output tensor is owned by the session's output borrow,
                // so the cache needs its own copy to outlive this call. The
                // self-attn half is 4 x 20 x seq x 64 floats; the encoder half
                // (4 x 20 x 1500 x 64) is copied once per utterance and then
                // reused as read-only input on every step.
                .try_extract_array::<f32>()?
                .to_owned())
        };
        let mut decoder = Vec::with_capacity(DECODER_LAYERS * 2);
        let mut encoder = Vec::with_capacity(DECODER_LAYERS * 2);
        for layer in 0..DECODER_LAYERS {
            decoder.push(take(format!("present.{layer}.decoder.key"))?);
            decoder.push(take(format!("present.{layer}.decoder.value"))?);
            encoder.push(take(format!("present.{layer}.encoder.key"))?);
            encoder.push(take(format!("present.{layer}.encoder.value"))?);
        }
        Ok(Self { decoder, encoder })
    }

    /// `past_key_values.N.{decoder,encoder}.{key,value}` in the order the
    /// export declares them: decoder key/value then encoder key/value, layer
    /// by layer.
    ///
    /// Returned as `(name, view)` pairs borrowing `self`, so the 16 input
    /// tensors are views of one cache with no per-step copy.
    /// The 16 `past_key_values.N.{decoder,encoder}.{key,value}` inputs, in the
    /// order the export declares them (layer by layer, decoder key/value then
    /// encoder key/value).
    ///
    /// Returned as owned name/value pairs the caller splices into a single
    /// `SessionInputs`: `ort`'s `inputs!` macro needs its names as literals, so
    /// 16 dynamically-named tensors cannot go through it.
    pub fn named_inputs<'a>(
        &'a self,
    ) -> Vec<(String, ort::session::SessionInputValue<'a>)> {
        let mut out = Vec::with_capacity(DECODER_LAYERS * 4);
        for layer in 0..DECODER_LAYERS {
            for (role, halves) in [("decoder", &self.decoder), ("encoder", &self.encoder)] {
                for (offset, half) in ["key", "value"].into_iter().enumerate() {
                    let name = format!("past_key_values.{layer}.{role}.{half}");
                    // The cache holds tensors ORT itself produced, so a
                    // failure here would be an ORT bug rather than bad caller
                    // data; `expect` keeps the hot path free of an error arm.
                    let view = ort::value::TensorRef::from_array_view(
                        halves[layer * 2 + offset].view(),
                    )
                    .expect("KV cache tensor must be convertible");
                    out.push((name, view.into()));
                }
            }
        }
        out
    }

    /// Replace the self-attention half from a `decoder_with_past_model` run,
    /// leaving the cross-attention half alone (see the type comment).
    pub fn update_decoder(
        &mut self,
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<(), WhisperError> {
        for layer in 0..DECODER_LAYERS {
            for (offset, half) in ["key", "value"].into_iter().enumerate() {
                let name = format!("present.{layer}.decoder.{half}");
                let value = outputs
                    .get(name.as_str())
                    .ok_or(WhisperError::OutputNotFound(name.clone()))?
                    .try_extract_array::<f32>()?
                    .to_owned();
                self.decoder[layer * 2 + offset] = value;
            }
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
    pub fn prompt_token(self) -> i64 {
        match self {
            Self::Polish => TOK_LANG_PL,
            Self::English => TOK_LANG_EN,
            Self::Auto => TOK_LANG_PL,
        }
    }

    /// Stable short spelling for log lines.
    pub fn as_str(self) -> &'static str {
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
    pub frontend: Frontend,
}
