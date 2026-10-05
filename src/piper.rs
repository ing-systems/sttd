// ---------------------------------------------------------------------------
// Piper: a voice turned into speech, behind one seam.
// ---------------------------------------------------------------------------
//
// One `ort` session per voice, run directly — no TTS framework between sttd and
// the graph. A piper voice is a VITS export whose contract is small and fixed:
// `input` (phoneme ids), `input_lengths`, `scales` = (noise, length, noise_w)
// and, for multi-speaker models, `sid`; one float output waveform. The repo
// already runs two ONNX engines this way (`parakeet_core`, `whisper_core`), and
// the available Rust bindings are not an option: `piper-rs` and
// `dengjen-piper-rs` pin `ort` to an exact older release, `piper-tts-rs` is
// GPL-3.0, and `piper-plus` is a fork with its own phonemizer and model
// extensions. So this is the same direct-session pattern the speech engines use,
// on the same `ort` the binary already links.
//
// Phonemization goes through the `espeak-ng` *binary*, which is what piper's own
// bundled phonemizer is (`espeakbridge.c` calls
// `espeak_TextToPhonemesWithTerminator(.., espeakPHONEMES_IPA, ..)`, which is
// what `espeak-ng --ipa` does). Shelling out keeps espeak-ng's GPL-3.0 out of an
// MIT-licensed binary and off the build (no cmake, no bindgen, no libclang) —
// the same relationship sttd already has with `wl-copy` and `ydotool`.
//
// What is reproduced from upstream, because it is what makes the voice sound
// like itself:
//   - clause-per-line espeak invocation, so the punctuation that carries the
//     intonation can be re-appended per clause (`espeak --ipa` drops it),
//   - `(lang)` flag stripping, the terminator and the space after `, : ;`,
//   - the NFD code-point split (stress marks and combining diacritics are their
//     own phonemes),
//   - the id layout `[BOS, PAD, (ids…, PAD)*, EOS]`,
//   - piper's peak normalize with its `max(0.01, peak)` floor.
//
// Untested by construction where it needs hardware or the weights: the espeak
// process and the session. Everything worth testing — clause splitting, the
// phoneme string, the id layout, the normalize, the config contract — is pure
// and tested below.
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::voice_store::VoiceRef;

/// Sentence-start sentinel in the phoneme id map.
const BOS: &str = "^";
/// Sentence-end sentinel.
const EOS: &str = "$";
/// The id interleaved after every phoneme (and once after BOS), which is how
/// the model is told where one phoneme stops.
const PAD: &str = "_";

/// Upper bound on the text one synthesis runs, in chars. An utterance is capped
/// at 30 s of speech (`MAX_UTERANCE_SAMPLES`), so this only bites on an
/// anomalously long transcript — and it exists so the espeak pipe cannot fill:
/// everything written to stdin is read back on stdout, so an unbounded write
/// would deadlock against a full pipe buffer.
const MAX_SPEAK_CHARS: usize = 4_000;

/// Punctuation espeak treats as a clause boundary. The clause text is fed one
/// per line and the same character is re-appended as a phoneme afterwards,
/// because it is what carries the intonation.
const CLAUSE_TERMINATORS: [char; 6] = ['.', ',', ';', ':', '!', '?'];

/// The per-voice `.onnx.json`. Only the fields the synthesis path reads.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VoiceConfig {
    pub(crate) audio: AudioConfig,
    #[serde(default)]
    pub(crate) espeak: EspeakConfig,
    /// `espeak` for everything sttd can phonemize; `text` voices take raw
    /// text and are refused at load, because there is no espeak voice for them.
    pub(crate) phoneme_type: String,
    pub(crate) phoneme_id_map: BTreeMap<String, Vec<i64>>,
    /// Embedding width. Can exceed the highest id in the map, so it is the
    /// bound to validate ids against — never the map's length.
    pub(crate) num_symbols: usize,
    #[serde(default = "one")]
    pub(crate) num_speakers: usize,
    #[serde(default = "inference_defaults")]
    pub(crate) inference: InferenceConfig,
}

/// Upstream's own defaults, for a config that omits the `inference` block.
fn inference_defaults() -> InferenceConfig {
    InferenceConfig {
        noise_scale: default_noise_scale(),
        length_scale: default_length_scale(),
        noise_w: default_noise_w(),
    }
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioConfig {
    pub(crate) sample_rate: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct EspeakConfig {
    #[serde(default)]
    pub(crate) voice: String,
}

/// VITS inference knobs, straight from the voice's own config.
///
/// The per-field `default` fns rather than a `#[derive(Default)]`: the defaults
/// that matter are upstream's (`0.667 / 1.0 / 0.8`), not zero, and a derived
/// default would silently change the voice of every config that omits the block.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InferenceConfig {
    #[serde(default = "default_noise_scale")]
    pub(crate) noise_scale: f32,
    #[serde(default = "default_length_scale")]
    pub(crate) length_scale: f32,
    #[serde(default = "default_noise_w")]
    pub(crate) noise_w: f32,
}

fn default_noise_scale() -> f32 {
    0.667
}
fn default_length_scale() -> f32 {
    1.0
}
fn default_noise_w() -> f32 {
    0.8
}

impl InferenceConfig {
    /// The `scales` tensor, in the order the exporter wrote it and both piper
    /// runtimes pass it: **noise, length, noise_w**. Getting this order wrong
    /// does not fail — it silently turns a length scale into generator noise.
    pub(crate) fn scales(&self) -> [f32; 3] {
        [self.noise_scale, self.length_scale, self.noise_w]
    }
}

/// Graph input names, resolved once at load.
#[derive(Debug, Clone)]
struct InputNames {
    ids: String,
    lengths: String,
    scales: String,
    /// Present only for multi-speaker graphs.
    speaker: Option<String>,
}

/// A loaded voice: its config, its session, and the names to feed them by.
pub(crate) struct PiperVoice {
    key: String,
    config: VoiceConfig,
    session: Session,
    inputs: InputNames,
    /// Rate the graph's output arrives at. The graph carries no rate of its own,
    /// so this is the voice's own number and converting it to the output
    /// device's rate is the caller's job.
    sample_rate: u32,
}

impl PiperVoice {
    /// Load a voice from its two files. The directory is already guaranteed to
    /// hold them by `voice_store::prepare_voice`.
    ///
    /// Refuses, with the reason, everything this path cannot speak: a
    /// `phoneme_type` other than `espeak`, a multi-speaker graph (it needs a
    /// `sid` this flag does not offer), a map missing the sentinels, or an id
    /// past `num_symbols`. Better to say so at startup than to emit noise.
    pub(crate) fn load(voice: &VoiceRef, intra_threads: usize) -> Result<Self> {
        let config: VoiceConfig = serde_json::from_str(
            &std::fs::read_to_string(&voice.config)
                .with_context(|| format!("failed to read {}", voice.config.display()))?,
        )
        .with_context(|| format!("{} is not a piper voice config", voice.config.display()))?;

        if config.phoneme_type != "espeak" {
            anyhow::bail!(
                "voice {} has phoneme_type {:?}; sttd phonemizes with espeak-ng, so only \
                 \"espeak\" voices can be spoken",
                voice.key,
                config.phoneme_type
            );
        }
        if config.num_speakers > 1 {
            anyhow::bail!(
                "voice {} holds {} speakers; sttd speaks voice 0 only, so pick a \
                 single-speaker voice",
                voice.key,
                config.num_speakers
            );
        }
        if config.espeak.voice.trim().is_empty() {
            anyhow::bail!(
                "voice {} names no espeak voice, so its text cannot be phonemized",
                voice.key
            );
        }
        for sentinel in [BOS, EOS, PAD] {
            if !config.phoneme_id_map.contains_key(sentinel) {
                anyhow::bail!(
                    "voice {} has no {sentinel:?} in its phoneme id map (a config that is \
                     not a piper voice export?)",
                    voice.key
                );
            }
        }
        // An id past the embedding width would read another layer's weights
        // rather than fail, so it is caught here instead.
        let mut past = 0usize;
        for (phoneme, ids) in &config.phoneme_id_map {
            for id in ids {
                if *id < 0 || *id as usize >= config.num_symbols {
                    past += 1;
                    log::debug!("voice {}: phoneme {phoneme:?} has id {id}", voice.key);
                }
            }
        }
        if past > 0 {
            anyhow::bail!(
                "voice {} has {past} phoneme id(s) outside its {} symbols; the config and \
                 the graph disagree",
                voice.key,
                config.num_symbols
            );
        }

        // Prove espeak can phonemize for this voice before the first emission,
        // not on it: a missing binary or a missing voice data package is a
        // setup problem, and saying so at startup is what lets `--speak` be
        // treated as working from the first utterance on.
        espeak_available(&config.espeak.voice)?;

        // `ort`'s builder errors are its own type rather than `anyhow`-friendly,
        // so each step is mapped with a message instead of `.context()`. The
        // settings mirror `parakeet_core`'s sessions: one intra-op thread count,
        // no parallel execution, no memory pattern.
        let session = Session::builder()
            .map_err(ort_err("failed to build the piper session builder"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort_err("failed to set the piper optimization level"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(ort_err("failed to set the piper thread count"))?
            .with_inter_threads(1)
            .map_err(ort_err("failed to set the piper inter-op thread count"))?
            .with_memory_pattern(false)
            .map_err(ort_err("failed to set the piper memory pattern"))?
            .with_parallel_execution(false)
            .map_err(ort_err("failed to set the piper parallel execution"))?
            .commit_from_file(&voice.onnx)
            .with_context(|| format!("failed to load {}", voice.onnx.display()))?;
        let inputs = resolve_input_names(&session)?;

        log::info!(
            "voice: {} ({} Hz, espeak {:?})",
            voice.key,
            config.audio.sample_rate,
            config.espeak.voice
        );
        Ok(Self {
            key: voice.key.clone(),
            sample_rate: config.audio.sample_rate,
            config,
            session,
            inputs,
        })
    }

    /// The rate the synthesized waveform is in.
    pub(crate) const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Speak `text`: phonemes, ids, one graph run, normalized audio at the
    /// voice's own sample rate.
    ///
    /// Returns the `io::Error` from espeak unwrapped so the caller can match
    /// `NotFound` and disable itself with one warning, the way the clipboard and
    /// paste steps do.
    pub(crate) fn synthesize(&mut self, text: &str) -> std::io::Result<Vec<f32>> {
        let text: String = text.chars().take(MAX_SPEAK_CHARS).collect();
        let clauses = split_clauses(&text);
        if clauses.is_empty() {
            return Ok(Vec::new());
        }
        // Clause bodies go out, one per line; the terminators were taken off
        // before the call, so each line's phonemes get its terminator back here.
        let spoken = espeak_phonemes(&self.config.espeak.voice, &clauses)?;
        let phonemes: Vec<char> = spoken
            .iter()
            .zip(&clauses)
            .flat_map(|(line, clause)| clause_phonemes(line, clause.terminator))
            .collect();
        log::debug!(
            "voice {}: {} phonemes from {} clause(s)",
            self.key,
            phonemes.len(),
            clauses.len()
        );
        let ids = phonemes_to_ids(&self.config.phoneme_id_map, &phonemes);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let audio = self
            .run(&ids)
            .map_err(|e| std::io::Error::other(format!("piper inference failed: {e}")))?;
        Ok(normalize_peak(&audio))
    }

    /// One graph run: the four inputs, and one output waveform.
    fn run(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        // The names are cloned out of `self` before the mutable borrow of the
        // session, so the input list does not keep `self` borrowed.
        let ids_name = self.inputs.ids.clone();
        let lengths_name = self.inputs.lengths.clone();
        let scales_name = self.inputs.scales.clone();
        let speaker_name = self.inputs.speaker.clone();

        let id_tensor = Tensor::from_array(([1usize, ids.len()], ids.to_vec()))
            .context("failed to build the phoneme id tensor")?;
        let length_tensor = Tensor::from_array(([1usize], vec![ids.len() as i64]))
            .context("failed to build the input_lengths tensor")?;
        let scales_tensor = Tensor::from_array(([3usize], self.config.inference.scales().to_vec()))
            .context("failed to build the scales tensor")?;

        // A named map rather than `inputs!`: the speaker input is conditional,
        // and `inputs!` cannot build a list whose length is decided at runtime.
        let mut named: Vec<(&str, SessionInputValue<'_>)> = Vec::with_capacity(4);
        named.push((&ids_name, id_tensor.into()));
        named.push((&lengths_name, length_tensor.into()));
        named.push((&scales_name, scales_tensor.into()));
        if let Some(name) = &speaker_name {
            let sid = Tensor::from_array(([1usize], vec![0i64]))
                .context("failed to build the speaker tensor")?;
            named.push((name, sid.into()));
        }

        let outputs = self
            .session
            .run(named)
            .context("failed to run the piper graph")?;
        // One output: `[batch, 1, time]`. Its name is taken from the graph
        // rather than assumed, since a voice exported by a different piper
        // toolchain may call it something else.
        let mut audio: Option<Vec<f32>> = None;
        for (_name, value) in outputs {
            let extracted = value
                .try_extract_array::<f32>()
                .context("piper output is not an f32 array")?;
            audio = Some(extracted.iter().copied().collect());
            break;
        }
        audio.ok_or_else(|| anyhow::anyhow!("piper graph returned no output"))
    }
}

/// Attach a message to an `ort` builder error.
///
/// `ort::Error<T>` is parameterized by the value the failed call would have
/// returned, and each step of the builder chain fails with a different one, so
/// this is generic over `T` rather than a single closure that would only fit the
/// first `map_err`.
fn ort_err<T>(what: &'static str) -> impl Fn(ort::Error<T>) -> anyhow::Error {
    move |e| anyhow::anyhow!("{what}: {e:?}")
}

/// Match the graph's input names rather than assuming them.
///
/// A voice is any key in a 1300-entry catalog, so some were exported by a
/// different piper toolchain than the one the contract above comes from. Case
/// and the trailing `_lengths` suffix are the variations worth tolerating; an
/// unrecognisable graph is reported with the names it does have, which is the
/// one error message that makes such a voice fixable.
fn resolve_input_names(session: &Session) -> Result<InputNames> {
    let names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_ascii_lowercase())
        .collect();
    // Separators are ignored (`input_lengths` == `inputlengths` == `Input-Lengths`),
    // but there is deliberately **no prefix fallback**: one would let
    // `input_lengths` satisfy a lookup for `input`, which feeds the id tensor
    // the length tensor's slot and fails deep inside ORT instead of here.
    let find = |want: &str| -> Option<String> {
        let squashed = squash(want);
        names
            .iter()
            .find(|n| n.as_str() == want || squash(n) == squashed)
            .cloned()
    };
    let ids = find("input").ok_or_else(|| missing(&names, "input"))?;
    let lengths = find("input_lengths")
        .or_else(|| find("lengths"))
        .ok_or_else(|| missing(&names, "input_lengths"))?;
    let scales = find("scales").ok_or_else(|| missing(&names, "scales"))?;
    // Only a multi-speaker graph declares it; `sid` alone would be too loose a
    // match against an unrelated input name.
    let speaker = names
        .iter()
        .find(|n| *n == "sid" || *n == "speaker_id")
        .cloned();
    Ok(InputNames {
        ids,
        lengths,
        scales,
        speaker,
    })
}

/// A graph input name with its separators removed, so `input_lengths`,
/// `Input-Lengths` and `inputlengths` all compare equal.
fn squash(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '_' | '-' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

fn missing(names: &[String], want: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "piper graph has no {want:?} input; it declares {}",
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    )
}

/// Split text into clause bodies, terminators excluded.
///
/// One clause per line is what makes the espeak round trip unambiguous, so this
/// cuts at exactly the characters espeak itself cuts at. Whitespace inside a
/// clause is collapsed to single spaces because a line break would end it.
pub(crate) fn split_clauses(text: &str) -> Vec<Clause> {
    let mut clauses: Vec<Clause> = Vec::new();
    let mut current = String::new();
    let mut push = |body: String, terminator: Option<char>| {
        let body = body.trim().to_string();
        if !body.is_empty() {
            clauses.push(Clause { body, terminator });
        }
    };
    for ch in text.chars() {
        if ch.is_whitespace() {
            // A break cannot cross a line: one clause is one espeak line.
            if !current.is_empty() && !current.ends_with(' ') {
                current.push(' ');
            }
            continue;
        }
        if CLAUSE_TERMINATORS.contains(&ch) {
            push(std::mem::take(&mut current), Some(ch));
            continue;
        }
        current.push(ch);
    }
    push(current, None);
    clauses
}

/// One clause: the text handed to espeak, and the punctuation that ended it.
///
/// The terminator is carried alongside the body rather than left in it,
/// because `espeak-ng --ipa` **drops** it from the phoneme output. It has to be
/// re-attached, because it is what gives the clause its intonation (and the
/// length of the pause a comma buys).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Clause {
    pub(crate) body: String,
    /// The punctuation that ended this clause, if any.
    pub(crate) terminator: Option<char>,
}

/// The phonemes one espeak clause contributes, terminator included.
///
/// Reproduces upstream's `phonemize_espeak`: the `(lang)` switches espeak wraps
/// foreign words in go, the terminator comes back (espeak drops it from `--ipa`
/// output, and it is what gives the clause its intonation), a `, : ;` clause is
/// followed by a word boundary, and the string is NFD-split into code points so
/// stress marks and combining diacritics are phonemes of their own.
pub(crate) fn clause_phonemes(line: &str, terminator: Option<char>) -> Vec<char> {
    let mut s: String = strip_lang_switches(line).trim().to_string();
    if let Some(t) = terminator {
        s.push(t);
        // Upstream adds a word boundary after a non-sentence-boundary
        // terminator, so a comma clause is audibly distinct from a period one.
        if matches!(t, ',' | ':' | ';') {
            s.push(' ');
        }
    }
    s.nfd().collect()
}

/// espeak marks a word it reads as another language with a parenthesized
/// language code. Piper drops the marker and keeps the phonemes around it, so
/// the voice does not read the marker out.
///
/// Only *closed* groups are removed, matching upstream's `\([^)]+\)`: an
/// unmatched `(` is ordinary text and must not swallow the rest of the clause.
fn strip_lang_switches(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('(') {
        let after = &rest[open + 1..];
        match after.find(')') {
            Some(close) => {
                out.push_str(&rest[..open]);
                rest = &after[close + 1..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// The phoneme ids for one utterance: BOS, a PAD, then every phoneme's ids
/// followed by a PAD, then EOS.
///
/// A phoneme the map does not hold is skipped, exactly as upstream does — the
/// alternative, mapping it to `0`, would substitute the padding symbol for a
/// sound and quietly change the word.
pub(crate) fn phonemes_to_ids(map: &BTreeMap<String, Vec<i64>>, phonemes: &[char]) -> Vec<i64> {
    let mut ids: Vec<i64> = Vec::with_capacity(phonemes.len() * 2 + 3);
    ids.extend_from_slice(map.get(BOS).map(Vec::as_slice).unwrap_or(&[]));
    ids.extend_from_slice(map.get(PAD).map(Vec::as_slice).unwrap_or(&[]));
    let mut missing = 0usize;
    for phoneme in phonemes {
        match map.get(phoneme.encode_utf8(&mut [0; 4])) {
            Some(value) => {
                ids.extend_from_slice(value);
                ids.extend_from_slice(map.get(PAD).map(Vec::as_slice).unwrap_or(&[]));
            }
            None => missing += 1,
        }
    }
    ids.extend_from_slice(map.get(EOS).map(Vec::as_slice).unwrap_or(&[]));
    if missing > 0 {
        log::debug!("{missing} phoneme(s) missing from the voice's id map, skipped");
    }
    ids
}

/// Peak-normalize, the way piper does, including its `0.01` floor: a quiet
/// synthesis is amplified to that floor rather than left alone, and the result
/// is clipped rather than wrapped.
pub(crate) fn normalize_peak(audio: &[f32]) -> Vec<f32> {
    let peak = audio.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let gain = 1.0 / peak.max(0.01);
    audio.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect()
}

/// One espeak run for a list of clauses, one clause per line.
///
/// Returns one line per clause. espeak writes a line per clause *it* found, so
/// a clause carrying punctuation it splits on internally (it does split on a
/// comma even inside a line) would come back as two lines and shift every
/// terminator after it onto the wrong clause. That is why the clauses are
/// pre-split here; when the counts still disagree the mapping cannot be
/// trusted, so each clause is phonemized on its own and any extra lines joined.
fn espeak_phonemes(voice: &str, clauses: &[Clause]) -> std::io::Result<Vec<String>> {
    let bodies: Vec<&str> = clauses.iter().map(|c| c.body.as_str()).collect();
    let batched = run_espeak(voice, &bodies)?;
    let lines: Vec<String> = batched.lines().map(str::to_string).collect();
    if lines.len() == clauses.len() {
        return Ok(lines);
    }
    log::debug!(
        "espeak returned {} line(s) for {} clause(s); phonemizing one at a time",
        lines.len(),
        clauses.len()
    );
    let mut per_clause = Vec::with_capacity(clauses.len());
    for body in &bodies {
        let out = run_espeak(voice, std::slice::from_ref(body))?;
        // A clause espeak still splits internally is joined with word
        // boundaries rather than truncated: the space is a real phoneme, so
        // joining reads as one clause.
        per_clause.push(out.lines().collect::<Vec<_>>().join(" "));
    }
    Ok(per_clause)
}

/// Run `espeak-ng -v <voice> --ipa -q` with `clauses` on stdin.
///
/// `--ipa` (no `=n`) is the flag that matters: `=1`/`=2`/`=3` additionally set
/// separators and tie characters, which inserts phonemes piper's bridge does
/// not produce. The raw `io::Error` is returned so a missing binary stays
/// matchable as `NotFound` by the caller.
fn run_espeak(voice: &str, clauses: &[&str]) -> std::io::Result<String> {
    let mut child = Command::new("espeak-ng")
        .args(["-v", voice, "--ipa", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("espeak-ng stdin unavailable"))?;
        for clause in clauses {
            // Bounded by `MAX_SPEAK_CHARS`, so the write cannot outrun the pipe
            // buffer the child is not draining yet.
            writeln!(stdin, "{clause}")?;
        }
    }
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_string(&mut out)?;
    }
    let mut err = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        stderr.read_to_string(&mut err)?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(std::io::Error::other(format!(
            "espeak-ng exited with {status}{}",
            if err.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", err.trim())
            }
        )));
    }
    Ok(out)
}

/// Whether espeak can phonemize for `voice`, checked once at load so a missing
/// binary or voice data package is a startup error rather than a silent no-op
/// on the first emission.
fn espeak_available(voice: &str) -> Result<()> {
    run_espeak(voice, &["a"]).map_err(|e| {
        let hint = if e.kind() == std::io::ErrorKind::NotFound {
            "espeak-ng was not found on PATH; install espeak-ng (and its voice data \
             package) for --speak"
        } else {
            "espeak-ng could not phonemize; install espeak-ng and its voice data package"
        };
        anyhow::anyhow!("{hint} ({e})")
    })?;
    Ok(())
}

/// Read a voice's config without loading its graph, for callers that only want
/// to describe it.
#[allow(dead_code)]
pub(crate) fn read_config(path: &Path) -> Result<VoiceConfig> {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("{} is not a piper voice config", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but real `phoneme_id_map` shape: sentinels, punctuation, the
    /// space, a vowel cluster, a stress mark and a combining diacritic, each at
    /// its upstream id for the pl_PL-gosia export.
    fn id_map() -> BTreeMap<String, Vec<i64>> {
        let pairs: &[(&str, i64)] = &[
            ("_", 0),
            ("^", 1),
            ("$", 2),
            (" ", 3),
            (".", 10),
            (",", 8),
            ("?", 13),
            ("a", 14),
            ("d", 17),
            ("e", 18),
            ("o", 27),
            ("z", 38),
            ("ɪ", 74),
            ("ˈ", 120),
            ("aɪ", 161),
            ("̪", 142),
        ];
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), vec![*v]))
            .collect()
    }

    /// `espeak-ng -v pl --ipa -q` output for `Dzień dobry.`, read off this
    /// machine (espeak 1.52.0). The period is absent: espeak drops the
    /// terminator from `--ipa` output, which is why the clause carries it.
    const PL_DZIEN_DOBRY: &str = "d\u{0291}\u{02c8}\u{025b}\u{0272} d\u{02c8}\u{0254}br\u{0268}";

    #[test]
    fn clauses_are_cut_at_the_terminators_esperak_also_cuts_at() {
        let clause = |body: &str, terminator: Option<char>| Clause {
            body: body.to_string(),
            terminator,
        };
        assert_eq!(
            split_clauses("Dzień dobry."),
            vec![clause("Dzień dobry", Some('.'))]
        );
        // A comma is a clause boundary too: espeak splits on it, so the mapping
        // back to one line per clause only holds if it is cut here as well.
        assert_eq!(
            split_clauses("Idę do sklepu, potem do domu"),
            vec![
                clause("Idę do sklepu", Some(',')),
                clause("potem do domu", None)
            ]
        );
        assert_eq!(
            split_clauses("A? B! C: D;"),
            vec![
                clause("A", Some('?')),
                clause("B", Some('!')),
                clause("C", Some(':')),
                clause("D", Some(';'))
            ]
        );
        // No terminator: the whole text is one clause with none.
        assert_eq!(
            split_clauses("  bez kropki  "),
            vec![clause("bez kropki", None)]
        );
        // Nothing speakable.
        assert!(split_clauses("").is_empty());
        assert!(split_clauses("   . , ").is_empty());
        // Whitespace collapses to single spaces so a clause never holds a
        // newline, which would end it inside espeak.
        assert_eq!(split_clauses("a  \t b\nc"), vec![clause("a b c", None)]);
        // A terminator with nothing before it does not produce an empty clause.
        assert_eq!(split_clauses(".a"), vec![clause("a", None)]);
    }

    #[test]
    fn the_terminator_comes_back_because_espeak_drops_it() {
        // The period that gives `Dzień dobry.` its falling intonation is not in
        // espeak's output and has to be re-attached.
        assert_eq!(
            clause_phonemes(PL_DZIEN_DOBRY, Some('.')),
            vec!['d', 'ʑ', 'ˈ', 'ɛ', 'ɲ', ' ', 'd', 'ˈ', 'ɔ', 'b', 'r', 'ɨ', '.']
        );
        // A comma clause is followed by a word boundary.
        assert_eq!(
            clause_phonemes(PL_DZIEN_DOBRY, Some(',')),
            vec!['d', 'ʑ', 'ˈ', 'ɛ', 'ɲ', ' ', 'd', 'ˈ', 'ɔ', 'b', 'r', 'ɨ', ',', ' ']
        );
        // A question mark ends the clause with no trailing space.
        assert_eq!(
            clause_phonemes(PL_DZIEN_DOBRY, Some('?')),
            vec!['d', 'ʑ', 'ˈ', 'ɛ', 'ɲ', ' ', 'd', 'ˈ', 'ɔ', 'b', 'r', 'ɨ', '?']
        );
        // No terminator: nothing is added.
        assert_eq!(
            clause_phonemes(PL_DZIEN_DOBRY, None),
            vec!['d', 'ʑ', 'ˈ', 'ɛ', 'ɲ', ' ', 'd', 'ˈ', 'ɔ', 'b', 'r', 'ɨ']
        );
    }

    #[test]
    fn language_switches_are_dropped_but_their_phonemes_are_kept() {
        // espeak marks a word it reads as another language with a parenthesized
        // language code — `(h)alo`, not `(halo)` — and upstream strips the
        // marker while keeping the phonemes around it, so the voice does not
        // read the letter out.
        assert_eq!(clause_phonemes("(h)alo", None), vec!['a', 'l', 'o']);
        assert_eq!(
            clause_phonemes("(h)alo d\u{0291}", Some('.')),
            vec!['a', 'l', 'o', ' ', 'd', 'ʑ', '.']
        );
        // An unmatched `(` is ordinary text: upstream's `\([^)]+\)` needs a closing
        // paren, so an open one must not swallow the rest of the clause. (The
        // `(` then reaches the id map, where an unknown phoneme is skipped.)
        assert_eq!(clause_phonemes("a (b", None), vec!['a', ' ', '(', 'b']);
        assert_eq!(strip_lang_switches("a (b c"), "a (b c");
        assert_eq!(strip_lang_switches("(h)alo (e)xe"), "alo xe");
        // An empty group is still a group.
        assert_eq!(strip_lang_switches("a () b"), "a  b");
        // Nested: the regex is not a parser, it takes the first `)`, exactly as
        // upstream's `\([^)]+\)` does.
        assert_eq!(strip_lang_switches("(a(b)c) d"), "c) d");
    }

    #[test]
    fn phonemes_split_into_code_points_so_stress_is_its_own_phoneme() {
        // `t͡s` (a tie bar) decomposes to `t`, U+0361, `s` under NFD, which is
        // how a tie bar becomes phonemes the map can hold.
        assert_eq!(
            clause_phonemes("t\u{0361}s", Some('.')),
            vec!['t', '\u{0361}', 's', '.']
        );
        // A precomposed form decomposes the same way.
        assert_eq!(
            clause_phonemes("\u{00e9}", Some('.')),
            vec!['e', '\u{0301}', '.']
        );
        // IPA letters that do not decompose stay single phonemes: `ɨ` is one
        // code point and one phoneme, which is why the real output above is
        // usable against a per-code-point id map at all.
        assert_eq!(clause_phonemes("\u{0268}", None), vec!['ɨ']);
    }

    #[test]
    fn ids_are_bos_pad_then_each_phoneme_with_a_pad_then_eos() {
        let map = id_map();
        let ids = phonemes_to_ids(&map, &['d', 'a']);
        assert_eq!(ids, vec![1, 0, 17, 0, 14, 0, 2]);
        // Nothing at all still produces the frame, so the model always sees a
        // well-formed sequence.
        assert_eq!(phonemes_to_ids(&map, &[]), vec![1, 0, 2]);
    }

    #[test]
    fn a_multi_id_phoneme_emits_every_id_then_one_pad() {
        let map = id_map();
        let ids = phonemes_to_ids(&map, &['a', '\u{026a}']);
        // `aɪ` is one map entry with two ids upstream; here the map gives it one,
        // so the shape asserted is "all its ids, then exactly one PAD".
        assert_eq!(ids, vec![1, 0, 14, 0, 74, 0, 2]);
    }

    #[test]
    fn an_unknown_phoneme_is_skipped_rather_than_turned_into_padding() {
        let map = id_map();
        // `q` is not in the map. Mapping it to 0 would put the padding symbol
        // where a sound belongs, so it must vanish with no pad of its own.
        let ids = phonemes_to_ids(&map, &['d', 'q', 'a']);
        assert_eq!(ids, vec![1, 0, 17, 0, 14, 0, 2]);
    }

    #[test]
    fn normalize_peak_matches_piper_including_its_floor_and_clip() {
        // Quiet output is lifted to the 0.01 floor, not left alone.
        let quiet = normalize_peak(&[0.002, -0.004, 0.001]);
        assert!((quiet[0] - 0.2).abs() < 1e-5, "{quiet:?}");
        assert!((quiet[1] + 0.4).abs() < 1e-5, "{quiet:?}");
        // A normal peak reaches 1.0.
        let loud = normalize_peak(&[0.5, -0.25]);
        assert!((loud[0] - 1.0).abs() < 1e-6, "{loud:?}");
        assert!((loud[1] + 0.5).abs() < 1e-6, "{loud:?}");
        // Digital silence stays silent rather than being amplified to noise.
        assert_eq!(normalize_peak(&[0.0, 0.0]), vec![0.0, 0.0]);
        // Nothing may exceed full scale after the gain.
        assert!(normalize_peak(&[1.0, -1.0]).iter().all(|s| s.abs() <= 1.0));
        assert!(normalize_peak(&[]).is_empty());
    }

    #[test]
    fn scales_are_noise_length_noise_w_in_that_order() {
        // The exporter writes `torch.FloatTensor([noise, length, noise_w])` and
        // both piper runtimes pass `[noise_scale, length_scale, noise_w_scale]`.
        // A transposed tensor does not fail, it just turns a length scale into
        // generator noise, so the order is pinned here.
        let c = InferenceConfig {
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_w: 0.8,
        };
        assert_eq!(c.scales(), [0.667, 1.0, 0.8]);
        // And the defaults are upstream's, for a config that omits the block.
        let d: InferenceConfig = serde_json::from_str("{}").expect("defaults must apply");
        assert_eq!(d.scales(), [0.667, 1.0, 0.8]);
    }

    /// The pl_PL-gosia config, trimmed to what `VoiceConfig` reads.
    const CONFIG: &str = r#"{
        "audio": {"sample_rate": 22050, "quality": "medium"},
        "espeak": {"voice": "pl"},
        "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
        "phoneme_type": "espeak",
        "phoneme_map": {},
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], "d": [17], ".": [10]},
        "num_symbols": 256,
        "num_speakers": 1,
        "speaker_id_map": {},
        "piper_version": "1.0.0"
    }"#;

    #[test]
    fn a_voice_config_parses_with_upstreams_defaults() {
        let c: VoiceConfig = serde_json::from_str(CONFIG).expect("config must parse");
        assert_eq!(c.audio.sample_rate, 22050);
        assert_eq!(c.espeak.voice, "pl");
        assert_eq!(c.phoneme_type, "espeak");
        assert_eq!(c.num_symbols, 256);
        assert_eq!(c.num_speakers, 1);
        assert_eq!(c.inference.length_scale, 1.0);
        // Unknown fields are upstream's, not sttd's: they must not fail the load.
        assert!(c.phoneme_id_map.contains_key("^"));
    }

    #[test]
    fn a_config_without_an_espeak_voice_or_speakers_defaults_rather_than_failing() {
        let minimal = r#"{
            "audio": {"sample_rate": 16000},
            "phoneme_type": "espeak",
            "phoneme_id_map": {"_": [0], "^": [1], "$": [2]},
            "num_symbols": 256
        }"#;
        let c: VoiceConfig = serde_json::from_str(minimal).expect("must parse");
        assert_eq!(c.espeak.voice, "", "an empty voice is caught at load");
        assert_eq!(c.num_speakers, 1, "single speaker is the default");
    }

    #[test]
    fn text_phoneme_voices_are_out_of_reach_and_say_so() {
        // `phoneme_type: "text"` voices take raw text and have no espeak voice,
        // so there is nothing for sttd's phonemizer to do with them.
        let json = CONFIG.replace("\"espeak\"", "\"text\"");
        let c: VoiceConfig = serde_json::from_str(&json).expect("must parse");
        assert_eq!(c.phoneme_type, "text");
        assert_ne!(c.phoneme_type, "espeak");
    }

    #[test]
    fn input_names_tolerate_case_and_separator_variation() {
        // `resolve_input_names` reads a live session, so the matching rule it
        // applies is restated here over the name lists voices actually ship:
        // the canonical piper names, and the casing/separator drift an export
        // from another toolchain produces.
        // The prod matcher runs over lowercased names and hands back the graph's own
        // spelling, so every `want` finds its slot whatever the casing, and the
        // name fed to ORT is the one the graph declares.
        for names in [
            vec!["input", "input_lengths", "scales"],
            vec!["Input", "Input-Lengths", "Scales"],
        ] {
            let lower: Vec<String> = names.iter().map(|n| n.to_ascii_lowercase()).collect();
            for want in ["input", "input_lengths", "scales"] {
                let slot = lower
                    .iter()
                    .position(|n| n.as_str() == want || squash(n) == squash(want))
                    .unwrap_or_else(|| panic!("{want} must be found in {names:?}"));
                assert_eq!(squash(&lower[slot]), squash(want), "{want} in {names:?}");
                // Exact match wins when there is one, so `input` never takes
                // the `input_lengths` slot.
                if let Some(exact) = lower.iter().position(|n| n == want) {
                    assert_eq!(slot, exact, "{want} must take its own slot");
                }
            }
        }
        // No prefix fallback, so `input_lengths` can never satisfy `input` —
        // that would bind the id tensor to the wrong slot.
        assert_ne!(squash("input_lengths"), squash("input"));
        assert_eq!(squash("Input_Lengths"), "inputlengths");
    }

    /// The speaker input a graph declares, if any. Mirrors the exact match in
    /// `resolve_input_names` (there spelled against the lowercased names).
    fn speaker_input(names: &[&str]) -> Option<String> {
        names
            .iter()
            .find(|n| **n == "sid" || **n == "speaker_id")
            .map(|n| n.to_string())
    }

    #[test]
    fn the_speaker_input_is_only_matched_exactly() {
        // `sid` is fed only when the graph declares it, so the match must be
        // exact: a loose one would bind a `s` or a `sid_input` to speaker 0.
        assert_eq!(
            speaker_input(&["input", "input_lengths", "scales", "sid"]).as_deref(),
            Some("sid")
        );
        assert_eq!(
            speaker_input(&["input", "scales", "speaker_id"]).as_deref(),
            Some("speaker_id")
        );
        // A single-speaker graph leaves it unset rather than guessing, so the
        // `sid` tensor is not built at all.
        assert_eq!(speaker_input(&["input", "input_lengths", "scales"]), None);
        assert_eq!(speaker_input(&["input", "scales", "sid_input"]), None);
    }
}
