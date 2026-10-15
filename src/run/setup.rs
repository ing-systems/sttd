//! What both input paths resolve from the flags before the first frame: the
//! model, the acoustic tunables, the worker's configuration, the dictionary,
//! the emitter, the metrics log and the translation step. Every `Cli` →
//! component mapping lives here, so no component below learns what a command
//! line is.

use std::time::Duration;

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::asr::{AsrConfig, LiveDictionary, PartialSlot};
use crate::cli::Cli;
use crate::control::DictionaryState;
use crate::device::{auto_threads, resolve_n_threads};
use crate::emit::{EmissionPolicy, SpeechOut, SystemEmitter};
use crate::frame::FrameProcessor;
use crate::gate::GateConfig;
use crate::language::Language;
use crate::metrics::{open_metrics_log, MetricsSlot};
use crate::model_store::{ensure_available, resolve_model, EngineKind, ModelRef};
use crate::speech::EngineConfig;
use crate::translate::{PlanFlags, RealTranslator, TranslationPlan, TranslationState};
use crate::utterance::Utterance;
use crate::whisper_core::types::DEFAULT_WINDOW_FRAMES;

/// Resolve/download the model and create the bounded utterance ring: the
/// pre-capture work both input paths share, so a bad `--model` or a bad
/// `--dump-dir` fails fast before any capture or heavy setup.
///
/// The ring is bounded (`ASR_QUEUE_CAP`, each utterance seconds of audio) and
/// the ASR worker sheds weakest-first past `ASR_SHED_KEEP`, so live speech
/// stays fresh instead of lagging minutes behind. It survives a device
/// rebuild: the main loop owns the producer, the worker the consumer, and
/// `FrameProcessor` just returns closed utterances.
pub(super) fn prepare_model(
    cli: &Cli,
) -> Result<(ModelRef, Producer<Utterance>, Consumer<Utterance>)> {
    let model = resolve_model_input(cli)?;
    log::info!("model: {model}");
    validate_model_flags(cli, &model)?;
    let (tx, rx) = RingBuffer::<Utterance>::new(crate::asr::ASR_QUEUE_CAP);
    Ok((model, tx, rx))
}

/// Resolve `--model` (spec or path, either engine) to a guaranteed-present
/// `ModelRef`: engine dispatch first (`resolve_model`), then
/// download-if-missing (`ensure_available`).
pub(super) fn resolve_model_input(cli: &Cli) -> Result<ModelRef> {
    let input = cli.model_input();
    let model = resolve_model(&input)?;
    ensure_available(&model)?;
    Ok(model)
}

/// Flags the selected engine ignores, so the no-effect is visible instead of
/// silent. Without the `cuda` feature both engines run on CPU, so `--gpu` is
/// accepted but has no provider behind it. With the `cuda` feature the
/// Parakeet and Whisper encoders use CUDA and the decoders stay on CPU, so
/// `--gpu` applies and is not reported here. `--language` only means something
/// to Whisper.
pub(super) fn ignored_model_flags(cli: &Cli, model: &ModelRef) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if cli.gpu && !cfg!(feature = "cuda") {
        flags.push("--gpu");
    }
    if model.kind == EngineKind::Parakeet && cli.language.is_some() {
        // Parakeet auto-detects per utterance, so a fixed language cannot apply.
        // Any explicit flag counts, including one that matches the default the
        // system locale would have supplied: the user typed it, so it is worth
        // saying that it does nothing here.
        flags.push("--language");
    }
    if cli.audio_ctx != DEFAULT_WINDOW_FRAMES && model.kind == EngineKind::Parakeet {
        // Parakeet sizes its window itself; a Whisper encoder follows the audio
        // because the graph is patched when the model loads.
        flags.push("--audio-ctx");
    }
    flags
}

/// Warn about flags the selected engine ignores, then log what that engine
/// does about the spoken language.
pub(super) fn validate_model_flags(cli: &Cli, model: &ModelRef) -> Result<()> {
    let ignored = ignored_model_flags(cli, model);
    if !ignored.is_empty() {
        log::warn!(
            "{} ignores flag(s): {} (no effect with this --model)",
            model.kind.as_str(),
            ignored.join(", ")
        );
    }
    match model.kind {
        EngineKind::Parakeet => {
            log::info!("Parakeet auto-detects the spoken language per utterance")
        }
        EngineKind::Whisper => match cli.language() {
            Language::Auto => {
                log::info!("Whisper detects the spoken language per utterance (--language auto)")
            }
            language => log::info!("Whisper decodes in {language} (--language)"),
        },
    }
    Ok(())
}

/// The worker's whole configuration, resolved once from `Cli` on this thread
/// before the worker starts. The `Cli` → config mapping lives here, not in
/// `asr`, so the worker module never learns what a command line is.
pub(super) fn asr_config(
    cli: &Cli,
    acoustic: &AcousticConfig,
    shed_stale: bool,
    dictionary: LiveDictionary,
) -> AsrConfig {
    AsrConfig {
        engine: EngineConfig {
            model_spec: cli.model_input(),
            n_threads: resolve_n_threads(cli.threads, auto_threads()),
            use_gpu: cli.gpu,
            language: cli.language(),
            audio_ctx: cli.audio_ctx,
        },
        // The transcript gate shares the acoustic thresholds, so both come
        // from one resolved config.
        gate: acoustic.gate.clone(),
        remove_hesitations: cli.remove_hesitations,
        // The live handle, not a `Vec` snapshot: the D-Bus thread swaps the
        // word list at runtime and the worker observes it per utterance.
        dictionary,
        shed_stale,
        partials: PartialSlot::default(),
        auto_unload: cli.auto_unload,
        auto_unload_after: Duration::from_secs(cli.auto_unload_after),
        sound: cli.sound,
        autopause: cli.autopause,
    }
}

/// The run's translation settings, read out of the flags. See
/// `translate::PlanFlags` for what each one is.
pub(super) fn translation_plan(cli: &Cli) -> TranslationPlan {
    TranslationPlan::from_flags(PlanFlags {
        translate: cli.translate.as_deref(),
        translate_model: cli.translate_model.as_deref(),
        language: cli.language(),
        intra_threads: resolve_n_threads(cli.threads, auto_threads()).max(1) as usize,
        voice: cli.speak.then(|| cli.voice.clone()),
        runtime_requests: !cli.no_dbus && cli.file.is_none(),
    })
}

/// Load `--dictionary` once at startup. A broken path warns and disables
/// correction (never fails transcription); success logs the entry count.
/// Large dictionaries warn: above `POSTCORR_WARN_DICT_WORDS` the false-positive
/// risk grows, at or above `POSTCORR_MAX_DICT_WORDS` fuzzy is off (exact stays).
pub(super) fn load_dictionary_or_warn(cli: &Cli) -> Vec<String> {
    let Some(path) = cli.dictionary.as_deref() else {
        return Vec::new();
    };
    match crate::dictionary::load_dictionary(path) {
        Ok(words) => {
            log::info!(
                "dictionary: {} entr(y/ies) from {}",
                words.len(),
                path.display()
            );
            if words.len() > crate::dictionary::POSTCORR_MAX_DICT_WORDS {
                log::warn!(
                    "dictionary: {} entries, fuzzy correction disabled (over {}); exact casing restore stays on",
                    words.len(),
                    crate::dictionary::POSTCORR_MAX_DICT_WORDS
                );
            } else if words.len() > crate::dictionary::POSTCORR_WARN_DICT_WORDS {
                log::warn!(
                    "dictionary: {} entries, false-positive risk grows; keep the list small and targeted",
                    words.len()
                );
            }
            words
        }
        Err(e) => {
            log::warn!(
                "dictionary: failed to load {}: {e:?} (correction disabled)",
                path.display()
            );
            Vec::new()
        }
    }
}

/// Emission policy (dump dir, clipboard, paste, stdout, spoken transcripts),
/// resolved once from `Cli`. Runs on the main thread before the worker spawns,
/// so a bad dump dir, voice key or voice download fails fast like the model load
/// it precedes.
pub(super) fn make_emitter(cli: &Cli) -> Result<SystemEmitter> {
    SystemEmitter::new(EmissionPolicy {
        dump_dir: cli.dump_dir.clone(),
        copy: cli.copy,
        paste: cli.paste,
        paste_mode: cli.paste_mode,
        stdout: cli.stdout,
        speech: make_speech(cli)?,
    })
}

/// The spoken-emission channel, when `--speak` is on: resolve the voice key,
/// download what is missing and load the graph, then hand it to its own thread.
///
/// Off by default, and inert without `--speak`: `--voice` is never resolved and
/// nothing is downloaded, because a flag that silently pulled 60 MiB nobody
/// asked for is worse than one that does nothing. The voice load is a few
/// hundred milliseconds, on the main thread, before the worker starts — the same
/// place and cost profile as the speech model's own load.
pub(super) fn make_speech(cli: &Cli) -> Result<Option<SpeechOut>> {
    if !cli.speak {
        // Warned rather than silently ignored, and only when the user actually
        // typed one: the default is not a thing they asked for.
        if cli.voice != crate::config::DEFAULT_VOICE {
            log::warn!("--voice is ignored without --speak");
        }
        return Ok(None);
    }
    let voice = crate::voice_store::prepare_voice(&cli.voice)
        .with_context(|| format!("failed to prepare voice {}", cli.voice))?;
    log::info!("speak: reading every emitted transcript aloud (voice={voice})");
    if !cli.autopause {
        // The capture side holds itself while sttd speaks, so a second utterance
        // cannot even be recorded during playback. That is right for
        // single-shot dictation and wrong for a continuous run, where it eats
        // the user's own words; `--autopause` is the pairing that works.
        log::warn!(
            "--speak without --autopause discards everything you say while sttd is speaking \
             (use --autopause, or headphones)"
        );
    }
    // One thread for the voice, half of `--threads` at most: a sentence is a
    // few seconds of audio and the graph is small, so it must not take the
    // whole machine away from the speech model decoding the next utterance.
    let speech_threads = (resolve_n_threads(cli.threads, 1) / 2).max(1) as usize;
    let loaded = crate::piper::PiperVoice::load(&voice, speech_threads)
        .context("failed to load the piper voice")?;
    Ok(Some(SpeechOut::spawn(loaded)))
}

/// The capture-side tunables, resolved once from `Cli` and shared by both
/// input paths: the microphone opens a `FrameProcessor` per device open (and
/// per rebuild), file mode opens one, and the same thresholds reach the ASR
/// worker through `AsrConfig::gate`. Keeping them in one struct is what
/// stops the two paths from drifting apart, and it keeps `Utterance` free of
/// config (only evidence).
pub(super) struct AcousticConfig {
    denoiser: crate::denoise::DenoiserKind,
    vad_threshold: f32,
    vad_adapt: bool,
    gate: GateConfig,
    hangover_frames: u32,
    target_rms: f32,
}

impl AcousticConfig {
    pub(super) const fn from_cli(cli: &Cli) -> Self {
        Self {
            denoiser: cli.denoiser,
            vad_threshold: cli.vad_threshold,
            vad_adapt: cli.vad_adapt,
            gate: GateConfig::new(cli.min_rms, cli.min_density),
            hangover_frames: crate::config::hangover_frames(cli.hangover_ms),
            target_rms: cli.target_rms,
        }
    }

    /// Build the frame processor for a given input rate. `FrameProcessor`
    /// owns all resampling (a shared hw -> 16 kHz hop by default, or
    /// hw -> 48 kHz -> denoise -> 48 kHz -> 16 kHz for the VAD branch plus
    /// hw -> 16 kHz for the model with `--denoiser deepfilter`).
    pub(super) fn make_frame_processor(&self, hw_rate: u32) -> Result<FrameProcessor> {
        FrameProcessor::new(
            hw_rate,
            self.denoiser,
            self.vad_threshold,
            self.vad_adapt,
            self.gate.clone(),
            self.hangover_frames,
            self.target_rms,
            crate::config::MAX_GAIN,
        )
    }
}

/// The live word list both input paths start from: `--dictionary` loaded once,
/// with the ASR booster hook attached by `LiveDictionary::new`. Returns the
/// hooked handle (what `dbus::spawn` serves) and the worker's side.
pub(super) fn live_dictionary(cli: &Cli) -> (DictionaryState, LiveDictionary) {
    LiveDictionary::new(DictionaryState::new(
        load_dictionary_or_warn(cli),
        cli.dictionary.clone().unwrap_or_default(),
    ))
}

/// The one `--metrics` writer. It is a slot both the worker and the
/// translation thread write through, because the worker writes a row for every
/// utterance it does not hand over and the thread writes the rows whose `text`
/// column is a translation.
pub(super) fn open_metrics(cli: &Cli) -> MetricsSlot {
    MetricsSlot::new(open_metrics_log(cli.metrics.as_deref()))
}

/// The translation step's shared state, resolved once from the flags.
///
/// With `--translate`, the translator is loaded here, on the main thread,
/// before the worker starts: it is ~114 MiB of weights and a setup problem (no
/// file, no write access) that should fail fast rather than on the first
/// utterance. A run without the flag loads nothing until the bus asks.
pub(super) fn start_translation(cli: &Cli) -> Result<TranslationState> {
    let state = TranslationState::new(translation_plan(cli), RealTranslator);
    state.load_startup()?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A `ModelRef` for flag-validation tests: `assets` is empty (a
    /// user-supplied directory), which is all the flag checks look at.
    fn model_ref(kind: EngineKind, dir: &str) -> ModelRef {
        ModelRef {
            dir: std::path::PathBuf::from(dir),
            assets: &[],
            kind,
        }
    }

    #[test]
    fn ignored_model_flags_are_empty_for_plain_cli() {
        // `--gpu` defaults to false and `--language` to the `Language`
        // default, so a plain CLI is clean for either engine.
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        for (kind, dir) in [
            (EngineKind::Parakeet, "models/parakeet-ultra-int8"),
            (EngineKind::Whisper, "models/whisper-large-v3-turbo-int8"),
        ] {
            let model = model_ref(kind, dir);
            assert!(
                ignored_model_flags(&cli, &model).is_empty(),
                "{}",
                kind.as_str()
            );
            assert!(validate_model_flags(&cli, &model).is_ok());
        }
    }

    /// `--gpu` is a no-op only in a CPU-only build. A `cuda` build runs the
    /// encoders on the CUDA EP, so the flag applies to both engines and no
    /// longer belongs in the ignored set.
    #[test]
    fn gpu_is_ignored_by_both_engines_without_cuda() {
        let cli = Cli::try_parse_from(["sttd", "--gpu"]).expect("flags must parse");
        let expected: Vec<&'static str> = if cfg!(feature = "cuda") {
            Vec::new()
        } else {
            vec!["--gpu"]
        };
        for kind in [EngineKind::Parakeet, EngineKind::Whisper] {
            let model = model_ref(kind, "models/x");
            assert_eq!(ignored_model_flags(&cli, &model), expected);
        }
    }

    /// Parakeet auto-detects per utterance, so a fixed `--language` cannot
    /// apply and must be reported rather than silently dropped.
    #[test]
    fn language_is_ignored_by_parakeet_but_honored_by_whisper() {
        let cli = Cli::try_parse_from(["sttd", "--language", "en"]).expect("flags must parse");
        let parakeet = model_ref(EngineKind::Parakeet, "models/parakeet-ultra-int8");
        let whisper = model_ref(EngineKind::Whisper, "models/whisper-large-v3-turbo-int8");
        assert_eq!(ignored_model_flags(&cli, &parakeet), vec!["--language"]);
        assert!(ignored_model_flags(&cli, &whisper).is_empty());

        // `auto` is a deviation from the default for Whisper's log line but a
        // Parakeet user who set it explicitly should still be told.
        let auto = Cli::try_parse_from(["sttd", "--language", "auto"]).expect("flags must parse");
        assert_eq!(auto.language, Some(Language::Auto));
        assert_eq!(auto.language(), Language::Auto);
        assert_eq!(
            ignored_model_flags(&auto, &parakeet),
            vec!["--language"],
            "explicit --language auto must still be reported under Parakeet"
        );
        assert!(ignored_model_flags(&auto, &whisper).is_empty());
    }

    #[test]
    fn audio_ctx_is_reported_only_where_the_window_cannot_follow_the_audio() {
        let parakeet = model_ref(EngineKind::Parakeet, "models/parakeet-ultra-int8");
        let whisper = model_ref(EngineKind::Whisper, "models/whisper-large-v3-turbo-int8");
        let cli = Cli::try_parse_from(["sttd", "--audio-ctx", "300"]).expect("must parse");

        // Parakeet sizes its own window.
        assert_eq!(ignored_model_flags(&cli, &parakeet), vec!["--audio-ctx"]);
        // Whisper's encoder is patched when the model loads, so the floor applies.
        assert!(ignored_model_flags(&cli, &whisper).is_empty());
        assert!(validate_model_flags(&cli, &parakeet).is_ok());

        // The default is not a setting, so nothing is reported for it.
        let plain = Cli::try_parse_from(["sttd"]).expect("must parse");
        assert!(ignored_model_flags(&plain, &whisper).is_empty());
        assert!(ignored_model_flags(&plain, &parakeet).is_empty());
    }

    /// Every code Whisper accepts goes through, and the unset default is the
    /// system locale's language rather than a fixed one. sttd keeps no list of
    /// languages: the model decides at load time which codes it has.
    #[test]
    fn language_flag_defaults_to_the_system_locale_and_takes_any_code() {
        let plain = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        assert_eq!(plain.language, None, "unset, not defaulted to a code");
        assert_eq!(
            plain.language(),
            crate::config::system_language(),
            "unset resolves through the locale"
        );

        for (value, expected) in [
            ("pl", Language::Explicit("pl".to_string())),
            ("en", Language::Explicit("en".to_string())),
            ("de", Language::Explicit("de".to_string())),
            ("yue", Language::Explicit("yue".to_string())),
            ("DE", Language::Explicit("de".to_string())),
            ("auto", Language::Auto),
            ("Auto", Language::Auto),
            (" pl ", Language::Explicit("pl".to_string())),
        ] {
            let cli = Cli::try_parse_from(["sttd", "--language", value])
                .unwrap_or_else(|e| panic!("--language {value} must parse: {e}"));
            assert_eq!(cli.language, Some(expected.clone()), "value={value:?}");
            assert_eq!(cli.language(), expected, "value={value:?}");
        }

        // A locale name, a word, a digit and an over-long code are not language
        // codes, so they are refused before a model is ever loaded. Which codes
        // the model *does* have is its own business, checked at load.
        for bad in ["x", "pl_PL", "english", "p1", "abcd", "", "en-US"] {
            assert!(
                Cli::try_parse_from(["sttd", "--language", bad]).is_err(),
                "must reject {bad:?}"
            );
        }
    }

    #[test]
    fn asr_options_carry_the_language_to_the_worker() {
        let cli = Cli::try_parse_from(["sttd", "--language", "en"]).expect("flags must parse");
        let dictionary = crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
            Vec::new(),
            Default::default(),
        ))
        .1;
        assert_eq!(
            asr_config(&cli, &AcousticConfig::from_cli(&cli), true, dictionary)
                .engine
                .language,
            Language::Explicit("en".to_string())
        );
    }

    #[test]
    fn asr_config_starts_with_an_empty_partial_slot() {
        let dictionary = || {
            crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
                Vec::new(),
                Default::default(),
            ))
            .1
        };
        let cli = Cli::try_parse_from(["sttd"]).expect("defaults must parse");
        let config = asr_config(&cli, &AcousticConfig::from_cli(&cli), true, dictionary());
        assert!(config.partials.take().is_none());
        // The capture side holds a clone of the same slot.
        let capture_side = config.partials.clone();
        capture_side.put(vec![0.0; 4]);
        assert_eq!(config.partials.take().map(|p| p.len()), Some(4));
    }

    #[test]
    fn acoustic_config_carries_the_cli_tunables() {
        let cli = Cli::try_parse_from([
            "sttd",
            "--vad-threshold",
            "0.4",
            "--vad-adapt",
            "off",
            "--min-rms",
            "0.01",
            "--min-density",
            "0.5",
            "--hangover-ms",
            "350",
            "--target-rms",
            "0.02",
        ])
        .expect("flags must parse");
        let acoustic = AcousticConfig::from_cli(&cli);
        assert_eq!(acoustic.vad_threshold, 0.4);
        assert!(!acoustic.vad_adapt);
        assert_eq!(acoustic.gate.min_speech_rms, 0.01);
        assert_eq!(acoustic.gate.min_density, 0.5);
        assert_eq!(acoustic.hangover_frames, 35);
        assert_eq!(acoustic.target_rms, 0.02);
        // The ASR worker gets the same thresholds, from the same builder.
        let dictionary = crate::asr::LiveDictionary::new(crate::control::DictionaryState::new(
            Vec::new(),
            Default::default(),
        ))
        .1;
        assert_eq!(
            asr_config(&cli, &acoustic, true, dictionary)
                .gate
                .min_density,
            0.5
        );
    }
}
