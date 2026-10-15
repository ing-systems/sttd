//! The run's translation settings, and loading a translator for one target.

use anyhow::{Context, Result};

use crate::translator_store::Prepared;

use super::engine::Translator;

/// The run's own translation settings, resolved once from the CLI.
///
/// A load needs all four, and at runtime only the bus is left to ask with — so
/// they are captured here rather than re-read from `Cli` on the bus thread. The
/// source language is the one Whisper was told to decode in, because that is what
/// the transcript is; it is fixed for the run, so a runtime switch changes the
/// target only.
pub(crate) struct TranslationPlan {
    /// `--language`, or `None` under `--language auto`: the source is then
    /// unknown per utterance, which a pair export can answer for and a
    /// multilingual one cannot.
    source: Option<String>,
    /// `--translate-model`, overriding which translator a direction resolves to.
    pub(super) model: Option<String>,
    /// The language `--translate` asked for, if any. What `Enable` reloads when
    /// nothing is loaded.
    pub(super) configured: Option<String>,
    intra_threads: usize,
    /// `--voice`, kept only under `--speak`, so a runtime switch can warn about
    /// the same voice/language mismatch the startup path warns about.
    voice: Option<String>,
}

/// The flags a [`TranslationPlan`] is read from, as `main` resolves them. Kept
/// apart from `Cli` so this module never depends on the command line.
pub(crate) struct PlanFlags<'a> {
    /// `--translate`, verbatim.
    pub(crate) translate: Option<&'a str>,
    /// `--translate-model`, verbatim.
    pub(crate) translate_model: Option<&'a str>,
    /// The language the engine is told to decode in (`--language` or the locale).
    pub(crate) language: crate::language::Language,
    /// `--threads`, already resolved (`0` = auto has been answered).
    pub(crate) intra_threads: usize,
    /// `--voice`, only under `--speak`.
    pub(crate) voice: Option<String>,
    /// Whether this run has any way to ask for a translation other than
    /// `--translate`: the session bus in microphone mode.
    pub(crate) runtime_requests: bool,
}

impl TranslationPlan {
    /// A plan for a test: one named source, no override, nothing configured and no
    /// load. Every direction it can resolve is a catalog lookup, so a bus test
    /// exercises the refusals without weights and without a network.
    #[cfg(test)]
    pub(crate) fn for_tests(source: &str) -> Self {
        Self {
            source: Some(source.to_string()),
            model: None,
            configured: None,
            intra_threads: 1,
            voice: None,
        }
    }

    /// A plan whose translator is named outright, so the catalog's pair table is
    /// bypassed: `reference_for` then validates the spec alone and any target is
    /// accepted. That is what lets a test drive the *load* half — a switch, a
    /// failure — against targets the real catalog would refuse before a thread ever
    /// started.
    #[cfg(test)]
    pub(crate) fn for_tests_with_model(source: &str, model: &str) -> Self {
        Self {
            model: Some(model.to_string()),
            ..Self::for_tests(source)
        }
    }

    /// A plan as `--translate <target>` makes it, so the startup path is drivable
    /// without weights.
    #[cfg(test)]
    pub(crate) fn for_tests_configured(source: &str, target: &str) -> Self {
        Self {
            configured: Some(target.to_string()),
            ..Self::for_tests_with_model(source, "nllb-200-distilled-600M")
        }
    }

    /// Read the plan out of the flags, warning about the combinations that
    /// cannot work. See the type for what each field is.
    ///
    /// The warnings are the startup ones: a `--translate` with no code cannot name
    /// a direction, and a `--translate-model` with nothing asking for a
    /// translation downloads nothing.
    ///
    /// The second is only *unused* when no one can ask at runtime, which is why the
    /// session bus counts as an asker: the override now applies to a
    /// `SetTargetLanguage` too, not only to a `--translate` named at startup. So
    /// `--translate-model` on its own warns under `--no-dbus` and in file mode, and
    /// stays quiet in microphone mode with the bus, where it does something.
    pub(crate) fn from_flags(flags: PlanFlags<'_>) -> Self {
        let configured = flags
            .translate
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase());
        let model = flags
            .translate_model
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if configured.is_none() {
            if flags.translate.is_some() {
                log::warn!("--translate needs a language code (e.g. en); the step is off");
            }
            if model.is_some() && !flags.runtime_requests {
                log::warn!("--translate-model is ignored without --translate");
            }
        }
        let source = match flags.language {
            crate::language::Language::Explicit(code) | crate::language::Language::System(code) => {
                Some(code)
            }
            crate::language::Language::Auto => None,
        };
        Self {
            source,
            model,
            configured,
            intra_threads: flags.intra_threads,
            voice: flags.voice,
        }
    }

    /// The direction this run translates `target` in.
    pub(super) fn direction<'a>(
        &'a self,
        target: &'a str,
    ) -> crate::translator_store::Direction<'a> {
        crate::translator_store::Direction {
            src: self.source.as_deref(),
            tgt: target,
        }
    }

    /// The source language as a log or a spec reads it.
    #[cfg(test)]
    pub(super) fn source_label(&self) -> &str {
        self.source.as_deref().unwrap_or("auto")
    }
}

/// Resolve, download and load the translator for `target`, on the calling thread.
///
/// The whole startup path for one language, and the whole runtime path too: the
/// direction comes from the plan, which the flags fixed. Whether a direction is
/// even possible is answered before anything is fetched — see
/// [`crate::translator_store::reference_for`] — so a refusal costs nothing.
pub(crate) fn load_for(plan: &TranslationPlan, target: &str) -> Result<Translator> {
    let direction = plan.direction(target);
    let prepared = match plan.model.as_deref() {
        Some(spec) => crate::translator_store::prepare(spec, direction)
            .with_context(|| format!("failed to prepare translator {spec}"))?,
        None => crate::translator_store::prepare_from_direction(direction)?,
    };
    log_translator_direction(&prepared, target);
    if let Some(voice) = plan.voice.as_deref() {
        warn_on_voice_language_mismatch(voice, target);
    }
    Translator::load(&prepared, plan.intra_threads).context("failed to load the translator")
}

/// Log the direction a run translates in, and warn when the engine does not decode
/// in the language that direction assumes.
///
/// A **multilingual** export is the only case with an assumption to state: it is
/// told which language the source is, so under an engine that decides for itself
/// the flag names what the *translator* should read, not what the engine should
/// hear. Parakeet detects the spoken language per utterance and ignores
/// `--language` for transcription, so without this line `--language pl` alongside
/// a multilingual model reads as "transcribe Polish" — and a Polish sentence
/// spoken in German would be translated as Polish.
///
/// A **pair** export has no such gap: its weights only do one direction, so
/// `--language auto` picks a many-source model and a named code picks the matching
/// pair, and either way the direction was already resolved rather than assumed.
/// Nothing is warned about, which is the point of keeping the check on `codes`
/// rather than on the translator existing.
fn log_translator_direction(prepared: &Prepared, target: &str) {
    log::info!(
        "translate: emitting every accepted transcript into {target} (translator {})",
        prepared.reference.spec
    );
    let Some((src, _)) = &prepared.codes else {
        return;
    };
    log::warn!(
        "the selected engine detects the spoken language per utterance and ignores \
         --language for transcription, but {NLLB_SPEC_IN_LOG} is told to read every \
         transcript as {src}; speech in any other language will be mistranslated"
    );
}

/// The spec name, spelled once here rather than interpolated into the format
/// string above: `translator_store` owns the constant and a renamed spec must not
/// leave a stale copy in a warning.
const NLLB_SPEC_IN_LOG: &str = crate::translator_store::NLLB_SPEC;

/// Warn when the `--speak` voice speaks a different language than the one being
/// written to the clipboard. Not an error: a translation can still be read aloud
/// in the wrong language, which is bad but not fatal, and the user may know
/// their voice speaks it.
fn warn_on_voice_language_mismatch(voice: &str, target: &str) {
    let Some(voice_lang) = crate::voice_store::voice_language(voice) else {
        return;
    };
    if voice_lang.eq_ignore_ascii_case(target) {
        return;
    }
    log::warn!(
        "--voice {voice} is a {voice_lang} voice but translation writes {target}: \
         sttd will read the translation aloud in {voice_lang} \
         (pick a {target} voice with --voice)"
    );
}
