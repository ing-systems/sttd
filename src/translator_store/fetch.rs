//! Where a translator lives on disk and upstream, and fetching what is
//! missing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{DEFAULT_MODELS_DIR, NLLB_REPO, OPUS_MT_REPO, TRANSLATORS_SUBDIR};

use super::catalog::NLLB_SPEC;
use super::TranslatorRef;

/// Encoder graph, inside a translator directory and inside the repository
/// (`onnx/`). Same name for both families.
pub(super) const ENCODER_FILE: &str = "encoder_model_quantized.onnx";
/// Merged decoder graph: the prompt step, and every later step as well, because
/// one graph takes both. Same two locations.
pub(super) const DECODER_FILE: &str = "decoder_model_merged_quantized.onnx";
/// Decoder graph for the steps *after* the prompt, the one that reads a
/// key/value cache instead of recomputing it. Same two locations.
///
/// The second half of the decode. The merged graph above re-runs the whole
/// sequence on every step, which is **O(n²) in output length**; this graph takes
/// one token per step against the cache the merged graph returned, which is
/// O(n). Both are published for both families, and the split is the same one
/// `whisper_core` runs — see `translate::engine::Translator::run_cached_step`.
pub(super) const DECODER_WITH_PAST_FILE: &str = "decoder_with_past_model_quantized.onnx";
/// Tokenizer, at the repository root.
pub(super) const TOKENIZER_FILE: &str = "tokenizer.json";
/// The export's `config.json`, also at the repository root. Under a KiB, and the
/// only place the model describes *itself*: its family, its decoder geometry
/// and its opening and stopping token ids.
pub(super) const CONFIG_FILE: &str = "config.json";

/// The five files every translator needs, as `(repo path, file name)`. The
/// repository nests the graphs under `onnx/` and keeps the tokenizer and the
/// config at the root; on disk all are flat. One file set for **both** families,
/// which is why it is a constant and not a per-family list: `Xenova` publishes
/// `onnx/encoder_model_quantized.onnx`,
/// `onnx/decoder_model_merged_quantized.onnx` and
/// `onnx/decoder_with_past_model_quantized.onnx` under `opus-mt-pl-en` and under
/// `nllb-200-distilled-600M` alike.
pub(super) const ASSETS: &[(&str, &str)] = &[
    ("onnx/encoder_model_quantized.onnx", ENCODER_FILE),
    ("onnx/decoder_model_merged_quantized.onnx", DECODER_FILE),
    (
        "onnx/decoder_with_past_model_quantized.onnx",
        DECODER_WITH_PAST_FILE,
    ),
    ("tokenizer.json", TOKENIZER_FILE),
    ("config.json", CONFIG_FILE),
];

/// Directory every downloaded translator resolves into.
fn translators_dir() -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{TRANSLATORS_SUBDIR}"))
}

/// Directory one translator spec resolves into.
pub(super) fn translator_dir(spec: &str) -> PathBuf {
    translators_dir().join(spec)
}

/// Which HF organization an export is published under. Both families are
/// converted by the same publisher, which is why one `ASSETS` list serves both.
fn repo_of(spec: &str) -> &'static str {
    if spec == NLLB_SPEC {
        NLLB_REPO
    } else {
        OPUS_MT_REPO
    }
}

/// Where one repository is fetched from. `resolve/main` rather than `main` so a
/// spec pinned to a revision upstream resolves.
fn repo_base(spec: &str) -> String {
    format!(
        "https://huggingface.co/{}/{spec}/resolve/main/",
        repo_of(spec)
    )
}

/// Where a published file of `spec` is fetched from.
///
/// `repo_base` and the asset's repository path, in one place, so a token in
/// [`ASSETS`] and the URL it resolves to cannot drift apart.
fn asset_url(spec: &str, repo_path: &str) -> String {
    format!("{}{repo_path}", repo_base(spec))
}

/// Every file a translator directory must hold.
pub(super) const REQUIRED_FILES: [&str; 5] = [
    ENCODER_FILE,
    DECODER_FILE,
    DECODER_WITH_PAST_FILE,
    TOKENIZER_FILE,
    CONFIG_FILE,
];

/// Whether `spec`'s directory is on disk and complete, without touching the
/// network. What a weights-gated test asks before it would download anything.
#[cfg(test)]
pub(crate) fn translator_is_present(spec: &str) -> bool {
    missing_in(&translator_dir(spec)).is_none()
}

/// The first required file that is not in `dir`, or `None` when the directory is
/// a complete translator.
pub(super) fn missing_in(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return Some(dir.to_path_buf());
    }
    REQUIRED_FILES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| !path.is_file())
}

/// Download whatever of a translator's files is missing, then validate that the
/// directory is complete.
pub(super) fn ensure_available(t: &TranslatorRef) -> Result<()> {
    if t.assets.is_empty() {
        // Nothing to fetch, so this is the "did the user point at a real
        // translator directory" check, already done at resolve; repeat it so the
        // function stands on its own.
        if let Some(missing) = missing_in(&t.dir) {
            anyhow::bail!(
                "{} is not a translator directory: {} is missing",
                t.dir.display(),
                missing.display()
            );
        }
        return Ok(());
    }
    std::fs::create_dir_all(&t.dir).with_context(|| install_hint(&t.dir))?;
    for (repo_path, file_name) in &t.assets {
        let dest = t.dir.join(file_name);
        if dest.exists() {
            continue;
        }
        let url = asset_url(&t.spec, repo_path);
        log::info!(
            "translator file not found at {}, downloading {url}...",
            dest.display()
        );
        crate::model_store::download(&url, &dest)
            .with_context(|| format!("{file_name}: {}", install_hint(&t.dir)))?;
    }
    Ok(())
}

/// What to say when a translator has to be fetched into a directory that is
/// not writable — the usual case being a packaged install's root-owned models
/// directory, where only the bundled speech model is present. Same wording as
/// the voice path, because the cause and the fix are identical.
fn install_hint(dir: &Path) -> String {
    format!(
        "cannot prepare translator directory {} (packaged model dirs are \
         root-owned: install the sttd package, run once with sudo, or point \
         --translate-model at a writable directory)",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator_store::catalog::pair_for;
    use crate::translator_store::catalog::spec_for;
    use crate::translator_store::reference::pair_for_spec;

    #[test]
    fn spec_and_repository_naming_follow_upstream() {
        let pair = pair_for("pl", "en").expect("pl-en is in the catalog");
        assert_eq!(spec_for(pair), "opus-mt-pl-en");
        assert_eq!(
            repo_base(&spec_for(pair)),
            "https://huggingface.co/Xenova/opus-mt-pl-en/resolve/main/"
        );
        // Each pair is its own repository, so two pairs never share one.
        assert_ne!(
            repo_base(&spec_for(pair)),
            repo_base(&spec_for(
                pair_for("de", "en").expect("de-en is in the catalog")
            ))
        );
        // The multilingual export lives in the same publisher, so it resolves too,
        // and it is not a pair: no `opus-mt-` prefix, and nothing to split.
        assert_eq!(
            repo_base(NLLB_SPEC),
            "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/"
        );
        assert!(pair_for_spec(NLLB_SPEC).is_none());
        // The `--language auto` fallback is listed as a source only.
        assert!(pair_for("mul", "en").is_some());
        assert!(pair_for("en", "mul").is_none());
    }

    #[test]
    fn repository_urls_split_by_folder_and_the_tokenizer_stays_at_the_root() {
        // The graphs are nested under `onnx/` upstream; the tokenizer is not.
        // Getting this wrong is a 404 at first use, so it is asserted directly --
        // for **both** families, because one asset list serves both and that only
        // holds while they agree.
        for spec in ["opus-mt-pl-en", NLLB_SPEC] {
            let base = format!("https://huggingface.co/Xenova/{spec}/resolve/main/");
            assert_eq!(
                asset_url(spec, "onnx/encoder_model_quantized.onnx"),
                format!("{base}onnx/encoder_model_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "onnx/decoder_model_merged_quantized.onnx"),
                format!("{base}onnx/decoder_model_merged_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "onnx/decoder_with_past_model_quantized.onnx"),
                format!("{base}onnx/decoder_with_past_model_quantized.onnx")
            );
            assert_eq!(
                asset_url(spec, "tokenizer.json"),
                format!("{base}tokenizer.json")
            );
            // Every catalog asset resolves to a real repository path.
            for (repo_path, _) in ASSETS {
                assert!(
                    asset_url(spec, repo_path).starts_with(&base),
                    "{repo_path} must sit under {spec}'s own repository"
                );
            }
        }
    }

    #[test]
    fn translators_live_beside_the_models_never_inside_one() {
        // The directory name is what keeps `model_store`'s engine detection
        // from ever being pointed at a translator.
        assert_eq!(
            translators_dir(),
            PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{TRANSLATORS_SUBDIR}"))
        );
        assert_ne!(
            translator_dir("opus-mt-pl-en"),
            translator_dir("parakeet-ultra-int8")
        );
        assert_ne!(
            translator_dir(NLLB_SPEC),
            translator_dir("parakeet-ultra-int8")
        );
    }
}
