use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{DEFAULT_MODEL_PATH, MODEL_BASE_URL};

/// The whisper model file this run should use, without touching the disk:
/// `WHISPER_MODEL_PATH` if set, the compiled-in default otherwise.
pub(crate) fn configured_path() -> PathBuf {
    std::env::var_os("WHISPER_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

/// `configured_path`, guaranteed to exist on return: downloads the model into
/// place if it isn't already there.
pub(crate) fn ensure_available() -> Result<PathBuf> {
    let path = configured_path();
    if path.exists() {
        return Ok(path);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    // Derive the download URL from the filename so `WHISPER_MODEL_PATH` can
    // point at any ggml model in this HF repo, not just the compiled-in
    // default. Previously the URL was hardcoded to `ggml-base.bin`, so
    // pointing the env var at a missing `ggml-large-v3.bin` would silently
    // download `base` under that filename.
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("model path has no filename: {}", path.display()))?;
    let Some(url) = download_url(file_name) else {
        anyhow::bail!(
            "model not found at {} and its filename doesn't look like a whisper.cpp \
             model (expected ggml-<name>.bin); download it manually",
            path.display()
        );
    };
    eprintln!(
        "[sttd] model not found at {}, downloading {url}...",
        path.display()
    );
    download(&url, &path)?;
    Ok(path)
}

/// Download URL for a `ggml-*.bin` model filename (full or quantized,
/// e.g. `ggml-large-v3-turbo-q8_0.bin`), or `None` when the filename
/// doesn't look like a whisper.cpp model.
pub(crate) fn download_url(file_name: &str) -> Option<String> {
    if !file_name.starts_with("ggml-") || !file_name.ends_with(".bin") {
        return None;
    }
    Some(format!("{MODEL_BASE_URL}{file_name}"))
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("model download request failed: {url}"))?;

    let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    if let Some(n) = total {
        eprintln!("[sttd] downloading {:.1} MiB...", n as f64 / 1_048_576.0);
    } else {
        eprintln!("[sttd] downloading (size unknown)...");
    }

    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&tmp)
        .with_context(|| format!("failed to create {}", tmp.display()))?;
    let bytes = std::io::copy(&mut reader, &mut out).context("download I/O failed")?;
    use std::io::Write;
    out.flush().ok();
    drop(out);
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("failed to move {} -> {}", tmp.display(), dest.display()))?;
    eprintln!(
        "[sttd] downloaded {} ({:.1} MiB)",
        dest.display(),
        bytes as f64 / 1_048_576.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_quantized_q8() {
        let file_name = Path::new(DEFAULT_MODEL_PATH)
            .file_name()
            .and_then(|n| n.to_str())
            .expect("default model path has a filename");
        assert_eq!(file_name, "ggml-large-v3-turbo-q8_0.bin");
        let url = download_url(file_name).expect("default model must have a download URL");
        assert_eq!(url, format!("{MODEL_BASE_URL}{file_name}"));
    }

    #[test]
    fn quantized_filenames_resolve_to_hf_urls() {
        for file_name in [
            "ggml-large-v3-turbo-q8_0.bin",
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3-turbo.bin",
        ] {
            let url = download_url(file_name).expect("ggml-*.bin must resolve");
            assert_eq!(url, format!("{MODEL_BASE_URL}{file_name}"));
        }
    }

    #[test]
    fn non_ggml_filenames_are_rejected() {
        for file_name in [
            "whisper-large-v3-turbo-q8_0.gguf",
            "model.bin",
            "ggml-large-v3-turbo-q8_0.gguf",
            "",
        ] {
            assert!(download_url(file_name).is_none(), "must reject {file_name:?}");
        }
    }
}
