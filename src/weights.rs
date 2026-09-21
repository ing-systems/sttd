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
    if !file_name.starts_with("ggml-") || !file_name.ends_with(".bin") {
        anyhow::bail!(
            "model not found at {} and its filename doesn't look like a whisper.cpp \
             model (expected ggml-<name>.bin); download it manually",
            path.display()
        );
    }
    let url = format!("{MODEL_BASE_URL}{file_name}");
    eprintln!(
        "[sttd] model not found at {}, downloading {url}...",
        path.display()
    );
    download(&url, &path)?;
    Ok(path)
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
