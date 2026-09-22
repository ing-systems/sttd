use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{DEFAULT_MODELS_DIR, MODEL_BASE_URL, model_file_for_spec, valid_model_specs};

/// Resolve a `--model` value to a filesystem path: an explicit path
/// (contains a path separator or ends in `.bin`) passes through untouched,
/// otherwise the value must be a catalog spec (see `MODEL_CATALOG`) and maps
/// to `models/ggml-<spec>.bin`.
pub(crate) fn resolve_model_path(input: &str) -> Result<PathBuf> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty --model value; expected one of: {}", valid_model_specs());
    }
    if trimmed.ends_with(".bin") || trimmed.contains('/') || trimmed.contains('\\') {
        return Ok(PathBuf::from(trimmed));
    }
    if let Some(file) = model_file_for_spec(trimmed) {
        return Ok(PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{file}")));
    }
    anyhow::bail!(
        "unknown model {trimmed:?}; expected a path (*.bin) or one of: {}",
        valid_model_specs()
    );
}

/// `model`, guaranteed to exist on return: downloads the model into
/// place if it isn't already there.
pub(crate) fn ensure_available(model: &Path) -> Result<PathBuf> {
    let path = model;
    if path.exists() {
        return Ok(path.to_path_buf());
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
    log::info!(
        "model not found at {}, downloading {url}...",
        path.display()
    );
    download(&url, path)?;
    Ok(path.to_path_buf())
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

/// One-line download progress for stderr, e.g.
/// `[==============>               ]  45.2% (360.0/800.0 MiB)`.
/// Without a total it degrades to `123.4 MiB downloaded...`.
pub(crate) fn format_download_progress(downloaded: u64, total: Option<u64>) -> String {
    const WIDTH: usize = 30;
    const MIB: f64 = 1_048_576.0;
    match total {
        Some(t) if t > 0 => {
            let frac = (downloaded as f64 / t as f64).clamp(0.0, 1.0);
            let filled = (frac * WIDTH as f64) as usize;
            let mut bar = String::with_capacity(WIDTH + 2);
            bar.push('[');
            for i in 0..WIDTH {
                if i < filled {
                    bar.push('=');
                } else if i == filled {
                    bar.push('>');
                } else {
                    bar.push(' ');
                }
            }
            bar.push(']');
            format!(
                "{bar} {:5.1}% ({:.1}/{:.1} MiB)",
                frac * 100.0,
                downloaded as f64 / MIB,
                t as f64 / MIB
            )
        }
        _ => format!("{:.1} MiB downloaded...", downloaded as f64 / MIB),
    }
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("model download request failed: {url}"))?;

    let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    if let Some(n) = total {
        log::info!("downloading {:.1} MiB...", n as f64 / 1_048_576.0);
    } else {
        log::info!("downloading (size unknown)...");
    }

    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&tmp)
        .with_context(|| format!("failed to create {}", tmp.display()))?;
    // Chunked copy (instead of `std::io::copy`) so a large model download
    // reports progress via the logger (timestamps replace the old `\r` bar).
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];
    let mut last_render = Instant::now() - Duration::from_secs(3600);
    let mut next_log_at: u64 = 64 * 1_048_576;
    // Initial 0% so even a fast download shows the bar shape.
    log::debug!("{}", format_download_progress(0, total));
    loop {
        let n = reader.read(&mut buf).context("download I/O failed")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("download I/O failed")?;
        downloaded += n as u64;
        if tty {
            if last_render.elapsed() >= Duration::from_millis(200) {
                log::debug!("{}", format_download_progress(downloaded, total));
                last_render = Instant::now();
            }
        } else if downloaded >= next_log_at {
            log::info!("{}", format_download_progress(downloaded, total));
            next_log_at += 64 * 1_048_576;
        }
    }
    if tty {
        log::debug!("{}", format_download_progress(downloaded, total));
    }
    out.flush().ok();
    drop(out);
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("failed to move {} -> {}", tmp.display(), dest.display()))?;
    log::info!(
        "downloaded {} ({:.1} MiB)",
        dest.display(),
        downloaded as f64 / 1_048_576.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODEL_BASE_URL;

    #[test]
    fn default_model_is_quantized_q8() {
        let resolved = resolve_model_path(crate::config::DEFAULT_MODEL_SPEC)
            .expect("default spec must resolve");
        let file_name = resolved
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

    #[test]
    fn specs_resolve_into_models_dir() {
        for (spec, file) in [
            ("tiny", "models/ggml-tiny.bin"),
            ("base-q5_1", "models/ggml-base-q5_1.bin"),
            ("small", "models/ggml-small.bin"),
            ("medium-q5_0", "models/ggml-medium-q5_0.bin"),
            ("large-v3-turbo-q8_0", "models/ggml-large-v3-turbo-q8_0.bin"),
            ("large-v3-turbo", "models/ggml-large-v3-turbo.bin"),
        ] {
            assert_eq!(
                resolve_model_path(spec).expect("known spec must resolve"),
                PathBuf::from(file),
                "spec={spec}"
            );
        }
        assert_eq!(
            resolve_model_path(crate::config::DEFAULT_MODEL_SPEC)
                .expect("default spec must resolve"),
            PathBuf::from("models/ggml-large-v3-turbo-q8_0.bin")
        );
    }

    #[test]
    fn paths_pass_through_untouched() {
        for path in [
            "models/custom.bin",
            "models/ggml-large-v3-turbo-q8_0.bin",
            "/abs/models/ggml-base.bin",
            "custom/ggml-tiny.bin",
        ] {
            assert_eq!(
                resolve_model_path(path).expect("path must pass through"),
                PathBuf::from(path),
                "path={path}"
            );
        }
    }

    #[test]
    fn unknown_specs_are_rejected_with_valid_list() {
        for bad in ["", "   ", "tiny-q5_0", "medium-q5_1", "large-v1-q8_0", "huge", "tiny.en"] {
            let err = resolve_model_path(bad).expect_err("must reject");
            let msg = err.to_string();
            assert!(msg.contains("large-v3-turbo-q8_0"), "msg={msg}");
        }
    }

    #[test]
    fn download_progress_renders_bar_with_known_total() {
        let total = Some(100 * 1_048_576);
        let start = format_download_progress(0, total);
        assert!(start.contains("0.0%"), "start={start}");
        assert!(start.starts_with('['), "start={start}");
        let half = format_download_progress(50 * 1_048_576, total);
        assert!(half.contains("50.0%"), "half={half}");
        assert!(half.contains("50.0/100.0 MiB"), "half={half}");
        let done = format_download_progress(100 * 1_048_576, total);
        assert!(done.contains("100.0%"), "done={done}");
        // Over-run clamps at 100%.
        let over = format_download_progress(150 * 1_048_576, total);
        assert!(over.contains("100.0%"), "over={over}");
    }

    #[test]
    fn download_progress_degrades_without_total() {
        let line = format_download_progress(10 * 1_048_576, None);
        assert!(line.contains("10.0 MiB downloaded"), "line={line}");
        let zero_total = format_download_progress(10 * 1_048_576, Some(0));
        assert!(
            zero_total.contains("10.0 MiB downloaded"),
            "zero_total={zero_total}"
        );
    }
}
