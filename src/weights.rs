use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{
    parakeet_entry_for_spec, valid_model_specs, DEFAULT_MODELS_DIR, PARAKEET_BASE_URL,
};
#[cfg(feature = "whisper")]
use crate::config::{MODEL_BASE_URL, model_file_for_spec};

/// Which ASR engine a resolved `--model` value maps to, and where its
/// weights live on disk. Whisper's engine takes a single `ggml-*.bin` file
/// (only with the `whisper` Cargo feature); Parakeet's takes a directory
/// (encoder + decoder_joint + `vocab.txt`) —
/// `parakeet-rs::ParakeetTDT::from_pretrained` only accepts a directory, so
/// there is no single-file equivalent to fall back to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelRef {
    /// `ggml-*.bin` for whisper.cpp (only with the `whisper` Cargo feature).
    #[cfg(feature = "whisper")]
    Whisper(PathBuf),
    /// Model directory for Parakeet. `assets` lists the files
    /// `ensure_available` must fetch if missing; empty for a directory the
    /// user pointed `--model` at directly (nothing is downloaded, only
    /// validated).
    Parakeet {
        dir: PathBuf,
        assets: &'static [&'static str],
    },
}

impl std::fmt::Display for ModelRef {
    /// The underlying filesystem location, for the `"model: {model}"` log
    /// line: a `.bin` path for whisper, the model directory for Parakeet.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "whisper")]
            Self::Whisper(path) => write!(f, "{}", path.display()),
            Self::Parakeet { dir, .. } => write!(f, "{}", dir.display()),
        }
    }
}

/// Resolve a `--model` value to an engine + location: an explicit `.bin`
/// path or directory path (contains a path separator) passes through
/// untouched, otherwise the value must be a catalog spec — whisper's
/// (`ggml-<spec>.bin`, only with the `whisper` Cargo feature) or Parakeet's
/// (a model directory under `models/`, see `PARAKEET_CATALOG`) — or the
/// `parakeet` alias.
pub(crate) fn resolve_model(input: &str) -> Result<ModelRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty --model value; expected one of: {}", valid_model_specs());
    }
    #[cfg(feature = "whisper")]
    if trimmed.ends_with(".bin") {
        return Ok(ModelRef::Whisper(PathBuf::from(trimmed)));
    }
    #[cfg(not(feature = "whisper"))]
    if trimmed.ends_with(".bin") {
        anyhow::bail!(
            "whisper model {trimmed:?} needs the `whisper` Cargo feature \
             (rebuild with `--features whisper`); available: {}",
            valid_model_specs()
        );
    }
    #[cfg(feature = "whisper")]
    if let Some(file) = model_file_for_spec(trimmed) {
        return Ok(ModelRef::Whisper(PathBuf::from(format!(
            "{DEFAULT_MODELS_DIR}/{file}"
        ))));
    }
    if let Some((_, dir, assets, _)) = parakeet_entry_for_spec(trimmed) {
        return Ok(ModelRef::Parakeet {
            dir: PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{dir}")),
            assets,
        });
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        // A manually-prepared model: a directory is Parakeet (nothing to
        // download, just validated at load time), anything else used to be
        // treated as a whisper `.bin` path even without the extension.
        let path = PathBuf::from(trimmed);
        if path.is_dir() {
            return Ok(ModelRef::Parakeet { dir: path, assets: &[] });
        }
        #[cfg(feature = "whisper")]
        return Ok(ModelRef::Whisper(path));
        #[cfg(not(feature = "whisper"))]
        anyhow::bail!(
            "model path {trimmed:?} is not an existing Parakeet directory, and \
             whisper support is not compiled in (rebuild with `--features \
             whisper`); expected one of: {}",
            valid_model_specs()
        );
    }
    anyhow::bail!(
        "unknown model {trimmed:?}; expected a path (*.bin with the `whisper` \
         feature, or a Parakeet model directory) or one of: {}",
        valid_model_specs()
    );
}

/// `model`, guaranteed to exist on return: downloads any missing weights
/// into place first.
pub(crate) fn ensure_available(model: &ModelRef) -> Result<()> {
    match model {
        #[cfg(feature = "whisper")]
        ModelRef::Whisper(path) => ensure_whisper_available(path),
        ModelRef::Parakeet { dir, assets } => ensure_parakeet_available(dir, assets),
    }
}

/// Whisper: downloads the model into place if it isn't already there.
/// Whisper-only (see the `whisper` Cargo feature).
#[cfg(feature = "whisper")]
fn ensure_whisper_available(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
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
    Ok(())
}

/// Parakeet: downloads any of `assets` missing from `dir`. An empty
/// `assets` (user-supplied directory) is only validated, never fetched —
/// see `resolve_model`.
fn ensure_parakeet_available(dir: &Path, assets: &'static [&'static str]) -> Result<()> {
    if assets.is_empty() {
        if !dir.is_dir() {
            anyhow::bail!(
                "Parakeet model directory not found: {} (expected an encoder, a \
                 decoder_joint and vocab.txt)",
                dir.display()
            );
        }
        if !dir.join("vocab.txt").exists() {
            anyhow::bail!(
                "{} is missing vocab.txt; not a valid Parakeet model directory",
                dir.display()
            );
        }
        return Ok(());
    }
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    for file in assets {
        let dest = dir.join(file);
        if dest.exists() {
            continue;
        }
        let url = parakeet_asset_url(file);
        log::info!("model not found at {}, downloading {url}...", dest.display());
        download(&url, &dest)?;
    }
    Ok(())
}

/// Download URL for a `ggml-*.bin` model filename (full or quantized,
/// e.g. `ggml-large-v3-turbo-q8_0.bin`), or `None` when the filename
/// doesn't look like a whisper.cpp model. Whisper-only (see the `whisper`
/// Cargo feature).
#[cfg(feature = "whisper")]
pub(crate) fn download_url(file_name: &str) -> Option<String> {
    if !file_name.starts_with("ggml-") || !file_name.ends_with(".bin") {
        return None;
    }
    Some(format!("{MODEL_BASE_URL}{file_name}"))
}

/// Download URL for one Parakeet asset filename (e.g.
/// `encoder-model.int8.onnx`).
pub(crate) fn parakeet_asset_url(file_name: &str) -> String {
    format!("{PARAKEET_BASE_URL}{file_name}")
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

/// Note: no resume support — an interrupted download restarts from zero on
/// the next `ensure_available` call (`.part` is recreated with
/// `File::create` every attempt). Same gap for whisper's 1.5 GiB
/// `ggml-large-v3-turbo.bin` and Parakeet's 2.4 GiB
/// `encoder-model.onnx.data`; not fixed here.
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
    use crate::config::{PARAKEET_BASE_URL, PARAKEET_DEFAULT_SPEC};
    #[cfg(feature = "whisper")]
    use crate::config::MODEL_BASE_URL;

    #[test]
    fn default_model_is_parakeet() {
        let resolved =
            resolve_model(crate::config::DEFAULT_MODEL_SPEC).expect("default spec must resolve");
        let ModelRef::Parakeet { dir, assets } = resolved else {
            panic!("default spec must resolve to a Parakeet model");
        };
        assert_eq!(dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
        assert!(assets.contains(&"vocab.txt"));
    }

    #[cfg(feature = "whisper")]
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

    #[cfg(feature = "whisper")]
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

    #[cfg(feature = "whisper")]
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
                resolve_model(spec).expect("known spec must resolve"),
                ModelRef::Whisper(PathBuf::from(file)),
                "spec={spec}"
            );
        }
    }

    #[cfg(not(feature = "whisper"))]
    #[test]
    fn whisper_specs_are_rejected_without_feature() {
        for spec in ["tiny", "small", "large-v3-turbo-q8_0"] {
            let err = resolve_model(spec).expect_err("whisper spec must reject");
            assert!(
                err.to_string().contains("whisper") || err.to_string().contains("unknown model"),
                "msg={}",
                err
            );
        }
        for path in ["models/custom.bin", "models/ggml-small.bin"] {
            let err = resolve_model(path).expect_err("whisper path must reject");
            assert!(
                err.to_string().contains("whisper"),
                "msg={}",
                err
            );
        }
    }

    #[test]
    fn parakeet_specs_and_alias_resolve_to_model_dir() {
        for spec in ["parakeet-tdt-0.6b-v3-int8", "parakeet"] {
            let resolved = resolve_model(spec).expect("parakeet spec must resolve");
            let ModelRef::Parakeet { dir, assets } = resolved else {
                panic!("spec={spec} must resolve to Parakeet");
            };
            assert_eq!(dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
            assert!(assets.contains(&"vocab.txt"));
            assert!(assets.iter().any(|f| f.contains("int8")));
        }
        let ModelRef::Parakeet { dir, assets } =
            resolve_model(PARAKEET_DEFAULT_SPEC).expect("default parakeet spec must resolve")
        else {
            panic!("must resolve to Parakeet");
        };
        assert_eq!(dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
        assert_eq!(assets.len(), 3);
    }

    #[test]
    fn parakeet_fp32_spec_resolves_with_external_data_file() {
        let ModelRef::Parakeet { dir, assets } =
            resolve_model("parakeet-tdt-0.6b-v3-fp32").expect("fp32 spec must resolve")
        else {
            panic!("must resolve to Parakeet");
        };
        assert_eq!(dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-fp32"));
        assert!(assets.contains(&"encoder-model.onnx.data"));
    }

    #[test]
    fn parakeet_asset_urls_derive_from_base_url() {
        for file in ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"] {
            assert_eq!(parakeet_asset_url(file), format!("{PARAKEET_BASE_URL}{file}"));
        }
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn paths_pass_through_untouched() {
        for path in ["models/custom.bin", "models/ggml-large-v3-turbo-q8_0.bin", "/abs/models/ggml-base.bin", "custom/ggml-tiny.bin"] {
            assert_eq!(
                resolve_model(path).expect("path must pass through"),
                ModelRef::Whisper(PathBuf::from(path)),
                "path={path}"
            );
        }
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn nonexistent_directory_path_is_treated_as_whisper() {
        // A path with a separator that isn't an existing directory falls
        // back to whisper (matches the pre-Parakeet behavior for any
        // path-like --model value, including ones without .bin).
        assert_eq!(
            resolve_model("models/does-not-exist-dir/model").expect("path must resolve"),
            ModelRef::Whisper(PathBuf::from("models/does-not-exist-dir/model"))
        );
    }

    #[cfg(not(feature = "whisper"))]
    #[test]
    fn nonexistent_directory_path_is_rejected_without_whisper() {
        let err = resolve_model("models/does-not-exist-dir/model").expect_err("must reject");
        assert!(err.to_string().contains("whisper"), "msg={err}");
    }

    #[test]
    fn existing_directory_path_resolves_to_parakeet_with_no_assets() {
        let tmp = std::env::temp_dir().join(format!(
            "sttd-weights-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let resolved = resolve_model(tmp.to_str().unwrap()).expect("dir path must resolve");
        assert_eq!(
            resolved,
            ModelRef::Parakeet {
                dir: tmp.clone(),
                assets: &[],
            }
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn unknown_specs_are_rejected_with_valid_list() {
        for bad in ["", "   ", "tiny-q5_0", "medium-q5_1", "large-v1-q8_0", "huge", "tiny.en"] {
            let err = resolve_model(bad).expect_err("must reject");
            let msg = err.to_string();
            assert!(msg.contains("parakeet"), "msg={msg}");
        }
        #[cfg(feature = "whisper")]
        for bad in ["tiny-q5_0", "medium-q5_1"] {
            let err = resolve_model(bad).expect_err("must reject");
            assert!(err.to_string().contains("large-v3-turbo-q8_0"), "msg={err}");
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
