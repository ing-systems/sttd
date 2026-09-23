use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{
    parakeet_entry_for_spec, valid_model_specs, DEFAULT_MODELS_DIR,
    PARAKEET_BASE_URL, PARAKEET_TOKENIZER_URL,
};

/// Which ASR model a resolved `--model` value maps to, and where its
/// weights live on disk. Parakeet takes a directory
/// (encoder + decoder_joint + `vocab.txt` + `tokenizer.json`) — the engine
/// only accepts a directory, so there is no single-file equivalent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelRef {
    /// Model directory. `assets` lists the files `ensure_available` must
    /// fetch if missing; empty for a directory the user pointed `--model`
    /// at directly (nothing is downloaded, only validated).
    pub(crate) dir: PathBuf,
    pub(crate) assets: &'static [&'static str],
}

impl std::fmt::Display for ModelRef {
    /// The underlying filesystem location, for the `"model: {model}"` log
    /// line: the model directory.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dir.display())
    }
}

/// Resolve a `--model` value to a model location: a directory path
/// (contains a path separator) passes through untouched, otherwise the
/// value must be a catalog spec — Parakeet's (a model directory under
/// `models/`, see `PARAKEET_CATALOG`) — or the `parakeet` alias.
pub(crate) fn resolve_model(input: &str) -> Result<ModelRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty --model value; expected one of: {}", valid_model_specs());
    }
    if let Some((_, dir, assets, _)) = parakeet_entry_for_spec(trimmed) {
        return Ok(ModelRef {
            dir: PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{dir}")),
            assets,
        });
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        // A manually-prepared model directory (nothing to download, just
        // validated at load time).
        let path = PathBuf::from(trimmed);
        if path.is_dir() {
            return Ok(ModelRef { dir: path, assets: &[] });
        }
        anyhow::bail!(
            "model path {trimmed:?} is not an existing model directory; \
             expected one of: {}",
            valid_model_specs()
        );
    }
    anyhow::bail!(
        "unknown model {trimmed:?}; expected a model directory or one of: {}",
        valid_model_specs()
    );
}

/// `model`, guaranteed to exist on return: downloads any missing weights
/// into place first.
pub(crate) fn ensure_available(model: &ModelRef) -> Result<()> {
    ensure_parakeet_available(&model.dir, model.assets)
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
    std::fs::create_dir_all(dir).with_context(|| install_hint(dir))?;
    for file in assets {
        let dest = dir.join(file);
        if dest.exists() {
            continue;
        }
        let url = parakeet_asset_url(file);
        log::info!("model not found at {}, downloading {url}...", dest.display());
        download(&url, &dest)
            .with_context(|| format!("{file}: {}", install_hint(dir)))?;
    }
    Ok(())
}

/// What to say when a catalog model has to be fetched into a directory that
/// isn't writable — the usual case being `DEFAULT_MODELS_DIR` baked to the
/// root-owned `/usr/share/sttd/models` of a packaged install, where only the
/// bundled default model is present. Keeps the raw `EACCES` from
/// `create_dir_all`/`File::create` from being the whole story.
fn install_hint(dir: &Path) -> String {
    format!(
        "cannot prepare model directory {} (packaged model dirs are root-owned: \
         install the sttd package, run once with sudo, or point --model at a \
         writable directory)",
        dir.display()
    )
}

/// Download URL for one Parakeet asset filename (e.g.
/// `encoder-model.int8.onnx`). `tokenizer.json` lives in the upstream
/// NVIDIA checkpoint, everything else in the ONNX export (see
/// `PARAKEET_BASE_URL` / `PARAKEET_TOKENIZER_URL`).
pub(crate) fn parakeet_asset_url(file_name: &str) -> String {
    if file_name == "tokenizer.json" {
        return PARAKEET_TOKENIZER_URL.to_string();
    }
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
/// `File::create` every attempt). Same gap for Parakeet's 2.4 GiB
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

    #[test]
    fn default_model_is_parakeet() {
        let resolved =
            resolve_model(crate::config::DEFAULT_MODEL_SPEC).expect("default spec must resolve");
        assert_eq!(resolved.dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
        assert!(resolved.assets.contains(&"vocab.txt"));
    }

    #[test]
    fn parakeet_specs_and_alias_resolve_to_model_dir() {
        for spec in ["parakeet-tdt-0.6b-v3-int8", "parakeet"] {
            let resolved = resolve_model(spec).expect("parakeet spec must resolve");
            assert_eq!(resolved.dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
            assert!(resolved.assets.contains(&"vocab.txt"));
            assert!(resolved.assets.iter().any(|f| f.contains("int8")));
        }
        let resolved =
            resolve_model(PARAKEET_DEFAULT_SPEC).expect("default parakeet spec must resolve");
        assert_eq!(resolved.dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
        assert_eq!(resolved.assets.len(), 4);
    }

    #[test]
    fn parakeet_fp32_spec_resolves_with_external_data_file() {
        let resolved =
            resolve_model("parakeet-tdt-0.6b-v3-fp32").expect("fp32 spec must resolve");
        assert_eq!(resolved.dir, PathBuf::from("models/parakeet-tdt-0.6b-v3-fp32"));
        assert!(resolved.assets.contains(&"encoder-model.onnx.data"));
    }

    #[test]
    fn parakeet_asset_urls_derive_from_base_url() {
        for file in ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"] {
            assert_eq!(parakeet_asset_url(file), format!("{PARAKEET_BASE_URL}{file}"));
        }
    }

    #[test]
    fn parakeet_tokenizer_url_points_upstream() {
        use crate::config::PARAKEET_TOKENIZER_URL;
        assert_eq!(parakeet_asset_url("tokenizer.json"), PARAKEET_TOKENIZER_URL);
        assert!(PARAKEET_TOKENIZER_URL.contains("nvidia/parakeet-tdt-0.6b-v3"));
    }

    #[test]
    fn nonexistent_directory_path_is_rejected() {
        let err = resolve_model("models/does-not-exist-dir/model").expect_err("must reject");
        assert!(err.to_string().contains("not an existing model directory"), "msg={err}");
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
        assert_eq!(resolved, ModelRef { dir: tmp.clone(), assets: &[] });
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn unknown_specs_are_rejected_with_valid_list() {
        for bad in ["", "   ", "huge", "tiny", "small", "large-v3-turbo-q8_0"] {
            let err = resolve_model(bad).expect_err("must reject");
            let msg = err.to_string();
            assert!(msg.contains("parakeet"), "msg={msg}");
        }
    }

    #[test]
    fn install_hint_names_the_directory_and_the_way_out() {
        // The raw EACCES from a root-owned /usr/share/sttd/models must not be
        // the whole story: name the directory, then the three ways forward.
        let hint = install_hint(Path::new("/usr/share/sttd/models/parakeet-tdt-0.6b-v3-fp32"));
        assert!(
            hint.contains("/usr/share/sttd/models/parakeet-tdt-0.6b-v3-fp32"),
            "hint={hint}"
        );
        for way_out in ["sttd package", "sudo", "--model"] {
            assert!(hint.contains(way_out), "missing {way_out:?} in hint={hint}");
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
