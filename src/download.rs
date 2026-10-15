// ---------------------------------------------------------------------------
// Fetching weights: the HTTP download and the per-file loop every store runs
// ---------------------------------------------------------------------------
//
// Three stores fetch files into a directory under the models dir — speech
// models (`model_store`), translators (`translator_store`) and piper voices
// (`voice_store`). What a store owns is *which* files and from *where*; the
// fetch itself is the same for all three and lives here: skip what is
// present, download the rest through a `.part` file, and turn a permission
// failure into a message that names the way out.

use std::path::Path;

use anyhow::{Context, Result};

/// What a store is fetching into, for its log lines and its error message.
pub(crate) struct Target<'a> {
    pub(crate) dir: &'a Path,
    /// The directory's kind in the error hint: "model", "translator", "voice".
    pub(crate) kind: &'a str,
    /// The flag that points sttd at a writable directory instead.
    pub(crate) flag: &'a str,
    /// What the "not found" log line calls one missing file.
    pub(crate) file_noun: &'a str,
}

/// Download every `(url, file name)` in `files` that `target.dir` does not hold
/// yet, creating the directory first. Files already present are never
/// re-fetched, so an interrupted run resumes at the first missing file.
pub(crate) fn fetch_missing<'f>(
    target: &Target<'_>,
    files: impl IntoIterator<Item = (String, &'f str)>,
) -> Result<()> {
    let hint = || install_hint(target.dir, target.kind, target.flag);
    std::fs::create_dir_all(target.dir).with_context(hint)?;
    for (url, file_name) in files {
        let dest = target.dir.join(file_name);
        if dest.exists() {
            continue;
        }
        log::info!(
            "{} not found at {}, downloading {url}...",
            target.file_noun,
            dest.display()
        );
        download(&url, &dest).with_context(|| format!("{file_name}: {}", hint()))?;
    }
    Ok(())
}

/// The error context for a directory sttd cannot create or write: the raw
/// EACCES from a root-owned `/usr/share/sttd/models` is not the whole story.
pub(crate) fn install_hint(dir: &Path, kind: &str, flag: &str) -> String {
    format!(
        "cannot prepare {kind} directory {} (packaged model dirs are root-owned: \
         install the sttd package, run once with sudo, or point {flag} at a \
         writable directory)",
        dir.display()
    )
}

/// One-line download progress for stderr, e.g.
/// `[==============>               ]  45.2% (360.0/800.0 MiB)`.
/// Without a total it degrades to `123.4 MiB downloaded...`.
fn format_download_progress(downloaded: u64, total: Option<u64>) -> String {
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
/// `File::create` every attempt). This bites on the 619 MiB encoder; not
/// fixed here.
pub(crate) fn download(url: &str, dest: &Path) -> Result<()> {
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

    #[test]
    fn install_hint_names_the_directory_and_the_way_out() {
        // The raw EACCES from a root-owned /usr/share/sttd/models must not be
        // the whole story: name the directory, then the three ways forward.
        let hint = install_hint(
            Path::new("/usr/share/sttd/models/parakeet-ultra-int8"),
            "model",
            "--model",
        );
        assert!(
            hint.contains("/usr/share/sttd/models/parakeet-ultra-int8"),
            "hint={hint}"
        );
        for way_out in ["sttd package", "sudo", "--model"] {
            assert!(hint.contains(way_out), "missing {way_out:?} in hint={hint}");
        }
    }

    #[test]
    fn every_store_names_its_own_kind_and_flag() {
        for (kind, flag) in [("translator", "--translate-model"), ("voice", "--voice")] {
            let hint = install_hint(Path::new("/x"), kind, flag);
            assert!(
                hint.contains(&format!("{kind} directory /x")),
                "hint={hint}"
            );
            assert!(hint.contains(flag), "hint={hint}");
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
