//! Owner-only file helpers shared by the dump directory, `--file` WAV output
//! and the last-transcript file.

use std::path::Path;

use anyhow::{Context, Result};

/// Create/truncate `path` with owner-only permissions (0600 on unix). Mic
/// audio and transcripts are sensitive; `File::create` honors umask (often
/// 0644) and keeps a pre-existing path's mode — neither is acceptable here.
pub(crate) fn owner_only_file(path: &Path) -> Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

/// Write `contents` to `path` with owner-only permissions (see
/// `owner_only_file`).
pub(crate) fn write_owner_only_text(path: &Path, contents: &str) -> Result<()> {
    use std::io::Write;
    let mut file = owner_only_file(path)?;
    file.write_all(contents.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}
