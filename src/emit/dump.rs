//! The `--dump-dir` side effect: owner-only WAV/TXT pairs and their
//! retention.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::DUMP_KEEP_FILES;
use crate::fsutil::write_owner_only_text;

/// Write `body` as `<dump_dir>/<tag>.txt`, the WAV's partner: the emitted
/// text, or for an utterance that was dropped rather than printed the reason
/// why, so `STTD_DUMP_DIR` output stays correlatable with the stderr log
/// (previously a dropped utterance left only a WAV with no indication of what
/// became of it or why). A no-op without a dump directory.
pub(super) fn dump_text(dump_dir: &Option<PathBuf>, tag: &str, body: &str) {
    if let Some(dir) = dump_dir {
        let txt_path = dir.join(format!("{tag}.txt"));
        if let Err(e) = write_owner_only_text(&txt_path, body) {
            log::warn!("failed to write {}: {e:?}", txt_path.display());
        }
    }
}

/// Ensure `dir` exists with owner-only permissions (0700 on unix), like
/// Murmure's recordings dir. Creation errors propagate (fatal at startup);
/// a chmod failure only warns — transcription never depends on it.
pub(super) fn ensure_owner_only_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir)
            .with_context(|| format!("failed to stat {}", dir.display()))?
            .permissions()
            .mode()
            & 0o777;
        if mode != 0o700 {
            if let Err(e) = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)) {
                log::warn!("failed to chmod 0700 {}: {e:?}", dir.display());
            }
        }
    }
    Ok(())
}

/// Delete utterance pairs past `DUMP_KEEP_FILES`, newest-first by tag name
/// (`utt_%06d` sorts chronologically). Observability hygiene only: I/O
/// errors warn and keep going, never fail transcription.
pub(super) fn prune_dump_dir(dir: &Path) {
    let stems = match dump_stems(dir) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("dump prune: failed to list {}: {e:?}", dir.display());
            return;
        }
    };
    for stem in stems_to_prune(stems, DUMP_KEEP_FILES) {
        for ext in ["wav", "txt"] {
            let path = dir.join(format!("{stem}.{ext}"));
            // `remove_file` already reports a missing file as `NotFound`; the
            // `exists()` probe this replaces only doubled the syscalls.
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("dump prune: failed to remove {}: {e:?}", path.display()),
            }
        }
    }
}

/// Sorted `utt_NNNNNN` stems of the pairs in `dir` (the `.wav` names; the
/// matching `.txt` is implied).
fn dump_stems(dir: &Path) -> Result<Vec<String>> {
    let mut stems = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("utt_") && name.ends_with(".wav") {
            stems.push(name.trim_end_matches(".wav").to_string());
        }
    }
    stems.sort();
    Ok(stems)
}

/// Pure selection: which stems to delete keeping the newest `keep`. Takes
/// the names already sorted by the caller (`dump_stems`), oldest first.
fn stems_to_prune(mut stems: Vec<String>, keep: usize) -> Vec<String> {
    if stems.len() > keep {
        stems.truncate(stems.len() - keep);
        stems
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::test_support::*;

    use crate::emit::system::{EmissionPolicy, SystemEmitter};
    use crate::emit::{Emission, Emitter};

    #[test]
    fn prune_keeps_newest_and_drops_oldest() {
        let stems: Vec<String> = (0..5).map(|i| format!("utt_{i:06}")).collect();
        assert!(stems_to_prune(stems.clone(), 5).is_empty());
        assert!(stems_to_prune(stems.clone(), 10).is_empty());
        assert_eq!(
            stems_to_prune(stems.clone(), 2),
            vec![
                "utt_000000".to_string(),
                "utt_000001".to_string(),
                "utt_000002".to_string()
            ]
        );
        assert!(stems_to_prune(Vec::new(), 2).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn dump_dir_and_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("sttd-emit-perms-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        ensure_owner_only_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "dir mode={mode:o}");
        let txt = dir.join("utt_000001.txt");
        write_owner_only_text(&txt, "Dzień dobry").unwrap();
        let mode = std::fs::metadata(&txt).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "file mode={mode:o}");
        assert_eq!(std::fs::read_to_string(&txt).unwrap(), "Dzień dobry");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dump_retention_prunes_oldest_pairs() {
        use crate::config::DUMP_KEEP_FILES;
        let dir = std::env::temp_dir().join(format!("sttd-emit-retain-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let evidence = evidence();
        {
            let mut emitter = SystemEmitter::new(EmissionPolicy {
                dump_dir: Some(dir.clone()),
                ..policy_off()
            })
            .unwrap();
            let pcm = vec![0.01f32; 1600];
            for i in 0..(DUMP_KEEP_FILES + 5) {
                let tag = format!("utt_{i:06}");
                emitter.emit(Emission {
                    tag: &tag,
                    pcm: &pcm,
                    text: "Dzień dobry",
                    verdict: gate().decide_transcript("Dzień dobry"),
                    evidence: &evidence,
                });
            }
            // Drop at scope end flushes the async queue.
        }
        let wavs: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .collect();
        assert_eq!(wavs.len(), DUMP_KEEP_FILES, "kept={}", wavs.len());
        assert!(!dir.join("utt_000000.wav").exists());
        assert!(!dir.join("utt_000000.txt").exists());
        let newest = format!("utt_{:06}.wav", DUMP_KEEP_FILES + 4);
        assert!(dir.join(newest).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
