// ---------------------------------------------------------------------------
// Voices: the piper voice index, `--voice` resolution and voice download.
// ---------------------------------------------------------------------------
//
// One flag, `--voice`, picks a piper TTS voice for `--speak`. The catalog is
// not a static table like the speech models': `rhasspy/piper-voices` publishes
// ~1300 voices, so the index is upstream's own `voices.json` (~245 KiB), cached
// under the models directory and fetched on first use. That file maps every
// voice key to the two files it needs (`.onnx` + `.onnx.json`) with sizes and
// digests, so resolution and `--list-voices` never guess a repository path.
//
// Layout mirrors `model_store` deliberately: a key resolves into
// `<models>/voices/<key>/` holding the two files flat, whatever nesting the
// repository uses upstream (`pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx`).
// The engine never sees the repository layout, and no renaming happens after
// download.
//
// Deliberately *not* part of `model_store`: a voice is not a speech engine, so
// it must not be reachable through `--model`, and `model_store`'s engine
// detection (`EngineKind::marker_file`) must never classify one.
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::config::{DEFAULT_MODELS_DIR, PIPER_VOICES_REPO, VOICES_INDEX_FILE, VOICES_SUBDIR};

/// Where a published file is fetched from. `resolve/main` rather than `main`
/// so a voice pinned to a revision upstream resolves.
pub(crate) fn repo_url(repo_path: &str) -> String {
    format!("https://huggingface.co/{PIPER_VOICES_REPO}/resolve/main/{repo_path}")
}

/// Directory holding every downloaded voice, and the cached index.
pub(crate) fn voices_dir() -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{VOICES_SUBDIR}"))
}

/// Directory one voice key resolves into.
pub(crate) fn voice_dir(key: &str) -> PathBuf {
    voices_dir().join(key)
}

/// What upstream says about one file of a voice.
#[derive(Debug, Clone, Deserialize)]
struct VoiceFile {
    #[serde(default)]
    size_bytes: u64,
}

/// One voice's row in `voices.json`.
///
/// Only the fields sttd reads. `phoneme_type` and `espeak.voice` are *not* here:
/// they live in the per-voice `.onnx.json`, which is not downloaded until the
/// voice is actually used — so `--list-voices` cannot filter by them and the
/// compatibility check happens at load (`piper::VoiceConfig`).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VoiceRow {
    pub(crate) key: String,
    #[serde(default)]
    pub(crate) language: VoiceLanguage,
    #[serde(default)]
    pub(crate) quality: String,
    #[serde(default)]
    pub(crate) num_speakers: u32,
    #[serde(default)]
    pub(crate) aliases: Vec<String>,
    #[serde(default)]
    files: BTreeMap<String, VoiceFile>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct VoiceLanguage {
    #[serde(default)]
    pub(crate) code: String,
    #[serde(default)]
    pub(crate) name_english: String,
}

impl VoiceRow {
    /// Repository path of the ONNX graph, e.g.
    /// `pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx`.
    fn onnx_path(&self) -> Option<&str> {
        self.file_path(".onnx")
    }

    /// Repository path of the config, e.g. `.../pl_PL-gosia-medium.onnx.json`.
    ///
    /// Matched separately from the graph because `.onnx.json` also ends in
    /// `.json` while the graph's own name ends in `.onnx`; the two suffixes
    /// cannot collide, so an exact suffix test is enough.
    fn config_path(&self) -> Option<&str> {
        self.file_path(".onnx.json")
    }

    fn file_path(&self, suffix: &str) -> Option<&str> {
        self.files
            .keys()
            .find(|p| p.ends_with(suffix))
            .map(String::as_str)
    }

    fn onnx_size(&self) -> u64 {
        self.onnx_path()
            .and_then(|p| self.files.get(p))
            .map(|f| f.size_bytes)
            .unwrap_or(0)
    }

    /// Whether this voice is something sttd can actually run: espeak
    /// phonemization, single speaker. Multi-speaker graphs need a `sid` input
    /// and a speaker choice, which `--voice` does not offer yet, so
    /// `--list-voices` marks them and `piper::PiperVoice::load` refuses one.
    pub(crate) const fn multi_speaker(&self) -> bool {
        self.num_speakers > 1
    }
}

/// The cached `voices.json`, keyed by voice key.
pub(crate) struct VoiceIndex {
    rows: BTreeMap<String, VoiceRow>,
}

impl VoiceIndex {
    /// The index, from the cache when it is there and freshly fetched when it
    /// is not. A cache that will not parse is refetched once: a truncated file
    /// (interrupted write, or a hand edit) should not need a manual delete.
    pub(crate) fn load() -> Result<Self> {
        let path = voices_dir().join(VOICES_INDEX_FILE);
        if let Some(index) = Self::read(&path) {
            return Ok(index);
        }
        log::info!("fetching the piper voice index from huggingface.co/{PIPER_VOICES_REPO}...",);
        std::fs::create_dir_all(voices_dir())
            .with_context(|| format!("failed to create {}", voices_dir().display()))?;
        let url = repo_url(VOICES_INDEX_FILE);
        crate::model_store::download(&url, &path)
            .with_context(|| format!("failed to download the voice index from {url}"))?;
        Self::read(&path).ok_or_else(|| {
            anyhow::anyhow!(
                "{} did not parse as a voice index; delete it to refetch",
                path.display()
            )
        })
    }

    fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        match serde_json::from_str::<BTreeMap<String, VoiceRow>>(&text) {
            Ok(rows) => Some(Self { rows }),
            Err(e) => {
                log::warn!("voice index {} unusable ({e}); refetching", path.display());
                None
            }
        }
    }

    /// The row `input` names: a voice key, or one of its upstream aliases
    /// (`pl-mls_6892-low` for `pl_PL-mls_6892-low`).
    pub(crate) fn row(&self, input: &str) -> Option<&VoiceRow> {
        let input = input.trim();
        if let Some(row) = self.rows.get(input) {
            return Some(row);
        }
        self.rows
            .values()
            .find(|r| r.aliases.iter().any(|a| a == input))
    }

    /// Every row, or those matching `filter` case-insensitively against the
    /// key, the language code or the language name — the substring-match idiom
    /// `--list-devices` already uses, which is what keeps an index this size
    /// usable.
    pub(crate) fn filtered(&self, filter: Option<&str>) -> Vec<&VoiceRow> {
        let needle = filter.map(str::trim).filter(|f| !f.is_empty());
        self.rows
            .values()
            .filter(|r| match needle {
                None => true,
                Some(f) => {
                    let f = f.to_ascii_lowercase();
                    r.key.to_ascii_lowercase().contains(&f)
                        || r.language.code.to_ascii_lowercase().contains(&f)
                        || r.language.name_english.to_ascii_lowercase().contains(&f)
                }
            })
            .collect()
    }
}

/// A resolved `--voice`: where a voice's two files live, and what still has to
/// be fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoiceRef {
    /// Voice key (`pl_PL-gosia-medium`), or the directory name for a
    /// user-supplied path.
    pub(crate) key: String,
    pub(crate) dir: PathBuf,
    /// ONNX graph inside `dir`.
    pub(crate) onnx: PathBuf,
    /// Config inside `dir`.
    pub(crate) config: PathBuf,
    /// `(repo path, file name)` pairs to fetch when missing. Empty for a
    /// user-supplied directory, which is validated but never downloaded into.
    assets: Vec<(String, String)>,
}

impl VoiceRef {
    /// Both files present, i.e. the voice is installed. The `--list-voices`
    /// `I` marker: "installed", not merely "known".
    pub(crate) fn is_installed(&self) -> bool {
        self.onnx.exists() && self.config.exists()
    }
}

impl std::fmt::Display for VoiceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key)
    }
}

/// Resolve a `--voice` value: a directory path (contains a path separator)
/// passes through untouched and is only validated; anything else must be a
/// voice key or alias in the index.
pub(crate) fn resolve_voice(input: &str, index: &VoiceIndex) -> Result<VoiceRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty --voice value; expected a piper voice key (see --list-voices)");
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        let dir = PathBuf::from(trimmed);
        if !dir.is_dir() {
            anyhow::bail!(
                "voice path {trimmed:?} is not an existing voice directory; expected one \
                 holding a .onnx graph and its .onnx.json config"
            );
        }
        let (onnx, config) = voice_files_in(&dir)?.ok_or_else(|| {
            anyhow::anyhow!(
                "{} holds no voice: expected exactly one .onnx graph and its .onnx.json config",
                dir.display()
            )
        })?;
        // The directory's own name is the voice's identity on a `--voice <path>`
        // run: there is no upstream key for it, and the name is what the log
        // line and the dump file name use.
        let key = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| trimmed.to_string());
        return Ok(VoiceRef {
            key,
            dir,
            onnx,
            config,
            assets: Vec::new(),
        });
    }
    let row = index.row(trimmed).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown voice {trimmed:?}; run `sttd --list-voices` for the catalog \
             (upstream: https://huggingface.co/{PIPER_VOICES_REPO}/tree/main)"
        )
    })?;
    let (onnx_repo, config_repo) = row.onnx_path().zip(row.config_path()).ok_or_else(|| {
        anyhow::anyhow!("voice {}: index row names no .onnx + .onnx.json", row.key)
    })?;
    let dir = voice_dir(&row.key);
    Ok(VoiceRef {
        key: row.key.clone(),
        onnx: dir.join(format!("{}.onnx", row.key)),
        config: dir.join(format!("{}.onnx.json", row.key)),
        dir,
        assets: vec![
            (onnx_repo.to_string(), format!("{}.onnx", row.key)),
            (config_repo.to_string(), format!("{}.onnx.json", row.key)),
        ],
    })
}

/// The single `.onnx` + its `.onnx.json` inside a user-supplied directory.
///
/// Two graphs is a directory holding more than one voice, not a voice: the file
/// names are not fixed (`--voice <path>` accepts any), so the count is the only
/// thing that can tell them apart.
fn voice_files_in(dir: &Path) -> Result<Option<(PathBuf, PathBuf)>> {
    let mut onnx: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // `.onnx.json` does not end in `.onnx`, so the two tests cannot collide.
        if name.ends_with(".onnx") {
            if onnx.is_some() {
                anyhow::bail!(
                    "{} holds more than one .onnx graph; point --voice at a single voice",
                    dir.display()
                );
            }
            onnx = Some(path);
        } else if name.ends_with(".onnx.json") {
            if config.is_some() {
                anyhow::bail!(
                    "{} holds more than one .onnx.json config; point --voice at a single voice",
                    dir.display()
                );
            }
            config = Some(path);
        }
    }
    Ok(onnx.zip(config))
}

/// A voice, guaranteed to exist on return: downloads whatever of its two files
/// is missing.
pub(crate) fn ensure_available(voice: &VoiceRef) -> Result<()> {
    if voice.assets.is_empty() {
        // Nothing to fetch, so this is the "did the user point --voice at a real
        // voice directory" check.
        if voice.onnx.exists() && voice.config.exists() {
            return Ok(());
        }
        anyhow::bail!(
            "{} is not a voice directory: {} is missing",
            voice.dir.display(),
            if voice.onnx.exists() {
                voice.config.display().to_string()
            } else {
                voice.onnx.display().to_string()
            }
        );
    }
    std::fs::create_dir_all(&voice.dir).with_context(|| install_hint(&voice.dir))?;
    for (repo_path, file_name) in &voice.assets {
        let dest = voice.dir.join(file_name);
        if dest.exists() {
            continue;
        }
        let url = repo_url(repo_path);
        log::info!(
            "voice not found at {}, downloading {url}...",
            dest.display()
        );
        crate::model_store::download(&url, &dest)
            .with_context(|| format!("{file_name}: {}", install_hint(&voice.dir)))?;
    }
    Ok(())
}

/// What to say when a voice has to be fetched into a directory that is not
/// writable — the usual case being a packaged install's root-owned models
/// directory, where only the bundled speech model is present. Same wording as
/// the speech-model path, because the cause and the fix are identical.
fn install_hint(dir: &Path) -> String {
    format!(
        "cannot prepare voice directory {} (packaged model dirs are root-owned: \
         install the sttd package, run once with sudo, or point --voice at a \
         writable directory)",
        dir.display()
    )
}

/// Render voice rows as an aligned table for `--list-voices`: `SPEC` / `LANG` /
/// `QUALITY` / `SIZE` / `SPK`, with a two-character leading marker — `*` when
/// the row is what `--voice` selects, `I` when both its files are on disk, `*I`
/// for both.
pub(crate) fn render_table(
    rows: &[&VoiceRow],
    refs: &[VoiceRef],
    selected: Option<&str>,
) -> String {
    let selected = selected.map(str::trim);
    let spk_w = rows
        .iter()
        .map(|r| r.num_speakers.to_string().len() + usize::from(r.multi_speaker()))
        .max()
        .unwrap_or(1)
        .max("SPK".len());
    let sizes: Vec<u64> = rows.iter().map(|r| r.onnx_size()).collect();
    let spec_w = rows
        .iter()
        .map(|r| r.key.len())
        .max()
        .unwrap_or(0)
        .max("SPEC".len());
    let lang_w = rows
        .iter()
        .map(|r| r.language.code.len())
        .max()
        .unwrap_or(0)
        .max("LANG".len());
    let quality_w = rows
        .iter()
        .map(|r| r.quality.len())
        .max()
        .unwrap_or(0)
        .max("QUALITY".len());
    let size_w = sizes
        .iter()
        .map(|s| format_size(*s).len())
        .max()
        .unwrap_or(0)
        .max("SIZE".len());

    let mut out = String::new();
    // `SIZE` is padded to `size_w`, which was already widened to hold the widest
    // rendered size, so the header lines up with the rows.
    out.push_str(&format!(
        "   {:<spec_w$}  {:<lang_w$}  {:<quality_w$}  {:>size_w$}  {:>spk_w$}",
        "SPEC", "LANG", "QUALITY", "SIZE", "SPK"
    ));
    out.push('\n');
    for (row, size) in rows.iter().zip(&sizes) {
        // Installed is asked of the resolved reference, which is what knows
        // where this row's two files live.
        let installed = refs
            .iter()
            .find(|v| v.key == row.key)
            .is_some_and(VoiceRef::is_installed);
        let marker = match (selected == Some(row.key.as_str()), installed) {
            (true, true) => "*I",
            (true, false) => "* ",
            (false, true) => "I ",
            (false, false) => "  ",
        };
        // A multi-speaker voice is listed but not usable (`piper::PiperVoice::load`
        // refuses one), so the speaker count says so rather than leaving the
        // user to discover it at load time.
        let speakers = if row.multi_speaker() {
            format!("{}*", row.num_speakers)
        } else {
            row.num_speakers.to_string()
        };
        out.push_str(&format!(
            "{marker} {:<spec_w$}  {:<lang_w$}  {:<quality_w$}  {:>size_w$}  {:>spk_w$}",
            row.key,
            row.language.code,
            row.quality,
            format_size(*size),
            speakers
        ));
        out.push('\n');
    }
    out
}

/// Human-readable download size for the table, e.g. `60.3 MiB`. Unknown
/// upstream (`0`) renders as `-` rather than a misleading `0.0 MiB`.
pub(crate) fn format_size(bytes: u64) -> String {
    const MIB: f64 = 1_048_576.0;
    if bytes == 0 {
        "-".to_string()
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

/// Resolve `--voice` and download whatever is missing, in one call: the
/// startup-time path, where a bad key or an unwritable directory has to fail
/// fast like the speech-model load it sits beside.
pub(crate) fn prepare_voice(input: &str) -> Result<VoiceRef> {
    let index = VoiceIndex::load()?;
    let voice = resolve_voice(input, &index)?;
    ensure_available(&voice)?;
    Ok(voice)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two real rows from upstream's `voices.json`, trimmed to the fields this
    /// module reads. Enough to exercise key lookup, the alias list, the
    /// repo-path extraction and the language/quality columns without shipping a
    /// 245 KiB fixture.
    const FIXTURE: &str = r#"{
        "pl_PL-gosia-medium": {
            "key": "pl_PL-gosia-medium",
            "name": "gosia",
            "language": {"code": "pl_PL", "family": "pl", "region": "PL",
                         "name_native": "Polski", "name_english": "Polish",
                         "country_english": "Poland"},
            "quality": "medium",
            "num_speakers": 1,
            "speaker_id_map": {},
            "files": {
                "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx": {"size_bytes": 63201294, "md5_digest": "ecf8"},
                "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json": {"size_bytes": 4814, "md5_digest": "e570"},
                "pl/pl_PL/gosia/medium/MODEL_CARD": {"size_bytes": 275, "md5_digest": "ff04"}
            },
            "aliases": []
        },
        "pl_PL-mls_6892-low": {
            "key": "pl_PL-mls_6892-low",
            "name": "mls_6892",
            "language": {"code": "pl_PL", "name_english": "Polish"},
            "quality": "low",
            "num_speakers": 1,
            "speaker_id_map": {},
            "files": {
                "pl/pl_PL/mls_6892/low/pl_PL-mls_6892-low.onnx": {"size_bytes": 63104526},
                "pl/pl_PL/mls_6892/low/pl_PL-mls_6892-low.onnx.json": {"size_bytes": 4157}
            },
            "aliases": ["pl-mls_6892-low"]
        },
        "en_US-hfc_female-medium": {
            "key": "en_US-hfc_female-medium",
            "name": "hfc_female",
            "language": {"code": "en_US", "name_english": "English"},
            "quality": "medium",
            "num_speakers": 1,
            "files": {
                "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx": {"size_bytes": 63104526},
                "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json": {"size_bytes": 4185}
            }
        },
        "de_DE-thorsten_emotional-medium": {
            "key": "de_DE-thorsten_emotional-medium",
            "name": "thorsten_emotional",
            "language": {"code": "de_DE", "name_english": "German"},
            "quality": "medium",
            "num_speakers": 8,
            "speaker_id_map": {"amused": 0},
            "files": {
                "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx": {"size_bytes": 76745905},
                "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx.json": {"size_bytes": 5031}
            }
        }
    }"#;

    fn index() -> VoiceIndex {
        let rows: BTreeMap<String, VoiceRow> =
            serde_json::from_str(FIXTURE).expect("fixture parses");
        VoiceIndex { rows }
    }

    #[test]
    fn rows_expose_sizes_speakers_and_the_two_repo_paths() {
        let index = index();
        let row = index.row("pl_PL-gosia-medium").expect("row must exist");
        assert_eq!(row.language.code, "pl_PL");
        assert_eq!(row.language.name_english, "Polish");
        assert_eq!(row.quality, "medium");
        assert!(!row.multi_speaker());
        // The graph is the `.onnx`, the config the `.onnx.json`; `MODEL_CARD`
        // must not be mistaken for either.
        assert_eq!(
            row.onnx_path(),
            Some("pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx")
        );
        assert_eq!(
            row.config_path(),
            Some("pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json")
        );
        assert_eq!(row.onnx_size(), 63_201_294);
        assert!(index
            .row("de_DE-thorsten_emotional-medium")
            .unwrap()
            .multi_speaker());
    }

    #[test]
    fn lookup_accepts_a_key_and_an_upstream_alias() {
        let index = index();
        assert_eq!(
            index.row("pl_PL-mls_6892-low").unwrap().key,
            "pl_PL-mls_6892-low"
        );
        // The old-style alias the repo still publishes for this voice.
        assert_eq!(
            index.row("pl-mls_6892-low").unwrap().key,
            "pl_PL-mls_6892-low"
        );
        // Surrounding whitespace is the user's, not the key's.
        assert_eq!(
            index.row("  pl_PL-gosia-medium \n").unwrap().key,
            "pl_PL-gosia-medium"
        );
        assert!(index.row("pl_PL-nonexistent-medium").is_none());
    }

    #[test]
    fn filter_matches_key_language_code_and_language_name() {
        let index = index();
        let keys = |f: Option<&str>| -> Vec<String> {
            index
                .filtered(f)
                .into_iter()
                .map(|r| r.key.clone())
                .collect()
        };
        assert_eq!(keys(None).len(), 4, "no filter lists everything");
        assert_eq!(keys(Some("")), keys(None), "a blank filter is no filter");
        assert_eq!(
            keys(Some("pl_PL")),
            vec!["pl_PL-gosia-medium", "pl_PL-mls_6892-low"]
        );
        // Language name, for the user who thinks in names not codes.
        assert_eq!(
            keys(Some("german")),
            vec!["de_DE-thorsten_emotional-medium"]
        );
        // Substring, case-insensitive: the `--list-devices` idiom.
        assert_eq!(keys(Some("GOSIA")), vec!["pl_PL-gosia-medium"]);
        assert!(keys(Some("klingon")).is_empty());
    }

    #[test]
    fn resolution_flattens_the_repository_layout_into_the_voice_directory() {
        let index = index();
        let voice = resolve_voice("pl_PL-gosia-medium", &index).expect("must resolve");
        assert_eq!(voice.key, "pl_PL-gosia-medium");
        assert_eq!(
            voice.dir,
            voice_dir("pl_PL-gosia-medium"),
            "a key resolves into the voices directory"
        );
        // Nested upstream (`pl/pl_PL/gosia/medium/…`), flat on disk, and named
        // after the key so the engine never sees the repository layout.
        assert_eq!(
            voice.onnx,
            voice_dir("pl_PL-gosia-medium").join("pl_PL-gosia-medium.onnx")
        );
        assert_eq!(
            voice.config,
            voice_dir("pl_PL-gosia-medium").join("pl_PL-gosia-medium.onnx.json")
        );
        let names: Vec<&str> = voice.assets.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["pl_PL-gosia-medium.onnx", "pl_PL-gosia-medium.onnx.json"]
        );
        let urls: Vec<String> = voice.assets.iter().map(|(p, _)| repo_url(p)).collect();
        assert_eq!(
            urls,
            vec![
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json",
            ]
        );
        // Installed-ness is disk state, which this machine may or may not have:
        // the voice was really downloaded during development, so this asserts
        // the *check* against both outcomes rather than one of them.
        let expected = voice.onnx.exists() && voice.config.exists();
        assert_eq!(
            voice.is_installed(),
            expected,
            "installed means both files present"
        );
    }

    #[test]
    fn an_alias_resolves_to_the_canonical_key_and_its_directory() {
        let index = index();
        let voice = resolve_voice("pl-mls_6892-low", &index).expect("must resolve");
        assert_eq!(voice.key, "pl_PL-mls_6892-low");
        assert_eq!(voice.dir, voice_dir("pl_PL-mls_6892-low"));
    }

    #[test]
    fn a_directory_path_passes_through_and_is_never_downloaded_into() {
        let index = index();
        let dir = std::env::temp_dir().join(format!("sttd-voice-path-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("custom.onnx"), b"x").unwrap();
        std::fs::write(dir.join("custom.onnx.json"), b"{}").unwrap();
        let voice = resolve_voice(dir.to_str().unwrap(), &index).expect("must resolve");
        assert_eq!(
            voice.key,
            dir.file_name().unwrap().to_string_lossy(),
            "the directory's own name is the identity on a path run"
        );
        assert_eq!(voice.onnx, dir.join("custom.onnx"));
        assert_eq!(voice.config, dir.join("custom.onnx.json"));
        assert!(
            voice.assets.is_empty(),
            "a path is validated, never fetched"
        );
        assert!(voice.is_installed());
        // Nothing to download, so validation passes.
        ensure_available(&voice).expect("a complete directory must validate");

        // A directory missing its config is not a voice: refused at resolution,
        // before anything is opened, with the expectation named in the error.
        std::fs::remove_file(dir.join("custom.onnx.json")).unwrap();
        let err = resolve_voice(dir.to_str().unwrap(), &index)
            .expect_err("a directory without a config is not a voice")
            .to_string();
        assert!(err.contains(".onnx.json"), "err={err}");

        // Neither is a directory with no graph in it.
        let empty = std::env::temp_dir().join(format!("sttd-voice-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve_voice(empty.to_str().unwrap(), &index).is_err());

        // Nor a path that does not exist.
        assert!(resolve_voice("/nonexistent/voice", &index).is_err());
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn unknown_and_empty_voices_are_refused_with_a_pointer_to_the_listing() {
        let index = index();
        let err = resolve_voice("pl_PL-nope-medium", &index)
            .expect_err("an unknown key must not resolve")
            .to_string();
        assert!(err.contains("--list-voices"), "err={err}");
        let err = resolve_voice("   ", &index)
            .expect_err("an empty value must not resolve")
            .to_string();
        assert!(err.contains("--list-voices"), "err={err}");
    }

    #[test]
    fn table_marks_the_selected_row_and_aligns_its_columns() {
        let index = index();
        let rows = index.filtered(Some("pl_PL"));
        let refs: Vec<VoiceRef> = rows
            .iter()
            .map(|r| resolve_voice(&r.key, &index).expect("must resolve"))
            .collect();
        let table = render_table(&rows, &refs, Some("pl_PL-gosia-medium"));
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), rows.len() + 1, "header plus one row per voice");
        for header in ["SPEC", "LANG", "QUALITY", "SIZE", "SPK"] {
            assert!(lines[0].contains(header), "header={:?}", lines[0]);
        }
        let selected = lines
            .iter()
            .find(|l| l.contains("pl_PL-gosia-medium"))
            .expect("row must exist");
        // `*` for selected; `I` depends on what is on this machine, so only the
        // selected half is asserted.
        assert!(selected.starts_with('*'), "row={selected:?}");
        assert_eq!(
            selected.trim_end().len(),
            selected.len(),
            "no trailing space"
        );
        // Size is rendered from the index, not the file system.
        assert!(selected.contains("60.3 MiB"), "row={selected:?}");
        let other = lines
            .iter()
            .find(|l| l.contains("pl_PL-mls_6892-low"))
            .expect("row must exist");
        assert!(other.starts_with("  "), "row={other:?}");
    }

    #[test]
    fn sizes_render_and_degrade() {
        assert_eq!(format_size(0), "-", "unknown upstream is not 0.0 MiB");
        assert_eq!(format_size(1_048_576), "1.0 MiB");
        assert_eq!(format_size(63_201_294), "60.3 MiB");
    }

    #[test]
    fn voices_live_beside_the_models_never_inside_one() {
        // The directory name is what keeps `model_store`'s engine detection
        // from ever being pointed at a voice.
        assert_eq!(
            voices_dir(),
            PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{VOICES_SUBDIR}"))
        );
        assert_ne!(
            voice_dir("pl_PL-gosia-medium"),
            voice_dir("parakeet-ultra-int8")
        );
    }
}
