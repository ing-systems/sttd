use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{
    DEFAULT_MODELS_DIR, PARAKEET_DEFAULT_SPEC, PARAKEET_TOKENIZER_URL, WHISPER_DEFAULT_SPEC,
};

/// Which engine a resolved `--model` value selects. Both engines load from a
/// directory of files, but the file names differ, so a directory the user
/// pointed `--model` at directly has to be classified by probing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineKind {
    /// Parakeet-TDT: `encoder-model` + `decoder_joint-model` + `vocab.txt`.
    Parakeet,
    /// Whisper large-v3-turbo: `encoder_model` + `decoder_model` +
    /// `decoder_with_past_model` + `tokenizer.json`.
    Whisper,
}

/// The models directory a catalog spec resolves into: the compile-time baked
/// `DEFAULT_MODELS_DIR` joined with the spec's directory name (which equals the
/// spec, see `CatalogEntry::dir`).
///
/// The one join in the codebase: the download path, `--list-models`' installed
/// check and the tests that reach a downloaded model all go through it, so they
/// cannot drift apart.
pub(crate) fn catalog_dir(spec: &str) -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODELS_DIR}/{spec}"))
}

/// One row of a model catalog: a `--model` spec and where its weights come from.
///
/// `dir` equals `spec` for every row today, which is what lets `catalog_dir`
/// take either. It stays a separate field because it is the download
/// destination, not an identity: a future export that ships under a different
/// directory name changes this column and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CatalogEntry {
    pub(crate) spec: &'static str,
    /// Directory under `DEFAULT_MODELS_DIR`. Also selects the download source.
    pub(crate) dir: &'static str,
    /// Files `ensure_available` fetches. Filenames match what the engine looks
    /// for, so no renaming after download.
    pub(crate) assets: &'static [&'static str],
    /// Stated upstream download size. Read by `--list-models` only.
    pub(crate) size: &'static str,
    /// Which engine's catalog this row belongs to.
    pub(crate) kind: EngineKind,
}

/// One engine's catalog: its rows, the short `--model` alias that stands for
/// one of them, and where its weights are published.
///
/// `alias` and `default_spec` are separate rather than one "alias target"
/// string because they are read separately: `default_spec` is what an alias
/// resolves to, and it is also the row the catalog's own default points at.
pub(crate) struct Catalog {
    pub(crate) engine: EngineKind,
    /// Short `--model` alias (`parakeet`, `whisper`).
    pub(crate) alias: &'static str,
    /// The row `alias` stands for.
    pub(crate) default_spec: &'static str,
    pub(crate) entries: &'static [CatalogEntry],
}

/// Parakeet-TDT-0.6B-v3 weights, published as loose ONNX files (no archive)
/// rather than a single `.bin`: the engine loads from a directory containing
/// an encoder, a decoder_joint, `vocab.txt` and `tokenizer.json`
/// (phrase-boosting; without it transcription works but boosting stays off).
/// Mel features are computed in Rust (`parakeet_core::frontend`), so no
/// `nemo128.onnx` and no `config.json`/`preprocessor_config.json` are needed.
///
/// One export, `parakeet-ultra-int8`, in its own Hugging Face repository. It
/// keeps the base architecture and tokenizer (128 mel features, subsampling 8, a
/// `vocab.txt` byte-identical to the base checkpoint), so the Rust frontend and
/// the `PreprocessorConfig::tdt_0_6b` recipe still apply unchanged.
///
/// The repository is not one flat file list: the ONNX weights sit in its
/// `int8/` folder, `vocab.txt` at its root. `tokenizer.json` is absent from the
/// export, so it comes from the upstream NVIDIA checkpoint
/// (`PARAKEET_TOKENIZER_URL`).
///
/// `(spec, repo resolve root, weights folder)`.
pub(crate) const PARAKEET_SOURCES: &[(&str, &str, &str)] = &[(
    "parakeet-ultra-int8",
    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/main/",
    "int8/",
)];

/// Parakeet's catalog. The ultra int8 export is the default variant.
pub(crate) const PARAKEET_CATALOG: Catalog = Catalog {
    engine: EngineKind::Parakeet,
    alias: "parakeet",
    default_spec: PARAKEET_DEFAULT_SPEC,
    entries: &[CatalogEntry {
        spec: "parakeet-ultra-int8",
        dir: "parakeet-ultra-int8",
        assets: &[
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "vocab.txt",
            "tokenizer.json",
        ],
        size: "638 MiB",
        kind: EngineKind::Parakeet,
    }],
};

/// Whisper large-v3-turbo weights, published as loose ONNX files: the engine
/// loads from a directory containing three graphs and `tokenizer.json`.
/// Mel features are computed in Rust (`whisper_core::frontend`), so no
/// `nemo128.onnx`-style preprocessor session and no `config.json` /
/// `preprocessor_config.json` are needed.
///
/// Three graphs, not one: `decoder_model` runs the prompt (and produces the
/// cross-attention KV cache), then `decoder_with_past_model` runs every
/// following token off that cache. The single-graph `decoder_model_merged`
/// alternative is deliberately not used — it re-runs cross-attention over all
/// 1500 encoder frames per token, which costs ~40 GFLOP per token here.
///
/// The ONNX layout is not one flat file list: the graphs sit in the
/// repository's `onnx/` folder, `tokenizer.json` at its root.
///
/// `(spec, repo resolve root, weights folder)`.
pub(crate) const WHISPER_SOURCES: &[(&str, &str, &str)] = &[(
    "whisper-large-v3-turbo-int8",
    "https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/",
    "onnx/",
)];

/// Whisper's catalog. The int8 variant is the default.
pub(crate) const WHISPER_CATALOG: Catalog = Catalog {
    engine: EngineKind::Whisper,
    alias: "whisper",
    default_spec: WHISPER_DEFAULT_SPEC,
    entries: &[CatalogEntry {
        spec: "whisper-large-v3-turbo-int8",
        dir: "whisper-large-v3-turbo-int8",
        assets: &[
            "encoder_model_int8.onnx",
            "decoder_model_int8.onnx",
            "decoder_with_past_model_int8.onnx",
            "tokenizer.json",
        ],
        size: "1443 MiB",
        kind: EngineKind::Whisper,
    }],
};

/// Every catalog, in `--model` resolution order: Parakeet first, so a spec that
/// somehow named a row in both would resolve to Parakeet rather than depend on
/// declaration order elsewhere.
pub(crate) const CATALOGS: &[&Catalog] = &[&PARAKEET_CATALOG, &WHISPER_CATALOG];

/// Download source for a Parakeet spec: `(repo resolve root, weights folder)`.
pub(crate) fn parakeet_source_for_spec(spec: &str) -> Option<(&'static str, &'static str)> {
    PARAKEET_SOURCES
        .iter()
        .find(|(s, ..)| *s == spec)
        .map(|(_, root, folder)| (*root, *folder))
}

/// Download source for a Whisper spec: `(repo resolve root, weights folder)`.
pub(crate) fn whisper_source_for_spec(spec: &str) -> Option<(&'static str, &'static str)> {
    WHISPER_SOURCES
        .iter()
        .find(|(s, ..)| *s == spec)
        .map(|(_, root, folder)| (*root, *folder))
}

impl Catalog {
    /// The row `spec` names: an exact row, or this catalog's short alias
    /// standing for its default.
    ///
    /// The one place the alias rule is implemented. `resolve_model` and
    /// `--list-models` both go through it, so a spec cannot resolve for
    /// `--model` but light up no row in `--list-models`, or the reverse.
    pub(crate) fn entry_for_spec(&self, spec: &str) -> Option<&'static CatalogEntry> {
        let spec = spec.trim();
        let spec = if spec == self.alias {
            self.default_spec
        } else {
            spec
        };
        self.entries.iter().find(|e| e.spec == spec)
    }

    /// Every value `--model` accepts for this catalog, for error messages:
    /// each spec, then the alias.
    pub(crate) fn specs(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries
            .iter()
            .map(|e| e.spec)
            .chain(std::iter::once(self.alias))
    }

    /// Whether this catalog has the weights for `entry` on disk, i.e. whether
    /// every asset it lists is present in its directory. The `--list-models`
    /// `I` marker: "installed", not merely "known".
    pub(crate) fn is_installed(&self, entry: &CatalogEntry) -> bool {
        // Non-empty on purpose: every real row lists assets (asserted by
        // `catalog_rows_are_self_consistent`), and `[].iter().all(..)` is
        // vacuously true, so an empty list would claim a directory holding
        // nothing is a complete install.
        !entry.assets.is_empty() && {
            let dir = catalog_dir(entry.dir);
            entry.assets.iter().all(|f| dir.join(f).exists())
        }
    }

    /// Render this catalog as an aligned table for `--list-models`: a `SPEC`
    /// / `DIR` / `SIZE` header plus one row per entry, columns padded to the
    /// widest cell.
    ///
    /// The leading two-character marker is `*` when the entry is the one
    /// `--model` selects, `I` when its weights are installed, `*I` for both and
    /// two spaces for neither.
    pub(crate) fn render_table(&self, selected: Option<&str>) -> String {
        let spec_w = self
            .entries
            .iter()
            .map(|e| e.spec.len())
            .max()
            .unwrap_or(0)
            .max("SPEC".len());
        let dir_w = self
            .entries
            .iter()
            .map(|e| e.dir.len())
            .max()
            .unwrap_or(0)
            .max("DIR".len());
        let size_w = self
            .entries
            .iter()
            .map(|e| e.size.len())
            .max()
            .unwrap_or(0)
            .max("SIZE".len());
        let mut out = String::new();
        out.push_str(&format!(
            "   {0:<1$}  {2:<3$}  {4:>5$}\n",
            "SPEC", spec_w, "DIR", dir_w, "SIZE", size_w
        ));
        for entry in self.entries {
            let marker = match (selected == Some(entry.spec), self.is_installed(entry)) {
                (true, true) => "*I",
                (true, false) => "* ",
                (false, true) => "I ",
                (false, false) => "  ",
            };
            out.push_str(&format!(
                "{marker} {0:<1$}  {2:<3$}  {4:>5$}\n",
                entry.spec, spec_w, entry.dir, dir_w, entry.size, size_w
            ));
        }
        out
    }
}

/// Comma-separated valid specs for `--model` error messages, over every catalog
/// in resolution order.
fn valid_model_specs() -> String {
    CATALOGS
        .iter()
        .flat_map(|c| c.specs())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The catalog entry a `--model` input names, in catalog order. `None` when the
/// value is not a spec or alias of any catalog (a path, which `resolve_model`
/// handles separately).
fn catalog_entry_for_input(input: &str) -> Option<&'static CatalogEntry> {
    CATALOGS.iter().find_map(|c| c.entry_for_spec(input))
}

impl EngineKind {
    /// Short name for log lines and the `--list-models` table heading.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet",
            Self::Whisper => "whisper",
        }
    }

    /// The file that identifies a model directory as this engine's.
    /// Note the deliberate naming difference between the two engines:
    /// Parakeet's graphs use hyphens (`encoder-model.int8.onnx`), Whisper's
    /// underscores (`encoder_model_int8.onnx`).
    const fn marker_file(self) -> &'static str {
        match self {
            Self::Parakeet => "decoder_joint-model.int8.onnx",
            Self::Whisper => "decoder_with_past_model_int8.onnx",
        }
    }

    /// Every file a directory of this kind must hold. Used both to validate a
    /// user-supplied directory and to describe the expected layout in errors.
    const fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::Parakeet => &[
                "encoder-model.int8.onnx",
                "decoder_joint-model.int8.onnx",
                "vocab.txt",
            ],
            Self::Whisper => &[
                "encoder_model_int8.onnx",
                "decoder_model_int8.onnx",
                "decoder_with_past_model_int8.onnx",
                "tokenizer.json",
            ],
        }
    }
}

/// Which ASR model a resolved `--model` value maps to, and where its
/// weights live on disk. Both engines take a directory of files — Parakeet an
/// encoder plus `decoder_joint`, Whisper an encoder plus two decoders — so
/// there is no single-file equivalent for either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelRef {
    /// Model directory. `assets` lists the files `ensure_available` must
    /// fetch if missing; empty for a directory the user pointed `--model`
    /// at directly (nothing is downloaded, only validated).
    pub(crate) dir: PathBuf,
    pub(crate) assets: &'static [&'static str],
    /// Which engine opens `dir`. `asr::load_model` dispatches on this.
    pub(crate) kind: EngineKind,
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
/// value must be a catalog spec — Parakeet's or Whisper's (a model directory
/// under `models/`, see `PARAKEET_CATALOG` / `WHISPER_CATALOG`) — or one of
/// the `parakeet` / `whisper` aliases.
pub(crate) fn resolve_model(input: &str) -> Result<ModelRef> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!(
            "empty --model value; expected one of: {}",
            valid_model_specs()
        );
    }
    if let Some(entry) = catalog_entry_for_input(trimmed) {
        return Ok(ModelRef {
            dir: catalog_dir(entry.spec),
            assets: entry.assets,
            kind: entry.kind,
        });
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        // A manually-prepared model directory (nothing to download, just
        // validated at load time). Which engine it holds is decided by its
        // files, since the two engines have different names.
        let path = PathBuf::from(trimmed);
        if path.is_dir() {
            let kind = detect_engine_kind(&path)?;
            return Ok(ModelRef {
                dir: path,
                assets: &[],
                kind,
            });
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

/// Classify a user-supplied model directory by probing for each engine's
/// marker file. Both markers present is ambiguous and rejected rather than
/// guessed; neither means the directory is not a model at all.
fn detect_engine_kind(dir: &Path) -> Result<EngineKind> {
    let parakeet = dir.join(EngineKind::Parakeet.marker_file()).exists();
    let whisper = dir.join(EngineKind::Whisper.marker_file()).exists();
    match (parakeet, whisper) {
        (true, false) => Ok(EngineKind::Parakeet),
        (false, true) => Ok(EngineKind::Whisper),
        (true, true) => anyhow::bail!(
            "{} holds both a Parakeet and a Whisper model directory's files; \
             point --model at one engine's directory",
            dir.display()
        ),
        (false, false) => anyhow::bail!(
            "{} is not a valid model directory; a Parakeet one holds {} and a \
             Whisper one holds {}",
            dir.display(),
            EngineKind::Parakeet.required_files().join(", "),
            EngineKind::Whisper.required_files().join(", ")
        ),
    }
}

/// `model`, guaranteed to exist on return: downloads any missing weights
/// into place first.
pub(crate) fn ensure_available(model: &ModelRef) -> Result<()> {
    ensure_available_for(&model.dir, model.assets, model.kind)
}

/// Downloads any of `assets` missing from `dir`. An empty `assets`
/// (user-supplied directory) is only validated, never fetched — see
/// `resolve_model`. For a catalog entry the directory name equals the spec,
/// which selects the download source; for a user-supplied directory `assets`
/// is empty and the spec is never used.
fn ensure_available_for(
    dir: &Path,
    assets: &'static [&'static str],
    kind: EngineKind,
) -> Result<()> {
    if assets.is_empty() {
        // Nothing to fetch, so this is the "did the user point --model at a
        // real model directory" check. Every required file must be there, not
        // just the marker `resolve_model` probed with: a directory holding an
        // encoder but no decoder would otherwise fail deep inside the engine
        // with an opaque ORT error.
        if !dir.is_dir() {
            anyhow::bail!(
                "{} model directory not found: {} (expected {})",
                kind.as_str(),
                dir.display(),
                kind.required_files().join(", ")
            );
        }
        let missing: Vec<&str> = kind
            .required_files()
            .iter()
            .copied()
            .filter(|f| !dir.join(f).exists())
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "{} is missing {}; not a valid {} model directory",
                dir.display(),
                missing.join(", "),
                kind.as_str()
            );
        }
        return Ok(());
    }
    std::fs::create_dir_all(dir).with_context(|| install_hint(dir))?;
    let spec = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    for file in assets {
        let dest = dir.join(file);
        if dest.exists() {
            continue;
        }
        let url = asset_url(kind, &spec, file);
        log::info!(
            "model not found at {}, downloading {url}...",
            dest.display()
        );
        download(&url, &dest).with_context(|| format!("{file}: {}", install_hint(dir)))?;
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
/// `encoder-model.int8.onnx`) of catalog `spec`. Three sources per spec,
/// because the export repository is not one flat file list: the ONNX weights
/// sit in its `int8/` folder, `vocab.txt` at its root, and `tokenizer.json` is
/// absent from it entirely, so it comes from the upstream NVIDIA checkpoint
/// (`PARAKEET_TOKENIZER_URL`, shared by both specs — see
/// `PARAKEET_SOURCES`).
///
/// The destination is always the flat `<models_dir>/<spec>/<file_name>`, so
/// the repository layout never leaks into the engine's expectations.
///
/// # Panics
///
/// Panics if `spec` has no `PARAKEET_SOURCES` row. Every catalog spec has one
/// (asserted in `config`'s tests); the empty `assets` path in
/// `ensure_parakeet_available` — a user-supplied directory whose name need
/// not be a spec — never reaches this function.
pub(crate) fn parakeet_asset_url(spec: &str, file_name: &str) -> String {
    let (root, folder) = parakeet_source_for_spec(spec)
        .unwrap_or_else(|| panic!("no PARAKEET_SOURCES row for parakeet spec {spec:?}"));
    match file_name {
        "tokenizer.json" => PARAKEET_TOKENIZER_URL.to_string(),
        "vocab.txt" => format!("{root}{file_name}"),
        _ => format!("{root}{folder}{file_name}"),
    }
}

/// Download URL for one Whisper asset filename (e.g.
/// `decoder_with_past_model_int8.onnx`) of catalog `spec`. The repository is
/// not one flat file list: the graphs sit in its `onnx/` folder and
/// `tokenizer.json` at its root.
///
/// # Panics
///
/// Panics if `spec` has no `WHISPER_SOURCES` row. Every whisper catalog spec
/// has one (asserted in `config`'s tests); the empty `assets` path in
/// `ensure_available_for` never reaches this function.
pub(crate) fn whisper_asset_url(spec: &str, file_name: &str) -> String {
    let (root, folder) = whisper_source_for_spec(spec)
        .unwrap_or_else(|| panic!("no WHISPER_SOURCES row for whisper spec {spec:?}"));
    if file_name.ends_with(".onnx") {
        format!("{root}{folder}{file_name}")
    } else {
        format!("{root}{file_name}")
    }
}

/// Download URL for one asset of `kind`, dispatching to that engine's
/// repository layout. Panics if `spec` has no source row for `kind` (see
/// `whisper_asset_url`).
fn asset_url(kind: EngineKind, spec: &str, file_name: &str) -> String {
    match kind {
        EngineKind::Parakeet => parakeet_asset_url(spec, file_name),
        EngineKind::Whisper => whisper_asset_url(spec, file_name),
    }
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
    use crate::config::{PARAKEET_DEFAULT_SPEC, WHISPER_DEFAULT_SPEC};

    /// A `SPEC` / `DIR` / `SIZE` header plus one row per entry, columns padded
    /// to the widest cell. Both engines ship as a directory of files rather
    /// than one, so all three columns are always present.
    #[test]
    fn catalog_table_aligns_columns_and_marks_rows() {
        for (catalog, spec) in [
            (&PARAKEET_CATALOG, "parakeet-ultra-int8"),
            (&WHISPER_CATALOG, "whisper-large-v3-turbo-int8"),
        ] {
            let table = catalog.render_table(Some(spec));
            let lines: Vec<&str> = table.lines().collect();
            assert_eq!(lines.len(), catalog.entries.len() + 1, "{spec}");
            assert!(lines[0].contains("SPEC"), "{spec} header={:?}", lines[0]);
            assert!(lines[0].contains("DIR"), "{spec} header={:?}", lines[0]);
            assert!(lines[0].contains("SIZE"), "{spec} header={:?}", lines[0]);
            let row = lines
                .iter()
                .find(|l| l.contains(spec))
                .unwrap_or_else(|| panic!("row for {spec} must exist"));
            // Selected is the `*`; whether `I` also lights up depends on what is
            // on this machine, so only the `*` half is asserted here.
            assert!(row.starts_with('*'), "{spec} row={row:?}");
            assert_eq!(row.trim_end().len(), row.len(), "no trailing space");
        }
    }

    /// The two markers are independent: `*` is what `--model` selects, `I` is
    /// whether the weights are present. `is_installed` is the seam — a spec
    /// resolves for `--model` whether or not anything is downloaded yet.
    #[test]
    fn installed_and_selected_are_independent_of_each_other() {
        let entry = WHISPER_CATALOG.entries[0];
        let selected = Some(entry.spec);
        // Every asset present means installed; the `*` marker is chosen
        // independently, so all four combinations are expressible.
        let header = format!("{}\n", WHISPER_CATALOG.render_table(selected));
        assert!(header.starts_with("   SPEC"), "header={header:?}");
        let row = WHISPER_CATALOG
            .render_table(selected)
            .lines()
            .nth(1)
            .expect("one row for a one-entry catalog")
            .to_string();
        let marker = &row[..2];
        assert!(matches!(marker, "*I" | "* "), "row={row:?}");
        // An entry with no assets can never be "installed": the check is over
        // the file list, so an empty list would claim a directory that holds
        // nothing is complete.
        let empty = CatalogEntry {
            spec: "empty",
            dir: "empty",
            assets: &[],
            size: "0 MiB",
            kind: EngineKind::Whisper,
        };
        assert!(
            !WHISPER_CATALOG.is_installed(&empty),
            "an entry listing no files must not report installed"
        );
    }

    /// Every spec must have a download-source row and a directory that equals
    /// its spec, and must list the files its engine cannot decode without. A
    /// spec with no source row would panic at download time instead of failing
    /// here.
    #[test]
    fn catalog_rows_are_self_consistent() {
        fn source_for(engine: EngineKind) -> fn(&str) -> Option<(&'static str, &'static str)> {
            match engine {
                EngineKind::Parakeet => parakeet_source_for_spec,
                EngineKind::Whisper => whisper_source_for_spec,
            }
        }
        for catalog in [&PARAKEET_CATALOG, &WHISPER_CATALOG] {
            assert_eq!(catalog.entries.len(), 1);
            assert!(
                catalog
                    .entries
                    .iter()
                    .any(|e| e.spec == catalog.default_spec),
                "{} must have a row for its default spec",
                catalog.engine.as_str()
            );
            for entry in catalog.entries {
                assert_eq!(
                    entry.dir, entry.spec,
                    "dir must equal spec for {}",
                    entry.spec
                );
                assert!(!entry.assets.is_empty(), "{} must list files", entry.spec);
                assert_eq!(entry.kind, catalog.engine, "{} kind", entry.spec);
                assert!(
                    source_for(catalog.engine)(entry.spec).is_some(),
                    "{} must have a source row",
                    entry.spec
                );
            }
            // The alias must resolve, and to the default row.
            assert_eq!(
                catalog.entry_for_spec(catalog.alias).map(|e| e.spec),
                Some(catalog.default_spec),
                "{} alias",
                catalog.engine.as_str()
            );
            assert!(catalog.entry_for_spec("bogus").is_none());
        }
    }

    /// A spec belongs to exactly one catalog: a Whisper spec must not be
    /// selectable in the Parakeet table, or the `*` marker would light up in
    /// both. The alias is per-catalog too, and both catalogs use the same short
    /// names for their own defaults.
    #[test]
    fn selection_is_per_catalog_and_exact() {
        assert_eq!(
            PARAKEET_CATALOG
                .entry_for_spec("parakeet-ultra-int8")
                .map(|e| e.spec),
            Some("parakeet-ultra-int8")
        );
        assert!(
            PARAKEET_CATALOG
                .entry_for_spec("whisper-large-v3-turbo-int8")
                .is_none(),
            "a Whisper spec must not resolve in the Parakeet catalog"
        );
        assert_eq!(
            PARAKEET_CATALOG.entry_for_spec("parakeet").map(|e| e.spec),
            Some(PARAKEET_DEFAULT_SPEC)
        );
        assert_eq!(
            WHISPER_CATALOG.entry_for_spec("whisper").map(|e| e.spec),
            Some(WHISPER_DEFAULT_SPEC)
        );
        assert!(PARAKEET_CATALOG.entry_for_spec("whisper").is_none());
        for bad in ["small", "", "   "] {
            assert!(PARAKEET_CATALOG.entry_for_spec(bad).is_none(), "{bad:?}");
        }
    }

    /// Both catalogs are consulted in `--model` resolution order, Parakeet
    /// first, so `resolve_model` and `--list-models` mark the same row.
    #[test]
    fn an_input_resolves_to_exactly_one_catalog() {
        for input in [
            "parakeet",
            "parakeet-ultra-int8",
            "whisper",
            "whisper-large-v3-turbo-int8",
        ] {
            let hits: Vec<EngineKind> = CATALOGS
                .iter()
                .filter(|c| c.entry_for_spec(input).is_some())
                .map(|c| c.engine)
                .collect();
            assert_eq!(hits.len(), 1, "{input:?} resolved in {hits:?}");
        }
        // Surrounding whitespace is trimmed, as `--model " parakeet "` is a
        // shell quoting accident rather than a different model.
        assert_eq!(
            catalog_entry_for_input("  parakeet  ").map(|e| e.spec),
            Some(PARAKEET_DEFAULT_SPEC)
        );
        assert!(catalog_entry_for_input("models/custom").is_none());
    }

    #[test]
    fn default_model_is_parakeet() {
        let resolved =
            resolve_model(crate::config::DEFAULT_MODEL_SPEC).expect("default spec must resolve");
        assert_eq!(resolved.dir, PathBuf::from("models/parakeet-ultra-int8"));
        assert_eq!(resolved.kind, EngineKind::Parakeet);
        assert!(resolved.assets.contains(&"vocab.txt"));
    }

    #[test]
    fn parakeet_specs_and_alias_resolve_to_model_dir() {
        for (spec, dir) in [
            ("parakeet-ultra-int8", "models/parakeet-ultra-int8"),
            ("parakeet", "models/parakeet-ultra-int8"),
        ] {
            let resolved = resolve_model(spec).expect("parakeet spec must resolve");
            assert_eq!(resolved.dir, PathBuf::from(dir));
            assert_eq!(resolved.kind, EngineKind::Parakeet);
            assert!(resolved.assets.contains(&"vocab.txt"));
            assert!(resolved.assets.iter().any(|f| f.contains("int8")));
        }
        let resolved =
            resolve_model(PARAKEET_DEFAULT_SPEC).expect("default parakeet spec must resolve");
        assert_eq!(resolved.dir, PathBuf::from("models/parakeet-ultra-int8"));
        assert_eq!(resolved.assets.len(), 4);
    }

    /// A model directory is told apart by its marker file, and the two
    /// exports spell their graphs differently (hyphens vs underscores), so a
    /// wrong classification here means loading the wrong engine's names.
    #[test]
    fn whisper_specs_and_alias_resolve_to_model_dir() {
        for (spec, dir) in [
            (
                "whisper-large-v3-turbo-int8",
                "models/whisper-large-v3-turbo-int8",
            ),
            ("whisper", "models/whisper-large-v3-turbo-int8"),
        ] {
            let resolved = resolve_model(spec).expect("whisper spec must resolve");
            assert_eq!(resolved.dir, PathBuf::from(dir));
            assert_eq!(resolved.kind, EngineKind::Whisper);
            assert_eq!(resolved.assets.len(), 4);
        }
        let resolved =
            resolve_model(WHISPER_DEFAULT_SPEC).expect("default whisper spec must resolve");
        assert_eq!(
            resolved.dir,
            PathBuf::from("models/whisper-large-v3-turbo-int8")
        );
        assert_eq!(resolved.kind, EngineKind::Whisper);
        for graph in [
            "encoder_model_int8.onnx",
            "decoder_model_int8.onnx",
            "decoder_with_past_model_int8.onnx",
        ] {
            assert!(
                resolved.assets.contains(&graph),
                "the decode loop needs {graph}, missing from the asset list"
            );
        }
        assert!(resolved.assets.contains(&"tokenizer.json"));
    }

    #[test]
    fn whisper_asset_urls_split_onnx_folder_from_root() {
        for entry in WHISPER_CATALOG.entries {
            let spec = entry.spec;
            let (root, folder) =
                whisper_source_for_spec(spec).expect("catalog spec must have a source row");
            assert_eq!(folder, "onnx/", "{spec} folder");
            for graph in [
                "encoder_model_int8.onnx",
                "decoder_model_int8.onnx",
                "decoder_with_past_model_int8.onnx",
            ] {
                let url = whisper_asset_url(spec, graph);
                assert_eq!(url, format!("{root}onnx/{graph}"), "{spec} {graph}");
                assert!(url.contains("/onnx/"), "{spec} url missing /onnx/: {url}");
            }
            // The tokenizer sits at the repository root, not in the folder.
            let tok = whisper_asset_url(spec, "tokenizer.json");
            assert_eq!(tok, format!("{root}tokenizer.json"), "{spec}");
            assert!(!tok.contains("/onnx/"), "{spec} tokenizer url: {tok}");
            assert!(
                tok.contains("onnx-community/whisper-large-v3-turbo"),
                "{spec} tokenizer must come from the whisper repo: {tok}"
            );
        }
    }

    /// Build a throwaway directory holding `files` (empty files are enough:
    /// classification and validation only test existence).
    fn temp_model_dir(tag: &str, files: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sttd-{tag}-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        dir
    }

    #[test]
    fn user_directory_is_classified_by_its_files() {
        let parakeet = temp_model_dir("parakeet-dir", EngineKind::Parakeet.required_files());
        let resolved = resolve_model(parakeet.to_str().unwrap()).expect("must resolve");
        assert_eq!(resolved.kind, EngineKind::Parakeet);
        assert!(resolved.assets.is_empty(), "a user dir is never fetched");
        // A resolved directory must then pass validation.
        ensure_available(&resolved).expect("a complete parakeet dir must validate");
        std::fs::remove_dir_all(&parakeet).ok();

        let whisper = temp_model_dir("whisper-dir", EngineKind::Whisper.required_files());
        let resolved = resolve_model(whisper.to_str().unwrap()).expect("must resolve");
        assert_eq!(resolved.kind, EngineKind::Whisper);
        assert!(resolved.assets.is_empty(), "a user dir is never fetched");
        ensure_available(&resolved).expect("a complete whisper dir must validate");
        std::fs::remove_dir_all(&whisper).ok();
    }

    #[test]
    fn ambiguous_or_empty_directory_is_rejected() {
        // Both marker files present: refuse rather than pick one.
        let both = temp_model_dir(
            "both-dir",
            &[
                EngineKind::Parakeet.marker_file(),
                EngineKind::Whisper.marker_file(),
            ],
        );
        let err = resolve_model(both.to_str().unwrap()).expect_err("must reject");
        assert!(err.to_string().contains("both"), "msg={err}");
        std::fs::remove_dir_all(&both).ok();

        // No marker file: not a model directory at all, and the error names
        // the files each engine expects.
        let empty = temp_model_dir("empty-dir", &["README.md"]);
        let err = resolve_model(empty.to_str().unwrap()).expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("encoder-model.int8.onnx"), "msg={msg}");
        assert!(msg.contains("encoder_model_int8.onnx"), "msg={msg}");
        std::fs::remove_dir_all(&empty).ok();
    }

    /// A directory holding a Whisper marker but not the rest: `resolve_model`
    /// classifies it, then validation must catch the gap *before* the engine
    /// opens a session, naming what is missing. Without the marker the
    /// classification itself rejects it (see `ambiguous_or_empty_directory_is_rejected`),
    /// so the marker is what makes this the interesting case.
    #[test]
    fn incomplete_user_directory_fails_validation_with_the_missing_file() {
        let partial = temp_model_dir(
            "partial-dir",
            &[
                "decoder_with_past_model_int8.onnx",
                "encoder_model_int8.onnx",
            ],
        );
        let resolved = resolve_model(partial.to_str().unwrap()).expect("classifies as whisper");
        assert_eq!(resolved.kind, EngineKind::Whisper);
        let err = ensure_available(&resolved).expect_err("must reject");
        let msg = err.to_string();
        // The two graphs it does have must not be reported as missing.
        assert!(msg.contains("decoder_model_int8.onnx"), "msg={msg}");
        assert!(msg.contains("tokenizer.json"), "msg={msg}");
        assert!(!msg.contains("encoder_model_int8.onnx"), "msg={msg}");
        assert!(msg.contains("whisper"), "msg={msg}");
        std::fs::remove_dir_all(&partial).ok();
    }

    #[test]
    fn parakeet_asset_urls_derive_from_each_spec_source() {
        // The export lives in its own repository, and the ONNX weights sit in
        // that repository's `int8/` folder.
        for entry in PARAKEET_CATALOG.entries {
            let spec = entry.spec;
            let (root, folder) =
                parakeet_source_for_spec(spec).expect("catalog spec must have a source row");
            assert_eq!(folder, "int8/", "{spec} folder");
            for file in ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"] {
                let url = parakeet_asset_url(spec, file);
                assert_eq!(url, format!("{root}{folder}{file}"), "{spec} {file}");
                let from_own_repo = url.starts_with(root);
                assert!(from_own_repo, "{spec} url must use its own repo: {url}");
                assert!(url.contains("/int8/"), "{spec} url missing /int8/: {url}");
            }
        }
    }

    #[test]
    fn parakeet_vocab_url_points_at_the_repository_root() {
        // `vocab.txt` sits next to `config.json` at the repo root, not in
        // `int8/` — so it must not come from the weights folder.
        for entry in PARAKEET_CATALOG.entries {
            let spec = entry.spec;
            let (root, folder) =
                parakeet_source_for_spec(spec).expect("catalog spec must have a source row");
            let url = parakeet_asset_url(spec, "vocab.txt");
            assert_eq!(url, format!("{root}vocab.txt"), "{spec}");
            assert!(
                !url.contains(folder),
                "{spec} vocab must not be under {folder}: {url}"
            );
            assert_ne!(url, parakeet_asset_url(spec, "encoder-model.int8.onnx"));
        }
    }

    #[test]
    fn parakeet_tokenizer_url_points_upstream() {
        for entry in PARAKEET_CATALOG.entries {
            let spec = entry.spec;
            assert_eq!(
                parakeet_asset_url(spec, "tokenizer.json"),
                PARAKEET_TOKENIZER_URL,
                "{spec}"
            );
        }
        assert!(PARAKEET_TOKENIZER_URL.contains("nvidia/parakeet-tdt-0.6b-v3"));
    }

    #[test]
    fn nonexistent_directory_path_is_rejected() {
        let err = resolve_model("models/does-not-exist-dir/model").expect_err("must reject");
        assert!(
            err.to_string().contains("not an existing model directory"),
            "msg={err}"
        );
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
        let hint = install_hint(Path::new("/usr/share/sttd/models/parakeet-ultra-int8"));
        assert!(
            hint.contains("/usr/share/sttd/models/parakeet-ultra-int8"),
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
