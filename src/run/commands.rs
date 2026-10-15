//! The one-shot commands that run instead of a transcription session:
//! `--list-models`, `--list-voices` and `--paste-last`.

use anyhow::{Context, Result};

use crate::cli::Cli;
use crate::model_store::CATALOGS;

/// Print known models — both catalogs — as one aligned table per engine, each
/// rendered by its own catalog (columns SPEC/DIR/SIZE, since both engines ship
/// as a directory of files rather than one), with a two-character leading
/// marker: `*` = currently selected (effective `--model` after CLI >
/// `STTD_MODEL` > compiled-in default), `I` = installed (weights present on
/// disk). A row can carry both (`*I`).
pub(crate) fn print_available_models(cli: &Cli) {
    let input = cli.model_input();

    let mut any_selected = false;
    for catalog in CATALOGS {
        let selected = catalog.entry_for_spec(&input).map(|e| e.spec);
        println!("{} models:", catalog.engine.as_str());
        print!("{}", catalog.render_table(selected));
        any_selected |= selected.is_some();
    }

    if !any_selected {
        let trimmed = input.trim();
        if !trimmed.is_empty() {
            println!("selected: {trimmed} (not in catalog)");
        }
    }
}

/// Print piper voices as an aligned table: `SPEC` / `LANG` / `QUALITY` /
/// `SIZE` / `SPK`, with the same leading marker as `--list-models` (`*` =
/// selected via `--voice`/`STTD_VOICE`, `I` = downloaded to disk).
///
/// An optional substring filters on the voice key or its language, because the
/// catalog is ~1300 rows — unfiltered it is a reference dump, not a listing.
/// The index comes from upstream's own `voices.json`, so the first call needs
/// the network; after that it is read from the cache beside the voices.
pub(crate) fn print_available_voices(cli: &Cli, filter: &str) -> Result<()> {
    let index = crate::voice_store::VoiceIndex::load()?;
    let rows = index.filtered(Some(filter));
    if rows.is_empty() {
        println!("no piper voice matches {filter:?}");
        return Ok(());
    }
    // Resolving is what knows where a row's two files live, and it is the same
    // path `--voice` takes, so the `I` markers cannot disagree with it.
    let refs: Vec<crate::voice_store::VoiceRef> = rows
        .iter()
        .filter_map(|row| crate::voice_store::resolve_voice(&row.key, &index).ok())
        .collect();
    println!(
        "piper voices ({} of {}), from https://huggingface.co/{}/tree/main:",
        rows.len(),
        index.filtered(None).len(),
        crate::config::PIPER_VOICES_REPO
    );
    print!(
        "{}",
        crate::voice_store::render_table(&rows, &refs, Some(cli.voice.trim()))
    );
    if rows.iter().any(|r| r.multi_speaker()) {
        println!("* multi-speaker: listed, but --speak refuses one (it has no speaker to pick)");
    }
    if refs.iter().all(|v| !v.is_installed()) {
        println!("none downloaded yet; `sttd --speak --voice <spec>` fetches one (~60 MiB)");
    }
    Ok(())
}

/// Re-copy and re-paste the last emitted transcription. One-shot: reads the
/// daemon's last-text file written on every emission. `type` mode types
/// directly without touching the clipboard; key modes copy via `wl-copy`
/// then paste via `ydotool`. Warns (no failure) when no emission exists yet.
pub(crate) fn run_paste_last(cli: &Cli) -> Result<()> {
    let path = crate::emit::last_text_path();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let text = text.trim();
    if text.is_empty() {
        log::warn!(
            "paste-last: no emitted transcription yet ({})",
            path.display()
        );
        return Ok(());
    }
    if !crate::emit::is_wayland_session_env() {
        log::warn!("paste-last: not a Wayland session; copy attempted anyway");
    }
    if cli.paste_mode == crate::emit::PasteMode::Type {
        crate::emit::type_via_ydotool(text)
            .context("paste-last: ydotool type failed (is ydotoold running?)")?;
        log::info!("paste-last: typed {} chars", text.chars().count());
        return Ok(());
    }
    crate::emit::copy_to_clipboard(text).context("paste-last: wl-copy failed")?;
    log::info!("paste-last: copied {} chars", text.chars().count());
    crate::emit::paste_via_ydotool(cli.paste_mode)
        .context("paste-last: ydotool failed (is ydotoold running?)")?;
    Ok(())
}
