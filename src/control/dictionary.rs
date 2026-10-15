//! The live word list.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use super::{lock_read, lock_write};

/// The live word list shared by the bus thread, the main loop and the ASR
/// worker.
///
/// `words` is double-wrapped in `Arc` so a reader clones one pointer per
/// utterance instead of copying the list, and no lock is ever held across
/// inference. `dirty` records that the live list diverged from `source`, so a
/// caller knows a `LoadFile` would discard its edits.
#[derive(Clone)]
pub(crate) struct DictionaryState {
    words: Arc<RwLock<Arc<Vec<String>>>>,
    source: Arc<RwLock<PathBuf>>,
    dirty: Arc<AtomicBool>,
    /// Called after every successful word swap so the ASR lane can update the
    /// ONNX boost automaton. Attachable after construction, so the wiring in
    /// `main` never has to change when the booster hook is added.
    on_words_swapped: Arc<dyn Fn(Arc<Vec<String>>) + Send + Sync>,
}

impl DictionaryState {
    /// Build the dictionary state. `source` is the dictionary file the words
    /// came from; an empty path means "no dictionary configured". The list
    /// starts clean, because a startup load matches its own source.
    pub(crate) fn new(words: Vec<String>, source: PathBuf) -> Self {
        Self {
            words: Arc::new(RwLock::new(Arc::new(words))),
            source: Arc::new(RwLock::new(source)),
            dirty: Arc::new(AtomicBool::new(false)),
            on_words_swapped: Arc::new(|words| {
                log::debug!(
                    "dictionary: {} word(s) swapped (no booster attached)",
                    words.len()
                );
            }),
        }
    }

    /// Attach the booster update callback (ASR lane).
    pub(crate) fn with_words_hook(
        mut self,
        hook: Arc<dyn Fn(Arc<Vec<String>>) + Send + Sync>,
    ) -> Self {
        self.on_words_swapped = hook;
        self
    }

    /// Snapshot of the live word list.
    pub(crate) fn words(&self) -> Arc<Vec<String>> {
        Arc::clone(&lock_read(&self.words))
    }

    /// Number of entries in the live list. Uncapped: the `--dictionary` caps
    /// apply in post-correction, not at swap time, so this can exceed
    /// `POSTCORR_MAX_DICT_WORDS`.
    pub(crate) fn count(&self) -> u32 {
        lock_read(&self.words).len() as u32
    }

    /// Whether the live list diverged from the source file.
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Current dictionary source path (empty when none is configured).
    pub(crate) fn source(&self) -> PathBuf {
        lock_read(&self.source).clone()
    }

    /// Replace the whole list from a caller-provided one. The source path stays
    /// unchanged, so the result counts as dirty: the server cannot know that a
    /// caller-supplied list matches the file. Returns the new count.
    pub(crate) fn replace_words(&self, words: Vec<String>) -> u32 {
        self.swap(crate::dictionary::sanitize_entries(
            words.iter().map(String::as_str),
        ));
        self.mark_dirty(true);
        self.count()
    }

    /// Merge entries into the live list.
    ///
    /// Sanitizing matches the file path: entries with digits, entries of more
    /// than two words, and duplicates within the input are dropped. A
    /// normalized collision with a live entry keeps the live entry, so an add
    /// never re-cases an existing word. Returns `(count, added)`, where
    /// `added` is 0 when every entry was already present.
    pub(crate) fn add_words(&self, words: Vec<String>) -> (u32, u32) {
        let clean = crate::dictionary::sanitize_entries(words.iter().map(String::as_str));
        let mut merged = lock_read(&self.words).as_ref().clone();
        let before = merged.len();
        let mut seen: HashSet<String> = merged
            .iter()
            .map(|w| crate::dictionary::normalize_word(w))
            .collect();
        for entry in clean {
            if seen.insert(crate::dictionary::normalize_word(&entry)) {
                merged.push(entry);
            }
        }
        let added = merged.len() - before;
        if added > 0 {
            self.swap(merged);
            self.mark_dirty(true);
        }
        (self.count(), added as u32)
    }

    /// Remove entries by normalized form, so `APPIMAGE` drops `AppImage` and
    /// `ZOLC` drops `żółć`. Returns `(count, removed)`, where `removed` is 0
    /// when nothing matched.
    pub(crate) fn remove_words(&self, words: Vec<String>) -> (u32, u32) {
        let drop: HashSet<String> = words
            .iter()
            .map(|w| crate::dictionary::normalize_word(w))
            .collect();
        let existing = lock_read(&self.words).as_ref().clone();
        let kept: Vec<String> = existing
            .iter()
            .filter(|w| !drop.contains(&crate::dictionary::normalize_word(w)))
            .cloned()
            .collect();
        let removed = existing.len() - kept.len();
        if removed > 0 {
            self.swap(kept);
            self.mark_dirty(true);
        }
        (self.count(), removed as u32)
    }

    /// Membership by normalized form, so `ZOLC` matches `żółć`.
    pub(crate) fn contains(&self, word: &str) -> bool {
        let key = crate::dictionary::normalize_word(word);
        lock_read(&self.words)
            .iter()
            .any(|w| crate::dictionary::normalize_word(w) == key)
    }

    /// Load a dictionary file and swap the live list.
    ///
    /// On success the path becomes the new source and the list is clean again.
    /// On failure the old words, the old source and the dirty flag all stay as
    /// they were.
    pub(crate) fn load_file(&self, path: &Path) -> anyhow::Result<u32> {
        let words = crate::dictionary::load_dictionary(path)?;
        self.swap(words);
        *lock_write(&self.source) = path.to_path_buf();
        self.mark_dirty(false);
        Ok(self.count())
    }

    /// Swap in an already-sanitized list and run the booster hook after it, so
    /// a reader can never see the new list with a stale automaton.
    fn swap(&self, words: Vec<String>) {
        let swapped = Arc::new(words);
        *lock_write(&self.words) = Arc::clone(&swapped);
        (self.on_words_swapped)(swapped);
    }

    fn mark_dirty(&self, dirty: bool) {
        self.dirty.store(dirty, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dictionary_with(words: &[&str]) -> DictionaryState {
        DictionaryState::new(
            words.iter().map(|w| w.to_string()).collect(),
            PathBuf::new(),
        )
    }

    /// A dictionary file in the test temp dir, per the `wav.rs` fixture
    /// pattern. The caller removes it.
    fn write_temp_dictionary(name: &str, body: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("sttd-dbus-test-{}-{name}", std::process::id()));
        std::fs::write(&path, body).expect("temp dictionary must be writable");
        path
    }

    #[test]
    fn replace_words_sanitizes_like_the_file_path() {
        let state = dictionary_with(&["old"]);
        let count = state.replace_words(vec![
            "AppImage".to_string(),
            "appimage".to_string(), // normalized duplicate
            "123".to_string(),      // digits rejected
            "a b c".to_string(),    // more than two words rejected
            "żółć".to_string(),
        ]);
        assert_eq!(count, 2);
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "żółć".to_string()]
        );
        assert!(state.is_dirty(), "a caller-supplied list diverges");
    }

    #[test]
    fn words_hook_runs_after_the_swap() {
        let seen: Arc<RwLock<Option<Arc<Vec<String>>>>> = Arc::new(RwLock::new(None));
        let sink = Arc::clone(&seen);
        let state = dictionary_with(&[])
            .with_words_hook(Arc::new(move |words| *lock_write(&sink) = Some(words)));
        state.replace_words(vec!["Parakeet".to_string()]);
        // The hook observes the already-swapped list.
        assert_eq!(
            *lock_read(&seen),
            Some(Arc::new(vec!["Parakeet".to_string()]))
        );
    }

    #[test]
    fn add_words_merges_and_counts() {
        let state = dictionary_with(&["AppImage"]);
        let (count, added) = state.add_words(vec![
            "Syntocinon".to_string(),
            "1Password".to_string(), // digits rejected
        ]);
        assert_eq!(added, 1);
        assert_eq!(count, 2);
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "Syntocinon".to_string()]
        );
        assert!(state.is_dirty());
    }

    #[test]
    fn add_words_keeps_the_existing_casing() {
        let state = dictionary_with(&["AppImage"]);
        let (count, added) = state.add_words(vec!["appimage".to_string()]);
        assert_eq!(added, 0, "a re-case is not a new entry");
        assert_eq!(count, 1);
        assert_eq!(*state.words(), vec!["AppImage".to_string()]);
    }

    #[test]
    fn add_words_of_nothing_leaves_the_dirty_flag_alone() {
        let state = dictionary_with(&["AppImage"]);
        let (_, added) = state.add_words(vec!["appimage".to_string()]);
        assert_eq!(added, 0);
        assert!(!state.is_dirty(), "nothing changed, so nothing diverged");
    }

    #[test]
    fn remove_words_matches_case_insensitively() {
        let state = dictionary_with(&["AppImage", "żółć", "Syntocinon"]);
        let (count, removed) = state.remove_words(vec!["APPIMAGE".to_string(), "ZOLC".to_string()]);
        assert_eq!(removed, 2);
        assert_eq!(count, 1);
        assert_eq!(*state.words(), vec!["Syntocinon".to_string()]);
        assert!(state.is_dirty());
    }

    #[test]
    fn remove_words_reports_zero_when_absent() {
        let state = dictionary_with(&["AppImage"]);
        let (count, removed) = state.remove_words(vec!["nothing-like-this".to_string()]);
        assert_eq!(removed, 0);
        assert_eq!(count, 1);
        assert!(!state.is_dirty(), "nothing changed, so nothing diverged");
    }

    #[test]
    fn contains_folds_case_and_accents() {
        let state = dictionary_with(&["AppImage", "żółć"]);
        assert!(state.contains("appimage"));
        assert!(state.contains("ZOLC"));
        assert!(state.contains("żółć"));
        assert!(state.contains("ŻÓŁĆ"), "uppercase diacritics fold too");
        assert!(!state.contains("syntocinon"));
    }

    #[test]
    fn count_reflects_every_mutation() {
        let state = dictionary_with(&["a", "b"]);
        assert_eq!(state.count(), 2);
        state.replace_words(vec!["only".to_string()]);
        assert_eq!(state.count(), 1);
        assert_eq!(state.add_words(vec!["two".to_string()]).0, 2);
        assert_eq!(state.remove_words(vec!["only".to_string()]).0, 1);
    }

    #[test]
    fn load_file_sets_the_source_and_clears_dirty() {
        let path = write_temp_dictionary("ok.txt", "AppImage\nSyntocinon\n");
        let state = dictionary_with(&["stale"]);
        state.replace_words(vec!["edited".to_string()]);
        assert!(state.is_dirty());

        let count = state.load_file(&path).expect("the temp file must load");
        assert_eq!(count, 2);
        assert_eq!(state.source(), path);
        assert!(!state.is_dirty(), "the list now matches the source");
        assert_eq!(
            *state.words(),
            vec!["AppImage".to_string(), "Syntocinon".to_string()]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_file_failure_keeps_the_old_state() {
        let path = std::env::temp_dir().join(format!("sttd-dbus-missing-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let state = dictionary_with(&["kept"]);
        state.replace_words(vec!["kept".to_string()]);
        assert!(state.is_dirty());

        assert!(state.load_file(&path).is_err());
        assert_eq!(*state.words(), vec!["kept".to_string()]);
        assert!(state.source().as_os_str().is_empty(), "source stays unset");
        assert!(state.is_dirty(), "a failed load changes nothing");
    }
}
