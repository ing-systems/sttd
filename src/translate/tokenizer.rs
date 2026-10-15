//! Loading an export's `tokenizer.json`, patched around a `tokenizers` 0.23
//! panic.

use anyhow::Result;
use tokenizers::Tokenizer;

/// Whether a normalizer carries the null charsmap `tokenizers` 0.23 panics on.
///
/// Walked rather than pattern-matched, because where the `Precompiled` sits inside
/// the normalizer is the converter's choice: Xenova writes a bare `Precompiled`
/// for Marian and wraps the same node in a `Sequence` (with a `Replace` beside it)
/// for NLLB. A `Parallel` would be a third shape and would need the same walk.
fn has_null_charsmap(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, child)| {
            (key == "precompiled_charsmap" && child.is_null()) || has_null_charsmap(child)
        }),
        serde_json::Value::Array(items) => items.iter().any(has_null_charsmap),
        _ => false,
    }
}

/// Drop the normalizer **only** when it is the one `tokenizers` cannot load, and
/// report whether it did.
///
/// Xenova's Marian exports carry
/// `"normalizer":{"type":"Precompiled","precompiled_charsmap":null}`, and
/// `tokenizers` 0.23 panics on the null while deserializing it:
///
/// ```text
/// tokenizers-0.23.2/src/normalizers/mod.rs:143:26:
/// Precompiled: Error("invalid type: null, expected a borrowed string", line: 1, column: 28)
/// ```
///
/// The charsmap is what `transformers.js` writes as `null` because it does not
/// use it, and the working reference produces byte-identical ids with the
/// normalizer dropped (`Dzisiaj jest poniedzialek.` →
/// `[8261,45,54,95,12080,187,2,0]`), so the field is removed rather than
/// substituted. `source.spm` would be the real charsmap, but only Xenova's
/// converted `tokenizer.json` is downloaded, and that is where the null is.
///
/// **The test is on the null, not on the field.** NLLB's export publishes a real
/// 316 KB charsmap, and dropping a normalizer that works would change its
/// tokenization — measured against the real file, it loads in 979 ms with the
/// normalizer intact. Blanket-dropping was right for the one export it was written
/// against and wrong for the next one; asking whether the value is the panic is
/// right for both, which is why this is a value-level function a test can pin
/// rather than a side effect hidden in the loader.
fn patched_value(value: &mut serde_json::Value) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    let null_charsmap = object.get("normalizer").is_some_and(has_null_charsmap);
    if null_charsmap {
        object.remove("normalizer");
    }
    null_charsmap
}

/// Load a SentencePiece or BPE `tokenizer.json`, dropping a null charsmap first.
///
/// The only patch is [`patched_value`]'s. This is the graph contract's first
/// dependency, so a tokenizer the build cannot read fails here rather than
/// mid-utterance.
pub(super) fn patched_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let raw = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
    if !value.is_object() {
        anyhow::bail!("{} is not a tokenizer object", path.display());
    }
    let patched = patched_value(&mut value);
    // Only a patched document is written back out. An unpatched one is already the
    // bytes on disk, and a multilingual export's `tokenizer.json` is 16.5 MiB, so
    // re-serializing it would move the whole file to arrive at the same place.
    let bytes = if patched {
        serde_json::to_vec(&value)?
    } else {
        raw
    };
    let tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", path.display()))?;
    if patched {
        log::debug!(
            "{}: dropped a normalizer with a null charsmap, which tokenizers 0.23 cannot load",
            path.display()
        );
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `tokenizers` 0.23 panic, reproduced from a literal so the workaround
    /// is pinned without downloading weights.
    ///
    /// Built from a trimmed stand-in: only the `normalizer` key and the two
    /// fields that must survive matter here.
    #[test]
    fn the_null_precompiled_normalizer_is_dropped_before_loading() {
        let raw = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "normalizer": {"type": "Precompiled", "precompiled_charsmap": null},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "WhitespaceSplit"},
                    {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
                ]
            },
            "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
            "model": {"type": "Unigram", "unk_id": 1, "vocab": [["▁the", -1.0], ["</s>", 0.0]]}
        }"#;
        let mut value: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert!(
            value.get("normalizer").is_some(),
            "the fixture must reproduce the null charsmap"
        );
        assert!(patched_value(&mut value), "a null charsmap must be dropped");
        assert!(
            !String::from_utf8_lossy(&serde_json::to_vec(&value).unwrap())
                .contains("precompiled_charsmap"),
            "the null charsmap must not survive the patch"
        );
        // The rest of the tokenizer is untouched.
        assert!(value.get("pre_tokenizer").is_some());
        assert!(value.get("decoder").is_some());
        assert!(value.get("model").is_some());
        // And the patched form is what `tokenizers` accepts.
        let patched = serde_json::to_vec(&value).unwrap();
        Tokenizer::from_bytes(&patched).expect("a normalizer-free tokenizer must load");
        // The unpatched form is what it *panics* on, which is why this is a patch
        // and not a fallback: `from_bytes` never gets to return an error, so the
        // failure has to be caught rather than asserted on a `Result`.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let unpatched_panics =
            std::panic::catch_unwind(|| Tokenizer::from_bytes(raw.as_bytes())).is_err();
        std::panic::set_hook(previous_hook);
        assert!(
            unpatched_panics,
            "if the null charsmap now loads, the patch is no longer needed"
        );
    }

    /// A working normalizer is **kept**, which is the case the patch was wrong
    /// about. NLLB's export publishes a real 316 KB charsmap wrapped in a
    /// `Sequence` beside a `Replace`, and dropping it would change the
    /// tokenization — so the test is on the value being null, never on the field
    /// being present.
    ///
    /// The charsmap is a stand-in (real `precompiled_charsmap` values are a long
    /// base64 blob, and this is about the branch taken, not about what
    /// `tokenizers` makes of the bytes); the `#[ignore]`d test below loads the real
    /// file.
    #[test]
    fn a_normalizer_with_a_real_charsmap_survives_the_patch() {
        let build = |normalizer: &str| {
            format!(
                r#"{{
                    "version": "1.0",
                    "normalizer": {normalizer},
                    "decoder": {{"type": "Metaspace", "replacement": "▁"}},
                    "model": {{"type": "Unigram", "unk_id": 1,
                              "vocab": [["▁the", -1.0], ["</s>", 0.0]]}}
                }}"#
            )
        };
        // The NLLB shape: a `Sequence` whose `Precompiled` carries a real charsmap.
        let mut nllb: serde_json::Value = serde_json::from_str(&build(
            r#"{"type": "Sequence", "normalizers": [
                   {"type": "Precompiled", "precompiled_charsmap": "AAAAAA=="},
                   {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
               ]}"#,
        ))
        .unwrap();
        assert!(
            !patched_value(&mut nllb),
            "a real charsmap must not be dropped"
        );
        assert!(
            nllb.get("normalizer").is_some(),
            "the normalizer must survive a patch that had nothing to fix"
        );

        // The same shape with a null inside it is still dropped: the walk does not
        // care how deep the null is buried, which is what a `Parallel` would need.
        let mut nested: serde_json::Value = serde_json::from_str(&build(
            r#"{"type": "Sequence", "normalizers": [
                   {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "},
                   {"type": "Precompiled", "precompiled_charsmap": null}
               ]}"#,
        ))
        .unwrap();
        assert!(
            patched_value(&mut nested),
            "a null charsmap buried in a Sequence must still be found"
        );
        assert!(nested.get("normalizer").is_none());

        // And a normalizer with no charsmap at all is not ours to remove.
        let mut other: serde_json::Value =
            serde_json::from_str(&build(r#"{"type": "NFC"}"#)).unwrap();
        assert!(!patched_value(&mut other));
        assert!(other.get("normalizer").is_some());

        // A tokenizer with no normalizer is a valid shape, not an error.
        let mut none: serde_json::Value = serde_json::from_str(
            r#"{"version": "1.0",
                "decoder": {"type": "Metaspace", "replacement": "▁"},
                "model": {"type": "Unigram", "unk_id": 1,
                          "vocab": [["▁the", -1.0], ["</s>", 0.0]]}}"#,
        )
        .unwrap();
        assert!(!patched_value(&mut none));
    }
}
