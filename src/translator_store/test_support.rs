//! Fixtures the resolution and model-facts tests share: an export directory
//! written to a temp path, and the two families' `config.json` files.

use std::path::{Path, PathBuf};

use super::fetch::{CONFIG_FILE, REQUIRED_FILES};

/// A stand-in export directory: the four files, zero bytes each, plus a
/// `config.json` describing the export the way its own would.
pub(super) fn write_dir(dir: &Path, names: &[&str], config: &str) {
    std::fs::create_dir_all(dir).unwrap();
    for name in names {
        if *name == CONFIG_FILE {
            std::fs::write(dir.join(name), config).unwrap();
            continue;
        }
        std::fs::write(dir.join(name), b"x").unwrap();
    }
}

pub(super) fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sttd-translator-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

pub(super) const ALL_FILES: [&str; 5] = REQUIRED_FILES;

/// `opus-mt-pl-en`'s own numbers, as literals. What sttd compiles in is only
/// the *absence* of these; the values arrive from the file.
pub(super) const MARIAN_CONFIG: &str = r#"{
    "model_type": "marian",
    "decoder_layers": 6, "decoder_attention_heads": 8, "d_model": 512,
    "decoder_start_token_id": 63429, "eos_token_id": 0, "pad_token_id": 63429
}"#;
/// `nllb-200-distilled-600M`'s own numbers. Note `decoder_start_token_id` and
/// `eos_token_id` are **both 2**: the config sttd must accept, and the one a
/// family-blind check would refuse.
pub(super) const NLLB_CONFIG: &str = r#"{
    "model_type": "m2m_100",
    "decoder_layers": 12, "decoder_attention_heads": 16, "d_model": 1024,
    "decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1
}"#;
