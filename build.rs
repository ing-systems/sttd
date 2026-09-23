//! Bakes the catalog model directory into the binary. Compile-time only:
//! `STTD_MODELS_DIR` is read while building and does nothing when `sttd`
//! runs — the packaged Arch build sets it to `/usr/share/sttd/models`
//! (see `packaging/PKGBUILD`); an unset or blank value keeps the dev
//! default `models` (CWD-relative, next to the checkout).
//!
//! `rerun-if-env-changed` is what makes flipping the variable rebuild the
//! crate: Cargo does not otherwise know the program reads an env var, so
//! without it an existing `target/` would keep the stale baked path.
fn main() {
    println!("cargo:rerun-if-env-changed=STTD_MODELS_DIR");
    let dir = std::env::var("STTD_MODELS_DIR").unwrap_or_default();
    let dir = dir.trim();
    println!(
        "cargo:rustc-env=STTD_MODELS_DIR={}",
        if dir.is_empty() { "models" } else { dir }
    );
}
