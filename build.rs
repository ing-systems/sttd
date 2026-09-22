// Link args scoped to the `sherpa` feature: sherpa-onnx-sys (shared mode)
// copies `libsherpa-onnx-*.so` next to the binary, so bake `$ORIGIN` into
// the runpath — without this the binary only runs with `LD_LIBRARY_PATH`
// pointed at its own directory. No-op without the feature.
fn main() {
    if std::env::var("CARGO_FEATURE_SHERPA").is_ok() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
}
