fn main() {
    // Link Apple Accelerate framework for cblas_sgemm — only when targeting macOS.
    // Uses TARGET env var (not cfg!) because build.rs runs on the host.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple") && !target.contains("wasm") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
