use std::env;
use std::path::PathBuf;

fn main() {
    // Locate the Zig-built static library relative to the workspace root.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zig_lib_dir: PathBuf = [&manifest_dir, "..", "..", "zig", "zig-out", "lib"]
        .iter()
        .collect();
    let zig_lib_dir = zig_lib_dir
        .canonicalize()
        .expect("zig-out/lib not found — run `zig build` in synapse/zig/ first");

    println!("cargo:rustc-link-search=native={}", zig_lib_dir.display());
    println!("cargo:rustc-link-lib=static=synapse_zig");
}
