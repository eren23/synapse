use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zig_dir: PathBuf = [&manifest_dir, "..", "..", "zig"].iter().collect();
    let zig_dir = zig_dir
        .canonicalize()
        .expect("synapse/zig/ directory not found");
    let zig_src_dir = zig_dir.join("src");
    let zig_lib_dir = zig_dir.join("zig-out").join("lib");
    let zig_cache_dir = zig_dir.join(".zig-cache");
    let zig_global_cache_dir = zig_dir.join(".zig-global-cache");

    std::fs::create_dir_all(&zig_cache_dir).expect("failed to create .zig-cache");
    std::fs::create_dir_all(&zig_global_cache_dir).expect("failed to create .zig-global-cache");

    // Re-run this build script when any Zig source file changes.
    println!("cargo:rerun-if-changed={}", zig_src_dir.display());

    // Also watch the build.zig file itself.
    println!(
        "cargo:rerun-if-changed={}",
        zig_dir.join("build.zig").display()
    );

    // Auto-build Zig if the library doesn't exist or sources changed.
    let lib_path = zig_lib_dir.join("libsynapse_zig.a");
    let needs_build = !lib_path.exists() || {
        // Check if any .zig source is newer than the library
        let lib_mtime = std::fs::metadata(&lib_path).and_then(|m| m.modified()).ok();
        lib_mtime.is_none() || walkdir_any_newer(&zig_src_dir, lib_mtime.unwrap())
    };

    if needs_build {
        eprintln!("synapse-sys: rebuilding Zig library...");
        let profile = if env::var("PROFILE").unwrap_or_default() == "release" {
            "ReleaseFast"
        } else {
            "Debug"
        };

        let status = Command::new("zig")
            .arg("build")
            .arg(format!("-Doptimize={profile}"))
            .arg("-Dtarget=native")
            .arg("--cache-dir")
            .arg(&zig_cache_dir)
            .arg("--global-cache-dir")
            .arg(&zig_global_cache_dir)
            .current_dir(&zig_dir)
            .status()
            .expect("Failed to run `zig build`. Is Zig installed?");

        if !status.success() {
            panic!("Zig build failed with status: {status}");
        }
    }

    let zig_lib_dir = zig_lib_dir
        .canonicalize()
        .expect("zig-out/lib not found after build");

    println!("cargo:rustc-link-search=native={}", zig_lib_dir.display());
    println!("cargo:rustc-link-lib=static=synapse_zig");
}

/// Check if any file under `dir` has a modification time newer than `reference`.
fn walkdir_any_newer(dir: &std::path::Path, reference: std::time::SystemTime) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return true; // Can't read → rebuild to be safe
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if walkdir_any_newer(&path, reference) {
                return true;
            }
        } else if path.extension().is_some_and(|ext| ext == "zig") {
            if let Ok(meta) = std::fs::metadata(&path) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > reference {
                        return true;
                    }
                }
            }
        }
    }
    false
}
