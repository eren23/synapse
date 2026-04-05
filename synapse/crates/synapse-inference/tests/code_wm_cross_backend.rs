//! Verify Code WM produces identical outputs with pure-rust and zig-ffi backends.
//! The pure-rust path is what WASM/ESP32 use, so matching zig-ffi proves
//! cross-platform zero drift.
//!
//! Run both:
//!   cargo test -p synapse-inference --test code_wm_cross_backend  # zig-ffi (default)
//!   cargo test -p synapse-inference --test code_wm_cross_backend --no-default-features --features pure-rust

use std::path::{Path, PathBuf};

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn repo_root() -> PathBuf {
    // synapse/crates/synapse-inference → synapse
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

#[test]
fn code_wm_golden_any_backend() {
    let root = repo_root();
    let weights = root.join("models/code_wm/g8.safetensors");
    let config = root.join("configs/code_wm_g8.json");
    let goldens = root.join("tests/fixtures/code_wm_reference_g8.safetensors");
    if !weights.exists() || !goldens.exists() {
        eprintln!("SKIP: missing artifacts at {}", root.display());
        return;
    }

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(&weights).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let g = load_safetensors(&goldens).unwrap();

    // Backend label: compile-time cfg detection.
    let backend = if cfg!(feature = "zig-ffi") { "zig-ffi" } else { "pure-rust" };
    println!("\n[backend = {backend}]");

    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens: Vec<i64> = g[&format!("{pfx}input_tokens")].data.iter().map(|&v| v as i64).collect();
        let action: &[f32] = &g[&format!("{pfx}action_input")].data;

        let z = model.encode(&tokens);
        let z_ref = &g[&format!("{pfx}encoder_final")].data;
        let ma = max_abs(&z, z_ref);
        let co = cosine(&z, z_ref);
        println!("  seed{seed} encoder: max_abs={ma:.3e} cos={co:.10}");
        assert!(ma < 5e-5 && co > 0.99999, "encoder drift");

        let za = model.encode_action(action);
        let za_ref = &g[&format!("{pfx}action_final")].data;
        let ma = max_abs(&za, za_ref);
        let co = cosine(&za, za_ref);
        println!("  seed{seed} action:  max_abs={ma:.3e} cos={co:.10}");
        assert!(ma < 5e-5 && co > 0.99999, "action drift");

        let zs = &g[&format!("{pfx}pred_z_state")].data;
        let za2 = &g[&format!("{pfx}pred_z_action")].data;
        let zn = model.predict(zs, za2);
        let zn_ref = &g[&format!("{pfx}pred_final")].data;
        let ma = max_abs(&zn, zn_ref);
        let co = cosine(&zn, zn_ref);
        println!("  seed{seed} pred:    max_abs={ma:.3e} cos={co:.10}");
        assert!(ma < 5e-5 && co > 0.99999, "predictor drift");
    }
}
