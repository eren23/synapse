//! Parity test for the CodeDeltaTok head.
//!
//! No trained weights are needed: `scripts/export_unixcoder_reference.py
//! random-cdt` dumps a fresh random-initialized PyTorch CDT model plus its
//! outputs on a handful of synthetic (h_b, h_a) pairs. We load the same
//! state dict into the Rust port and assert forward-pass outputs match.
//!
//! Skips cleanly if the fixture is missing.

use std::path::PathBuf;

use synapse_inference::models::text_encoder::{CodeDeltaTokConfig, CodeDeltaTokHead};
use synapse_inference::weight_loading::{load_safetensors, WeightMapper};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn cdt_head_matches_pytorch_random_init() {
    let fixture_path = repo_root()
        .join("crates/synapse-inference/tests/fixtures/cdt_random.safetensors");
    if !fixture_path.exists() {
        eprintln!("SKIP: {} missing. Run \
                   `python scripts/export_unixcoder_reference.py random-cdt \
                   --out crates/synapse-inference/tests/fixtures/cdt_random.safetensors`.",
                  fixture_path.display());
        return;
    }
    let fx = load_safetensors(&fixture_path).expect("load cdt fixture");

    let mut head = CodeDeltaTokHead::from_config(CodeDeltaTokConfig::paper_default());
    let result = head.load_weights(fx.clone(), &WeightMapper::code_deltatok())
        .expect("load_weights");
    assert!(
        result.missing.is_empty(),
        "Missing CDT targets: {:?}", result.missing,
    );
    // Tolerated leftovers: the four `inputs.*` / `golden.*` parity tensors.
    for key in &result.unexpected {
        let ok = matches!(
            key.as_str(),
            "inputs.h_b" | "inputs.h_a" | "golden.delta" | "golden.recon",
        );
        assert!(ok, "Unexpected CDT source key: {key}");
    }

    let d = head.config.feature_dim;
    let k = head.config.num_delta_tokens;

    let h_b = fx.get("inputs.h_b").unwrap();
    let h_a = fx.get("inputs.h_a").unwrap();
    let delta_ref = fx.get("golden.delta").unwrap();
    let recon_ref = fx.get("golden.recon").unwrap();
    let batch = h_b.shape[0];
    assert_eq!(h_b.shape, vec![batch, d]);
    assert_eq!(delta_ref.shape, vec![batch, k, d]);
    assert_eq!(recon_ref.shape, vec![batch, d]);

    let mut worst_delta_cos = 1.0f32;
    let mut worst_recon_cos = 1.0f32;
    let mut worst_delta_abs = 0.0f32;
    let mut worst_recon_abs = 0.0f32;

    for b in 0..batch {
        let hb = &h_b.data[b * d..(b + 1) * d];
        let ha = &h_a.data[b * d..(b + 1) * d];

        let delta = head.encode(hb, ha);
        let recon = head.decode(&delta, hb);

        let delta_t = &delta_ref.data[b * k * d..(b + 1) * k * d];
        let recon_t = &recon_ref.data[b * d..(b + 1) * d];

        let dc = cosine(&delta, delta_t);
        let rc = cosine(&recon, recon_t);
        let da = max_abs(&delta, delta_t);
        let ra = max_abs(&recon, recon_t);

        worst_delta_cos = worst_delta_cos.min(dc);
        worst_recon_cos = worst_recon_cos.min(rc);
        worst_delta_abs = worst_delta_abs.max(da);
        worst_recon_abs = worst_recon_abs.max(ra);
    }

    println!(
        "CDT parity (random init, batch={batch}): \
         worst delta cos = {worst_delta_cos:.6}, recon cos = {worst_recon_cos:.6}, \
         delta max_abs = {worst_delta_abs:.5}, recon max_abs = {worst_recon_abs:.5}",
    );

    assert!(worst_delta_cos >= 0.9999,
            "worst delta cosine {worst_delta_cos} < 0.9999");
    assert!(worst_recon_cos >= 0.9999,
            "worst recon cosine {worst_recon_cos} < 0.9999");
    assert!(worst_delta_abs < 1e-3,
            "worst delta max_abs {worst_delta_abs} >= 1e-3");
    assert!(worst_recon_abs < 1e-3,
            "worst recon max_abs {worst_recon_abs} >= 1e-3");
}
