//! Sanity test for in-memory Q4_0 quantization of the CodeDeltaTok head.
//!
//! Loads the trained fp32 head, quantizes it in memory, compares delta and
//! recon against the fp32 reference computed by `cdt_trained_parity.rs`'s
//! fixture. Q4 is expected to drift — we assert cosine ≥ 0.98 and
//! ‖Δ‖ / ‖ref‖ < 0.2 on both outputs, which is well above the random
//! baseline.
//!
//! Skips when the `cdt_paper.safetensors` fixture is missing (it isn't
//! tracked in the repo; regenerate with
//! `python scripts/export_unixcoder_reference.py convert-cdt ...`).

use std::path::PathBuf;

use synapse_inference::models::text_encoder::{
    CodeDeltaTokConfig, CodeDeltaTokHead, Q4CodeDeltaTokHead,
};
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

fn rel_l2(a: &[f32], ref_: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(ref_).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
    let den: f32 = ref_.iter().map(|y| y * y).sum::<f32>().sqrt() + 1e-30;
    num / den
}

#[test]
fn cdt_q4_head_tracks_fp32() {
    let fixture_path = repo_root()
        .join("crates/synapse-inference/tests/fixtures/cdt_paper.safetensors");
    if !fixture_path.exists() {
        eprintln!("SKIP: {} missing.", fixture_path.display());
        return;
    }
    let fx = load_safetensors(&fixture_path).expect("load cdt fixture");

    let mut fp32 = CodeDeltaTokHead::from_config(CodeDeltaTokConfig::paper_default());
    fp32.load_weights(fx.clone(), &WeightMapper::code_deltatok())
        .expect("fp32 load_weights");

    // Quantize in memory — no additional safetensors round-trip.
    let q4 = Q4CodeDeltaTokHead::from_fp32(&fp32);
    let q4_bytes = q4.q4_memory_bytes();
    println!(
        "Q4 linear-storage footprint: {:.1} MB  ({:.1}x smaller than fp32 linears)",
        q4_bytes as f32 / 1e6,
        (76.0 * 4.0) / (q4_bytes as f32 / 1e6),
    );

    let d = fp32.config.feature_dim;
    let k = fp32.config.num_delta_tokens;

    let h_b = fx.get("parity.h_b").unwrap();
    let h_a = fx.get("parity.h_a").unwrap();
    let batch = h_b.shape[0];

    let mut worst_delta_cos = 1.0f32;
    let mut worst_recon_cos = 1.0f32;
    let mut worst_delta_rel = 0.0f32;
    let mut worst_recon_rel = 0.0f32;

    for b in 0..batch {
        let hb = &h_b.data[b * d..(b + 1) * d];
        let ha = &h_a.data[b * d..(b + 1) * d];

        let delta_ref = fp32.encode(hb, ha);
        let recon_ref = fp32.decode(&delta_ref, hb);

        let delta_q4 = q4.encode(hb, ha);
        let recon_q4 = q4.decode(&delta_q4, hb);
        assert_eq!(delta_q4.len(), k * d);
        assert_eq!(recon_q4.len(), d);

        worst_delta_cos = worst_delta_cos.min(cosine(&delta_q4, &delta_ref));
        worst_recon_cos = worst_recon_cos.min(cosine(&recon_q4, &recon_ref));
        worst_delta_rel = worst_delta_rel.max(rel_l2(&delta_q4, &delta_ref));
        worst_recon_rel = worst_recon_rel.max(rel_l2(&recon_q4, &recon_ref));
    }

    println!(
        "Q4 drift: delta cos = {worst_delta_cos:.5}  rel_l2 = {worst_delta_rel:.4},  \
         recon cos = {worst_recon_cos:.5}  rel_l2 = {worst_recon_rel:.4}",
    );

    assert!(worst_delta_cos >= 0.98,
            "worst Q4 delta cos {worst_delta_cos} < 0.98 — quantization broke something");
    assert!(worst_recon_cos >= 0.98,
            "worst Q4 recon cos {worst_recon_cos} < 0.98");
    assert!(worst_delta_rel < 0.20,
            "worst Q4 delta rel_l2 {worst_delta_rel} >= 0.20");
    assert!(worst_recon_rel < 0.20,
            "worst Q4 recon rel_l2 {worst_recon_rel} >= 0.20");
}
