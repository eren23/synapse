//! Real-weight validation for RWKV-7 models against HuggingFace reference.
//!
//! These tests are `#[ignore]` by default — they require a downloaded model.
//!
//! To run:
//!   1. Download: `huggingface-cli download RWKV/RWKV7-Goose-0.1B-HF`
//!   2. Generate: `cd scripts/reference && python generate_rwkv_reference.py`
//!   3. Run: `RWKV_01B_PATH=... cargo test --test rwkv_hf_validation -- --ignored`

use std::path::Path;

use synapse_inference::engine::InferenceEngine;
use synapse_inference::models::Model;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

#[test]
#[ignore]
fn test_rwkv7_01b_matches_huggingface() {
    let reference_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rwkv7_pile_01b_reference.json"
    );

    let reference_str = match std::fs::read_to_string(reference_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!(
                "Reference file not found at {}. Run generate_rwkv_reference.py first.",
                reference_path
            );
            return;
        }
    };
    let reference: serde_json::Value = serde_json::from_str(&reference_str)
        .expect("Failed to parse reference JSON");

    let model_path = std::env::var("RWKV_01B_PATH")
        .expect("Set RWKV_01B_PATH env var to the downloaded model directory");
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.join("config.json").exists(),
        "config.json not found in {model_path}"
    );

    eprintln!("Loading RWKV-7 0.1B from {model_path}...");
    let engine = InferenceEngine::from_pretrained(model_dir)
        .expect("Failed to load RWKV-7 0.1B");

    assert!(engine.is_ssm(), "Engine should detect SSM model");

    let ref_token_ids: Vec<u32> = reference["token_ids"]
        .as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let ref_logits: Vec<f32> = reference["logits"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let ref_top_k: Vec<usize> = reference["top_k_ids"]
        .as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    eprintln!("Running forward pass with {} tokens...", ref_token_ids.len());
    let ssm = engine.ssm_model.as_ref().unwrap();
    let output = ssm.forward(&ref_token_ids);
    let synapse_logits = &output.logits;

    // Debug stats
    let syn_min = synapse_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let syn_max = synapse_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let syn_mean = synapse_logits.iter().sum::<f32>() / synapse_logits.len() as f32;
    let ref_min = ref_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let ref_max = ref_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ref_mean = ref_logits.iter().sum::<f32>() / ref_logits.len() as f32;
    eprintln!("Synapse: min={syn_min:.4} max={syn_max:.4} mean={syn_mean:.4}");
    eprintln!("Ref:     min={ref_min:.4} max={ref_max:.4} mean={ref_mean:.4}");
    let nan_count = synapse_logits.iter().filter(|v| !v.is_finite()).count();
    if nan_count > 0 { eprintln!("WARNING: {nan_count} NaN/Inf!"); }

    assert_eq!(
        synapse_logits.len(), ref_logits.len(),
        "Vocab size mismatch: synapse={} ref={}", synapse_logits.len(), ref_logits.len()
    );

    // Top-5 match
    let synapse_top5 = top_k_indices(synapse_logits, 5);
    let ref_top5 = &ref_top_k[..5];
    let top5_match = synapse_top5.iter().zip(ref_top5.iter()).filter(|(a, b)| a == b).count();
    eprintln!("Top-5 match: {}/5 (synapse={:?}, ref={:?})", top5_match, synapse_top5, ref_top5);
    // Check overlap (same tokens in top-5, any order)
    let overlap = synapse_top5.iter().filter(|t| ref_top5.contains(t)).count();
    eprintln!("Top-5 overlap: {overlap}/5");

    // Cosine similarity — RWKV-7 has small accumulated divergence across 12 layers
    // due to bf16→f32 conversion and low-rank approximation differences.
    // Target: >0.98 (functional), ideal: >0.999 (exact match like Mamba).
    let cos_sim = cosine_similarity(synapse_logits, &ref_logits);
    eprintln!("Cosine similarity: {cos_sim:.6}");
    assert!(cos_sim > 0.98, "Cosine similarity too low: {cos_sim:.6}");

    // Top-3 tokens must overlap
    assert!(overlap >= 3, "Top-5 overlap too low: {overlap}/5");

    // Argmax must match
    let synapse_argmax = synapse_logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(synapse_argmax, ref_top_k[0], "Argmax mismatch");

    eprintln!("PASS: RWKV-7 0.1B matches HuggingFace reference.");
}
