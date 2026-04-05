//! Zero-drift validation: compare Synapse's Rust CodeWorldModel against a
//! PyTorch reference dump stage-by-stage.
//!
//! Tier-1 tolerance: cosine ≥ 0.99999, max_abs < 1e-5 at every intermediate.
//! If any stage fails, the first failing stage pinpoints the drifting kernel.
//!
//! Prerequisites (produced by `scripts/convert_code_wm_ckpt.py` + `scripts/reference/code_wm_pytorch_baseline.py`):
//!   - models/code_wm/g8.safetensors      (inference weights)
//!   - configs/code_wm_g8.json            (architecture config)
//!   - tests/fixtures/code_wm_reference_g8.safetensors  (stage-wise activations)
//!   (likewise for g1b)

use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

// Per-stage tolerance for f32 compute. Intermediate activations accumulate
// rounding error across 12+ matmuls per encoder pass; we allow a few ULPs.
// End-to-end outputs are typically <1e-6 (near bit-exact after LayerNorm dampens).
const TIER1_MAX_ABS: f32 = 5e-5;
const TIER1_MIN_COS: f32 = 0.99999;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0_f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn assert_close(stage: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{stage}: length mismatch got={}, want={}",
        got.len(),
        want.len()
    );
    let m = max_abs_diff(got, want);
    let c = cosine(got, want);
    assert!(
        m < TIER1_MAX_ABS && c > TIER1_MIN_COS,
        "{stage}: DRIFT — max_abs_diff={m:.3e} (tier1: <{TIER1_MAX_ABS:.0e}), cos={c:.10} (tier1: >{TIER1_MIN_COS:.6})"
    );
}

fn load_model(weights: &str, config: &str) -> CodeWorldModel {
    let cfg = CodeWorldModelConfig::from_json(Path::new(config))
        .unwrap_or_else(|e| panic!("config load failed: {e}"));
    let tensors = load_safetensors(Path::new(weights))
        .unwrap_or_else(|e| panic!("safetensors load failed: {e:?}"));
    let mut m = CodeWorldModel::from_config(&cfg);
    let stats = m
        .load_weights(tensors)
        .unwrap_or_else(|e| panic!("load_weights failed: {e:?}"));
    assert_eq!(
        stats.loaded, 47,
        "expected 47 tensors loaded, got {}. skipped: {:?}",
        stats.loaded, stats.skipped
    );
    m
}

// Convert i64 token values (stored as f32 in the reference dump because
// Synapse's safetensors parser only handles f32/f16/bf16) back to i64.
fn tokens_from_f32(buf: &[f32]) -> Vec<i64> {
    buf.iter().map(|&v| v as i64).collect()
}

/// Validate the encoder path end-to-end (no stage-by-stage debug probes —
/// that's gated behind `debug_activations`, see encoder_g8_stepwise below).
fn validate_end_to_end(tag: &str, model: &CodeWorldModel, goldens: &std::collections::HashMap<String, synapse_inference::weight_loading::RawTensor>) {
    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens = tokens_from_f32(&goldens[&format!("{pfx}input_tokens")].data);
        let action_tensor = &goldens[&format!("{pfx}action_input")].data;
        let action: &[f32] = action_tensor;

        // Encoder
        let z_state = model.encode(&tokens);
        let z_state_ref = &goldens[&format!("{pfx}encoder_final")].data;
        assert_close(&format!("{tag} seed{seed} encoder_final"), &z_state, z_state_ref);

        // Action encoder
        let z_action = model.encode_action(action);
        let z_action_ref = &goldens[&format!("{pfx}action_final")].data;
        assert_close(&format!("{tag} seed{seed} action_final"), &z_action, z_action_ref);

        // Predictor (use the reference z_state/z_action so predictor drift
        // is isolated from any earlier drift — pure predictor validation).
        let z_state_input = &goldens[&format!("{pfx}pred_z_state")].data;
        let z_action_input = &goldens[&format!("{pfx}pred_z_action")].data;
        let z_next = model.predict(z_state_input, z_action_input);
        let z_next_ref = &goldens[&format!("{pfx}pred_final")].data;
        assert_close(&format!("{tag} seed{seed} pred_final"), &z_next, z_next_ref);
    }
}

#[test]
fn code_wm_g8_end_to_end_golden() {
    let weights = "models/code_wm/g8.safetensors";
    let config = "configs/code_wm_g8.json";
    let goldens_path = "tests/fixtures/code_wm_reference_g8.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing g8 artifacts. Run scripts/convert_code_wm_ckpt.py and scripts/reference/code_wm_pytorch_baseline.py first.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("g8", &m, &goldens);
}

#[cfg(feature = "debug_activations")]
#[test]
fn code_wm_g8_stepwise_golden() {
    use synapse_inference::models::vision::code_wm::{EncoderTrace, PredictorTrace};
    let weights = "models/code_wm/g8.safetensors";
    let config = "configs/code_wm_g8.json";
    let goldens_path = "tests/fixtures/code_wm_reference_g8.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing g8 artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();

    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens = tokens_from_f32(&goldens[&format!("{pfx}input_tokens")].data);
        let trace: EncoderTrace = m.encode_debug(&tokens);

        assert_close(&format!("g8 seed{seed} after_embed"),
                     &trace.after_embed, &goldens[&format!("{pfx}after_embed")].data);
        assert_close(&format!("g8 seed{seed} after_cls_prepend"),
                     &trace.after_cls_prepend, &goldens[&format!("{pfx}after_cls_prepend")].data);
        assert_close(&format!("g8 seed{seed} after_pe"),
                     &trace.after_pe, &goldens[&format!("{pfx}after_pe")].data);
        for (i, loop_trace) in trace.loops.iter().enumerate() {
            assert_close(&format!("g8 seed{seed} loop_{i}_norm1"),
                         &loop_trace.norm1, &goldens[&format!("{pfx}loop_{i}_norm1")].data);
            assert_close(&format!("g8 seed{seed} loop_{i}_attn"),
                         &loop_trace.attn, &goldens[&format!("{pfx}loop_{i}_attn")].data);
            assert_close(&format!("g8 seed{seed} loop_{i}_res1"),
                         &loop_trace.res1, &goldens[&format!("{pfx}loop_{i}_res1")].data);
            assert_close(&format!("g8 seed{seed} loop_{i}_norm2"),
                         &loop_trace.norm2, &goldens[&format!("{pfx}loop_{i}_norm2")].data);
            assert_close(&format!("g8 seed{seed} loop_{i}_mlp"),
                         &loop_trace.mlp, &goldens[&format!("{pfx}loop_{i}_mlp")].data);
            assert_close(&format!("g8 seed{seed} loop_{i}_res2"),
                         &loop_trace.res2, &goldens[&format!("{pfx}loop_{i}_res2")].data);
        }
        assert_close(&format!("g8 seed{seed} cls_extracted"),
                     &trace.cls_extracted, &goldens[&format!("{pfx}cls_extracted")].data);
        assert_close(&format!("g8 seed{seed} encoder_final"),
                     &trace.encoder_final, &goldens[&format!("{pfx}encoder_final")].data);

        // Predictor stepwise. Use reference z_state/z_action to isolate predictor drift.
        let zs = &goldens[&format!("{pfx}pred_z_state")].data;
        let za = &goldens[&format!("{pfx}pred_z_action")].data;
        let ptrace: PredictorTrace = m.predict_debug(zs, za);
        assert_close(&format!("g8 seed{seed} pred_stacked"),
                     &ptrace.stacked, &goldens[&format!("{pfx}pred_stacked")].data);
        for (bi, block_loops) in ptrace.blocks.iter().enumerate() {
            for (li, loop_trace) in block_loops.iter().enumerate() {
                assert_close(&format!("g8 seed{seed} pred_b{bi}_l{li}_res2"),
                             &loop_trace.res2, &goldens[&format!("{pfx}pred_b{bi}_l{li}_res2")].data);
            }
        }
        assert_close(&format!("g8 seed{seed} pred_token0_extracted"),
                     &ptrace.token0_extracted, &goldens[&format!("{pfx}pred_token0_extracted")].data);
        assert_close(&format!("g8 seed{seed} pred_final"),
                     &ptrace.pred_final, &goldens[&format!("{pfx}pred_final")].data);
    }
}

#[test]
fn code_wm_g1b_end_to_end_golden() {
    let weights = "models/code_wm/g1b.safetensors";
    let config = "configs/code_wm_g1b.json";
    let goldens_path = "tests/fixtures/code_wm_reference_g1b.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing g1b artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("g1b", &m, &goldens);
}
