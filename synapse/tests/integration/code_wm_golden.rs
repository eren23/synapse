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

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig, PoolMode};
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
    // Cls variants (g8/g1b/g10/expa) have 47 tensors. Attn variants
    // (ema15k / phase4-contrast-*) add 5 state_encoder.attn_pool.* tensors → 52.
    let expected = match cfg.pool_mode {
        PoolMode::Cls => 47,
        PoolMode::Attn => 52,
    };
    assert_eq!(
        stats.loaded, expected,
        "expected {expected} tensors loaded, got {}. skipped: {:?}",
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

/// ExpA: 192d, 2.4M params, trained on 500K samples, 15K steps (research champion).
#[test]
fn code_wm_expa_end_to_end_golden() {
    let weights = "models/code_wm/expa.safetensors";
    let config = "configs/code_wm_expa.json";
    let goldens_path = "tests/fixtures/code_wm_reference_expa.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing expa artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("expa", &m, &goldens);
}

/// G10: 128d, 1.1M params, 500K samples training (minimum-size fallback).
#[test]
fn code_wm_g10_end_to_end_golden() {
    let weights = "models/code_wm/g10.safetensors";
    let config = "configs/code_wm_g10.json";
    let goldens_path = "tests/fixtures/code_wm_reference_g10.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing g10 artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("g10", &m, &goldens);
}

// ─── Phase 2–4 attn-pool variants ─────────────────────────────────────
//
// These four checkpoints come from the tap's Phase 2–4 training runs
// (Crucible community tap, commit c6492c8). All share the same
// architecture as g8/g1b/g10 (128d, 4 heads, 6 encoder loops, 2×6
// predictor loops) but were trained with WM_POOL_MODE=attn (the tap
// default), so they add a learned attention-pooling readout head.
//
// Expected: 52 tensors loaded (47 standard + 5 attn_pool), parity
// against the Python reference at cos ≥ 0.99999 / max_abs < 5e-5.

/// ema-frozen-15k: best predictor from the Phase 3 run (val_dcos 0.9948).
#[test]
fn code_wm_ema15k_end_to_end_golden() {
    let weights = "models/code_wm/ema15k.safetensors";
    let config = "configs/code_wm_ema15k.json";
    let goldens_path = "tests/fixtures/code_wm_reference_ema15k.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing ema15k artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("ema15k", &m, &goldens);
}

/// phase4-contrast-high: best retriever from the Phase 4 run (λ=1.0
/// supervised contrastive on deltas, beats BoW on by_joint MRR).
#[test]
fn code_wm_contrast_high_end_to_end_golden() {
    let weights = "models/code_wm/contrast_high.safetensors";
    let config = "configs/code_wm_contrast_high.json";
    let goldens_path = "tests/fixtures/code_wm_reference_contrast_high.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing contrast_high artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("contrast_high", &m, &goldens);
}

/// phase4-contrast-mid: λ=0.5 ablation row.
#[test]
fn code_wm_contrast_mid_end_to_end_golden() {
    let weights = "models/code_wm/contrast_mid.safetensors";
    let config = "configs/code_wm_contrast_mid.json";
    let goldens_path = "tests/fixtures/code_wm_reference_contrast_mid.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing contrast_mid artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("contrast_mid", &m, &goldens);
}

/// phase4-contrast-low: λ=0.1 ablation row.
#[test]
fn code_wm_contrast_low_end_to_end_golden() {
    let weights = "models/code_wm/contrast_low.safetensors";
    let config = "configs/code_wm_contrast_low.json";
    let goldens_path = "tests/fixtures/code_wm_reference_contrast_low.safetensors";
    if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
        eprintln!("SKIP: missing contrast_low artifacts.");
        return;
    }
    let m = load_model(weights, config);
    let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
    validate_end_to_end("contrast_low", &m, &goldens);
}

// ─── Phase 5 — variance-corrected sweep + 15K λ ladder ────────────────
//
// Phase 5 ran a full seed-variance + horizon sweep on top of the Phase 4
// results. Same architecture (attn pool, model_dim=128, ema_decay=0.99999,
// no bounded_residual), so no Rust changes — these all use the same
// AttentionPooling head + 47-tensor + 5 attn_pool loader path as the
// Phase 4 contrast_* variants.
//
// Production-relevant headlines (per the Phase 5 Session Report):
//   p5_contrast_high_15k    — NEW retrieval champion (λ=1.0 × 15K, +0.0094 over BoW on cross-repo)
//   p5_contrast_extreme_15k — best in-distribution val at 15K (λ=2.0)
//   p5_ema15k_s{2,3}        — predictor seed variance for the existing ema15k
//
// Helper macro to keep the 10 new tests compact.
macro_rules! p5_golden_test {
    ($fn_name:ident, $variant:literal, $tag:literal) => {
        #[test]
        fn $fn_name() {
            let weights = concat!("models/code_wm/", $variant, ".safetensors");
            let config = concat!("configs/code_wm_", $variant, ".json");
            let goldens_path = concat!("tests/fixtures/code_wm_reference_", $variant, ".safetensors");
            if !Path::new(weights).exists() || !Path::new(goldens_path).exists() {
                eprintln!(concat!("SKIP: missing ", $variant, " artifacts."));
                return;
            }
            let m = load_model(weights, config);
            let goldens = load_safetensors(Path::new(goldens_path)).unwrap();
            validate_end_to_end($tag, &m, &goldens);
        }
    };
}

// Production retrieval champion — λ=1.0 × 15K, the only CodeWM checkpoint
// that clearly beats BoW on the 20-repo cross-repo eval (+0.0094, single seed).
p5_golden_test!(code_wm_p5_contrast_high_15k_golden, "p5_contrast_high_15k", "p5_contrast_high_15k");

// 15K λ=2.0 — best in-distribution val at 15K (peak 0.9917, single seed).
p5_golden_test!(code_wm_p5_contrast_extreme_15k_golden, "p5_contrast_extreme_15k", "p5_contrast_extreme_15k");

// Predictor seed variance: ema-frozen-15K seeds 43 + 44.
// Phase 5 confirmed predictor std=0.0015 across 3 seeds (very tight).
p5_golden_test!(code_wm_p5_ema15k_s2_golden, "p5_ema15k_s2", "p5_ema15k_s2");
p5_golden_test!(code_wm_p5_ema15k_s3_golden, "p5_ema15k_s3", "p5_ema15k_s3");

// 3K λ=2.0 — primary 3-seed variance row.
// Phase 5: λ=2.0 has 6× tighter variance than λ=1.0 at 3K (0.0010 vs 0.0063).
p5_golden_test!(code_wm_p5_contrast_extreme_3k_golden, "p5_contrast_extreme_3k", "p5_contrast_extreme_3k");
p5_golden_test!(code_wm_p5_contrast_extreme_3k_s2_golden, "p5_contrast_extreme_3k_s2", "p5_contrast_extreme_3k_s2");
p5_golden_test!(code_wm_p5_contrast_extreme_3k_s3_golden, "p5_contrast_extreme_3k_s3", "p5_contrast_extreme_3k_s3");

// 3K λ=1.0 seed sweep — seeds 43/44 (the original phase4-contrast-high is seed 42).
p5_golden_test!(code_wm_p5_contrast_high_3k_s2_golden, "p5_contrast_high_3k_s2", "p5_contrast_high_3k_s2");
p5_golden_test!(code_wm_p5_contrast_high_3k_s3_golden, "p5_contrast_high_3k_s3", "p5_contrast_high_3k_s3");

// 3K λ=5.0 — top of the lambda ladder.
p5_golden_test!(code_wm_p5_contrast_mega_3k_golden, "p5_contrast_mega_3k", "p5_contrast_mega_3k");
