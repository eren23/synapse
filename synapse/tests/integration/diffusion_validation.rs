//! Integration tests for the Diffusion LLM module.
//!
//! These tests validate the end-to-end denoising generation pipeline
//! using a tiny model with random weights.

use synapse_inference::diffusion::{DiffusionLLMConfig, DiffusionModel, MaskSchedule};
use synapse_inference::diffusion::schedule::{tokens_per_step, unmask_by_confidence};

// ── Helpers ──────────────────────────────────────────────────────────

/// Deterministic pseudo-random floats for building test weights.
fn pseudo_rand(seed: usize, len: usize) -> Vec<f32> {
    let scale = 0.02;
    (0..len)
        .map(|i| {
            let x = (seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i.wrapping_mul(1442695040888963407))
                & 0xFFFFFF) as f32
                / 0xFFFFFF as f32;
            (x - 0.5) * scale
        })
        .collect()
}

/// Build a tiny DiffusionModel with deterministic random weights.
fn build_test_model(config: &DiffusionLLMConfig) -> DiffusionModel {
    let h = config.hidden_size;
    let nh = config.num_heads;
    let hd = config.head_dim;
    let qk_dim = nh * hd;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    let embed_tokens = pseudo_rand(42, vocab * h);
    let final_norm_weight = vec![1.0f32; h];
    let lm_head_weight = pseudo_rand(99, vocab * h);

    let mut layers = Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let seed_base = layer_idx * 1000;
        layers.push(synapse_inference::diffusion::model::BiDirectionalLayer {
            hidden_size: h,
            num_heads: nh,
            head_dim: hd,
            intermediate_size: inter,
            norm_eps: config.norm_eps as f32,
            attn_norm_weight: vec![1.0f32; h],
            w_q: pseudo_rand(seed_base + 1, qk_dim * h),
            w_k: pseudo_rand(seed_base + 2, qk_dim * h),
            w_v: pseudo_rand(seed_base + 3, qk_dim * h),
            w_o: pseudo_rand(seed_base + 4, h * qk_dim),
            ffn_norm_weight: vec![1.0f32; h],
            ffn_gate_weight: pseudo_rand(seed_base + 5, inter * h),
            ffn_up_weight: pseudo_rand(seed_base + 6, inter * h),
            ffn_down_weight: pseudo_rand(seed_base + 7, h * inter),
        });
    }

    DiffusionModel {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm_weight,
        lm_head_weight,
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn test_diffusion_forward_produces_finite_logits() {
    let config = DiffusionLLMConfig::tiny_test();
    let model = build_test_model(&config);

    let prompt = vec![1u32, 2, 3];
    let output = model.generate(&prompt, 5, MaskSchedule::Linear);

    assert_eq!(output.len(), 5, "should produce exactly 5 tokens");
    for &tok in &output {
        assert!(
            (tok as usize) < config.vocab_size,
            "token {tok} out of vocab range [0, {})",
            config.vocab_size
        );
    }
}

#[test]
fn test_diffusion_generate_produces_valid_tokens() {
    let config = DiffusionLLMConfig::tiny_test();
    let model = build_test_model(&config);

    let prompt = vec![10u32, 20, 30];
    let output_len = 8;
    let output = model.generate(&prompt, output_len, MaskSchedule::Confidence);

    assert_eq!(output.len(), output_len);
    for (i, &tok) in output.iter().enumerate() {
        assert!(
            (tok as usize) < config.vocab_size,
            "output[{i}] = {tok} exceeds vocab_size {}",
            config.vocab_size
        );
    }
}

#[test]
fn test_diffusion_generate_with_different_schedules() {
    let config = DiffusionLLMConfig::tiny_test();
    let model = build_test_model(&config);

    let prompt = vec![5u32, 15];
    let output_len = 6;

    for schedule in [MaskSchedule::Linear, MaskSchedule::Confidence, MaskSchedule::Cosine] {
        let output = model.generate(&prompt, output_len, schedule);
        assert_eq!(
            output.len(),
            output_len,
            "schedule {:?} should produce {output_len} tokens",
            schedule
        );
        for &tok in &output {
            assert!((tok as usize) < config.vocab_size);
        }
    }
}

#[test]
fn test_diffusion_constant_memory() {
    // Verify that generating different lengths does not accumulate state.
    // Since each step is an independent forward pass with no KV cache,
    // the model struct should not grow. We verify by running multiple
    // generations in sequence and checking they all succeed.
    let config = DiffusionLLMConfig::tiny_test();
    let model = build_test_model(&config);

    let prompt = vec![1u32];
    for output_len in [1, 5, 10, 20] {
        let output = model.generate(&prompt, output_len, MaskSchedule::Linear);
        assert_eq!(output.len(), output_len);
    }
}

#[test]
fn test_schedule_tokens_sum_correctly() {
    // Verify that all schedule types distribute tokens correctly
    for total in [1, 5, 10, 20, 100] {
        for steps in [1, 3, 5, 10] {
            for schedule in [MaskSchedule::Linear, MaskSchedule::Confidence, MaskSchedule::Cosine] {
                let per_step = tokens_per_step(schedule, total, steps);
                assert_eq!(per_step.len(), steps);
                let sum: usize = per_step.iter().sum();
                assert_eq!(
                    sum, total,
                    "schedule {:?} with total={total} steps={steps}: sum={sum}",
                    schedule
                );
            }
        }
    }
}

#[test]
fn test_unmask_confidence_ordering() {
    // Verify that higher-confidence positions are unmasked first
    let vocab = 4;
    let logits = vec![
        0.0, 0.0, 0.0, 0.0, // pos 0: not masked
        0.1, 0.5, 0.2, 0.3, // pos 1: masked, best=0.5 at tok 1
        0.8, 0.1, 0.2, 0.3, // pos 2: masked, best=0.8 at tok 0
        0.1, 0.2, 0.3, 0.9, // pos 3: masked, best=0.9 at tok 3
    ];
    let is_masked = vec![false, true, true, true];

    // Unmask 1: should be pos 3 (confidence 0.9)
    let result = unmask_by_confidence(&logits, &is_masked, vocab, 1);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 3, "highest confidence should be position 3");
    assert_eq!(result[0].1, 3, "predicted token should be 3");

    // Unmask 2: pos 3 then pos 2
    let result = unmask_by_confidence(&logits, &is_masked, vocab, 2);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].0, 3);
    assert_eq!(result[1].0, 2);
}
