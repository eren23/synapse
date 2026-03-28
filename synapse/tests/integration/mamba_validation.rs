//! Integration tests for MambaModel using the public API.

use synapse_inference::model::Model;
use synapse_inference::ssm::{MambaBlock, MambaConfig, MambaModel};

fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
            (f32::from_bits(bits) - 1.5) * 0.2
        })
        .collect()
}

fn build_test_model() -> MambaModel {
    let config = MambaConfig::tiny_test();
    let d_model = config.d_model;
    let d_inner = config.d_inner();
    let d_state = config.d_state;
    let d_conv = config.d_conv;
    let vocab = config.vocab_size;

    let embed_tokens = pseudo_random_vec(100, vocab * d_model);
    let final_norm_weight = vec![1.0f32; d_model];
    let lm_head_weight = pseudo_random_vec(200, vocab * d_model);

    let mut blocks = Vec::new();
    for layer_idx in 0..config.num_layers {
        let seed_base = (layer_idx as u64 + 1) * 1000;
        blocks.push(MambaBlock {
            d_model,
            d_inner,
            d_state,
            d_conv,
            norm_weight: vec![1.0f32; d_model],
            norm_eps: config.norm_eps as f32,
            in_proj_weight: pseudo_random_vec(seed_base + 1, 2 * d_inner * d_model),
            in_proj_bias: vec![],
            conv1d_weight: pseudo_random_vec(seed_base + 2, d_inner * d_conv),
            conv1d_bias: vec![0.0f32; d_inner],
            x_proj_weight: pseudo_random_vec(seed_base + 3, (2 * d_state + 1) * d_inner),
            dt_proj_weight: pseudo_random_vec(seed_base + 4, d_inner),
            dt_proj_bias: vec![0.0f32; d_inner],
            a_log: pseudo_random_vec(seed_base + 5, d_inner * d_state)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            d_param: vec![1.0f32; d_inner],
            out_proj_weight: pseudo_random_vec(seed_base + 6, d_model * d_inner),
            out_proj_bias: vec![],
        });
    }

    MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
}

#[test]
fn test_mamba_forward_produces_finite_logits() {
    let model = build_test_model();
    let vocab = model.config.vocab_size;

    let output = model.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 1, vocab]);
    assert_eq!(output.logits.len(), vocab);
    for (i, &v) in output.logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
    }
}

#[test]
fn test_mamba_prefill_then_decode() {
    let model = build_test_model();
    let vocab = model.config.vocab_size;

    // Prefill
    model.reset_state();
    let out1 = model.prefill(&[1, 2, 3]);
    assert_eq!(out1.shape, [1, 1, vocab]);
    for (i, &v) in out1.logits.iter().enumerate() {
        assert!(v.is_finite(), "prefill logit[{i}] = {v} is not finite");
    }

    // Decode step 1
    let out2 = model.decode_one(4);
    assert_eq!(out2.shape, [1, 1, vocab]);
    for (i, &v) in out2.logits.iter().enumerate() {
        assert!(v.is_finite(), "decode1 logit[{i}] = {v} is not finite");
    }

    // Decode step 2
    let out3 = model.decode_one(5);
    assert_eq!(out3.shape, [1, 1, vocab]);
    for (i, &v) in out3.logits.iter().enumerate() {
        assert!(v.is_finite(), "decode2 logit[{i}] = {v} is not finite");
    }
}

#[test]
fn test_mamba_constant_memory_state() {
    let model = build_test_model();
    let mem_before = model.state_memory_bytes();
    assert!(mem_before > 0, "state memory should be nonzero");

    model.reset_state();
    let _ = model.prefill(&[1, 2, 3]);
    let _ = model.decode_one(4);
    let _ = model.decode_one(5);

    let mem_after = model.state_memory_bytes();
    assert_eq!(
        mem_before, mem_after,
        "state memory must stay constant: before={mem_before}, after={mem_after}"
    );
}

#[test]
fn test_mamba_reset_state() {
    let model = build_test_model();
    let tokens = &[1u32, 2, 3];

    // Run 1: prefill + decode
    model.reset_state();
    let out_prefill_1 = model.prefill(tokens);
    let out_decode_1 = model.decode_one(4);

    // Reset and run again with the same sequence
    model.reset_state();
    let out_prefill_2 = model.prefill(tokens);
    let out_decode_2 = model.decode_one(4);

    // Prefill outputs should be identical
    assert_eq!(out_prefill_1.logits.len(), out_prefill_2.logits.len());
    for (i, (&a, &b)) in out_prefill_1
        .logits
        .iter()
        .zip(out_prefill_2.logits.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "prefill logit[{i}] mismatch after reset: {a} vs {b}"
        );
    }

    // Decode outputs should be identical
    assert_eq!(out_decode_1.logits.len(), out_decode_2.logits.len());
    for (i, (&a, &b)) in out_decode_1
        .logits
        .iter()
        .zip(out_decode_2.logits.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "decode logit[{i}] mismatch after reset: {a} vs {b}"
        );
    }
}
