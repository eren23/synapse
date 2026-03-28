//! Integration tests for RwkvModel using the public API.

use synapse_inference::model::Model;
use synapse_inference::ssm::{RwkvBlock, RwkvConfig, RwkvModel};

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

fn build_test_model() -> RwkvModel {
    let config = RwkvConfig::tiny_test();
    let h = config.hidden_size;
    let nh = config.num_heads;
    let hs = config.head_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    let embed_tokens = pseudo_random_vec(100, vocab * h);
    let final_norm_weight = vec![1.0f32; h];
    let final_norm_bias = vec![0.0f32; h];
    let lm_head_weight = pseudo_random_vec(200, vocab * h);

    let mut blocks = Vec::new();
    for layer_idx in 0..config.num_layers {
        let seed_base = (layer_idx as u64 + 1) * 1000;
        blocks.push(RwkvBlock {
            hidden_size: h,
            num_heads: nh,
            head_size: hs,
            intermediate_size: inter,
            norm_eps: config.norm_eps as f32,
            ln1_weight: vec![1.0f32; h],
            ln1_bias: vec![0.0f32; h],
            time_mix_x: vec![0.5f32; h],
            receptance_weight: pseudo_random_vec(seed_base + 1, h * h),
            key_weight: pseudo_random_vec(seed_base + 2, h * h),
            value_weight: pseudo_random_vec(seed_base + 3, h * h),
            gate_weight: pseudo_random_vec(seed_base + 4, h * h),
            output_weight: pseudo_random_vec(seed_base + 5, h * h),
            time_decay: pseudo_random_vec(seed_base + 6, nh * hs)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            att_ln_weight: vec![1.0f32; h],
            att_ln_bias: vec![0.0f32; h],
            ln2_weight: vec![1.0f32; h],
            ln2_bias: vec![0.0f32; h],
            channel_mix_x: vec![0.5f32; h],
            ffn_receptance_weight: pseudo_random_vec(seed_base + 7, h * h),
            ffn_key_weight: pseudo_random_vec(seed_base + 8, inter * h),
            ffn_value_weight: pseudo_random_vec(seed_base + 9, h * inter),
        });
    }

    RwkvModel::new(
        config,
        embed_tokens,
        blocks,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
    )
}

#[test]
fn test_rwkv_forward_produces_finite_logits() {
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
fn test_rwkv_prefill_then_decode() {
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
fn test_rwkv_constant_memory_state() {
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
fn test_rwkv_reset_state() {
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
