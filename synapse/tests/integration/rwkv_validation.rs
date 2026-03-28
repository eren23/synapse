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
    let dr = config.decay_rank;
    let ar = config.alpha_rank;
    let gr = config.gate_rank;
    let vocab = config.vocab_size;

    let embed_tokens = pseudo_random_vec(100, vocab * h);
    let final_norm_weight = vec![1.0f32; h];
    let final_norm_bias = vec![0.0f32; h];
    let lm_head_weight = pseudo_random_vec(200, vocab * h);

    let mut blocks = Vec::new();
    for layer_idx in 0..config.num_layers {
        let s = (layer_idx as u64 + 1) * 1000;
        blocks.push(RwkvBlock {
            hidden_size: h, num_heads: nh, head_size: hs,
            intermediate_size: inter, decay_rank: dr, alpha_rank: ar, gate_rank: gr,
            norm_eps: config.norm_eps as f32,
            ln1_weight: vec![1.0f32; h], ln1_bias: vec![0.0f32; h],
            x_r: pseudo_random_vec(s+10, h), x_k: pseudo_random_vec(s+11, h),
            x_v: pseudo_random_vec(s+12, h), x_w: pseudo_random_vec(s+13, h),
            x_a: pseudo_random_vec(s+14, h), x_g: pseudo_random_vec(s+15, h),
            r_proj: pseudo_random_vec(s+1, h*h), k_proj: pseudo_random_vec(s+2, h*h),
            v_proj: pseudo_random_vec(s+3, h*h), o_proj: pseudo_random_vec(s+5, h*h),
            w0: pseudo_random_vec(s+20, h),
            w1: pseudo_random_vec(s+21, h*dr), w2: pseudo_random_vec(s+22, dr*h),
            a0: pseudo_random_vec(s+30, h),
            a1: pseudo_random_vec(s+31, h*ar), a2: pseudo_random_vec(s+32, ar*h),
            g1: pseudo_random_vec(s+40, h*gr), g2: pseudo_random_vec(s+41, gr*h),
            k_k: vec![1.0f32; h], k_a: vec![1.0f32; h],
            r_k: pseudo_random_vec(s+50, nh*hs),
            g_norm_weight: vec![1.0f32; h], g_norm_bias: vec![0.0f32; h],
            v_rank: 0,
            v0: vec![],
            v1: vec![],
            v2: vec![],
            ln2_weight: vec![1.0f32; h], ln2_bias: vec![0.0f32; h],
            ffn_x_k: pseudo_random_vec(s+60, h),
            ffn_key_weight: pseudo_random_vec(s+8, inter*h),
            ffn_value_weight: pseudo_random_vec(s+9, h*inter),
        });
    }

    RwkvModel::new(config, embed_tokens, blocks, final_norm_weight, final_norm_bias, lm_head_weight)
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
