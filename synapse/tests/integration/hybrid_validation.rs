//! Integration tests for HybridModel (Qwen3.5-style DeltaNet + GQA).

use synapse_inference::model::Model;
use synapse_inference::ssm::{
    DeltaNetDecoderLayer, GqaDecoderLayer, HybridConfig, HybridLayer, HybridModel,
};

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

fn make_rope_tables(max_pos: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half_d = head_dim / 2;
    let mut cos = vec![0.0f32; max_pos * half_d];
    let mut sin = vec![0.0f32; max_pos * half_d];
    for pos in 0..max_pos {
        for i in 0..half_d {
            let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos[pos * half_d + i] = angle.cos();
            sin[pos * half_d + i] = angle.sin();
        }
    }
    (cos, sin)
}

fn build_test_model() -> HybridModel {
    let config = HybridConfig::tiny_test();
    let d = config.hidden_size;
    let vocab = config.vocab_size;
    let nh_dn = config.deltanet_num_heads;
    let hd_dn = config.deltanet_head_dim;
    let ck = config.deltanet_conv_kernel;
    let nq = config.num_attention_heads;
    let nkv = config.num_kv_heads;
    let hd_gqa = config.gqa_head_dim;
    let im = config.intermediate_size;
    let nh_hd_dn = nh_dn * hd_dn;

    let embed_tokens = pseudo_random_vec(100, vocab * d);
    let final_norm_weight = vec![1.0f32; d];
    let lm_head_weight = pseudo_random_vec(200, vocab * d);

    let max_kv_seq = 64;
    let (rope_cos, rope_sin) = make_rope_tables(max_kv_seq, hd_gqa);

    let mut layers = Vec::new();
    for layer_idx in 0..config.num_layers {
        let seed_base = (layer_idx as u64 + 1) * 1000;
        if config.is_full_attention(layer_idx) {
            layers.push(HybridLayer::Gqa(GqaDecoderLayer {
                hidden_size: d,
                num_q_heads: nq,
                num_kv_heads: nkv,
                head_dim: hd_gqa,
                intermediate_size: im,
                norm_eps: config.norm_eps as f32,
                attn_norm_weight: vec![1.0; d],
                w_q: pseudo_random_vec(seed_base + 1, nq * hd_gqa * d),
                w_k: pseudo_random_vec(seed_base + 2, nkv * hd_gqa * d),
                w_v: pseudo_random_vec(seed_base + 3, nkv * hd_gqa * d),
                w_o: pseudo_random_vec(seed_base + 4, d * nq * hd_gqa),
                q_norm_weight: vec![1.0; hd_gqa],
                k_norm_weight: vec![1.0; hd_gqa],
                ffn_norm_weight: vec![1.0; d],
                ffn_gate_weight: pseudo_random_vec(seed_base + 5, im * d),
                ffn_up_weight: pseudo_random_vec(seed_base + 6, im * d),
                ffn_down_weight: pseudo_random_vec(seed_base + 7, d * im),
            }));
        } else {
            layers.push(HybridLayer::DeltaNet(DeltaNetDecoderLayer {
                hidden_size: d,
                num_heads: nh_dn,
                head_dim: hd_dn,
                intermediate_size: im,
                conv_kernel: ck,
                norm_eps: config.norm_eps as f32,
                attn_norm_weight: vec![1.0; d],
                qkv_weight: pseudo_random_vec(seed_base + 1, 3 * nh_hd_dn * d),
                gate_proj_weight: pseudo_random_vec(seed_base + 2, nh_hd_dn * d),
                beta_proj_weight: pseudo_random_vec(seed_base + 3, nh_dn * d),
                alpha_proj_weight: pseudo_random_vec(seed_base + 4, nh_dn * d),
                q_conv_weight: pseudo_random_vec(seed_base + 5, nh_hd_dn * ck),
                q_conv_bias: vec![0.0; nh_hd_dn],
                k_conv_weight: pseudo_random_vec(seed_base + 6, nh_hd_dn * ck),
                k_conv_bias: vec![0.0; nh_hd_dn],
                v_conv_weight: pseudo_random_vec(seed_base + 7, nh_hd_dn * ck),
                v_conv_bias: vec![0.0; nh_hd_dn],
                o_norm_weight: vec![1.0; nh_hd_dn],
                o_proj_weight: pseudo_random_vec(seed_base + 8, d * nh_hd_dn),
                ffn_norm_weight: vec![1.0; d],
                ffn_gate_weight: pseudo_random_vec(seed_base + 9, im * d),
                ffn_up_weight: pseudo_random_vec(seed_base + 10, im * d),
                ffn_down_weight: pseudo_random_vec(seed_base + 11, d * im),
            }));
        }
    }

    HybridModel::new(
        config,
        embed_tokens,
        layers,
        final_norm_weight,
        Some(lm_head_weight),
        rope_cos,
        rope_sin,
        max_kv_seq,
    )
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn test_hybrid_forward_produces_finite_logits() {
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
fn test_hybrid_prefill_then_decode() {
    let model = build_test_model();
    let vocab = model.config.vocab_size;

    model.reset_state();
    let out1 = model.prefill(&[10, 20, 30]);
    assert_eq!(out1.shape, [1, 1, vocab]);
    for (i, &v) in out1.logits.iter().enumerate() {
        assert!(v.is_finite(), "prefill logit[{i}] = {v} is not finite");
    }

    // Decode several more tokens
    for tok in 40..45u32 {
        let out = model.decode_one(tok);
        assert_eq!(out.shape, [1, 1, vocab]);
        for (i, &v) in out.logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "decode tok={tok} logit[{i}] = {v} is not finite"
            );
        }
    }
}

#[test]
fn test_hybrid_deltanet_state_constant() {
    let model = build_test_model();

    // DeltaNet state allocation is constant regardless of sequence length
    let dn_before = model.deltanet_state_memory_bytes();
    assert!(dn_before > 0);

    model.reset_state();
    let _ = model.prefill(&[1, 2, 3, 4, 5]);
    let dn_after_prefill = model.deltanet_state_memory_bytes();
    assert_eq!(dn_before, dn_after_prefill);

    // Decode more tokens — DeltaNet memory still constant
    for tok in 6..15u32 {
        let _ = model.decode_one(tok);
    }
    let dn_after_decode = model.deltanet_state_memory_bytes();
    assert_eq!(dn_before, dn_after_decode);

    // But the KV cache logical length should have grown
    // 5 prefill + 9 decode = 14 tokens
    assert_eq!(model.kv_cache_len(), 14);
}

#[test]
fn test_hybrid_reset_state() {
    let model = build_test_model();

    // First run
    model.reset_state();
    let out1 = model.prefill(&[7, 8, 9]);

    // Reset and run again
    model.reset_state();
    let out2 = model.prefill(&[7, 8, 9]);

    // Should be deterministic
    assert_eq!(out1.logits.len(), out2.logits.len());
    for (i, (&a, &b)) in out1.logits.iter().zip(out2.logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit[{i}] differs after reset: {a} vs {b}"
        );
    }
}
