//! KV-cache correctness test: verify that incremental (decode-step) forward passes
//! produce bit-exact results compared to full-context forward passes.
//!
//! Since the current engine recomputes the full context on each decode step
//! (no KV-cache optimization yet), this test verifies the fundamental invariant:
//! the last-position logits from a full forward on [t0..tN] must exactly match
//! the last-position logits from decode_step on the same tokens.

use std::collections::HashMap;

use synapse_inference::config::*;
use synapse_inference::generation::{GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

fn test_config() -> ModelConfig {
    ModelConfig {
        name: "KVCacheTest".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 256,
            max_sequence_length: 64,
            tie_word_embeddings: true,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 128,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 64,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();
    let nl = cfg.architecture.num_layers;

    let fake = |shape: Vec<usize>, seed: u32| -> RawTensor {
        let n: usize = shape.iter().product();
        RawTensor {
            data: AlignedBuffer::from_vec(gen_weights(n, seed)),
            shape,
        }
    };

    let mut w = HashMap::new();
    w.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h], 1));
    for i in 0..nl {
        let s = (i as u32 + 1) * 100;
        w.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            fake(vec![h], s),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            fake(vec![q_dim, h], s + 1),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            fake(vec![kv_dim, h], s + 2),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            fake(vec![kv_dim, h], s + 3),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            fake(vec![h, q_dim], s + 4),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.q_norm.weight"),
            fake(vec![cfg.attention.head_dim()], s + 5),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.k_norm.weight"),
            fake(vec![cfg.attention.head_dim()], s + 6),
        );
        w.insert(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            fake(vec![h], s + 7),
        );
        w.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            fake(vec![inter, h], s + 8),
        );
        w.insert(
            format!("model.layers.{i}.mlp.up_proj.weight"),
            fake(vec![inter, h], s + 9),
        );
        w.insert(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            fake(vec![h, inter], s + 10),
        );
    }
    w.insert("model.norm.weight".into(), fake(vec![h], 9999));
    w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
    w
}

fn build_model(cfg: &ModelConfig) -> synapse_inference::model::CausalLM {
    let mut model = ModelBuilder::from_config(cfg);
    let weights = generate_fake_hf_weights(cfg);
    let mapper = WeightMapper::qwen3();
    let result = model.load_weights(weights, &mapper).unwrap();
    assert!(
        result.missing.is_empty(),
        "Missing keys: {:?}",
        result.missing
    );
    model
}

/// Verify that prefill logits match full forward logits (bit-exact).
#[test]
fn kvcache_prefill_matches_full_forward() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);
    let vocab = cfg.architecture.vocab_size;

    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4],
        vec![10, 20, 30],
        vec![5, 5, 5, 5, 5, 5],
        vec![0, 100, 200, 50],
    ];

    for prompt in &prompts {
        let prefill_logits = pipeline.prefill(prompt);

        let full_output = model.forward(prompt);
        let seq_len = full_output.shape[1];
        let full_last_logits = &full_output.logits[(seq_len - 1) * vocab..seq_len * vocab];

        assert_eq!(prefill_logits.len(), full_last_logits.len());

        for (i, (&a, &b)) in prefill_logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() == 0.0,
                "Logit {i} not bit-exact: prefill={a}, full={b} (prompt={:?})",
                prompt
            );
        }
    }
}

/// Verify that decode_step produces bit-exact last-position logits
/// compared to a full forward pass on the same tokens.
#[test]
fn kvcache_decode_step_matches_full_forward() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);
    let vocab = cfg.architecture.vocab_size;

    // Simulate an autoregressive sequence: start with prompt, add tokens one at a time
    let initial_prompt = vec![1u32, 2, 3, 4, 5];

    // Generate some tokens to extend the sequence
    for extra in 0..5 {
        let mut all_tokens = initial_prompt.clone();
        for _ in 0..=extra {
            all_tokens.push(42 + extra);
        }

        // decode_step on the full sequence
        let decode_logits = pipeline.decode_step(&all_tokens);

        // Full forward on the same sequence
        let full_output = model.forward(&all_tokens);
        let seq_len = full_output.shape[1];
        let full_last_logits = &full_output.logits[(seq_len - 1) * vocab..seq_len * vocab];

        assert_eq!(decode_logits.len(), full_last_logits.len());

        for (i, (&a, &b)) in decode_logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() == 0.0,
                "Decode step logit {i} not bit-exact: decode={a}, full={b} (seq_len={})",
                all_tokens.len()
            );
        }
    }
}

/// Verify that extending a sequence preserves earlier-position logits.
/// Forward on [1,2,3] should give the same logits at positions 0,1,2
/// as forward on [1,2,3,4] at positions 0,1,2.
#[test]
fn kvcache_prefix_logits_consistent() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let vocab = cfg.architecture.vocab_size;

    let short_seq = vec![1u32, 2, 3];
    let long_seq = vec![1u32, 2, 3, 4];

    let short_out = model.forward(&short_seq);
    let long_out = model.forward(&long_seq);

    // Causal attention means positions 0,1,2 in long_seq see same context as short_seq.
    // However, the pure forward (no KV cache) recomputes everything, so the logits
    // at positions 0,1,2 should be identical.
    for pos in 0..short_seq.len() {
        let short_logits = &short_out.logits[pos * vocab..(pos + 1) * vocab];
        let long_logits = &long_out.logits[pos * vocab..(pos + 1) * vocab];

        for (i, (&a, &b)) in short_logits.iter().zip(long_logits.iter()).enumerate() {
            assert!(
                (a - b).abs() == 0.0,
                "Position {} logit {i} differs: short={a}, long={b}",
                pos
            );
        }
    }
}

/// Verify greedy generation produces same tokens whether done in one call
/// or by manually calling decode_step in a loop.
#[test]
fn kvcache_manual_decode_matches_pipeline() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_new_tokens = 10;

    // Pipeline generation (greedy)
    let config = GenerationConfig {
        max_new_tokens: num_new_tokens,
        ..Default::default()
    };
    let pipeline_output = pipeline.generate(&prompt, config, None);
    let pipeline_generated = &pipeline_output.token_ids[prompt.len()..];

    // Manual decode loop (greedy)
    let mut all_tokens = prompt.clone();
    let mut manual_generated = Vec::new();

    // First: prefill to get first token
    let logits = pipeline.prefill(&all_tokens);
    let first_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    all_tokens.push(first_token);
    manual_generated.push(first_token);

    // Then: decode loop
    for _ in 1..num_new_tokens {
        let logits = pipeline.decode_step(&all_tokens);
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        all_tokens.push(token);
        manual_generated.push(token);
    }

    assert_eq!(
        pipeline_generated,
        &manual_generated[..],
        "Manual decode loop must produce same tokens as pipeline"
    );
}
