//! E2E inference test: Build Qwen3-architecture model, generate 50 tokens greedy,
//! verify coherent output (deterministic, valid tokens, top-1 self-agreement >= 95%).

use std::collections::HashMap;

use synapse_inference::config::*;
use synapse_inference::generation::{GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

/// Qwen3-architecture config with reduced dimensions for fast testing.
/// Same architecture choices (GQA, RMSNorm, SwiGLU, RoPE) as Qwen3-0.6B.
fn qwen3_test_config() -> ModelConfig {
    ModelConfig {
        name: "Qwen3-E2E-Test".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 256,
            max_sequence_length: 128,
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
            base: 1_000_000.0,
            max_position_embeddings: 128,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

/// Deterministic pseudo-random weight generator.
fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

/// Generate fake HuggingFace-format weights matching the config.
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
        w.insert(format!("model.layers.{i}.input_layernorm.weight"), fake(vec![h], s));
        w.insert(format!("model.layers.{i}.self_attn.q_proj.weight"), fake(vec![q_dim, h], s + 1));
        w.insert(format!("model.layers.{i}.self_attn.k_proj.weight"), fake(vec![kv_dim, h], s + 2));
        w.insert(format!("model.layers.{i}.self_attn.v_proj.weight"), fake(vec![kv_dim, h], s + 3));
        w.insert(format!("model.layers.{i}.self_attn.o_proj.weight"), fake(vec![h, q_dim], s + 4));
        w.insert(format!("model.layers.{i}.self_attn.q_norm.weight"), fake(vec![cfg.attention.head_dim()], s + 5));
        w.insert(format!("model.layers.{i}.self_attn.k_norm.weight"), fake(vec![cfg.attention.head_dim()], s + 6));
        w.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h], s + 7));
        w.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h], s + 8));
        w.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h], s + 9));
        w.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter], s + 10));
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
    assert!(result.missing.is_empty(), "Missing keys: {:?}", result.missing);
    model
}

#[test]
fn inference_e2e_generate_50_tokens_greedy() {
    let cfg = qwen3_test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);
    let prompt = vec![1u32, 2, 3, 4, 5];

    // Generate 50 tokens with greedy sampling
    let config = GenerationConfig {
        max_new_tokens: 50,
        ..Default::default()
    };
    let output = pipeline.generate(&prompt, config, None);

    // Verify correct token counts
    assert_eq!(output.num_generated_tokens, 50);
    assert_eq!(output.num_prompt_tokens, 5);
    assert_eq!(output.token_ids.len(), 55); // 5 prompt + 50 generated

    // All generated tokens must be in valid vocab range
    for (i, &t) in output.token_ids.iter().enumerate() {
        assert!(
            (t as usize) < cfg.architecture.vocab_size,
            "Token {i} = {t} exceeds vocab size {}",
            cfg.architecture.vocab_size
        );
    }

    // Prompt tokens preserved
    assert_eq!(&output.token_ids[..5], &[1, 2, 3, 4, 5]);
}

#[test]
fn inference_e2e_greedy_deterministic() {
    let cfg = qwen3_test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);
    let prompt = vec![1u32, 2, 3, 4, 5];

    // Run greedy generation 5 times — all must produce identical output
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for _ in 0..5 {
        let config = GenerationConfig {
            max_new_tokens: 50,
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config, None);
        runs.push(output.token_ids.clone());
    }

    for (i, run) in runs.iter().enumerate() {
        assert_eq!(
            run, &runs[0],
            "Run {i} differs from run 0: greedy must be deterministic"
        );
    }
}

#[test]
fn inference_e2e_top1_self_agreement() {
    let cfg = qwen3_test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);

    // Test across multiple prompts
    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4, 5],
        vec![10, 20, 30],
        vec![0, 0, 0, 0],
        vec![100, 200, 50, 75, 25, 10],
    ];

    for prompt in &prompts {
        let config1 = GenerationConfig {
            max_new_tokens: 50,
            ..Default::default()
        };
        let config2 = GenerationConfig {
            max_new_tokens: 50,
            ..Default::default()
        };

        let out1 = pipeline.generate(prompt, config1, None);
        let out2 = pipeline.generate(prompt, config2, None);

        let total = out1.token_ids.len();
        let agree = out1
            .token_ids
            .iter()
            .zip(out2.token_ids.iter())
            .filter(|(a, b)| a == b)
            .count();
        let agreement = agree as f64 / total as f64;

        assert!(
            agreement >= 0.95,
            "Top-1 agreement {:.1}% < 95% for prompt {:?}",
            agreement * 100.0,
            prompt
        );
    }
}

#[test]
fn inference_e2e_output_logits_finite() {
    let cfg = qwen3_test_config();
    let model = build_model(&cfg);
    let prompt = vec![1u32, 2, 3, 4, 5];

    let output = model.forward(&prompt);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Forward pass produced non-finite logits"
    );
}

#[test]
fn inference_e2e_timing_positive() {
    let cfg = qwen3_test_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);
    let prompt = vec![1u32, 2, 3];

    let config = GenerationConfig {
        max_new_tokens: 10,
        ..Default::default()
    };
    let output = pipeline.generate(&prompt, config, None);

    assert!(output.elapsed.as_nanos() > 0, "Elapsed time should be positive");
    assert!(output.tokens_per_sec > 0.0, "Tokens/sec should be positive");
}
