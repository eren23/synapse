//! Prefill throughput benchmark: measure tokens/sec during the prefill phase
//! (single forward pass over the full prompt).
//!
//! Threshold: >= 500 tok/s (Qwen3 architecture, f32).
//! Debug mode: >= 100 tok/s (~5x lower).

use std::collections::HashMap;
use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::generation::GenerationPipeline;
use synapse_inference::model::ModelBuilder;
use synapse_inference::weight_loading::{RawTensor, WeightMapper};

/// Qwen3-architecture benchmark config with reduced dimensions.
/// Preserves the same architecture (GQA, RMSNorm, SwiGLU, RoPE) but uses
/// smaller dimensions so the benchmark completes quickly while testing
/// the same code paths as the full model.
fn bench_config() -> ModelConfig {
    ModelConfig {
        name: "Qwen3-PrefillBench".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 128,
            num_layers: 4,
            vocab_size: 512,
            max_sequence_length: 256,
            tie_word_embeddings: true,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 256,
        },
        position: PositionConfig::RoPE {
            base: 1_000_000.0,
            max_position_embeddings: 256,
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
            data: gen_weights(n, seed),
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
        w.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h], s + 5));
        w.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h], s + 6));
        w.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h], s + 7));
        w.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter], s + 8));
    }
    w.insert("model.norm.weight".into(), fake(vec![h], 9999));
    w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
    w
}

fn build_model(cfg: &ModelConfig) -> synapse_inference::model::CausalLM {
    let mut model = ModelBuilder::from_config(cfg);
    let weights = generate_fake_hf_weights(cfg);
    let mapper = WeightMapper::qwen3();
    let result = model.load_weights(weights, &mapper);
    assert!(result.missing.is_empty(), "Missing keys: {:?}", result.missing);
    model
}

#[test]
fn prefill_throughput_500_tok_per_sec() {
    let cfg = bench_config();
    let model = build_model(&cfg);
    let pipeline = GenerationPipeline::new(&model);

    let seq_lengths = [16, 32, 64, 128];
    let warmup_iters = 2;
    let bench_iters = 5;

    let mut total_tokens = 0usize;
    let mut total_elapsed = std::time::Duration::ZERO;

    for &seq_len in &seq_lengths {
        let tokens: Vec<u32> = (0..seq_len).map(|i| (i % cfg.architecture.vocab_size) as u32).collect();

        // Warmup
        for _ in 0..warmup_iters {
            let _ = pipeline.prefill(&tokens);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..bench_iters {
            let _ = pipeline.prefill(&tokens);
        }
        let elapsed = start.elapsed();

        let toks = bench_iters * seq_len;
        let tps = toks as f64 / elapsed.as_secs_f64();
        eprintln!(
            "  seq_len={seq_len}: {toks} tokens in {:.3}s = {:.0} tok/s",
            elapsed.as_secs_f64(),
            tps
        );

        total_tokens += toks;
        total_elapsed += elapsed;
    }

    let overall_tps = total_tokens as f64 / total_elapsed.as_secs_f64();
    eprintln!(
        "Prefill throughput: {total_tokens} tokens in {:.3}s = {:.0} tok/s",
        total_elapsed.as_secs_f64(),
        overall_tps
    );

    let threshold = if cfg!(debug_assertions) { 100.0 } else { 500.0 };
    assert!(
        overall_tps >= threshold,
        "Prefill throughput {:.0} tok/s < {:.0} tok/s threshold",
        overall_tps,
        threshold
    );
}
