//! Inference throughput benchmark: measure decode tokens/sec for f32 and INT8.
//!
//! Thresholds:
//! - f32 decode: >= 20 tok/s (debug: >= 4 tok/s)
//! - INT8 decode: >= 35 tok/s (debug: >= 7 tok/s)

use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::generation::{GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::quantize_model;
use synapse_inference::weight_loading::AlignedBuffer;

fn bench_config() -> ModelConfig {
    ModelConfig {
        name: "Qwen3-InferenceBench".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 128,
            num_layers: 4,
            vocab_size: 512,
            max_sequence_length: 128,
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
            base: 10000.0,
            max_position_embeddings: 128,
            style: Default::default(),
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

fn fill_model_weights(model: &mut synapse_inference::model::CausalLM) {
    let cfg = &model.config;
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    model.embed_tokens = AlignedBuffer::from_vec(gen_weights(vocab * h, 1));
    model.final_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.w_q = AlignedBuffer::from_vec(gen_weights(q_dim * h, s + 1));
        layer.w_k = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 2));
        layer.w_v = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 3));
        layer.w_o = AlignedBuffer::from_vec(gen_weights(h * q_dim, s + 4));
        layer.ffn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.ffn_gate = AlignedBuffer::from_vec(gen_weights(inter * h, s + 5));
        layer.ffn_up = AlignedBuffer::from_vec(gen_weights(inter * h, s + 6));
        layer.ffn_down = AlignedBuffer::from_vec(gen_weights(h * inter, s + 7));
    }
}

/// Measure decode throughput for the f32 model using the generation pipeline.
#[test]
fn inference_throughput_f32_decode_20_tok_per_sec() {
    let cfg = bench_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let pipeline = GenerationPipeline::new(&model);
    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    // Warmup
    let warmup_config = GenerationConfig {
        max_new_tokens: 3,
        ..Default::default()
    };
    let _ = pipeline.generate(&prompt, warmup_config, None);

    // Benchmark
    let config = GenerationConfig {
        max_new_tokens: num_tokens,
        ..Default::default()
    };
    let start = Instant::now();
    let output = pipeline.generate(&prompt, config, None);
    let elapsed = start.elapsed();

    let tps = output.num_generated_tokens as f64 / elapsed.as_secs_f64();
    eprintln!(
        "f32 decode: {} tokens in {:.3}s = {:.1} tok/s",
        output.num_generated_tokens,
        elapsed.as_secs_f64(),
        tps
    );

    let threshold = if cfg!(debug_assertions) { 4.0 } else { 20.0 };
    assert!(
        tps >= threshold,
        "f32 decode throughput {:.1} tok/s < {:.0} tok/s threshold",
        tps,
        threshold
    );
}

/// Measure decode throughput for the INT8 quantized model.
#[test]
fn inference_throughput_int8_decode_35_tok_per_sec() {
    let cfg = bench_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let quantized = quantize_model(&model);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    // Warmup
    for _ in 0..2 {
        let _ = quantized.forward(&prompt);
    }

    // Benchmark: manually simulate decode loop
    let mut all_tokens = prompt.clone();
    let vocab = cfg.architecture.vocab_size;

    let start = Instant::now();
    for _ in 0..num_tokens {
        let output = quantized.forward(&all_tokens);
        let seq_len = output.shape[1];
        let last_logits = &output.logits[(seq_len - 1) * vocab..seq_len * vocab];
        let token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        all_tokens.push(token);
    }
    let elapsed = start.elapsed();

    let tps = num_tokens as f64 / elapsed.as_secs_f64();
    eprintln!(
        "INT8 decode: {num_tokens} tokens in {:.3}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tps
    );

    let threshold = if cfg!(debug_assertions) { 7.0 } else { 35.0 };
    assert!(
        tps >= threshold,
        "INT8 decode throughput {:.1} tok/s < {:.0} tok/s threshold",
        tps,
        threshold
    );
}
