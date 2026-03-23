//! Benchmark any model via config.
//!
//! Usage:
//!   cargo run --example model_benchmark --release
//!   cargo run --example model_benchmark --release -- --config configs/llama3.2_1b.json
//!
//! Runs prefill and decode benchmarks, reporting throughput in tok/s.
//! Uses random weights (replace with real checkpoint loading for production).

use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::generation::{GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::{f32_model_memory_bytes, quantize_model};

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

    model.embed_tokens = gen_weights(vocab * h, 1);
    model.final_norm_weight = vec![1.0f32; h];

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = vec![1.0f32; h];
        layer.w_q = gen_weights(q_dim * h, s + 1);
        layer.w_k = gen_weights(kv_dim * h, s + 2);
        layer.w_v = gen_weights(kv_dim * h, s + 3);
        layer.w_o = gen_weights(h * q_dim, s + 4);
        layer.ffn_norm_weight = vec![1.0f32; h];
        layer.ffn_gate = gen_weights(inter * h, s + 5);
        layer.ffn_up = gen_weights(inter * h, s + 6);
        layer.ffn_down = gen_weights(h * inter, s + 7);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // ── Parse config ─────────────────────────────────────────────────
    let config_json = if args.len() >= 3 && args[1] == "--config" {
        std::fs::read_to_string(&args[2]).expect("Failed to read config file")
    } else {
        // Default: use a small Qwen3 config for demo
        include_str!("../configs/qwen3_0.6b.json").to_string()
    };

    let mut cfg = ModelConfig::from_json(&config_json).expect("Failed to parse config");

    // Scale down for demo if using the full model (too slow for CPU benchmarking)
    if cfg.architecture.hidden_size > 256 {
        println!("Scaling down {} for CPU benchmarking...", cfg.name);
        cfg.architecture.hidden_size = 128;
        cfg.architecture.num_layers = 4;
        cfg.architecture.vocab_size = 512;
        cfg.architecture.max_sequence_length = 128;
        cfg.attention = AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
        };
        cfg.ffn = FFNConfig::SwiGLU {
            intermediate_size: 256,
        };
        cfg.position = PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 128,
        };
    }

    println!("═══════════════════════════════════════════════");
    println!("  Synapse Model Benchmark");
    println!("═══════════════════════════════════════════════");
    println!("  Model:      {}", cfg.name);
    println!("  Hidden:     {}", cfg.architecture.hidden_size);
    println!("  Layers:     {}", cfg.architecture.num_layers);
    println!("  Vocab:      {}", cfg.architecture.vocab_size);
    println!("  Heads:      {}/{} (Q/KV)",
        cfg.attention.num_heads(), cfg.attention.num_kv_heads());
    println!("  Head dim:   {}", cfg.attention.head_dim());
    println!("  FFN inter:  {}", cfg.ffn.intermediate_size());
    println!("═══════════════════════════════════════════════");

    // ── Build model ──────────────────────────────────────────────────
    print!("Building model...");
    let start = Instant::now();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);
    let build_time = start.elapsed();
    println!(" {:.3}s ({} params)", build_time.as_secs_f64(), model.param_count());

    let f32_mem = f32_model_memory_bytes(&model);
    println!("  f32 memory: {:.2} MB", f32_mem as f64 / (1024.0 * 1024.0));

    // ── Quantize ─────────────────────────────────────────────────────
    print!("Quantizing to INT8...");
    let start = Instant::now();
    let quantized = quantize_model(&model);
    let quant_time = start.elapsed();
    let int8_mem = quantized.memory_bytes();
    println!(" {:.3}s", quant_time.as_secs_f64());
    println!("  INT8 memory: {:.2} MB ({:.1}% of f32)",
        int8_mem as f64 / (1024.0 * 1024.0),
        int8_mem as f64 / f32_mem as f64 * 100.0);

    // ── Prefill benchmark ────────────────────────────────────────────
    let pipeline = GenerationPipeline::new(&model);
    let seq_lengths = [8, 16, 32, 64];

    println!("\n── Prefill Benchmark (f32) ──────────────────");
    for &seq_len in &seq_lengths {
        let tokens: Vec<u32> = (0..seq_len)
            .map(|i| (i % cfg.architecture.vocab_size) as u32)
            .collect();

        // Warmup
        let _ = pipeline.prefill(&tokens);

        let iters = 5;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = pipeline.prefill(&tokens);
        }
        let elapsed = start.elapsed();
        let tps = (iters * seq_len) as f64 / elapsed.as_secs_f64();
        println!("  seq_len={seq_len:>3}: {tps:>8.0} tok/s");
    }

    // ── Decode benchmark (f32) ───────────────────────────────────────
    println!("\n── Decode Benchmark (f32) ───────────────────");
    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    let config = GenerationConfig {
        max_new_tokens: num_tokens,
        ..Default::default()
    };
    let start = Instant::now();
    let output = pipeline.generate(&prompt, config);
    let elapsed = start.elapsed();
    let tps = output.num_generated_tokens as f64 / elapsed.as_secs_f64();
    println!("  {num_tokens} tokens: {tps:.1} tok/s");

    // ── Decode benchmark (INT8) ──────────────────────────────────────
    println!("\n── Decode Benchmark (INT8) ──────────────────");
    let vocab = cfg.architecture.vocab_size;
    let mut all_tokens = prompt.clone();

    let start = Instant::now();
    for _ in 0..num_tokens {
        let out = quantized.forward(&all_tokens);
        let seq_len = out.shape[1];
        let last_logits = &out.logits[(seq_len - 1) * vocab..seq_len * vocab];
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
    println!("  {num_tokens} tokens: {tps:.1} tok/s");

    println!("\n═══════════════════════════════════════════════");
    println!("  Benchmark complete.");
    println!("═══════════════════════════════════════════════");
}
