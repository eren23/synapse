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
use synapse_inference::kv_cache::KVCache;
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::{f32_model_memory_bytes, quantize_model};
use synapse_inference::weight_loading::AlignedBuffer;

fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

fn aligned_weights(len: usize, seed: u32) -> AlignedBuffer {
    AlignedBuffer::from_vec(gen_weights(len, seed))
}

fn ones_aligned(len: usize) -> AlignedBuffer {
    AlignedBuffer::from_vec(vec![1.0f32; len])
}

fn fill_model_weights(model: &mut synapse_inference::model::CausalLM) {
    let cfg = &model.config;
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    model.embed_tokens = aligned_weights(vocab * h, 1);
    model.final_norm_weight = ones_aligned(h);
    if model.lm_head_weight.is_some() {
        model.lm_head_weight = Some(aligned_weights(vocab * h, 2));
    }

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = ones_aligned(h);
        layer.w_q = aligned_weights(q_dim * h, s + 1);
        layer.w_k = aligned_weights(kv_dim * h, s + 2);
        layer.w_v = aligned_weights(kv_dim * h, s + 3);
        layer.w_o = aligned_weights(h * q_dim, s + 4);
        layer.ffn_norm_weight = ones_aligned(h);
        layer.ffn_gate = aligned_weights(inter * h, s + 5);
        layer.ffn_up = aligned_weights(inter * h, s + 6);
        layer.ffn_down = aligned_weights(h * inter, s + 7);
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

    let full_scale = args.iter().any(|a| a == "--full-scale");

    // Scale down for demo if using the full model (too slow for CPU benchmarking)
    if !full_scale && cfg.architecture.hidden_size > 256 {
        println!("Scaling down {} for CPU benchmarking...", cfg.name);
        cfg.architecture.hidden_size = 128;
        cfg.architecture.num_layers = 4;
        cfg.architecture.vocab_size = 512;
        cfg.architecture.max_sequence_length = 128;
        cfg.attention = match &cfg.attention {
            AttentionConfig::MHA { .. } => AttentionConfig::MHA {
                num_heads: 4,
                head_dim: 32,
            },
            AttentionConfig::MQA { .. } => AttentionConfig::MQA {
                num_heads: 4,
                head_dim: 32,
            },
            AttentionConfig::GQA { .. } => AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
            },
            AttentionConfig::SlidingWindow { window_size, .. } => {
                AttentionConfig::SlidingWindow {
                    num_heads: 4,
                    num_kv_heads: 2,
                    head_dim: 32,
                    window_size: (*window_size).min(128),
                }
            }
            AttentionConfig::Bidirectional { .. } => AttentionConfig::Bidirectional {
                num_heads: 4,
                head_dim: 32,
            },
        };
        cfg.ffn = match &cfg.ffn {
            FFNConfig::GeGLU { .. } => FFNConfig::GeGLU {
                intermediate_size: 256,
            },
            _ => FFNConfig::SwiGLU {
                intermediate_size: 256,
            },
        };
        cfg.position = PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 128,
            style: Default::default(),
            scaling: Default::default(),
        };
    }

    println!("═══════════════════════════════════════════════");
    println!("  Synapse Model Benchmark");
    println!("═══════════════════════════════════════════════");
    println!("  Model:      {}", cfg.name);
    println!("  Hidden:     {}", cfg.architecture.hidden_size);
    println!("  Layers:     {}", cfg.architecture.num_layers);
    println!("  Vocab:      {}", cfg.architecture.vocab_size);
    println!(
        "  Heads:      {}/{} (Q/KV)",
        cfg.attention.num_heads(),
        cfg.attention.num_kv_heads()
    );
    println!("  Head dim:   {}", cfg.attention.head_dim());
    println!("  FFN inter:  {}", cfg.ffn.intermediate_size());
    println!("═══════════════════════════════════════════════");

    // ── Build model ──────────────────────────────────────────────────
    print!("Building model...");
    let start = Instant::now();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);
    let build_time = start.elapsed();
    println!(
        " {:.3}s ({} params)",
        build_time.as_secs_f64(),
        model.param_count()
    );

    let f32_mem = f32_model_memory_bytes(&model);
    println!("  f32 memory: {:.2} MB", f32_mem as f64 / (1024.0 * 1024.0));

    // ── Quantize ─────────────────────────────────────────────────────
    print!("Quantizing to INT8...");
    let start = Instant::now();
    let quantized = quantize_model(&model);
    let quant_time = start.elapsed();
    let int8_mem = quantized.memory_bytes();
    println!(" {:.3}s", quant_time.as_secs_f64());
    println!(
        "  INT8 memory: {:.2} MB ({:.1}% of f32)",
        int8_mem as f64 / (1024.0 * 1024.0),
        int8_mem as f64 / f32_mem as f64 * 100.0
    );

    // ── SIMD Prefill benchmark ─────────────────────────────────────────
    let pipeline = GenerationPipeline::new(&model);
    let seq_lengths = [8, 16, 32, 64, 128];

    println!("\n── SIMD Prefill Benchmark (f32) ─────────────");
    let mut total_prefill_tokens = 0usize;
    let mut total_prefill_time = std::time::Duration::ZERO;
    for &seq_len in &seq_lengths {
        if seq_len > cfg.architecture.max_sequence_length {
            continue;
        }
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
        total_prefill_tokens += iters * seq_len;
        total_prefill_time += elapsed;
    }
    let overall_prefill_tps = total_prefill_tokens as f64 / total_prefill_time.as_secs_f64();
    println!("  ────────────────────────────");
    println!("  overall:      {overall_prefill_tps:>8.0} tok/s");

    // ── Decode benchmark (f32, no cache) ────────────────────────────
    println!("\n── Decode Benchmark (f32, no cache) ─────────");
    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    let config = GenerationConfig {
        max_new_tokens: num_tokens,
        ..Default::default()
    };
    let start = Instant::now();
    let output = pipeline.generate(&prompt, config, None);
    let elapsed = start.elapsed();
    let tps = output.num_generated_tokens as f64 / elapsed.as_secs_f64();
    println!("  {num_tokens} tokens: {tps:.1} tok/s (full recompute)");

    // ── KV-cache decode benchmark (f32) ─────────────────────────────
    println!("\n── KV-Cache Decode Benchmark (f32) ──────────");
    let cache_num_tokens = 20;
    let cache = KVCache::new(
        cfg.architecture.num_layers,
        prompt.len() + cache_num_tokens,
        cfg.attention.num_kv_heads(),
        cfg.attention.head_dim(),
    )
    .expect("Failed to create KV-cache");
    let kv_mem = cache.expected_allocation_bytes();
    let mut state = synapse_inference::model::ModelState::KvCache(cache);

    let config = GenerationConfig {
        max_new_tokens: cache_num_tokens,
        ..Default::default()
    };
    let start = Instant::now();
    let output = pipeline.generate(&prompt, config, Some(&mut state));
    let elapsed = start.elapsed();
    let cached_tps = output.num_generated_tokens as f64 / elapsed.as_secs_f64();
    let speedup = cached_tps / tps;
    println!(
        "  {cache_num_tokens} tokens: {cached_tps:.1} tok/s (cached, {speedup:.1}x vs recompute)"
    );
    println!(
        "  KV-cache memory: {:.2} MB",
        kv_mem as f64 / (1024.0 * 1024.0)
    );

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

    // ── Memory summary ──────────────────────────────────────────────
    println!("\n── Memory Summary ──────────────────────────");
    let f32_mb = f32_mem as f64 / (1024.0 * 1024.0);
    let int8_mb = int8_mem as f64 / (1024.0 * 1024.0);
    let kv_mb = kv_mem as f64 / (1024.0 * 1024.0);
    let total_mb = f32_mb + kv_mb;
    println!("  f32 weights:  {f32_mb:.2} MB");
    println!(
        "  INT8 weights: {int8_mb:.2} MB ({:.1}% of f32)",
        int8_mem as f64 / f32_mem as f64 * 100.0
    );
    println!(
        "  KV-cache:     {kv_mb:.2} MB ({}L, {}ctx)",
        cfg.architecture.num_layers,
        prompt.len() + cache_num_tokens
    );
    println!("  Total (f32+KV): {total_mb:.2} MB");

    // ── Phase 4 Summary ─────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════");
    println!("  Phase 4 Metrics Summary");
    println!("═══════════════════════════════════════════════");
    println!(
        "  {:.<30} {:>8.0} tok/s",
        "SIMD prefill (avg)", overall_prefill_tps
    );
    println!("  {:.<30} {:>8.1} tok/s", "KV-cache decode", cached_tps);
    println!("  {:.<30} {:>8.1}x", "KV-cache vs recompute", speedup);
    println!("  {:.<30} {:>8.2} MB", "f32 model memory", f32_mb);
    println!("  {:.<30} {:>8.2} MB", "INT8 model memory", int8_mb);
    println!("  {:.<30} {:>8.2} MB", "KV-cache memory", kv_mb);
    println!(
        "  {:.<30} {:>8.1}%",
        "INT8 compression ratio",
        int8_mem as f64 / f32_mem as f64 * 100.0
    );
    println!("───────────────────────────────────────────────");
    println!("  Phase 4 targets (Qwen3-0.6B full scale):");
    println!("    SIMD decode  >= 5 tok/s    (from 0.3)");
    println!("    SIMD prefill >= 50 tok/s   (pp128, from 5)");
    println!("    Metal decode >= 30 tok/s   (if Metal enabled)");
    println!("    llama.cpp gap <= 5x        (from ~270x)");
    println!("═══════════════════════════════════════════════");
}
