//! Interactive chat with a Qwen3-0.6B-architecture model.
//!
//! Usage:
//!   cargo run --example qwen3_chat --release
//!
//! This example builds a Qwen3-architecture model with random weights
//! (since no real checkpoint is included) and demonstrates the full
//! generation pipeline: prompt → prefill → decode → output.
//!
//! With real weights, replace the fake weight loading with:
//!   `load_safetensors(Path::new("path/to/model.safetensors"))`

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use synapse_inference::config::*;
use synapse_inference::generation::{CombinedSampler, GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::weight_loading::{RawTensor, WeightMapper};

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

fn main() {
    // ── Load config ──────────────────────────────────────────────────
    let config_json = include_str!("../configs/qwen3_0.6b.json");
    let mut cfg = ModelConfig::from_json(config_json).expect("Failed to parse Qwen3 config");

    // Use a smaller config for demo (full 0.6B is too slow without optimized backend)
    cfg.architecture.hidden_size = 64;
    cfg.architecture.num_layers = 4;
    cfg.architecture.vocab_size = 256;
    cfg.architecture.max_sequence_length = 128;
    cfg.attention = AttentionConfig::GQA {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
    };
    cfg.ffn = FFNConfig::SwiGLU {
        intermediate_size: 128,
    };
    cfg.position = PositionConfig::RoPE {
        base: 1_000_000.0,
        max_position_embeddings: 128,
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Synapse Qwen3 Chat (demo with random wts)  ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Architecture: {} layers, h={}, vocab={}",
        cfg.architecture.num_layers,
        cfg.architecture.hidden_size,
        cfg.architecture.vocab_size,
    );
    println!("║  Attention: GQA, FFN: SwiGLU, Norm: RMSNorm ║");
    println!("║  Type 'quit' to exit                        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    // ── Build model with fake weights ────────────────────────────────
    print!("Loading model...");
    io::stdout().flush().unwrap();

    let mut model = ModelBuilder::from_config(&cfg);
    let weights = generate_fake_hf_weights(&cfg);
    let mapper = WeightMapper::qwen3();
    let result = model.load_weights(weights, &mapper);
    assert!(result.missing.is_empty(), "Missing keys: {:?}", result.missing);
    println!(" done ({} params)", model.param_count());

    let pipeline = GenerationPipeline::new(&model);

    // ── Chat loop ────────────────────────────────────────────────────
    let stdin = io::stdin();
    let mut line_reader = stdin.lock().lines();

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let line = match line_reader.next() {
            Some(Ok(l)) => l,
            _ => break,
        };

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            println!("Goodbye!");
            break;
        }

        // Simple tokenization: map each byte to a token ID
        let prompt_tokens: Vec<u32> = line
            .bytes()
            .map(|b| (b as u32) % cfg.architecture.vocab_size as u32)
            .collect();

        let config = GenerationConfig {
            max_new_tokens: 32,
            combined: Some(CombinedSampler {
                temperature: 0.8,
                top_k: 10,
                top_p: 0.9,
                repetition_penalty: 1.2,
            }),
            seed: Some(42),
            on_token: Some(Box::new(|_token| {
                // Streaming: print a dot for each generated token
                print!(".");
                io::stdout().flush().unwrap();
            })),
            ..Default::default()
        };

        let output = pipeline.generate(&prompt_tokens, config);

        println!();
        println!(
            "Generated {} tokens in {:.3}s ({:.1} tok/s)",
            output.num_generated_tokens,
            output.elapsed.as_secs_f64(),
            output.tokens_per_sec
        );
        println!(
            "Token IDs: {:?}",
            &output.token_ids[output.num_prompt_tokens..]
        );
    }
}
