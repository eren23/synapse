//! Interactive chat with either a real Qwen3 checkpoint or a tiny demo model.
//!
//! Usage:
//!   cargo run --example qwen3_chat --release -- --model-dir /path/to/Qwen3-0.6B
//!   cargo run --example qwen3_chat --release -- --demo

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use synapse_inference::config::*;
use synapse_inference::engine::InferenceEngine;
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
            // MANUAL FIX: RawTensor.data changed from Vec<f32> to AlignedBuffer in swarm output
            data: synapse_inference::weight_loading::AlignedBuffer::from_vec(gen_weights(n, seed)),
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

fn demo_engine() -> InferenceEngine {
    let config_json = include_str!("../configs/qwen3_0.6b.json");
    let mut cfg = ModelConfig::from_json(config_json).expect("Failed to parse demo config");

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

    let mut model = ModelBuilder::from_config(&cfg);
    let weights = generate_fake_hf_weights(&cfg);
    let mapper = WeightMapper::qwen3();
    let result = model.load_weights(weights, &mapper).expect("Demo weights should load");
    assert!(result.missing.is_empty(), "Missing keys: {:?}", result.missing);
    assert!(result.unexpected.is_empty(), "Unexpected keys: {:?}", result.unexpected);

    InferenceEngine {
        model,
        config: cfg,
        tokenizer: None,
    }
}

enum Mode {
    Demo,
    Chat(PathBuf),
    Verify(PathBuf),
}

fn parse_args() -> Result<Mode, String> {
    let mut args = std::env::args().skip(1);
    let mut model_dir = None;
    let mut verify = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => return Ok(Mode::Demo),
            "--verify" => verify = true,
            "--model-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--model-dir requires a path".to_string())?;
                model_dir = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("Usage:");
                println!("  cargo run --example qwen3_chat --release -- --model-dir /path/to/Qwen3-0.6B");
                println!("  cargo run --example qwen3_chat --release -- --model-dir /path --verify");
                println!("  cargo run --example qwen3_chat --release -- --demo");
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    match (model_dir, verify) {
        (Some(dir), true) => Ok(Mode::Verify(dir)),
        (Some(dir), false) => Ok(Mode::Chat(dir)),
        (None, _) => Ok(Mode::Demo),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = parse_args().map_err(io::Error::other)?;

    match mode {
        Mode::Verify(dir) => run_verify(dir)?,
        Mode::Chat(dir) => run_pretrained_chat(dir)?,
        Mode::Demo => run_demo_chat(),
    }

    Ok(())
}

fn run_verify(model_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    print!("Loading checkpoint from {}...", model_dir.display());
    io::stdout().flush()?;
    let engine = InferenceEngine::from_pretrained(&model_dir)?;
    println!(" done ({} params)", engine.param_count());

    let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let tokens = engine.encode(prompt)?;
    println!("Prompt: {prompt:?}");
    println!("Token IDs ({} tokens): {:?}", tokens.len(), tokens);

    let h = engine.config.architecture.hidden_size;
    let vocab = engine.config.architecture.vocab_size;

    // Step 1: Embedding - compare first 8 values at positions 0 and last
    let id0 = tokens[0] as usize;
    println!("\nEmbed[0,:8]:  {:?}", &engine.model.embed_tokens[id0 * h..id0 * h + 8]);
    let id_last = *tokens.last().unwrap() as usize;
    println!("Embed[-1,:8]: {:?}", &engine.model.embed_tokens[id_last * h..id_last * h + 8]);

    // Full forward for logits
    let output = engine.model.forward(&tokens);
    let seq_len = output.shape[1];
    let last_logits = &output.logits[(seq_len - 1) * vocab..seq_len * vocab];

    let mut indexed: Vec<(usize, f32)> = last_logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop-10 next-token logits (compare against HF transformers):");
    println!("{:<10} {:<15} {}", "Token ID", "Logit", "Decoded");
    println!("{}", "-".repeat(45));
    let tokenizer = engine.tokenizer().unwrap();
    for &(id, logit) in indexed.iter().take(10) {
        let decoded = tokenizer.decode(&[id as u32]).unwrap_or_default();
        println!("{:<10} {:<15.6} {:?}", id, logit, decoded);
    }

    // Print specific token logits for direct comparison
    println!("\nSpecific token logits for comparison:");
    for tid in [151667u32, 151644, 151668, 2784, 151645, 47611] {
        let logit = last_logits[tid as usize];
        let decoded = tokenizer.decode(&[tid]).unwrap_or_default();
        println!("  {tid:<10} {logit:<15.6} {decoded:?}");
    }

    Ok(())
}

fn run_pretrained_chat(model_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    print!("Loading checkpoint from {}...", model_dir.display());
    io::stdout().flush()?;
    let engine = InferenceEngine::from_pretrained(&model_dir)?;
    println!(" done ({} params)", engine.param_count());
    println!("Type 'quit' to exit.");

    let tokenizer = engine.tokenizer().expect("pretrained engine has tokenizer").clone();
    let stop_sequences = tokenizer.encode("<|im_end|>")?;
    let eos_token_id = tokenizer.eos_token_id();
    let stdin = io::stdin();
    let mut line_reader = stdin.lock().lines();

    loop {
        print!("\n> ");
        io::stdout().flush()?;

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

        let prompt = format!(
            "<|im_start|>user\n{line}<|im_end|>\n<|im_start|>assistant\n"
        );
        let prompt_tokens = engine.encode(&prompt)?;
        let stream_tokenizer = tokenizer.clone();
        let pipeline = GenerationPipeline::new(&engine.model);

        let max_new_tokens = 256;
        let max_seq = prompt_tokens.len() + max_new_tokens;
        let mut cache = engine.create_kv_cache(max_seq)?;

        // Qwen3 generates <think>...</think> before the answer.
        // Show "thinking..." while hidden, then stream the actual answer.
        let think_id = tokenizer.encode("<think>").ok().and_then(|v| v.first().copied());
        let end_think_id = tokenizer.encode("</think>").ok().and_then(|v| v.first().copied());
        let in_think = std::cell::Cell::new(false);
        let think_shown = std::cell::Cell::new(false);

        let config = GenerationConfig {
            max_new_tokens,
            eos_token_id,
            stop_sequences: vec![stop_sequences.clone()],
            combined: Some(CombinedSampler {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
            }),
            seed: Some(42),
            on_token: Some(Box::new(move |token| {
                if Some(token) == think_id {
                    in_think.set(true);
                    if !think_shown.get() {
                        print!("(thinking...) ");
                        let _ = io::stdout().flush();
                        think_shown.set(true);
                    }
                    return;
                }
                if Some(token) == end_think_id {
                    in_think.set(false);
                    print!("\r              \r"); // clear "thinking..."
                    let _ = io::stdout().flush();
                    return;
                }
                if in_think.get() {
                    return;
                }
                if let Ok(piece) = stream_tokenizer.decode_token_piece(token) {
                    print!("{piece}");
                    let _ = io::stdout().flush();
                }
            })),
            ..Default::default()
        };

        let output = pipeline.generate(&prompt_tokens, config, Some(&mut cache));
        println!();
        println!(
            "Generated {} tokens in {:.3}s ({:.1} tok/s)",
            output.num_generated_tokens,
            output.elapsed.as_secs_f64(),
            output.tokens_per_sec
        );
    }

    Ok(())
}

fn run_demo_chat() {
    let engine = demo_engine();
    let pipeline = GenerationPipeline::new(&engine.model);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Synapse Qwen3 Chat (demo with random wts)  ║");
    println!("╠══════════════════════════════════════════════╣");
    println!(
        "║  Architecture: {} layers, h={}, vocab={}",
        engine.config.architecture.num_layers,
        engine.config.architecture.hidden_size,
        engine.config.architecture.vocab_size,
    );
    println!("║  Attention: GQA, FFN: SwiGLU, Norm: RMSNorm ║");
    println!("║  Use --model-dir for a real checkpoint      ║");
    println!("║  Type 'quit' to exit                        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

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

        let prompt_tokens: Vec<u32> = line
            .bytes()
            .map(|b| (b as u32) % engine.config.architecture.vocab_size as u32)
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
                print!(".");
                io::stdout().flush().unwrap();
            })),
            ..Default::default()
        };

        let output = pipeline.generate(&prompt_tokens, config, None);

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
