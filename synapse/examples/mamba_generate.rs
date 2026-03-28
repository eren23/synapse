//! Text completion with a Mamba checkpoint.
//!
//! Usage:
//!   cargo run --example mamba_generate --release -- --model-dir models/mamba-130m --prompt "The capital of France is"
//!   cargo run --example mamba_generate --release -- --model-dir models/mamba-130m --prompt "Once upon a time" --max-tokens 100
//!   cargo run --example mamba_generate --release -- --model-dir models/mamba-130m --interactive

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use synapse_inference::engine::InferenceEngine;
use synapse_inference::generation::{GenerationConfig, TemperatureSampler};
use synapse_inference::model::{Model, ModelState};
use synapse_inference::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut prompt = String::from("The meaning of life is");
    let mut max_tokens: usize = 50;
    let mut interactive = false;
    let mut temperature: f32 = 0.8;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = Some(PathBuf::from(&args[i])); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--max-tokens" => { i += 1; max_tokens = args[i].parse().expect("invalid --max-tokens"); }
            "--interactive" | "-i" => { interactive = true; }
            "--temperature" | "-t" => { i += 1; temperature = args[i].parse().expect("invalid --temperature"); }
            "--help" | "-h" => {
                eprintln!("Usage: mamba_generate [OPTIONS]");
                eprintln!("  --model-dir DIR    Path to Mamba model directory (required)");
                eprintln!("  --prompt TEXT       Input prompt (default: \"The meaning of life is\")");
                eprintln!("  --max-tokens N      Max tokens to generate (default: 50)");
                eprintln!("  --temperature T     Sampling temperature (default: 0.8, 0 = greedy)");
                eprintln!("  --interactive, -i   Interactive mode (type prompts, Ctrl-D to quit)");
                return;
            }
            other => { eprintln!("Unknown argument: {other}"); return; }
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("Error: --model-dir is required");
        eprintln!("  Example: cargo run --example mamba_generate --release -- --model-dir models/mamba-130m");
        std::process::exit(1);
    });

    // Load model
    eprint!("Loading model from {}... ", model_dir.display());
    io::stderr().flush().ok();
    let t0 = Instant::now();
    let engine = InferenceEngine::from_pretrained(&model_dir)
        .unwrap_or_else(|e| { eprintln!("\nFailed: {e}"); std::process::exit(1); });
    eprintln!("done ({:.1}s)", t0.elapsed().as_secs_f32());

    // Get tokenizer — try model dir first, fall back to GPT-NeoX from HF cache
    let tokenizer = engine.tokenizer.clone().unwrap_or_else(|| {
        eprintln!("No tokenizer in model dir, trying EleutherAI/gpt-neox-20b from HF cache...");
        let neox_paths = [
            dirs::home_dir().map(|h| h.join(".cache/huggingface/hub/models--EleutherAI--gpt-neox-20b")),
        ];
        for p in neox_paths.iter().flatten() {
            if let Ok(entries) = std::fs::read_dir(p.join("snapshots")) {
                for entry in entries.flatten() {
                    let snap = entry.path();
                    if let Ok(tok) = Tokenizer::from_model_dir(&snap) {
                        eprintln!("Loaded tokenizer from {}", snap.display());
                        return tok;
                    }
                }
            }
        }
        eprintln!("No tokenizer found. Download with: python3 -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')\"");
        std::process::exit(1);
    });

    if !engine.is_ssm() {
        eprintln!("Not an SSM model. Use qwen3_chat for transformer models.");
        std::process::exit(1);
    }

    eprintln!("Model: {} | SSM | max_tokens={max_tokens} temperature={temperature}", engine.config.name);

    if interactive {
        eprintln!("Interactive mode. Type a prompt, press Enter. Ctrl-D to quit.\n");
        let stdin = io::stdin();
        loop {
            eprint!("> ");
            io::stderr().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let line = line.trim();
            if line.is_empty() { continue; }
            generate(
                engine.ssm_model.as_ref().unwrap().as_ref(),
                &tokenizer, line, max_tokens, temperature,
            );
            eprintln!();
        }
    } else {
        generate(
            engine.ssm_model.as_ref().unwrap().as_ref(),
            &tokenizer, &prompt, max_tokens, temperature,
        );
        println!();
    }
}

fn generate(
    model: &dyn Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) {
    let token_ids = tokenizer.encode(prompt).unwrap_or_else(|e| {
        eprintln!("Tokenization failed: {e}");
        std::process::exit(1);
    });

    let pipeline = synapse_inference::generation::GenerationPipeline::new(model);

    let tok_clone = tokenizer.clone();

    // Print prompt
    eprint!("{prompt}");
    io::stderr().flush().ok();

    let t0 = Instant::now();

    let sampler: Option<Box<dyn synapse_inference::generation::Sampler>> =
        if temperature > 0.01 {
            Some(Box::new(TemperatureSampler { temperature }))
        } else {
            None // greedy
        };

    let gen_config = GenerationConfig {
        max_new_tokens: max_tokens,
        sampler,
        on_token: Some(Box::new(move |token_id: u32| {
            if let Ok(text) = tok_clone.decode(&[token_id]) {
                eprint!("{text}");
                io::stderr().flush().ok();
            }
        })),
        ..Default::default()
    };

    let mut state = ModelState::Recurrent;
    let output = pipeline.generate(&token_ids, gen_config, Some(&mut state));

    let gen_tokens = output.num_generated_tokens;
    let tok_per_sec = output.tokens_per_sec;

    eprintln!(
        "\n--- {gen_tokens} tokens in {:.2}s ({tok_per_sec:.1} tok/s) ---",
        output.elapsed.as_secs_f64()
    );
}

/// Minimal home directory detection (avoids adding the `dirs` crate).
mod dirs {
    use std::path::PathBuf;
    pub fn home_dir() -> Option<PathBuf> {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}
