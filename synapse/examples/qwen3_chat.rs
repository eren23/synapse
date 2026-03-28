//! Interactive chat with either a real Qwen3 checkpoint or a tiny demo model.
//!
//! Usage:
//!   cargo run --example qwen3_chat --release -- --model-dir /path/to/Qwen3-0.6B
//!   cargo run --example qwen3_chat --release -- --demo

use std::cell::Cell;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

use synapse_inference::capabilities::CapabilityReport;
use synapse_inference::chat_template::ChatMessage;
use synapse_inference::config::*;
use synapse_inference::engine::InferenceEngine;
use synapse_inference::generation::{CombinedSampler, GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelBuilder;
use synapse_inference::model_adapter::ModelAdapterKind;
use synapse_inference::model_adapter::ThinkingMode;
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
        style: Default::default(),
        scaling: Default::default(),
    };

    let mut model = ModelBuilder::from_config(&cfg);
    let weights = generate_fake_hf_weights(&cfg);
    let mapper = WeightMapper::qwen3();
    let result = model
        .load_weights(weights, &mapper)
        .expect("Demo weights should load");
    assert!(
        result.missing.is_empty(),
        "Missing keys: {:?}",
        result.missing
    );
    assert!(
        result.unexpected.is_empty(),
        "Unexpected keys: {:?}",
        result.unexpected
    );

    InferenceEngine {
        model,
        quantized_model: None,
        config: cfg,
        model_adapter_kind: ModelAdapterKind::Qwen3,
        tokenizer: None,
        chat_template: None,
        #[cfg(feature = "metal")]
        backend: synapse_inference::metal::ComputeBackend::auto(),
        #[cfg(feature = "metal")]
        metal_model_bufs_cell: None,
    }
}

enum Mode {
    Demo,
    Chat {
        dir: PathBuf,
        quantize: bool,
        speculative: bool,
        thinking_mode: Option<ThinkingMode>,
        profile_stages: bool,
        max_new_tokens: usize,
        prompt: Option<String>,
    },
    Verify(PathBuf),
    InspectPrompt {
        dir: PathBuf,
        prompt: String,
        thinking_mode: Option<ThinkingMode>,
    },
    Capabilities(Option<PathBuf>),
}

struct PreparedPrompt {
    prompt: String,
    prompt_tokens: Vec<u32>,
    render_elapsed: Duration,
    encode_elapsed: Duration,
    reasoning_start_id: Option<u32>,
    reasoning_end_id: Option<u32>,
}

fn parse_args() -> Result<Mode, String> {
    let mut args = std::env::args().skip(1);
    let mut model_dir = None;
    let mut verify = false;
    let mut inspect_prompt = false;
    let mut quantize = false;
    let mut speculative = false;
    let mut capabilities = false;
    let mut thinking_mode = None;
    let mut profile_stages = false;
    let mut max_new_tokens = 256usize;
    let mut prompt = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => return Ok(Mode::Demo),
            "--verify" => verify = true,
            "--inspect-prompt" => inspect_prompt = true,
            "--capabilities" => capabilities = true,
            "--quantize" | "-q" => quantize = true,
            "--speculative" | "-s" => speculative = true,
            "--profile-stages" => profile_stages = true,
            "--max-new-tokens" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-new-tokens requires a value".to_string())?;
                max_new_tokens = value
                    .parse::<usize>()
                    .map_err(|_| "--max-new-tokens must be an integer".to_string())?;
            }
            "--thinking" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--thinking requires a mode".to_string())?;
                thinking_mode = Some(ThinkingMode::parse_cli(&value)?);
            }
            "--prompt" => {
                prompt = Some(
                    args.next()
                        .ok_or_else(|| "--prompt requires a value".to_string())?,
                );
            }
            "--model-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--model-dir requires a path".to_string())?;
                model_dir = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("Usage:");
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path/to/Qwen3-0.6B"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --verify"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --quantize"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --thinking auto"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --prompt \"hello\""
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --prompt \"hello\" --profile-stages"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --prompt \"hello\" --max-new-tokens 32"
                );
                println!(
                    "  cargo run --example qwen3_chat --release -- --model-dir /path --inspect-prompt --prompt \"hello\""
                );
                println!("  cargo run --example qwen3_chat --release -- --capabilities");
                println!("  cargo run --example qwen3_chat --release -- --capabilities --model-dir /path");
                println!("  cargo run --example qwen3_chat --release -- --demo");
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    if capabilities {
        return Ok(Mode::Capabilities(model_dir));
    }
    if verify && inspect_prompt {
        return Err("--verify and --inspect-prompt are mutually exclusive".to_string());
    }
    if inspect_prompt {
        let dir = model_dir.ok_or_else(|| "--inspect-prompt requires --model-dir".to_string())?;
        return Ok(Mode::InspectPrompt {
            dir,
            prompt: prompt.unwrap_or_else(|| "hello".to_string()),
            thinking_mode,
        });
    }

    match (model_dir, verify) {
        (Some(dir), true) => Ok(Mode::Verify(dir)),
        (Some(dir), false) => Ok(Mode::Chat {
            dir,
            quantize,
            speculative,
            thinking_mode,
            profile_stages,
            max_new_tokens,
            prompt,
        }),
        (None, _) => Ok(Mode::Demo),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = parse_args().map_err(io::Error::other)?;

    match mode {
        Mode::Verify(dir) => run_verify(dir)?,
        Mode::InspectPrompt {
            dir,
            prompt,
            thinking_mode,
        } => run_inspect_prompt(dir, prompt, thinking_mode)?,
        Mode::Chat {
            dir,
            quantize,
            speculative,
            thinking_mode,
            profile_stages,
            max_new_tokens,
            prompt,
        } => run_pretrained_chat(
            dir,
            quantize,
            speculative,
            thinking_mode,
            profile_stages,
            max_new_tokens,
            prompt,
        )?,
        Mode::Capabilities(dir) => run_capabilities(dir)?,
        Mode::Demo => run_demo_chat(),
    }

    Ok(())
}

fn prepare_prompt(
    engine: &InferenceEngine,
    tokenizer: &synapse_inference::tokenizer::Tokenizer,
    line: &str,
    thinking_mode: ThinkingMode,
) -> Result<PreparedPrompt, Box<dyn std::error::Error>> {
    let render_start = Instant::now();
    let prompt = engine.format_chat_with_mode(
        &[ChatMessage {
            role: "user".into(),
            content: line.to_string(),
        }],
        thinking_mode,
    )?;
    let render_elapsed = render_start.elapsed();

    let encode_start = Instant::now();
    let prompt_tokens = engine.encode(&prompt)?;
    let encode_elapsed = encode_start.elapsed();

    let (reasoning_start_id, reasoning_end_id) = if let Some(markers) = engine.reasoning_markers() {
        let start_id = tokenizer
            .encode(markers.start)
            .ok()
            .filter(|ids| ids.len() == 1)
            .and_then(|ids| ids.first().copied());
        let end_id = tokenizer
            .encode(markers.end)
            .ok()
            .filter(|ids| ids.len() == 1)
            .and_then(|ids| ids.first().copied());
        (start_id, end_id)
    } else {
        (None, None)
    };

    Ok(PreparedPrompt {
        prompt,
        prompt_tokens,
        render_elapsed,
        encode_elapsed,
        reasoning_start_id,
        reasoning_end_id,
    })
}

fn print_prompt_inspection(
    engine: &InferenceEngine,
    prepared: &PreparedPrompt,
    thinking_mode: ThinkingMode,
) {
    println!("Chat: family={} thinking={}", engine.model_family(), thinking_mode);
    println!("Rendered prompt ({} chars): {:?}", prepared.prompt.len(), prepared.prompt);
    println!(
        "Prompt tokens ({}): {:?}",
        prepared.prompt_tokens.len(),
        prepared.prompt_tokens
    );
    println!(
        "Reasoning markers: start={:?} end={:?}",
        prepared.reasoning_start_id, prepared.reasoning_end_id
    );
}

fn print_stage_profile(
    render_elapsed: Duration,
    encode_elapsed: Duration,
    output: &synapse_inference::generation::GenerationOutput,
    hidden_tokens: usize,
    visible_tokens: usize,
) {
    let decode_elapsed = output.elapsed.saturating_sub(output.prefill_elapsed);
    println!(
        "Profile: render={:.1}ms encode={:.1}ms prefill={:.1}ms decode={:.1}ms hidden_tokens={} visible_tokens={}",
        render_elapsed.as_secs_f64() * 1000.0,
        encode_elapsed.as_secs_f64() * 1000.0,
        output.prefill_elapsed.as_secs_f64() * 1000.0,
        decode_elapsed.as_secs_f64() * 1000.0,
        hidden_tokens,
        visible_tokens,
    );
}

fn run_capabilities(model_dir: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let report = if let Some(dir) = model_dir {
        let engine = InferenceEngine::from_pretrained(&dir)?;
        engine.capability_report()
    } else {
        CapabilityReport::for_current_build()
    };
    println!("{}", report.to_json()?);
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
    println!(
        "\nEmbed[0,:8]:  {:?}",
        &engine.model.embed_tokens[id0 * h..id0 * h + 8]
    );
    let id_last = *tokens.last().unwrap() as usize;
    println!(
        "Embed[-1,:8]: {:?}",
        &engine.model.embed_tokens[id_last * h..id_last * h + 8]
    );

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

fn run_inspect_prompt(
    model_dir: PathBuf,
    prompt: String,
    thinking_mode: Option<ThinkingMode>,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("Loading checkpoint from {}...", model_dir.display());
    io::stdout().flush()?;
    let engine = InferenceEngine::from_pretrained(&model_dir)?;
    println!(" done ({} params)", engine.param_count());

    let thinking_mode = thinking_mode.unwrap_or_else(|| engine.default_cli_thinking_mode());
    let tokenizer = engine
        .tokenizer()
        .expect("pretrained engine has tokenizer")
        .clone();
    let prepared = prepare_prompt(&engine, &tokenizer, &prompt, thinking_mode)?;

    println!("{}", engine.runtime_plan().log_line());
    print_prompt_inspection(&engine, &prepared, thinking_mode);
    Ok(())
}

fn run_pretrained_chat(
    model_dir: PathBuf,
    quantize: bool,
    speculative: bool,
    thinking_mode: Option<ThinkingMode>,
    profile_stages: bool,
    max_new_tokens: usize,
    prompt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("Loading checkpoint from {}...", model_dir.display());
    io::stdout().flush()?;
    let mut engine = InferenceEngine::from_pretrained(&model_dir)?;
    println!(" done ({} params)", engine.param_count());

    if quantize {
        print!("Quantizing to INT8...");
        io::stdout().flush()?;
        engine.quantize();
        println!(" done");
    }
    let thinking_mode = thinking_mode.unwrap_or_else(|| engine.default_cli_thinking_mode());
    println!("{}", engine.runtime_plan().log_line());
    println!(
        "Chat: family={} thinking={}",
        engine.model_family(),
        thinking_mode
    );

    let tokenizer = engine
        .tokenizer()
        .expect("pretrained engine has tokenizer")
        .clone();
    let stop_sequences = tokenizer.encode("<|im_end|>").unwrap_or_default();
    let eos_token_id = tokenizer.eos_token_id();

    let run_prompt = |line: &str| -> Result<(), Box<dyn std::error::Error>> {
        let prepared = prepare_prompt(&engine, &tokenizer, line, thinking_mode)?;
        let stream_tokenizer = tokenizer.clone();
        let pipeline = engine.generation_pipeline();

        let max_seq = prepared.prompt_tokens.len() + max_new_tokens;
        let mut state = engine.create_state(max_seq)?;

        let in_think = Rc::new(Cell::new(false));
        let think_shown = Rc::new(Cell::new(false));
        let hidden_tokens = Rc::new(Cell::new(0usize));
        let visible_tokens = Rc::new(Cell::new(0usize));
        let reasoning_start_id = prepared.reasoning_start_id;
        let reasoning_end_id = prepared.reasoning_end_id;

        let in_think_cb = Rc::clone(&in_think);
        let think_shown_cb = Rc::clone(&think_shown);
        let hidden_tokens_cb = Rc::clone(&hidden_tokens);
        let visible_tokens_cb = Rc::clone(&visible_tokens);

        let config = GenerationConfig {
            max_new_tokens,
            eos_token_id,
            stop_sequences: vec![stop_sequences.clone()],
            speculative_k: if speculative { 4 } else { 0 },
            speculative_draft_layers: 0,
            combined: Some(CombinedSampler {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
            }),
            seed: Some(42),
            on_token: Some(Box::new(move |token| {
                if Some(token) == reasoning_start_id {
                    hidden_tokens_cb.set(hidden_tokens_cb.get() + 1);
                    in_think_cb.set(true);
                    if !think_shown_cb.get() {
                        print!("(thinking...) ");
                        let _ = io::stdout().flush();
                        think_shown_cb.set(true);
                    }
                    return;
                }
                if Some(token) == reasoning_end_id {
                    hidden_tokens_cb.set(hidden_tokens_cb.get() + 1);
                    in_think_cb.set(false);
                    print!("\r              \r");
                    let _ = io::stdout().flush();
                    return;
                }
                if in_think_cb.get() {
                    hidden_tokens_cb.set(hidden_tokens_cb.get() + 1);
                    return;
                }
                visible_tokens_cb.set(visible_tokens_cb.get() + 1);
                if let Ok(piece) = stream_tokenizer.decode_token_piece(token) {
                    print!("{piece}");
                    let _ = io::stdout().flush();
                }
            })),
            ..Default::default()
        };

        let output = pipeline.generate(&prepared.prompt_tokens, config, Some(&mut state));
        println!();
        let mode_str = if engine.is_quantized() { "INT8" } else { "f32" };
        println!(
            "Prefill: {} tokens in {:.0}ms ({:.0} tok/s) | Decode: {} tokens at {:.1} tok/s | {}",
            output.num_prompt_tokens,
            output.prefill_elapsed.as_millis(),
            output.prefill_tokens_per_sec,
            output.num_generated_tokens,
            output.tokens_per_sec,
            mode_str,
        );
        if profile_stages {
            print_prompt_inspection(&engine, &prepared, thinking_mode);
            print_stage_profile(
                prepared.render_elapsed,
                prepared.encode_elapsed,
                &output,
                hidden_tokens.get(),
                visible_tokens.get(),
            );
        }

        Ok(())
    };

    if let Some(prompt) = prompt {
        run_prompt(&prompt)?;
        return Ok(());
    }

    println!("Type 'quit' to exit.");

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
        run_prompt(&line)?;
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
