//! Interactive chat with LFM2.5-350M.
//!
//! Usage:
//!   cargo run --release -p synapse-inference --example lfm25_chat -- <model-dir>
//!   cargo run --release --features metal -p synapse-inference --example lfm25_chat -- <model-dir>
//!
//! Example:
//!   cargo run --release --features metal -p synapse-inference --example lfm25_chat -- \
//!     models/lfm25-350m-ready

use std::io::{self, Write};
use std::path::Path;

use synapse_inference::models::ssm::hybrid::config::HybridConfig;
use synapse_inference::models::ssm::hybrid::model::HybridModel;
use synapse_inference::models::traits::{Model, ModelState};
use synapse_inference::tokenizer::Tokenizer;
use synapse_inference::weight_loading::load_gguf_with_raw_q4;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).expect(
        "Usage: lfm25_chat <model-dir>\n  model-dir should contain: *.gguf, tokenizer.json, config.json"
    );
    let model_path = Path::new(model_dir);

    // Find GGUF file — prefer Q8_0 > Q6_K > Q4_K > Q4_0 for quality
    let mut ggufs: Vec<_> = std::fs::read_dir(model_path)
        .expect("can't read model dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "gguf"))
        .map(|e| e.path())
        .collect();
    ggufs.sort_by_key(|p| {
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.contains("F16") || name.contains("F32") { 0 }
        else if name.contains("Q8") { 1 }
        else if name.contains("Q6") { 2 }
        else if name.contains("Q4_K") { 3 }
        else { 4 } // Q4_0
    });
    let gguf_path = ggufs.first().expect("no .gguf file found in model dir").clone();

    eprintln!("Loading model from {}...", gguf_path.display());
    let t0 = std::time::Instant::now();

    let (weights, _q4_raw) = load_gguf_with_raw_q4(&gguf_path).expect("Failed to load GGUF");
    let config = HybridConfig::lfm25_350m();
    let model = HybridModel::from_weights_lfm25(config, &weights, 2048)
        .expect("Failed to build model");

    let tokenizer = Tokenizer::from_model_dir(model_path)
        .expect("Failed to load tokenizer (need tokenizer.json in model dir)");

    eprintln!("Loaded in {:.1}s", t0.elapsed().as_secs_f32());
    eprintln!("  embed_norm: {} elements", model.embed_norm_weight.len());
    eprintln!("  final_norm[:5]: {:?}", &model.final_norm_weight[..5.min(model.final_norm_weight.len())]);
    eprintln!();

    // Tokenize using Python's tokenizers library (our BPE has merge issues)
    // Falls back to our Rust tokenizer if Python not available
    let tokenize_prompt = |user_msg: &str| -> Vec<u32> {
        // Try Python tokenizers first (correct BPE merges)
        let py_code = format!(
            "from tokenizers import Tokenizer; t = Tokenizer.from_file('{}'); \
             enc = t.encode('<|im_start|>user\\n{}<|im_end|>\\n<|im_start|>assistant\\n'); \
             print(','.join(str(i) for i in enc.ids))",
            model_path.join("tokenizer.json").display(),
            user_msg.replace("'", "\\'"),
        );
        if let Ok(output) = std::process::Command::new("python3")
            .args(["-c", &py_code])
            .output()
        {
            if output.status.success() {
                let s = String::from_utf8_lossy(&output.stdout);
                let ids: Vec<u32> = s.trim().split(',')
                    .filter_map(|x| x.parse().ok())
                    .collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
        }
        // Fallback: raw encode (may have BPE merge issues)
        eprintln!("[warn: using fallback tokenizer — install `pip3 install tokenizers` for best results]");
        let prompt = format!(
            "<|startoftext|><|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        );
        tokenizer.encode(&prompt).unwrap_or_default()
    };

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        let token_ids = tokenize_prompt(input);
        eprintln!("[prompt {} tokens: {:?}...]", token_ids.len(), &token_ids[..token_ids.len().min(20)]);

        // Prefill
        model.reset_state();
        let mut state = ModelState::Recurrent;
        let out = model.forward_prefill(&token_ids, &mut state);

        // Debug: compare key logit values with HF reference
        eprintln!("[logits[0..5]: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]]",
            out.logits[0], out.logits[1], out.logits[2], out.logits[3], out.logits[4]);
        eprintln!("[token 1098 (The): {:.4}, token 708 (\\n): {:.4}]",
            out.logits[1098], out.logits[708]);
        eprintln!("[HF reference: logits[0..5] = [-4.24, 1.86, 6.44, -3.90, -3.58], token 1098=22.45, token 708=9.45]");

        // Greedy decode
        let mut next = sample(&out.logits, 0.7, 50);
        let max_tokens = 256;
        let eos_tokens: Vec<u32> = vec![7]; // eos_token_id=7 from config

        let t0 = std::time::Instant::now();
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            if eos_tokens.contains(&next) {
                break;
            }
            generated.push(next);

            // Print token as we decode (streaming)
            if let Ok(piece) = tokenizer.decode_token_piece(next) {
                print!("{piece}");
                io::stdout().flush().unwrap();
            }
            // Debug: show top-5 tokens for first few steps
            if generated.len() <= 3 {
                eprint!("[kv_len={}] ", model.kv_cache_len());
                let mut top5: Vec<(usize, f32)> = out.logits.iter().enumerate()
                    .map(|(i, &v)| (i, v)).collect();
                top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                top5.truncate(5);
                let top5_str: Vec<String> = top5.iter()
                    .map(|(id, score)| format!("{}({:.2})", id, score))
                    .collect();
                eprint!("[top5: {}] ", top5_str.join(", "));
            }

            let out = model.forward_one(next, &mut state);
            next = sample(&out.logits, 0.7, 50);
        }

        let elapsed = t0.elapsed();
        let tps = if !generated.is_empty() {
            generated.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!("\n[{} tokens in {:.1}s = {:.1} tok/s]\n", generated.len(), elapsed.as_secs_f32(), tps);
    }
}

fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32).unwrap_or(0);
    }

    // Top-K + temperature sampling
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v / temperature)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax
    let max_val = indexed[0].1;
    let exps: Vec<f64> = indexed.iter().map(|(_, v)| ((*v - max_val) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();

    // Sample
    let r: f64 = {
        use std::time::SystemTime;
        let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
        (t.subsec_nanos() as f64) / 4_294_967_296.0
    };
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return indexed[i].0 as u32;
        }
    }
    indexed[0].0 as u32
}
