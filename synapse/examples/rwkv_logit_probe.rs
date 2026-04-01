//! Compare RWKV logits and tokenization against a Python/HF baseline.
//!
//! ```text
//! cargo run --example rwkv_logit_probe --release -- \
//!   --model-dir models/rwkv7-pile-0.1b --prompt "hello"
//! ```

use std::path::PathBuf;

use synapse_inference::engine::InferenceEngine;
use synapse_inference::models::ModelState;
use synapse_inference::tokenizer::Tokenizer;

fn top_k_indices(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs
        .into_iter()
        .take(k)
        .map(|(i, v)| (i as u32, v))
        .collect()
}

fn main() {
    let mut model_dir: Option<PathBuf> = None;
    let mut prompt = String::from("hello");
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                if i < args.len() {
                    model_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--prompt" => {
                i += 1;
                if i < args.len() {
                    prompt = args[i].clone();
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("Usage: rwkv_logit_probe --model-dir DIR [--prompt TEXT]");
        std::process::exit(1);
    });

    let engine = InferenceEngine::from_pretrained(&model_dir).unwrap_or_else(|e| {
        eprintln!("Load failed: {e}");
        std::process::exit(1);
    });

    let tokenizer = engine.tokenizer.clone().unwrap_or_else(|| {
        Tokenizer::from_model_dir(&model_dir).unwrap_or_else(|e| {
            eprintln!("No tokenizer on engine and from_model_dir failed: {e}");
            std::process::exit(1);
        })
    });

    let ssm = engine
        .ssm_model
        .as_ref()
        .expect("Expected SSM (RWKV) model");

    let ids = tokenizer.encode(&prompt).unwrap_or_else(|e| {
        eprintln!("encode: {e}");
        std::process::exit(1);
    });

    println!("SYNAPSE_TOKEN_IDS:{}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));

    let mut state = ModelState::Recurrent;
    let out = ssm.forward_prefill(&ids, &mut state);
    let vocab = out.shape[2];
    assert_eq!(out.logits.len(), vocab, "logits length");

    println!("SYNAPSE_VOCAB_SIZE:{vocab}");
    for (rank, (tid, logit)) in top_k_indices(&out.logits, 15).iter().enumerate() {
        let piece = tokenizer
            .decode(&[*tid])
            .unwrap_or_else(|_| format!("<decode err {tid}>"));
        let esc = piece.replace('\n', "\\n");
        println!("SYNAPSE_TOP_{}: id={} logit={:.4} decode={:?}", rank + 1, tid, logit, esc);
    }
}
