//! Encode code snippets with `microsoft/unixcoder-base` in pure Rust.
//!
//! Reproduces the feature-extraction call used by the codewm3 paper's
//! tap (`collectors/precompute_backbone_features.py`): CLS pool of the
//! frozen UniXcoder encoder, 768-dim f32.
//!
//! Usage:
//!
//! ```bash
//! # Point at a local UniXcoder dir (must contain model.safetensors,
//! # config.json, vocab.json, merges.txt).
//! cargo run --release -p synapse-inference --example unixcoder_embed -- \
//!     --model-dir ~/.cache/huggingface/hub/models--microsoft--unixcoder-base/snapshots/* \
//!     --before before.py --after after.py
//! ```
//!
//! Prints the two 768-dim CLS features' mean/norm and their cosine
//! similarity — the same quantity the paper reports as
//! `cos(h_b, h_a) = 0.766 ± 0.029`.
//!
//! NOTE: Synapse's GPT-2-style BPE tokenizer does not yet reproduce
//! `RobertaTokenizer.add_prefix_space=False` byte-for-byte, so token ids
//! on novel inputs can differ from HuggingFace by ~1–2 pieces. The
//! encoder itself is bit-exact with HF (see `tests/unixcoder_parity.rs`);
//! when you need publication-grade features, tokenize with HF and feed
//! the ids in directly rather than relying on this example's tokenizer
//! path.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use synapse_inference::models::text_encoder::{
    parse_roberta_config, unixcoder_base, CodeDeltaTokConfig, CodeDeltaTokHead, RoBERTaEncoder,
};
use synapse_inference::tokenizer::Tokenizer;
use synapse_inference::weight_loading::{load_safetensors, WeightMapper};

fn parse_args() -> Result<Args, String> {
    let mut model_dir: Option<PathBuf> = None;
    let mut before_path: Option<PathBuf> = None;
    let mut after_path: Option<PathBuf> = None;
    let mut cdt_ckpt: Option<PathBuf> = None;
    let mut max_length: usize = 512;

    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model-dir" => model_dir = it.next().map(PathBuf::from),
            "--before"    => before_path = it.next().map(PathBuf::from),
            "--after"     => after_path = it.next().map(PathBuf::from),
            "--cdt-ckpt"  => cdt_ckpt = it.next().map(PathBuf::from),
            "--max-length" => max_length = it.next().and_then(|v| v.parse().ok()).unwrap_or(512),
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    Ok(Args {
        model_dir: model_dir.ok_or("--model-dir is required")?,
        before_path: before_path.ok_or("--before is required")?,
        after_path: after_path.ok_or("--after is required")?,
        cdt_ckpt,
        max_length,
    })
}

fn print_help() {
    println!(
        "unixcoder_embed --model-dir <dir> --before <file> --after <file> \
         [--cdt-ckpt <safetensors>] [--max-length 512]"
    );
}

struct Args {
    model_dir: PathBuf,
    before_path: PathBuf,
    after_path: PathBuf,
    cdt_ckpt: Option<PathBuf>,
    max_length: usize,
}

fn first_safetensors(dir: &Path) -> Option<PathBuf> {
    let p = dir.join("model.safetensors");
    if p.exists() { return Some(p); }
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return Some(path);
        }
    }
    None
}

/// HF RoBERTa/UniXcoder encode convention: `<s> … </s>` then pad with
/// `<pad>`. ids 0, 2, 1 respectively in UniXcoder's vocab.
fn encode_unixcoder(tok: &Tokenizer, text: &str, max_length: usize,
                    bos: u32, eos: u32, pad: u32) -> (Vec<i64>, Vec<i64>) {
    let body = tok.encode(text).unwrap_or_default();
    let reserve = 2; // room for <s> and </s>
    let body_len = body.len().min(max_length.saturating_sub(reserve));
    let mut ids: Vec<u32> = Vec::with_capacity(max_length);
    ids.push(bos);
    ids.extend_from_slice(&body[..body_len]);
    ids.push(eos);
    let real_len = ids.len();
    while ids.len() < max_length {
        ids.push(pad);
    }
    let input_ids: Vec<i64> = ids.into_iter().map(|id| id as i64).collect();
    let attention_mask: Vec<i64> = (0..max_length)
        .map(|i| if i < real_len { 1 } else { 0 })
        .collect();
    (input_ids, attention_mask)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); print_help(); return ExitCode::from(2); }
    };

    // Load config (fall back to unixcoder_base() if absent).
    let config_path = args.model_dir.join("config.json");
    let config = if config_path.exists() {
        let text = fs::read_to_string(&config_path).expect("read config.json");
        parse_roberta_config(&text).expect("parse config.json")
    } else {
        unixcoder_base()
    };

    let weights_path = first_safetensors(&args.model_dir)
        .expect("no *.safetensors in model-dir");
    println!("Loading weights: {}", weights_path.display());
    let weights = load_safetensors(&weights_path).expect("load weights");

    let mut model = RoBERTaEncoder::from_config(config);
    let result = model.load_weights(weights, &WeightMapper::unixcoder())
        .expect("load_weights");
    if !result.missing.is_empty() {
        eprintln!("WARN: missing targets: {:?}", result.missing);
    }

    let tok = Tokenizer::from_model_dir(&args.model_dir).expect("load tokenizer");

    // UniXcoder's fixed special-token ids.
    let bos = 0u32; // <s>
    let eos = 2u32; // </s>
    let pad = 1u32; // <pad>

    let before = fs::read_to_string(&args.before_path).expect("read --before");
    let after  = fs::read_to_string(&args.after_path).expect("read --after");

    let (ids_b, mask_b) = encode_unixcoder(&tok, &before, args.max_length, bos, eos, pad);
    let (ids_a, mask_a) = encode_unixcoder(&tok, &after,  args.max_length, bos, eos, pad);

    let h_b = model.cls_feature(&ids_b, &mask_b);
    let h_a = model.cls_feature(&ids_a, &mask_a);

    let norm_b: f32 = h_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_a: f32 = h_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = cosine(&h_b, &h_a);

    println!(
        "Before  CLS:  ‖h_b‖ = {:.4}   mean = {:.4e}",
        norm_b,
        h_b.iter().sum::<f32>() / h_b.len() as f32,
    );
    println!(
        "After   CLS:  ‖h_a‖ = {:.4}   mean = {:.4e}",
        norm_a,
        h_a.iter().sum::<f32>() / h_a.len() as f32,
    );
    println!("cos(h_b, h_a) = {cos:.4}   (paper reports 0.766 ± 0.029)");

    if let Some(path) = args.cdt_ckpt {
        println!("\nLoading CDT head: {}", path.display());
        let weights = load_safetensors(&path).expect("load CDT safetensors");
        let mut head = CodeDeltaTokHead::from_config(CodeDeltaTokConfig::paper_default());
        let result = head
            .load_weights(weights, &WeightMapper::code_deltatok())
            .expect("CDT load_weights");
        if !result.missing.is_empty() {
            eprintln!("WARN: missing CDT targets: {:?}", result.missing);
        }
        let delta = head.encode(&h_b, &h_a);
        let recon = head.decode(&delta, &h_b);

        let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        let recon_cos = cosine(&recon, &h_a);
        println!(
            "Delta token: ‖d‖ = {delta_norm:.4} (dim = {}), recon cos(recon, h_a) = {recon_cos:.4}",
            delta.len(),
        );
    }

    ExitCode::SUCCESS
}
