//! Fully native Code WM pipeline: walk a directory → tokenize .py files in Rust
//! → encode with CodeWM → compute pairwise cosine → find top-k similar.
//!
//! Zero Python runtime dependency. Single binary does tokenization + encoding + retrieval.
//!
//! Usage:
//!   cargo run --release --example code_wm_pipeline -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       scripts \
//!       [top_k=3] [max_len=512]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use synapse_code_tokenizer::tokenize;
use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-30 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn collect_py_files(root: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= max_files { break; }
        if let Ok(entries) = fs::read_dir(&dir) {
            let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let p = entry.path();
                if p.is_dir() { stack.push(p); }
                else if p.extension().map(|e| e == "py").unwrap_or(false) {
                    out.push(p);
                    if out.len() >= max_files { break; }
                }
            }
        }
    }
    out.sort();
    out
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let dir = args.next().expect("arg 3: directory with .py files");
    let top_k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_len: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let max_files = 64;

    // Load model
    let t0 = Instant::now();
    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Model loaded in {load_ms:.1}ms (dim={}, vocab={})", cfg.model_dim, cfg.vocab_size);

    // Walk dir, tokenize + encode each .py file
    let files = collect_py_files(Path::new(&dir), max_files);
    println!("Found {} .py files in {dir}", files.len());
    if files.is_empty() { return; }

    let mut embeddings = Vec::with_capacity(files.len());
    let mut names = Vec::with_capacity(files.len());
    let mut tok_ms_total = 0.0;
    let mut enc_ms_total = 0.0;
    let root = Path::new(&dir).canonicalize().unwrap();

    for path in &files {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let t = Instant::now();
        let toks_u16 = tokenize(&source, max_len);
        tok_ms_total += t.elapsed().as_secs_f64() * 1000.0;

        // Convert u16 → i64 for the model (vocab values are small integers, fit trivially)
        let toks_i64: Vec<i64> = toks_u16.iter().map(|&t| t as i64).collect();

        let t = Instant::now();
        let z = model.encode(&toks_i64);
        enc_ms_total += t.elapsed().as_secs_f64() * 1000.0;

        embeddings.push(l2_norm(&z));
        names.push(path.strip_prefix(&root).unwrap_or(path).display().to_string());
    }

    let n = embeddings.len();
    println!(
        "\nEncoded {n} files — tokenize: {tok_ms_total:.1}ms ({:.2}ms/file), encode: {enc_ms_total:.1}ms ({:.2}ms/file)",
        tok_ms_total / n as f64,
        enc_ms_total / n as f64,
    );

    // Top-k pairwise retrieval
    println!("\nTop-{top_k} nearest neighbors per file:\n");
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, dot(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let hits: String = sims.iter().take(top_k)
            .map(|(j, c)| format!("{}({:.3})", names[*j], c))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {:<50}  →  {hits}", names[i]);
    }
}
