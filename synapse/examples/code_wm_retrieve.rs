//! Code WM retrieval demo: semantic code similarity via AST embeddings.
//!
//! Loads a tokenized corpus (produced by scripts/tokenize_code_dir.py),
//! encodes each file with Code WM, and computes cosine similarity between
//! every pair. Prints the top-k most similar file for each query.
//!
//! Usage:
//!   cargo run --release --example code_wm_retrieve -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       tests/fixtures/code_corpus.safetensors \
//!       tests/fixtures/code_corpus_filenames.json \
//!       [top_k=3]
//!
//! G1b is the classification champion (95% edit-type accuracy). G8 also works
//! if you prefer the predictor-capable variant.

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-30 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}

fn cosine_normed(a: &[f32], b: &[f32]) -> f32 {
    // Both are assumed L2-normalized.
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors");
    let config_path = args.next().expect("arg 2: config .json");
    let corpus_path = args.next().expect("arg 3: tokenized corpus .safetensors");
    let filenames_path = args.next().expect("arg 4: filenames .json");
    let top_k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    // Model
    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path)).unwrap();
    let tensors = load_safetensors(Path::new(&weights_path)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();
    println!(
        "Loaded model: dim={}, vocab={}, loops={} (encoder)",
        cfg.model_dim, cfg.vocab_size, cfg.encoder_loops
    );

    // Corpus tokens (stored as f32 because Synapse safetensors loader is f32-only).
    let corpus = load_safetensors(Path::new(&corpus_path)).unwrap();
    let tokens_f32 = &corpus["tokens"];
    let max_len = corpus["tokens"].shape[1];
    let n_files = corpus["tokens"].shape[0];
    println!("Corpus: {n_files} files × {max_len} tokens");

    // Filenames sidecar.
    let filenames_json = fs::read_to_string(&filenames_path).unwrap();
    let filenames_meta: serde_json::Value = serde_json::from_str(&filenames_json).unwrap();
    let filenames: Vec<String> = filenames_meta["filenames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(filenames.len(), n_files, "filename count mismatch");

    // Encode every file → embeddings [n_files, model_dim].
    let t0 = Instant::now();
    let mut embeddings = Vec::with_capacity(n_files);
    for i in 0..n_files {
        let row_f32 = &tokens_f32.data[i * max_len..(i + 1) * max_len];
        let tokens_i64: Vec<i64> = row_f32.iter().map(|&v| v as i64).collect();
        let z = model.encode(&tokens_i64);
        embeddings.push(l2_normalize(&z));
    }
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Encoded {n_files} files in {encode_ms:.1}ms ({:.2}ms/file)\n",
        encode_ms / n_files as f64
    );

    // For each file, find top-k most similar others.
    let k = top_k.min(n_files - 1);
    for q in 0..n_files {
        let q_emb = &embeddings[q];
        let mut sims: Vec<(usize, f32)> = (0..n_files)
            .filter(|&j| j != q)
            .map(|j| (j, cosine_normed(q_emb, &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Query: {}", filenames[q]);
        for (rank, (idx, sim)) in sims.iter().take(k).enumerate() {
            println!("  #{}  cos={:.4}  {}", rank + 1, sim, filenames[*idx]);
        }
        println!();
    }
}
