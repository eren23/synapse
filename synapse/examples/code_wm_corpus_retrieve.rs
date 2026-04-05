//! Real-world retrieval analysis on a large Python corpus.
//!
//! Loads a pre-tokenized file index (from scripts/build_file_index.py),
//! encodes every file with Code WM, and finds top-k nearest neighbors.
//! Prints qualitative examples + summary statistics.
//!
//! Usage:
//!   cargo run --release --example code_wm_corpus_retrieve -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       tests/fixtures/file_index.safetensors \
//!       tests/fixtures/file_index_meta.json \
//!       [num_examples=15]

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-30 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn first_dir(path: &str) -> String {
    path.split('/').next().unwrap_or(path).to_string()
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let index_path = args.next().expect("arg 3: file_index.safetensors");
    let meta_path = args.next().expect("arg 4: file_index_meta.json");
    let num_examples: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);

    // Load model
    let t0 = Instant::now();
    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Model loaded in {load_ms:.1}ms");

    // Load index
    let index = load_safetensors(Path::new(&index_path)).unwrap();
    let tokens_t = &index["tokens"];
    let n = tokens_t.shape[0];
    let max_len = tokens_t.shape[1];

    let meta_str = fs::read_to_string(&meta_path).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    let entries = meta["entries"].as_array().unwrap();
    let paths: Vec<String> = entries.iter().map(|e| e["path"].as_str().unwrap().to_string()).collect();
    let previews: Vec<String> = entries.iter()
        .map(|e| e["preview"].as_str().unwrap_or("").to_string()).collect();

    println!("Corpus: {n} files × {max_len} tokens");

    // Encode all
    let t0 = Instant::now();
    let mut embeddings = Vec::with_capacity(n);
    for i in 0..n {
        let row = &tokens_t.data[i * max_len..(i + 1) * max_len];
        let toks_i64: Vec<i64> = row.iter().map(|&v| v as i64).collect();
        let z = model.encode(&toks_i64);
        embeddings.push(l2_norm(&z));
    }
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Encoded {n} files in {enc_ms:.1}ms ({:.1}ms/file)\n", enc_ms / n as f64);

    // Qualitative: top-5 neighbors for num_examples random queries
    println!("=== Qualitative: top-3 neighbors for {num_examples} random queries ===\n");
    let stride = n / num_examples.max(1);
    for i in (0..n).step_by(stride).take(num_examples) {
        let mut sims: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, dot(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Q: {}", paths[i]);
        println!("   preview: {}", previews[i].replace('\n', " | "));
        for (rank, (j, c)) in sims.iter().take(3).enumerate() {
            println!("   #{}  cos={:.3}  {}", rank + 1, c, paths[*j]);
        }
        println!();
    }

    // Quantitative: separation using top-level directory as "category"
    // (e.g. transformers/, torch/, numpy/, etc. — likely-related files share a dir)
    println!("=== Quantitative: within-pkg vs between-pkg cosine ===\n");
    let cats: Vec<String> = paths.iter().map(|p| first_dir(p)).collect();
    let mut within_sum = 0.0_f64; let mut within_n = 0usize;
    let mut between_sum = 0.0_f64; let mut between_n = 0usize;
    for i in 0..n {
        for j in (i+1)..n {
            let c = dot(&embeddings[i], &embeddings[j]) as f64;
            if cats[i] == cats[j] {
                within_sum += c; within_n += 1;
            } else {
                between_sum += c; between_n += 1;
            }
        }
    }
    let within_mean = within_sum / within_n as f64;
    let between_mean = between_sum / between_n as f64;
    println!("Within-package cos:  mean={:.4}  (n={})", within_mean, within_n);
    println!("Between-package cos: mean={:.4}  (n={})", between_mean, between_n);
    println!("Separation:          {:+.4}  (positive = packages cluster)", within_mean - between_mean);

    // Count unique top-level packages
    let mut pkgs: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for c in &cats { pkgs.insert(c); }
    println!("\nUnique top-level packages in corpus: {}", pkgs.len());
}
