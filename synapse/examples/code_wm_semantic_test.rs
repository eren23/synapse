//! Semantic similarity sanity test for Code WM.
//!
//! Loads a curated set of Python snippets (created by scripts/tokenize_snippets.py)
//! with known categories (sort/str/math/io/http). Encodes each with Code WM,
//! then measures whether within-category cosine > between-category cosine.
//!
//! Strong result = the model actually understands code semantics.
//!
//! Usage:
//!   cargo run --release --example code_wm_semantic_test -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       tests/fixtures/snippets.safetensors \
//!       tests/fixtures/snippets_meta.json

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-30 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors");
    let config_path = args.next().expect("arg 2: config .json");
    let snippets_path = args.next().expect("arg 3: snippets .safetensors");
    let meta_path = args.next().expect("arg 4: snippets metadata .json");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path)).unwrap();
    let tensors = load_safetensors(Path::new(&weights_path)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let snippets = load_safetensors(Path::new(&snippets_path)).unwrap();
    let tokens = &snippets["tokens"];
    let n = tokens.shape[0];
    let max_len = tokens.shape[1];

    let meta_json = fs::read_to_string(&meta_path).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta_json).unwrap();
    let snippets_meta = meta["snippets"].as_array().unwrap();
    let names: Vec<String> = snippets_meta.iter().map(|m| m["name"].as_str().unwrap().to_string()).collect();
    let categories: Vec<String> = snippets_meta.iter().map(|m| m["category"].as_str().unwrap().to_string()).collect();

    println!("Encoding {n} snippets (max_len={max_len})...\n");
    let mut embeddings = Vec::with_capacity(n);
    for i in 0..n {
        let row_f32 = &tokens.data[i * max_len..(i + 1) * max_len];
        let tokens_i64: Vec<i64> = row_f32.iter().map(|&v| v as i64).collect();
        let z = model.encode(&tokens_i64);
        embeddings.push(l2_norm(&z));
    }

    // ── Full pairwise similarity matrix ──
    println!("Pairwise cosine matrix (normalized):\n");
    print!("             ");
    for j in 0..n {
        print!("{:>6}", &names[j][..6.min(names[j].len())]);
    }
    println!();
    for i in 0..n {
        print!("{:<12} ", names[i]);
        for j in 0..n {
            let c = dot(&embeddings[i], &embeddings[j]);
            // Color-ish: bold if same category (via > prefix), blank if diag
            if i == j {
                print!("  1.00");
            } else if categories[i] == categories[j] {
                print!("*{:.2}", c);
            } else {
                print!(" {:.2}", c);
            }
        }
        println!("  [{}]", categories[i]);
    }

    // ── Within vs between category statistics ──
    let mut within: HashMap<String, Vec<f32>> = HashMap::new();
    let mut between: Vec<f32> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let c = dot(&embeddings[i], &embeddings[j]);
            if categories[i] == categories[j] {
                within.entry(categories[i].clone()).or_default().push(c);
            } else {
                between.push(c);
            }
        }
    }

    println!("\n\nWithin-category cosine (higher = model recognizes semantics):");
    let mut within_all = Vec::new();
    for (cat, sims) in &within {
        let mean = sims.iter().sum::<f32>() / sims.len() as f32;
        let min = sims.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  {cat:<6}  mean={mean:.3}  min={min:.3}  max={max:.3}  (n={})", sims.len());
        within_all.extend(sims);
    }
    let w_mean = within_all.iter().sum::<f32>() / within_all.len() as f32;
    let b_mean = between.iter().sum::<f32>() / between.len() as f32;
    println!("  ────");
    println!("  overall within:  mean={w_mean:.3}  (n={})", within_all.len());
    println!("  overall between: mean={b_mean:.3}  (n={})", between.len());
    println!("  separation:      {:+.3}  (positive = model clusters by category)", w_mean - b_mean);

    // ── Top-3 nearest for each query ──
    println!("\n\nTop-3 nearest neighbors per snippet:");
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, dot(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let hits: String = sims.iter().take(3)
            .map(|(j, c)| {
                let mark = if categories[*j] == categories[i] { "✓" } else { " " };
                format!("{mark}{}({:.2})", names[*j], c)
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {:<14} [{}]  →  {}", names[i], categories[i], hits);
    }
}
