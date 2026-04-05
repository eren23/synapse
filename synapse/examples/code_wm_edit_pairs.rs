//! Before/after edit similarity test for Code WM.
//!
//! Code WM was trained on CommitPackFT edits (before→after + action). For each
//! pair, we encode both snippets and check that cos(before, after) is high
//! (pairs cluster tighter than random snippets).
//!
//! Metrics:
//!   pair_cos    — cos(before_i, after_i) per pair
//!   delta_norm  — ||after_i - before_i|| per pair
//!   random_cos  — cos(before_i, after_j) for i≠j (baseline)
//!
//! Strong result: pair_cos mean >> random_cos mean.
//!
//! Usage:
//!   cargo run --release --example code_wm_edit_pairs -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       tests/fixtures/edit_pairs.safetensors \
//!       tests/fixtures/edit_pairs_meta.json

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

fn delta_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let pairs = args.next().expect("arg 3: pairs .safetensors");
    let meta = args.next().expect("arg 4: pairs meta .json");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let pairs_t = load_safetensors(Path::new(&pairs)).unwrap();
    let tokens = &pairs_t["tokens"];
    let n_rows = tokens.shape[0];
    let max_len = tokens.shape[1];
    let num_pairs = n_rows / 2;

    let meta_json = fs::read_to_string(&meta).unwrap();
    let meta_parsed: serde_json::Value = serde_json::from_str(&meta_json).unwrap();
    let pair_meta = meta_parsed["pairs"].as_array().unwrap();

    // Encode all snippets (alternating before/after).
    let mut embeds_raw = Vec::with_capacity(n_rows);
    let mut embeds_norm = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row: Vec<i64> = tokens.data[i * max_len..(i + 1) * max_len].iter().map(|&v| v as i64).collect();
        let z = model.encode(&row);
        embeds_norm.push(l2_norm(&z));
        embeds_raw.push(z);
    }

    println!("=== Edit-pair analysis (G1b) ===");
    println!("Training: model was trained on CommitPackFT before→after edits.");
    println!("Expected: cos(before, after) close to 1.0; delta_norm small.\n");

    let mut pair_cos = Vec::new();
    let mut pair_delta = Vec::new();
    println!("{:<20} {:<14} {:>8}  {:>10}  {:>10}", "pair", "category", "cos", "delta_L2", "base_cos");
    println!("{}", "─".repeat(72));
    for p in 0..num_pairs {
        let before = &embeds_norm[p * 2];
        let after = &embeds_norm[p * 2 + 1];
        let c = dot(before, after);
        let d = delta_l2(&embeds_raw[p * 2], &embeds_raw[p * 2 + 1]);
        pair_cos.push(c);
        pair_delta.push(d);
        let name = pair_meta[p]["pair"].as_str().unwrap();
        let cat = pair_meta[p]["category"].as_str().unwrap();
        // Baseline: cos of this pair's `before` against a different pair's `after`.
        let other = (p + 7) % num_pairs;  // arbitrary non-adjacent pair
        let base = dot(before, &embeds_norm[other * 2 + 1]);
        println!("{:<20} {:<14} {:>8.4}  {:>10.4}  {:>10.4}", name, cat, c, d, base);
    }

    // Aggregate: paired vs unpaired cosines.
    let mut unpaired = Vec::new();
    for i in 0..n_rows {
        for j in 0..n_rows {
            // unpaired = different pair (skip before/after of same pair, and self)
            if i == j { continue; }
            if i / 2 == j / 2 { continue; }  // same pair
            unpaired.push(dot(&embeds_norm[i], &embeds_norm[j]));
        }
    }

    let pair_mean = pair_cos.iter().sum::<f32>() / pair_cos.len() as f32;
    let pair_min = pair_cos.iter().cloned().fold(f32::INFINITY, f32::min);
    let pair_max = pair_cos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let unpaired_mean = unpaired.iter().sum::<f32>() / unpaired.len() as f32;
    let unpaired_max = unpaired.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n=== Summary ===");
    println!("Paired   cos: mean={pair_mean:.4}  min={pair_min:.4}  max={pair_max:.4}  (n={})", pair_cos.len());
    println!("Unpaired cos: mean={unpaired_mean:.4}  max={unpaired_max:.4}  (n={})", unpaired.len());
    println!("Separation:   {:+.4} (paired - unpaired mean)", pair_mean - unpaired_mean);
    let delta_mean = pair_delta.iter().sum::<f32>() / pair_delta.len() as f32;
    println!("Delta  norm: mean={delta_mean:.3}  (||after - before||_2 in latent space)");

    // How often is the correct "after" the top match for its "before"?
    let mut top1_correct = 0;
    let mut top3_correct = 0;
    for p in 0..num_pairs {
        let before = &embeds_norm[p * 2];
        // Rank all OTHER embeddings (including after of same pair).
        let mut sims: Vec<(usize, f32)> = (0..n_rows).filter(|&j| j != p * 2)
            .map(|j| (j, dot(before, &embeds_norm[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let correct = p * 2 + 1;
        if sims[0].0 == correct { top1_correct += 1; }
        if sims.iter().take(3).any(|(j, _)| *j == correct) { top3_correct += 1; }
    }
    println!("\nRetrieval test: does 'before' find its own 'after' as top match?");
    println!("  top-1 accuracy: {}/{} = {:.1}%", top1_correct, num_pairs, 100.0 * top1_correct as f32 / num_pairs as f32);
    println!("  top-3 accuracy: {}/{} = {:.1}%", top3_correct, num_pairs, 100.0 * top3_correct as f32 / num_pairs as f32);
}
