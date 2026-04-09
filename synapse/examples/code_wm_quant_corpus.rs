//! Real-code quantization benchmark for Code WM.
//!
//! Loads a Code WM variant, quantizes it to INT8 / Q4 / Q4-full, and encodes
//! every file in a pre-tokenized Python corpus with all four precisions. Reports
//! per-file cosine drift (f32 → quantized) distribution across the corpus plus
//! per-precision encode latency.
//!
//! Rationale: the existing `code_wm_{int8,q4}_compare.rs` examples compare
//! quantized Rust output against a PyTorch reference dump generated from
//! **3 synthetic random-token sequences**. That's a small sample on an easier
//! distribution than real Python code. This example uses the existing
//! 500-file corpus fixture (`file_index.safetensors`, same one consumed by
//! `code_wm_corpus_retrieve.rs`) and does a Rust-only comparison so the
//! ~5e-7 Rust↔PyTorch baseline drift never enters the measurement.
//!
//! Usage:
//!   cargo run --release --example code_wm_quant_corpus -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       tests/fixtures/file_index.safetensors

use std::env;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::quantization::vision::int8_code_wm::quantize_code_wm;
use synapse_inference::quantization::vision::q4_code_wm::quantize_code_wm_q4;
use synapse_inference::quantization::vision::q4_code_wm_full::quantize_code_wm_q4_full;
use synapse_inference::weight_loading::load_safetensors;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

/// Nearest-rank percentile. Input is mutated (sorted) in place.
fn percentile(xs: &mut [f32], p: f64) -> f32 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().copied().sum::<f32>() / (xs.len() as f32).max(1.0)
}

fn frac_below(xs: &[f32], threshold: f32) -> f32 {
    let n = xs.iter().filter(|&&x| x < threshold).count();
    n as f32 / (xs.len() as f32).max(1.0)
}

fn print_stats(label: &str, cos: Vec<f32>) {
    let n = cos.len();
    let avg = mean(&cos);
    let p05 = percentile(&mut cos.clone(), 0.05);
    let p50 = percentile(&mut cos.clone(), 0.50);
    let p95 = percentile(&mut cos.clone(), 0.95);
    let min = cos.iter().copied().fold(f32::INFINITY, f32::min);
    let max = cos.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let pct_below_999 = frac_below(&cos, 0.999) * 100.0;
    let pct_below_99 = frac_below(&cos, 0.99) * 100.0;
    println!(
        "  {label:<10} n={n:>4}  min={min:.6}  p5={p05:.6}  p50={p50:.6}  p95={p95:.6}  max={max:.6}  mean={avg:.6}  <0.999={pct_below_999:>5.2}%  <0.99={pct_below_99:>5.2}%"
    );
}

fn encode_all(
    label: &str,
    encode: impl Fn(&[i64]) -> Vec<f32>,
    tokens_data: &[f32],
    n: usize,
    max_len: usize,
) -> (Vec<Vec<f32>>, f64) {
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &tokens_data[i * max_len..(i + 1) * max_len];
        let toks: Vec<i64> = row.iter().map(|&v| v as i64).collect();
        out.push(encode(&toks));
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_file_ms = total_ms / n as f64;
    println!("  {label:<10} encoded {n} files in {total_ms:>8.1} ms  ({per_file_ms:>6.3} ms/file)");
    (out, per_file_ms)
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors");
    let config_path = args.next().expect("arg 2: config .json");
    let index_path = args.next().expect("arg 3: file_index .safetensors");

    println!("Loading config: {config_path}");
    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path))
        .unwrap_or_else(|e| panic!("config load failed: {e}"));
    println!("  pool_mode={:?}", cfg.pool_mode);

    println!("Loading f32 weights: {weights_path}");
    let tensors = load_safetensors(Path::new(&weights_path))
        .unwrap_or_else(|e| panic!("safetensors load failed: {e:?}"));
    let mut f32_model = CodeWorldModel::from_config(&cfg);
    f32_model
        .load_weights(tensors)
        .unwrap_or_else(|e| panic!("load_weights failed: {e:?}"));

    println!("Quantizing to INT8 / Q4 / Q4-full ...");
    let int8_model = quantize_code_wm(&f32_model);
    let q4_model = quantize_code_wm_q4(&f32_model);
    let q4f_model = quantize_code_wm_q4_full(&f32_model);

    println!("  f32:     {:>8.1} KB", 2984.0);
    println!(
        "  INT8:    {:>8.1} KB",
        int8_model.memory_bytes() as f64 / 1024.0
    );
    println!(
        "  Q4:      {:>8.1} KB",
        q4_model.memory_bytes() as f64 / 1024.0
    );
    println!(
        "  Q4-full: {:>8.1} KB",
        q4f_model.memory_bytes() as f64 / 1024.0
    );

    println!("\nLoading corpus: {index_path}");
    let index = load_safetensors(Path::new(&index_path))
        .unwrap_or_else(|e| panic!("corpus load failed: {e:?}"));
    let tokens_t = &index["tokens"];
    let n = tokens_t.shape[0];
    let max_len = tokens_t.shape[1];
    println!("Corpus: {n} files × {max_len} tokens");

    // Encode all files with each precision. Rust-only, no PyTorch — the
    // only thing these 4 runs disagree on is the quantization, so cosine(f32, q)
    // is a clean measurement of quantization error alone.
    println!("\nEncoding {n} files with 4 precisions ...");
    let (z_f32, f32_ms) = encode_all("f32", |t| f32_model.encode(t), &tokens_t.data, n, max_len);
    let (z_int8, int8_ms) = encode_all("INT8", |t| int8_model.encode(t), &tokens_t.data, n, max_len);
    let (z_q4, q4_ms) = encode_all("Q4", |t| q4_model.encode(t), &tokens_t.data, n, max_len);
    let (z_q4f, q4f_ms) = encode_all("Q4-full", |t| q4f_model.encode(t), &tokens_t.data, n, max_len);

    // Per-file cosine drift f32 → quantized.
    let int8_cos: Vec<f32> = z_f32.iter().zip(&z_int8).map(|(a, b)| cosine(a, b)).collect();
    let q4_cos: Vec<f32> = z_f32.iter().zip(&z_q4).map(|(a, b)| cosine(a, b)).collect();
    let q4f_cos: Vec<f32> = z_f32.iter().zip(&z_q4f).map(|(a, b)| cosine(a, b)).collect();

    println!("\n=== Cosine drift f32 → quantized (across {n} files) ===");
    print_stats("INT8", int8_cos);
    print_stats("Q4", q4_cos);
    print_stats("Q4-full", q4f_cos);

    println!("\n=== Latency summary ===");
    println!("  f32:     {f32_ms:>6.3} ms/file  (baseline)");
    println!(
        "  INT8:    {int8_ms:>6.3} ms/file  ({:.2}× f32)",
        f32_ms / int8_ms
    );
    println!(
        "  Q4:      {q4_ms:>6.3} ms/file  ({:.2}× f32)",
        f32_ms / q4_ms
    );
    println!(
        "  Q4-full: {q4f_ms:>6.3} ms/file  ({:.2}× f32)",
        f32_ms / q4f_ms
    );
}
