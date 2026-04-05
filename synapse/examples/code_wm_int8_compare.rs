//! Compare INT8-quantized Code WM against f32 reference.
//!
//! Loads the f32 model, quantizes it to INT8, then compares encoder/action/
//! predictor outputs against the PyTorch goldens. Reports cosine similarity,
//! max_abs_diff, and total in-memory bytes for each variant.
//!
//! Usage:
//!   cargo run --release --example code_wm_int8_compare -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       tests/fixtures/code_wm_reference_g8.safetensors

use std::env;
use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::quantization::vision::int8_code_wm::quantize_code_wm;
use synapse_inference::weight_loading::load_safetensors;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn tokens_from_f32(buf: &[f32]) -> Vec<i64> {
    buf.iter().map(|&v| v as i64).collect()
}

fn bytes_f32(m: &CodeWorldModel) -> usize {
    let cfg = &m.config;
    // Embedding + CLS + PE
    let mut total = 4 * (cfg.vocab_size * cfg.model_dim + cfg.model_dim + (cfg.max_seq_len + 1) * cfg.model_dim);
    // 1 encoder block + predictor_depth blocks
    let block = {
        let d = cfg.model_dim;
        let mlp = cfg.mlp_hidden;
        // norm1, norm2: 2 × (D + D) f32
        // attn in_proj: 3D×D weight + 3D bias
        // attn out_proj: D×D weight + D bias
        // mlp up: mlp×D weight + mlp bias
        // mlp down: D×mlp weight + D bias
        4 * (2 * (d + d) + 3 * d * d + 3 * d + d * d + d + mlp * d + mlp + d * mlp + d)
    };
    total += block * (1 + cfg.predictor_depth);
    // encoder final norm, predictor final norm: 2 × (D+D)
    total += 4 * (2 * (cfg.model_dim + cfg.model_dim));
    // action encoder: fc1 (D × action_dim + D), fc2 (D×D + D)
    total += 4 * (cfg.model_dim * cfg.action_dim + cfg.model_dim + cfg.model_dim * cfg.model_dim + cfg.model_dim);
    total
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors");
    let config_path = args.next().expect("arg 2: config .json");
    let goldens_path = args.next().expect("arg 3: reference dump .safetensors");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path)).unwrap();
    let tensors = load_safetensors(Path::new(&weights_path)).unwrap();
    let mut f32_model = CodeWorldModel::from_config(&cfg);
    f32_model.load_weights(tensors).unwrap();

    let int8_model = quantize_code_wm(&f32_model);
    let f32_bytes = bytes_f32(&f32_model);
    let int8_bytes = int8_model.memory_bytes();

    println!("Code WM size comparison:");
    println!("  f32:  {:.1} KB", f32_bytes as f64 / 1024.0);
    println!("  INT8: {:.1} KB", int8_bytes as f64 / 1024.0);
    println!("  ratio: {:.2}x smaller\n", f32_bytes as f64 / int8_bytes as f64);

    let goldens = load_safetensors(Path::new(&goldens_path)).unwrap();

    println!("Quality vs PyTorch goldens (3 seeds):");
    println!("  stage      | f32 cos        | INT8 cos       | f32 max_abs   | INT8 max_abs");
    println!("  -----------|----------------|----------------|---------------|--------------");
    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens = tokens_from_f32(&goldens[&format!("{pfx}input_tokens")].data);
        let action: &[f32] = &goldens[&format!("{pfx}action_input")].data;
        let zs_ref = &goldens[&format!("{pfx}pred_z_state")].data;
        let za_ref = &goldens[&format!("{pfx}pred_z_action")].data;

        // Encoder
        let z_f32 = f32_model.encode(&tokens);
        let z_i8 = int8_model.encode(&tokens);
        let z_ref = &goldens[&format!("{pfx}encoder_final")].data;
        println!(
            "  enc s={}    | {:.10} | {:.10} | {:.3e}     | {:.3e}",
            seed,
            cosine(&z_f32, z_ref),
            cosine(&z_i8, z_ref),
            max_abs(&z_f32, z_ref),
            max_abs(&z_i8, z_ref)
        );

        // Action
        let a_f32 = f32_model.encode_action(action);
        let a_i8 = int8_model.encode_action(action);
        let a_ref = &goldens[&format!("{pfx}action_final")].data;
        println!(
            "  act s={}    | {:.10} | {:.10} | {:.3e}     | {:.3e}",
            seed,
            cosine(&a_f32, a_ref),
            cosine(&a_i8, a_ref),
            max_abs(&a_f32, a_ref),
            max_abs(&a_i8, a_ref)
        );

        // Predictor (using reference z_state/z_action to isolate predictor drift)
        let p_f32 = f32_model.predict(zs_ref, za_ref);
        let p_i8 = int8_model.predict(zs_ref, za_ref);
        let p_ref = &goldens[&format!("{pfx}pred_final")].data;
        println!(
            "  pred s={}   | {:.10} | {:.10} | {:.3e}     | {:.3e}",
            seed,
            cosine(&p_f32, p_ref),
            cosine(&p_i8, p_ref),
            max_abs(&p_f32, p_ref),
            max_abs(&p_i8, p_ref)
        );
    }
}
