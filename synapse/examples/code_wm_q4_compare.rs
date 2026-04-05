//! Compare f32, INT8, and Q4 quantized Code WM side-by-side.
//!
//! Usage:
//!   cargo run --release --example code_wm_q4_compare -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       tests/fixtures/code_wm_reference_g8.safetensors

use std::env;
use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::quantization::vision::int8_code_wm::quantize_code_wm;
use synapse_inference::quantization::vision::q4_code_wm::quantize_code_wm_q4;
use synapse_inference::weight_loading::load_safetensors;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    for i in 0..a.len() { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0f32, f32::max)
}

fn tokens_from_f32(buf: &[f32]) -> Vec<i64> {
    buf.iter().map(|&v| v as i64).collect()
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let goldens = args.next().expect("arg 3: reference dump .safetensors");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut f32_model = CodeWorldModel::from_config(&cfg);
    f32_model.load_weights(tensors).unwrap();

    let int8_model = quantize_code_wm(&f32_model);
    let q4_model = quantize_code_wm_q4(&f32_model);

    // Sizes
    let int8_bytes = int8_model.memory_bytes();
    let q4_bytes = q4_model.memory_bytes();
    // Approximate f32 size from the checkpoint params
    let f32_bytes = (cfg.vocab_size * cfg.model_dim
        + cfg.model_dim  // cls
        + (cfg.max_seq_len + 1) * cfg.model_dim  // PE
        + 4 * 2 * cfg.model_dim  // final norms
        + 3 * (
            // per block: norm×2, attn_in_proj + bias, attn_out_proj + bias, mlp_up + bias, mlp_down + bias
            4*cfg.model_dim + 3*cfg.model_dim*cfg.model_dim + 3*cfg.model_dim
                + cfg.model_dim*cfg.model_dim + cfg.model_dim
                + cfg.mlp_hidden*cfg.model_dim + cfg.mlp_hidden
                + cfg.model_dim*cfg.mlp_hidden + cfg.model_dim
        )
        + cfg.model_dim*cfg.action_dim + cfg.model_dim  // action fc1
        + cfg.model_dim*cfg.model_dim + cfg.model_dim  // action fc2
    ) * 4;

    println!("Code WM size comparison:");
    println!("  f32:   {:>6.1} KB        (baseline)", f32_bytes as f64 / 1024.0);
    println!("  INT8:  {:>6.1} KB   {:.2}x smaller", int8_bytes as f64 / 1024.0, f32_bytes as f64 / int8_bytes as f64);
    println!("  Q4:    {:>6.1} KB   {:.2}x smaller", q4_bytes as f64 / 1024.0, f32_bytes as f64 / q4_bytes as f64);

    let goldens = load_safetensors(Path::new(&goldens)).unwrap();

    println!("\nQuality vs PyTorch goldens (seed 0):");
    println!("{}", "─".repeat(86));
    println!("  stage      | f32 cos        | INT8 cos       | Q4 cos         | Q4 max_abs  ");
    println!("{}", "─".repeat(86));

    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens = tokens_from_f32(&goldens[&format!("{pfx}input_tokens")].data);
        let action: &[f32] = &goldens[&format!("{pfx}action_input")].data;
        let zs = &goldens[&format!("{pfx}pred_z_state")].data;
        let za = &goldens[&format!("{pfx}pred_z_action")].data;

        // Encoder
        let z_f32 = f32_model.encode(&tokens);
        let z_i8 = int8_model.encode(&tokens);
        let z_q4 = q4_model.encode(&tokens);
        let z_ref = &goldens[&format!("{pfx}encoder_final")].data;
        println!(
            "  enc s={seed}    | {:.10} | {:.10} | {:.10} | {:.3e}",
            cosine(&z_f32, z_ref), cosine(&z_i8, z_ref), cosine(&z_q4, z_ref), max_abs(&z_q4, z_ref)
        );

        // Action
        let a_f32 = f32_model.encode_action(action);
        let a_i8 = int8_model.encode_action(action);
        let a_q4 = q4_model.encode_action(action);
        let a_ref = &goldens[&format!("{pfx}action_final")].data;
        println!(
            "  act s={seed}    | {:.10} | {:.10} | {:.10} | {:.3e}",
            cosine(&a_f32, a_ref), cosine(&a_i8, a_ref), cosine(&a_q4, a_ref), max_abs(&a_q4, a_ref)
        );

        // Predictor
        let p_f32 = f32_model.predict(zs, za);
        let p_i8 = int8_model.predict(zs, za);
        let p_q4 = q4_model.predict(zs, za);
        let p_ref = &goldens[&format!("{pfx}pred_final")].data;
        println!(
            "  pred s={seed}   | {:.10} | {:.10} | {:.10} | {:.3e}",
            cosine(&p_f32, p_ref), cosine(&p_i8, p_ref), cosine(&p_q4, p_ref), max_abs(&p_q4, p_ref)
        );
    }
}
