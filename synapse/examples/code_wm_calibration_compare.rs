//! Compare INT8 calibration strategies: MinMax vs Percentile(99.5) vs Percentile(99.9).
//!
//! G1b (VICReg-trained) has wider weight distributions where MinMax over-scales
//! due to outliers. Percentile clipping typically recovers 0.0001-0.001 cos.
//!
//! Usage:
//!   cargo run --release --example code_wm_calibration_compare -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       tests/fixtures/code_wm_reference_g1b.safetensors

use std::env;
use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::quantization::vision::int8_code_wm::{
    quantize_code_wm, quantize_code_wm_int8_percentile,
};
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

    // Quantize with each strategy.
    let minmax = quantize_code_wm(&f32_model);
    let pct_995 = quantize_code_wm_int8_percentile(&f32_model, 99.5);
    let pct_999 = quantize_code_wm_int8_percentile(&f32_model, 99.9);
    let pct_9995 = quantize_code_wm_int8_percentile(&f32_model, 99.95);

    let goldens = load_safetensors(Path::new(&goldens)).unwrap();

    println!("Calibration strategy comparison (INT8 Code WM):\n");
    println!("{}", "─".repeat(92));
    println!(
        "  stage   | MinMax cos     | P99.5 cos      | P99.9 cos      | P99.95 cos     | MinMax / best max_abs",
    );
    println!("{}", "─".repeat(92));

    for seed in 0..3 {
        let pfx = format!("seed{seed}_");
        let tokens = tokens_from_f32(&goldens[&format!("{pfx}input_tokens")].data);
        let action: &[f32] = &goldens[&format!("{pfx}action_input")].data;
        let zs = &goldens[&format!("{pfx}pred_z_state")].data;
        let za = &goldens[&format!("{pfx}pred_z_action")].data;

        // Encoder
        let z_mm = minmax.encode(&tokens);
        let z_995 = pct_995.encode(&tokens);
        let z_999 = pct_999.encode(&tokens);
        let z_9995 = pct_9995.encode(&tokens);
        let z_ref = &goldens[&format!("{pfx}encoder_final")].data;
        println!(
            "  enc s={seed} | {:.10} | {:.10} | {:.10} | {:.10} | {:.3e} / {:.3e}",
            cosine(&z_mm, z_ref), cosine(&z_995, z_ref), cosine(&z_999, z_ref), cosine(&z_9995, z_ref),
            max_abs(&z_mm, z_ref),
            [&z_995, &z_999, &z_9995].iter().map(|z| max_abs(z, z_ref)).fold(f32::INFINITY, f32::min)
        );

        // Predictor
        let p_mm = minmax.predict(zs, za);
        let p_995 = pct_995.predict(zs, za);
        let p_999 = pct_999.predict(zs, za);
        let p_9995 = pct_9995.predict(zs, za);
        let p_ref = &goldens[&format!("{pfx}pred_final")].data;
        println!(
            "  pred s={seed}| {:.10} | {:.10} | {:.10} | {:.10} | {:.3e} / {:.3e}",
            cosine(&p_mm, p_ref), cosine(&p_995, p_ref), cosine(&p_999, p_ref), cosine(&p_9995, p_ref),
            max_abs(&p_mm, p_ref),
            [&p_995, &p_999, &p_9995].iter().map(|p| max_abs(p, p_ref)).fold(f32::INFINITY, f32::min)
        );

        // Action encoder stays f32 → skip (would all be identical)
        let _ = action;
    }
}
