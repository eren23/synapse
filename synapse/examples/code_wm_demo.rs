//! Code WM end-to-end demo: load weights, encode tokens, predict next latent.
//!
//! Usage:
//!   cargo run --release --example code_wm_demo -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       [tests/fixtures/code_wm_reference_g8.safetensors]
//!
//! If a reference dump path is given, also reports cosine similarity + max-abs
//! diff against the PyTorch golden for seed 0.

use std::env;
use std::path::Path;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig, PoolMode};
use synapse_inference::weight_loading::{load_safetensors, RawTensor};

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0_f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors path");
    let config_path = args.next().expect("arg 2: config .json path");
    let goldens_path = args.next();

    println!("Loading config: {config_path}");
    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path)).unwrap();
    println!("  vocab_size={}, model_dim={}, num_heads={}, encoder_loops={}, predictor_depth={}, predictor_loops={}",
             cfg.vocab_size, cfg.model_dim, cfg.num_heads, cfg.encoder_loops, cfg.predictor_depth, cfg.predictor_loops);

    println!("Loading weights: {weights_path}");
    let tensors = load_safetensors(Path::new(&weights_path)).unwrap();
    println!("  {} tensors", tensors.len());

    let mut model = CodeWorldModel::from_config(&cfg);
    let stats = model.load_weights(tensors).unwrap();
    println!("  loaded={} skipped={}", stats.loaded, stats.skipped.len());
    // Cls variants (g8/g1b/g10/expa) have 47 tensors; attn variants add 5
    // state_encoder.attn_pool.* tensors for a total of 52.
    let expected = match cfg.pool_mode {
        PoolMode::Cls => 47,
        PoolMode::Attn => 52,
    };
    assert_eq!(stats.loaded, expected, "expected {expected} tensors loaded, got {}", stats.loaded);

    // Run encode + encode_action + predict.
    if let Some(goldens_path) = goldens_path {
        println!("\nRunning against goldens: {goldens_path}");
        let goldens: std::collections::HashMap<String, RawTensor> =
            load_safetensors(Path::new(&goldens_path)).unwrap();
        let tokens: Vec<i64> = goldens["seed0_input_tokens"]
            .data
            .iter()
            .map(|&v| v as i64)
            .collect();
        let action: &[f32] = &goldens["seed0_action_input"].data;

        let z_state = model.encode(&tokens);
        let z_state_ref = &goldens["seed0_encoder_final"].data;
        println!(
            "  encoder:   cos={:.10} max_abs={:.3e} norm={:.6}",
            cosine(&z_state, z_state_ref),
            max_abs(&z_state, z_state_ref),
            z_state.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        let z_action = model.encode_action(action);
        let z_action_ref = &goldens["seed0_action_final"].data;
        println!(
            "  action:    cos={:.10} max_abs={:.3e} norm={:.6}",
            cosine(&z_action, z_action_ref),
            max_abs(&z_action, z_action_ref),
            z_action.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Isolated predictor: use reference z_state/z_action as input
        // so predictor drift isn't polluted by earlier drift.
        let z_state_in = &goldens["seed0_pred_z_state"].data;
        let z_action_in = &goldens["seed0_pred_z_action"].data;
        let z_next = model.predict(z_state_in, z_action_in);
        let z_next_ref = &goldens["seed0_pred_final"].data;
        println!(
            "  predictor: cos={:.10} max_abs={:.3e} norm={:.6}",
            cosine(&z_next, z_next_ref),
            max_abs(&z_next, z_next_ref),
            z_next.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Full chained path
        let z_full = model.predict(&z_state, &z_action);
        println!(
            "  chained:   cos={:.10} max_abs={:.3e} norm={:.6}",
            cosine(&z_full, z_next_ref),
            max_abs(&z_full, z_next_ref),
            z_full.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
    } else {
        // Synthetic demo: run with a random token sequence.
        let tokens: Vec<i64> = (0..32).map(|i| (i * 7) as i64 % cfg.vocab_size as i64).collect();
        let action: Vec<f32> = (0..cfg.action_dim)
            .map(|i| if i % 3 == 0 { 1.0 } else if i % 3 == 1 { 0.0 } else { 0.5 })
            .collect();

        let z_state = model.encode(&tokens);
        let z_action = model.encode_action(&action);
        let z_next = model.predict(&z_state, &z_action);

        println!("\nSynthetic inference:");
        println!(
            "  z_state:   shape=[{}] norm={:.4}",
            z_state.len(),
            z_state.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        println!(
            "  z_action:  shape=[{}] norm={:.4}",
            z_action.len(),
            z_action.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        println!(
            "  z_next:    shape=[{}] norm={:.4}",
            z_next.len(),
            z_next.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
    }
}
