//! World Model Rollout Benchmark
//!
//! Demonstrates real-time latent dynamics prediction.
//! Benchmarks the hot path: state→action→state prediction.
//!
//! Usage: cargo run --example world_model_rollout --release

use std::time::Instant;
use synapse_inference::models::vision::vit::ViTConfig;
use synapse_inference::models::{RealtimeRollout, WorldModel, WorldModelConfig};

fn main() {
    // Small dynamics model for real-time inference
    // Typical robotics world model: 128-dim latent, 2-4 dynamics layers
    let config = WorldModelConfig {
        encoder: ViTConfig {
            image_size: 64,
            patch_size: 8,
            channels: 3,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 4,
            intermediate_size: 256,
            num_classes: 0,
        },
        latent_dim: 128,
        dynamics_num_layers: 2,
        dynamics_num_heads: 4,
        dynamics_hidden_size: 128,
        action_dim: 4, // e.g., [dx, dy, dz, gripper]
    };

    println!("=== Synapse World Model Rollout Benchmark ===");
    println!("  Latent dim: {}", config.latent_dim);
    println!(
        "  Dynamics: {} layers, {} hidden",
        config.dynamics_num_layers, config.dynamics_hidden_size
    );
    println!("  Action dim: {}", config.action_dim);
    println!();

    // Build model with random weights (no pretrained checkpoint needed for benchmark)
    let mut model = WorldModel::from_config(&config);
    fill_random_weights(&mut model, &config);

    let mut rollout = RealtimeRollout::new(model);

    // Create a fake observation image
    let image: Vec<f32> = (0..64 * 64 * 3).map(|i| (i as f32 * 0.001).sin()).collect();

    // Benchmark: encode observation
    let start = Instant::now();
    rollout.reset(&image, 64, 64);
    let encode_time = start.elapsed();
    println!(
        "Encode observation (64x64): {:.1}ms",
        encode_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Initial state L2 norm: {:.4}",
        l2_norm(&rollout.state().embedding)
    );
    println!();

    // Benchmark: single dynamics step (THE HOT PATH)
    let action = vec![0.1f32, -0.2, 0.05, 1.0]; // sample action
    let warmup = 10;
    let iterations = 100;

    // Warmup
    for _ in 0..warmup {
        rollout.step(&action);
    }
    rollout.reset(&image, 64, 64); // reset after warmup

    // Benchmark single step
    let start = Instant::now();
    for _ in 0..iterations {
        rollout.step(&action);
    }
    let total = start.elapsed();
    let per_step = total.as_secs_f64() * 1000.0 / iterations as f64;
    println!("Dynamics step (state→action→state):");
    println!("  {:.2}ms per step ({} iterations)", per_step, iterations);
    println!("  {:.0} steps/sec", 1000.0 / per_step);
    if per_step < 10.0 {
        println!("  ✓ Under 10ms target — suitable for real-time control");
    } else if per_step < 50.0 {
        println!("  ~ Under 50ms — suitable for planning, not real-time control");
    } else {
        println!("  ✗ Over 50ms — needs optimization for real-time use");
    }
    println!();

    // Benchmark: multi-step planning (batch rollout without advancing state)
    rollout.reset(&image, 64, 64);
    let plan_steps = 50;
    let actions: Vec<Vec<f32>> = (0..plan_steps)
        .map(|i| {
            vec![
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.15).cos(),
                0.05,
                if i % 10 < 5 { 1.0 } else { 0.0 },
            ]
        })
        .collect();

    let start = Instant::now();
    let trajectory = rollout.plan(&actions);
    let plan_time = start.elapsed();
    println!("Planning rollout ({} steps):", plan_steps);
    println!(
        "  {:.1}ms total ({:.2}ms per step)",
        plan_time.as_secs_f64() * 1000.0,
        plan_time.as_secs_f64() * 1000.0 / plan_steps as f64
    );
    println!("  Trajectory length: {}", trajectory.len());
    println!(
        "  Final state L2 norm: {:.4}",
        l2_norm(&trajectory.last().unwrap().embedding)
    );
    println!();

    // Show state doesn't advance after plan()
    println!(
        "After plan(): rollout.steps() = {} (unchanged)",
        rollout.steps()
    );
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn fill_random_weights(model: &mut WorldModel, config: &WorldModelConfig) {
    use synapse_inference::weight_loading::AlignedBuffer;

    let h = config.encoder.hidden_size;
    let lat = config.latent_dim;
    let dyn_h = config.dynamics_hidden_size;
    let act = config.action_dim;
    let dyn_inter = config.dynamics_intermediate_size();

    // State projection [encoder_dim, latent_dim]
    model.state_proj = AlignedBuffer::from_vec(gen(h * lat, 1));
    // Action embedding [action_dim, dynamics_hidden]
    model.action_embed = AlignedBuffer::from_vec(gen(act * dyn_h, 2));
    // Output projection [dynamics_hidden, latent_dim]
    model.output_proj = AlignedBuffer::from_vec(gen(dyn_h * lat, 3));
    // Dynamics norm
    model.dynamics_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; dyn_h]);

    // Dynamics layers
    for (i, layer) in model.dynamics_layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.w_q = AlignedBuffer::from_vec(gen(dyn_h * dyn_h, s + 1));
        layer.w_k = AlignedBuffer::from_vec(gen(dyn_h * dyn_h, s + 2));
        layer.w_v = AlignedBuffer::from_vec(gen(dyn_h * dyn_h, s + 3));
        layer.w_o = AlignedBuffer::from_vec(gen(dyn_h * dyn_h, s + 4));
        layer.ffn_up = AlignedBuffer::from_vec(gen(dyn_inter * dyn_h, s + 5));
        layer.ffn_down = AlignedBuffer::from_vec(gen(dyn_h * dyn_inter, s + 6));
        layer.attn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; dyn_h]);
        layer.ffn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; dyn_h]);
    }

    // Encoder — fill with small random weights
    let enc_h = config.encoder.hidden_size;
    let patch_dim = config.encoder.patch_size * config.encoder.patch_size * config.encoder.channels;
    let num_patches = (config.encoder.image_size / config.encoder.patch_size).pow(2);
    model.encoder.patch_proj = AlignedBuffer::from_vec(gen(enc_h * patch_dim, 10));
    model.encoder.cls_token = AlignedBuffer::from_vec(gen(enc_h, 11));
    model.encoder.pos_embed = AlignedBuffer::from_vec(gen((num_patches + 1) * enc_h, 12));
    model.encoder.final_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; enc_h]);

    let enc_inter = config.encoder.intermediate_size;
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 1000;
        layer.w_q = AlignedBuffer::from_vec(gen(enc_h * enc_h, s + 1));
        layer.w_k = AlignedBuffer::from_vec(gen(enc_h * enc_h, s + 2));
        layer.w_v = AlignedBuffer::from_vec(gen(enc_h * enc_h, s + 3));
        layer.w_o = AlignedBuffer::from_vec(gen(enc_h * enc_h, s + 4));
        layer.ffn_up = AlignedBuffer::from_vec(gen(enc_inter * enc_h, s + 5));
        layer.ffn_down = AlignedBuffer::from_vec(gen(enc_h * enc_inter, s + 6));
        layer.attn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; enc_h]);
        layer.ffn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; enc_h]);
    }
}

fn gen(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.1 - 0.05
        })
        .collect()
}
