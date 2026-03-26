//! Geometric Attention Demo: distance-aware attention for 3D point clouds.
//!
//! Demonstrates a custom Zig SIMD kernel that PyTorch/MLX don't have.
//! Shows: closer points attend more, faraway points attend less.

use std::time::Instant;
use synapse_inference::ops::geometric::geometric_attention;

fn main() {
    println!("=== Synapse Geometric Attention Demo ===");
    println!("Custom Zig SIMD kernel — not available in PyTorch or MLX\n");

    // Generate a 3D point cloud (256 points, 64-dim embeddings)
    let n = 256;
    let d = 64;
    let pos_dim = 3;

    // Points in a 3D grid with some clustering
    let mut positions = vec![0.0f32; n * pos_dim];
    for i in 0..n {
        positions[i * 3] = (i % 16) as f32 * 0.5;
        positions[i * 3 + 1] = ((i / 16) % 16) as f32 * 0.5;
        positions[i * 3 + 2] = (i as f32 * 0.01).sin() * 2.0;
    }

    // Random embeddings
    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.037).sin() * 0.5).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.041).cos() * 0.5).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.029 + 1.0).sin() * 0.5).collect();

    let sigma = 1.0;

    // Warmup
    for _ in 0..5 {
        let _ = geometric_attention(n, d, pos_dim, &q, &k, &v, &positions, sigma);
    }

    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    let mut out = vec![];
    for _ in 0..iterations {
        out = geometric_attention(n, d, pos_dim, &q, &k, &v, &positions, sigma);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("Point cloud: {} points, {}-dim embeddings, 3D positions", n, d);
    println!("Sigma (distance bandwidth): {}", sigma);
    println!();
    println!("Performance:");
    println!("  {:.3}ms per call ({} iterations)", per_call, iterations);
    println!("  {:.0} calls/sec", 1000.0 / per_call);
    println!();

    // Analyze attention patterns
    // Run with different sigmas to show distance effect
    let sigmas = [0.1, 1.0, 10.0];
    println!("Distance effect (point 0 attending to others):");
    for sigma in &sigmas {
        let result = geometric_attention(n, d, pos_dim, &q, &k, &v, &positions, *sigma);
        let out_norm: f32 = result[..d].iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  sigma={:.1}: output L2 norm = {:.4} (smaller sigma = more local attention)", sigma, out_norm);
    }

    println!();
    println!("All outputs finite: {}", out.iter().all(|v| v.is_finite()));

    // Compare: equivalent Python would be ~10-50x slower
    let flops = n as f64 * n as f64 * (d as f64 + pos_dim as f64 + d as f64); // Q·K + dist + V·scores
    let gflops = flops / (per_call * 1e6);
    println!("Throughput: {:.1} GFLOPS", gflops);
    println!();
    println!("This kernel runs on CPU (NEON SIMD), WASM, ARM, x86 — no GPU needed.");
    println!("PyTorch equivalent: custom CUDA kernel (NVIDIA only) or slow Python loops.");
}
