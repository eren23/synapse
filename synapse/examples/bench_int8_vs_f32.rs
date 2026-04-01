//! Benchmark: INT8 GEMV vs f32 GEMV at edge-model dimensions.
//!
//! Usage: cargo run -p synapse --release --example bench_int8_vs_f32

use std::time::Instant;
fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "zig-ffi")]
    { synapse_core::sgemm(m, n, k, a, b).expect("sgemm failed") }
    #[cfg(not(feature = "zig-ffi"))]
    { synapse_inference::ops::pure_rust_ops::matmul_t(a, b, m, k, n) }
}
use synapse_inference::quantization::QuantizedLinear;

fn bench_gemv(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    // f32 GEMV: x[m,k] @ W^T[k,n] → [m,n]
    let x = vec![0.1f32; m * k];
    let w = vec![0.02f32; n * k];
    let start = Instant::now();
    for _ in 0..iters {
        let _ = matmul_t(&x, &w, m, k, n);
    }
    let f32_us = start.elapsed().as_micros() as f64 / iters as f64;

    // INT8 GEMV (full path: quantize activations + INT8 kernel)
    let ql = QuantizedLinear::from_f32(&w, n, k);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ql.forward(&x, m);
    }
    let int8_us = start.elapsed().as_micros() as f64 / iters as f64;

    // INT8 with pre-quantized activations (amortized Q/K/V case)
    let (x_i8, scales_x) = synapse_core::quantize_per_channel_int8(&x, m, k)
        .expect("quantize failed");
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ql.forward_pre_quantized(&x_i8, &scales_x, m);
    }
    let int8_preq_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = f32_us / int8_us;
    let speedup_preq = f32_us / int8_preq_us;
    println!(
        "{label:>28}  f32={f32_us:>8.1}us  int8={int8_us:>8.1}us  int8_preq={int8_preq_us:>8.1}us  \
         speedup={speedup:.2}x  preq={speedup_preq:.2}x"
    );
}

fn main() {
    println!("INT8 vs f32 GEMV Benchmark");
    println!("==========================\n");

    println!("Predictor GEMV (M=1, single-token predict)\n");
    let iters = 10000;
    bench_gemv("64x64 (hybrid pred)", 1, 64, 64, iters);
    bench_gemv("192x192 (baseline pred)", 1, 192, 192, iters);
    bench_gemv("256x64 (hybrid FFN up)", 1, 256, 64, iters);
    bench_gemv("768x192 (baseline FFN up)", 1, 768, 192, iters);
    bench_gemv("1024x192 (pred QKV)", 1, 1024, 192, iters);
    bench_gemv("3072x192 (pred QKV fused)", 1, 3072, 192, iters);
    bench_gemv("2048x64 (hybrid pred FFN)", 1, 2048, 64, iters);
    bench_gemv("2048x192 (base pred FFN)", 1, 2048, 192, iters);

    println!("\nPredictor GEMV (M=3, 3-token predict sequence)\n");
    let iters = 5000;
    bench_gemv("3x64x64 (hybrid)", 3, 64, 64, iters);
    bench_gemv("3x192x192 (baseline)", 3, 192, 192, iters);
    bench_gemv("3x3072x192 (pred QKV)", 3, 3072, 192, iters);

    println!("\nEncoder GEMV (full sequence)\n");
    let iters = 200;
    bench_gemv("261x64x64 (hybrid enc)", 261, 64, 64, iters);
    bench_gemv("257x192x192 (base enc)", 257, 192, 192, iters);
    bench_gemv("261x256x64 (hyb FFN up)", 261, 256, 64, iters);
    bench_gemv("257x768x192 (base FFN)", 257, 768, 192, iters);
}
