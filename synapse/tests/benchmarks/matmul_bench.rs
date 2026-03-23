//! Matrix multiplication benchmark at various sizes.

use std::time::Instant;
use synapse_autograd::Tensor;

fn bench_matmul(m: usize, k: usize, n: usize, iterations: usize) -> f64 {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect();
    let a = Tensor::new(a_data, vec![m, k]);
    let b = Tensor::new(b_data, vec![k, n]);

    // Warmup
    for _ in 0..3 {
        let _ = a.matmul(&b);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }
    let elapsed = start.elapsed();

    let flops_per_iter = 2.0 * m as f64 * k as f64 * n as f64;
    let total_flops = flops_per_iter * iterations as f64;
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;

    eprintln!(
        "  matmul [{} x {}] @ [{} x {}]: {:.3}ms/iter, {:.2} GFLOPS",
        m, k, k, n,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        gflops
    );

    gflops
}

#[test]
fn matmul_small_batch() {
    eprintln!("Matrix multiplication benchmarks:");
    let gflops = bench_matmul(64, 256, 128, 100);
    assert!(gflops > 0.0, "GFLOPS should be positive");
}

#[test]
fn matmul_medium() {
    let gflops = bench_matmul(128, 512, 256, 50);
    assert!(gflops > 0.0);
}

#[test]
fn matmul_large() {
    let gflops = bench_matmul(256, 1024, 512, 10);
    assert!(gflops > 0.0);
}

#[test]
fn matmul_correctness() {
    // Verify matmul produces correct results
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = a.matmul(&b);

    assert_eq!(c.shape, vec![2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert!((c.data[0] - 58.0).abs() < 1e-4);
    assert!((c.data[1] - 64.0).abs() < 1e-4);
    assert!((c.data[2] - 139.0).abs() < 1e-4);
    assert!((c.data[3] - 154.0).abs() < 1e-4);
}
