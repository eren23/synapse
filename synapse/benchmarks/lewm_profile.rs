use std::time::Instant;
use synapse_inference::models::vision::lewm::{LeWMConfig, LeWorldModel};

fn main() {
    let config = LeWMConfig::pusht();
    let weights_path = "/tmp/lewm-pusht/pusht/lejepa_weights.safetensors";
    let weights = synapse_inference::weight_loading::load_safetensors(
        std::path::Path::new(weights_path)
    ).expect("load");
    let mut model = LeWorldModel::from_config(&config);
    let _ = model.load_weights(weights);

    let image = vec![0.5f32; 224 * 224 * 3];
    let z = model.encode(&image, 224, 224);
    let action = vec![0.1f32; 10];

    // Profile: what fraction of time is matmul_t vs everything else?
    // Test 1: Just the matmul sizes used in LEWM
    let h = 192usize;
    let inner = 1024usize;
    let inter = 2048usize;

    let a3xh = vec![0.1f32; 3 * h];      // [3, 192]
    let w_qkv = vec![0.1f32; 3*inner * h]; // [3072, 192]
    let w_up = vec![0.1f32; inter * h];    // [2048, 192]
    let w_down = vec![0.1f32; h * inter];  // [192, 2048]
    let w_out = vec![0.1f32; h * inner];   // [192, 1024]
    let a3xi = vec![0.1f32; 3 * inner];    // [3, 1024]
    let a3xinter = vec![0.1f32; 3 * inter]; // [3, 2048]

    // Warmup
    for _ in 0..5 {
        let _ = synapse_inference::ops::matmul::matmul_t(&a3xh, &w_qkv, 3, h, 3*inner);
    }

    let runs = 100;
    
    println!("═══════════════════════════════════════════════════");
    println!("  LEWM Matmul Profile (seq_len=3)");
    println!("═══════════════════════════════════════════════════\n");

    // Individual matmul timings
    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::matmul::matmul_t(&a3xh, &w_qkv, 3, h, 3*inner);
    }
    let qkv_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::matmul::matmul_t(&a3xi, &w_out, 3, inner, h);
    }
    let out_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::matmul::matmul_t(&a3xh, &w_up, 3, h, inter);
    }
    let up_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::matmul::matmul_t(&a3xinter, &w_down, 3, inter, h);
    }
    let down_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    let per_layer_us = qkv_us + out_us + up_us + down_us;
    let total_6_layers = per_layer_us * 6.0;

    println!("Per matmul (Zig SIMD, {runs} runs avg):");
    println!("  QKV  [3,192]×[3072,192]^T: {qkv_us:.0}µs");
    println!("  Out  [3,1024]×[192,1024]^T: {out_us:.0}µs");
    println!("  Up   [3,192]×[2048,192]^T:  {up_us:.0}µs");
    println!("  Down [3,2048]×[192,2048]^T: {down_us:.0}µs");
    println!("  Per layer (4 matmuls):       {per_layer_us:.0}µs");
    println!("  6 layers total:              {total_6_layers:.0}µs = {:.1}ms", total_6_layers / 1000.0);

    // Compare: pure-Rust naive matmul for same sizes
    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::pure_rust_ops::matmul_t(&a3xh, &w_qkv, 3, h, 3*inner);
    }
    let qkv_pure_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    let t = Instant::now();
    for _ in 0..runs {
        let _ = synapse_inference::ops::pure_rust_ops::matmul_t(&a3xh, &w_up, 3, h, inter);
    }
    let up_pure_us = t.elapsed().as_nanos() as f64 / runs as f64 / 1000.0;

    println!("\nPure Rust naive (no SIMD, no FFI):");
    println!("  QKV  [3,192]×[3072,192]^T: {qkv_pure_us:.0}µs ({:.1}x vs Zig)", qkv_us / qkv_pure_us);
    println!("  Up   [3,192]×[2048,192]^T:  {up_pure_us:.0}µs ({:.1}x vs Zig)", up_us / up_pure_us);

    // Full predict_next timing breakdown
    println!("\n─────────────────────────────────────────────────");
    let _ = model.predict_next(&z, &action); // warmup
    let t = Instant::now();
    for _ in 0..20 { let _ = model.predict_next(&z, &action); }
    let predict_ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
    println!("Full predict_next: {predict_ms:.1}ms");
    println!("Matmul accounts for: {:.1}ms ({:.0}% of total)", total_6_layers / 1000.0, total_6_layers / 1000.0 / predict_ms * 100.0);
}
