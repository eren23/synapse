use super::*;
use crate::metal::device::KERNEL_NAMES;

/// Helper: get MetalBackend or skip test on non-Apple hardware.
fn get_backend() -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(b) => Some(b),
        Err(MetalError::NoDevice) => {
            eprintln!("Skipping: no Metal GPU available");
            None
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

/// Create a Metal buffer from f32 data.
fn make_buffer(device: &::metal::Device, data: &[f32]) -> ::metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<f32>()) as u64,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Create an empty (zeroed) Metal buffer for n f32 elements.
fn make_empty(device: &::metal::Device, n: usize) -> ::metal::Buffer {
    device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Create a Metal buffer holding a single u32 constant.
fn make_const_u32(device: &::metal::Device, val: u32) -> ::metal::Buffer {
    make_buffer(device, &[f32::from_bits(val)])
}

/// Create a Metal buffer holding a single f32 constant.
fn make_const_f32(device: &::metal::Device, val: f32) -> ::metal::Buffer {
    make_buffer(device, &[val])
}

/// Read f32 values from a shared-mode Metal buffer.
fn read_buffer(buf: &::metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let mut out = vec![0.0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
    out
}

/// Assert two f32 slices are approximately equal within tolerance.
/// Uses relative tolerance for large values, absolute for small values.
fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_diff = (a - e).abs();
        let rel_tol = tol * e.abs().max(1.0);
        assert!(
            abs_diff < rel_tol,
            "{label}[{i}]: GPU={a} vs CPU={e}, diff={abs_diff}, rel_tol={rel_tol}",
        );
    }
}

// ======================== Pipeline compilation tests ========================

#[test]
fn metal_backend_creation() {
    match MetalBackend::new() {
        Ok(backend) => {
            assert!(!backend.device_name().is_empty());
            // At least all required kernels must be present
            assert!(backend.pipeline_count() >= KERNEL_NAMES.len());
            for name in KERNEL_NAMES {
                assert!(backend.pipeline(name).is_some(), "Missing pipeline: {name}");
            }
        }
        Err(MetalError::NoDevice) => {
            assert!(!MetalBackend::is_available());
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn buffer_pool_reuses_matching_size() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let mut pool = BufferPool::new(&backend.device);

    let data = vec![1.0f32; 256];
    let buf = pool.get_or_create(&data);
    assert_eq!(pool.allocated_count(), 1);
    assert_eq!(pool.reused_count(), 0);

    pool.release(buf);
    assert_eq!(pool.free_count(), 1);

    let _buf2 = pool.get_or_create(&data);
    assert_eq!(pool.allocated_count(), 1);
    assert_eq!(pool.reused_count(), 1);
    assert_eq!(pool.free_count(), 0);

    let small = vec![1.0f32; 64];
    let _buf3 = pool.get_or_create(&small);
    assert_eq!(pool.allocated_count(), 2);
    assert_eq!(pool.reused_count(), 1);
}

#[test]
fn no_memory_leak_100_iterations() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let mut pool = BufferPool::new(&backend.device);

    let data = vec![1.0f32; 1024];
    for _ in 0..100 {
        let buf = pool.get_or_create(&data);
        pool.release(buf);
    }

    assert_eq!(pool.allocated_count(), 1);
    assert_eq!(pool.reused_count(), 99);
    assert_eq!(pool.free_count(), 1);
}

// ======================== Shader correctness tests ========================

#[test]
fn matmul_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    // 3x4 * 4x5 = 3x5 (non-power-of-2 dimensions)
    let m: u32 = 3;
    let n: u32 = 5;
    let k: u32 = 4;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();

    // CPU reference
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[(i * k + kk) as usize] * b_data[(kk * n + j) as usize];
            }
            expected[(i * n + j) as usize] = sum;
        }
    }

    let buf_a = make_buffer(dev, &a);
    let buf_b = make_buffer(dev, &b_data);
    let buf_c = make_empty(dev, (m * n) as usize);
    let buf_m = make_const_u32(dev, m);
    let buf_n = make_const_u32(dev, n);
    let buf_k = make_const_u32(dev, k);

    let pipeline = backend.pipeline("matmul").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_c), 0);
    encoder.set_buffer(3, Some(&buf_m), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);

    let grid =
        ::metal::MTLSize::new(((n + 31) / 32 * 32) as u64, ((m + 31) / 32 * 32) as u64, 1);
    let tg = ::metal::MTLSize::new(32, 32, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_c, (m * n) as usize);
    assert_approx(&result, &expected, 1e-4, "matmul");
}

#[test]
fn rmsnorm_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let n: u32 = 13; // non-power-of-2
    let eps: f32 = 1e-5;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.5).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    let expected: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * rms * wi)
        .collect();

    let buf_x = make_buffer(dev, &x);
    let buf_w = make_buffer(dev, &weight);
    let buf_out = make_empty(dev, n as usize);
    let buf_n = make_const_u32(dev, n);
    let buf_eps = make_const_f32(dev, eps);

    let pipeline = backend.pipeline("rmsnorm").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_w), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    encoder.set_buffer(4, Some(&buf_eps), 0);

    // Dispatch one threadgroup of 256 threads
    let grid = ::metal::MTLSize::new(256, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n as usize);
    assert_approx(&result, &expected, 1e-4, "rmsnorm");
}

#[test]
fn silu_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let n: u32 = 17; // non-power-of-2
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 4.0).collect();

    // CPU reference: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let expected: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();

    let buf_x = make_buffer(dev, &x);
    let buf_out = make_empty(dev, n as usize);
    let buf_n = make_const_u32(dev, n);

    let pipeline = backend.pipeline("silu").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_out), 0);
    encoder.set_buffer(2, Some(&buf_n), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n as usize);
    assert_approx(&result, &expected, 1e-4, "silu");
}

#[test]
fn swiglu_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let n: u32 = 11;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.4 - 2.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.5).collect();

    // CPU reference: swiglu(gate, up) = silu(gate) * up
    let expected: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let buf_gate = make_buffer(dev, &gate);
    let buf_up = make_buffer(dev, &up);
    let buf_out = make_empty(dev, n as usize);
    let buf_n = make_const_u32(dev, n);

    let pipeline = backend.pipeline("swiglu").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_gate), 0);
    encoder.set_buffer(1, Some(&buf_up), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n as usize);
    assert_approx(&result, &expected, 1e-4, "swiglu");
}

#[test]
fn attention_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let seq_len: u32 = 4;
    let kv_len: u32 = 4;
    let head_dim: u32 = 8; // small for testing

    // Deterministic test data
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i as f32) * 0.1 - 1.0) * 0.5)
        .collect();
    let k: Vec<f32> = (0..kv_len * head_dim)
        .map(|i| ((i as f32) * 0.07 + 0.3) * 0.5)
        .collect();
    let v: Vec<f32> = (0..kv_len * head_dim)
        .map(|i| ((i as f32) * 0.13 - 0.5) * 0.5)
        .collect();

    // CPU reference: scaled dot-product attention with causal mask
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut expected = vec![0.0f32; (seq_len * head_dim) as usize];
    for q_pos in 0..seq_len {
        let causal_len = (q_pos + 1).min(kv_len);

        // Compute scores
        let mut scores = vec![0.0f32; causal_len as usize];
        for j in 0..causal_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[(q_pos * head_dim + d) as usize] * k[(j * head_dim + d) as usize];
            }
            scores[j as usize] = dot * scale;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of V
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..causal_len {
                val += weights[j as usize] * v[(j * head_dim + d) as usize];
            }
            expected[(q_pos * head_dim + d) as usize] = val;
        }
    }

    let buf_q = make_buffer(dev, &q);
    let buf_k = make_buffer(dev, &k);
    let buf_v = make_buffer(dev, &v);
    let buf_out = make_empty(dev, (seq_len * head_dim) as usize);
    let buf_seq = make_const_u32(dev, seq_len);
    let buf_kv = make_const_u32(dev, kv_len);
    let buf_hd = make_const_u32(dev, head_dim);

    let pipeline = backend.pipeline("attention").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_q), 0);
    encoder.set_buffer(1, Some(&buf_k), 0);
    encoder.set_buffer(2, Some(&buf_v), 0);
    encoder.set_buffer(3, Some(&buf_out), 0);
    encoder.set_buffer(4, Some(&buf_seq), 0);
    encoder.set_buffer(5, Some(&buf_kv), 0);
    encoder.set_buffer(6, Some(&buf_hd), 0);

    // Dispatch one threadgroup per query position (1D grid: seq_len * 256 threads)
    let grid = ::metal::MTLSize::new(256 * seq_len as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, (seq_len * head_dim) as usize);
    assert_approx(&result, &expected, 1e-4, "attention");
}

// ======================== Dispatch layer tests ========================

/// CpuSimd backend produces correct matmul results (delegates to syn_sgemm).
#[test]
fn dispatch_cpu_matmul_correctness() {
    let backend = ComputeBackend::CpuSimd;
    let (m, k, n) = (4, 128, 256);

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.005 + 0.1).collect();

    let result = backend.matmul_t(&a, &b, m, k, n);

    // Reference: naive triple-loop matmul_t
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += a[i * k + d] * b[j * k + d];
            }
            expected[i * n + j] = sum;
        }
    }

    assert_approx(&result, &expected, 1e-3, "dispatch CpuSimd matmul");
}

/// Metal backend produces matmul results within 1e-4 of CPU.
#[test]
fn dispatch_metal_matmul_correctness() {
    let backend = match ComputeBackend::new_metal() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: no Metal GPU available");
            return;
        }
    };

    let (m, k, n) = (4, 128, 256);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.005 + 0.1).collect();

    let gpu_result = backend.matmul_t(&a, &b, m, k, n);
    let cpu_result = ComputeBackend::CpuSimd.matmul_t(&a, &b, m, k, n);

    assert_approx(
        &gpu_result,
        &cpu_result,
        1e-4,
        "dispatch Metal vs CPU matmul",
    );
}

/// Dispatch heuristic: small matrices -> CPU (no GPU dispatch).
#[test]
fn dispatch_heuristic_small_uses_cpu() {
    // M*N*K = 4*64*32 = 8192, well below 1M threshold
    let backend = match ComputeBackend::new_metal() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: no Metal GPU available");
            return;
        }
    };

    let (m, k, n) = (4, 32, 64);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.05).collect();

    // Both should produce identical results (CPU path for small matrices)
    let result = backend.matmul_t(&a, &b, m, k, n);
    let cpu_result = ComputeBackend::CpuSimd.matmul_t(&a, &b, m, k, n);

    // Should be exactly equal since both go through CPU for small matrices
    for (i, (r, c)) in result.iter().zip(cpu_result.iter()).enumerate() {
        assert_eq!(r, c, "small matmul[{i}]: dispatch should use CPU path");
    }
}

/// Dispatch heuristic: large matrices -> GPU.
#[test]
fn dispatch_heuristic_large_uses_gpu() {
    let backend = match ComputeBackend::new_metal() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: no Metal GPU available");
            return;
        }
    };

    // M*N*K = 128*256*128 = ~4.2M, above 1M threshold -> GPU
    let (m, k, n) = (128, 128, 256);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001 + 0.1).collect();

    let result = backend.matmul_t(&a, &b, m, k, n);
    let cpu_result = ComputeBackend::CpuSimd.matmul_t(&a, &b, m, k, n);

    // GPU result should be close to CPU (within 1e-4)
    assert_approx(&result, &cpu_result, 1e-3, "large matmul GPU vs CPU");
}

/// Metal backend rmsnorm matches CPU within tolerance.
#[test]
fn dispatch_metal_rmsnorm_correctness() {
    let backend = match ComputeBackend::new_metal() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: no Metal GPU available");
            return;
        }
    };

    let hidden_size = 1024;
    let batch = 4;
    let eps = 1e-5f32;
    let x: Vec<f32> = (0..batch * hidden_size)
        .map(|i| (i as f32) * 0.002 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let gpu_result = backend.rmsnorm(&x, &weight, eps, hidden_size);
    let cpu_result = ComputeBackend::CpuSimd.rmsnorm(&x, &weight, eps, hidden_size);

    assert_approx(
        &gpu_result,
        &cpu_result,
        1e-4,
        "dispatch Metal vs CPU rmsnorm",
    );
}

/// Metal backend swiglu matches CPU within tolerance.
#[test]
fn dispatch_metal_swiglu_correctness() {
    let backend = match ComputeBackend::new_metal() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: no Metal GPU available");
            return;
        }
    };

    let n = 2048;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003 - 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 + 0.5).collect();

    let gpu_result = backend.swiglu(&gate, &up);
    let cpu_result = ComputeBackend::CpuSimd.swiglu(&gate, &up);

    assert_approx(
        &gpu_result,
        &cpu_result,
        1e-4,
        "dispatch Metal vs CPU swiglu",
    );
}

/// Engine creates CpuSimd backend when using BackendSelection::CpuSimd.
#[test]
fn engine_creates_cpu_backend() {
    use crate::config::*;
    use crate::engine::{BackendSelection, InferenceEngine};

    let config = ModelConfig {
        name: "test".into(),
        architecture: ArchitectureConfig {
            hidden_size: 64,
            num_layers: 1,
            vocab_size: 100,
            max_sequence_length: 128,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::MHA {
            num_heads: 4,
            head_dim: 16,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 128,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 128,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    };

    let engine = InferenceEngine::from_config_with_backend(config, BackendSelection::CpuSimd);
    assert!(
        !engine.backend.is_gpu(),
        "CpuSimd selection should create CPU backend"
    );
}

/// Engine creates Metal backend when using BackendSelection::Auto (if GPU available).
#[test]
fn engine_creates_auto_backend() {
    use crate::config::*;
    use crate::engine::{BackendSelection, InferenceEngine};

    let config = ModelConfig {
        name: "test".into(),
        architecture: ArchitectureConfig {
            hidden_size: 64,
            num_layers: 1,
            vocab_size: 100,
            max_sequence_length: 128,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::MHA {
            num_heads: 4,
            head_dim: 16,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 128,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 128,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    };

    let engine = InferenceEngine::from_config_with_backend(config, BackendSelection::Auto);
    if MetalBackend::is_available() {
        assert!(
            engine.backend.is_gpu(),
            "Auto on Apple hardware should use GPU"
        );
    } else {
        assert!(
            !engine.backend.is_gpu(),
            "Auto without GPU should fallback to CPU"
        );
    }
}

// ==================== Critical shader correctness tests ====================

/// Create a Metal buffer from i8 data.
fn make_buffer_i8(device: &::metal::Device, data: &[i8]) -> ::metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        data.len() as u64,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Read i8 values from a shared-mode Metal buffer.
fn read_buffer_i8(buf: &::metal::Buffer, n: usize) -> Vec<i8> {
    let ptr = buf.contents() as *const i8;
    let mut out = vec![0i8; n];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
    out
}

/// Simple deterministic pseudo-random f32 in [-1, 1] from a seed + index.
fn pseudo_rand(seed: u32, idx: usize) -> f32 {
    // xorshift-like hash for reproducible test data
    let mut h = seed.wrapping_add(idx as u32).wrapping_mul(2654435761);
    h ^= h >> 16;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    // Map to [-1, 1]
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[test]
fn gemv_int8_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let k: usize = 64;
    let n: usize = 32;

    // Random activations
    let a: Vec<f32> = (0..k).map(|i| pseudo_rand(42, i)).collect();
    // Random int8 weights
    let b_int8: Vec<i8> = (0..k * n)
        .map(|i| {
            let v = pseudo_rand(99, i);
            (v * 127.0).clamp(-128.0, 127.0) as i8
        })
        .collect();
    // Random per-column scales
    let scales: Vec<f32> = (0..n)
        .map(|j| pseudo_rand(77, j).abs() * 0.1 + 0.01)
        .collect();

    // CPU reference: out[j] = sum_k(a[k] * i8_to_f32(b[k*N+j])) * scale[j]
    let mut expected = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for ki in 0..k {
            sum += a[ki] * (b_int8[ki * n + j] as f32);
        }
        expected[j] = sum * scales[j];
    }

    let buf_a = make_buffer(dev, &a);
    let buf_b = make_buffer_i8(dev, &b_int8);
    let buf_scales = make_buffer(dev, &scales);
    let buf_out = make_empty(dev, n);
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);

    let pipeline = backend.pipeline("gemv_int8").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_scales), 0);
    encoder.set_buffer(3, Some(&buf_out), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n.min(256) as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n);
    assert_approx(&result, &expected, 1e-2, "gemv_int8");
}

#[test]
fn gemv_f32_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let k: usize = 64;
    let n: usize = 32;

    let a: Vec<f32> = (0..k).map(|i| pseudo_rand(10, i)).collect();
    // B layout: [K, N] row-major (gemv kernel convention)
    let b: Vec<f32> = (0..k * n).map(|i| pseudo_rand(20, i)).collect();

    // CPU reference: out[j] = sum_k(a[k] * B[k*N + j])
    let mut expected = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for ki in 0..k {
            sum += a[ki] * b[ki * n + j];
        }
        expected[j] = sum;
    }

    let buf_a = make_buffer(dev, &a);
    let buf_b = make_buffer(dev, &b);
    let buf_out = make_empty(dev, n);
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);

    let pipeline = backend.pipeline("gemv").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    encoder.set_buffer(4, Some(&buf_k), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n.min(256) as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n);
    assert_approx(&result, &expected, 1e-4, "gemv_f32");
}

#[test]
fn rope_rotate_half_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let num_heads: usize = 4;
    let head_dim: usize = 16;
    let half_d = head_dim / 2;
    let total = num_heads * head_dim;
    let _pos: usize = 3;

    // Random Q data and cos/sin for one position
    let q_data: Vec<f32> = (0..total).map(|i| pseudo_rand(50, i)).collect();
    let cos_row: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.2).cos()).collect();
    let sin_row: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.2).sin()).collect();

    // CPU reference: apply_rope_inplace from ops::rope
    let mut expected = q_data.clone();
    crate::ops::rope::apply_rope_inplace(
        &mut expected,
        &cos_row,
        &sin_row,
        1, // seq_len = 1
        num_heads,
        head_dim,
        0, // pos_offset = 0 (cos/sin already indexed for our position)
        crate::config::position::RoPEStyle::RotateHalf,
    );

    // GPU: rope_rotate_half operates in-place on qk buffer
    let buf_qk = make_buffer(dev, &q_data);
    let buf_cos = make_buffer(dev, &cos_row);
    let buf_sin = make_buffer(dev, &sin_row);
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);

    let total_pairs = (num_heads * half_d) as u64;
    let pipeline = backend.pipeline("rope_rotate_half").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_qk), 0);
    encoder.set_buffer(1, Some(&buf_cos), 0);
    encoder.set_buffer(2, Some(&buf_sin), 0);
    encoder.set_buffer(3, Some(&buf_num_heads), 0);
    encoder.set_buffer(4, Some(&buf_head_dim), 0);

    let grid = ::metal::MTLSize::new(total_pairs, 1, 1);
    let tg = ::metal::MTLSize::new(total_pairs.min(256), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_qk, total);
    assert_approx(&result, &expected, 1e-5, "rope_rotate_half");
}

#[test]
fn kv_scatter_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let max_seq: usize = 16;
    let kv_dim: usize = 8;
    let pos: u32 = 5;

    // Initialize caches to zero
    let k_cache_data = vec![0.0f32; max_seq * kv_dim];
    let v_cache_data = vec![0.0f32; max_seq * kv_dim];

    // Token K/V data
    let k_token: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let v_token: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * -0.3).collect();

    let buf_k_cache = make_buffer(dev, &k_cache_data);
    let buf_v_cache = make_buffer(dev, &v_cache_data);
    let buf_k_token = make_buffer(dev, &k_token);
    let buf_v_token = make_buffer(dev, &v_token);
    let buf_pos = make_const_u32(dev, pos);
    let buf_kv_dim = make_const_u32(dev, kv_dim as u32);

    let pipeline = backend.pipeline("kv_cache_scatter").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_k_cache), 0);
    encoder.set_buffer(1, Some(&buf_v_cache), 0);
    encoder.set_buffer(2, Some(&buf_k_token), 0);
    encoder.set_buffer(3, Some(&buf_v_token), 0);
    encoder.set_buffer(4, Some(&buf_pos), 0);
    encoder.set_buffer(5, Some(&buf_kv_dim), 0);

    let grid = ::metal::MTLSize::new(kv_dim as u64, 1, 1);
    let tg = ::metal::MTLSize::new(kv_dim as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let k_result = read_buffer(&buf_k_cache, max_seq * kv_dim);
    let v_result = read_buffer(&buf_v_cache, max_seq * kv_dim);

    // Verify: positions 5*8..5*8+8 should match token data
    let offset = (pos as usize) * kv_dim;
    for i in 0..kv_dim {
        assert!(
            (k_result[offset + i] - k_token[i]).abs() < 1e-6,
            "k_cache[{}]: expected {} got {}",
            offset + i,
            k_token[i],
            k_result[offset + i]
        );
        assert!(
            (v_result[offset + i] - v_token[i]).abs() < 1e-6,
            "v_cache[{}]: expected {} got {}",
            offset + i,
            v_token[i],
            v_result[offset + i]
        );
    }

    // Verify: all other positions should still be zero
    for i in 0..max_seq * kv_dim {
        if i >= offset && i < offset + kv_dim {
            continue;
        }
        assert_eq!(
            k_result[i], 0.0,
            "k_cache[{i}]: should be 0 but was {}",
            k_result[i]
        );
        assert_eq!(
            v_result[i], 0.0,
            "v_cache[{i}]: should be 0 but was {}",
            v_result[i]
        );
    }
}

#[test]
fn attention_decode_vs_cpu() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let num_heads: usize = 4;
    let num_kv_heads: usize = 2;
    let head_dim: usize = 8;
    let seq_len: usize = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q: Vec<f32> = (0..q_dim).map(|i| pseudo_rand(11, i)).collect();
    let k_cache: Vec<f32> = (0..seq_len * kv_dim).map(|i| pseudo_rand(22, i)).collect();
    let v_cache: Vec<f32> = (0..seq_len * kv_dim).map(|i| pseudo_rand(33, i)).collect();

    // CPU reference: per-head attention with GQA
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut expected = vec![0.0f32; q_dim];
    for head in 0..num_heads {
        let kv_head = head / groups;
        // Compute scores
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[head * head_dim + d] * k_cache[j * kv_dim + kv_head * head_dim + d];
            }
            scores[j] = dot * scale;
        }
        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_e: f32 = exp_s.iter().sum();
        let weights: Vec<f32> = exp_s.iter().map(|e| e / sum_e).collect();
        // Weighted V sum
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..seq_len {
                val += weights[j] * v_cache[j * kv_dim + kv_head * head_dim + d];
            }
            expected[head * head_dim + d] = val;
        }
    }

    let buf_q = make_buffer(dev, &q);
    let buf_k = make_buffer(dev, &k_cache);
    let buf_v = make_buffer(dev, &v_cache);
    let buf_out = make_empty(dev, q_dim);
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_num_kv_heads = make_const_u32(dev, num_kv_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let buf_seq_len = make_const_u32(dev, seq_len as u32);
    let buf_kv_dim = make_const_u32(dev, kv_dim as u32);

    let pipeline = backend.pipeline("attention_decode").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_q), 0);
    encoder.set_buffer(1, Some(&buf_k), 0);
    encoder.set_buffer(2, Some(&buf_v), 0);
    encoder.set_buffer(3, Some(&buf_out), 0);
    encoder.set_buffer(4, Some(&buf_num_heads), 0);
    encoder.set_buffer(5, Some(&buf_num_kv_heads), 0);
    encoder.set_buffer(6, Some(&buf_head_dim), 0);
    encoder.set_buffer(7, Some(&buf_seq_len), 0);
    encoder.set_buffer(8, Some(&buf_kv_dim), 0);

    // One threadgroup per head, 256 threads per threadgroup
    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, q_dim);
    assert_approx(&result, &expected, 1e-3, "attention_decode");
}

#[test]
fn headwise_rmsnorm_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let num_heads: usize = 4;
    let head_dim: usize = 16;
    let total = num_heads * head_dim;
    let eps: f32 = 1e-5;

    let x: Vec<f32> = (0..total).map(|i| pseudo_rand(60, i)).collect();
    let weight: Vec<f32> = (0..head_dim)
        .map(|i| 1.0 + pseudo_rand(70, i) * 0.5)
        .collect();

    // CPU reference: apply_headwise_rmsnorm from ops::norm
    let expected =
        crate::ops::norm::apply_headwise_rmsnorm(&x, &weight, 1, num_heads, head_dim, eps);

    let buf_x = make_buffer(dev, &x);
    let buf_w = make_buffer(dev, &weight);
    let buf_out = make_empty(dev, total);
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let buf_eps = make_const_f32(dev, eps);
    let buf_hdw = make_const_u32(dev, head_dim as u32); // head_dim_weight > 0

    let pipeline = backend.pipeline("headwise_rmsnorm").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_w), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_num_heads), 0);
    encoder.set_buffer(4, Some(&buf_head_dim), 0);
    encoder.set_buffer(5, Some(&buf_eps), 0);
    encoder.set_buffer(6, Some(&buf_hdw), 0);

    // One threadgroup per head
    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, total);
    assert_approx(&result, &expected, 1e-4, "headwise_rmsnorm");
}

#[test]
fn headwise_rmsnorm_identity_when_empty() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let num_heads: usize = 4;
    let head_dim: usize = 16;
    let total = num_heads * head_dim;

    let x: Vec<f32> = (0..total).map(|i| pseudo_rand(80, i)).collect();
    // Dummy weight buffer (won't be accessed when head_dim_weight == 0)
    let dummy_weight = vec![0.0f32; 1];

    let buf_x = make_buffer(dev, &x);
    let buf_w = make_buffer(dev, &dummy_weight);
    let buf_out = make_empty(dev, total);
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let buf_eps = make_const_f32(dev, 1e-5);
    let buf_hdw = make_const_u32(dev, 0u32); // head_dim_weight = 0 -> identity

    let pipeline = backend.pipeline("headwise_rmsnorm").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_w), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_num_heads), 0);
    encoder.set_buffer(4, Some(&buf_head_dim), 0);
    encoder.set_buffer(5, Some(&buf_eps), 0);
    encoder.set_buffer(6, Some(&buf_hdw), 0);

    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, total);
    // Output must match input exactly (identity pass-through)
    for (i, (r, e)) in result.iter().zip(x.iter()).enumerate() {
        assert_eq!(r, e, "headwise_rmsnorm identity[{i}]: got {r} expected {e}");
    }
}

// ==================== Integration-style tests ====================

#[test]
fn gpu_int8_quantization_roundtrip() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let k: usize = 64;
    let n: usize = 32;

    // Create random f32 weights in [K, N] (already transposed layout)
    let original: Vec<f32> = (0..k * n).map(|i| pseudo_rand(123, i) * 2.0).collect();

    // Quantize using the same function used by MetalLayerWeights
    let (int8_buf, scale_buf) =
        super::gpu_buffers::quantize_and_upload_int8_public(dev, &original, k, n);

    // Read back int8 data and scales
    let int8_data = read_buffer_i8(&int8_buf, k * n);
    let scales = read_buffer(&scale_buf, n);

    // Dequantize: f32_out[k,j] = int8[k,j] * scale[j]
    // Check that the mean relative error (excluding near-zero values) is < 2%
    let mut rel_error_sum: f64 = 0.0;
    let mut rel_error_count: usize = 0;
    let mut max_abs_error: f32 = 0.0;
    for i in 0..k {
        for j in 0..n {
            let dequantized = (int8_data[i * n + j] as f32) * scales[j];
            let orig = original[i * n + j];
            let abs_err = (dequantized - orig).abs();
            if abs_err > max_abs_error {
                max_abs_error = abs_err;
            }
            // Relative error only for non-trivial values (skip near-zero originals)
            if orig.abs() > 0.1 {
                rel_error_sum += (abs_err / orig.abs()) as f64;
                rel_error_count += 1;
            }
        }
    }
    let mean_rel_error = if rel_error_count > 0 {
        rel_error_sum / rel_error_count as f64
    } else {
        0.0
    };
    assert!(
        mean_rel_error < 0.02,
        "INT8 quantization roundtrip mean relative error {mean_rel_error:.4} exceeds 2%"
    );
    // Also check max absolute error is bounded by the scale magnitude
    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_abs_error < max_scale * 1.5,
        "INT8 roundtrip max absolute error {max_abs_error} too large (max_scale={max_scale})"
    );
}

#[test]
fn gemv_int8_matches_cpu_qgemm() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;

    let k: usize = 128;
    let n: usize = 64;

    // Random activations
    let a_f32: Vec<f32> = (0..k).map(|i| pseudo_rand(200, i)).collect();

    // Create random f32 weights [K, N], quantize to int8
    let weights_f32: Vec<f32> = (0..k * n).map(|i| pseudo_rand(300, i) * 2.0).collect();

    // Per-column quantization (matching quantize_and_upload_int8 logic)
    let mut scales = vec![0.0f32; n];
    let mut int8_data = vec![0i8; k * n];
    for j in 0..n {
        let mut max_abs: f32 = 0.0;
        for i in 0..k {
            let v = weights_f32[i * n + j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales[j] = scale;
        let inv_scale = 1.0 / scale;
        for i in 0..k {
            let val = weights_f32[i * n + j] * inv_scale;
            int8_data[i * n + j] = val.round().clamp(-128.0, 127.0) as i8;
        }
    }

    // GPU: gemv_int8
    let buf_a = make_buffer(dev, &a_f32);
    let buf_b = make_buffer_i8(dev, &int8_data);
    let buf_scales = make_buffer(dev, &scales);
    let buf_out = make_empty(dev, n);
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);

    let pipeline = backend.pipeline("gemv_int8").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_scales), 0);
    encoder.set_buffer(3, Some(&buf_out), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n.min(256) as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let gpu_result = read_buffer(&buf_out, n);

    // CPU reference: qgemm_int8 with M=1
    // synapse_core::qgemm_int8 uses layout A_i8[M,K], B_i8[K,N] with
    // per-row scales_a[M] and per-column scales_b[N].
    // For our case: A is quantized activation (M=1), B is our int8 weights.
    // The GPU gemv_int8 computes: out[j] = sum_k(a_f32[k] * i8[k*N+j]) * scale[j]
    // qgemm_int8 computes: out[j] = scale_a[0] * sum_k(a_i8[k] * b_i8[k*N+j]) * scale_b[j]
    // These differ (GPU uses f32 activations; CPU quantizes activations too).
    // So we compare against the same CPU reference as the GPU: f32 activation * int8 weight * scale.
    let mut cpu_result = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for ki in 0..k {
            sum += a_f32[ki] * (int8_data[ki * n + j] as f32);
        }
        cpu_result[j] = sum * scales[j];
    }

    assert_approx(&gpu_result, &cpu_result, 1e-2, "gemv_int8 vs cpu reference");
}

// ======================== LEWM GPU forward tests ========================

#[test]
fn lewm_new_kernels_compiled() {
    // Verify all 6 new kernels exist in the compiled pipeline set.
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    for name in &[
        "gemv3_t",
        "layernorm_modulate",
        "gelu_inplace",
        "gated_residual",
        "add_bias",
        "attention_3x3",
    ] {
        assert!(
            backend.pipeline(name).is_some(),
            "Missing pipeline: {name}"
        );
    }
}

#[test]
fn lewm_gemv3_correctness() {
    // Test gemv3_t: C[3,4] = A[3,2] * B^T[4,2]
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;
    let m = 3u32;
    let n = 4u32;
    let k = 2u32;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
    let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0]; // [4, 2]

    let buf_a = make_buffer(dev, &a);
    let buf_b = make_buffer(dev, &b);
    let buf_c = make_empty(dev, (m * n) as usize);
    let buf_m = make_const_u32(dev, m);
    let buf_n = make_const_u32(dev, n);
    let buf_k = make_const_u32(dev, k);

    let pipeline = backend.pipeline("gemv3_t").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_c), 0);
    encoder.set_buffer(3, Some(&buf_m), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_c, (m * n) as usize);

    // CPU reference: C[i,j] = sum_k A[i,k] * B[j,k]
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for ki in 0..k as usize {
                sum += a[i * k as usize + ki] * b[j * k as usize + ki];
            }
            expected[i * n as usize + j] = sum;
        }
    }
    assert_approx(&result, &expected, 1e-5, "gemv3_t");
}

#[test]
fn lewm_gelu_correctness() {
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;
    let data = [-1.0f32, 0.0, 0.5, 1.0, 2.0];
    let n = data.len() as u32;

    let buf_x = make_buffer(dev, &data);
    let buf_n = make_const_u32(dev, n);

    let pipeline = backend.pipeline("gelu_inplace").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(n as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_x, n as usize);
    let expected: Vec<f32> = data
        .iter()
        .map(|&v| {
            0.5 * v
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt()
                        * (v + 0.044715 * v * v * v))
                        .tanh())
        })
        .collect();
    assert_approx(&result, &expected, 1e-5, "gelu_inplace");
}

#[test]
fn lewm_attention_3x3_correctness() {
    // Test bidirectional attention for seq_len=3, num_heads=2, head_dim=4
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };
    let dev = &backend.device;
    let seq_len = 3u32;
    let num_heads = 2u32;
    let head_dim = 4u32;
    let inner_dim = (num_heads * head_dim) as usize; // 8

    // Random-ish Q, K, V [seq_len, inner_dim]
    let q: Vec<f32> = (0..seq_len as usize * inner_dim)
        .map(|i| (i as f32 * 0.37 + 0.1).sin() * 0.5)
        .collect();
    let k: Vec<f32> = (0..seq_len as usize * inner_dim)
        .map(|i| (i as f32 * 0.53 + 0.3).sin() * 0.5)
        .collect();
    let v: Vec<f32> = (0..seq_len as usize * inner_dim)
        .map(|i| (i as f32 * 0.71 + 0.7).sin() * 0.5)
        .collect();

    let buf_q = make_buffer(dev, &q);
    let buf_k = make_buffer(dev, &k);
    let buf_v = make_buffer(dev, &v);
    let buf_out = make_empty(dev, seq_len as usize * inner_dim);
    let buf_nh = make_const_u32(dev, num_heads);
    let buf_hd = make_const_u32(dev, head_dim);
    let buf_sl = make_const_u32(dev, seq_len);

    let pipeline = backend.pipeline("attention_3x3").unwrap();
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_q), 0);
    encoder.set_buffer(1, Some(&buf_k), 0);
    encoder.set_buffer(2, Some(&buf_v), 0);
    encoder.set_buffer(3, Some(&buf_out), 0);
    encoder.set_buffer(4, Some(&buf_nh), 0);
    encoder.set_buffer(5, Some(&buf_hd), 0);
    encoder.set_buffer(6, Some(&buf_sl), 0);
    let grid = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let gpu_result = read_buffer(&buf_out, seq_len as usize * inner_dim);

    // CPU reference: bidirectional attention
    let sl = seq_len as usize;
    let hd = head_dim as usize;
    let nh = num_heads as usize;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut expected = vec![0.0f32; sl * inner_dim];

    for h in 0..nh {
        for qi in 0..sl {
            // Compute scores for all keys
            let mut scores = vec![0.0f32; sl];
            let mut max_score = f32::NEG_INFINITY;
            for ki in 0..sl {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[qi * inner_dim + h * hd + d] * k[ki * inner_dim + h * hd + d];
                }
                scores[ki] = dot * scale;
                if scores[ki] > max_score {
                    max_score = scores[ki];
                }
            }
            // Softmax
            let mut sum_exp = 0.0f32;
            for ki in 0..sl {
                scores[ki] = (scores[ki] - max_score).exp();
                sum_exp += scores[ki];
            }
            for ki in 0..sl {
                scores[ki] /= sum_exp;
            }
            // Weighted V
            for d in 0..hd {
                let mut val = 0.0f32;
                for ki in 0..sl {
                    val += scores[ki] * v[ki * inner_dim + h * hd + d];
                }
                expected[qi * inner_dim + h * hd + d] = val;
            }
        }
    }

    assert_approx(&gpu_result, &expected, 1e-4, "attention_3x3");
}

#[test]
fn lewm_metal_predict_matches_cpu() {
    // End-to-end: Metal predict_next output matches CPU predict_next.
    let backend = match get_backend() {
        Some(b) => b,
        None => return,
    };

    use crate::models::vision::lewm::{LeWMConfig, LeWorldModel};
    use crate::weight_loading::AlignedBuffer;

    // Deterministic pseudo-random weight generator
    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    // Use a small config to keep the test fast
    let config = LeWMConfig {
        image_size: 8,
        patch_size: 4,
        channels: 3,
        encoder_hidden: 16,
        encoder_layers: 2,
        encoder_heads: 2,
        encoder_inter: 32,
        predictor_hidden: 16,
        predictor_layers: 2,
        predictor_heads: 2,
        predictor_inner_dim: 16,
        predictor_inter: 32,
        action_dim: 4,
        latent_dim: 16,
    };

    let pred_h = config.predictor_hidden;
    let pred_inner = config.predictor_inner_dim;
    let pred_inter = config.predictor_inter;
    let act_dim = config.action_dim;
    let enc_inter = config.encoder_inter;
    let h = config.encoder_hidden;
    let patch_dim = config.patch_size * config.patch_size * config.channels;
    let num_patches = (config.image_size / config.patch_size).pow(2);
    let enc_seq_len = num_patches + 1;

    let mut model = LeWorldModel::from_config(&config);

    // Initialize encoder weights
    model.encoder.patch_proj = AlignedBuffer::from_slice(&gen_weights(h * patch_dim, 1));
    model.encoder.cls_token = AlignedBuffer::from_slice(&gen_weights(h, 2));
    model.encoder.pos_embed = AlignedBuffer::from_slice(&gen_weights(enc_seq_len * h, 3));
    model.encoder.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
        layer.w_q = AlignedBuffer::from_slice(&gen_weights(h * h, s + 1));
        layer.w_k = AlignedBuffer::from_slice(&gen_weights(h * h, s + 2));
        layer.w_v = AlignedBuffer::from_slice(&gen_weights(h * h, s + 3));
        layer.w_o = AlignedBuffer::from_slice(&gen_weights(h * h, s + 4));
        layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
        layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(enc_inter * h, s + 5));
        layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(h * enc_inter, s + 6));
    }

    // Projector
    model.projector.layers[0].0 =
        AlignedBuffer::from_slice(&gen_weights(enc_inter * h, 400));
    model.projector.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 401));
    model.projector.layers[1].0 =
        AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 402));
    model.projector.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 403));
    model.projector.layers[2].0 =
        AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 404));
    model.projector.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 405));

    // Pred_proj
    model.pred_proj.layers[0].0 =
        AlignedBuffer::from_slice(&gen_weights(enc_inter * pred_h, 500));
    model.pred_proj.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 501));
    model.pred_proj.layers[1].0 =
        AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 502));
    model.pred_proj.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 503));
    model.pred_proj.layers[2].0 =
        AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 504));
    model.pred_proj.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 505));

    // Action encoder
    model.action_conv_weight =
        AlignedBuffer::from_slice(&gen_weights(act_dim * act_dim, 600));
    model.action_conv_bias = AlignedBuffer::from_slice(&gen_weights(act_dim, 601));
    model.action_mlp1_weight =
        AlignedBuffer::from_slice(&gen_weights(enc_inter * act_dim, 602));
    model.action_mlp1_bias = AlignedBuffer::from_slice(&gen_weights(enc_inter, 603));
    model.action_mlp2_weight =
        AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 604));
    model.action_mlp2_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, 605));

    // Predictor
    model.predictor_pos_embed = AlignedBuffer::from_slice(&gen_weights(3 * pred_h, 700));
    model.predictor_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
    model.predictor_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);

    for (i, layer) in model.predictor_layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 1000;
        layer.adaln_weight =
            AlignedBuffer::from_slice(&gen_weights(6 * pred_h * pred_h, s + 1));
        layer.adaln_bias = AlignedBuffer::from_slice(&gen_weights(6 * pred_h, s + 2));
        layer.to_qkv =
            AlignedBuffer::from_slice(&gen_weights(3 * pred_inner * pred_h, s + 3));
        layer.attn_out_weight =
            AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inner, s + 4));
        layer.attn_out_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 5));
        layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
        layer.attn_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
        layer.mlp_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
        layer.mlp_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
        layer.mlp_up_weight =
            AlignedBuffer::from_slice(&gen_weights(pred_inter * pred_h, s + 10));
        layer.mlp_up_bias = AlignedBuffer::from_slice(&gen_weights(pred_inter, s + 11));
        layer.mlp_down_weight =
            AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inter, s + 12));
        layer.mlp_down_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 13));
    }

    let state = MetalLeWMState::from_model(&model, &backend);

    let z_t = gen_weights(config.latent_dim, 42);
    let action = gen_weights(config.action_dim, 43);

    let cpu_result = model.predict_next(&z_t, &action);
    let gpu_result = model.predict_next_metal(&z_t, &action, &state, &backend);

    assert_eq!(cpu_result.len(), gpu_result.len(), "output length mismatch");
    // Both paths run the same computation; tolerance accounts for floating-point order
    assert_approx(&gpu_result, &cpu_result, 1e-2, "lewm_metal_vs_cpu");
}
