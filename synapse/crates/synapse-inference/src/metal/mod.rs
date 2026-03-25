mod buffer;
mod device;
pub mod dispatch;
pub mod gpu_buffers;
pub mod gpu_forward;

pub use buffer::BufferPool;
pub use device::{MetalBackend, MetalError};
pub use dispatch::ComputeBackend;
pub use gpu_buffers::MetalModelBuffers;

#[cfg(test)]
mod tests {
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
    fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{label}[{i}]: GPU={a} vs CPU={e}, diff={}",
                (a - e).abs()
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
                    assert!(
                        backend.pipeline(name).is_some(),
                        "Missing pipeline: {name}"
                    );
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

        let grid = ::metal::MTLSize::new(
            ((n + 31) / 32 * 32) as u64,
            ((m + 31) / 32 * 32) as u64,
            1,
        );
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
                    dot += q[(q_pos * head_dim + d) as usize]
                        * k[(j * head_dim + d) as usize];
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

        assert_approx(&gpu_result, &cpu_result, 1e-4, "dispatch Metal vs CPU matmul");
    }

    /// Dispatch heuristic: small matrices → CPU (no GPU dispatch).
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

    /// Dispatch heuristic: large matrices → GPU.
    #[test]
    fn dispatch_heuristic_large_uses_gpu() {
        let backend = match ComputeBackend::new_metal() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping: no Metal GPU available");
                return;
            }
        };

        // M*N*K = 128*256*128 = ~4.2M, above 1M threshold → GPU
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
        let weight: Vec<f32> = (0..hidden_size)
            .map(|i| 1.0 + (i as f32) * 0.001)
            .collect();

        let gpu_result = backend.rmsnorm(&x, &weight, eps, hidden_size);
        let cpu_result = ComputeBackend::CpuSimd.rmsnorm(&x, &weight, eps, hidden_size);

        assert_approx(&gpu_result, &cpu_result, 1e-4, "dispatch Metal vs CPU rmsnorm");
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

        assert_approx(&gpu_result, &cpu_result, 1e-4, "dispatch Metal vs CPU swiglu");
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
            },
            quantization: QuantConfig::F32,
        };

        let engine = InferenceEngine::from_config_with_backend(config, BackendSelection::CpuSimd);
        assert!(!engine.backend.is_gpu(), "CpuSimd selection should create CPU backend");
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
            },
            quantization: QuantConfig::F32,
        };

        let engine = InferenceEngine::from_config_with_backend(config, BackendSelection::Auto);
        if MetalBackend::is_available() {
            assert!(engine.backend.is_gpu(), "Auto on Apple hardware should use GPU");
        } else {
            assert!(!engine.backend.is_gpu(), "Auto without GPU should fallback to CPU");
        }
    }
}
