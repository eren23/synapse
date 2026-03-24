use super::buffer::BufferPool;
use super::device::MetalBackend;
use std::cell::RefCell;

/// Threshold for GPU dispatch: M*N*K > 1M elements → GPU path.
const GPU_DISPATCH_THRESHOLD: usize = 1_000_000;

/// Backend dispatcher that routes operations to CPU (Zig SIMD) or GPU (Metal)
/// based on matrix size heuristics.
pub enum ComputeBackend {
    /// CPU path using Zig SIMD kernels via FFI.
    CpuSimd,
    /// Metal GPU path with buffer pooling.
    Metal {
        backend: MetalBackend,
        pool: RefCell<BufferPool>,
    },
}

impl ComputeBackend {
    /// Create a Metal GPU backend. Returns `Err` if no GPU is available.
    pub fn new_metal() -> Result<Self, super::device::MetalError> {
        let backend = MetalBackend::new()?;
        let pool = RefCell::new(BufferPool::new(&backend.device));
        Ok(ComputeBackend::Metal { backend, pool })
    }

    /// Try to create a Metal backend, falling back to CpuSimd if unavailable.
    pub fn auto() -> Self {
        match Self::new_metal() {
            Ok(b) => b,
            Err(_) => ComputeBackend::CpuSimd,
        }
    }

    /// Returns true if this backend dispatches to GPU for large operations.
    pub fn is_gpu(&self) -> bool {
        matches!(self, ComputeBackend::Metal { .. })
    }

    /// Whether a matmul of given dimensions should go to GPU.
    ///
    /// M=1 (single-token decode) always stays on CPU — GPU kernel launch
    /// overhead dominates for single-row operations. Metal is only used
    /// for batched operations (prefill, training) where M > 1.
    fn should_use_gpu(&self, m: usize, n: usize, k: usize) -> bool {
        matches!(self, ComputeBackend::Metal { .. })
            && m > 1
            && m * n * k > GPU_DISPATCH_THRESHOLD
    }

    /// y = A * B^T  where A is [m, k], B is [n, k] → y is [m, n].
    ///
    /// Dispatches to Metal GPU for large matrices, CPU SIMD otherwise.
    pub fn matmul_t(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k, "matmul_t: a.len() != m*k");
        debug_assert_eq!(b.len(), n * k, "matmul_t: b.len() != n*k");

        if self.should_use_gpu(m, n, k) {
            if let ComputeBackend::Metal { backend, pool } = self {
                return gpu_matmul_t(a, b, m, k, n, backend, &mut pool.borrow_mut());
            }
        }
        cpu_matmul_t(a, b, m, k, n)
    }

    /// RMS normalization over the last dimension.
    ///
    /// `x` is `[batch, hidden_size]`, `weight` is `[hidden_size]`.
    /// Dispatches to GPU when batch * hidden_size > threshold.
    pub fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
        let batch = x.len() / hidden_size;
        // Use hidden_size^2 * batch as proxy for compute cost
        if self.should_use_gpu(batch, hidden_size, hidden_size) {
            if let ComputeBackend::Metal { backend, pool } = self {
                return gpu_rmsnorm(x, weight, eps, hidden_size, backend, &mut pool.borrow_mut());
            }
        }
        cpu_rmsnorm(x, weight, eps, hidden_size)
    }

    /// Fused SwiGLU: out = silu(gate) * up.
    ///
    /// All slices are the same length. Dispatches to GPU for large vectors.
    pub fn swiglu(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        let n = gate.len();
        // SwiGLU is elementwise; use n as the size proxy
        if matches!(self, ComputeBackend::Metal { .. }) && n > GPU_DISPATCH_THRESHOLD {
            if let ComputeBackend::Metal { backend, pool } = self {
                return gpu_swiglu(gate, up, backend, &mut pool.borrow_mut());
            }
        }
        cpu_swiglu(gate, up)
    }

    /// Scaled dot-product attention with causal mask (single head).
    ///
    /// Q: [seq_len, head_dim], K: [kv_len, head_dim], V: [kv_len, head_dim].
    /// Dispatches to GPU for large attention windows.
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        if self.should_use_gpu(seq_len, kv_len, head_dim) {
            if let ComputeBackend::Metal { backend, pool } = self {
                return gpu_attention(
                    q, k, v, seq_len, kv_len, head_dim, backend,
                    &mut pool.borrow_mut(),
                );
            }
        }
        cpu_attention(q, k, v, seq_len, kv_len, head_dim)
    }
}

// ── CPU paths (Zig SIMD FFI) ─────────────────────────────────────────

fn cpu_matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let status = unsafe {
        synapse_sys::syn_sgemm(
            m, n, k,
            a.as_ptr(), k, 0,
            b.as_ptr(), k, 1,
            out.as_mut_ptr(), n,
        )
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm failed: {status}");
    out
}

fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    unsafe {
        for i in 0..n {
            let off = i * hidden_size;
            let row_ptr = x.as_ptr().add(off);
            let out_ptr = out.as_mut_ptr().add(off);

            synapse_sys::syn_vmul(out_ptr, row_ptr, row_ptr, hidden_size);
            let mut sum_sq = 0.0f32;
            synapse_sys::syn_vreduce_sum(out_ptr, hidden_size, &mut sum_sq);
            let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();
            synapse_sys::syn_vmul(out_ptr, row_ptr, weight.as_ptr(), hidden_size);
            for j in 0..hidden_size {
                *out_ptr.add(j) *= scale;
            }
        }
    }
    out
}

fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let len = gate.len();
    let mut out = vec![0.0f32; len];
    let status = unsafe {
        synapse_sys::syn_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len)
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
    out
}

fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * head_dim];

    for q_pos in 0..seq_len {
        let causal_len = (q_pos + 1).min(kv_len);
        let mut scores = vec![0.0f32; causal_len];
        for j in 0..causal_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_pos * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }

        // Softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // Weighted sum of V
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..causal_len {
                val += scores[j] * v[j * head_dim + d];
            }
            out[q_pos * head_dim + d] = val;
        }
    }
    out
}

// ── GPU paths (Metal) ────────────────────────────────────────────────

/// Helper: create a Metal buffer from f32 data.
fn make_buffer(device: &::metal::Device, data: &[f32]) -> ::metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<f32>()) as u64,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Helper: create an empty Metal buffer for n f32 elements.
fn make_empty(device: &::metal::Device, n: usize) -> ::metal::Buffer {
    device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Helper: create a Metal buffer holding a single u32 constant.
fn make_const_u32(device: &::metal::Device, val: u32) -> ::metal::Buffer {
    make_buffer(device, &[f32::from_bits(val)])
}

/// Helper: create a Metal buffer holding a single f32 constant.
fn make_const_f32(device: &::metal::Device, val: f32) -> ::metal::Buffer {
    make_buffer(device, &[val])
}

/// Helper: read f32 values from a shared-mode Metal buffer.
fn read_buffer(buf: &::metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let mut out = vec![0.0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
    out
}

fn gpu_matmul_t(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let dev = &backend.device;

    // Metal matmul kernel expects B as [K, N], but our B is [N, K] (transposed).
    // Ensure the transposed weight is cached first, then borrow everything.
    pool.get_or_create_transposed_weight(b, n, k);

    // Now all borrows are separate — get pointers to cached data
    let buf_b_ptr = b.as_ptr() as usize;
    let buf_a = pool.get_or_create(a);
    let buf_c = pool.create_empty(m * n);
    let buf_m = make_const_u32(dev, m as u32);
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);

    let pipeline = backend.pipeline("matmul").expect("matmul pipeline missing");
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    let buf_b = pool.get_cached_weight(buf_b_ptr).expect("weight should be cached");
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(&buf_c), 0);
    encoder.set_buffer(3, Some(&buf_m), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);

    let grid = ::metal::MTLSize::new(
        ((n as u32 + 31) / 32 * 32) as u64,
        ((m as u32 + 31) / 32 * 32) as u64,
        1,
    );
    let tg = ::metal::MTLSize::new(32, 32, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_c, m * n);

    pool.release(buf_a);
    // buf_b is cached — don't release it
    pool.release(buf_c);

    result
}

fn gpu_rmsnorm(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    hidden_size: usize,
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let dev = &backend.device;
    let batch = x.len() / hidden_size;

    let buf_x = pool.get_or_create(x);
    let buf_w = pool.get_or_create(weight);
    let buf_out = pool.create_empty(x.len());
    let buf_n = make_const_u32(dev, hidden_size as u32);
    let buf_eps = make_const_f32(dev, eps);

    let pipeline = backend.pipeline("rmsnorm").expect("rmsnorm pipeline missing");
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_x), 0);
    encoder.set_buffer(1, Some(&buf_w), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    encoder.set_buffer(4, Some(&buf_eps), 0);

    // One threadgroup per row, 256 threads per threadgroup
    let threadgroups = ::metal::MTLSize::new(batch as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, x.len());

    pool.release(buf_x);
    pool.release(buf_w);
    pool.release(buf_out);

    result
}

fn gpu_swiglu(
    gate: &[f32],
    up: &[f32],
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let dev = &backend.device;
    let n = gate.len();

    let buf_gate = pool.get_or_create(gate);
    let buf_up = pool.get_or_create(up);
    let buf_out = pool.create_empty(n);
    let buf_n = make_const_u32(dev, n as u32);

    let pipeline = backend.pipeline("swiglu").expect("swiglu pipeline missing");
    let cmd_buf = backend.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_gate), 0);
    encoder.set_buffer(1, Some(&buf_up), 0);
    encoder.set_buffer(2, Some(&buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, n);

    pool.release(buf_gate);
    pool.release(buf_up);
    pool.release(buf_out);

    result
}

fn gpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let dev = &backend.device;

    let buf_q = pool.get_or_create(q);
    let buf_k = pool.get_or_create(k);
    let buf_v = pool.get_or_create(v);
    let buf_out = pool.create_empty(seq_len * head_dim);
    let buf_seq = make_const_u32(dev, seq_len as u32);
    let buf_kv = make_const_u32(dev, kv_len as u32);
    let buf_hd = make_const_u32(dev, head_dim as u32);

    let pipeline = backend.pipeline("attention").expect("attention pipeline missing");
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

    // One threadgroup per query position, 256 threads each
    let threadgroups = ::metal::MTLSize::new(seq_len as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_out, seq_len * head_dim);

    pool.release(buf_q);
    pool.release(buf_k);
    pool.release(buf_v);
    pool.release(buf_out);

    result
}
