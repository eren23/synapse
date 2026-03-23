use crate::registry::{AttentionVariant, FFNVariant, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError};

/// A single decoder transformer layer (pre-norm architecture).
///
/// Forward: norm → attention → residual → norm → FFN → residual.
pub struct DecoderLayer {
    pub attn_norm: Box<dyn NormVariant>,
    pub attention: Box<dyn AttentionVariant>,
    pub ffn_norm: Box<dyn NormVariant>,
    pub ffn: Box<dyn FFNVariant>,
    pub hidden_size: usize,

    // ── Weights (64-byte aligned for SIMD) ───────────────────────────
    pub attn_norm_weight: AlignedBuffer,
    pub w_q: AlignedBuffer,
    pub w_k: AlignedBuffer,
    pub w_v: AlignedBuffer,
    pub w_o: AlignedBuffer,
    pub q_norm_weight: AlignedBuffer,
    pub k_norm_weight: AlignedBuffer,
    pub ffn_norm_weight: AlignedBuffer,
    pub ffn_gate: AlignedBuffer,
    pub ffn_up: AlignedBuffer,
    pub ffn_down: AlignedBuffer,
}

impl DecoderLayer {
    /// Pre-norm forward: norm→attention→residual→norm→FFN→residual.
    ///
    /// `x` is `[seq_len, hidden_size]` (flat). Returns same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out = self.apply_attention(&normed, seq_len);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Number of trainable parameters in this layer.
    pub fn param_count(&self) -> usize {
        let h = self.hidden_size;
        let q_dim = self.attention.num_heads() * self.attention.head_dim();
        let kv_dim = self.attention.num_kv_heads() * self.attention.head_dim();
        let inter = self.ffn.intermediate_size();

        let norms = 2 * h + 2 * self.attention.head_dim(); // attn_norm + q/k norm + ffn_norm
        let attn = q_dim * h + kv_dim * h + kv_dim * h + h * q_dim;
        let ffn = if is_gated_ffn(self.ffn.name()) {
            3 * inter * h
        } else {
            2 * inter * h
        };

        norms + attn + ffn
    }

    /// Weight keys this layer expects (relative to `layers[i].`).
    pub fn weight_keys(&self, layer_idx: usize) -> Vec<String> {
        let i = layer_idx;
        let mut keys = vec![
            format!("layers[{i}].attn_norm.weight"),
            format!("layers[{i}].attention.w_q"),
            format!("layers[{i}].attention.w_k"),
            format!("layers[{i}].attention.w_v"),
            format!("layers[{i}].attention.w_o"),
            format!("layers[{i}].attention.q_norm"),
            format!("layers[{i}].attention.k_norm"),
            format!("layers[{i}].ffn_norm.weight"),
        ];
        if is_gated_ffn(self.ffn.name()) {
            keys.push(format!("layers[{i}].ffn.w_gate"));
        }
        keys.push(format!("layers[{i}].ffn.w_up"));
        keys.push(format!("layers[{i}].ffn.w_down"));
        keys
    }

    /// Assign a weight by its field name (e.g. "attention.w_q").
    pub fn set_weight(&mut self, field: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        self.validate_weight_shape(field, &tensor.shape)?;
        match field {
            "attn_norm.weight" => self.attn_norm_weight = tensor.data.clone(),
            "attention.w_q" => self.w_q = tensor.data.clone(),
            "attention.w_k" => self.w_k = tensor.data.clone(),
            "attention.w_v" => self.w_v = tensor.data.clone(),
            "attention.w_o" => self.w_o = tensor.data.clone(),
            "attention.q_norm" => self.q_norm_weight = tensor.data.clone(),
            "attention.k_norm" => self.k_norm_weight = tensor.data.clone(),
            "ffn_norm.weight" => self.ffn_norm_weight = tensor.data.clone(),
            "ffn.w_gate" => self.ffn_gate = tensor.data.clone(),
            "ffn.w_up" => self.ffn_up = tensor.data.clone(),
            "ffn.w_down" => self.ffn_down = tensor.data.clone(),
            _ => {}
        }
        Ok(())
    }

    fn validate_weight_shape(&self, field: &str, actual: &[usize]) -> Result<(), WeightError> {
        let h = self.hidden_size;
        let q_dim = self.attention.num_heads() * self.attention.head_dim();
        let kv_dim = self.attention.num_kv_heads() * self.attention.head_dim();
        let inter = self.ffn.intermediate_size();

        let expected = match field {
            "attn_norm.weight" | "ffn_norm.weight" => vec![h],
            "attention.w_q" => vec![q_dim, h],
            "attention.w_k" | "attention.w_v" => vec![kv_dim, h],
            "attention.w_o" => vec![h, q_dim],
            "attention.q_norm" | "attention.k_norm" => vec![self.attention.head_dim()],
            "ffn.w_gate" | "ffn.w_up" => vec![inter, h],
            "ffn.w_down" => vec![h, inter],
            _ => return Ok(()),
        };

        if actual != expected {
            return Err(WeightError::ShapeMismatch(format!(
                "{field}: expected {:?}, got {:?}",
                expected, actual
            )));
        }

        Ok(())
    }

    // ── Attention ────────────────────────────────────────────────────

    fn apply_attention(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q, K, V projections: x is [seq_len, h]
        let q = matmul_t(x, &self.w_q, seq_len, h, q_dim);
        let k = matmul_t(x, &self.w_k, seq_len, h, kv_dim);
        let v = matmul_t(x, &self.w_v, seq_len, h, kv_dim);
        let q = apply_headwise_rmsnorm(
            &q,
            &self.q_norm_weight,
            seq_len,
            num_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );
        let k = apply_headwise_rmsnorm(
            &k,
            &self.k_norm_weight,
            seq_len,
            num_kv_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );

        // Multi-head causal attention with GQA support
        let mut attn_output = vec![0.0f32; seq_len * q_dim];

        for head in 0..num_heads {
            let kv_head = head / groups;

            for t in 0..seq_len {
                // Compute scores for position t attending to 0..=t
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + head * head_dim + d]
                            * k[s * kv_dim + kv_head * head_dim + d];
                    }
                    scores[s] = dot * scale;
                }

                softmax_slice(&mut scores[..=t]);

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_head * head_dim + d];
                    }
                    attn_output[t * q_dim + head * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        matmul_t(&attn_output, &self.w_o, seq_len, q_dim, h)
    }

    // ── FFN ──────────────────────────────────────────────────────────

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                let len = hidden.len();
                let status = unsafe {
                    synapse_sys::syn_swiglu(
                        hidden.as_mut_ptr(),
                        gate.as_ptr(),
                        up.as_ptr(),
                        len,
                    )
                };
                debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            "GeGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = gelu(gate[i]) * up[i];
                }
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            _ => {
                // GELU or others: y = activation(x @ up^T) @ down^T
                let mut activated = matmul_t(x, &self.ffn_up, tokens, h, inter);
                for v in activated.iter_mut() {
                    *v = gelu(*v);
                }
                matmul_t(&activated, &self.ffn_down, tokens, inter, h)
            }
        }
    }
}

// ── Math helpers ─────────────────────────────────────────────────────

/// Whether this FFN variant is gated (3 weight matrices vs 2).
pub(crate) fn is_gated_ffn(name: &str) -> bool {
    matches!(name, "SwiGLU" | "GeGLU")
}

/// Apply normalization (dispatch on variant name).
pub(crate) fn apply_norm(
    x: &[f32],
    weight: &[f32],
    norm: &dyn NormVariant,
    hidden_size: usize,
) -> Vec<f32> {
    let eps = norm.eps() as f32;
    match norm.name() {
        "RMSNorm" => rmsnorm(x, weight, eps, hidden_size),
        "LayerNorm" => layernorm(x, weight, eps, hidden_size),
        _ => x.to_vec(),
    }
}

/// y = A * B^T  where A is [m, k], B is [n, k] → y is [m, n].
///
/// Dispatches to the Zig SIMD tiled GEMM (`syn_sgemm`) via FFI.
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];
    // syn_sgemm: C = op(A) * op(B), row-major.
    //   A [m, k] no-transpose, lda = k
    //   B [n, k] transposed → [k, n], ldb = k
    //   C [m, n], ldc = n
    let status = unsafe {
        synapse_sys::syn_sgemm(
            m, n, k,
            a.as_ptr(), k, 0,   // A: no transpose
            b.as_ptr(), k, 1,   // B: transpose
            out.as_mut_ptr(), n, // C
        )
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm failed: {status}");
    out
}

/// Naive triple-loop reference implementation of y = A * B^T.
///
/// Kept for test comparison against the SIMD path.
pub(crate) fn matmul_t_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t_naive: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t_naive: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += a_row[d] * b_row[d];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// RMS normalization over the last dimension (SIMD via Zig FFI).
///
/// Uses `syn_vmul` / `syn_vreduce_sum` for zero-copy SIMD on each row,
/// avoiding tensor-handle allocation overhead that dominates at small sizes.
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];

    unsafe {
        for i in 0..n {
            let off = i * hidden_size;
            let row_ptr = x.as_ptr().add(off);
            let out_ptr = out.as_mut_ptr().add(off);

            // SIMD: out = x ⊙ x  (reuse output as scratch for squared values)
            synapse_sys::syn_vmul(out_ptr, row_ptr, row_ptr, hidden_size);

            // SIMD: ms = Σ(x²)
            let mut sum_sq = 0.0f32;
            synapse_sys::syn_vreduce_sum(out_ptr, hidden_size, &mut sum_sq);

            let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

            // SIMD: out = x ⊙ weight
            synapse_sys::syn_vmul(out_ptr, row_ptr, weight.as_ptr(), hidden_size);

            // Scale by normalization factor.  At 1024 elements this is auto-
            // vectorized by LLVM and negligible relative to the SIMD ops above.
            for j in 0..hidden_size {
                *out_ptr.add(j) *= scale;
            }
        }
    }
    out
}

/// Naive scalar RMS normalization (reference for test comparison).
///
/// Uses `black_box` on the accumulator to prevent LLVM auto-vectorization,
/// giving a fair scalar-vs-SIMD benchmark comparison.
#[inline(never)]
pub(crate) fn rmsnorm_naive(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mut ms = 0.0f32;
        for j in 0..hidden_size {
            ms += slice[j] * slice[j];
            ms = std::hint::black_box(ms);
        }
        ms /= hidden_size as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = std::hint::black_box(slice[j] * scale) * weight[j];
        }
    }
    out
}

/// Layer normalization over the last dimension (gamma only, no beta).
fn layernorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mean: f32 = slice.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (slice[j] - mean) * scale * weight[j];
        }
    }
    out
}

fn apply_headwise_rmsnorm(
    x: &[f32],
    weight: &[f32],
    _rows: usize,
    _heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    if weight.is_empty() {
        return x.to_vec();
    }

    // Data is already contiguous per-head: [rows * heads, head_dim].
    // Delegate to SIMD rmsnorm which normalizes over the last dimension.
    rmsnorm(x, weight, eps, head_dim)
}

/// Naive scalar headwise RMS normalization (reference for test comparison).
pub(crate) fn apply_headwise_rmsnorm_naive(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    if weight.is_empty() {
        return x.to_vec();
    }

    let mut out = vec![0.0f32; x.len()];
    let stride = heads * head_dim;
    for row in 0..rows {
        let row_offset = row * stride;
        for head in 0..heads {
            let head_offset = row_offset + head * head_dim;
            let slice = &x[head_offset..head_offset + head_dim];
            let ms = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for idx in 0..head_dim {
                out[head_offset + idx] = slice[idx] * scale * weight[idx];
            }
        }
    }
    out
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

pub(crate) fn softmax_slice(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

pub(crate) fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub(crate) fn add_vecs_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate deterministic pseudo-random f32 values in [-1, 1].
    fn pseudo_rand(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    /// Assert element-wise closeness: |a - e| <= atol + rtol * |e|.
    /// Same semantics as numpy.allclose.
    fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
        const ATOL: f32 = 1e-5;
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let bound = ATOL + rtol * e.abs();
            assert!(
                diff <= bound,
                "{label}[{i}]: simd={a}, naive={e}, diff={diff} > bound={bound} \
                 (atol={ATOL}, rtol={rtol})"
            );
        }
    }

    // ── Correctness tests: SIMD matches naive ─────────────────────────
    // Tolerance: SIMD tiling reorders accumulation vs the scalar loop,
    // causing O(k · eps) relative error.  Use 1e-5 for small k, 1e-4
    // for k ≥ 512 (consistent with the full-model 1e-4 requirement).

    #[test]
    fn matmul_simd_vs_naive_1x512_x_512x1024() {
        let (m, k, n) = (1, 512, 1024);
        let a = pseudo_rand(m * k, 42);
        let b = pseudo_rand(n * k, 123);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[1,512]x[512,1024]");
    }

    #[test]
    fn matmul_simd_vs_naive_4x1024_x_1024x3072() {
        let (m, k, n) = (4, 1024, 3072);
        let a = pseudo_rand(m * k, 7);
        let b = pseudo_rand(n * k, 99);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[4,1024]x[1024,3072]");
    }

    #[test]
    fn matmul_simd_vs_naive_128x1024_x_1024x1024() {
        let (m, k, n) = (128, 1024, 1024);
        let a = pseudo_rand(m * k, 314);
        let b = pseudo_rand(n * k, 159);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[128,1024]x[1024,1024]");
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn matmul_edge_m1_single_token() {
        let (m, k, n) = (1, 64, 128);
        let a = pseudo_rand(m * k, 1);
        let b = pseudo_rand(n * k, 2);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-5, "m=1 single token");
    }

    #[test]
    fn matmul_edge_k1() {
        let (m, k, n) = (8, 1, 16);
        let a = pseudo_rand(m * k, 10);
        let b = pseudo_rand(n * k, 20);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-5, "k=1");
    }

    #[test]
    fn matmul_edge_non_power_of_2() {
        let (m, k, n) = (13, 67, 101);
        let a = pseudo_rand(m * k, 55);
        let b = pseudo_rand(n * k, 77);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "non-pow2 [13,67]x[67,101]");
    }

    #[test]
    fn matmul_edge_small_1x1() {
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let simd = matmul_t(&a, &b, 1, 1, 1);
        let naive = matmul_t_naive(&a, &b, 1, 1, 1);
        assert_close(&simd, &naive, 1e-5, "1x1");
    }

    // ── RMSNorm: SIMD vs naive correctness ────────────────────────────

    #[test]
    fn rmsnorm_simd_vs_naive_1x1024() {
        let hidden = 1024;
        let x = pseudo_rand(1 * hidden, 42);
        let w = pseudo_rand(hidden, 100).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [1,1024]");
    }

    #[test]
    fn rmsnorm_simd_vs_naive_4x1024() {
        let hidden = 1024;
        let x = pseudo_rand(4 * hidden, 7);
        let w = pseudo_rand(hidden, 200).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [4,1024]");
    }

    #[test]
    fn rmsnorm_simd_vs_naive_128x1024() {
        let hidden = 1024;
        let x = pseudo_rand(128 * hidden, 314);
        let w = pseudo_rand(hidden, 300).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [128,1024]");
    }

    #[test]
    fn rmsnorm_weighted_vs_unweighted() {
        let hidden = 64;
        let x = pseudo_rand(2 * hidden, 55);
        let ones = vec![1.0f32; hidden];
        let gamma = pseudo_rand(hidden, 77).iter().map(|v| v.abs() + 0.5).collect::<Vec<_>>();

        let out_unit = rmsnorm(&x, &ones, 1e-5, hidden);
        let out_weighted = rmsnorm(&x, &gamma, 1e-5, hidden);

        // Weighted output should equal unit output * gamma element-wise.
        let out_unit_naive = rmsnorm_naive(&x, &ones, 1e-5, hidden);
        let out_weighted_naive = rmsnorm_naive(&x, &gamma, 1e-5, hidden);

        // Verify the naive weighted = naive_unit * gamma
        for i in 0..out_weighted_naive.len() {
            let j = i % hidden;
            let expected = out_unit_naive[i] * gamma[j];
            assert!(
                (out_weighted_naive[i] - expected).abs() < 1e-5,
                "naive weighted[{i}] mismatch: {} vs {}",
                out_weighted_naive[i], expected,
            );
        }

        // Verify SIMD matches naive for both
        assert_close(&out_unit, &out_unit_naive, 1e-5, "rmsnorm unit weight");
        assert_close(&out_weighted, &out_weighted_naive, 1e-5, "rmsnorm weighted");
    }

    #[test]
    fn rmsnorm_edge_hidden1_batch1() {
        let x = vec![3.0f32];
        let w = vec![2.0f32];
        let simd = rmsnorm(&x, &w, 1e-5, 1);
        let naive = rmsnorm_naive(&x, &w, 1e-5, 1);
        assert_close(&simd, &naive, 1e-5, "rmsnorm h=1 b=1");
    }

    #[test]
    fn headwise_rmsnorm_simd_vs_naive() {
        let (rows, heads, head_dim) = (4, 8, 128);
        let total = rows * heads * head_dim;
        let x = pseudo_rand(total, 42);
        let w = pseudo_rand(head_dim, 99).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let eps = 1e-5;

        let simd = apply_headwise_rmsnorm(&x, &w, rows, heads, head_dim, eps);
        let naive = apply_headwise_rmsnorm_naive(&x, &w, rows, heads, head_dim, eps);
        assert_close(&simd, &naive, 1e-5, "headwise rmsnorm");
    }

    // ── Benchmark: RMSNorm SIMD >= 4× throughput vs naive ───────────

    #[test]
    fn bench_rmsnorm_simd_vs_naive_throughput() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping rmsnorm throughput benchmark in debug mode");
            return;
        }

        let hidden = 1024;
        let batch = 1;
        let total = batch * hidden;
        let x = pseudo_rand(total, 42);
        let w = pseudo_rand(hidden, 99).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();

        // Warm up
        let _ = rmsnorm(&x, &w, 1e-5, hidden);
        let _ = rmsnorm_naive(&x, &w, 1e-5, hidden);

        let iters = 500;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = rmsnorm_naive(&x, &w, 1e-5, hidden);
        }
        let naive_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = rmsnorm(&x, &w, 1e-5, hidden);
        }
        let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = naive_ns / simd_ns;
        eprintln!(
            "rmsnorm [{batch},{hidden}]: naive={naive_ns:.0}ns, \
             simd={simd_ns:.0}ns, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 4.0,
            "RMSNorm SIMD speedup {speedup:.1}× is below 4× threshold \
             (naive={naive_ns:.0}ns, simd={simd_ns:.0}ns)"
        );
    }

    // ── Benchmark: matmul SIMD >= 4× throughput vs naive ────────────

    #[test]
    fn bench_simd_vs_naive_throughput() {
        // Only meaningful in release mode — skip in debug to avoid timeout.
        if cfg!(debug_assertions) {
            eprintln!("Skipping throughput benchmark in debug mode");
            return;
        }

        let (m, k, n) = (1024, 1024, 3072);
        let a = pseudo_rand(m * k, 42);
        let b = pseudo_rand(n * k, 99);

        // Warm up
        let _ = matmul_t(&a, &b, m, k, n);
        let _ = matmul_t_naive(&a, &b, m, k, n);

        let iters = 5;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_t_naive(&a, &b, m, k, n);
        }
        let naive_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_t(&a, &b, m, k, n);
        }
        let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = naive_ns / simd_ns;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let naive_gflops = flops / naive_ns;
        let simd_gflops = flops / simd_ns;

        eprintln!(
            "matmul_t [{m},{k}]×[{k},{n}]: naive={naive_gflops:.2} GFLOP/s, \
             simd={simd_gflops:.2} GFLOP/s, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 4.0,
            "SIMD speedup {speedup:.1}× is below 4× threshold \
             (naive={naive_gflops:.2}, simd={simd_gflops:.2} GFLOP/s)"
        );
    }

    // ── SwiGLU fused vs manual correctness ──────────────────────────

    /// Manual silu(gate)*up reference (scalar, not using FFI).
    fn swiglu_manual(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect()
    }

    /// Fused SwiGLU via syn_swiglu FFI.
    fn swiglu_fused(gate: &[f32], up: &[f32]) -> Vec<f32> {
        let len = gate.len();
        let mut out = vec![0.0f32; len];
        let status = unsafe {
            synapse_sys::syn_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len)
        };
        assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
        out
    }

    #[test]
    fn swiglu_fused_vs_manual_1024() {
        let gate = pseudo_rand(1024, 42);
        let up = pseudo_rand(1024, 99);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [1024]");
    }

    #[test]
    fn swiglu_fused_vs_manual_3072() {
        let gate = pseudo_rand(3072, 7);
        let up = pseudo_rand(3072, 13);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [3072]");
    }

    #[test]
    fn swiglu_fused_vs_manual_11008() {
        let gate = pseudo_rand(11008, 314);
        let up = pseudo_rand(11008, 159);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [11008]");
    }

    // ── GeGLU / GELU paths still work ───────────────────────────────

    #[test]
    fn geglu_path_correctness() {
        let len = 1024;
        let gate = pseudo_rand(len, 55);
        let up = pseudo_rand(len, 77);
        let mut result = vec![0.0f32; len];
        for i in 0..len {
            result[i] = gelu(gate[i]) * up[i];
        }
        // Sanity: GeGLU output should differ from SwiGLU output
        let swiglu_result = swiglu_manual(&gate, &up);
        let differs = result.iter().zip(swiglu_result.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(differs, "GeGLU and SwiGLU outputs should differ");

        // Verify GeGLU values are reasonable (not NaN/Inf)
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "GeGLU[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn gelu_path_correctness() {
        let len = 1024;
        let x = pseudo_rand(len, 42);
        let mut activated = x.clone();
        for v in activated.iter_mut() {
            *v = gelu(*v);
        }
        // GELU should be approximately x for large positive x, ~0 for large negative x
        for (i, (&orig, &act)) in x.iter().zip(activated.iter()).enumerate() {
            assert!(act.is_finite(), "GELU[{i}] is not finite: {act}");
            if orig > 3.0 {
                // For large positive inputs, gelu(x) ≈ x
                assert!(
                    (act - orig).abs() < 0.1,
                    "GELU[{i}]: expected ~{orig}, got {act}"
                );
            }
            if orig < -3.0 {
                // For large negative inputs, gelu(x) ≈ 0
                assert!(act.abs() < 0.1, "GELU[{i}]: expected ~0, got {act}");
            }
        }
    }

    // ── Benchmark: syn_swiglu >= 2× throughput vs separate silu+mul ─

    /// Scalar SwiGLU: silu(gate) * up, one element at a time.
    /// Uses `black_box` on the exp input to prevent LLVM from batching
    /// multiple exp() calls into SIMD — same strategy as `rmsnorm_naive`.
    #[inline(never)]
    fn swiglu_separate_scalar(dst: &mut [f32], gate: &[f32], up: &[f32]) {
        for i in 0..dst.len() {
            let g = gate[i];
            let neg_g = std::hint::black_box(-g);
            let s = g / (1.0 + neg_g.exp());
            dst[i] = std::hint::black_box(s) * up[i];
        }
    }

    #[test]
    fn bench_swiglu_fused_vs_separate_throughput() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping swiglu throughput benchmark in debug mode");
            return;
        }

        let len = 3072; // [1, 3072] single-token FFN intermediate
        let gate = pseudo_rand(len, 42);
        let up = pseudo_rand(len, 99);
        let mut dst = vec![0.0f32; len];

        // Warm up
        for _ in 0..100 {
            swiglu_separate_scalar(&mut dst, &gate, &up);
            unsafe {
                synapse_sys::syn_swiglu(dst.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len);
            }
        }

        // Use min-of-runs: noise only adds latency, minimum is most
        // representative (standard microbenchmark practice).
        let runs = 5;
        let iters_per_run = 2000;

        let mut best_separate = f64::MAX;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..iters_per_run {
                swiglu_separate_scalar(&mut dst, &gate, &up);
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters_per_run as f64;
            if ns < best_separate {
                best_separate = ns;
            }
        }

        let mut best_fused = f64::MAX;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..iters_per_run {
                unsafe {
                    synapse_sys::syn_swiglu(
                        dst.as_mut_ptr(),
                        gate.as_ptr(),
                        up.as_ptr(),
                        len,
                    );
                }
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters_per_run as f64;
            if ns < best_fused {
                best_fused = ns;
            }
        }

        let speedup = best_separate / best_fused;
        eprintln!(
            "swiglu [1,{len}]: separate={best_separate:.0}ns, \
             fused={best_fused:.0}ns, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 2.0,
            "syn_swiglu speedup {speedup:.1}× is below 2× threshold \
             (separate={best_separate:.0}ns, fused={best_fused:.0}ns)"
        );
    }
}
