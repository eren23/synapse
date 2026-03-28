//! DeltaNet and GQA decoder layers for hybrid models (Qwen3.5).
//!
//! Each layer type is self-contained with its own weights and forward methods,
//! avoiding coupling with the transformer model builder infrastructure.

use crate::ops::activation::{silu, softmax_slice};
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use crate::ssm::deltanet::{deltanet_seq, deltanet_step, l2_normalize};
use crate::ssm::deltanet_state::DeltaNetLayerState;

// ---------------------------------------------------------------------------
// DeltaNet Decoder Layer
// ---------------------------------------------------------------------------

/// A single DeltaNet (gated linear attention) decoder layer.
///
/// Forward path:
/// 1. RMSNorm
/// 2. QKV projection (combined weight)
/// 3. Conv1d on Q, K, V
/// 4. L2 normalise Q, K
/// 5. Compute alpha (decay) and beta (update) from separate projections
/// 6. DeltaNet recurrence per head
/// 7. Output norm + SiLU gate
/// 8. Output projection
/// 9. Residual
/// 10. RMSNorm -> SwiGLU FFN -> Residual
pub struct DeltaNetDecoderLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub conv_kernel: usize,
    pub norm_eps: f32,

    // Attention norm
    pub attn_norm_weight: Vec<f32>, // [hidden_size]

    // Combined QKV projection: [3 * num_heads * head_dim, hidden_size]
    pub qkv_weight: Vec<f32>,

    // Gate/beta/alpha projections from hidden_size
    pub gate_proj_weight: Vec<f32>,  // [num_heads * head_dim, hidden_size] (z gate)
    pub beta_proj_weight: Vec<f32>,  // [num_heads, hidden_size] (update gate)
    pub alpha_proj_weight: Vec<f32>, // [num_heads, hidden_size] (decay gate)

    // Conv1d weights for Q, K, V
    pub q_conv_weight: Vec<f32>, // [num_heads * head_dim, conv_kernel]
    pub q_conv_bias: Vec<f32>,   // [num_heads * head_dim]
    pub k_conv_weight: Vec<f32>,
    pub k_conv_bias: Vec<f32>,
    pub v_conv_weight: Vec<f32>,
    pub v_conv_bias: Vec<f32>,

    // Output norm + projection
    pub o_norm_weight: Vec<f32>, // [num_heads * head_dim] (RMSNorm)
    pub o_proj_weight: Vec<f32>, // [hidden_size, num_heads * head_dim]

    // FFN norm
    pub ffn_norm_weight: Vec<f32>, // [hidden_size]

    // SwiGLU FFN
    pub ffn_gate_weight: Vec<f32>, // [intermediate_size, hidden_size]
    pub ffn_up_weight: Vec<f32>,   // [intermediate_size, hidden_size]
    pub ffn_down_weight: Vec<f32>, // [hidden_size, intermediate_size]
}

impl DeltaNetDecoderLayer {
    /// Apply conv1d to a single channel `x[i]`, shifting the state and computing
    /// the dot product with the kernel.
    ///
    /// `conv_state` layout: `[channels, conv_kernel]`.
    #[inline]
    fn conv1d_step_single(
        x_val: f32,
        channel: usize,
        conv_state: &mut [f32],
        conv_weight: &[f32],
        conv_bias: &[f32],
        conv_kernel: usize,
    ) -> f32 {
        let buf = &mut conv_state[channel * conv_kernel..(channel + 1) * conv_kernel];
        // Shift left
        buf.copy_within(1.., 0);
        // Insert new value
        buf[conv_kernel - 1] = x_val;
        // Dot product with kernel
        let w = &conv_weight[channel * conv_kernel..(channel + 1) * conv_kernel];
        let sum: f32 = buf.iter().zip(w.iter()).map(|(&b, &k)| b * k).sum();
        sum + conv_bias[channel]
    }

    /// Apply conv1d to all channels for a single timestep, using the rolling
    /// state buffers in `layer_state`.
    fn conv1d_step_all(
        q_raw: &[f32],
        k_raw: &[f32],
        v_raw: &[f32],
        state: &mut DeltaNetLayerState,
        q_conv_weight: &[f32],
        q_conv_bias: &[f32],
        k_conv_weight: &[f32],
        k_conv_bias: &[f32],
        v_conv_weight: &[f32],
        v_conv_bias: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let channels = q_raw.len();
        let ck = state.conv_kernel;
        let mut q_out = vec![0.0f32; channels];
        let mut k_out = vec![0.0f32; channels];
        let mut v_out = vec![0.0f32; channels];

        for i in 0..channels {
            q_out[i] = Self::conv1d_step_single(
                q_raw[i],
                i,
                &mut state.q_conv_state,
                q_conv_weight,
                q_conv_bias,
                ck,
            );
            k_out[i] = Self::conv1d_step_single(
                k_raw[i],
                i,
                &mut state.k_conv_state,
                k_conv_weight,
                k_conv_bias,
                ck,
            );
            v_out[i] = Self::conv1d_step_single(
                v_raw[i],
                i,
                &mut state.v_conv_state,
                v_conv_weight,
                v_conv_bias,
                ck,
            );
        }
        (q_out, k_out, v_out)
    }

    /// Single-token forward pass for the DeltaNet attention sub-block.
    ///
    /// Returns the attention output vector `[hidden_size]`.
    fn attn_forward_one(&self, normed: &[f32], state: &mut DeltaNetLayerState) -> Vec<f32> {
        let nh = self.num_heads;
        let hd = self.head_dim;
        let total_qkv_dim = 3 * nh * hd;

        // 1. QKV projection: [1, hidden_size] x [3*nh*hd, hidden_size]^T -> [3*nh*hd]
        let qkv = matmul_t(normed, &self.qkv_weight, 1, self.hidden_size, total_qkv_dim);
        let q_raw = &qkv[0..nh * hd];
        let k_raw = &qkv[nh * hd..2 * nh * hd];
        let v_raw = &qkv[2 * nh * hd..3 * nh * hd];

        // 2. Conv1d step on Q, K, V
        let (q_conv, k_conv, v_conv) = Self::conv1d_step_all(
            q_raw,
            k_raw,
            v_raw,
            state,
            &self.q_conv_weight,
            &self.q_conv_bias,
            &self.k_conv_weight,
            &self.k_conv_bias,
            &self.v_conv_weight,
            &self.v_conv_bias,
        );

        // 3. Compute gate (z), alpha, beta from the normed input
        let gate_vec = matmul_t(normed, &self.gate_proj_weight, 1, self.hidden_size, nh * hd);
        let alpha_raw = matmul_t(normed, &self.alpha_proj_weight, 1, self.hidden_size, nh);
        let beta_raw = matmul_t(normed, &self.beta_proj_weight, 1, self.hidden_size, nh);

        // 4. Per-head DeltaNet step
        let mut attn_out = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let q_h = l2_normalize(&q_conv[h * hd..(h + 1) * hd]);
            let k_h = l2_normalize(&k_conv[h * hd..(h + 1) * hd]);
            let v_h = &v_conv[h * hd..(h + 1) * hd];

            // Sigmoid gates
            let alpha = 1.0 / (1.0 + (-alpha_raw[h]).exp());
            let beta = 1.0 / (1.0 + (-beta_raw[h]).exp());

            let memory =
                &mut state.memory[h * hd * hd..(h + 1) * hd * hd];
            let o_h = deltanet_step(&q_h, &k_h, v_h, alpha, beta, memory, hd);
            attn_out[h * hd..(h + 1) * hd].copy_from_slice(&o_h);
        }

        // 5. Output norm (per-head RMSNorm over the concatenated heads)
        let normed_out = rmsnorm(&attn_out, &self.o_norm_weight, self.norm_eps, nh * hd);

        // 6. SiLU gate: element-wise silu(gate) * normed_out
        let gated: Vec<f32> = normed_out
            .iter()
            .zip(gate_vec.iter())
            .map(|(&o, &g)| o * silu(g))
            .collect();

        // 7. Output projection: [1, nh*hd] x [hidden_size, nh*hd]^T -> [hidden_size]
        matmul_t(&gated, &self.o_proj_weight, 1, nh * hd, self.hidden_size)
    }

    /// Sequence forward pass for the DeltaNet attention sub-block.
    ///
    /// Processes `seq_len` tokens and returns `[seq_len * hidden_size]`.
    fn attn_forward_seq(
        &self,
        normed: &[f32],
        seq_len: usize,
        state: &mut DeltaNetLayerState,
    ) -> Vec<f32> {
        let nh = self.num_heads;
        let hd = self.head_dim;
        let total_qkv_dim = 3 * nh * hd;

        // 1. QKV projection: [seq_len, hidden_size] x [3*nh*hd, hidden_size]^T
        let qkv = matmul_t(
            normed,
            &self.qkv_weight,
            seq_len,
            self.hidden_size,
            total_qkv_dim,
        );

        // 2. Conv1d + L2 normalize per timestep (sequential due to conv state)
        let mut q_all = vec![0.0f32; seq_len * nh * hd];
        let mut k_all = vec![0.0f32; seq_len * nh * hd];
        let mut v_all = vec![0.0f32; seq_len * nh * hd];

        for t in 0..seq_len {
            let off = t * total_qkv_dim;
            let q_raw = &qkv[off..off + nh * hd];
            let k_raw = &qkv[off + nh * hd..off + 2 * nh * hd];
            let v_raw = &qkv[off + 2 * nh * hd..off + 3 * nh * hd];

            let (q_conv, k_conv, v_conv) = Self::conv1d_step_all(
                q_raw,
                k_raw,
                v_raw,
                state,
                &self.q_conv_weight,
                &self.q_conv_bias,
                &self.k_conv_weight,
                &self.k_conv_bias,
                &self.v_conv_weight,
                &self.v_conv_bias,
            );

            let t_off = t * nh * hd;
            q_all[t_off..t_off + nh * hd].copy_from_slice(&q_conv);
            k_all[t_off..t_off + nh * hd].copy_from_slice(&k_conv);
            v_all[t_off..t_off + nh * hd].copy_from_slice(&v_conv);
        }

        // 3. Compute gate, alpha, beta for all timesteps
        let gate_all = matmul_t(
            normed,
            &self.gate_proj_weight,
            seq_len,
            self.hidden_size,
            nh * hd,
        );
        let alpha_all = matmul_t(
            normed,
            &self.alpha_proj_weight,
            seq_len,
            self.hidden_size,
            nh,
        );
        let beta_all = matmul_t(
            normed,
            &self.beta_proj_weight,
            seq_len,
            self.hidden_size,
            nh,
        );

        // 4. Per-head DeltaNet sequence
        // We need to reshape from [seq_len, nh*hd] interleaved to per-head
        // [seq_len, hd] before calling deltanet_seq.
        let mut attn_out = vec![0.0f32; seq_len * nh * hd];

        for h in 0..nh {
            // Extract per-head Q, K, V: [seq_len * hd]
            let mut q_h = vec![0.0f32; seq_len * hd];
            let mut k_h = vec![0.0f32; seq_len * hd];
            let mut v_h = vec![0.0f32; seq_len * hd];
            let mut alpha_h = vec![0.0f32; seq_len];
            let mut beta_h = vec![0.0f32; seq_len];

            for t in 0..seq_len {
                // L2 normalize Q and K per-head per-timestep
                let q_slice = &q_all[t * nh * hd + h * hd..t * nh * hd + (h + 1) * hd];
                let k_slice = &k_all[t * nh * hd + h * hd..t * nh * hd + (h + 1) * hd];
                let v_slice = &v_all[t * nh * hd + h * hd..t * nh * hd + (h + 1) * hd];

                let q_norm = l2_normalize(q_slice);
                let k_norm = l2_normalize(k_slice);

                q_h[t * hd..(t + 1) * hd].copy_from_slice(&q_norm);
                k_h[t * hd..(t + 1) * hd].copy_from_slice(&k_norm);
                v_h[t * hd..(t + 1) * hd].copy_from_slice(v_slice);

                // Sigmoid gates
                alpha_h[t] = 1.0 / (1.0 + (-alpha_all[t * nh + h]).exp());
                beta_h[t] = 1.0 / (1.0 + (-beta_all[t * nh + h]).exp());
            }

            let memory = &mut state.memory[h * hd * hd..(h + 1) * hd * hd];
            let o_h = deltanet_seq(&q_h, &k_h, &v_h, &alpha_h, &beta_h, memory, seq_len, hd);

            // Write back interleaved
            for t in 0..seq_len {
                let dst = t * nh * hd + h * hd;
                attn_out[dst..dst + hd].copy_from_slice(&o_h[t * hd..(t + 1) * hd]);
            }
        }

        // 5. Output norm + SiLU gate + output projection, per timestep
        let mut out = vec![0.0f32; seq_len * self.hidden_size];
        for t in 0..seq_len {
            let t_off = t * nh * hd;
            let attn_t = &attn_out[t_off..t_off + nh * hd];

            // Output norm
            let normed_t = rmsnorm(attn_t, &self.o_norm_weight, self.norm_eps, nh * hd);

            // SiLU gate
            let gate_t = &gate_all[t_off..t_off + nh * hd];
            let gated: Vec<f32> = normed_t
                .iter()
                .zip(gate_t.iter())
                .map(|(&o, &g)| o * silu(g))
                .collect();

            // Output projection
            let proj = matmul_t(&gated, &self.o_proj_weight, 1, nh * hd, self.hidden_size);
            let dst = t * self.hidden_size;
            out[dst..dst + self.hidden_size].copy_from_slice(&proj);
        }

        out
    }

    /// SwiGLU FFN: gate_proj -> SiLU, up_proj, element multiply, down_proj.
    fn ffn_forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let im = self.intermediate_size;

        // Norm
        let normed = rmsnorm(x, &self.ffn_norm_weight, self.norm_eps, h);

        // Gate and up projections
        let gate = matmul_t(&normed, &self.ffn_gate_weight, seq_len, h, im);
        let up = matmul_t(&normed, &self.ffn_up_weight, seq_len, h, im);

        // SiLU(gate) * up
        let fused: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        // Down projection
        matmul_t(&fused, &self.ffn_down_weight, seq_len, im, h)
    }

    /// Process a single token through this DeltaNet layer.
    ///
    /// `hidden`: input `[hidden_size]`.
    /// Returns output `[hidden_size]` with residual connections.
    pub fn forward_one(&self, hidden: &[f32], state: &mut DeltaNetLayerState) -> Vec<f32> {
        // 1. Attention sub-block
        let normed = rmsnorm(hidden, &self.attn_norm_weight, self.norm_eps, self.hidden_size);
        let attn_out = self.attn_forward_one(&normed, state);

        // Residual
        let after_attn: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(&h, &a)| h + a)
            .collect();

        // 2. FFN sub-block
        let ffn_out = self.ffn_forward(&after_attn, 1);

        // Residual
        after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(&h, &f)| h + f)
            .collect()
    }

    /// Process a sequence of tokens through this DeltaNet layer.
    ///
    /// `hidden`: input `[seq_len * hidden_size]`.
    /// Returns output `[seq_len * hidden_size]` with residual connections.
    pub fn forward_seq(
        &self,
        hidden: &[f32],
        seq_len: usize,
        state: &mut DeltaNetLayerState,
    ) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-block
        let normed = rmsnorm(hidden, &self.attn_norm_weight, self.norm_eps, h);
        let attn_out = self.attn_forward_seq(&normed, seq_len, state);

        // Residual
        let after_attn: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(&h, &a)| h + a)
            .collect();

        // 2. FFN sub-block
        let ffn_out = self.ffn_forward(&after_attn, seq_len);

        // Residual
        after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(&h, &f)| h + f)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GQA Decoder Layer
// ---------------------------------------------------------------------------

/// State for a single GQA layer's KV cache.
pub struct KvLayerState {
    /// Key cache: `[max_seq * kv_dim]` (flat, row-major).
    pub k_cache: Vec<f32>,
    /// Value cache: `[max_seq * kv_dim]` (flat, row-major).
    pub v_cache: Vec<f32>,
    /// Maximum sequence length the cache can hold.
    pub max_seq: usize,
    /// Dimension per row: `num_kv_heads * head_dim`.
    pub kv_dim: usize,
    /// Current number of cached tokens.
    pub len: usize,
}

impl KvLayerState {
    pub fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        KvLayerState {
            k_cache: vec![0.0f32; max_seq * kv_dim],
            v_cache: vec![0.0f32; max_seq * kv_dim],
            max_seq,
            kv_dim,
            len: 0,
        }
    }

    /// Append a single token's K and V to the cache.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert!(self.len < self.max_seq, "KV cache overflow");
        let off = self.len * self.kv_dim;
        self.k_cache[off..off + self.kv_dim].copy_from_slice(k);
        self.v_cache[off..off + self.kv_dim].copy_from_slice(v);
        self.len += 1;
    }

    /// Append multiple tokens' K and V to the cache.
    pub fn append_seq(&mut self, k: &[f32], v: &[f32], seq_len: usize) {
        debug_assert!(
            self.len + seq_len <= self.max_seq,
            "KV cache overflow: {} + {} > {}",
            self.len,
            seq_len,
            self.max_seq
        );
        let off = self.len * self.kv_dim;
        let size = seq_len * self.kv_dim;
        self.k_cache[off..off + size].copy_from_slice(&k[..size]);
        self.v_cache[off..off + size].copy_from_slice(&v[..size]);
        self.len += seq_len;
    }

    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.len = 0;
    }

    /// Memory in bytes used by this cache entry.
    pub fn memory_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * 4
    }
}

/// A single GQA (Grouped Query Attention) decoder layer.
///
/// Uses standard softmax attention with RoPE and a KV cache. Self-contained
/// so it does not depend on the transformer model builder infrastructure.
pub struct GqaDecoderLayer {
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,

    // Attention norm
    pub attn_norm_weight: Vec<f32>, // [hidden_size]

    // Q, K, V projections
    pub w_q: Vec<f32>, // [num_q_heads * head_dim, hidden_size]
    pub w_k: Vec<f32>, // [num_kv_heads * head_dim, hidden_size]
    pub w_v: Vec<f32>, // [num_kv_heads * head_dim, hidden_size]
    pub w_o: Vec<f32>, // [hidden_size, num_q_heads * head_dim]

    // Per-head Q/K norms (Qwen3 style)
    pub q_norm_weight: Vec<f32>, // [head_dim]
    pub k_norm_weight: Vec<f32>, // [head_dim]

    // FFN
    pub ffn_norm_weight: Vec<f32>,  // [hidden_size]
    pub ffn_gate_weight: Vec<f32>,  // [intermediate_size, hidden_size]
    pub ffn_up_weight: Vec<f32>,    // [intermediate_size, hidden_size]
    pub ffn_down_weight: Vec<f32>,  // [hidden_size, intermediate_size]
}

impl GqaDecoderLayer {
    /// Apply RoPE rotation to a single head's Q or K vector in-place.
    ///
    /// Uses the rotate-half convention: pairs `(i, i + head_dim/2)`.
    fn apply_rope_head(
        head_vec: &mut [f32],
        head_dim: usize,
        cos_row: &[f32],
        sin_row: &[f32],
    ) {
        let half_d = head_dim / 2;
        for i in 0..half_d {
            let x_first = head_vec[i];
            let x_second = head_vec[half_d + i];
            head_vec[i] = x_first * cos_row[i] - x_second * sin_row[i];
            head_vec[half_d + i] = x_second * cos_row[i] + x_first * sin_row[i];
        }
    }

    /// Per-head RMSNorm (Qwen3 style): norm each head's vector with the
    /// shared `weight` of length `head_dim`.
    fn head_rmsnorm(data: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        let num_heads = data.len() / head_dim;
        for h in 0..num_heads {
            let off = h * head_dim;
            let sum_sq: f32 = data[off..off + head_dim].iter().map(|v| v * v).sum();
            let scale = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
            for j in 0..head_dim {
                data[off + j] *= weight[j] * scale;
            }
        }
    }

    /// Single-token forward through the GQA attention sub-block.
    fn attn_forward_one(
        &self,
        normed: &[f32],
        kv_state: &mut KvLayerState,
        rope_cos: &[f32],
        rope_sin: &[f32],
        position: usize,
    ) -> Vec<f32> {
        let nq = self.num_q_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let half_d = hd / 2;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;

        // Project Q, K, V
        let mut q = matmul_t(normed, &self.w_q, 1, self.hidden_size, q_dim);
        let mut k = matmul_t(normed, &self.w_k, 1, self.hidden_size, kv_dim);
        let v = matmul_t(normed, &self.w_v, 1, self.hidden_size, kv_dim);

        // Per-head Q/K norms
        Self::head_rmsnorm(&mut q, &self.q_norm_weight, hd, self.norm_eps);
        Self::head_rmsnorm(&mut k, &self.k_norm_weight, hd, self.norm_eps);

        // RoPE
        let cos_row = &rope_cos[position * half_d..(position + 1) * half_d];
        let sin_row = &rope_sin[position * half_d..(position + 1) * half_d];
        for h in 0..nq {
            Self::apply_rope_head(&mut q[h * hd..(h + 1) * hd], hd, cos_row, sin_row);
        }
        for h in 0..nkv {
            Self::apply_rope_head(&mut k[h * hd..(h + 1) * hd], hd, cos_row, sin_row);
        }

        // Append to KV cache
        kv_state.append(&k, &v);
        let seq_kv = kv_state.len;

        // GQA: each Q head group shares a KV head
        let heads_per_group = nq / nkv;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out = vec![0.0f32; q_dim];

        for qh in 0..nq {
            let kv_h = qh / heads_per_group;
            let q_slice = &q[qh * hd..(qh + 1) * hd];

            // Compute attention scores: Q @ K^T for all cached positions
            let mut scores = vec![0.0f32; seq_kv];
            for pos in 0..seq_kv {
                let k_pos = &kv_state.k_cache[pos * kv_dim + kv_h * hd..pos * kv_dim + (kv_h + 1) * hd];
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_slice[d] * k_pos[d];
                }
                scores[pos] = dot * scale;
            }

            // Softmax
            softmax_slice(&mut scores);

            // Weighted sum of V
            for d in 0..hd {
                let mut val = 0.0f32;
                for pos in 0..seq_kv {
                    val += scores[pos]
                        * kv_state.v_cache[pos * kv_dim + kv_h * hd + d];
                }
                attn_out[qh * hd + d] = val;
            }
        }

        // Output projection
        matmul_t(&attn_out, &self.w_o, 1, q_dim, self.hidden_size)
    }

    /// Sequence (prefill) forward through the GQA attention sub-block.
    fn attn_forward_seq(
        &self,
        normed: &[f32],
        seq_len: usize,
        kv_state: &mut KvLayerState,
        rope_cos: &[f32],
        rope_sin: &[f32],
        pos_offset: usize,
    ) -> Vec<f32> {
        let nq = self.num_q_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let half_d = hd / 2;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;

        // Project Q, K, V for all tokens
        let mut q = matmul_t(normed, &self.w_q, seq_len, self.hidden_size, q_dim);
        let mut k = matmul_t(normed, &self.w_k, seq_len, self.hidden_size, kv_dim);
        let v = matmul_t(normed, &self.w_v, seq_len, self.hidden_size, kv_dim);

        // Per-head norms + RoPE per token
        for t in 0..seq_len {
            let pos = pos_offset + t;
            let cos_row = &rope_cos[pos * half_d..(pos + 1) * half_d];
            let sin_row = &rope_sin[pos * half_d..(pos + 1) * half_d];

            // Q heads norm + RoPE
            for h in 0..nq {
                let off = t * q_dim + h * hd;
                let sum_sq: f32 = q[off..off + hd].iter().map(|v| v * v).sum();
                let scale = 1.0 / (sum_sq / hd as f32 + self.norm_eps).sqrt();
                for j in 0..hd {
                    q[off + j] *= self.q_norm_weight[j] * scale;
                }
                Self::apply_rope_head(&mut q[off..off + hd], hd, cos_row, sin_row);
            }

            // KV heads norm + RoPE
            for h in 0..nkv {
                let off = t * kv_dim + h * hd;
                let sum_sq: f32 = k[off..off + hd].iter().map(|v| v * v).sum();
                let scale = 1.0 / (sum_sq / hd as f32 + self.norm_eps).sqrt();
                for j in 0..hd {
                    k[off + j] *= self.k_norm_weight[j] * scale;
                }
                Self::apply_rope_head(&mut k[off..off + hd], hd, cos_row, sin_row);
            }
        }

        // Append all tokens to KV cache
        kv_state.append_seq(&k, &v, seq_len);
        let seq_kv = kv_state.len;

        // GQA attention with causal mask
        let heads_per_group = nq / nkv;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for qh in 0..nq {
            let kv_h = qh / heads_per_group;

            for qi in 0..seq_len {
                // Causal: can attend up to position (pos_offset + qi) in the
                // KV cache. Since we just appended seq_len tokens starting at
                // pos_offset, the causal bound is pos_offset + qi + 1.
                let causal_len = (pos_offset + qi + 1).min(seq_kv);
                let q_off = qi * q_dim + qh * hd;
                let q_slice = &q[q_off..q_off + hd];

                let mut scores = vec![0.0f32; causal_len];
                for pos in 0..causal_len {
                    let k_pos_off = pos * kv_dim + kv_h * hd;
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q_slice[d] * kv_state.k_cache[k_pos_off + d];
                    }
                    scores[pos] = dot * scale;
                }

                softmax_slice(&mut scores);

                for d in 0..hd {
                    let mut val = 0.0f32;
                    for pos in 0..causal_len {
                        val += scores[pos]
                            * kv_state.v_cache[pos * kv_dim + kv_h * hd + d];
                    }
                    attn_out[qi * q_dim + qh * hd + d] = val;
                }
            }
        }

        // Output projection
        matmul_t(&attn_out, &self.w_o, seq_len, q_dim, self.hidden_size)
    }

    /// SwiGLU FFN sub-block.
    fn ffn_forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let im = self.intermediate_size;

        let normed = rmsnorm(x, &self.ffn_norm_weight, self.norm_eps, h);
        let gate = matmul_t(&normed, &self.ffn_gate_weight, seq_len, h, im);
        let up = matmul_t(&normed, &self.ffn_up_weight, seq_len, h, im);

        let fused: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        matmul_t(&fused, &self.ffn_down_weight, seq_len, im, h)
    }

    /// Process a single token through this GQA layer.
    pub fn forward_one(
        &self,
        hidden: &[f32],
        kv_state: &mut KvLayerState,
        rope_cos: &[f32],
        rope_sin: &[f32],
        position: usize,
    ) -> Vec<f32> {
        // Attention sub-block
        let normed = rmsnorm(hidden, &self.attn_norm_weight, self.norm_eps, self.hidden_size);
        let attn_out = self.attn_forward_one(&normed, kv_state, rope_cos, rope_sin, position);

        // Residual
        let after_attn: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(&h, &a)| h + a)
            .collect();

        // FFN sub-block
        let ffn_out = self.ffn_forward(&after_attn, 1);

        // Residual
        after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(&h, &f)| h + f)
            .collect()
    }

    /// Process a sequence of tokens through this GQA layer (prefill).
    pub fn forward_seq(
        &self,
        hidden: &[f32],
        seq_len: usize,
        kv_state: &mut KvLayerState,
        rope_cos: &[f32],
        rope_sin: &[f32],
        pos_offset: usize,
    ) -> Vec<f32> {
        let h = self.hidden_size;

        // Attention sub-block
        let normed = rmsnorm(hidden, &self.attn_norm_weight, self.norm_eps, h);
        let attn_out = self.attn_forward_seq(
            &normed, seq_len, kv_state, rope_cos, rope_sin, pos_offset,
        );

        // Residual
        let after_attn: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(&h, &a)| h + a)
            .collect();

        // FFN sub-block
        let ffn_out = self.ffn_forward(&after_attn, seq_len);

        // Residual
        after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(&h, &f)| h + f)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
                (f32::from_bits(bits) - 1.5) * 0.2
            })
            .collect()
    }

    fn make_deltanet_layer() -> DeltaNetDecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = 16;
        let intermediate_size = 128;
        let conv_kernel = 4;
        let nh_hd = num_heads * head_dim;

        DeltaNetDecoderLayer {
            hidden_size,
            num_heads,
            head_dim,
            intermediate_size,
            conv_kernel,
            norm_eps: 1e-6,
            attn_norm_weight: vec![1.0; hidden_size],
            qkv_weight: pseudo_random_vec(1, 3 * nh_hd * hidden_size),
            gate_proj_weight: pseudo_random_vec(2, nh_hd * hidden_size),
            beta_proj_weight: pseudo_random_vec(3, num_heads * hidden_size),
            alpha_proj_weight: pseudo_random_vec(4, num_heads * hidden_size),
            q_conv_weight: pseudo_random_vec(5, nh_hd * conv_kernel),
            q_conv_bias: vec![0.0; nh_hd],
            k_conv_weight: pseudo_random_vec(6, nh_hd * conv_kernel),
            k_conv_bias: vec![0.0; nh_hd],
            v_conv_weight: pseudo_random_vec(7, nh_hd * conv_kernel),
            v_conv_bias: vec![0.0; nh_hd],
            o_norm_weight: vec![1.0; nh_hd],
            o_proj_weight: pseudo_random_vec(8, hidden_size * nh_hd),
            ffn_norm_weight: vec![1.0; hidden_size],
            ffn_gate_weight: pseudo_random_vec(9, intermediate_size * hidden_size),
            ffn_up_weight: pseudo_random_vec(10, intermediate_size * hidden_size),
            ffn_down_weight: pseudo_random_vec(11, hidden_size * intermediate_size),
        }
    }

    fn make_gqa_layer() -> GqaDecoderLayer {
        let hidden_size = 64;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let intermediate_size = 128;

        GqaDecoderLayer {
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            norm_eps: 1e-6,
            attn_norm_weight: vec![1.0; hidden_size],
            w_q: pseudo_random_vec(20, num_q_heads * head_dim * hidden_size),
            w_k: pseudo_random_vec(21, num_kv_heads * head_dim * hidden_size),
            w_v: pseudo_random_vec(22, num_kv_heads * head_dim * hidden_size),
            w_o: pseudo_random_vec(23, hidden_size * num_q_heads * head_dim),
            q_norm_weight: vec![1.0; head_dim],
            k_norm_weight: vec![1.0; head_dim],
            ffn_norm_weight: vec![1.0; hidden_size],
            ffn_gate_weight: pseudo_random_vec(24, intermediate_size * hidden_size),
            ffn_up_weight: pseudo_random_vec(25, intermediate_size * hidden_size),
            ffn_down_weight: pseudo_random_vec(26, hidden_size * intermediate_size),
        }
    }

    #[test]
    fn deltanet_layer_forward_one_produces_finite_output() {
        let layer = make_deltanet_layer();
        let mut state = DeltaNetLayerState::new(4, 16, 4);
        let input = pseudo_random_vec(100, 64);

        let output = layer.forward_one(&input, &mut state);

        assert_eq!(output.len(), 64);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn deltanet_layer_forward_seq_produces_finite_output() {
        let layer = make_deltanet_layer();
        let mut state = DeltaNetLayerState::new(4, 16, 4);
        let seq_len = 3;
        let input = pseudo_random_vec(101, seq_len * 64);

        let output = layer.forward_seq(&input, seq_len, &mut state);

        assert_eq!(output.len(), seq_len * 64);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    fn make_rope_tables(max_pos: usize, half_d: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cos = vec![0.0f32; max_pos * half_d];
        let mut sin = vec![0.0f32; max_pos * half_d];
        let head_dim = half_d * 2;
        for pos in 0..max_pos {
            for i in 0..half_d {
                let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_d + i] = angle.cos();
                sin[pos * half_d + i] = angle.sin();
            }
        }
        (cos, sin)
    }

    #[test]
    fn gqa_layer_forward_one_produces_finite_output() {
        let layer = make_gqa_layer();
        let mut kv = KvLayerState::new(32, 2, 16);
        let (cos, sin) = make_rope_tables(32, 8);
        let input = pseudo_random_vec(200, 64);

        let output = layer.forward_one(&input, &mut kv, &cos, &sin, 0);

        assert_eq!(output.len(), 64);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
        assert_eq!(kv.len, 1);
    }

    #[test]
    fn gqa_layer_forward_seq_produces_finite_output() {
        let layer = make_gqa_layer();
        let mut kv = KvLayerState::new(32, 2, 16);
        let (cos, sin) = make_rope_tables(32, 8);
        let seq_len = 5;
        let input = pseudo_random_vec(201, seq_len * 64);

        let output = layer.forward_seq(&input, seq_len, &mut kv, &cos, &sin, 0);

        assert_eq!(output.len(), seq_len * 64);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
        assert_eq!(kv.len, seq_len);
    }

    #[test]
    fn kv_cache_grows_with_tokens() {
        let layer = make_gqa_layer();
        let mut kv = KvLayerState::new(32, 2, 16);
        let (cos, sin) = make_rope_tables(32, 8);

        let input = pseudo_random_vec(300, 64);
        let _ = layer.forward_one(&input, &mut kv, &cos, &sin, 0);
        assert_eq!(kv.len, 1);

        let _ = layer.forward_one(&input, &mut kv, &cos, &sin, 1);
        assert_eq!(kv.len, 2);

        let _ = layer.forward_one(&input, &mut kv, &cos, &sin, 2);
        assert_eq!(kv.len, 3);
    }
}
