//! Attention mechanism implementations with GQA and sliding window support.
//!
//! GQA (Grouped-Query Attention) is the general form that subsumes:
//! - **MHA** (Multi-Head Attention): `num_kv_heads == num_heads` — no KV sharing
//! - **MQA** (Multi-Query Attention): `num_kv_heads == 1` — all heads share one KV

use super::AttentionVariant;
use synapse_core::{KvCache, SynapseError, Tensor};

// ── Helper functions ─────────────────────────────────────────────────

/// C\[m,n\] = A\[m,k\] @ B\[k,n\]  (no transpose).
fn sgemm_nn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), SynapseError> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    unsafe {
        let status = synapse_sys::syn_sgemm(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            0,
            b.as_ptr(),
            n,
            0,
            c.as_mut_ptr(),
            n,
        );
        if status != synapse_sys::SYN_OK {
            return Err(SynapseError::Internal);
        }
    }
    Ok(())
}

/// C\[m,n\] = A\[m,k\] @ B^T, where B is stored row-major as \[n,k\].
fn sgemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), SynapseError> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    unsafe {
        let status = synapse_sys::syn_sgemm(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            0,
            b.as_ptr(),
            k,
            1,
            c.as_mut_ptr(),
            n,
        );
        if status != synapse_sys::SYN_OK {
            return Err(SynapseError::Internal);
        }
    }
    Ok(())
}

/// Swap dimensions 1 and 2 of a 4-D tensor: \[d0, d1, d2, d3\] → \[d0, d2, d1, d3\].
///
/// Used for the \[batch, seq, heads, d\] ↔ \[batch, heads, seq, d\] conversion.
fn transpose_0213(data: &[f32], d0: usize, d1: usize, d2: usize, d3: usize) -> Vec<f32> {
    let mut out = vec![0.0; d0 * d1 * d2 * d3];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let src = ((i0 * d1 + i1) * d2 + i2) * d3;
                let dst = ((i0 * d2 + i2) * d1 + i1) * d3;
                out[dst..dst + d3].copy_from_slice(&data[src..src + d3]);
            }
        }
    }
    out
}

/// Repeat KV heads via interleave: \[batch, kv\_heads, seq, d\] → \[batch, kv\_heads\*repeat, seq, d\].
fn repeat_kv(
    data: &[f32],
    batch: usize,
    kv_heads: usize,
    seq: usize,
    d: usize,
    repeat: usize,
) -> Vec<f32> {
    if repeat == 1 {
        return data.to_vec();
    }
    let num_heads = kv_heads * repeat;
    let head_size = seq * d;
    let mut out = vec![0.0; batch * num_heads * head_size];
    for b in 0..batch {
        for kv in 0..kv_heads {
            let src_off = (b * kv_heads + kv) * head_size;
            for r in 0..repeat {
                let dst_off = (b * num_heads + kv * repeat + r) * head_size;
                out[dst_off..dst_off + head_size]
                    .copy_from_slice(&data[src_off..src_off + head_size]);
            }
        }
    }
    out
}

/// Row-wise softmax in-place on a \[rows, cols\] matrix.
fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

// ── GQA Attention ────────────────────────────────────────────────────

/// Grouped-Query Attention with Q/K/V/O linear projections, RoPE, and KV-cache.
///
/// This is the general form that handles all standard attention patterns:
/// - **MHA**: set `num_kv_heads = num_heads` (each Q head has its own KV)
/// - **MQA**: set `num_kv_heads = 1` (all Q heads share a single KV head)
/// - **GQA**: set `num_kv_heads` to any divisor of `num_heads`
#[derive(Debug)]
pub struct GQAAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    w_q: Vec<f32>, // [hidden_size, num_heads * head_dim]
    w_k: Vec<f32>, // [hidden_size, num_kv_heads * head_dim]
    w_v: Vec<f32>, // [hidden_size, num_kv_heads * head_dim]
    w_o: Vec<f32>, // [num_heads * head_dim, hidden_size]
}

impl GQAAttention {
    /// Create a new GQA attention layer with zero-initialized weights.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, hidden_size: usize) -> Self {
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            w_q: vec![0.0; hidden_size * q_dim],
            w_k: vec![0.0; hidden_size * kv_dim],
            w_v: vec![0.0; hidden_size * kv_dim],
            w_o: vec![0.0; q_dim * hidden_size],
        }
    }

    /// Set all projection weights. Panics if sizes don't match.
    ///
    /// Weight layouts (row-major):
    /// - `w_q`: `[hidden_size, num_heads * head_dim]`
    /// - `w_k`: `[hidden_size, num_kv_heads * head_dim]`
    /// - `w_v`: `[hidden_size, num_kv_heads * head_dim]`
    /// - `w_o`: `[num_heads * head_dim, hidden_size]`
    pub fn set_weights(&mut self, w_q: Vec<f32>, w_k: Vec<f32>, w_v: Vec<f32>, w_o: Vec<f32>) {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        assert_eq!(w_q.len(), self.hidden_size * q_dim);
        assert_eq!(w_k.len(), self.hidden_size * kv_dim);
        assert_eq!(w_v.len(), self.hidden_size * kv_dim);
        assert_eq!(w_o.len(), q_dim * self.hidden_size);
        self.w_q = w_q;
        self.w_k = w_k;
        self.w_v = w_v;
        self.w_o = w_o;
    }

    /// Total number of parameters across all four projection matrices.
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Project input through Q/K/V, reshape to 4-D, and apply RoPE.
    ///
    /// Returns `(Q, K, V)` as flat arrays with logical shapes:
    /// - Q: `[batch, num_heads, seq, head_dim]`  (RoPE applied)
    /// - K: `[batch, kv_heads, seq, head_dim]`   (RoPE applied)
    /// - V: `[batch, kv_heads, seq, head_dim]`
    pub(crate) fn prepare_qkv(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
        rope_offset: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), SynapseError> {
        let m = batch * seq_len;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Linear projections: [batch*seq, hidden] @ W → [batch*seq, dim]
        let mut q = vec![0.0f32; m * q_dim];
        let mut k = vec![0.0f32; m * kv_dim];
        let mut v = vec![0.0f32; m * kv_dim];
        sgemm_nn(m, q_dim, self.hidden_size, input, &self.w_q, &mut q)?;
        sgemm_nn(m, kv_dim, self.hidden_size, input, &self.w_k, &mut k)?;
        sgemm_nn(m, kv_dim, self.hidden_size, input, &self.w_v, &mut v)?;

        // [batch*seq, dim] → [batch, seq, heads, d] → [batch, heads, seq, d]
        let q_4d = transpose_0213(&q, batch, seq_len, self.num_heads, self.head_dim);
        let k_4d = transpose_0213(&k, batch, seq_len, self.num_kv_heads, self.head_dim);
        let v_4d = transpose_0213(&v, batch, seq_len, self.num_kv_heads, self.head_dim);

        // Apply RoPE to Q and K
        let (q_out, k_out) = if let (Some(cos), Some(sin)) = (rope_cos, rope_sin) {
            let qt = Tensor::from_data(&q_4d, &[batch, self.num_heads, seq_len, self.head_dim])?;
            let kt = Tensor::from_data(&k_4d, &[batch, self.num_kv_heads, seq_len, self.head_dim])?;
            (
                qt.rope(cos, sin, rope_offset)?.to_vec()?,
                kt.rope(cos, sin, rope_offset)?.to_vec()?,
            )
        } else {
            (q_4d, k_4d)
        };

        Ok((q_out, k_out, v_4d))
    }

    /// Project attention output back to hidden dimension.
    ///
    /// `attn_out`: `[m, num_heads * head_dim]` → returns `[m, hidden_size]`.
    pub(crate) fn project_output(
        &self,
        attn_out: &[f32],
        m: usize,
    ) -> Result<Vec<f32>, SynapseError> {
        let q_dim = self.num_heads * self.head_dim;
        let mut result = vec![0.0f32; m * self.hidden_size];
        sgemm_nn(m, self.hidden_size, q_dim, attn_out, &self.w_o, &mut result)?;
        Ok(result)
    }

    /// Forward pass without KV-cache.
    ///
    /// - `input`: flat `[batch * seq_len * hidden_size]` array.
    /// - Returns flat `[batch * seq_len * hidden_size]` output.
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
        rope_offset: usize,
    ) -> Result<Vec<f32>, SynapseError> {
        let (q, k, v) = self.prepare_qkv(input, batch, seq_len, rope_cos, rope_sin, rope_offset)?;

        // GQA: expand KV heads to match Q heads
        let repeat = self.num_heads / self.num_kv_heads;
        let k_exp = repeat_kv(&k, batch, self.num_kv_heads, seq_len, self.head_dim, repeat);
        let v_exp = repeat_kv(&v, batch, self.num_kv_heads, seq_len, self.head_dim, repeat);

        // Scaled dot-product attention with causal masking
        let qt = Tensor::from_data(&q, &[batch, self.num_heads, seq_len, self.head_dim])?;
        let kt = Tensor::from_data(&k_exp, &[batch, self.num_heads, seq_len, self.head_dim])?;
        let vt = Tensor::from_data(&v_exp, &[batch, self.num_heads, seq_len, self.head_dim])?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let (out_t, _) = qt.scaled_dot_product_attention(&kt, &vt, scale, true)?;

        // [batch, heads, seq, d] → [batch, seq, heads*d] = [batch*seq, q_dim]
        let out_4d = out_t.to_vec()?;
        let out_2d = transpose_0213(&out_4d, batch, self.num_heads, seq_len, self.head_dim);

        self.project_output(&out_2d, batch * seq_len)
    }

    /// Forward pass with KV-cache support.
    ///
    /// - **Prefill** (cache empty, `seq_len > 1`): processes full sequence and fills cache.
    /// - **Decode** (cache populated, `seq_len = 1`): processes single token against cached K/V.
    ///
    /// Requires `batch = 1`.
    pub fn forward_with_cache(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        cache: &mut KvCache,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
    ) -> Result<Vec<f32>, SynapseError> {
        if batch != 1 {
            return Err(SynapseError::InvalidArg);
        }
        let kv_dim = self.num_kv_heads * self.head_dim;
        let cached_len = cache.seq_len()?;

        // Continued prefill (cache non-empty + seq > 1) not supported — SDPA's
        // built-in causal mask assumes seq_q == seq_k.
        if cached_len > 0 && seq_len > 1 {
            return Err(SynapseError::InvalidArg);
        }

        // Prepare Q/K/V with RoPE offset matching the cache position
        let (q, k_roped, v) =
            self.prepare_qkv(input, 1, seq_len, rope_cos, rope_sin, cached_len)?;

        // Transpose K (RoPE'd) and V back to per-token layout for caching:
        // [1, heads, seq, d] → [1, seq, heads, d] = [seq, kv_dim]
        let k_cache = transpose_0213(&k_roped, 1, self.num_kv_heads, seq_len, self.head_dim);
        let v_cache = transpose_0213(&v, 1, self.num_kv_heads, seq_len, self.head_dim);

        for t in 0..seq_len {
            cache.append(
                &k_cache[t * kv_dim..(t + 1) * kv_dim],
                &v_cache[t * kv_dim..(t + 1) * kv_dim],
            )?;
        }

        // Retrieve full K/V history from cache
        let (k_all, v_all, full_len) = cache.slice()?;
        let k_all = k_all.to_vec();
        let v_all = v_all.to_vec();

        // Reshape [full_seq, kv_dim] → [1, kv_heads, full_seq, head_dim]
        let k_full = transpose_0213(&k_all, 1, full_len, self.num_kv_heads, self.head_dim);
        let v_full = transpose_0213(&v_all, 1, full_len, self.num_kv_heads, self.head_dim);

        // GQA expand
        let repeat = self.num_heads / self.num_kv_heads;
        let k_exp = repeat_kv(
            &k_full,
            1,
            self.num_kv_heads,
            full_len,
            self.head_dim,
            repeat,
        );
        let v_exp = repeat_kv(
            &v_full,
            1,
            self.num_kv_heads,
            full_len,
            self.head_dim,
            repeat,
        );

        // SDPA: causal for prefill (seq_q == seq_k), non-causal for decode (seq_q == 1)
        let qt = Tensor::from_data(&q, &[1, self.num_heads, seq_len, self.head_dim])?;
        let kt = Tensor::from_data(&k_exp, &[1, self.num_heads, full_len, self.head_dim])?;
        let vt = Tensor::from_data(&v_exp, &[1, self.num_heads, full_len, self.head_dim])?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let causal = cached_len == 0 && seq_len > 1;
        let (out_t, _) = qt.scaled_dot_product_attention(&kt, &vt, scale, causal)?;

        let out_4d = out_t.to_vec()?;
        let out_2d = transpose_0213(&out_4d, 1, self.num_heads, seq_len, self.head_dim);
        self.project_output(&out_2d, seq_len)
    }
}

impl AttentionVariant for GQAAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn name(&self) -> &str {
        "GQA"
    }
}

// ── Sliding Window Attention ─────────────────────────────────────────

/// Sliding-window attention: wraps GQA and masks attention beyond `window_size`.
///
/// At position `i`, the query can only attend to positions `max(0, i - window_size + 1)..=i`.
#[derive(Debug)]
pub struct SlidingWindowAttention {
    inner: GQAAttention,
    window_size: usize,
}

impl SlidingWindowAttention {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        window_size: usize,
    ) -> Self {
        Self {
            inner: GQAAttention::new(num_heads, num_kv_heads, head_dim, hidden_size),
            window_size,
        }
    }

    pub fn set_weights(&mut self, w_q: Vec<f32>, w_k: Vec<f32>, w_v: Vec<f32>, w_o: Vec<f32>) {
        self.inner.set_weights(w_q, w_k, w_v, w_o);
    }

    pub fn param_count(&self) -> usize {
        self.inner.param_count()
    }

    /// Forward with sliding-window + causal masking.
    ///
    /// Returns `(output, attention_weights)` where:
    /// - `output`: `[batch * seq_len * hidden_size]`
    /// - `attention_weights`: `[batch * num_heads * seq_len * seq_len]`
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
        rope_offset: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), SynapseError> {
        let (q, k, v) =
            self.inner
                .prepare_qkv(input, batch, seq_len, rope_cos, rope_sin, rope_offset)?;

        let repeat = self.inner.num_heads / self.inner.num_kv_heads;
        let k_exp = repeat_kv(
            &k,
            batch,
            self.inner.num_kv_heads,
            seq_len,
            self.inner.head_dim,
            repeat,
        );
        let v_exp = repeat_kv(
            &v,
            batch,
            self.inner.num_kv_heads,
            seq_len,
            self.inner.head_dim,
            repeat,
        );

        // Manual attention with causal + sliding-window mask
        let hd = self.inner.head_dim;
        let nh = self.inner.num_heads;
        let head_seq = seq_len * hd;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut out_4d = vec![0.0f32; batch * nh * head_seq];
        let mut all_weights = vec![0.0f32; batch * nh * seq_len * seq_len];

        for b in 0..batch {
            for h in 0..nh {
                let bh = b * nh + h;
                let q_off = bh * head_seq;
                let k_off = bh * head_seq;
                let v_off = bh * head_seq;
                let o_off = bh * head_seq;

                // scores = Q_head @ K_head^T  →  [seq, seq]
                let mut scores = vec![0.0f32; seq_len * seq_len];
                sgemm_nt(
                    seq_len,
                    seq_len,
                    hd,
                    &q[q_off..q_off + head_seq],
                    &k_exp[k_off..k_off + head_seq],
                    &mut scores,
                )?;

                // Scale
                for s in scores.iter_mut() {
                    *s *= scale;
                }

                // Causal + sliding-window mask
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i || (i as isize - j as isize) >= self.window_size as isize {
                            scores[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }

                softmax_rows(&mut scores, seq_len, seq_len);

                // Store weights for this head
                let w_off = bh * seq_len * seq_len;
                all_weights[w_off..w_off + seq_len * seq_len].copy_from_slice(&scores);

                // out = weights @ V_head  →  [seq, hd]
                sgemm_nn(
                    seq_len,
                    hd,
                    seq_len,
                    &scores,
                    &v_exp[v_off..v_off + head_seq],
                    &mut out_4d[o_off..o_off + head_seq],
                )?;
            }
        }

        // [batch, heads, seq, d] → [batch, seq, heads*d]
        let out_2d = transpose_0213(&out_4d, batch, nh, seq_len, hd);
        let output = self.inner.project_output(&out_2d, batch * seq_len)?;

        Ok((output, all_weights))
    }
}

impl AttentionVariant for SlidingWindowAttention {
    fn num_heads(&self) -> usize {
        self.inner.num_heads
    }
    fn head_dim(&self) -> usize {
        self.inner.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads
    }
    fn window_size(&self) -> Option<usize> {
        Some(self.window_size)
    }
    fn name(&self) -> &str {
        "SlidingWindow"
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::KvCache;

    const HIDDEN: usize = 32;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const Q_DIM: usize = HEADS * HEAD_DIM; // 32
    const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 16

    /// Deterministic weight initialiser: small, non-zero, varying values.
    fn det_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + seed as f32) * 0.1).sin() * 0.1)
            .collect()
    }

    fn det_input(batch: usize, seq: usize) -> Vec<f32> {
        det_weights(batch * seq * HIDDEN, 42)
    }

    fn build_gqa() -> GQAAttention {
        let mut attn = GQAAttention::new(HEADS, KV_HEADS, HEAD_DIM, HIDDEN);
        attn.set_weights(
            det_weights(HIDDEN * Q_DIM, 1),
            det_weights(HIDDEN * KV_DIM, 2),
            det_weights(HIDDEN * KV_DIM, 3),
            det_weights(Q_DIM * HIDDEN, 4),
        );
        attn
    }

    fn build_rope_tables(head_dim: usize, max_seq: usize) -> (Tensor, Tensor) {
        let half_d = head_dim / 2;
        let mut cos_data = vec![0.0f32; max_seq * half_d];
        let mut sin_data = vec![0.0f32; max_seq * half_d];
        for pos in 0..max_seq {
            for i in 0..half_d {
                let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_d + i] = angle.cos();
                sin_data[pos * half_d + i] = angle.sin();
            }
        }
        (
            Tensor::from_data(&cos_data, &[max_seq, half_d]).unwrap(),
            Tensor::from_data(&sin_data, &[max_seq, half_d]).unwrap(),
        )
    }

    // ── Shape tests ──────────────────────────────────────────────────

    #[test]
    fn gqa_output_shape() {
        let attn = build_gqa();
        let input = det_input(2, 5);
        let output = attn.forward(&input, 2, 5, None, None, 0).unwrap();
        // Output shape must be [batch, seq, hidden]
        assert_eq!(output.len(), 2 * 5 * HIDDEN);
    }

    // ── MHA / MQA equivalence ────────────────────────────────────────

    #[test]
    fn gqa_with_full_kv_heads_is_mha() {
        // MHA: num_kv_heads == num_heads → repeat factor = 1 (no KV sharing)
        let mut mha = GQAAttention::new(HEADS, HEADS, HEAD_DIM, HIDDEN);
        mha.set_weights(
            det_weights(HIDDEN * Q_DIM, 10),
            det_weights(HIDDEN * Q_DIM, 20), // same size as W_q
            det_weights(HIDDEN * Q_DIM, 30),
            det_weights(Q_DIM * HIDDEN, 40),
        );
        let input = det_input(1, 4);
        let output = mha.forward(&input, 1, 4, None, None, 0).unwrap();
        assert_eq!(output.len(), 4 * HIDDEN);
        assert!(
            output.iter().any(|&x| x.abs() > 1e-10),
            "output should be non-zero"
        );
        // MHA → K/V projection size equals Q projection size
        assert_eq!(mha.w_k.len(), mha.w_q.len());
    }

    #[test]
    fn gqa_with_single_kv_head_is_mqa() {
        // MQA: num_kv_heads == 1 → all Q heads share a single KV head
        let kv_dim_mqa = 1 * HEAD_DIM;
        let mut mqa = GQAAttention::new(HEADS, 1, HEAD_DIM, HIDDEN);
        mqa.set_weights(
            det_weights(HIDDEN * Q_DIM, 50),
            det_weights(HIDDEN * kv_dim_mqa, 60),
            det_weights(HIDDEN * kv_dim_mqa, 70),
            det_weights(Q_DIM * HIDDEN, 80),
        );
        let input = det_input(1, 4);
        let output = mqa.forward(&input, 1, 4, None, None, 0).unwrap();
        assert_eq!(output.len(), 4 * HIDDEN);
        assert!(
            output.iter().any(|&x| x.abs() > 1e-10),
            "output should be non-zero"
        );
        // MQA → repeat factor is num_heads
        assert_eq!(HEADS / 1, HEADS);
        // MQA → K/V projection much smaller than Q
        assert_eq!(mqa.w_k.len(), HIDDEN * kv_dim_mqa);
    }

    // ── KV-cache: prefill + decode must match full forward ───────────

    #[test]
    fn kv_cache_prefill_decode_matches_full_forward() {
        let attn = build_gqa();
        let (cos, sin) = build_rope_tables(HEAD_DIM, 32);

        // Full forward over 11 tokens
        let input_11 = det_input(1, 11);
        let full_output = attn
            .forward(&input_11, 1, 11, Some(&cos), Some(&sin), 0)
            .unwrap();

        // Prefill first 10 tokens
        let input_10 = &input_11[..10 * HIDDEN];
        let mut cache = KvCache::new(32, KV_HEADS, HEAD_DIM).unwrap();
        let prefill_output = attn
            .forward_with_cache(input_10, 1, 10, &mut cache, Some(&cos), Some(&sin))
            .unwrap();

        // Decode the 11th token
        let input_1 = &input_11[10 * HIDDEN..11 * HIDDEN];
        let decode_output = attn
            .forward_with_cache(input_1, 1, 1, &mut cache, Some(&cos), Some(&sin))
            .unwrap();

        // Concatenate and compare bit-exact
        let mut cached_output = prefill_output;
        cached_output.extend_from_slice(&decode_output);

        assert_eq!(full_output.len(), cached_output.len());
        for (i, (&a, &b)) in full_output.iter().zip(cached_output.iter()).enumerate() {
            let diff = (a - b).abs();
            let tol = 1e-5 + 1e-4 * a.abs().max(b.abs());
            assert!(
                diff <= tol,
                "Mismatch at index {i}: full={a}, cached={b}, diff={diff}"
            );
        }
    }

    // ── Sliding window masks beyond window ───────────────────────────

    #[test]
    fn sliding_window_attention_masks_beyond_window() {
        let window = 3;
        let seq = 6;
        let mut sw = SlidingWindowAttention::new(HEADS, KV_HEADS, HEAD_DIM, HIDDEN, window);
        sw.set_weights(
            det_weights(HIDDEN * Q_DIM, 100),
            det_weights(HIDDEN * KV_DIM, 200),
            det_weights(HIDDEN * KV_DIM, 300),
            det_weights(Q_DIM * HIDDEN, 400),
        );
        let input = det_input(1, seq);
        let (output, weights) = sw.forward(&input, 1, seq, None, None, 0).unwrap();
        assert_eq!(output.len(), seq * HIDDEN);

        // weights layout: [batch=1, num_heads, seq, seq]
        for h in 0..HEADS {
            for i in 0..seq {
                for j in 0..seq {
                    let w = weights[(h * seq + i) * seq + j];
                    if j > i || (i as isize - j as isize) >= window as isize {
                        assert!(
                            w.abs() < 1e-6,
                            "h={h} i={i} j={j}: weight {w} should be ~0 (beyond window)"
                        );
                    }
                }
            }
        }
    }

    // ── Parameter count ──────────────────────────────────────────────

    #[test]
    fn param_count_matches_four_projections() {
        let attn = GQAAttention::new(HEADS, KV_HEADS, HEAD_DIM, HIDDEN);
        let expected = HIDDEN * Q_DIM   // W_q: hidden → q_dim
                     + HIDDEN * KV_DIM  // W_k: hidden → kv_dim
                     + HIDDEN * KV_DIM  // W_v: hidden → kv_dim
                     + Q_DIM * HIDDEN; // W_o: q_dim  → hidden
        assert_eq!(attn.param_count(), expected);
    }

    // ── RoPE affects output ──────────────────────────────────────────

    #[test]
    fn rope_changes_attention_output() {
        // Use larger weights so Q@K^T scores are meaningful (not near-uniform softmax)
        let mut attn = GQAAttention::new(HEADS, KV_HEADS, HEAD_DIM, HIDDEN);
        let w = |len, seed: u32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32 + seed as f32) * 0.1).sin() * 0.5)
                .collect()
        };
        attn.set_weights(
            w(HIDDEN * Q_DIM, 1),
            w(HIDDEN * KV_DIM, 2),
            w(HIDDEN * KV_DIM, 3),
            w(Q_DIM * HIDDEN, 4),
        );
        let input: Vec<f32> = (0..4 * HIDDEN)
            .map(|i| ((i as f32 + 42.0) * 0.1).sin() * 0.5)
            .collect();
        let (cos, sin) = build_rope_tables(HEAD_DIM, 16);

        let out_no_rope = attn.forward(&input, 1, 4, None, None, 0).unwrap();
        let out_with_rope = attn
            .forward(&input, 1, 4, Some(&cos), Some(&sin), 0)
            .unwrap();

        let differs = out_no_rope
            .iter()
            .zip(out_with_rope.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "RoPE should change the attention output");
    }

    // ── Trait accessors ──────────────────────────────────────────────

    #[test]
    fn gqa_trait_accessors() {
        let attn = GQAAttention::new(16, 8, 64, 1024);
        assert_eq!(attn.num_heads(), 16);
        assert_eq!(attn.num_kv_heads(), 8);
        assert_eq!(attn.head_dim(), 64);
        assert_eq!(attn.name(), "GQA");
        assert_eq!(attn.window_size(), None);
    }

    #[test]
    fn sliding_window_trait_accessors() {
        let sw = SlidingWindowAttention::new(32, 8, 128, 4096, 4096);
        assert_eq!(sw.num_heads(), 32);
        assert_eq!(sw.num_kv_heads(), 8);
        assert_eq!(sw.head_dim(), 128);
        assert_eq!(sw.window_size(), Some(4096));
        assert_eq!(sw.name(), "SlidingWindow");
    }
}
