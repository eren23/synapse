//! Multi-head attention module.

use synapse_autograd::Tensor;

use crate::dropout::Dropout;
use crate::linear::Linear;
use crate::module::Module;
use crate::positional::RotaryPositionalEmbedding;

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub dropout: Dropout,
    pub rope: Option<RotaryPositionalEmbedding>,
    training: bool,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention layer.
    ///
    /// * `d_model` – model dimension (must be divisible by `n_heads`)
    /// * `n_heads` – number of attention heads
    /// * `dropout_p` – dropout probability applied to attention weights
    pub fn new(d_model: usize, n_heads: usize, dropout_p: f32) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            d_model,
            n_heads
        );
        let d_head = d_model / n_heads;
        MultiHeadAttention {
            d_model,
            n_heads,
            d_head,
            w_q: Linear::new(d_model, d_model, true),
            w_k: Linear::new(d_model, d_model, true),
            w_v: Linear::new(d_model, d_model, true),
            w_o: Linear::new(d_model, d_model, true),
            dropout: Dropout::new(dropout_p),
            rope: None,
            training: true,
        }
    }

    /// Enable rotary positional embedding (builder pattern).
    pub fn with_rope(mut self, rope: RotaryPositionalEmbedding) -> Self {
        self.rope = Some(rope);
        self
    }

    /// Forward pass with separate query, key, value and optional causal mask.
    ///
    /// * `query` – `[B, Sq, D]`
    /// * `key`   – `[B, Sk, D]`
    /// * `value` – `[B, Sk, D]`
    /// * `causal` – apply causal (autoregressive) mask
    ///
    /// Returns `[B, Sq, D]`.
    pub fn forward_with_mask(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
    ) -> Tensor {
        assert_eq!(query.shape.len(), 3, "query must be [B, S, D]");
        assert_eq!(key.shape.len(), 3, "key must be [B, S, D]");
        assert_eq!(value.shape.len(), 3, "value must be [B, S, D]");
        assert_eq!(query.shape[2], self.d_model);
        assert_eq!(key.shape[2], self.d_model);
        assert_eq!(value.shape[2], self.d_model);

        let batch = query.shape[0];
        let sq = query.shape[1];
        let sk = key.shape[1];

        // QKV projection: [B, S, D] -> [B, S, D]
        let q = project_3d(&self.w_q, query, batch, sq, self.d_model);
        let k = project_3d(&self.w_k, key, batch, sk, self.d_model);
        let v = project_3d(&self.w_v, value, batch, sk, self.d_model);

        // Split heads: [B, S, D] -> [B, H, S, D/H]
        let mut q = split_heads(&q, batch, sq, self.n_heads, self.d_head);
        let mut k = split_heads(&k, batch, sk, self.n_heads, self.d_head);
        let v = split_heads(&v, batch, sk, self.n_heads, self.d_head);

        // Optional RoPE on Q and K
        if let Some(ref rope) = self.rope {
            q = rope.apply(&q, 0);
            k = rope.apply(&k, 0);
        }

        // Scaled dot-product attention: [B, H, Sq, D/H]
        let attn_out = self.scaled_dot_product_attention(&q, &k, &v, causal);

        // Concat heads: [B, H, Sq, D/H] -> [B, Sq, D]
        let concat = concat_heads(&attn_out, batch, sq, self.n_heads, self.d_head);

        // Output projection: [B, Sq, D] -> [B, Sq, D]
        project_3d(&self.w_o, &concat, batch, sq, self.d_model)
    }

    /// Scaled dot-product attention at the Tensor level.
    ///
    /// Q, K, V: `[B, H, Sq/Sk, D/H]` → output `[B, H, Sq, D/H]`
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Tensor {
        let d = q.shape[3];
        let scale = 1.0 / (d as f32).sqrt();

        // scores = Q @ K^T * scale → [B, H, Sq, Sk]
        let kt = k.transpose_dims(2, 3);
        let mut scores = batched_matmul_4d(q, &kt).scale(scale);

        if causal {
            apply_causal_mask(&mut scores);
        }

        // attn_weights = softmax(scores, axis=-1) → [B, H, Sq, Sk]
        let attn_weights = scores.softmax_axis(3);

        // Apply dropout to attention weights
        let attn_weights = self.dropout.forward(&attn_weights);

        // output = attn_weights @ V → [B, H, Sq, D/H]
        batched_matmul_4d(&attn_weights, v)
    }
}

impl Module for MultiHeadAttention {
    /// Self-attention (non-causal): query = key = value = input.
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, input, input, false)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters_mut());
        params.extend(self.w_k.parameters_mut());
        params.extend(self.w_v.parameters_mut());
        params.extend(self.w_o.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.dropout.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "MultiHeadAttention"
    }
}

// ── Helpers ───────────────────────────────────────────────────────

/// Apply a Linear layer to a 3D input by flattening B*S.
fn project_3d(linear: &Linear, input: &Tensor, batch: usize, seq: usize, d: usize) -> Tensor {
    let flat = input.reshape(&[batch * seq, d]);
    let out = linear.forward(&flat);
    out.reshape(&[batch, seq, linear.out_features()])
}

/// Reshape [B, S, H*D_h] → [B, H, S, D_h].
fn split_heads(x: &Tensor, batch: usize, seq: usize, heads: usize, d_head: usize) -> Tensor {
    let d_model = heads * d_head;
    let mut out = vec![0.0f32; batch * heads * seq * d_head];

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                for d in 0..d_head {
                    let src = b * seq * d_model + s * d_model + h * d_head + d;
                    let dst = ((b * heads + h) * seq + s) * d_head + d;
                    out[dst] = x.data[src];
                }
            }
        }
    }

    Tensor::new(out, vec![batch, heads, seq, d_head])
}

/// Reshape [B, H, S, D_h] → [B, S, H*D_h].
fn concat_heads(x: &Tensor, batch: usize, seq: usize, heads: usize, d_head: usize) -> Tensor {
    let d_model = heads * d_head;
    let mut out = vec![0.0f32; batch * seq * d_model];

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                for d in 0..d_head {
                    let src = ((b * heads + h) * seq + s) * d_head + d;
                    let dst = b * seq * d_model + s * d_model + h * d_head + d;
                    out[dst] = x.data[src];
                }
            }
        }
    }

    Tensor::new(out, vec![batch, seq, d_model])
}

/// Batched matmul over [B, H] leading dimensions.
/// a: [B, H, M, K] × b: [B, H, K, N] → [B, H, M, N]
fn batched_matmul_4d(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 4);
    assert_eq!(b.shape.len(), 4);
    let (batch, heads, m, k) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
    assert_eq!(b.shape[0], batch);
    assert_eq!(b.shape[1], heads);
    assert_eq!(b.shape[2], k);
    let n = b.shape[3];

    let mut data = vec![0.0f32; batch * heads * m * n];

    for bi in 0..batch {
        for hi in 0..heads {
            let a_base = (bi * heads + hi) * m * k;
            let b_base = (bi * heads + hi) * k * n;
            let o_base = (bi * heads + hi) * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a.data[a_base + i * k + p] * b.data[b_base + p * n + j];
                    }
                    data[o_base + i * n + j] = sum;
                }
            }
        }
    }

    Tensor::new(data, vec![batch, heads, m, n])
}

/// Set scores[b,h,i,j] = -inf where j > i (causal mask).
fn apply_causal_mask(scores: &mut Tensor) {
    let (b, h, sq, sk) = (
        scores.shape[0],
        scores.shape[1],
        scores.shape[2],
        scores.shape[3],
    );
    for bi in 0..b {
        for hi in 0..h {
            for i in 0..sq {
                for j in (i + 1)..sk {
                    let idx = ((bi * h + hi) * sq + i) * sk + j;
                    scores.data[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2654435761);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    // ── Output shape ──────────────────────────────────────────────

    #[test]
    fn test_self_attention_output_shape() {
        let mha = MultiHeadAttention::new(64, 4, 0.0);
        let input = make_tensor(&[2, 8, 64], 1);
        let output = mha.forward(&input);
        assert_eq!(output.shape, vec![2, 8, 64]);
    }

    #[test]
    fn test_cross_attention_output_shape() {
        let mha = MultiHeadAttention::new(32, 4, 0.0);
        let query = make_tensor(&[2, 6, 32], 1);
        let key = make_tensor(&[2, 10, 32], 2);
        let value = make_tensor(&[2, 10, 32], 3);
        let output = mha.forward_with_mask(&query, &key, &value, false);
        assert_eq!(output.shape, vec![2, 6, 32]);
    }

    #[test]
    fn test_single_head_output_shape() {
        let mha = MultiHeadAttention::new(16, 1, 0.0);
        let input = make_tensor(&[1, 4, 16], 42);
        let output = mha.forward(&input);
        assert_eq!(output.shape, vec![1, 4, 16]);
    }

    // ── Parameter count ───────────────────────────────────────────

    #[test]
    fn test_parameter_count() {
        let d_model = 64;
        let mha = MultiHeadAttention::new(d_model, 4, 0.0);
        let params = mha.parameters();
        let total: usize = params.iter().map(|p| p.numel()).sum();
        // 4 weight matrices [d_model, d_model] + 4 bias vectors [d_model]
        let expected = 4 * d_model * d_model + 4 * d_model;
        assert_eq!(total, expected);
    }

    #[test]
    fn test_parameter_count_d128() {
        let d_model = 128;
        let mha = MultiHeadAttention::new(d_model, 8, 0.0);
        let params = mha.parameters();
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 4 * d_model * d_model + 4 * d_model);
    }

    // ── Training mode / dropout ───────────────────────────────────

    #[test]
    fn test_training_mode_default() {
        let mha = MultiHeadAttention::new(32, 4, 0.1);
        assert!(mha.is_training());
        assert!(mha.dropout.is_training());
    }

    #[test]
    fn test_set_training_propagates() {
        let mut mha = MultiHeadAttention::new(32, 4, 0.1);
        mha.set_training(false);
        assert!(!mha.is_training());
        assert!(!mha.dropout.is_training());
    }

    #[test]
    fn test_dropout_active_in_training() {
        // With high dropout, outputs should differ between runs during training
        let mha = MultiHeadAttention::new(32, 4, 0.5);
        let input = make_tensor(&[1, 4, 32], 1);
        let out1 = mha.forward(&input);
        let out2 = mha.forward(&input);
        // Very high probability they differ (dropout is random)
        let differs = out1
            .data
            .iter()
            .zip(&out2.data)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "dropout should cause different outputs in training mode"
        );
    }

    #[test]
    fn test_dropout_inactive_in_inference() {
        let mut mha = MultiHeadAttention::new(32, 4, 0.5);
        mha.set_training(false);
        let input = make_tensor(&[1, 4, 32], 1);
        let out1 = mha.forward(&input);
        let out2 = mha.forward(&input);
        assert_eq!(
            out1.data, out2.data,
            "inference mode should be deterministic"
        );
    }

    // ── Causal masking ────────────────────────────────────────────

    #[test]
    fn test_causal_mask_position_independence() {
        // Output at position i should depend only on positions <= i.
        // Verify by changing a future token and checking that earlier outputs don't change.
        let mut mha = MultiHeadAttention::new(16, 2, 0.0);
        mha.set_training(false);

        let input1 = make_tensor(&[1, 4, 16], 1);
        let out1 = mha.forward_with_mask(&input1, &input1, &input1, true);

        // Modify position 3 (last token)
        let mut input2_data = input1.data.clone();
        for d in 0..16 {
            input2_data[3 * 16 + d] = 99.0;
        }
        let input2 = Tensor::new(input2_data, vec![1, 4, 16]);
        let out2 = mha.forward_with_mask(&input2, &input2, &input2, true);

        // Positions 0, 1, 2 should be unchanged
        for pos in 0..3 {
            for d in 0..16 {
                let idx = pos * 16 + d;
                assert!(
                    (out1.data[idx] - out2.data[idx]).abs() < 1e-5,
                    "causal: position {} should not depend on future token (dim {}, diff {})",
                    pos,
                    d,
                    (out1.data[idx] - out2.data[idx]).abs()
                );
            }
        }

        // Position 3 should differ
        let pos3_differs =
            (0..16).any(|d| (out1.data[3 * 16 + d] - out2.data[3 * 16 + d]).abs() > 1e-5);
        assert!(
            pos3_differs,
            "position 3 output should change when its input changes"
        );
    }

    #[test]
    fn test_non_causal_sees_future() {
        // Without causal mask, changing a future token SHOULD affect earlier positions.
        let mut mha = MultiHeadAttention::new(16, 2, 0.0);
        mha.set_training(false);

        let input1 = make_tensor(&[1, 4, 16], 1);
        let out1 = mha.forward_with_mask(&input1, &input1, &input1, false);

        let mut input2_data = input1.data.clone();
        for d in 0..16 {
            input2_data[3 * 16 + d] = 99.0;
        }
        let input2 = Tensor::new(input2_data, vec![1, 4, 16]);
        let out2 = mha.forward_with_mask(&input2, &input2, &input2, false);

        // Position 0 should be affected by the change at position 3
        let pos0_differs = (0..16).any(|d| (out1.data[d] - out2.data[d]).abs() > 1e-5);
        assert!(
            pos0_differs,
            "non-causal attention: earlier positions should see future tokens"
        );
    }

    // ── Gradient flow through all projections ─────────────────────

    #[test]
    fn test_gradient_flows_all_projections() {
        // Perturb each projection's weights and verify the output changes,
        // confirming gradient connectivity.
        let mut mha = MultiHeadAttention::new(16, 2, 0.0);
        mha.set_training(false);
        let input = make_tensor(&[1, 4, 16], 1);
        let base_out = mha.forward(&input);

        let eps = 0.1;
        let projection_names = ["w_q", "w_k", "w_v", "w_o"];

        for (idx, name) in projection_names.iter().enumerate() {
            let mut perturbed = MultiHeadAttention::new(16, 2, 0.0);
            perturbed.set_training(false);

            // Copy all weights from original
            copy_params(&mha, &mut perturbed);

            // Perturb one projection
            let target = match idx {
                0 => &mut perturbed.w_q,
                1 => &mut perturbed.w_k,
                2 => &mut perturbed.w_v,
                3 => &mut perturbed.w_o,
                _ => unreachable!(),
            };
            for val in target.weight.data.iter_mut() {
                *val += eps;
            }

            let perturbed_out = perturbed.forward(&input);
            let max_diff: f32 = base_out
                .data
                .iter()
                .zip(&perturbed_out.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff > 1e-6,
                "gradient should flow through {}: max_diff = {}",
                name,
                max_diff
            );
        }
    }

    // ── RoPE ──────────────────────────────────────────────────────

    #[test]
    fn test_with_rope_output_shape() {
        let d_model = 32;
        let n_heads = 4;
        let d_head = d_model / n_heads;
        let rope = RotaryPositionalEmbedding::new(128, d_head);
        let mut mha = MultiHeadAttention::new(d_model, n_heads, 0.0).with_rope(rope);
        mha.set_training(false);

        let input = make_tensor(&[2, 8, d_model], 1);
        let output = mha.forward(&input);
        assert_eq!(output.shape, vec![2, 8, d_model]);
    }

    #[test]
    fn test_rope_changes_output() {
        let d_model = 16;
        let n_heads = 2;
        let d_head = d_model / n_heads;

        let mut mha_no_rope = MultiHeadAttention::new(d_model, n_heads, 0.0);
        mha_no_rope.set_training(false);

        let rope = RotaryPositionalEmbedding::new(64, d_head);
        let mut mha_rope = MultiHeadAttention::new(d_model, n_heads, 0.0);
        mha_rope.set_training(false);
        // Copy weights so only difference is RoPE
        copy_params(&mha_no_rope, &mut mha_rope);
        mha_rope.rope = Some(rope);

        let input = make_tensor(&[1, 4, d_model], 1);
        let out_no_rope = mha_no_rope.forward(&input);
        let out_rope = mha_rope.forward(&input);

        let differs = out_no_rope
            .data
            .iter()
            .zip(&out_rope.data)
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "RoPE should change the output");
    }

    #[test]
    fn test_rope_causal_output_shape() {
        let d_model = 32;
        let n_heads = 4;
        let d_head = d_model / n_heads;
        let rope = RotaryPositionalEmbedding::new(128, d_head);
        let mut mha = MultiHeadAttention::new(d_model, n_heads, 0.0).with_rope(rope);
        mha.set_training(false);

        let input = make_tensor(&[2, 8, d_model], 1);
        let output = mha.forward_with_mask(&input, &input, &input, true);
        assert_eq!(output.shape, vec![2, 8, d_model]);
    }

    // ── Module trait ──────────────────────────────────────────────

    #[test]
    fn test_module_name() {
        let mha = MultiHeadAttention::new(32, 4, 0.0);
        assert_eq!(mha.name(), "MultiHeadAttention");
    }

    #[test]
    fn test_module_as_trait_object() {
        let mha: Box<dyn Module> = Box::new(MultiHeadAttention::new(16, 2, 0.0));
        let input = make_tensor(&[1, 4, 16], 1);
        let output = mha.forward(&input);
        assert_eq!(output.shape, vec![1, 4, 16]);
    }

    // ── Helper ────────────────────────────────────────────────────

    fn copy_params(src: &MultiHeadAttention, dst: &mut MultiHeadAttention) {
        dst.w_q.weight = src.w_q.weight.clone();
        dst.w_q.bias = src.w_q.bias.clone();
        dst.w_k.weight = src.w_k.weight.clone();
        dst.w_k.bias = src.w_k.bias.clone();
        dst.w_v.weight = src.w_v.weight.clone();
        dst.w_v.bias = src.w_v.bias.clone();
        dst.w_o.weight = src.w_o.weight.clone();
        dst.w_o.bias = src.w_o.bias.clone();
    }
}
