use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── Scaled Dot-Product Attention ──────────────────────────────────

pub struct ScaledDotProductAttentionBackward {
    input_ids: Vec<VariableId>,
    q_data: Tensor,       // [B, H, Sq, D]
    k_data: Tensor,       // [B, H, Sk, D]
    v_data: Tensor,       // [B, H, Sk, D]
    attn_weights: Tensor, // [B, H, Sq, Sk]
    scale: f32,
    causal: bool,
}

impl GradFn for ScaledDotProductAttentionBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // grad_output: [B, H, Sq, D]

        // grad_V = attn_weights^T × grad_output  → [B, H, Sk, D]
        let attn_t = self.attn_weights.transpose_dims(2, 3);
        let grad_v = batched_matmul_4d(&attn_t, grad_output);

        // grad_attn = grad_output × V^T  → [B, H, Sq, Sk]
        let vt = self.v_data.transpose_dims(2, 3);
        let grad_attn = batched_matmul_4d(grad_output, &vt);

        // grad_scores = softmax_backward(grad_attn, attn_weights)
        // dx = s * (dout - sum(dout * s, axis=-1, keepdim=true))
        let s = &self.attn_weights;
        let ds = grad_attn.mul(s);
        let sum_ds = ds.sum_axis(3, true).broadcast_to(&s.shape);
        let mut grad_scores = s.mul(&grad_attn.sub(&sum_ds));

        // grad_scores *= scale
        grad_scores = grad_scores.scale(self.scale);

        // Apply causal mask to grad_scores (zero out masked positions)
        if self.causal {
            zero_causal_mask(&mut grad_scores);
        }

        // grad_Q = grad_scores × K  → [B, H, Sq, D]
        let grad_q = batched_matmul_4d(&grad_scores, &self.k_data);

        // grad_K = grad_scores^T × Q  → [B, H, Sk, D]
        let grad_scores_t = grad_scores.transpose_dims(2, 3);
        let grad_k = batched_matmul_4d(&grad_scores_t, &self.q_data);

        vec![Some(grad_q), Some(grad_k), Some(grad_v)]
    }

    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Helpers ───────────────────────────────────────────────────────

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

/// Set scores[b,h,i,j] = -inf where j > i (causal mask for forward).
fn apply_causal_mask(scores: &mut Tensor) {
    let (b, h, sq, sk) = (scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]);
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

/// Zero out grad[b,h,i,j] where j > i (causal mask for backward).
fn zero_causal_mask(grad: &mut Tensor) {
    let (b, h, sq, sk) = (grad.shape[0], grad.shape[1], grad.shape[2], grad.shape[3]);
    for bi in 0..b {
        for hi in 0..h {
            for i in 0..sq {
                for j in (i + 1)..sk {
                    let idx = ((bi * h + hi) * sq + i) * sk + j;
                    grad.data[idx] = 0.0;
                }
            }
        }
    }
}

// ── Graph method ─────────────────────────────────────────────────

impl Graph {
    /// Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V
    ///
    /// Q: [B, H, Sq, D], K: [B, H, Sk, D], V: [B, H, Sk, D]
    /// Returns: [B, H, Sq, D]
    pub fn scaled_dot_product_attention(
        &mut self,
        q: VariableId,
        k: VariableId,
        v: VariableId,
        causal: bool,
    ) -> VariableId {
        let q_data = self.variables[&q].data.clone();
        let k_data = self.variables[&k].data.clone();
        let v_data = self.variables[&v].data.clone();

        let d = q_data.shape[3];
        let scale = 1.0 / (d as f32).sqrt();

        // scores = Q @ K^T * scale  → [B, H, Sq, Sk]
        let kt = k_data.transpose_dims(2, 3);
        let mut scores = batched_matmul_4d(&q_data, &kt).scale(scale);

        // Apply causal mask
        if causal {
            apply_causal_mask(&mut scores);
        }

        // attn_weights = softmax(scores, axis=-1)  → [B, H, Sq, Sk]
        let attn_weights = scores.softmax_axis(3);

        // output = attn_weights @ V  → [B, H, Sq, D]
        let output = batched_matmul_4d(&attn_weights, &v_data);

        if !self.should_track(&[q, k, v]) {
            return self.untracked(output);
        }

        self.record_op(
            Box::new(ScaledDotProductAttentionBackward {
                input_ids: vec![q, k, v],
                q_data,
                k_data,
                v_data,
                attn_weights,
                scale,
                causal,
            }),
            &[q, k, v],
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward::backward;
    use crate::grad_check::grad_check;

    /// Deterministic pseudo-random tensor with values in [-0.5, 0.5].
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

    #[test]
    fn test_sdpa_grad_check_1_1_4_8_non_causal() {
        let inputs = vec![
            make_tensor(&[1, 1, 4, 8], 1),
            make_tensor(&[1, 1, 4, 8], 2),
            make_tensor(&[1, 1, 4, 8], 3),
        ];
        assert!(
            grad_check(
                |g, v| g.scaled_dot_product_attention(v[0], v[1], v[2], false),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [1,1,4,8] non-causal"
        );
    }

    #[test]
    fn test_sdpa_grad_check_1_1_4_8_causal() {
        let inputs = vec![
            make_tensor(&[1, 1, 4, 8], 4),
            make_tensor(&[1, 1, 4, 8], 5),
            make_tensor(&[1, 1, 4, 8], 6),
        ];
        assert!(
            grad_check(
                |g, v| g.scaled_dot_product_attention(v[0], v[1], v[2], true),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [1,1,4,8] causal"
        );
    }

    #[test]
    fn test_sdpa_grad_check_2_4_16_32_non_causal() {
        let inputs = vec![
            make_tensor(&[2, 4, 16, 32], 10),
            make_tensor(&[2, 4, 16, 32], 20),
            make_tensor(&[2, 4, 16, 32], 30),
        ];
        assert!(
            grad_check(
                |g, v| g.scaled_dot_product_attention(v[0], v[1], v[2], false),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [2,4,16,32] non-causal"
        );
    }

    #[test]
    fn test_sdpa_grad_check_2_4_16_32_causal() {
        let inputs = vec![
            make_tensor(&[2, 4, 16, 32], 40),
            make_tensor(&[2, 4, 16, 32], 50),
            make_tensor(&[2, 4, 16, 32], 60),
        ];
        assert!(
            grad_check(
                |g, v| g.scaled_dot_product_attention(v[0], v[1], v[2], true),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [2,4,16,32] causal"
        );
    }

    #[test]
    fn test_sdpa_gradient_shapes_match_inputs() {
        let mut g = Graph::new();
        let q = g.variable(make_tensor(&[2, 4, 16, 32], 1), true);
        let k = g.variable(make_tensor(&[2, 4, 16, 32], 2), true);
        let v = g.variable(make_tensor(&[2, 4, 16, 32], 3), true);
        let out = g.scaled_dot_product_attention(q, k, v, false);

        assert_eq!(g.data(out).shape, vec![2, 4, 16, 32]);

        backward(&mut g, out);

        assert_eq!(g.grad(q).unwrap().shape, vec![2, 4, 16, 32]);
        assert_eq!(g.grad(k).unwrap().shape, vec![2, 4, 16, 32]);
        assert_eq!(g.grad(v).unwrap().shape, vec![2, 4, 16, 32]);
    }

    #[test]
    fn test_sdpa_gradient_shapes_causal() {
        let mut g = Graph::new();
        let q = g.variable(make_tensor(&[1, 1, 4, 8], 7), true);
        let k = g.variable(make_tensor(&[1, 1, 4, 8], 8), true);
        let v = g.variable(make_tensor(&[1, 1, 4, 8], 9), true);
        let out = g.scaled_dot_product_attention(q, k, v, true);

        assert_eq!(g.data(out).shape, vec![1, 1, 4, 8]);

        backward(&mut g, out);

        assert_eq!(g.grad(q).unwrap().shape, vec![1, 1, 4, 8]);
        assert_eq!(g.grad(k).unwrap().shape, vec![1, 1, 4, 8]);
        assert_eq!(g.grad(v).unwrap().shape, vec![1, 1, 4, 8]);
    }
}
