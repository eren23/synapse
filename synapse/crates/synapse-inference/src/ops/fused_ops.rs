//! Fused kernels for edge inference on memory-constrained targets (ESP32, WASM).
//!
//! Each fused kernel eliminates intermediate buffer allocations by combining
//! multiple operations into a single pass. This is critical on ESP32 (32MB)
//! where every allocation counts, and on WASM where bandwidth is limited.
//!
//! All kernels are tested against unfused reference implementations.

/// Fused LayerNorm + adaLN modulation.
///
/// Combines normalization and affine modulation into a single pass,
/// eliminating the intermediate normalized buffer.
///
/// - `x`: `[seq_len, hidden]` input tensor
/// - `norm_weight`: `[hidden]` LayerNorm scale parameters
/// - `scale`: `[hidden]` adaLN scale (applied as `1 + scale`)
/// - `shift`: `[hidden]` adaLN shift
///
/// Produces: `normed[i] * (1.0 + scale[j]) + shift[j]` in one pass.
pub fn fused_layernorm_modulate(
    x: &[f32],
    norm_weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let off = t * hidden;
        let row = &x[off..off + hidden];
        // Compute mean and variance in one pass (Welford-style two-sum)
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for j in 0..hidden {
            sum += row[j];
            sum_sq += row[j] * row[j];
        }
        let mean = sum / hidden as f32;
        let var = sum_sq / hidden as f32 - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Normalize + weight + modulate in single pass
        for j in 0..hidden {
            let normed = (row[j] - mean) * inv_std * norm_weight[j];
            out[off + j] = normed * (1.0 + scale[j]) + shift[j];
        }
    }
    out
}

/// Fused gated residual + LayerNorm + adaLN modulation.
///
/// Combines: `residual += gate * proj`, then LayerNorm + modulate on the
/// updated residual. Eliminates 2 intermediate buffers (updated residual
/// copy + normed output).
///
/// - `residual`: `[seq_len, hidden]` (modified **in-place** with gated projection)
/// - `proj`: `[seq_len, hidden]` projection output
/// - `gate`: `[hidden]` gating vector
/// - `norm_weight`: `[hidden]` LayerNorm scale parameters
/// - `scale`: `[hidden]` adaLN scale
/// - `shift`: `[hidden]` adaLN shift
///
/// Returns the modulated output `[seq_len, hidden]`.
pub fn fused_gated_residual_layernorm_modulate(
    residual: &mut [f32],
    proj: &[f32],
    gate: &[f32],
    norm_weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let off = t * hidden;
        // Step 1: Apply gated residual in-place
        for j in 0..hidden {
            residual[off + j] += gate[j] * proj[off + j];
        }
        // Step 2: LayerNorm + modulate on the updated residual
        let row = &residual[off..off + hidden];
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for j in 0..hidden {
            sum += row[j];
            sum_sq += row[j] * row[j];
        }
        let mean = sum / hidden as f32;
        let var = sum_sq / hidden as f32 - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..hidden {
            let normed = (row[j] - mean) * inv_std * norm_weight[j];
            out[off + j] = normed * (1.0 + scale[j]) + shift[j];
        }
    }
    out
}

/// Online softmax attention: Q x K -> softmax -> x V without materializing
/// the full N x N attention matrix.
///
/// Uses the online softmax algorithm (Milakov & Gimelshein, 2018) to maintain
/// a running maximum and exponential sum, processing one key at a time per query.
///
/// Memory: O(seq_len * head_dim) instead of O(seq_len^2).
///
/// - `q`, `k`, `v`: `[seq_len, num_heads * head_dim]` (interleaved by head)
///
/// Returns `[seq_len, num_heads * head_dim]`.
pub fn online_softmax_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let qk_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * qk_dim];

    for head in 0..num_heads {
        for qi in 0..seq_len {
            // Online softmax: track running max and sum
            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut acc = vec![0.0f32; head_dim]; // weighted sum accumulator

            for ki in 0..seq_len {
                // Compute score = q[qi] . k[ki] * scale
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[qi * qk_dim + head * head_dim + d]
                        * k[ki * qk_dim + head * head_dim + d];
                }
                score *= scale;

                // Online softmax update
                if score > max_score {
                    let correction = (max_score - score).exp();
                    // Rescale existing accumulator and sum
                    for d in 0..head_dim {
                        acc[d] *= correction;
                    }
                    sum_exp = sum_exp * correction + 1.0;
                    max_score = score;
                } else {
                    sum_exp += (score - max_score).exp();
                }

                // Accumulate weighted value
                let w = (score - max_score).exp();
                for d in 0..head_dim {
                    acc[d] += w * v[ki * qk_dim + head * head_dim + d];
                }
            }

            // Normalize by softmax denominator
            if sum_exp > 0.0 {
                for d in 0..head_dim {
                    output[qi * qk_dim + head * head_dim + d] = acc[d] / sum_exp;
                }
            }
        }
    }
    output
}

/// Fused linear projection + bias + GELU activation.
///
/// Combines matmul_t, bias addition, and GELU into a single pass, eliminating
/// the intermediate bias-added buffer.
///
/// - `x`: `[seq_len, in_dim]` input
/// - `weight`: `[out_dim, in_dim]` row-major (transposed layout for matmul_t)
/// - `bias`: `[out_dim]`
///
/// Returns `[seq_len, out_dim]` with GELU applied.
pub fn fused_linear_bias_gelu(
    x: &[f32],
    weight: &[f32], // [out_dim, in_dim] row-major (for matmul_t)
    bias: &[f32],   // [out_dim]
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * out_dim];
    for i in 0..seq_len {
        for j in 0..out_dim {
            let mut sum = 0.0f32;
            for p in 0..in_dim {
                sum += x[i * in_dim + p] * weight[j * in_dim + p];
            }
            if j < bias.len() {
                sum += bias[j];
            }
            // Inline GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            out[i * out_dim + j] = 0.5
                * sum
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt()
                        * (sum + 0.044715 * sum * sum * sum))
                        .tanh());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    /// Reference unfused LayerNorm (matches lewm.rs pattern).
    fn ref_layernorm(x: &[f32], weight: &[f32], eps: f32, hidden: usize) -> Vec<f32> {
        let rows = x.len() / hidden;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let off = r * hidden;
            let row = &x[off..off + hidden];
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            let var: f32 =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let scale = 1.0 / (var + eps).sqrt();
            for j in 0..hidden {
                out[off + j] = (row[j] - mean) * scale * weight[j];
            }
        }
        out
    }

    /// Reference unfused adaLN modulation.
    fn ref_modulate(
        normed: &[f32],
        scale: &[f32],
        shift: &[f32],
        seq_len: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                out[idx] = normed[idx] * (1.0 + scale[j]) + shift[j];
            }
        }
        out
    }

    #[test]
    fn fused_layernorm_modulate_matches_unfused() {
        let hidden = 64;
        let seq_len = 4;
        let x: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32) * 0.01 - 1.0)
            .collect();
        let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.001).collect();
        let scale: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
        let shift: Vec<f32> = (0..hidden).map(|i| -0.5 + i as f32 * 0.005).collect();

        let normed = ref_layernorm(&x, &weight, 1e-6, hidden);
        let expected = ref_modulate(&normed, &scale, &shift, seq_len, hidden);
        let fused = fused_layernorm_modulate(&x, &weight, &scale, &shift, seq_len, hidden, 1e-6);

        assert!(
            approx_eq(&fused, &expected, 1e-4),
            "max diff: {}",
            fused
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn online_softmax_attention_matches_naive() {
        let seq_len = 8;
        let num_heads = 2;
        let head_dim = 4;
        let dim = num_heads * head_dim;
        let q: Vec<f32> = (0..seq_len * dim)
            .map(|i| (i as f32) * 0.1 - 2.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * dim)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.02).collect();

        let naive =
            crate::ops::attention::bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);
        let online = online_softmax_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        assert!(
            approx_eq(&online, &naive, 1e-4),
            "max diff: {}",
            online
                .iter()
                .zip(&naive)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn fused_gated_residual_layernorm_modulate_matches_unfused() {
        let hidden = 32;
        let seq_len = 3;
        let mut residual: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.01).collect();
        let proj: Vec<f32> = (0..seq_len * hidden)
            .map(|i| (i as f32) * 0.005 - 0.4)
            .collect();
        let gate: Vec<f32> = (0..hidden).map(|i| 0.5 + i as f32 * 0.01).collect();
        let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.001).collect();
        let scale: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.02).collect();
        let shift: Vec<f32> = (0..hidden).map(|i| -0.3 + i as f32 * 0.01).collect();

        // Unfused reference
        let mut ref_residual = residual.clone();
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                ref_residual[idx] += gate[j] * proj[idx];
            }
        }
        let normed = ref_layernorm(&ref_residual, &weight, 1e-6, hidden);
        let expected = ref_modulate(&normed, &scale, &shift, seq_len, hidden);

        let fused = fused_gated_residual_layernorm_modulate(
            &mut residual,
            &proj,
            &gate,
            &weight,
            &scale,
            &shift,
            seq_len,
            hidden,
            1e-6,
        );

        assert!(
            approx_eq(&fused, &expected, 1e-4),
            "max diff: {}",
            fused
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn fused_linear_bias_gelu_matches_unfused() {
        let in_dim = 16;
        let out_dim = 8;
        let seq_len = 2;
        let x: Vec<f32> = (0..seq_len * in_dim)
            .map(|i| i as f32 * 0.1 - 0.5)
            .collect();
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.05).collect();

        // Unfused reference: matmul_t + bias + gelu
        let mut expected =
            crate::ops::matmul::matmul_t(&x, &weight, seq_len, in_dim, out_dim);
        for t in 0..seq_len {
            for j in 0..out_dim {
                expected[t * out_dim + j] += bias[j];
            }
        }
        for v in expected.iter_mut() {
            *v = crate::ops::activation::gelu(*v);
        }

        let fused = fused_linear_bias_gelu(&x, &weight, &bias, seq_len, in_dim, out_dim);
        assert!(
            approx_eq(&fused, &expected, 1e-4),
            "max diff: {}",
            fused
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn online_attention_handles_single_token() {
        let seq_len = 1;
        let num_heads = 2;
        let head_dim = 4;
        let dim = num_heads * head_dim;
        let q = vec![1.0f32; dim];
        let k = vec![1.0f32; dim];
        let v: Vec<f32> = (0..dim).map(|i| i as f32).collect();

        let result = online_softmax_attention(&q, &k, &v, seq_len, num_heads, head_dim);
        // With seq_len=1, output should just be v (softmax of single element = 1.0)
        assert!(approx_eq(&result, &v, 1e-5));
    }
}
