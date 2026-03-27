//! Pure-Rust fallback ops for WASM and embedded targets.
//! These are correctness-first, not performance-optimized.

/// Matrix multiply: C[m,n] = A[m,k] * B^T[n,k]
pub fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Matrix multiply: C[m,n] = A[m,k] * B[k,n] (non-transposed)
pub fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Quantize f32 data to INT8 per-channel with min/max scaling.
/// Returns (int8_data, scales) where scales has length `channels`.
pub fn quantize_per_channel_int8(
    data: &[f32],
    channels: usize,
    channel_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut int8_data = vec![0i8; channels * channel_size];
    let mut scales = vec![0.0f32; channels];
    for ch in 0..channels {
        let off = ch * channel_size;
        let slice = &data[off..off + channel_size];
        let max_abs = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[ch] = scale;
        let inv_scale = 1.0 / scale;
        for j in 0..channel_size {
            int8_data[off + j] = (slice[j] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (int8_data, scales)
}

/// INT8 GEMM: C[m,n] = diag(scales_a) * (A_i8[m,k] @ B_i8[k,n]) * diag(scales_b)
/// B is stored in transposed layout [k,n] (same as QuantizedLinear's internal format).
pub fn qgemm_int8(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    b: &[i8],       // [k, n] layout
    scales_a: &[f32], // [m]
    scales_b: &[f32], // [n]
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += (a[i * k + p] as i32) * (b[p * n + j] as i32);
            }
            out[i * n + j] = (acc as f32) * scales_a[i] * scales_b[j];
        }
    }
    out
}

/// Fused SwiGLU: out[i] = silu(gate[i]) * up[i]
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    debug_assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect()
}

/// RMS normalization over the last dimension.
///
/// `x` is `[batch, hidden_size]`, `weight` is `[hidden_size]`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];

    for i in 0..n {
        let off = i * hidden_size;
        let row = &x[off..off + hidden_size];

        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

        for j in 0..hidden_size {
            out[off + j] = row[j] * weight[j] * scale;
        }
    }
    out
}

/// Fused causal attention: Q*K^T -> scale -> causal mask -> softmax -> *V
///
/// Q: [seq_q, head_dim], K: [seq_kv, head_dim], V: [seq_kv, head_dim]
/// Returns [seq_q, head_dim]
pub fn fused_attention(
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_q * head_dim];

    for qi in 0..seq_q {
        let causal_len = (qi + 1).min(seq_kv);
        let mut scores = vec![0.0f32; causal_len];

        // Q * K^T with causal mask
        for ki in 0..causal_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            scores[ki] = dot * scale;
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
            for ki in 0..causal_len {
                val += scores[ki] * v[ki * head_dim + d];
            }
            out[qi * head_dim + d] = val;
        }
    }

    out
}

/// Bidirectional (non-causal) attention: Q*K^T -> scale -> softmax -> *V
///
/// Q: [seq_q, head_dim], K: [seq_kv, head_dim], V: [seq_kv, head_dim]
/// Returns [seq_q, head_dim]
pub fn fused_attention_bidi(
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_q * head_dim];

    for qi in 0..seq_q {
        let mut scores = vec![0.0f32; seq_kv];

        // Q * K^T (no mask - bidirectional)
        for ki in 0..seq_kv {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            scores[ki] = dot * scale;
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
            for ki in 0..seq_kv {
                val += scores[ki] * v[ki * head_dim + d];
            }
            out[qi * head_dim + d] = val;
        }
    }

    out
}

/// Geometric attention with Gaussian distance bias (pure-Rust fallback).
///
/// score[i,j] = softmax(Q[i]*K[j]/sqrt(d) + exp(-||pos_i - pos_j||^2 / 2*sigma^2))
/// out[i] = sum_j score[i,j] * V[j]
pub fn geometric_attention(
    n: usize,
    d: usize,
    pos_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[f32],
    sigma: f32,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut out = vec![0.0f32; n * d];

    for i in 0..n {
        let mut scores = vec![0.0f32; n];
        for j in 0..n {
            // Dot-product attention score
            let mut dot = 0.0f32;
            for dd in 0..d {
                dot += q[i * d + dd] * k[j * d + dd];
            }
            dot *= scale;

            // Distance bias
            let mut dist_sq = 0.0f32;
            for p in 0..pos_dim {
                let diff = positions[i * pos_dim + p] - positions[j * pos_dim + p];
                dist_sq += diff * diff;
            }
            let bias = (-dist_sq / two_sigma_sq).exp();

            scores[j] = dot + bias;
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
        for dd in 0..d {
            let mut val = 0.0f32;
            for j in 0..n {
                val += scores[j] * v[j * d + dd];
            }
            out[i * d + dd] = val;
        }
    }

    out
}

/// GELU activation (approximation matching PyTorch).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place softmax over a mutable slice.
pub fn softmax(x: &mut [f32]) {
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

/// Layer normalization (weight only, no bias) over rows of `hidden_size`.
pub fn layernorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let rows = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let off = r * hidden_size;
        let row = &x[off..off + hidden_size];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (row[j] - mean) * scale * weight[j];
        }
    }
    out
}

/// Layer normalization with weight and bias over rows of `hidden_size`.
pub fn layernorm_with_bias(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let rows = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let off = r * hidden_size;
        let row = &x[off..off + hidden_size];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (row[j] - mean) * scale * weight[j] + if j < bias.len() { bias[j] } else { 0.0 };
        }
    }
    out
}

/// Multi-head bidirectional attention (no causal mask).
///
/// Q, K, V: `[seq_len, num_heads * head_dim]` interleaved by head.
/// Returns `[seq_len, num_heads * head_dim]`.
pub fn bidirectional_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, num_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let qk_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * qk_dim];
    for head in 0..num_heads {
        for qi in 0..seq_len {
            let mut scores = vec![0.0f32; seq_len];
            for ki in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * qk_dim + head * head_dim + d]
                         * k[ki * qk_dim + head * head_dim + d];
                }
                scores[ki] = dot * scale;
            }
            softmax(&mut scores);
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..seq_len {
                    val += scores[ki] * v[ki * qk_dim + head * head_dim + d];
                }
                output[qi * qk_dim + head * head_dim + d] = val;
            }
        }
    }
    output
}

/// C[m,n] = A[m,k] @ B[k,n]  (no transpose), writes into mutable slice.
pub fn sgemm_nn_into(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// C[m,n] = A[m,k] @ B^T, where B is stored row-major as [n,k], writes into mutable slice.
pub fn sgemm_nt_into(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_rust_matmul_t_matches_reference() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // [3, 3] identity
        let out = matmul_t(&a, &b, 2, 3, 3);
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6); // row 0 * col 0 of I
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn pure_rust_matmul_nn_matches_reference() {
        // A=[2,3] * B=[3,2] -> C=[2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [3, 2]
        let out = matmul_nn(&a, &b, 2, 3, 2);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6); // 1*1 + 2*0 + 3*0
        assert!((out[1] - 2.0).abs() < 1e-6); // 1*0 + 2*1 + 3*0
    }

    #[test]
    fn pure_rust_int8_quantize_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25]; // [2 channels, 3 each]
        let (int8, scales) = quantize_per_channel_int8(&data, 2, 3);
        assert_eq!(int8.len(), 6);
        assert_eq!(scales.len(), 2);
        // Channel 0: max_abs=1.0, scale=1/127, so 1.0 -> 127
        assert_eq!(int8[0], 127);
        assert_eq!(int8[1], -127);
    }

    #[test]
    fn pure_rust_qgemm_matches_f32_within_tolerance() {
        // Small matrix: verify INT8 GEMM approximates f32 matmul
        let a_f32 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let b_f32 = vec![5.0f32, 6.0, 7.0, 8.0]; // [2, 2]
        let ref_out = matmul_t(&a_f32, &b_f32, 2, 2, 2);

        // Quantize and compute
        let (a_int8, scales_a) = quantize_per_channel_int8(&a_f32, 2, 2);
        // For qgemm_int8, b needs to be in [k, n] layout (transposed from [n, k])
        let b_transposed = vec![5.0f32, 7.0, 6.0, 8.0];
        let (b_int8_t, scales_b) = quantize_per_channel_int8(&b_transposed, 2, 2);
        let out = qgemm_int8(2, 2, 2, &a_int8, &b_int8_t, &scales_a, &scales_b);
        assert!(out.iter().all(|v| v.is_finite()));
        // The quantized result should be in the same ballpark as the f32 reference
        assert!((out[0] - ref_out[0]).abs() < 2.0, "qgemm diverges too much from reference");
    }

    #[test]
    fn pure_rust_swiglu_matches_manual() {
        let gate = vec![1.0, -1.0, 0.5, 2.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        let out = swiglu(&gate, &up);
        for i in 0..4 {
            let expected = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
            assert!((out[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn pure_rust_rmsnorm_unit_rms() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let out = rmsnorm(&x, &weight, 1e-8, 4);
        let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "RMS should be ~1.0, got {rms}");
    }

    #[test]
    fn pure_rust_fused_attention_basic() {
        // 2-token sequence, 4-dim heads
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [2, 4]
        let k = q.clone();
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect(); // [2, 4]
        let out = fused_attention(2, 2, 4, &q, &k, &v);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pure_rust_fused_attention_bidi_basic() {
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = fused_attention_bidi(2, 2, 4, &q, &k, &v);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pure_rust_geometric_attention_basic() {
        let n = 4;
        let d = 4;
        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07 + 1.0).sin()).collect();
        let positions = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            10.0, 10.0, 0.0,
        ];
        let out = geometric_attention(n, d, 3, &q, &k, &v, &positions, 1.0);
        assert_eq!(out.len(), n * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pure_rust_gelu_matches_reference() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.8412).abs() < 0.01);
        assert!((gelu(-1.0) - (-0.1588)).abs() < 0.01);
    }

    #[test]
    fn pure_rust_silu_matches_reference() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((silu(1.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn pure_rust_softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be monotonically increasing
        for i in 1..x.len() {
            assert!(x[i] >= x[i - 1]);
        }
    }

    #[test]
    fn pure_rust_layernorm_unit_variance() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let out = layernorm(&x, &weight, 1e-8, 4);
        // Mean should be ~0
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn pure_rust_layernorm_with_bias_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.5; 4];
        let out = layernorm_with_bias(&x, &weight, &bias, 1e-8, 4);
        // Mean should be ~0.5 (since weight=1 and bias=0.5)
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn pure_rust_bidirectional_attention_basic() {
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [2, 4]
        let k = q.clone();
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = bidirectional_attention(&q, &k, &v, 2, 1, 4);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
