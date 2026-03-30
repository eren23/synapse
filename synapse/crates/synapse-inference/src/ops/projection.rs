//! Projection GEMV with fused bias for small-K linear layers.
//!
//! Dispatch: Zig SIMD FFI when available, pure-Rust fallback otherwise.
//! Optimized for LEWM input_proj/cond_proj: M in {1,3}, N=192, K in [48,192].

/// Projection GEMV: output[m,n] = input[m,k] * weight[n,k]^T + bias[n]
///
/// `input` is `[m * k]`, `weight` is `[n * k]` (row-major, each row = one output neuron),
/// `bias` is `[n]` (pass empty slice for no bias). Returns `[m * n]` f32 output.
pub(crate) fn projection_gemv_bias(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    if m == 0 || n == 0 || k == 0 {
        return vec![0.0f32; m * n];
    }
    debug_assert_eq!(input.len(), m * k, "projection_gemv_bias: input.len() != m*k");
    debug_assert_eq!(weight.len(), n * k, "projection_gemv_bias: weight.len() != n*k");
    debug_assert!(
        bias.is_empty() || bias.len() == n,
        "projection_gemv_bias: bias.len() must be 0 or n"
    );

    #[cfg(feature = "zig-ffi")]
    {
        let bias_ptr = if bias.is_empty() {
            std::ptr::null()
        } else {
            bias.as_ptr()
        };
        let mut out = vec![0.0f32; m * n];
        let status = unsafe {
            synapse_sys::syn_projection_gemv_bias(
                m,
                n,
                k,
                input.as_ptr(),
                weight.as_ptr(),
                bias_ptr,
                out.as_mut_ptr(),
            )
        };
        debug_assert_eq!(
            status,
            synapse_sys::SYN_OK,
            "syn_projection_gemv_bias failed: {status}"
        );
        return out;
    }

    #[cfg(not(feature = "zig-ffi"))]
    {
        return projection_gemv_bias_fallback(input, weight, bias, m, n, k);
    }
}

/// Pure-Rust fallback: matmul_t + bias addition.
#[cfg(not(feature = "zig-ffi"))]
fn projection_gemv_bias_fallback(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = super::pure_rust_ops::matmul_t(input, weight, m, k, n);
    if !bias.is_empty() {
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias[col];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_identity_no_bias() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let weight = vec![1.0f32, 0.0, 0.0, 1.0]; // [2, 2] identity
        let out = projection_gemv_bias(&input, &weight, &[], 2, 2, 2);
        assert_eq!(out.len(), 4);
        for (i, (&expected, &got)) in input.iter().zip(out.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "mismatch at {i}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn projection_with_bias() {
        // input [1, 3] * weight [2, 3]^T + bias [2]
        let input = vec![1.0f32, 2.0, 3.0]; // [1, 3]
        let weight = vec![
            1.0, 0.0, 0.0, // row 0: picks input[0] => 1.0
            0.0, 1.0, 0.0, // row 1: picks input[1] => 2.0
        ]; // [2, 3]
        let bias = vec![10.0, 20.0];
        let out = projection_gemv_bias(&input, &weight, &bias, 1, 2, 3);
        assert!((out[0] - 11.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 22.0).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn projection_m3_batch() {
        // M=3 (typical LEWM input sequence), K=4, N=2
        let input = vec![
            1.0, 1.0, 1.0, 1.0, // row 0: sum = 4
            2.0, 2.0, 2.0, 2.0, // row 1: sum = 8
            0.5, 0.5, 0.5, 0.5, // row 2: sum = 2
        ]; // [3, 4]
        let weight = vec![
            1.0, 1.0, 1.0, 1.0, // row 0: all ones => dot = sum of input row
            0.0, 0.0, 0.0, 1.0, // row 1: picks last element
        ]; // [2, 4]
        let bias = vec![0.0, 100.0];
        let out = projection_gemv_bias(&input, &weight, &bias, 3, 2, 4);
        assert_eq!(out.len(), 6);
        assert!((out[0] - 4.0).abs() < 1e-5); // row0, col0: 4 + 0
        assert!((out[1] - 101.0).abs() < 1e-5); // row0, col1: 1 + 100
        assert!((out[2] - 8.0).abs() < 1e-5); // row1, col0: 8 + 0
        assert!((out[3] - 102.0).abs() < 1e-5); // row1, col1: 2 + 100
        assert!((out[4] - 2.0).abs() < 1e-5); // row2, col0: 2 + 0
        assert!((out[5] - 100.5).abs() < 1e-5); // row2, col1: 0.5 + 100
    }

    #[test]
    fn projection_empty_dims() {
        let out = projection_gemv_bias(&[], &[], &[], 0, 192, 48);
        assert!(out.is_empty());
    }
}
