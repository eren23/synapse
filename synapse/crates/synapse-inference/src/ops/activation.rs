//! Activation functions, elementwise ops, and softmax.
//!
//! Priority rule: always prefer batched/vectorized calls over scalar loops.
//! - For slices: use `*_inplace` or `batched_*` to amortise call overhead.
//! - Scalar versions exist only for single-value calls or non-`zig-ffi` fallbacks.

/// In-place SiLU: `x / (1 + exp(-x))` applied to every element of `slice`.
/// Prefer this over scalar `silu()` for vectorized SIMD path via FFI.
#[cfg(feature = "zig-ffi")]
#[allow(dead_code)] // exposed for explicit in-place usage
pub(crate) fn silu_inplace(slice: &mut [f32]) {
    // syn_silu works in-place: dst and src can alias.
    unsafe {
        synapse_sys::syn_silu(slice.as_mut_ptr(), slice.as_ptr(), slice.len());
    }
}

#[cfg(not(feature = "zig-ffi"))]
#[allow(dead_code)] // only compiled when zig-ffi is disabled
pub(crate) fn silu_inplace(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Scalar SiLU: `x / (1 + exp(-x))`.
///
/// For vectorised paths, prefer `silu_inplace(&mut slice)`.
#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────

/// Sigmoid applied to a single scalar.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Batched sigmoid: `sigmoid(x[i])` for every element of `input`.
/// Uses Zig SIMD under `zig-ffi`, pure Rust otherwise.
pub(crate) fn batched_sigmoid(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "zig-ffi")]
    {
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            synapse_sys::syn_sigmoid(out.as_mut_ptr(), input.as_ptr(), input.len());
        }
        out
    }
    #[cfg(not(feature = "zig-ffi"))]
    {
        input.iter().map(|&x| sigmoid(x)).collect()
    }
}

// ── Tanh ─────────────────────────────────────────────────────────────────────

/// Tanh applied to a single scalar (uses the stdlib intrinsic).
#[allow(dead_code)] // used via the batched_tanh fallback below
pub(crate) fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

/// Batched tanh: `x.tanh()` for every element of `input`.
/// Uses Zig SIMD under `zig-ffi`, pure Rust otherwise.
pub(crate) fn batched_tanh(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "zig-ffi")]
    {
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            synapse_sys::syn_tanh_act(out.as_mut_ptr(), input.as_ptr(), input.len());
        }
        out
    }
    #[cfg(not(feature = "zig-ffi"))]
    {
        input.iter().map(|&x| tanh_scalar(x)).collect()
    }
}

// ── Softplus ─────────────────────────────────────────────────────────────────

/// Softplus applied to a single scalar: `log(1 + exp(x))`, numerically stable.
#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Batched softplus: `softplus(input[i])` for every element.
/// Uses Zig SIMD under `zig-ffi`, pure Rust otherwise.
pub(crate) fn batched_softplus(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "zig-ffi")]
    {
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            synapse_sys::syn_softplus(out.as_mut_ptr(), input.as_ptr(), input.len());
        }
        out
    }
    #[cfg(not(feature = "zig-ffi"))]
    {
        input.iter().map(|&x| softplus(x)).collect()
    }
}

// ── GELU ─────────────────────────────────────────────────────────────────────

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ── Softmax ───────────────────────────────────────────────────────────────────

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

/// Whether this FFN variant is gated (3 weight matrices vs 2).
pub(crate) fn is_gated_ffn(name: &str) -> bool {
    matches!(name, "SwiGLU" | "GeGLU")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_at_zero_is_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7, "silu(0) should be 0");
    }

    #[test]
    fn silu_positive_less_than_input() {
        for &x in &[0.5f32, 1.0, 2.0, 5.0] {
            let y = silu(x);
            assert!(y > 0.0, "silu({x}) should be positive, got {y}");
            assert!(y < x, "silu({x}) should be less than input, got {y}");
        }
    }

    #[test]
    fn silu_inplace_matches_scalar() {
        let mut buf = vec![1.0f32, -0.5, 3.0, 0.0];
        let expected: Vec<f32> = buf.iter().map(|&x| silu(x)).collect();
        silu_inplace(&mut buf);
        for (got, exp) in buf.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "silu_inplace mismatch: got {got}, expected {exp}");
        }
    }

    #[test]
    fn sigmoid_bounds() {
        for &x in &[-10.0f32, -1.0, 0.0, 1.0, 10.0] {
            let y = sigmoid(x);
            assert!(y > 0.0 && y < 1.0, "sigmoid({x}) = {y} out of (0,1)");
        }
    }

    #[test]
    fn batched_sigmoid_matches_scalar() {
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let out = batched_sigmoid(&input);
        for (i, (&got, &exp)) in out.iter().zip(input.iter()).enumerate() {
            assert!((got - sigmoid(exp)).abs() < 1e-6, "batched_sigmoid[{i}] mismatch");
        }
    }

    #[test]
    fn batched_tanh_bounds() {
        let input = vec![-5.0f32, -1.0, 0.0, 1.0, 5.0];
        let out = batched_tanh(&input);
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0, "batched_tanh[{i}] = {v} out of [-1,1]");
            assert!((v - tanh_scalar(input[i])).abs() < 1e-6, "batched_tanh[{i}] mismatch");
        }
    }

    #[test]
    fn softplus_stable_large_input() {
        // softplus(50) should ≈ 50
        let out = batched_softplus(&[50.0f32, 100.0]);
        assert!((out[0] - 50.0).abs() < 1.0, "softplus(50) ≈ 50, got {}", out[0]);
        assert!((out[1] - 100.0).abs() < 1.0, "softplus(100) ≈ 100, got {}", out[1]);
    }

    /// Length ≥ 4 exercises Zig SIMD chunks (VEC_LEN=4), not only the scalar tail.
    #[test]
    fn batched_softplus_matches_scalar_long_slice() {
        let input = vec![-30.0f32, 0.0, 2.5, 50.0, 100.0, -1.0, 15.0, 21.0];
        let out = batched_softplus(&input);
        for (i, (&got, &x)) in out.iter().zip(input.iter()).enumerate() {
            let exp = softplus(x);
            assert!(
                (got - exp).abs() < 1e-4,
                "batched_softplus[{i}] x={x}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn gelu_at_zero_is_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-7, "gelu(0) should be 0");
    }

    #[test]
    fn gelu_known_value() {
        // gelu(1.0) ≈ 0.8413 per the tanh approximation
        let result = gelu(1.0);
        assert!((result - 0.8413).abs() < 1e-3, "gelu(1.0) should be ~0.8413, got {result}");
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax_slice(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1.0, got {sum}");
    }

    #[test]
    fn softmax_max_element_has_highest_prob() {
        let mut x = vec![0.5f32, 3.0, 1.0, 2.0];
        softmax_slice(&mut x);
        let max_idx = x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 1, "largest input should produce largest softmax output");
    }

    #[test]
    fn is_gated_ffn_recognizes_known_variants() {
        assert!(is_gated_ffn("SwiGLU"), "SwiGLU should be gated");
        assert!(is_gated_ffn("GeGLU"), "GeGLU should be gated");
        assert!(!is_gated_ffn("GELU"), "GELU should not be gated");
        assert!(!is_gated_ffn("ReLU"), "ReLU should not be gated");
        assert!(!is_gated_ffn("SiLU"), "SiLU should not be gated");
    }
}
