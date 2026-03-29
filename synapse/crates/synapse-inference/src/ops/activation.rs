pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
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
    fn gelu_at_zero_is_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-7, "gelu(0) should be 0");
    }

    #[test]
    fn gelu_known_value() {
        // gelu(1.0) ≈ 0.8413 per the tanh approximation
        let result = gelu(1.0);
        assert!(
            (result - 0.8413).abs() < 1e-3,
            "gelu(1.0) should be ~0.8413, got {result}"
        );
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
        // original index 1 (value 3.0) should have the highest probability
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
