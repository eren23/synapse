use crate::registry::NormVariant;

/// RMS normalization over the last dimension (SIMD via Zig FFI).
///
/// Uses `syn_vmul` / `syn_vreduce_sum` for zero-copy SIMD on each row,
/// avoiding tensor-handle allocation overhead that dominates at small sizes.
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];

    unsafe {
        for i in 0..n {
            let off = i * hidden_size;
            let row_ptr = x.as_ptr().add(off);
            let out_ptr = out.as_mut_ptr().add(off);

            // SIMD: out = x * x  (reuse output as scratch for squared values)
            synapse_sys::syn_vmul(out_ptr, row_ptr, row_ptr, hidden_size);

            // SIMD: ms = sum(x^2)
            let mut sum_sq = 0.0f32;
            synapse_sys::syn_vreduce_sum(out_ptr, hidden_size, &mut sum_sq);

            let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

            // SIMD: out = x * weight
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

/// Layer normalization over the last dimension (gamma only, no beta).
pub(crate) fn layernorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
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

pub(crate) fn apply_headwise_rmsnorm(
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
