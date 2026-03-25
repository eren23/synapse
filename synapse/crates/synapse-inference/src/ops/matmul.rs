/// y = A * B^T  where A is [m, k], B is [n, k] -> y is [m, n].
///
/// Dispatches to the Zig SIMD tiled GEMM (`syn_sgemm`) via FFI.
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];
    // syn_sgemm: C = op(A) * op(B), row-major.
    //   A [m, k] no-transpose, lda = k
    //   B [n, k] transposed -> [k, n], ldb = k
    //   C [m, n], ldc = n
    let status = unsafe {
        synapse_sys::syn_sgemm(
            m, n, k,
            a.as_ptr(), k, 0,   // A: no transpose
            b.as_ptr(), k, 1,   // B: transpose
            out.as_mut_ptr(), n, // C
        )
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm failed: {status}");
    out
}

/// y = A * B  where A is [m, k], B is [k, n] -> y is [m, n].
///
/// Non-transposed variant of [`matmul_t`]. Used for score*V in cached decode
/// where scores are `[1, seq_len]` and V is `[seq_len, head_dim]`.
pub(crate) fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_nn: a.len() != m*k");
    debug_assert_eq!(b.len(), k * n, "matmul_nn: b.len() != k*n");
    let mut out = vec![0.0f32; m * n];
    // syn_sgemm: C = op(A) * op(B), row-major.
    //   A [m, k] no-transpose, lda = k
    //   B [k, n] no-transpose, ldb = n
    //   C [m, n], ldc = n
    let status = unsafe {
        synapse_sys::syn_sgemm(
            m, n, k,
            a.as_ptr(), k, 0,   // A: no transpose
            b.as_ptr(), n, 0,   // B: no transpose
            out.as_mut_ptr(), n, // C
        )
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm (nn) failed: {status}");
    out
}
