// ── Apple Accelerate BLAS (macOS) ─────────────────────────────────
// cblas_sgemm from Accelerate.framework — hand-tuned by Apple for all
// Apple Silicon matrix sizes. Significantly faster than our Zig kernel
// for small M (LEWM predict) and competitive for large M (LLM prefill).
#[cfg(target_os = "macos")]
mod accelerate {
    // CBLAS enums (CblasRowMajor=101, CblasNoTrans=111, CblasTrans=112)
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    extern "C" {
        fn cblas_sgemm(
            order: i32, trans_a: i32, trans_b: i32,
            m: i32, n: i32, k: i32,
            alpha: f32, a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32, c: *mut f32, ldc: i32,
        );
    }

    /// C[m,n] = A[m,k] * B^T[n,k] via Apple Accelerate cblas_sgemm.
    pub fn sgemm_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), k as i32,
                0.0, out.as_mut_ptr(), n as i32,
            );
        }
    }

    /// C[m,n] = A[m,k] * B[k,n] via Apple Accelerate cblas_sgemm.
    pub fn sgemm_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0, out.as_mut_ptr(), n as i32,
            );
        }
    }
}

/// y = A * B^T  where A is [m, k], B is [n, k] -> y is [m, n].
///
/// Dispatch order:
/// 1. Apple Accelerate (macOS) — fastest for all sizes
/// 2. Zig SIMD (zig-ffi feature) — cross-platform SIMD
/// 3. Pure-Rust scalar — WASM/embedded fallback
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];

    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm_t(a, b, m, k, n, &mut out);
        return out;
    }

    #[cfg(all(not(target_os = "macos"), feature = "zig-ffi"))]
    {
        let status = unsafe {
            synapse_sys::syn_sgemm(m, n, k, a.as_ptr(), k, 0, b.as_ptr(), k, 1, out.as_mut_ptr(), n)
        };
        debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm failed: {status}");
        return out;
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "zig-ffi")))]
    {
        return super::pure_rust_ops::matmul_t(a, b, m, k, n);
    }
}

/// y = A * B  where A is [m, k], B is [k, n] -> y is [m, n].
///
/// Non-transposed variant. Same dispatch as matmul_t.
pub(crate) fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_nn: a.len() != m*k");
    debug_assert_eq!(b.len(), k * n, "matmul_nn: b.len() != k*n");
    let mut out = vec![0.0f32; m * n];

    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm_nn(a, b, m, k, n, &mut out);
        return out;
    }

    #[cfg(all(not(target_os = "macos"), feature = "zig-ffi"))]
    {
        let status = unsafe {
            synapse_sys::syn_sgemm(m, n, k, a.as_ptr(), k, 0, b.as_ptr(), n, 0, out.as_mut_ptr(), n)
        };
        debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm (nn) failed: {status}");
        return out;
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "zig-ffi")))]
    {
        return super::pure_rust_ops::matmul_nn(a, b, m, k, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity matrix stored row-major as [n, k] for matmul_t (B is [n, k]).
    fn identity_for_t(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n { m[i * n + i] = 1.0; }
        m
    }

    /// Identity matrix stored row-major as [k, n] for matmul_nn (B is [k, n]).
    fn identity_for_nn(n: usize) -> Vec<f32> {
        identity_for_t(n) // same layout for square identity
    }

    #[test]
    fn matmul_t_identity() {
        // A [2, 3] * I^T [3, 3] = A  (I stored as [3, 3], row-major)
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let i3 = identity_for_t(3);
        let out = matmul_t(&a, &i3, 2, 3, 3);
        assert_eq!(out.len(), 6);
        for (i, (&expected, &got)) in a.iter().zip(out.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-5, "matmul_t identity mismatch at {i}: {got} != {expected}");
        }
    }

    #[test]
    fn matmul_t_known_product() {
        // [1,2; 3,4] * [1,0; 0,1]^T = [1,2; 3,4]
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // [2, 2] identity (also its own transpose)
        let out = matmul_t(&a, &b, 2, 2, 2);
        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (i, (&e, &g)) in expected.iter().zip(out.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "matmul_t known product mismatch at {i}: {g} != {e}");
        }
    }

    #[test]
    fn matmul_nn_identity() {
        // A [2, 3] * I [3, 3] = A
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let i3 = identity_for_nn(3);
        let out = matmul_nn(&a, &i3, 2, 3, 3);
        assert_eq!(out.len(), 6);
        for (i, (&expected, &got)) in a.iter().zip(out.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-5, "matmul_nn identity mismatch at {i}: {got} != {expected}");
        }
    }

    #[test]
    fn matmul_t_m1_single_row() {
        // M=1 edge case: single query against a 4-row key matrix.
        // q = [1, 0, 0, 0], k_rows = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        // => scores = [1, 0, 0, 0]
        let q = vec![1.0f32, 0.0, 0.0, 0.0]; // [1, 4]
        let k = vec![
            1.0f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]; // [4, 4]
        let out = matmul_t(&q, &k, 1, 4, 4);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-5, "score[0] should be 1.0, got {}", out[0]);
        assert!(out[1].abs() < 1e-5, "score[1] should be 0.0, got {}", out[1]);
        assert!(out[2].abs() < 1e-5, "score[2] should be 0.0, got {}", out[2]);
        assert!(out[3].abs() < 1e-5, "score[3] should be 0.0, got {}", out[3]);
    }
}
