//! Ternary (2-bit) quantization for linear layers.
//!
//! Each weight is quantized to {-1, 0, +1} * scale, where the scale is computed
//! per-row as the mean absolute value of the non-zero weights. Two bits per weight
//! are packed into u8 bytes, giving 4 weights per byte and 16 weights per block.
//!
//! Encoding:
//!   0b00 = -1 (multiply by -scale)
//!   0b01 =  0 (skip)
//!   0b10 = +1 (multiply by +scale)
//!   0b11 = reserved / treated as 0
//!
//! Block size is 16 weights packed into 4 bytes plus a f32 scale, giving 8 bytes
//! per block = 2 bits/weight + overhead. Compared to f32 (4 bytes/weight), ternary
//! blocks use 0.5 bytes/weight plus scale overhead — well under 25% of f32 memory
//! for any reasonably wide layer.

use std::mem;

/// A single ternary block: 16 weights packed into 4 bytes with one f32 scale.
///
/// Each 2-bit group encodes one weight:
///   0b00 = -1, 0b01 = 0, 0b10 = +1, 0b11 = 0 (reserved)
///
/// `bits[b]` holds weights `[4b .. 4b+4)`:
///   bits 1:0 → weight 4b
///   bits 3:2 → weight 4b+1
///   bits 5:4 → weight 4b+2
///   bits 7:6 → weight 4b+3
#[repr(C)]
#[derive(Clone, Debug)]
pub struct TernaryBlock {
    /// Per-block scale: dequantized value = ternary_code * scale.
    pub scale: f32,
    /// 16 weights packed as 2-bit codes, 4 per byte.
    pub bits: [u8; 4],
}

impl TernaryBlock {
    /// Encode 16 f32 weights (already thresholded) into a ternary block.
    ///
    /// `ternary[i]` must be -1, 0, or +1.
    fn encode(ternary: &[i8; 16], scale: f32) -> Self {
        let mut bits = [0u8; 4];
        for i in 0..16 {
            let code: u8 = match ternary[i] {
                -1 => 0b00,
                1 => 0b10,
                _ => 0b01, // 0 or any other value
            };
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            bits[byte_idx] |= code << bit_shift;
        }
        TernaryBlock { scale, bits }
    }

    /// Decode the 16 ternary codes from this block into i8 values {-1, 0, +1}.
    fn decode_codes(&self) -> [i8; 16] {
        let mut codes = [0i8; 16];
        for i in 0..16 {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let code = (self.bits[byte_idx] >> bit_shift) & 0b11;
            codes[i] = match code {
                0b00 => -1,
                0b10 => 1,
                _ => 0, // 0b01 (zero) or 0b11 (reserved)
            };
        }
        codes
    }
}

/// A linear layer with 2-bit ternary quantized weights.
///
/// Weights are stored as a flat array of [`TernaryBlock`]s in row-major order:
/// `blocks[row * blocks_per_row + b]` covers columns `[b*16 .. (b+1)*16)` of `row`.
///
/// The forward pass accumulates via addition/subtraction only (no multiplications
/// on individual weights), multiplying by the block scale at the end of each block.
pub struct TernaryLinear {
    pub blocks: Vec<TernaryBlock>,
    pub blocks_per_row: usize,
    pub out_features: usize,
    pub in_features: usize,
}

impl TernaryLinear {
    /// Quantize an f32 weight matrix `[out_features, in_features]` to ternary blocks.
    ///
    /// Ternarization is performed per-row:
    ///   threshold = 0.5 * mean(|w|) for all weights in the row
    ///   ternary(w) = +1 if w > threshold, -1 if w < -threshold, else 0
    ///   scale = mean(|w|) for non-zero entries (fallback: mean(|w|) of all)
    ///
    /// `in_features` is padded to a multiple of 16 internally (zero-padded columns).
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(
            weights.len(),
            out_features * in_features,
            "TernaryLinear::from_f32: weights.len() != out_features * in_features"
        );

        let padded_k = (in_features + 15) / 16 * 16;
        let blocks_per_row = padded_k / 16;
        let mut blocks = Vec::with_capacity(out_features * blocks_per_row);

        for row in 0..out_features {
            let row_slice = &weights[row * in_features..(row + 1) * in_features];

            // Compute per-row mean absolute value.
            let mean_abs = if in_features == 0 {
                0.0f32
            } else {
                row_slice.iter().map(|w| w.abs()).sum::<f32>() / in_features as f32
            };
            let threshold = 0.5 * mean_abs;

            // Assign ternary codes and compute non-zero scale.
            let mut ternary_row = vec![0i8; padded_k];
            let mut nonzero_sum = 0.0f32;
            let mut nonzero_count = 0usize;

            for (i, &w) in row_slice.iter().enumerate() {
                if w > threshold {
                    ternary_row[i] = 1;
                    nonzero_sum += w.abs();
                    nonzero_count += 1;
                } else if w < -threshold {
                    ternary_row[i] = -1;
                    nonzero_sum += w.abs();
                    nonzero_count += 1;
                }
                // else: zero (already 0)
            }
            // Padded columns remain 0.

            let scale = if nonzero_count > 0 {
                nonzero_sum / nonzero_count as f32
            } else {
                // All weights are near-zero; use mean_abs as fallback.
                mean_abs
            };

            // Pack into blocks of 16.
            for b in 0..blocks_per_row {
                let start = b * 16;
                let mut chunk = [0i8; 16];
                chunk.copy_from_slice(&ternary_row[start..start + 16]);
                blocks.push(TernaryBlock::encode(&chunk, scale));
            }
        }

        TernaryLinear {
            blocks,
            blocks_per_row,
            out_features,
            in_features,
        }
    }

    /// Create an empty (zero-sized) TernaryLinear, used for absent gate weights.
    pub fn empty() -> Self {
        TernaryLinear {
            blocks: Vec::new(),
            blocks_per_row: 0,
            out_features: 0,
            in_features: 0,
        }
    }

    /// Returns true if this layer has no weights.
    pub fn is_empty(&self) -> bool {
        self.out_features == 0 || self.in_features == 0
    }

    /// Forward pass: `x [m, in_features]` -> `[m, out_features]`.
    ///
    /// Decodes ternary codes on-the-fly and accumulates via add/subtract/skip,
    /// multiplying by the block scale at the end of each block.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;

        if m == 0 || k == 0 || n == 0 {
            return vec![];
        }

        debug_assert_eq!(
            x.len(),
            m * k,
            "TernaryLinear::forward: x.len() != m * in_features"
        );

        let padded_k = self.blocks_per_row * 16;
        let mut out = vec![0.0f32; m * n];

        for i in 0..m {
            let x_row = &x[i * k..(i + 1) * k];
            for j in 0..n {
                let mut acc = 0.0f32;
                for b in 0..self.blocks_per_row {
                    let block = &self.blocks[j * self.blocks_per_row + b];
                    let codes = block.decode_codes();
                    let mut block_acc = 0.0f32;
                    for wi in 0..16 {
                        let col = b * 16 + wi;
                        // Only accumulate within actual (non-padded) columns.
                        if col < k {
                            match codes[wi] {
                                1 => block_acc += x_row[col],
                                -1 => block_acc -= x_row[col],
                                _ => {}
                            }
                        }
                    }
                    acc += block_acc * block.scale;
                }
                out[i * n + j] = acc;
            }
        }
        let _ = padded_k; // suppress unused warning
        out
    }

    /// Reconstruct the f32 weight matrix (lossy).
    ///
    /// Returns `[out_features, in_features]` row-major.
    pub fn dequantize(&self) -> Vec<f32> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0f32; self.out_features * self.in_features];
        for row in 0..self.out_features {
            for b in 0..self.blocks_per_row {
                let block = &self.blocks[row * self.blocks_per_row + b];
                let codes = block.decode_codes();
                for wi in 0..16 {
                    let col = b * 16 + wi;
                    if col < self.in_features {
                        out[row * self.in_features + col] = codes[wi] as f32 * block.scale;
                    }
                }
            }
        }
        out
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * mem::size_of::<TernaryBlock>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic pseudo-random f32 in [-1, 1].
    fn pseudo_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Frobenius-norm-based relative error: ||a - b||_F / ||b||_F.
    fn frobenius_relative_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&va, &vb) in a.iter().zip(b.iter()) {
            diff_sq += ((va - vb) as f64).powi(2);
            ref_sq += (vb as f64).powi(2);
        }
        if ref_sq == 0.0 {
            return 0.0;
        }
        (diff_sq / ref_sq).sqrt() as f32
    }

    /// Reference f32 matmul: Y = X @ W^T, where W is [n, k] and X is [m, k].
    fn matmul_f32_ref(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for d in 0..k {
                    sum += x[i * k + d] * w[j * k + d];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    #[test]
    fn test_ternary_linear_forward_small() {
        // 4 input rows, 8 output features, 32 input features.
        let m = 4;
        let out_features = 8;
        let in_features = 32;

        let x = pseudo_random_vec(m * in_features, 42);
        let w = pseudo_random_vec(out_features * in_features, 123);

        let tl = TernaryLinear::from_f32(&w, out_features, in_features);
        let y = tl.forward(&x, m);

        // Output must be the right size.
        assert_eq!(y.len(), m * out_features, "output size mismatch");

        // All values must be finite.
        for &v in &y {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }

        // Sanity: the ternary output should be in the right ballpark compared
        // to the f32 reference. Ternary is lossy; allow up to 80% relative error.
        let y_ref = matmul_f32_ref(&x, &w, m, out_features, in_features);
        let rel_err = frobenius_relative_error(&y, &y_ref);
        assert!(
            rel_err < 0.8,
            "forward small: Frobenius relative error {rel_err:.4} >= 0.8"
        );
    }

    #[test]
    fn test_ternary_dequantize_roundtrip() {
        let out_features = 4;
        let in_features = 32;

        let w = pseudo_random_vec(out_features * in_features, 999);
        let tl = TernaryLinear::from_f32(&w, out_features, in_features);
        let dq = tl.dequantize();

        assert_eq!(dq.len(), out_features * in_features, "dequantize size mismatch");

        // All values must be finite.
        for &v in &dq {
            assert!(v.is_finite(), "dequantized value is non-finite: {v}");
        }

        // Every dequantized value must be exactly -scale, 0, or +scale for its block.
        for row in 0..out_features {
            for b in 0..tl.blocks_per_row {
                let block = &tl.blocks[row * tl.blocks_per_row + b];
                let scale = block.scale;
                for wi in 0..16 {
                    let col = b * 16 + wi;
                    if col < in_features {
                        let val = dq[row * in_features + col];
                        let ok = val == 0.0
                            || (val - scale).abs() < 1e-6
                            || (val + scale).abs() < 1e-6;
                        assert!(
                            ok,
                            "dequantized value {val} is not in {{-{scale}, 0, +{scale}}} \
                             (row={row}, col={col})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ternary_linear_empty() {
        let tl = TernaryLinear::empty();
        assert!(tl.is_empty());

        // forward on empty should return empty vec
        let result = tl.forward(&[], 0);
        assert!(result.is_empty(), "empty forward should return empty vec");

        // forward with m>0 on empty layer
        let x = vec![1.0f32; 4];
        let result2 = tl.forward(&x, 1);
        assert!(
            result2.is_empty(),
            "forward on empty layer should return empty vec"
        );
    }

    #[test]
    fn test_ternary_memory_compression() {
        let out_features = 64;
        let in_features = 128;

        let w = pseudo_random_vec(out_features * in_features, 42);
        let tl = TernaryLinear::from_f32(&w, out_features, in_features);

        let f32_bytes = out_features * in_features * std::mem::size_of::<f32>();
        let ternary_bytes = tl.memory_bytes();

        // Ternary should use strictly less than 25% of f32 memory.
        let ratio = ternary_bytes as f32 / f32_bytes as f32;
        assert!(
            ratio < 0.25,
            "ternary memory ratio {ratio:.4} >= 0.25 (f32={f32_bytes}B, ternary={ternary_bytes}B)"
        );
    }
}
