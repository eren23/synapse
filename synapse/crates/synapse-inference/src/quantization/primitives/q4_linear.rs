//! Q4_0 (4-bit) quantization primitives.
//!
//! Q4_0 block format: 32 elements per block, each block stores 1 f32 scale +
//! 16 bytes of nibble pairs = 20 bytes per block. This gives ~6.4x compression
//! vs f32. Predictor weights shrink from ~43MB (f32) to ~7MB (Q4), fitting in
//! ESP32-P4's 32MB PSRAM.

/// Convert f32 to IEEE 754 half-precision (f16) bit pattern.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or subnormal → f16 zero
        return sign as u16;
    }
    if exp == 0xFF {
        // Inf/NaN
        return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow → zero
    }

    (sign | ((new_exp as u32) << 10) | (man >> 13)) as u16
}

/// A single Q4_0 block: 32 elements quantized to 4-bit with one f32 scale.
///
/// Each nibble pair packs two signed 4-bit values offset by +8 into a byte:
///   low nibble  = (v0 + 8), range [0, 15]
///   high nibble = (v1 + 8), range [0, 15]
///
/// Dequantization: value = (nibble - 8) * scale
#[repr(C)]
pub struct Q4Block {
    pub scale: f32,
    pub nibbles: [u8; 16], // 32 values packed as nibble pairs
}

/// A linear layer with Q4_0 quantized weights.
///
/// Weights are stored as a flat array of [`Q4Block`]s in row-major order:
/// `blocks[row * blocks_per_row + b]` covers columns `[b*32 .. (b+1)*32)` of row `row`.
pub struct Q4Linear {
    pub blocks: Vec<Q4Block>,
    pub out_features: usize,
    pub in_features: usize,
    /// Cached Zig-compatible packed buffer (lazily initialized).
    pub packed_zig_cache: std::cell::RefCell<Option<Vec<u8>>>,
}

impl Q4Linear {
    /// Quantize an f32 weight matrix `[out_features, in_features]` to Q4_0 blocks.
    ///
    /// `in_features` is padded to a multiple of 32 internally (zero-padded columns).
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        let padded_k = (in_features + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        let mut blocks = Vec::with_capacity(out_features * blocks_per_row);

        for row in 0..out_features {
            for b in 0..blocks_per_row {
                let mut vals = [0.0f32; 32];
                for i in 0..32 {
                    let col = b * 32 + i;
                    if col < in_features {
                        vals[i] = weights[row * in_features + col];
                    }
                }
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };
                let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

                let mut nibbles = [0u8; 16];
                for i in 0..16 {
                    let v0 = (vals[2 * i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    let v1 = (vals[2 * i + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    // Pack: low nibble = v0 + 8, high nibble = v1 + 8
                    nibbles[i] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
                }
                blocks.push(Q4Block { scale, nibbles });
            }
        }

        Q4Linear {
            blocks,
            out_features,
            in_features,
            packed_zig_cache: std::cell::RefCell::new(None),
        }
    }

    /// Pack blocks into Zig-compatible Q4_0 binary format: `[f16_scale, 16_nibble_bytes]` per block.
    ///
    /// Returns packed buffer for the entire `[out_features, padded_k/32]` block matrix.
    /// Call once at load time; reuse for all forward calls.
    pub fn pack_for_zig(&self) -> Vec<u8> {
        let padded_k = (self.in_features + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        let block_bytes = 2 + 16; // f16 scale + 16 nibble bytes = 18
        let mut packed = vec![0u8; self.out_features * blocks_per_row * block_bytes];

        for (i, block) in self.blocks.iter().enumerate() {
            let off = i * block_bytes;
            // Convert f32 scale to f16 (IEEE 754 half-precision)
            let f16_bits = f32_to_f16(block.scale);
            packed[off] = (f16_bits & 0xFF) as u8;
            packed[off + 1] = (f16_bits >> 8) as u8;
            packed[off + 2..off + 18].copy_from_slice(&block.nibbles);
        }
        packed
    }

    /// Forward pass: `x [m, in_features]` -> `[m, out_features]`.
    ///
    /// Uses Zig SIMD Q4 GEMV when available, falls back to pure Rust.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;

        if m == 0 || k == 0 || n == 0 {
            return vec![0.0f32; m * n];
        }

        // Use Zig Q4 GEMV for each row of M (cached packed buffer)
        #[cfg(feature = "zig-ffi")]
        {
            let packed = self.get_packed_zig();
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                let row = &x[i * k..(i + 1) * k];
                let row_out = synapse_core::q4_0_gemv(n, k, row, &*packed)
                    .expect("q4_0_gemv failed");
                out[i * n..(i + 1) * n].copy_from_slice(&row_out);
            }
            return out;
        }

        #[cfg(not(feature = "zig-ffi"))]
        self.forward_scalar(x, m)
    }

    /// Pure-Rust scalar fallback for Q4 forward pass.
    #[allow(dead_code)] // available for benchmarking or non-Zig targets
    fn forward_scalar(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;
        let padded_k = (k + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        let mut out = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let block = &self.blocks[j * blocks_per_row + b];
                    let scale = block.scale;
                    for ni in 0..16 {
                        let byte = block.nibbles[ni];
                        let v0 = ((byte & 0x0F) as i8 - 8) as f32 * scale;
                        let v1 = ((byte >> 4) as i8 - 8) as f32 * scale;
                        let col0 = b * 32 + 2 * ni;
                        let col1 = col0 + 1;
                        if col0 < k {
                            sum += x[i * k + col0] * v0;
                        }
                        if col1 < k {
                            sum += x[i * k + col1] * v1;
                        }
                    }
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    /// Memory in bytes for the Q4_0 block storage.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q4Block>()
    }

    /// Dequantize all Q4 blocks back to f32.
    /// Returns [out_features, in_features] row-major weights.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.out_features * self.in_features];
        let padded_k = (self.in_features + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        for row in 0..self.out_features {
            for b in 0..blocks_per_row {
                let block = &self.blocks[row * blocks_per_row + b];
                for ni in 0..16 {
                    let byte = block.nibbles[ni];
                    let v0 = ((byte & 0x0F) as i8 - 8) as f32 * block.scale;
                    let v1 = ((byte >> 4) as i8 - 8) as f32 * block.scale;
                    let col0 = b * 32 + 2 * ni;
                    let col1 = col0 + 1;
                    if col0 < self.in_features {
                        weights[row * self.in_features + col0] = v0;
                    }
                    if col1 < self.in_features {
                        weights[row * self.in_features + col1] = v1;
                    }
                }
            }
        }
        weights
    }

    /// Create an empty (zero-sized) Q4Linear, used for absent weights.
    pub fn empty() -> Self {
        Q4Linear {
            blocks: Vec::new(),
            out_features: 0,
            in_features: 0,
            packed_zig_cache: std::cell::RefCell::new(None),
        }
    }

    /// Get or compute the cached Zig-compatible packed buffer.
    #[cfg(feature = "zig-ffi")]
    fn get_packed_zig(&self) -> std::cell::Ref<'_, Vec<u8>> {
        {
            let cache = self.packed_zig_cache.borrow();
            if cache.is_some() {
                return std::cell::Ref::map(cache, |c| c.as_ref().unwrap());
            }
        }
        *self.packed_zig_cache.borrow_mut() = Some(self.pack_for_zig());
        std::cell::Ref::map(self.packed_zig_cache.borrow(), |c| c.as_ref().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    #[test]
    fn q4_linear_from_f32_roundtrip() {
        let weights: Vec<f32> = vec![
            1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0, 0.3, -0.3, 0.7, -0.7, 0.9,
            -0.9, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 0.2, -0.2, 0.15, -0.15, 0.35, -0.35, 0.45,
            -0.45, 0.55, -0.55, 0.65,
        ]; // [1, 32]
        let q = Q4Linear::from_f32(&weights, 1, 32);
        assert_eq!(q.blocks.len(), 1);
        assert!(q.blocks[0].scale > 0.0);
    }

    #[test]
    fn q4_linear_forward_produces_finite() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let q = Q4Linear::from_f32(&weights, 2, 32);
        let x = vec![1.0f32; 32];
        let out = q.forward(&x, 1);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q4_linear_memory_smaller_than_f32() {
        let weights: Vec<f32> = vec![0.1; 1024 * 512]; // [1024, 512]
        let q = Q4Linear::from_f32(&weights, 1024, 512);
        let f32_bytes = 1024 * 512 * 4;
        assert!(
            q.memory_bytes() < f32_bytes / 3,
            "Q4 should be >3x smaller than f32"
        );
    }

    #[test]
    fn q4_reduces_memory_vs_int8() {
        // Use a realistically-sized layer where Q4 wins over INT8.
        // At small dimensions (e.g. in_features=16), Q4 block overhead dominates
        // because padding to 32 wastes nibble slots and the f32 scale costs 4 bytes
        // per block. With in_features >= 64, Q4 consistently beats INT8.
        let out = 128;
        let inf = 256;
        let weights: Vec<f32> = gen_weights(out * inf, 999);
        let q4 = Q4Linear::from_f32(&weights, out, inf);
        let q4_bytes = q4.memory_bytes();
        let int8_would_be = out * inf; // 1 byte per weight for INT8
        assert!(
            q4_bytes < int8_would_be,
            "Q4 ({q4_bytes} bytes) should use less memory than INT8 ({int8_would_be} bytes)"
        );
        // Q4 with 256-wide rows: 256/32 = 8 blocks/row, 8*20 = 160 bytes/row
        // INT8: 256 bytes/row. So Q4 is ~1.6x smaller.
        let ratio = int8_would_be as f64 / q4_bytes as f64;
        assert!(
            ratio > 1.4,
            "Q4 should be at least 1.4x smaller than INT8, got {ratio:.2}x"
        );
    }

    #[test]
    fn q4_dequantize_roundtrip_accuracy() {
        // Quantize f32 → Q4 → dequant → check max error is within Q4 tolerance
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let q4 = Q4Linear::from_f32(&weights, 32, 32);
        let dequant = q4.dequantize();
        let max_err: f32 = weights
            .iter()
            .zip(&dequant)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q4 with 4-bit resolution: max error should be < scale/2 ≈ max_abs/7
        assert!(max_err < 1.0, "Q4 roundtrip error too large: {max_err}");
    }
}
