/// Convert f16 (IEEE 754 half-precision) bit patterns to f32 values.
pub fn f16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&bits| f16_bits_to_f32(bits)).collect()
}

/// Convert bf16 (bfloat16) bit patterns to f32 values.
pub fn bf16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&bits| bf16_bits_to_f32(bits)).collect()
}

/// Convert a single f16 bit pattern to f32.
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut e = 1u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let m = (m & 0x3ff) << 13;
            let e = 127 - 15 + 1 - e;
            f32::from_bits((sign << 31) | (e << 23) | m)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13))
    } else {
        // Normal
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

/// Convert a single bf16 bit pattern to f32.
/// bf16 is the upper 16 bits of an f32.
pub(crate) fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Transpose a 2D row-major matrix.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        // f16 1.0 = 0x3C00
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        // f16 -1.0 = 0xBC00
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        // f16 0.0 = 0x0000
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        // f16 0.5 = 0x3800
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        // f16 inf = 0x7C00
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn bf16_known_values() {
        // bf16 1.0 = upper 16 bits of f32 1.0 (0x3F800000) = 0x3F80
        assert_eq!(bf16_bits_to_f32(0x3F80), 1.0);
        // bf16 -1.0 = 0xBF80
        assert_eq!(bf16_bits_to_f32(0xBF80), -1.0);
        // bf16 0.0 = 0x0000
        assert_eq!(bf16_bits_to_f32(0x0000), 0.0);
        // bf16 2.0 = 0x4000
        assert_eq!(bf16_bits_to_f32(0x4000), 2.0);
    }

    #[test]
    fn bf16_to_f32_within_tolerance() {
        // bf16 truncates mantissa, so round-trip introduces error.
        // Encode 3.14 as bf16: f32 bits = 0x4048F5C3, upper 16 = 0x4049
        let original = 3.14f32;
        let bf16_bits = (original.to_bits() >> 16) as u16;
        let recovered = bf16_bits_to_f32(bf16_bits);
        // bf16 has 7 mantissa bits → relative error ≤ 2^-7 ≈ 0.0078
        // For |x| ≈ 3.14, max absolute error ≈ 0.025
        assert!(
            (original - recovered).abs() < 1e-1,
            "bf16 roundtrip: {original} → {recovered}"
        );
    }

    #[test]
    fn f16_batch_conversion() {
        let bits = [0x3C00u16, 0x4000, 0x3800]; // 1.0, 2.0, 0.5
        let result = f16_to_f32(&bits);
        assert_eq!(result, vec![1.0, 2.0, 0.5]);
    }

    #[test]
    fn bf16_batch_conversion() {
        let bits = [0x3F80u16, 0x4000, 0xBF80]; // 1.0, 2.0, -1.0
        let result = bf16_to_f32(&bits);
        assert_eq!(result, vec![1.0, 2.0, -1.0]);
    }

    #[test]
    fn transpose_2x3() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let result = transpose(&data, 2, 3);
        // Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity_1x1() {
        let data = [42.0];
        assert_eq!(transpose(&data, 1, 1), vec![42.0]);
    }

    #[test]
    fn transpose_roundtrip() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&data, 2, 3);
        let tt = transpose(&t, 3, 2);
        assert_eq!(tt, data.to_vec());
    }
}
