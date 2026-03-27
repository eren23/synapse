pub use crate::config::position::RoPEStyle;


/// Apply RoPE rotation to Q or K vectors in-place (rotate-half convention).
///
/// `qk` layout: `[seq_len, num_heads * head_dim]` (flat, heads contiguous).
/// Uses the HuggingFace "rotate_half" convention: pairs dimension `i` with
/// dimension `i + head_dim/2` (first-half/second-half), NOT adjacent pairs.
///
/// cos/sin tables are `[max_pos, head_dim / 2]` (one entry per pair).
/// `pos_offset` is 0 for full-sequence forward, or the cache length for
/// single-token decode steps.
pub(crate) fn apply_rope_inplace(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    style: RoPEStyle,
) {
    let half_d = head_dim / 2;
    let total_dim = num_heads * head_dim;
    for t in 0..seq_len {
        let pos = pos_offset + t;
        let cos_row = pos * half_d;
        for head in 0..num_heads {
            let base = t * total_dim + head * head_dim;
            match style {
                RoPEStyle::RotateHalf => {
                    // Pairs (i, i + d/2): Qwen3, LLaMA 3, Mistral
                    for i in 0..half_d {
                        let idx_first = base + i;
                        let idx_second = base + half_d + i;
                        let cos_val = cos[cos_row + i];
                        let sin_val = sin[cos_row + i];
                        let x_first = qk[idx_first];
                        let x_second = qk[idx_second];
                        qk[idx_first] = x_first * cos_val - x_second * sin_val;
                        qk[idx_second] = x_second * cos_val + x_first * sin_val;
                    }
                }
                RoPEStyle::Interleaved => {
                    // Pairs (2i, 2i+1): GPT-NeoX
                    for i in 0..half_d {
                        let idx_even = base + 2 * i;
                        let idx_odd = base + 2 * i + 1;
                        let cos_val = cos[cos_row + i];
                        let sin_val = sin[cos_row + i];
                        let x_even = qk[idx_even];
                        let x_odd = qk[idx_odd];
                        qk[idx_even] = x_even * cos_val - x_odd * sin_val;
                        qk[idx_odd] = x_odd * cos_val + x_even * sin_val;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build flat cos/sin tables of shape [max_pos, half_d] filled with a
    /// constant value. Using cos=1.0, sin=0.0 gives the identity rotation.
    fn make_rope_tables(max_pos: usize, half_d: usize, cos_val: f32, sin_val: f32) -> (Vec<f32>, Vec<f32>) {
        (vec![cos_val; max_pos * half_d], vec![sin_val; max_pos * half_d])
    }

    #[test]
    fn rope_at_position_zero_is_near_identity() {
        // cos=1, sin=0 at all positions => rotation is identity regardless of pos.
        let head_dim = 8;
        let half_d = head_dim / 2;
        let (cos, sin) = make_rope_tables(16, half_d, 1.0, 0.0);

        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [1, 1, head_dim]
        let mut qk = original.clone();

        apply_rope_inplace(&mut qk, &cos, &sin, 1, 1, head_dim, 0, RoPEStyle::RotateHalf);

        for (i, (&orig, &rotated)) in original.iter().zip(qk.iter()).enumerate() {
            assert!(
                (rotated - orig).abs() < 1e-6,
                "pos=0 with cos=1/sin=0 should be identity at index {i}: got {rotated}, expected {orig}"
            );
        }
    }

    #[test]
    fn rope_different_positions_produce_different_output() {
        // Use real frequencies: theta_i = 1 / (10000^(2i/d))
        let head_dim = 8;
        let half_d = head_dim / 2;
        let max_pos = 32;

        let mut cos_table = vec![0.0f32; max_pos * half_d];
        let mut sin_table = vec![0.0f32; max_pos * half_d];
        for pos in 0..max_pos {
            for i in 0..half_d {
                let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table[pos * half_d + i] = angle.cos();
                sin_table[pos * half_d + i] = angle.sin();
            }
        }

        let input: Vec<f32> = (1..=head_dim as u32).map(|x| x as f32).collect();

        let mut qk0 = input.clone();
        apply_rope_inplace(&mut qk0, &cos_table, &sin_table, 1, 1, head_dim, 0, RoPEStyle::RotateHalf);

        let mut qk10 = input.clone();
        apply_rope_inplace(&mut qk10, &cos_table, &sin_table, 1, 1, head_dim, 10, RoPEStyle::RotateHalf);

        let diff: f32 = qk0.iter().zip(qk10.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "pos=0 and pos=10 should produce different RoPE outputs, diff={diff}");
    }

    #[test]
    fn rope_interleaved_at_identity_is_noop() {
        let head_dim = 8;
        let half_d = head_dim / 2;
        let (cos, sin) = make_rope_tables(16, half_d, 1.0, 0.0);

        let original = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let mut qk = original.clone();

        apply_rope_inplace(&mut qk, &cos, &sin, 1, 1, head_dim, 0, RoPEStyle::Interleaved);

        for (i, (&orig, &rotated)) in original.iter().zip(qk.iter()).enumerate() {
            assert!(
                (rotated - orig).abs() < 1e-6,
                "Interleaved with cos=1/sin=0 should be identity at index {i}"
            );
        }
    }
}
