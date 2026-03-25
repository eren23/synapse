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
