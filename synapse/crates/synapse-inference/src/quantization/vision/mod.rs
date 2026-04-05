pub mod full_q_lewm;
pub mod int8_code_wm;
pub mod int8_lewm;
pub mod q4_code_wm;
pub mod q4_code_wm_full;
pub mod q4_lewm;
pub mod ternary_lewm;

pub use full_q_lewm::{FullyQuantizedLeWM, quantize_lewm_full, Q4FullLeWM, quantize_lewm_q4_full};
pub use int8_code_wm::{load_and_quantize as load_and_quantize_code_wm, quantize_code_wm, QuantizedCodeWorldModel, QuantizedTransformerBlock};
pub use q4_code_wm::{quantize_code_wm_q4, Q4CodeWorldModel, Q4TransformerBlock};
pub use q4_code_wm_full::{quantize_code_wm_q4_full, Int8Table, Q4FullCodeWorldModel, Q4FullTransformerBlock};
pub use int8_lewm::{quantize_lewm, QuantizedAdaLNLayer, QuantizedLeWM};
pub use q4_lewm::{cached_q4_lewm, quantize_lewm_q4, CachedQ4AdaLNLayer, CachedQ4LeWM, CachedQ4Linear, QuantizedQ4AdaLNLayer, QuantizedQ4LeWM};
pub use ternary_lewm::{TernaryLeWM, TernaryAdaLNLayer, quantize_lewm_ternary};

/// Apply input_proj: `[seq_len, latent_dim]` -> `[seq_len, predictor_hidden]`.
pub(super) fn apply_input_proj(
    weight: &[f32], bias: &[f32],
    seq: &[f32], seq_len: usize, latent: usize, hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = crate::ops::matmul::matmul_t(
            &seq[t * latent..(t + 1) * latent], weight, 1, latent, hidden,
        );
        out[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }
    if !bias.is_empty() {
        for t in 0..seq_len {
            for j in 0..hidden {
                out[t * hidden + j] += bias[j];
            }
        }
    }
    out
}

/// Apply cond_proj: `[latent_dim]` -> `[predictor_hidden]`.
pub(super) fn apply_cond_proj(
    weight: &[f32], bias: &[f32],
    cond: &[f32], latent: usize, hidden: usize,
) -> Vec<f32> {
    let mut out = crate::ops::matmul::matmul_t(cond, weight, 1, latent, hidden);
    if !bias.is_empty() {
        for j in 0..hidden {
            out[j] += bias[j];
        }
    }
    out
}
