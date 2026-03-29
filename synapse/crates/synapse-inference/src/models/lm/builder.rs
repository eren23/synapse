use crate::config::position::{RoPEScaling, RoPEStyle};
use crate::config::{ModelConfig, PositionConfig};
use crate::registry::{create_attention, create_ffn, create_norm};
use crate::weight_loading::AlignedBuffer;

use super::causal_lm::CausalLM;
use super::decoder_layer::DecoderLayer;

/// Assembles a [`CausalLM`] from a [`ModelConfig`].
///
/// The returned model has the correct structure (layers, norms, attention
/// geometry) but weights are uninitialized (empty buffers). Call
/// [`CausalLM::load_weights`] to populate them before inference.
pub struct ModelBuilder;

impl ModelBuilder {
    /// Build a `CausalLM` from a model configuration.
    ///
    /// Creates embedding + N decoder layers + final norm + lm_head.
    /// Handles `tie_word_embeddings`: when true, `lm_head_weight` is `None`.
    /// Precomputes RoPE cos/sin tables shared across all layers.
    pub fn from_config(config: &ModelConfig) -> CausalLM {
        let arch = &config.architecture;
        let num_layers = arch.num_layers;

        // Models that use per-head Q/K norms (e.g. Qwen3).
        // LLaMA and Mistral do not have per-head norms.
        let has_head_norms = !matches!(
            config.name.as_str(),
            "llama" | "mistral" | "phi" | "phi3" | "gemma" | "gemma2" | "qwen2" | "qwen2.5" | "vit"
        );

        let rope_style = match &config.position {
            PositionConfig::RoPE { style, .. } => *style,
            _ => RoPEStyle::default(),
        };

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DecoderLayer {
                attn_norm: create_norm(&config.norm),
                attention: create_attention(&config.attention),
                ffn_norm: create_norm(&config.norm),
                ffn: create_ffn(&config.ffn),
                hidden_size: arch.hidden_size,
                has_head_norms,
                rope_style,
                attn_norm_weight: AlignedBuffer::new_zeroed(0),
                w_q: AlignedBuffer::new_zeroed(0),
                w_k: AlignedBuffer::new_zeroed(0),
                w_v: AlignedBuffer::new_zeroed(0),
                w_o: AlignedBuffer::new_zeroed(0),
                q_norm_weight: AlignedBuffer::new_zeroed(0),
                k_norm_weight: AlignedBuffer::new_zeroed(0),
                q_bias: AlignedBuffer::new_zeroed(0),
                k_bias: AlignedBuffer::new_zeroed(0),
                v_bias: AlignedBuffer::new_zeroed(0),
                ffn_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_gate: AlignedBuffer::new_zeroed(0),
                ffn_up: AlignedBuffer::new_zeroed(0),
                ffn_down: AlignedBuffer::new_zeroed(0),
            });
        }

        // Precompute RoPE cos/sin tables
        let (rope_cos, rope_sin) = precompute_rope_tables(config);

        CausalLM {
            final_norm: create_norm(&config.norm),
            layers,
            embed_tokens: AlignedBuffer::new_zeroed(0),
            final_norm_weight: AlignedBuffer::new_zeroed(0),
            lm_head_weight: if arch.tie_word_embeddings {
                None
            } else {
                Some(AlignedBuffer::new_zeroed(0))
            },
            config: config.clone(),
            rope_cos,
            rope_sin,
        }
    }
}

/// Precompute RoPE cosine and sine tables from model config.
///
/// Returns `(cos, sin)` each shaped `[max_pos, head_dim / 2]` (flat).
fn precompute_rope_tables(config: &ModelConfig) -> (Vec<f32>, Vec<f32>) {
    let (base, max_pos, scaling) = match &config.position {
        PositionConfig::RoPE {
            base,
            max_position_embeddings,
            scaling,
            ..
        } => (*base, *max_position_embeddings, *scaling),
        _ => (
            10_000.0,
            config.architecture.max_sequence_length,
            RoPEScaling::None,
        ),
    };
    let head_dim = config.attention.head_dim();
    let half_d = head_dim / 2;

    // Apply dynamic NTK scaling to base frequency
    let effective_base = match scaling {
        RoPEScaling::Dynamic { factor } => {
            base * factor.powf((head_dim as f64) / ((head_dim as f64) - 2.0))
        }
        _ => base,
    };

    // Linear scaling factor applied to frequencies
    let linear_factor = match scaling {
        RoPEScaling::Linear { factor } => factor as f32,
        _ => 1.0,
    };

    let mut cos_data = vec![0.0f32; max_pos * half_d];
    let mut sin_data = vec![0.0f32; max_pos * half_d];

    for pos in 0..max_pos {
        for i in 0..half_d {
            let freq = 1.0 / (effective_base as f32).powf(2.0 * i as f32 / head_dim as f32);
            let scaled_freq = freq / linear_factor;
            let angle = pos as f32 * scaled_freq;
            cos_data[pos * half_d + i] = angle.cos();
            sin_data[pos * half_d + i] = angle.sin();
        }
    }

    (cos_data, sin_data)
}
