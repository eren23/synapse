//! Configuration for Diffusion LLM (non-autoregressive text generation).

/// Configuration for a bidirectional diffusion language model.
///
/// Unlike autoregressive models, diffusion LLMs generate all tokens
/// simultaneously by iteratively denoising a fully masked sequence.
#[derive(Debug, Clone)]
pub struct DiffusionLLMConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub mask_token_id: u32,
    /// T: total denoising steps
    pub num_denoise_steps: usize,
    pub norm_eps: f64,
}

impl DiffusionLLMConfig {
    /// A tiny configuration for unit testing.
    pub fn tiny_test() -> Self {
        DiffusionLLMConfig {
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 128,
            mask_token_id: 0,
            num_denoise_steps: 5,
            norm_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_test_config_is_consistent() {
        let cfg = DiffusionLLMConfig::tiny_test();
        assert_eq!(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
        assert!(cfg.num_denoise_steps > 0);
        assert!(cfg.vocab_size > 0);
    }
}
