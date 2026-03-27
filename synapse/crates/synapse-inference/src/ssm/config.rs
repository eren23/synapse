/// Configuration for a Mamba SSM model.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f64,
}

impl MambaConfig {
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    pub fn mamba_130m() -> Self {
        MambaConfig {
            d_model: 768,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            num_layers: 24,
            vocab_size: 50280,
            norm_eps: 1e-5,
        }
    }

    pub fn mamba_370m() -> Self {
        MambaConfig {
            d_model: 1024,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            num_layers: 48,
            vocab_size: 50280,
            norm_eps: 1e-5,
        }
    }

    pub fn tiny_test() -> Self {
        MambaConfig {
            d_model: 64,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            num_layers: 2,
            vocab_size: 128,
            norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_inner_is_expand_times_d_model() {
        let cfg = MambaConfig::mamba_130m();
        assert_eq!(cfg.d_inner(), cfg.expand * cfg.d_model);
        assert_eq!(cfg.d_inner(), 1536);
    }

    #[test]
    fn tiny_test_config_has_expected_values() {
        let cfg = MambaConfig::tiny_test();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.d_state, 4);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.expand, 2);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.vocab_size, 128);
        assert_eq!(cfg.d_inner(), 128);
    }

    #[test]
    fn mamba_370m_has_larger_dims_than_130m() {
        let small = MambaConfig::mamba_130m();
        let large = MambaConfig::mamba_370m();
        assert!(large.d_model > small.d_model);
        assert!(large.num_layers > small.num_layers);
    }
}
