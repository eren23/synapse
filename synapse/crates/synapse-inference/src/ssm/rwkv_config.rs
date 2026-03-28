/// Configuration for an RWKV-7 model.
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    pub hidden_size: usize,       // d_model (e.g., 768 for 0.1B, 1024 for 0.4B)
    pub num_heads: usize,         // number of attention heads
    pub head_size: usize,         // per-head dimension (typically 64)
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize, // FFN hidden size (typically 3.5-4x hidden_size)
    pub norm_eps: f64,
}

impl RwkvConfig {
    pub fn rwkv7_0_1b() -> Self {
        RwkvConfig {
            hidden_size: 768,
            num_heads: 12,
            head_size: 64,
            num_layers: 12,
            vocab_size: 50304,
            intermediate_size: 2688,
            norm_eps: 1e-5,
        }
    }

    pub fn rwkv7_0_4b() -> Self {
        RwkvConfig {
            hidden_size: 1024,
            num_heads: 16,
            head_size: 64,
            num_layers: 24,
            vocab_size: 50304,
            intermediate_size: 3584,
            norm_eps: 1e-5,
        }
    }

    pub fn tiny_test() -> Self {
        RwkvConfig {
            hidden_size: 64,
            num_heads: 2,
            head_size: 32,
            num_layers: 2,
            vocab_size: 128,
            intermediate_size: 128,
            norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rwkv7_0_1b_has_expected_values() {
        let cfg = RwkvConfig::rwkv7_0_1b();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_heads, 12);
        assert_eq!(cfg.head_size, 64);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.vocab_size, 50304);
        assert_eq!(cfg.intermediate_size, 2688);
        assert_eq!(cfg.num_heads * cfg.head_size, cfg.hidden_size);
    }

    #[test]
    fn rwkv7_0_4b_has_expected_values() {
        let cfg = RwkvConfig::rwkv7_0_4b();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_size, 64);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads * cfg.head_size, cfg.hidden_size);
    }

    #[test]
    fn tiny_test_config_has_expected_values() {
        let cfg = RwkvConfig::tiny_test();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_heads, 2);
        assert_eq!(cfg.head_size, 32);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.vocab_size, 128);
        assert_eq!(cfg.intermediate_size, 128);
        assert_eq!(cfg.num_heads * cfg.head_size, cfg.hidden_size);
    }

    #[test]
    fn rwkv7_0_4b_has_larger_dims_than_0_1b() {
        let small = RwkvConfig::rwkv7_0_1b();
        let large = RwkvConfig::rwkv7_0_4b();
        assert!(large.hidden_size > small.hidden_size);
        assert!(large.num_layers > small.num_layers);
        assert!(large.num_heads > small.num_heads);
    }
}
