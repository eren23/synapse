//! Configuration for Qwen3.5-style hybrid models that combine DeltaNet (linear
//! attention) layers with GQA (full attention) layers.
//!
//! In a typical Qwen3.5 layout with `full_attention_interval = 4`, layers
//! repeat as `[DeltaNet, DeltaNet, DeltaNet, GQA] x N`. DeltaNet layers use
//! constant-size recurrent state while GQA layers use a traditional KV cache.

/// Configuration for a hybrid DeltaNet + GQA model.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f64,

    // GQA (full attention) parameters
    /// Number of query heads for GQA layers.
    pub num_attention_heads: usize,
    /// Number of KV heads for GQA layers (< num_attention_heads for grouped query).
    pub num_kv_heads: usize,
    /// Head dimension for GQA layers.
    pub gqa_head_dim: usize,

    // DeltaNet (linear attention) parameters
    /// Number of heads for DeltaNet layers.
    pub deltanet_num_heads: usize,
    /// Head dimension for DeltaNet layers.
    pub deltanet_head_dim: usize,
    /// Conv1d kernel size for DeltaNet layers (typically 4).
    pub deltanet_conv_kernel: usize,

    // FFN (same for both layer types)
    /// Intermediate (hidden) size of the SwiGLU FFN.
    pub intermediate_size: usize,

    // Layer type pattern
    /// Every Nth layer is a full-attention (GQA) layer. E.g. 4 means indices
    /// 3, 7, 11, ... are GQA and the rest are DeltaNet.
    pub full_attention_interval: usize,
}

impl HybridConfig {
    /// Returns `true` if `layer_idx` is a full-attention (GQA) layer.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval == 0
    }

    /// Number of DeltaNet (linear attention) layers in the model.
    pub fn num_deltanet_layers(&self) -> usize {
        self.num_layers - self.num_gqa_layers()
    }

    /// Number of GQA (full attention) layers in the model.
    pub fn num_gqa_layers(&self) -> usize {
        self.num_layers / self.full_attention_interval
    }

    /// Qwen3.5-0.8B configuration.
    pub fn qwen35_0_8b() -> Self {
        HybridConfig {
            hidden_size: 1024,
            num_layers: 24,
            vocab_size: 151936,
            norm_eps: 1e-6,
            num_attention_heads: 16,
            num_kv_heads: 8,
            gqa_head_dim: 64,
            deltanet_num_heads: 16,
            deltanet_head_dim: 64,
            deltanet_conv_kernel: 4,
            intermediate_size: 3072,
            full_attention_interval: 4,
        }
    }

    /// Tiny configuration for unit tests (4 layers: 3 DeltaNet + 1 GQA).
    pub fn tiny_test() -> Self {
        HybridConfig {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 128,
            norm_eps: 1e-6,
            num_attention_heads: 4,
            num_kv_heads: 2,
            gqa_head_dim: 16,
            deltanet_num_heads: 4,
            deltanet_head_dim: 16,
            deltanet_conv_kernel: 4,
            intermediate_size: 128,
            full_attention_interval: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_full_attention_pattern() {
        let cfg = HybridConfig::qwen35_0_8b();
        // GQA at indices 3, 7, 11, 15, 19, 23
        for i in 0..24 {
            let expected = (i + 1) % 4 == 0;
            assert_eq!(
                cfg.is_full_attention(i),
                expected,
                "layer {i}: expected is_full_attention={expected}"
            );
        }
    }

    #[test]
    fn test_layer_counts() {
        let cfg = HybridConfig::qwen35_0_8b();
        assert_eq!(cfg.num_gqa_layers(), 6);
        assert_eq!(cfg.num_deltanet_layers(), 18);
        assert_eq!(cfg.num_gqa_layers() + cfg.num_deltanet_layers(), cfg.num_layers);
    }

    #[test]
    fn test_tiny_config_counts() {
        let cfg = HybridConfig::tiny_test();
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.num_gqa_layers(), 1);
        assert_eq!(cfg.num_deltanet_layers(), 3);
        assert!(cfg.is_full_attention(3));
        assert!(!cfg.is_full_attention(0));
        assert!(!cfg.is_full_attention(1));
        assert!(!cfg.is_full_attention(2));
    }
}
