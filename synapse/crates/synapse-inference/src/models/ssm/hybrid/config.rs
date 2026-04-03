//! Configuration for hybrid models that combine different layer types
//! (DeltaNet, GQA, LIV Conv) in an interleaved pattern.
//!
//! Supports two layer-pattern modes:
//! 1. **Interval** (Qwen3.5): `full_attention_interval = 4` → `[DN,DN,DN,GQA] x N`
//! 2. **Explicit** (LFM2.5): `layer_types = Some(vec![Conv,Conv,Gqa,...])` for
//!    arbitrary patterns.

/// The kind of decoder layer at a given position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// DeltaNet gated linear attention (constant recurrent state).
    DeltaNet,
    /// Grouped Query Attention (KV cache).
    Gqa,
    /// Gated depthwise convolution (LFM2.5 LIV Conv).
    LivConv,
}

/// Configuration for a hybrid model combining multiple layer types.
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

    // LIV Conv parameters
    /// Inner dimension for LIV Conv layers. Typically == hidden_size.
    pub livconv_inner_size: usize,
    /// Depthwise conv kernel size for LIV Conv layers (typically 4).
    pub livconv_kernel_size: usize,

    // FFN (same for all layer types)
    /// Intermediate (hidden) size of the SwiGLU FFN.
    pub intermediate_size: usize,

    // Layer type pattern
    /// Every Nth layer is a full-attention (GQA) layer. E.g. 4 means indices
    /// 3, 7, 11, ... are GQA and the rest are DeltaNet.
    /// Used only when `layer_types` is `None`.
    pub full_attention_interval: usize,

    /// Explicit per-layer type assignment. When `Some`, overrides
    /// `full_attention_interval`. Length must equal `num_layers`.
    pub layer_types: Option<Vec<LayerKind>>,

    /// RoPE theta for GQA layers (default 10000.0).
    pub rope_theta: f32,

    /// Whether embeddings are tied to the LM head.
    pub tie_embedding: bool,
}

impl HybridConfig {
    /// Returns the kind of layer at position `layer_idx`.
    ///
    /// Uses `layer_types` if set, otherwise falls back to `full_attention_interval`.
    pub fn layer_kind(&self, layer_idx: usize) -> LayerKind {
        if let Some(ref types) = self.layer_types {
            types[layer_idx]
        } else if (layer_idx + 1) % self.full_attention_interval == 0 {
            LayerKind::Gqa
        } else {
            LayerKind::DeltaNet
        }
    }

    /// Returns `true` if `layer_idx` is a full-attention (GQA) layer.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_kind(layer_idx) == LayerKind::Gqa
    }

    /// Number of DeltaNet (linear attention) layers in the model.
    pub fn num_deltanet_layers(&self) -> usize {
        (0..self.num_layers).filter(|&i| self.layer_kind(i) == LayerKind::DeltaNet).count()
    }

    /// Number of GQA (full attention) layers in the model.
    pub fn num_gqa_layers(&self) -> usize {
        (0..self.num_layers).filter(|&i| self.layer_kind(i) == LayerKind::Gqa).count()
    }

    /// Number of LIV Conv layers in the model.
    pub fn num_livconv_layers(&self) -> usize {
        (0..self.num_layers).filter(|&i| self.layer_kind(i) == LayerKind::LivConv).count()
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
            livconv_inner_size: 0,
            livconv_kernel_size: 4,
            intermediate_size: 3072,
            full_attention_interval: 4,
            layer_types: None,
            rope_theta: 10000.0,
            tie_embedding: false,
        }
    }

    /// LFM2.5-350M configuration (LiquidAI).
    ///
    /// 16 layers: 10 gated LIV conv + 6 GQA attention.
    /// Non-regular interleaving: C,C,A,C,C,A,C,C,A,C,A,C,A,C,A,C
    pub fn lfm25_350m() -> Self {
        use LayerKind::{Gqa, LivConv as C};
        HybridConfig {
            hidden_size: 1024,
            num_layers: 16,
            vocab_size: 65536,
            norm_eps: 1e-5,
            num_attention_heads: 16,
            num_kv_heads: 8,
            gqa_head_dim: 64, // 1024 / 16
            deltanet_num_heads: 0,
            deltanet_head_dim: 0,
            deltanet_conv_kernel: 0,
            livconv_inner_size: 1024,
            livconv_kernel_size: 3, // conv_L_cache=3 = kernel_size
            intermediate_size: 6656,
            full_attention_interval: 1, // unused — layer_types takes precedence
            layer_types: Some(vec![
                C, C, Gqa, C, C, Gqa, C, C, Gqa, C, Gqa, C, Gqa, C, Gqa, C,
            ]),
            rope_theta: 1_000_000.0,
            tie_embedding: true,
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
            livconv_inner_size: 0,
            livconv_kernel_size: 4,
            intermediate_size: 128,
            full_attention_interval: 4,
            layer_types: None,
            rope_theta: 10000.0,
            tie_embedding: false,
        }
    }

    /// Tiny LIV Conv test config (4 layers: 2 LivConv + 2 GQA).
    pub fn tiny_test_livconv() -> Self {
        use LayerKind::{Gqa, LivConv as C};
        HybridConfig {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 128,
            norm_eps: 1e-6,
            num_attention_heads: 4,
            num_kv_heads: 2,
            gqa_head_dim: 16,
            deltanet_num_heads: 0,
            deltanet_head_dim: 0,
            deltanet_conv_kernel: 0,
            livconv_inner_size: 64,
            livconv_kernel_size: 4,
            intermediate_size: 128,
            full_attention_interval: 1, // unused
            layer_types: Some(vec![C, Gqa, C, Gqa]),
            rope_theta: 10000.0,
            tie_embedding: false,
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
        assert_eq!(cfg.num_livconv_layers(), 0);
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

    #[test]
    fn test_lfm25_layer_pattern() {
        let cfg = HybridConfig::lfm25_350m();
        assert_eq!(cfg.num_layers, 16);
        assert_eq!(cfg.num_gqa_layers(), 6);
        assert_eq!(cfg.num_livconv_layers(), 10);
        assert_eq!(cfg.num_deltanet_layers(), 0);

        // Verify the exact interleaving: C,C,A,C,C,A,C,C,A,C,A,C,A,C,A,C
        let expected = [
            LayerKind::LivConv, LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv, LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv, LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv, LayerKind::Gqa,
            LayerKind::LivConv,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(cfg.layer_kind(i), exp, "layer {i} mismatch");
        }
    }

    #[test]
    fn test_tiny_livconv_config() {
        let cfg = HybridConfig::tiny_test_livconv();
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.num_livconv_layers(), 2);
        assert_eq!(cfg.num_gqa_layers(), 2);
        assert_eq!(cfg.num_deltanet_layers(), 0);
        assert_eq!(cfg.layer_kind(0), LayerKind::LivConv);
        assert_eq!(cfg.layer_kind(1), LayerKind::Gqa);
        assert_eq!(cfg.layer_kind(2), LayerKind::LivConv);
        assert_eq!(cfg.layer_kind(3), LayerKind::Gqa);
    }
}
