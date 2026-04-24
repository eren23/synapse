//! `microsoft/unixcoder-base` preset for [`super::roberta::RoBERTaConfig`].

use super::roberta::RoBERTaConfig;

/// Configuration for `microsoft/unixcoder-base`.
///
/// Matches the HuggingFace `config.json` exactly (2026-04 snapshot). UniXcoder
/// is architecturally a RoBERTa-base encoder, differing only in vocabulary
/// and pretraining data; see the official config:
/// `https://huggingface.co/microsoft/unixcoder-base/blob/main/config.json`.
pub fn unixcoder_base() -> RoBERTaConfig {
    RoBERTaConfig {
        vocab_size: 51416,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 1026,
        type_vocab_size: 10,
        layer_norm_eps: 1e-5,
        pad_token_id: 1,
    }
}
