//! Encoder-only text transformers (RoBERTa family, UniXcoder, CodeBERT).
//!
//! Unlike [`crate::models::vision::vit::EncoderLayer`] which is pre-norm,
//! these use the HuggingFace RoBERTa post-norm convention
//! (`attn → add → LN → FFN → add → LN`) and require padding-aware attention
//! so the `[CLS]` feature matches HF numerics bit-for-bit.
//!
//! The primary entry point is [`unixcoder_base`], which returns a
//! [`RoBERTaConfig`] for `microsoft/unixcoder-base` — the frozen backbone used
//! by the CodeDeltaTok paper (codewm3). Feed the whole file through
//! [`RoBERTaEncoder::cls_feature`] to reproduce the tap's
//! `precompute_backbone_features.py` output.

pub mod code_deltatok;
pub mod code_deltatok_q4;
pub mod roberta;
pub mod roberta_q4;
pub mod unixcoder;

pub use code_deltatok::{CodeDeltaTokConfig, CodeDeltaTokHead, DeltaTokBlock};
pub use code_deltatok_q4::{Q4CodeDeltaTokHead, Q4DeltaTokBlock};
pub use roberta::{
    parse_roberta_config, RoBERTaConfig, RoBERTaEmbeddings, RoBERTaEncoder, RoBERTaLayer,
};
pub use roberta_q4::{Q4RoBERTaEncoder, Q4RoBERTaLayer};
pub use unixcoder::unixcoder_base;
