//! Shared math operations used across f32 and quantized inference paths.

pub mod activation;
pub mod attention;
pub mod fused_ops;
pub mod geometric;
pub mod matmul;
pub mod norm;
pub mod patch_embed;
pub mod projection;
pub mod pure_rust_ops;
pub mod rope;
pub mod vector;
