pub mod ir;
pub mod pass;
pub mod fusion;
pub mod fuse_attention;
pub mod fuse_layernorm_residual;
pub mod dead_code;
pub mod constant_fold;
pub mod scheduler;

pub use ir::{DType, Graph, NodeId, NodeKind, NodeMeta, OpKind};
pub use pass::{OptimizationPass, run_passes, run_passes_to_fixpoint};
pub use fusion::{FuseMatMulBiasRelu, FuseConvBatchNorm, FuseElementWise};
pub use fuse_attention::FuseAttention;
pub use fuse_layernorm_residual::FuseLayerNormResidual;
pub use dead_code::DeadCodeElimination;
pub use constant_fold::ConstantFolding;
pub use scheduler::MemoryOptimalScheduler;
