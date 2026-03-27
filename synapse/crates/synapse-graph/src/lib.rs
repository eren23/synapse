pub mod constant_fold;
pub mod dead_code;
pub mod fuse_attention;
pub mod fuse_layernorm_residual;
pub mod fusion;
pub mod ir;
pub mod pass;
pub mod scheduler;

pub use constant_fold::ConstantFolding;
pub use dead_code::DeadCodeElimination;
pub use fuse_attention::FuseAttention;
pub use fuse_layernorm_residual::FuseLayerNormResidual;
pub use fusion::{FuseConvBatchNorm, FuseElementWise, FuseMatMulBiasRelu};
pub use ir::{DType, Graph, NodeId, NodeKind, NodeMeta, OpKind};
pub use pass::{run_passes, run_passes_to_fixpoint, OptimizationPass};
pub use scheduler::MemoryOptimalScheduler;
