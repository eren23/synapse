pub mod ir;
pub mod pass;
pub mod fusion;
pub mod dead_code;
pub mod constant_fold;
pub mod scheduler;

pub use ir::{DType, Graph, NodeId, NodeKind, NodeMeta, OpKind};
pub use pass::{OptimizationPass, run_passes, run_passes_to_fixpoint};
pub use fusion::{FuseMatMulBiasRelu, FuseConvBatchNorm, FuseElementWise};
pub use dead_code::DeadCodeElimination;
pub use constant_fold::ConstantFolding;
pub use scheduler::MemoryOptimalScheduler;
