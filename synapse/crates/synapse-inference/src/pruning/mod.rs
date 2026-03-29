//! Model surgery: sensitivity analysis, layer removal, weight pruning, and SSM-aware pruning.
//!
//! The pruning pipeline follows a principled order:
//! 1. **Sensitivity analysis** — measure layer importance via output divergence
//! 2. **Layer removal** — drop near-identity layers (ShortGPT-style)
//! 3. **Weight pruning** — Wanda-style magnitude × activation pruning
//! 4. **SSM-aware pruning** — channel/head pruning for Mamba/RWKV
//! 5. **Quantize** — apply INT8/Q4 quantization (via existing quantization module)
//!
//! For models <200M params, research shows prune-then-quantize gives optimal
//! compression for edge deployment (ESP32, WASM).

pub mod layer_removal;
pub mod sensitivity;
pub mod ssm_pruning;
pub mod wanda;
pub mod pipeline;

pub use sensitivity::{LayerImportance, SensitivityAnalyzer};
pub use layer_removal::LayerRemover;
pub use wanda::WandaPruner;
pub use ssm_pruning::{MambaChannelPruner, RwkvHeadPruner};
pub use pipeline::{PruningStrategy, SurgeonPipeline, SurgeryReport};
