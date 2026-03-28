//! SSM (State Space Model) and RNN-based architectures.
//!
//! Provides configuration, per-layer recurrent state management, and the
//! kernels needed for single-step decode and sequence prefill.
//!
//! Supported architectures:
//! - **Mamba**: Selective State Space Model with selective scan.
//! - **RWKV-7**: RNN-based architecture with WKV recurrence.
//! - **DeltaNet**: Gated linear attention (used in Qwen3.5 hybrid models).

pub mod config;
pub mod deltanet;
pub mod deltanet_state;
pub mod mamba_block;
pub mod mamba_model;
pub mod rwkv_block;
pub mod rwkv_config;
pub mod rwkv_model;
pub mod rwkv_state;
pub mod selective_scan;
pub mod state;
pub mod wkv;

pub use config::MambaConfig;
pub use deltanet::{deltanet_seq, deltanet_step, l2_normalize};
pub use deltanet_state::DeltaNetLayerState;
pub use mamba_block::MambaBlock;
pub use mamba_model::MambaModel;
pub use rwkv_block::RwkvBlock;
pub use rwkv_config::RwkvConfig;
pub use rwkv_model::RwkvModel;
pub use rwkv_state::{RwkvLayerState, RwkvState};
pub use selective_scan::{compute_delta, selective_scan_seq, selective_scan_step};
pub use state::{MambaLayerState, RecurrentState};
pub use wkv::{wkv_seq, wkv_step};
