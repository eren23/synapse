//! Mamba SSM (State Space Model) support.
//!
//! Provides configuration, per-layer recurrent state management, and the
//! selective scan kernels needed for single-step decode and sequence prefill.

pub mod config;
pub mod mamba_block;
pub mod selective_scan;
pub mod state;

pub use config::MambaConfig;
pub use mamba_block::MambaBlock;
pub use selective_scan::{compute_delta, selective_scan_seq, selective_scan_step};
pub use state::{MambaLayerState, RecurrentState};
