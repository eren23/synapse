//! SSM (State Space Model) and RNN-based architectures.

pub mod deltanet;
pub mod deltanet_state;
pub mod hybrid;
pub mod mamba;
pub mod rwkv;

pub use deltanet::{deltanet_seq, deltanet_step, l2_normalize};
pub use deltanet_state::DeltaNetLayerState;
pub use hybrid::config::HybridConfig;
pub use hybrid::layer::{DeltaNetDecoderLayer, GqaDecoderLayer, KvLayerState};
pub use hybrid::model::{HybridLayer, HybridModel, HybridState};
pub use mamba::block::MambaBlock;
pub use mamba::config::MambaConfig;
pub use mamba::model::MambaModel;
pub use mamba::selective_scan::{compute_delta, selective_scan_seq, selective_scan_step};
pub use mamba::state::{MambaLayerState, RecurrentState};
pub use rwkv::block::RwkvBlock;
pub use rwkv::config::RwkvConfig;
pub use rwkv::model::RwkvModel;
pub use rwkv::state::{RwkvLayerState, RwkvState};
pub use rwkv::wkv::{wkv7_seq, wkv7_step};
