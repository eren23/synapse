pub mod lm;
pub mod ssm;
pub mod traits;
pub mod vision;

// Flatten LM types into the `models::` namespace (public API surface).
pub use lm::{CausalLM, DecoderLayer, LoadResult, ModelBuilder, ModelOutput};

// Re-export traits at models:: level
pub use traits::{Model, ModelState};

// Re-export vision types at models:: level
pub use vision::{
    parse_clip_config, parse_clip_config_json, CLIPConfig, CLIPModel,
    JEPAConfig, JEPAModel,
    AdaLNTransformerLayer, LeWMConfig, LeWorldModel,
    parse_vit_config, parse_vit_config_json, parse_vit_labels, parse_vit_labels_json,
    ViTConfig, ViTModel, ViTOutput,
    LatentState, RealtimeRollout, WorldModel, WorldModelConfig,
};

// Re-export SSM types at models:: level
pub use ssm::{
    MambaConfig, MambaBlock, MambaModel,
    compute_delta, selective_scan_seq, selective_scan_step,
    MambaLayerState, RecurrentState,
    RwkvConfig, RwkvBlock, RwkvModel,
    RwkvLayerState, RwkvState,
    wkv7_seq, wkv7_step,
    deltanet_seq, deltanet_step, l2_normalize,
    DeltaNetLayerState,
    HybridConfig,
    DeltaNetDecoderLayer, GqaDecoderLayer, KvLayerState,
    HybridLayer, HybridModel, HybridState,
};
