//! LEWM model loading and inference for ESP32.
//!
//! Uses Q4 quantization (~7MB weights) to fit in 32MB PSRAM.

use std::time::Instant;
use synapse_inference::model::lewm::{LeWMConfig, LeWorldModel};

/// Loaded LEWM model ready for inference on ESP32.
pub struct Esp32LeWM {
    config: LeWMConfig,
    // For now, use f32 model. When Q4 binary loading is ready,
    // switch to QuantizedQ4LeWM.
    model: LeWorldModel,
}

/// Timing information for a single inference call.
#[derive(Debug, serde::Serialize)]
pub struct InferenceMetrics {
    pub operation: String,
    pub latency_ms: f64,
    pub output_dim: usize,
}

impl Esp32LeWM {
    /// Create a model with default PushT config and zeroed weights.
    /// For testing without real weights.
    pub fn new_zeroed() -> Self {
        let config = LeWMConfig::pusht();
        let model = LeWorldModel::from_config(&config);
        Esp32LeWM { config, model }
    }

    /// Load model from compact binary data.
    /// Format: [u32 header_len][JSON header][weight data]
    /// TODO: Implement when weight conversion script is ready.
    pub fn from_binary(_data: &[u8]) -> Result<Self, String> {
        // Placeholder -- will be implemented with convert_lewm_q4_esp32.py
        Err("Binary loading not yet implemented. Use new_zeroed() for testing.".into())
    }

    /// Encode an image to a latent state.
    /// image: flat [H*W*3] f32 pixel data, normalized to [0,1].
    pub fn encode(&self, image: &[f32], height: usize, width: usize) -> (Vec<f32>, InferenceMetrics) {
        let start = Instant::now();
        let latent = self.model.encode(image, height, width);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let metrics = InferenceMetrics {
            operation: "encode".into(),
            latency_ms: ms,
            output_dim: latent.len(),
        };
        (latent, metrics)
    }

    /// Predict next latent state given current state and action.
    pub fn predict_next(&self, state: &[f32], action: &[f32]) -> (Vec<f32>, InferenceMetrics) {
        let start = Instant::now();
        let next = self.model.predict_next(state, action);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let metrics = InferenceMetrics {
            operation: "predict".into(),
            latency_ms: ms,
            output_dim: next.len(),
        };
        (next, metrics)
    }

    /// Multi-step rollout.
    pub fn rollout(&self, state: &[f32], actions: &[Vec<f32>]) -> (Vec<Vec<f32>>, InferenceMetrics) {
        let start = Instant::now();
        let trajectory = self.model.rollout(state, actions);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let steps = trajectory.len();
        let metrics = InferenceMetrics {
            operation: format!("rollout_{steps}_steps"),
            latency_ms: ms,
            output_dim: self.config.latent_dim,
        };
        (trajectory, metrics)
    }

    pub fn config(&self) -> &LeWMConfig { &self.config }
    pub fn latent_dim(&self) -> usize { self.config.latent_dim }
    pub fn action_dim(&self) -> usize { self.config.action_dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esp32_model_creates_with_zeroed_weights() {
        let model = Esp32LeWM::new_zeroed();
        assert_eq!(model.latent_dim(), 192);
        assert_eq!(model.action_dim(), 10);
    }

    #[test]
    fn esp32_model_config_matches_pusht() {
        let model = Esp32LeWM::new_zeroed();
        let cfg = model.config();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.predictor_layers, 6);
        assert_eq!(cfg.predictor_heads, 16);
    }

    #[test]
    fn esp32_model_binary_loading_not_yet_implemented() {
        let result = Esp32LeWM::from_binary(&[0u8; 64]);
        assert!(result.is_err());
    }

    #[test]
    fn esp32_inference_metrics_serializes() {
        let m = InferenceMetrics {
            operation: "predict".into(),
            latency_ms: 42.5,
            output_dim: 192,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("predict"));
        assert!(json.contains("42.5"));
    }
}
