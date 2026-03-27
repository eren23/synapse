//! HTTP server for LEWM inference over WiFi.
//!
//! Endpoints:
//!   POST /encode   -- image -> latent
//!   POST /predict  -- latent + action -> next latent
//!   POST /rollout  -- latent + actions -> trajectory
//!   GET  /status   -- model info + memory usage

use serde::{Deserialize, Serialize};

/// Request for /encode endpoint.
#[derive(Deserialize)]
pub struct EncodeRequest {
    /// Base64-encoded image data (224x224x3 RGB, f32 normalized to [0,1])
    pub image: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// Request for /predict endpoint.
#[derive(Deserialize)]
pub struct PredictRequest {
    pub latent: Vec<f32>,
    pub action: Vec<f32>,
}

/// Request for /rollout endpoint.
#[derive(Deserialize)]
pub struct RolloutRequest {
    pub latent: Vec<f32>,
    pub actions: Vec<Vec<f32>>,
}

/// Response for any inference endpoint.
#[derive(Serialize)]
pub struct InferenceResponse {
    pub result: serde_json::Value,
    pub latency_ms: f64,
    pub operation: String,
}

/// Response for /status endpoint.
#[derive(Serialize)]
pub struct StatusResponse {
    pub model: String,
    pub latent_dim: usize,
    pub action_dim: usize,
    pub quantization: String,
    pub backend: String,
}

/// Handle an encode request.
pub fn handle_encode(model: &crate::model::Esp32LeWM, req: EncodeRequest) -> InferenceResponse {
    let (latent, metrics) = model.encode(&req.image, req.height, req.width);
    InferenceResponse {
        result: serde_json::json!({ "latent": latent }),
        latency_ms: metrics.latency_ms,
        operation: metrics.operation,
    }
}

/// Handle a predict request.
pub fn handle_predict(model: &crate::model::Esp32LeWM, req: PredictRequest) -> InferenceResponse {
    let (next, metrics) = model.predict_next(&req.latent, &req.action);
    InferenceResponse {
        result: serde_json::json!({ "next_latent": next }),
        latency_ms: metrics.latency_ms,
        operation: metrics.operation,
    }
}

/// Handle a rollout request.
pub fn handle_rollout(model: &crate::model::Esp32LeWM, req: RolloutRequest) -> InferenceResponse {
    let (trajectory, metrics) = model.rollout(&req.latent, &req.actions);
    InferenceResponse {
        result: serde_json::json!({ "trajectory": trajectory }),
        latency_ms: metrics.latency_ms,
        operation: metrics.operation,
    }
}

/// Handle a status request.
pub fn handle_status(model: &crate::model::Esp32LeWM) -> StatusResponse {
    StatusResponse {
        model: "LeWorldModel (PushT)".into(),
        latent_dim: model.latent_dim(),
        action_dim: model.action_dim(),
        quantization: "Q4_0".into(),
        backend: if cfg!(feature = "esp32") { "ESP32-P4 PIE".into() } else { "pure-rust".into() },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_status_returns_model_info() {
        let model = crate::model::Esp32LeWM::new_zeroed();
        let status = handle_status(&model);
        assert_eq!(status.latent_dim, 192);
        assert_eq!(status.action_dim, 10);
        assert_eq!(status.model, "LeWorldModel (PushT)");
        assert_eq!(status.quantization, "Q4_0");
    }

    #[test]
    fn status_response_serializes_to_json() {
        let model = crate::model::Esp32LeWM::new_zeroed();
        let status = handle_status(&model);
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("LeWorldModel"));
        assert!(json.contains("192"));
    }

    #[test]
    fn request_types_deserialize() {
        let json = r#"{"latent":[0.1,0.2],"action":[0.3]}"#;
        let req: PredictRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.latent.len(), 2);
        assert_eq!(req.action.len(), 1);

        let json = r#"{"latent":[0.1],"actions":[[0.2],[0.3]]}"#;
        let req: RolloutRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.actions.len(), 2);

        let json = r#"{"image":[0.5,0.5,0.5],"height":1,"width":1}"#;
        let req: EncodeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image.len(), 3);
        assert_eq!(req.height, 1);
    }

    #[test]
    fn inference_response_serializes() {
        let resp = InferenceResponse {
            result: serde_json::json!({"latent": [0.1, 0.2]}),
            latency_ms: 5.0,
            operation: "predict".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("predict"));
        assert!(json.contains("5.0"));
    }
}
