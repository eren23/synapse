//! HTTP server for inference over WiFi.
//!
//! Endpoints:
//!   POST /encode   -- image -> latent  (LeWM only)
//!   POST /predict  -- latent + action -> next latent  (LeWM only)
//!   POST /rollout  -- latent + actions -> trajectory  (LeWM only)
//!   POST /llm/generate -- prompt tokens -> generated tokens  (Mamba/RWKV)
//!   GET  /model/info   -- model metadata
//!   GET  /status   -- model info + memory usage

use serde::{Deserialize, Serialize};
use crate::model::{Esp32Model, ModelInfo};

// ---------------------------------------------------------------------------
// LeWM request/response types (unchanged)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LLM request type
// ---------------------------------------------------------------------------

/// Request for /llm/generate endpoint.
#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
}

// ---------------------------------------------------------------------------
// LeWM handlers (unchanged, still take &Esp32LeWM)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Multi-model handlers
// ---------------------------------------------------------------------------

/// Handle a generate request. Only works for LLM model variants (Mamba, RWKV).
pub fn handle_generate(model: &Esp32Model, req: GenerateRequest) -> InferenceResponse {
    match model {
        Esp32Model::Mamba(m) => {
            let result = m.generate(&req.prompt_tokens, req.max_tokens, req.temperature);
            InferenceResponse {
                result: serde_json::json!({
                    "tokens": result.tokens,
                    "tokens_per_sec": result.tokens_per_sec,
                }),
                latency_ms: result.latency_ms,
                operation: "generate".into(),
            }
        }
        Esp32Model::Rwkv(r) => {
            let result = r.generate(&req.prompt_tokens, req.max_tokens, req.temperature);
            InferenceResponse {
                result: serde_json::json!({
                    "tokens": result.tokens,
                    "tokens_per_sec": result.tokens_per_sec,
                }),
                latency_ms: result.latency_ms,
                operation: "generate".into(),
            }
        }
        Esp32Model::LeWM(_) => InferenceResponse {
            result: serde_json::json!({ "error": "not a language model" }),
            latency_ms: 0.0,
            operation: "generate".into(),
        },
    }
}

/// Handle a model info request.
pub fn handle_model_info(model: &Esp32Model) -> ModelInfo {
    model.model_info()
}

/// Handle a status request for any model type.
pub fn handle_status(model: &Esp32Model) -> StatusResponse {
    let backend = if cfg!(feature = "esp32") {
        "ESP32-P4 PIE".into()
    } else {
        "pure-rust".into()
    };

    match model {
        Esp32Model::LeWM(m) => StatusResponse {
            model: "LeWorldModel (PushT)".into(),
            latent_dim: m.latent_dim(),
            action_dim: m.action_dim(),
            quantization: "Q4_0".into(),
            backend,
        },
        Esp32Model::Mamba(_) => {
            let info = model.model_info();
            StatusResponse {
                model: info.name,
                latent_dim: 0,
                action_dim: 0,
                quantization: info.quantization,
                backend,
            }
        }
        Esp32Model::Rwkv(_) => {
            let info = model.model_info();
            StatusResponse {
                model: info.name,
                latent_dim: 0,
                action_dim: 0,
                quantization: info.quantization,
                backend,
            }
        }
    }
}

/// Handle a status request for an Esp32LeWM specifically (backwards compatible).
pub fn handle_lewm_status(model: &crate::model::Esp32LeWM) -> StatusResponse {
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
    use crate::model::{Esp32LeWM, Esp32Mamba, Esp32Rwkv};

    #[test]
    fn handle_status_returns_lewm_info() {
        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
        let status = handle_status(&model);
        assert_eq!(status.latent_dim, 192);
        assert_eq!(status.action_dim, 10);
        assert_eq!(status.model, "LeWorldModel (PushT)");
        assert_eq!(status.quantization, "Q4_0");
    }

    #[test]
    fn handle_status_returns_mamba_info() {
        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let status = handle_status(&model);
        assert!(status.model.contains("Mamba"));
        assert_eq!(status.quantization, "Q4_0");
    }

    #[test]
    fn handle_status_returns_rwkv_info() {
        let model = Esp32Model::Rwkv(Esp32Rwkv::new_zeroed());
        let status = handle_status(&model);
        assert!(status.model.contains("RWKV"));
        assert_eq!(status.quantization, "Q4_0");
    }

    #[test]
    fn handle_lewm_status_backwards_compat() {
        let model = Esp32LeWM::new_zeroed();
        let status = handle_lewm_status(&model);
        assert_eq!(status.latent_dim, 192);
        assert_eq!(status.action_dim, 10);
    }

    #[test]
    fn status_response_serializes_to_json() {
        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
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
    fn generate_request_deserializes() {
        let json = r#"{"prompt_tokens":[1,2,3],"max_tokens":10,"temperature":0.7}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(req.max_tokens, 10);
        assert!((req.temperature - 0.7).abs() < 1e-6);
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

    #[test]
    fn handle_generate_mamba_returns_tokens() {
        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let req = GenerateRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 5,
            temperature: 1.0,
        };
        let resp = handle_generate(&model, req);
        assert_eq!(resp.operation, "generate");
        let tokens = resp.result["tokens"].as_array().unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn handle_generate_rwkv_returns_tokens() {
        let model = Esp32Model::Rwkv(Esp32Rwkv::new_zeroed());
        let req = GenerateRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 5,
            temperature: 1.0,
        };
        let resp = handle_generate(&model, req);
        assert_eq!(resp.operation, "generate");
        let tokens = resp.result["tokens"].as_array().unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn handle_generate_lewm_returns_error() {
        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
        let req = GenerateRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 5,
            temperature: 1.0,
        };
        let resp = handle_generate(&model, req);
        let err = resp.result["error"].as_str().unwrap();
        assert_eq!(err, "not a language model");
    }

    #[test]
    fn handle_model_info_delegates() {
        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let info = handle_model_info(&model);
        assert_eq!(info.model_type, "mamba");

        let model = Esp32Model::Rwkv(Esp32Rwkv::new_zeroed());
        let info = handle_model_info(&model);
        assert_eq!(info.model_type, "rwkv");

        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
        let info = handle_model_info(&model);
        assert_eq!(info.model_type, "lewm");
    }
}
