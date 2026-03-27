use serde::{Deserialize, Serialize};

/// Configuration for weight quantization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum QuantConfig {
    F32,
    F16,
    INT8 {
        calibration_method: String,
        calibration_samples: usize,
    },
    Ternary,
}
