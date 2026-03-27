use serde::{Deserialize, Serialize};
#[cfg(feature = "zig-ffi")]
use synapse_core::{capability_summary, CapabilityRuntimeProfile, CapabilitySupportLevel};

const STATUS_MANIFEST_JSON: &str = include_str!("../../../status/public_status.json");

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProfile {
    NativePerf,
    ArmCompact,
    WasmPortable,
}

impl RuntimeProfile {
    pub fn for_current_target() -> Self {
        if cfg!(target_arch = "wasm32") {
            Self::WasmPortable
        } else if cfg!(target_arch = "aarch64") && !cfg!(target_os = "macos") {
            Self::ArmCompact
        } else {
            Self::NativePerf
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportLevel {
    Stable,
    Beta,
    Experimental,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelSupportLevel {
    Validated,
    BenchmarkedLocal,
    ConfigReady,
    InProgress,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureStatus {
    pub id: String,
    pub label: String,
    pub support: SupportLevel,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelProfile {
    pub id: String,
    pub label: String,
    pub status: ModelSupportLevel,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactBudget {
    pub id: String,
    pub label: String,
    pub max_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NativeKernelInfo {
    pub abi_version: u32,
    pub target_arch: String,
    pub target_os: String,
    pub simd_backend: String,
    pub runtime_profile: RuntimeProfile,
    pub support_level: SupportLevel,
    pub feature_bits: u64,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityReport {
    pub manifest_version: u32,
    pub last_verified: String,
    pub runtime_profile: RuntimeProfile,
    pub target: String,
    pub summary: String,
    pub backends: Vec<String>,
    pub quantization: Vec<String>,
    pub loaded_model: Option<String>,
    pub model_families: Vec<ModelProfile>,
    pub features: Vec<FeatureStatus>,
    pub artifact_budgets: Vec<ArtifactBudget>,
    pub native_kernel: Option<NativeKernelInfo>,
}

impl CapabilityReport {
    pub fn for_current_build() -> Self {
        Self::for_model_name(None)
    }

    pub fn for_model_name(model_name: Option<&str>) -> Self {
        let manifest: Manifest =
            serde_json::from_str(STATUS_MANIFEST_JSON).expect("status manifest should parse");
        let runtime_profile = RuntimeProfile::for_current_target();
        let profile = manifest
            .runtime_profiles
            .iter()
            .find(|item| item.id == runtime_profile)
            .expect("runtime profile should exist in status manifest");

        let summary = match runtime_profile {
            RuntimeProfile::WasmPortable => manifest.positioning.wasm_runtime.clone(),
            _ => manifest.positioning.native_runtime.clone(),
        };

        Self {
            manifest_version: manifest.manifest_version,
            last_verified: manifest.last_verified,
            runtime_profile,
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            summary,
            backends: profile.backends.clone(),
            quantization: profile.quantization.clone(),
            loaded_model: model_name.map(str::to_owned),
            model_families: manifest
                .model_families
                .into_iter()
                .map(|item| ModelProfile {
                    id: item.id,
                    label: item.label,
                    status: item.status,
                    notes: item.notes,
                })
                .collect(),
            features: manifest
                .features
                .into_iter()
                .map(|item| FeatureStatus {
                    id: item.id,
                    label: item.label,
                    support: item.support,
                    details: item.details,
                })
                .collect(),
            artifact_budgets: manifest
                .artifact_budgets
                .into_iter()
                .map(|item| ArtifactBudget {
                    id: item.id,
                    label: item.label,
                    max_bytes: item.max_bytes,
                })
                .collect(),
            native_kernel: native_kernel_info(),
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(feature = "zig-ffi")]
fn native_kernel_info() -> Option<NativeKernelInfo> {
    if cfg!(target_arch = "wasm32") {
        return None;
    }

    let summary = capability_summary().ok()?;
    Some(NativeKernelInfo {
        abi_version: summary.abi_version,
        target_arch: summary.target_arch.as_str().to_owned(),
        target_os: summary.target_os.as_str().to_owned(),
        simd_backend: summary.simd_backend.as_str().to_owned(),
        runtime_profile: match summary.runtime_profile {
            CapabilityRuntimeProfile::NativePerf => RuntimeProfile::NativePerf,
            CapabilityRuntimeProfile::ArmCompact => RuntimeProfile::ArmCompact,
            CapabilityRuntimeProfile::WasmPortable => RuntimeProfile::WasmPortable,
        },
        support_level: match summary.support_level {
            CapabilitySupportLevel::Stable => SupportLevel::Stable,
            CapabilitySupportLevel::Beta => SupportLevel::Beta,
            CapabilitySupportLevel::Experimental => SupportLevel::Experimental,
        },
        feature_bits: summary.feature_bits,
        feature_names: summary
            .feature_names()
            .into_iter()
            .map(str::to_owned)
            .collect(),
    })
}

#[cfg(not(feature = "zig-ffi"))]
fn native_kernel_info() -> Option<NativeKernelInfo> {
    None
}

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    manifest_version: u32,
    last_verified: String,
    positioning: Positioning,
    runtime_profiles: Vec<ManifestRuntimeProfile>,
    model_families: Vec<ManifestModelFamily>,
    features: Vec<ManifestFeature>,
    artifact_budgets: Vec<ManifestArtifactBudget>,
}

#[derive(Debug, Clone, Deserialize)]
struct Positioning {
    native_runtime: String,
    wasm_runtime: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestRuntimeProfile {
    id: RuntimeProfile,
    backends: Vec<String>,
    quantization: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestModelFamily {
    id: String,
    label: String,
    status: ModelSupportLevel,
    notes: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestFeature {
    id: String,
    label: String,
    support: SupportLevel,
    details: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestArtifactBudget {
    id: String,
    label: String,
    max_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::engine::InferenceEngine;

    const QWEN3_JSON: &str = include_str!("../../../configs/qwen3_0.6b.json");

    #[test]
    fn current_build_report_has_runtime_profile() {
        let report = CapabilityReport::for_current_build();
        assert!(!report.backends.is_empty());
        assert!(!report.quantization.is_empty());
        assert_eq!(report.manifest_version, 2);
        assert_eq!(report.model_families.len(), 5);
        if !cfg!(target_arch = "wasm32") && cfg!(feature = "zig-ffi") {
            assert!(report.native_kernel.is_some());
        }
    }

    #[test]
    fn report_round_trips_to_json() {
        let report = CapabilityReport::for_model_name(Some("Qwen3-0.6B"));
        let json = report.to_json().expect("report should serialize");
        let parsed: CapabilityReport = serde_json::from_str(&json).expect("report should parse");
        assert_eq!(parsed.loaded_model.as_deref(), Some("Qwen3-0.6B"));
    }

    #[test]
    fn native_targets_default_to_native_perf_or_arm_compact() {
        let profile = RuntimeProfile::for_current_target();
        assert!(matches!(
            profile,
            RuntimeProfile::NativePerf | RuntimeProfile::ArmCompact | RuntimeProfile::WasmPortable
        ));
    }

    #[test]
    fn inference_engine_report_includes_loaded_model_name() {
        let config = ModelConfig::from_json(QWEN3_JSON).expect("config should parse");
        let engine = InferenceEngine::from_config(config);
        let report = engine.capability_report();
        assert_eq!(report.loaded_model.as_deref(), Some("Qwen3-0.6B"));
    }
}
