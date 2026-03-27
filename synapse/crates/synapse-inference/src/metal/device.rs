use ::metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, Library};
use std::collections::HashMap;

/// Errors from Metal backend operations.
#[derive(Debug)]
pub enum MetalError {
    /// No Metal-compatible GPU found.
    NoDevice,
    /// Shader compilation failed.
    ShaderCompilation(String),
    /// Pipeline creation failed.
    PipelineCreation(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::NoDevice => write!(f, "No Metal-compatible GPU device found"),
            MetalError::ShaderCompilation(msg) => write!(f, "Shader compilation failed: {msg}"),
            MetalError::PipelineCreation(msg) => write!(f, "Pipeline creation failed: {msg}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Metal shader sources loaded from external .metal files and inline utilities.
const SHADER_SOURCE: &str = concat!(
    include_str!("shaders/matmul.metal"),
    "\n",
    include_str!("shaders/rmsnorm.metal"),
    "\n",
    include_str!("shaders/attention.metal"),
    "\n",
    include_str!("shaders/silu.metal"),
    "\n",
    include_str!("shaders/rope.metal"),
    "\n",
    include_str!("shaders/kv_scatter.metal"),
    "\n",
    include_str!("shaders/attention_decode.metal"),
    "\n",
    include_str!("shaders/headwise_rmsnorm.metal"),
    "\n",
    include_str!("shaders/gemv.metal"),
    "\n",
    include_str!("shaders/gemv_int8.metal"),
    "\n",
    // LEWM shaders (3 optimization levels):
    //   lewm_gemv3.metal        — individual kernel ops, used by v1 dispatch (~5ms)
    //   lewm_fused_layer.metal  — one-dispatch-per-layer, scalar dots (~1.07ms, fastest)
    //   lewm_fused_simd.metal   — one-dispatch-per-layer, float4 vectorized (~1.09ms)
    include_str!("shaders/lewm_gemv3.metal"),
    "\n",
    include_str!("shaders/lewm_fused_layer.metal"),
    "\n",
    include_str!("shaders/lewm_fused_simd.metal"),
    "\n",
    // Simple utility kernels kept inline
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    out[tid] = a[tid] * b[tid];
}

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    out[tid] = a[tid] + b[tid];
}

kernel void softmax(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float max_val = x[0];
    for (uint i = 1; i < n; i++) {
        max_val = max(max_val, x[i]);
    }
    float sum = 0.0;
    for (uint i = 0; i < n; i++) {
        sum += exp(x[i] - max_val);
    }
    out[tid] = exp(x[tid] - max_val) / sum;
}
"#,
);

/// Names of compiled compute kernels.
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "matmul",
    "rmsnorm",
    "silu",
    "swiglu",
    "attention",
    "elementwise_mul",
    "elementwise_add",
    "softmax",
    "rope_rotate_half",
    "kv_cache_scatter",
    "attention_decode",
    "headwise_rmsnorm",
    "gemv",
    "gemv_int8",
    "gemv3_t",
    "layernorm_modulate",
    "gelu_inplace",
    "gated_residual",
    "add_bias",
    "attention_3x3",
    "adaln_layer_fused",
    "adaln_layer_fused_simd",
];

/// Optional kernels that require specific Metal feature support.
/// These are registered if available but don't cause errors if missing.
pub(crate) const OPTIONAL_KERNEL_NAMES: &[&str] = &["matmul_simd"];

/// Apple Metal GPU backend for accelerated inference.
///
/// Wraps a Metal `Device`, `CommandQueue`, and pre-compiled `ComputePipelineState`
/// for each inference kernel (matmul, rmsnorm, silu, elementwise ops, softmax).
pub struct MetalBackend {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub pipelines: HashMap<String, ComputePipelineState>,
    _library: Library,
}

impl MetalBackend {
    /// Create a new Metal backend, detecting the GPU and compiling all shaders.
    ///
    /// Returns `Err(MetalError::NoDevice)` on non-Apple hardware.
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        let mut pipelines = HashMap::new();
        for &name in KERNEL_NAMES {
            let function = library
                .get_function(name, None)
                .map_err(|e| MetalError::ShaderCompilation(format!("'{name}': {e}")))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation(format!("'{name}': {e}")))?;
            pipelines.insert(name.to_string(), pipeline);
        }

        // Register optional kernels (e.g. simdgroup_matrix variants) if available
        for &name in OPTIONAL_KERNEL_NAMES {
            if let Ok(function) = library.get_function(name, None) {
                if let Ok(pipeline) = device.new_compute_pipeline_state_with_function(&function) {
                    pipelines.insert(name.to_string(), pipeline);
                }
            }
        }

        Ok(Self {
            device,
            command_queue,
            pipelines,
            _library: library,
        })
    }

    /// Check if a Metal-compatible GPU is available on this system.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Get the name of the GPU device.
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Get a compiled pipeline by kernel name.
    pub fn pipeline(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    /// Get the number of compiled pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }
}
