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

/// Metal shader source for inference compute kernels.
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += a[gid.y * K + k] * b[k * N + gid.x];
    }
    c[gid.y * N + gid.x] = sum;
}

kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float sum_sq = 0.0;
    for (uint i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = rsqrt(sum_sq / float(n) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}

kernel void silu(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float val = x[tid];
    out[tid] = val / (1.0 + exp(-val));
}

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
"#;

/// Names of compiled compute kernels.
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "matmul",
    "rmsnorm",
    "silu",
    "elementwise_mul",
    "elementwise_add",
    "softmax",
];

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
