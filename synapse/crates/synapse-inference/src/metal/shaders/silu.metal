#include <metal_stdlib>
using namespace metal;

/// Elementwise SiLU activation: out[i] = x[i] * sigmoid(x[i])
///
/// Dispatch: threadgroups = ceil(n/256), threads_per_threadgroup = 256
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

/// Fused SwiGLU: out[i] = silu(gate[i]) * up[i]
/// Combines SiLU activation on the gate projection with elementwise multiply
/// of the up projection in a single kernel to halve memory traffic.
///
/// Dispatch: threadgroups = ceil(n/256), threads_per_threadgroup = 256
kernel void swiglu(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float g = gate[tid];
    float silu_g = g / (1.0 + exp(-g));
    out[tid] = silu_g * up[tid];
}
