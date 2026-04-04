#include <metal_stdlib>
using namespace metal;

/// Single-token causal depthwise conv1d step for all channels.
///
/// Per-channel: shift rolling state left by 1, insert new input value,
/// compute dot product with kernel weights.
///
/// state:  [channels * kernel_size]  (rolling buffer, READ+WRITE — persists across steps)
/// x_in:   [channels]               (new input values for this timestep)
/// weight: [channels * kernel_size]  (conv kernel weights)
/// out:    [channels]               (conv output)
///
/// Dispatch: threads = channels, threadgroup_size = min(256, channels)
kernel void conv1d_step(
    device float* state [[buffer(0)]],
    device const float* x_in [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) return;

    uint base = tid * kernel_size;

    // Shift left: state[0..K-2] = state[1..K-1]
    for (uint k = 0; k < kernel_size - 1; k++) {
        state[base + k] = state[base + k + 1];
    }
    // Insert new value at end
    state[base + kernel_size - 1] = x_in[tid];

    // Dot product with kernel
    float sum = 0.0;
    for (uint k = 0; k < kernel_size; k++) {
        sum += state[base + k] * weight[base + k];
    }
    out[tid] = sum;
}
