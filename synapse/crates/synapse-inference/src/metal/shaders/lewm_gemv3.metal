// Skinny GEMV for M<=3: C[M,N] = A[M,K] * B^T[N,K]
// B is stored row-major as [N, K] (transposed layout).
// One thread per output column j, computes all M rows.
// Optimised for LEWM predict_next where seq_len=3.

#include <metal_stdlib>
using namespace metal;

kernel void gemv3_t(
    device const float* a [[buffer(0)]],    // [M, K] row-major
    device const float* b [[buffer(1)]],    // [N, K] row-major (transposed B)
    device float* c [[buffer(2)]],          // [M, N] output
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    for (uint i = 0; i < M; i++) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += a[i * K + k] * b[tid * K + k];
        }
        c[i * N + tid] = sum;
    }
}

// Fused LayerNorm + adaLN modulate for one row.
// Computes: out[j] = ((x[j] - mean) * inv_std * weight[j]) * (1 + scale[j]) + shift[j]
// Each thread handles one element across all rows. The per-row mean/variance
// computation is serial (hidden=192 is tiny).
kernel void layernorm_modulate(
    device const float* x [[buffer(0)]],       // [seq_len * hidden]
    device const float* weight [[buffer(1)]],  // [hidden]
    device const float* scale [[buffer(2)]],   // [hidden]
    device const float* shift [[buffer(3)]],   // [hidden]
    device float* out [[buffer(4)]],           // [seq_len * hidden]
    constant uint& hidden [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint row = tid / hidden;
    uint col = tid % hidden;
    if (row >= seq_len) return;

    // Compute mean and var for this row (serial — hidden is only 192)
    float mean = 0.0;
    for (uint j = 0; j < hidden; j++) {
        mean += x[row * hidden + j];
    }
    mean /= float(hidden);

    float var = 0.0;
    for (uint j = 0; j < hidden; j++) {
        float d = x[row * hidden + j] - mean;
        var += d * d;
    }
    var /= float(hidden);
    float inv_std = rsqrt(var + eps);

    float normed = (x[row * hidden + col] - mean) * inv_std * weight[col];
    out[row * hidden + col] = normed * (1.0 + scale[col]) + shift[col];
}

// GELU activation in-place
kernel void gelu_inplace(
    device float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float v = x[tid];
    x[tid] = 0.5 * v * (1.0 + tanh(sqrt(2.0 / M_PI_F) * (v + 0.044715 * v * v * v)));
}

// Gated residual: residual[i] += gate[i % hidden] * proj[i]
kernel void gated_residual(
    device float* residual [[buffer(0)]],
    device const float* proj [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total) return;
    residual[tid] += gate[tid % hidden] * proj[tid];
}

// Add bias: x[i] += bias[i % out_dim]
kernel void add_bias(
    device float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& out_dim [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total) return;
    x[tid] += bias[tid % out_dim];
}

// Bidirectional multi-head attention for seq_len=3.
// Each thread handles one head. Computes 3x3 attention scores, softmax, and V weighting.
// Q, K, V are [seq_len * num_heads * head_dim] in head-interleaved layout after QKV split.
// Actually Q/K/V are stored as [seq_len, inner_dim] = [seq_len, num_heads * head_dim].
// For head h at position t: offset = t * inner_dim + h * head_dim.
//
// Bidirectional: every position attends to every other position (no causal mask).
kernel void attention_3x3(
    device const float* q [[buffer(0)]],  // [seq_len * inner_dim]
    device const float* k [[buffer(1)]],  // [seq_len * inner_dim]
    device const float* v [[buffer(2)]],  // [seq_len * inner_dim]
    device float* out [[buffer(3)]],      // [seq_len * inner_dim]
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint tid [[thread_position_in_grid]])  // one thread per head
{
    if (tid >= num_heads) return;

    uint h = tid;
    uint inner_dim = num_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    // For each query position
    for (uint qi = 0; qi < seq_len; qi++) {
        // Compute attention scores for this query against all keys
        float scores[8]; // max seq_len (supports up to 8, we use 3)
        float max_score = -1e9;

        for (uint ki = 0; ki < seq_len; ki++) {
            float dot = 0.0;
            for (uint d = 0; d < head_dim; d++) {
                dot += q[qi * inner_dim + h * head_dim + d]
                     * k[ki * inner_dim + h * head_dim + d];
            }
            scores[ki] = dot * scale;
            max_score = max(max_score, scores[ki]);
        }

        // Softmax
        float sum_exp = 0.0;
        for (uint ki = 0; ki < seq_len; ki++) {
            scores[ki] = exp(scores[ki] - max_score);
            sum_exp += scores[ki];
        }
        float inv_sum = 1.0 / sum_exp;
        for (uint ki = 0; ki < seq_len; ki++) {
            scores[ki] *= inv_sum;
        }

        // Weighted sum of V
        for (uint d = 0; d < head_dim; d++) {
            float val = 0.0;
            for (uint ki = 0; ki < seq_len; ki++) {
                val += scores[ki] * v[ki * inner_dim + h * head_dim + d];
            }
            out[qi * inner_dim + h * head_dim + d] = val;
        }
    }
}
