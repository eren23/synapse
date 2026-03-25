//! Shared attention computations used by both f32 and quantized inference paths.
//!
//! These functions operate on **pre-projected** Q, K, V tensors: the caller
//! handles the linear projections (f32 matmul_t vs QuantizedLinear::forward),
//! then delegates the core attention logic here.

use crate::config::position::RoPEStyle;
use crate::kv_cache::KVCacheLayer;
use crate::ops::activation::softmax_slice;
use crate::ops::matmul::{matmul_nn, matmul_t};
use crate::ops::norm::apply_headwise_rmsnorm;
use crate::ops::rope::apply_rope_inplace;

/// Minimum sequence length to use SIMD gather+matmul for attention.
/// Below this threshold, scalar dot products avoid gather overhead.
pub const ATTENTION_SIMD_THRESHOLD: usize = 16;

/// Compute single-token cached attention given pre-projected Q, K, V.
///
/// Handles: headwise norms -> RoPE -> cache append -> Q*K^T -> softmax -> score*V.
/// Returns attention output `[q_dim]` where `q_dim = num_heads * head_dim`.
pub fn cached_attention_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache_layer: &mut KVCacheLayer,
    pos: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    rope_style: RoPEStyle,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    eps: f32,
    window_size: Option<usize>,
) -> Vec<f32> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Apply headwise norms
    let mut q = apply_headwise_rmsnorm(q, q_norm_weight, 1, num_heads, head_dim, eps);
    let mut k = apply_headwise_rmsnorm(k, k_norm_weight, 1, num_kv_heads, head_dim, eps);

    // Apply RoPE at the correct position
    apply_rope_inplace(&mut q, rope_cos, rope_sin, 1, num_heads, head_dim, pos, rope_style);
    apply_rope_inplace(&mut k, rope_cos, rope_sin, 1, num_kv_heads, head_dim, pos, rope_style);

    // Append RoPE'd K and raw V to cache
    cache_layer
        .append(&k, v)
        .expect("KV cache append failed");

    // Get full cached K/V (all positions up to and including this one)
    let (cached_k, cached_v, seq_len) = cache_layer
        .slice()
        .expect("KV cache slice failed");

    // Sliding window: limit attention to the last `window_size` positions
    let (effective_k, effective_v, effective_len) = if let Some(ws) = window_size {
        if seq_len > ws {
            let offset = (seq_len - ws) * kv_dim;
            (&cached_k[offset..], &cached_v[offset..], ws)
        } else {
            (cached_k, cached_v, seq_len)
        }
    } else {
        (cached_k, cached_v, seq_len)
    };

    // Compute attention: single Q against effective cached K/V
    let mut attn_output = vec![0.0f32; q_dim];

    // For longer sequences, gather K/V per kv_head into contiguous buffers
    // and use SIMD matmul for Q*K^T. For short sequences, scalar is faster
    // (avoids gather overhead).
    if effective_len >= ATTENTION_SIMD_THRESHOLD {
        // SIMD path: gather + matmul_t
        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for kv_head in 0..num_kv_heads {
            let mut k_buf = vec![0.0f32; effective_len * head_dim];
            let mut v_buf = vec![0.0f32; effective_len * head_dim];
            for s in 0..effective_len {
                let off = s * kv_dim + kv_head * head_dim;
                k_buf[s * head_dim..(s + 1) * head_dim]
                    .copy_from_slice(&effective_k[off..off + head_dim]);
                v_buf[s * head_dim..(s + 1) * head_dim]
                    .copy_from_slice(&effective_v[off..off + head_dim]);
            }
            k_heads.push(k_buf);
            v_heads.push(v_buf);
        }

        for head in 0..num_heads {
            let kv_head = head / groups;
            let q_head = &q[head * head_dim..(head + 1) * head_dim];

            // Q*K^T via SIMD: [1, head_dim] x [effective_len, head_dim]^T = [1, effective_len]
            let mut scores = matmul_t(q_head, &k_heads[kv_head], 1, head_dim, effective_len);
            for s in &mut scores {
                *s *= scale;
            }
            softmax_slice(&mut scores);

            // score*V via SIMD: [1, effective_len] x [effective_len, head_dim] -> [1, head_dim]
            let sv = matmul_nn(&scores, &v_heads[kv_head], 1, effective_len, head_dim);
            attn_output[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&sv);
        }
    } else {
        // Scalar path for short sequences (avoids gather overhead)
        for head in 0..num_heads {
            let kv_head = head / groups;
            let mut scores = vec![0.0f32; effective_len];
            for s in 0..effective_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[head * head_dim + d]
                        * effective_k[s * kv_dim + kv_head * head_dim + d];
                }
                scores[s] = dot * scale;
            }
            softmax_slice(&mut scores);
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..effective_len {
                    sum += scores[s]
                        * effective_v[s * kv_dim + kv_head * head_dim + d];
                }
                attn_output[head * head_dim + d] = sum;
            }
        }
    }

    attn_output
}

/// Compute batched causal attention with cache populate.
///
/// Handles: headwise norms -> RoPE -> cache populate -> fused causal attention.
/// Returns attention output `[seq_len * q_dim]` where `q_dim = num_heads * head_dim`.
pub fn cached_attention_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache_layer: &mut KVCacheLayer,
    rope_cos: &[f32],
    rope_sin: &[f32],
    rope_style: RoPEStyle,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let groups = num_heads / num_kv_heads;

    let mut q = apply_headwise_rmsnorm(q, q_norm_weight, seq_len, num_heads, head_dim, eps);
    let mut k = apply_headwise_rmsnorm(k, k_norm_weight, seq_len, num_kv_heads, head_dim, eps);

    // Apply RoPE
    apply_rope_inplace(&mut q, rope_cos, rope_sin, seq_len, num_heads, head_dim, 0, rope_style);
    apply_rope_inplace(&mut k, rope_cos, rope_sin, seq_len, num_kv_heads, head_dim, 0, rope_style);

    // Populate KV cache with each position's RoPE'd K and raw V
    for t in 0..seq_len {
        let k_token = &k[t * kv_dim..(t + 1) * kv_dim];
        let v_token = &v[t * kv_dim..(t + 1) * kv_dim];
        cache_layer
            .append(k_token, v_token)
            .expect("KV cache append failed during prefill");
    }

    // Batched causal attention via fused SIMD kernel.
    // Gather per-head Q/K/V into contiguous buffers for the fused kernel.
    let mut attn_output = vec![0.0f32; seq_len * q_dim];

    for head in 0..num_heads {
        let kv_head = head / groups;

        // Gather Q for this head: [seq_len, head_dim]
        let mut q_head = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            let src = t * q_dim + head * head_dim;
            q_head[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&q[src..src + head_dim]);
        }

        // Gather K for this kv_head: [seq_len, head_dim]
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            let src = t * kv_dim + kv_head * head_dim;
            k_head[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&k[src..src + head_dim]);
        }

        // Gather V for this kv_head: [seq_len, head_dim]
        let mut v_head = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            let src = t * kv_dim + kv_head * head_dim;
            v_head[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&v[src..src + head_dim]);
        }

        // Fused causal attention: Q*K^T -> scale -> mask -> softmax -> *V
        let head_out = synapse_core::fused_attention(
            seq_len, seq_len, head_dim, &q_head, &k_head, &v_head,
        )
        .expect("fused attention failed");

        // Scatter output back to interleaved layout
        for t in 0..seq_len {
            let dst = t * q_dim + head * head_dim;
            attn_output[dst..dst + head_dim]
                .copy_from_slice(&head_out[t * head_dim..(t + 1) * head_dim]);
        }
    }

    attn_output
}
