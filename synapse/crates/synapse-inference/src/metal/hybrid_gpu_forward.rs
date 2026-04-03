//! GPU-native forward pass for hybrid LIV Conv + GQA decoder layers.
//!
//! Encodes ALL layers into a single Metal command buffer with zero CPU-GPU
//! round trips. Each layer dispatches either the LIV Conv path or the GQA path
//! based on `layer_kinds[i]`.

use super::device::MetalBackend;
use super::hybrid_gpu_buffers::*;
use crate::models::ssm::hybrid::config::LayerKind;

/// Run ALL hybrid decoder layers on GPU in a single command buffer.
///
/// Encodes rmsnorm, gemv, swiglu, conv1d_step, rope, kv_scatter,
/// attention_decode, headwise_rmsnorm, and elementwise_add dispatches for
/// every layer into ONE command buffer. Single commit + waitUntilCompleted.
pub fn hybrid_forward_all_layers(
    bufs: &mut MetalHybridBuffers,
    hidden: &[f32],
    backend: &MetalBackend,
) -> Vec<f32> {
    let dev = &backend.device;
    let num_layers = bufs.layer_weights.len();
    let config_h = bufs.consts.h.length() as usize; // just for assertion
    let _ = config_h;
    let pos = bufs.pos;

    // 1. Write hidden state into scratch.x via shared memory
    unsafe {
        let ptr = bufs.scratch.x.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), ptr, hidden.len());
    }

    // 2. Update seq_len_buf to pos + 1 (sequence length after kv_scatter)
    unsafe {
        let ptr = bufs.consts.seq_len_buf.contents() as *mut u32;
        *ptr = (pos + 1) as u32;
    }

    // 3. Fetch all pipelines
    let gemv_pl = backend.pipeline("gemv").expect("gemv pipeline");
    let gemv_q4_pl = backend.pipeline("gemv_q4").expect("gemv_q4 pipeline");
    let rmsnorm_pl = backend.pipeline("rmsnorm").expect("rmsnorm pipeline");
    let add_pl = backend
        .pipeline("elementwise_add")
        .expect("elementwise_add pipeline");
    let swiglu_pl = backend.pipeline("swiglu").expect("swiglu pipeline");
    let rope_pl = backend
        .pipeline("rope_rotate_half")
        .expect("rope_rotate_half pipeline");
    let kv_scatter_pl = backend
        .pipeline("kv_cache_scatter")
        .expect("kv_cache_scatter pipeline");
    let attn_decode_pl = backend
        .pipeline("attention_decode")
        .expect("attention_decode pipeline");
    let headwise_rmsnorm_pl = backend
        .pipeline("headwise_rmsnorm")
        .expect("headwise_rmsnorm pipeline");
    let conv1d_step_pl = backend
        .pipeline("conv1d_step")
        .expect("conv1d_step pipeline");
    let silu_pl = backend.pipeline("silu").expect("silu pipeline");
    let mul_pl = backend
        .pipeline("elementwise_mul")
        .expect("elementwise_mul pipeline");

    // 4. Create ONE command buffer for all layers
    let cmd_buf = backend.command_queue.new_command_buffer();

    let c = &bufs.consts;

    for i in 0..num_layers {
        let scratch = &bufs.scratch;
        let kind = bufs.layer_kinds[i];

        match kind {
            LayerKind::Gqa => {
                let lw = match &bufs.layer_weights[i] {
                    MetalHybridLayerWeights::Gqa(g) => g,
                    _ => panic!("layer kind/weight mismatch at layer {i}"),
                };
                let kv_idx = bufs.kv_indices[i].expect("GQA layer must have kv_index");
                let kv = &bufs.kv_layers[kv_idx];

                encode_gqa_layer(
                    cmd_buf,
                    scratch,
                    lw,
                    kv,
                    c,
                    &bufs.rope_cos,
                    &bufs.rope_sin,
                    pos,
                    gemv_pl,
                    gemv_q4_pl,
                    rmsnorm_pl,
                    add_pl,
                    swiglu_pl,
                    rope_pl,
                    kv_scatter_pl,
                    attn_decode_pl,
                    headwise_rmsnorm_pl,
                    dev,
                );
            }
            LayerKind::LivConv => {
                let lw = match &bufs.layer_weights[i] {
                    MetalHybridLayerWeights::LivConv(l) => l,
                    _ => panic!("layer kind/weight mismatch at layer {i}"),
                };
                let conv_idx = bufs.conv_indices[i].expect("LivConv layer must have conv_index");
                let conv_state = &bufs.conv_states[conv_idx];

                encode_livconv_layer(
                    cmd_buf,
                    scratch,
                    lw,
                    conv_state,
                    c,
                    gemv_pl,
                    gemv_q4_pl,
                    rmsnorm_pl,
                    add_pl,
                    swiglu_pl,
                    conv1d_step_pl,
                    silu_pl,
                    mul_pl,
                    dev,
                );
            }
            LayerKind::DeltaNet => {
                panic!("DeltaNet layers not supported by Metal hybrid GPU path");
            }
        }
    }

    // 5. ONE commit + wait
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // 6. Update position
    bufs.pos += 1;

    // 7. Read scratch.x back to CPU
    read_buffer(&bufs.scratch.x, hidden.len())
}

// ── GQA layer encoder ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn encode_gqa_layer(
    cmd_buf: &::metal::CommandBufferRef,
    scratch: &HybridScratchBuffers,
    lw: &MetalGqaWeights,
    kv: &MetalHybridKVCacheLayer,
    c: &HybridConstantBuffers,
    rope_cos: &::metal::Buffer,
    rope_sin: &::metal::Buffer,
    pos: usize,
    gemv_pl: &::metal::ComputePipelineState,
    gemv_q4_pl: &::metal::ComputePipelineState,
    rmsnorm_pl: &::metal::ComputePipelineState,
    add_pl: &::metal::ComputePipelineState,
    swiglu_pl: &::metal::ComputePipelineState,
    rope_pl: &::metal::ComputePipelineState,
    kv_scatter_pl: &::metal::ComputePipelineState,
    attn_decode_pl: &::metal::ComputePipelineState,
    headwise_rmsnorm_pl: &::metal::ComputePipelineState,
    dev: &::metal::Device,
) {
    // Read constant values from the buffers for dispatch grid sizing.
    // These are u32 stored as f32 bit patterns.
    let q_dim = read_const_u32(&c.q_dim) as usize;
    let kv_dim = read_const_u32(&c.kv_dim) as usize;
    let h = read_const_u32(&c.h) as usize;
    let inter = read_const_u32(&c.inter) as usize;
    let num_heads = read_const_u32(&c.num_heads) as usize;
    let num_kv_heads = read_const_u32(&c.num_kv_heads) as usize;
    let head_dim = read_const_u32(&c.head_dim) as usize;
    let half_d = head_dim / 2;

    // ── Attention sub-layer ─────────────────────────────────────

    // enc1: norm_out = rmsnorm(x, attn_norm)
    encode_rmsnorm_fast(
        cmd_buf, rmsnorm_pl, &scratch.x, &lw.attn_norm, &scratch.norm_out,
        &c.h, &c.eps,
    );

    // enc2-4: Q/K/V = gemv(norm_out, W) — Q4 when available
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.wq, &lw.q4_wq, &scratch.q,
        q_dim, &c.q_dim, &c.h,
    );
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.wk, &lw.q4_wk, &scratch.k,
        kv_dim, &c.kv_dim, &c.h,
    );
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.wv, &lw.q4_wv, &scratch.v,
        kv_dim, &c.kv_dim, &c.h,
    );

    // enc5-6: headwise rmsnorm on Q/K
    encode_headwise_rmsnorm_fast(
        cmd_buf, headwise_rmsnorm_pl, &scratch.q, &lw.q_norm, &scratch.q,
        num_heads, head_dim, &c.num_heads, &c.head_dim, &c.eps, &c.head_dim,
    );
    encode_headwise_rmsnorm_fast(
        cmd_buf, headwise_rmsnorm_pl, &scratch.k, &lw.k_norm, &scratch.k,
        num_kv_heads, head_dim, &c.num_kv_heads, &c.head_dim, &c.eps, &c.head_dim,
    );

    // enc7-8: RoPE on Q and K
    encode_rope_fast(
        cmd_buf, rope_pl, &scratch.q, rope_cos, rope_sin,
        pos, num_heads, half_d, &c.num_heads, &c.head_dim,
    );
    encode_rope_fast(
        cmd_buf, rope_pl, &scratch.k, rope_cos, rope_sin,
        pos, num_kv_heads, half_d, &c.num_kv_heads, &c.head_dim,
    );

    // enc9: kv_scatter
    encode_kv_scatter(
        cmd_buf, kv_scatter_pl,
        &kv.k_cache, &kv.v_cache, &scratch.k, &scratch.v,
        pos, kv_dim, dev,
    );

    // enc10: attention_decode
    encode_attention_decode(
        cmd_buf, attn_decode_pl,
        &scratch.q, &kv.k_cache, &kv.v_cache, &scratch.attn_out,
        num_heads, num_kv_heads, head_dim,
        &c.seq_len_buf, kv_dim, dev,
    );

    // enc11: O = gemv(attn_out, wo)
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.attn_out, &lw.wo, &lw.q4_wo, &scratch.o,
        h, &c.h, &c.q_dim,
    );

    // enc12: residual = x + O
    encode_add_fast(
        cmd_buf, add_pl, &scratch.x, &scratch.o, &scratch.residual,
        h, &c.h,
    );

    // ── FFN sub-layer ───────────────────────────────────────────

    // enc13: norm_out = rmsnorm(residual, ffn_norm)
    encode_rmsnorm_fast(
        cmd_buf, rmsnorm_pl, &scratch.residual, &lw.ffn_norm, &scratch.norm_out,
        &c.h, &c.eps,
    );

    // enc14-15: gate/up projections — Q4 when available
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.ffn_gate, &lw.q4_ffn_gate, &scratch.gate_buf,
        inter, &c.inter, &c.h,
    );
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.ffn_up, &lw.q4_ffn_up, &scratch.up_buf,
        inter, &c.inter, &c.h,
    );

    // enc16: swiglu
    encode_swiglu_fast(
        cmd_buf, swiglu_pl, &scratch.gate_buf, &scratch.up_buf, &scratch.ffn_hidden,
        inter, &c.inter,
    );

    // enc17: down = gemv(ffn_hidden, w_down) — Q4 when available
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.ffn_hidden, &lw.ffn_down, &lw.q4_ffn_down, &scratch.down_buf,
        h, &c.h, &c.inter,
    );

    // enc18: x = residual + down
    encode_add_fast(
        cmd_buf, add_pl, &scratch.residual, &scratch.down_buf, &scratch.x,
        h, &c.h,
    );
}

// ── LIV Conv layer encoder ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn encode_livconv_layer(
    cmd_buf: &::metal::CommandBufferRef,
    scratch: &HybridScratchBuffers,
    lw: &MetalLivConvWeights,
    conv_state: &::metal::Buffer,
    c: &HybridConstantBuffers,
    gemv_pl: &::metal::ComputePipelineState,
    gemv_q4_pl: &::metal::ComputePipelineState,
    rmsnorm_pl: &::metal::ComputePipelineState,
    add_pl: &::metal::ComputePipelineState,
    swiglu_pl: &::metal::ComputePipelineState,
    conv1d_step_pl: &::metal::ComputePipelineState,
    silu_pl: &::metal::ComputePipelineState,
    mul_pl: &::metal::ComputePipelineState,
    _dev: &::metal::Device,
) {
    let h = read_const_u32(&c.h) as usize;
    let inner = read_const_u32(&c.inner) as usize;
    let inter = read_const_u32(&c.inter) as usize;
    let proj_dim = read_const_u32(&c.proj_dim) as usize;

    // ── Conv sub-layer ──────────────────────────────────────────

    // enc1: norm_out = rmsnorm(x, attn_norm)
    encode_rmsnorm_fast(
        cmd_buf, rmsnorm_pl, &scratch.x, &lw.attn_norm, &scratch.norm_out,
        &c.h, &c.eps,
    );

    // enc2: proj_out = gemv(norm_out, in_proj)  → [proj_dim] (= 3*inner)
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.in_proj, &lw.q4_in_proj, &scratch.proj_out,
        proj_dim, &c.proj_dim, &c.h,
    );

    // The in_proj outputs [3*inner] into proj_out:
    //   x_conv  = proj_out[0..inner]           (byte offset 0)
    //   gate1   = proj_out[inner..2*inner]     (byte offset inner*4)
    //   gate2   = proj_out[2*inner..3*inner]   (byte offset 2*inner*4)
    //
    // We use Metal buffer offsets to reference sub-regions.

    let inner_bytes = (inner * std::mem::size_of::<f32>()) as u64;

    // enc3: gate1_silu = silu(gate1) → into conv_out (temporary)
    // We use silu on the gate1 region of proj_out, writing to conv_out.
    encode_silu_offset(
        cmd_buf, silu_pl, &scratch.proj_out, inner_bytes,
        &scratch.conv_out, 0,
        inner, &c.inner,
    );

    // enc4: gated_in = gate1_silu * x_conv → into conv_out (in-place reuse)
    // Multiply conv_out (= silu(gate1)) with proj_out[0..inner] (= x_conv)
    encode_mul_offset(
        cmd_buf, mul_pl,
        &scratch.conv_out, 0,       // silu(gate1)
        &scratch.proj_out, 0,       // x_conv
        &scratch.conv_out, 0,       // output: gated_in → reuse conv_out
        inner, &c.inner,
    );

    // enc5: conv1d_step: state update + dot product
    // Input: conv_out (gated_in), output: conv_proj (temporary, reuse as [inner])
    // We reuse conv_proj as temp [inner] buffer (it's [hidden] but inner <= hidden).
    // Actually, let's use a dedicated buffer path: conv_out is gated input,
    // we need a separate output. Use the first `inner` elements of conv_proj.
    encode_conv1d_step(
        cmd_buf, conv1d_step_pl,
        conv_state,        // [inner * kernel_size] rolling state
        &scratch.conv_out, // x_in: gated input [inner]
        &lw.conv_weight,   // weight: [inner * kernel_size]
        &scratch.norm_out, // output: reuse norm_out as temp [inner] (it's [hidden] >= inner)
        &c.channels,
        &c.kernel_size,
        inner,
    );

    // enc6: post_silu = silu(conv_output) → into conv_out
    encode_silu_offset(
        cmd_buf, silu_pl, &scratch.norm_out, 0,
        &scratch.conv_out, 0,
        inner, &c.inner,
    );

    // enc7: post_gate = post_silu * gate2 → into conv_out
    // gate2 is proj_out[2*inner..3*inner]
    encode_mul_offset(
        cmd_buf, mul_pl,
        &scratch.conv_out, 0,           // silu(conv_output)
        &scratch.proj_out, 2 * inner_bytes, // gate2
        &scratch.conv_out, 0,           // output: post_gate
        inner, &c.inner,
    );

    // enc8: conv_proj = gemv(post_gate, out_proj) → [hidden]
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.conv_out, &lw.out_proj, &lw.q4_out_proj, &scratch.conv_proj,
        h, &c.h, &c.inner,
    );

    // enc9: residual = x + conv_proj
    encode_add_fast(
        cmd_buf, add_pl, &scratch.x, &scratch.conv_proj, &scratch.residual,
        h, &c.h,
    );

    // ── FFN sub-layer ───────────────────────────────────────────

    // enc10: norm_out = rmsnorm(residual, ffn_norm)
    encode_rmsnorm_fast(
        cmd_buf, rmsnorm_pl, &scratch.residual, &lw.ffn_norm, &scratch.norm_out,
        &c.h, &c.eps,
    );

    // enc11-12: gate/up projections — Q4 when available
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.ffn_gate, &lw.q4_ffn_gate, &scratch.gate_buf,
        inter, &c.inter, &c.h,
    );
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.norm_out, &lw.ffn_up, &lw.q4_ffn_up, &scratch.up_buf,
        inter, &c.inter, &c.h,
    );

    // enc13: swiglu
    encode_swiglu_fast(
        cmd_buf, swiglu_pl, &scratch.gate_buf, &scratch.up_buf, &scratch.ffn_hidden,
        inter, &c.inter,
    );

    // enc14: down = gemv(ffn_hidden, w_down) — Q4 when available
    encode_gemv_q4_or_f32(
        cmd_buf, gemv_pl, gemv_q4_pl, &scratch.ffn_hidden, &lw.ffn_down, &lw.q4_ffn_down, &scratch.down_buf,
        h, &c.h, &c.inter,
    );

    // enc15: x = residual + down
    encode_add_fast(
        cmd_buf, add_pl, &scratch.residual, &scratch.down_buf, &scratch.x,
        h, &c.h,
    );
}

// ── Encoder helpers (matching gpu_forward.rs buffer slot assignments) ─

/// GEMV: out[N] = A[K] @ B[K, N] — pre-allocated constant buffers.
fn encode_gemv_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b: &::metal::Buffer,
    buf_c: &::metal::Buffer,
    n: usize,
    buf_n: &::metal::Buffer,
    buf_k: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_c), 0);
    encoder.set_buffer(3, Some(buf_n), 0);
    encoder.set_buffer(4, Some(buf_k), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// GEMV dispatch: use Q4 kernel when raw Q4 buffer is available, else f32.
/// The gemv_q4 shader has the same buffer slot layout (a=0, b=1, c=2, N=3, K=4)
/// but b is raw Q4_0 block bytes (NOT transposed) instead of f32 [K,N].
fn encode_gemv_q4_or_f32(
    cmd_buf: &::metal::CommandBufferRef,
    gemv_pl: &::metal::ComputePipelineState,
    gemv_q4_pl: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b_f32: &::metal::Buffer,
    buf_b_q4: &Option<::metal::Buffer>,
    buf_c: &::metal::Buffer,
    n: usize,
    buf_n: &::metal::Buffer,
    buf_k: &::metal::Buffer,
) {
    if let Some(q4_buf) = buf_b_q4 {
        // Use Q4 GEMV: 4x less memory bandwidth
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(gemv_q4_pl);
        encoder.set_buffer(0, Some(buf_a), 0);
        encoder.set_buffer(1, Some(q4_buf), 0);
        encoder.set_buffer(2, Some(buf_c), 0);
        encoder.set_buffer(3, Some(buf_n), 0);
        encoder.set_buffer(4, Some(buf_k), 0);
        let grid = ::metal::MTLSize::new(n as u64, 1, 1);
        let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    } else {
        encode_gemv_fast(cmd_buf, gemv_pl, buf_a, buf_b_f32, buf_c, n, buf_n, buf_k);
    }
}

/// RMSNorm: out = rmsnorm(x, weight, eps) — pre-allocated constants.
fn encode_rmsnorm_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &::metal::Buffer,
    buf_w: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    buf_n: &::metal::Buffer,
    buf_eps: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_w), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(buf_n), 0);
    encoder.set_buffer(4, Some(buf_eps), 0);
    let threadgroups = ::metal::MTLSize::new(1, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

/// SwiGLU: out[i] = silu(gate[i]) * up[i] — pre-allocated constants.
fn encode_swiglu_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_gate: &::metal::Buffer,
    buf_up: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    n: usize,
    buf_n: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_gate), 0);
    encoder.set_buffer(1, Some(buf_up), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Elementwise add: out = a + b — pre-allocated constant.
fn encode_add_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    n: usize,
    buf_n: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// RoPE rotation (rotate-half): in-place rotation of Q or K.
fn encode_rope_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_qk: &::metal::Buffer,
    buf_cos: &::metal::Buffer,
    buf_sin: &::metal::Buffer,
    pos: usize,
    num_heads: usize,
    half_d: usize,
    buf_num_heads: &::metal::Buffer,
    buf_head_dim: &::metal::Buffer,
) {
    let cos_offset = (pos * half_d * std::mem::size_of::<f32>()) as u64;
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_qk), 0);
    encoder.set_buffer(1, Some(buf_cos), cos_offset);
    encoder.set_buffer(2, Some(buf_sin), cos_offset);
    encoder.set_buffer(3, Some(buf_num_heads), 0);
    encoder.set_buffer(4, Some(buf_head_dim), 0);
    let total_pairs = num_heads * half_d;
    let grid = ::metal::MTLSize::new(total_pairs as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(total_pairs as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// KV scatter: write one token's K/V into the cache at position `pos`.
fn encode_kv_scatter(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    k_cache: &::metal::Buffer,
    v_cache: &::metal::Buffer,
    k_token: &::metal::Buffer,
    v_token: &::metal::Buffer,
    pos: usize,
    kv_dim: usize,
    dev: &::metal::Device,
) {
    let buf_pos = make_const_u32(dev, pos as u32);
    let buf_kv_dim = make_const_u32(dev, kv_dim as u32);

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(k_cache), 0);
    encoder.set_buffer(1, Some(v_cache), 0);
    encoder.set_buffer(2, Some(k_token), 0);
    encoder.set_buffer(3, Some(v_token), 0);
    encoder.set_buffer(4, Some(&buf_pos), 0);
    encoder.set_buffer(5, Some(&buf_kv_dim), 0);

    let grid = ::metal::MTLSize::new(kv_dim as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(kv_dim as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Attention decode: one Q-token against all cached K/V.
#[allow(clippy::too_many_arguments)]
fn encode_attention_decode(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_q: &::metal::Buffer,
    k_cache: &::metal::Buffer,
    v_cache: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_buf: &::metal::Buffer,
    kv_dim: usize,
    dev: &::metal::Device,
) {
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_num_kv_heads = make_const_u32(dev, num_kv_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let buf_kv_dim = make_const_u32(dev, kv_dim as u32);

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_q), 0);
    encoder.set_buffer(1, Some(k_cache), 0);
    encoder.set_buffer(2, Some(v_cache), 0);
    encoder.set_buffer(3, Some(buf_out), 0);
    encoder.set_buffer(4, Some(&buf_num_heads), 0);
    encoder.set_buffer(5, Some(&buf_num_kv_heads), 0);
    encoder.set_buffer(6, Some(&buf_head_dim), 0);
    encoder.set_buffer(7, Some(seq_len_buf), 0);
    encoder.set_buffer(8, Some(&buf_kv_dim), 0);

    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

/// Per-head RMSNorm (Qwen3 style).
#[allow(clippy::too_many_arguments)]
fn encode_headwise_rmsnorm_fast(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &::metal::Buffer,
    buf_w: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    num_heads: usize,
    _head_dim: usize,
    buf_num_heads: &::metal::Buffer,
    buf_head_dim: &::metal::Buffer,
    buf_eps: &::metal::Buffer,
    buf_hdw: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_w), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(buf_num_heads), 0);
    encoder.set_buffer(4, Some(buf_head_dim), 0);
    encoder.set_buffer(5, Some(buf_eps), 0);
    encoder.set_buffer(6, Some(buf_hdw), 0);
    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

/// Conv1d step: shift state, insert new value, dot product with kernel.
fn encode_conv1d_step(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    state: &::metal::Buffer,
    x_in: &::metal::Buffer,
    weight: &::metal::Buffer,
    out: &::metal::Buffer,
    channels_buf: &::metal::Buffer,
    kernel_size_buf: &::metal::Buffer,
    channels: usize,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(state), 0);
    encoder.set_buffer(1, Some(x_in), 0);
    encoder.set_buffer(2, Some(weight), 0);
    encoder.set_buffer(3, Some(out), 0);
    encoder.set_buffer(4, Some(channels_buf), 0);
    encoder.set_buffer(5, Some(kernel_size_buf), 0);

    let grid = ::metal::MTLSize::new(channels as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(channels as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// SiLU with byte offsets into input/output buffers.
fn encode_silu_offset(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &::metal::Buffer,
    x_offset: u64,
    buf_out: &::metal::Buffer,
    out_offset: u64,
    n: usize,
    buf_n: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), x_offset);
    encoder.set_buffer(1, Some(buf_out), out_offset);
    encoder.set_buffer(2, Some(buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Elementwise multiply with byte offsets: out = a * b.
#[allow(clippy::too_many_arguments)]
fn encode_mul_offset(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    a_offset: u64,
    buf_b: &::metal::Buffer,
    b_offset: u64,
    buf_out: &::metal::Buffer,
    out_offset: u64,
    n: usize,
    buf_n: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), a_offset);
    encoder.set_buffer(1, Some(buf_b), b_offset);
    encoder.set_buffer(2, Some(buf_out), out_offset);
    encoder.set_buffer(3, Some(buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

// ── Utility ─────────────────────────────────────────────────────────

fn make_const_u32(device: &::metal::Device, val: u32) -> ::metal::Buffer {
    let data = [f32::from_bits(val)];
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        4,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

fn read_buffer(buf: &::metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let mut out = vec![0.0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
    out
}

/// Read a u32 value from a constant buffer (stored as f32 bit pattern).
fn read_const_u32(buf: &::metal::Buffer) -> u32 {
    let ptr = buf.contents() as *const u32;
    unsafe { *ptr }
}
