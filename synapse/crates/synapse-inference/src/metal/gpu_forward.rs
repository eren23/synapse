//! GPU-native forward pass for decoder layers.
//!
//! Phase 1: Keeps Q/K/V projection in one command buffer (CMD1), and chains
//! the entire FFN sub-layer (O proj → residual → norm → gate/up → swiglu → down → residual)
//! into a second command buffer (CMD2). This halves commit+wait from 4 to 2 per layer.
//!
//! Attention stays on CPU (RoPE + KV cache + dot products are sequential and tiny).

use super::buffer::BufferPool;
use super::device::MetalBackend;
use crate::kv_cache::KVCacheLayer;
use crate::model::decoder_layer::DecoderLayer;
use crate::ops::attention::cached_attention_decode;
use crate::ops::norm::apply_norm;

/// Run one decoder layer with GPU-accelerated matmuls.
///
/// CMD1: Q/K/V projections (1 commit+wait)
/// CPU: bias, headwise norm, RoPE, cache, attention
/// CMD2: O proj → residual → rmsnorm → gate → up → swiglu → down → residual (1 commit+wait)
///
/// Total: 2 commit+wait per layer (was 4).
pub fn gpu_forward_one(
    layer: &DecoderLayer,
    hidden: &[f32],
    cache_layer: &mut KVCacheLayer,
    pos: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let h = layer.hidden_size;
    let num_heads = layer.attention.num_heads();
    let num_kv_heads = layer.attention.num_kv_heads();
    let head_dim = layer.attention.head_dim();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let inter = layer.ffn.intermediate_size();

    // ── 1. Attention sub-layer ─────────────────────────────────────
    // RMSNorm on CPU (cheap, [1, h])
    let normed = apply_norm(hidden, &layer.attn_norm_weight, &*layer.attn_norm, h);

    // CMD1: Q/K/V projections batched in one command buffer
    let (q, k, v) = gpu_batch_qkv(
        &normed,
        &layer.w_q,
        &layer.w_k,
        &layer.w_v,
        &layer.q_bias,
        &layer.k_bias,
        &layer.v_bias,
        h,
        q_dim,
        kv_dim,
        backend,
        pool,
    );

    // Attention on CPU (RoPE + cache + dot products)
    let attn_out = cached_attention_decode(
        &q,
        &k,
        &v,
        num_heads,
        num_kv_heads,
        head_dim,
        cache_layer,
        pos,
        rope_cos,
        rope_sin,
        layer.rope_style,
        &layer.q_norm_weight,
        &layer.k_norm_weight,
        layer.attn_norm.eps() as f32,
        layer.attention.window_size(),
    );

    // ── 2. FFN sub-layer — ALL on GPU in ONE command buffer ────────
    // CMD2: O proj → residual_add → rmsnorm → gate → up → swiglu → down → residual_add
    gpu_ffn_chained(hidden, &attn_out, layer, h, q_dim, inter, backend, pool)
}

/// Batch Q/K/V projections into one GPU command buffer.
fn gpu_batch_qkv(
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    q_bias: &[f32],
    k_bias: &[f32],
    v_bias: &[f32],
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dev = &backend.device;

    pool.get_or_create_transposed_weight(w_q, q_dim, h);
    pool.get_or_create_transposed_weight(w_k, kv_dim, h);
    pool.get_or_create_transposed_weight(w_v, kv_dim, h);

    let wq_ptr = w_q.as_ptr() as usize;
    let wk_ptr = w_k.as_ptr() as usize;
    let wv_ptr = w_v.as_ptr() as usize;

    let buf_x = pool.get_or_create(x);
    let buf_q = pool.create_empty(q_dim);
    let buf_k = pool.create_empty(kv_dim);
    let buf_v = pool.create_empty(kv_dim);

    let pipeline = backend.pipeline("matmul").expect("matmul pipeline");
    let cmd_buf = backend.command_queue.new_command_buffer();

    for (buf_out, w_ptr, n) in [
        (&buf_q, wq_ptr, q_dim),
        (&buf_k, wk_ptr, kv_dim),
        (&buf_v, wv_ptr, kv_dim),
    ] {
        encode_matmul(
            cmd_buf,
            pipeline,
            &buf_x,
            pool.get_cached_weight(w_ptr).unwrap(),
            buf_out,
            1,
            n,
            h,
            dev,
        );
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut q = read_buffer(&buf_q, q_dim);
    let mut k = read_buffer(&buf_k, kv_dim);
    let mut v = read_buffer(&buf_v, kv_dim);

    add_bias_inplace(&mut q, q_bias);
    add_bias_inplace(&mut k, k_bias);
    add_bias_inplace(&mut v, v_bias);

    pool.release(buf_x);
    pool.release(buf_q);
    pool.release(buf_k);
    pool.release(buf_v);

    (q, k, v)
}

/// Chained FFN: O proj → residual → rmsnorm → gate/up → swiglu → down → residual
/// ALL in ONE command buffer (one commit+wait).
fn gpu_ffn_chained(
    x_hidden: &[f32], // layer input [h] (for residual)
    attn_out: &[f32], // attention output [q_dim]
    layer: &DecoderLayer,
    h: usize,
    q_dim: usize,
    inter: usize,
    backend: &MetalBackend,
    pool: &mut BufferPool,
) -> Vec<f32> {
    let dev = &backend.device;

    // Ensure all weights cached
    pool.get_or_create_transposed_weight(&layer.w_o, h, q_dim);
    if !layer.ffn_gate.is_empty() {
        pool.get_or_create_transposed_weight(&layer.ffn_gate, inter, h);
    }
    pool.get_or_create_transposed_weight(&layer.ffn_up, inter, h);
    pool.get_or_create_transposed_weight(&layer.ffn_down, h, inter);

    let wo_ptr = layer.w_o.as_ptr() as usize;
    let gate_ptr = layer.ffn_gate.as_ptr() as usize;
    let up_ptr = layer.ffn_up.as_ptr() as usize;
    let down_ptr = layer.ffn_down.as_ptr() as usize;

    // Scratch buffers (allocated from pool, reused across layers via pool)
    let buf_attn = pool.get_or_create(attn_out);
    let buf_x = pool.get_or_create(x_hidden);
    let buf_o = pool.create_empty(h);
    let buf_residual = pool.create_empty(h);
    let buf_norm_r = pool.create_empty(h);
    let buf_gate = pool.create_empty(inter);
    let buf_up = pool.create_empty(inter);
    let buf_hidden = pool.create_empty(inter);
    let buf_down = pool.create_empty(h);
    let buf_x_out = pool.create_empty(h);

    let matmul_pl = backend.pipeline("matmul").expect("matmul pipeline");
    let add_pl = backend
        .pipeline("elementwise_add")
        .expect("elementwise_add pipeline");
    let rmsnorm_pl = backend.pipeline("rmsnorm").expect("rmsnorm pipeline");
    let swiglu_pl = backend.pipeline("swiglu").expect("swiglu pipeline");

    let cmd_buf = backend.command_queue.new_command_buffer();

    // enc1: O = matmul(attn_out, w_o)  [q_dim → h]
    encode_matmul(
        cmd_buf,
        matmul_pl,
        &buf_attn,
        pool.get_cached_weight(wo_ptr).unwrap(),
        &buf_o,
        1,
        h,
        q_dim,
        dev,
    );

    // enc2: residual = x + O
    encode_add(cmd_buf, add_pl, &buf_x, &buf_o, &buf_residual, h, dev);

    // enc3: norm_r = rmsnorm(residual, ffn_norm_weight)
    let buf_ffn_norm_w = pool.get_or_create(&layer.ffn_norm_weight);
    encode_rmsnorm(
        cmd_buf,
        rmsnorm_pl,
        &buf_residual,
        &buf_ffn_norm_w,
        &buf_norm_r,
        h,
        layer.ffn_norm.eps() as f32,
        dev,
    );

    // enc4+5: gate = matmul(norm_r, w_gate), up = matmul(norm_r, w_up)
    if !layer.ffn_gate.is_empty() {
        encode_matmul(
            cmd_buf,
            matmul_pl,
            &buf_norm_r,
            pool.get_cached_weight(gate_ptr).unwrap(),
            &buf_gate,
            1,
            inter,
            h,
            dev,
        );
    }
    encode_matmul(
        cmd_buf,
        matmul_pl,
        &buf_norm_r,
        pool.get_cached_weight(up_ptr).unwrap(),
        &buf_up,
        1,
        inter,
        h,
        dev,
    );

    // enc6: hidden = swiglu(gate, up)
    encode_swiglu(
        cmd_buf,
        swiglu_pl,
        &buf_gate,
        &buf_up,
        &buf_hidden,
        inter,
        dev,
    );

    // enc7: down = matmul(hidden, w_down)
    encode_matmul(
        cmd_buf,
        matmul_pl,
        &buf_hidden,
        pool.get_cached_weight(down_ptr).unwrap(),
        &buf_down,
        1,
        h,
        inter,
        dev,
    );

    // enc8: x_out = residual + down
    encode_add(
        cmd_buf,
        add_pl,
        &buf_residual,
        &buf_down,
        &buf_x_out,
        h,
        dev,
    );

    // ONE commit + wait for entire FFN sub-layer
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let result = read_buffer(&buf_x_out, h);

    pool.release(buf_attn);
    pool.release(buf_x);
    pool.release(buf_o);
    pool.release(buf_residual);
    pool.release(buf_norm_r);
    pool.release(buf_ffn_norm_w);
    pool.release(buf_gate);
    pool.release(buf_up);
    pool.release(buf_hidden);
    pool.release(buf_down);
    pool.release(buf_x_out);

    result
}

// ── Encoder helpers ─────────────────────────────────────────────────

fn encode_matmul(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b: &::metal::Buffer,
    buf_c: &::metal::Buffer,
    m: usize,
    n: usize,
    k: usize,
    dev: &::metal::Device,
) {
    let buf_m = make_const_u32(dev, m as u32);
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_c), 0);
    encoder.set_buffer(3, Some(&buf_m), 0);
    encoder.set_buffer(4, Some(&buf_n), 0);
    encoder.set_buffer(5, Some(&buf_k), 0);

    let grid = ::metal::MTLSize::new(
        ((n as u32 + 31) / 32 * 32) as u64,
        ((m as u32 + 31) / 32 * 32) as u64,
        1,
    );
    let tg = ::metal::MTLSize::new(32, 32, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

fn encode_add(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    n: usize,
    dev: &::metal::Device,
) {
    let buf_n = make_const_u32(dev, n as u32);
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

fn encode_rmsnorm(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &::metal::Buffer,
    buf_w: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    hidden_size: usize,
    eps: f32,
    dev: &::metal::Device,
) {
    let buf_n = make_const_u32(dev, hidden_size as u32);
    let buf_eps = make_const_f32(dev, eps);
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_w), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    encoder.set_buffer(4, Some(&buf_eps), 0);
    // One threadgroup per row, 256 threads
    let threadgroups = ::metal::MTLSize::new(1, 1, 1); // batch=1 for M=1
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

fn encode_swiglu(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_gate: &::metal::Buffer,
    buf_up: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    n: usize,
    dev: &::metal::Device,
) {
    let buf_n = make_const_u32(dev, n as u32);
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_gate), 0);
    encoder.set_buffer(1, Some(buf_up), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(&buf_n), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

// ── Phase 3: GPU-resident all-layers-in-one-command-buffer ──────────

use super::gpu_buffers::MetalModelBuffers;

/// Run ALL decoder layers on GPU in a single command buffer.
///
/// Encodes rmsnorm, gemv, rope, kv_scatter, attention_decode, headwise_rmsnorm,
/// swiglu, and elementwise_add dispatches for every layer into ONE command buffer.
/// Single commit + waitUntilCompleted at the end.
pub fn gpu_forward_all_layers(
    model_bufs: &mut MetalModelBuffers,
    hidden: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    h: usize,
    inter: usize,
    has_head_norms: bool,
    eps: f32,
    backend: &MetalBackend,
) -> Vec<f32> {
    let dev = &backend.device;
    let num_layers = model_bufs.layer_weights.len();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let half_d = head_dim / 2;
    let pos = model_bufs.kv_cache.pos;

    // 1. Write hidden state into scratch.x via shared memory pointer
    unsafe {
        let ptr = model_bufs.scratch.x.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), ptr, h);
    }

    // 2. Update seq_len_buf to pos + 1 (sequence length after kv_scatter)
    unsafe {
        let ptr = model_bufs.seq_len_buf.contents() as *mut u32;
        *ptr = (pos + 1) as u32;
    }

    // 3. Fetch all pipelines
    let gemv_pl = backend.pipeline("gemv").expect("gemv pipeline");
    let gemv_int8_pl = backend.pipeline("gemv_int8").expect("gemv_int8 pipeline");
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

    // 4. Create ONE command buffer for all layers
    let cmd_buf = backend.command_queue.new_command_buffer();

    let c = &model_bufs.consts;

    for i in 0..num_layers {
        let lw = &model_bufs.layer_weights[i];
        let kv = &model_bufs.kv_cache.layers[i];
        let scratch = &model_bufs.scratch;

        // ── Attention sub-layer ─────────────────────────────────────

        // enc1: norm_x = rmsnorm(x, attn_norm_weight)
        encode_rmsnorm_fast(
            cmd_buf,
            rmsnorm_pl,
            &scratch.x,
            &lw.attn_norm,
            &scratch.norm_x,
            &c.h,
            &c.eps,
        );

        // enc2-4: Q/K/V = gemv(norm_x, W) — use INT8 if available
        if lw.has_int8 {
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.norm_x,
                lw.wq_int8.as_ref().unwrap(),
                lw.wq_scale.as_ref().unwrap(),
                &scratch.q,
                q_dim,
                &c.q_dim,
                &c.h,
            );
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.norm_x,
                lw.wk_int8.as_ref().unwrap(),
                lw.wk_scale.as_ref().unwrap(),
                &scratch.k,
                kv_dim,
                &c.kv_dim,
                &c.h,
            );
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.norm_x,
                lw.wv_int8.as_ref().unwrap(),
                lw.wv_scale.as_ref().unwrap(),
                &scratch.v,
                kv_dim,
                &c.kv_dim,
                &c.h,
            );
        } else {
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.norm_x,
                &lw.wq,
                &scratch.q,
                q_dim,
                &c.q_dim,
                &c.h,
            );
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.norm_x,
                &lw.wk,
                &scratch.k,
                kv_dim,
                &c.kv_dim,
                &c.h,
            );
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.norm_x,
                &lw.wv,
                &scratch.v,
                kv_dim,
                &c.kv_dim,
                &c.h,
            );
        }

        // enc5-7: bias add (in-place)
        if let Some(ref bias) = lw.q_bias {
            encode_add_fast(
                cmd_buf, add_pl, &scratch.q, bias, &scratch.q, q_dim, &c.q_dim,
            );
        }
        if let Some(ref bias) = lw.k_bias {
            encode_add_fast(
                cmd_buf, add_pl, &scratch.k, bias, &scratch.k, kv_dim, &c.kv_dim,
            );
        }
        if let Some(ref bias) = lw.v_bias {
            encode_add_fast(
                cmd_buf, add_pl, &scratch.v, bias, &scratch.v, kv_dim, &c.kv_dim,
            );
        }

        // enc8-9: headwise rmsnorm on Q/K
        if has_head_norms {
            if let Some(ref qn) = lw.q_norm {
                encode_headwise_rmsnorm_fast(
                    cmd_buf,
                    headwise_rmsnorm_pl,
                    &scratch.q,
                    qn,
                    &scratch.q,
                    num_heads,
                    head_dim,
                    &c.num_heads,
                    &c.head_dim,
                    &c.eps,
                    &c.head_dim,
                );
            }
            if let Some(ref kn) = lw.k_norm {
                encode_headwise_rmsnorm_fast(
                    cmd_buf,
                    headwise_rmsnorm_pl,
                    &scratch.k,
                    kn,
                    &scratch.k,
                    num_kv_heads,
                    head_dim,
                    &c.num_kv_heads,
                    &c.head_dim,
                    &c.eps,
                    &c.head_dim,
                );
            }
        }

        // enc10-11: RoPE on Q and K
        encode_rope_fast(
            cmd_buf,
            rope_pl,
            &scratch.q,
            &model_bufs.rope_cos,
            &model_bufs.rope_sin,
            pos,
            num_heads,
            half_d,
            &c.num_heads,
            &c.head_dim,
        );
        encode_rope_fast(
            cmd_buf,
            rope_pl,
            &scratch.k,
            &model_bufs.rope_cos,
            &model_bufs.rope_sin,
            pos,
            num_kv_heads,
            half_d,
            &c.num_kv_heads,
            &c.head_dim,
        );

        // enc12: kv_scatter
        encode_kv_scatter(
            cmd_buf,
            kv_scatter_pl,
            &kv.k_cache,
            &kv.v_cache,
            &scratch.k,
            &scratch.v,
            pos,
            kv_dim,
            dev,
        );

        // enc13: attention_decode
        encode_attention_decode(
            cmd_buf,
            attn_decode_pl,
            &scratch.q,
            &kv.k_cache,
            &kv.v_cache,
            &scratch.attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            &model_bufs.seq_len_buf,
            kv_dim,
            dev,
        );

        // enc14: O = gemv(attn_out, wo)
        if lw.has_int8 {
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.attn_out,
                lw.wo_int8.as_ref().unwrap(),
                lw.wo_scale.as_ref().unwrap(),
                &scratch.o,
                h,
                &c.h,
                &c.q_dim,
            );
        } else {
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.attn_out,
                &lw.wo,
                &scratch.o,
                h,
                &c.h,
                &c.q_dim,
            );
        }

        // enc15: residual = x + O
        encode_add_fast(
            cmd_buf,
            add_pl,
            &scratch.x,
            &scratch.o,
            &scratch.residual,
            h,
            &c.h,
        );

        // ── FFN sub-layer ───────────────────────────────────────────

        // enc16: norm_r = rmsnorm(residual, ffn_norm)
        encode_rmsnorm_fast(
            cmd_buf,
            rmsnorm_pl,
            &scratch.residual,
            &lw.ffn_norm,
            &scratch.norm_r,
            &c.h,
            &c.eps,
        );

        // enc17-18: gate/up
        if lw.has_int8 {
            if lw.has_gate {
                if let (Some(ref gi), Some(ref gs)) = (&lw.gate_int8, &lw.gate_scale) {
                    encode_gemv_int8(
                        cmd_buf,
                        gemv_int8_pl,
                        &scratch.norm_r,
                        gi,
                        gs,
                        &scratch.gate_buf,
                        inter,
                        &c.inter,
                        &c.h,
                    );
                }
            }
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.norm_r,
                lw.up_int8.as_ref().unwrap(),
                lw.up_scale.as_ref().unwrap(),
                &scratch.up_buf,
                inter,
                &c.inter,
                &c.h,
            );
        } else {
            if lw.has_gate {
                encode_gemv_fast(
                    cmd_buf,
                    gemv_pl,
                    &scratch.norm_r,
                    &lw.gate,
                    &scratch.gate_buf,
                    inter,
                    &c.inter,
                    &c.h,
                );
            }
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.norm_r,
                &lw.up,
                &scratch.up_buf,
                inter,
                &c.inter,
                &c.h,
            );
        }

        // enc19: swiglu
        encode_swiglu_fast(
            cmd_buf,
            swiglu_pl,
            &scratch.gate_buf,
            &scratch.up_buf,
            &scratch.hidden,
            inter,
            &c.inter,
        );

        // enc20: down = gemv(hidden, w_down)
        if lw.has_int8 {
            encode_gemv_int8(
                cmd_buf,
                gemv_int8_pl,
                &scratch.hidden,
                lw.down_int8.as_ref().unwrap(),
                lw.down_scale.as_ref().unwrap(),
                &scratch.down_buf,
                h,
                &c.h,
                &c.inter,
            );
        } else {
            encode_gemv_fast(
                cmd_buf,
                gemv_pl,
                &scratch.hidden,
                &lw.down,
                &scratch.down_buf,
                h,
                &c.h,
                &c.inter,
            );
        }

        // enc21: x = residual + down
        encode_add_fast(
            cmd_buf,
            add_pl,
            &scratch.residual,
            &scratch.down_buf,
            &scratch.x,
            h,
            &c.h,
        );
    }

    // 5. ONE commit + wait
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // 6. Update KV cache position
    model_bufs.kv_cache.pos += 1;

    // 7. Read scratch.x back to CPU
    read_buffer(&model_bufs.scratch.x, h)
}

// ── New encoder helpers for Phase 3 kernels ─────────────────────────

/// GEMV with pre-allocated constant buffers (no per-dispatch allocation).
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

fn encode_gemv(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b: &::metal::Buffer,
    buf_c: &::metal::Buffer,
    n: usize,
    k: usize,
    dev: &::metal::Device,
) {
    let buf_n = make_const_u32(dev, n as u32);
    let buf_k = make_const_u32(dev, k as u32);
    encode_gemv_fast(cmd_buf, pipeline, buf_a, buf_b, buf_c, n, &buf_n, &buf_k);
}

fn encode_rope(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_qk: &::metal::Buffer,
    buf_cos: &::metal::Buffer,
    buf_sin: &::metal::Buffer,
    pos: usize,
    num_heads: usize,
    head_dim: usize,
    half_d: usize,
    dev: &::metal::Device,
) {
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let cos_offset = (pos * half_d * std::mem::size_of::<f32>()) as u64;
    let sin_offset = cos_offset;

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_qk), 0);
    encoder.set_buffer(1, Some(buf_cos), cos_offset);
    encoder.set_buffer(2, Some(buf_sin), sin_offset);
    encoder.set_buffer(3, Some(&buf_num_heads), 0);
    encoder.set_buffer(4, Some(&buf_head_dim), 0);

    let total_pairs = num_heads * half_d;
    let grid = ::metal::MTLSize::new(total_pairs as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(total_pairs as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

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

    // One threadgroup per head, 256 threads per threadgroup
    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

fn encode_headwise_rmsnorm(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &::metal::Buffer,
    buf_w: &::metal::Buffer,
    buf_out: &::metal::Buffer,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    head_dim_weight: usize,
    dev: &::metal::Device,
) {
    let buf_num_heads = make_const_u32(dev, num_heads as u32);
    let buf_head_dim = make_const_u32(dev, head_dim as u32);
    let buf_eps = make_const_f32(dev, eps);
    let buf_hdw = make_const_u32(dev, head_dim_weight as u32);

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_w), 0);
    encoder.set_buffer(2, Some(buf_out), 0);
    encoder.set_buffer(3, Some(&buf_num_heads), 0);
    encoder.set_buffer(4, Some(&buf_head_dim), 0);
    encoder.set_buffer(5, Some(&buf_eps), 0);
    encoder.set_buffer(6, Some(&buf_hdw), 0);

    // One threadgroup per head, 256 threads per threadgroup
    let threadgroups = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, tg);
    encoder.end_encoding();
}

// ── Utility ─────────────────────────────────────────────────────────

fn add_bias_inplace(x: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    for (val, &b) in x.iter_mut().zip(bias.iter()) {
        *val += b;
    }
}

/// INT8 GEMV encoder: y[N] = A_f32 @ dequant(B_int8) * scales
fn encode_gemv_int8(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &::metal::Buffer,
    buf_b_int8: &::metal::Buffer,
    buf_scales: &::metal::Buffer,
    buf_c: &::metal::Buffer,
    n: usize,
    buf_n: &::metal::Buffer,
    buf_k: &::metal::Buffer,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b_int8), 0);
    encoder.set_buffer(2, Some(buf_scales), 0);
    encoder.set_buffer(3, Some(buf_c), 0);
    encoder.set_buffer(4, Some(buf_n), 0);
    encoder.set_buffer(5, Some(buf_k), 0);
    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

// ── Fast encoder variants (zero-allocation, pre-allocated constants) ─

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

// ── Utility ─────────────────────────────────────────────────────────

fn make_const_u32(device: &::metal::Device, val: u32) -> ::metal::Buffer {
    let data = [f32::from_bits(val)];
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        4,
        ::metal::MTLResourceOptions::StorageModeShared,
    )
}

fn make_const_f32(device: &::metal::Device, val: f32) -> ::metal::Buffer {
    device.new_buffer_with_data(
        &val as *const f32 as *const _,
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
