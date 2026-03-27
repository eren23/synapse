//! GPU-accelerated LEWM predict_next using a single Metal command buffer.
//!
//! Encodes all 6 adaLN predictor layers into one command buffer so the GPU
//! processes them back-to-back with zero CPU-GPU synchronization between layers.
//! This targets sub-0.76ms latency (PyTorch MPS baseline) for seq_len=3, hidden=192.

use ::metal::{Buffer, Device, MTLResourceOptions};

use super::device::MetalBackend;
use crate::model::lewm::{AdaLNTransformerLayer, LeWMConfig, LeWorldModel};

// ── GPU weight buffers ──────────────────────────────────────────────

/// Pre-uploaded weights for one adaLN predictor layer on GPU.
pub struct MetalAdaLNWeights {
    pub adaln_weight: Buffer,       // [1152, 192] row-major (N=1152, K=192)
    pub adaln_bias: Buffer,         // [1152]
    pub to_qkv: Buffer,            // [3072, 192] row-major (N=3072, K=192)
    pub attn_out_weight: Buffer,    // [192, 1024] row-major (N=192, K=1024)
    pub attn_out_bias: Buffer,      // [192]
    pub attn_norm_weight: Buffer,   // [192]
    pub mlp_norm_weight: Buffer,    // [192]
    pub mlp_up_weight: Buffer,      // [2048, 192] row-major (N=2048, K=192)
    pub mlp_up_bias: Buffer,        // [2048]
    pub mlp_down_weight: Buffer,    // [192, 2048] row-major (N=192, K=2048)
    pub mlp_down_bias: Buffer,      // [192]
    pub has_adaln_bias: bool,
    pub has_attn_out_bias: bool,
    pub has_mlp_up_bias: bool,
    pub has_mlp_down_bias: bool,
}

/// Pre-uploaded LEWM predictor weights on GPU.
pub struct MetalLeWMWeights {
    pub layers: Vec<MetalAdaLNWeights>,
    pub pos_embed: Buffer,           // [seq_len * hidden]
    pub final_norm_weight: Buffer,   // [hidden]
    pub final_norm_bias: Buffer,     // [hidden]
    pub has_final_norm_bias: bool,
}

/// Pre-allocated scratch buffers for the LEWM GPU forward pass.
pub struct LeWMScratchBuffers {
    pub seq: Buffer,         // [seq_len * hidden] = [3 * 192]
    pub mod_params: Buffer,  // [6 * hidden] = [1152]
    pub normed: Buffer,      // [seq_len * hidden]
    pub qkv: Buffer,         // [seq_len * 3 * inner_dim] = [3 * 3072]
    pub q: Buffer,           // [seq_len * inner_dim] = [3 * 1024]
    pub k: Buffer,           // [seq_len * inner_dim]
    pub v: Buffer,           // [seq_len * inner_dim]
    pub attn_out: Buffer,    // [seq_len * inner_dim]
    pub proj: Buffer,        // [seq_len * hidden]
    pub ffn_inter: Buffer,   // [seq_len * inter] = [3 * 2048]
    pub ffn_out: Buffer,     // [seq_len * hidden]
    pub conditioning: Buffer, // [hidden] — action embedding for adaLN (persists across layers)
}

/// Pre-allocated constant buffers for dimensions that never change.
pub struct LeWMConstBuffers {
    pub hidden: Buffer,    // 192
    pub inner_dim: Buffer, // 1024
    pub inter: Buffer,     // 2048
    pub mod_dim: Buffer,   // 1152 (6 * hidden)
    pub seq_len: Buffer,   // 3
    pub num_heads: Buffer, // 16
    pub head_dim: Buffer,  // 64
    pub one: Buffer,       // 1
    pub three: Buffer,     // 3
    pub qkv_dim: Buffer,   // 3072 (3 * inner_dim)
    pub eps: Buffer,       // 1e-6
    // Totals for dispatch
    pub seq_x_hidden: Buffer,    // 3 * 192 = 576
    pub seq_x_inner: Buffer,     // 3 * 1024 = 3072
    pub seq_x_inter: Buffer,     // 3 * 2048 = 6144
    pub seq_x_qkv_dim: Buffer,   // 3 * 3072 = 9216
}

/// Everything needed for GPU-accelerated LEWM predict_next.
pub struct MetalLeWMState {
    pub weights: MetalLeWMWeights,
    pub scratch: LeWMScratchBuffers,
    pub consts: LeWMConstBuffers,
    // Dimension values stored for runtime dispatch
    pub hidden: usize,
    pub inner_dim: usize,
    pub inter: usize,
    pub num_heads: usize,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn upload(device: &Device, data: &[f32]) -> Buffer {
    if data.is_empty() {
        // Metal doesn't support zero-length buffers; allocate 4 bytes as placeholder.
        return alloc_empty(device, 1);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_empty(device: &Device, n: usize) -> Buffer {
    device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_const_u32(device: &Device, val: u32) -> Buffer {
    let data = [f32::from_bits(val)];
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_const_f32(device: &Device, val: f32) -> Buffer {
    device.new_buffer_with_data(
        &val as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_buffer(buf: &Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let mut out = vec![0.0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
    out
}

// ── Weight upload ───────────────────────────────────────────────────

impl MetalAdaLNWeights {
    /// Upload one adaLN layer's weights to GPU.
    /// Weight matrices are stored in their original row-major layout [N, K]
    /// for use with gemv3_t which expects B as [N, K].
    pub fn from_layer(layer: &AdaLNTransformerLayer, device: &Device) -> Self {
        let has_adaln_bias = !layer.adaln_bias.is_empty();
        let has_attn_out_bias = !layer.attn_out_bias.is_empty();
        let has_mlp_up_bias = !layer.mlp_up_bias.is_empty();
        let has_mlp_down_bias = !layer.mlp_down_bias.is_empty();

        MetalAdaLNWeights {
            adaln_weight: upload(device, &layer.adaln_weight),
            adaln_bias: if has_adaln_bias {
                upload(device, &layer.adaln_bias)
            } else {
                alloc_empty(device, 1)
            },
            to_qkv: upload(device, &layer.to_qkv),
            attn_out_weight: upload(device, &layer.attn_out_weight),
            attn_out_bias: if has_attn_out_bias {
                upload(device, &layer.attn_out_bias)
            } else {
                alloc_empty(device, 1)
            },
            attn_norm_weight: upload(device, &layer.attn_norm_weight),
            mlp_norm_weight: upload(device, &layer.mlp_norm_weight),
            mlp_up_weight: upload(device, &layer.mlp_up_weight),
            mlp_up_bias: if has_mlp_up_bias {
                upload(device, &layer.mlp_up_bias)
            } else {
                alloc_empty(device, 1)
            },
            mlp_down_weight: upload(device, &layer.mlp_down_weight),
            mlp_down_bias: if has_mlp_down_bias {
                upload(device, &layer.mlp_down_bias)
            } else {
                alloc_empty(device, 1)
            },
            has_adaln_bias,
            has_attn_out_bias,
            has_mlp_up_bias,
            has_mlp_down_bias,
        }
    }
}

impl MetalLeWMWeights {
    /// Upload all predictor weights from a LeWorldModel to GPU.
    pub fn from_model(model: &LeWorldModel, device: &Device) -> Self {
        let layers: Vec<MetalAdaLNWeights> = model
            .predictor_layers
            .iter()
            .map(|l| MetalAdaLNWeights::from_layer(l, device))
            .collect();

        let has_final_norm_bias = !model.predictor_norm_bias.is_empty();

        MetalLeWMWeights {
            layers,
            pos_embed: upload(device, &model.predictor_pos_embed),
            final_norm_weight: upload(device, &model.predictor_norm_weight),
            final_norm_bias: if has_final_norm_bias {
                upload(device, &model.predictor_norm_bias)
            } else {
                alloc_empty(device, 1)
            },
            has_final_norm_bias,
        }
    }
}

impl LeWMScratchBuffers {
    pub fn new(config: &LeWMConfig, device: &Device) -> Self {
        let seq_len = 3;
        let h = config.predictor_hidden;
        let inner = config.predictor_inner_dim;
        let inter = config.predictor_inter;

        LeWMScratchBuffers {
            seq: alloc_empty(device, seq_len * h),
            mod_params: alloc_empty(device, 6 * h),
            normed: alloc_empty(device, seq_len * h),
            qkv: alloc_empty(device, seq_len * 3 * inner),
            q: alloc_empty(device, seq_len * inner),
            k: alloc_empty(device, seq_len * inner),
            v: alloc_empty(device, seq_len * inner),
            attn_out: alloc_empty(device, seq_len * inner),
            proj: alloc_empty(device, seq_len * h),
            ffn_inter: alloc_empty(device, seq_len * inter),
            ffn_out: alloc_empty(device, seq_len * h),
            conditioning: alloc_empty(device, h),
        }
    }
}

impl LeWMConstBuffers {
    pub fn new(config: &LeWMConfig, device: &Device) -> Self {
        let h = config.predictor_hidden;
        let inner = config.predictor_inner_dim;
        let inter = config.predictor_inter;
        let seq_len = 3usize;
        let num_heads = config.predictor_heads;
        let head_dim = inner / num_heads;
        let mod_dim = 6 * h;
        let qkv_dim = 3 * inner;

        LeWMConstBuffers {
            hidden: make_const_u32(device, h as u32),
            inner_dim: make_const_u32(device, inner as u32),
            inter: make_const_u32(device, inter as u32),
            mod_dim: make_const_u32(device, mod_dim as u32),
            seq_len: make_const_u32(device, seq_len as u32),
            num_heads: make_const_u32(device, num_heads as u32),
            head_dim: make_const_u32(device, head_dim as u32),
            one: make_const_u32(device, 1),
            three: make_const_u32(device, 3),
            qkv_dim: make_const_u32(device, qkv_dim as u32),
            eps: make_const_f32(device, 1e-6),
            seq_x_hidden: make_const_u32(device, (seq_len * h) as u32),
            seq_x_inner: make_const_u32(device, (seq_len * inner) as u32),
            seq_x_inter: make_const_u32(device, (seq_len * inter) as u32),
            seq_x_qkv_dim: make_const_u32(device, (seq_len * qkv_dim) as u32),
        }
    }
}

impl MetalLeWMState {
    /// Create the full GPU state from a LeWorldModel, uploading all weights.
    pub fn from_model(model: &LeWorldModel, backend: &MetalBackend) -> Self {
        let device = &backend.device;
        MetalLeWMState {
            weights: MetalLeWMWeights::from_model(model, device),
            scratch: LeWMScratchBuffers::new(&model.config, device),
            consts: LeWMConstBuffers::new(&model.config, device),
            hidden: model.config.predictor_hidden,
            inner_dim: model.config.predictor_inner_dim,
            inter: model.config.predictor_inter,
            num_heads: model.config.predictor_heads,
        }
    }
}

// ── Encoder helpers ─────────────────────────────────────────────────

/// Encode a gemv3_t dispatch: C[M,N] = A[M,K] * B^T[N,K]
fn encode_gemv3(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_a: &Buffer,
    buf_b: &Buffer,
    buf_c: &Buffer,
    buf_m: &Buffer,
    buf_n: &Buffer,
    buf_k: &Buffer,
    n: usize,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_c), 0);
    encoder.set_buffer(3, Some(buf_m), 0);
    encoder.set_buffer(4, Some(buf_n), 0);
    encoder.set_buffer(5, Some(buf_k), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Encode add_bias dispatch: x[i] += bias[i % out_dim]
fn encode_add_bias(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &Buffer,
    buf_bias: &Buffer,
    buf_out_dim: &Buffer,
    buf_total: &Buffer,
    total: usize,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_bias), 0);
    encoder.set_buffer(2, Some(buf_out_dim), 0);
    encoder.set_buffer(3, Some(buf_total), 0);

    let grid = ::metal::MTLSize::new(total as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(total as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Encode gelu_inplace dispatch.
fn encode_gelu(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_x: &Buffer,
    buf_n: &Buffer,
    n: usize,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_x), 0);
    encoder.set_buffer(1, Some(buf_n), 0);

    let grid = ::metal::MTLSize::new(n as u64, 1, 1);
    let tg = ::metal::MTLSize::new(256.min(n as u64), 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

/// Encode attention_3x3 dispatch.
fn encode_attention_3x3(
    cmd_buf: &::metal::CommandBufferRef,
    pipeline: &::metal::ComputePipelineState,
    buf_q: &Buffer,
    buf_k: &Buffer,
    buf_v: &Buffer,
    buf_out: &Buffer,
    buf_num_heads: &Buffer,
    buf_head_dim: &Buffer,
    buf_seq_len: &Buffer,
    num_heads: usize,
) {
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buf_q), 0);
    encoder.set_buffer(1, Some(buf_k), 0);
    encoder.set_buffer(2, Some(buf_v), 0);
    encoder.set_buffer(3, Some(buf_out), 0);
    encoder.set_buffer(4, Some(buf_num_heads), 0);
    encoder.set_buffer(5, Some(buf_head_dim), 0);
    encoder.set_buffer(6, Some(buf_seq_len), 0);

    // One thread per head
    let grid = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    let tg = ::metal::MTLSize::new(num_heads as u64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
}

// ── Main forward pass ───────────────────────────────────────────────

/// Run all 6 predictor layers in one Metal command buffer.
///
/// Input: `seq` is `[3, 192]` = `[z_t + pos_embed, a_embed + pos_embed, zeros + pos_embed]`.
/// `conditioning` is `[192]` (action embedding for adaLN modulation).
/// Returns: the full sequence output `[3 * hidden]` (caller extracts target position).
///
/// The entire forward pass — all 6 layers, all matmuls, norms, attention, activations —
/// is encoded into a single command buffer. ONE commit, ONE wait.
pub fn lewm_predict_metal(
    seq: &[f32],           // [3 * 192] = z_t + action_embed + zeros (with pos_embed added)
    conditioning: &[f32],  // [192] action embedding for adaLN
    state: &MetalLeWMState,
    backend: &MetalBackend,
) -> Vec<f32> {
    let config_h = state.hidden;
    let inner = state.inner_dim;
    let inter = state.inter;
    let seq_len = 3;
    let num_heads = state.num_heads;
    let mod_dim = 6 * config_h;

    let scratch = &state.scratch;
    let consts = &state.consts;

    // Upload input sequence to scratch.seq (shared memory)
    unsafe {
        let ptr = scratch.seq.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(seq.as_ptr(), ptr, seq_len * config_h);
    }

    // Upload conditioning to dedicated buffer (persists across all layers)
    unsafe {
        let ptr = scratch.conditioning.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conditioning.as_ptr(), ptr, config_h);
    }

    // Fetch pipelines
    let gemv3_pl = backend.pipeline("gemv3_t").expect("gemv3_t pipeline");
    let ln_mod_pl = backend
        .pipeline("layernorm_modulate")
        .expect("layernorm_modulate pipeline");
    let gelu_pl = backend
        .pipeline("gelu_inplace")
        .expect("gelu_inplace pipeline");
    let gated_res_pl = backend
        .pipeline("gated_residual")
        .expect("gated_residual pipeline");
    let add_bias_pl = backend.pipeline("add_bias").expect("add_bias pipeline");
    let attn_pl = backend
        .pipeline("attention_3x3")
        .expect("attention_3x3 pipeline");

    // Create ONE command buffer for all 6 layers
    let cmd_buf = backend.command_queue.new_command_buffer();

    for layer_idx in 0..state.weights.layers.len() {
        let lw = &state.weights.layers[layer_idx];

        // ── Step 1: adaLN modulation ─────────────────────────────────
        // conditioning [1, hidden] × adaln_weight [mod_dim, hidden]^T → mod_params [1, mod_dim]
        // Using gemv3_t with M=1
        encode_gemv3(
            cmd_buf,
            gemv3_pl,
            &scratch.conditioning,
            &lw.adaln_weight,
            &scratch.mod_params,
            &consts.one,
            &consts.mod_dim,
            &consts.hidden,
            mod_dim,
        );

        // Add adaLN bias if present
        if lw.has_adaln_bias {
            encode_add_bias(
                cmd_buf,
                add_bias_pl,
                &scratch.mod_params,
                &lw.adaln_bias,
                &consts.mod_dim,
                &consts.mod_dim,
                mod_dim,
            );
        }

        // mod_params layout: [scale1(192), shift1(192), gate1(192), scale2(192), shift2(192), gate2(192)]
        // We use buffer offsets to access each sub-vector.
        let scale1_off = 0u64;
        let shift1_off = (config_h * 4) as u64;
        let gate1_off = (2 * config_h * 4) as u64;
        let scale2_off = (3 * config_h * 4) as u64;
        let shift2_off = (4 * config_h * 4) as u64;
        let gate2_off = (5 * config_h * 4) as u64;

        // ── Step 2: Pre-attention layernorm + modulate ───────────────
        // LayerNorm(seq) * (1 + scale1) + shift1 → normed
        {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ln_mod_pl);
            encoder.set_buffer(0, Some(&scratch.seq), 0);
            encoder.set_buffer(1, Some(&lw.attn_norm_weight), 0);
            encoder.set_buffer(2, Some(&scratch.mod_params), scale1_off);
            encoder.set_buffer(3, Some(&scratch.mod_params), shift1_off);
            encoder.set_buffer(4, Some(&scratch.normed), 0);
            encoder.set_buffer(5, Some(&consts.hidden), 0);
            encoder.set_buffer(6, Some(&consts.seq_len), 0);
            encoder.set_buffer(7, Some(&consts.eps), 0);

            let total = seq_len * config_h;
            let grid = ::metal::MTLSize::new(total as u64, 1, 1);
            let tg = ::metal::MTLSize::new(256.min(total as u64), 1, 1);
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }

        // ── Step 3: Q/K/V projections ────────────────────────────────
        // to_qkv is stored as [3*inner_dim, hidden] row-major.
        // We split it into three separate projections using buffer byte offsets
        // so Q, K, V land in contiguous buffers for the attention kernel.
        {
            let k_weight_off = (inner * config_h * 4) as u64;
            let v_weight_off = (2 * inner * config_h * 4) as u64;

            // Q: normed [3, 192] × to_qkv[0..inner, :] → q [3, 1024]
            encode_gemv3(
                cmd_buf,
                gemv3_pl,
                &scratch.normed,
                &lw.to_qkv,
                &scratch.q,
                &consts.three,
                &consts.inner_dim,
                &consts.hidden,
                inner,
            );

            // K: normed [3, 192] × to_qkv[inner..2*inner, :] → k [3, 1024]
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(gemv3_pl);
                encoder.set_buffer(0, Some(&scratch.normed), 0);
                encoder.set_buffer(1, Some(&lw.to_qkv), k_weight_off);
                encoder.set_buffer(2, Some(&scratch.k), 0);
                encoder.set_buffer(3, Some(&consts.three), 0);
                encoder.set_buffer(4, Some(&consts.inner_dim), 0);
                encoder.set_buffer(5, Some(&consts.hidden), 0);

                let grid = ::metal::MTLSize::new(inner as u64, 1, 1);
                let tg = ::metal::MTLSize::new(256.min(inner as u64), 1, 1);
                encoder.dispatch_threads(grid, tg);
                encoder.end_encoding();
            }

            // V: normed [3, 192] × to_qkv[2*inner..3*inner, :] → v [3, 1024]
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(gemv3_pl);
                encoder.set_buffer(0, Some(&scratch.normed), 0);
                encoder.set_buffer(1, Some(&lw.to_qkv), v_weight_off);
                encoder.set_buffer(2, Some(&scratch.v), 0);
                encoder.set_buffer(3, Some(&consts.three), 0);
                encoder.set_buffer(4, Some(&consts.inner_dim), 0);
                encoder.set_buffer(5, Some(&consts.hidden), 0);

                let grid = ::metal::MTLSize::new(inner as u64, 1, 1);
                let tg = ::metal::MTLSize::new(256.min(inner as u64), 1, 1);
                encoder.dispatch_threads(grid, tg);
                encoder.end_encoding();
            }
        }

        // ── Step 5: Bidirectional attention (3x3, 16 heads) ─────────
        encode_attention_3x3(
            cmd_buf,
            attn_pl,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            &scratch.attn_out,
            &consts.num_heads,
            &consts.head_dim,
            &consts.seq_len,
            num_heads,
        );

        // ── Step 6: Output projection ───────────────────────────────
        // attn_out [3, 1024] × attn_out_weight [192, 1024]^T → proj [3, 192]
        encode_gemv3(
            cmd_buf,
            gemv3_pl,
            &scratch.attn_out,
            &lw.attn_out_weight,
            &scratch.proj,
            &consts.three,
            &consts.hidden,
            &consts.inner_dim,
            config_h,
        );

        // Add attn output bias
        if lw.has_attn_out_bias {
            encode_add_bias(
                cmd_buf,
                add_bias_pl,
                &scratch.proj,
                &lw.attn_out_bias,
                &consts.hidden,
                &consts.seq_x_hidden,
                seq_len * config_h,
            );
        }

        // ── Step 7: Gated residual (attention) ──────────────────────
        // seq[i] += gate1[i % hidden] * proj[i]
        {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(gated_res_pl);
            encoder.set_buffer(0, Some(&scratch.seq), 0);
            encoder.set_buffer(1, Some(&scratch.proj), 0);
            encoder.set_buffer(2, Some(&scratch.mod_params), gate1_off);
            encoder.set_buffer(3, Some(&consts.hidden), 0);
            encoder.set_buffer(4, Some(&consts.seq_x_hidden), 0);

            let total = seq_len * config_h;
            let grid = ::metal::MTLSize::new(total as u64, 1, 1);
            let tg = ::metal::MTLSize::new(256.min(total as u64), 1, 1);
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }

        // ── Step 8: Pre-FFN layernorm + modulate ────────────────────
        {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ln_mod_pl);
            encoder.set_buffer(0, Some(&scratch.seq), 0);
            encoder.set_buffer(1, Some(&lw.mlp_norm_weight), 0);
            encoder.set_buffer(2, Some(&scratch.mod_params), scale2_off);
            encoder.set_buffer(3, Some(&scratch.mod_params), shift2_off);
            encoder.set_buffer(4, Some(&scratch.normed), 0);
            encoder.set_buffer(5, Some(&consts.hidden), 0);
            encoder.set_buffer(6, Some(&consts.seq_len), 0);
            encoder.set_buffer(7, Some(&consts.eps), 0);

            let total = seq_len * config_h;
            let grid = ::metal::MTLSize::new(total as u64, 1, 1);
            let tg = ::metal::MTLSize::new(256.min(total as u64), 1, 1);
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }

        // ── Step 9: FFN up projection ───────────────────────────────
        // normed [3, 192] × mlp_up_weight [2048, 192]^T → ffn_inter [3, 2048]
        encode_gemv3(
            cmd_buf,
            gemv3_pl,
            &scratch.normed,
            &lw.mlp_up_weight,
            &scratch.ffn_inter,
            &consts.three,
            &consts.inter,
            &consts.hidden,
            inter,
        );

        // Add FFN up bias
        if lw.has_mlp_up_bias {
            encode_add_bias(
                cmd_buf,
                add_bias_pl,
                &scratch.ffn_inter,
                &lw.mlp_up_bias,
                &consts.inter,
                &consts.seq_x_inter,
                seq_len * inter,
            );
        }

        // ── Step 10: GELU activation ────────────────────────────────
        encode_gelu(
            cmd_buf,
            gelu_pl,
            &scratch.ffn_inter,
            &consts.seq_x_inter,
            seq_len * inter,
        );

        // ── Step 11: FFN down projection ────────────────────────────
        // ffn_inter [3, 2048] × mlp_down_weight [192, 2048]^T → ffn_out [3, 192]
        encode_gemv3(
            cmd_buf,
            gemv3_pl,
            &scratch.ffn_inter,
            &lw.mlp_down_weight,
            &scratch.ffn_out,
            &consts.three,
            &consts.hidden,
            &consts.inter,
            config_h,
        );

        // Add FFN down bias
        if lw.has_mlp_down_bias {
            encode_add_bias(
                cmd_buf,
                add_bias_pl,
                &scratch.ffn_out,
                &lw.mlp_down_bias,
                &consts.hidden,
                &consts.seq_x_hidden,
                seq_len * config_h,
            );
        }

        // ── Step 12: Gated residual (FFN) ───────────────────────────
        // seq[i] += gate2[i % hidden] * ffn_out[i]
        {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(gated_res_pl);
            encoder.set_buffer(0, Some(&scratch.seq), 0);
            encoder.set_buffer(1, Some(&scratch.ffn_out), 0);
            encoder.set_buffer(2, Some(&scratch.mod_params), gate2_off);
            encoder.set_buffer(3, Some(&consts.hidden), 0);
            encoder.set_buffer(4, Some(&consts.seq_x_hidden), 0);

            let total = seq_len * config_h;
            let grid = ::metal::MTLSize::new(total as u64, 1, 1);
            let tg = ::metal::MTLSize::new(256.min(total as u64), 1, 1);
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }
    }

    // ── Final: single commit + wait ─────────────────────────────────
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read the full sequence output
    read_buffer(&scratch.seq, seq_len * config_h)
}

/// Fused Metal LEWM predict: ONE kernel dispatch per layer = 6 total.
///
/// Uses `adaln_layer_fused` kernel that does the entire adaLN transformer
/// layer in a single dispatch (adaLN + norm + QKV + attention + output proj +
/// gated residual + norm + FFN + gated residual).
///
/// Compared to `lewm_predict_metal` which uses ~65 dispatches (10+ per layer),
/// this reduces GPU scheduling overhead dramatically for small seq_len=3 workloads.
pub fn lewm_predict_metal_fused(
    seq: &[f32],           // [3 * 192]
    conditioning: &[f32],  // [192]
    state: &MetalLeWMState,
    backend: &MetalBackend,
) -> Vec<f32> {
    let config_h = state.hidden;
    let seq_len = 3;
    let scratch = &state.scratch;

    // Upload input sequence
    unsafe {
        let ptr = scratch.seq.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(seq.as_ptr(), ptr, seq_len * config_h);
    }

    // Upload conditioning
    unsafe {
        let ptr = scratch.conditioning.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conditioning.as_ptr(), ptr, config_h);
    }

    let fused_pl = backend.pipeline("adaln_layer_fused").expect("adaln_layer_fused pipeline");

    // Use work-loop pattern: launch moderate thread count, each thread processes
    // multiple elements via stride loop. Matches GPU's preferred occupancy without
    // wasting cycles on idle threads during small steps.
    let num_threads: u64 = 3072; // Optimal: matches SEQ*INNER, good work/thread balance
    let cmd_buf = backend.command_queue.new_command_buffer();

    // 6 layers, ONE dispatch each
    for layer_weights in &state.weights.layers {
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fused_pl);

        // Input/output
        encoder.set_buffer(0, Some(&scratch.seq), 0);
        encoder.set_buffer(1, Some(&scratch.conditioning), 0);

        // Layer weights
        encoder.set_buffer(2, Some(&layer_weights.adaln_weight), 0);
        encoder.set_buffer(3, Some(&layer_weights.adaln_bias), 0);
        encoder.set_buffer(4, Some(&layer_weights.to_qkv), 0);
        encoder.set_buffer(5, Some(&layer_weights.attn_out_weight), 0);
        encoder.set_buffer(6, Some(&layer_weights.attn_out_bias), 0);
        encoder.set_buffer(7, Some(&layer_weights.attn_norm_weight), 0);
        encoder.set_buffer(8, Some(&layer_weights.mlp_norm_weight), 0);
        encoder.set_buffer(9, Some(&layer_weights.mlp_up_weight), 0);
        encoder.set_buffer(10, Some(&layer_weights.mlp_up_bias), 0);
        encoder.set_buffer(11, Some(&layer_weights.mlp_down_weight), 0);
        encoder.set_buffer(12, Some(&layer_weights.mlp_down_bias), 0);

        // Scratch buffers
        encoder.set_buffer(13, Some(&scratch.mod_params), 0);
        encoder.set_buffer(14, Some(&scratch.normed), 0);
        encoder.set_buffer(15, Some(&scratch.qkv), 0);
        encoder.set_buffer(16, Some(&scratch.attn_out), 0);
        encoder.set_buffer(17, Some(&scratch.ffn_inter), 0);

        let grid = ::metal::MTLSize::new(num_threads, 1, 1);
        let tg = ::metal::MTLSize::new(num_threads.min(256), 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    read_buffer(&scratch.seq, seq_len * config_h)
}

/// V3: Fused Metal with vectorized float4 dot products.
/// Uses `adaln_layer_fused_simd` kernel — aggressively vectorized matmuls
/// with 3-way unrolled float4 accumulation. Same single-dispatch-per-layer
/// pattern but much faster inner loops.
pub fn lewm_predict_metal_fused_v3(
    seq: &[f32],
    conditioning: &[f32],
    state: &MetalLeWMState,
    backend: &MetalBackend,
) -> Vec<f32> {
    let config_h = state.hidden;
    let seq_len = 3;
    let scratch = &state.scratch;

    unsafe {
        let ptr = scratch.seq.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(seq.as_ptr(), ptr, seq_len * config_h);
    }
    unsafe {
        let ptr = scratch.conditioning.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conditioning.as_ptr(), ptr, config_h);
    }

    let fused_pl = backend.pipeline("adaln_layer_fused_simd").expect("adaln_layer_fused_simd pipeline");

    // Use enough threads for the largest step (QKV: 9216 outputs)
    // but reasonable for GPU occupancy. 4096 = good balance.
    let num_threads: u64 = 4096;
    let cmd_buf = backend.command_queue.new_command_buffer();

    for layer_weights in &state.weights.layers {
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fused_pl);

        encoder.set_buffer(0, Some(&scratch.seq), 0);
        encoder.set_buffer(1, Some(&scratch.conditioning), 0);
        encoder.set_buffer(2, Some(&layer_weights.adaln_weight), 0);
        encoder.set_buffer(3, Some(&layer_weights.adaln_bias), 0);
        encoder.set_buffer(4, Some(&layer_weights.to_qkv), 0);
        encoder.set_buffer(5, Some(&layer_weights.attn_out_weight), 0);
        encoder.set_buffer(6, Some(&layer_weights.attn_out_bias), 0);
        encoder.set_buffer(7, Some(&layer_weights.attn_norm_weight), 0);
        encoder.set_buffer(8, Some(&layer_weights.mlp_norm_weight), 0);
        encoder.set_buffer(9, Some(&layer_weights.mlp_up_weight), 0);
        encoder.set_buffer(10, Some(&layer_weights.mlp_up_bias), 0);
        encoder.set_buffer(11, Some(&layer_weights.mlp_down_weight), 0);
        encoder.set_buffer(12, Some(&layer_weights.mlp_down_bias), 0);
        encoder.set_buffer(13, Some(&scratch.mod_params), 0);
        encoder.set_buffer(14, Some(&scratch.normed), 0);
        encoder.set_buffer(15, Some(&scratch.qkv), 0);
        encoder.set_buffer(16, Some(&scratch.attn_out), 0);
        encoder.set_buffer(17, Some(&scratch.ffn_inter), 0);
        // Buffer 18: padded_a (not used in v3, pass normed as placeholder)
        encoder.set_buffer(18, Some(&scratch.normed), 0);

        let grid = ::metal::MTLSize::new(num_threads, 1, 1);
        let tg = ::metal::MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    read_buffer(&scratch.seq, seq_len * config_h)
}
