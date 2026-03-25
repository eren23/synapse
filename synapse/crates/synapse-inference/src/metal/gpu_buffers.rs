//! GPU-resident model buffers for Phase 3 all-layers-in-one-command-buffer decode.
//!
//! Pre-uploads all weights to Metal shared buffers at init, allocates persistent
//! scratch buffers and a GPU-side KV cache, so that the decode loop can encode
//! ALL layers into a single command buffer with zero CPU-GPU round trips.

use ::metal::{Buffer, Device, MTLResourceOptions};

use crate::model::causal_lm::CausalLM;
use crate::model::decoder_layer::DecoderLayer;

/// Pre-uploaded weight buffers for one decoder layer.
pub struct MetalLayerWeights {
    pub wq: Buffer,            // pre-transposed [K, q_dim]
    pub wk: Buffer,            // pre-transposed [K, kv_dim]
    pub wv: Buffer,
    pub wo: Buffer,            // pre-transposed [q_dim, h]
    pub gate: Buffer,          // pre-transposed [h, inter]
    pub up: Buffer,
    pub down: Buffer,          // pre-transposed [inter, h]
    pub attn_norm: Buffer,     // [h]
    pub ffn_norm: Buffer,      // [h]
    pub q_bias: Option<Buffer>,  // [q_dim] or None
    pub k_bias: Option<Buffer>,  // [kv_dim] or None
    pub v_bias: Option<Buffer>,  // [kv_dim] or None
    pub q_norm: Option<Buffer>,  // [head_dim] or None
    pub k_norm: Option<Buffer>,  // [head_dim] or None
    /// Whether this layer has a gate projection (SwiGLU/GeGLU).
    pub has_gate: bool,
}

/// GPU-resident KV cache.
pub struct MetalKVCache {
    pub layers: Vec<MetalKVCacheLayer>,
    pub pos: usize,
    pub max_seq: usize,
}

pub struct MetalKVCacheLayer {
    pub k_cache: Buffer,   // [max_seq * kv_dim]
    pub v_cache: Buffer,   // [max_seq * kv_dim]
    pub kv_dim: usize,
}

/// Reusable scratch buffers for one decode step.
pub struct ScratchBuffers {
    pub x: Buffer,         // [h] -- layer input/output
    pub norm_x: Buffer,    // [h]
    pub q: Buffer,         // [q_dim]
    pub k: Buffer,         // [kv_dim]
    pub v: Buffer,         // [kv_dim]
    pub attn_out: Buffer,  // [q_dim]
    pub o: Buffer,         // [h]
    pub residual: Buffer,  // [h]
    pub norm_r: Buffer,    // [h]
    pub gate_buf: Buffer,  // [inter]
    pub up_buf: Buffer,    // [inter]
    pub hidden: Buffer,    // [inter]
    pub down_buf: Buffer,  // [h]
}

/// All GPU-resident model buffers.
pub struct MetalModelBuffers {
    pub layer_weights: Vec<MetalLayerWeights>,
    pub scratch: ScratchBuffers,
    pub kv_cache: MetalKVCache,
    pub rope_cos: Buffer,
    pub rope_sin: Buffer,
    pub seq_len_buf: Buffer,
    /// Pre-allocated constant buffers to avoid per-dispatch Metal buffer allocation.
    pub consts: ConstantBuffers,
}

/// Pre-allocated Metal constant buffers for dimension values that never change.
/// Eliminates ~1500 buffer allocations per token.
pub struct ConstantBuffers {
    pub h: Buffer,            // hidden_size
    pub q_dim: Buffer,        // num_heads * head_dim
    pub kv_dim: Buffer,       // num_kv_heads * head_dim
    pub inter: Buffer,        // intermediate_size
    pub head_dim: Buffer,
    pub num_heads: Buffer,
    pub num_kv_heads: Buffer,
    pub eps: Buffer,          // f32
    pub one: Buffer,          // M=1 for matmul
}

// ---- Helpers ---------------------------------------------------------------

/// Create a Metal buffer from f32 data.
fn upload(device: &Device, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Create a zeroed Metal buffer for `n` f32 elements.
fn alloc_empty(device: &Device, n: usize) -> Buffer {
    device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Transpose a weight matrix from logical [n, k] (row-major) to [k, n] (row-major).
/// This matches the layout expected by the matmul/gemv kernels where B is [K, N].
fn transpose(data: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut bt = vec![0.0f32; k * n];
    for i in 0..n {
        for j in 0..k {
            bt[j * n + i] = data[i * k + j];
        }
    }
    bt
}

/// Upload a transposed weight to GPU. Returns a Buffer with layout [k, n].
fn upload_transposed(device: &Device, data: &[f32], n: usize, k: usize) -> Buffer {
    let bt = transpose(data, n, k);
    upload(device, &bt)
}

/// Upload bias if non-empty, else return None.
fn upload_bias(device: &Device, data: &[f32]) -> Option<Buffer> {
    if data.is_empty() {
        None
    } else {
        Some(upload(device, data))
    }
}

/// Upload per-head norm weights if non-empty, else return None.
fn upload_norm(device: &Device, data: &[f32]) -> Option<Buffer> {
    if data.is_empty() {
        None
    } else {
        Some(upload(device, data))
    }
}

// ---- Implementations -------------------------------------------------------

impl MetalLayerWeights {
    /// Transpose weights and upload them to GPU for one decoder layer.
    pub fn from_decoder_layer(layer: &DecoderLayer, device: &Device) -> Self {
        let h = layer.hidden_size;
        let num_heads = layer.attention.num_heads();
        let num_kv_heads = layer.attention.num_kv_heads();
        let head_dim = layer.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter = layer.ffn.intermediate_size();

        // Weight matrices: stored as [out_dim, in_dim] in the model (row-major).
        // The gemv kernel wants B as [K, N] where K=in_dim, N=out_dim.
        // So we transpose [out_dim, in_dim] -> [in_dim, out_dim].
        let wq = upload_transposed(device, &layer.w_q, q_dim, h);
        let wk = upload_transposed(device, &layer.w_k, kv_dim, h);
        let wv = upload_transposed(device, &layer.w_v, kv_dim, h);
        let wo = upload_transposed(device, &layer.w_o, h, q_dim);

        let has_gate = !layer.ffn_gate.is_empty();
        let gate = if has_gate {
            upload_transposed(device, &layer.ffn_gate, inter, h)
        } else {
            alloc_empty(device, 1) // dummy
        };
        let up = upload_transposed(device, &layer.ffn_up, inter, h);
        let down = upload_transposed(device, &layer.ffn_down, h, inter);

        let attn_norm = upload(device, &layer.attn_norm_weight);
        let ffn_norm = upload(device, &layer.ffn_norm_weight);

        let q_bias = upload_bias(device, &layer.q_bias);
        let k_bias = upload_bias(device, &layer.k_bias);
        let v_bias = upload_bias(device, &layer.v_bias);
        let q_norm = upload_norm(device, &layer.q_norm_weight);
        let k_norm = upload_norm(device, &layer.k_norm_weight);

        Self {
            wq, wk, wv, wo,
            gate, up, down,
            attn_norm, ffn_norm,
            q_bias, k_bias, v_bias,
            q_norm, k_norm,
            has_gate,
        }
    }
}

impl MetalKVCache {
    /// Allocate a GPU-resident KV cache for all layers.
    pub fn new(num_layers: usize, max_seq: usize, kv_dim: usize, device: &Device) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(MetalKVCacheLayer {
                k_cache: alloc_empty(device, max_seq * kv_dim),
                v_cache: alloc_empty(device, max_seq * kv_dim),
                kv_dim,
            });
        }
        Self {
            layers,
            pos: 0,
            max_seq,
        }
    }

    /// Truncate the cache to a given position (for speculative decoding rollback).
    pub fn truncate_to(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Populate the GPU KV cache from a CPU KV cache after prefill.
    ///
    /// Copies each layer's K and V data from the Zig-backed CPU cache into
    /// the Metal shared-memory buffers so the GPU decode loop can access them.
    pub fn populate_from_cpu_cache(&mut self, cpu_cache: &crate::kv_cache::KVCache) {
        let seq_len = cpu_cache.current_len().expect("failed to read cache length");
        for (i, gpu_layer) in self.layers.iter().enumerate() {
            let (k_data, v_data, layer_len) = cpu_cache.get(i).expect("failed to read CPU KV cache layer");
            debug_assert_eq!(layer_len, seq_len);
            let copy_len = seq_len * gpu_layer.kv_dim;
            unsafe {
                let k_ptr = gpu_layer.k_cache.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_ptr, copy_len);
                let v_ptr = gpu_layer.v_cache.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, copy_len);
            }
        }
        self.pos = seq_len;
    }
}

impl ScratchBuffers {
    /// Allocate persistent scratch buffers for the decode step.
    pub fn new(h: usize, q_dim: usize, kv_dim: usize, inter: usize, device: &Device) -> Self {
        Self {
            x: alloc_empty(device, h),
            norm_x: alloc_empty(device, h),
            q: alloc_empty(device, q_dim),
            k: alloc_empty(device, kv_dim),
            v: alloc_empty(device, kv_dim),
            attn_out: alloc_empty(device, q_dim),
            o: alloc_empty(device, h),
            residual: alloc_empty(device, h),
            norm_r: alloc_empty(device, h),
            gate_buf: alloc_empty(device, inter),
            up_buf: alloc_empty(device, inter),
            hidden: alloc_empty(device, inter),
            down_buf: alloc_empty(device, h),
        }
    }
}

impl MetalModelBuffers {
    /// Upload all model weights, allocate scratch buffers and KV cache.
    pub fn from_causal_lm(model: &CausalLM, max_seq: usize, device: &Device) -> Self {
        let num_layers = model.layers.len();
        let h = model.config.architecture.hidden_size;
        let num_heads = model.config.attention.num_heads();
        let num_kv_heads = model.config.attention.num_kv_heads();
        let head_dim = model.config.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter = model.config.ffn.intermediate_size();

        // Upload layer weights
        let layer_weights: Vec<MetalLayerWeights> = model
            .layers
            .iter()
            .map(|layer| MetalLayerWeights::from_decoder_layer(layer, device))
            .collect();

        // Scratch buffers
        let scratch = ScratchBuffers::new(h, q_dim, kv_dim, inter, device);

        // KV cache
        let kv_cache = MetalKVCache::new(num_layers, max_seq, kv_dim, device);

        // RoPE tables
        let rope_cos = upload(device, &model.rope_cos);
        let rope_sin = upload(device, &model.rope_sin);

        // seq_len buffer (single u32, written before each decode step)
        let seq_len_buf = alloc_empty(device, 1);

        // Pre-allocate constant buffers
        let eps = model.layers.first().map_or(1e-6, |l| l.attn_norm.eps() as f32);
        let consts = ConstantBuffers::new(
            h, q_dim, kv_dim, inter, head_dim, num_heads, num_kv_heads, eps, device,
        );

        Self {
            layer_weights,
            scratch,
            kv_cache,
            rope_cos,
            rope_sin,
            seq_len_buf,
            consts,
        }
    }
}

impl ConstantBuffers {
    fn new(
        h: usize, q_dim: usize, kv_dim: usize, inter: usize,
        head_dim: usize, num_heads: usize, num_kv_heads: usize,
        eps: f32, device: &Device,
    ) -> Self {
        let make_u32 = |val: u32| -> Buffer {
            let data = [f32::from_bits(val)];
            device.new_buffer_with_data(
                data.as_ptr() as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let make_f32 = |val: f32| -> Buffer {
            device.new_buffer_with_data(
                &val as *const f32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            )
        };

        Self {
            h: make_u32(h as u32),
            q_dim: make_u32(q_dim as u32),
            kv_dim: make_u32(kv_dim as u32),
            inter: make_u32(inter as u32),
            head_dim: make_u32(head_dim as u32),
            num_heads: make_u32(num_heads as u32),
            num_kv_heads: make_u32(num_kv_heads as u32),
            eps: make_f32(eps),
            one: make_u32(1),
        }
    }
}
