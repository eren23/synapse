//! GPU-resident model buffers for hybrid LIV Conv + GQA decode.
//!
//! Pre-uploads all weights to Metal shared buffers at init, allocates persistent
//! scratch buffers, conv state buffers, and a GPU-side KV cache (for GQA layers
//! only), so that the decode loop can encode ALL layers into a single command
//! buffer with zero CPU-GPU round trips.

use ::metal::{Buffer, Device, MTLResourceOptions};

use super::device::MetalBackend;
use crate::models::ssm::hybrid::config::{HybridConfig, LayerKind};
use crate::models::ssm::hybrid::layer::{GqaDecoderLayer, LivConvDecoderLayer};
use crate::models::ssm::hybrid::model::{HybridLayer, HybridModel};

// ── Helpers ─────────────────────────────────────────────────────────

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
/// This matches the layout expected by the gemv kernel where B is [K, N].
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

// ── Per-layer weight structs ────────────────────────────────────────

/// Upload raw bytes (Q4 block data) to a Metal buffer.
fn upload_bytes(device: &Device, data: &[u8]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        data.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Repack Q4_0 raw bytes from row-major [out, in/32 blocks] to the layout
/// expected by gemv_q4: each row's blocks are contiguous (row j = blocks_per_row * 18 bytes).
/// GGUF already stores in this layout, so this is a no-op identity copy.
/// But we need to handle the GGUF [in, out] vs our [out, in] convention.
///
/// GGUF shape for a weight `[N, K]` means N output rows, K input cols.
/// The raw Q4 data has N rows of (K/32) blocks each = N * (K/32) * 18 bytes.
/// gemv_q4 kernel expects: row j at offset j * (K/32) * 18.
/// This matches GGUF layout directly — no repacking needed.
fn upload_q4_for_gemv(device: &Device, raw: &[u8]) -> Buffer {
    upload_bytes(device, raw)
}

/// Weights for one LIV Conv layer on GPU.
///
/// Has both f32 transposed weights (for prefill) and optional raw Q4 (for decode GEMV).
pub struct MetalLivConvWeights {
    pub attn_norm: Buffer,
    pub in_proj: Buffer,       // f32 transposed
    pub conv_weight: Buffer,
    pub out_proj: Buffer,      // f32 transposed
    pub ffn_norm: Buffer,
    pub ffn_gate: Buffer,      // f32 transposed
    pub ffn_up: Buffer,        // f32 transposed
    pub ffn_down: Buffer,      // f32 transposed
    // Raw Q4 for GEMV decode (uploaded directly from GGUF, NOT transposed)
    pub q4_in_proj: Option<Buffer>,
    pub q4_out_proj: Option<Buffer>,
    pub q4_ffn_gate: Option<Buffer>,
    pub q4_ffn_up: Option<Buffer>,
    pub q4_ffn_down: Option<Buffer>,
}

/// Weights for one GQA layer on GPU.
pub struct MetalGqaWeights {
    pub attn_norm: Buffer,
    pub wq: Buffer,
    pub wk: Buffer,
    pub wv: Buffer,
    pub wo: Buffer,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub ffn_norm: Buffer,
    pub ffn_gate: Buffer,
    pub ffn_up: Buffer,
    pub ffn_down: Buffer,
    // Raw Q4 for GEMV decode
    pub q4_wq: Option<Buffer>,
    pub q4_wk: Option<Buffer>,
    pub q4_wv: Option<Buffer>,
    pub q4_wo: Option<Buffer>,
    pub q4_ffn_gate: Option<Buffer>,
    pub q4_ffn_up: Option<Buffer>,
    pub q4_ffn_down: Option<Buffer>,
}

/// Discriminated union of per-layer GPU weights.
pub enum MetalHybridLayerWeights {
    LivConv(MetalLivConvWeights),
    Gqa(MetalGqaWeights),
}

/// KV cache for a single GQA layer.
pub struct MetalHybridKVCacheLayer {
    pub k_cache: Buffer,
    pub v_cache: Buffer,
    pub kv_dim: usize,
}

// ── Scratch buffers ─────────────────────────────────────────────────

/// Reusable scratch buffers for one hybrid decode step.
pub struct HybridScratchBuffers {
    // Shared (carries across layers)
    pub x: Buffer,        // [hidden]
    pub residual: Buffer, // [hidden]

    // Norm output
    pub norm_out: Buffer, // [hidden]

    // GQA-specific
    pub q: Buffer,        // [q_dim]
    pub k: Buffer,        // [kv_dim]
    pub v: Buffer,        // [kv_dim]
    pub attn_out: Buffer, // [q_dim]
    pub o: Buffer,        // [hidden]

    // LIV Conv-specific
    pub proj_out: Buffer,  // [3 * inner] (or max(proj_dim, q_dim))
    pub conv_out: Buffer,  // [inner]
    pub conv_proj: Buffer, // [hidden]

    // FFN (shared across all layer types)
    pub gate_buf: Buffer,   // [inter]
    pub up_buf: Buffer,     // [inter]
    pub ffn_hidden: Buffer, // [inter]
    pub down_buf: Buffer,   // [hidden]
}

/// Pre-allocated Metal constant buffers for dimension values that never change.
pub struct HybridConstantBuffers {
    pub h: Buffer,            // hidden_size
    pub q_dim: Buffer,        // num_heads * head_dim
    pub kv_dim: Buffer,       // num_kv_heads * head_dim
    pub inter: Buffer,        // intermediate_size
    pub inner: Buffer,        // livconv_inner_size
    pub proj_dim: Buffer,     // 3 * inner (LIV Conv input projection)
    pub head_dim: Buffer,     // GQA head dimension
    pub num_heads: Buffer,    // num_attention_heads
    pub num_kv_heads: Buffer, // num_kv_heads
    pub eps: Buffer,          // f32 norm epsilon
    pub one: Buffer,          // M=1 for matmul
    pub kernel_size: Buffer,  // conv kernel size
    pub channels: Buffer,     // inner_size (channels for conv1d_step)
    pub seq_len_buf: Buffer,  // updated each decode step
}

// ── Main struct ─────────────────────────────────────────────────────

/// All GPU-resident model buffers for a hybrid LIV Conv + GQA model.
pub struct MetalHybridBuffers {
    pub layer_weights: Vec<MetalHybridLayerWeights>,
    pub scratch: HybridScratchBuffers,
    /// Conv rolling state: one `[inner * kernel_size]` buffer per LIV Conv layer.
    pub conv_states: Vec<Buffer>,
    /// KV cache layers (one per GQA layer).
    pub kv_layers: Vec<MetalHybridKVCacheLayer>,
    /// Precomputed RoPE cos table.
    pub rope_cos: Buffer,
    /// Precomputed RoPE sin table.
    pub rope_sin: Buffer,
    /// Current position (number of tokens decoded so far).
    pub pos: usize,
    /// Maximum sequence length for KV cache.
    pub max_seq: usize,
    /// Pre-allocated constant buffers.
    pub consts: HybridConstantBuffers,
    /// Maps layer index → conv_states index (None for GQA layers).
    pub conv_indices: Vec<Option<usize>>,
    /// Maps layer index → kv_layers index (None for LIV Conv layers).
    pub kv_indices: Vec<Option<usize>>,
    /// Layer kind for each layer index (for dispatch).
    pub layer_kinds: Vec<LayerKind>,
    // ── LM head (final norm + output projection) on GPU ──
    /// Final RMSNorm weight: `[hidden]`.
    pub final_norm: Buffer,
    /// LM head weight: f32 transposed `[hidden, vocab]` for f32 GEMV.
    pub lm_head_f32: Buffer,
    /// LM head weight: raw Q4 for Q4 GEMV (None if Q6_K/f32).
    pub lm_head_q4: Option<Buffer>,
    /// Logits output scratch: `[vocab]`.
    pub logits_buf: Buffer,
    /// Vocab size constant buffer.
    pub vocab_const: Buffer,
}

// ── Implementations ─────────────────────────────────────────────────

use crate::weight_loading::RawQ4Tensor;
use std::collections::HashMap;

impl MetalLivConvWeights {
    fn from_layer(
        layer: &LivConvDecoderLayer,
        layer_idx: usize,
        q4_map: &HashMap<String, RawQ4Tensor>,
        device: &Device,
    ) -> Self {
        let h = layer.hidden_size;
        let inner = layer.inner_size;
        let inter = layer.intermediate_size;
        let proj_dim = layer.input_proj_weight.len() / h;

        let in_proj = upload_transposed(device, &layer.input_proj_weight, proj_dim, h);
        let out_proj = upload_transposed(device, &layer.output_proj_weight, h, inner);
        let conv_weight = upload(device, &layer.conv_weight);
        let ffn_gate = upload_transposed(device, &layer.ffn_gate_weight, inter, h);
        let ffn_up = upload_transposed(device, &layer.ffn_up_weight, inter, h);
        let ffn_down = upload_transposed(device, &layer.ffn_down_weight, h, inter);
        let attn_norm = upload(device, &layer.attn_norm_weight);
        let ffn_norm = upload(device, &layer.ffn_norm_weight);

        // Upload raw Q4 if available (NOT transposed — gemv_q4 reads row-major)
        let b = format!("blk.{layer_idx}");
        let q4_in_proj = q4_map.get(&format!("{b}.shortconv.in_proj.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_out_proj = q4_map.get(&format!("{b}.shortconv.out_proj.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_gate = q4_map.get(&format!("{b}.ffn_gate.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_up = q4_map.get(&format!("{b}.ffn_up.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_down = q4_map.get(&format!("{b}.ffn_down.weight")).map(|t| upload_q4_for_gemv(device, &t.data));

        MetalLivConvWeights {
            attn_norm, in_proj, conv_weight, out_proj,
            ffn_norm, ffn_gate, ffn_up, ffn_down,
            q4_in_proj, q4_out_proj, q4_ffn_gate, q4_ffn_up, q4_ffn_down,
        }
    }
}

impl MetalGqaWeights {
    fn from_layer(
        layer: &GqaDecoderLayer,
        layer_idx: usize,
        q4_map: &HashMap<String, RawQ4Tensor>,
        device: &Device,
    ) -> Self {
        let h = layer.hidden_size;
        let q_dim = layer.num_q_heads * layer.head_dim;
        let kv_dim = layer.num_kv_heads * layer.head_dim;
        let inter = layer.intermediate_size;

        // Transpose for GEMV: [out_dim, in_dim] → [in_dim, out_dim]
        let wq = upload_transposed(device, &layer.w_q, q_dim, h);
        let wk = upload_transposed(device, &layer.w_k, kv_dim, h);
        let wv = upload_transposed(device, &layer.w_v, kv_dim, h);
        let wo = upload_transposed(device, &layer.w_o, h, q_dim);

        let q_norm = upload(device, &layer.q_norm_weight);
        let k_norm = upload(device, &layer.k_norm_weight);

        let attn_norm = upload(device, &layer.attn_norm_weight);
        let ffn_norm = upload(device, &layer.ffn_norm_weight);

        let ffn_gate = upload_transposed(device, &layer.ffn_gate_weight, inter, h);
        let ffn_up = upload_transposed(device, &layer.ffn_up_weight, inter, h);
        let ffn_down = upload_transposed(device, &layer.ffn_down_weight, h, inter);

        // Upload raw Q4 if available
        let b = format!("blk.{layer_idx}");
        let q4_wq = q4_map.get(&format!("{b}.attn_q.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_wk = q4_map.get(&format!("{b}.attn_k.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_wv = q4_map.get(&format!("{b}.attn_v.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_wo = q4_map.get(&format!("{b}.attn_output.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_gate = q4_map.get(&format!("{b}.ffn_gate.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_up = q4_map.get(&format!("{b}.ffn_up.weight")).map(|t| upload_q4_for_gemv(device, &t.data));
        let q4_ffn_down = q4_map.get(&format!("{b}.ffn_down.weight")).map(|t| upload_q4_for_gemv(device, &t.data));

        MetalGqaWeights {
            attn_norm, wq, wk, wv, wo, q_norm, k_norm,
            ffn_norm, ffn_gate, ffn_up, ffn_down,
            q4_wq, q4_wk, q4_wv, q4_wo,
            q4_ffn_gate, q4_ffn_up, q4_ffn_down,
        }
    }
}

impl HybridScratchBuffers {
    fn new(config: &HybridConfig, device: &Device) -> Self {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.gqa_head_dim;
        let kv_dim = config.num_kv_heads * config.gqa_head_dim;
        let inter = config.intermediate_size;
        let inner = config.livconv_inner_size;
        // proj_out must be large enough for the largest projection:
        // LIV Conv: 3 * inner (or 2 * inner). GQA: q_dim.
        let proj_out_size = (3 * inner).max(q_dim);

        HybridScratchBuffers {
            x: alloc_empty(device, h),
            residual: alloc_empty(device, h),
            norm_out: alloc_empty(device, h),
            q: alloc_empty(device, q_dim),
            k: alloc_empty(device, kv_dim),
            v: alloc_empty(device, kv_dim),
            attn_out: alloc_empty(device, q_dim),
            o: alloc_empty(device, h),
            proj_out: alloc_empty(device, proj_out_size),
            conv_out: alloc_empty(device, inner.max(1)),
            conv_proj: alloc_empty(device, h),
            gate_buf: alloc_empty(device, inter),
            up_buf: alloc_empty(device, inter),
            ffn_hidden: alloc_empty(device, inter),
            down_buf: alloc_empty(device, h),
        }
    }
}

impl HybridConstantBuffers {
    fn new(config: &HybridConfig, device: &Device) -> Self {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.gqa_head_dim;
        let kv_dim = config.num_kv_heads * config.gqa_head_dim;
        let inter = config.intermediate_size;
        let inner = config.livconv_inner_size;
        let proj_dim = 3 * inner; // double-gated default
        let head_dim = config.gqa_head_dim;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let eps = config.norm_eps as f32;
        let kernel_size = config.livconv_kernel_size;

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

        HybridConstantBuffers {
            h: make_u32(h as u32),
            q_dim: make_u32(q_dim as u32),
            kv_dim: make_u32(kv_dim as u32),
            inter: make_u32(inter as u32),
            inner: make_u32(inner as u32),
            proj_dim: make_u32(proj_dim as u32),
            head_dim: make_u32(head_dim as u32),
            num_heads: make_u32(num_heads as u32),
            num_kv_heads: make_u32(num_kv_heads as u32),
            eps: make_f32(eps),
            one: make_u32(1),
            kernel_size: make_u32(kernel_size as u32),
            channels: make_u32(inner as u32),
            seq_len_buf: alloc_empty(device, 1),
        }
    }
}

impl MetalHybridBuffers {
    /// Upload all model weights (f32 + raw Q4), allocate scratch, conv states, KV cache.
    ///
    /// `q4_map`: raw Q4 tensors from GGUF. When present, Q4 GEMV is used for decode
    /// instead of f32 GEMV, giving ~4x bandwidth reduction.
    pub fn from_hybrid_model(
        model: &HybridModel,
        max_seq: usize,
        backend: &MetalBackend,
    ) -> Self {
        Self::from_hybrid_model_with_q4(model, max_seq, backend, &HashMap::new())
    }

    /// Upload with optional raw Q4 data for Q4 GEMV decode.
    pub fn from_hybrid_model_with_q4(
        model: &HybridModel,
        max_seq: usize,
        backend: &MetalBackend,
        q4_map: &HashMap<String, RawQ4Tensor>,
    ) -> Self {
        let device = &backend.device;
        let config = &model.config;
        let num_layers = config.num_layers;

        // ── Layer kinds and index maps ──────────────────────────────
        let mut layer_kinds = Vec::with_capacity(num_layers);
        let mut conv_indices = Vec::with_capacity(num_layers);
        let mut kv_indices = Vec::with_capacity(num_layers);
        let mut conv_count = 0usize;
        let mut kv_count = 0usize;

        for i in 0..num_layers {
            let kind = config.layer_kind(i);
            layer_kinds.push(kind);
            match kind {
                LayerKind::LivConv => {
                    conv_indices.push(Some(conv_count));
                    kv_indices.push(None);
                    conv_count += 1;
                }
                LayerKind::Gqa => {
                    conv_indices.push(None);
                    kv_indices.push(Some(kv_count));
                    kv_count += 1;
                }
                LayerKind::DeltaNet => {
                    // DeltaNet layers are not supported by this GPU path
                    conv_indices.push(None);
                    kv_indices.push(None);
                }
            }
        }

        // ── Upload layer weights ────────────────────────────────────
        let layer_weights: Vec<MetalHybridLayerWeights> = model
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| match layer {
                HybridLayer::LivConv(l) => {
                    MetalHybridLayerWeights::LivConv(MetalLivConvWeights::from_layer(l, i, q4_map, device))
                }
                HybridLayer::Gqa(g) => {
                    MetalHybridLayerWeights::Gqa(MetalGqaWeights::from_layer(g, i, q4_map, device))
                }
                HybridLayer::DeltaNet(_) => {
                    panic!("DeltaNet layers not supported by Metal hybrid GPU path");
                }
            })
            .collect();

        // ── Conv rolling states ─────────────────────────────────────
        let inner = config.livconv_inner_size;
        let kernel_size = config.livconv_kernel_size;
        let conv_states: Vec<Buffer> = (0..conv_count)
            .map(|_| alloc_empty(device, inner * kernel_size))
            .collect();

        // ── KV cache layers ─────────────────────────────────────────
        let kv_dim = config.num_kv_heads * config.gqa_head_dim;
        let kv_layers: Vec<MetalHybridKVCacheLayer> = (0..kv_count)
            .map(|_| MetalHybridKVCacheLayer {
                k_cache: alloc_empty(device, max_seq * kv_dim),
                v_cache: alloc_empty(device, max_seq * kv_dim),
                kv_dim,
            })
            .collect();

        // ── RoPE tables ─────────────────────────────────────────────
        let rope_cos = upload(device, &model.rope_cos);
        let rope_sin = upload(device, &model.rope_sin);

        // ── Scratch buffers ─────────────────────────────────────────
        let scratch = HybridScratchBuffers::new(config, device);

        // ── Constant buffers ────────────────────────────────────────
        let consts = HybridConstantBuffers::new(config, device);

        // ── LM head + final norm on GPU ───────────────────────────
        let final_norm = upload(device, &model.final_norm_weight);
        let vocab = config.vocab_size;
        let h = config.hidden_size;
        let lm_head_data = model.lm_head_weight.as_deref().unwrap_or(&model.embed_tokens);
        let lm_head_f32 = upload_transposed(device, lm_head_data, vocab, h);
        let lm_head_q4 = q4_map.get("output.weight").map(|t| upload_q4_for_gemv(device, &t.data));
        let logits_buf = alloc_empty(device, vocab);
        let vocab_const = {
            let val = vocab as u32;
            device.new_buffer_with_data(
                &val as *const u32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            )
        };

        MetalHybridBuffers {
            layer_weights,
            scratch,
            conv_states,
            kv_layers,
            rope_cos,
            rope_sin,
            pos: 0,
            max_seq,
            consts,
            conv_indices,
            kv_indices,
            layer_kinds,
            final_norm,
            lm_head_f32,
            lm_head_q4,
            logits_buf,
            vocab_const,
        }
    }

    /// Copy CPU-side conv states and KV caches to GPU buffers after prefill.
    ///
    /// Must be called after CPU prefill and before GPU decode begins.
    pub fn populate_from_cpu_state(&mut self, model: &HybridModel, _device: &Device) {
        use crate::models::ssm::hybrid::model::LayerState;

        // We need to access the model's internal state. Since HybridModel
        // holds state via RefCell, we borrow it here.
        let state = model.state.borrow();
        self.pos = state.position;

        for (i, layer_state) in state.layer_states.iter().enumerate() {
            match layer_state {
                LayerState::Conv(cs) => {
                    if let Some(conv_idx) = self.conv_indices[i] {
                        let gpu_buf = &self.conv_states[conv_idx];
                        let copy_len = cs.conv_state.len();
                        unsafe {
                            let ptr = gpu_buf.contents() as *mut f32;
                            std::ptr::copy_nonoverlapping(
                                cs.conv_state.as_ptr(),
                                ptr,
                                copy_len,
                            );
                        }
                    }
                }
                LayerState::Kv(ks) => {
                    if let Some(kv_idx) = self.kv_indices[i] {
                        let gpu_kv = &self.kv_layers[kv_idx];
                        let copy_len = ks.len * ks.kv_dim;
                        unsafe {
                            let k_ptr = gpu_kv.k_cache.contents() as *mut f32;
                            std::ptr::copy_nonoverlapping(
                                ks.k_cache.as_ptr(),
                                k_ptr,
                                copy_len,
                            );
                            let v_ptr = gpu_kv.v_cache.contents() as *mut f32;
                            std::ptr::copy_nonoverlapping(
                                ks.v_cache.as_ptr(),
                                v_ptr,
                                copy_len,
                            );
                        }
                    }
                }
                LayerState::DeltaNet(_) => {
                    // DeltaNet state is not supported by this GPU path
                }
            }
        }
    }

    /// Reset all conv states to zero and reset position to 0.
    pub fn reset(&mut self) {
        for conv_buf in &self.conv_states {
            unsafe {
                let ptr = conv_buf.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, conv_buf.length() as usize);
            }
        }
        for kv in &self.kv_layers {
            unsafe {
                let k_ptr = kv.k_cache.contents() as *mut u8;
                std::ptr::write_bytes(k_ptr, 0, kv.k_cache.length() as usize);
                let v_ptr = kv.v_cache.contents() as *mut u8;
                std::ptr::write_bytes(v_ptr, 0, kv.v_cache.length() as usize);
            }
        }
        self.pos = 0;
    }
}
