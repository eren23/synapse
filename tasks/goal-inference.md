# Swarm Goal

Synapse Phase 3 — Modular LLM Inference Engine with INT8 Quantization

Build a production-grade, model-agnostic inference engine on top of the
existing Synapse framework. The engine uses a **component registry** pattern
where every architectural element (attention, normalization, FFN, positional
encoding, quantization) is a pluggable trait with config-driven instantiation.
Reference model: **Qwen3-0.6B** (28 layers, GQA, RoPE, RMSNorm, SwiGLU).
The architecture is designed so that LLaMA 3.2, Mistral, Gemma 3, and future
hybrid models (Qwen 3.5, Mamba-hybrids) can be added by implementing new
trait variants — not by rewriting the engine.

Includes SIMD-accelerated INT8 quantized matmul kernels in Zig, a KV-Cache
with pre-allocated memory management, safetensors and GGUF weight loading,
pretrained tokenizer loading, and a configurable token generation pipeline
with multiple sampling strategies.

**Builds on top of Synapse Phase 1 (tensor engine, autograd, training) and
Phase 2 (transformer stack, attention, LayerNorm, RoPE).** Inference-only —
no gradient computation, no backward passes.

Targeting ~15,000–20,000 new lines with comprehensive unit tests, benchmark
tests with hard pass/fail performance thresholds, and an end-to-end demo
that loads and runs Qwen3-0.6B from HuggingFace safetensors weights.

**CRITICAL RULE: Every task MUST include its own tests. No implementation
without tests. Every benchmark MUST have a hard pass/fail threshold. If a
benchmark does not meet its threshold, the task FAILS.**

---

## 0) Phase 3 Overview

### What Already Exists (DO NOT REWRITE)

Phase 3 builds on Phases 1 and 2. The following are already implemented:

- **Zig SIMD kernels**: matmul (tiled SGEMM), softmax (online stable),
  elementwise ops, reductions, activations (relu, sigmoid, tanh, gelu)
- **Zig allocators**: arena, pool, aligned, tracking
- **Zig FFI exports**: Full C ABI with opaque handles and status codes
- **Phase 2 Zig kernels**: fused attention, LayerNorm, RoPE
- **Rust autograd + nn**: Variable, Graph, backward, Module trait,
  Sequential, Linear, Conv2d, BatchNorm, Dropout, Embedding,
  MultiHeadAttention, TransformerEncoder/Decoder, LayerNorm, positional
  encodings
- **Rust optim + data + train**: full training pipeline
- **Rust graph**: IR, fusion passes, DCE, constant folding, scheduler
- **Checkpoint**: binary save/load (Synapse format)

### What's New in Phase 3

Phase 3 adds an **inference-only execution path** that bypasses autograd
entirely. No gradient tracking, no tape recording, no backward graph
construction. This enables:
- Lower memory usage (no saved activations for backward)
- Faster execution (no graph overhead)
- KV-Cache (impossible with standard autograd since it mutates state)

### Why a Component Registry?

Modern LLMs have converged on similar building blocks but with subtle
variations:

| Model | Attention | Norm | FFN | Position | Special |
|-------|-----------|------|-----|----------|---------|
| Qwen3 | GQA (16Q/8KV) | RMSNorm | SwiGLU | RoPE (θ=1M) | — |
| LLaMA 3.2 | GQA (32Q/8KV) | RMSNorm | SwiGLU | RoPE (θ=500K) | — |
| Mistral | GQA + sliding window | RMSNorm | SwiGLU | RoPE | Window=4096 |
| Gemma 3 | Sliding + global | RMSNorm | GeGLU | RoPE | Logit soft-cap |
| Qwen3.5 | Hybrid linear+full | RMSNorm | SwiGLU | mRoPE (partial) | Vision, MTP |
| Future | Linear RNN + full | varies | MoE | varies | State-space |

A hardcoded architecture serves one model. A component registry serves all
of them with the same engine. Each component is a Rust trait with a factory
that reads from `ModelConfig`. Adding a new model means:
1. Write its `model_config.json`
2. Implement any missing trait variants (if new architecture features)
3. Load weights → run

### Sample Usage (Target API)

```rust
use synapse_inference::prelude::*;

fn main() -> Result<()> {
    // Load model from HuggingFace safetensors
    let engine = InferenceEngine::from_pretrained(
        "./models/Qwen3-0.6B",
        InferenceConfig {
            quantization: Quantization::INT8,  // or F32, F16
            max_seq_len: 2048,
            device: Device::CPU,
        },
    )?;

    // Generate text
    let output = engine.generate(
        "Explain quantum computing in simple terms:",
        GenerationConfig {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            stop_tokens: vec![engine.eos_token_id()],
        },
    )?;

    println!("{}", output.text);
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.num_tokens, output.elapsed_secs, output.tokens_per_sec);
    Ok(())
}
```

### Sample Usage (Model Config — Qwen3-0.6B)

```json
{
  "model_type": "causal_lm",
  "architecture": {
    "num_layers": 28,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": true
  },
  "attention": {
    "variant": "gqa",
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "bias": false,
    "sliding_window": null
  },
  "normalization": {
    "variant": "rms_norm",
    "eps": 1e-6
  },
  "ffn": {
    "variant": "swiglu",
    "activation": "silu"
  },
  "position_encoding": {
    "variant": "rope",
    "theta": 1000000.0,
    "partial_rotary_factor": 1.0,
    "rope_scaling": null
  },
  "weight_format": "safetensors",
  "tokenizer_format": "bpe"
}
```

### Sample Config (LLaMA 3.2 1B — different model, same engine)

```json
{
  "model_type": "causal_lm",
  "architecture": {
    "num_layers": 16,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": true
  },
  "attention": {
    "variant": "gqa",
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 64,
    "bias": false,
    "sliding_window": null
  },
  "normalization": {
    "variant": "rms_norm",
    "eps": 1e-5
  },
  "ffn": {
    "variant": "swiglu",
    "activation": "silu"
  },
  "position_encoding": {
    "variant": "rope",
    "theta": 500000.0,
    "partial_rotary_factor": 1.0,
    "rope_scaling": null
  },
  "weight_format": "safetensors",
  "tokenizer_format": "bpe"
}
```

---

## 1) Architecture

```
Phase 3 additions (NEW)
─────────────────────────────────────────────────────────────
│                                                             │
│  synapse-inference (NEW CRATE)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  InferenceEngine                                     │    │
│  │    ├── ModelConfig (JSON → component assembly)       │    │
│  │    ├── WeightLoader (safetensors, GGUF)              │    │
│  │    ├── Tokenizer (BPE vocab, SentencePiece)          │    │
│  │    └── GenerationPipeline (sample → decode loop)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Component Registry (trait-based, config-driven)            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ AttentionVar  │ │ NormVariant  │ │ FFNVariant   │        │
│  │  ├─ MHA      │ │  ├─ RMSNorm  │ │  ├─ GELU     │        │
│  │  ├─ GQA  ◄───┤ │  ├─ LayerN.  │ │  ├─ SwiGLU◄──│        │
│  │  ├─ MQA      │ │  └─ (future) │ │  ├─ GeGLU    │        │
│  │  ├─ Sliding   │ └──────────────┘ │  └─ (future) │        │
│  │  └─ (future) │                    └──────────────┘        │
│  └──────────────┘                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ PosVariant   │ │ QuantVariant │ │ WeightFormat │        │
│  │  ├─ RoPE ◄───┤ │  ├─ F32     │ │  ├─ Safetens.◄│        │
│  │  ├─ Learned  │ │  ├─ F16     │ │  ├─ GGUF     │        │
│  │  ├─ Sinusoid │ │  ├─ INT8 ◄──│ │  └─ (future) │        │
│  │  └─ (future) │ │  └─ (future) │ └──────────────┘        │
│  └──────────────┘ └──────────────┘                          │
│                                                             │
│  KV-Cache (pre-allocated, per-layer, append-only)           │
│  ┌──────────────────────────────────────────────┐           │
│  │  Layer 0: K [max_seq, n_kv_heads, head_dim]  │           │
│  │           V [max_seq, n_kv_heads, head_dim]  │           │
│  │  Layer 1: K [...] V [...]                    │           │
│  │  ...                                         │           │
│  │  Layer N: K [...] V [...]                    │           │
│  │  current_seq_len: usize (append pointer)     │           │
│  └──────────────────────────────────────────────┘           │
│                                                             │
│  Zig kernels (NEW)                                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  ops/rmsnorm.zig    — SIMD RMSNorm           │           │
│  │  ops/silu.zig       — SiLU + SwiGLU fused    │           │
│  │  ops/quantize.zig   — INT8 quantize/deq.     │           │
│  │  ops/qmatmul.zig    — INT8 quantized GEMM    │           │
│  │  ops/kvcache.zig    — KV-Cache append/read    │           │
│  └──────────────────────────────────────────────┘           │
─────────────────────────────────────────────────────────────
```

---

## 2) New Files

```
synapse/
├── zig/
│   ├── src/ops/
│   │   ├── rmsnorm.zig             # RMSNorm: x * rsqrt(mean(x²) + eps) * gamma
│   │   ├── silu.zig                # SiLU (x * sigmoid(x)) + fused SwiGLU
│   │   ├── quantize.zig            # Per-channel INT8 quantize + dequantize
│   │   ├── qmatmul.zig             # INT8 quantized tiled GEMM
│   │   └── kvcache.zig             # Pre-allocated KV-Cache append/slice ops
│   │
│   └── tests/
│       ├── test_rmsnorm.zig
│       ├── test_silu.zig
│       ├── test_quantize.zig
│       ├── test_qmatmul.zig
│       ├── test_kvcache.zig
│       ├── bench_rmsnorm.zig        # MUST pass: SIMD >=4x vs scalar
│       ├── bench_silu.zig           # MUST pass: fused SwiGLU >=1.5x vs separate
│       ├── bench_qmatmul.zig        # MUST pass: INT8 >=2x vs f32 matmul
│       └── bench_kvcache.zig        # MUST pass: cached >=3x vs recompute
│
├── crates/
│   └── synapse-inference/           # NEW CRATE
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs               # Public API, prelude
│           │
│           ├── config/
│           │   ├── mod.rs
│           │   ├── model_config.rs  # ModelConfig struct (JSON deserialize)
│           │   ├── attention.rs     # AttentionConfig enum
│           │   ├── norm.rs          # NormConfig enum
│           │   ├── ffn.rs           # FFNConfig enum
│           │   ├── position.rs      # PositionConfig enum
│           │   └── quantization.rs  # QuantConfig enum
│           │
│           ├── registry/
│           │   ├── mod.rs
│           │   ├── attention.rs     # AttentionVariant trait + GQA, MHA, MQA, SlidingWindow
│           │   ├── norm.rs          # NormVariant trait + RMSNorm, LayerNorm
│           │   ├── ffn.rs           # FFNVariant trait + SwiGLU, GELU, GeGLU
│           │   ├── position.rs      # PositionVariant trait + RoPE, Learned, Sinusoidal
│           │   └── factory.rs       # Config → trait object factory functions
│           │
│           ├── model/
│           │   ├── mod.rs
│           │   ├── causal_lm.rs     # CausalLM: embedding → N×(attn+ffn+norm) → lm_head
│           │   ├── decoder_layer.rs # Single transformer decoder layer (generic)
│           │   └── builder.rs       # ModelConfig → assembled CausalLM
│           │
│           ├── kv_cache/
│           │   ├── mod.rs
│           │   ├── cache.rs         # KVCache: per-layer pre-allocated K/V buffers
│           │   └── strategy.rs      # CacheStrategy trait (+ future: paged, sliding window)
│           │
│           ├── quantization/
│           │   ├── mod.rs
│           │   ├── int8.rs          # Per-channel INT8 quantize/dequantize (wraps Zig)
│           │   ├── quantized_linear.rs # QuantizedLinear: INT8 weights + f32 compute
│           │   └── calibration.rs   # Post-training quantization calibration
│           │
│           ├── weight_loading/
│           │   ├── mod.rs
│           │   ├── safetensors.rs   # Safetensors parser (mmap-based, zero-copy where possible)
│           │   ├── gguf.rs          # GGUF format parser
│           │   ├── weight_map.rs    # Maps HF layer names → Synapse module paths
│           │   └── converter.rs     # Dtype conversion, transpose, reshaping on load
│           │
│           ├── tokenizer/
│           │   ├── mod.rs
│           │   ├── bpe.rs           # Load BPE from vocab.json + merges.txt
│           │   ├── sentencepiece.rs # Load SentencePiece .model files
│           │   ├── vocabulary.rs    # ID ↔ token mapping, special tokens
│           │   └── pre_tokenizer.rs # Whitespace, byte-level pre-tokenization
│           │
│           ├── generation/
│           │   ├── mod.rs
│           │   ├── pipeline.rs      # GenerationPipeline: tokenize → prefill → decode → detokenize
│           │   ├── sampler.rs       # Greedy, TopK, TopP, Temperature, RepetitionPenalty
│           │   ├── stopping.rs      # StopCondition: EOS, max_length, stop_sequences
│           │   └── output.rs        # GenerationOutput: text, tokens, timing stats
│           │
│           └── engine.rs            # InferenceEngine: top-level orchestrator
│
├── tests/
│   ├── integration/
│   │   ├── inference_e2e.rs         # Qwen3-0.6B load + generate
│   │   ├── quantization_accuracy.rs # INT8 vs f32 output comparison
│   │   ├── kvcache_correctness.rs   # Cached vs uncached output identical
│   │   └── config_driven_assembly.rs # Load different configs, verify assembly
│   │
│   └── benchmarks/
│       ├── inference_throughput.rs   # Tokens/sec on Qwen3-0.6B
│       ├── prefill_throughput.rs     # Prompt processing tokens/sec
│       ├── quantization_speedup.rs   # INT8 vs f32 speed comparison
│       └── memory_usage.rs           # Peak memory during inference
│
├── examples/
│   ├── qwen3_chat.rs                # Interactive chat with Qwen3-0.6B
│   └── model_benchmark.rs           # Benchmark any model via config
│
└── configs/                          # Model config files
    ├── qwen3_0.6b.json
    ├── llama3.2_1b.json              # Ready for when weights are available
    └── mistral_7b.json               # Ready for when weights are available
```

---

## 3) New FFI Functions

Extends `synapse.h` with inference-specific exports.

```c
// --- RMSNorm ---
syn_status_t syn_rmsnorm_forward(
    syn_tensor_t* out,              // same shape as input
    const syn_tensor_t* input,      // [*, hidden_size]
    const syn_tensor_t* weight,     // [hidden_size] (gamma)
    float eps                       // typically 1e-6
);

// --- SiLU + SwiGLU ---
syn_status_t syn_silu(
    syn_tensor_t* out,              // same shape as input
    const syn_tensor_t* input
);

syn_status_t syn_swiglu(
    syn_tensor_t* out,              // [*, intermediate_size]
    const syn_tensor_t* gate,       // [*, intermediate_size]
    const syn_tensor_t* up          // [*, intermediate_size]
    // Computes: silu(gate) * up
);

// --- INT8 Quantization ---
syn_status_t syn_quantize_per_channel_int8(
    int8_t* out_data,               // quantized output
    float* out_scales,              // per-channel scale factors
    const syn_tensor_t* input,      // f32 input tensor
    size_t channel_axis             // axis for per-channel quantization
);

syn_status_t syn_dequantize_per_channel_int8(
    syn_tensor_t* out,              // f32 output
    const int8_t* data,             // INT8 input
    const float* scales,            // per-channel scales
    size_t num_elements,
    size_t channel_size
);

syn_status_t syn_qgemm_int8(
    float* out,                     // f32 output [M, N]
    const int8_t* a,                // INT8 matrix [M, K]
    const int8_t* b,                // INT8 matrix [K, N]
    const float* scale_a,           // per-row scales for A [M]
    const float* scale_b,           // per-column scales for B [N]
    size_t M, size_t N, size_t K
);

// --- KV-Cache ---
syn_status_t syn_kvcache_create(
    syn_tensor_t** key_cache,       // [max_seq, n_kv_heads, head_dim]
    syn_tensor_t** val_cache,       // [max_seq, n_kv_heads, head_dim]
    size_t max_seq_len,
    size_t n_kv_heads,
    size_t head_dim
);

syn_status_t syn_kvcache_append(
    syn_tensor_t* key_cache,
    syn_tensor_t* val_cache,
    const syn_tensor_t* new_keys,   // [1, n_kv_heads, head_dim] (single token)
    const syn_tensor_t* new_vals,   // [1, n_kv_heads, head_dim]
    size_t position                  // current sequence position
);

syn_status_t syn_kvcache_slice(
    syn_tensor_t* key_out,          // [seq_len, n_kv_heads, head_dim]
    syn_tensor_t* val_out,
    const syn_tensor_t* key_cache,
    const syn_tensor_t* val_cache,
    size_t start,
    size_t end
);
```

---

## 4) Component Registry — Trait Definitions

### AttentionVariant

```rust
/// Trait for all attention implementations.
/// The engine calls this generically — the variant handles head geometry,
/// masking, and KV-cache interaction internally.
pub trait AttentionVariant: Send + Sync {
    fn forward(
        &self,
        hidden_states: &Tensor,     // [batch, seq, hidden]
        kv_cache: Option<&mut KVCacheLayer>,
        position: usize,
        causal: bool,
    ) -> Result<Tensor>;

    fn num_params(&self) -> usize;
    fn name(&self) -> &str;
}

/// GQA: num_kv_heads < num_heads. KV heads are repeated to match Q heads.
pub struct GQAAttention {
    w_q: Tensor,        // [hidden, num_heads * head_dim]
    w_k: Tensor,        // [hidden, num_kv_heads * head_dim]
    w_v: Tensor,        // [hidden, num_kv_heads * head_dim]
    w_o: Tensor,        // [num_heads * head_dim, hidden]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Box<dyn PositionVariant>,
}

/// MHA: num_kv_heads == num_heads (standard).
/// MQA: num_kv_heads == 1 (single KV head shared by all Q heads).
/// Both are special cases of GQA — implemented via GQAAttention with
/// appropriate head counts. No separate structs needed.

/// SlidingWindowAttention: GQA but only attends to last W tokens.
pub struct SlidingWindowAttention {
    inner: GQAAttention,
    window_size: usize,
}

// FUTURE EXTENSION POINTS (not implemented in Phase 3):
// - LinearAttention: for Qwen 3.5 / Mamba-hybrid models
//   Uses linear RNN to maintain compressed state instead of full KV-cache.
//   Would implement AttentionVariant with different internal state.
// - HybridAttention: alternates Linear and Full attention layers
//   (e.g., 3:1 ratio). The DecoderLayer would hold Vec<Box<dyn AttentionVariant>>
//   with per-layer variant selection from config.
```

### NormVariant

```rust
pub trait NormVariant: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
}

/// RMSNorm: x * rsqrt(mean(x²) + eps) * gamma
/// Used by Qwen3, LLaMA, Mistral. No mean subtraction (unlike LayerNorm).
pub struct RMSNorm {
    weight: Tensor,     // gamma [hidden_size]
    eps: f32,
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
/// Used by GPT-2, BERT, Phase 2 transformers.
pub struct LayerNormInfer {
    weight: Tensor,     // gamma
    bias: Tensor,       // beta
    eps: f32,
}

// FUTURE: GroupNorm, DeepNorm (scaling factor for deep transformers)
```

### FFNVariant

```rust
pub trait FFNVariant: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn num_params(&self) -> usize;
    fn name(&self) -> &str;
}

/// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
/// Used by Qwen3, LLaMA, Mistral. Two parallel projections with gating.
pub struct SwiGLUFFN {
    gate_proj: Tensor,  // [hidden, intermediate]
    up_proj: Tensor,    // [hidden, intermediate]
    down_proj: Tensor,  // [intermediate, hidden]
}

/// Standard FFN: down_proj(activation(up_proj(x)))
/// Used by GPT-2 (GELU), Phase 2 transformers.
pub struct StandardFFN {
    up_proj: Tensor,    // [hidden, intermediate]
    down_proj: Tensor,  // [intermediate, hidden]
    activation: ActivationType,
}

/// GeGLU: down_proj(gelu(gate_proj(x)) * up_proj(x))
/// Used by Gemma 3.
pub struct GeGLUFFN {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

// FUTURE: MoEFFN — Mixture of Experts with top-k router.
// Would implement FFNVariant, routing tokens to N expert FFNs.
// Needed for Qwen3.5-35B-A3B, Mixtral, etc.
```

### PositionVariant

```rust
pub trait PositionVariant: Send + Sync {
    /// Apply positional information to Q and K tensors.
    /// Some variants (RoPE) rotate Q/K directly.
    /// Others (learned) add to hidden states before projection.
    fn apply(&self, q: &Tensor, k: &Tensor, position: usize) -> Result<(Tensor, Tensor)>;
    fn name(&self) -> &str;
}

/// RoPE: Rotary Positional Embedding.
/// Rotates pairs of dimensions by position-dependent angles.
pub struct RoPEPosition {
    cos_cache: Tensor,  // [max_seq, head_dim/2]
    sin_cache: Tensor,  // [max_seq, head_dim/2]
    theta: f64,
    partial_rotary_factor: f64,  // 1.0 for full, 0.25 for partial (Qwen3.5)
}

/// Learned: adds learned position embedding to hidden states.
pub struct LearnedPosition {
    embeddings: Tensor, // [max_seq, hidden_size]
}

// FUTURE:
// - mRoPE: Multi-dimensional RoPE with sections (Qwen 3.5 vision)
// - ALiBi: Attention with Linear Biases (no position encoding,
//   adds bias to attention scores based on distance)
// - NTK-aware RoPE scaling: for context length extrapolation
```

### QuantVariant

```rust
pub enum Quantization {
    F32,                // No quantization — full precision
    F16,                // Half precision (if hardware supports)
    INT8 {              // 8-bit integer quantization
        calibration: CalibrationMethod,
    },
    // FUTURE: INT4, GPTQ, AWQ, GGML quantization formats
}

pub enum CalibrationMethod {
    MinMax,             // Simple min/max per-channel scaling
    Percentile(f32),    // Clip outliers at given percentile
    // FUTURE: MSE-optimal, entropy-based
}
```

---

## 5) Generic Decoder Layer

The core insight: every modern causal LM is `N × DecoderLayer` where each
layer is `norm → attention → residual → norm → ffn → residual`. The only
differences are which norm, which attention, and which FFN.

```rust
/// A single decoder layer assembled from registry components.
/// This struct is the SAME for Qwen3, LLaMA, Mistral, etc.
pub struct DecoderLayer {
    attn_norm: Box<dyn NormVariant>,
    attention: Box<dyn AttentionVariant>,
    ffn_norm: Box<dyn NormVariant>,
    ffn: Box<dyn FFNVariant>,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        hidden: &Tensor,
        kv_cache: Option<&mut KVCacheLayer>,
        position: usize,
    ) -> Result<Tensor> {
        // Pre-norm architecture (used by all modern LLMs)
        let normed = self.attn_norm.forward(hidden)?;
        let attn_out = self.attention.forward(&normed, kv_cache, position, true)?;
        let hidden = hidden.add(&attn_out)?;  // residual

        let normed = self.ffn_norm.forward(&hidden)?;
        let ffn_out = self.ffn.forward(&normed)?;
        let hidden = hidden.add(&ffn_out)?;   // residual

        Ok(hidden)
    }
}

/// Complete causal language model.
pub struct CausalLM {
    embedding: Tensor,              // [vocab_size, hidden_size]
    layers: Vec<DecoderLayer>,
    final_norm: Box<dyn NormVariant>,
    lm_head: Option<Tensor>,       // None if tie_word_embeddings=true
    config: ModelConfig,
}

impl CausalLM {
    pub fn forward(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let mut hidden = self.embed(token_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, Some(kv_cache.layer_mut(i)), pos)?;
        }
        hidden = self.final_norm.forward(&hidden)?;
        self.lm_head_forward(&hidden) // logits [seq, vocab]
    }
}
```

---

## 6) Optimization Targets

**These are HARD pass/fail thresholds. If a benchmark does not meet its
target, the task FAILS.**

| # | Module | Metric | Threshold | How to Measure |
|---|--------|--------|-----------|----------------|
| 1 | `ops/rmsnorm.zig` | SIMD RMSNorm vs scalar | **>=4x** on [64, 1024] | `bench_rmsnorm.zig`: 200 iterations |
| 2 | `ops/silu.zig` | Fused SwiGLU vs separate silu+mul | **>=1.5x** on [64, 3072] | `bench_silu.zig`: 200 iterations |
| 3 | `ops/qmatmul.zig` | INT8 GEMM vs f32 GEMM | **>=2x** on 512×512 | `bench_qmatmul.zig`: 100 iterations |
| 4 | `ops/qmatmul.zig` | INT8 quantization error | **<=1% relative** vs f32 output | Same benchmark, verify accuracy |
| 5 | `ops/kvcache.zig` | Cached generation vs recompute | **>=3x** at seq_len=256 | `bench_kvcache.zig`: 100 tokens |
| 6 | `synapse-inference` | Prefill throughput | **>=500 tokens/sec** on Qwen3-0.6B (f32) | `prefill_throughput.rs` |
| 7 | `synapse-inference` | Decode throughput | **>=20 tokens/sec** on Qwen3-0.6B (f32) | `inference_throughput.rs` |
| 8 | `synapse-inference` | INT8 decode throughput | **>=35 tokens/sec** on Qwen3-0.6B (INT8) | Same benchmark, INT8 mode |
| 9 | `synapse-inference` | Peak memory Qwen3-0.6B f32 | **<=3 GB** | `memory_usage.rs` |
| 10 | `synapse-inference` | Peak memory Qwen3-0.6B INT8 | **<=1.5 GB** | Same benchmark, INT8 |
| 11 | `synapse-inference` | Config assembly | Load + assemble model in **<=2 sec** (no weights) | `config_driven_assembly.rs` |

Use `cfg!(debug_assertions)` for debug-mode thresholds at ~5x lower values,
following the pattern established in Phase 1.

### Correctness Thresholds (non-negotiable)

| Module | Requirement |
|--------|-------------|
| RMSNorm | Max relative error **<= 1e-5** vs scalar reference |
| SiLU/SwiGLU | Max relative error **<= 1e-5** vs scalar reference |
| INT8 quantize/dequantize | Round-trip error **<= 0.5%** of original range per channel |
| INT8 GEMM | Max relative error **<= 1%** vs f32 GEMM for matrix sizes up to 1024 |
| KV-Cache | Cached output **bit-exact** with uncached full recompute |
| Safetensors loader | Loaded weights **bit-exact** with reference (Python torch.load) |
| GGUF loader | Loaded weights match dequantized reference within **1e-4** |
| Weight mapping | All model weights loaded — **zero missing, zero unexpected** keys |
| Tokenizer | Encode/decode roundtrip **lossless** for in-vocabulary text |
| Generation | Greedy output **deterministic** (same input → same output, always) |
| Component registry | Config → model assembly produces correct parameter count |

---

## 7) Task Decomposition — 14 Tasks

**CRITICAL RULES FOR EVERY TASK:**
1. Every task MUST write tests alongside implementation. No code without tests.
2. Every benchmark task MUST include both naive baseline AND optimized implementation.
3. Every task MUST list its pass/fail criteria. The judge uses these to accept/reject.
4. Dependencies must be respected. A task cannot start until its dependencies are complete.

### Dependency Graph

```
WAVE 1 (fully parallel — Zig kernels, no inter-dependencies):
  Task 1: Zig RMSNorm kernel
  Task 2: Zig SiLU + fused SwiGLU kernel
  Task 3: Zig INT8 quantization + quantized GEMM
  Task 4: Zig KV-Cache memory management

WAVE 2 (FFI + Rust foundation):
  Task 5: Zig FFI exports + Rust FFI bindings ────── depends: Tasks 1-4
  Task 6: ModelConfig + component registry ─────────  depends: none (pure Rust types)

WAVE 3 (Rust inference components — many parallel):
  Task 7: NormVariant + FFNVariant implementations ── depends: Tasks 5, 6
  Task 8: AttentionVariant + GQA implementation ───── depends: Tasks 5, 6
  Task 9: KV-Cache Rust module ────────────────────── depends: Task 5
  Task 10: Safetensors + GGUF weight loaders ──────── depends: Task 6
  Task 11: Pretrained tokenizer loader ────────────── depends: none (pure Rust)

WAVE 4 (assembly + generation):
  Task 12: CausalLM model builder ─────────────────── depends: Tasks 7, 8, 9, 10
  Task 13: Generation pipeline + sampling ─────────── depends: Tasks 11, 12
  Task 14: INT8 quantization + E2E benchmarks ─────── depends: Task 13
```

---

### Task 1: Zig RMSNorm Kernel

**Implement:**
- `zig/src/ops/rmsnorm.zig`: SIMD-vectorized RMS normalization.
  Formula: `output = x * rsqrt(mean(x²) + eps) * gamma`
  No mean subtraction (unlike LayerNorm) — simpler and faster.
  Uses SIMD for the squared-sum reduction and the element-wise multiply.
  **MUST include scalar reference implementation for benchmark comparison.**

**Tests (mandatory):**
- `zig/tests/test_rmsnorm.zig`: Correctness vs scalar for shapes:
  [64, 256], [32, 1024], [1, 4096]. Output norm approximately 1.0.
  Special cases: all-zeros (should not crash), constant input, large values.
  Compare to LayerNorm output to verify they differ (no mean subtraction).
- `zig/tests/bench_rmsnorm.zig`: SIMD vs scalar on [64, 1024], 200 iters.

**Pass/fail:**
- All correctness tests pass (max relative error <= 1e-5).
- **SIMD >=4x throughput** vs scalar.
- No inf/nan for inputs in [-1000, 1000].
- Differs from LayerNorm output (verifies no accidental mean subtraction).

**Dependencies:** None.

---

### Task 2: Zig SiLU + Fused SwiGLU Kernel

**Implement:**
- `zig/src/ops/silu.zig`:
  - `silu(x)`: `x * sigmoid(x)` — SIMD vectorized. Reuses existing
    sigmoid from vec_ops but fused into single pass.
  - `swiglu(gate, up)`: `silu(gate) * up` — fused kernel that avoids
    intermediate allocation. Single pass over both inputs.
  **MUST include separate (non-fused) implementation for benchmark.**

**Tests (mandatory):**
- `zig/tests/test_silu.zig`: SiLU correctness vs `x / (1 + exp(-x))`
  reference. SwiGLU correctness vs `silu(gate) * up` computed separately.
  Shapes: [64, 3072], [1, 1024]. Edge cases: x=0 (SiLU(0)=0), large |x|.
- `zig/tests/bench_silu.zig`: Fused SwiGLU vs separate silu-then-mul
  on [64, 3072], 200 iterations.

**Pass/fail:**
- SiLU max relative error <= 1e-5 vs reference.
- **Fused SwiGLU >=1.5x** vs separate computation.
- SiLU(0) == 0 exactly.
- No inf/nan for inputs in [-100, 100].

**Dependencies:** None.

---

### Task 3: Zig INT8 Quantization + Quantized GEMM

**Implement:**
- `zig/src/ops/quantize.zig`:
  - `quantize_per_channel_int8`: For each channel (row or column), compute
    scale = max(|x|) / 127, then round(x / scale) → int8. Store scales.
  - `dequantize_per_channel_int8`: int8 * scale → f32.
  - Round-trip must preserve values within quantization error.

- `zig/src/ops/qmatmul.zig`: INT8 tiled GEMM.
  - Inputs: A [M,K] int8 with scales_a [M], B [K,N] int8 with scales_b [N]
  - Computation: accumulate int8*int8 → int32, then scale to f32 at tile boundary
  - Tiled with 8x8 micro-kernel using SIMD integer multiply-accumulate
  - **MUST include naive int8 triple-loop AND f32 GEMM for comparison**

**Tests (mandatory):**
- `zig/tests/test_quantize.zig`: Round-trip error <= 0.5% of range per
  channel. Symmetric quantization (zero-centered). Handles all-zeros channel.
- `zig/tests/test_qmatmul.zig`: INT8 GEMM correctness vs f32 GEMM for
  sizes: 8x8, 64x64, 128x512, 512x512. Max relative error <= 1%.
- `zig/tests/bench_qmatmul.zig`: INT8 vs f32 GEMM on 512x512, 100 iters.
  Also benchmark quantize + GEMM + dequantize end-to-end.

**Pass/fail:**
- Round-trip quantization error <= 0.5% per channel.
- **INT8 GEMM >=2x throughput** vs f32 GEMM on 512x512.
- INT8 GEMM output within 1% relative error of f32 GEMM.
- End-to-end (quantize+gemm+dequantize) still >=1.5x vs pure f32.

**Dependencies:** None.

---

### Task 4: Zig KV-Cache Memory Management

**Implement:**
- `zig/src/ops/kvcache.zig`:
  - Pre-allocated contiguous buffer: K [max_seq, n_kv_heads, head_dim],
    V [max_seq, n_kv_heads, head_dim]. Allocated once at engine init.
  - `kvcache_append`: Copy new K/V for single token into position `pos`.
    Must be O(1) — just a memcpy into the right offset.
  - `kvcache_slice`: Return view of K/V from position 0..seq_len.
    Zero-copy pointer + length return.
  - `kvcache_reset`: Reset position counter to 0 (no deallocation).
  - Per-layer: the Zig side manages N independent caches (one per layer).

**Tests (mandatory):**
- `zig/tests/test_kvcache.zig`: Create cache, append tokens one by one,
  verify slice returns correct accumulated K/V. Reset and reuse.
  Verify pre-allocated memory is not re-allocated on append.
- `zig/tests/bench_kvcache.zig`: Generate 256 tokens with KV-cache
  (incremental attention on cached K/V) vs without (full recompute at
  each step). Measure total wall time.

**Pass/fail:**
- Appended K/V values correctly readable via slice.
- **Cached generation >=3x faster** than full recompute at seq_len=256.
- No memory allocation after initial create (verified by tracking allocator).
- Reset + reuse produces same results as fresh cache.

**Dependencies:** None.

---

### Task 5: Zig FFI Exports + Rust FFI Bindings

**Implement:**
- Extend `zig/src/ffi/exports.zig` with all new functions:
  `syn_rmsnorm_forward`, `syn_silu`, `syn_swiglu`,
  `syn_quantize_per_channel_int8`, `syn_dequantize_per_channel_int8`,
  `syn_qgemm_int8`, `syn_kvcache_create`, `syn_kvcache_append`,
  `syn_kvcache_slice`, `syn_kvcache_reset`.
- Update `synapse.h` with new declarations.
- Extend `crates/synapse-sys/src/lib.rs` with new extern declarations.
- Add safe Rust wrappers in synapse-core for each new operation.

**Tests (mandatory):**
- FFI round-trip: create Rust tensors → call each new FFI function →
  verify results match direct Zig calls.
- Error propagation: null pointers, shape mismatches → proper error codes.
- No panics cross the boundary (test with invalid inputs).

**Pass/fail:**
- All FFI functions callable from Rust.
- **No panics** across FFI boundary.
- **No memory leaks** in 10K create/call/destroy cycles.
- Error codes correctly map to `Result::Err` in Rust.

**Dependencies:** Tasks 1, 2, 3, 4.

---

### Task 6: ModelConfig + Component Registry Types

**Implement:**
- `crates/synapse-inference/src/config/model_config.rs`: `ModelConfig`
  struct that deserializes from JSON. Contains nested configs for
  architecture, attention, norm, FFN, position, quantization.
  Uses `serde` + `serde_json`.
- `crates/synapse-inference/src/config/attention.rs`: `AttentionConfig`
  enum (GQA, MHA, MQA, SlidingWindow) with parameters.
- `crates/synapse-inference/src/config/norm.rs`: `NormConfig` enum.
- `crates/synapse-inference/src/config/ffn.rs`: `FFNConfig` enum.
- `crates/synapse-inference/src/config/position.rs`: `PositionConfig` enum.
- `crates/synapse-inference/src/config/quantization.rs`: `QuantConfig` enum.
- `crates/synapse-inference/src/registry/`: Trait definitions for
  `AttentionVariant`, `NormVariant`, `FFNVariant`, `PositionVariant`.
- `crates/synapse-inference/src/registry/factory.rs`: Config → trait
  object factory. Given a `NormConfig::RMSNorm { eps: 1e-6 }`, returns
  `Box<dyn NormVariant>`.

**Tests (mandatory):**
- Deserialize Qwen3-0.6B config from JSON. Verify all fields parsed.
- Deserialize LLaMA config. Verify different values parsed correctly.
- Factory creates correct variant for each config enum value.
- Unknown config values produce clear errors (not panics).
- Round-trip: serialize → deserialize → compare.

**Pass/fail:**
- Qwen3 config parses correctly.
- Factory returns correct trait objects for all implemented variants.
- Serde errors are descriptive (include field name and expected type).

**Dependencies:** None (pure Rust types, no Zig calls).

---

### Task 7: NormVariant + FFNVariant Implementations

**Implement:**
- `crates/synapse-inference/src/registry/norm.rs`:
  - `RMSNorm` implementing `NormVariant`. Calls Zig `syn_rmsnorm_forward`.
  - `LayerNormInfer` implementing `NormVariant`. Calls existing Zig LayerNorm.
- `crates/synapse-inference/src/registry/ffn.rs`:
  - `SwiGLUFFN` implementing `FFNVariant`. Three weight matrices
    (gate_proj, up_proj, down_proj). Forward: `down(swiglu(gate(x), up(x)))`.
  - `StandardFFN` implementing `FFNVariant`. Two weight matrices + activation.
  - `GeGLUFFN` implementing `FFNVariant`. Like SwiGLU but with GELU gating.

**Tests (mandatory):**
- RMSNorm: output shape matches input. Output approximately unit norm.
  Weights correctly scale output.
- SwiGLU: output shape [batch, seq, hidden]. Intermediate shape
  [batch, seq, intermediate]. Parameter count = 3 * hidden * intermediate.
- GeGLU: same structure as SwiGLU but different activation verified.
- All variants work through the `NormVariant`/`FFNVariant` trait interface.

**Pass/fail:**
- Output shapes correct for all variants.
- Parameter counts match expected formulas.
- Trait dispatch works (can call `.forward()` on `Box<dyn NormVariant>`).
- RMSNorm and LayerNorm produce different outputs for same input.

**Dependencies:** Tasks 5 (FFI), 6 (registry types).

---

### Task 8: AttentionVariant + GQA Implementation

**Implement:**
- `crates/synapse-inference/src/registry/attention.rs`:
  - `GQAAttention` implementing `AttentionVariant`.
    - Q projection: hidden → num_heads * head_dim
    - K projection: hidden → num_kv_heads * head_dim
    - V projection: hidden → num_kv_heads * head_dim
    - O projection: num_heads * head_dim → hidden
    - GQA head repeat: expand KV heads to match Q heads via repeat_interleave
    - Applies RoPE to Q and K before attention
    - Supports KV-cache: on first call (prefill), computes full attention.
      On subsequent calls (decode), computes attention for single token
      against all cached K/V.
    - Causal masking for prefill.
  - `SlidingWindowAttention` implementing `AttentionVariant`.
    Wraps GQAAttention but masks attention beyond window_size positions.
  - MHA and MQA are just GQA with different head counts — document this.

**Tests (mandatory):**
- GQA output shape [batch, seq, hidden].
- Verify GQA with num_kv_heads=num_heads produces same result as MHA.
- Verify GQA with num_kv_heads=1 produces same result as MQA.
- KV-cache: run prefill(10 tokens), then decode(1 token), verify output
  matches full forward(11 tokens).
- Sliding window: attention weights are zero beyond window.
- Parameter count = 4 linear projections with correct shapes.

**Pass/fail:**
- GQA/MHA/MQA produce correct output shapes.
- **KV-cache output matches full recompute** (bit-exact for f32).
- Sliding window correctly restricts attention span.
- RoPE correctly applied (verify positions affect output).

**Dependencies:** Tasks 5 (FFI), 6 (registry types).

---

### Task 9: KV-Cache Rust Module

**Implement:**
- `crates/synapse-inference/src/kv_cache/cache.rs`:
  - `KVCache`: manages N layers of `KVCacheLayer`.
  - `KVCacheLayer`: wraps Zig-allocated pre-allocated K/V buffers.
  - Methods: `append(layer, new_k, new_v)`, `get(layer, 0..seq_len)`,
    `reset()`, `current_len()`.
  - Pre-allocates for max_seq_len at init. No re-allocation during use.
- `crates/synapse-inference/src/kv_cache/strategy.rs`:
  - `CacheStrategy` trait for future extensibility.
  - `PreAllocatedStrategy`: current approach.
  - Document future: `PagedStrategy` (vLLM-style paged attention),
    `SlidingWindowStrategy` (only cache last W tokens).

**Tests (mandatory):**
- Create cache for 28 layers, max_seq=2048, 8 KV heads, head_dim=128.
- Append single tokens sequentially, verify retrieval.
- Full capacity: fill to max_seq_len, verify correctness.
- Reset and reuse: verify clean state.
- Memory: total allocation = 2 * num_layers * max_seq * n_kv * head_dim * sizeof(f32).

**Pass/fail:**
- Append + retrieve produces correct values.
- No allocation after init (verified).
- Memory usage matches expected formula.
- Reset produces clean state.

**Dependencies:** Task 5 (Zig KV-cache FFI).

---

### Task 10: Safetensors + GGUF Weight Loaders

**Implement:**
- `crates/synapse-inference/src/weight_loading/safetensors.rs`:
  - Parse safetensors format: JSON header (tensor names, shapes, dtypes,
    offsets) + raw binary data.
  - Memory-map the file for zero-copy access to large weight files.
  - Support dtypes: f32, f16, bf16 (convert f16/bf16 → f32 on load).
  - Return `HashMap<String, Tensor>` of loaded weights.

- `crates/synapse-inference/src/weight_loading/gguf.rs`:
  - Parse GGUF format: metadata key-value pairs + tensor info + binary data.
  - Support quantization types: F32, F16, Q8_0 (dequantize to f32 on load).
  - Return `HashMap<String, Tensor>` of loaded weights.

- `crates/synapse-inference/src/weight_loading/weight_map.rs`:
  - `WeightMapper`: maps HuggingFace layer names to Synapse module paths.
  - Qwen3 mapping: `model.layers.{i}.self_attn.q_proj.weight` →
    `layers[i].attention.w_q`, etc.
  - Configurable via pattern rules (regex-based).
  - Validation: after loading, verify zero missing keys and zero unexpected keys.

- `crates/synapse-inference/src/weight_loading/converter.rs`:
  - Dtype conversion: bf16 → f32, f16 → f32.
  - Shape operations: transpose weight matrices if layout differs.

**Tests (mandatory):**
- Create a small safetensors file (2 tensors), load it, verify values.
- Create a small GGUF file (2 tensors), load it, verify values.
- Weight mapper: apply Qwen3 mapping, verify all keys resolve.
- Weight mapper: detect missing key → error. Detect extra key → warning.
- bf16 → f32 conversion: verify values within 1e-3 of original f32.
- Benchmark: load time for a ~1.2GB safetensors file (simulated).

**Pass/fail:**
- Safetensors: loaded weights **bit-exact** with Python reference.
- GGUF: loaded weights within 1e-4 of reference.
- Weight mapping: **zero missing, zero unexpected** keys for Qwen3 mapping.
- bf16/f16 conversion correct.
- Load time <= 5 seconds for 1.2GB file (mmap, not full read).

**Dependencies:** Task 6 (ModelConfig for weight mapping rules).

---

### Task 11: Pretrained Tokenizer Loader

**Implement:**
- `crates/synapse-inference/src/tokenizer/bpe.rs`:
  - Load BPE tokenizer from HuggingFace format:
    - `tokenizer.json` (HF tokenizers format — contains vocab, merges, special tokens)
    - OR `vocab.json` + `merges.txt` (legacy GPT-2 format)
  - Encode: text → BPE merge algorithm → token IDs.
  - Decode: token IDs → text (handle byte-level tokens, special tokens).
  - Special tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` (Qwen3 chat).

- `crates/synapse-inference/src/tokenizer/sentencepiece.rs`:
  - Load SentencePiece `.model` protobuf files.
  - Unigram tokenization.
  - For LLaMA/Mistral compatibility.

- `crates/synapse-inference/src/tokenizer/vocabulary.rs`:
  - `Vocabulary`: bidirectional `token ↔ id` mapping.
  - Special token registry (PAD, BOS, EOS, UNK + model-specific).

- `crates/synapse-inference/src/tokenizer/pre_tokenizer.rs`:
  - Byte-level BPE pre-tokenization (GPT-2/Qwen style).
  - Whitespace splitting with preservation.

**Tests (mandatory):**
- BPE: encode("Hello, world!") → known token IDs (compare with HF tokenizers).
- Decode: round-trip for ASCII, Unicode, emoji, multi-byte UTF-8.
- Special tokens: EOS token is not split by BPE.
- Vocabulary: size matches expected (151936 for Qwen3).
- SentencePiece: basic encode/decode (if .model file available).
- Edge cases: empty string, single character, very long input (>4K tokens).

**Pass/fail:**
- **Encode/decode roundtrip lossless** for in-vocabulary text.
- Special tokens handled correctly (not merged by BPE).
- Token IDs match HuggingFace tokenizers output for 10 test strings.
- Unicode handling correct (no mojibake).

**Dependencies:** None (pure Rust, no Zig dependency).

---

### Task 12: CausalLM Model Builder

**Implement:**
- `crates/synapse-inference/src/model/builder.rs`:
  - `ModelBuilder::from_config(config: &ModelConfig) -> CausalLM`
  - Uses factory functions from registry to create components.
  - Assembles: embedding + N × DecoderLayer + final_norm + lm_head.
  - Handle `tie_word_embeddings`: lm_head shares embedding weight matrix.

- `crates/synapse-inference/src/model/decoder_layer.rs`:
  - `DecoderLayer` struct with trait-object components.
  - Pre-norm forward pass (norm → attention → residual → norm → ffn → residual).

- `crates/synapse-inference/src/model/causal_lm.rs`:
  - `CausalLM` struct: embedding, layers, final_norm, lm_head.
  - `forward(token_ids, kv_cache) → logits`
  - `load_weights(weights: HashMap<String, Tensor>, mapper: &WeightMapper)`

- `crates/synapse-inference/src/engine.rs`:
  - `InferenceEngine::from_pretrained(path, config) → Self`
  - Orchestrates: read model_config → build model → load weights →
    init KV-cache → init tokenizer.

**Tests (mandatory):**
- Build Qwen3-0.6B model from config (no weights). Verify:
  - 28 layers created
  - Each layer has RMSNorm, GQA(16Q/8KV), SwiGLU
  - Total parameter count matches expected (~600M)
  - tie_word_embeddings: lm_head is None
- Build LLaMA config. Verify different layer count and head geometry.
- Forward pass with random weights: output shape [batch, seq, vocab].
- Weight loading: create fake weights with correct names, load into model,
  verify weights are in the right modules.

**Pass/fail:**
- Parameter count matches expected (within 1% for tied embeddings).
- Output shape [1, seq_len, vocab_size] from forward pass.
- Weight loading: **zero missing, zero unexpected** keys.
- **Config assembly <= 2 seconds** (no weights, just struct creation).

**Dependencies:** Tasks 7 (norm+ffn), 8 (attention), 9 (kv-cache), 10 (weight loading).

---

### Task 13: Generation Pipeline + Sampling

**Implement:**
- `crates/synapse-inference/src/generation/pipeline.rs`:
  - `GenerationPipeline`: tokenize → prefill → decode loop → detokenize.
  - Prefill: process entire prompt in one forward pass, populate KV-cache.
  - Decode: generate one token at a time, appending to KV-cache.
  - Streaming: optional callback per generated token.

- `crates/synapse-inference/src/generation/sampler.rs`:
  - `GreedySampler`: argmax of logits.
  - `TemperatureSampler`: logits / temperature → softmax → sample.
  - `TopKSampler`: keep top K logits, zero rest, sample.
  - `TopPSampler`: sort logits, cumsum probabilities, keep until cumsum >= p.
  - `RepetitionPenalty`: divide logits of already-generated tokens by penalty.
  - `CombinedSampler`: chain temperature → top_k → top_p → repetition penalty.

- `crates/synapse-inference/src/generation/stopping.rs`:
  - `StopCondition`: EOS token, max_length, stop_sequences (string matching).

- `crates/synapse-inference/src/generation/output.rs`:
  - `GenerationOutput`: text, token_ids, num_tokens, elapsed time, tokens/sec.

**Tests (mandatory):**
- Greedy: same input always produces same output (deterministic).
- Temperature=0: equivalent to greedy.
- Temperature=1.0: output has entropy (not always same token).
- TopK=1: equivalent to greedy.
- TopP=0.0: equivalent to greedy.
- Repetition penalty: repeated tokens get lower probability.
- Stop on EOS: generation stops when EOS is produced.
- Stop on max_length: generation stops at exactly max_length.
- Prefill + decode produces same logits as single forward for short sequences.

**Pass/fail:**
- Greedy is **deterministic** (100 runs, same output).
- Temperature/TopK/TopP produce valid probability distributions (sum=1, non-negative).
- Stop conditions work correctly.
- Streaming callback receives each token.

**Dependencies:** Tasks 11 (tokenizer), 12 (model builder).

---

### Task 14: INT8 Quantization + End-to-End Benchmarks

**Implement:**
- `crates/synapse-inference/src/quantization/int8.rs`:
  - `quantize_model(model: &CausalLM) -> QuantizedCausalLM`
  - Quantizes all Linear layers to INT8 (weights only, activations stay f32).
  - Per-channel quantization of weight matrices.
- `crates/synapse-inference/src/quantization/quantized_linear.rs`:
  - `QuantizedLinear`: stores INT8 weights + f32 scales.
  - Forward: dequantize-on-the-fly or call INT8 GEMM directly.
- `crates/synapse-inference/src/quantization/calibration.rs`:
  - `MinMaxCalibration`: compute per-channel min/max from weight values.
  - `PercentileCalibration`: clip outlier channels.

- Integration tests + benchmarks:
  - `tests/integration/inference_e2e.rs`: Load Qwen3-0.6B from safetensors,
    generate 50 tokens with greedy sampling. Verify output is coherent
    (not garbage — basic perplexity check or known-output comparison).
  - `tests/integration/quantization_accuracy.rs`: Compare INT8 vs f32
    model output logits. Top-1 token agreement rate.
  - `tests/integration/kvcache_correctness.rs`: Generate with KV-cache,
    verify output matches generation without KV-cache (full recompute).
  - `tests/integration/config_driven_assembly.rs`: Load Qwen3 config and
    LLaMA config, build both models, verify different architecture.
  - `tests/benchmarks/inference_throughput.rs`: Measure tokens/sec.
  - `tests/benchmarks/prefill_throughput.rs`: Measure prompt processing speed.
  - `tests/benchmarks/quantization_speedup.rs`: INT8 vs f32 throughput.
  - `tests/benchmarks/memory_usage.rs`: Peak memory during inference.

- Examples:
  - `examples/qwen3_chat.rs`: Interactive chat loop with Qwen3-0.6B.
  - `examples/model_benchmark.rs`: Load any model via config, benchmark
    prefill + decode throughput.

**Tests (mandatory):**
- All integration tests and benchmarks listed above.
- Examples run to completion without errors.

**Pass/fail:**
- **Qwen3-0.6B generates coherent text** (greedy output for known prompt
  matches reference, or top-1 agreement >= 95% with HuggingFace output).
- **INT8 top-1 agreement >= 99%** with f32 model.
- **KV-cache output matches** non-cached output exactly.
- **Config assembly**: Qwen3 and LLaMA both build correctly with different
  architectures from the same engine.
- Throughput thresholds (see Section 6).
- Memory thresholds (see Section 6).
- **All existing Phase 1 and Phase 2 tests still pass** (no regressions).

**Dependencies:** Task 13 (generation pipeline).

---

## 8) Success Metrics

| Metric | Target |
|--------|--------|
| Tasks completed | 14/14 |
| Unit tests | All pass, 100% pass rate |
| Benchmark thresholds | All 11 hard thresholds met |
| Memory safety | Zero leaks (Zig tracking allocator + Rust Drop) |
| FFI safety | Zero panics crossing boundary |
| Qwen3-0.6B inference | Generates coherent text from real weights |
| INT8 quantization | >=99% top-1 agreement with f32 |
| KV-Cache correctness | Bit-exact with full recompute |
| Component registry | Same engine runs Qwen3 and LLaMA configs |
| Phase 1+2 regression | All existing tests still pass |
| New lines | ~15,000–20,000 |
| New crate | `synapse-inference` |
| Test coverage | Every new module has unit tests |
| Benchmark coverage | Every perf-critical module has pass/fail benchmark |

---

## 9) Key Architectural Decisions

1. **Component registry with trait objects, not generics.** Using
   `Box<dyn AttentionVariant>` instead of generic type parameters because:
   - Model architecture is determined at runtime from config JSON
   - Different layers could theoretically have different components
   - Trade-off: ~1 indirect call per layer per token (negligible vs matmul cost)

2. **Inference-only crate, separate from training.** `synapse-inference`
   does not depend on `synapse-autograd` or `synapse-train`. It uses
   `synapse-core` (Tensor) and `synapse-sys` (FFI) directly. This means:
   - No gradient overhead
   - No tape recording
   - Simpler memory model (no saved activations)
   - KV-cache mutation is safe (no autograd graph to corrupt)

3. **Pre-allocated KV-cache.** Allocate max_seq_len * all_layers at init.
   Wastes some memory for short sequences but eliminates all runtime
   allocation during generation. This is the standard approach (vLLM,
   llama.cpp). Future: paged attention for dynamic allocation.

4. **Weight-only INT8 quantization.** Weights are stored as INT8,
   activations remain f32. This is simpler than full INT8 inference
   (which requires activation calibration datasets) and still provides
   ~2x memory reduction and speedup from INT8 GEMM. Matches the most
   common deployment approach.

5. **Safetensors as primary format.** This is the HuggingFace standard
   and what Qwen3 ships in. GGUF support added for llama.cpp ecosystem
   compatibility. Weight mapping rules are per-model-family so adding
   a new model means writing a mapping, not a new loader.

6. **Generic decoder layer, not generic model.** The `DecoderLayer`
   struct is identical for all models — only the trait objects inside
   differ. This means the forward pass logic (pre-norm residual
   architecture) is written once. Adding a new model family means:
   - Implement any missing trait variants (e.g., new attention type)
   - Write a weight mapping config
   - Done — no new forward pass code

7. **Extensibility documentation in code.** Every trait has doc comments
   explaining what future variants would look like (Linear attention,
   MoE FFN, mRoPE, paged KV-cache, etc.). This serves as a roadmap
   for Phase 4+ without any dead code.

---

## 10) Future Extension Roadmap (NOT implemented in Phase 3)

These are documented here so future swarm tasks know exactly what to add
and where to plug it in:

### Phase 4a: Hybrid Attention (for Qwen 3.5)
- Implement `LinearAttention` variant (linear RNN-based attention)
- Implement `HybridLayerSchedule` — per-layer variant selection from config
- Add `layer_types` array to ModelConfig (like Qwen3.5's config)
- Add mRoPE with `mrope_section` and `partial_rotary_factor`
- ~5k lines, plugs into existing `AttentionVariant` trait

### Phase 4b: Vision Encoder (for Qwen 3.5 multimodal)
- Implement ViT patch embedder
- Add vision config to ModelConfig
- Cross-modal projection layer
- ~4k lines, new module in synapse-inference

### Phase 4c: MoE FFN (for Mixtral, Qwen3.5-35B-A3B)
- Implement `MoEFFN` variant with top-k router
- Expert parallelism for multi-core
- ~3k lines, plugs into existing `FFNVariant` trait

### Phase 4d: Advanced Quantization (INT4, GPTQ, AWQ)
- INT4 quantized GEMM kernels in Zig
- GPTQ/AWQ weight format loading
- ~4k lines, plugs into existing `QuantVariant` enum

### Phase 4e: Speculative Decoding / Multi-Token Prediction
- Draft model generates N tokens speculatively
- Main model verifies in single forward pass
- MTP head (like Qwen3.5's `mtp_num_hidden_layers`)
- ~3k lines, plugs into generation pipeline
