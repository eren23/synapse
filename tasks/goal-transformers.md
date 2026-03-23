# Swarm Goal

Synapse Phase 2 — Transformer & Attention Stack (Zig + Rust)

Extend the Synapse neural network training framework with a complete
transformer stack: SIMD-vectorized attention kernels in Zig, LayerNorm,
rotary positional embeddings, multi-head attention, transformer encoder/decoder
blocks, text data utilities, and new graph fusion passes for attention patterns.
**Builds on top of the existing Synapse codebase** — reuses existing matmul,
softmax, activation kernels, autograd engine, optimizer, and training loop.
Targeting ~15,000–20,000 new lines with comprehensive unit tests, benchmark
tests with hard pass/fail performance thresholds, and end-to-end training
examples (text classification + Vision Transformer).

**CRITICAL RULE: Every task MUST include its own tests. No implementation
without tests. Every benchmark MUST have a hard pass/fail threshold. If a
benchmark does not meet its threshold, the task FAILS.**

---

## 0) Phase 2 Overview

Synapse Phase 1 delivered a complete CNN/MLP training pipeline: Zig SIMD
tensor engine, Rust autograd, neural network layers, optimizers, data loading,
graph IR with operator fusion, and a PyTorch-like training API. Phase 2 adds
the single most impactful missing capability: **transformer/attention support**.

Without transformers, Synapse cannot train any modern architecture (BERT, GPT,
Vision Transformer, etc.). This phase adds them at every layer of the stack:
Zig kernels → C ABI FFI → Rust autograd backward ops → Rust nn modules →
graph fusion passes → data utilities → examples.

### What Already Exists (DO NOT REWRITE)

Phase 2 builds on top of Phase 1. The following are already implemented and
must be reused, not reimplemented:

- **Zig SIMD kernels**: matmul (tiled SGEMM), softmax (online stable),
  elementwise ops, reductions, conv2d, pooling, batchnorm, activations
- **Zig allocators**: arena, pool, aligned, tracking
- **Zig FFI exports**: Full C ABI with opaque handles and status codes
- **Rust autograd**: Variable, Graph, backward(), GradFn trait, grad_check,
  all existing backward ops (arithmetic, matmul, conv, pool, batchnorm, etc.)
- **Rust nn**: Module trait, Sequential, Linear, Conv2d, BatchNorm, Dropout,
  ReLU/Sigmoid/Tanh/GELU/Softmax, Embedding, LSTMCell, GRUCell, pooling
- **Rust optim**: SGD, Adam, RMSProp, LR schedulers, gradient clipping
- **Rust data**: Dataset trait, DataLoader, samplers, transforms, collation
- **Rust train**: Trainer, callbacks, metrics, checkpoints, progress
- **Rust graph**: IR, fusion passes, DCE, constant folding, scheduler

### Sample Usage (Target API)

```rust
use synapse::prelude::*;

fn main() -> Result<()> {
    // Transformer encoder for sequence classification
    let model = Sequential::new(vec![
        Box::new(Embedding::new(vocab_size, 256)),
        Box::new(SinusoidalPositionalEncoding::new(256, max_seq_len)),
        Box::new(TransformerEncoder::new(TransformerEncoderConfig {
            d_model: 256,
            n_heads: 8,
            d_ff: 1024,
            n_layers: 4,
            dropout: 0.1,
            activation: Activation::GELU,
        })),
        // Pool over sequence dimension, then classify
        Box::new(MeanPool1d),
        Box::new(Linear::new(256, num_classes, true)?),
    ]);

    let mut optimizer = Adam::new(model.parameters(), AdamConfig {
        lr: 1e-4,
        betas: (0.9, 0.98),
        weight_decay: 0.01,
        ..Default::default()
    });

    // Warmup + cosine decay (standard transformer schedule)
    let scheduler = WarmupCosineScheduler::new(
        &optimizer, warmup_steps: 500, total_steps: 10000,
    );

    // Training loop (existing Trainer handles the rest)
    let mut trainer = Trainer::new(TrainerConfig { epochs: 10 });
    trainer.fit(&mut model);
    Ok(())
}
```

### Sample Usage (Zig Fused Attention Kernel)

```zig
const synapse = @import("synapse");
const attention = synapse.ops.attention;
const Tensor = synapse.tensor.Tensor;

pub fn benchmark_attention() !void {
    var arena = synapse.alloc.ArenaAllocator.init(
        std.heap.page_allocator, 64 * 1024 * 1024,
    );
    defer arena.deinit();

    const alloc = arena.allocator();
    // Q, K, V: [batch=8, heads=8, seq=128, d_head=32]
    const q = try Tensor(f32).randn(alloc, &.{ 8, 8, 128, 32 });
    const k = try Tensor(f32).randn(alloc, &.{ 8, 8, 128, 32 });
    const v = try Tensor(f32).randn(alloc, &.{ 8, 8, 128, 32 });

    // Fused: Q×K^T/√d → causal mask → softmax → ×V in one pass
    var output = try Tensor(f32).zeros(alloc, &.{ 8, 8, 128, 32 });
    attention.scaled_dot_product(q, k, v, &output, .{
        .causal = true,
        .scale = 1.0 / @sqrt(32.0),
    });
}
```

---

## 1) Architecture Changes

```
Existing Synapse (unchanged)
─────────────────────────────────────────────────────
│  synapse-train, synapse-nn, synapse-data           │
│  synapse-autograd, synapse-optim, synapse-graph     │
│  synapse-core, synapse-sys (FFI)                    │
│  libsynapse_zig.a (tensor, alloc, simd, ops)        │
─────────────────────────────────────────────────────

Phase 2 additions (NEW)
─────────────────────────────────────────────────────
│  synapse-nn:                                        │
│    + LayerNorm, MultiHeadAttention,                 │
│    + TransformerEncoderLayer, TransformerEncoder,    │
│    + TransformerDecoderLayer, TransformerDecoder,    │
│    + SinusoidalPositionalEncoding, LearnablePosEmb,  │
│    + RotaryPositionalEmbedding, MeanPool1d          │
│                                                     │
│  synapse-autograd/ops:                              │
│    + attention.rs (scaled_dot_product backward)     │
│    + layernorm.rs (LayerNorm backward)              │
│    + rope.rs (RoPE backward)                        │
│                                                     │
│  synapse-graph:                                     │
│    + FuseAttention pass                             │
│    + FuseLayerNormResidual pass                     │
│                                                     │
│  synapse-data:                                      │
│    + tokenizer.rs (whitespace + BPE)                │
│    + text_dataset.rs (line-based text dataset)      │
│    + SequencePadCollate (pad to max length)         │
│                                                     │
│  Zig layer:                                         │
│    + ops/attention.zig (fused SDPA kernel)          │
│    + ops/layernorm.zig (SIMD layer normalization)   │
│    + ops/rope.zig (rotary positional embeddings)    │
│    + ffi/exports.zig (new FFI exports)              │
│                                                     │
│  Examples:                                          │
│    + examples/text_classification.rs                │
│    + examples/vision_transformer.rs                 │
─────────────────────────────────────────────────────
```

---

## 2) New Files

```
synapse/
├── zig/
│   ├── src/
│   │   └── ops/
│   │       ├── attention.zig           # Fused scaled dot-product attention kernel
│   │       ├── layernorm.zig           # Layer normalization (Welford single-pass)
│   │       └── rope.zig                # Rotary positional embedding computation
│   │
│   └── tests/
│       ├── test_attention.zig          # Attention correctness vs naive
│       ├── test_layernorm.zig          # LayerNorm correctness
│       ├── test_rope.zig               # RoPE correctness
│       ├── bench_attention.zig         # MUST pass: fused >=3x vs naive
│       └── bench_layernorm.zig         # MUST pass: SIMD >=4x vs scalar
│
├── crates/
│   ├── synapse-autograd/src/ops/
│   │   ├── attention.rs                # Scaled dot-product attention backward
│   │   ├── layernorm.rs                # LayerNorm backward
│   │   └── rope.rs                     # RoPE backward
│   │
│   ├── synapse-nn/src/
│   │   ├── layernorm.rs                # LayerNorm module (weight + bias)
│   │   ├── attention.rs                # MultiHeadAttention module
│   │   ├── transformer.rs              # TransformerEncoder/DecoderLayer + stacked
│   │   └── positional.rs               # Sinusoidal, learnable, RoPE positional encodings
│   │
│   ├── synapse-data/src/
│   │   ├── tokenizer.rs                # Whitespace + byte-pair encoding tokenizer
│   │   └── text_dataset.rs             # Line-based text dataset with vocab building
│   │
│   └── synapse-graph/src/
│       ├── fuse_attention.rs           # FuseAttention optimization pass
│       └── fuse_layernorm_residual.rs  # FuseLayerNormResidual optimization pass
│
├── tests/
│   ├── integration/
│   │   ├── transformer_e2e.rs          # Train transformer on synthetic seq classification
│   │   ├── attention_correctness.rs    # Gradient checks for all attention ops
│   │   └── transformer_graph_opt.rs    # Attention fusion correctness
│   │
│   └── benchmarks/
│       ├── attention_bench.rs          # Fused vs naive attention throughput
│       └── transformer_throughput.rs   # End-to-end transformer training throughput
│
└── examples/
    ├── text_classification.rs          # Transformer text classifier
    └── vision_transformer.rs           # ViT on CIFAR-10
```

---

## 3) New FFI Functions

Extends `synapse.h` with new exports. Same conventions as Phase 1: opaque
handles, status codes, no panics crossing the boundary.

```c
// --- Layer Normalization ---
syn_status_t syn_layernorm_forward(
    syn_tensor_t* out,              // [*, normalized_shape]
    const syn_tensor_t* input,      // [*, normalized_shape]
    const syn_tensor_t* gamma,      // [normalized_shape]
    const syn_tensor_t* beta,       // [normalized_shape]
    size_t normalized_dim,          // number of trailing dims to normalize
    float eps                       // default 1e-5
);

// --- Fused Scaled Dot-Product Attention ---
syn_status_t syn_scaled_dot_product_attention(
    syn_tensor_t* out,              // [batch, heads, seq_q, d_head]
    syn_tensor_t* attn_weights,     // [batch, heads, seq_q, seq_k] (optional, NULL to skip)
    const syn_tensor_t* query,      // [batch, heads, seq_q, d_head]
    const syn_tensor_t* key,        // [batch, heads, seq_k, d_head]
    const syn_tensor_t* value,      // [batch, heads, seq_k, d_head]
    float scale,                    // typically 1/sqrt(d_head)
    int causal                      // 1 for causal masking, 0 for no mask
);

// --- Rotary Positional Embedding ---
syn_status_t syn_rope_forward(
    syn_tensor_t* out,              // same shape as input
    const syn_tensor_t* input,      // [batch, heads, seq, d_head]
    const syn_tensor_t* cos_table,  // [max_seq, d_head/2]
    const syn_tensor_t* sin_table,  // [max_seq, d_head/2]
    size_t offset                   // position offset for KV-cache scenarios
);

// --- Causal Mask Generation ---
syn_status_t syn_causal_mask(
    syn_tensor_t* out,              // [seq, seq] filled with 0/-inf
    size_t seq_len
);
```

---

## 4) New Rust Trait Implementations

### LayerNorm (synapse-nn)

```rust
pub struct LayerNorm {
    weight: Variable,           // gamma, shape [normalized_shape]
    bias: Variable,             // beta,  shape [normalized_shape]
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    // Normalizes over last N dims matching normalized_shape
}
```

### MultiHeadAttention (synapse-nn)

```rust
pub struct MultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    d_head: usize,              // d_model / n_heads
    w_q: Linear,                // [d_model, d_model]
    w_k: Linear,                // [d_model, d_model]
    w_v: Linear,                // [d_model, d_model]
    w_o: Linear,                // [d_model, d_model]
    dropout: Dropout,
    rope: Option<RotaryPositionalEmbedding>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout: f32) -> Self;
    pub fn forward_with_mask(
        &self, query: &Variable, key: &Variable, value: &Variable,
        causal: bool,
    ) -> Result<Variable>;
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    // Self-attention: query = key = value = input
}
```

### TransformerEncoderLayer (synapse-nn)

```rust
pub struct TransformerEncoderConfig {
    pub d_model: usize,         // embedding dimension
    pub n_heads: usize,         // attention heads
    pub d_ff: usize,            // feed-forward hidden dim (typically 4*d_model)
    pub n_layers: usize,        // number of encoder layers
    pub dropout: f32,           // dropout rate
    pub activation: Activation, // GELU or ReLU for FFN
}

pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,           // post-attention
    norm2: LayerNorm,           // post-FFN
    ff1: Linear,                // d_model -> d_ff
    ff2: Linear,                // d_ff -> d_model
    dropout: Dropout,
    activation: Box<dyn Module>,
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    // Pre-norm architecture:
    // x = x + self_attn(norm1(x))
    // x = x + ff2(dropout(activation(ff1(norm2(x)))))
}

pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    final_norm: LayerNorm,
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Variable) -> Result<Variable>;
}
```

### TransformerDecoderLayer (synapse-nn)

```rust
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,      // causal self-attention
    cross_attn: MultiHeadAttention,     // cross-attention to encoder output
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    dropout: Dropout,
    activation: Box<dyn Module>,
}

impl TransformerDecoderLayer {
    pub fn forward_with_memory(
        &self, tgt: &Variable, memory: &Variable, causal: bool,
    ) -> Result<Variable>;
    // x = x + self_attn(norm1(x))  [causal]
    // x = x + cross_attn(norm2(x), memory, memory)
    // x = x + ff2(dropout(activation(ff1(norm3(x)))))
}

pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    final_norm: LayerNorm,
}
```

### Positional Encodings (synapse-nn)

```rust
pub struct SinusoidalPositionalEncoding {
    encoding: Tensor,           // precomputed [max_len, d_model]
}

pub struct LearnablePositionalEmbedding {
    embedding: Embedding,       // [max_len, d_model] learnable
}

pub struct RotaryPositionalEmbedding {
    cos_table: Tensor,          // [max_len, d_head/2]
    sin_table: Tensor,          // [max_len, d_head/2]
    d_head: usize,
}
```

### Autograd Ops (synapse-autograd)

```rust
// attention.rs
pub struct ScaledDotProductAttentionBackward {
    // Saves Q, K, V, attention_weights for backward
}
impl GradFn for ScaledDotProductAttentionBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    // Returns: grad_query, grad_key, grad_value
}

// layernorm.rs
pub struct LayerNormBackward {
    // Saves input, mean, rstd, weight for backward
}
impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    // Returns: grad_input, grad_weight, grad_bias
}

// rope.rs
pub struct RoPEBackward {
    // Saves cos_table, sin_table for backward
}
impl GradFn for RoPEBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    // RoPE backward: rotate gradient by -theta (inverse rotation)
}
```

### Graph Optimization Passes (synapse-graph)

```rust
// FuseAttention: Q_proj + K_proj + V_proj + MatMul + Scale + Mask + Softmax + MatMul
// → single FusedAttention node
pub struct FuseAttention;
impl OptimizationPass for FuseAttention {
    fn name(&self) -> &str { "fuse_attention" }
    fn run(&self, graph: &mut ComputeGraph) -> Result<bool>;
}

// FuseLayerNormResidual: LayerNorm(x + residual) → FusedLayerNormResidual
pub struct FuseLayerNormResidual;
impl OptimizationPass for FuseLayerNormResidual {
    fn name(&self) -> &str { "fuse_layernorm_residual" }
    fn run(&self, graph: &mut ComputeGraph) -> Result<bool>;
}
```

---

## 5) Optimization Targets

**These are HARD pass/fail thresholds. If a benchmark does not meet its
target, the task FAILS. Every task that has a benchmark threshold must
include both the naive baseline implementation AND the optimized
implementation, and the benchmark must compare them.**

| # | Module | Metric | Threshold | How to Measure |
|---|--------|--------|-----------|----------------|
| 1 | `ops/attention.zig` | Fused SDPA vs naive (separate matmul+softmax+matmul) | **>=3x** on [8, 8, 128, 32] | `bench_attention.zig`: 50 iterations, compare fused vs naive |
| 2 | `ops/attention.zig` | Fused SDPA with causal mask vs without | **<=10% overhead** for masking | Same benchmark, causal=true vs causal=false |
| 3 | `ops/layernorm.zig` | SIMD LayerNorm vs scalar | **>=4x** on [64, 128, 256] | `bench_layernorm.zig`: 100 iterations |
| 4 | `ops/layernorm.zig` | Welford single-pass vs two-pass | **>=1.5x** on [64, 128, 256] | Same benchmark |
| 5 | `ops/rope.zig` | SIMD RoPE vs scalar | **>=3x** on [8, 8, 128, 32] | Benchmark in `test_rope.zig` |
| 6 | `synapse-graph` | Fused attention vs unfused graph | **>=1.3x** on 4-layer transformer | `transformer_graph_opt.rs` integration test |
| 7 | `synapse-graph` | Fused LayerNorm+residual vs unfused | **>=1.2x** | Same integration test |
| 8 | `synapse-autograd` | Attention backward overhead | **<=10%** overhead vs forward-only | `attention_bench.rs`: forward+backward vs forward |
| 9 | `synapse-train` | Transformer throughput | **>=2000 tokens/sec** on 4-layer encoder, d=256, seq=128 | `transformer_throughput.rs` |
| 10 | `synapse-train` | Sequence classification accuracy | **>85%** in 5 epochs on synthetic task | `transformer_e2e.rs` |

### Correctness Thresholds (non-negotiable)

| Module | Requirement |
|--------|-------------|
| Fused attention | Max relative error **<= 1e-4** vs naive (Q×K^T→softmax→×V) for seq_len up to 512 |
| LayerNorm | Output mean **<= 1e-5**, variance within **1e-4** of 1.0 over normalized dims |
| RoPE | Max relative error **<= 1e-5** vs scalar reference implementation |
| All new autograd ops | `grad_check` passes: analytical vs numerical gradient relative error **< 1e-3** |
| Causal masking | Attention weights are **exactly 0.0** for future positions (not approximately zero) |
| Softmax in attention | **No inf/nan** for sequences up to 2048 tokens |
| FFI boundary | **Zero panics** crossing FFI for all new exports |
| Memory | **Zero leaks** detected by Zig tracking allocator in all test scenarios |

---

## 6) Task Decomposition — 12 Tasks

**CRITICAL RULES FOR EVERY TASK:**
1. Every task MUST write tests alongside implementation. No code without tests.
2. Every benchmark task MUST include both naive baseline AND optimized implementation.
3. Every task MUST list its pass/fail criteria. The judge uses these to accept/reject.
4. Dependencies must be respected. A task cannot start until its dependencies are complete.

### Dependency Graph

```
WAVE 1 (fully parallel, no dependencies on each other — only on existing Phase 1 code):
  Task 1: Zig LayerNorm kernel
  Task 2: Zig Fused Attention kernel
  Task 3: Zig RoPE kernel

WAVE 2 (depends on Wave 1):
  Task 4: Zig FFI Exports for new ops ──── depends: Tasks 1, 2, 3
  Task 5: Rust FFI bindings for new ops ── depends: Task 4

WAVE 3 (Rust autograd + nn, many parallel):
  Task 6: Rust Autograd — Attention ops ── depends: Task 5
  Task 7: Rust Autograd — LayerNorm+RoPE ─ depends: Task 5 (parallel with 6)
  Task 8: Rust NN — Positional Encodings ─ depends: Task 7
  Task 9: Rust NN — MultiHeadAttention ─── depends: Tasks 6, 8

WAVE 4 (integration):
  Task 10: Rust NN — Transformer blocks ── depends: Task 9
  Task 11: Rust Data — Text utilities ──── depends: none (parallel with all)
  Task 12: Rust Graph + Examples + E2E ─── depends: Tasks 10, 11
```

---

### Task 1: Zig LayerNorm Kernel

**Implement:**
- `zig/src/ops/layernorm.zig`: SIMD-vectorized layer normalization using
  Welford's single-pass algorithm for numerically stable mean+variance.
  Normalizes over trailing dimensions. Applies affine transform (gamma*x+beta).
  Handles arbitrary tensor shapes.
  **MUST also include naive two-pass implementation (mean pass, variance pass)
  AND a scalar reference for benchmark comparison.**

**Tests (mandatory):**
- `zig/tests/test_layernorm.zig`: Correctness vs scalar reference for shapes:
  [64, 256], [32, 128, 512], [8, 16, 32, 64]. Output mean <= 1e-5, variance
  within 1e-4 of 1.0 over normalized dims. Special values: all-zeros input,
  constant input (var=0 + eps), large values (±1000).
- `zig/tests/bench_layernorm.zig`: SIMD vs scalar on [64, 128, 256],
  100 iterations. Welford vs two-pass on same shape.

**Pass/fail:**
- All correctness tests pass.
- **SIMD >=4x throughput** vs scalar reference.
- **Welford single-pass >=1.5x** vs two-pass.
- Numerically stable: no inf/nan for inputs in [-1000, 1000].

**Dependencies:** None (uses existing SIMD dispatch and tensor infrastructure).

---

### Task 2: Zig Fused Attention Kernel

**Implement:**
- `zig/src/ops/attention.zig`: Fused scaled dot-product attention:
  1. Compute S = Q × K^T (reuse existing tiled SGEMM)
  2. Scale: S = S / sqrt(d_head)
  3. Optional causal mask: S[i][j] = -inf where j > i
  4. Softmax over last dimension (reuse existing online softmax)
  5. Output = S × V (reuse existing tiled SGEMM)

  The "fused" part: steps 1-5 happen in a single function call with tiled
  memory access patterns that keep intermediate S in cache rather than writing
  to main memory between steps. Tile over the sequence dimension.

  **MUST also include naive implementation (separate matmul → scale → mask →
  softmax → matmul) for benchmark comparison.**

  Shapes: Q [batch, heads, seq_q, d_head], K [batch, heads, seq_k, d_head],
  V [batch, heads, seq_k, d_head] → output [batch, heads, seq_q, d_head].

  Optionally output attention weights [batch, heads, seq_q, seq_k].

**Tests (mandatory):**
- `zig/tests/test_attention.zig`: Correctness vs naive for shapes:
  [1,1,8,32], [2,4,32,64], [8,8,128,32]. Causal vs non-causal.
  Verify causal mask: attention weights are exactly 0.0 for future positions.
  Max relative error <= 1e-4 vs naive.
  Edge cases: seq_len=1, d_head=1, batch=1.
- `zig/tests/bench_attention.zig`: Fused vs naive on [8,8,128,32],
  50 iterations. Causal vs non-causal overhead.

**Pass/fail:**
- Correctness within 1e-4 relative error for all shapes.
- **Fused >=3x** vs naive separate-step implementation.
- **Causal masking overhead <=10%** vs non-causal.
- Causal mask positions are exactly 0.0, not approximately zero.
- No inf/nan for sequences up to 2048 tokens.

**Dependencies:** None (reuses existing matmul.zig and softmax.zig internally).

---

### Task 3: Zig RoPE Kernel

**Implement:**
- `zig/src/ops/rope.zig`: Rotary positional embedding computation.
  Given input tensor [batch, heads, seq, d_head] and precomputed cos/sin
  tables [max_seq, d_head/2], applies rotation:
  ```
  x_rotated[..., 2i]   = x[..., 2i] * cos[pos, i] - x[..., 2i+1] * sin[pos, i]
  x_rotated[..., 2i+1] = x[..., 2i] * sin[pos, i] + x[..., 2i+1] * cos[pos, i]
  ```
  SIMD-vectorized: pairs of elements processed as complex rotations.
  Position offset parameter for KV-cache scenarios.
  **MUST include scalar reference implementation for benchmark comparison.**

  Also implement cos/sin table generation:
  ```
  theta_i = 1 / (10000 ^ (2i / d_head))
  cos_table[pos, i] = cos(pos * theta_i)
  sin_table[pos, i] = sin(pos * theta_i)
  ```

**Tests (mandatory):**
- `zig/tests/test_rope.zig`: Correctness vs scalar reference for d_head=32,
  64, 128. Verify rotation is invertible (apply then apply with negated sin).
  Offset parameter shifts positions correctly. Edge: seq_len=1, d_head=2.
- Benchmark: SIMD RoPE vs scalar on [8, 8, 128, 32].

**Pass/fail:**
- Max relative error **<= 1e-5** vs scalar reference.
- **SIMD >=3x** vs scalar on [8, 8, 128, 32].
- Rotation invertibility verified (apply forward then backward = identity within 1e-5).

**Dependencies:** None (uses existing SIMD dispatch).

---

### Task 4: Zig FFI Exports for Transformer Ops

**Implement:**
- Extend `zig/src/ffi/exports.zig` with new exported functions:
  - `syn_layernorm_forward`
  - `syn_scaled_dot_product_attention`
  - `syn_rope_forward`
  - `syn_causal_mask`
- Update `synapse.h` with new function declarations.
- All functions use opaque handles, return `syn_status_t`, no panics escape.
- Null pointer checks on all inputs.

**Tests (mandatory):**
- Round-trip test: create tensors via FFI, call each new function, verify
  result via FFI, destroy. All via C ABI exported functions.
- Invalid input test: null pointers, shape mismatches produce proper error codes.

**Pass/fail:**
- All FFI functions callable via C ABI.
- **No panics** escape across FFI boundary.
- Error codes returned for invalid inputs (null, shape mismatch).
- `zig build` still produces valid `libsynapse_zig.a`.

**Dependencies:** Tasks 1, 2, 3.

---

### Task 5: Rust FFI Bindings for Transformer Ops

**Implement:**
- Extend `crates/synapse-sys/src/lib.rs` with new `extern "C"` declarations
  matching the updated `synapse.h`.
- Extend `crates/synapse-core/` with safe Rust wrappers for:
  - `Tensor::layernorm(gamma, beta, eps)`
  - `Tensor::scaled_dot_product_attention(key, value, scale, causal)`
  - `Tensor::rope(cos_table, sin_table, offset)`
- All wrap status codes into `Result<T, SynapseError>`.

**Tests (mandatory):**
- FFI roundtrip: create Rust tensors, call each new FFI function through
  safe wrappers, verify results match expected output.
- Error propagation: shape mismatches produce proper Rust errors.

**Pass/fail:**
- All new FFI functions correctly bridged.
- **No memory leaks** in create/call/destroy cycle (10K iterations).
- Zig errors correctly map to `Result::Err`.

**Dependencies:** Task 4.

---

### Task 6: Rust Autograd — Attention Backward

**Implement:**
- `crates/synapse-autograd/src/ops/attention.rs`:
  `ScaledDotProductAttentionBackward` implementing `GradFn`.
  Backward: given grad_output [B,H,Sq,D] and saved Q,K,V,attn_weights:
  ```
  grad_V = attn_weights^T × grad_output
  grad_attn = grad_output × V^T
  grad_scores = softmax_backward(grad_attn, attn_weights)
  grad_scores *= scale
  [apply causal mask to grad_scores if causal]
  grad_Q = grad_scores × K
  grad_K = grad_scores^T × Q
  ```

**Tests (mandatory):**
- `grad_check` for scaled dot-product attention: numerical vs analytical
  for Q, K, V gradients. Shapes: [1,1,4,8], [2,4,16,32].
  Both causal and non-causal.
- Verify gradient shapes match input shapes.

**Pass/fail:**
- **grad_check passes** (relative error < 1e-3) for all three gradients (Q, K, V).
- Works for both causal and non-causal attention.
- No shape mismatches.

**Dependencies:** Task 5.

---

### Task 7: Rust Autograd — LayerNorm + RoPE Backward

**Implement:**
- `crates/synapse-autograd/src/ops/layernorm.rs`:
  `LayerNormBackward` implementing `GradFn`.
  Saves: input, mean, rstd (reciprocal std), weight.
  Returns: grad_input, grad_weight, grad_bias.

- `crates/synapse-autograd/src/ops/rope.rs`:
  `RoPEBackward` implementing `GradFn`.
  RoPE backward is the inverse rotation (negate sin component):
  ```
  grad_input[..., 2i]   = grad[..., 2i] * cos[pos,i] + grad[..., 2i+1] * sin[pos,i]
  grad_input[..., 2i+1] = -grad[..., 2i] * sin[pos,i] + grad[..., 2i+1] * cos[pos,i]
  ```

**Tests (mandatory):**
- `grad_check` for LayerNorm: shapes [32, 64], [8, 16, 32]. Verify
  grad_input, grad_weight, grad_bias all pass.
- `grad_check` for RoPE: shapes [2, 4, 16, 32]. Verify grad_input passes.
- Edge cases: constant input to LayerNorm (near-zero variance).

**Pass/fail:**
- **grad_check passes** for all LayerNorm gradients (< 1e-3).
- **grad_check passes** for RoPE gradient (< 1e-3).
- Handles edge cases without inf/nan.

**Dependencies:** Task 5.

---

### Task 8: Rust NN — Positional Encodings

**Implement:**
- `crates/synapse-nn/src/positional.rs`:
  - `SinusoidalPositionalEncoding`: Precomputes [max_len, d_model] table
    using standard sin/cos formula. Added to input via broadcast add.
    Not learnable.
  - `LearnablePositionalEmbedding`: Wraps existing `Embedding` module
    with indices [0..seq_len].
  - `RotaryPositionalEmbedding`: Precomputes cos/sin tables. Applies
    RoPE to Q and K inside attention (not to input directly). Uses
    Zig RoPE kernel via autograd op from Task 7.
  - `MeanPool1d`: Averages over sequence dimension. Utility for
    classification heads.

**Tests (mandatory):**
- Sinusoidal: output shape [batch, seq, d_model]. Values match
  hand-computed sin/cos for first few positions.
- Learnable: parameters count = max_len * d_model.
- RoPE: cos/sin table shapes correct. After applying RoPE, vectors at
  different positions produce different dot products (relative position
  matters). Test with known values.
- MeanPool1d: correct shape reduction [B, S, D] → [B, D].

**Pass/fail:**
- All output shapes correct.
- Sinusoidal values within 1e-6 of reference.
- RoPE tables match reference theta formula.
- All modules implement Module trait correctly.

**Dependencies:** Task 7 (for RoPE autograd op).

---

### Task 9: Rust NN — MultiHeadAttention

**Implement:**
- `crates/synapse-nn/src/attention.rs`:
  - `MultiHeadAttention` with configurable d_model, n_heads, dropout, and
    optional RoPE.
  - QKV projections via three `Linear` layers (or packed QKV linear).
  - Split heads: reshape [B, S, D] → [B, H, S, D/H].
  - Scaled dot-product attention (calls autograd op from Task 6).
  - Concat heads: reshape [B, H, S, D/H] → [B, S, D].
  - Output projection via `Linear`.
  - `forward_with_mask()` for causal self-attention.
  - Standard `forward()` for non-causal self-attention (query=key=value=input).

**Tests (mandatory):**
- Output shape: [B, S, D] → [B, S, D] for self-attention.
- Parameter count: 4 * d_model^2 + 4 * d_model (weights + biases for Q,K,V,O).
- Training mode: dropout active. Inference: dropout inactive.
- Causal attention: verify output at position i depends only on positions <= i
  (set input positions > i to large values, verify output unchanged).
- Gradient flows through all parameters (no disconnected components).

**Pass/fail:**
- Output shapes correct for all configurations.
- Parameter counts match formula.
- Causal masking verified.
- Gradients flow to all 4 projection matrices.

**Dependencies:** Tasks 6 (attention autograd), 8 (positional encodings for RoPE).

---

### Task 10: Rust NN — Transformer Encoder/Decoder Blocks

**Implement:**
- `crates/synapse-nn/src/transformer.rs`:
  - `TransformerEncoderLayer`: Pre-norm architecture:
    ```
    x = x + self_attn(norm1(x))
    x = x + ff2(dropout(activation(ff1(norm2(x)))))
    ```
  - `TransformerEncoder`: Stack of N encoder layers + final LayerNorm.
  - `TransformerDecoderLayer`: Pre-norm with causal self-attention +
    cross-attention:
    ```
    x = x + self_attn(norm1(x))        [causal]
    x = x + cross_attn(norm2(x), memory, memory)
    x = x + ff2(dropout(activation(ff1(norm3(x)))))
    ```
  - `TransformerDecoder`: Stack of N decoder layers + final LayerNorm.
  - `TransformerEncoderConfig` / `TransformerDecoderConfig` structs.
  - Configurable activation (ReLU or GELU) for FFN.

- Update `crates/synapse-nn/src/layernorm.rs` (new file):
  - `LayerNorm` module wrapping the autograd op from Task 7.

**Tests (mandatory):**
- Encoder output shape: [B, S, D] → [B, S, D].
- Decoder output shape: [B, Tgt_S, D] given memory [B, Src_S, D].
- Parameter count matches expected (per layer: 4*D^2 + 8*D + 2*D*Dff + 2*Dff + 4*D).
- Training/inference mode propagates through all sub-modules.
- Stack of 4 layers: forward pass succeeds, backward produces gradients.
- Residual connections verified: if all weights are zero-initialized,
  output approximately equals input (identity residual path).

**Pass/fail:**
- All output shapes correct.
- Parameter counts match formula.
- Mode propagation works for all sub-modules.
- Backward produces non-zero gradients for all parameters.
- Residual path verification passes.

**Dependencies:** Task 9 (MultiHeadAttention), Task 7 (LayerNorm).

---

### Task 11: Rust Data — Text Utilities

**Implement:**
- `crates/synapse-data/src/tokenizer.rs`:
  - `WhitespaceTokenizer`: Splits on whitespace. Builds vocabulary from
    corpus. Maps tokens to integer IDs. Includes `<PAD>`, `<UNK>`,
    `<BOS>`, `<EOS>` special tokens.
  - `BPETokenizer`: Byte-pair encoding. Train from corpus (configurable
    vocab size). Encode/decode methods.
  - `Vocabulary`: word-to-id and id-to-word mappings.
- `crates/synapse-data/src/text_dataset.rs`:
  - `TextClassificationDataset`: Loads line-based text files where each
    line is `label\ttext`. Tokenizes with provided tokenizer. Stores
    as (token_ids, label) pairs.
  - `SequencePadCollate`: Custom collation function that pads sequences
    to max length in batch. Returns (padded_tokens [B, max_len],
    labels [B], lengths [B]).

**Tests (mandatory):**
- WhitespaceTokenizer: encode/decode roundtrip. `<UNK>` for unknown words.
  Vocabulary size matches unique tokens + specials.
- BPE: train on small corpus, verify merges reduce vocab. Encode/decode
  roundtrip for in-vocabulary text.
- TextClassificationDataset: loads sample file, returns correct token_ids
  and labels. Length matches line count.
- SequencePadCollate: batch of variable-length sequences → padded tensor
  with correct `<PAD>` values. Lengths tensor correct.

**Pass/fail:**
- Encode/decode roundtrip is lossless for in-vocabulary text.
- `<UNK>` correctly handles out-of-vocabulary tokens.
- Padding produces correct shapes and values.
- BPE training terminates and reduces input.

**Dependencies:** None (uses existing Dataset trait and Tensor). Can run
in parallel with everything.

---

### Task 12: Rust Graph Fusion + Examples + End-to-End Tests

**Implement:**
- `crates/synapse-graph/src/fuse_attention.rs`: `FuseAttention` pass that
  detects QKV projection → split heads → matmul → scale → softmax → matmul →
  concat → output projection pattern and replaces with single FusedAttention
  node. Add `OpKind::FusedAttention` to IR.
- `crates/synapse-graph/src/fuse_layernorm_residual.rs`:
  `FuseLayerNormResidual` pass that detects Add(x, residual) → LayerNorm
  and replaces with single FusedLayerNormResidual node. Add
  `OpKind::FusedLayerNormResidual` to IR.
- Update `synapse-graph` to register new passes in the pipeline.

- `examples/text_classification.rs`: Train a 2-layer transformer encoder
  on a synthetic text classification task (random word patterns → binary
  label). Demonstrates: tokenizer → dataset → dataloader → transformer →
  training loop.
- `examples/vision_transformer.rs`: Vision Transformer on synthetic
  CIFAR-10-like data. Patch embedding (conv2d with kernel=patch_size) →
  positional encoding → transformer encoder → classification head.

- Integration tests:
  - `tests/integration/transformer_e2e.rs`: Train 4-layer transformer on
    synthetic sequence classification (1000 sequences, vocab=500, seq_len=32,
    d_model=64, 4 heads, d_ff=256). Must reach >85% accuracy in 5 epochs.
  - `tests/integration/attention_correctness.rs`: grad_check for full
    MultiHeadAttention module (not just the kernel — the whole module
    including projections).
  - `tests/integration/transformer_graph_opt.rs`: Build transformer graph,
    apply fusion passes, verify fused output matches unfused within 1e-4.
    Benchmark fused vs unfused.
  - `tests/benchmarks/attention_bench.rs`: Fused vs naive attention
    throughput through Rust+FFI.
  - `tests/benchmarks/transformer_throughput.rs`: End-to-end transformer
    training, measure tokens/sec.

**Tests (mandatory):**
- All tests listed above. Every benchmark has a hard threshold.
- Examples run to completion without errors.

**Pass/fail:**
- **Transformer E2E: >85% accuracy** in 5 epochs on synthetic task.
- **Attention fusion >=1.3x** vs unfused graph on 4-layer transformer.
- **LayerNorm+residual fusion >=1.2x** vs unfused.
- **Transformer throughput >=2000 tokens/sec** (4-layer, d=256, seq=128,
  batch=32). Use `cfg!(debug_assertions)` for debug-mode threshold of 200
  tokens/sec.
- **Attention bench**: fused kernel >=2x vs naive through Rust FFI.
- grad_check passes for full MultiHeadAttention module.
- Examples run without error.
- **All existing Phase 1 tests still pass** (no regressions).

**Dependencies:** Tasks 10, 11.

---

## 7) Success Metrics

| Metric | Target |
|--------|--------|
| Tasks completed | 12/12 |
| Unit tests | All pass, 100% pass rate |
| Benchmark thresholds | All 10 hard thresholds met |
| Memory safety | Zero leaks (Zig tracking allocator + Rust Drop) |
| FFI safety | Zero panics crossing boundary |
| Autograd correctness | grad_check passes for attention, layernorm, RoPE |
| Numerical stability | No inf/nan in attention softmax for seq up to 2048 |
| Transformer E2E accuracy | >85% in 5 epochs |
| Transformer throughput | >=2000 tokens/sec |
| Phase 1 regression | All existing tests still pass |
| New lines | ~15,000–20,000 |
| Test coverage | Every new module has unit tests |
| Benchmark coverage | Every perf-critical module has pass/fail benchmark |

---

## 8) Key Architectural Decisions

1. **Pre-norm transformer architecture.** LayerNorm before attention/FFN
   (not after). This is more stable for training and is the modern default
   (GPT-2+, LLaMA, etc.). It also enables LayerNorm+residual fusion since
   the residual add feeds directly into the next norm.

2. **Fused attention kernel in Zig, not Rust.** The inner loop of attention
   (matmul → scale → mask → softmax → matmul) is memory-bandwidth-bound.
   Fusing in Zig with tiled access keeps intermediates in L1/L2 cache.
   Rust calls this as a single FFI function.

3. **RoPE as a separate module, applied inside attention.** RoPE is applied
   to Q and K after projection but before the dot product. It's not a
   positional encoding added to the input — it's a rotation applied to
   query/key pairs. This matches LLaMA/modern architectures.

4. **Separate encoder and decoder.** Encoder-only (BERT-like) and
   encoder-decoder (T5-like) are both supported. Decoder-only (GPT-like)
   is an encoder with causal masking. The decoder adds cross-attention
   to encoder output.

5. **BPE tokenizer is minimal.** Not a production tokenizer — designed for
   demonstrations and synthetic tasks. Real workloads would use an external
   tokenizer (sentencepiece, tiktoken) and feed integer IDs directly.

6. **Graph fusion is pattern-based.** Like Phase 1's fusion passes, the
   attention fusion pass detects specific node patterns in the IR and
   replaces them. It does not attempt general fusion — only the known
   transformer patterns (QKV+attention, layernorm+residual).

7. **Build-mode-aware benchmark thresholds.** Following the pattern
   established in Phase 1 (and my fixes), benchmarks use
   `cfg!(debug_assertions)` to set realistic thresholds for both debug
   and release builds. Debug thresholds are ~10x lower.
