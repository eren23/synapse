# Swarm Goal

Synapse Phase 4 — Performance Optimization: Wire SIMD Kernels, KV-Cache, Metal GPU

Close the ~1000x performance gap between Synapse and llama.cpp on Qwen3-0.6B.
Current state (2026-03-23 benchmarks on Apple M4):

| Metric | Synapse (f32, CPU) | llama.cpp (BF16, Metal) | llama.cpp (Q4_K_M, Metal) |
|--------|-------------------|------------------------|--------------------------|
| Prefill tok/s (pp128) | **~5** | 5,368 | 5,518 |
| Decode tok/s (tg64) | **~0.3** | 82 | 173 |
| Model size | 1,938 MB | 1,138 MB | 373 MB |

The gap is caused by three compounding problems:
1. **Zig SIMD kernels exist but are never called** — the inference hot path uses
   naive triple-nested Rust loops (`matmul_t()`) instead of `syn_sgemm()` FFI
2. **No KV-cache in generation** — the decode loop re-runs the full forward pass
   on ALL tokens every step (quadratic complexity)
3. **No GPU offload** — llama.cpp uses Metal on M4, Synapse is CPU-only

Phase 4 wires up the existing SIMD kernels, implements proper KV-cache,
adds Apple Metal GPU compute shaders, and makes INT8 quantization actually
fast (currently zero speedup because dequantization happens in the inner loop).

**Builds on top of Synapse Phase 3 (inference engine, component registry,
weight loading, generation pipeline).** Modifies existing code — few new
files, mostly rewiring hot paths.

**CRITICAL RULE: Every task MUST include its own tests. No implementation
without tests. Every benchmark MUST have a hard pass/fail threshold. If a
benchmark does not meet its threshold, the task FAILS.**

---

## 0) Phase 4 Overview

### Root Cause Analysis: Why Synapse Is ~1000x Slower

The Phase 3 inference engine built correct abstractions but with placeholder
implementations in the hot path. Here is exactly what happens when you call
`engine.generate()`:

#### Current Hot Path (SLOW)

```
engine.generate("Hello")
  → pipeline.generate(tokens, config)
    → LOOP for each new token:                    ← No KV-cache: O(n²) total work
        model.forward(ALL_TOKENS_SO_FAR)          ← Reprocesses entire prefix
          → for each layer:
              norm (pure Rust scalar loop)         ← Should call syn_rmsnorm_forward()
              matmul_t(x, w_q) for Q projection   ← NAIVE TRIPLE-NESTED RUST LOOP
              matmul_t(x, w_k) for K projection   ← Should call syn_sgemm()
              matmul_t(x, w_v) for V projection   ← Should call syn_sgemm()
              attention scores (nested Rust loop)  ← Should use fused attention kernel
              matmul_t(attn, w_o) for output       ← Should call syn_sgemm()
              norm (pure Rust scalar loop)
              matmul_t(x, w_gate) for FFN gate     ← Should call syn_sgemm()
              matmul_t(x, w_up) for FFN up         ← Should call syn_sgemm()
              silu activation (scalar)              ← Should call syn_silu() / syn_swiglu()
              matmul_t(hidden, w_down) for FFN down ← Should call syn_sgemm()
          → final_norm (scalar)
          → matmul_t(hidden, lm_head)              ← Should call syn_sgemm()
        extract last logit
        sample token
        append to ALL_TOKENS
```

**9 matmuls per layer × 28 layers = 252 matmuls per forward pass, ALL using
naive Rust loops.** Plus no KV-cache means this entire forward pass runs on
the full sequence every decode step.

#### Target Hot Path (FAST)

```
engine.generate("Hello")
  → pipeline.prefill(tokens)                       ← Process prompt once
      → for each layer:
          syn_rmsnorm_forward() via FFI             ← Zig SIMD
          syn_sgemm() for Q/K/V projections         ← Zig SIMD tiled GEMM
          syn_attention_forward() via FFI            ← Zig fused attention
          syn_sgemm() for output projection          ← Zig SIMD
          syn_rmsnorm_forward() via FFI
          syn_swiglu() for fused FFN                ← Zig fused SwiGLU
          syn_sgemm() for down projection            ← Zig SIMD
          kv_cache.append(k, v)                     ← Store K/V for reuse
  → LOOP for each new token:                        ← KV-cache: O(n) per step
      model.forward_one(new_token, kv_cache)       ← Only process 1 token
        → for each layer:
            (same SIMD ops but on single token)
            kv_cache.append(k, v)                   ← Append 1 K/V entry
            attention against full cached K/V        ← Read from cache, not recompute
```

### What Already Exists (DO NOT REWRITE)

Phase 4 builds on Phases 1–3. The following are already implemented:

**Zig SIMD kernels (EXIST, need to be WIRED IN):**
- `zig/src/ops/matmul.zig` — Tiled SGEMM with SIMD, cache-blocking
- `zig/src/ops/rmsnorm.zig` — SIMD RMSNorm with 2x unrolled loops
- `zig/src/ops/silu.zig` — SIMD SiLU + fused SwiGLU
- `zig/src/ops/quantize.zig` — SIMD INT8 quantize/dequantize
- `zig/src/ops/qmatmul.zig` — INT8 quantized GEMM
- `zig/src/ops/kvcache.zig` — KV-Cache append/slice operations
- `zig/src/ops/attention.zig` — Fused attention kernel (from Phase 2)

**Zig FFI bindings (EXIST, need to be CALLED):**
- `crates/synapse-sys/src/lib.rs` — C ABI declarations:
  - `syn_sgemm()` — SIMD matrix multiply
  - `syn_rmsnorm_forward()` — SIMD RMSNorm
  - `syn_silu()` — SIMD SiLU activation
  - `syn_swiglu()` — Fused SwiGLU kernel
  - `syn_qgemm_int8()` — Quantized INT8 matmul
  - `syn_kvcache_create/append/slice()` — KV-cache ops
  - `syn_attention_forward()` — Fused attention

**Rust inference engine (MODIFY, don't rewrite):**
- `synapse-inference` crate — all config, registry, model, generation code
- Component registry with trait-based dispatch
- Weight loading (safetensors + GGUF)
- Generation pipeline with sampling strategies

### What's New in Phase 4

Phase 4 does NOT add new architecture — it makes existing architecture fast:

1. **Wire Zig SIMD kernels into the decoder layer hot path**
2. **Implement KV-cache in the generation decode loop**
3. **Fix INT8 quantization to use vectorized `syn_qgemm_int8()`**
4. **Add Apple Metal GPU compute shaders** for matmul + attention
5. **Memory-map weights** instead of copying into Vec

---

## 1) Architecture

```
Phase 4 changes (PERFORMANCE)
─────────────────────────────────────────────────────────────
│                                                             │
│  Inference Hot Path (REWIRE)                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DecoderLayer.forward()                              │    │
│  │    BEFORE: matmul_t() (naive Rust)                   │    │
│  │    AFTER:  syn_sgemm() (Zig SIMD FFI)               │    │
│  │                                                       │    │
│  │    BEFORE: scalar norm loops                          │    │
│  │    AFTER:  syn_rmsnorm_forward() (Zig SIMD FFI)     │    │
│  │                                                       │    │
│  │    BEFORE: scalar activation                          │    │
│  │    AFTER:  syn_swiglu() (Zig fused SIMD FFI)        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Generation Pipeline (REWIRE)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  BEFORE: model.forward(ALL_TOKENS) every step        │    │
│  │  AFTER:  prefill once → forward_one(token) + cache   │    │
│  │                                                       │    │
│  │  KV-Cache integration:                                │    │
│  │    prefill: store all K/V in cache                   │    │
│  │    decode:  append 1 K/V, attend against full cache  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Metal GPU Backend (NEW)                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  metal/                                               │    │
│  │    ├── device.rs     # MTLDevice, command queues      │    │
│  │    ├── buffer.rs     # GPU buffer management          │    │
│  │    ├── shaders/                                       │    │
│  │    │   ├── matmul.metal    # Tiled GEMM shader       │    │
│  │    │   ├── rmsnorm.metal   # RMSNorm shader          │    │
│  │    │   ├── attention.metal # Fused attention shader   │    │
│  │    │   └── silu.metal      # SiLU/SwiGLU shader      │    │
│  │    └── dispatch.rs   # CPU↔GPU routing based on size  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  INT8 Quantized Inference (FIX)                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  BEFORE: i8→f32 cast in inner loop (zero speedup)    │    │
│  │  AFTER:  syn_qgemm_int8() (Zig SIMD INT8 GEMM)     │    │
│  │         + Metal INT8 shader path                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Weight Loading (OPTIMIZE)                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  BEFORE: mmap → copy ALL into Vec<f32>               │    │
│  │  AFTER:  mmap → keep as &[u8] slice, convert lazily  │    │
│  │         (or convert once into aligned buffer)         │    │
│  └─────────────────────────────────────────────────────┘    │
─────────────────────────────────────────────────────────────
```

---

## 2) Files to Modify

```
synapse/
├── crates/synapse-inference/src/
│   ├── model/
│   │   ├── decoder_layer.rs     # REWIRE: replace matmul_t() with syn_sgemm() FFI
│   │   │                        # REWIRE: replace scalar norm with syn_rmsnorm_forward()
│   │   │                        # REWIRE: replace scalar activation with syn_swiglu()
│   │   │                        # ADD: forward_one() for single-token decode with KV-cache
│   │   └── causal_lm.rs        # ADD: forward_one() that delegates to layer.forward_one()
│   │
│   ├── generation/
│   │   └── pipeline.rs          # REWIRE: decode loop to use KV-cache + forward_one()
│   │                            # ADD: prefill() stores K/V in cache
│   │                            # ADD: decode_step() processes single token
│   │
│   ├── kv_cache/
│   │   ├── cache.rs             # MODIFY: wire to syn_kvcache_append/slice FFI
│   │   └── strategy.rs          # ADD: CacheStrategy for sliding window eviction
│   │
│   ├── quantization/
│   │   └── quantized_linear.rs  # REWIRE: replace scalar i8→f32 loop with syn_qgemm_int8()
│   │
│   ├── weight_loading/
│   │   └── safetensors.rs       # OPTIMIZE: keep mmap'd data, avoid Vec<f32> copy
│   │
│   ├── metal/                   # NEW DIRECTORY
│   │   ├── mod.rs               # Metal backend toggle, feature-gated
│   │   ├── device.rs            # MTLDevice wrapper, command queue, pipeline state
│   │   ├── buffer.rs            # GPU buffer pool, CPU↔GPU transfer
│   │   ├── dispatch.rs          # Backend dispatcher: CPU (Zig SIMD) vs GPU (Metal)
│   │   │                        # Heuristic: matrices > threshold → GPU, else CPU
│   │   └── shaders/
│   │       ├── matmul.metal     # Tiled GEMM (threadgroup shared memory)
│   │       ├── rmsnorm.metal    # Parallel RMSNorm (reduction + normalize)
│   │       ├── attention.metal  # Fused QKV attention + softmax
│   │       └── silu.metal       # SiLU + fused SwiGLU elementwise
│   │
│   ├── engine.rs                # ADD: backend selection (CPU-SIMD vs Metal)
│   └── lib.rs                   # EXPORT: metal module behind feature flag
│
├── tests/
│   ├── integration/
│   │   ├── kvcache_decode.rs    # NEW: verify KV-cache decode matches full recompute
│   │   └── metal_correctness.rs # NEW: Metal vs CPU output comparison (max err < 1e-5)
│   │
│   └── benchmarks/
│       ├── simd_vs_naive.rs     # NEW: syn_sgemm vs matmul_t throughput comparison
│       ├── kvcache_speedup.rs   # NEW: cached decode vs recompute speedup
│       └── metal_throughput.rs  # NEW: Metal GPU throughput on Qwen3-0.6B
│
├── Cargo.toml                   # ADD: metal-rs dependency (feature-gated)
│
└── bench_vs_llamacpp.sh         # UPDATE: include Synapse Phase 4 numbers
```

---

## 3) Component Details

### 3a) Wire Zig SIMD into Decoder Layer

The critical change: replace `matmul_t()` in `decoder_layer.rs` with FFI
calls to the existing Zig SIMD kernels.

**Current code** (`decoder_layer.rs` lines 262-279):
```rust
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += a_row[d] * b_row[d];  // NAIVE: no SIMD, no tiling, no cache blocking
            }
            out[i * n + j] = sum;
        }
    }
    out
}
```

**Target code**:
```rust
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    unsafe {
        synapse_sys::syn_sgemm(
            out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            m, n, k,
            false,  // a not transposed
            true,   // b transposed (weights stored as [out_dim, in_dim])
        );
    }
    out
}
```

This is a **drop-in replacement** — same function signature, same semantics,
but calls the Zig SIMD tiled GEMM instead of scalar loops.

Similarly for norm and activations:
```rust
// Replace scalar RMSNorm with:
fn apply_rmsnorm(out: &mut [f32], input: &[f32], weight: &[f32], eps: f32, hidden_size: usize) {
    unsafe {
        synapse_sys::syn_rmsnorm_forward(
            out.as_mut_ptr(), input.as_ptr(), weight.as_ptr(), eps,
            input.len() / hidden_size, hidden_size,
        );
    }
}

// Replace scalar SwiGLU with:
fn apply_swiglu(out: &mut [f32], gate: &[f32], up: &[f32], len: usize) {
    unsafe {
        synapse_sys::syn_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len);
    }
}
```

**Tests:**
- SIMD matmul output matches naive matmul_t within 1e-5 relative error
- SIMD RMSNorm output matches scalar within 1e-5
- SIMD SwiGLU output matches scalar within 1e-5
- Full model forward pass: SIMD output matches naive output within 1e-4

**Benchmark (HARD thresholds):**
- `syn_sgemm` ≥ 4× throughput vs `matmul_t` on [1024, 1024] × [1024, 3072]
- `syn_rmsnorm_forward` ≥ 4× throughput vs scalar on [1, 1024]
- `syn_swiglu` ≥ 2× throughput vs separate silu + elementwise mul

---

### 3b) KV-Cache Integration in Generation Pipeline

The generation decode loop currently reprocesses ALL tokens every step.
This must be changed to: prefill once, then decode one token at a time
with KV-cache.

**Current code** (`generation/pipeline.rs` lines ~115-146):
```rust
while !stop_checker.should_stop(...) {
    let output = self.model.forward(&all_tokens);  // RE-RUNS FULL SEQUENCE
    let last_logits = &output.logits[(seq_len - 1) * vocab_size..];
    all_tokens.push(sampled_token);
}
```

**Target code**:
```rust
// 1. Prefill: process entire prompt, populate KV-cache
let mut kv_cache = KVCache::new(config.max_seq_len, &model.config);
let logits = self.model.forward_prefill(&prompt_tokens, &mut kv_cache);

// 2. Decode: one token at a time, read+append KV-cache
let mut next_token = sample(logits);
while !stop_checker.should_stop(...) {
    let logits = self.model.forward_one(next_token, &mut kv_cache);
    next_token = sample(logits);
    generated.push(next_token);
}
```

**New methods needed:**
- `CausalLM::forward_prefill(tokens: &[u32], cache: &mut KVCache) -> Logits`
  - Processes full prompt, stores all K/V in cache
- `CausalLM::forward_one(token: u32, cache: &mut KVCache) -> Logits`
  - Processes single token, appends K/V to cache, attends against full cache
- `DecoderLayer::forward_one(hidden: &[f32], cache: &mut KVCacheLayer, pos: usize) -> Vec<f32>`
  - Same as forward but for seq_len=1, reads cached K/V for attention

**KV-cache structure** (already exists in `kv_cache/cache.rs`, needs wiring):
```rust
pub struct KVCache {
    layers: Vec<KVCacheLayer>,  // One per decoder layer
}

pub struct KVCacheLayer {
    key: Vec<f32>,    // [max_seq, n_kv_heads, head_dim] pre-allocated
    value: Vec<f32>,  // [max_seq, n_kv_heads, head_dim]
    seq_len: usize,   // Current number of cached entries
}
```

**Tests:**
- Generate 20 tokens with KV-cache: output token IDs IDENTICAL to
  full-recompute generation (bit-for-bit deterministic)
- KV-cache values at each position match full forward pass K/V at same position
- KV-cache memory: exactly `2 * num_layers * max_seq * n_kv_heads * head_dim * 4` bytes

**Benchmark (HARD thresholds):**
- Decode throughput with KV-cache ≥ 10× vs without cache (at 64 generated tokens)
- Prefill throughput unchanged (±10%) vs current

---

### 3c) Fix INT8 Quantized Inference

Current INT8 gives ZERO speedup because it dequantizes i8→f32 in the
inner loop. Replace with the Zig `syn_qgemm_int8()` which does true
INT8 matmul with INT32 accumulation.

**Current code** (`quantization/quantized_linear.rs` lines 72-94):
```rust
for d in 0..k {
    sum += x_row[d] * w_row[d] as f32;  // CASTS i8→f32 EVERY ELEMENT
}
out[i * n + j] = sum * self.scales[j];
```

**Target code**:
```rust
unsafe {
    synapse_sys::syn_qgemm_int8(
        out.as_mut_ptr(),
        x_quantized.as_ptr(),   // INT8 input (quantize x on the fly)
        self.weights_int8.as_ptr(),
        self.scales_x.as_ptr(),
        self.scales_w.as_ptr(),
        m, n, k,
    );
}
```

**Tests:**
- INT8 matmul output vs f32 matmul: top-1 token agreement ≥ 99%
- Max absolute error ≤ 0.5 for logits

**Benchmark (HARD thresholds):**
- INT8 decode throughput ≥ 1.5× vs f32 decode throughput
- INT8 memory ≤ 50% of f32 memory

---

### 3d) Apple Metal GPU Backend

Add Metal compute shader support for the heaviest operations. This is
the single biggest potential speedup (100x+) since llama.cpp's Metal
backend is what makes it fast.

**Architecture:**
- Feature-gated: `#[cfg(feature = "metal")]`
- Dependency: `metal-rs` crate for MTLDevice/MTLBuffer/MTLComputePipeline
- Dispatch heuristic: matrices larger than threshold → GPU, smaller → CPU (Zig SIMD)

**Metal shaders needed (MSL — Metal Shading Language):**

1. **matmul.metal** — Tiled GEMM using threadgroup shared memory
   ```metal
   kernel void matmul_tiled(
       device const float* A [[buffer(0)]],
       device const float* B [[buffer(1)]],
       device float* C [[buffer(2)]],
       constant uint& M [[buffer(3)]],
       constant uint& N [[buffer(4)]],
       constant uint& K [[buffer(5)]],
       uint2 gid [[threadgroup_position_in_grid]],
       uint2 tid [[thread_position_in_threadgroup]]
   ) {
       // Tile size 32x32, shared memory blocking
       threadgroup float As[32][32];
       threadgroup float Bs[32][32];
       // ... tiled matmul with SIMD group operations
   }
   ```

2. **rmsnorm.metal** — Parallel reduction + normalize
3. **attention.metal** — Fused scaled dot-product attention
4. **silu.metal** — Elementwise SiLU + fused SwiGLU

**Rust integration:**
```rust
pub struct MetalBackend {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<&'static str, metal::ComputePipelineState>,
    buffer_pool: BufferPool,
}

impl MetalBackend {
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let buf_a = self.buffer_pool.get_or_create(a);
        let buf_b = self.buffer_pool.get_or_create(b);
        let buf_c = self.buffer_pool.create_empty(m * n);

        let encoder = self.queue.new_command_buffer().compute_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines["matmul_tiled"]);
        encoder.set_buffer(0, &buf_a);
        encoder.set_buffer(1, &buf_b);
        encoder.set_buffer(2, &buf_c);
        // ... dispatch threadgroups
        encoder.end_encoding();
        // ...
    }
}
```

**Backend dispatcher:**
```rust
pub enum ComputeBackend {
    CpuSimd,         // Zig SIMD via FFI (default)
    Metal(MetalBackend),  // Apple GPU (feature-gated)
}

impl ComputeBackend {
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            Self::CpuSimd => {
                let mut out = vec![0.0f32; m * n];
                unsafe { synapse_sys::syn_sgemm(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, n, k, false, true); }
                out
            }
            Self::Metal(metal) => metal.matmul(a, b, m, k, n),
        }
    }
}
```

**Tests:**
- Metal matmul output matches CPU output within 1e-4 relative error
- Full forward pass: Metal vs CPU max logit difference < 0.01
- Buffer pool: no GPU memory leaks after 100 forward passes

**Benchmark (HARD thresholds):**
- Metal matmul ≥ 50× throughput vs CPU matmul_t on [1024, 1024] × [1024, 3072]
- Metal full decode ≥ 20 tok/s on Qwen3-0.6B (from current 0.3)
- Metal prefill ≥ 500 tok/s on Qwen3-0.6B pp128 (from current 5)

---

### 3e) Weight Loading Optimization

Current safetensors loader mmaps the file then immediately copies ALL
data into `Vec<f32>`. For a 1.5GB model this wastes ~1.5GB of heap.

**Options (choose based on complexity):**

**Option A (minimal): Aligned copy, no double-alloc**
- Mmap file → convert weights directly into SIMD-aligned buffers
- Avoid the intermediate `bytes_to_f32()` → `Vec<f32>` copy chain
- Use aligned allocation (`std::alloc::alloc` with 64-byte alignment) for
  cache-line-friendly access in SIMD kernels

**Option B (advanced): Keep weights mmap'd**
- For f32 safetensors: keep the mmap, cast `&[u8]` → `&[f32]` directly
- For bf16/f16: convert once into aligned buffer
- Requires all weight consumers to accept `&[f32]` slices, not own `Vec<f32>`

**Tests:**
- Model loads correctly with new loading path
- Weight values identical to current loading

**Benchmark (HARD thresholds):**
- Model load time ≤ 80% of current (for f32 safetensors)
- Peak RSS during loading ≤ 1.2× model size (currently ~2× due to copy)

---

## 4) Optimization Targets

**These are HARD pass/fail thresholds. If a benchmark does not meet its
target, the task FAILS.**

### CPU-SIMD Targets (Zig FFI wiring)

| # | What | Metric | Threshold | How to Measure |
|---|------|--------|-----------|----------------|
| 1 | SGEMM | syn_sgemm vs matmul_t | **≥4×** on [1024,1024]×[1024,3072] | `simd_vs_naive.rs` |
| 2 | RMSNorm | syn_rmsnorm vs scalar | **≥4×** on [1, 1024] | Same benchmark |
| 3 | SwiGLU | syn_swiglu vs separate | **≥2×** on [1, 3072] | Same benchmark |
| 4 | Prefill | Synapse pp128 | **≥50 tok/s** (from ~5) | `prefill_throughput.rs` |
| 5 | Decode (f32) | Synapse tg64 with KV-cache | **≥3 tok/s** (from ~0.3) | `inference_throughput.rs` |

### KV-Cache Targets

| # | What | Metric | Threshold | How to Measure |
|---|------|--------|-----------|----------------|
| 6 | Decode speedup | Cached vs recompute | **≥10×** at 64 tokens generated | `kvcache_speedup.rs` |
| 7 | Correctness | Cached vs full tokens | **Bit-exact** token IDs | `kvcache_decode.rs` |
| 8 | Memory | Cache memory overhead | **≤ 50 MB** for Qwen3-0.6B 2048 ctx | Same benchmark |

### INT8 Targets

| # | What | Metric | Threshold | How to Measure |
|---|------|--------|-----------|----------------|
| 9 | INT8 decode | INT8 vs f32 throughput | **≥1.5×** speedup | `inference_throughput.rs` |
| 10 | INT8 accuracy | Top-1 agreement with f32 | **≥99%** on 100 tokens | `quantization_accuracy.rs` |

### Metal GPU Targets

| # | What | Metric | Threshold | How to Measure |
|---|------|--------|-----------|----------------|
| 11 | Metal matmul | GPU vs CPU throughput | **≥50×** on [1024,1024]×[1024,3072] | `metal_throughput.rs` |
| 12 | Metal decode | Qwen3-0.6B tok/s | **≥20 tok/s** | Same benchmark |
| 13 | Metal prefill | Qwen3-0.6B pp128 | **≥500 tok/s** | Same benchmark |
| 14 | Metal correctness | GPU vs CPU max error | **≤1e-4** relative | `metal_correctness.rs` |

### Overall Target

| # | What | Metric | Threshold |
|---|------|--------|-----------|
| 15 | Full pipeline (Metal) | Qwen3-0.6B decode | **≥30 tok/s** |
| 16 | Full pipeline (CPU-SIMD) | Qwen3-0.6B decode | **≥5 tok/s** |
| 17 | llama.cpp gap | Synapse/llama.cpp ratio (Metal) | **≤5×** (from ~270×) |

Use `cfg!(debug_assertions)` for debug-mode thresholds at ~5× lower values.

---

## 5) Task Dependency Graph

```
Task A: Wire Zig SIMD into decoder layer  ─────┐
Task B: KV-cache in generation pipeline   ──────┼── Task E: Integration + benchmarks
Task C: Fix INT8 with syn_qgemm_int8     ──────┤
Task D: Apple Metal GPU backend           ──────┘
Task F: Weight loading optimization       ─────── (independent, can run parallel)
```

**Parallel group 1** (no deps): Tasks A, B, C, D, F — all independent
**Serial**: Task E depends on A + B + C + D

Estimated task count when decomposed: ~10-12 tasks from these 5 deliverables.

---

## 6) Verification Plan

After all tasks complete:

1. `cargo test -p synapse-inference` — all tests pass
2. `cargo test --test kvcache_decode` — cached output matches full recompute
3. `cargo test --test metal_correctness` — Metal output matches CPU
4. `cargo run --example model_benchmark --release -- --full-scale` — shows improved numbers
5. `cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b` — interactive chat works at usable speed
6. `./bench_vs_llamacpp.sh` — Synapse within 5× of llama.cpp (Metal path)
7. All existing Phase 3 tests still pass (regression)

---

## 7) Risk Mitigation

**Risk: Zig FFI function signatures don't match expected shapes**
- Mitigation: Read `synapse-sys/src/lib.rs` and `synapse.h` carefully.
  Write shim functions if signatures need adaptation.

**Risk: Metal shaders are complex to debug**
- Mitigation: Test each shader independently with known inputs before
  integrating into the forward pass. Use Metal debugging tools.

**Risk: KV-cache changes break existing generation**
- Mitigation: Keep the old `forward()` method intact. Add `forward_one()`
  as a new method. Pipeline can fallback to old path.

**Risk: Mmap weight loading breaks on non-aligned data**
- Mitigation: Always copy into aligned buffer for SIMD. Only skip copy
  for f32 safetensors on aligned mmaps.
