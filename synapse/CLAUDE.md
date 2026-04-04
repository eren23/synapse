# Synapse — Development Guide

## Build

```bash
cargo build --release          # Auto-rebuilds Zig via build.rs
cargo build --release --features metal   # With Metal GPU
```

Zig is rebuilt automatically when `zig/src/**/*.zig` files change. Manual rebuild: `cd zig && zig build -Doptimize=ReleaseFast`.

## Test

```bash
cargo test -p synapse-inference --lib      # 510+ library tests (540+ with --features metal)
cargo test --test multi_model_validation   # 17 multi-architecture tests
cargo test --release                       # All tests (some benchmarks are flaky on timing)
```

Known flaky benchmarks (timing-sensitive, pre-existing):
- `attention_bench_fused_2x_vs_naive` — 1.99x vs 2.0x threshold
- `bench_layernorm_residual_fusion_speedup` — 1.15x vs 1.20x threshold

## Architecture

9 crates under `crates/`:
- **synapse-inference** — Main inference engine (models, generation, quantization, chat templates)
- **synapse-core** — FFI wrappers for Zig ops (Tensor, KvCache)
- **synapse-sys** — Raw C bindings to `libsynapse_zig.a` (auto-rebuild in build.rs)
- **synapse-nn** — Neural network modules (Linear, Embedding, Transformer)
- **synapse-autograd** — Tape-based automatic differentiation
- **synapse-optim** — Optimizers (SGD, Adam) + schedulers
- **synapse-data** — DataLoader, Dataset, Sampler
- **synapse-graph** — Graph IR + optimization passes
- **synapse-train** — Training loop with callbacks

### synapse-inference module layout

```
src/
  engine/               — InferenceEngine (loading, config parsing, orchestration)
  models/
    lm/                 — Autoregressive LLMs (CausalLM, DecoderLayer, ModelBuilder)
    vision/             — Image/world models (ViT, CLIP, JEPA, LeWM, WorldModel)
    ssm/
      mamba/            — Selective State Space Model
      rwkv/             — RWKV-7 time-mixing RNN
      hybrid/           — Generalized hybrid: DeltaNet+GQA (Qwen3.5), LIV Conv+GQA (LFM2.5)
    traits.rs           — Model, ModelState, ModelOutput interfaces
  quantization/
    primitives/         — Q4Linear, QuantizedLinear, TernaryLinear, calibration
    lm/                 — Quantized CausalLM (INT8, ternary)
    ssm/                — Quantized Mamba (INT8, Q4), RWKV (Q4)
    vision/             — Quantized LeWM (INT8, Q4, ternary, full)
  pruning/              — Sensitivity analysis, WANDA, layer removal, SSM pruning
  generation/           — Sampling pipeline, speculative decoding
  registry/             — Factory pattern (attention, FFN, norm, position variants)
  ops/                  — Math kernels (pure Rust + Zig FFI dispatch)
  metal/                — Apple Silicon GPU backend (24 compute shaders + hybrid GPU forward)
  weight_loading/       — safetensors, GGUF (f32 + raw Q4), sharded weights
  config/               — Model config JSON schema
  kv_cache/             — KV cache for transformer decode
```

## Key Files

| File | Role |
|------|------|
| `zig/src/ops/matmul.zig` | f32 SGEMM + GEMV kernel |
| `zig/src/ops/qmatmul.zig` | INT8 + Q4 GEMV kernels |
| `zig/src/ops/attention.zig` | Fused tiled attention |
| `crates/synapse-inference/src/models/lm/decoder_layer.rs` | Core decoder: attention, FFN, RoPE |
| `crates/synapse-inference/src/models/lm/causal_lm.rs` | Full model: forward, prefill, decode |
| `crates/synapse-inference/src/generation/pipeline.rs` | Generation loop, speculative decoding |
| `crates/synapse-inference/src/engine/mod.rs` | High-level InferenceEngine |
| `crates/synapse-inference/src/engine/loading.rs` | Model loading (from_pretrained) |
| `crates/synapse-inference/src/quantization/lm/int8.rs` | INT8 quantized model |
| `crates/synapse-inference/src/weight_loading/` | safetensors + GGUF loading |
| `crates/synapse-inference/src/metal/dispatch.rs` | Metal GPU dispatch |
| `crates/synapse-inference/src/metal/hybrid_gpu_forward.rs` | GPU-resident hybrid forward (LIV Conv + GQA, Q4 GEMV) |
| `crates/synapse-inference/src/metal/hybrid_gpu_buffers.rs` | GPU weight upload + scratch for hybrid models |
| `crates/synapse-inference/src/models/ssm/hybrid/layer.rs` | DeltaNet, GQA, LIV Conv decoder layers |
| `crates/synapse-inference/src/models/ssm/hybrid/model.rs` | HybridModel: forward, weight loading, GPU dispatch |
| `crates/synapse-inference/src/model_adapter.rs` | Model-family adapter (thinking modes, prompt formatting) |
| `scripts/benchmark_matrix.py` | Canonical benchmark matrix runner |

## GPU Architecture (Metal)

The Metal backend lives in `crates/synapse-inference/src/metal/` with this structure:

| File | Role |
|------|------|
| `metal/shaders/*.metal` | 15 Metal compute shaders (matmul, gemv, gemv_int8, **gemv_q4**, rmsnorm, headwise_rmsnorm, silu, swiglu, attention, attention_decode, rope_rotate_half, kv_cache_scatter, **conv1d_step**, lewm_gemv3, plus inline elementwise_mul/add/softmax) |
| `metal/device.rs` | `MetalBackend` struct: detects GPU, compiles all shaders from `SHADER_SOURCE` into `ComputePipelineState` pipelines, indexed by `KERNEL_NAMES` |
| `metal/gpu_forward.rs` | GPU-resident forward pass for CausalLM: encodes all decoder layers into a single Metal command buffer with zero CPU-GPU round-trips |
| `metal/hybrid_gpu_forward.rs` | **GPU-resident forward for hybrid models**: encodes LIV Conv + GQA layers into one command buffer, uses Q4 GEMV when raw Q4 data available |
| `metal/hybrid_gpu_buffers.rs` | **MetalHybridBuffers**: pre-uploads f32 + raw Q4 weights, conv state buffers, KV cache for hybrid models |
| `metal/lewm_forward.rs` | GPU-accelerated LEWM predict_next: encodes all 6 adaLN predictor layers into one command buffer with zero CPU-GPU sync |
| `metal/gpu_buffers.rs` | `MetalModelBuffers`: pre-uploads all weights (f32 + INT8 quantized) to GPU for CausalLM |
| `metal/buffer.rs` | `BufferPool`: reuses Metal buffers by size, caches transposed weights |
| `metal/dispatch.rs` | `ComputeBackend` enum: routes operations to CPU (Zig SIMD) or GPU (Metal) based on matrix size heuristics |
| `metal/mod.rs` | Module exports + 30 tests (shader correctness, dispatch, LEWM GPU, integration) |

### Adding a New Metal Shader

1. Write the `.metal` file in `metal/shaders/` (e.g., `shaders/my_kernel.metal`)
2. Add it to the `SHADER_SOURCE` concatenation in `device.rs`: `include_str!("shaders/my_kernel.metal")`
3. Add the kernel function name to `KERNEL_NAMES` in `device.rs`
4. Access it via `backend.pipeline("my_kernel")` in Rust code
5. Add a correctness test in `metal/mod.rs` following the existing pattern (GPU vs CPU reference comparison)

If the kernel requires specific Metal feature support and may not compile on all hardware, add it to `OPTIONAL_KERNEL_NAMES` instead.

## Adding a New Model

### Standard transformer (CausalLM)
1. Add `WeightMapper::new_model()` in `weight_loading/weight_map.rs`
2. Update `from_model_type()` to recognize the model type string
3. Update `has_head_norms` in `models/lm/builder.rs` if needed
4. Add config JSON in `configs/`
5. Add test in `tests/integration/multi_model_validation.rs`

### Hybrid model (new layer type + GQA)
1. Add a new `LayerKind` variant in `models/ssm/hybrid/config.rs`
2. Implement the layer struct in `models/ssm/hybrid/layer.rs` (follow `LivConvDecoderLayer` pattern)
3. Add the layer to `HybridLayer` enum and wire match arms in `model.rs` (prefill + decode_one)
4. Add `from_weights_<model>()` with the correct weight name mapping
5. Add `from_pretrained_<model>()` in `engine/loading.rs`
6. Add config constructor and test config in `config.rs`

### LFM2.5-350M (GGUF)
```bash
# Download
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('LiquidAI/LFM2.5-350M-GGUF', 'LFM2.5-350M-Q4_0.gguf', cache_dir='models/lfm25-350m')"

# Run inference (CPU)
cargo run --release -p synapse-inference --example lfm25_inference -- \
  (find models/lfm25-350m -name '*.gguf')

# Run inference (Metal GPU with Q4 GEMV)
cargo run --release --features metal -p synapse-inference --example lfm25_inference -- \
  (find models/lfm25-350m -name '*.gguf')

# CPU-only (disable GPU even with metal feature)
NO_Q4_GPU=1 cargo run --release --features metal -p synapse-inference --example lfm25_inference -- ...
```

## Performance

| Model | Backend | Decode tok/s | Notes |
|-------|---------|-------------|-------|
| Qwen3-0.6B INT8 | Metal GPU | 27.3 | GPU-resident, all-layers-in-one-cmd-buffer |
| LFM2.5-350M Q4 | Metal GPU (Q4 GEMV) | **67** | Hybrid LIV Conv+GQA, raw Q4 on GPU |
| LFM2.5-350M Q4 | Accelerate BLAS (CPU) | 48 | f32 dequantized, Apple BLAS |
| LFM2.5-350M Q4 | llama.cpp Metal | 357 | Reference baseline |

Run `scripts/bench_suite.sh` for full comparison.

## Limitations & Experimental Features

### LEWM on ESP32-P4

**Build & Flash** (ESP-IDF C firmware at `synapse-esp32/esp-idf-app/`):
```bash
source ~/.espressif/esp-idf/v5.4/export.sh   # use bash, not fish
cd synapse/synapse-esp32/esp-idf-app
rm -f sdkconfig                               # only needed after sdkconfig.defaults changes
idf.py set-target esp32p4
cat sdkconfig.credentials >> sdkconfig        # REQUIRED: injects WiFi creds (gitignored file)
idf.py build && idf.py -p /dev/cu.usbmodem* flash
```
WiFi credentials are in `sdkconfig.credentials` (gitignored, not tracked). If it doesn't exist, create it:
```
CONFIG_LEWM_WIFI_SSID="YourSSID"
CONFIG_LEWM_WIFI_PASS="YourPassword"
```

**Current performance** (64d hybrid ALAL, 2026-04-03):
- predict_next: 145ms, encode: 817ms, total: 962ms
- Perfectly deterministic (aligned PIE alloc)
- Host↔ESP32 cosine: 0.9999

| Item | Status | Details |
|------|--------|---------|
| `/rollout_fused` up to 50 steps | Working | `MAX_PREDICTOR_SEQ_LEN=150` (50 steps × 3 tokens) |
| Pure Rust path | By design | Slowest path; correctness-first fallback for WASM and `zig-ffi`-disabled builds |
| WANDA pruning | Experimental | Quality degrades at 40%+ sparsity |
| Slim 48d variant | Projected only | Not tested on ESP32 hardware |

### Kernel Dispatch Tiers

| Tier | Path | Use when |
|------|------|---------|
| **Pure Rust** | `ops/pure_rust_ops.rs` | WASM, `zig-ffi`-disabled, correctness-first |
| **Fused Ops** | `ops/fused_ops.rs` | ESP32/WASM with fused layernorm/attention kernels |
| **Zig FFI** | `zig/src/ops/*.zig` | SIMD-accelerated; cross-platform (ESP32, macOS, Linux) |
| **Fused Rollout** | `zig/src/ops/fused_lewm_rollout.zig` | All rollout steps in one pass; flag-controlled optimizations |
| **Apple Accelerate** | BLAS dispatch | Fastest CPU path for matmul on macOS |
| **Metal GPU** | `metal/shaders/*.metal` | GPU-resident decode; Q4 GEMV for bandwidth-bound ops |

### LEWM Rollout Optimization Flags

Set via `model.set_fuse_mode(mode)`. Bitfield flags combine freely:

| Flag | Value | Effect |
|------|-------|--------|
| `FUSED_ROLLOUT` | 0x01 | Batch all N steps as seq_len=N*3 (1.9x) |
| `ESP_FUSED` | 0x02 | Single-pass bias+GELU/residual loops |
| `PREPACK_WEIGHTS` | 0x04 | Pre-pack weight matrices (stub) |
| `BLAS_ACCELERATE` | 0x08 | Apple Accelerate cblas_sgemm on macOS |
| `SHARED_ADALN` | 0x10 | Shared adaLN modulation |
| `QUANT_INT8` | 0x20 | INT8 GEMM dispatch (stub) |
| `QUANT_Q4` | 0x40 | Q4 GEMV dispatch (stub) |

Recommended: `0x1B` (FUSED_ROLLOUT + ESP + BLAS + SHARED_ADALN) = **2.7x** on macOS.

### Known Flaky Benchmarks

- `attention_bench_fused_2x_vs_naive` — 1.99x vs 2.0x threshold
- `bench_layernorm_residual_fusion_speedup` — 1.15x vs 1.20x threshold
