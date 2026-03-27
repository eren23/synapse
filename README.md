# Synapse

Edge-native inference stack built from scratch in Rust + Zig + Metal.

Synapse is a modular local inference engine with native SIMD kernels, optional Metal acceleration, an embeddable C boundary, and a pure-Rust WASM runtime for browser demos. The strongest near-term wedge is local inference across native and browser targets, not generic framework sprawl.

## Positioning Snapshot

<!-- status:root-positioning:start -->
Edge-native inference stack for local ML across native and browser targets.

- Native builds use Rust orchestration with Zig SIMD kernels and optional Metal acceleration.
- Browser builds use a pure-Rust WASM runtime for portability and client-side demos.
- Public benchmark rows are measured on Apple Silicon and synced from status/benchmark_matrix.json.
<!-- status:root-positioning:end -->

## Benchmark Snapshot

<!-- status:root-benchmark:start -->
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
<!-- status:root-benchmark:end -->

## Runtime Profiles

<!-- status:root-profiles:start -->
| Runtime Profile | Support | Targets | Backends | Quantization |
|-----------------|---------|---------|----------|--------------|
| Native Performance | Stable | aarch64-apple-darwin, x86_64-unknown-linux-gnu | cpu_simd, metal | f32, f16, int8, q4_0, q4_k, q6_k, q8_0 |
| ARM Compact | Beta | aarch64-unknown-linux-musl, aarch64-unknown-linux-gnu | cpu_simd | f32, int8, q4_0, q4_k |
| WASM Portable | Stable | wasm32-unknown-unknown | pure_rust_wasm | f32 |
<!-- status:root-profiles:end -->

## Artifact Budgets

<!-- status:root-artifacts:start -->
| Artifact | Current | Budget | Status |
|----------|---------|--------|--------|
| WASM core | ~158 KB | ~160 KB | ok |
| WASM JS wrapper | ~20 KB | ~32 KB | ok |
<!-- status:root-artifacts:end -->

## Architecture

```
synapse/
├── zig/src/ops/           # SIMD kernels (ARM NEON): matmul, RMSNorm, SiLU, INT8, KV-cache
├── crates/
│   ├── synapse-core/      # Core tensor ops, Zig FFI bindings
│   ├── synapse-inference/  # Inference engine
│   │   ├── config/        # Model config (JSON + HuggingFace format parser)
│   │   ├── registry/      # Pluggable components: attention, norm, FFN, position
│   │   ├── model/         # CausalLM, DecoderLayer, ModelBuilder
│   │   ├── generation/    # Pipeline, sampler, stopping conditions
│   │   ├── weight_loading/# Safetensors + GGUF, weight mapping per model
│   │   ├── tokenizer/     # BPE tokenizer (HuggingFace format)
│   │   ├── kv_cache/      # Pre-allocated KV-cache with append/slice
│   │   ├── quantization/  # INT8 per-channel quantization
│   │   └── metal/         # Apple Metal GPU backend (shaders + dispatch)
│   ├── synapse-nn/        # Neural network layers (training)
│   ├── synapse-autograd/  # Automatic differentiation
│   └── synapse-train/     # Training loop, optimizers
├── tests/
│   ├── integration/       # E2E inference, KV-cache, quantization accuracy
│   └── benchmarks/        # Throughput, memory, SIMD vs naive comparisons
├── examples/
│   ├── qwen3_chat.rs      # Interactive chat with real or demo models
│   └── model_benchmark.rs # Benchmark any model via config
└── configs/               # Model configs: Qwen3-0.6B, LLaMA-3.2-1B, Mistral-7B
```

### Component Registry

Every architectural element is a pluggable trait with config-driven instantiation:

| Component | Variants |
|-----------|----------|
| Attention | GQA, MHA, MQA, SlidingWindow |
| Normalization | RMSNorm, LayerNorm |
| FFN | SwiGLU, GELU, GeGLU |
| Position | RoPE, Learned, Sinusoidal |
| Quantization | F32, F16, INT8 |
| Weights | Safetensors, GGUF |

Adding a new model = write its config JSON + weight mapper. No engine changes.

## Quick Start

```bash
# Demo mode (random weights, no downloads)
cargo run --example qwen3_chat --release -- --demo

# Real Qwen3-0.6B (download model first)
# pip install huggingface_hub
# python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='/tmp/qwen3-0.6b')"
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Run benchmarks
cargo run --example model_benchmark --release -- --full-scale

# Run tests
cargo test -p synapse-inference
cargo test --test inference_e2e

# Benchmark vs llama.cpp
./bench_vs_llamacpp.sh
```

## Development Phases

| Phase | Status | What |
|-------|--------|------|
| **Phase 1** | Done | Zig SIMD tensor engine, Rust autograd, training framework (~30k lines) |
| **Phase 2** | Done | Transformer stack, attention kernels, LayerNorm, RoPE (~15k lines) |
| **Phase 3** | Done | Inference engine, component registry, INT8 quantization, Qwen3 support (~14k lines) |
| **Phase 4** | Done | Wire SIMD kernels, KV-cache, Metal GPU shaders, benchmark harness |
| **Phase 4.5** | **TODO** | Wire Metal shaders into forward path, fix output correctness |
| **Phase 5** | Planned | Q4_K block quantization, Flash Attention, simdgroup_matrix, match llama.cpp |

## TODO (Next Steps)

### Correctness (blocking)
- [ ] Debug gibberish output — forward pass produces wrong tokens with real Qwen3 weights
  - Likely: attention masking, RoPE application, or weight loading order bug
  - Test: compare logits at each layer against HuggingFace reference implementation

### Performance (Phase 4.5 — tighten native path)
- [ ] Improve Metal/native path consistency with the published benchmark surface
- [ ] Add reproducible benchmark generation so public numbers come from a single manifest
- [ ] Keep browser and native claims explicitly separated in docs and site copy
- [ ] Target: make the edge/native story coherent before chasing broad parity claims

### Multi-Model Support (Phase 3.5)
- [ ] Generic weight mappers for LLaMA 3.2, Mistral 7B, Phi-3
- [ ] SentencePiece tokenizer (for LLaMA/Mistral)
- [ ] Config parser: dynamic norm/FFN detection (LayerNorm for Phi-3, GELU FFN)
- [ ] Sliding window attention kernel (Mistral)
- [ ] Engine auto-detection from HuggingFace config.json model_type field

### Performance (Phase 5 — match llama.cpp)
- [ ] Q4_K block quantization (4-bit weights, ~4x less memory bandwidth)
- [ ] Metal simdgroup_matrix (hardware matrix multiply on M-series)
- [ ] Flash Attention with tiled online softmax
- [ ] Kernel fusion (RMSNorm + matmul, attention + softmax in single dispatch)
- [ ] GGUF native inference (skip safetensors → f32 conversion)

## Built With

- **Rust** — inference engine, autograd, training framework
- **Zig** — SIMD kernels (ARM NEON), custom allocators, FFI exports
- **Metal Shading Language** — GPU compute shaders for Apple Silicon
- **Swarm development** — Phases 1–4 built using [attoswarm](https://github.com/attocode) parallel agent orchestration
