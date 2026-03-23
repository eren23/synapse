# Synapse

High-performance LLM inference engine built from scratch in Rust + Zig + Metal.

Synapse is a modular inference engine with SIMD-vectorized kernels (Zig/ARM NEON), Apple Metal GPU compute shaders, INT8 quantization, KV-cache, and a pluggable component registry that supports multiple model architectures from a single engine.

**~53,000 lines** across Rust (36k), Zig (17k), and Metal (300).

## Current State

Reference model: **Qwen3-0.6B** (596M params, loaded from HuggingFace safetensors).

### Benchmark (Apple M4, 2026-03-23)

| Metric | Synapse (CPU+SIMD) | llama.cpp (Metal) |
|--------|:------------------:|:-----------------:|
| Prefill tok/s (pp128) | 86 | 5,368 |
| Decode tok/s (KV-cache) | 2.5 | 82 |
| INT8 decode tok/s | 1.4 | — |
| Model memory (f32) | 1,938 MB | 1,138 MB (BF16) |

Gap is ~60x — primarily because Metal GPU shaders are built but not yet wired into the forward path. CPU SIMD path went from 5 to 86 tok/s prefill after Phase 4 optimization.

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

### Performance (Phase 4.5 — wire Metal into hot path)
- [ ] Route decoder layer matmuls through Metal GPU backend (shaders already written)
- [ ] Metal dispatch heuristic: large matrices → GPU, small → CPU SIMD
- [ ] Double-buffered Metal pipeline (overlap compute layer N with data transfer N+1)
- [ ] Target: 20+ tok/s decode, 500+ tok/s prefill

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
