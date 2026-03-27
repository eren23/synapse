# Synapse

**Edge-native inference stack built from scratch in Rust + Zig + Metal.**

Modular local inference engine with Zig SIMD kernels, Metal GPU acceleration, a pure-Rust WASM runtime for browser delivery, and an ESP32-P4 target for microcontroller inference. One codebase, three deployment targets.

## Benchmarks

| Model | Quantization | Prefill (tok/s) | Decode (tok/s) |
|-------|-------------|-----------------|----------------|
| Qwen3-0.6B | f32 | 11 | 7.3 |
| Qwen3-0.6B | INT8 | 23 | **27.3** |
| LLaMA 3.2-1B | f32 | 1 | 2.1 |
| LLaMA 3.2-1B | INT8 | 8 | 9.7 |

> Measured end-to-end on Apple Silicon. Full matrix in [`synapse/status/benchmark_matrix.md`](synapse/status/benchmark_matrix.md).

## Deployment Targets

| Target | Backend | Quantization | Use Case |
|--------|---------|-------------|----------|
| **Desktop** | Zig SIMD (NEON/AVX2) + Metal GPU | f32, f16, INT8, Q4 | Development, serving |
| **Browser** | Pure Rust (WASM) | f32, INT8 | Embeddable widget, client-side demos |
| **ESP32-P4** | Pure Rust (RISC-V) | INT8, Q4 | Edge inference via WiFi HTTP |

WASM budget: ~160KB core + ~32KB JS wrapper.

## Supported Models

| Model Family | Type | Status |
|--------------|------|--------|
| **Qwen3** | LLM (GQA) | Validated — benchmarked, logits verified |
| **LLaMA 3.2** | LLM (GQA) | Validated — benchmarked locally |
| **Mistral 7B** | LLM (Sliding Window) | Config ready — synthetic tests passing |
| **Phi-3** | LLM (GQA) | In progress |
| **Gemma** | LLM (MHA, GeGLU) | Config ready — synthetic tests passing |
| **ViT** | Vision | Validated |
| **CLIP** | Vision+Text | Supported |
| **DINOv2** | Vision | Supported |
| **LEWM** | World Model | Validated — runs on all 3 targets |

Adding a new model = write a config JSON + weight mapper. No engine changes.

## Component Registry

Every architectural element is a pluggable trait with config-driven instantiation:

| Component | Variants |
|-----------|----------|
| Attention | GQA, MHA, MQA, SlidingWindow |
| Normalization | RMSNorm, LayerNorm |
| FFN | SwiGLU, GELU, GeGLU |
| Position | RoPE, Learned, Sinusoidal |
| Quantization | f32, f16, INT8, Q4_0, Q4_K, Q6_K, Q8_0 |
| Weights | safetensors, GGUF |

## Quick Start

```bash
# Build (Zig kernels auto-rebuild)
cd synapse && cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Chat
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Chat with INT8 quantization
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize

# With Metal GPU (macOS)
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b --quantize

# Demo mode (random weights, no downloads)
cargo run --example qwen3_chat --release -- --demo

# Build for browser
wasm-pack build -p synapse-wasm --release

# Build for ESP32
cargo build -p synapse-esp32
```

## World Models (LEWM)

Latent Emergent World Model — ViT encoder + DiT predictor for latent state prediction.

| Operation | Latency (Apple Silicon) |
|-----------|------------------------|
| Encode (224x224 -> 192d) | 26.9ms |
| Predict (single step) | 12.8ms |
| Rollout (50 steps) | 609ms |

- **Browser**: 69MB checkpoint, interactive trajectory rollouts (`synapse/web/index.html`)
- **ESP32-P4**: Phone camera -> WiFi HTTP -> LEWM inference -> JSON response
- **Quantization**: INT8 (~4x smaller), Q4 (~6.4x compression, ~7MB weights)

## Architecture

```
synapse/
├── crates/
│   ├── synapse-inference/    # Models, generation, quantization, chat templates
│   │   ├── model/            # CausalLM, DecoderLayer, ModelBuilder
│   │   ├── generation/       # Pipeline, sampler, speculative decoding
│   │   ├── weight_loading/   # safetensors + GGUF, per-model weight mappers
│   │   ├── tokenizer/        # BPE tokenizer (HuggingFace format)
│   │   ├── kv_cache/         # Pre-allocated KV cache
│   │   ├── quantization/     # INT8 per-channel quantization
│   │   ├── metal/            # Metal GPU backend (13 shaders, zero-roundtrip forward)
│   │   ├── lewm/             # World model (ViT encoder + DiT predictor)
│   │   └── diffusion/        # Diffusion pipeline (scaffolding)
│   ├── synapse-core/         # FFI wrappers for Zig tensor ops
│   ├── synapse-sys/          # Raw C bindings (auto-rebuild via build.rs)
│   ├── synapse-nn/           # Neural network modules
│   ├── synapse-autograd/     # Tape-based autodiff
│   ├── synapse-optim/        # SGD, Adam, RMSProp + schedulers
│   ├── synapse-data/         # DataLoader, Dataset, Sampler
│   ├── synapse-graph/        # Graph IR + optimization passes
│   └── synapse-train/        # Training loop + callbacks
├── synapse-wasm/             # Browser WASM runtime (pure Rust, zero FFI)
├── synapse-esp32/            # ESP32-P4 edge target (WiFi HTTP server)
├── zig/src/ops/              # SIMD kernels: matmul, qmatmul, attention, RoPE, RMSNorm
├── configs/                  # Model configs (Qwen3, LLaMA, Mistral, Phi-3, Gemma)
├── scripts/                  # Benchmark suite + logit verification
└── web/                      # Browser LEWM demo
```

## Testing

```bash
cargo test -p synapse-inference --lib      # 332 unit tests
cargo test --test multi_model_validation   # 17 multi-architecture tests
cargo test --release                       # Full suite including benchmarks
```

## Development History

| Phase | What |
|-------|------|
| 1 | Zig SIMD tensor engine, Rust autograd, training framework |
| 2 | Transformer stack, attention kernels, RoPE |
| 3 | Inference engine, component registry, INT8 quantization, Qwen3 |
| 4 | SIMD kernel wiring, KV cache, Metal GPU shaders |
| 5 | Multi-model support (LLaMA, Mistral, Phi-3, Gemma), GGUF loading, Q4 quantization |
| 6 | LEWM world models, WASM runtime, ESP32 target, speculative decoding |

## Built With

- **Rust** — inference engine, autograd, training framework
- **Zig** — SIMD kernels (ARM NEON + AVX2), C ABI FFI
- **Metal Shading Language** — GPU compute shaders for Apple Silicon
- **Swarm development** — built using [attoswarm](https://github.com/attocode) parallel agent orchestration

## License

MIT
