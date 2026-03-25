# Synapse

Synapse is a modular LLM inference engine built in Rust with Zig SIMD kernels and Metal GPU acceleration. It loads HuggingFace checkpoints directly and runs transformer models on Apple Silicon and x86 hardware.

## Key Features

- **Zig SIMD kernels** -- hand-tuned GEMV for NEON (ARM) and AVX2 (x86), including fused attention
- **Quantization** -- INT8 per-channel symmetric, Q4_0 native 4-bit GEMV, GGUF format support (Q4_K, Q6_K, Q8_0, etc.)
- **Metal GPU** -- GPU-accelerated prefill with pre-transposed weight caching
- **5 model families** -- Qwen3, LLaMA 3.2, Mistral, Phi-3, Gemma
- **Chat templates** -- minijinja-based rendering loaded from `tokenizer_config.json`
- **Speculative decoding** -- draft-model-free speculative generation
- **9-crate workspace** -- clean separation of inference, training, autograd, data loading, and graph IR

## Performance

On Qwen3-0.6B (596M params), Apple M5:

| Configuration | Prefill | Decode |
|---------------|---------|--------|
| f32 CPU | 18 tok/s | 6.6 tok/s |
| INT8 CPU | 31 tok/s | **14.6 tok/s** |
| Metal f32 | 19 tok/s | 6.5 tok/s |

INT8 decode reaches **6.3x improvement** over the 2.3 tok/s baseline.

## Quick Start

```bash
# Build
cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Run chat
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b
```

See [Installation](getting-started/installation.md) for prerequisites and detailed setup.
