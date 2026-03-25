# Quick Start

This guide walks through running your first chat session with Synapse.

## 1. Download a Model

Use the HuggingFace CLI to download Qwen3-0.6B:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b
```

This downloads the safetensors weights, tokenizer, and config files (~1.2 GB).

## 2. Run Chat

```bash
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b
```

The engine loads the model, detects the chat template from `tokenizer_config.json`, and starts an interactive session.

## 3. Enable INT8 Quantization

For faster inference, add the `--quantize` flag:

```bash
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize
```

This converts weights to INT8 at load time. On Apple M5, decode speed goes from 6.6 tok/s to 14.6 tok/s.

## 4. Enable Metal GPU

Build with Metal and run:

```bash
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b
```

Metal accelerates the prefill phase (prompt processing). Combine with `--quantize` for best throughput.

## 5. Use Other Models

Synapse supports multiple model families. Download and point to any supported checkpoint:

```bash
# LLaMA 3.2 1B
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /tmp/llama-3.2-1b
cargo run --example qwen3_chat --release -- --model-dir /tmp/llama-3.2-1b
```

The engine auto-detects the model architecture from `config.json`. See [Supported Models](models.md) for the full list.
