# Supported Models

Synapse supports 5 model families through architecture-specific weight mappers and config parsers.

## Model Matrix

| Model | Config File | Weight Mapper | Validated | Notes |
|-------|-------------|---------------|-----------|-------|
| Qwen3 | `qwen3_0.6b.json` | `qwen3()` | Yes (logits verified) | Per-head Q/K norms |
| LLaMA 3.2 | `llama3.2_1b.json` | `llama()` | Config only | `rope_scaling` support |
| Mistral 7B | `mistral_7b.json` | `mistral()` | Config only | Sliding window attention |
| Phi-3 | -- | `phi()` | Config only | Fused projections not yet supported |
| Gemma | -- | `gemma()` | Config only | Same as LLaMA weight naming |

**Validated** means end-to-end logit comparison against HuggingFace Transformers using `scripts/verify_logits.py`.

## How Model Loading Works

1. Synapse reads `config.json` from the model directory to determine the architecture
2. The appropriate weight mapper translates HuggingFace parameter names to Synapse's internal layer names
3. Weights are loaded from safetensors (or GGUF) files
4. The tokenizer and chat template are loaded from `tokenizer_config.json`

## Adding a New Model

To add support for a new transformer architecture:

1. Create a config JSON in `configs/` with the model's hyperparameters
2. Write a weight mapper function in the inference crate that maps HF weight names to Synapse layer names
3. Add any architecture-specific attention or normalization logic
4. Validate with `scripts/verify_logits.py` against HuggingFace output

## GGUF Models

Synapse can also load models in GGUF format, which includes pre-quantized weights:

```bash
cargo run --example qwen3_chat --release -- --model-dir /path/to/gguf/
```

Supported GGUF quantization types: F32, F16, Q8_0, Q4_0, Q4_1, Q4_K, Q6_K.
