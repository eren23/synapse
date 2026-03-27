# Synapse Benchmark Matrix

- Generated: `2026-03-27T06:59:08+00:00`
- Host: `Apple Silicon`
- Git commit: `f00068a66cdfb116dbdbec5cc9a8c756f57b18de`

## Test Suites

| Suite | Category | Status | Duration (s) | Summary |
|-------|----------|--------|--------------|---------|
| Inference library tests | tests | ok | 0.938 | ok. 260 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.90s |
| Multi-model validation | tests | ok | 0.033 | ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| Quantization speedup isolated matmul | benchmarks | ok | 18.918 | ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 18.07s |
| Quantization speedup full model | benchmarks | ok | 0.066 | ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.03s |
| Prefill throughput benchmark | benchmarks | ok | 0.578 | ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s |
| KV-cache speedup benchmark | benchmarks | ok | 0.601 | ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s |

## Measured Local End-to-End

| Family | Checkpoint | Configuration | Prompt | Status | Prefill | Decode | Notes |
|--------|------------|---------------|--------|--------|---------|--------|-------|
| Qwen3 | Qwen3-0.6B | f32 CPU | hello | ok | 11.0 | 7.3 |  |
| Qwen3 | Qwen3-0.6B | INT8 CPU | hello | ok | 23.0 | 27.3 |  |
| Qwen3 | Qwen3-0.6B | INT8 CPU | repro_repeat_after_me | ok | 44.0 | 24.7 |  |
| Qwen3 | Qwen3-0.6B | INT8 CPU | repro_repeat_after_me | ok | 27.0 | 22.3 |  |
| Qwen3 | Qwen3-0.6B | f32 metal-feature build | hello | failed |  |  | command exited non-zero |
| Qwen3 | Qwen3-0.6B | INT8 metal-feature build | hello | failed |  |  | command exited non-zero |
| LLaMA 3.2 | LLaMA 3.2-1B | f32 CPU | hello | ok | 1.0 | 2.1 |  |
| LLaMA 3.2 | LLaMA 3.2-1B | INT8 CPU | hello | ok | 8.0 | 9.7 |  |

## Synthetic / Config-Validated

| Family | Checkpoint | Configuration | Prompt | Status | Prefill | Decode | Notes |
|--------|------------|---------------|--------|--------|---------|--------|-------|
| Qwen3 | synthetic scaled config | synthetic scaled | synthetic_default | ok | 17150.0 | 6093.1 | Qwen3 synthetic scaled config |
| LLaMA 3.2 | synthetic scaled config | synthetic scaled | synthetic_default | ok | 19883.0 | 6120.9 | LLaMA 3.2 synthetic scaled config |
| Mistral 7B | synthetic scaled config | synthetic scaled | synthetic_default | failed |  |  | Mistral 7B synthetic scaled config |

## Exploratory Local

| Family | Checkpoint | Configuration | Prompt | Status | Prefill | Decode | Notes |
|--------|------------|---------------|--------|--------|---------|--------|-------|
| Qwen2.5 | qwen2.5-0.5b | f32 CPU | hello | ok | 53.0 | 11.0 |  |
| TinyLlama | tinyllama-1.1b | f32 CPU | hello | failed |  |  | command exited non-zero |
| Qwen3 GGUF | qwen3-0.6b-gguf | f32 CPU | hello | failed |  |  | command exited non-zero |
