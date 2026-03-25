# Testing

Synapse has 225+ tests across the workspace, covering unit tests, integration tests, and multi-architecture validation.

## Test Suite Overview

| Category | Count | Scope |
|----------|-------|-------|
| Library tests (`--lib`) | 210 | Unit tests in inference crate |
| Integration tests | 15+ | End-to-end inference, KV cache, quantization |
| Benchmark tests | varies | Matmul, attention, throughput, memory |

## Running Tests

Run the core inference tests:

```bash
cargo test -p synapse-inference --lib
```

Run all workspace tests:

```bash
cargo test --workspace
```

Run a specific test by name:

```bash
cargo test -p synapse-inference --lib test_qwen3_logits
```

## What the Tests Cover

**Model loading**: Verifies config parsing, weight mapping, and tensor shape validation for all 5 model families.

**Quantization accuracy**: Compares INT8 quantized output against f32 reference, checking that error stays within tolerance.

**KV cache correctness**: Ensures incremental decode produces the same results as full-sequence prefill.

**Attention**: Tests masked attention, causal masking, and multi-head splitting.

**GGUF loading**: Validates dequantization for each supported quantization type (Q4_0, Q4_1, Q4_K, Q6_K, Q8_0).

**Chat templates**: Tests template rendering for each model family and edge cases (empty messages, system-only, etc.).

## Integration Tests

Located in `tests/`:

```bash
cargo test -p synapse-inference --test '*'
```

These run full inference pipelines and verify output token sequences.

## Logit Verification

To verify Synapse output against HuggingFace Transformers:

```bash
python scripts/verify_logits.py --model-dir /path/to/qwen3-0.6b --prompt "Hello"
```

This script:
1. Runs the same prompt through both HuggingFace and Synapse
2. Compares logits element-wise
3. Reports max absolute error and cosine similarity

## Benchmark Scripts

Full performance benchmark:

```bash
bash scripts/bench_suite.sh --model-dir /path/to/qwen3-0.6b
```

Compare against llama.cpp:

```bash
bash bench_vs_llamacpp.sh /path/to/model.gguf
```

These scripts output structured results to `benchmark_results_*.md`.
