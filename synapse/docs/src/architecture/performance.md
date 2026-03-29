# Performance

<!-- status:docs-performance-note:start -->
Measured on Apple Silicon (Tiered local matrix on this machine). Last verified: 2026-03-27.
<!-- status:docs-performance-note:end -->

## End-to-End Throughput

<!-- status:docs-performance-benchmark:start -->
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
<!-- status:docs-performance-benchmark:end -->

The public table is generated from measured local rows only. Synthetic config benchmarks, exploratory checkpoints, and fallback runs are preserved in `status/benchmark_matrix.md` instead of being mixed into the headline numbers.

## Evidence Tiers

Synapse now separates performance evidence into three lanes:

1. **Measured local end-to-end**: a real checkpoint loaded on this machine, with throughput captured from the runtime path that actually executed.
2. **Synthetic / config-validated**: the architecture exercised through fake-weight tests or scaled synthetic benchmarks. This is useful for regression detection and shared-kernel comparison, but it is not a published real-model claim.
3. **Exploratory local**: extra checkpoints present locally and useful for investigation, but not part of the official support surface yet.

This keeps the public table short and defensible while still preserving the broader matrix in artifacts.

## Why Qwen3 Improved First

The recent decode recovery came from shared decode infrastructure, not a Qwen-only shortcut:

- the `M=1` INT8 GEMV kernel is shared by cached decode projections,
- the quantized LM-head path is shared by quantized causal-LM decode,
- the cached decode flow now avoids redundant input quantization across Q/K/V.

Qwen3 shows the win first because it is the real checkpoint path that has been exercised most deeply on this machine. Other decoder-only families that use the same causal-LM stack should benefit from the same kernel work once their end-to-end loading path is benchmarked locally. The docs should treat that as an architecture-based expectation, not as a numeric promise.

## Metal Reporting Rules

Metal support is runtime-reported, not feature-flag-assumed.

- A `--features metal` build only counts as a Metal benchmark if the `Runtime:` line reports `backend=metal`.
- If the metal-feature build falls back to `cpu_simd`, the run stays in the raw matrix artifact as a fallback row and is not promoted into the public performance table.
- This prevents stale GPU claims from surviving after backend routing changes.

## Running the Matrix

Canonical matrix runner:

```bash
python3 scripts/benchmark_matrix.py --include-exploratory
```

Human-facing wrapper:

```bash
bash scripts/bench_suite.sh --include-exploratory
```

The matrix writes:

- `status/benchmark_matrix.json`
- `status/benchmark_matrix.md`
- `status/public_status.json`

Then the docs sync step consumes `status/public_status.json`:

```bash
python3 scripts/sync_public_status.py --check
```

## Artifact Budgets

<!-- status:docs-performance-artifacts:start -->
| Artifact | Current | Budget | Status |
|----------|---------|--------|--------|
| WASM core | ~519 KB | ~160 KB | over |
| WASM JS wrapper | ~43 KB | ~32 KB | over |
<!-- status:docs-performance-artifacts:end -->
