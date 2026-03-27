# Testing

Synapse uses three different validation lanes:

1. **Core correctness tests** for the inference crate
2. **Multi-model validation** for architecture coverage across supported families
3. **Benchmark matrix runs** for performance evidence and public-status synchronization

## Core Commands

Run the core inference tests:

```bash
cargo test -p synapse-inference --lib
```

Run the multi-family validation suite:

```bash
cargo test --test multi_model_validation --release
```

Run targeted performance tests:

```bash
cargo test --test quantization_speedup --release -- --nocapture
cargo test --test prefill_throughput --release -- --nocapture
cargo test --test kvcache_speedup --release -- --nocapture
```

## Benchmark Matrix

The canonical benchmark/reporting entrypoint is:

```bash
python3 scripts/benchmark_matrix.py --include-exploratory
```

That command:

- runs the selected test and benchmark suites,
- measures real local checkpoint rows where available,
- records synthetic and exploratory rows separately,
- writes the machine-readable artifact to `status/benchmark_matrix.json`,
- writes the human summary to `status/benchmark_matrix.md`,
- updates the public snapshot in `status/public_status.json`.

The shell wrapper is just a convenience alias:

```bash
bash scripts/bench_suite.sh --include-exploratory
```

## Public Docs Sync

README and mdBook status blocks are rendered from `status/public_status.json`.

Check that the public files are in sync:

```bash
python3 scripts/sync_public_status.py --check
```

Build the docs after status regeneration:

```bash
cd docs && mdbook build
```

## Evidence Rules

- Public performance tables may contain only measured local end-to-end rows.
- Synthetic rows are useful for regression tracking, but they do not count as public real-model claims.
- Exploratory checkpoints are kept in the artifact appendix, not the headline support table.
- Metal rows count only when the runtime line confirms `backend=metal`.
