# Contributing

## Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run the test suite: `cargo test -p synapse-inference --lib`
5. Submit a pull request

## Before Submitting

- **Run tests**: All 210+ tests must pass. Run `cargo test -p synapse-inference --lib`.
- **Run clippy**: `cargo clippy --workspace` should produce no warnings.
- **Format code**: `cargo fmt --all` to match the project style.
- **Test with a model**: If your change affects inference, verify with a real model run.

## Code Style

- Follow existing patterns in the codebase. Read nearby code before writing new code.
- No unnecessary abstractions -- prefer direct, readable implementations.
- Use descriptive variable names. Avoid single-letter names outside of loop indices and mathematical formulas.
- Keep functions focused. If a function exceeds ~100 lines, consider splitting it.
- Document public APIs with doc comments.

## Adding a New Model

To add support for a new transformer architecture:

1. **Config**: Add a config JSON in `configs/` with the model's hyperparameters (hidden_size, num_heads, etc.)
2. **Weight mapper**: Write a mapper function in the inference crate that translates HuggingFace weight names to Synapse layer names. See existing mappers (e.g., `qwen3()`, `llama()`) as reference.
3. **Architecture-specific logic**: If the model uses non-standard attention (e.g., sliding window, grouped query), add the necessary dispatch.
4. **Test**: Add unit tests for config loading and weight mapping.
5. **Validate**: Run `scripts/verify_logits.py` to compare output against HuggingFace Transformers.

## Adding Zig Kernels

If you need to add or modify SIMD kernels:

1. Edit files in `zig/src/ops/`
2. Add the FFI export in `zig/src/ffi/exports.zig`
3. Add the Rust binding in `crates/synapse-sys/src/lib.rs`
4. Add the safe wrapper in `crates/synapse-core/src/lib.rs`
5. Test: `cargo test -p synapse-core`

## Reporting Issues

Include:
- Hardware (chip, RAM)
- OS version
- Rust and Zig versions
- Steps to reproduce
- Error output or unexpected behavior
