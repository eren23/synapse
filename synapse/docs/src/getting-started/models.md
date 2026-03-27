# Supported Models

Synapse tracks model families through two separate signals:

- **Status**: support maturity for that family
- **Evidence**: the strongest proof currently available for the status claim

That split matters. A family can be structurally supported and exercised through tests without being presented as a fully benchmarked real-checkpoint path yet.

## Model Matrix

<!-- status:docs-model-matrix:start -->
| Model Family | Status | Evidence | Notes |
|--------------|--------|----------|-------|
| Qwen3 | Validated | Benchmarked Local | Real checkpoint benchmarked locally; logits verified |
| LLaMA 3.2 | Benchmarked Local | Benchmarked Local | Real checkpoint benchmarked locally on this machine |
| Mistral 7B | Config Ready | Synthetic Validated | Sliding-window config path present; synthetic correctness tests pass, but the scaled synthetic throughput benchmark is currently failing |
| Phi-3 | In Progress | Synthetic Validated | Weight-mapper support in progress; synthetic validation passing |
| Gemma | Config Ready | Synthetic Validated | Same core transformer path; synthetic validation passing |
<!-- status:docs-model-matrix:end -->

## Status and Evidence Levels

- **Validated**: the family has the strongest end-to-end proof available in this repo, such as logit verification against a reference implementation.
- **Benchmarked local**: a real checkpoint was loaded and benchmarked locally, but full reference-logit validation may not be published yet.
- **Config ready**: the architectural path, config, and builder logic exist, but the family should not yet be read as a public end-to-end benchmark guarantee.
- **In progress**: the family is partially wired and covered by some tests, but the public support story is still incomplete.

Evidence values are intentionally narrower:

- **Benchmarked local**: backed by a real local checkpoint row in the benchmark artifact
- **Logits verified**: backed by explicit reference verification
- **Synthetic validated**: backed by fake-weight tests or scaled synthetic benchmark runs
- **Exploratory local**: observed locally, but intentionally kept outside the main public support surface

## How Model Loading Works

1. Synapse reads `config.json` from the model directory to determine the architecture.
2. The weight mapper translates HuggingFace parameter names to Synapse internal layer names.
3. Weights are loaded from safetensors or GGUF files.
4. Tokenizer and chat-template data are loaded from the model directory when available.

## Why Some Families Get Fast First

Performance work lands at two different layers:

- **Shared infrastructure**: decode kernels, quantized linear paths, LM-head projection, cache handling
- **Family-specific integration**: weight mapping, tokenizer/template behavior, RoPE variants, sliding-window attention, model loading details

That means one family can show a large measured gain before another family has a public benchmark row, even when both are expected to benefit from the same kernel work. Synapse should publish those gains only after the family-specific loading path has been exercised locally.

## Adding a New Model

To add support for a new transformer family:

1. Add the config or HuggingFace config parser path.
2. Add the weight mapper.
3. Wire any family-specific attention, normalization, or prompt behavior.
4. Cover it in synthetic validation first.
5. Promote it to benchmarked status only after a real checkpoint runs successfully in the local matrix.
