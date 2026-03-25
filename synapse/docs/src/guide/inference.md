# Inference API

Synapse provides two levels of API: a high-level `InferenceEngine` for quick usage and a lower-level `GenerationPipeline` for fine-grained control.

## InferenceEngine (High-Level)

The simplest way to run inference:

```rust
use synapse_inference::InferenceEngine;

// Load model, tokenizer, and chat template from a HuggingFace directory
let engine = InferenceEngine::from_pretrained("/path/to/model")?;

// Generate text
let config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};
let output = engine.generate_text("What is Rust?", &config)?;
println!("{}", output);
```

### Quantization

Convert the loaded model to INT8 for faster inference:

```rust
let engine = InferenceEngine::from_pretrained("/path/to/model")?;
engine.quantize(); // In-place INT8 conversion
```

After quantization, all subsequent `generate_text` calls use INT8 kernels automatically.

## GenerationPipeline (Low-Level)

For more control over the generation loop:

```rust
use synapse_inference::{GenerationPipeline, CausalLM, KvCache};

let model = CausalLM::load("/path/to/model")?;
let mut kv_cache = KvCache::new(&model.config);
let pipeline = GenerationPipeline::new(&model, &mut kv_cache);

// Run prefill
let logits = pipeline.prefill(&input_ids)?;

// Decode token by token
let next_token = pipeline.decode_step(sampled_token)?;
```

## ModelRef Dispatch

Synapse uses a `ModelRef` enum to dispatch between f32 and INT8 models at runtime:

```rust
enum ModelRef {
    F32(CausalLM<f32>),
    Int8(CausalLM<QuantizedInt8>),
}
```

This avoids generic proliferation while keeping the hot path monomorphic.

## Speculative Decoding

Enable speculative decoding for faster generation with large models:

```rust
let config = GenerationConfig {
    speculative_k: 4, // Draft 4 tokens ahead
    ..Default::default()
};
let output = engine.generate_text(prompt, &config)?;
```

Speculative decoding drafts multiple tokens and verifies them in a single forward pass, reducing the number of sequential decode steps.

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 256 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | 0 | Top-K sampling (0 = disabled) |
| `repetition_penalty` | 1.0 | Penalize repeated tokens |
| `speculative_k` | 0 | Speculative decoding lookahead (0 = disabled) |
