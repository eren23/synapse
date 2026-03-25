# Chat Templates

Synapse uses minijinja-based chat template rendering to format conversations for each model family.

## How Templates Work

When loading a model with `InferenceEngine::from_pretrained()`, Synapse:

1. Reads `tokenizer_config.json` from the model directory
2. Extracts the `chat_template` field (a Jinja2 template string)
3. Compiles it with minijinja for fast rendering

If the template is missing or incompatible, Synapse falls back to **ChatML** format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

## API Usage

```rust
use synapse_inference::ChatTemplate;

let template = ChatTemplate::from_tokenizer_config("/path/to/tokenizer_config.json")?;

let messages = vec![
    ("system", "You are a helpful assistant."),
    ("user", "What is Rust?"),
];

let formatted = template.apply(&messages)?;
// Returns the full prompt string ready for tokenization
```

## Supported Template Formats

| Model | Template Style | Special Tokens |
|-------|---------------|----------------|
| Qwen3 | ChatML variant | `<|im_start|>`, `<|im_end|>` |
| LLaMA 3.2 | Llama-style | `<|begin_of_text|>`, `<|eot_id|>` |
| Mistral | Mistral-style | `[INST]`, `[/INST]` |
| Phi-3 | ChatML variant | `<|user|>`, `<|assistant|>` |
| Gemma | Gemma-style | `<start_of_turn>`, `<end_of_turn>` |

## Multi-Turn Conversations

Templates handle multi-turn conversations automatically. Pass the full message history:

```rust
let messages = vec![
    ("system", "You are a helpful assistant."),
    ("user", "What is Rust?"),
    ("assistant", "Rust is a systems programming language."),
    ("user", "What about Zig?"),
];

let formatted = template.apply(&messages)?;
```

The template engine handles role tags, turn separators, and generation prompts for each model family.

## Custom Templates

To override the default template, pass a custom Jinja2 string:

```rust
let template = ChatTemplate::from_string("{% for msg in messages %}...")?;
```

This is useful for experimenting with prompt formats or using models without a bundled template.
