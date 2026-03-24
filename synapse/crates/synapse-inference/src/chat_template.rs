//! Chat template support for HuggingFace-style `tokenizer_config.json`.
//!
//! Many HuggingFace models ship a Jinja2 `chat_template` field that describes
//! how to format a list of `{role, content}` messages into a single prompt
//! string.  This module parses that field and renders it via `minijinja`.

use std::path::Path;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single message in a multi-turn conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A parsed chat template that can format messages into a prompt string.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The raw Jinja2 template string (from `tokenizer_config.json`).
    pub template: String,
    /// Beginning-of-sequence token, if any.
    pub bos_token: Option<String>,
    /// End-of-sequence token, if any.
    pub eos_token: Option<String>,
    /// Whether to prepend the BOS token to the rendered prompt.
    pub add_bos_token: bool,
    /// Whether to append the EOS token to the rendered prompt.
    pub add_eos_token: bool,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl ChatTemplate {
    /// Load a chat template from a HuggingFace `tokenizer_config.json` file.
    ///
    /// The file is expected to contain at least a `chat_template` string field.
    /// `bos_token` and `eos_token` may be plain strings **or** objects with a
    /// `content` field (both forms appear in the wild).
    pub fn from_tokenizer_config(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&data)?;

        let template = json
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or("tokenizer_config.json missing `chat_template` field")?
            .to_string();

        let bos_token = extract_token(&json, "bos_token");
        let eos_token = extract_token(&json, "eos_token");
        let add_bos_token = json
            .get("add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let add_eos_token = json
            .get("add_eos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            template,
            bos_token,
            eos_token,
            add_bos_token,
            add_eos_token,
        })
    }

    /// Render a list of chat messages into a prompt string using the Jinja2
    /// template.
    ///
    /// The template receives the following variables:
    /// - `messages` -- the list of `{role, content}` dicts
    /// - `add_generation_prompt` -- always `true` (we want the assistant turn)
    /// - `bos_token` / `eos_token` -- the special tokens (empty string if absent)
    pub fn apply(&self, messages: &[ChatMessage]) -> Result<String, Box<dyn std::error::Error>> {
        // Try to render with the model's template, fall back to ChatML if it
        // uses Python methods that minijinja doesn't support (e.g. startswith).
        match self.try_render(messages, &self.template) {
            Ok(rendered) => Ok(rendered),
            Err(_) => {
                // Fall back to simple ChatML format
                let fallback = Self::default_qwen3();
                self.try_render(messages, &fallback.template)
            }
        }
    }

    fn try_render(
        &self,
        messages: &[ChatMessage],
        template_str: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut env = minijinja::Environment::new();
        env.add_template("chat", template_str)?;
        let tmpl = env.get_template("chat")?;

        let bos = self.bos_token.as_deref().unwrap_or("");
        let eos = self.eos_token.as_deref().unwrap_or("");

        let rendered = tmpl.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => bos,
            eos_token => eos,
        })?;

        let mut result = String::new();
        if self.add_bos_token {
            if let Some(ref bos) = self.bos_token {
                result.push_str(bos);
            }
        }
        result.push_str(&rendered);
        if self.add_eos_token {
            if let Some(ref eos) = self.eos_token {
                result.push_str(eos);
            }
        }
        Ok(result)
    }

    /// Fallback template matching Qwen3 / ChatML format.
    ///
    /// Useful when no `tokenizer_config.json` is available.
    pub fn default_qwen3() -> Self {
        Self {
            template: concat!(
                "{% for message in messages %}",
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}",
                "{% endfor %}",
                "{% if add_generation_prompt %}",
                "{{'<|im_start|>assistant\n'}}",
                "{% endif %}",
            )
            .to_string(),
            bos_token: None,
            eos_token: Some("<|im_end|>".to_string()),
            add_bos_token: false,
            add_eos_token: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a special token field that may be a plain string or an object with
/// a `content` key (e.g. `{"content": "<s>", "lstrip": false, ...}`).
fn extract_token(json: &serde_json::Value, key: &str) -> Option<String> {
    let val = json.get(key)?;
    if let Some(s) = val.as_str() {
        return Some(s.to_string());
    }
    if let Some(obj) = val.as_object() {
        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
            return Some(content.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_qwen3_template() {
        let tmpl = ChatTemplate::default_qwen3();
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hello!".into(),
            },
        ];
        let result = tmpl.apply(&messages).expect("template should render");
        assert!(result.contains("<|im_start|>user\nHello!<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_multi_turn_template() {
        let tmpl = ChatTemplate::default_qwen3();
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "How are you?".into(),
            },
        ];
        let result = tmpl.apply(&messages).expect("template should render");
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_bos_token_prepend() {
        let tmpl = ChatTemplate {
            template: "{{ messages[0]['content'] }}".to_string(),
            bos_token: Some("<s>".to_string()),
            eos_token: None,
            add_bos_token: true,
            add_eos_token: false,
        };
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "test".into(),
        }];
        let result = tmpl.apply(&messages).expect("template should render");
        assert_eq!(result, "<s>test");
    }

    #[test]
    fn test_from_tokenizer_config_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config_path = dir.path().join("tokenizer_config.json");

        let json_content = serde_json::json!({
            "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "add_bos_token": false,
            "add_eos_token": false
        });
        std::fs::write(&config_path, json_content.to_string()).unwrap();

        let tmpl =
            ChatTemplate::from_tokenizer_config(&config_path).expect("should parse config");
        assert_eq!(tmpl.bos_token.as_deref(), Some("<|endoftext|>"));
        assert_eq!(tmpl.eos_token.as_deref(), Some("<|im_end|>"));
        assert!(!tmpl.add_bos_token);
        assert!(!tmpl.add_eos_token);

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let result = tmpl.apply(&messages).expect("template should render");
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_token_as_object() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config_path = dir.path().join("tokenizer_config.json");

        let json_content = serde_json::json!({
            "chat_template": "{{ messages[0]['content'] }}",
            "bos_token": {
                "content": "<s>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            },
            "eos_token": {
                "content": "</s>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            },
            "add_bos_token": true,
            "add_eos_token": true
        });
        std::fs::write(&config_path, json_content.to_string()).unwrap();

        let tmpl =
            ChatTemplate::from_tokenizer_config(&config_path).expect("should parse config");
        assert_eq!(tmpl.bos_token.as_deref(), Some("<s>"));
        assert_eq!(tmpl.eos_token.as_deref(), Some("</s>"));
        assert!(tmpl.add_bos_token);
        assert!(tmpl.add_eos_token);

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let result = tmpl.apply(&messages).expect("template should render");
        assert_eq!(result, "<s>hi</s>");
    }
}
