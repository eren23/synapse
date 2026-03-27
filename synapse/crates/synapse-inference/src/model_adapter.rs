use std::error::Error;
use std::fmt;

use crate::chat_template::{ChatMessage, ChatTemplate, ChatTemplateOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingMode {
    #[default]
    Auto,
    Disabled,
}

impl ThinkingMode {
    pub fn parse_cli(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "disabled" => Ok(Self::Disabled),
            _ => Err(format!(
                "Unknown thinking mode: {value}. Expected one of: auto, disabled"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for ThinkingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningMarkers {
    pub start: &'static str,
    pub end: &'static str,
}

pub trait ModelAdapter: Sync {
    fn family(&self) -> &'static str;

    fn default_cli_thinking_mode(&self) -> ThinkingMode {
        ThinkingMode::Auto
    }

    fn reasoning_markers(&self) -> Option<ReasoningMarkers> {
        None
    }

    fn format_chat_prompt(
        &self,
        template: Option<&ChatTemplate>,
        messages: &[ChatMessage],
        thinking_mode: ThinkingMode,
    ) -> Result<String, Box<dyn Error>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelAdapterKind {
    Generic,
    Qwen3,
}

impl ModelAdapterKind {
    pub fn from_model_name(name: &str) -> Self {
        let normalized = name.to_ascii_lowercase();
        if normalized.contains("qwen3") {
            Self::Qwen3
        } else {
            Self::Generic
        }
    }

    pub fn family(self) -> &'static str {
        adapter_for_kind(self).family()
    }
}

static GENERIC_ADAPTER: GenericAdapter = GenericAdapter;
static QWEN3_ADAPTER: Qwen3Adapter = Qwen3Adapter;

pub fn adapter_for_kind(kind: ModelAdapterKind) -> &'static dyn ModelAdapter {
    match kind {
        ModelAdapterKind::Generic => &GENERIC_ADAPTER,
        ModelAdapterKind::Qwen3 => &QWEN3_ADAPTER,
    }
}

struct GenericAdapter;

impl ModelAdapter for GenericAdapter {
    fn family(&self) -> &'static str {
        "generic"
    }

    fn format_chat_prompt(
        &self,
        template: Option<&ChatTemplate>,
        messages: &[ChatMessage],
        _thinking_mode: ThinkingMode,
    ) -> Result<String, Box<dyn Error>> {
        if let Some(template) = template {
            return template.apply_with_options(messages, ChatTemplateOptions::default());
        }

        if messages.len() == 1 && messages[0].role == "user" {
            return Ok(messages[0].content.clone());
        }

        let prompt = messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(prompt)
    }
}

struct Qwen3Adapter;

impl ModelAdapter for Qwen3Adapter {
    fn family(&self) -> &'static str {
        "qwen3"
    }

    fn default_cli_thinking_mode(&self) -> ThinkingMode {
        ThinkingMode::Disabled
    }

    fn reasoning_markers(&self) -> Option<ReasoningMarkers> {
        Some(ReasoningMarkers {
            start: "<think>",
            end: "</think>",
        })
    }

    fn format_chat_prompt(
        &self,
        template: Option<&ChatTemplate>,
        messages: &[ChatMessage],
        thinking_mode: ThinkingMode,
    ) -> Result<String, Box<dyn Error>> {
        let template = template
            .cloned()
            .unwrap_or_else(ChatTemplate::default_qwen3);
        template.apply_with_options(
            messages,
            ChatTemplateOptions {
                enable_thinking: Some(!matches!(thinking_mode, ThinkingMode::Disabled)),
                ..Default::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_message(content: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: content.into(),
        }]
    }

    #[test]
    fn generic_adapter_uses_raw_content_without_template() {
        let prompt = adapter_for_kind(ModelAdapterKind::Generic)
            .format_chat_prompt(None, &user_message("hello"), ThinkingMode::Auto)
            .unwrap();
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn qwen3_adapter_appends_empty_think_block_when_disabled() {
        let prompt = adapter_for_kind(ModelAdapterKind::Qwen3)
            .format_chat_prompt(None, &user_message("hello"), ThinkingMode::Disabled)
            .unwrap();

        assert!(prompt.contains("<|im_start|>assistant\n"));
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen3_adapter_keeps_auto_prompt_unchanged() {
        let prompt = adapter_for_kind(ModelAdapterKind::Qwen3)
            .format_chat_prompt(None, &user_message("hello"), ThinkingMode::Auto)
            .unwrap();

        assert!(prompt.contains("<|im_start|>assistant\n"));
        assert!(!prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen3_adapter_exposes_reasoning_markers() {
        assert_eq!(
            adapter_for_kind(ModelAdapterKind::Qwen3).reasoning_markers(),
            Some(ReasoningMarkers {
                start: "<think>",
                end: "</think>",
            })
        );
        assert_eq!(
            adapter_for_kind(ModelAdapterKind::Generic).reasoning_markers(),
            None
        );
    }

    #[test]
    fn qwen3_is_detected_case_insensitively() {
        assert_eq!(
            ModelAdapterKind::from_model_name("Qwen3-0.6B"),
            ModelAdapterKind::Qwen3
        );
        assert_eq!(
            ModelAdapterKind::from_model_name("qwen3"),
            ModelAdapterKind::Qwen3
        );
        assert_eq!(
            ModelAdapterKind::from_model_name("llama"),
            ModelAdapterKind::Generic
        );
    }
}
