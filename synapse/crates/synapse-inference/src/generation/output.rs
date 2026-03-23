use std::time::Duration;

/// Result of a generation run.
pub struct GenerationOutput {
    /// The generated text (empty if no detokenizer is available).
    pub text: String,
    /// All token IDs: prompt tokens followed by generated tokens.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excludes prompt tokens).
    pub num_generated_tokens: usize,
    /// Number of prompt tokens consumed during prefill.
    pub num_prompt_tokens: usize,
    /// Wall-clock time for the entire generation.
    pub elapsed: Duration,
    /// Generated tokens per second (excludes prefill).
    pub tokens_per_sec: f64,
}

impl GenerationOutput {
    pub fn new(
        text: String,
        token_ids: Vec<u32>,
        num_prompt_tokens: usize,
        num_generated_tokens: usize,
        elapsed: Duration,
    ) -> Self {
        let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            num_generated_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        Self {
            text,
            token_ids,
            num_generated_tokens,
            num_prompt_tokens,
            elapsed,
            tokens_per_sec,
        }
    }
}
