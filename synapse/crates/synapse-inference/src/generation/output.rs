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
    /// Wall-clock time for the entire generation (prefill + decode).
    pub elapsed: Duration,
    /// Wall-clock time for prefill only (time to first token).
    pub prefill_elapsed: Duration,
    /// Generated tokens per second (decode phase only, excludes prefill).
    pub tokens_per_sec: f64,
    /// Prefill throughput: prompt tokens per second.
    pub prefill_tokens_per_sec: f64,
}

impl GenerationOutput {
    pub fn new(
        text: String,
        token_ids: Vec<u32>,
        num_prompt_tokens: usize,
        num_generated_tokens: usize,
        elapsed: Duration,
        prefill_elapsed: Duration,
    ) -> Self {
        let decode_elapsed = elapsed.saturating_sub(prefill_elapsed);
        let tokens_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
            num_generated_tokens as f64 / decode_elapsed.as_secs_f64()
        } else {
            0.0
        };
        let prefill_tokens_per_sec = if prefill_elapsed.as_secs_f64() > 0.0 {
            num_prompt_tokens as f64 / prefill_elapsed.as_secs_f64()
        } else {
            0.0
        };
        Self {
            text,
            token_ids,
            num_generated_tokens,
            num_prompt_tokens,
            elapsed,
            prefill_elapsed,
            tokens_per_sec,
            prefill_tokens_per_sec,
        }
    }
}
