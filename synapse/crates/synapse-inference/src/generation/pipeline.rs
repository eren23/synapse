use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::model::CausalLM;

use super::output::GenerationOutput;
use super::sampler::{CombinedSampler, GreedySampler, Sampler};
use super::stopping::{StopChecker, StopCondition};

/// Configuration for the generation pipeline.
pub struct GenerationConfig {
    /// Maximum number of tokens to generate (excludes prompt).
    pub max_new_tokens: usize,
    /// EOS token ID. Generation stops when this token is produced.
    pub eos_token_id: Option<u32>,
    /// Additional stop sequences (as token ID subsequences).
    pub stop_sequences: Vec<Vec<u32>>,
    /// Sampling strategy. Defaults to greedy if `None`.
    pub sampler: Option<Box<dyn Sampler>>,
    /// Combined sampler config (alternative to `sampler`).
    pub combined: Option<CombinedSampler>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Streaming callback: called with each generated token ID.
    pub on_token: Option<Box<dyn FnMut(u32)>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_token_id: None,
            stop_sequences: Vec::new(),
            sampler: None,
            combined: None,
            seed: None,
            on_token: None,
        }
    }
}

/// Token generation pipeline: tokenize → prefill → decode loop → detokenize.
///
/// Operates on raw token IDs since no tokenizer is integrated yet.
/// The pipeline runs a forward pass on the full prompt (prefill), then
/// generates tokens one at a time (decode), sampling from the logits
/// of the last position.
pub struct GenerationPipeline<'a> {
    model: &'a CausalLM,
}

impl<'a> GenerationPipeline<'a> {
    pub fn new(model: &'a CausalLM) -> Self {
        Self { model }
    }

    /// Run generation given prompt token IDs.
    ///
    /// Returns `GenerationOutput` with the generated tokens and timing info.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        mut config: GenerationConfig,
    ) -> GenerationOutput {
        let start = Instant::now();
        let num_prompt_tokens = prompt_tokens.len();

        // Build stop conditions
        let mut conditions = vec![StopCondition::MaxLength(config.max_new_tokens)];
        if let Some(eos) = config.eos_token_id {
            conditions.push(StopCondition::EosToken(eos));
        }
        if !config.stop_sequences.is_empty() {
            conditions.push(StopCondition::StopSequences(
                config.stop_sequences.clone(),
            ));
        }
        let stop_checker = StopChecker::new(conditions);

        // Initialize RNG
        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // All tokens: prompt + generated
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // ── Prefill ──────────────────────────────────────────────────
        // Run forward on the full prompt to get initial logits.
        let prefill_output = self.model.forward(prompt_tokens);
        let vocab_size = prefill_output.shape[2];

        // Get logits for the last prompt token position
        let last_pos_logits = &prefill_output.logits
            [(num_prompt_tokens - 1) * vocab_size..num_prompt_tokens * vocab_size];
        let mut logits_buf: Vec<f32> = last_pos_logits.to_vec();

        // Sample first token
        let first_token = self.sample_token(
            &mut logits_buf,
            &generated_tokens,
            &mut config,
            &mut rng,
        );
        all_tokens.push(first_token);
        generated_tokens.push(first_token);
        if let Some(ref mut cb) = config.on_token {
            cb(first_token);
        }

        // ── Decode loop ──────────────────────────────────────────────
        // Generate tokens one at a time until a stop condition is met.
        while !stop_checker.should_stop(
            *generated_tokens.last().unwrap(),
            &generated_tokens,
            generated_tokens.len(),
        ) {
            // Forward pass on all tokens so far.
            // (Without KV-cache integration, we re-process everything.
            //  With KV-cache, this would be a single-token forward.)
            let output = self.model.forward(&all_tokens);
            let seq_len = output.shape[1];

            // Logits for the last position
            let last_logits =
                &output.logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
            logits_buf.clear();
            logits_buf.extend_from_slice(last_logits);

            let token = self.sample_token(
                &mut logits_buf,
                &generated_tokens,
                &mut config,
                &mut rng,
            );
            all_tokens.push(token);
            generated_tokens.push(token);

            if let Some(ref mut cb) = config.on_token {
                cb(token);
            }
        }

        let elapsed = start.elapsed();
        GenerationOutput::new(
            String::new(), // no tokenizer for detokenization
            all_tokens,
            num_prompt_tokens,
            generated_tokens.len(),
            elapsed,
        )
    }

    /// Prefill-only: run a single forward pass on the prompt and return the
    /// logits for the last position.
    pub fn prefill(&self, prompt_tokens: &[u32]) -> Vec<f32> {
        let output = self.model.forward(prompt_tokens);
        let vocab_size = output.shape[2];
        let seq_len = output.shape[1];
        output.logits[(seq_len - 1) * vocab_size..seq_len * vocab_size].to_vec()
    }

    /// Decode-only: given all tokens so far, run forward and return the
    /// logits for the last position.
    pub fn decode_step(&self, all_tokens: &[u32]) -> Vec<f32> {
        let output = self.model.forward(all_tokens);
        let vocab_size = output.shape[2];
        let seq_len = output.shape[1];
        output.logits[(seq_len - 1) * vocab_size..seq_len * vocab_size].to_vec()
    }

    fn sample_token(
        &self,
        logits: &mut [f32],
        generated_tokens: &[u32],
        config: &mut GenerationConfig,
        rng: &mut StdRng,
    ) -> u32 {
        if let Some(ref combined) = config.combined {
            combined.sample_with_history(logits, generated_tokens, rng)
        } else if let Some(ref sampler) = config.sampler {
            sampler.sample(logits, rng)
        } else {
            GreedySampler.sample(logits, rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::model::ModelBuilder;
    use crate::weight_loading::{RawTensor, WeightMapper};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "TinyTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 32,
                num_layers: 2,
                vocab_size: 64,
                max_sequence_length: 16,
                tie_word_embeddings: true,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 8,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 64,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 16,
            },
            quantization: QuantConfig::F32,
        }
    }

    fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let fake = |shape: Vec<usize>| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: (0..n)
                    .map(|i| (i as f32 * 0.001) % 0.1 + 0.01)
                    .collect(),
                shape,
            }
        };

        let mut w = HashMap::new();
        w.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h]));
        for i in 0..nl {
            w.insert(
                format!("model.layers.{i}.input_layernorm.weight"),
                fake(vec![h]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                fake(vec![q_dim, h]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                fake(vec![kv_dim, h]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                fake(vec![kv_dim, h]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                fake(vec![h, q_dim]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                fake(vec![cfg.attention.head_dim()]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                fake(vec![cfg.attention.head_dim()]),
            );
            w.insert(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                fake(vec![h]),
            );
            w.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                fake(vec![inter, h]),
            );
            w.insert(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                fake(vec![inter, h]),
            );
            w.insert(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                fake(vec![h, inter]),
            );
        }
        w.insert("model.norm.weight".into(), fake(vec![h]));
        w.insert("lm_head.weight".into(), fake(vec![vocab, h]));
        w
    }

    fn build_tiny_model() -> CausalLM {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);
        let weights = generate_fake_hf_weights(&cfg);
        let mapper = WeightMapper::qwen3();
        let result = model.load_weights(weights, &mapper).unwrap();
        assert!(result.missing.is_empty(), "Missing: {:?}", result.missing);
        model
    }

    #[test]
    fn greedy_deterministic_across_runs() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let mut results = Vec::new();
        for _ in 0..10 {
            let config = GenerationConfig {
                max_new_tokens: 5,
                seed: Some(0),
                ..Default::default()
            };
            let output = pipeline.generate(&prompt, config);
            results.push(output.token_ids.clone());
        }

        // All runs should produce identical output
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r, &results[0], "Run {i} differs from run 0");
        }
    }

    #[test]
    fn eos_stops_generation() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        // Use greedy sampling. Even if the model doesn't naturally produce token 0,
        // we test the mechanism with a large max_new_tokens.
        let config = GenerationConfig {
            max_new_tokens: 100,
            eos_token_id: Some(0), // very unlikely but tests the code path
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);

        // Should stop at max_new_tokens if EOS never emitted, or at EOS
        assert!(output.num_generated_tokens <= 100);
    }

    #[test]
    fn max_length_stops_generation() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let config = GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);

        assert_eq!(output.num_generated_tokens, 3);
        assert_eq!(output.num_prompt_tokens, 3);
        assert_eq!(output.token_ids.len(), 6); // 3 prompt + 3 generated
    }

    #[test]
    fn streaming_callback_receives_each_token() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let streamed = Arc::new(Mutex::new(Vec::new()));
        let streamed_clone = Arc::clone(&streamed);

        let config = GenerationConfig {
            max_new_tokens: 5,
            on_token: Some(Box::new(move |token| {
                streamed_clone.lock().unwrap().push(token);
            })),
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);

        let streamed_tokens = streamed.lock().unwrap();
        let generated = &output.token_ids[3..]; // skip prompt
        assert_eq!(&*streamed_tokens, generated, "Streamed tokens must match generated tokens");
        assert_eq!(streamed_tokens.len(), 5);
    }

    #[test]
    fn prefill_decode_matches_full_forward() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3, 4];

        // Prefill: get last-position logits from prompt
        let prefill_logits = pipeline.prefill(&prompt);

        // Full forward on same tokens should give same last-position logits
        let full_output = model.forward(&prompt);
        let vocab = full_output.shape[2];
        let seq_len = full_output.shape[1];
        let full_last_logits =
            &full_output.logits[(seq_len - 1) * vocab..seq_len * vocab];

        assert_eq!(prefill_logits.len(), full_last_logits.len());
        for (i, (&a, &b)) in prefill_logits.iter().zip(full_last_logits.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Logit {i} mismatch: prefill={a}, full={b}"
            );
        }
    }

    #[test]
    fn output_has_valid_timing() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let config = GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);

        assert!(output.elapsed.as_nanos() > 0, "Elapsed time should be positive");
        assert!(output.tokens_per_sec > 0.0, "Tokens/sec should be positive");
    }

    #[test]
    fn temperature_sampling_with_pipeline() {
        use super::super::sampler::TemperatureSampler;

        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let config = GenerationConfig {
            max_new_tokens: 5,
            sampler: Some(Box::new(TemperatureSampler { temperature: 1.0 })),
            seed: Some(42),
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);
        assert_eq!(output.num_generated_tokens, 5);
    }

    #[test]
    fn combined_sampler_with_pipeline() {
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        let config = GenerationConfig {
            max_new_tokens: 5,
            combined: Some(CombinedSampler {
                temperature: 0.8,
                top_k: 10,
                top_p: 0.9,
                repetition_penalty: 1.2,
            }),
            seed: Some(42),
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config);
        assert_eq!(output.num_generated_tokens, 5);
    }
}
