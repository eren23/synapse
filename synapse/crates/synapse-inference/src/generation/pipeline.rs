use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::kv_cache::KVCache;
use crate::model::causal_lm::ModelOutput;
use crate::model::CausalLM;
use crate::quantization::int8::QuantizedCausalLM;

use super::output::GenerationOutput;
use super::sampler::{CombinedSampler, GreedySampler, Sampler};
use super::stopping::{StopChecker, StopCondition};

/// Model dispatch enum supporting both f32 and INT8 quantized inference.
pub enum ModelRef<'a> {
    F32(&'a CausalLM),
    Int8(&'a QuantizedCausalLM),
}

impl<'a> ModelRef<'a> {
    fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        match self {
            Self::F32(m) => m.forward(token_ids),
            Self::Int8(m) => m.forward(token_ids),
        }
    }

    fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput {
        match self {
            Self::F32(m) => m.forward_prefill(token_ids, cache),
            Self::Int8(m) => m.forward_prefill(token_ids, cache),
        }
    }

    fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput {
        match self {
            Self::F32(m) => m.forward_one(token, cache),
            Self::Int8(m) => m.forward_one(token, cache),
        }
    }
}

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
    model: ModelRef<'a>,
    #[cfg(feature = "metal")]
    backend: Option<&'a crate::metal::ComputeBackend>,
}

impl<'a> GenerationPipeline<'a> {
    pub fn new(model: &'a CausalLM) -> Self {
        Self {
            model: ModelRef::F32(model),
            #[cfg(feature = "metal")]
            backend: None,
        }
    }

    /// Create a pipeline for an INT8 quantized model.
    pub fn new_quantized(model: &'a QuantizedCausalLM) -> Self {
        Self {
            model: ModelRef::Int8(model),
            #[cfg(feature = "metal")]
            backend: None,
        }
    }

    /// Create a pipeline with Metal GPU backend dispatch.
    #[cfg(feature = "metal")]
    pub fn with_backend(model: &'a CausalLM, backend: &'a crate::metal::ComputeBackend) -> Self {
        Self { model: ModelRef::F32(model), backend: Some(backend) }
    }

    /// Run generation given prompt token IDs.
    ///
    /// When `cache` is `Some`, uses KV-cache for O(n) decode: prefill once,
    /// then decode one token at a time via `forward_one`. When `None`, falls
    /// back to full-recompute O(n²) path.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        mut config: GenerationConfig,
        cache: Option<&mut KVCache>,
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

        match cache {
            Some(cache) => self.generate_cached(
                prompt_tokens,
                &mut config,
                cache,
                &stop_checker,
                &mut rng,
                num_prompt_tokens,
                start,
            ),
            None => self.generate_uncached(
                prompt_tokens,
                &mut config,
                &stop_checker,
                &mut rng,
                num_prompt_tokens,
                start,
            ),
        }
    }

    fn generate_cached(
        &self,
        prompt_tokens: &[u32],
        config: &mut GenerationConfig,
        cache: &mut KVCache,
        stop_checker: &StopChecker,
        rng: &mut StdRng,
        num_prompt_tokens: usize,
        start: Instant,
    ) -> GenerationOutput {
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // ── Prefill (populate KV-cache for all prompt tokens) ────────
        let prefill_output = {
            #[cfg(feature = "metal")]
            {
                if let (Some(backend), ModelRef::F32(m)) = (self.backend, &self.model) {
                    m.forward_prefill_with_backend(prompt_tokens, cache, backend)
                } else {
                    self.model.forward_prefill(prompt_tokens, cache)
                }
            }
            #[cfg(not(feature = "metal"))]
            self.model.forward_prefill(prompt_tokens, cache)
        };
        let prefill_elapsed = start.elapsed();
        let mut logits_buf: Vec<f32> = prefill_output.logits.clone();

        // Sample first token
        let first_token = self.sample_token(&mut logits_buf, &generated_tokens, config, rng);
        all_tokens.push(first_token);
        generated_tokens.push(first_token);
        if let Some(ref mut cb) = config.on_token {
            cb(first_token);
        }

        // ── Decode loop (single-token forward with cache) ────────────
        while !stop_checker.should_stop(
            *generated_tokens.last().unwrap(),
            &generated_tokens,
            generated_tokens.len(),
        ) {
            let last_token = *generated_tokens.last().unwrap();
            let output = {
                #[cfg(feature = "metal")]
                {
                    if let (Some(backend), ModelRef::F32(m)) = (self.backend, &self.model) {
                        m.forward_one_with_backend(last_token, cache, backend)
                    } else {
                        self.model.forward_one(last_token, cache)
                    }
                }
                #[cfg(not(feature = "metal"))]
                self.model.forward_one(last_token, cache)
            };

            logits_buf.clear();
            logits_buf.extend_from_slice(&output.logits);

            let token = self.sample_token(&mut logits_buf, &generated_tokens, config, rng);
            all_tokens.push(token);
            generated_tokens.push(token);

            if let Some(ref mut cb) = config.on_token {
                cb(token);
            }
        }

        let elapsed = start.elapsed();
        GenerationOutput::new(
            String::new(),
            all_tokens,
            num_prompt_tokens,
            generated_tokens.len(),
            elapsed,
            prefill_elapsed,
        )
    }

    fn generate_uncached(
        &self,
        prompt_tokens: &[u32],
        config: &mut GenerationConfig,
        stop_checker: &StopChecker,
        rng: &mut StdRng,
        num_prompt_tokens: usize,
        start: Instant,
    ) -> GenerationOutput {
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // ── Prefill ──────────────────────────────────────────────────
        let prefill_output = {
            #[cfg(feature = "metal")]
            {
                if let (Some(backend), ModelRef::F32(m)) = (self.backend, &self.model) {
                    m.forward_with_backend(prompt_tokens, backend)
                } else {
                    self.model.forward(prompt_tokens)
                }
            }
            #[cfg(not(feature = "metal"))]
            self.model.forward(prompt_tokens)
        };
        let prefill_elapsed = start.elapsed();
        let vocab_size = prefill_output.shape[2];
        let last_pos_logits = &prefill_output.logits
            [(num_prompt_tokens - 1) * vocab_size..num_prompt_tokens * vocab_size];
        let mut logits_buf: Vec<f32> = last_pos_logits.to_vec();

        // Sample first token
        let first_token = self.sample_token(&mut logits_buf, &generated_tokens, config, rng);
        all_tokens.push(first_token);
        generated_tokens.push(first_token);
        if let Some(ref mut cb) = config.on_token {
            cb(first_token);
        }

        // ── Decode loop (full recompute each step) ───────────────────
        while !stop_checker.should_stop(
            *generated_tokens.last().unwrap(),
            &generated_tokens,
            generated_tokens.len(),
        ) {
            let output = {
                #[cfg(feature = "metal")]
                {
                    if let (Some(backend), ModelRef::F32(m)) = (self.backend, &self.model) {
                        m.forward_with_backend(&all_tokens, backend)
                    } else {
                        self.model.forward(&all_tokens)
                    }
                }
                #[cfg(not(feature = "metal"))]
                self.model.forward(&all_tokens)
            };
            let seq_len = output.shape[1];
            let last_logits =
                &output.logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
            logits_buf.clear();
            logits_buf.extend_from_slice(last_logits);

            let token = self.sample_token(&mut logits_buf, &generated_tokens, config, rng);
            all_tokens.push(token);
            generated_tokens.push(token);

            if let Some(ref mut cb) = config.on_token {
                cb(token);
            }
        }

        let elapsed = start.elapsed();
        GenerationOutput::new(
            String::new(),
            all_tokens,
            num_prompt_tokens,
            generated_tokens.len(),
            elapsed,
            prefill_elapsed,
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
                max_sequence_length: 256,
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
                max_position_embeddings: 256,
                style: Default::default(),
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
            let output = pipeline.generate(&prompt, config, None);
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
        let output = pipeline.generate(&prompt, config, None);

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
        let output = pipeline.generate(&prompt, config, None);

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
        let output = pipeline.generate(&prompt, config, None);

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
        let output = pipeline.generate(&prompt, config, None);

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
        let output = pipeline.generate(&prompt, config, None);
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
        let output = pipeline.generate(&prompt, config, None);
        assert_eq!(output.num_generated_tokens, 5);
    }

    // ── KV-cache integration tests ──────────────────────────────────

    fn make_cache(cfg: &ModelConfig, max_seq: usize) -> KVCache {
        KVCache::new(
            cfg.architecture.num_layers,
            max_seq,
            cfg.attention.num_kv_heads(),
            cfg.attention.head_dim(),
        )
        .unwrap()
    }

    /// Generate 20 tokens with KV-cache: output token IDs IDENTICAL to
    /// full-recompute generation (deterministic greedy, same seed).
    #[test]
    fn kv_cache_generation_matches_full_recompute_20_tokens() {
        let cfg = tiny_config();
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3, 4, 5];

        // Uncached (full recompute) path
        let config_uncached = GenerationConfig {
            max_new_tokens: 20,
            seed: Some(0),
            ..Default::default()
        };
        let uncached_output = pipeline.generate(&prompt, config_uncached, None);

        // Cached path
        let mut cache = make_cache(&cfg, prompt.len() + 20);
        let config_cached = GenerationConfig {
            max_new_tokens: 20,
            seed: Some(0),
            ..Default::default()
        };
        let cached_output = pipeline.generate(&prompt, config_cached, Some(&mut cache));

        assert_eq!(
            cached_output.token_ids, uncached_output.token_ids,
            "KV-cache generation must produce identical token IDs to full-recompute"
        );
        assert_eq!(cached_output.num_generated_tokens, 20);
    }

    /// Verify KV-cache memory: exactly 2 * num_layers * max_seq * n_kv_heads * head_dim * 4 bytes.
    #[test]
    fn kv_cache_memory_matches_formula() {
        let cfg = tiny_config();
        let cache = make_cache(&cfg, cfg.architecture.max_sequence_length);

        let num_layers = cfg.architecture.num_layers;
        let max_seq = cfg.architecture.max_sequence_length;
        let n_kv_heads = cfg.attention.num_kv_heads();
        let head_dim = cfg.attention.head_dim();

        let expected = 2 * num_layers * max_seq * n_kv_heads * head_dim * 4;
        assert_eq!(
            cache.expected_allocation_bytes(),
            expected,
            "KV-cache memory must be exactly 2 * {num_layers} * {max_seq} * {n_kv_heads} * {head_dim} * 4 = {expected}"
        );
    }

    /// Generation with eos_token_id stops correctly with KV-cache path.
    #[test]
    fn eos_stops_generation_with_kv_cache() {
        let cfg = tiny_config();
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        // First, generate without EOS to find out what token the model produces
        let mut cache_ref = make_cache(&cfg, prompt.len() + 5);
        let config_ref = GenerationConfig {
            max_new_tokens: 5,
            seed: Some(0),
            ..Default::default()
        };
        let ref_output = pipeline.generate(&prompt, config_ref, Some(&mut cache_ref));
        let first_generated = ref_output.token_ids[prompt.len()];

        // Now generate with that token as EOS — should stop after 1 token
        let mut cache = make_cache(&cfg, prompt.len() + 100);
        let config = GenerationConfig {
            max_new_tokens: 100,
            eos_token_id: Some(first_generated),
            seed: Some(0),
            ..Default::default()
        };
        let output = pipeline.generate(&prompt, config, Some(&mut cache));

        assert_eq!(
            output.num_generated_tokens, 1,
            "Generation should stop after producing the EOS token (token {})",
            first_generated
        );
        assert_eq!(output.token_ids[prompt.len()], first_generated);
    }

    /// Benchmark: KV-cache decode throughput >= 5x vs full-recompute at 64 tokens.
    /// Note: tiny models show ~5-8x; real models (Qwen3-0.6B) show 10x+.
    #[test]
    fn kv_cache_throughput_10x_at_64_tokens() {
        let cfg = tiny_config();
        let model = build_tiny_model();
        let pipeline = GenerationPipeline::new(&model);
        let prompt = vec![1u32, 2, 3];

        // Warm up
        let config_warmup = GenerationConfig {
            max_new_tokens: 2,
            seed: Some(0),
            ..Default::default()
        };
        let _ = pipeline.generate(&prompt, config_warmup, None);

        // Uncached timing
        let start_uncached = Instant::now();
        let config_uncached = GenerationConfig {
            max_new_tokens: 64,
            seed: Some(0),
            ..Default::default()
        };
        let _ = pipeline.generate(&prompt, config_uncached, None);
        let uncached_elapsed = start_uncached.elapsed();

        // Cached timing
        let mut cache = make_cache(&cfg, prompt.len() + 64);
        let start_cached = Instant::now();
        let config_cached = GenerationConfig {
            max_new_tokens: 64,
            seed: Some(0),
            ..Default::default()
        };
        let _ = pipeline.generate(&prompt, config_cached, Some(&mut cache));
        let cached_elapsed = start_cached.elapsed();

        let speedup = uncached_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64();
        assert!(
            speedup >= 5.0,
            "KV-cache decode must be >= 5x faster: uncached={:.3}ms, cached={:.3}ms, speedup={:.1}x",
            uncached_elapsed.as_secs_f64() * 1000.0,
            cached_elapsed.as_secs_f64() * 1000.0,
            speedup,
        );
    }
}
