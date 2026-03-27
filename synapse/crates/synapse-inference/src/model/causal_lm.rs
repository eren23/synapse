use std::collections::{HashMap, HashSet};

use crate::config::ModelConfig;
use crate::kv_cache::KVCache;
#[cfg(feature = "metal")]
use crate::model::decoder_layer::apply_norm_dispatch;
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::registry::NormVariant;
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

use super::DecoderLayer;

/// Output from a forward pass.
pub struct ModelOutput {
    /// Flat logits: `[seq_len * vocab_size]`.
    pub logits: Vec<f32>,
    /// Logical shape `[batch, seq_len, vocab_size]`.
    pub shape: [usize; 3],
}

/// Result of weight loading: lists of missing and unexpected keys.
pub struct LoadResult {
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
}

/// A causal language model: embedding → N × DecoderLayer → norm → lm_head.
pub struct CausalLM {
    pub config: ModelConfig,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: Box<dyn NormVariant>,

    // ── Weights (64-byte aligned for SIMD) ───────────────────────────
    pub embed_tokens: AlignedBuffer,
    pub final_norm_weight: AlignedBuffer,
    /// `None` when `tie_word_embeddings` is true (reuses `embed_tokens`).
    pub lm_head_weight: Option<AlignedBuffer>,

    // ── RoPE precomputed tables (shared across all layers) ───────────
    /// Cosine table: `[max_position_embeddings, head_dim / 2]`.
    pub rope_cos: Vec<f32>,
    /// Sine table: `[max_position_embeddings, head_dim / 2]`.
    pub rope_sin: Vec<f32>,
}

impl CausalLM {
    /// Total number of unique trainable parameters.
    pub fn param_count(&self) -> usize {
        let arch = &self.config.architecture;
        let h = arch.hidden_size;

        let embed = arch.vocab_size * h;
        let layers: usize = self.layers.iter().map(|l| l.param_count()).sum();
        let norm = h;
        let lm_head = if arch.tie_word_embeddings {
            0
        } else {
            arch.vocab_size * h
        };

        embed + layers + norm + lm_head
    }

    /// All target weight keys this model expects (Synapse naming).
    pub fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = vec!["embed_tokens.weight".to_string()];
        for (i, layer) in self.layers.iter().enumerate() {
            keys.extend(layer.weight_keys(i));
        }
        keys.push("norm.weight".to_string());
        if !self.config.architecture.tie_word_embeddings {
            keys.push("lm_head.weight".to_string());
        }
        keys
    }

    /// Load weights from source tensors using a name mapper.
    ///
    /// Returns missing (expected but not provided) and unexpected (provided but
    /// not expected) keys. Both should be empty for a correct checkpoint.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<LoadResult, WeightError> {
        let source_keys: Vec<String> = weights.keys().cloned().collect();
        let mapping = mapper.map_keys(&source_keys);

        let expected: HashSet<String> = self.expected_weight_keys().into_iter().collect();
        let mut loaded = HashSet::new();

        for (source, target) in &mapping.mapping {
            if let Some(raw) = weights.get(source) {
                self.set_weight(target, raw)?;
                if expected.contains(target) {
                    loaded.insert(target.clone());
                }
            }
        }

        let missing: Vec<String> = expected.difference(&loaded).cloned().collect();
        let unexpected = mapping.unmapped;

        Ok(LoadResult {
            missing,
            unexpected,
        })
    }

    /// Forward pass: token_ids → logits.
    ///
    /// `token_ids` is a 1-D slice of token indices for a single sequence.
    /// Returns `ModelOutput` with shape `[1, seq_len, vocab_size]`.
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        // 1. Embedding lookup → [seq_len, h]
        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }
        if let Some(scale) = self.config.architecture.embed_scale {
            for v in &mut x {
                *v *= scale;
            }
        }

        // 2. Decoder layers
        for layer in &self.layers {
            x = layer.forward(&x, seq_len, &self.rope_cos, &self.rope_sin);
        }

        // 3. Final norm
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [seq_len, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, seq_len, h, vocab);

        ModelOutput {
            logits,
            shape: [1, seq_len, vocab],
        }
    }

    /// Forward pass dispatched through ComputeBackend.
    #[cfg(feature = "metal")]
    pub fn forward_with_backend(
        &self,
        token_ids: &[u32],
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        for layer in &self.layers {
            x = layer.forward_with_backend(&x, seq_len, backend, &self.rope_cos, &self.rope_sin);
        }

        x = apply_norm_dispatch(&x, &self.final_norm_weight, &*self.final_norm, h, backend);

        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = backend.matmul_t(&x, lm_weight, seq_len, h, vocab);

        ModelOutput {
            logits,
            shape: [1, seq_len, vocab],
        }
    }

    /// Prefill forward pass: process all prompt tokens and populate the KV cache.
    ///
    /// Runs batched attention across all positions (fast) while simultaneously
    /// populating the KV cache for subsequent decode steps.
    ///
    /// After this call, `cache` holds K/V for all `token_ids.len()` positions,
    /// ready for subsequent [`forward_one`] decode steps.
    pub fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        // 1. Embedding lookup → [seq_len, h]
        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }
        if let Some(scale) = self.config.architecture.embed_scale {
            for v in &mut x {
                *v *= scale;
            }
        }

        // 2. Decoder layers — batched forward with cache populate
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_prefill_batched(
                &x,
                seq_len,
                cache.layer_mut(i),
                &self.rope_cos,
                &self.rope_sin,
            );
        }

        // 3. Final norm (last token only)
        let last_hidden = &x[(seq_len - 1) * h..seq_len * h];
        let normed = apply_norm(last_hidden, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [1, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&normed, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Single-token decode step using the KV cache.
    ///
    /// Embeds the token, runs through all decoder layers via
    /// [`DecoderLayer::forward_one`] (each layer appends its K/V to the cache
    /// and attends against the full cached history), then returns logits.
    pub fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let pos = cache.current_len().expect("failed to query cache length");

        // 1. Embedding lookup → [1, h]
        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }
        if let Some(scale) = self.config.architecture.embed_scale {
            for v in &mut x {
                *v *= scale;
            }
        }

        // 2. Decoder layers with KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_one(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin);
        }

        // 3. Final norm
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [1, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Draft forward: runs only the first `n_draft_layers` layers.
    /// Used by self-speculative decoding as a fast approximation.
    /// Populates the KV cache for the draft layers only.
    pub fn forward_one_draft(
        &self,
        token: u32,
        cache: &mut KVCache,
        n_draft_layers: usize,
    ) -> ModelOutput {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let pos = cache.current_len().expect("failed to query cache length");

        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // Only run the first n_draft_layers
        let n = n_draft_layers.min(self.layers.len());
        for (i, layer) in self.layers[..n].iter().enumerate() {
            x = layer.forward_one(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin);
        }
        // Remaining layers: just append zeros to cache to keep positions in sync
        let kv_dim = self.config.attention.num_kv_heads() * self.config.attention.head_dim();
        let zeros = vec![0.0f32; kv_dim];
        for i in n..self.layers.len() {
            cache
                .layer_mut(i)
                .append(&zeros, &zeros)
                .expect("KV cache append failed for draft padding");
        }

        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Prefill forward pass dispatched through ComputeBackend.
    #[cfg(feature = "metal")]
    pub fn forward_prefill_with_backend(
        &self,
        token_ids: &[u32],
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // Use batched prefill (same as CPU path - batched attention + cache populate)
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_prefill_batched(
                &x,
                seq_len,
                cache.layer_mut(i),
                &self.rope_cos,
                &self.rope_sin,
            );
        }

        let last_hidden = &x[(seq_len - 1) * h..seq_len * h];
        let normed = apply_norm_dispatch(
            last_hidden,
            &self.final_norm_weight,
            &*self.final_norm,
            h,
            backend,
        );

        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = backend.matmul_t(&normed, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Single-token decode with backend dispatch.
    #[cfg(feature = "metal")]
    pub fn forward_one_with_backend(
        &self,
        token: u32,
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let pos = cache.current_len().expect("failed to query cache length");

        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // Use GPU-native forward with batched command buffers when Metal is available
        if let crate::metal::ComputeBackend::Metal {
            ref backend,
            ref pool,
        } = backend
        {
            let mut pool = pool.borrow_mut();
            for (i, layer) in self.layers.iter().enumerate() {
                x = crate::metal::gpu_forward::gpu_forward_one(
                    layer,
                    &x,
                    cache.layer_mut(i),
                    pos,
                    &self.rope_cos,
                    &self.rope_sin,
                    backend,
                    &mut pool,
                );
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_one(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin);
            }
        }

        x = apply_norm_dispatch(&x, &self.final_norm_weight, &*self.final_norm, h, backend);

        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = backend.matmul_t(&x, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// GPU-resident single-token decode: all layers in one command buffer.
    ///
    /// Embedding lookup and final norm + LM head stay on CPU.
    /// The 28-layer decoder runs entirely on GPU in a single commit+wait.
    #[cfg(feature = "metal")]
    pub fn forward_one_gpu_resident(
        &self,
        token: u32,
        model_bufs: &mut crate::metal::gpu_buffers::MetalModelBuffers,
        backend: &crate::metal::MetalBackend,
    ) -> ModelOutput {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let num_heads = self.config.attention.num_heads();
        let num_kv_heads = self.config.attention.num_kv_heads();
        let head_dim = self.config.attention.head_dim();
        let inter = self.config.ffn.intermediate_size();
        let has_head_norms = self.layers.first().map_or(false, |l| l.has_head_norms);
        let eps = self
            .layers
            .first()
            .map_or(1e-6, |l| l.attn_norm.eps() as f32);

        // 1. Embedding lookup on CPU -> [h]
        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // 2. All decoder layers on GPU in one command buffer
        x = crate::metal::gpu_forward::gpu_forward_all_layers(
            model_bufs,
            &x,
            num_heads,
            num_kv_heads,
            head_dim,
            h,
            inter,
            has_head_norms,
            eps,
            backend,
        );

        // 3. Final norm on CPU
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection on CPU
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn set_weight(&mut self, key: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        self.validate_weight_shape(key, &tensor.shape)?;
        match key {
            "embed_tokens.weight" => self.embed_tokens = tensor.data.clone(),
            "norm.weight" => self.final_norm_weight = tensor.data.clone(),
            "lm_head.weight" => {
                if !self.config.architecture.tie_word_embeddings {
                    self.lm_head_weight = Some(tensor.data.clone());
                }
                // For tied models, accept the key but reuse embed_tokens.
            }
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        layer.set_weight(field, tensor)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_weight_shape(&self, key: &str, actual: &[usize]) -> Result<(), WeightError> {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let expected = match key {
            "embed_tokens.weight" | "lm_head.weight" => vec![vocab, h],
            "norm.weight" => vec![h],
            _ => return Ok(()),
        };

        if actual != expected {
            return Err(WeightError::ShapeMismatch(format!(
                "{key}: expected {:?}, got {:?}",
                expected, actual
            )));
        }

        Ok(())
    }
}

/// Parse `"layers[5].attention.w_q"` → `(5, "attention.w_q")`.
fn parse_layer_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("layers[")?;
    let bracket = rest.find(']')?;
    let idx: usize = rest[..bracket].parse().ok()?;
    let field = rest[bracket + 1..].strip_prefix('.')?;
    Some((idx, field))
}

impl super::traits::Model for CausalLM {
    fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        CausalLM::forward(self, token_ids)
    }

    fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput {
        CausalLM::forward_prefill(self, token_ids, cache)
    }

    #[cfg(feature = "metal")]
    fn forward_prefill_gpu(
        &self,
        token_ids: &[u32],
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        CausalLM::forward_prefill_with_backend(self, token_ids, cache, backend)
    }

    fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput {
        CausalLM::forward_one(self, token, cache)
    }

    fn forward_one_draft(&self, token: u32, cache: &mut KVCache, n_layers: usize) -> ModelOutput {
        CausalLM::forward_one_draft(self, token, cache, n_layers)
    }

    #[cfg(feature = "metal")]
    fn forward_one_gpu(
        &self,
        token: u32,
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        CausalLM::forward_one_with_backend(self, token, cache, backend)
    }

    #[cfg(feature = "metal")]
    fn forward_one_gpu_resident(
        &self,
        token: u32,
        model_bufs: &mut crate::metal::gpu_buffers::MetalModelBuffers,
        backend: &crate::metal::MetalBackend,
    ) -> ModelOutput {
        CausalLM::forward_one_gpu_resident(self, token, model_bufs, backend)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::kv_cache::KVCache;
    use crate::model::ModelBuilder;
    use crate::weight_loading::{RawTensor, WeightMapper};

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "CausalLMTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 64,
                num_layers: 4,
                vocab_size: 256,
                max_sequence_length: 64,
                tie_word_embeddings: true,
                embed_scale: None,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 128,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 64,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let fake = |shape: Vec<usize>, seed: u32| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(n, seed)),
                shape,
            }
        };

        let mut w = HashMap::new();
        w.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h], 1));
        for i in 0..nl {
            let s = (i as u32 + 1) * 100;
            w.insert(
                format!("model.layers.{i}.input_layernorm.weight"),
                fake(vec![h], s),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                fake(vec![q_dim, h], s + 1),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                fake(vec![kv_dim, h], s + 2),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                fake(vec![kv_dim, h], s + 3),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                fake(vec![h, q_dim], s + 4),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 5),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 6),
            );
            w.insert(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                fake(vec![h], s + 7),
            );
            w.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                fake(vec![inter, h], s + 8),
            );
            w.insert(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                fake(vec![inter, h], s + 9),
            );
            w.insert(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                fake(vec![h, inter], s + 10),
            );
        }
        w.insert("model.norm.weight".into(), fake(vec![h], 9999));
        w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
        w
    }

    fn build_test_model(cfg: &ModelConfig) -> CausalLM {
        let mut model = ModelBuilder::from_config(cfg);
        let weights = generate_fake_hf_weights(cfg);
        let mapper = WeightMapper::qwen3();
        let result = model.load_weights(weights, &mapper).unwrap();
        assert!(
            result.missing.is_empty(),
            "Missing keys: {:?}",
            result.missing
        );
        model
    }

    fn make_cache(cfg: &ModelConfig) -> KVCache {
        KVCache::new(
            cfg.architecture.num_layers,
            cfg.architecture.max_sequence_length,
            cfg.attention.num_kv_heads(),
            cfg.attention.head_dim(),
        )
        .unwrap()
    }

    /// forward_prefill() logits must match forward() last-position logits within 1e-5.
    #[test]
    fn forward_prefill_matches_forward() {
        let cfg = test_config();
        let model = build_test_model(&cfg);
        let vocab = cfg.architecture.vocab_size;

        let prompts: Vec<Vec<u32>> =
            vec![vec![1, 2, 3, 4], vec![10, 20, 30], vec![5, 5, 5, 5, 5, 5]];

        for prompt in &prompts {
            let mut cache = make_cache(&cfg);
            let prefill_out = model.forward_prefill(prompt, &mut cache);

            let full_out = model.forward(prompt);
            let seq_len = full_out.shape[1];
            let full_last_logits = &full_out.logits[(seq_len - 1) * vocab..seq_len * vocab];

            assert_eq!(prefill_out.logits.len(), full_last_logits.len());
            for (i, (&a, &b)) in prefill_out
                .logits
                .iter()
                .zip(full_last_logits.iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-5,
                    "Logit {i} mismatch: prefill={a}, full={b} (prompt={:?})",
                    prompt
                );
            }
        }
    }

    /// forward_prefill(prompt) then forward_one(token) must match
    /// forward(prompt ++ [token]) last-position logits within 1e-5.
    #[test]
    fn prefill_then_one_matches_full_forward() {
        let cfg = test_config();
        let model = build_test_model(&cfg);
        let vocab = cfg.architecture.vocab_size;

        let prompt = vec![1u32, 2, 3, 4];
        let next_token = 42u32;

        // Cached path: prefill + single decode step
        let mut cache = make_cache(&cfg);
        let _ = model.forward_prefill(&prompt, &mut cache);
        let one_out = model.forward_one(next_token, &mut cache);

        // Reference: full forward on concatenated sequence
        let mut full_tokens = prompt.clone();
        full_tokens.push(next_token);
        let full_out = model.forward(&full_tokens);
        let seq_len = full_out.shape[1];
        let full_last_logits = &full_out.logits[(seq_len - 1) * vocab..seq_len * vocab];

        assert_eq!(one_out.logits.len(), full_last_logits.len());
        for (i, (&a, &b)) in one_out
            .logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Logit {i} mismatch: cached={a}, full={b}",
            );
        }
    }

    /// Multi-step decode: forward_prefill(prompt) then forward_one() × 3 must match
    /// forward(prompt ++ decode_tokens) last-position logits within 1e-5.
    /// Verifies KV-cache accumulation across multiple decode steps.
    #[test]
    fn multi_step_decode_matches_full_forward() {
        let cfg = test_config();
        let model = build_test_model(&cfg);
        let vocab = cfg.architecture.vocab_size;

        let prompt = vec![1u32, 2, 3];
        let decode_tokens = vec![10u32, 20, 30];

        // Cached path: prefill + multiple decode steps
        let mut cache = make_cache(&cfg);
        let _ = model.forward_prefill(&prompt, &mut cache);

        let mut cached_logits = Vec::new();
        for &tok in &decode_tokens {
            let out = model.forward_one(tok, &mut cache);
            cached_logits = out.logits;
        }

        // Verify cache accumulated correctly
        let expected_len = prompt.len() + decode_tokens.len();
        assert_eq!(
            cache.current_len().unwrap(),
            expected_len,
            "Cache should hold prompt + decode tokens"
        );

        // Reference: full forward on all tokens
        let mut all_tokens = prompt.clone();
        all_tokens.extend_from_slice(&decode_tokens);
        let full_out = model.forward(&all_tokens);
        let seq_len = full_out.shape[1];
        let full_last_logits = &full_out.logits[(seq_len - 1) * vocab..seq_len * vocab];

        assert_eq!(cached_logits.len(), full_last_logits.len());
        for (i, (&a, &b)) in cached_logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Logit {i} mismatch after multi-step decode: cached={a}, full={b}",
            );
        }
    }
}
