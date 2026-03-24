//! KV-cache decode correctness: verify that KV-cache-based generation produces
//! identical output to full-recompute generation, with K/V value validation
//! and tests across varying prompt lengths.
//!
//! Test cases:
//! 1. Generate 20 tokens via cached (forward_prefill + forward_one) vs
//!    full-recompute (forward on all tokens each step) — bit-exact token IDs.
//! 2. KV-cache values at shared prefix positions match within 1e-6.
//! 3. Deterministic: identical runs produce identical output.
//! 4. Varying prompt lengths: 1, 8, 128 tokens.

use std::collections::HashMap;

use synapse_inference::config::*;
use synapse_inference::kv_cache::KVCache;
use synapse_inference::model::ModelBuilder;
use synapse_inference::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

fn test_config() -> ModelConfig {
    ModelConfig {
        name: "KVDecodeTest".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 256,
            max_sequence_length: 256,
            tie_word_embeddings: true,
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
            max_position_embeddings: 256,
            style: Default::default(),
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
        w.insert(format!("model.layers.{i}.input_layernorm.weight"), fake(vec![h], s));
        w.insert(format!("model.layers.{i}.self_attn.q_proj.weight"), fake(vec![q_dim, h], s + 1));
        w.insert(format!("model.layers.{i}.self_attn.k_proj.weight"), fake(vec![kv_dim, h], s + 2));
        w.insert(format!("model.layers.{i}.self_attn.v_proj.weight"), fake(vec![kv_dim, h], s + 3));
        w.insert(format!("model.layers.{i}.self_attn.o_proj.weight"), fake(vec![h, q_dim], s + 4));
        w.insert(format!("model.layers.{i}.self_attn.q_norm.weight"), fake(vec![cfg.attention.head_dim()], s + 5));
        w.insert(format!("model.layers.{i}.self_attn.k_norm.weight"), fake(vec![cfg.attention.head_dim()], s + 6));
        w.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h], s + 7));
        w.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h], s + 8));
        w.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h], s + 9));
        w.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter], s + 10));
    }
    w.insert("model.norm.weight".into(), fake(vec![h], 9999));
    w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
    w
}

fn build_model(cfg: &ModelConfig) -> synapse_inference::model::CausalLM {
    let mut model = ModelBuilder::from_config(cfg);
    let weights = generate_fake_hf_weights(cfg);
    let mapper = WeightMapper::qwen3();
    let result = model.load_weights(weights, &mapper).unwrap();
    assert!(result.missing.is_empty(), "Missing keys: {:?}", result.missing);
    model
}

fn make_cache(cfg: &ModelConfig, max_seq: usize) -> KVCache {
    KVCache::new(
        cfg.architecture.num_layers,
        max_seq,
        cfg.attention.num_kv_heads(),
        cfg.attention.head_dim(),
    )
    .unwrap()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// Generate tokens via KV-cache path: forward_prefill + forward_one loop.
fn generate_cached(
    model: &synapse_inference::model::CausalLM,
    prompt: &[u32],
    num_tokens: usize,
    cache: &mut KVCache,
) -> Vec<u32> {
    let prefill_out = model.forward_prefill(prompt, cache);
    let mut all_tokens = prompt.to_vec();
    all_tokens.push(argmax(&prefill_out.logits));

    for _ in 1..num_tokens {
        let out = model.forward_one(*all_tokens.last().unwrap(), cache);
        all_tokens.push(argmax(&out.logits));
    }
    all_tokens
}

/// Generate tokens via full-recompute path: forward on all tokens each step.
fn generate_recompute(
    model: &synapse_inference::model::CausalLM,
    prompt: &[u32],
    num_tokens: usize,
) -> Vec<u32> {
    let vocab = model.config.architecture.vocab_size;
    let full_out = model.forward(prompt);
    let seq_len = full_out.shape[1];
    let last_logits = &full_out.logits[(seq_len - 1) * vocab..seq_len * vocab];
    let mut all_tokens = prompt.to_vec();
    all_tokens.push(argmax(last_logits));

    for _ in 1..num_tokens {
        let out = model.forward(&all_tokens);
        let seq_len = out.shape[1];
        let last_logits = &out.logits[(seq_len - 1) * vocab..seq_len * vocab];
        all_tokens.push(argmax(last_logits));
    }
    all_tokens
}

// ── Test 1 & 3: Bit-exact token IDs ────────────────────────────────

/// Generate 20 tokens via KV-cache (forward_prefill + forward_one) and via
/// full-recompute (forward on all tokens each step). Assert token IDs are
/// IDENTICAL (bit-exact, not approximate).
#[test]
fn kvcache_decode_tokens_identical_to_recompute() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    let mut cache = make_cache(&cfg, prompt.len() + num_tokens);
    let cached = generate_cached(&model, &prompt, num_tokens, &mut cache);
    let recompute = generate_recompute(&model, &prompt, num_tokens);

    assert_eq!(
        cached, recompute,
        "KV-cache decode must produce bit-exact identical token IDs to full-recompute"
    );
    assert_eq!(cached.len(), prompt.len() + num_tokens);
}

// ── Test 4: KV-cache values match at shared prefix positions ────────

/// Verify KV-cache values at each position satisfy the causal invariant:
/// prefilling [t0..tN] and [t0..tN, extra...] produces identical K/V at
/// positions 0..N-1 within 1e-6.
#[test]
fn kvcache_values_match_at_shared_positions() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let stride = cfg.attention.num_kv_heads() * cfg.attention.head_dim();

    let short = vec![1u32, 2, 3, 4, 5];
    let long = vec![1u32, 2, 3, 4, 5, 100, 200];

    let mut cache_short = make_cache(&cfg, short.len() + 1);
    let _ = model.forward_prefill(&short, &mut cache_short);

    let mut cache_long = make_cache(&cfg, long.len() + 1);
    let _ = model.forward_prefill(&long, &mut cache_long);

    for layer in 0..cfg.architecture.num_layers {
        let (k_short, v_short, len_short) = cache_short.get(layer).unwrap();
        let (k_long, v_long, len_long) = cache_long.get(layer).unwrap();

        assert_eq!(len_short, short.len());
        assert_eq!(len_long, long.len());

        for pos in 0..short.len() {
            let ks = &k_short[pos * stride..(pos + 1) * stride];
            let kl = &k_long[pos * stride..(pos + 1) * stride];
            for (i, (&a, &b)) in ks.iter().zip(kl.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Layer {layer} pos {pos} K[{i}]: short={a} vs long={b}"
                );
            }

            let vs = &v_short[pos * stride..(pos + 1) * stride];
            let vl = &v_long[pos * stride..(pos + 1) * stride];
            for (i, (&a, &b)) in vs.iter().zip(vl.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Layer {layer} pos {pos} V[{i}]: short={a} vs long={b}"
                );
            }
        }

        // Sanity: K/V values should be finite
        for &v in k_short.iter().chain(v_short.iter()) {
            assert!(v.is_finite(), "Layer {layer}: non-finite KV-cache value");
        }
    }
}

// ── Test 5: Deterministic with seed ─────────────────────────────────

/// Two identical KV-cache decode runs produce identical token sequences
/// (deterministic greedy, same weights, same prompt).
#[test]
fn kvcache_decode_deterministic() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_tokens = 20;

    let mut cache1 = make_cache(&cfg, prompt.len() + num_tokens);
    let run1 = generate_cached(&model, &prompt, num_tokens, &mut cache1);

    let mut cache2 = make_cache(&cfg, prompt.len() + num_tokens);
    let run2 = generate_cached(&model, &prompt, num_tokens, &mut cache2);

    assert_eq!(run1, run2, "KV-cache decode must be deterministic across runs");

    // Also verify KV-cache contents are identical
    for layer in 0..cfg.architecture.num_layers {
        let (k1, v1, len1) = cache1.get(layer).unwrap();
        let (k2, v2, len2) = cache2.get(layer).unwrap();
        assert_eq!(len1, len2);
        assert_eq!(k1, k2, "Layer {layer} K-cache differs between runs");
        assert_eq!(v1, v2, "Layer {layer} V-cache differs between runs");
    }
}

// ── Test 6: Varying prompt lengths ──────────────────────────────────

/// KV-cache decode produces identical tokens to full-recompute for
/// varying prompt lengths: 1, 8, and 128 tokens.
#[test]
fn kvcache_decode_varying_prompt_lengths() {
    let cfg = test_config();
    let model = build_model(&cfg);
    let num_tokens = 20;

    let prompts: Vec<Vec<u32>> = vec![
        vec![42],                                      // 1 token
        vec![1, 2, 3, 4, 5, 6, 7, 8],                 // 8 tokens
        (0..128).map(|i| (i % 256) as u32).collect(),  // 128 tokens
    ];

    for prompt in &prompts {
        let mut cache = make_cache(&cfg, prompt.len() + num_tokens);
        let cached = generate_cached(&model, prompt, num_tokens, &mut cache);
        let recompute = generate_recompute(&model, prompt, num_tokens);

        assert_eq!(
            cached, recompute,
            "Mismatch for prompt length {}: cached tokens != recompute tokens",
            prompt.len()
        );
        assert_eq!(
            cached.len(),
            prompt.len() + num_tokens,
            "Wrong total length for prompt length {}",
            prompt.len()
        );
    }
}
