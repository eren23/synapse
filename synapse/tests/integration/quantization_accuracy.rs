//! Quantization accuracy test: INT8 vs f32 logits comparison.
//! Top-1 agreement must be >= 99% across multiple input sequences.

use synapse_inference::config::*;
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::{quantize_model, QuantizedCausalLM};
use synapse_inference::weight_loading::AlignedBuffer;

fn test_config() -> ModelConfig {
    ModelConfig {
        name: "QuantAccuracyTest".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 32,
            num_layers: 2,
            vocab_size: 64,
            max_sequence_length: 32,
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
            max_position_embeddings: 32,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

/// Deterministic pseudo-random weight generator with Xavier-scale magnitude.
fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

fn fill_model_weights(model: &mut synapse_inference::model::CausalLM) {
    let cfg = &model.config;
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    model.embed_tokens = AlignedBuffer::from_vec(gen_weights(vocab * h, 1));
    model.final_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.w_q = AlignedBuffer::from_vec(gen_weights(q_dim * h, s + 1));
        layer.w_k = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 2));
        layer.w_v = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 3));
        layer.w_o = AlignedBuffer::from_vec(gen_weights(h * q_dim, s + 4));
        layer.ffn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.ffn_gate = AlignedBuffer::from_vec(gen_weights(inter * h, s + 5));
        layer.ffn_up = AlignedBuffer::from_vec(gen_weights(inter * h, s + 6));
        layer.ffn_down = AlignedBuffer::from_vec(gen_weights(h * inter, s + 7));
    }
}

/// Argmax of a logit slice.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn quantization_accuracy_top1_agreement_99_percent() {
    let cfg = test_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let quantized: QuantizedCausalLM = quantize_model(&model);
    let vocab = cfg.architecture.vocab_size;

    // Test sequences of varying lengths — enough positions for robust statistics
    let test_sequences: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4],
        vec![10, 20, 30, 40, 50],
        vec![0, 1, 2],
        vec![5, 5, 5, 5, 5, 5],
        vec![42, 17, 3, 12, 7],
        vec![33, 11, 22, 44],
        vec![60, 0, 30, 15],
        vec![7, 14, 21, 28, 35],
        vec![50, 25, 12, 6, 3],
        vec![2, 4, 8, 16, 32],
        vec![63, 62, 61, 60],
        vec![1, 1, 1, 1, 1],
        vec![40, 41, 42, 43, 44, 45],
        vec![55, 10, 33, 22],
        vec![9, 18, 27, 36, 45],
        vec![3, 6, 9, 12],
    ];

    let mut total_positions = 0usize;
    let mut agree_positions = 0usize;

    for seq in &test_sequences {
        let f32_output = model.forward(seq);
        let q_output = quantized.forward(seq);
        let seq_len = seq.len();

        assert_eq!(f32_output.logits.len(), q_output.logits.len());

        // Compare top-1 prediction at each position
        for pos in 0..seq_len {
            let f32_logits = &f32_output.logits[pos * vocab..(pos + 1) * vocab];
            let q_logits = &q_output.logits[pos * vocab..(pos + 1) * vocab];

            let f32_top1 = argmax(f32_logits);
            let q_top1 = argmax(q_logits);

            if f32_top1 == q_top1 {
                agree_positions += 1;
            }
            total_positions += 1;
        }
    }

    let agreement = agree_positions as f64 / total_positions as f64;
    eprintln!(
        "INT8 vs f32 top-1 agreement: {}/{} = {:.1}%",
        agree_positions,
        total_positions,
        agreement * 100.0
    );

    assert!(
        agreement >= 0.99,
        "Top-1 agreement {:.1}% < 99% ({}/{})",
        agreement * 100.0,
        agree_positions,
        total_positions
    );
}

#[test]
fn quantization_accuracy_logits_finite() {
    let cfg = test_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let quantized = quantize_model(&model);

    let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let output = quantized.forward(&token_ids);

    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Quantized model produced non-finite logits"
    );
}

#[test]
fn quantization_accuracy_nrmse_below_1_percent() {
    let cfg = test_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let quantized = quantize_model(&model);

    let token_ids: Vec<u32> = vec![1, 2, 3, 4];
    let f32_output = model.forward(&token_ids);
    let q_output = quantized.forward(&token_ids);

    // Compute normalized RMSE
    let l2_err: f32 = f32_output
        .logits
        .iter()
        .zip(q_output.logits.iter())
        .map(|(f, q)| (f - q).powi(2))
        .sum();
    let l2_ref: f32 = f32_output.logits.iter().map(|f| f.powi(2)).sum();

    let nrmse = if l2_ref > 0.0 {
        (l2_err / l2_ref).sqrt()
    } else {
        0.0
    };

    eprintln!("Quantized model NRMSE: {nrmse:.6}");
    assert!(
        nrmse < 0.01,
        "Quantized model NRMSE {nrmse:.6} exceeds 1%"
    );
}
