//! Transformer training throughput benchmark.
//!
//! End-to-end: 4-layer encoder, d=256, seq=128, batch=32.
//! Measures tokens/sec through frozen backbone + trained classification head.
//! Target: >= 2000 tokens/sec (release) / >= 200 tokens/sec (debug).

use std::time::Instant;

use synapse_autograd::{backward, Graph, Tensor};
use synapse_nn::init::xavier_uniform;
use synapse_nn::{
    Activation, Embedding, MeanPool1d, Module, SinusoidalPositionalEncoding, TransformerEncoder,
    TransformerEncoderConfig,
};
use synapse_optim::{Adam, Optimizer, Param};

// Release mode uses the full model; debug uses a smaller model so the test
// completes in reasonable time while still exercising the full pipeline.
const VOCAB_SIZE: usize = 512;

#[cfg(not(debug_assertions))]
const SEQ_LEN: usize = 128;
#[cfg(debug_assertions)]
const SEQ_LEN: usize = 64;

#[cfg(not(debug_assertions))]
const D_MODEL: usize = 256;
#[cfg(debug_assertions)]
const D_MODEL: usize = 64;

const N_HEADS: usize = 4;

#[cfg(not(debug_assertions))]
const D_FF: usize = 512;
#[cfg(debug_assertions)]
const D_FF: usize = 128;

const N_LAYERS: usize = 4;
const NUM_CLASSES: usize = 10;

#[cfg(not(debug_assertions))]
const BATCH_SIZE: usize = 32;
#[cfg(debug_assertions)]
const BATCH_SIZE: usize = 16;

struct TransformerModel {
    embedding: Embedding,
    pos_encoding: SinusoidalPositionalEncoding,
    encoder: TransformerEncoder,
    pool: MeanPool1d,
    fc_w: Tensor,
    fc_b: Tensor,
    optimizer: Adam,
}

impl TransformerModel {
    fn new() -> Self {
        let config = TransformerEncoderConfig {
            d_model: D_MODEL,
            n_heads: N_HEADS,
            d_ff: D_FF,
            n_layers: N_LAYERS,
            dropout: 0.0,
            activation: Activation::GELU,
        };

        let mut encoder = TransformerEncoder::new(&config);
        encoder.set_training(false);

        let mut embedding = Embedding::new(VOCAB_SIZE, D_MODEL);
        embedding.set_training(false);

        TransformerModel {
            embedding,
            pos_encoding: SinusoidalPositionalEncoding::new(SEQ_LEN + 1, D_MODEL),
            encoder,
            pool: MeanPool1d::new(),
            fc_w: xavier_uniform(&[D_MODEL, NUM_CLASSES]),
            fc_b: Tensor::zeros(&[1, NUM_CLASSES]),
            optimizer: Adam::new(0.001),
        }
    }

    fn backbone(&self, input: &Tensor) -> Tensor {
        let embedded = self.embedding.forward(input);
        let positioned = self.pos_encoding.forward(&embedded);
        let encoded = self.encoder.forward(&positioned);
        self.pool.forward(&encoded)
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        let features = self.backbone(input);

        let mut graph = Graph::new();
        let x = graph.variable(features, false);
        let t = graph.variable(target.clone(), false);
        let fc_w_var = graph.variable(self.fc_w.clone(), true);
        let fc_b_var = graph.variable(self.fc_b.clone(), true);

        let logits = graph.matmul(x, fc_w_var);
        let logits = graph.add(logits, fc_b_var);
        let loss = graph.cross_entropy_loss(logits, t);
        let loss_val = graph.data(loss).data[0];

        backward(&mut graph, loss);

        let shapes = [self.fc_w.shape.clone(), self.fc_b.shape.clone()];
        let mut params: Vec<Param> = [(&self.fc_w, fc_w_var), (&self.fc_b, fc_b_var)]
            .iter()
            .map(|(tensor, var)| {
                let grad = graph.grad(*var).map(|g| g.data.clone());
                let mut p = Param::new(tensor.data.clone());
                p.grad = grad;
                p
            })
            .collect();

        self.optimizer.step(&mut params);

        self.fc_w = Tensor::new(params[0].data.clone(), shapes[0].clone());
        self.fc_b = Tensor::new(params[1].data.clone(), shapes[1].clone());

        loss_val
    }
}

fn make_batch(seed: u32) -> (Tensor, Tensor) {
    let mut state = seed.wrapping_mul(2654435761);
    let input_data: Vec<f32> = (0..BATCH_SIZE * SEQ_LEN)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state % VOCAB_SIZE as u32) as f32
        })
        .collect();

    let mut target_data = vec![0.0f32; BATCH_SIZE * NUM_CLASSES];
    for i in 0..BATCH_SIZE {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let class = (state as usize) % NUM_CLASSES;
        target_data[i * NUM_CLASSES + class] = 1.0;
    }

    (
        Tensor::new(input_data, vec![BATCH_SIZE, SEQ_LEN]),
        Tensor::new(target_data, vec![BATCH_SIZE, NUM_CLASSES]),
    )
}

#[test]
fn transformer_throughput_tokens_per_sec() {
    let mut model = TransformerModel::new();

    // Generate training batches — more steps in release, fewer in debug
    let n_batches = if cfg!(debug_assertions) { 5 } else { 10 };
    let batches: Vec<(Tensor, Tensor)> = (0..n_batches).map(|i| make_batch(i as u32 + 42)).collect();

    // Warmup
    for (input, target) in &batches[..1] {
        model.train_step(input, target);
    }

    // Benchmark
    let num_steps = batches.len();
    let start = Instant::now();
    for (input, target) in &batches {
        model.train_step(input, target);
    }
    let elapsed = start.elapsed();

    let total_tokens = num_steps * BATCH_SIZE * SEQ_LEN;
    let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    eprintln!(
        "Transformer throughput: {} tokens in {:.3}s = {:.0} tokens/sec",
        total_tokens,
        elapsed.as_secs_f64(),
        tokens_per_sec
    );
    eprintln!(
        "  Config: {}L, d={}, {}H, ff={}, seq={}, batch={}",
        N_LAYERS, D_MODEL, N_HEADS, D_FF, SEQ_LEN, BATCH_SIZE
    );

    // Build-mode-aware threshold
    let threshold = if cfg!(debug_assertions) {
        200.0
    } else {
        2000.0
    };
    assert!(
        tokens_per_sec >= threshold,
        "Expected >= {:.0} tokens/sec, got {:.0}",
        threshold,
        tokens_per_sec
    );
}
