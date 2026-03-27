//! End-to-end transformer test: Train 4-layer encoder on synthetic sequence
//! classification. Must reach >85% accuracy in 5 epochs.
//!
//! Architecture: Embedding(500, 64) → SinusoidalPE → TransformerEncoder(4L, d=64, 4H, ff=256)
//!             → MeanPool1d → Linear(64, NUM_CLASSES)
//! Only the classification head is trained through autograd (linear probe).

use synapse_autograd::{backward, Graph, NoGradGuard, Tensor};
use synapse_nn::init::xavier_uniform;
use synapse_nn::{
    Activation, Embedding, MeanPool1d, Module, SinusoidalPositionalEncoding, TransformerEncoder,
    TransformerEncoderConfig,
};
use synapse_optim::{Adam, Optimizer, Param};
use synapse_train::{TrainLoop, Trainer, TrainerConfig};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const VOCAB_SIZE: usize = 500;
const SEQ_LEN: usize = 32;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const D_FF: usize = 256;
const N_LAYERS: usize = 4;
const NUM_CLASSES: usize = 4;
const BATCH_SIZE: usize = 32;

struct TransformerClassifier {
    embedding: Embedding,
    pos_encoding: SinusoidalPositionalEncoding,
    encoder: TransformerEncoder,
    pool: MeanPool1d,
    fc_w: Tensor,
    fc_b: Tensor,
    optimizer: Adam,
    train_data: Vec<(Tensor, Tensor)>,
    val_data: Vec<(Tensor, Tensor)>,
}

impl TransformerClassifier {
    fn new(train_data: Vec<(Tensor, Tensor)>, val_data: Vec<(Tensor, Tensor)>) -> Self {
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

        TransformerClassifier {
            embedding,
            pos_encoding: SinusoidalPositionalEncoding::new(SEQ_LEN + 1, D_MODEL),
            encoder,
            pool: MeanPool1d::new(),
            fc_w: xavier_uniform(&[D_MODEL, NUM_CLASSES]),
            fc_b: Tensor::zeros(&[1, NUM_CLASSES]),
            optimizer: Adam::new(0.003),
            train_data,
            val_data,
        }
    }

    fn backbone(&self, input: &Tensor) -> Tensor {
        let embedded = self.embedding.forward(input);
        let positioned = self.pos_encoding.forward(&embedded);
        let encoded = self.encoder.forward(&positioned);
        self.pool.forward(&encoded)
    }

    fn predict(&self, input: &Tensor) -> Tensor {
        let features = self.backbone(input);
        features.matmul(&self.fc_w).add_broadcast(&self.fc_b)
    }
}

impl TrainLoop for TransformerClassifier {
    fn train_batches(&self) -> Vec<(Tensor, Tensor)> {
        self.train_data.clone()
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

    fn validate(&mut self) -> Option<f32> {
        let _guard = NoGradGuard::new();
        let mut correct = 0usize;
        let mut total = 0usize;

        for (input, target) in &self.val_data {
            let logits = self.predict(input);
            let batch = input.shape[0];
            for i in 0..batch {
                let pred = (0..NUM_CLASSES)
                    .max_by(|&a, &b| {
                        logits.data[i * NUM_CLASSES + a]
                            .partial_cmp(&logits.data[i * NUM_CLASSES + b])
                            .unwrap()
                    })
                    .unwrap();
                let actual = (0..NUM_CLASSES)
                    .max_by(|&a, &b| {
                        target.data[i * NUM_CLASSES + a]
                            .partial_cmp(&target.data[i * NUM_CLASSES + b])
                            .unwrap()
                    })
                    .unwrap();
                if pred == actual {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = correct as f32 / total.max(1) as f32;
        eprintln!(
            "  val accuracy: {:.2}% ({}/{})",
            accuracy * 100.0,
            correct,
            total
        );
        Some(1.0 - accuracy)
    }
}

/// Generate synthetic sequence classification data.
///
/// Each class uses a distinctive, non-overlapping token range so that
/// mean-pooled embeddings are linearly separable even with a frozen backbone.
///
/// Class 0: tokens in [10, 50)   (92% signal, 8% random)
/// Class 1: tokens in [120, 180) (92% signal, 8% random)
/// Class 2: tokens in [250, 330) (92% signal, 8% random)
/// Class 3: tokens in [380, 460) (92% signal, 8% random)
fn generate_data(n_samples: usize, batch_size: usize, seed: u64) -> Vec<(Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_batches = n_samples / batch_size;

    let class_ranges: [(usize, usize); NUM_CLASSES] =
        [(10, 50), (120, 180), (250, 330), (380, 460)];

    let mut batches = Vec::with_capacity(n_batches);

    for _ in 0..n_batches {
        let mut input_data = Vec::with_capacity(batch_size * SEQ_LEN);
        let mut target_data = Vec::with_capacity(batch_size * NUM_CLASSES);

        for _ in 0..batch_size {
            let class = rng.gen_range(0..NUM_CLASSES);
            let (lo, hi) = class_ranges[class];

            for _ in 0..SEQ_LEN {
                let word = if rng.gen::<f32>() < 0.92 {
                    rng.gen_range(lo..hi)
                } else {
                    rng.gen_range(2..VOCAB_SIZE)
                };
                input_data.push(word as f32);
            }

            let mut one_hot = vec![0.0f32; NUM_CLASSES];
            one_hot[class] = 1.0;
            target_data.extend_from_slice(&one_hot);
        }

        batches.push((
            Tensor::new(input_data, vec![batch_size, SEQ_LEN]),
            Tensor::new(target_data, vec![batch_size, NUM_CLASSES]),
        ));
    }

    batches
}

#[test]
fn transformer_e2e_4_layer_above_85_percent() {
    let train_data = generate_data(1024, BATCH_SIZE, 42);
    let val_data = generate_data(256, BATCH_SIZE, 123);

    let mut model = TransformerClassifier::new(train_data, val_data);
    let mut trainer = Trainer::new(TrainerConfig { epochs: 5 });
    let history = trainer.fit(&mut model);

    assert_eq!(history.epochs.len(), 5);

    let final_val_loss = history.epochs.last().unwrap().val_loss.unwrap();
    let final_accuracy = 1.0 - final_val_loss;
    eprintln!("Final accuracy: {:.2}%", final_accuracy * 100.0);
    assert!(
        final_accuracy >= 0.85,
        "Expected accuracy >= 85%, got {:.2}%",
        final_accuracy * 100.0
    );
}
