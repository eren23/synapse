//! Text classification: Transformer encoder on synthetic binary classification.
//!
//! Pipeline: Embedding → SinusoidalPositionalEncoding → TransformerEncoder(2 layers,
//!           d_model=64, 4 heads, d_ff=256) → MeanPool1d → Linear → cross-entropy loss
//! Optimizer: Adam with linear warmup.
//! Demonstrates the Trainer API with a transformer backbone.

use synapse_autograd::{backward, Graph, NoGradGuard, Tensor};
use synapse_nn::init::xavier_uniform;
use synapse_nn::{
    Activation, Embedding, MeanPool1d, Module, SinusoidalPositionalEncoding, TransformerEncoder,
    TransformerEncoderConfig,
};
use synapse_optim::{Adam, LinearWarmup, Optimizer, Param};
use synapse_train::{TrainLoop, Trainer, TrainerConfig};

use rand::Rng;

const VOCAB_SIZE: usize = 100;
const SEQ_LEN: usize = 16;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const D_FF: usize = 256;
const N_LAYERS: usize = 2;
const NUM_CLASSES: usize = 2;
const BATCH_SIZE: usize = 32;

struct TextClassifier {
    // Backbone (eager forward, frozen during training)
    embedding: Embedding,
    pos_encoding: SinusoidalPositionalEncoding,
    encoder: TransformerEncoder,
    pool: MeanPool1d,
    // Classification head: weights in [in, out] layout for direct matmul
    fc_w: Tensor, // [D_MODEL, NUM_CLASSES]
    fc_b: Tensor, // [1, NUM_CLASSES]
    optimizer: Adam,
    warmup: LinearWarmup,
    train_data: Vec<(Tensor, Tensor)>,
    val_data: Vec<(Tensor, Tensor)>,
}

impl TextClassifier {
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

        TextClassifier {
            embedding,
            pos_encoding: SinusoidalPositionalEncoding::new(SEQ_LEN + 1, D_MODEL),
            encoder,
            pool: MeanPool1d::new(),
            fc_w: xavier_uniform(&[D_MODEL, NUM_CLASSES]),
            fc_b: Tensor::zeros(&[1, NUM_CLASSES]),
            optimizer: Adam::new(0.001),
            warmup: LinearWarmup::new(0.001, 50),
            train_data,
            val_data,
        }
    }

    /// Eager forward through the frozen backbone.
    fn backbone(&self, input: &Tensor) -> Tensor {
        let embedded = self.embedding.forward(input); // [B, S] -> [B, S, D]
        let positioned = self.pos_encoding.forward(&embedded); // [B, S, D]
        let encoded = self.encoder.forward(&positioned); // [B, S, D]
        self.pool.forward(&encoded) // [B, D]
    }

    /// Eager forward for inference.
    fn predict(&self, input: &Tensor) -> Tensor {
        let features = self.backbone(input);
        features.matmul(&self.fc_w).add_broadcast(&self.fc_b)
    }
}

impl TrainLoop for TextClassifier {
    fn train_batches(&self) -> Vec<(Tensor, Tensor)> {
        self.train_data.clone()
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        // Advance warmup and update optimizer lr
        self.warmup.step();
        self.optimizer.lr = self.warmup.get_lr();

        // Backbone forward (eager, frozen)
        let features = self.backbone(input); // [B, D]

        // Classification head through graph for gradient computation
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

        // Collect params + grads, optimizer step
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
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;

        for (input, target) in &self.val_data {
            let logits = self.predict(input);
            let batch = input.shape[0];

            // Cross-entropy loss (eager)
            let log_sm = logits.log_softmax_axis(1);
            let per_elem: f32 = target
                .data
                .iter()
                .zip(log_sm.data.iter())
                .map(|(&t, &ls)| t * ls)
                .sum();
            total_loss += -per_elem / batch as f32;

            // Accuracy
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

        let n = self.val_data.len().max(1) as f32;
        let avg_loss = total_loss / n;
        let accuracy = correct as f32 / total.max(1) as f32;
        println!(
            "  Validation: loss={:.4}, accuracy={:.2}%",
            avg_loss,
            accuracy * 100.0
        );
        Some(avg_loss)
    }
}

/// Generate synthetic binary text classification data.
///
/// Class 0: sequences dominated by word IDs in [5..30]
/// Class 1: sequences dominated by word IDs in [50..75]
fn generate_data(n_samples: usize, batch_size: usize) -> Vec<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let n_batches = n_samples / batch_size;
    let mut batches = Vec::with_capacity(n_batches);

    for _ in 0..n_batches {
        let mut input_data = Vec::with_capacity(batch_size * SEQ_LEN);
        let mut target_data = Vec::with_capacity(batch_size * NUM_CLASSES);

        for _ in 0..batch_size {
            let class = rng.gen_range(0..NUM_CLASSES);
            let (lo, hi) = if class == 0 { (5, 30) } else { (50, 75) };

            for _ in 0..SEQ_LEN {
                // 70% class-specific words, 30% random noise
                let word = if rng.gen::<f32>() < 0.7 {
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

fn main() {
    println!("Text Classification with Transformer Encoder");
    println!(
        "Model: Embedding({}) → SinPE → TransformerEncoder({}L, d={}, {}H, ff={}) → MeanPool → Linear({})",
        VOCAB_SIZE, N_LAYERS, D_MODEL, N_HEADS, D_FF, NUM_CLASSES
    );
    println!("Optimizer: Adam with linear warmup (50 steps)");
    println!();

    println!("Generating synthetic data...");
    let train_data = generate_data(1600, BATCH_SIZE);
    let val_data = generate_data(320, BATCH_SIZE);

    println!(
        "Training: {} batches x {} = {} samples",
        train_data.len(),
        BATCH_SIZE,
        train_data.len() * BATCH_SIZE
    );
    println!(
        "Validation: {} batches x {} = {} samples",
        val_data.len(),
        BATCH_SIZE,
        val_data.len() * BATCH_SIZE
    );
    println!();

    let mut model = TextClassifier::new(train_data, val_data);
    let mut trainer = Trainer::new(TrainerConfig { epochs: 5 });

    let history = trainer.fit(&mut model);

    println!("\nTraining Summary:");
    for e in &history.epochs {
        println!(
            "  Epoch {}: train_loss={:.4}, val_loss={:.4}, time={:.2}s",
            e.epoch + 1,
            e.train_loss,
            e.val_loss.unwrap_or(f32::NAN),
            e.duration_secs
        );
    }

    // Verify loss decreased
    if history.epochs.len() >= 2 {
        let first = history.epochs.first().unwrap().train_loss;
        let last = history.epochs.last().unwrap().train_loss;
        if last < first {
            println!("\nTraining loss decreased: {:.4} -> {:.4}", first, last);
        }
    }
}
