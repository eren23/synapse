//! Vision Transformer (ViT) on synthetic CIFAR-10-like data.
//!
//! Pipeline: Conv2d(3, d_model, patch_size) → reshape patches → SinusoidalPositionalEncoding
//!         → TransformerEncoder(2 layers) → MeanPool1d → Linear(d_model, num_classes)
//! Uses a manual training loop with Adam optimizer.

use synapse_autograd::{backward, Graph, Tensor};
use synapse_nn::init::xavier_uniform;
use synapse_nn::{
    Activation, Conv2d, MeanPool1d, Module, SinusoidalPositionalEncoding, TransformerEncoder,
    TransformerEncoderConfig,
};
use synapse_optim::{Adam, Optimizer, Param};

use rand::Rng;

const NUM_CLASSES: usize = 10;
const IMG_C: usize = 3;
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const PATCH_SIZE: usize = 8;
const NUM_PATCHES: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 16
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const D_FF: usize = 256;
const N_LAYERS: usize = 2;
const BATCH_SIZE: usize = 16;

struct ViTModel {
    // Patch embedding: Conv2d with kernel_size=patch_size, stride=patch_size
    patch_conv: Conv2d,
    // Positional encoding
    pos_encoding: SinusoidalPositionalEncoding,
    // Transformer encoder
    encoder: TransformerEncoder,
    // Mean pooling
    pool: MeanPool1d,
    // Classification head: [D_MODEL, NUM_CLASSES] layout for direct matmul
    fc_w: Tensor,
    fc_b: Tensor,
    optimizer: Adam,
}

impl ViTModel {
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

        // Conv2d: [N, 3, 32, 32] -> [N, d_model, 4, 4] with kernel=8, stride=8
        let mut patch_conv = Conv2d::new(
            IMG_C,
            D_MODEL,
            (PATCH_SIZE, PATCH_SIZE),
            (PATCH_SIZE, PATCH_SIZE),
            (0, 0),
            true,
        );
        patch_conv.set_training(false);

        ViTModel {
            patch_conv,
            pos_encoding: SinusoidalPositionalEncoding::new(NUM_PATCHES + 1, D_MODEL),
            encoder,
            pool: MeanPool1d::new(),
            fc_w: xavier_uniform(&[D_MODEL, NUM_CLASSES]),
            fc_b: Tensor::zeros(&[1, NUM_CLASSES]),
            optimizer: Adam::new(0.001),
        }
    }

    /// Eager forward through the frozen backbone.
    /// Input: [B, 3, 32, 32] -> Output: [B, D_MODEL]
    fn backbone(&self, input: &Tensor) -> Tensor {
        let batch = input.shape[0];

        // Patch embedding via Conv2d: [B, 3, 32, 32] -> [B, d_model, 4, 4]
        let patches = self.patch_conv.forward(input);
        let grid_h = patches.shape[2]; // 4
        let grid_w = patches.shape[3]; // 4
        let n_patches = grid_h * grid_w; // 16

        // Reshape [B, d_model, H', W'] -> [B, num_patches, d_model]
        // Need to transpose from [B, C, H, W] to [B, H*W, C]
        let mut seq_data = vec![0.0f32; batch * n_patches * D_MODEL];
        for b in 0..batch {
            for h in 0..grid_h {
                for w in 0..grid_w {
                    let patch_idx = h * grid_w + w;
                    for c in 0..D_MODEL {
                        let src = b * (D_MODEL * grid_h * grid_w)
                            + c * (grid_h * grid_w)
                            + h * grid_w
                            + w;
                        let dst = b * (n_patches * D_MODEL) + patch_idx * D_MODEL + c;
                        seq_data[dst] = patches.data[src];
                    }
                }
            }
        }
        let sequence = Tensor::new(seq_data, vec![batch, n_patches, D_MODEL]);

        // Positional encoding: [B, num_patches, d_model]
        let positioned = self.pos_encoding.forward(&sequence);

        // Transformer encoder: [B, num_patches, d_model]
        let encoded = self.encoder.forward(&positioned);

        // Mean pool: [B, num_patches, d_model] -> [B, d_model]
        self.pool.forward(&encoded)
    }

    /// Full forward for inference.
    fn predict(&self, input: &Tensor) -> Tensor {
        let features = self.backbone(input);
        features.matmul(&self.fc_w).add_broadcast(&self.fc_b)
    }

    /// Single training step: returns loss value.
    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        // Backbone forward (eager, frozen)
        let features = self.backbone(input); // [B, D_MODEL]

        // Classification head through graph
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

        // Update classifier params
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

/// Generate synthetic CIFAR-10-like data with class prototypes.
fn generate_data(n_samples: usize, batch_size: usize) -> Vec<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let n_batches = n_samples / batch_size;
    let pixels = IMG_C * IMG_H * IMG_W;

    // Each class has a random prototype image
    let prototypes: Vec<Vec<f32>> = (0..NUM_CLASSES)
        .map(|_| (0..pixels).map(|_| rng.gen_range(-1.0..1.0f32)).collect())
        .collect();

    let noise_std = 0.5f32;
    let mut batches = Vec::with_capacity(n_batches);

    for _ in 0..n_batches {
        let mut input_data = Vec::with_capacity(batch_size * pixels);
        let mut target_data = Vec::with_capacity(batch_size * NUM_CLASSES);

        for _ in 0..batch_size {
            let class = rng.gen_range(0..NUM_CLASSES);
            for &val in &prototypes[class] {
                input_data.push(val + rng.gen_range(-noise_std..noise_std));
            }
            let mut one_hot = vec![0.0f32; NUM_CLASSES];
            one_hot[class] = 1.0;
            target_data.extend_from_slice(&one_hot);
        }

        batches.push((
            Tensor::new(input_data, vec![batch_size, IMG_C, IMG_H, IMG_W]),
            Tensor::new(target_data, vec![batch_size, NUM_CLASSES]),
        ));
    }

    batches
}

fn main() {
    println!("Vision Transformer (ViT) on Synthetic CIFAR-10");
    println!(
        "Model: Conv2d(3→{}, patch={}) → SinPE → TransformerEncoder({}L, d={}, {}H, ff={}) → MeanPool → Linear({})",
        D_MODEL, PATCH_SIZE, N_LAYERS, D_MODEL, N_HEADS, D_FF, NUM_CLASSES
    );
    println!(
        "Image: {}x{}x{}, {} patches of {}x{}",
        IMG_H, IMG_W, IMG_C, NUM_PATCHES, PATCH_SIZE, PATCH_SIZE
    );
    println!();

    println!("Generating synthetic data...");
    let train_data = generate_data(480, BATCH_SIZE);
    let val_data = generate_data(160, BATCH_SIZE);

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

    let mut model = ViTModel::new();
    let epochs = 5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (input, target) in &train_data {
            total_loss += model.train_step(input, target);
        }
        let avg_loss = total_loss / train_data.len() as f32;

        // Validation accuracy
        let mut correct = 0usize;
        let mut total = 0usize;
        for (input, target) in &val_data {
            let logits = model.predict(input);
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
        println!(
            "Epoch {}/{}: loss={:.4}, val_accuracy={:.2}%",
            epoch + 1,
            epochs,
            avg_loss,
            accuracy * 100.0
        );
    }
}
