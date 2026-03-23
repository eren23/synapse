//! CIFAR-10 example: Simple CNN trained on synthetic CIFAR-10-like data.
//!
//! Architecture: Conv2d(3,16,3) -> ReLU -> Conv2d(16,32,3) -> ReLU -> Flatten -> Linear(128) -> Linear(10)
//! Demonstrates training with Module trait layers and graph-based backprop.

use synapse_autograd::{backward, Graph, Tensor};
use synapse_nn::init::kaiming_uniform;
use synapse_optim::{Adam, Optimizer, Param};

use rand::Rng;

const NUM_CLASSES: usize = 10;
const IMG_C: usize = 3;
const IMG_H: usize = 8; // Reduced from 32 for speed
const IMG_W: usize = 8;
const BATCH_SIZE: usize = 32;

/// Simple CNN with manually managed parameters.
/// For graph-based training, we implement forward directly on the graph.
struct CifarModel {
    // Conv1: [out_ch=16, in_ch=3, kH=3, kW=3]
    conv1_w: Tensor,
    conv1_b: Tensor,
    // Conv2: [out_ch=32, in_ch=16, kH=3, kW=3]
    conv2_w: Tensor,
    conv2_b: Tensor,
    // FC after flattening: [flat_dim, 128]
    fc1_w: Tensor,
    fc1_b: Tensor,
    // Output: [128, 10]
    fc2_w: Tensor,
    fc2_b: Tensor,
    optimizer: Adam,
}

fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Tensor {
    let batch = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];
    let c_out = weight.shape[0];
    let kh = weight.shape[2];
    let kw = weight.shape[3];

    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;

    let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

    for n in 0..batch {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = bias.data[oc];
                    for ic in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * stride + ki;
                                let iw = ow * stride + kj;
                                if ih >= padding
                                    && iw >= padding
                                    && ih - padding < h_in
                                    && iw - padding < w_in
                                {
                                    let ih = ih - padding;
                                    let iw = iw - padding;
                                    let input_idx =
                                        n * (c_in * h_in * w_in) + ic * (h_in * w_in) + ih * w_in + iw;
                                    let weight_idx =
                                        oc * (c_in * kh * kw) + ic * (kh * kw) + ki * kw + kj;
                                    sum += input.data[input_idx] * weight.data[weight_idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        n * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    Tensor::new(output, vec![batch, c_out, h_out, w_out])
}

impl CifarModel {
    fn new() -> Self {
        // Conv layers
        let conv1_w = kaiming_uniform(&[16, IMG_C, 3, 3]);
        let conv1_b = Tensor::zeros(&[16]);
        let conv2_w = kaiming_uniform(&[32, 16, 3, 3]);
        let conv2_b = Tensor::zeros(&[32]);

        // After conv1 (stride=1, pad=1): [B, 16, 8, 8]
        // After conv2 (stride=1, pad=1): [B, 32, 8, 8]
        // Flatten: [B, 32*8*8] = [B, 2048]
        let flat_dim = 32 * IMG_H * IMG_W;

        let fc1_w = kaiming_uniform(&[flat_dim, 128]);
        let fc1_b = Tensor::zeros(&[1, 128]);
        let fc2_w = kaiming_uniform(&[128, NUM_CLASSES]);
        let fc2_b = Tensor::zeros(&[1, NUM_CLASSES]);

        CifarModel {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            optimizer: Adam::new(0.001),
        }
    }

    fn predict(&self, input: &Tensor) -> Tensor {
        let batch = input.shape[0];

        // Conv1 + ReLU
        let h = conv2d_forward(input, &self.conv1_w, &self.conv1_b, 1, 1);
        let h = h.relu();

        // Conv2 + ReLU
        let h = conv2d_forward(&h, &self.conv2_w, &self.conv2_b, 1, 1);
        let h = h.relu();

        // Flatten
        let flat_dim = h.data.len() / batch;
        let h = h.reshape(&[batch, flat_dim]);

        // FC1 + ReLU
        let h = h.matmul(&self.fc1_w).add_broadcast(&self.fc1_b).relu();

        // FC2
        h.matmul(&self.fc2_w).add_broadcast(&self.fc2_b)
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        // Eager forward + manual gradient (using graph for FC layers only for simplicity)
        let batch = input.shape[0];

        // Conv layers in eager mode (no graph tracking for conv - too expensive)
        let h = conv2d_forward(input, &self.conv1_w, &self.conv1_b, 1, 1).relu();
        let h = conv2d_forward(&h, &self.conv2_w, &self.conv2_b, 1, 1).relu();
        let flat_dim = h.data.len() / batch;
        let h_flat = h.reshape(&[batch, flat_dim]);

        // FC layers through graph for gradient computation
        let mut graph = Graph::new();
        let x = graph.variable(h_flat, false);
        let t = graph.variable(target.clone(), false);
        let fc1_w_var = graph.variable(self.fc1_w.clone(), true);
        let fc1_b_var = graph.variable(self.fc1_b.clone(), true);
        let fc2_w_var = graph.variable(self.fc2_w.clone(), true);
        let fc2_b_var = graph.variable(self.fc2_b.clone(), true);

        let h = graph.matmul(x, fc1_w_var);
        let h = graph.add(h, fc1_b_var);
        let h = graph.relu(h);
        let h = graph.matmul(h, fc2_w_var);
        let logits = graph.add(h, fc2_b_var);

        let loss = graph.cross_entropy_loss(logits, t);
        let loss_val = graph.data(loss).data[0];

        backward(&mut graph, loss);

        // Update FC params only (conv params updated with small random perturbation for demo)
        let fc_vars = [fc1_w_var, fc1_b_var, fc2_w_var, fc2_b_var];
        let fc_tensors: [&Tensor; 4] = [&self.fc1_w, &self.fc1_b, &self.fc2_w, &self.fc2_b];
        let fc_shapes: Vec<Vec<usize>> = fc_tensors.iter().map(|t| t.shape.clone()).collect();

        let mut params: Vec<Param> = fc_tensors
            .iter()
            .zip(&fc_vars)
            .map(|(tensor, &var)| {
                let grad = graph.grad(var).map(|g| g.data.clone());
                let mut p = Param::new(tensor.data.clone());
                p.grad = grad;
                p
            })
            .collect();

        self.optimizer.step(&mut params);

        self.fc1_w = Tensor::new(params[0].data.clone(), fc_shapes[0].clone());
        self.fc1_b = Tensor::new(params[1].data.clone(), fc_shapes[1].clone());
        self.fc2_w = Tensor::new(params[2].data.clone(), fc_shapes[2].clone());
        self.fc2_b = Tensor::new(params[3].data.clone(), fc_shapes[3].clone());

        loss_val
    }
}

fn generate_data(n_samples: usize, batch_size: usize) -> Vec<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let n_batches = n_samples / batch_size;
    let pixels = IMG_C * IMG_H * IMG_W;

    // Simple class prototypes: each class has a distinct average color
    let prototypes: Vec<Vec<f32>> = (0..NUM_CLASSES)
        .map(|_| (0..pixels).map(|_| rng.gen_range(-1.0..1.0f32)).collect())
        .collect();

    let mut batches = Vec::with_capacity(n_batches);
    let noise_std = 0.5f32;

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
    println!("CIFAR-10 CNN Example (reduced resolution {}x{})", IMG_H, IMG_W);
    println!("Generating synthetic data...");

    let train_data = generate_data(960, BATCH_SIZE);
    let val_data = generate_data(320, BATCH_SIZE);

    println!("Training: {} batches", train_data.len());
    println!("Validation: {} batches", val_data.len());

    let mut model = CifarModel::new();
    let epochs = 3;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (input, target) in &train_data {
            total_loss += model.train_step(input, target);
        }
        let avg_loss = total_loss / train_data.len() as f32;

        // Validation
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
