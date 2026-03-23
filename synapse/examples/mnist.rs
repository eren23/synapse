//! MNIST example: MLP trained on synthetic MNIST-like data.
//!
//! Demonstrates a 3-layer MLP (784 -> 256 -> 128 -> 10) with cross-entropy loss.
//! Uses the Trainer API with EarlyStopping callback.

use synapse_autograd::{backward, Graph, NoGradGuard, Tensor};
use synapse_nn::init::kaiming_uniform;
use synapse_optim::{Adam, Optimizer, Param};
use synapse_train::{
    EarlyStopping, Trainer, TrainerConfig, TrainLoop,
};

use rand::Rng;

const NUM_CLASSES: usize = 10;
const INPUT_DIM: usize = 784;
const HIDDEN1: usize = 256;
const HIDDEN2: usize = 128;
const BATCH_SIZE: usize = 64;

struct MnistModel {
    // Weights: [in, out] layout for direct matmul
    w1: Tensor, b1: Tensor,
    w2: Tensor, b2: Tensor,
    w3: Tensor, b3: Tensor,
    optimizer: Adam,
    train_data: Vec<(Tensor, Tensor)>,
    val_data: Vec<(Tensor, Tensor)>,
}

impl MnistModel {
    fn new(train_data: Vec<(Tensor, Tensor)>, val_data: Vec<(Tensor, Tensor)>) -> Self {
        MnistModel {
            w1: kaiming_uniform(&[INPUT_DIM, HIDDEN1]),
            b1: Tensor::zeros(&[1, HIDDEN1]),
            w2: kaiming_uniform(&[HIDDEN1, HIDDEN2]),
            b2: Tensor::zeros(&[1, HIDDEN2]),
            w3: kaiming_uniform(&[HIDDEN2, NUM_CLASSES]),
            b3: Tensor::zeros(&[1, NUM_CLASSES]),
            optimizer: Adam::new(0.001),
            train_data,
            val_data,
        }
    }

    fn forward_graph(&self, graph: &mut Graph, x: usize, params: &[usize]) -> usize {
        let h = graph.matmul(x, params[0]);
        let h = graph.add(h, params[1]);
        let h = graph.relu(h);
        let h = graph.matmul(h, params[2]);
        let h = graph.add(h, params[3]);
        let h = graph.relu(h);
        let h = graph.matmul(h, params[4]);
        graph.add(h, params[5])
    }

    fn predict(&self, input: &Tensor) -> Tensor {
        // Eager forward (no graph)
        let h = input.matmul(&self.w1).add_broadcast(&self.b1).relu();
        let h = h.matmul(&self.w2).add_broadcast(&self.b2).relu();
        h.matmul(&self.w3).add_broadcast(&self.b3)
    }
}

impl TrainLoop for MnistModel {
    fn train_batches(&self) -> Vec<(Tensor, Tensor)> {
        self.train_data.clone()
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        let mut graph = Graph::new();
        let x = graph.variable(input.clone(), false);
        let t = graph.variable(target.clone(), false);

        let param_vars: Vec<usize> = [&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3]
            .iter()
            .map(|p| graph.variable((*p).clone(), true))
            .collect();

        let logits = self.forward_graph(&mut graph, x, &param_vars);
        let loss = graph.cross_entropy_loss(logits, t);
        let loss_val = graph.data(loss).data[0];

        backward(&mut graph, loss);

        let shapes: Vec<Vec<usize>> = [&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3]
            .iter()
            .map(|p| p.shape.clone())
            .collect();

        let mut params: Vec<Param> = [&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3]
            .iter()
            .zip(&param_vars)
            .map(|(tensor, &var)| {
                let grad = graph.grad(var).map(|g| g.data.clone());
                let mut p = Param::new(tensor.data.clone());
                p.grad = grad;
                p
            })
            .collect();

        self.optimizer.step(&mut params);

        self.w1 = Tensor::new(params[0].data.clone(), shapes[0].clone());
        self.b1 = Tensor::new(params[1].data.clone(), shapes[1].clone());
        self.w2 = Tensor::new(params[2].data.clone(), shapes[2].clone());
        self.b2 = Tensor::new(params[3].data.clone(), shapes[3].clone());
        self.w3 = Tensor::new(params[4].data.clone(), shapes[4].clone());
        self.b3 = Tensor::new(params[5].data.clone(), shapes[5].clone());

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

            // Compute cross-entropy loss eagerly
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

fn generate_prototypes() -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..NUM_CLASSES)
        .map(|_| (0..INPUT_DIM).map(|_| rng.gen_range(-1.0..1.0f32)).collect())
        .collect()
}

/// Generate synthetic data from shared prototypes.
fn generate_data(
    prototypes: &[Vec<f32>],
    n_samples: usize,
    batch_size: usize,
) -> Vec<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();

    let noise_std = 0.3f32;
    let n_batches = n_samples / batch_size;
    let mut batches = Vec::with_capacity(n_batches);

    for _ in 0..n_batches {
        let mut input_data = Vec::with_capacity(batch_size * INPUT_DIM);
        let mut target_data = Vec::with_capacity(batch_size * NUM_CLASSES);

        for _ in 0..batch_size {
            let class = rng.gen_range(0..NUM_CLASSES);
            let proto = &prototypes[class];
            for &val in proto {
                input_data.push(val + rng.gen_range(-noise_std..noise_std));
            }
            let mut one_hot = vec![0.0f32; NUM_CLASSES];
            one_hot[class] = 1.0;
            target_data.extend_from_slice(&one_hot);
        }

        batches.push((
            Tensor::new(input_data, vec![batch_size, INPUT_DIM]),
            Tensor::new(target_data, vec![batch_size, NUM_CLASSES]),
        ));
    }

    batches
}

fn main() {
    println!("Generating synthetic MNIST-like data...");
    let prototypes = generate_prototypes();
    let train_data = generate_data(&prototypes, 6400, BATCH_SIZE);
    let val_data = generate_data(&prototypes, 1280, BATCH_SIZE);

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

    let mut model = MnistModel::new(train_data, val_data);
    let mut trainer = Trainer::new(TrainerConfig { epochs: 5 });
    trainer.add_callback(Box::new(EarlyStopping::new(3, 0.001)));

    let history = trainer.fit(&mut model);

    for e in &history.epochs {
        println!(
            "Epoch {}: train_loss={:.4}, val_loss={:.4}, time={:.2}s",
            e.epoch + 1,
            e.train_loss,
            e.val_loss.unwrap_or(f32::NAN),
            e.duration_secs
        );
    }
}
