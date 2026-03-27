//! End-to-end MNIST test: Train MLP for 3 epochs, verify accuracy > 90%.

use synapse_autograd::{backward, Graph, NoGradGuard, Tensor};
use synapse_nn::init::kaiming_uniform;
use synapse_optim::{Adam, Optimizer, Param};
use synapse_train::{TrainLoop, Trainer, TrainerConfig};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NUM_CLASSES: usize = 10;
const INPUT_DIM: usize = 784;
const HIDDEN1: usize = 256;
const HIDDEN2: usize = 128;
const BATCH_SIZE: usize = 64;

struct MnistMLP {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
    optimizer: Adam,
    train_data: Vec<(Tensor, Tensor)>,
    val_data: Vec<(Tensor, Tensor)>,
}

impl MnistMLP {
    fn predict(&self, input: &Tensor) -> Tensor {
        let h = input.matmul(&self.w1).add_broadcast(&self.b1).relu();
        let h = h.matmul(&self.w2).add_broadcast(&self.b2).relu();
        h.matmul(&self.w3).add_broadcast(&self.b3)
    }
}

impl TrainLoop for MnistMLP {
    fn train_batches(&self) -> Vec<(Tensor, Tensor)> {
        self.train_data.clone()
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        let mut graph = Graph::new();
        let x = graph.variable(input.clone(), false);
        let t = graph.variable(target.clone(), false);

        // Register params
        let w1_var = graph.variable(self.w1.clone(), true);
        let b1_var = graph.variable(self.b1.clone(), true);
        let w2_var = graph.variable(self.w2.clone(), true);
        let b2_var = graph.variable(self.b2.clone(), true);
        let w3_var = graph.variable(self.w3.clone(), true);
        let b3_var = graph.variable(self.b3.clone(), true);

        let h = graph.matmul(x, w1_var);
        let h = graph.add(h, b1_var);
        let h = graph.relu(h);
        let h = graph.matmul(h, w2_var);
        let h = graph.add(h, b2_var);
        let h = graph.relu(h);
        let h = graph.matmul(h, w3_var);
        let logits = graph.add(h, b3_var);

        let loss = graph.cross_entropy_loss(logits, t);
        let loss_val = graph.data(loss).data[0];

        backward(&mut graph, loss);

        let vars = [w1_var, b1_var, w2_var, b2_var, w3_var, b3_var];
        let tensors: [&Tensor; 6] = [&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3];
        let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();

        let mut params: Vec<Param> = tensors
            .iter()
            .zip(&vars)
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
        // Return negative accuracy as "loss" (lower is better for callbacks)
        Some(1.0 - accuracy)
    }
}

fn generate_prototypes(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..NUM_CLASSES)
        .map(|_| {
            (0..INPUT_DIM)
                .map(|_| rng.gen_range(-1.0..1.0f32))
                .collect()
        })
        .collect()
}

fn generate_data(
    prototypes: &[Vec<f32>],
    n_samples: usize,
    batch_size: usize,
    seed: u64,
) -> (Vec<(Tensor, Tensor)>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let noise_std = 0.2f32;
    let n_batches = n_samples / batch_size;
    let mut batches = Vec::with_capacity(n_batches);
    let mut all_targets = Vec::with_capacity(n_samples);

    for _ in 0..n_batches {
        let mut input_data = Vec::with_capacity(batch_size * INPUT_DIM);
        let mut target_data = Vec::with_capacity(batch_size * NUM_CLASSES);

        for _ in 0..batch_size {
            let class = rng.gen_range(0..NUM_CLASSES);
            all_targets.push(class);
            for &val in &prototypes[class] {
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

    (batches, all_targets)
}

#[test]
fn mnist_e2e_3_epochs_above_90_percent() {
    let prototypes = generate_prototypes(99);
    let (train_data, _) = generate_data(&prototypes, 6400, BATCH_SIZE, 42);
    let (val_data, _val_targets) = generate_data(&prototypes, 1280, BATCH_SIZE, 123);

    let mut model = MnistMLP {
        w1: kaiming_uniform(&[INPUT_DIM, HIDDEN1]),
        b1: Tensor::zeros(&[1, HIDDEN1]),
        w2: kaiming_uniform(&[HIDDEN1, HIDDEN2]),
        b2: Tensor::zeros(&[1, HIDDEN2]),
        w3: kaiming_uniform(&[HIDDEN2, NUM_CLASSES]),
        b3: Tensor::zeros(&[1, NUM_CLASSES]),
        optimizer: Adam::new(0.002),
        train_data,
        val_data,
    };

    let mut trainer = Trainer::new(TrainerConfig { epochs: 3 });
    let history = trainer.fit(&mut model);

    assert_eq!(history.epochs.len(), 3);

    // Check final accuracy via val_loss (which is 1.0 - accuracy)
    let final_val_loss = history.epochs.last().unwrap().val_loss.unwrap();
    let final_accuracy = 1.0 - final_val_loss;
    eprintln!("Final accuracy: {:.2}%", final_accuracy * 100.0);
    assert!(
        final_accuracy >= 0.90,
        "Expected accuracy >= 90%, got {:.2}%",
        final_accuracy * 100.0
    );
}
