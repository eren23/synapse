//! XOR example: 2-layer MLP solving the XOR problem.
//!
//! Demonstrates graph-based training with Synapse.
//! Target: loss < 0.01 within 1000 steps.

use synapse_autograd::{backward, Graph, Tensor};
use synapse_nn::init::xavier_uniform;
use synapse_optim::{Optimizer, Param, SGD};

fn main() {
    // XOR dataset: 4 samples
    let inputs = Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let hidden = 16;

    // Weights stored as [in_features, out_features] for direct matmul
    let mut w1 = xavier_uniform(&[2, hidden]);
    let mut b1 = Tensor::zeros(&[1, hidden]);
    let mut w2 = xavier_uniform(&[hidden, 1]);
    let mut b2 = Tensor::zeros(&[1, 1]);

    let mut optimizer = SGD::new(1.0);

    let mut final_loss = f32::INFINITY;

    for step in 0..1000 {
        let mut graph = Graph::new();
        let x = graph.variable(inputs.clone(), false);
        let t = graph.variable(targets.clone(), false);

        let w1_var = graph.variable(w1.clone(), true);
        let b1_var = graph.variable(b1.clone(), true);
        let w2_var = graph.variable(w2.clone(), true);
        let b2_var = graph.variable(b2.clone(), true);

        // h = relu(x @ w1 + b1)
        let h = graph.matmul(x, w1_var);
        let h = graph.add(h, b1_var);
        let h = graph.relu(h);

        // out = sigmoid(h @ w2 + b2)
        let out = graph.matmul(h, w2_var);
        let out = graph.add(out, b2_var);
        let out = graph.sigmoid(out);

        // loss = mse(out, t)
        let loss = graph.mse_loss(out, t);
        let loss_val = graph.data(loss).data[0];
        final_loss = loss_val;

        backward(&mut graph, loss);

        // Collect params and gradients for optimizer
        let var_ids = [w1_var, b1_var, w2_var, b2_var];
        let tensors: [&Tensor; 4] = [&w1, &b1, &w2, &b2];
        let mut params: Vec<Param> = tensors
            .iter()
            .zip(&var_ids)
            .map(|(tensor, &var)| {
                let grad = graph.grad(var).unwrap();
                Param::with_grad(tensor.data.clone(), grad.data.clone())
            })
            .collect();

        optimizer.step(&mut params);

        // Write updated parameters back
        w1 = Tensor::new(params[0].data.clone(), w1.shape.clone());
        b1 = Tensor::new(params[1].data.clone(), b1.shape.clone());
        w2 = Tensor::new(params[2].data.clone(), w2.shape.clone());
        b2 = Tensor::new(params[3].data.clone(), b2.shape.clone());

        if step % 100 == 0 {
            println!("Step {:4}: loss = {:.6}", step, loss_val);
        }
    }

    println!("\nFinal loss: {:.6}", final_loss);

    // Verify predictions
    let mut graph = Graph::new();
    let x = graph.variable(inputs.clone(), false);
    let w1_var = graph.variable(w1, false);
    let b1_var = graph.variable(b1, false);
    let w2_var = graph.variable(w2, false);
    let b2_var = graph.variable(b2, false);

    let h = graph.matmul(x, w1_var);
    let h = graph.add(h, b1_var);
    let h = graph.relu(h);
    let out = graph.matmul(h, w2_var);
    let out = graph.add(out, b2_var);
    let out = graph.sigmoid(out);

    let predictions = graph.data(out);
    println!("\nPredictions:");
    println!("  0 XOR 0 = {:.4} (expected 0)", predictions.data[0]);
    println!("  0 XOR 1 = {:.4} (expected 1)", predictions.data[1]);
    println!("  1 XOR 0 = {:.4} (expected 1)", predictions.data[2]);
    println!("  1 XOR 1 = {:.4} (expected 0)", predictions.data[3]);

    assert!(
        final_loss < 0.01,
        "XOR did not converge: loss = {}",
        final_loss
    );
}
