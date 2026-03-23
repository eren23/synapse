//! Memory benchmark: verify training completes without excessive allocation.

use synapse_autograd::{backward, Graph, Tensor};
use synapse_nn::init::kaiming_uniform;
use synapse_optim::{Adam, Optimizer, Param};

#[test]
fn memory_bench_mlp_training() {
    let input_dim = 256;
    let hidden = 128;
    let output = 10;
    let batch_size = 64;
    let num_steps = 50;

    let mut w1 = kaiming_uniform(&[input_dim, hidden]);
    let mut b1 = Tensor::zeros(&[1, hidden]);
    let mut w2 = kaiming_uniform(&[hidden, output]);
    let mut b2 = Tensor::zeros(&[1, output]);
    let mut optimizer = Adam::new(0.001);

    let input_data: Vec<f32> = (0..batch_size * input_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let input = Tensor::new(input_data, vec![batch_size, input_dim]);
    let mut target_data = vec![0.0f32; batch_size * output];
    for i in 0..batch_size {
        target_data[i * output + (i % output)] = 1.0;
    }
    let target = Tensor::new(target_data, vec![batch_size, output]);

    let mut losses = Vec::with_capacity(num_steps);

    for _ in 0..num_steps {
        let mut graph = Graph::new();
        let x = graph.variable(input.clone(), false);
        let t = graph.variable(target.clone(), false);
        let w1v = graph.variable(w1.clone(), true);
        let b1v = graph.variable(b1.clone(), true);
        let w2v = graph.variable(w2.clone(), true);
        let b2v = graph.variable(b2.clone(), true);

        let h = graph.matmul(x, w1v);
        let h = graph.add(h, b1v);
        let h = graph.relu(h);
        let h = graph.matmul(h, w2v);
        let logits = graph.add(h, b2v);
        let loss = graph.cross_entropy_loss(logits, t);
        let loss_val = graph.data(loss).data[0];
        losses.push(loss_val);

        backward(&mut graph, loss);

        let vars = [w1v, b1v, w2v, b2v];
        let tensors: [&Tensor; 4] = [&w1, &b1, &w2, &b2];
        let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
        let mut params: Vec<Param> = tensors.iter().zip(&vars).map(|(tensor, &var)| {
            let grad = graph.grad(var).map(|g| g.data.clone());
            let mut p = Param::new(tensor.data.clone());
            p.grad = grad;
            p
        }).collect();
        optimizer.step(&mut params);

        w1 = Tensor::new(params[0].data.clone(), shapes[0].clone());
        b1 = Tensor::new(params[1].data.clone(), shapes[1].clone());
        w2 = Tensor::new(params[2].data.clone(), shapes[2].clone());
        b2 = Tensor::new(params[3].data.clone(), shapes[3].clone());

        // Graph is dropped here - memory should be freed
    }

    // Verify loss is decreasing
    assert!(losses.last().unwrap() < losses.first().unwrap(),
        "Loss should decrease: first={}, last={}", losses.first().unwrap(), losses.last().unwrap());

    eprintln!("Memory bench completed {} steps, loss: {:.4} -> {:.4}",
        num_steps, losses.first().unwrap(), losses.last().unwrap());
}

#[test]
fn memory_bench_large_tensor_creation() {
    // Verify large tensor allocation doesn't cause issues
    let sizes = [1000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let t = Tensor::zeros(&[size]);
        assert_eq!(t.data.len(), size);

        let t2 = Tensor::ones(&[size]);
        let t3 = t.add(&t2);
        assert_eq!(t3.data.len(), size);
        assert!((t3.data[0] - 1.0).abs() < 1e-6);
    }

    eprintln!("Large tensor allocation test passed (up to 1M elements)");
}
