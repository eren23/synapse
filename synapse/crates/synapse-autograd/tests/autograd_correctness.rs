use std::time::Instant;
use synapse_autograd::{backward, grad_check, Graph, NoGradGuard, Tensor};

// ── Helper: deterministic pseudo-random data ───────────────────────

fn pseudo_rand(n: usize, offset: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i + offset) * 2654435761) as f32; // Knuth multiplicative hash
            (v % 1000.0) / 500.0 - 1.0 // range roughly [-1, 1]
        })
        .collect()
}

fn make_tensor(shape: &[usize], offset: usize) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(pseudo_rand(n, offset), shape.to_vec())
}

// ── Arithmetic ─────────────────────────────────────────────────────

#[test]
fn test_add_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        Tensor::new(vec![4.0, 5.0, 6.0], vec![3]),
    ];
    assert!(grad_check(|g, v| g.add(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_sub_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        Tensor::new(vec![4.0, 5.0, 6.0], vec![3]),
    ];
    assert!(grad_check(|g, v| g.sub(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_mul_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        Tensor::new(vec![4.0, 5.0, 6.0], vec![3]),
    ];
    assert!(grad_check(|g, v| g.mul(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_div_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        Tensor::new(vec![4.0, 5.0, 6.0], vec![3]),
    ];
    assert!(grad_check(|g, v| g.div(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_neg_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, -2.0, 3.0], vec![3])];
    assert!(grad_check(|g, v| g.neg(v[0]), &inputs, 1e-3, 1e-2));
}

// ── Broadcasting gradient reduction ────────────────────────────────

#[test]
fn test_add_broadcast_grad_check() {
    // [3,4] + [4] → broadcasting
    let inputs = vec![
        Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4],
        ),
        Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![4]),
    ];
    assert!(grad_check(|g, v| g.add(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_mul_broadcast_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        Tensor::new(vec![0.5, 1.0, 1.5], vec![3]),
    ];
    assert!(grad_check(|g, v| g.mul(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_div_broadcast_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        Tensor::new(vec![2.0, 3.0, 4.0], vec![3]),
    ];
    assert!(grad_check(|g, v| g.div(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_broadcasting_gradient_reduction_correct() {
    // Verify gradient shapes are reduced correctly
    let mut g = Graph::new();
    let a = g.variable(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        true,
    );
    let b = g.variable(Tensor::new(vec![0.5, 1.0, 1.5], vec![3]), true);
    let c = g.add(a, b); // [2,3] + [3] → [2,3]
    backward(&mut g, c);
    assert_eq!(g.grad(a).unwrap().shape, vec![2, 3]);
    assert_eq!(g.grad(b).unwrap().shape, vec![3]);
    // grad_b should be sum of grad along axis 0: [2.0, 2.0, 2.0] (ones summed over batch)
    assert_eq!(g.grad(b).unwrap().data, vec![2.0, 2.0, 2.0]);
}

// ── MatMul ─────────────────────────────────────────────────────────

#[test]
fn test_matmul_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
    ];
    assert!(grad_check(|g, v| g.matmul(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_matmul_gradient_shapes_non_square() {
    let mut g = Graph::new();
    // A: [3,5], B: [5,2] → C: [3,2]
    let a = g.variable(make_tensor(&[3, 5], 0), true);
    let b = g.variable(make_tensor(&[5, 2], 100), true);
    let c = g.matmul(a, b);
    backward(&mut g, c);
    assert_eq!(g.grad(a).unwrap().shape, vec![3, 5]);
    assert_eq!(g.grad(b).unwrap().shape, vec![5, 2]);
}

#[test]
fn test_matmul_non_square_grad_check() {
    let inputs = vec![make_tensor(&[4, 3], 0), make_tensor(&[3, 7], 50)];
    assert!(grad_check(|g, v| g.matmul(v[0], v[1]), &inputs, 1e-3, 1e-2));
}

// ── Reduce ─────────────────────────────────────────────────────────

#[test]
fn test_sum_all_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    assert!(grad_check(|g, v| g.sum_all(v[0]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_sum_axis_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(
        |g, v| g.sum_axis(v[0], 0, false),
        &inputs,
        1e-3,
        1e-2
    ));
    assert!(grad_check(
        |g, v| g.sum_axis(v[0], 1, false),
        &inputs,
        1e-3,
        1e-2
    ));
    assert!(grad_check(
        |g, v| g.sum_axis(v[0], 0, true),
        &inputs,
        1e-3,
        1e-2
    ));
}

#[test]
fn test_mean_all_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    assert!(grad_check(|g, v| g.mean_all(v[0]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_mean_axis_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(
        |g, v| g.mean_axis(v[0], 0, false),
        &inputs,
        1e-3,
        1e-2
    ));
    assert!(grad_check(
        |g, v| g.mean_axis(v[0], 1, false),
        &inputs,
        1e-3,
        1e-2
    ));
}

// ── Activations ────────────────────────────────────────────────────

#[test]
fn test_relu_grad_check() {
    // Avoid values near 0 where relu is non-differentiable
    let inputs = vec![Tensor::new(vec![-2.0, 1.5, -0.5, 3.0], vec![4])];
    assert!(grad_check(|g, v| g.relu(v[0]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_sigmoid_grad_check() {
    let inputs = vec![Tensor::new(vec![-1.0, 0.5, 1.0, 2.0], vec![4])];
    assert!(grad_check(|g, v| g.sigmoid(v[0]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_tanh_grad_check() {
    let inputs = vec![Tensor::new(vec![-1.0, 0.5, 1.0, 2.0], vec![4])];
    assert!(grad_check(|g, v| g.tanh_op(v[0]), &inputs, 1e-3, 1e-2));
}

#[test]
fn test_gelu_grad_check() {
    let inputs = vec![Tensor::new(vec![-1.0, 0.5, 1.0, 2.0], vec![4])];
    assert!(grad_check(|g, v| g.gelu(v[0]), &inputs, 1e-3, 1e-2));
}

// ── Reshape / Transpose ────────────────────────────────────────────

#[test]
fn test_reshape_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(
        |g, v| g.reshape(v[0], &[3, 2]),
        &inputs,
        1e-3,
        1e-2,
    ));
}

#[test]
fn test_transpose_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(
        |g, v| g.transpose(v[0], 0, 1),
        &inputs,
        1e-3,
        1e-2,
    ));
}

// ── Conv2d ─────────────────────────────────────────────────────────

#[test]
fn test_conv2d_grad_check() {
    // Small conv: input [1,1,4,4], weight [1,1,3,3], stride=1, pad=0 → [1,1,2,2]
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.1).collect();
    let weight_data: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
    let inputs = vec![
        Tensor::new(input_data, vec![1, 1, 4, 4]),
        Tensor::new(weight_data, vec![1, 1, 3, 3]),
    ];
    assert!(grad_check(
        |g, v| g.conv2d(v[0], v[1], 1, 0),
        &inputs,
        1e-3,
        5e-2,
    ));
}

// ── Pool ───────────────────────────────────────────────────────────

#[test]
fn test_max_pool2d_grad_check() {
    // Ensure unique max values in each window to avoid non-differentiable points
    let inputs = vec![Tensor::new(
        vec![
            1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
        ],
        vec![1, 1, 4, 4],
    )];
    assert!(grad_check(
        |g, v| g.max_pool2d(v[0], 2, 2),
        &inputs,
        1e-3,
        1e-2,
    ));
}

#[test]
fn test_avg_pool2d_grad_check() {
    let inputs = vec![Tensor::new(
        (1..=16).map(|x| x as f32).collect(),
        vec![1, 1, 4, 4],
    )];
    assert!(grad_check(
        |g, v| g.avg_pool2d(v[0], 2, 2),
        &inputs,
        1e-3,
        1e-2,
    ));
}

// ── BatchNorm ──────────────────────────────────────────────────────

#[test]
fn test_batch_norm_grad_check() {
    // N=4, C=3 (need N>2 for non-degenerate gradients)
    let inputs = vec![
        Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![4, 3],
        ),
        Tensor::new(vec![1.0, 1.0, 1.0], vec![3]), // gamma
        Tensor::new(vec![0.0, 0.0, 0.0], vec![3]), // beta
    ];
    assert!(grad_check(
        |g, v| g.batch_norm(v[0], v[1], v[2], 1e-5),
        &inputs,
        1e-3,
        5e-2,
    ));
}

// ── Softmax ────────────────────────────────────────────────────────

#[test]
fn test_softmax_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(|g, v| g.softmax(v[0], 1), &inputs, 1e-3, 1e-2,));
}

#[test]
fn test_log_softmax_grad_check() {
    let inputs = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])];
    assert!(grad_check(
        |g, v| g.log_softmax(v[0], 1),
        &inputs,
        1e-3,
        1e-2,
    ));
}

// ── Loss ───────────────────────────────────────────────────────────

#[test]
fn test_mse_loss_grad_check() {
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        Tensor::new(vec![1.5, 2.5, 2.0, 5.0], vec![4]),
    ];
    assert!(grad_check(
        |g, v| g.mse_loss(v[0], v[1]),
        &inputs,
        1e-3,
        1e-2,
    ));
}

#[test]
fn test_cross_entropy_loss_grad_check() {
    // pred: logits [2,3], target: one-hot [2,3]
    let inputs = vec![
        Tensor::new(vec![1.0, 2.0, 0.5, 0.5, 1.5, 1.0], vec![2, 3]),
        Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
    ];
    assert!(grad_check(
        |g, v| g.cross_entropy_loss(v[0], v[1]),
        &inputs,
        1e-3,
        5e-2,
    ));
}

// ── Benchmark: 10-layer MLP forward+backward ───────────────────────

fn build_mlp(
    g: &mut Graph,
    input: synapse_autograd::variable::VariableId,
    weights: &[synapse_autograd::variable::VariableId],
    biases: &[synapse_autograd::variable::VariableId],
) -> synapse_autograd::variable::VariableId {
    let mut x = input;
    let n_layers = weights.len();
    for i in 0..n_layers {
        x = g.matmul(x, weights[i]);
        x = g.add(x, biases[i]); // broadcasting: [batch, out] + [1, out]
        if i < n_layers - 1 {
            x = g.relu(x);
        }
    }
    g.sum_all(x)
}

#[test]
fn benchmark_mlp_forward_backward() {
    let batch = 64;
    let dims = [784, 128, 128, 128, 128, 128, 128, 128, 128, 128, 10];
    // Use smaller hidden dims to keep test fast

    let make_weight = |rows: usize, cols: usize, seed: usize| -> Tensor {
        let n = rows * cols;
        let scale = (2.0 / (rows + cols) as f32).sqrt();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let v = ((i + seed) as f64 * 0.00017 + 0.3).sin() as f32;
                v * scale
            })
            .collect();
        Tensor::new(data, vec![rows, cols])
    };

    // Forward-only (no grad)
    let forward_time = {
        let _guard = NoGradGuard::new();
        let mut g = Graph::new();
        let input = g.variable(make_weight(batch, dims[0], 999), false);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..10 {
            weights.push(g.variable(make_weight(dims[i], dims[i + 1], i * 1000), false));
            biases.push(g.variable(
                Tensor::new(vec![0.0; dims[i + 1]], vec![1, dims[i + 1]]),
                false,
            ));
        }
        let start = Instant::now();
        let _out = build_mlp(&mut g, input, &weights, &biases);
        start.elapsed()
    };

    // Forward + backward (with grad)
    let (forward_tracked_time, backward_time) = {
        let mut g = Graph::new();
        let input = g.variable(make_weight(batch, dims[0], 999), false);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..10 {
            weights.push(g.variable(make_weight(dims[i], dims[i + 1], i * 1000), true));
            biases.push(g.variable(
                Tensor::new(vec![0.0; dims[i + 1]], vec![1, dims[i + 1]]),
                true,
            ));
        }
        let start = Instant::now();
        let out = build_mlp(&mut g, input, &weights, &biases);
        let fwd = start.elapsed();
        let bwd_start = Instant::now();
        backward(&mut g, out);
        let bwd = bwd_start.elapsed();
        (fwd, bwd)
    };

    let fwd_ns = forward_time.as_nanos().max(1);
    let fwd_tracked_ns = forward_tracked_time.as_nanos().max(1);
    let overhead_pct = ((fwd_tracked_ns as f64 - fwd_ns as f64) / fwd_ns as f64) * 100.0;

    eprintln!("Forward (no grad):  {:?}", forward_time);
    eprintln!("Forward (tracked):  {:?}", forward_tracked_time);
    eprintln!("Backward:           {:?}", backward_time);
    eprintln!("Forward overhead:   {:.1}%", overhead_pct);

    // Autograd tracking overhead should be small relative to computation
    // Using generous threshold since timing can be noisy in tests
    assert!(
        overhead_pct < 25.0,
        "Forward tracking overhead too high: {:.1}%",
        overhead_pct
    );
}
