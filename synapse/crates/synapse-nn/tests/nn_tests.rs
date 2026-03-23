//! Comprehensive tests for synapse-nn.

use synapse_autograd::Tensor;
use synapse_nn::*;
use synapse_nn::module::Module;

// ═══════════════════════════════════════════════════════════════════════
// Init tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_xavier_uniform_shape() {
    let t = synapse_nn::init::xavier_uniform(&[64, 128]);
    assert_eq!(t.shape, vec![64, 128]);
    assert_eq!(t.numel(), 64 * 128);
}

#[test]
fn test_xavier_normal_shape() {
    let t = synapse_nn::init::xavier_normal(&[32, 16]);
    assert_eq!(t.shape, vec![32, 16]);
}

#[test]
fn test_kaiming_uniform_shape() {
    let t = synapse_nn::init::kaiming_uniform(&[64, 3, 3, 3]);
    assert_eq!(t.shape, vec![64, 3, 3, 3]);
    assert_eq!(t.numel(), 64 * 3 * 3 * 3);
}

#[test]
fn test_kaiming_normal_shape() {
    let t = synapse_nn::init::kaiming_normal(&[32, 16, 5, 5]);
    assert_eq!(t.shape, vec![32, 16, 5, 5]);
}

#[test]
fn test_xavier_uniform_range() {
    let shape = [256, 128];
    let t = synapse_nn::init::xavier_uniform(&shape);
    let a = (6.0f32 / (128.0 + 256.0)).sqrt();
    for &v in &t.data {
        assert!(v >= -a && v <= a, "value {} outside [-{}, {}]", v, a, a);
    }
}

#[test]
fn test_kaiming_uniform_range() {
    let shape = [64, 3, 3, 3];
    let t = synapse_nn::init::kaiming_uniform(&shape);
    let fan_in = 3 * 3 * 3;
    let a = (6.0f32 / fan_in as f32).sqrt();
    for &v in &t.data {
        assert!(v >= -a - 1e-5 && v <= a + 1e-5, "value {} outside range", v);
    }
}

#[test]
fn test_calculate_fans() {
    // 2D
    assert_eq!(synapse_nn::init::calculate_fans(&[64, 128]), (128, 64));
    // 4D
    assert_eq!(synapse_nn::init::calculate_fans(&[64, 3, 5, 5]), (3 * 25, 64 * 25));
}

// ═══════════════════════════════════════════════════════════════════════
// Linear tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_linear_output_shape() {
    let layer = Linear::new(128, 64, true);
    let input = Tensor::ones(&[8, 128]);
    let output = layer.forward(&input);
    assert_eq!(output.shape, vec![8, 64]);
}

#[test]
fn test_linear_no_bias_output_shape() {
    let layer = Linear::new(32, 16, false);
    let input = Tensor::ones(&[4, 32]);
    let output = layer.forward(&input);
    assert_eq!(output.shape, vec![4, 16]);
}

#[test]
fn test_linear_parameter_count() {
    // With bias: in*out + out = 128*64 + 64 = 8256
    let layer = Linear::new(128, 64, true);
    let params = layer.parameters();
    assert_eq!(params.len(), 2);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 128 * 64 + 64);
}

#[test]
fn test_linear_no_bias_parameter_count() {
    let layer = Linear::new(128, 64, false);
    let params = layer.parameters();
    assert_eq!(params.len(), 1);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 128 * 64);
}

// ═══════════════════════════════════════════════════════════════════════
// Activation tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_relu_output_shape() {
    let relu = ReLU::new();
    let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]);
    let output = relu.forward(&input);
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_output_shape() {
    let sig = Sigmoid::new();
    let input = Tensor::zeros(&[4, 8]);
    let output = sig.forward(&input);
    assert_eq!(output.shape, vec![4, 8]);
    // sigmoid(0) = 0.5
    for &v in &output.data {
        assert!((v - 0.5).abs() < 1e-6);
    }
}

#[test]
fn test_tanh_output_shape() {
    let tanh = Tanh::new();
    let input = Tensor::zeros(&[3, 5]);
    let output = tanh.forward(&input);
    assert_eq!(output.shape, vec![3, 5]);
    for &v in &output.data {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_gelu_output_shape() {
    let gelu = GELU::new();
    let input = Tensor::ones(&[2, 4]);
    let output = gelu.forward(&input);
    assert_eq!(output.shape, vec![2, 4]);
    // GELU(1.0) ≈ 0.8412
    for &v in &output.data {
        assert!((v - 0.8412).abs() < 0.01);
    }
}

#[test]
fn test_softmax_output_shape_and_sum() {
    let sm = Softmax::new(1);
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let output = sm.forward(&input);
    assert_eq!(output.shape, vec![2, 3]);
    // Each row should sum to 1
    for row in 0..2 {
        let sum: f32 = (0..3).map(|col| output.data[row * 3 + col]).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum = {}", sum);
    }
}

#[test]
fn test_activations_no_parameters() {
    assert!(ReLU::new().parameters().is_empty());
    assert!(Sigmoid::new().parameters().is_empty());
    assert!(Tanh::new().parameters().is_empty());
    assert!(GELU::new().parameters().is_empty());
    assert!(Softmax::new(0).parameters().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Conv2d tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_conv2d_output_shape() {
    // Input: [1, 3, 32, 32], kernel 3x3, stride 1, pad 1
    // Output: [1, 16, 32, 32]
    let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (1, 1), true);
    let input = Tensor::ones(&[1, 3, 32, 32]);
    let output = conv.forward(&input);
    assert_eq!(output.shape, vec![1, 16, 32, 32]);
}

#[test]
fn test_conv2d_output_shape_no_padding() {
    // Input: [2, 3, 8, 8], kernel 3x3, stride 1, pad 0
    // H_out = (8 - 3)/1 + 1 = 6
    let conv = Conv2d::new(3, 8, (3, 3), (1, 1), (0, 0), true);
    let input = Tensor::ones(&[2, 3, 8, 8]);
    let output = conv.forward(&input);
    assert_eq!(output.shape, vec![2, 8, 6, 6]);
}

#[test]
fn test_conv2d_output_shape_stride2() {
    // Input: [1, 3, 32, 32], kernel 3x3, stride 2, pad 1
    // H_out = (32 + 2 - 3)/2 + 1 = 16
    let conv = Conv2d::new(3, 32, (3, 3), (2, 2), (1, 1), true);
    let input = Tensor::ones(&[1, 3, 32, 32]);
    let output = conv.forward(&input);
    assert_eq!(output.shape, vec![1, 32, 16, 16]);
}

#[test]
fn test_conv2d_parameter_count() {
    // weight: 16*3*3*3 = 432, bias: 16 -> total 448
    let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (0, 0), true);
    let params = conv.parameters();
    assert_eq!(params.len(), 2);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 16 * 3 * 3 * 3 + 16);
}

#[test]
fn test_conv2d_no_bias_parameter_count() {
    let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (0, 0), false);
    let params = conv.parameters();
    assert_eq!(params.len(), 1);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 16 * 3 * 3 * 3);
}

#[test]
fn test_conv2d_5x5_kernel() {
    // Input: [1, 1, 10, 10], kernel 5x5, stride 1, pad 2
    // H_out = (10 + 4 - 5)/1 + 1 = 10
    let conv = Conv2d::new(1, 4, (5, 5), (1, 1), (2, 2), false);
    let input = Tensor::ones(&[1, 1, 10, 10]);
    let output = conv.forward(&input);
    assert_eq!(output.shape, vec![1, 4, 10, 10]);
}

// ═══════════════════════════════════════════════════════════════════════
// BatchNorm tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_batchnorm1d_output_shape_2d() {
    let bn = BatchNorm1d::new(16);
    let input = Tensor::ones(&[8, 16]);
    let output = bn.forward(&input);
    assert_eq!(output.shape, vec![8, 16]);
}

#[test]
fn test_batchnorm1d_output_shape_3d() {
    let bn = BatchNorm1d::new(16);
    let input = Tensor::ones(&[4, 16, 10]);
    let output = bn.forward(&input);
    assert_eq!(output.shape, vec![4, 16, 10]);
}

#[test]
fn test_batchnorm2d_output_shape() {
    let bn = BatchNorm2d::new(32);
    let input = Tensor::ones(&[4, 32, 8, 8]);
    let output = bn.forward(&input);
    assert_eq!(output.shape, vec![4, 32, 8, 8]);
}

#[test]
fn test_batchnorm1d_parameter_count() {
    // gamma + beta = 16 + 16 = 32
    let bn = BatchNorm1d::new(16);
    let params = bn.parameters();
    assert_eq!(params.len(), 2);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 32);
}

#[test]
fn test_batchnorm2d_parameter_count() {
    let bn = BatchNorm2d::new(64);
    let params = bn.parameters();
    assert_eq!(params.len(), 2);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 128);
}

#[test]
fn test_batchnorm2d_normalizes() {
    // After batchnorm with default gamma=1, beta=0, output should be ~normalized
    let bn = BatchNorm2d::new(2);
    // Create input with known statistics
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0,  // channel 0, batch 0
            5.0, 6.0, 7.0, 8.0,  // channel 1, batch 0
            5.0, 6.0, 7.0, 8.0,  // channel 0, batch 1
            1.0, 2.0, 3.0, 4.0,  // channel 1, batch 1
        ],
        vec![2, 2, 2, 2],
    );
    let output = bn.forward(&input);
    assert_eq!(output.shape, vec![2, 2, 2, 2]);
    // Output should be centered (mean ≈ 0 per channel)
}

// ═══════════════════════════════════════════════════════════════════════
// Dropout tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_dropout_inference_identity() {
    let mut dropout = Dropout::new(0.5);
    dropout.set_training(false);
    let input = Tensor::ones(&[10, 10]);
    let output = dropout.forward(&input);
    assert_eq!(output.data, input.data);
}

#[test]
fn test_dropout_training_zeros_fraction() {
    let dropout = Dropout::new(0.5);
    let input = Tensor::ones(&[1000]);
    // Run 1000 trials and check zero fraction
    let mut total_zeros = 0;
    let trials = 100;
    for _ in 0..trials {
        let output = dropout.forward(&input);
        total_zeros += output.data.iter().filter(|&&x| x == 0.0).count();
    }
    let zero_fraction = total_zeros as f64 / (1000.0 * trials as f64);
    // Should be approximately p=0.5, within ±5%
    assert!(
        (zero_fraction - 0.5).abs() < 0.05,
        "dropout zero fraction = {} (expected ~0.5)",
        zero_fraction
    );
}

#[test]
fn test_dropout_p03_fraction() {
    let dropout = Dropout::new(0.3);
    let input = Tensor::ones(&[1000]);
    let mut total_zeros = 0;
    let trials = 100;
    for _ in 0..trials {
        let output = dropout.forward(&input);
        total_zeros += output.data.iter().filter(|&&x| x == 0.0).count();
    }
    let zero_fraction = total_zeros as f64 / (1000.0 * trials as f64);
    assert!(
        (zero_fraction - 0.3).abs() < 0.05,
        "dropout zero fraction = {} (expected ~0.3)",
        zero_fraction
    );
}

#[test]
fn test_dropout_scaling() {
    // Non-zero elements should be scaled by 1/(1-p)
    let dropout = Dropout::new(0.5);
    let input = Tensor::ones(&[1000]);
    let output = dropout.forward(&input);
    for &v in &output.data {
        assert!(v == 0.0 || (v - 2.0).abs() < 1e-6, "unexpected value {}", v);
    }
}

#[test]
fn test_dropout_output_shape() {
    let dropout = Dropout::new(0.5);
    let input = Tensor::ones(&[8, 128]);
    let output = dropout.forward(&input);
    assert_eq!(output.shape, vec![8, 128]);
}

#[test]
fn test_dropout_no_parameters() {
    assert!(Dropout::new(0.5).parameters().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Pooling tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_maxpool2d_output_shape() {
    // Input: [1, 1, 4, 4], kernel 2x2, stride 2 -> [1, 1, 2, 2]
    let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0));
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        vec![1, 1, 4, 4],
    );
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![1, 1, 2, 2]);
    assert_eq!(output.data, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn test_maxpool2d_batch() {
    let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0));
    let input = Tensor::ones(&[4, 3, 8, 8]);
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![4, 3, 4, 4]);
}

#[test]
fn test_avgpool2d_output_shape() {
    let pool = AvgPool2d::new((2, 2), (2, 2), (0, 0));
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        vec![1, 1, 4, 4],
    );
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![1, 1, 2, 2]);
    // avg(1,2,5,6) = 3.5, avg(3,4,7,8) = 5.5, avg(9,10,13,14) = 11.5, avg(11,12,15,16) = 13.5
    assert_eq!(output.data, vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn test_avgpool2d_stride1() {
    // Input: [1, 1, 4, 4], kernel 2x2, stride 1 -> [1, 1, 3, 3]
    let pool = AvgPool2d::new((2, 2), (1, 1), (0, 0));
    let input = Tensor::ones(&[1, 1, 4, 4]);
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![1, 1, 3, 3]);
}

#[test]
fn test_adaptive_avg_pool2d_output_shape() {
    let pool = AdaptiveAvgPool2d::new((1, 1));
    let input = Tensor::ones(&[2, 16, 7, 7]);
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![2, 16, 1, 1]);
    // Average of all 1s = 1
    for &v in &output.data {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_adaptive_avg_pool2d_to_4x4() {
    let pool = AdaptiveAvgPool2d::new((4, 4));
    let input = Tensor::ones(&[1, 3, 16, 16]);
    let output = pool.forward(&input);
    assert_eq!(output.shape, vec![1, 3, 4, 4]);
}

#[test]
fn test_pooling_no_parameters() {
    assert!(MaxPool2d::new((2, 2), (2, 2), (0, 0)).parameters().is_empty());
    assert!(AvgPool2d::new((2, 2), (2, 2), (0, 0)).parameters().is_empty());
    assert!(AdaptiveAvgPool2d::new((1, 1)).parameters().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Flatten tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_flatten_default() {
    let flatten = Flatten::default();
    let input = Tensor::ones(&[4, 3, 8, 8]);
    let output = flatten.forward(&input);
    assert_eq!(output.shape, vec![4, 3 * 8 * 8]);
}

#[test]
fn test_flatten_custom_dims() {
    let flatten = Flatten::new(2, -1);
    let input = Tensor::ones(&[4, 3, 8, 8]);
    let output = flatten.forward(&input);
    assert_eq!(output.shape, vec![4, 3, 64]);
}

#[test]
fn test_flatten_all() {
    let flatten = Flatten::new(0, -1);
    let input = Tensor::ones(&[2, 3, 4]);
    let output = flatten.forward(&input);
    assert_eq!(output.shape, vec![24]);
}

#[test]
fn test_flatten_no_parameters() {
    assert!(Flatten::default().parameters().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Embedding tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_embedding_output_shape() {
    let emb = Embedding::new(100, 32);
    // Input: 3 indices
    let input = Tensor::new(vec![0.0, 5.0, 99.0], vec![3]);
    let output = emb.forward(&input);
    assert_eq!(output.shape, vec![3, 32]);
}

#[test]
fn test_embedding_2d_input() {
    let emb = Embedding::new(50, 16);
    // Input: [2, 3] batch of indices
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let output = emb.forward(&input);
    assert_eq!(output.shape, vec![2, 3, 16]);
}

#[test]
fn test_embedding_lookup_correct() {
    let emb = Embedding::new(10, 4);
    let input = Tensor::new(vec![3.0], vec![1]);
    let output = emb.forward(&input);
    // Output should match row 3 of weight
    for i in 0..4 {
        assert_eq!(output.data[i], emb.weight.data[3 * 4 + i]);
    }
}

#[test]
fn test_embedding_parameter_count() {
    let emb = Embedding::new(100, 64);
    let params = emb.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].numel(), 100 * 64);
}

// ═══════════════════════════════════════════════════════════════════════
// RNN tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_lstm_cell_output_shape() {
    let lstm = LSTMCell::new(32, 64);
    let input = Tensor::ones(&[4, 32]);
    let h = Tensor::zeros(&[4, 64]);
    let c = Tensor::zeros(&[4, 64]);
    let (h_new, c_new) = lstm.forward_cell(&input, &h, &c);
    assert_eq!(h_new.shape, vec![4, 64]);
    assert_eq!(c_new.shape, vec![4, 64]);
}

#[test]
fn test_lstm_cell_module_forward() {
    let lstm = LSTMCell::new(16, 32);
    let input = Tensor::ones(&[8, 16]);
    let output = lstm.forward(&input);
    assert_eq!(output.shape, vec![8, 32]);
}

#[test]
fn test_lstm_cell_parameter_count() {
    let lstm = LSTMCell::new(32, 64);
    let params = lstm.parameters();
    assert_eq!(params.len(), 4);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    // weight_ih: 4*64*32 = 8192, weight_hh: 4*64*64 = 16384
    // bias_ih: 4*64 = 256, bias_hh: 4*64 = 256
    // total = 25088
    assert_eq!(total, 4 * 64 * 32 + 4 * 64 * 64 + 4 * 64 + 4 * 64);
}

#[test]
fn test_gru_cell_output_shape() {
    let gru = GRUCell::new(32, 64);
    let input = Tensor::ones(&[4, 32]);
    let h = Tensor::zeros(&[4, 64]);
    let h_new = gru.forward_cell(&input, &h);
    assert_eq!(h_new.shape, vec![4, 64]);
}

#[test]
fn test_gru_cell_module_forward() {
    let gru = GRUCell::new(16, 32);
    let input = Tensor::ones(&[8, 16]);
    let output = gru.forward(&input);
    assert_eq!(output.shape, vec![8, 32]);
}

#[test]
fn test_gru_cell_parameter_count() {
    let gru = GRUCell::new(32, 64);
    let params = gru.parameters();
    assert_eq!(params.len(), 4);
    let total: usize = params.iter().map(|p| p.numel()).sum();
    // weight_ih: 3*64*32, weight_hh: 3*64*64, bias_ih: 3*64, bias_hh: 3*64
    assert_eq!(total, 3 * 64 * 32 + 3 * 64 * 64 + 3 * 64 + 3 * 64);
}

// ═══════════════════════════════════════════════════════════════════════
// Sequential tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_sequential_chains_correctly() {
    let seq = Sequential::new()
        .add(Box::new(Linear::new(128, 64, true)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(Linear::new(64, 10, true)));

    let input = Tensor::ones(&[4, 128]);
    let output = seq.forward(&input);
    assert_eq!(output.shape, vec![4, 10]);
}

#[test]
fn test_sequential_parameter_count() {
    let seq = Sequential::new()
        .add(Box::new(Linear::new(128, 64, true)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(Linear::new(64, 10, true)));

    let params = seq.parameters();
    let total: usize = params.iter().map(|p| p.numel()).sum();
    // Linear(128,64): 128*64+64 = 8256
    // ReLU: 0
    // Linear(64,10): 64*10+10 = 650
    assert_eq!(total, 128 * 64 + 64 + 64 * 10 + 10);
}

#[test]
fn test_sequential_training_mode_propagation() {
    let mut seq = Sequential::new()
        .add(Box::new(Linear::new(10, 5, true)))
        .add(Box::new(Dropout::new(0.5)))
        .add(Box::new(Linear::new(5, 2, true)));

    assert!(seq.is_training());

    seq.set_training(false);
    assert!(!seq.is_training());

    // Dropout should now be in eval mode -> pass through
    let input = Tensor::ones(&[4, 10]);
    let output1 = seq.forward(&input);
    let output2 = seq.forward(&input);
    // In eval mode, outputs should be deterministic
    assert_eq!(output1.data, output2.data);
}

#[test]
fn test_sequential_len() {
    let seq = Sequential::new()
        .add(Box::new(Linear::new(10, 5, true)))
        .add(Box::new(ReLU::new()));
    assert_eq!(seq.len(), 2);
}

// ═══════════════════════════════════════════════════════════════════════
// ModuleList tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_module_list() {
    let mut ml = ModuleList::new();
    ml.push(Box::new(Linear::new(10, 5, true)));
    ml.push(Box::new(ReLU::new()));
    ml.push(Box::new(Linear::new(5, 2, true)));
    assert_eq!(ml.len(), 3);

    let input = Tensor::ones(&[4, 10]);
    let output = ml.forward(&input);
    assert_eq!(output.shape, vec![4, 2]);
}

#[test]
fn test_module_list_training_propagation() {
    let mut ml = ModuleList::new();
    ml.push(Box::new(Dropout::new(0.5)));
    ml.set_training(false);
    assert!(!ml.is_training());
}

// ═══════════════════════════════════════════════════════════════════════
// Integration: 5-layer CNN
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_5_layer_cnn_output_shape() {
    // Build a 5-layer CNN for 32x32x3 input
    let seq = Sequential::new()
        // Layer 1: Conv2d(3, 16, 3, stride=1, pad=1) + ReLU + MaxPool2d(2,2)
        // 32x32 -> 32x32 -> 16x16
        .add(Box::new(Conv2d::new(3, 16, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(16)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        // Layer 2: Conv2d(16, 32, 3, stride=1, pad=1) + ReLU + MaxPool2d(2,2)
        // 16x16 -> 16x16 -> 8x8
        .add(Box::new(Conv2d::new(16, 32, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(32)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        // Layer 3: Conv2d(32, 64, 3, stride=1, pad=1) + ReLU + MaxPool2d(2,2)
        // 8x8 -> 8x8 -> 4x4
        .add(Box::new(Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(64)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        // Layer 4: Conv2d(64, 128, 3, stride=1, pad=1) + ReLU
        // 4x4 -> 4x4
        .add(Box::new(Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(128)))
        .add(Box::new(ReLU::new()))
        // Layer 5: AdaptiveAvgPool2d(1,1) + Flatten + Linear(128, 10)
        .add(Box::new(AdaptiveAvgPool2d::new((1, 1))))
        .add(Box::new(Flatten::default()))
        .add(Box::new(Linear::new(128, 10, true)));

    let input = Tensor::ones(&[2, 3, 32, 32]);
    let output = seq.forward(&input);
    assert_eq!(output.shape, vec![2, 10]);

    // Count total parameters
    let params = seq.parameters();
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert!(total > 0);
    // Expected: conv1(16*3*3*3+16) + bn1(32) + conv2(32*16*3*3+32) + bn2(64) +
    //           conv3(64*32*3*3+64) + bn3(128) + conv4(128*64*3*3+128) + bn4(256) +
    //           linear(128*10+10) = lots
    let expected = (16 * 3 * 3 * 3 + 16) + 32
        + (32 * 16 * 3 * 3 + 32) + 64
        + (64 * 32 * 3 * 3 + 64) + 128
        + (128 * 64 * 3 * 3 + 128) + 256
        + (128 * 10 + 10);
    assert_eq!(total, expected);
}

// ═══════════════════════════════════════════════════════════════════════
// Output shape formula verification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_conv2d_output_formula() {
    // Verify formula: H_out = (H + 2*pad - kernel) / stride + 1
    for &(h, k, s, p, expected) in &[
        (32, 3, 1, 1, 32),
        (32, 3, 2, 1, 16),
        (28, 5, 1, 0, 24),
        (28, 5, 1, 2, 28),
        (7, 3, 1, 0, 5),
        (14, 3, 2, 1, 7),
    ] {
        let conv = Conv2d::new(1, 1, (k, k), (s, s), (p, p), false);
        let input = Tensor::ones(&[1, 1, h, h]);
        let output = conv.forward(&input);
        assert_eq!(
            output.shape[2], expected,
            "H={} K={} S={} P={}: got {} expected {}",
            h, k, s, p, output.shape[2], expected
        );
        assert_eq!(output.shape[3], expected);
    }
}

#[test]
fn test_pool_output_formula() {
    // Same formula for pooling
    for &(h, k, s, expected) in &[
        (8, 2, 2, 4),
        (16, 2, 2, 8),
        (7, 3, 1, 5),
        (32, 4, 4, 8),
    ] {
        let pool = MaxPool2d::new((k, k), (s, s), (0, 0));
        let input = Tensor::ones(&[1, 1, h, h]);
        let output = pool.forward(&input);
        assert_eq!(
            output.shape[2], expected,
            "H={} K={} S={}: got {} expected {}",
            h, k, s, output.shape[2], expected
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Mode propagation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_mode_propagation_through_sequential() {
    let mut seq = Sequential::new()
        .add(Box::new(Linear::new(10, 10, true)))
        .add(Box::new(BatchNorm1d::new(10)))
        .add(Box::new(Dropout::new(0.5)))
        .add(Box::new(ReLU::new()));

    // Initially training
    assert!(seq.is_training());

    // Switch to eval
    seq.set_training(false);
    assert!(!seq.is_training());

    // Switch back to training
    seq.set_training(true);
    assert!(seq.is_training());
}

// ═══════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_linear_single_sample() {
    let layer = Linear::new(4, 2, true);
    let input = Tensor::ones(&[1, 4]);
    let output = layer.forward(&input);
    assert_eq!(output.shape, vec![1, 2]);
}

#[test]
fn test_conv2d_single_channel() {
    let conv = Conv2d::new(1, 1, (3, 3), (1, 1), (1, 1), true);
    let input = Tensor::ones(&[1, 1, 5, 5]);
    let output = conv.forward(&input);
    assert_eq!(output.shape, vec![1, 1, 5, 5]);
}

#[test]
fn test_empty_sequential() {
    let seq = Sequential::new();
    let input = Tensor::ones(&[2, 3]);
    let output = seq.forward(&input);
    assert_eq!(output.data, input.data);
    assert_eq!(output.shape, input.shape);
}

#[test]
fn test_module_name() {
    assert_eq!(Linear::new(1, 1, false).name(), "Linear");
    assert_eq!(Conv2d::new(1, 1, (1, 1), (1, 1), (0, 0), false).name(), "Conv2d");
    assert_eq!(BatchNorm1d::new(1).name(), "BatchNorm1d");
    assert_eq!(BatchNorm2d::new(1).name(), "BatchNorm2d");
    assert_eq!(Dropout::new(0.5).name(), "Dropout");
    assert_eq!(ReLU::new().name(), "ReLU");
    assert_eq!(Sigmoid::new().name(), "Sigmoid");
    assert_eq!(Tanh::new().name(), "Tanh");
    assert_eq!(GELU::new().name(), "GELU");
    assert_eq!(Softmax::new(0).name(), "Softmax");
    assert_eq!(MaxPool2d::new((2, 2), (2, 2), (0, 0)).name(), "MaxPool2d");
    assert_eq!(AvgPool2d::new((2, 2), (2, 2), (0, 0)).name(), "AvgPool2d");
    assert_eq!(AdaptiveAvgPool2d::new((1, 1)).name(), "AdaptiveAvgPool2d");
    assert_eq!(Flatten::default().name(), "Flatten");
    assert_eq!(Embedding::new(10, 4).name(), "Embedding");
    assert_eq!(LSTMCell::new(4, 8).name(), "LSTMCell");
    assert_eq!(GRUCell::new(4, 8).name(), "GRUCell");
    assert_eq!(Sequential::new().name(), "Sequential");
    assert_eq!(ModuleList::new().name(), "ModuleList");
}
