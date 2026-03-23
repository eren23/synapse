//! Benchmark: Forward pass of a 5-layer CNN on 32x32x3 input.

use criterion::{criterion_group, criterion_main, Criterion};
use synapse_autograd::Tensor;
use synapse_nn::module::Module;
use synapse_nn::*;

fn build_5_layer_cnn() -> Sequential {
    Sequential::new()
        .add(Box::new(Conv2d::new(3, 16, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(16)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        .add(Box::new(Conv2d::new(16, 32, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(32)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        .add(Box::new(Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(64)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))))
        .add(Box::new(Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), true)))
        .add(Box::new(BatchNorm2d::new(128)))
        .add(Box::new(ReLU::new()))
        .add(Box::new(AdaptiveAvgPool2d::new((1, 1))))
        .add(Box::new(Flatten::default()))
        .add(Box::new(Linear::new(128, 10, true)))
}

fn bench_cnn_forward(c: &mut Criterion) {
    let cnn = build_5_layer_cnn();
    let input = Tensor::ones(&[1, 3, 32, 32]);

    c.bench_function("5-layer CNN forward 32x32x3", |b| {
        b.iter(|| cnn.forward(&input))
    });
}

criterion_group!(benches, bench_cnn_forward);
criterion_main!(benches);
