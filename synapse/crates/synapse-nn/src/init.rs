//! Weight initialization strategies (Xavier/Glorot, Kaiming/He).

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use synapse_autograd::Tensor;

/// Calculate fan_in and fan_out from a weight tensor shape.
/// For 2D [out, in]: fan_in=in, fan_out=out
/// For 4D [out, in, kH, kW]: fan_in=in*kH*kW, fan_out=out*kH*kW
pub fn calculate_fans(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        1 => (shape[0], shape[0]),
        2 => (shape[1], shape[0]),
        n if n >= 3 => {
            let receptive_field: usize = shape[2..].iter().product();
            let fan_in = shape[1] * receptive_field;
            let fan_out = shape[0] * receptive_field;
            (fan_in, fan_out)
        }
        _ => panic!("cannot compute fans for shape {:?}", shape),
    }
}

/// Xavier (Glorot) uniform initialization: U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
pub fn xavier_uniform(shape: &[usize]) -> Tensor {
    let (fan_in, fan_out) = calculate_fans(shape);
    let a = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let dist = Uniform::new(-a, a);
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

/// Xavier (Glorot) normal initialization: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
pub fn xavier_normal(shape: &[usize]) -> Tensor {
    let (fan_in, fan_out) = calculate_fans(shape);
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

/// Kaiming (He) uniform initialization: U(-a, a) where a = sqrt(6 / fan_in)
/// Designed for ReLU networks.
pub fn kaiming_uniform(shape: &[usize]) -> Tensor {
    let (fan_in, _) = calculate_fans(shape);
    let a = (6.0 / fan_in as f64).sqrt();
    let dist = Uniform::new(-a, a);
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

/// Kaiming (He) normal initialization: N(0, std) where std = sqrt(2 / fan_in)
/// Designed for ReLU networks.
pub fn kaiming_normal(shape: &[usize]) -> Tensor {
    let (fan_in, _) = calculate_fans(shape);
    let std = (2.0 / fan_in as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

/// Fill a tensor with random values from N(0, std).
pub fn randn(shape: &[usize], std: f32) -> Tensor {
    let dist = Normal::new(0.0, std as f64).unwrap();
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

/// Fill a tensor with random uniform values in [low, high).
pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Tensor {
    let mut rng = rand::thread_rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(low..high)).collect();
    Tensor::new(data, shape.to_vec())
}
