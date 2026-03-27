//! 2D Convolution layer with Kaiming initialization.

use synapse_autograd::Tensor;

use crate::init::kaiming_uniform;
use crate::module::Module;

pub struct Conv2d {
    pub weight: Tensor,       // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Option<Tensor>, // [out_channels]
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    training: bool,
}

impl Conv2d {
    /// Create a Conv2d layer.
    /// weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let weight = kaiming_uniform(&[out_channels, in_channels, kernel_size.0, kernel_size.1]);
        let bias_tensor = if bias {
            // Initialize bias uniformly in [-k, k] where k = 1/sqrt(in_channels * kH * kW)
            let fan_in = in_channels * kernel_size.0 * kernel_size.1;
            let k = 1.0 / (fan_in as f32).sqrt();
            let n = out_channels;
            let mut rng = rand::thread_rng();
            use rand::Rng;
            let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-k..k)).collect();
            Some(Tensor::new(data, vec![out_channels]))
        } else {
            None
        };
        Conv2d {
            weight,
            bias: bias_tensor,
            stride,
            padding,
            training: true,
        }
    }

    pub fn out_channels(&self) -> usize {
        self.weight.shape[0]
    }

    pub fn in_channels(&self) -> usize {
        self.weight.shape[1]
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        (self.weight.shape[2], self.weight.shape[3])
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let (kh, kw) = self.kernel_size();
        let h_out = (h + 2 * self.padding.0 - kh) / self.stride.0 + 1;
        let w_out = (w + 2 * self.padding.1 - kw) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for Conv2d {
    /// Forward: input [N, C_in, H, W] -> output [N, C_out, H_out, W_out]
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 4, "Conv2d expects 4D input [N, C, H, W]");
        let batch = input.shape[0];
        let c_in = input.shape[1];
        let h_in = input.shape[2];
        let w_in = input.shape[3];
        assert_eq!(c_in, self.in_channels());

        let c_out = self.out_channels();
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let (h_out, w_out) = self.output_size(h_in, w_in);
        let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

        // Direct convolution: for each output position, sum over kernel
        for n in 0..batch {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ic in 0..c_in {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = oh * sh + khi;
                                    let iw = ow * sw + kwi;
                                    // Check padding bounds
                                    if ih >= ph && iw >= pw {
                                        let ih = ih - ph;
                                        let iw = iw - pw;
                                        if ih < h_in && iw < w_in {
                                            let input_idx = n * (c_in * h_in * w_in)
                                                + ic * (h_in * w_in)
                                                + ih * w_in
                                                + iw;
                                            let weight_idx = oc * (c_in * kh * kw)
                                                + ic * (kh * kw)
                                                + khi * kw
                                                + kwi;
                                            sum += input.data[input_idx]
                                                * self.weight.data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(ref bias) = self.bias {
                            sum += bias.data[oc];
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

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Conv2d"
    }
}
