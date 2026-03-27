//! Pooling layers: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.

use synapse_autograd::Tensor;

use crate::module::Module;

// ── MaxPool2d ─────────────────────────────────────────────────────────

pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    training: bool,
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }

    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let w_out = (w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for MaxPool2d {
    /// Input: [N, C, H, W] -> Output: [N, C, H_out, W_out]
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 4, "MaxPool2d expects 4D input");
        let (batch, c, h, w) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (h_out, w_out) = self.output_size(h, w);

        let mut output = vec![f32::NEG_INFINITY; batch * c * h_out * w_out];

        for n in 0..batch {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * sh + khi;
                                let iw = ow * sw + kwi;
                                if ih >= ph && iw >= pw {
                                    let ih = ih - ph;
                                    let iw = iw - pw;
                                    if ih < h && iw < w {
                                        let idx = n * (c * h * w) + ci * (h * w) + ih * w + iw;
                                        max_val = max_val.max(input.data[idx]);
                                    }
                                }
                            }
                        }
                        let out_idx =
                            n * (c * h_out * w_out) + ci * (h_out * w_out) + oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, c, h_out, w_out])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "MaxPool2d"
    }
}

// ── AvgPool2d ─────────────────────────────────────────────────────────

pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    training: bool,
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }

    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let w_out = (w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for AvgPool2d {
    /// Input: [N, C, H, W] -> Output: [N, C, H_out, W_out]
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 4, "AvgPool2d expects 4D input");
        let (batch, c, h, w) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (h_out, w_out) = self.output_size(h, w);

        let mut output = vec![0.0f32; batch * c * h_out * w_out];

        for n in 0..batch {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * sh + khi;
                                let iw = ow * sw + kwi;
                                if ih >= ph && iw >= pw {
                                    let ih = ih - ph;
                                    let iw = iw - pw;
                                    if ih < h && iw < w {
                                        let idx = n * (c * h * w) + ci * (h * w) + ih * w + iw;
                                        sum += input.data[idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                        let out_idx =
                            n * (c * h_out * w_out) + ci * (h_out * w_out) + oh * w_out + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, c, h_out, w_out])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "AvgPool2d"
    }
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────

pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
    training: bool,
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d {
            output_size,
            training: true,
        }
    }
}

impl Module for AdaptiveAvgPool2d {
    /// Input: [N, C, H, W] -> Output: [N, C, out_H, out_W]
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 4, "AdaptiveAvgPool2d expects 4D input");
        let (batch, c, h_in, w_in) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (h_out, w_out) = self.output_size;

        let mut output = vec![0.0f32; batch * c * h_out * w_out];

        for n in 0..batch {
            for ci in 0..c {
                for oh in 0..h_out {
                    // Compute adaptive window: [start_h, end_h)
                    let start_h = (oh * h_in) / h_out;
                    let end_h = ((oh + 1) * h_in) / h_out;
                    for ow in 0..w_out {
                        let start_w = (ow * w_in) / w_out;
                        let end_w = ((ow + 1) * w_in) / w_out;

                        let mut sum = 0.0f32;
                        let count = (end_h - start_h) * (end_w - start_w);
                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let idx =
                                    n * (c * h_in * w_in) + ci * (h_in * w_in) + ih * w_in + iw;
                                sum += input.data[idx];
                            }
                        }

                        let out_idx =
                            n * (c * h_out * w_out) + ci * (h_out * w_out) + oh * w_out + ow;
                        output[out_idx] = sum / count as f32;
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, c, h_out, w_out])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "AdaptiveAvgPool2d"
    }
}
