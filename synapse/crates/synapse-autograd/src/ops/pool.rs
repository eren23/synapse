use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── MaxPool2d ──────────────────────────────────────────────────────

pub struct MaxPool2dBackward {
    input_ids: Vec<VariableId>,
    max_indices: Vec<usize>, // flat indices into input for each output element
    input_shape: Vec<usize>,
}

impl GradFn for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; n];
        for (i, &idx) in self.max_indices.iter().enumerate() {
            grad_input[idx] += grad_output.data[i];
        }
        vec![Some(Tensor::new(grad_input, self.input_shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── AvgPool2d ──────────────────────────────────────────────────────

pub struct AvgPool2dBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
    kernel_size: usize,
    stride: usize,
}

impl GradFn for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let batch = self.input_shape[0];
        let channels = self.input_shape[1];
        let h = self.input_shape[2];
        let w = self.input_shape[3];
        let ks = self.kernel_size;
        let h_out = (h - ks) / self.stride + 1;
        let w_out = (w - ks) / self.stride + 1;
        let window = (ks * ks) as f32;
        let mut grad_input = vec![0.0f32; self.input_shape.iter().product()];
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_output.data[n * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow] / window;
                        for ki in 0..ks {
                            for kj in 0..ks {
                                let ih = oh * self.stride + ki;
                                let iw = ow * self.stride + kj;
                                grad_input[n * channels * h * w + c * h * w + ih * w + iw] += g;
                            }
                        }
                    }
                }
            }
        }
        vec![Some(Tensor::new(grad_input, self.input_shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn max_pool2d(&mut self, input: VariableId, kernel_size: usize, stride: usize) -> VariableId {
        let input_data = &self.variables[&input].data;
        let batch = input_data.shape[0];
        let channels = input_data.shape[1];
        let h = input_data.shape[2];
        let w = input_data.shape[3];
        let h_out = (h - kernel_size) / stride + 1;
        let w_out = (w - kernel_size) / stride + 1;
        let out_numel = batch * channels * h_out * w_out;
        let mut out_data = vec![0.0f32; out_numel];
        let mut max_indices = vec![0usize; out_numel];
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let out_idx = n * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;
                        for ki in 0..kernel_size {
                            for kj in 0..kernel_size {
                                let ih = oh * stride + ki;
                                let iw = ow * stride + kj;
                                let in_idx = n * channels * h * w + c * h * w + ih * w + iw;
                                if input_data.data[in_idx] > max_val {
                                    max_val = input_data.data[in_idx];
                                    max_idx = in_idx;
                                }
                            }
                        }
                        out_data[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }
        }
        let input_shape = input_data.shape.clone();
        let output = Tensor::new(out_data, vec![batch, channels, h_out, w_out]);
        if !self.should_track(&[input]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(MaxPool2dBackward { input_ids: vec![input], max_indices, input_shape }),
            &[input],
            output,
        )
    }

    pub fn avg_pool2d(&mut self, input: VariableId, kernel_size: usize, stride: usize) -> VariableId {
        let input_data = &self.variables[&input].data;
        let batch = input_data.shape[0];
        let channels = input_data.shape[1];
        let h = input_data.shape[2];
        let w = input_data.shape[3];
        let h_out = (h - kernel_size) / stride + 1;
        let w_out = (w - kernel_size) / stride + 1;
        let window = (kernel_size * kernel_size) as f32;
        let mut out_data = vec![0.0f32; batch * channels * h_out * w_out];
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ki in 0..kernel_size {
                            for kj in 0..kernel_size {
                                let ih = oh * stride + ki;
                                let iw = ow * stride + kj;
                                sum += input_data.data[n * channels * h * w + c * h * w + ih * w + iw];
                            }
                        }
                        out_data[n * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow] = sum / window;
                    }
                }
            }
        }
        let input_shape = input_data.shape.clone();
        let output = Tensor::new(out_data, vec![batch, channels, h_out, w_out]);
        if !self.should_track(&[input]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(AvgPool2dBackward { input_ids: vec![input], input_shape, kernel_size, stride }),
            &[input],
            output,
        )
    }
}
