use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── im2col / col2im helpers ────────────────────────────────────────

fn im2col(
    input: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * pad - kh) / stride + 1;
    let w_out = (w + 2 * pad - kw) / stride + 1;
    let col_rows = batch * h_out * w_out;
    let col_cols = c * kh * kw;
    let mut cols = vec![0.0f32; col_rows * col_cols];
    for ni in 0..batch {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = ni * h_out * w_out + oh * w_out + ow;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * stride + ki) as isize - pad as isize;
                            let iw = (ow * stride + kj) as isize - pad as isize;
                            let col = ci * kh * kw + ki * kw + kj;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                cols[row * col_cols + col] = input[ni * c * h * w
                                    + ci as usize * h * w
                                    + ih as usize * w
                                    + iw as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    cols
}

fn col2im(
    cols: &[f32],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * pad - kh) / stride + 1;
    let w_out = (w + 2 * pad - kw) / stride + 1;
    let col_cols = c * kh * kw;
    let mut output = vec![0.0f32; batch * c * h * w];
    for ni in 0..batch {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = ni * h_out * w_out + oh * w_out + ow;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * stride + ki) as isize - pad as isize;
                            let iw = (ow * stride + kj) as isize - pad as isize;
                            let col = ci * kh * kw + ki * kw + kj;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                output[ni * c * h * w
                                    + ci as usize * h * w
                                    + ih as usize * w
                                    + iw as usize] += cols[row * col_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }
    output
}

// ── Conv2d backward ────────────────────────────────────────────────

pub struct Conv2dBackward {
    input_ids: Vec<VariableId>,
    input_data: Tensor,
    weight_data: Tensor,
    stride: usize,
    padding: usize,
}

impl GradFn for Conv2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let batch = self.input_data.shape[0];
        let c_in = self.input_data.shape[1];
        let h = self.input_data.shape[2];
        let w = self.input_data.shape[3];
        let c_out = self.weight_data.shape[0];
        let kh = self.weight_data.shape[2];
        let kw = self.weight_data.shape[3];
        let h_out = (h + 2 * self.padding - kh) / self.stride + 1;
        let w_out = (w + 2 * self.padding - kw) / self.stride + 1;

        // Recompute im2col columns
        let cols_data = im2col(
            &self.input_data.data,
            batch,
            c_in,
            h,
            w,
            kh,
            kw,
            self.stride,
            self.padding,
        );
        let cols = Tensor::new(cols_data, vec![batch * h_out * w_out, c_in * kh * kw]);

        // Rearrange grad_output [N,Cout,Hout,Wout] → [N*Hout*Wout, Cout]
        let mut dout_mat_data = vec![0.0f32; batch * h_out * w_out * c_out];
        for n in 0..batch {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let src = n * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        let dst = (n * h_out * w_out + oh * w_out + ow) * c_out + co;
                        dout_mat_data[dst] = grad_output.data[src];
                    }
                }
            }
        }
        let dout_mat = Tensor::new(dout_mat_data, vec![batch * h_out * w_out, c_out]);

        // Weight as matrix [Cout, Cin*kH*kW]
        let weight_mat = self.weight_data.reshape(&[c_out, c_in * kh * kw]);

        // grad_weight = dout^T @ cols → [Cout, Cin*kH*kW]
        let grad_weight_mat = dout_mat.transpose_2d().matmul(&cols);
        let grad_weight = grad_weight_mat.reshape(&self.weight_data.shape);

        // grad_cols = dout @ weight → [N*Hout*Wout, Cin*kH*kW]
        let grad_cols = dout_mat.matmul(&weight_mat);

        // grad_input via col2im
        let grad_input_data = col2im(
            &grad_cols.data,
            batch,
            c_in,
            h,
            w,
            kh,
            kw,
            self.stride,
            self.padding,
        );
        let grad_input = Tensor::new(grad_input_data, self.input_data.shape.clone());

        vec![Some(grad_input), Some(grad_weight)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph method ───────────────────────────────────────────────────

impl Graph {
    /// Conv2d: input [N,Cin,H,W] * weight [Cout,Cin,kH,kW] → [N,Cout,Hout,Wout]
    pub fn conv2d(
        &mut self,
        input: VariableId,
        weight: VariableId,
        stride: usize,
        padding: usize,
    ) -> VariableId {
        let input_data = self.variables[&input].data.clone();
        let weight_data = self.variables[&weight].data.clone();
        let batch = input_data.shape[0];
        let c_in = input_data.shape[1];
        let h = input_data.shape[2];
        let w = input_data.shape[3];
        let c_out = weight_data.shape[0];
        let kh = weight_data.shape[2];
        let kw = weight_data.shape[3];
        let h_out = (h + 2 * padding - kh) / stride + 1;
        let w_out = (w + 2 * padding - kw) / stride + 1;

        // im2col → [N*Hout*Wout, Cin*kH*kW]
        let cols_data = im2col(&input_data.data, batch, c_in, h, w, kh, kw, stride, padding);
        let cols = Tensor::new(cols_data, vec![batch * h_out * w_out, c_in * kh * kw]);

        // weight_mat [Cout, Cin*kH*kW]
        let weight_mat = weight_data.reshape(&[c_out, c_in * kh * kw]);

        // output_mat = cols @ weight_mat^T → [N*Hout*Wout, Cout]
        let output_mat = cols.matmul(&weight_mat.transpose_2d());

        // Rearrange to [N, Cout, Hout, Wout]
        let mut out_data = vec![0.0f32; batch * c_out * h_out * w_out];
        for n in 0..batch {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let src = (n * h_out * w_out + oh * w_out + ow) * c_out + co;
                        let dst = n * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        out_data[dst] = output_mat.data[src];
                    }
                }
            }
        }
        let output = Tensor::new(out_data, vec![batch, c_out, h_out, w_out]);

        if !self.should_track(&[input, weight]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(Conv2dBackward {
                input_ids: vec![input, weight],
                input_data,
                weight_data,
                stride,
                padding,
            }),
            &[input, weight],
            output,
        )
    }
}
