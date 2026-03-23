use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

pub struct LayerNormBackward {
    input_ids: Vec<VariableId>, // [input, weight, bias]
    x_hat: Tensor,              // normalized input, same shape as input
    rstd: Tensor,               // reciprocal std [outer]
    weight_data: Tensor,        // [D]
    norm_size: usize,           // size of normalized dimension
}

impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let d = self.norm_size;
        let df = d as f32;
        let outer = grad_output.numel() / d;

        // Flatten to [outer, D]
        let go = grad_output.reshape(&[outer, d]);
        let xh = self.x_hat.reshape(&[outer, d]);

        // grad_bias = sum(grad_output, axis=0) → [D]
        let grad_bias = go.sum_axis(0, false);

        // grad_weight = sum(grad_output * x_hat, axis=0) → [D]
        let grad_weight = go.mul(&xh).sum_axis(0, false);

        // dx_hat = grad_output * weight  [outer, D]
        let w_broad = self.weight_data.reshape(&[1, d]).broadcast_to(&[outer, d]);
        let dx_hat = go.mul(&w_broad);

        // grad_input = rstd/D * (D*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
        let sum_dx = dx_hat.sum_axis(1, true).broadcast_to(&[outer, d]);
        let sum_dx_xh = dx_hat.mul(&xh).sum_axis(1, true).broadcast_to(&[outer, d]);

        let term = dx_hat.scale(df).sub(&sum_dx).sub(&xh.mul(&sum_dx_xh));
        let rstd_broad = self.rstd.reshape(&[outer, 1]).broadcast_to(&[outer, d]);
        let grad_input = rstd_broad.scale(1.0 / df).mul(&term);

        vec![
            Some(grad_input.reshape(&grad_output.shape)),
            Some(grad_weight),
            Some(grad_bias),
        ]
    }

    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

impl Graph {
    /// Layer normalization over the last dimension.
    /// input: [*, D], weight: [D], bias: [D]
    pub fn layer_norm(
        &mut self,
        input: VariableId,
        weight: VariableId,
        bias: VariableId,
        eps: f32,
    ) -> VariableId {
        let input_data = self.variables[&input].data.clone();
        let weight_data = self.variables[&weight].data.clone();
        let bias_data = self.variables[&bias].data.clone();

        let ndim = input_data.shape.len();
        let d = input_data.shape[ndim - 1];
        let outer = input_data.numel() / d;

        // Flatten to [outer, D]
        let flat = input_data.reshape(&[outer, d]);

        // mean [outer, 1], variance [outer, 1]
        let mean = flat.mean_axis(1, true);
        let diff = flat.sub(&mean.broadcast_to(&[outer, d]));
        let var = diff.mul(&diff).mean_axis(1, true);

        // rstd [outer]
        let rstd_data: Vec<f32> = var.data.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect();
        let rstd = Tensor::new(rstd_data, vec![outer]);

        // x_hat = diff * rstd  [outer, D]
        let rstd_broad = rstd.reshape(&[outer, 1]).broadcast_to(&[outer, d]);
        let x_hat = diff.mul(&rstd_broad);

        // output = x_hat * weight + bias  [outer, D]
        let w_broad = weight_data.reshape(&[1, d]).broadcast_to(&[outer, d]);
        let b_broad = bias_data.reshape(&[1, d]).broadcast_to(&[outer, d]);
        let output = x_hat.mul(&w_broad).add(&b_broad).reshape(&input_data.shape);

        let x_hat_orig = x_hat.reshape(&input_data.shape);

        if !self.should_track(&[input, weight, bias]) {
            return self.untracked(output);
        }

        self.record_op(
            Box::new(LayerNormBackward {
                input_ids: vec![input, weight, bias],
                x_hat: x_hat_orig,
                rstd,
                weight_data,
                norm_size: d,
            }),
            &[input, weight, bias],
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward::backward;
    use crate::grad_check::grad_check;

    fn make_tensor(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2654435761);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    #[test]
    fn test_layernorm_grad_check_32_64() {
        let d = 64;
        let inputs = vec![
            make_tensor(&[32, d], 1),
            make_tensor(&[d], 2),
            make_tensor(&[d], 3),
        ];
        assert!(
            grad_check(
                |g, v| g.layer_norm(v[0], v[1], v[2], 1e-5),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [32, 64]"
        );
    }

    #[test]
    fn test_layernorm_grad_check_8_16_32() {
        let d = 32;
        let inputs = vec![
            make_tensor(&[8, 16, d], 10),
            make_tensor(&[d], 20),
            make_tensor(&[d], 30),
        ];
        assert!(
            grad_check(
                |g, v| g.layer_norm(v[0], v[1], v[2], 1e-5),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for [8, 16, 32]"
        );
    }

    #[test]
    fn test_layernorm_constant_input_no_inf_nan() {
        // Edge case: constant input → near-zero variance
        let d = 16;
        let input = Tensor::new(vec![3.0; 8 * d], vec![8, d]);
        let weight = Tensor::ones(&[d]);
        let bias = Tensor::zeros(&[d]);

        let mut g = Graph::new();
        let vi = g.variable(input, true);
        let vw = g.variable(weight, true);
        let vb = g.variable(bias, true);
        let out = g.layer_norm(vi, vw, vb, 1e-5);

        backward(&mut g, out);

        for &v in [vi, vw, vb].iter() {
            let grad = g.grad(v).expect("gradient should exist");
            for &val in &grad.data {
                assert!(val.is_finite(), "gradient contains inf or nan");
            }
        }
    }

    #[test]
    fn test_layernorm_output_shape() {
        let mut g = Graph::new();
        let input = g.variable(make_tensor(&[4, 8, 16], 1), true);
        let weight = g.variable(make_tensor(&[16], 2), true);
        let bias = g.variable(make_tensor(&[16], 3), true);
        let out = g.layer_norm(input, weight, bias, 1e-5);

        assert_eq!(g.data(out).shape, vec![4, 8, 16]);

        backward(&mut g, out);

        assert_eq!(g.grad(input).unwrap().shape, vec![4, 8, 16]);
        assert_eq!(g.grad(weight).unwrap().shape, vec![16]);
        assert_eq!(g.grad(bias).unwrap().shape, vec![16]);
    }
}
