use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

pub struct RoPEBackward {
    input_ids: Vec<VariableId>, // [input]
    cos_table: Tensor,          // [S, D/2]
    sin_table: Tensor,          // [S, D/2]
}

impl GradFn for RoPEBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // RoPE backward is the inverse rotation (negate sin component):
        // grad_input[..., 2i]   = grad[..., 2i] * cos[pos,i] + grad[..., 2i+1] * sin[pos,i]
        // grad_input[..., 2i+1] = -grad[..., 2i] * sin[pos,i] + grad[..., 2i+1] * cos[pos,i]
        let shape = &grad_output.shape;
        assert_eq!(shape.len(), 4);
        let (batch, heads, seq, d_head) = (shape[0], shape[1], shape[2], shape[3]);
        let half_d = d_head / 2;

        let mut grad_input = vec![0.0f32; grad_output.numel()];

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    let base = ((b * heads + h) * seq + s) * d_head;
                    for i in 0..half_d {
                        let cos_val = self.cos_table.data[s * half_d + i];
                        let sin_val = self.sin_table.data[s * half_d + i];
                        let g_even = grad_output.data[base + 2 * i];
                        let g_odd = grad_output.data[base + 2 * i + 1];

                        grad_input[base + 2 * i] = g_even * cos_val + g_odd * sin_val;
                        grad_input[base + 2 * i + 1] = -g_even * sin_val + g_odd * cos_val;
                    }
                }
            }
        }

        vec![Some(Tensor::new(grad_input, shape.clone()))]
    }

    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

impl Graph {
    /// Rotary position embedding.
    /// input: [B, H, S, D], cos_table: [S, D/2], sin_table: [S, D/2]
    ///
    /// Forward:
    ///   x_rot[..., 2i]   = x[..., 2i] * cos[pos,i] - x[..., 2i+1] * sin[pos,i]
    ///   x_rot[..., 2i+1] = x[..., 2i] * sin[pos,i] + x[..., 2i+1] * cos[pos,i]
    pub fn rope(
        &mut self,
        input: VariableId,
        cos_table: &Tensor,
        sin_table: &Tensor,
    ) -> VariableId {
        let input_data = self.variables[&input].data.clone();
        let shape = &input_data.shape;
        assert_eq!(shape.len(), 4);
        let (batch, heads, seq, d_head) = (shape[0], shape[1], shape[2], shape[3]);
        let half_d = d_head / 2;

        let mut output = vec![0.0f32; input_data.numel()];

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    let base = ((b * heads + h) * seq + s) * d_head;
                    for i in 0..half_d {
                        let cos_val = cos_table.data[s * half_d + i];
                        let sin_val = sin_table.data[s * half_d + i];
                        let x_even = input_data.data[base + 2 * i];
                        let x_odd = input_data.data[base + 2 * i + 1];

                        output[base + 2 * i] = x_even * cos_val - x_odd * sin_val;
                        output[base + 2 * i + 1] = x_even * sin_val + x_odd * cos_val;
                    }
                }
            }
        }

        let out_tensor = Tensor::new(output, shape.clone());

        if !self.should_track(&[input]) {
            return self.untracked(out_tensor);
        }

        self.record_op(
            Box::new(RoPEBackward {
                input_ids: vec![input],
                cos_table: cos_table.clone(),
                sin_table: sin_table.clone(),
            }),
            &[input],
            out_tensor,
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

    /// Generate cos/sin tables using standard RoPE frequencies.
    fn make_rope_tables(seq: usize, half_d: usize) -> (Tensor, Tensor) {
        let mut cos_data = vec![0.0f32; seq * half_d];
        let mut sin_data = vec![0.0f32; seq * half_d];
        for s in 0..seq {
            for i in 0..half_d {
                let theta = (s as f32) / 10000.0f32.powf(2.0 * i as f32 / (2 * half_d) as f32);
                cos_data[s * half_d + i] = theta.cos();
                sin_data[s * half_d + i] = theta.sin();
            }
        }
        (
            Tensor::new(cos_data, vec![seq, half_d]),
            Tensor::new(sin_data, vec![seq, half_d]),
        )
    }

    #[test]
    fn test_rope_grad_check_2_4_16_32() {
        let (batch, heads, seq, d_head) = (2, 4, 16, 32);
        let half_d = d_head / 2;
        let (cos_table, sin_table) = make_rope_tables(seq, half_d);

        let inputs = vec![make_tensor(&[batch, heads, seq, d_head], 42)];
        assert!(
            grad_check(
                |g, v| g.rope(v[0], &cos_table, &sin_table),
                &inputs,
                1e-3,
                5e-2,
            ),
            "grad_check failed for RoPE [2,4,16,32]"
        );
    }

    #[test]
    fn test_rope_output_shape() {
        let (batch, heads, seq, d_head) = (2, 4, 8, 16);
        let half_d = d_head / 2;
        let (cos_table, sin_table) = make_rope_tables(seq, half_d);

        let mut g = Graph::new();
        let input = g.variable(make_tensor(&[batch, heads, seq, d_head], 1), true);
        let out = g.rope(input, &cos_table, &sin_table);

        assert_eq!(g.data(out).shape, vec![2, 4, 8, 16]);

        backward(&mut g, out);
        assert_eq!(g.grad(input).unwrap().shape, vec![2, 4, 8, 16]);
    }

    #[test]
    fn test_rope_inverse_roundtrip() {
        // Applying RoPE forward then backward (inverse rotation) should recover the input.
        let (batch, heads, seq, d_head) = (1, 1, 4, 8);
        let half_d = d_head / 2;
        let (cos_table, sin_table) = make_rope_tables(seq, half_d);

        let input = make_tensor(&[batch, heads, seq, d_head], 99);

        let mut g = Graph::new();
        let vi = g.variable(input.clone(), false);
        let out = g.rope(vi, &cos_table, &sin_table);
        let out_data = g.data(out).clone();

        // Apply inverse rotation manually
        let mut recovered = vec![0.0f32; out_data.numel()];
        for s in 0..seq {
            let base = s * d_head;
            for i in 0..half_d {
                let cos_val = cos_table.data[s * half_d + i];
                let sin_val = sin_table.data[s * half_d + i];
                let y_even = out_data.data[base + 2 * i];
                let y_odd = out_data.data[base + 2 * i + 1];
                recovered[base + 2 * i] = y_even * cos_val + y_odd * sin_val;
                recovered[base + 2 * i + 1] = -y_even * sin_val + y_odd * cos_val;
            }
        }

        for (a, b) in input.data.iter().zip(&recovered) {
            assert!((a - b).abs() < 1e-5, "roundtrip mismatch: {} vs {}", a, b);
        }
    }
}
