//! Recurrent cells: LSTMCell, GRUCell.

use synapse_autograd::Tensor;

use crate::init::xavier_uniform;
use crate::module::Module;

// ── Helper: slice rows from a 2D tensor ──────────────────────────────

fn slice_rows(t: &Tensor, row_start: usize, row_end: usize) -> Tensor {
    let cols = t.shape[1];
    let data = t.data[row_start * cols..row_end * cols].to_vec();
    Tensor::new(data, vec![row_end - row_start, cols])
}

// ── LSTMCell ──────────────────────────────────────────────────────────

/// Single LSTM cell: (h', c') = lstm_cell(input, h, c)
///
/// Gates: i, f, g, o computed from concatenated [input, h]:
///   [i, f, g, o] = W_ih @ input^T + W_hh @ h^T + bias
///   i = sigmoid(i_gate), f = sigmoid(f_gate), g = tanh(g_gate), o = sigmoid(o_gate)
///   c' = f * c + i * g
///   h' = o * tanh(c')
pub struct LSTMCell {
    pub weight_ih: Tensor, // [4*hidden_size, input_size]
    pub weight_hh: Tensor, // [4*hidden_size, hidden_size]
    pub bias_ih: Tensor,   // [4*hidden_size]
    pub bias_hh: Tensor,   // [4*hidden_size]
    pub input_size: usize,
    pub hidden_size: usize,
    training: bool,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_ih = xavier_uniform(&[4 * hidden_size, input_size]);
        let weight_hh = xavier_uniform(&[4 * hidden_size, hidden_size]);
        let bias_ih = Tensor::zeros(&[4 * hidden_size]);
        let bias_hh = Tensor::zeros(&[4 * hidden_size]);
        LSTMCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            training: true,
        }
    }

    /// Forward: input [batch, input_size], h [batch, hidden_size], c [batch, hidden_size]
    /// Returns: (h', c') both [batch, hidden_size]
    pub fn forward_cell(&self, input: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        let batch = input.shape[0];
        assert_eq!(input.shape, vec![batch, self.input_size]);
        assert_eq!(h.shape, vec![batch, self.hidden_size]);
        assert_eq!(c.shape, vec![batch, self.hidden_size]);

        let hs = self.hidden_size;

        // gates = input @ W_ih^T + h @ W_hh^T + bias_ih + bias_hh
        // shape: [batch, 4*hidden_size]
        let gates = input
            .matmul(&self.weight_ih.transpose_2d())
            .add(&h.matmul(&self.weight_hh.transpose_2d()));

        // Add biases (broadcast from [4*hs] to [batch, 4*hs])
        let bias = self.bias_ih.add(&self.bias_hh);
        let bias_2d = bias.reshape(&[1, 4 * hs]);
        let gates = gates.add_broadcast(&bias_2d);

        // Split into 4 gates
        let i_gate = slice_rows(&gates.transpose_2d(), 0, hs).transpose_2d();
        let f_gate = slice_rows(&gates.transpose_2d(), hs, 2 * hs).transpose_2d();
        let g_gate = slice_rows(&gates.transpose_2d(), 2 * hs, 3 * hs).transpose_2d();
        let o_gate = slice_rows(&gates.transpose_2d(), 3 * hs, 4 * hs).transpose_2d();

        let i = i_gate.sigmoid();
        let f = f_gate.sigmoid();
        let g = g_gate.tanh_act();
        let o = o_gate.sigmoid();

        // c' = f * c + i * g
        let c_new = f.mul(c).add(&i.mul(&g));
        // h' = o * tanh(c')
        let h_new = o.mul(&c_new.tanh_act());

        (h_new, c_new)
    }
}

impl Module for LSTMCell {
    /// Forward with zero initial hidden state.
    /// Input: [batch, input_size] -> Output: [batch, hidden_size] (h only)
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch = input.shape[0];
        let h = Tensor::zeros(&[batch, self.hidden_size]);
        let c = Tensor::zeros(&[batch, self.hidden_size]);
        let (h_new, _c_new) = self.forward_cell(input, &h, &c);
        h_new
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.weight_ih,
            &mut self.weight_hh,
            &mut self.bias_ih,
            &mut self.bias_hh,
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "LSTMCell"
    }
}

// ── GRUCell ───────────────────────────────────────────────────────────

/// Single GRU cell: h' = gru_cell(input, h)
///
/// r = sigmoid(W_ir @ input + W_hr @ h + bias_r)
/// z = sigmoid(W_iz @ input + W_hz @ h + bias_z)
/// n = tanh(W_in @ input + r * (W_hn @ h) + bias_n)
/// h' = (1 - z) * n + z * h
pub struct GRUCell {
    pub weight_ih: Tensor, // [3*hidden_size, input_size]
    pub weight_hh: Tensor, // [3*hidden_size, hidden_size]
    pub bias_ih: Tensor,   // [3*hidden_size]
    pub bias_hh: Tensor,   // [3*hidden_size]
    pub input_size: usize,
    pub hidden_size: usize,
    training: bool,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_ih = xavier_uniform(&[3 * hidden_size, input_size]);
        let weight_hh = xavier_uniform(&[3 * hidden_size, hidden_size]);
        let bias_ih = Tensor::zeros(&[3 * hidden_size]);
        let bias_hh = Tensor::zeros(&[3 * hidden_size]);
        GRUCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            training: true,
        }
    }

    /// Forward: input [batch, input_size], h [batch, hidden_size]
    /// Returns: h' [batch, hidden_size]
    pub fn forward_cell(&self, input: &Tensor, h: &Tensor) -> Tensor {
        let batch = input.shape[0];
        assert_eq!(input.shape, vec![batch, self.input_size]);
        assert_eq!(h.shape, vec![batch, self.hidden_size]);

        let hs = self.hidden_size;

        // input_gates = input @ W_ih^T + bias_ih  -> [batch, 3*hs]
        let bias_ih_2d = self.bias_ih.reshape(&[1, 3 * hs]);
        let ig = input
            .matmul(&self.weight_ih.transpose_2d())
            .add_broadcast(&bias_ih_2d);

        // hidden_gates = h @ W_hh^T + bias_hh -> [batch, 3*hs]
        let bias_hh_2d = self.bias_hh.reshape(&[1, 3 * hs]);
        let hg = h
            .matmul(&self.weight_hh.transpose_2d())
            .add_broadcast(&bias_hh_2d);

        // Split into r, z, n components
        // We need to slice columns: [batch, 3*hs] -> 3x [batch, hs]
        let ig_t = ig.transpose_2d(); // [3*hs, batch]
        let hg_t = hg.transpose_2d();

        let ig_r = slice_rows(&ig_t, 0, hs).transpose_2d();
        let ig_z = slice_rows(&ig_t, hs, 2 * hs).transpose_2d();
        let ig_n = slice_rows(&ig_t, 2 * hs, 3 * hs).transpose_2d();

        let hg_r = slice_rows(&hg_t, 0, hs).transpose_2d();
        let hg_z = slice_rows(&hg_t, hs, 2 * hs).transpose_2d();
        let hg_n = slice_rows(&hg_t, 2 * hs, 3 * hs).transpose_2d();

        let r = ig_r.add(&hg_r).sigmoid();
        let z = ig_z.add(&hg_z).sigmoid();
        let n = ig_n.add(&r.mul(&hg_n)).tanh_act();

        // h' = (1 - z) * n + z * h
        let ones = Tensor::ones(&z.shape);
        let h_new = ones.sub(&z).mul(&n).add(&z.mul(h));

        h_new
    }
}

impl Module for GRUCell {
    /// Forward with zero initial hidden state.
    /// Input: [batch, input_size] -> Output: [batch, hidden_size]
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch = input.shape[0];
        let h = Tensor::zeros(&[batch, self.hidden_size]);
        self.forward_cell(input, &h)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.weight_ih,
            &mut self.weight_hh,
            &mut self.bias_ih,
            &mut self.bias_hh,
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "GRUCell"
    }
}
