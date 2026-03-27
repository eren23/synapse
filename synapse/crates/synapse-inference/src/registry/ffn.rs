use super::FFNVariant;

/// Activation function variants for StandardFFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    GeLU,
    SiLU,
}

impl Activation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::GeLU => {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            }
            Activation::SiLU => x / (1.0 + (-x).exp()),
        }
    }
}

// ── helpers ────────────────────────────────────────────────────────────

/// C = A[m,k] @ B^T[k,n]  where B is stored row-major as [n,k].
fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = s;
        }
    }
    c
}

// ── SwiGLU FFN ─────────────────────────────────────────────────────────

/// SwiGLU FFN: `down_proj @ swiglu(gate_proj @ x, up_proj @ x)`
///
/// Weight matrices:
/// - gate_proj: `[intermediate_size, hidden_size]`
/// - up_proj:   `[intermediate_size, hidden_size]`
/// - down_proj: `[hidden_size, intermediate_size]`
///
/// The FFI forward path calls `syn_sgemm` + `syn_swiglu`.
#[derive(Debug, Clone)]
pub struct SwiGLUFFN {
    hidden_size: usize,
    intermediate_size_val: usize,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl SwiGLUFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            gate_proj: vec![0.0; intermediate_size * hidden_size],
            up_proj: vec![0.0; intermediate_size * hidden_size],
            down_proj: vec![0.0; hidden_size * intermediate_size],
        }
    }

    pub fn with_weights(
        hidden_size: usize,
        intermediate_size: usize,
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
    ) -> Self {
        assert_eq!(gate_proj.len(), intermediate_size * hidden_size);
        assert_eq!(up_proj.len(), intermediate_size * hidden_size);
        assert_eq!(down_proj.len(), hidden_size * intermediate_size);
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn param_count(&self) -> usize {
        3 * self.hidden_size * self.intermediate_size_val
    }

    pub fn output_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.hidden_size]
    }

    pub fn intermediate_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.intermediate_size_val]
    }

    /// Reference forward pass (pure Rust).
    /// Input: `[tokens, hidden_size]` where `tokens = batch * seq`.
    pub fn forward(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        assert_eq!(input.len(), tokens * self.hidden_size);
        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;

        // gate = input[m,k] @ gate_proj^T[k,n]
        let gate = matmul_a_bt(input, &self.gate_proj, m, n, k);
        let up = matmul_a_bt(input, &self.up_proj, m, n, k);

        // swiglu: silu(gate) * up
        let mut hidden = vec![0.0f32; m * n];
        for i in 0..m * n {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            hidden[i] = silu * up[i];
        }

        // output = hidden[m,n] @ down_proj^T[n,k]
        matmul_a_bt(&hidden, &self.down_proj, m, k, n)
    }

    /// Forward pass using Zig FFI (`syn_sgemm` + `syn_swiglu`).
    ///
    /// # Safety
    /// Requires the synapse Zig library to be linked.
    pub unsafe fn forward_ffi(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        use synapse_sys::*;

        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;

        let mut gate = vec![0.0f32; m * n];
        let mut up = vec![0.0f32; m * n];
        let mut hidden = vec![0.0f32; m * n];
        let mut output = vec![0.0f32; m * k];

        syn_sgemm(
            m,
            n,
            k,
            input.as_ptr(),
            k,
            0,
            self.gate_proj.as_ptr(),
            k,
            1,
            gate.as_mut_ptr(),
            n,
        );
        syn_sgemm(
            m,
            n,
            k,
            input.as_ptr(),
            k,
            0,
            self.up_proj.as_ptr(),
            k,
            1,
            up.as_mut_ptr(),
            n,
        );
        syn_swiglu(hidden.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), m * n);
        syn_sgemm(
            m,
            k,
            n,
            hidden.as_ptr(),
            n,
            0,
            self.down_proj.as_ptr(),
            n,
            1,
            output.as_mut_ptr(),
            k,
        );

        output
    }
}

impl FFNVariant for SwiGLUFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size_val
    }
    fn name(&self) -> &str {
        "SwiGLU"
    }
}

// ── Standard FFN ───────────────────────────────────────────────────────

/// Standard FFN: `w2 @ activation(w1 @ x)`
///
/// Weight matrices:
/// - w1: `[intermediate_size, hidden_size]`
/// - w2: `[hidden_size, intermediate_size]`
#[derive(Debug, Clone)]
pub struct StandardFFN {
    hidden_size: usize,
    intermediate_size_val: usize,
    activation: Activation,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl StandardFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize, activation: Activation) -> Self {
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            activation,
            w1: vec![0.0; intermediate_size * hidden_size],
            w2: vec![0.0; hidden_size * intermediate_size],
        }
    }

    pub fn with_weights(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        w1: Vec<f32>,
        w2: Vec<f32>,
    ) -> Self {
        assert_eq!(w1.len(), intermediate_size * hidden_size);
        assert_eq!(w2.len(), hidden_size * intermediate_size);
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            activation,
            w1,
            w2,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn param_count(&self) -> usize {
        2 * self.hidden_size * self.intermediate_size_val
    }

    pub fn output_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.hidden_size]
    }

    pub fn intermediate_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.intermediate_size_val]
    }

    /// Reference forward pass (pure Rust).
    pub fn forward(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        assert_eq!(input.len(), tokens * self.hidden_size);
        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;

        let mut hidden = matmul_a_bt(input, &self.w1, m, n, k);
        for v in hidden.iter_mut() {
            *v = self.activation.apply(*v);
        }

        matmul_a_bt(&hidden, &self.w2, m, k, n)
    }

    /// Forward pass using Zig FFI (`syn_sgemm` + activation).
    ///
    /// # Safety
    /// Requires the synapse Zig library to be linked.
    pub unsafe fn forward_ffi(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        use synapse_sys::*;

        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;
        let mn = m * n;

        let mut hidden = vec![0.0f32; mn];
        let mut output = vec![0.0f32; m * k];

        syn_sgemm(
            m,
            n,
            k,
            input.as_ptr(),
            k,
            0,
            self.w1.as_ptr(),
            k,
            1,
            hidden.as_mut_ptr(),
            n,
        );

        // Apply activation via the matching FFI function.
        let src = hidden.clone();
        match self.activation {
            Activation::ReLU => {
                syn_relu(hidden.as_mut_ptr(), src.as_ptr(), mn);
            }
            Activation::GeLU => {
                syn_gelu(hidden.as_mut_ptr(), src.as_ptr(), mn);
            }
            Activation::SiLU => {
                syn_silu(hidden.as_mut_ptr(), src.as_ptr(), mn);
            }
        }

        syn_sgemm(
            m,
            k,
            n,
            hidden.as_ptr(),
            n,
            0,
            self.w2.as_ptr(),
            n,
            1,
            output.as_mut_ptr(),
            k,
        );

        output
    }
}

impl FFNVariant for StandardFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size_val
    }
    fn name(&self) -> &str {
        "GELU" // kept for backward compat with existing FFNConfig::GELU
    }
}

// ── GeGLU FFN ──────────────────────────────────────────────────────────

/// GeGLU FFN: `down_proj @ (gelu(gate_proj @ x) * up_proj @ x)`
///
/// Same structure as SwiGLU but with GELU gating instead of SiLU.
#[derive(Debug, Clone)]
pub struct GeGLUFFN {
    hidden_size: usize,
    intermediate_size_val: usize,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl GeGLUFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            gate_proj: vec![0.0; intermediate_size * hidden_size],
            up_proj: vec![0.0; intermediate_size * hidden_size],
            down_proj: vec![0.0; hidden_size * intermediate_size],
        }
    }

    pub fn with_weights(
        hidden_size: usize,
        intermediate_size: usize,
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
    ) -> Self {
        assert_eq!(gate_proj.len(), intermediate_size * hidden_size);
        assert_eq!(up_proj.len(), intermediate_size * hidden_size);
        assert_eq!(down_proj.len(), hidden_size * intermediate_size);
        Self {
            hidden_size,
            intermediate_size_val: intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn param_count(&self) -> usize {
        3 * self.hidden_size * self.intermediate_size_val
    }

    pub fn output_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.hidden_size]
    }

    pub fn intermediate_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        vec![batch, seq, self.intermediate_size_val]
    }

    /// Reference forward pass (pure Rust).
    pub fn forward(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        assert_eq!(input.len(), tokens * self.hidden_size);
        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;

        let gate = matmul_a_bt(input, &self.gate_proj, m, n, k);
        let up = matmul_a_bt(input, &self.up_proj, m, n, k);

        // GeGLU: gelu(gate) * up
        let mut hidden = vec![0.0f32; m * n];
        let gelu = Activation::GeLU;
        for i in 0..m * n {
            hidden[i] = gelu.apply(gate[i]) * up[i];
        }

        matmul_a_bt(&hidden, &self.down_proj, m, k, n)
    }

    /// Forward pass using Zig FFI (`syn_sgemm` + `syn_gelu` + `syn_mul`).
    ///
    /// # Safety
    /// Requires the synapse Zig library to be linked.
    pub unsafe fn forward_ffi(&self, input: &[f32], tokens: usize) -> Vec<f32> {
        use synapse_sys::*;

        let m = tokens;
        let k = self.hidden_size;
        let n = self.intermediate_size_val;
        let mn = m * n;

        let mut gate = vec![0.0f32; mn];
        let mut up = vec![0.0f32; mn];
        let mut output = vec![0.0f32; m * k];

        syn_sgemm(
            m,
            n,
            k,
            input.as_ptr(),
            k,
            0,
            self.gate_proj.as_ptr(),
            k,
            1,
            gate.as_mut_ptr(),
            n,
        );
        syn_sgemm(
            m,
            n,
            k,
            input.as_ptr(),
            k,
            0,
            self.up_proj.as_ptr(),
            k,
            1,
            up.as_mut_ptr(),
            n,
        );

        // gelu(gate)
        let gate_src = gate.clone();
        syn_gelu(gate.as_mut_ptr(), gate_src.as_ptr(), mn);

        // hidden = gelu(gate) * up
        syn_mul(gate.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), mn);

        syn_sgemm(
            m,
            k,
            n,
            gate.as_ptr(),
            n,
            0,
            self.down_proj.as_ptr(),
            n,
            1,
            output.as_mut_ptr(),
            k,
        );

        output
    }
}

impl FFNVariant for GeGLUFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size_val
    }
    fn name(&self) -> &str {
        "GeGLU"
    }
}

// ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── SwiGLU tests ───────────────────────────────────────────────

    #[test]
    fn swiglu_output_shape() {
        let ffn = SwiGLUFFN::new(64, 128);
        assert_eq!(ffn.output_shape(2, 4), vec![2, 4, 64]);
    }

    #[test]
    fn swiglu_intermediate_shape() {
        let ffn = SwiGLUFFN::new(64, 128);
        assert_eq!(ffn.intermediate_shape(2, 4), vec![2, 4, 128]);
    }

    #[test]
    fn swiglu_param_count() {
        let hidden = 64;
        let intermediate = 128;
        let ffn = SwiGLUFFN::new(hidden, intermediate);
        assert_eq!(ffn.param_count(), 3 * hidden * intermediate);
    }

    #[test]
    fn swiglu_forward_output_len() {
        let hidden = 4;
        let intermediate = 8;
        // Use small identity-ish weights so output is non-trivial.
        let gate_proj = vec![0.1f32; intermediate * hidden];
        let up_proj = vec![0.1f32; intermediate * hidden];
        let down_proj = vec![0.1f32; hidden * intermediate];

        let ffn = SwiGLUFFN::with_weights(hidden, intermediate, gate_proj, up_proj, down_proj);
        let tokens = 3;
        let input = vec![1.0f32; tokens * hidden];
        let out = ffn.forward(&input, tokens);
        assert_eq!(out.len(), tokens * hidden);
    }

    // ── GeGLU tests ────────────────────────────────────────────────

    #[test]
    fn geglu_output_shape() {
        let ffn = GeGLUFFN::new(64, 128);
        assert_eq!(ffn.output_shape(2, 4), vec![2, 4, 64]);
    }

    #[test]
    fn geglu_intermediate_shape() {
        let ffn = GeGLUFFN::new(64, 128);
        assert_eq!(ffn.intermediate_shape(2, 4), vec![2, 4, 128]);
    }

    #[test]
    fn geglu_param_count() {
        let hidden = 64;
        let intermediate = 128;
        let ffn = GeGLUFFN::new(hidden, intermediate);
        assert_eq!(ffn.param_count(), 3 * hidden * intermediate);
    }

    #[test]
    fn swiglu_and_geglu_produce_different_output() {
        let hidden = 4;
        let intermediate = 8;

        // Same non-zero weights for both.
        let gate = vec![0.1f32; intermediate * hidden];
        let up = vec![0.2f32; intermediate * hidden];
        let down = vec![0.1f32; hidden * intermediate];

        let swiglu =
            SwiGLUFFN::with_weights(hidden, intermediate, gate.clone(), up.clone(), down.clone());
        let geglu = GeGLUFFN::with_weights(hidden, intermediate, gate, up, down);

        let tokens = 2;
        let input = vec![1.0f32; tokens * hidden];

        let out_sw = swiglu.forward(&input, tokens);
        let out_ge = geglu.forward(&input, tokens);

        let diff: f32 = out_sw
            .iter()
            .zip(out_ge.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "SwiGLU and GeGLU should produce different outputs, diff = {diff}"
        );
    }

    // ── StandardFFN tests ──────────────────────────────────────────

    #[test]
    fn standard_ffn_output_shape() {
        let ffn = StandardFFN::new(64, 256, Activation::GeLU);
        assert_eq!(ffn.output_shape(2, 4), vec![2, 4, 64]);
    }

    #[test]
    fn standard_ffn_param_count() {
        let hidden = 64;
        let intermediate = 256;
        let ffn = StandardFFN::new(hidden, intermediate, Activation::ReLU);
        assert_eq!(ffn.param_count(), 2 * hidden * intermediate);
    }

    #[test]
    fn standard_ffn_forward_output_len() {
        let hidden = 4;
        let intermediate = 8;
        let w1 = vec![0.1f32; intermediate * hidden];
        let w2 = vec![0.1f32; hidden * intermediate];

        let ffn = StandardFFN::with_weights(hidden, intermediate, Activation::ReLU, w1, w2);
        let tokens = 3;
        let input = vec![1.0f32; tokens * hidden];
        let out = ffn.forward(&input, tokens);
        assert_eq!(out.len(), tokens * hidden);
    }

    // ── Trait dispatch ─────────────────────────────────────────────

    #[test]
    fn ffn_trait_dispatch() {
        let variants: Vec<Box<dyn FFNVariant>> = vec![
            Box::new(SwiGLUFFN::new(64, 128)),
            Box::new(StandardFFN::new(64, 256, Activation::GeLU)),
            Box::new(GeGLUFFN::new(64, 128)),
        ];

        assert_eq!(variants[0].name(), "SwiGLU");
        assert_eq!(variants[0].intermediate_size(), 128);

        assert_eq!(variants[1].name(), "GELU");
        assert_eq!(variants[1].intermediate_size(), 256);

        assert_eq!(variants[2].name(), "GeGLU");
        assert_eq!(variants[2].intermediate_size(), 128);
    }

    // ── FFI round-trip tests ───────────────────────────────────────

    #[test]
    fn swiglu_ffi_matches_reference() {
        let hidden = 4;
        let intermediate = 8;
        let gate = vec![0.1f32; intermediate * hidden];
        let up = vec![0.2f32; intermediate * hidden];
        let down = vec![0.1f32; hidden * intermediate];

        let ffn = SwiGLUFFN::with_weights(hidden, intermediate, gate, up, down);
        let tokens = 2;
        let input = vec![1.0f32; tokens * hidden];

        let ref_out = ffn.forward(&input, tokens);
        let ffi_out = unsafe { ffn.forward_ffi(&input, tokens) };

        for i in 0..ref_out.len() {
            assert!(
                (ref_out[i] - ffi_out[i]).abs() < 1e-4,
                "Mismatch at {i}: ref={}, ffi={}",
                ref_out[i],
                ffi_out[i]
            );
        }
    }

    #[test]
    fn geglu_ffi_matches_reference() {
        let hidden = 4;
        let intermediate = 8;
        let gate = vec![0.1f32; intermediate * hidden];
        let up = vec![0.2f32; intermediate * hidden];
        let down = vec![0.1f32; hidden * intermediate];

        let ffn = GeGLUFFN::with_weights(hidden, intermediate, gate, up, down);
        let tokens = 2;
        let input = vec![1.0f32; tokens * hidden];

        let ref_out = ffn.forward(&input, tokens);
        let ffi_out = unsafe { ffn.forward_ffi(&input, tokens) };

        for i in 0..ref_out.len() {
            assert!(
                (ref_out[i] - ffi_out[i]).abs() < 1e-4,
                "Mismatch at {i}: ref={}, ffi={}",
                ref_out[i],
                ffi_out[i]
            );
        }
    }
}
