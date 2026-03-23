use crate::registry::{AttentionVariant, FFNVariant, NormVariant};

/// A single decoder transformer layer (pre-norm architecture).
///
/// Forward: norm → attention → residual → norm → FFN → residual.
pub struct DecoderLayer {
    pub attn_norm: Box<dyn NormVariant>,
    pub attention: Box<dyn AttentionVariant>,
    pub ffn_norm: Box<dyn NormVariant>,
    pub ffn: Box<dyn FFNVariant>,
    pub hidden_size: usize,

    // ── Weights ──────────────────────────────────────────────────────
    pub attn_norm_weight: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

impl DecoderLayer {
    /// Pre-norm forward: norm→attention→residual→norm→FFN→residual.
    ///
    /// `x` is `[seq_len, hidden_size]` (flat). Returns same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out = self.apply_attention(&normed, seq_len);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Number of trainable parameters in this layer.
    pub fn param_count(&self) -> usize {
        let h = self.hidden_size;
        let q_dim = self.attention.num_heads() * self.attention.head_dim();
        let kv_dim = self.attention.num_kv_heads() * self.attention.head_dim();
        let inter = self.ffn.intermediate_size();

        let norms = 2 * h; // attn_norm + ffn_norm
        let attn = q_dim * h + kv_dim * h + kv_dim * h + h * q_dim;
        let ffn = if is_gated_ffn(self.ffn.name()) {
            3 * inter * h
        } else {
            2 * inter * h
        };

        norms + attn + ffn
    }

    /// Weight keys this layer expects (relative to `layers[i].`).
    pub fn weight_keys(&self, layer_idx: usize) -> Vec<String> {
        let i = layer_idx;
        let mut keys = vec![
            format!("layers[{i}].attn_norm.weight"),
            format!("layers[{i}].attention.w_q"),
            format!("layers[{i}].attention.w_k"),
            format!("layers[{i}].attention.w_v"),
            format!("layers[{i}].attention.w_o"),
            format!("layers[{i}].ffn_norm.weight"),
        ];
        if is_gated_ffn(self.ffn.name()) {
            keys.push(format!("layers[{i}].ffn.w_gate"));
        }
        keys.push(format!("layers[{i}].ffn.w_up"));
        keys.push(format!("layers[{i}].ffn.w_down"));
        keys
    }

    /// Assign a weight by its field name (e.g. "attention.w_q").
    pub fn set_weight(&mut self, field: &str, data: &[f32]) {
        match field {
            "attn_norm.weight" => self.attn_norm_weight = data.to_vec(),
            "attention.w_q" => self.w_q = data.to_vec(),
            "attention.w_k" => self.w_k = data.to_vec(),
            "attention.w_v" => self.w_v = data.to_vec(),
            "attention.w_o" => self.w_o = data.to_vec(),
            "ffn_norm.weight" => self.ffn_norm_weight = data.to_vec(),
            "ffn.w_gate" => self.ffn_gate = data.to_vec(),
            "ffn.w_up" => self.ffn_up = data.to_vec(),
            "ffn.w_down" => self.ffn_down = data.to_vec(),
            _ => {}
        }
    }

    // ── Attention ────────────────────────────────────────────────────

    fn apply_attention(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q, K, V projections: x is [seq_len, h]
        let q = matmul_t(x, &self.w_q, seq_len, h, q_dim);
        let k = matmul_t(x, &self.w_k, seq_len, h, kv_dim);
        let v = matmul_t(x, &self.w_v, seq_len, h, kv_dim);

        // Multi-head causal attention with GQA support
        let mut attn_output = vec![0.0f32; seq_len * q_dim];

        for head in 0..num_heads {
            let kv_head = head / groups;

            for t in 0..seq_len {
                // Compute scores for position t attending to 0..=t
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + head * head_dim + d]
                            * k[s * kv_dim + kv_head * head_dim + d];
                    }
                    scores[s] = dot * scale;
                }

                softmax_slice(&mut scores[..=t]);

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_head * head_dim + d];
                    }
                    attn_output[t * q_dim + head * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        matmul_t(&attn_output, &self.w_o, seq_len, q_dim, h)
    }

    // ── FFN ──────────────────────────────────────────────────────────

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = silu(gate[i]) * up[i];
                }
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            "GeGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = gelu(gate[i]) * up[i];
                }
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            _ => {
                // GELU or others: y = activation(x @ up^T) @ down^T
                let mut activated = matmul_t(x, &self.ffn_up, tokens, h, inter);
                for v in activated.iter_mut() {
                    *v = gelu(*v);
                }
                matmul_t(&activated, &self.ffn_down, tokens, inter, h)
            }
        }
    }
}

// ── Math helpers ─────────────────────────────────────────────────────

/// Whether this FFN variant is gated (3 weight matrices vs 2).
pub(crate) fn is_gated_ffn(name: &str) -> bool {
    matches!(name, "SwiGLU" | "GeGLU")
}

/// Apply normalization (dispatch on variant name).
pub(crate) fn apply_norm(
    x: &[f32],
    weight: &[f32],
    norm: &dyn NormVariant,
    hidden_size: usize,
) -> Vec<f32> {
    let eps = norm.eps() as f32;
    match norm.name() {
        "RMSNorm" => rmsnorm(x, weight, eps, hidden_size),
        "LayerNorm" => layernorm(x, weight, eps, hidden_size),
        _ => x.to_vec(),
    }
}

/// y = A * B^T  where A is [m, k], B is [n, k] → y is [m, n].
pub(crate) fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += a_row[d] * b_row[d];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// RMS normalization over the last dimension.
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = slice[j] * scale * weight[j];
        }
    }
    out
}

/// Layer normalization over the last dimension (gamma only, no beta).
fn layernorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mean: f32 = slice.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (slice[j] - mean) * scale * weight[j];
        }
    }
    out
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

pub(crate) fn softmax_slice(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

pub(crate) fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub(crate) fn add_vecs_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}
