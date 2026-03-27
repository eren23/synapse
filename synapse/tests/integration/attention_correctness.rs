//! Gradient correctness for the full MultiHeadAttention module.
//!
//! Implements the complete MHA forward pass (Q/K/V/O projections, split heads,
//! scaled dot-product attention, concat heads) through the autograd graph and
//! verifies analytical gradients against numerical (central finite differences)
//! for every parameter: W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o.

use synapse_autograd::{backward, grad_check, Graph, Tensor};

/// Deterministic pseudo-random tensor with values in [-0.3, 0.3].
/// Kept small to avoid numerical issues with softmax saturation.
fn make_tensor(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed.wrapping_mul(2654435761);
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32 / u32::MAX as f32 - 0.5) * 0.6
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Build the full multi-head attention forward pass through the autograd graph.
///
/// vars layout:
///   [0] input:  [B, S, D]
///   [1] W_q:    [D, D]
///   [2] b_q:    [D]
///   [3] W_k:    [D, D]
///   [4] b_k:    [D]
///   [5] W_v:    [D, D]
///   [6] b_v:    [D]
///   [7] W_o:    [D, D]
///   [8] b_o:    [D]
///
/// Returns a scalar (sum of all output elements) for grad_check.
fn mha_graph(
    g: &mut Graph,
    vars: &[synapse_autograd::VariableId],
    batch: usize,
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
) -> synapse_autograd::VariableId {
    let d_head = d_model / n_heads;

    let input = vars[0];
    let w_q = vars[1];
    let b_q = vars[2];
    let w_k = vars[3];
    let b_k = vars[4];
    let w_v = vars[5];
    let b_v = vars[6];
    let w_o = vars[7];
    let b_o = vars[8];

    // Flatten: [B, S, D] -> [B*S, D]
    let flat = g.reshape(input, &[batch * seq_len, d_model]);

    // Q projection: [B*S, D] @ [D, D] + [D] -> [B*S, D]
    let q = g.matmul(flat, w_q);
    let q = g.add(q, b_q);

    // K projection
    let k = g.matmul(flat, w_k);
    let k = g.add(k, b_k);

    // V projection
    let v = g.matmul(flat, w_v);
    let v = g.add(v, b_v);

    // Split heads: [B*S, D] -> [B, S, H, D_h] -> transpose(1,2) -> [B, H, S, D_h]
    let q = g.reshape(q, &[batch, seq_len, n_heads, d_head]);
    let q = g.transpose(q, 1, 2);

    let k = g.reshape(k, &[batch, seq_len, n_heads, d_head]);
    let k = g.transpose(k, 1, 2);

    let v = g.reshape(v, &[batch, seq_len, n_heads, d_head]);
    let v = g.transpose(v, 1, 2);

    // Scaled dot-product attention: [B, H, S, D_h] -> [B, H, S, D_h]
    let attn = g.scaled_dot_product_attention(q, k, v, false);

    // Concat heads: transpose(1,2) -> [B, S, H, D_h] -> reshape -> [B*S, D]
    let attn = g.transpose(attn, 1, 2);
    let attn = g.reshape(attn, &[batch * seq_len, d_model]);

    // Output projection: [B*S, D] @ [D, D] + [D] -> [B*S, D]
    let out = g.matmul(attn, w_o);
    let out = g.add(out, b_o);

    // Reduce to scalar for grad_check
    g.sum_all(out)
}

// ── grad_check tests for the full MHA module ─────────────────────────

#[test]
fn test_mha_grad_check_small() {
    let batch = 1;
    let seq_len = 2;
    let d_model = 4;
    let n_heads = 2;

    let inputs = vec![
        make_tensor(&[batch, seq_len, d_model], 1), // input
        make_tensor(&[d_model, d_model], 10),       // W_q
        make_tensor(&[d_model], 11),                // b_q
        make_tensor(&[d_model, d_model], 20),       // W_k
        make_tensor(&[d_model], 21),                // b_k
        make_tensor(&[d_model, d_model], 30),       // W_v
        make_tensor(&[d_model], 31),                // b_v
        make_tensor(&[d_model, d_model], 40),       // W_o
        make_tensor(&[d_model], 41),                // b_o
    ];

    let pass = grad_check(
        |g, v| mha_graph(g, v, batch, seq_len, d_model, n_heads),
        &inputs,
        1e-3,
        5e-2,
    );
    assert!(pass, "grad_check failed for small MHA (B=1, S=2, D=4, H=2)");
}

#[test]
fn test_mha_grad_check_medium() {
    let batch = 2;
    let seq_len = 4;
    let d_model = 8;
    let n_heads = 2;

    let inputs = vec![
        make_tensor(&[batch, seq_len, d_model], 100),
        make_tensor(&[d_model, d_model], 110),
        make_tensor(&[d_model], 111),
        make_tensor(&[d_model, d_model], 120),
        make_tensor(&[d_model], 121),
        make_tensor(&[d_model, d_model], 130),
        make_tensor(&[d_model], 131),
        make_tensor(&[d_model, d_model], 140),
        make_tensor(&[d_model], 141),
    ];

    let pass = grad_check(
        |g, v| mha_graph(g, v, batch, seq_len, d_model, n_heads),
        &inputs,
        1e-3,
        5e-2,
    );
    assert!(
        pass,
        "grad_check failed for medium MHA (B=2, S=4, D=8, H=2)"
    );
}

#[test]
fn test_mha_grad_check_4_heads() {
    let batch = 1;
    let seq_len = 3;
    let d_model = 8;
    let n_heads = 4;

    let inputs = vec![
        make_tensor(&[batch, seq_len, d_model], 200),
        make_tensor(&[d_model, d_model], 210),
        make_tensor(&[d_model], 211),
        make_tensor(&[d_model, d_model], 220),
        make_tensor(&[d_model], 221),
        make_tensor(&[d_model, d_model], 230),
        make_tensor(&[d_model], 231),
        make_tensor(&[d_model, d_model], 240),
        make_tensor(&[d_model], 241),
    ];

    let pass = grad_check(
        |g, v| mha_graph(g, v, batch, seq_len, d_model, n_heads),
        &inputs,
        1e-3,
        5e-2,
    );
    assert!(
        pass,
        "grad_check failed for MHA with 4 heads (B=1, S=3, D=8, H=4)"
    );
}

// ── Verify gradient existence and finiteness for all parameters ──────

#[test]
fn test_mha_all_params_have_finite_gradients() {
    let batch = 2;
    let seq_len = 4;
    let d_model = 8;
    let n_heads = 2;

    let input = make_tensor(&[batch, seq_len, d_model], 300);
    let w_q = make_tensor(&[d_model, d_model], 310);
    let b_q = make_tensor(&[d_model], 311);
    let w_k = make_tensor(&[d_model, d_model], 320);
    let b_k = make_tensor(&[d_model], 321);
    let w_v = make_tensor(&[d_model, d_model], 330);
    let b_v = make_tensor(&[d_model], 331);
    let w_o = make_tensor(&[d_model, d_model], 340);
    let b_o = make_tensor(&[d_model], 341);

    let mut g = Graph::new();
    let vars = [
        g.variable(input, true),
        g.variable(w_q, true),
        g.variable(b_q, true),
        g.variable(w_k, true),
        g.variable(b_k, true),
        g.variable(w_v, true),
        g.variable(b_v, true),
        g.variable(w_o, true),
        g.variable(b_o, true),
    ];

    let output = mha_graph(&mut g, &vars, batch, seq_len, d_model, n_heads);
    backward(&mut g, output);

    let param_names = [
        "input", "W_q", "b_q", "W_k", "b_k", "W_v", "b_v", "W_o", "b_o",
    ];

    for (i, &var) in vars.iter().enumerate() {
        let grad = g.grad(var);
        assert!(
            grad.is_some(),
            "gradient should exist for {}",
            param_names[i]
        );
        let grad = grad.unwrap();
        assert!(
            !grad.data.is_empty(),
            "gradient for {} should be non-empty",
            param_names[i]
        );
        for (j, &val) in grad.data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "gradient for {} contains non-finite at index {}: {}",
                param_names[i],
                j,
                val
            );
        }
        // Verify gradient is non-trivial (not all zeros)
        let max_abs: f32 = grad.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-10,
            "gradient for {} is all zeros (max_abs={})",
            param_names[i],
            max_abs
        );
    }
}
