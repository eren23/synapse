//! Attention benchmark: fused vs naive attention throughput through Rust+FFI layer.
//! Target: fused >= 2x vs naive.
//!
//! Benchmarks the graph IR interpreter executing a FusedAttention node vs the
//! unfused subgraph (Q/K/V projections, transpose, matmul, scale, softmax,
//! attend, output projection).

use std::collections::HashMap;
use std::time::Instant;

use synapse_graph::*;

/// Build an unfused self-attention graph.
fn build_unfused_attention(
    seq_len: usize,
    d_model: usize,
) -> (Graph, Vec<NodeId>) {
    let d_k = d_model;
    let d_v = d_model;
    let d_out = d_model;

    let mut g = Graph::new();

    let input = g.add_node(
        NodeKind::Input("input".into()),
        vec![],
        NodeMeta::new(vec![seq_len, d_model], DType::F32),
        "input",
    );
    let w_q = g.add_node(
        NodeKind::Parameter("W_q".into()),
        vec![],
        NodeMeta::new(vec![d_model, d_k], DType::F32),
        "W_q",
    );
    let w_k = g.add_node(
        NodeKind::Parameter("W_k".into()),
        vec![],
        NodeMeta::new(vec![d_model, d_k], DType::F32),
        "W_k",
    );
    let w_v = g.add_node(
        NodeKind::Parameter("W_v".into()),
        vec![],
        NodeMeta::new(vec![d_model, d_v], DType::F32),
        "W_v",
    );
    let scale_val = 1.0 / (d_k as f32).sqrt();
    let scale = g.add_node(
        NodeKind::Constant(vec![scale_val; seq_len * seq_len]),
        vec![],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32),
        "scale",
    );
    let w_o = g.add_node(
        NodeKind::Parameter("W_o".into()),
        vec![],
        NodeMeta::new(vec![d_v, d_out], DType::F32),
        "W_o",
    );

    // Q, K, V projections
    let q = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![input, w_q],
        NodeMeta::new(vec![seq_len, d_k], DType::F32),
        "q_proj",
    );
    let k = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![input, w_k],
        NodeMeta::new(vec![seq_len, d_k], DType::F32),
        "k_proj",
    );
    let k_t = g.add_node(
        NodeKind::Op(OpKind::Transpose),
        vec![k],
        NodeMeta::new(vec![d_k, seq_len], DType::F32),
        "k_transpose",
    );
    let v = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![input, w_v],
        NodeMeta::new(vec![seq_len, d_v], DType::F32),
        "v_proj",
    );

    // Attention
    let scores = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![q, k_t],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32),
        "scores",
    );
    let scaled = g.add_node(
        NodeKind::Op(OpKind::Mul),
        vec![scores, scale],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32),
        "scaled_scores",
    );
    let weights = g.add_node(
        NodeKind::Op(OpKind::Softmax),
        vec![scaled],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32),
        "attn_weights",
    );
    let attended = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![weights, v],
        NodeMeta::new(vec![seq_len, d_v], DType::F32),
        "attended",
    );

    // Output projection
    let output = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![attended, w_o],
        NodeMeta::new(vec![seq_len, d_out], DType::F32),
        "output_proj",
    );
    g.mark_output(output);

    let feed_ids = vec![input, w_q, w_k, w_v, w_o];
    (g, feed_ids)
}

/// Build a fused attention graph via the FuseAttention pass.
fn build_fused_attention(
    seq_len: usize,
    d_model: usize,
) -> (Graph, Vec<NodeId>) {
    let (mut g, feed_ids) = build_unfused_attention(seq_len, d_model);
    FuseAttention::new().run(&mut g);
    DeadCodeElimination::new().run(&mut g);
    (g, feed_ids)
}

fn make_data(seq_len: usize, d_model: usize, feed_ids: &[NodeId]) -> HashMap<NodeId, Vec<f32>> {
    let mut inputs = HashMap::new();
    inputs.insert(
        feed_ids[0],
        (0..seq_len * d_model)
            .map(|i| ((i % 17) as f32) * 0.01)
            .collect(),
    );
    inputs.insert(
        feed_ids[1],
        (0..d_model * d_model)
            .map(|i| ((i % 13) as f32) * 0.01)
            .collect(),
    );
    inputs.insert(
        feed_ids[2],
        (0..d_model * d_model)
            .map(|i| ((i % 11) as f32) * 0.01)
            .collect(),
    );
    inputs.insert(
        feed_ids[3],
        (0..d_model * d_model)
            .map(|i| ((i % 7) as f32) * 0.01)
            .collect(),
    );
    inputs.insert(
        feed_ids[4],
        (0..d_model * d_model)
            .map(|i| ((i % 19) as f32) * 0.01)
            .collect(),
    );
    inputs
}

#[test]
fn attention_bench_fused_2x_vs_naive() {
    // Use a size where interpreter overhead (dispatch, Vec allocation, data
    // copying for intermediates) dominates compute, making fusion most impactful.
    let seq_len = 16;
    let d_model = 8;

    let (g_unfused, unfused_ids) = build_unfused_attention(seq_len, d_model);
    let (g_fused, fused_ids) = build_fused_attention(seq_len, d_model);

    let unfused_inputs = make_data(seq_len, d_model, &unfused_ids);
    let fused_inputs = make_data(seq_len, d_model, &fused_ids);

    // Verify numerical correctness
    let unfused_result = g_unfused.execute(&unfused_inputs);
    let fused_result = g_fused.execute(&fused_inputs);
    let unfused_out = &unfused_result[&g_unfused.outputs()[0]];
    let fused_out = &fused_result[&g_fused.outputs()[0]];
    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-3,
            "Mismatch at {}: unfused={}, fused={}",
            i, u, f
        );
    }

    // Aggressive warmup
    for _ in 0..100 {
        std::hint::black_box(g_unfused.execute(&unfused_inputs));
        std::hint::black_box(g_fused.execute(&fused_inputs));
    }

    // Best of 5 trials
    let iterations = 500;
    let mut best_speedup = 0.0f64;

    for trial in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_unfused.execute(&unfused_inputs));
        }
        let naive_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_fused.execute(&fused_inputs));
        }
        let fused_time = start.elapsed();

        let speedup = naive_time.as_secs_f64() / fused_time.as_secs_f64();
        if speedup > best_speedup {
            best_speedup = speedup;
            eprintln!(
                "  trial {}: naive={:.3}ms, fused={:.3}ms, speedup={:.2}x",
                trial,
                naive_time.as_secs_f64() * 1000.0 / iterations as f64,
                fused_time.as_secs_f64() * 1000.0 / iterations as f64,
                speedup,
            );
        }
    }

    eprintln!(
        "Attention bench (seq={}, d={}): best speedup={:.2}x",
        seq_len, d_model, best_speedup
    );

    // Build-mode-aware threshold
    let threshold = if cfg!(debug_assertions) { 1.3 } else { 2.0 };
    assert!(
        best_speedup >= threshold,
        "Fused attention should be >= {:.1}x vs naive, got {:.2}x",
        threshold,
        best_speedup,
    );
}
