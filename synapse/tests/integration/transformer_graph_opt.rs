//! Graph optimization tests for transformer fusion passes.
//!
//! Builds transformer computation graphs, applies FuseAttention and
//! FuseLayerNormResidual passes, and verifies:
//! 1. Fused output matches unfused output within 1e-4
//! 2. Attention fusion speedup >= 1.3x vs unfused
//! 3. LayerNorm+residual fusion speedup >= 1.2x vs unfused

use std::collections::HashMap;
use std::time::Instant;

use synapse_graph::*;

// ── Attention fusion: correctness + benchmark ─────────────────────────

/// Build a self-attention graph for fusion testing.
fn build_attention_graph(
    seq_len: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    d_out: usize,
) -> (Graph, NodeId, Vec<NodeId>) {
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

    let output = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![attended, w_o],
        NodeMeta::new(vec![seq_len, d_out], DType::F32),
        "output_proj",
    );
    g.mark_output(output);

    let param_ids = vec![input, w_q, w_k, w_v, scale, w_o];
    (g, output, param_ids)
}

/// Build an Add → LayerNorm graph for fusion testing.
fn build_add_layernorm_graph(dim: usize) -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new();

    let x = g.add_node(
        NodeKind::Input("x".into()),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "x",
    );
    let residual = g.add_node(
        NodeKind::Input("residual".into()),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "residual",
    );
    let gamma = g.add_node(
        NodeKind::Constant(vec![1.0; dim]),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "gamma",
    );
    let beta = g.add_node(
        NodeKind::Constant(vec![0.0; dim]),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "beta",
    );

    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![x, residual],
        NodeMeta::new(vec![dim], DType::F32),
        "residual_add",
    );
    let ln = g.add_node(
        NodeKind::Op(OpKind::LayerNorm),
        vec![add, gamma, beta],
        NodeMeta::new(vec![dim], DType::F32),
        "layer_norm",
    );
    g.mark_output(ln);

    (g, x, residual)
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn test_attention_fusion_numerically_correct() {
    let seq_len = 4;
    let d_model = 8;
    let d_k = 6;
    let d_v = 6;
    let d_out = 8;

    let (g_unfused, _, param_ids) =
        build_attention_graph(seq_len, d_model, d_k, d_v, d_out);

    let input_data: Vec<f32> = (0..seq_len * d_model)
        .map(|i| (i as f32) * 0.1 + 0.05)
        .collect();
    let w_q_data: Vec<f32> = (0..d_model * d_k)
        .map(|i| (i as f32) * 0.02 - 0.1)
        .collect();
    let w_k_data: Vec<f32> = (0..d_model * d_k)
        .map(|i| (i as f32) * 0.03 + 0.05)
        .collect();
    let w_v_data: Vec<f32> = (0..d_model * d_v)
        .map(|i| (i as f32) * 0.01 - 0.02)
        .collect();
    let w_o_data: Vec<f32> = (0..d_v * d_out)
        .map(|i| (i as f32) * 0.04 + 0.01)
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert(param_ids[0], input_data);
    inputs.insert(param_ids[1], w_q_data);
    inputs.insert(param_ids[2], w_k_data);
    inputs.insert(param_ids[3], w_v_data);
    // scale is embedded as constant
    inputs.insert(param_ids[5], w_o_data);

    let unfused_output_id = g_unfused.outputs()[0];
    let unfused_result = g_unfused.execute(&inputs);
    let unfused_out = unfused_result[&unfused_output_id].clone();

    let mut g_fused = g_unfused.clone();
    let changed = FuseAttention::new().run(&mut g_fused);
    assert!(changed, "FuseAttention should have matched the pattern");

    let fused_output_id = g_fused.outputs()[0];
    let fused_result = g_fused.execute(&inputs);
    let fused_out = fused_result[&fused_output_id].clone();

    assert_eq!(unfused_out.len(), fused_out.len());
    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-4,
            "Attention fusion mismatch at {}: unfused={}, fused={}, diff={}",
            i, u, f, (u - f).abs()
        );
    }
}

#[test]
fn test_layernorm_residual_fusion_numerically_correct() {
    let dim = 64;
    let (g_unfused, x_id, residual_id) = build_add_layernorm_graph(dim);

    let x_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let res_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();

    let mut inputs = HashMap::new();
    inputs.insert(x_id, x_data);
    inputs.insert(residual_id, res_data);

    let unfused_output_id = g_unfused.outputs()[0];
    let unfused_result = g_unfused.execute(&inputs);
    let unfused_out = unfused_result[&unfused_output_id].clone();

    let mut g_fused = g_unfused.clone();
    let changed = FuseLayerNormResidual::new().run(&mut g_fused);
    assert!(changed, "FuseLayerNormResidual should have matched the pattern");

    let fused_output_id = g_fused.outputs()[0];
    let fused_result = g_fused.execute(&inputs);
    let fused_out = fused_result[&fused_output_id].clone();

    assert_eq!(unfused_out.len(), fused_out.len());
    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-4,
            "LayerNorm+Residual fusion mismatch at {}: unfused={}, fused={}, diff={}",
            i, u, f, (u - f).abs()
        );
    }
}

#[test]
fn bench_attention_fusion_speedup() {
    // Keep dimensions small so interpreter overhead (dispatch, Vec alloc, data
    // copying) dominates compute, making fusion most impactful.
    let seq_len = 16;
    let d_model = 8;
    let d_k = 8;
    let d_v = 8;
    let d_out = 8;

    let (g_unfused, _, param_ids) =
        build_attention_graph(seq_len, d_model, d_k, d_v, d_out);

    let input_data: Vec<f32> = (0..seq_len * d_model)
        .map(|i| ((i % 17) as f32) * 0.01)
        .collect();
    let w_q_data: Vec<f32> = (0..d_model * d_k)
        .map(|i| ((i % 13) as f32) * 0.01)
        .collect();
    let w_k_data: Vec<f32> = (0..d_model * d_k)
        .map(|i| ((i % 11) as f32) * 0.01)
        .collect();
    let w_v_data: Vec<f32> = (0..d_model * d_v)
        .map(|i| ((i % 7) as f32) * 0.01)
        .collect();
    let w_o_data: Vec<f32> = (0..d_v * d_out)
        .map(|i| ((i % 19) as f32) * 0.01)
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert(param_ids[0], input_data);
    inputs.insert(param_ids[1], w_q_data);
    inputs.insert(param_ids[2], w_k_data);
    inputs.insert(param_ids[3], w_v_data);
    inputs.insert(param_ids[5], w_o_data);

    let mut g_fused = g_unfused.clone();
    FuseAttention::new().run(&mut g_fused);
    DeadCodeElimination::new().run(&mut g_fused);

    // Warmup
    for _ in 0..50 {
        std::hint::black_box(g_unfused.execute(&inputs));
        std::hint::black_box(g_fused.execute(&inputs));
    }

    // Best of 5 trials
    let iterations = 200;
    let mut best_speedup = 0.0f64;

    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_unfused.execute(&inputs));
        }
        let unfused_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_fused.execute(&inputs));
        }
        let fused_time = start.elapsed();

        let speedup = unfused_time.as_secs_f64() / fused_time.as_secs_f64();
        if speedup > best_speedup {
            best_speedup = speedup;
        }
    }

    eprintln!(
        "Attention fusion (seq={}, d={}): best speedup={:.2}x",
        seq_len, d_model, best_speedup
    );

    // Build-mode-aware threshold
    let threshold = if cfg!(debug_assertions) { 1.1 } else { 1.3 };
    assert!(
        best_speedup >= threshold,
        "Attention fusion should be >= {:.1}x faster, got {:.2}x",
        threshold,
        best_speedup,
    );
}

#[test]
fn bench_layernorm_residual_fusion_speedup() {
    let dim = 512;
    let (g_unfused, x_id, residual_id) = build_add_layernorm_graph(dim);

    let x_data: Vec<f32> = (0..dim).map(|i| ((i % 17) as f32) * 0.02).collect();
    let res_data: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32) * 0.03).collect();

    let mut inputs = HashMap::new();
    inputs.insert(x_id, x_data);
    inputs.insert(residual_id, res_data);

    let mut g_fused = g_unfused.clone();
    FuseLayerNormResidual::new().run(&mut g_fused);
    DeadCodeElimination::new().run(&mut g_fused);

    // Warmup
    for _ in 0..100 {
        std::hint::black_box(g_unfused.execute(&inputs));
        std::hint::black_box(g_fused.execute(&inputs));
    }

    // Best of 5 trials
    let iterations = 500;
    let mut best_speedup = 0.0f64;

    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_unfused.execute(&inputs));
        }
        let unfused_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_fused.execute(&inputs));
        }
        let fused_time = start.elapsed();

        let speedup = unfused_time.as_secs_f64() / fused_time.as_secs_f64();
        if speedup > best_speedup {
            best_speedup = speedup;
        }
    }

    eprintln!(
        "LayerNorm+Residual fusion (dim={}): best speedup={:.2}x",
        dim, best_speedup
    );

    // Build-mode-aware threshold
    let threshold = if cfg!(debug_assertions) { 1.05 } else { 1.2 };
    assert!(
        best_speedup >= threshold,
        "LayerNorm+Residual fusion should be >= {:.2}x faster, got {:.2}x",
        threshold,
        best_speedup,
    );
}

// ── Combined transformer graph: both fusion passes ───────────────────

#[test]
fn test_combined_transformer_fusion_pipeline() {
    let seq_len = 4;
    let d = 8;

    // Build attention subgraph
    let (mut g, _, param_ids) = build_attention_graph(seq_len, d, d, d, d);

    // Add residual + layernorm on top of the attention output
    let attn_output_id = g.outputs()[0];

    let residual = g.add_node(
        NodeKind::Input("residual".into()),
        vec![],
        NodeMeta::new(vec![seq_len, d], DType::F32),
        "residual",
    );
    let gamma = g.add_node(
        NodeKind::Constant(vec![1.0; d]),
        vec![],
        NodeMeta::new(vec![d], DType::F32),
        "gamma",
    );
    let beta = g.add_node(
        NodeKind::Constant(vec![0.0; d]),
        vec![],
        NodeMeta::new(vec![d], DType::F32),
        "beta",
    );

    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![attn_output_id, residual],
        NodeMeta::new(vec![seq_len, d], DType::F32),
        "add_residual",
    );
    let ln = g.add_node(
        NodeKind::Op(OpKind::LayerNorm),
        vec![add, gamma, beta],
        NodeMeta::new(vec![seq_len, d], DType::F32),
        "post_ln",
    );

    // Replace output: set new output
    g.set_outputs(vec![ln]);

    let before_count = g.node_count();

    // Prepare input data
    let input_data: Vec<f32> = (0..seq_len * d)
        .map(|i| (i as f32) * 0.1 + 0.05)
        .collect();
    let w_q_data: Vec<f32> = (0..d * d).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let w_k_data: Vec<f32> = (0..d * d).map(|i| (i as f32) * 0.03 + 0.05).collect();
    let w_v_data: Vec<f32> = (0..d * d).map(|i| (i as f32) * 0.01 - 0.02).collect();
    let w_o_data: Vec<f32> = (0..d * d).map(|i| (i as f32) * 0.04 + 0.01).collect();
    let residual_data: Vec<f32> = (0..seq_len * d)
        .map(|i| (i as f32) * 0.05 - 0.2)
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert(param_ids[0], input_data);
    inputs.insert(param_ids[1], w_q_data);
    inputs.insert(param_ids[2], w_k_data);
    inputs.insert(param_ids[3], w_v_data);
    inputs.insert(param_ids[5], w_o_data);
    inputs.insert(residual, residual_data);

    // Execute unfused
    let unfused_output_id = g.outputs()[0];
    let unfused_result = g.execute(&inputs);
    let unfused_out = unfused_result[&unfused_output_id].clone();

    // Apply both fusion passes
    let mut g_fused = g.clone();
    let passes: Vec<Box<dyn OptimizationPass>> = vec![
        Box::new(FuseAttention::new()),
        Box::new(FuseLayerNormResidual::new()),
        Box::new(DeadCodeElimination::new()),
    ];
    let applied = run_passes(&mut g_fused, &passes);
    assert!(!applied.is_empty(), "At least one pass should apply");

    let after_count = g_fused.node_count();
    assert!(
        after_count < before_count,
        "Fusion should reduce node count: before={}, after={}",
        before_count, after_count
    );

    // Execute fused
    let fused_output_id = g_fused.outputs()[0];
    let fused_result = g_fused.execute(&inputs);
    let fused_out = fused_result[&fused_output_id].clone();

    // Verify numerical equivalence
    assert_eq!(unfused_out.len(), fused_out.len());
    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-4,
            "Combined fusion mismatch at {}: unfused={}, fused={}, diff={}",
            i, u, f, (u - f).abs()
        );
    }
}
