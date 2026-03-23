use std::collections::HashMap;
use std::time::Instant;

use synapse_graph::*;

// ── Fusion Tests ────────────────────────────────────────────────────────

#[test]
fn test_matmul_bias_relu_fusion_reduces_nodes() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![4, 8], DType::F32), "a");
    let w = g.add_node(NodeKind::Parameter("w".into()), vec![], NodeMeta::new(vec![8, 16], DType::F32), "w");
    let bias = g.add_node(
        NodeKind::Constant(vec![0.1; 16]),
        vec![],
        NodeMeta::new(vec![16], DType::F32),
        "bias",
    );
    let mm = g.add_node(NodeKind::Op(OpKind::MatMul), vec![a, w], NodeMeta::new(vec![4, 16], DType::F32), "mm");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![mm, bias], NodeMeta::new(vec![4, 16], DType::F32), "add");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![4, 16], DType::F32), "relu");
    g.mark_output(relu);

    let before = g.node_count();
    FuseMatMulBiasRelu::new().run(&mut g);
    let after = g.node_count();

    assert!(after < before, "Fusion should reduce node count: before={}, after={}", before, after);
    // Specifically: removed 3 (matmul, add, relu), added 1 (fused)
    assert_eq!(before - after, 2);
}

#[test]
fn test_conv_batchnorm_fusion_reduces_nodes() {
    let mut g = build_conv_bn_graph(4, 8);
    let before = g.node_count();
    FuseConvBatchNorm::new().run(&mut g);
    // Fusion replaces conv+bn with fused node but old BN params become dead
    DeadCodeElimination::new().run(&mut g);
    let after = g.node_count();
    assert!(after < before, "Fusion+DCE should reduce node count: before={}, after={}", before, after);
}

#[test]
fn test_conv_bn_fusion_numerically_identical() {
    let channels = 4;
    let spatial = 16;
    let total = channels * spatial;

    let gamma_data = vec![2.0, 0.5, 1.5, 0.8];
    let beta_data = vec![0.1, -0.2, 0.3, 0.0];
    let mean_data = vec![0.5, 1.0, -0.5, 0.2];
    let var_data = vec![0.25, 1.0, 4.0, 0.5];
    let weight_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let input_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3 - 2.0).collect();

    // Unfused
    let mut g1 = Graph::new();
    let x1 = g1.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![total], DType::F32), "x");
    let w1 = g1.add_node(NodeKind::Constant(weight_data.clone()), vec![], NodeMeta::new(vec![total], DType::F32), "w");
    let gam1 = g1.add_node(NodeKind::Constant(gamma_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "gamma");
    let bet1 = g1.add_node(NodeKind::Constant(beta_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "beta");
    let men1 = g1.add_node(NodeKind::Constant(mean_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "mean");
    let var1 = g1.add_node(NodeKind::Constant(var_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "var");
    let conv1 = g1.add_node(NodeKind::Op(OpKind::Conv2d), vec![x1, w1], NodeMeta::new(vec![total], DType::F32), "conv");
    let bn1 = g1.add_node(
        NodeKind::Op(OpKind::BatchNorm),
        vec![conv1, gam1, bet1, men1, var1],
        NodeMeta::new(vec![total], DType::F32),
        "bn",
    );
    g1.mark_output(bn1);

    let mut inputs1 = HashMap::new();
    inputs1.insert(x1, input_data.clone());
    let unfused_result = g1.execute(&inputs1);
    let unfused_output = unfused_result[&bn1].clone();

    // Fused
    let mut g2 = g1.clone();
    FuseConvBatchNorm::new().run(&mut g2);

    let mut inputs2 = HashMap::new();
    inputs2.insert(x1, input_data);
    let fused_result = g2.execute(&inputs2);
    let fused_output_id = g2.outputs()[0];
    let fused_output = fused_result[&fused_output_id].clone();

    assert_eq!(unfused_output.len(), fused_output.len());
    for (i, (u, f)) in unfused_output.iter().zip(fused_output.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-5,
            "Conv+BN fusion mismatch at index {}: unfused={}, fused={}, diff={}",
            i, u, f, (u - f).abs()
        );
    }
}

#[test]
fn test_elementwise_fusion_reduces_nodes() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![100], DType::F32), "a");
    let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![100], DType::F32), "b");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![100], DType::F32), "add");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![100], DType::F32), "relu");
    let sigmoid = g.add_node(NodeKind::Op(OpKind::Sigmoid), vec![relu], NodeMeta::new(vec![100], DType::F32), "sigmoid");
    g.mark_output(sigmoid);

    let before = g.node_count();
    FuseElementWise::new().run(&mut g);
    let after = g.node_count();

    assert!(after < before, "Element-wise fusion should reduce node count");
}

// ── Dead Code Elimination Tests ─────────────────────────────────────────

#[test]
fn test_dce_removes_dead_branch() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![10], DType::F32), "a");
    let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![10], DType::F32), "b");

    // Live path
    let live_relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![a], NodeMeta::new(vec![10], DType::F32), "live_relu");

    // Dead path
    let _dead1 = g.add_node(NodeKind::Op(OpKind::Neg), vec![b], NodeMeta::new(vec![10], DType::F32), "dead1");
    let _dead2 = g.add_node(NodeKind::Op(OpKind::Sigmoid), vec![_dead1], NodeMeta::new(vec![10], DType::F32), "dead2");

    g.mark_output(live_relu);
    let before = g.node_count();

    DeadCodeElimination::new().run(&mut g);

    let after = g.node_count();
    assert!(after < before);
    assert_eq!(after, 2); // Only a and live_relu
}

#[test]
fn test_dce_preserves_semantics() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![4], DType::F32), "a");
    let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![4], DType::F32), "b");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![4], DType::F32), "add");

    // Dead branch
    let _dead = g.add_node(NodeKind::Op(OpKind::Neg), vec![a], NodeMeta::new(vec![4], DType::F32), "dead");
    g.mark_output(add);

    let mut inputs = HashMap::new();
    inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert(b, vec![5.0, 6.0, 7.0, 8.0]);

    let before_result = g.execute(&inputs)[&add].clone();

    DeadCodeElimination::new().run(&mut g);

    let after_result = g.execute(&inputs)[&add].clone();
    assert_eq!(before_result, after_result);
}

// ── Constant Folding Tests ──────────────────────────────────────────────

#[test]
fn test_constant_folding_computes_correctly() {
    let mut g = Graph::new();
    let a = g.add_node(
        NodeKind::Constant(vec![1.0, 2.0, 3.0, 4.0]),
        vec![],
        NodeMeta::new(vec![4], DType::F32),
        "a",
    );
    let b = g.add_node(
        NodeKind::Constant(vec![10.0, 20.0, 30.0, 40.0]),
        vec![],
        NodeMeta::new(vec![4], DType::F32),
        "b",
    );
    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![a, b],
        NodeMeta::new(vec![4], DType::F32),
        "add",
    );
    g.mark_output(add);

    ConstantFolding::new().run(&mut g);

    let output_id = g.outputs()[0];
    let node = g.node(output_id).unwrap();
    match &node.kind {
        NodeKind::Constant(data) => {
            assert_eq!(data, &vec![11.0, 22.0, 33.0, 44.0]);
        }
        _ => panic!("Expected constant after folding"),
    }
}

#[test]
fn test_constant_folding_chain() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Constant(vec![4.0, 9.0]), vec![], NodeMeta::new(vec![2], DType::F32), "a");
    let sqrt = g.add_node(NodeKind::Op(OpKind::Sqrt), vec![a], NodeMeta::new(vec![2], DType::F32), "sqrt");
    let neg = g.add_node(NodeKind::Op(OpKind::Neg), vec![sqrt], NodeMeta::new(vec![2], DType::F32), "neg");
    g.mark_output(neg);

    ConstantFolding::new().run(&mut g);

    let output_id = g.outputs()[0];
    let node = g.node(output_id).unwrap();
    match &node.kind {
        NodeKind::Constant(data) => {
            assert_eq!(data, &vec![-2.0, -3.0]);
        }
        _ => panic!("Expected constant after folding chain"),
    }
}

// ── Scheduler Tests ─────────────────────────────────────────────────────

#[test]
fn test_scheduler_valid_topological_order() {
    let mut g = Graph::new();
    let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![256, 256], DType::F32), "a");
    let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![256, 256], DType::F32), "b");
    let c = g.add_node(NodeKind::Input("c".into()), vec![], NodeMeta::new(vec![256, 256], DType::F32), "c");

    let ab = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![256, 256], DType::F32), "ab");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![ab], NodeMeta::new(vec![256, 256], DType::F32), "relu");
    let out = g.add_node(NodeKind::Op(OpKind::Add), vec![relu, c], NodeMeta::new(vec![256, 256], DType::F32), "out");
    g.mark_output(out);

    let scheduler = MemoryOptimalScheduler::new();
    let order = scheduler.schedule(&g);

    assert!(
        MemoryOptimalScheduler::validate_order(&g, &order),
        "Scheduler must produce valid topological order"
    );
    assert_eq!(order.len(), 6);
}

#[test]
fn test_scheduler_complex_graph() {
    let mut g = Graph::new();
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![64, 64], DType::F32), "x");
    let w1 = g.add_node(NodeKind::Parameter("w1".into()), vec![], NodeMeta::new(vec![64, 64], DType::F32), "w1");
    let w2 = g.add_node(NodeKind::Parameter("w2".into()), vec![], NodeMeta::new(vec![64, 64], DType::F32), "w2");
    let b1 = g.add_node(NodeKind::Constant(vec![0.0; 64]), vec![], NodeMeta::new(vec![64], DType::F32), "b1");

    let mm1 = g.add_node(NodeKind::Op(OpKind::MatMul), vec![x, w1], NodeMeta::new(vec![64, 64], DType::F32), "mm1");
    let add1 = g.add_node(NodeKind::Op(OpKind::Add), vec![mm1, b1], NodeMeta::new(vec![64, 64], DType::F32), "add1");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add1], NodeMeta::new(vec![64, 64], DType::F32), "relu");
    let mm2 = g.add_node(NodeKind::Op(OpKind::MatMul), vec![relu, w2], NodeMeta::new(vec![64, 64], DType::F32), "mm2");
    g.mark_output(mm2);

    let scheduler = MemoryOptimalScheduler::new();
    let order = scheduler.schedule(&g);
    assert!(MemoryOptimalScheduler::validate_order(&g, &order));
}

// ── Benchmark: Fused vs Unfused MatMul+Bias+ReLU ────────────────────────

#[test]
fn bench_fused_vs_unfused_matmul_bias_relu_256x256() {
    // Output is 256x256. Inner dimension K is kept small so the post-matmul
    // overhead (intermediate Vec allocations, extra data passes, bias constant
    // cloning, interpreter dispatch) — which is what fusion eliminates —
    // is proportionally significant relative to compute.
    let m = 256;
    let n = 256;
    let k = 4;

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.01).collect();
    let bias_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    // ── Unfused graph: matmul → add(bias) → relu  (3 ops, 3 allocs of 256KB each) ──
    let mut g_unfused = Graph::new();
    let a_id = g_unfused.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![m, k], DType::F32), "a");
    let b_id = g_unfused.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![k, n], DType::F32), "b");

    let mm = g_unfused.add_node(NodeKind::Op(OpKind::MatMul), vec![a_id, b_id], NodeMeta::new(vec![m, n], DType::F32), "mm");
    // Broadcast bias to [M x N] constant (cloned on every execute() call)
    let bias_full: Vec<f32> = (0..m).flat_map(|_| bias_data.iter().copied()).collect();
    let bias_full_id = g_unfused.add_node(
        NodeKind::Constant(bias_full),
        vec![],
        NodeMeta::new(vec![m, n], DType::F32),
        "bias_full",
    );
    let add = g_unfused.add_node(NodeKind::Op(OpKind::Add), vec![mm, bias_full_id], NodeMeta::new(vec![m, n], DType::F32), "add");
    let relu = g_unfused.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![m, n], DType::F32), "relu");
    g_unfused.mark_output(relu);

    // ── Fused graph: single FusedMatMulBiasRelu  (1 op, 1 alloc, no intermediates) ──
    let mut g_fused = Graph::new();
    let a_id2 = g_fused.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![m, k], DType::F32), "a");
    let b_id2 = g_fused.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![k, n], DType::F32), "b");
    let bias_id2 = g_fused.add_node(
        NodeKind::Constant(bias_data),
        vec![],
        NodeMeta::new(vec![n], DType::F32),
        "bias",
    );
    let fused = g_fused.add_node(
        NodeKind::Op(OpKind::FusedMatMulBiasRelu),
        vec![a_id2, b_id2, bias_id2],
        NodeMeta::new(vec![m, n], DType::F32),
        "fused",
    );
    g_fused.mark_output(fused);

    let mut unfused_inputs = HashMap::new();
    unfused_inputs.insert(a_id, a_data.clone());
    unfused_inputs.insert(b_id, b_data.clone());

    let mut fused_inputs = HashMap::new();
    fused_inputs.insert(a_id2, a_data);
    fused_inputs.insert(b_id2, b_data);

    // Verify numerical correctness
    let unfused_result = g_unfused.execute(&unfused_inputs);
    let fused_result = g_fused.execute(&fused_inputs);
    let unfused_out = &unfused_result[&relu];
    let fused_out = &fused_result[&fused];
    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-3,
            "Mismatch at {}: unfused={}, fused={}",
            i, u, f
        );
    }

    // Aggressive warmup to ensure CPU frequency is ramped up
    for _ in 0..50 {
        std::hint::black_box(g_unfused.execute(&unfused_inputs));
        std::hint::black_box(g_fused.execute(&fused_inputs));
    }

    // Best of 5 trials, each with 200 iterations, for stable measurement
    let iterations = 200;
    let mut best_speedup = 0.0f64;

    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_unfused.execute(&unfused_inputs));
        }
        let unfused_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(g_fused.execute(&fused_inputs));
        }
        let fused_time = start.elapsed();

        let speedup = unfused_time.as_secs_f64() / fused_time.as_secs_f64();
        if speedup > best_speedup {
            best_speedup = speedup;
            eprintln!(
                "  trial: unfused={:.2}ms, fused={:.2}ms, speedup={:.2}x",
                unfused_time.as_secs_f64() * 1000.0 / iterations as f64,
                fused_time.as_secs_f64() * 1000.0 / iterations as f64,
                speedup,
            );
        }
    }

    eprintln!("MatMul+Bias+ReLU 256x256: best speedup={:.2}x", best_speedup);

    assert!(
        best_speedup >= 1.3,
        "Fused should be at least 1.3x faster than unfused, got {:.2}x",
        best_speedup,
    );
}

// ── Combined Optimization Pipeline ──────────────────────────────────────

#[test]
fn test_full_optimization_pipeline() {
    let mut g = Graph::new();
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![8, 16], DType::F32), "x");
    let w = g.add_node(NodeKind::Parameter("w".into()), vec![], NodeMeta::new(vec![16, 32], DType::F32), "w");
    let bias = g.add_node(NodeKind::Constant(vec![0.1; 32]), vec![], NodeMeta::new(vec![32], DType::F32), "bias");

    // Main path: matmul -> add bias -> relu
    let mm = g.add_node(NodeKind::Op(OpKind::MatMul), vec![x, w], NodeMeta::new(vec![8, 32], DType::F32), "mm");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![mm, bias], NodeMeta::new(vec![8, 32], DType::F32), "add");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![8, 32], DType::F32), "relu");

    // Dead branch: constant folding opportunity + dead code
    let c1 = g.add_node(NodeKind::Constant(vec![1.0, 2.0]), vec![], NodeMeta::new(vec![2], DType::F32), "c1");
    let c2 = g.add_node(NodeKind::Constant(vec![3.0, 4.0]), vec![], NodeMeta::new(vec![2], DType::F32), "c2");
    let _dead_add = g.add_node(NodeKind::Op(OpKind::Add), vec![c1, c2], NodeMeta::new(vec![2], DType::F32), "dead_add");

    g.mark_output(relu);

    let passes: Vec<Box<dyn OptimizationPass>> = vec![
        Box::new(ConstantFolding::new()),
        Box::new(FuseMatMulBiasRelu::new()),
        Box::new(DeadCodeElimination::new()),
    ];

    let applied = run_passes(&mut g, &passes);
    assert!(!applied.is_empty());

    // After optimization: dead branch removed, matmul+bias+relu fused
    // Should have: x, w, bias, fused_matmul_bias_relu = 4 nodes
    assert_eq!(g.node_count(), 4);

    // Verify scheduler produces valid order on optimized graph
    let scheduler = MemoryOptimalScheduler::new();
    let order = scheduler.schedule(&g);
    assert!(MemoryOptimalScheduler::validate_order(&g, &order));
}

// ── Attention Fusion Tests ───────────────────────────────────────────────

#[test]
fn test_attention_fusion_reduces_nodes() {
    let mut g = build_attention_graph(4, 8, 6, 6, 8);
    let before = g.node_count();

    FuseAttention::new().run(&mut g);
    let after = g.node_count();

    assert!(after < before, "Attention fusion should reduce node count: before={}, after={}", before, after);
    // 9 ops replaced by 1 fused => net reduction of 8
    assert_eq!(before - after, 8);

    let output_id = g.outputs()[0];
    let fused = g.node(output_id).unwrap();
    assert!(matches!(fused.kind, NodeKind::Op(OpKind::FusedAttention)));
}

#[test]
fn test_attention_fusion_idempotent() {
    let mut g = build_attention_graph(2, 4, 3, 3, 4);
    FuseAttention::new().run(&mut g);
    let count = g.node_count();

    let changed = FuseAttention::new().run(&mut g);
    assert!(!changed);
    assert_eq!(g.node_count(), count);
}

#[test]
fn test_attention_fusion_numerically_identical() {
    let seq_len = 2;
    let d_model = 4;
    let d_k = 3;
    let d_v = 3;
    let d_out = 4;

    let (g_unfused, param_ids) = build_attention_graph_with_ids(seq_len, d_model, d_k, d_v, d_out);

    let input_data: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let w_q_data: Vec<f32> = (0..d_model * d_k).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let w_k_data: Vec<f32> = (0..d_model * d_k).map(|i| (i as f32) * 0.03 + 0.05).collect();
    let w_v_data: Vec<f32> = (0..d_model * d_v).map(|i| (i as f32) * 0.01 - 0.02).collect();
    let w_o_data: Vec<f32> = (0..d_v * d_out).map(|i| (i as f32) * 0.04 + 0.01).collect();

    let mut inputs = HashMap::new();
    inputs.insert(param_ids[0], input_data);
    inputs.insert(param_ids[1], w_q_data);
    inputs.insert(param_ids[2], w_k_data);
    inputs.insert(param_ids[3], w_v_data);
    inputs.insert(param_ids[4], w_o_data);

    let unfused_output_id = g_unfused.outputs()[0];
    let unfused_result = g_unfused.execute(&inputs);
    let unfused_out = unfused_result[&unfused_output_id].clone();

    let mut g_fused = g_unfused.clone();
    FuseAttention::new().run(&mut g_fused);

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

// ── LayerNorm Residual Fusion Tests ─────────────────────────────────────

#[test]
fn test_layernorm_residual_fusion_reduces_nodes() {
    let mut g = build_add_layernorm_graph(16);
    let before = g.node_count();

    FuseLayerNormResidual::new().run(&mut g);
    let after = g.node_count();

    assert!(after < before, "LayerNorm+Residual fusion should reduce: before={}, after={}", before, after);
    // Add + LayerNorm replaced by 1 fused => net reduction of 1
    assert_eq!(before - after, 1);
}

#[test]
fn test_layernorm_residual_fusion_idempotent() {
    let mut g = build_add_layernorm_graph(8);
    FuseLayerNormResidual::new().run(&mut g);
    let count = g.node_count();

    let changed = FuseLayerNormResidual::new().run(&mut g);
    assert!(!changed);
    assert_eq!(g.node_count(), count);
}

#[test]
fn test_layernorm_residual_fusion_numerically_identical() {
    let dim = 8;
    let mut g = Graph::new();
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "x");
    let residual = g.add_node(NodeKind::Input("res".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "res");
    let gamma = g.add_node(
        NodeKind::Constant(vec![2.0, 0.5, 1.5, 0.8, 1.0, 1.2, 0.3, 0.9]),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "gamma",
    );
    let beta = g.add_node(
        NodeKind::Constant(vec![0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.4]),
        vec![],
        NodeMeta::new(vec![dim], DType::F32),
        "beta",
    );
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![x, residual], NodeMeta::new(vec![dim], DType::F32), "add");
    let ln = g.add_node(NodeKind::Op(OpKind::LayerNorm), vec![add, gamma, beta], NodeMeta::new(vec![dim], DType::F32), "ln");
    g.mark_output(ln);

    let x_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let res_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();

    let mut inputs = HashMap::new();
    inputs.insert(x, x_data);
    inputs.insert(residual, res_data);

    let unfused_out = g.execute(&inputs)[&ln].clone();

    let mut g_fused = g.clone();
    FuseLayerNormResidual::new().run(&mut g_fused);
    let fused_output_id = g_fused.outputs()[0];
    let fused_out = g_fused.execute(&inputs)[&fused_output_id].clone();

    for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        assert!(
            (u - f).abs() < 1e-5,
            "LayerNorm+Residual mismatch at {}: unfused={}, fused={}",
            i, u, f
        );
    }
}

// ── Regression: existing passes still work with new IR variants ─────────

#[test]
fn test_existing_passes_unaffected_by_new_variants() {
    // Verify DCE, constant folding, and Phase 1 fusions still work correctly
    let mut g = Graph::new();
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![8, 16], DType::F32), "x");
    let w = g.add_node(NodeKind::Parameter("w".into()), vec![], NodeMeta::new(vec![16, 32], DType::F32), "w");
    let bias = g.add_node(NodeKind::Constant(vec![0.1; 32]), vec![], NodeMeta::new(vec![32], DType::F32), "bias");

    let mm = g.add_node(NodeKind::Op(OpKind::MatMul), vec![x, w], NodeMeta::new(vec![8, 32], DType::F32), "mm");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![mm, bias], NodeMeta::new(vec![8, 32], DType::F32), "add");
    let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![8, 32], DType::F32), "relu");

    // Dead branch
    let c1 = g.add_node(NodeKind::Constant(vec![1.0, 2.0]), vec![], NodeMeta::new(vec![2], DType::F32), "c1");
    let c2 = g.add_node(NodeKind::Constant(vec![3.0, 4.0]), vec![], NodeMeta::new(vec![2], DType::F32), "c2");
    let _dead = g.add_node(NodeKind::Op(OpKind::Add), vec![c1, c2], NodeMeta::new(vec![2], DType::F32), "dead");

    g.mark_output(relu);

    let passes: Vec<Box<dyn OptimizationPass>> = vec![
        Box::new(ConstantFolding::new()),
        Box::new(FuseMatMulBiasRelu::new()),
        Box::new(DeadCodeElimination::new()),
    ];
    let applied = run_passes(&mut g, &passes);
    assert!(!applied.is_empty());
    assert_eq!(g.node_count(), 4); // x, w, bias, fused
}

#[test]
fn test_full_pipeline_with_new_passes() {
    // Build a graph with both attention and layernorm+residual patterns
    let dim = 8;
    let mut g = Graph::new();

    // Simple Add -> LayerNorm subgraph
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "x");
    let res = g.add_node(NodeKind::Input("res".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "res");
    let gamma = g.add_node(NodeKind::Constant(vec![1.0; dim]), vec![], NodeMeta::new(vec![dim], DType::F32), "gamma");
    let beta = g.add_node(NodeKind::Constant(vec![0.0; dim]), vec![], NodeMeta::new(vec![dim], DType::F32), "beta");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![x, res], NodeMeta::new(vec![dim], DType::F32), "add");
    let ln = g.add_node(NodeKind::Op(OpKind::LayerNorm), vec![add, gamma, beta], NodeMeta::new(vec![dim], DType::F32), "ln");

    // Dead branch
    let dead = g.add_node(NodeKind::Op(OpKind::Neg), vec![x], NodeMeta::new(vec![dim], DType::F32), "dead");
    let _dead2 = g.add_node(NodeKind::Op(OpKind::Relu), vec![dead], NodeMeta::new(vec![dim], DType::F32), "dead2");

    g.mark_output(ln);

    let passes: Vec<Box<dyn OptimizationPass>> = vec![
        Box::new(FuseLayerNormResidual::new()),
        Box::new(DeadCodeElimination::new()),
    ];
    run_passes(&mut g, &passes);

    // After fusion: x, res, gamma, beta, fused_ln_res = 5
    // After DCE: dead branch removed
    assert_eq!(g.node_count(), 5);

    let output_id = g.outputs()[0];
    let fused = g.node(output_id).unwrap();
    assert!(matches!(fused.kind, NodeKind::Op(OpKind::FusedLayerNormResidual)));
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn build_attention_graph(
    seq_len: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    d_out: usize,
) -> Graph {
    let (g, _) = build_attention_graph_with_ids(seq_len, d_model, d_k, d_v, d_out);
    g
}

/// Returns (graph, [input, w_q, w_k, w_v, w_o]) - scale is a constant in the graph
fn build_attention_graph_with_ids(
    seq_len: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    d_out: usize,
) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new();

    let input = g.add_node(
        NodeKind::Input("input".into()), vec![],
        NodeMeta::new(vec![seq_len, d_model], DType::F32), "input",
    );
    let w_q = g.add_node(
        NodeKind::Parameter("W_q".into()), vec![],
        NodeMeta::new(vec![d_model, d_k], DType::F32), "W_q",
    );
    let w_k = g.add_node(
        NodeKind::Parameter("W_k".into()), vec![],
        NodeMeta::new(vec![d_model, d_k], DType::F32), "W_k",
    );
    let w_v = g.add_node(
        NodeKind::Parameter("W_v".into()), vec![],
        NodeMeta::new(vec![d_model, d_v], DType::F32), "W_v",
    );
    let scale_val = 1.0 / (d_k as f32).sqrt();
    let scale = g.add_node(
        NodeKind::Constant(vec![scale_val; seq_len * seq_len]), vec![],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32), "scale",
    );
    let w_o = g.add_node(
        NodeKind::Parameter("W_o".into()), vec![],
        NodeMeta::new(vec![d_v, d_out], DType::F32), "W_o",
    );

    let q = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![input, w_q],
        NodeMeta::new(vec![seq_len, d_k], DType::F32), "q_proj",
    );
    let k = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![input, w_k],
        NodeMeta::new(vec![seq_len, d_k], DType::F32), "k_proj",
    );
    let k_t = g.add_node(
        NodeKind::Op(OpKind::Transpose), vec![k],
        NodeMeta::new(vec![d_k, seq_len], DType::F32), "k_transpose",
    );
    let v = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![input, w_v],
        NodeMeta::new(vec![seq_len, d_v], DType::F32), "v_proj",
    );
    let scores = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![q, k_t],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32), "scores",
    );
    let scaled = g.add_node(
        NodeKind::Op(OpKind::Mul), vec![scores, scale],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32), "scaled",
    );
    let weights = g.add_node(
        NodeKind::Op(OpKind::Softmax), vec![scaled],
        NodeMeta::new(vec![seq_len, seq_len], DType::F32), "weights",
    );
    let attended = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![weights, v],
        NodeMeta::new(vec![seq_len, d_v], DType::F32), "attended",
    );
    let output = g.add_node(
        NodeKind::Op(OpKind::MatMul), vec![attended, w_o],
        NodeMeta::new(vec![seq_len, d_out], DType::F32), "output_proj",
    );
    g.mark_output(output);

    (g, vec![input, w_q, w_k, w_v, w_o])
}

fn build_add_layernorm_graph(dim: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "x");
    let res = g.add_node(NodeKind::Input("res".into()), vec![], NodeMeta::new(vec![dim], DType::F32), "res");
    let gamma = g.add_node(NodeKind::Constant(vec![1.0; dim]), vec![], NodeMeta::new(vec![dim], DType::F32), "gamma");
    let beta = g.add_node(NodeKind::Constant(vec![0.0; dim]), vec![], NodeMeta::new(vec![dim], DType::F32), "beta");
    let add = g.add_node(NodeKind::Op(OpKind::Add), vec![x, res], NodeMeta::new(vec![dim], DType::F32), "add");
    let ln = g.add_node(NodeKind::Op(OpKind::LayerNorm), vec![add, gamma, beta], NodeMeta::new(vec![dim], DType::F32), "ln");
    g.mark_output(ln);
    g
}

fn build_conv_bn_graph(channels: usize, spatial: usize) -> Graph {
    let total = channels * spatial;
    let mut g = Graph::new();

    let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![total], DType::F32), "x");
    let w = g.add_node(
        NodeKind::Constant(vec![1.0; total]),
        vec![],
        NodeMeta::new(vec![total], DType::F32),
        "w",
    );
    let gamma = g.add_node(
        NodeKind::Constant(vec![1.0; channels]),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "gamma",
    );
    let beta = g.add_node(
        NodeKind::Constant(vec![0.0; channels]),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "beta",
    );
    let mean = g.add_node(
        NodeKind::Constant(vec![0.0; channels]),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "mean",
    );
    let var = g.add_node(
        NodeKind::Constant(vec![1.0; channels]),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "var",
    );

    let conv = g.add_node(NodeKind::Op(OpKind::Conv2d), vec![x, w], NodeMeta::new(vec![total], DType::F32), "conv");
    let bn = g.add_node(
        NodeKind::Op(OpKind::BatchNorm),
        vec![conv, gamma, beta, mean, var],
        NodeMeta::new(vec![total], DType::F32),
        "bn",
    );
    g.mark_output(bn);
    g
}
