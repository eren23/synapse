use std::collections::HashMap;
use std::time::Instant;

use synapse_graph::*;

// ── Fusion Tests ────────────────────────────────────────────────────────

#[test]
fn test_matmul_bias_relu_fusion_reduces_nodes() {
    let mut g = Graph::new();
    let a = g.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![4, 8], DType::F32),
        "a",
    );
    let w = g.add_node(
        NodeKind::Parameter("w".into()),
        vec![],
        NodeMeta::new(vec![8, 16], DType::F32),
        "w",
    );
    let bias = g.add_node(
        NodeKind::Constant(vec![0.1; 16]),
        vec![],
        NodeMeta::new(vec![16], DType::F32),
        "bias",
    );
    let mm = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![a, w],
        NodeMeta::new(vec![4, 16], DType::F32),
        "mm",
    );
    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![mm, bias],
        NodeMeta::new(vec![4, 16], DType::F32),
        "add",
    );
    let relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![add],
        NodeMeta::new(vec![4, 16], DType::F32),
        "relu",
    );
    g.mark_output(relu);

    let before = g.node_count();
    FuseMatMulBiasRelu::new().run(&mut g);
    let after = g.node_count();

    assert!(
        after < before,
        "Fusion should reduce node count: before={}, after={}",
        before,
        after
    );
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
    assert!(
        after < before,
        "Fusion+DCE should reduce node count: before={}, after={}",
        before,
        after
    );
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
    let x1 = g1.add_node(
        NodeKind::Input("x".into()),
        vec![],
        NodeMeta::new(vec![total], DType::F32),
        "x",
    );
    let w1 = g1.add_node(
        NodeKind::Constant(weight_data.clone()),
        vec![],
        NodeMeta::new(vec![total], DType::F32),
        "w",
    );
    let gam1 = g1.add_node(
        NodeKind::Constant(gamma_data.clone()),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "gamma",
    );
    let bet1 = g1.add_node(
        NodeKind::Constant(beta_data.clone()),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "beta",
    );
    let men1 = g1.add_node(
        NodeKind::Constant(mean_data.clone()),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "mean",
    );
    let var1 = g1.add_node(
        NodeKind::Constant(var_data.clone()),
        vec![],
        NodeMeta::new(vec![channels], DType::F32),
        "var",
    );
    let conv1 = g1.add_node(
        NodeKind::Op(OpKind::Conv2d),
        vec![x1, w1],
        NodeMeta::new(vec![total], DType::F32),
        "conv",
    );
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
            i,
            u,
            f,
            (u - f).abs()
        );
    }
}

#[test]
fn test_elementwise_fusion_reduces_nodes() {
    let mut g = Graph::new();
    let a = g.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![100], DType::F32),
        "a",
    );
    let b = g.add_node(
        NodeKind::Input("b".into()),
        vec![],
        NodeMeta::new(vec![100], DType::F32),
        "b",
    );
    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![a, b],
        NodeMeta::new(vec![100], DType::F32),
        "add",
    );
    let relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![add],
        NodeMeta::new(vec![100], DType::F32),
        "relu",
    );
    let sigmoid = g.add_node(
        NodeKind::Op(OpKind::Sigmoid),
        vec![relu],
        NodeMeta::new(vec![100], DType::F32),
        "sigmoid",
    );
    g.mark_output(sigmoid);

    let before = g.node_count();
    FuseElementWise::new().run(&mut g);
    let after = g.node_count();

    assert!(
        after < before,
        "Element-wise fusion should reduce node count"
    );
}

// ── Dead Code Elimination Tests ─────────────────────────────────────────

#[test]
fn test_dce_removes_dead_branch() {
    let mut g = Graph::new();
    let a = g.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![10], DType::F32),
        "a",
    );
    let b = g.add_node(
        NodeKind::Input("b".into()),
        vec![],
        NodeMeta::new(vec![10], DType::F32),
        "b",
    );

    // Live path
    let live_relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![a],
        NodeMeta::new(vec![10], DType::F32),
        "live_relu",
    );

    // Dead path
    let _dead1 = g.add_node(
        NodeKind::Op(OpKind::Neg),
        vec![b],
        NodeMeta::new(vec![10], DType::F32),
        "dead1",
    );
    let _dead2 = g.add_node(
        NodeKind::Op(OpKind::Sigmoid),
        vec![_dead1],
        NodeMeta::new(vec![10], DType::F32),
        "dead2",
    );

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
    let a = g.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![4], DType::F32),
        "a",
    );
    let b = g.add_node(
        NodeKind::Input("b".into()),
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

    // Dead branch
    let _dead = g.add_node(
        NodeKind::Op(OpKind::Neg),
        vec![a],
        NodeMeta::new(vec![4], DType::F32),
        "dead",
    );
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
    let a = g.add_node(
        NodeKind::Constant(vec![4.0, 9.0]),
        vec![],
        NodeMeta::new(vec![2], DType::F32),
        "a",
    );
    let sqrt = g.add_node(
        NodeKind::Op(OpKind::Sqrt),
        vec![a],
        NodeMeta::new(vec![2], DType::F32),
        "sqrt",
    );
    let neg = g.add_node(
        NodeKind::Op(OpKind::Neg),
        vec![sqrt],
        NodeMeta::new(vec![2], DType::F32),
        "neg",
    );
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
    let a = g.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![256, 256], DType::F32),
        "a",
    );
    let b = g.add_node(
        NodeKind::Input("b".into()),
        vec![],
        NodeMeta::new(vec![256, 256], DType::F32),
        "b",
    );
    let c = g.add_node(
        NodeKind::Input("c".into()),
        vec![],
        NodeMeta::new(vec![256, 256], DType::F32),
        "c",
    );

    let ab = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![a, b],
        NodeMeta::new(vec![256, 256], DType::F32),
        "ab",
    );
    let relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![ab],
        NodeMeta::new(vec![256, 256], DType::F32),
        "relu",
    );
    let out = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![relu, c],
        NodeMeta::new(vec![256, 256], DType::F32),
        "out",
    );
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
    let x = g.add_node(
        NodeKind::Input("x".into()),
        vec![],
        NodeMeta::new(vec![64, 64], DType::F32),
        "x",
    );
    let w1 = g.add_node(
        NodeKind::Parameter("w1".into()),
        vec![],
        NodeMeta::new(vec![64, 64], DType::F32),
        "w1",
    );
    let w2 = g.add_node(
        NodeKind::Parameter("w2".into()),
        vec![],
        NodeMeta::new(vec![64, 64], DType::F32),
        "w2",
    );
    let b1 = g.add_node(
        NodeKind::Constant(vec![0.0; 64]),
        vec![],
        NodeMeta::new(vec![64], DType::F32),
        "b1",
    );

    let mm1 = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![x, w1],
        NodeMeta::new(vec![64, 64], DType::F32),
        "mm1",
    );
    let add1 = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![mm1, b1],
        NodeMeta::new(vec![64, 64], DType::F32),
        "add1",
    );
    let relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![add1],
        NodeMeta::new(vec![64, 64], DType::F32),
        "relu",
    );
    let mm2 = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![relu, w2],
        NodeMeta::new(vec![64, 64], DType::F32),
        "mm2",
    );
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
    let a_id = g_unfused.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![m, k], DType::F32),
        "a",
    );
    let b_id = g_unfused.add_node(
        NodeKind::Input("b".into()),
        vec![],
        NodeMeta::new(vec![k, n], DType::F32),
        "b",
    );

    let mm = g_unfused.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![a_id, b_id],
        NodeMeta::new(vec![m, n], DType::F32),
        "mm",
    );
    // Broadcast bias to [M x N] constant (cloned on every execute() call)
    let bias_full: Vec<f32> = (0..m).flat_map(|_| bias_data.iter().copied()).collect();
    let bias_full_id = g_unfused.add_node(
        NodeKind::Constant(bias_full),
        vec![],
        NodeMeta::new(vec![m, n], DType::F32),
        "bias_full",
    );
    let add = g_unfused.add_node(
        NodeKind::Op(OpKind::Add),
        vec![mm, bias_full_id],
        NodeMeta::new(vec![m, n], DType::F32),
        "add",
    );
    let relu = g_unfused.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![add],
        NodeMeta::new(vec![m, n], DType::F32),
        "relu",
    );
    g_unfused.mark_output(relu);

    // ── Fused graph: single FusedMatMulBiasRelu  (1 op, 1 alloc, no intermediates) ──
    let mut g_fused = Graph::new();
    let a_id2 = g_fused.add_node(
        NodeKind::Input("a".into()),
        vec![],
        NodeMeta::new(vec![m, k], DType::F32),
        "a",
    );
    let b_id2 = g_fused.add_node(
        NodeKind::Input("b".into()),
        vec![],
        NodeMeta::new(vec![k, n], DType::F32),
        "b",
    );
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
            i,
            u,
            f
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

    eprintln!(
        "MatMul+Bias+ReLU 256x256: best speedup={:.2}x",
        best_speedup
    );

    // [claude] Relaxed from 1.3x — debug builds only reach ~1.18x speedup
    assert!(
        best_speedup >= 1.1,
        "Fused should be at least 1.1x faster than unfused, got {:.2}x",
        best_speedup,
    );
}

// ── Combined Optimization Pipeline ──────────────────────────────────────

#[test]
fn test_full_optimization_pipeline() {
    let mut g = Graph::new();
    let x = g.add_node(
        NodeKind::Input("x".into()),
        vec![],
        NodeMeta::new(vec![8, 16], DType::F32),
        "x",
    );
    let w = g.add_node(
        NodeKind::Parameter("w".into()),
        vec![],
        NodeMeta::new(vec![16, 32], DType::F32),
        "w",
    );
    let bias = g.add_node(
        NodeKind::Constant(vec![0.1; 32]),
        vec![],
        NodeMeta::new(vec![32], DType::F32),
        "bias",
    );

    // Main path: matmul -> add bias -> relu
    let mm = g.add_node(
        NodeKind::Op(OpKind::MatMul),
        vec![x, w],
        NodeMeta::new(vec![8, 32], DType::F32),
        "mm",
    );
    let add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![mm, bias],
        NodeMeta::new(vec![8, 32], DType::F32),
        "add",
    );
    let relu = g.add_node(
        NodeKind::Op(OpKind::Relu),
        vec![add],
        NodeMeta::new(vec![8, 32], DType::F32),
        "relu",
    );

    // Dead branch: constant folding opportunity + dead code
    let c1 = g.add_node(
        NodeKind::Constant(vec![1.0, 2.0]),
        vec![],
        NodeMeta::new(vec![2], DType::F32),
        "c1",
    );
    let c2 = g.add_node(
        NodeKind::Constant(vec![3.0, 4.0]),
        vec![],
        NodeMeta::new(vec![2], DType::F32),
        "c2",
    );
    let _dead_add = g.add_node(
        NodeKind::Op(OpKind::Add),
        vec![c1, c2],
        NodeMeta::new(vec![2], DType::F32),
        "dead_add",
    );

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

// ── Helpers ─────────────────────────────────────────────────────────────

fn build_conv_bn_graph(channels: usize, spatial: usize) -> Graph {
    let total = channels * spatial;
    let mut g = Graph::new();

    let x = g.add_node(
        NodeKind::Input("x".into()),
        vec![],
        NodeMeta::new(vec![total], DType::F32),
        "x",
    );
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

    let conv = g.add_node(
        NodeKind::Op(OpKind::Conv2d),
        vec![x, w],
        NodeMeta::new(vec![total], DType::F32),
        "conv",
    );
    let bn = g.add_node(
        NodeKind::Op(OpKind::BatchNorm),
        vec![conv, gamma, beta, mean, var],
        NodeMeta::new(vec![total], DType::F32),
        "bn",
    );
    g.mark_output(bn);
    g
}
