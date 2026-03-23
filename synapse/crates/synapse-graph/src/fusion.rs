use crate::ir::{Graph, NodeId, NodeKind, NodeMeta, OpKind};
use crate::pass::OptimizationPass;

// ── MatMul + Bias + ReLU Fusion ────────────────────────────────────────

/// Fuses a MatMul -> Add (bias) -> ReLU pattern into FusedMatMulBiasRelu.
pub struct FuseMatMulBiasRelu;

impl FuseMatMulBiasRelu {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseMatMulBiasRelu {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FuseMatMulBiasRelu {
    fn name(&self) -> &str {
        "FuseMatMulBiasRelu"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Find all ReLU nodes
        let relu_ids: Vec<NodeId> = graph
            .node_ids()
            .into_iter()
            .filter(|&id| {
                graph
                    .node(id)
                    .map(|n| matches!(n.kind, NodeKind::Op(OpKind::Relu)))
                    .unwrap_or(false)
            })
            .collect();

        for relu_id in relu_ids {
            let relu_node = match graph.node(relu_id) {
                Some(n) => n,
                None => continue,
            };
            if relu_node.inputs.len() != 1 {
                continue;
            }
            let add_id = relu_node.inputs[0];

            // Check if input is Add with single use
            let add_node = match graph.node(add_id) {
                Some(n) if matches!(n.kind, NodeKind::Op(OpKind::Add)) => n,
                _ => continue,
            };
            if !graph.has_single_user(add_id) {
                continue;
            }
            if add_node.inputs.len() != 2 {
                continue;
            }

            let add_input_0 = add_node.inputs[0];
            let add_input_1 = add_node.inputs[1];

            // One of the add inputs should be a MatMul
            let (matmul_id, bias_id) = if is_matmul(graph, add_input_0) {
                (add_input_0, add_input_1)
            } else if is_matmul(graph, add_input_1) {
                (add_input_1, add_input_0)
            } else {
                continue;
            };

            // MatMul must have single use (only consumed by this Add)
            if !graph.has_single_user(matmul_id) {
                continue;
            }

            let matmul_node = graph.node(matmul_id).unwrap();
            if matmul_node.inputs.len() != 2 {
                continue;
            }
            let mm_a = matmul_node.inputs[0];
            let mm_b = matmul_node.inputs[1];
            let output_meta = relu_node.meta.clone();

            // Create fused node: inputs = [A, B, bias]
            let fused_id = graph.add_node(
                NodeKind::Op(OpKind::FusedMatMulBiasRelu),
                vec![mm_a, mm_b, bias_id],
                output_meta,
                "fused_matmul_bias_relu",
            );

            graph.replace_all_uses(relu_id, fused_id);
            graph.remove_node(relu_id);
            graph.remove_node(add_id);
            graph.remove_node(matmul_id);
            changed = true;
        }
        changed
    }
}

fn is_matmul(graph: &Graph, id: NodeId) -> bool {
    graph
        .node(id)
        .map(|n| matches!(n.kind, NodeKind::Op(OpKind::MatMul)))
        .unwrap_or(false)
}

// ── Conv + BatchNorm Fusion ─────────────────────────────────────────────

/// Fuses Conv2d -> BatchNorm by folding BN parameters into conv weights.
///
/// BN: y = gamma * (conv_out - mean) / sqrt(var + eps) + beta
/// Fused: y = (gamma / sqrt(var + eps)) * conv_out + (beta - gamma * mean / sqrt(var + eps))
///        = scale * conv_out + fused_bias
pub struct FuseConvBatchNorm;

impl FuseConvBatchNorm {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseConvBatchNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FuseConvBatchNorm {
    fn name(&self) -> &str {
        "FuseConvBatchNorm"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        let bn_ids: Vec<NodeId> = graph
            .node_ids()
            .into_iter()
            .filter(|&id| {
                graph
                    .node(id)
                    .map(|n| matches!(n.kind, NodeKind::Op(OpKind::BatchNorm)))
                    .unwrap_or(false)
            })
            .collect();

        for bn_id in bn_ids {
            let bn_node = match graph.node(bn_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            // BatchNorm inputs: [input, gamma, beta, mean, var]
            if bn_node.inputs.len() != 5 {
                continue;
            }

            let conv_id = bn_node.inputs[0];
            let conv_node = match graph.node(conv_id) {
                Some(n) if matches!(n.kind, NodeKind::Op(OpKind::Conv2d)) => n,
                _ => continue,
            };
            if !graph.has_single_user(conv_id) {
                continue;
            }

            // Conv2d inputs: [input, weights]
            if conv_node.inputs.len() < 2 {
                continue;
            }
            let conv_input = conv_node.inputs[0];
            let conv_weight_id = conv_node.inputs[1];

            let gamma_id = bn_node.inputs[1];
            let beta_id = bn_node.inputs[2];
            let mean_id = bn_node.inputs[3];
            let var_id = bn_node.inputs[4];

            // Get constant values for BN params and conv weights
            let conv_weight = get_constant_data(graph, conv_weight_id);
            let gamma = get_constant_data(graph, gamma_id);
            let beta = get_constant_data(graph, beta_id);
            let mean = get_constant_data(graph, mean_id);
            let var = get_constant_data(graph, var_id);

            let (conv_weight, gamma, beta, mean, var) =
                match (conv_weight, gamma, beta, mean, var) {
                    (Some(w), Some(g), Some(b), Some(m), Some(v)) => (w, g, b, m, v),
                    _ => continue,
                };

            let eps = 1e-5_f32;
            let channels = gamma.len();

            // Compute fused weight and bias
            let weight_per_channel = conv_weight.len() / channels;
            let mut fused_weight = conv_weight.clone();
            let mut fused_bias = vec![0.0f32; channels];

            for c in 0..channels {
                let scale = gamma[c] / (var[c] + eps).sqrt();
                fused_bias[c] = beta[c] - gamma[c] * mean[c] / (var[c] + eps).sqrt();
                for i in 0..weight_per_channel {
                    fused_weight[c * weight_per_channel + i] *= scale;
                }
            }

            // Create fused weight and bias constants
            let conv_weight_meta = graph.node(conv_weight_id).unwrap().meta.clone();
            let fused_w_id = graph.add_node(
                NodeKind::Constant(fused_weight),
                vec![],
                conv_weight_meta,
                "fused_conv_bn_weight",
            );
            let fused_b_id = graph.add_node(
                NodeKind::Constant(fused_bias),
                vec![],
                NodeMeta::new(vec![channels], crate::ir::DType::F32),
                "fused_conv_bn_bias",
            );

            // Create fused conv+bn node
            let fused_id = graph.add_node(
                NodeKind::Op(OpKind::FusedConvBatchNorm),
                vec![conv_input, fused_w_id, fused_b_id],
                bn_node.meta.clone(),
                "fused_conv_batchnorm",
            );

            graph.replace_all_uses(bn_id, fused_id);
            graph.remove_node(bn_id);
            graph.remove_node(conv_id);
            changed = true;
        }
        changed
    }
}

fn get_constant_data(graph: &Graph, id: NodeId) -> Option<Vec<f32>> {
    graph.node(id).and_then(|n| match &n.kind {
        NodeKind::Constant(data) => Some(data.clone()),
        NodeKind::Parameter(_) => None,
        _ => None,
    })
}

// ── Sequential Element-wise Fusion ──────────────────────────────────────

/// Fuses chains of element-wise operations into a single FusedElementWise node.
pub struct FuseElementWise;

impl FuseElementWise {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseElementWise {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FuseElementWise {
    fn name(&self) -> &str {
        "FuseElementWise"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Look for chains of element-wise ops where intermediate results have single users
        let all_ids = graph.node_ids();

        for &start_id in &all_ids {
            let start_node = match graph.node(start_id) {
                Some(n) => n,
                None => continue,
            };

            let start_op = match &start_node.kind {
                NodeKind::Op(op) if op.is_element_wise() => op.clone(),
                _ => continue,
            };

            // Walk the chain forward: follow single-use element-wise outputs
            let mut chain = vec![(start_id, start_op.clone())];
            let mut current_id = start_id;

            loop {
                let users = graph.users(current_id);
                if users.len() != 1 {
                    break;
                }
                let next_id = users[0];
                let next_node = match graph.node(next_id) {
                    Some(n) => n,
                    None => break,
                };
                let next_op = match &next_node.kind {
                    NodeKind::Op(op) if op.is_element_wise() => op.clone(),
                    _ => break,
                };
                // The next op must consume current_id as its first input
                if next_node.inputs.is_empty() || next_node.inputs[0] != current_id {
                    break;
                }
                chain.push((next_id, next_op));
                current_id = next_id;
            }

            if chain.len() < 2 {
                continue;
            }

            // Collect ops and all external inputs
            let ops: Vec<OpKind> = chain.iter().map(|(_, op)| op.clone()).collect();
            let first_id = chain[0].0;
            let last_id = chain[chain.len() - 1].0;

            // Gather all inputs: first node's inputs + additional inputs from binary ops
            let first_node = graph.node(first_id).unwrap();
            let mut all_inputs: Vec<NodeId> = first_node.inputs.clone();

            for &(nid, _) in chain.iter().skip(1) {
                let n = graph.node(nid).unwrap();
                // For binary ops, the second input is an external value
                for inp in n.inputs.iter().skip(1) {
                    all_inputs.push(*inp);
                }
            }

            let output_meta = graph.node(last_id).unwrap().meta.clone();

            let fused_id = graph.add_node(
                NodeKind::Op(OpKind::FusedElementWise(ops)),
                all_inputs,
                output_meta,
                "fused_elementwise",
            );

            graph.replace_all_uses(last_id, fused_id);

            // Remove all chain nodes
            for (nid, _) in &chain {
                graph.remove_node(*nid);
            }
            changed = true;
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use std::collections::HashMap;

    #[test]
    fn test_fuse_matmul_bias_relu() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![2, 3], DType::F32), "a");
        let w = g.add_node(NodeKind::Parameter("w".into()), vec![], NodeMeta::new(vec![3, 4], DType::F32), "w");
        let bias = g.add_node(NodeKind::Constant(vec![0.1; 4]), vec![], NodeMeta::new(vec![4], DType::F32), "bias");

        let mm = g.add_node(NodeKind::Op(OpKind::MatMul), vec![a, w], NodeMeta::new(vec![2, 4], DType::F32), "mm");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![mm, bias], NodeMeta::new(vec![2, 4], DType::F32), "add");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![2, 4], DType::F32), "relu");
        g.mark_output(relu);

        assert_eq!(g.node_count(), 6);

        let pass = FuseMatMulBiasRelu::new();
        let changed = pass.run(&mut g);
        assert!(changed);

        // MatMul, Add, ReLU replaced by 1 fused node => 6 - 3 + 1 = 4
        assert_eq!(g.node_count(), 4);

        let output_id = g.outputs()[0];
        let fused = g.node(output_id).unwrap();
        assert!(matches!(fused.kind, NodeKind::Op(OpKind::FusedMatMulBiasRelu)));
        assert_eq!(fused.inputs.len(), 3); // A, W, bias
    }

    #[test]
    fn test_fuse_conv_batchnorm() {
        let mut g = Graph::new();
        let channels = 2;
        let spatial = 4;
        let total = channels * spatial;

        let x = g.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![total], DType::F32), "x");
        let w = g.add_node(NodeKind::Constant(vec![1.0; total]), vec![], NodeMeta::new(vec![total], DType::F32), "w");
        let gamma = g.add_node(NodeKind::Constant(vec![1.0; channels]), vec![], NodeMeta::new(vec![channels], DType::F32), "gamma");
        let beta = g.add_node(NodeKind::Constant(vec![0.0; channels]), vec![], NodeMeta::new(vec![channels], DType::F32), "beta");
        let mean = g.add_node(NodeKind::Constant(vec![0.0; channels]), vec![], NodeMeta::new(vec![channels], DType::F32), "mean");
        let var = g.add_node(NodeKind::Constant(vec![1.0; channels]), vec![], NodeMeta::new(vec![channels], DType::F32), "var");

        let conv = g.add_node(NodeKind::Op(OpKind::Conv2d), vec![x, w], NodeMeta::new(vec![total], DType::F32), "conv");
        let bn = g.add_node(NodeKind::Op(OpKind::BatchNorm), vec![conv, gamma, beta, mean, var], NodeMeta::new(vec![total], DType::F32), "bn");
        g.mark_output(bn);

        let pass = FuseConvBatchNorm::new();
        let changed = pass.run(&mut g);
        assert!(changed);

        // After DCE the dead BN params are removed, reducing count
        crate::dead_code::DeadCodeElimination::new().run(&mut g);
        // Should have: x, fused_w, fused_b, fused_conv_bn = 4 nodes
        assert!(g.node_count() <= 4);

        let output_id = g.outputs()[0];
        let fused = g.node(output_id).unwrap();
        assert!(matches!(fused.kind, NodeKind::Op(OpKind::FusedConvBatchNorm)));
    }

    #[test]
    fn test_conv_bn_fusion_numerically_identical() {
        let channels = 3;
        let spatial = 4;
        let total = channels * spatial;

        let gamma_data = vec![2.0, 0.5, 1.5];
        let beta_data = vec![0.1, -0.2, 0.3];
        let mean_data = vec![0.5, 1.0, -0.5];
        let var_data = vec![0.25, 1.0, 4.0];
        let weight_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let input_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.5 - 1.0).collect();

        // Build unfused graph
        let mut g_unfused = Graph::new();
        let x1 = g_unfused.add_node(NodeKind::Input("x".into()), vec![], NodeMeta::new(vec![total], DType::F32), "x");
        let w1 = g_unfused.add_node(NodeKind::Constant(weight_data.clone()), vec![], NodeMeta::new(vec![total], DType::F32), "w");
        let gam1 = g_unfused.add_node(NodeKind::Constant(gamma_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "gamma");
        let bet1 = g_unfused.add_node(NodeKind::Constant(beta_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "beta");
        let men1 = g_unfused.add_node(NodeKind::Constant(mean_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "mean");
        let var1 = g_unfused.add_node(NodeKind::Constant(var_data.clone()), vec![], NodeMeta::new(vec![channels], DType::F32), "var");
        let conv1 = g_unfused.add_node(NodeKind::Op(OpKind::Conv2d), vec![x1, w1], NodeMeta::new(vec![total], DType::F32), "conv");
        let bn1 = g_unfused.add_node(NodeKind::Op(OpKind::BatchNorm), vec![conv1, gam1, bet1, men1, var1], NodeMeta::new(vec![total], DType::F32), "bn");
        g_unfused.mark_output(bn1);

        let mut inputs1 = HashMap::new();
        inputs1.insert(x1, input_data.clone());
        let result_unfused = g_unfused.execute(&inputs1);
        let unfused_output = result_unfused[&bn1].clone();

        // Build fused graph
        let mut g_fused = g_unfused.clone();
        FuseConvBatchNorm::new().run(&mut g_fused);

        let mut inputs2 = HashMap::new();
        inputs2.insert(x1, input_data);
        let result_fused = g_fused.execute(&inputs2);
        let fused_output_id = g_fused.outputs()[0];
        let fused_output = result_fused[&fused_output_id].clone();

        // Check numerical equivalence
        assert_eq!(unfused_output.len(), fused_output.len());
        for (i, (u, f)) in unfused_output.iter().zip(fused_output.iter()).enumerate() {
            assert!(
                (u - f).abs() < 1e-5,
                "Mismatch at index {}: unfused={}, fused={}, diff={}",
                i, u, f, (u - f).abs()
            );
        }
    }

    #[test]
    fn test_fuse_elementwise_chain() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![4], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![4], DType::F32), "b");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![4], DType::F32), "add");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![4], DType::F32), "relu");
        let neg = g.add_node(NodeKind::Op(OpKind::Neg), vec![relu], NodeMeta::new(vec![4], DType::F32), "neg");
        g.mark_output(neg);

        assert_eq!(g.node_count(), 5);

        let pass = FuseElementWise::new();
        let changed = pass.run(&mut g);
        assert!(changed);

        // add, relu, neg fused into 1 -> 5 - 3 + 1 = 3
        assert_eq!(g.node_count(), 3);
    }
}
