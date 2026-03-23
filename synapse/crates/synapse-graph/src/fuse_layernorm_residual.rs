use crate::ir::{Graph, NodeId, NodeKind, OpKind};
use crate::pass::OptimizationPass;

/// Fuses Add(x, residual) -> LayerNorm into a single FusedLayerNormResidual node.
///
/// Detected pattern:
///   sum = Add(x, residual)
///   output = LayerNorm(sum, gamma, beta)
///
/// Replaces with: FusedLayerNormResidual(x, residual, gamma, beta)
pub struct FuseLayerNormResidual;

impl FuseLayerNormResidual {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseLayerNormResidual {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FuseLayerNormResidual {
    fn name(&self) -> &str {
        "fuse_layernorm_residual"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Find all LayerNorm nodes
        let ln_ids: Vec<NodeId> = graph
            .node_ids()
            .into_iter()
            .filter(|&id| {
                graph
                    .node(id)
                    .map(|n| matches!(n.kind, NodeKind::Op(OpKind::LayerNorm)))
                    .unwrap_or(false)
            })
            .collect();

        for ln_id in ln_ids {
            let ln_node = match graph.node(ln_id) {
                Some(n) => n,
                None => continue,
            };
            // LayerNorm inputs: [input, gamma, beta]
            if ln_node.inputs.len() != 3 {
                continue;
            }

            let add_id = ln_node.inputs[0];
            let gamma_id = ln_node.inputs[1];
            let beta_id = ln_node.inputs[2];

            // Check if input is Add with single user
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

            let x_id = add_node.inputs[0];
            let residual_id = add_node.inputs[1];
            let output_meta = ln_node.meta.clone();

            // Create fused node: inputs = [x, residual, gamma, beta]
            let fused_id = graph.add_node(
                NodeKind::Op(OpKind::FusedLayerNormResidual),
                vec![x_id, residual_id, gamma_id, beta_id],
                output_meta,
                "fused_layernorm_residual",
            );

            graph.replace_all_uses(ln_id, fused_id);
            graph.remove_node(ln_id);
            graph.remove_node(add_id);
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

    fn build_add_layernorm_graph(dim: usize) -> (Graph, NodeId, NodeId, NodeId) {
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

        (g, x, residual, ln)
    }

    #[test]
    fn test_fuse_layernorm_residual() {
        let (mut g, _x, _residual, _ln) = build_add_layernorm_graph(8);

        // x, residual, gamma, beta, add, layernorm = 6 nodes
        assert_eq!(g.node_count(), 6);

        let pass = FuseLayerNormResidual::new();
        let changed = pass.run(&mut g);
        assert!(changed);

        // add + layernorm replaced by 1 fused node => 6 - 2 + 1 = 5
        assert_eq!(g.node_count(), 5);

        let output_id = g.outputs()[0];
        let fused = g.node(output_id).unwrap();
        assert!(matches!(
            fused.kind,
            NodeKind::Op(OpKind::FusedLayerNormResidual)
        ));
        assert_eq!(fused.inputs.len(), 4); // x, residual, gamma, beta
    }

    #[test]
    fn test_fuse_layernorm_residual_idempotent() {
        let (mut g, _, _, _) = build_add_layernorm_graph(8);

        let pass = FuseLayerNormResidual::new();
        pass.run(&mut g);
        let count_after_first = g.node_count();

        let changed = pass.run(&mut g);
        assert!(!changed, "Second run should not change anything");
        assert_eq!(g.node_count(), count_after_first);
    }

    #[test]
    fn test_fuse_layernorm_residual_numerically_identical() {
        let dim = 8;
        let (g_unfused, x_id, residual_id, _) = build_add_layernorm_graph(dim);

        let x_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let res_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();

        let mut inputs = HashMap::new();
        inputs.insert(x_id, x_data);
        inputs.insert(residual_id, res_data);

        let unfused_output_id = g_unfused.outputs()[0];
        let unfused_result = g_unfused.execute(&inputs);
        let unfused_out = unfused_result[&unfused_output_id].clone();

        let mut g_fused = g_unfused.clone();
        FuseLayerNormResidual::new().run(&mut g_fused);

        let fused_output_id = g_fused.outputs()[0];
        let fused_result = g_fused.execute(&inputs);
        let fused_out = fused_result[&fused_output_id].clone();

        assert_eq!(unfused_out.len(), fused_out.len());
        for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
            assert!(
                (u - f).abs() < 1e-5,
                "LayerNorm+Residual fusion mismatch at {}: unfused={}, fused={}, diff={}",
                i, u, f, (u - f).abs()
            );
        }
    }

    #[test]
    fn test_no_fuse_when_add_has_multiple_users() {
        let mut g = Graph::new();
        let dim = 4;

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
            "add",
        );
        let ln = g.add_node(
            NodeKind::Op(OpKind::LayerNorm),
            vec![add, gamma, beta],
            NodeMeta::new(vec![dim], DType::F32),
            "ln",
        );
        // Add has a second user, preventing fusion
        let relu = g.add_node(
            NodeKind::Op(OpKind::Relu),
            vec![add],
            NodeMeta::new(vec![dim], DType::F32),
            "relu",
        );
        g.mark_output(ln);
        g.mark_output(relu);

        let before = g.node_count();
        let pass = FuseLayerNormResidual::new();
        let changed = pass.run(&mut g);
        assert!(!changed, "Should not fuse when Add has multiple users");
        assert_eq!(g.node_count(), before);
    }
}
