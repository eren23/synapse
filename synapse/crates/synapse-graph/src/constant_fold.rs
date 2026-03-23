use std::collections::HashMap;

use crate::ir::{Graph, NodeId, NodeKind};
use crate::pass::OptimizationPass;

/// Folds subgraphs where all inputs are constants into a single Constant node.
pub struct ConstantFolding;

impl ConstantFolding {
    pub fn new() -> Self {
        Self
    }

    /// Check if all inputs to a node are constants.
    fn all_inputs_constant(&self, graph: &Graph, id: NodeId) -> bool {
        let node = match graph.node(id) {
            Some(n) => n,
            None => return false,
        };
        if !matches!(node.kind, NodeKind::Op(_)) {
            return false;
        }
        node.inputs.iter().all(|&inp| {
            graph
                .node(inp)
                .map(|n| matches!(n.kind, NodeKind::Constant(_)))
                .unwrap_or(false)
        })
    }
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        loop {
            let candidates: Vec<NodeId> = graph
                .node_ids()
                .into_iter()
                .filter(|&id| self.all_inputs_constant(graph, id))
                .collect();

            if candidates.is_empty() {
                break;
            }

            for id in candidates {
                let node = match graph.node(id) {
                    Some(n) => n.clone(),
                    None => continue,
                };

                // Gather constant input values
                let mut input_values = HashMap::new();
                for &inp in &node.inputs {
                    if let Some(inp_node) = graph.node(inp) {
                        if let NodeKind::Constant(data) = &inp_node.kind {
                            input_values.insert(inp, data.clone());
                        }
                    }
                }

                // Evaluate
                let result = graph.execute(&input_values);
                if let Some(val) = result.get(&id) {
                    // Replace op node with a constant holding the computed value
                    let new_id = graph.add_node(
                        NodeKind::Constant(val.clone()),
                        vec![],
                        node.meta.clone(),
                        format!("{}_folded", node.name),
                    );
                    graph.replace_all_uses(id, new_id);
                    graph.remove_node(id);
                    changed = true;
                }
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    #[test]
    fn test_fold_constant_add() {
        let mut g = Graph::new();
        let a = g.add_node(
            NodeKind::Constant(vec![1.0, 2.0, 3.0]),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "a",
        );
        let b = g.add_node(
            NodeKind::Constant(vec![4.0, 5.0, 6.0]),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "b",
        );
        let add = g.add_node(
            NodeKind::Op(OpKind::Add),
            vec![a, b],
            NodeMeta::new(vec![3], DType::F32),
            "add",
        );
        g.mark_output(add);

        assert_eq!(g.node_count(), 3);

        let cf = ConstantFolding::new();
        let changed = cf.run(&mut g);

        assert!(changed);
        // The add node should be replaced with a constant
        let output_id = g.outputs()[0];
        let output_node = g.node(output_id).unwrap();
        match &output_node.kind {
            NodeKind::Constant(data) => {
                assert_eq!(data, &vec![5.0, 7.0, 9.0]);
            }
            _ => panic!("expected constant after folding"),
        }
    }

    #[test]
    fn test_fold_chain() {
        let mut g = Graph::new();
        let a = g.add_node(
            NodeKind::Constant(vec![2.0, 4.0]),
            vec![],
            NodeMeta::new(vec![2], DType::F32),
            "a",
        );
        let b = g.add_node(
            NodeKind::Constant(vec![3.0, 5.0]),
            vec![],
            NodeMeta::new(vec![2], DType::F32),
            "b",
        );
        let add = g.add_node(
            NodeKind::Op(OpKind::Add),
            vec![a, b],
            NodeMeta::new(vec![2], DType::F32),
            "add",
        );
        // neg(add) should also fold since add becomes constant
        let neg = g.add_node(
            NodeKind::Op(OpKind::Neg),
            vec![add],
            NodeMeta::new(vec![2], DType::F32),
            "neg",
        );
        g.mark_output(neg);

        let cf = ConstantFolding::new();
        cf.run(&mut g);

        let output_id = g.outputs()[0];
        let output_node = g.node(output_id).unwrap();
        match &output_node.kind {
            NodeKind::Constant(data) => {
                assert_eq!(data, &vec![-5.0, -9.0]);
            }
            _ => panic!("expected constant after folding chain"),
        }
    }

    #[test]
    fn test_no_fold_with_input() {
        let mut g = Graph::new();
        let x = g.add_node(
            NodeKind::Input("x".into()),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "x",
        );
        let c = g.add_node(
            NodeKind::Constant(vec![1.0, 1.0, 1.0]),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "c",
        );
        let add = g.add_node(
            NodeKind::Op(OpKind::Add),
            vec![x, c],
            NodeMeta::new(vec![3], DType::F32),
            "add",
        );
        g.mark_output(add);

        let cf = ConstantFolding::new();
        let changed = cf.run(&mut g);
        assert!(!changed); // Can't fold because x is an input
    }

    #[test]
    fn test_fold_computes_correctly() {
        let mut g = Graph::new();
        let a = g.add_node(
            NodeKind::Constant(vec![9.0, 16.0, 25.0]),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "a",
        );
        let sqrt = g.add_node(
            NodeKind::Op(OpKind::Sqrt),
            vec![a],
            NodeMeta::new(vec![3], DType::F32),
            "sqrt",
        );
        g.mark_output(sqrt);

        ConstantFolding::new().run(&mut g);

        let output_id = g.outputs()[0];
        let output_node = g.node(output_id).unwrap();
        match &output_node.kind {
            NodeKind::Constant(data) => {
                assert_eq!(data, &vec![3.0, 4.0, 5.0]);
            }
            _ => panic!("expected constant"),
        }
    }
}
