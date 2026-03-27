use std::collections::HashSet;

use crate::ir::{Graph, NodeId};
use crate::pass::OptimizationPass;

/// Removes nodes that are not reachable from any graph output.
pub struct DeadCodeElimination;

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self
    }

    /// Collect all node ids reachable from the graph outputs.
    fn reachable(&self, graph: &Graph) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = graph.outputs().to_vec();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(node) = graph.node(id) {
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
        }
        visited
    }
}

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let reachable = self.reachable(graph);
        let all_ids = graph.node_ids();
        let mut removed = false;

        for id in all_ids {
            if !reachable.contains(&id) {
                graph.remove_node(id);
                removed = true;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    #[test]
    fn test_dce_removes_dead_branch() {
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

        // Live path: a -> relu -> output
        let relu = g.add_node(
            NodeKind::Op(OpKind::Relu),
            vec![a],
            NodeMeta::new(vec![4], DType::F32),
            "relu",
        );

        // Dead path: b -> neg -> sigmoid (not connected to output)
        let neg = g.add_node(
            NodeKind::Op(OpKind::Neg),
            vec![b],
            NodeMeta::new(vec![4], DType::F32),
            "neg",
        );
        let _sig = g.add_node(
            NodeKind::Op(OpKind::Sigmoid),
            vec![neg],
            NodeMeta::new(vec![4], DType::F32),
            "sig",
        );

        g.mark_output(relu);
        assert_eq!(g.node_count(), 5);

        let dce = DeadCodeElimination::new();
        let changed = dce.run(&mut g);

        assert!(changed);
        assert_eq!(g.node_count(), 2); // Only a and relu remain
        assert!(g.node(a).is_some());
        assert!(g.node(relu).is_some());
    }

    #[test]
    fn test_dce_no_change_when_all_reachable() {
        let mut g = Graph::new();
        let a = g.add_node(
            NodeKind::Input("a".into()),
            vec![],
            NodeMeta::new(vec![4], DType::F32),
            "a",
        );
        let relu = g.add_node(
            NodeKind::Op(OpKind::Relu),
            vec![a],
            NodeMeta::new(vec![4], DType::F32),
            "relu",
        );
        g.mark_output(relu);

        let dce = DeadCodeElimination::new();
        let changed = dce.run(&mut g);
        assert!(!changed);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_dce_preserves_semantics() {
        use std::collections::HashMap;

        let mut g = Graph::new();
        let a = g.add_node(
            NodeKind::Input("a".into()),
            vec![],
            NodeMeta::new(vec![3], DType::F32),
            "a",
        );
        let b = g.add_node(
            NodeKind::Input("b".into()),
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

        // Dead branch
        let _neg = g.add_node(
            NodeKind::Op(OpKind::Neg),
            vec![a],
            NodeMeta::new(vec![3], DType::F32),
            "neg",
        );
        g.mark_output(add);

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0]);
        inputs.insert(b, vec![4.0, 5.0, 6.0]);
        let before = g.execute(&inputs)[&add].clone();

        DeadCodeElimination::new().run(&mut g);

        let after = g.execute(&inputs)[&add].clone();
        assert_eq!(before, after);
    }
}
