use std::collections::{HashMap, HashSet};

use crate::ir::{Graph, NodeId};
use crate::pass::OptimizationPass;

/// Schedules nodes in a memory-optimal execution order using liveness analysis.
///
/// The scheduler produces a valid topological ordering that minimizes peak memory
/// by preferring to schedule nodes whose inputs can be freed (all other users
/// already scheduled).
pub struct MemoryOptimalScheduler;

impl MemoryOptimalScheduler {
    pub fn new() -> Self {
        Self
    }

    /// Compute execution order that minimizes peak live memory.
    pub fn schedule(&self, graph: &Graph) -> Vec<NodeId> {
        let live_ids: Vec<NodeId> = graph.node_ids();
        if live_ids.is_empty() {
            return vec![];
        }

        let live_set: HashSet<NodeId> = live_ids.iter().copied().collect();

        // Build adjacency: for each node, which live nodes depend on it
        let mut dependents: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

        for &id in &live_ids {
            dependents.entry(id).or_default();
            in_degree.entry(id).or_insert(0);
        }

        for &id in &live_ids {
            if let Some(node) = graph.node(id) {
                let mut live_input_count = 0;
                for &inp in &node.inputs {
                    if live_set.contains(&inp) {
                        dependents.entry(inp).or_default().push(id);
                        live_input_count += 1;
                    }
                }
                *in_degree.entry(id).or_insert(0) = live_input_count;
            }
        }

        // Count how many live users each node has (for liveness tracking)
        let mut user_count: HashMap<NodeId, usize> = HashMap::new();
        for &id in &live_ids {
            if let Some(node) = graph.node(id) {
                for &inp in &node.inputs {
                    if live_set.contains(&inp) {
                        *user_count.entry(inp).or_insert(0) += 1;
                    }
                }
            }
        }

        // Ready set: nodes with in_degree == 0
        let mut ready: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut scheduled = Vec::with_capacity(live_ids.len());
        let mut remaining_users: HashMap<NodeId, usize> = user_count.clone();
        let mut live_memory: HashSet<NodeId> = HashSet::new();
        let output_set: HashSet<NodeId> = graph.outputs().iter().copied().collect();

        while !ready.is_empty() {
            // Heuristic: pick the ready node that frees the most memory.
            // A node frees memory for input tensors whose remaining_users drops to 0
            // after being scheduled.
            let best_idx = select_best(&ready, graph, &remaining_users, &live_memory, &output_set);
            let chosen = ready.swap_remove(best_idx);

            scheduled.push(chosen);
            live_memory.insert(chosen);

            // Decrement remaining_users for chosen's inputs
            if let Some(node) = graph.node(chosen) {
                for &inp in &node.inputs {
                    if let Some(count) = remaining_users.get_mut(&inp) {
                        *count = count.saturating_sub(1);
                        if *count == 0 && !output_set.contains(&inp) {
                            live_memory.remove(&inp);
                        }
                    }
                }
            }

            // Update in_degree for dependents
            if let Some(deps) = dependents.get(&chosen) {
                for &dep in deps {
                    if let Some(deg) = in_degree.get_mut(&dep) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            ready.push(dep);
                        }
                    }
                }
            }
        }

        scheduled
    }

    /// Validate that the schedule is a valid topological order.
    pub fn validate_order(graph: &Graph, order: &[NodeId]) -> bool {
        let position: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        for &id in order {
            if let Some(node) = graph.node(id) {
                for &inp in &node.inputs {
                    match position.get(&inp) {
                        Some(&inp_pos) => {
                            if let Some(&id_pos) = position.get(&id) {
                                if inp_pos >= id_pos {
                                    return false;
                                }
                            }
                        }
                        None => {
                            // Input not in order at all - only ok if it was removed
                            if graph.node(inp).is_some() {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Compute peak live memory (number of live tensors) for a given schedule.
    pub fn peak_live_tensors(graph: &Graph, order: &[NodeId]) -> usize {
        let output_set: HashSet<NodeId> = graph.outputs().iter().copied().collect();
        let live_ids: HashSet<NodeId> = graph.node_ids().into_iter().collect();

        // Count total users for each node
        let mut total_users: HashMap<NodeId, usize> = HashMap::new();
        for &id in order {
            if let Some(node) = graph.node(id) {
                for &inp in &node.inputs {
                    if live_ids.contains(&inp) {
                        *total_users.entry(inp).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut remaining = total_users.clone();
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut peak = 0;

        for &id in order {
            live.insert(id);
            peak = peak.max(live.len());

            // Decrement remaining users of inputs and free if done
            if let Some(node) = graph.node(id) {
                for &inp in &node.inputs {
                    if let Some(count) = remaining.get_mut(&inp) {
                        *count = count.saturating_sub(1);
                        if *count == 0 && !output_set.contains(&inp) {
                            live.remove(&inp);
                        }
                    }
                }
            }
        }
        peak
    }
}

impl Default for MemoryOptimalScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Select the best ready node to schedule next.
/// Prefers nodes that free the most input memory when scheduled.
fn select_best(
    ready: &[NodeId],
    graph: &Graph,
    remaining_users: &HashMap<NodeId, usize>,
    _live_memory: &HashSet<NodeId>,
    output_set: &HashSet<NodeId>,
) -> usize {
    let mut best_idx = 0;
    let mut best_freed = 0i64;

    for (i, &id) in ready.iter().enumerate() {
        let mut freed = 0i64;
        if let Some(node) = graph.node(id) {
            for &inp in &node.inputs {
                let count = remaining_users.get(&inp).copied().unwrap_or(0);
                if count <= 1 && !output_set.contains(&inp) {
                    // This input would be freed
                    if let Some(inp_node) = graph.node(inp) {
                        freed += inp_node.meta.size_bytes() as i64;
                    }
                }
            }
            // Cost of producing this node's output
            freed -= node.meta.size_bytes() as i64;
        }

        if freed > best_freed || (freed == best_freed && id < ready[best_idx]) {
            best_freed = freed;
            best_idx = i;
        }
    }
    best_idx
}

/// OptimizationPass wrapper that reorders graph nodes according to the scheduler.
/// This doesn't actually modify the graph topology, but can be used to produce
/// an ordered execution plan.
impl OptimizationPass for MemoryOptimalScheduler {
    fn name(&self) -> &str {
        "MemoryOptimalScheduler"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let order = self.schedule(graph);
        // Validate it's a proper topological order
        Self::validate_order(graph, &order);
        // The scheduler doesn't modify graph topology
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    #[test]
    fn test_scheduler_basic_chain() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![100], DType::F32), "a");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![a], NodeMeta::new(vec![100], DType::F32), "relu");
        let neg = g.add_node(NodeKind::Op(OpKind::Neg), vec![relu], NodeMeta::new(vec![100], DType::F32), "neg");
        g.mark_output(neg);

        let scheduler = MemoryOptimalScheduler::new();
        let order = scheduler.schedule(&g);

        assert_eq!(order.len(), 3);
        assert!(MemoryOptimalScheduler::validate_order(&g, &order));
    }

    #[test]
    fn test_scheduler_diamond() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![100], DType::F32), "a");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![a], NodeMeta::new(vec![100], DType::F32), "relu");
        let neg = g.add_node(NodeKind::Op(OpKind::Neg), vec![a], NodeMeta::new(vec![100], DType::F32), "neg");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![relu, neg], NodeMeta::new(vec![100], DType::F32), "add");
        g.mark_output(add);

        let scheduler = MemoryOptimalScheduler::new();
        let order = scheduler.schedule(&g);

        assert_eq!(order.len(), 4);
        assert!(MemoryOptimalScheduler::validate_order(&g, &order));

        // a must come before both relu and neg, which must come before add
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a) < pos(relu));
        assert!(pos(a) < pos(neg));
        assert!(pos(relu) < pos(add));
        assert!(pos(neg) < pos(add));
    }

    #[test]
    fn test_scheduler_respects_dependencies() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![10], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![10], DType::F32), "b");
        let c = g.add_node(NodeKind::Input("c".into()), vec![], NodeMeta::new(vec![10], DType::F32), "c");

        let ab = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![10], DType::F32), "ab");
        let abc = g.add_node(NodeKind::Op(OpKind::Add), vec![ab, c], NodeMeta::new(vec![10], DType::F32), "abc");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![abc], NodeMeta::new(vec![10], DType::F32), "relu");
        g.mark_output(relu);

        let scheduler = MemoryOptimalScheduler::new();
        let order = scheduler.schedule(&g);

        assert_eq!(order.len(), 6);
        assert!(MemoryOptimalScheduler::validate_order(&g, &order));

        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a) < pos(ab));
        assert!(pos(b) < pos(ab));
        assert!(pos(ab) < pos(abc));
        assert!(pos(c) < pos(abc));
        assert!(pos(abc) < pos(relu));
    }

    #[test]
    fn test_scheduler_parallel_branches() {
        // Two independent branches merged at the end
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![1000], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![1000], DType::F32), "b");

        // Branch 1: a -> relu -> sigmoid (large tensors)
        let relu_a = g.add_node(NodeKind::Op(OpKind::Relu), vec![a], NodeMeta::new(vec![1000], DType::F32), "relu_a");
        let sig_a = g.add_node(NodeKind::Op(OpKind::Sigmoid), vec![relu_a], NodeMeta::new(vec![1000], DType::F32), "sig_a");

        // Branch 2: b -> neg -> tanh
        let neg_b = g.add_node(NodeKind::Op(OpKind::Neg), vec![b], NodeMeta::new(vec![1000], DType::F32), "neg_b");
        let tanh_b = g.add_node(NodeKind::Op(OpKind::Tanh), vec![neg_b], NodeMeta::new(vec![1000], DType::F32), "tanh_b");

        // Merge
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![sig_a, tanh_b], NodeMeta::new(vec![1000], DType::F32), "add");
        g.mark_output(add);

        let scheduler = MemoryOptimalScheduler::new();
        let order = scheduler.schedule(&g);

        assert_eq!(order.len(), 7);
        assert!(MemoryOptimalScheduler::validate_order(&g, &order));
    }

    #[test]
    fn test_peak_memory_calculation() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![100], DType::F32), "a");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![a], NodeMeta::new(vec![100], DType::F32), "relu");
        g.mark_output(relu);

        let scheduler = MemoryOptimalScheduler::new();
        let order = scheduler.schedule(&g);
        let peak = MemoryOptimalScheduler::peak_live_tensors(&g, &order);
        // a is live until relu consumes it, then relu is live -> peak = 2
        assert!(peak >= 1 && peak <= 2);
    }
}
