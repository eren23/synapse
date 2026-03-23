use std::collections::HashMap;

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

/// Reverse-mode automatic differentiation.
///
/// Walks the computation graph from `output_id` in reverse topological order,
/// accumulating gradients. Fan-out is handled by summing gradients from all
/// consumers of a node.
pub fn backward(graph: &mut Graph, output_id: VariableId) {
    let output_node_idx = graph.variables[&output_id]
        .node_idx
        .expect("cannot run backward on untracked variable");

    // Topological sort: leaves first, output last.
    let topo_order = graph.topological_sort(output_node_idx);

    // Accumulated gradient per node index.
    let mut grad_map: HashMap<usize, Tensor> = HashMap::new();

    // Seed the output gradient with ones.
    let output_shape = graph.variables[&output_id].data.shape.clone();
    grad_map.insert(output_node_idx, Tensor::ones(&output_shape));

    // Walk in reverse topological order (output → leaves).
    for &node_idx in topo_order.iter().rev() {
        let grad_output = match grad_map.get(&node_idx) {
            Some(g) => g.clone(),
            None => continue,
        };

        let node = &graph.nodes[node_idx];

        if let Some(ref grad_fn) = node.grad_fn {
            let input_grads = grad_fn.backward(&grad_output);
            let parent_indices = node.parent_indices.clone();

            for (i, &parent_idx) in parent_indices.iter().enumerate() {
                if let Some(Some(g)) = input_grads.get(i) {
                    grad_map
                        .entry(parent_idx)
                        .and_modify(|existing| *existing = existing.add(g))
                        .or_insert_with(|| g.clone());
                }
            }
        }
    }

    // Store computed gradients into variables.
    for (&node_idx, grad) in &grad_map {
        let var_id = graph.nodes[node_idx].var_id;
        if graph.variables[&var_id].requires_grad {
            graph.variable_mut(var_id).grad = Some(grad.clone());
        }
    }
}
