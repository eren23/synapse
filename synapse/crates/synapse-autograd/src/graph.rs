use std::collections::HashMap;

use crate::function::GradFn;
use crate::no_grad::is_grad_enabled;
use crate::tensor::Tensor;
use crate::variable::{Variable, VariableId};

/// A node in the computation graph.
pub struct Node {
    /// Backward function (None for leaf variables).
    pub grad_fn: Option<Box<dyn GradFn>>,
    /// Indices of parent (input) nodes in the graph's node list.
    pub parent_indices: Vec<usize>,
    /// The variable this node produces.
    pub var_id: VariableId,
}

/// Computation graph built as a Vec of nodes.
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) variables: HashMap<VariableId, Variable>,
    pub(crate) var_to_node: HashMap<VariableId, usize>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            variables: HashMap::new(),
            var_to_node: HashMap::new(),
        }
    }

    /// Create and register a leaf variable.
    pub fn variable(&mut self, data: Tensor, requires_grad: bool) -> VariableId {
        let mut var = Variable::new(data, requires_grad);
        let idx = self.nodes.len();
        var.node_idx = Some(idx);
        let id = var.id;
        self.nodes.push(Node {
            grad_fn: None,
            parent_indices: vec![],
            var_id: id,
        });
        self.var_to_node.insert(id, idx);
        self.variables.insert(id, var);
        id
    }

    /// Record an operation node in the graph.
    pub(crate) fn record_op(
        &mut self,
        grad_fn: Box<dyn GradFn>,
        input_ids: &[VariableId],
        output_data: Tensor,
    ) -> VariableId {
        let parent_indices: Vec<usize> = input_ids
            .iter()
            .map(|id| {
                *self
                    .var_to_node
                    .get(id)
                    .expect("input variable not in graph")
            })
            .collect();

        let idx = self.nodes.len();
        let var = Variable::with_node(output_data, true, idx);
        let id = var.id;

        self.nodes.push(Node {
            grad_fn: Some(grad_fn),
            parent_indices,
            var_id: id,
        });
        self.var_to_node.insert(id, idx);
        self.variables.insert(id, var);
        id
    }

    /// Create an untracked variable (no graph node).
    pub(crate) fn untracked(&mut self, output_data: Tensor) -> VariableId {
        let var = Variable::new(output_data, false);
        let id = var.id;
        self.variables.insert(id, var);
        id
    }

    /// Whether the operation should be tracked in the graph.
    pub(crate) fn should_track(&self, ids: &[VariableId]) -> bool {
        is_grad_enabled() && ids.iter().any(|id| self.variables[id].requires_grad)
    }

    /// Topological sort reachable from `root`, returning node indices
    /// in dependency order (leaves first, root last).
    pub fn topological_sort(&self, root: usize) -> Vec<usize> {
        let mut visited = vec![false; self.nodes.len()];
        let mut order = Vec::new();
        self.topo_dfs(root, &mut visited, &mut order);
        order
    }

    fn topo_dfs(&self, node: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        for &parent in &self.nodes[node].parent_indices {
            self.topo_dfs(parent, visited, order);
        }
        order.push(node);
    }

    /// Get the data tensor of a variable.
    pub fn data(&self, var: VariableId) -> &Tensor {
        &self.variables[&var].data
    }

    /// Get the gradient tensor of a variable, if computed.
    pub fn grad(&self, var: VariableId) -> Option<&Tensor> {
        self.variables[&var].grad.as_ref()
    }

    pub(crate) fn variable_mut(&mut self, var: VariableId) -> &mut Variable {
        self.variables.get_mut(&var).expect("variable not found")
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
