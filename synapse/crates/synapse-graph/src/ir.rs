use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a node in the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Data types supported by the graph IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }
}

/// Metadata attached to each node: shape and dtype.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeMeta {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl NodeMeta {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}

/// The kind of operation a node represents.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    // Linear algebra
    MatMul,
    // Element-wise
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sqrt,
    // Activations
    Relu,
    Sigmoid,
    Tanh,
    // Neural network
    Conv2d,
    BatchNorm,
    // Fused operations
    FusedMatMulBiasRelu,
    FusedConvBatchNorm,
    FusedElementWise(Vec<OpKind>),
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::MatMul => write!(f, "MatMul"),
            OpKind::Add => write!(f, "Add"),
            OpKind::Sub => write!(f, "Sub"),
            OpKind::Mul => write!(f, "Mul"),
            OpKind::Div => write!(f, "Div"),
            OpKind::Neg => write!(f, "Neg"),
            OpKind::Sqrt => write!(f, "Sqrt"),
            OpKind::Relu => write!(f, "Relu"),
            OpKind::Sigmoid => write!(f, "Sigmoid"),
            OpKind::Tanh => write!(f, "Tanh"),
            OpKind::Conv2d => write!(f, "Conv2d"),
            OpKind::BatchNorm => write!(f, "BatchNorm"),
            OpKind::FusedMatMulBiasRelu => write!(f, "FusedMatMulBiasRelu"),
            OpKind::FusedConvBatchNorm => write!(f, "FusedConvBatchNorm"),
            OpKind::FusedElementWise(ops) => {
                write!(f, "FusedElementWise(")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl OpKind {
    /// Returns true if this is an element-wise unary or binary operation.
    pub fn is_element_wise(&self) -> bool {
        matches!(
            self,
            OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::Div
                | OpKind::Neg
                | OpKind::Sqrt
                | OpKind::Relu
                | OpKind::Sigmoid
                | OpKind::Tanh
        )
    }

    /// Returns true if this is a unary element-wise op.
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            OpKind::Neg | OpKind::Sqrt | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh
        )
    }
}

/// The kind of a node in the graph.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// A computation operation.
    Op(OpKind),
    /// A constant value embedded in the graph.
    Constant(Vec<f32>),
    /// A trainable parameter (e.g., weights, biases).
    Parameter(String),
    /// An external input to the graph.
    Input(String),
}

/// A node in the computation graph.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub inputs: Vec<NodeId>,
    pub meta: NodeMeta,
    pub name: String,
}

/// The computation graph: an arena of nodes with tracked outputs.
#[derive(Clone, Debug)]
pub struct Graph {
    nodes: Vec<Option<Node>>,
    outputs: Vec<NodeId>,
    next_id: usize,
    id_to_index: HashMap<NodeId, usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
            id_to_index: HashMap::new(),
        }
    }

    /// Add a node to the graph and return its id.
    pub fn add_node(&mut self, kind: NodeKind, inputs: Vec<NodeId>, meta: NodeMeta, name: impl Into<String>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let index = self.nodes.len();
        self.id_to_index.insert(id, index);
        self.nodes.push(Some(Node {
            id,
            kind,
            inputs,
            meta,
            name: name.into(),
        }));
        id
    }

    /// Mark a node as a graph output.
    pub fn mark_output(&mut self, id: NodeId) {
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
    }

    /// Get the graph outputs.
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Set the graph outputs.
    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    /// Get a reference to a node by id.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.id_to_index.get(&id).and_then(|&idx| self.nodes[idx].as_ref())
    }

    /// Get a mutable reference to a node by id.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.id_to_index.get(&id).copied().and_then(|idx| self.nodes[idx].as_mut())
    }

    /// Remove a node from the graph.
    pub fn remove_node(&mut self, id: NodeId) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            self.nodes[idx] = None;
            self.id_to_index.remove(&id);
        }
    }

    /// Replace a node's inputs.
    pub fn set_inputs(&mut self, id: NodeId, inputs: Vec<NodeId>) {
        if let Some(node) = self.node_mut(id) {
            node.inputs = inputs;
        }
    }

    /// Get all live (non-removed) node ids.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().filter_map(|n| n.as_ref().map(|n| n.id)).collect()
    }

    /// Count live nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Get all nodes that use `target` as an input.
    pub fn users(&self, target: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .filter(|n| n.inputs.contains(&target))
            .map(|n| n.id)
            .collect()
    }

    /// Check if a node has exactly one user.
    pub fn has_single_user(&self, id: NodeId) -> bool {
        self.users(id).len() == 1
    }

    /// Remap all references from `old_id` to `new_id` in node inputs and outputs.
    pub fn replace_all_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        for slot in &mut self.nodes {
            if let Some(node) = slot {
                for input in &mut node.inputs {
                    if *input == old_id {
                        *input = new_id;
                    }
                }
            }
        }
        for output in &mut self.outputs {
            if *output == old_id {
                *output = new_id;
            }
        }
    }

    /// Topological sort of all live nodes (dependencies before dependents).
    pub fn topological_order(&self) -> Vec<NodeId> {
        let live_ids: HashSet<NodeId> = self.node_ids().into_iter().collect();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &id in &live_ids {
            in_degree.entry(id).or_insert(0);
            if let Some(node) = self.node(id) {
                for &inp in &node.inputs {
                    if live_ids.contains(&inp) {
                        *in_degree.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }

        // Use a deterministic ordering: sort by NodeId
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort();

        let mut order = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id);
            // Find nodes whose in-degree decreases
            for &other in &live_ids {
                if let Some(node) = self.node(other) {
                    if node.inputs.contains(&id) {
                        let deg = in_degree.get_mut(&other).unwrap();
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            // Only add once, when reaching 0
                            // Need to check it's not already in order
                            queue.push(other);
                            queue.sort();
                        }
                    }
                }
            }
        }
        order
    }

    /// Execute the graph, returning computed values for each node.
    /// This is a simple interpreter for benchmarking and testing.
    pub fn execute(&self, inputs: &HashMap<NodeId, Vec<f32>>) -> HashMap<NodeId, Vec<f32>> {
        let order = self.topological_order();
        let mut values: HashMap<NodeId, Vec<f32>> = inputs.clone();

        for id in order {
            let node = match self.node(id) {
                Some(n) => n,
                None => continue,
            };

            if values.contains_key(&id) {
                continue;
            }

            match &node.kind {
                NodeKind::Constant(data) => {
                    values.insert(id, data.clone());
                }
                NodeKind::Parameter(_) | NodeKind::Input(_) => {
                    // Should already be in inputs map
                    if !values.contains_key(&id) {
                        values.insert(id, vec![0.0; node.meta.numel()]);
                    }
                }
                NodeKind::Op(op) => {
                    let result = execute_op(op, &node.inputs, &values, &node.meta);
                    values.insert(id, result);
                }
            }
        }
        values
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a single operation given its inputs.
fn execute_op(
    op: &OpKind,
    input_ids: &[NodeId],
    values: &HashMap<NodeId, Vec<f32>>,
    meta: &NodeMeta,
) -> Vec<f32> {
    match op {
        OpKind::Add => {
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        }
        OpKind::Sub => {
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
        }
        OpKind::Mul => {
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
        }
        OpKind::Div => {
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
        }
        OpKind::Neg => {
            let a = &values[&input_ids[0]];
            a.iter().map(|x| -x).collect()
        }
        OpKind::Sqrt => {
            let a = &values[&input_ids[0]];
            a.iter().map(|x| x.sqrt()).collect()
        }
        OpKind::Relu => {
            let a = &values[&input_ids[0]];
            a.iter().map(|x| x.max(0.0)).collect()
        }
        OpKind::Sigmoid => {
            let a = &values[&input_ids[0]];
            a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
        }
        OpKind::Tanh => {
            let a = &values[&input_ids[0]];
            a.iter().map(|x| x.tanh()).collect()
        }
        OpKind::MatMul => {
            // input_ids[0] = A [M x K], input_ids[1] = B [K x N]
            // output = [M x N]
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            let m = meta.shape[0];
            let n = meta.shape[1];
            let k = a.len() / m; // infer K from A
            matmul_f32(a, b, m, k, n)
        }
        OpKind::FusedMatMulBiasRelu => {
            // input_ids[0] = A [M x K], input_ids[1] = B [K x N], input_ids[2] = bias [N]
            let a = &values[&input_ids[0]];
            let b = &values[&input_ids[1]];
            let bias = &values[&input_ids[2]];
            let m = meta.shape[0];
            let n = meta.shape[1];
            let k = a.len() / m;
            fused_matmul_bias_relu_f32(a, b, bias, m, k, n)
        }
        OpKind::Conv2d => {
            // Simplified: treat as matmul-like for testing
            let a = &values[&input_ids[0]];
            let w = &values[&input_ids[1]];
            a.iter().zip(w.iter().cycle()).map(|(x, y)| x * y).take(meta.numel()).collect()
        }
        OpKind::BatchNorm => {
            // input_ids: [input, gamma, beta, mean, var]
            // y = gamma * (x - mean) / sqrt(var + eps) + beta
            let x = &values[&input_ids[0]];
            let gamma = &values[&input_ids[1]];
            let beta = &values[&input_ids[2]];
            let mean = &values[&input_ids[3]];
            let var = &values[&input_ids[4]];
            let eps = 1e-5_f32;
            let channels = gamma.len();
            let spatial = x.len() / channels;
            let mut out = vec![0.0f32; x.len()];
            for c in 0..channels {
                let inv_std = 1.0 / (var[c] + eps).sqrt();
                for s in 0..spatial {
                    let idx = c * spatial + s;
                    if idx < x.len() {
                        out[idx] = gamma[c] * (x[idx] - mean[c]) * inv_std + beta[c];
                    }
                }
            }
            out
        }
        OpKind::FusedConvBatchNorm => {
            // input_ids: [input, fused_weight, fused_bias]
            let x = &values[&input_ids[0]];
            let w = &values[&input_ids[1]];
            let bias = &values[&input_ids[2]];
            let channels = bias.len();
            let spatial = x.len() / channels;
            let mut out = vec![0.0f32; x.len()];
            for c in 0..channels {
                for s in 0..spatial {
                    let idx = c * spatial + s;
                    if idx < x.len() && idx < w.len() {
                        out[idx] = x[idx] * w[idx] + bias[c];
                    }
                }
            }
            out
        }
        OpKind::FusedElementWise(ops) => {
            // Apply chain of element-wise ops
            let mut current = values[&input_ids[0]].clone();
            let mut input_idx = 1;
            for sub_op in ops {
                match sub_op {
                    OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
                        let other = &values[&input_ids[input_idx]];
                        input_idx += 1;
                        current = match sub_op {
                            OpKind::Add => current.iter().zip(other).map(|(a, b)| a + b).collect(),
                            OpKind::Sub => current.iter().zip(other).map(|(a, b)| a - b).collect(),
                            OpKind::Mul => current.iter().zip(other).map(|(a, b)| a * b).collect(),
                            OpKind::Div => current.iter().zip(other).map(|(a, b)| a / b).collect(),
                            _ => unreachable!(),
                        };
                    }
                    OpKind::Neg => {
                        current = current.iter().map(|x| -x).collect();
                    }
                    OpKind::Sqrt => {
                        current = current.iter().map(|x| x.sqrt()).collect();
                    }
                    OpKind::Relu => {
                        current = current.iter().map(|x| x.max(0.0)).collect();
                    }
                    OpKind::Sigmoid => {
                        current = current.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                    }
                    OpKind::Tanh => {
                        current = current.iter().map(|x| x.tanh()).collect();
                    }
                    _ => {}
                }
            }
            current
        }
    }
}

/// Simple row-major matrix multiply: C[M x N] = A[M x K] * B[K x N].
fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

/// Fused matmul + bias + relu: computes C = relu(A*B + bias) in a single kernel.
/// Applies bias+relu per row immediately after that row's matmul completes,
/// avoiding intermediate allocations entirely.
fn fused_matmul_bias_relu_f32(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
        // Fuse bias + relu while row i data is cache-hot
        let row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            row[j] = (row[j] + bias[j]).max(0.0);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic() {
        let mut g = Graph::new();
        let x = g.add_node(
            NodeKind::Input("x".into()),
            vec![],
            NodeMeta::new(vec![2, 3], DType::F32),
            "x",
        );
        let w = g.add_node(
            NodeKind::Parameter("w".into()),
            vec![],
            NodeMeta::new(vec![3, 4], DType::F32),
            "w",
        );
        let mm = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![x, w],
            NodeMeta::new(vec![2, 4], DType::F32),
            "matmul",
        );
        g.mark_output(mm);

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.outputs(), &[mm]);
        assert_eq!(g.users(x), vec![mm]);
        assert_eq!(g.users(w), vec![mm]);
    }

    #[test]
    fn test_topological_order() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![4], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![4], DType::F32), "b");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![4], DType::F32), "add");
        let relu = g.add_node(NodeKind::Op(OpKind::Relu), vec![add], NodeMeta::new(vec![4], DType::F32), "relu");
        g.mark_output(relu);

        let order = g.topological_order();
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a) < pos(add));
        assert!(pos(b) < pos(add));
        assert!(pos(add) < pos(relu));
    }

    #[test]
    fn test_execute_add() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![3], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![3], DType::F32), "b");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![3], DType::F32), "add");
        g.mark_output(add);

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0]);
        inputs.insert(b, vec![4.0, 5.0, 6.0]);
        let result = g.execute(&inputs);
        assert_eq!(result[&add], vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_replace_all_uses() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Input("a".into()), vec![], NodeMeta::new(vec![4], DType::F32), "a");
        let b = g.add_node(NodeKind::Input("b".into()), vec![], NodeMeta::new(vec![4], DType::F32), "b");
        let add = g.add_node(NodeKind::Op(OpKind::Add), vec![a, b], NodeMeta::new(vec![4], DType::F32), "add");
        g.mark_output(add);

        let c = g.add_node(NodeKind::Input("c".into()), vec![], NodeMeta::new(vec![4], DType::F32), "c");
        g.replace_all_uses(a, c);

        assert_eq!(g.node(add).unwrap().inputs[0], c);
    }
}
