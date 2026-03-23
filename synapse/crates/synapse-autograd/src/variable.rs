use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::Tensor;

pub type VariableId = usize;

static NEXT_VARIABLE_ID: AtomicUsize = AtomicUsize::new(0);

fn next_variable_id() -> VariableId {
    NEXT_VARIABLE_ID.fetch_add(1, Ordering::SeqCst)
}

/// A variable in the computation graph, wrapping a tensor with optional gradient.
#[derive(Clone, Debug)]
pub struct Variable {
    pub id: VariableId,
    pub data: Tensor,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    /// Index of this variable's node in the computation graph (None for untracked).
    pub(crate) node_idx: Option<usize>,
}

impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        Variable {
            id: next_variable_id(),
            data,
            requires_grad,
            grad: None,
            node_idx: None,
        }
    }

    pub(crate) fn with_node(data: Tensor, requires_grad: bool, node_idx: usize) -> Self {
        Variable {
            id: next_variable_id(),
            data,
            requires_grad,
            grad: None,
            node_idx: Some(node_idx),
        }
    }
}
