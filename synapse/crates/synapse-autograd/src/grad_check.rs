use crate::backward::backward;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

/// Check analytical gradients against numerical gradients via central finite differences.
///
/// `build_graph` builds a computation from input variables and returns the scalar output.
/// Returns `true` if all gradients match within `tol`.
pub fn grad_check<F>(build_graph: F, inputs: &[Tensor], eps: f32, tol: f32) -> bool
where
    F: Fn(&mut Graph, &[VariableId]) -> VariableId,
{
    // Compute analytical gradients.
    let mut g = Graph::new();
    let var_ids: Vec<VariableId> = inputs.iter().map(|t| g.variable(t.clone(), true)).collect();
    let output = build_graph(&mut g, &var_ids);
    backward(&mut g, output);

    let analytical_grads: Vec<Option<Tensor>> =
        var_ids.iter().map(|&id| g.grad(id).cloned()).collect();

    // Compare against numerical gradients.
    for (input_idx, input) in inputs.iter().enumerate() {
        let analytical = match &analytical_grads[input_idx] {
            Some(g) => g,
            None => continue,
        };

        for elem_idx in 0..input.numel() {
            let mut plus_input = input.clone();
            plus_input.data[elem_idx] += eps;
            let val_plus = eval_at(&build_graph, inputs, input_idx, &plus_input);

            let mut minus_input = input.clone();
            minus_input.data[elem_idx] -= eps;
            let val_minus = eval_at(&build_graph, inputs, input_idx, &minus_input);

            // Use f64 for the finite difference to minimize rounding error.
            let numerical = (val_plus - val_minus) / (2.0 * eps as f64);
            let anal = analytical.data[elem_idx] as f64;

            if (numerical - anal).abs() > tol as f64 {
                return false;
            }
        }
    }

    true
}

/// Evaluate the graph with one input replaced, returning the output sum as f64.
fn eval_at<F>(
    build_graph: &F,
    inputs: &[Tensor],
    replace_idx: usize,
    replacement: &Tensor,
) -> f64
where
    F: Fn(&mut Graph, &[VariableId]) -> VariableId,
{
    let mut g = Graph::new();
    let var_ids: Vec<VariableId> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i == replace_idx {
                g.variable(replacement.clone(), false)
            } else {
                g.variable(t.clone(), false)
            }
        })
        .collect();
    let output = build_graph(&mut g, &var_ids);
    // Sum in f64 for precision.
    g.data(output).data.iter().map(|&x| x as f64).sum()
}
