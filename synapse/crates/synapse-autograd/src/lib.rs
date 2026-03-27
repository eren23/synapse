pub mod backward;
pub mod function;
pub mod grad_check;
pub mod graph;
pub mod no_grad;
pub mod ops;
pub mod tensor;
pub mod variable;

pub use backward::backward;
pub use function::GradFn;
pub use grad_check::grad_check;
pub use graph::Graph;
pub use no_grad::{is_grad_enabled, NoGradGuard};
pub use tensor::Tensor;
pub use variable::{Variable, VariableId};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Linear graph: y = a*b + c ────────────────────────────────────

    #[test]
    fn test_linear_backward() {
        let mut g = Graph::new();
        let a = g.variable(Tensor::scalar(2.0), true);
        let b = g.variable(Tensor::scalar(3.0), true);
        let c = g.variable(Tensor::scalar(1.0), true);

        let ab = g.mul(a, b);
        let y = g.add(ab, c);

        backward(&mut g, y);

        // dy/da = b = 3, dy/db = a = 2, dy/dc = 1
        assert_eq!(g.grad(a).unwrap().data[0], 3.0);
        assert_eq!(g.grad(b).unwrap().data[0], 2.0);
        assert_eq!(g.grad(c).unwrap().data[0], 1.0);
    }

    // ── Fan-out: y = x + x ───────────────────────────────────────────

    #[test]
    fn test_fan_out() {
        let mut g = Graph::new();
        let x = g.variable(Tensor::scalar(5.0), true);
        let y = g.add(x, x);

        backward(&mut g, y);

        // dy/dx = 2
        assert_eq!(g.grad(x).unwrap().data[0], 2.0);
    }

    // ── Diamond: z = (x+y)*(x-y) = x² - y² ─────────────────────────

    #[test]
    fn test_diamond() {
        let mut g = Graph::new();
        let x = g.variable(Tensor::scalar(3.0), true);
        let y = g.variable(Tensor::scalar(2.0), true);

        let sum = g.add(x, y);
        let diff = g.sub(x, y);
        let z = g.mul(sum, diff);

        backward(&mut g, z);

        // dz/dx = 2x = 6, dz/dy = -2y = -4
        assert!((g.grad(x).unwrap().data[0] - 6.0).abs() < 1e-6);
        assert!((g.grad(y).unwrap().data[0] - (-4.0)).abs() < 1e-6);
    }

    // ── no_grad prevents graph construction ──────────────────────────

    #[test]
    fn test_no_grad_prevents_graph() {
        let mut g = Graph::new();
        let x = g.variable(Tensor::scalar(2.0), true);

        let nodes_before = g.nodes.len();
        let y;
        {
            let _guard = NoGradGuard::new();
            y = g.add(x, x);
        }

        // Forward still works.
        assert_eq!(g.data(y).data[0], 4.0);
        // No new node was added.
        assert_eq!(g.nodes.len(), nodes_before);
        // Output variable is untracked.
        assert!(g.variables[&y].node_idx.is_none());
    }

    #[test]
    fn test_no_grad_restores() {
        assert!(is_grad_enabled());
        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }

    // ── grad_check: numerical vs analytical ──────────────────────────

    #[test]
    fn test_grad_check_linear() {
        let inputs = vec![
            Tensor::scalar(2.0),
            Tensor::scalar(3.0),
            Tensor::scalar(1.0),
        ];

        let pass = grad_check(
            |g, vars| {
                let ab = g.mul(vars[0], vars[1]);
                g.add(ab, vars[2])
            },
            &inputs,
            1e-2,
            1e-4,
        );
        assert!(pass, "grad_check failed for linear graph");
    }

    #[test]
    fn test_grad_check_diamond() {
        let inputs = vec![Tensor::scalar(3.0), Tensor::scalar(2.0)];

        let pass = grad_check(
            |g, vars| {
                let sum = g.add(vars[0], vars[1]);
                let diff = g.sub(vars[0], vars[1]);
                g.mul(sum, diff)
            },
            &inputs,
            1e-2,
            1e-4,
        );
        assert!(pass, "grad_check failed for diamond graph");
    }

    #[test]
    fn test_grad_check_fan_out() {
        let inputs = vec![Tensor::scalar(4.0)];

        let pass = grad_check(|g, vars| g.add(vars[0], vars[0]), &inputs, 1e-2, 1e-4);
        assert!(pass, "grad_check failed for fan-out graph");
    }

    // ── Multi-element tensors ────────────────────────────────────────

    #[test]
    fn test_multi_element_backward() {
        let mut g = Graph::new();
        let a = g.variable(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]), true);
        let b = g.variable(Tensor::new(vec![4.0, 5.0, 6.0], vec![3]), true);

        let y = g.mul(a, b); // [4, 10, 18]
        backward(&mut g, y);

        // Seeded with ones → equivalent to sum(a*b)
        // d/da = b, d/db = a
        assert_eq!(g.grad(a).unwrap().data, vec![4.0, 5.0, 6.0]);
        assert_eq!(g.grad(b).unwrap().data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_grad_check_multi_element() {
        let inputs = vec![
            Tensor::new(vec![1.0, 2.0], vec![2]),
            Tensor::new(vec![3.0, 4.0], vec![2]),
        ];

        let pass = grad_check(|g, vars| g.mul(vars[0], vars[1]), &inputs, 1e-2, 1e-4);
        assert!(pass, "grad_check failed for multi-element tensors");
    }
}
