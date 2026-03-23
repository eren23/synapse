use crate::ir::Graph;

/// Trait for optimization passes that transform a graph.
pub trait OptimizationPass {
    /// A human-readable name for this pass.
    fn name(&self) -> &str;

    /// Apply the pass to the graph, returning true if the graph was modified.
    fn run(&self, graph: &mut Graph) -> bool;
}

/// Run a sequence of passes on a graph, returning the names of passes that modified it.
pub fn run_passes(graph: &mut Graph, passes: &[Box<dyn OptimizationPass>]) -> Vec<String> {
    let mut applied = Vec::new();
    for pass in passes {
        if pass.run(graph) {
            applied.push(pass.name().to_string());
        }
    }
    applied
}

/// Run passes repeatedly until no pass modifies the graph (fixed-point iteration).
pub fn run_passes_to_fixpoint(graph: &mut Graph, passes: &[Box<dyn OptimizationPass>]) -> usize {
    let mut total_iterations = 0;
    loop {
        let mut changed = false;
        for pass in passes {
            if pass.run(graph) {
                changed = true;
            }
        }
        total_iterations += 1;
        if !changed {
            break;
        }
    }
    total_iterations
}
