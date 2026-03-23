use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether gradient tracking is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|f| f.get())
}

/// RAII guard that disables gradient tracking for its lifetime.
///
/// Supports nesting: restores the previous state on drop.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = GRAD_ENABLED.with(|f| {
            let p = f.get();
            f.set(false);
            p
        });
        NoGradGuard { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|f| f.set(self.prev));
    }
}
