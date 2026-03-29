/// State for a single DeltaNet (linear attention) layer.
///
/// The memory state S has shape [num_heads, head_dim, head_dim].
/// Unlike KV cache, this does NOT grow with sequence length.
pub struct DeltaNetLayerState {
    /// Memory state: [num_heads * head_dim * head_dim]
    pub memory: Vec<f32>,
    /// Conv1d rolling buffers for Q, K, V: each [num_heads * head_dim, conv_kernel]
    pub q_conv_state: Vec<f32>,
    pub k_conv_state: Vec<f32>,
    pub v_conv_state: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
}

impl DeltaNetLayerState {
    pub fn new(num_heads: usize, head_dim: usize, conv_kernel: usize) -> Self {
        DeltaNetLayerState {
            memory: vec![0.0; num_heads * head_dim * head_dim],
            q_conv_state: vec![0.0; num_heads * head_dim * conv_kernel],
            k_conv_state: vec![0.0; num_heads * head_dim * conv_kernel],
            v_conv_state: vec![0.0; num_heads * head_dim * conv_kernel],
            num_heads,
            head_dim,
            conv_kernel,
        }
    }

    pub fn reset(&mut self) {
        self.memory.fill(0.0);
        self.q_conv_state.fill(0.0);
        self.k_conv_state.fill(0.0);
        self.v_conv_state.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        (self.memory.len()
            + self.q_conv_state.len()
            + self.k_conv_state.len()
            + self.v_conv_state.len())
            * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_correct_sizes() {
        let num_heads = 4;
        let head_dim = 8;
        let conv_kernel = 4;
        let state = DeltaNetLayerState::new(num_heads, head_dim, conv_kernel);

        assert_eq!(state.memory.len(), num_heads * head_dim * head_dim);
        assert_eq!(
            state.q_conv_state.len(),
            num_heads * head_dim * conv_kernel
        );
        assert_eq!(
            state.k_conv_state.len(),
            num_heads * head_dim * conv_kernel
        );
        assert_eq!(
            state.v_conv_state.len(),
            num_heads * head_dim * conv_kernel
        );
        assert_eq!(state.num_heads, num_heads);
        assert_eq!(state.head_dim, head_dim);
        assert_eq!(state.conv_kernel, conv_kernel);

        // All buffers should be initialised to zero
        assert!(state.memory.iter().all(|&v| v == 0.0));
        assert!(state.q_conv_state.iter().all(|&v| v == 0.0));
        assert!(state.k_conv_state.iter().all(|&v| v == 0.0));
        assert!(state.v_conv_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_clears_all_buffers() {
        let mut state = DeltaNetLayerState::new(2, 4, 4);

        // Dirty all buffers
        state.memory[0] = 1.5;
        state.q_conv_state[0] = 2.5;
        state.k_conv_state[0] = 3.5;
        state.v_conv_state[0] = 4.5;

        state.reset();

        assert!(state.memory.iter().all(|&v| v == 0.0), "memory not cleared");
        assert!(
            state.q_conv_state.iter().all(|&v| v == 0.0),
            "q_conv_state not cleared"
        );
        assert!(
            state.k_conv_state.iter().all(|&v| v == 0.0),
            "k_conv_state not cleared"
        );
        assert!(
            state.v_conv_state.iter().all(|&v| v == 0.0),
            "v_conv_state not cleared"
        );
    }

    #[test]
    fn memory_bytes_is_correct() {
        let num_heads = 2;
        let head_dim = 4;
        let conv_kernel = 4;
        let state = DeltaNetLayerState::new(num_heads, head_dim, conv_kernel);

        // memory:        2 * 4 * 4 = 32 elements
        // q/k/v_conv:    2 * 4 * 4 = 32 elements each  => 3 * 32 = 96 elements
        // total:         32 + 96 = 128 elements * 4 bytes = 512 bytes
        let expected = (num_heads * head_dim * head_dim
            + 3 * num_heads * head_dim * conv_kernel)
            * 4;
        assert_eq!(state.memory_bytes(), expected);
    }
}
