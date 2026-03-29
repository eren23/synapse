/// Per-layer recurrent state for a single Mamba block.
///
/// Holds the SSM hidden state `h` of shape `[d_inner, d_state]` and the
/// rolling convolution buffer of shape `[d_inner, d_conv]`.
pub struct MambaLayerState {
    /// SSM hidden state, layout: `[d_inner * d_state]` (row-major).
    pub ssm_state: Vec<f32>,
    /// Rolling convolution buffer, layout: `[d_inner * d_conv]` (row-major).
    pub conv_state: Vec<f32>,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
}

impl MambaLayerState {
    pub fn new(d_inner: usize, d_state: usize, d_conv: usize) -> Self {
        MambaLayerState {
            ssm_state: vec![0.0f32; d_inner * d_state],
            conv_state: vec![0.0f32; d_inner * d_conv],
            d_inner,
            d_state,
            d_conv,
        }
    }

    /// Zero out both state buffers.
    pub fn reset(&mut self) {
        self.ssm_state.iter_mut().for_each(|v| *v = 0.0);
        self.conv_state.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Aggregated recurrent state for an entire Mamba model (all layers).
pub struct RecurrentState {
    pub layers: Vec<MambaLayerState>,
    position: usize,
}

impl RecurrentState {
    pub fn new(num_layers: usize, d_inner: usize, d_state: usize, d_conv: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| MambaLayerState::new(d_inner, d_state, d_conv))
            .collect();
        RecurrentState { layers, position: 0 }
    }

    /// Number of tokens processed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Advance the position counter by `n` tokens.
    pub fn advance(&mut self, n: usize) {
        self.position += n;
    }

    /// Reset all layer states and the position counter.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.position = 0;
    }

    /// Total heap memory used by all state buffers (in bytes).
    pub fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.ssm_state.len() + l.conv_state.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_state_initialised_to_zero() {
        let s = MambaLayerState::new(8, 4, 4);
        assert!(s.ssm_state.iter().all(|&v| v == 0.0));
        assert!(s.conv_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn layer_state_reset_clears_values() {
        let mut s = MambaLayerState::new(4, 2, 2);
        s.ssm_state[0] = 1.0;
        s.conv_state[0] = 2.0;
        s.reset();
        assert!(s.ssm_state.iter().all(|&v| v == 0.0));
        assert!(s.conv_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn recurrent_state_position_tracking() {
        let mut rs = RecurrentState::new(2, 8, 4, 4);
        assert_eq!(rs.position(), 0);
        rs.advance(5);
        assert_eq!(rs.position(), 5);
        rs.advance(3);
        assert_eq!(rs.position(), 8);
        rs.reset();
        assert_eq!(rs.position(), 0);
    }

    #[test]
    fn recurrent_state_memory_bytes_is_nonzero() {
        let rs = RecurrentState::new(2, 8, 4, 4);
        // 2 layers * (8*4 + 8*4) * 4 bytes = 2 * 64 * 4 = 512
        assert_eq!(rs.memory_bytes(), 512);
    }

    #[test]
    fn recurrent_state_reset_clears_all_layers() {
        let mut rs = RecurrentState::new(3, 4, 2, 2);
        for layer in &mut rs.layers {
            layer.ssm_state[0] = 99.0;
        }
        rs.reset();
        for layer in &rs.layers {
            assert!(layer.ssm_state.iter().all(|&v| v == 0.0));
        }
    }
}
