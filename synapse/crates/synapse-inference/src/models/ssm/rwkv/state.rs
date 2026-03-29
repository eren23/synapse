/// Per-layer recurrent state for a single RWKV block.
///
/// Holds the WKV accumulator state of shape `[num_heads, head_size, head_size]`
/// and the previous token vectors for token shift in time mixing and channel mixing.
pub struct RwkvLayerState {
    /// WKV numerator accumulator: `[num_heads * head_size * head_size]` (row-major).
    pub wkv_state: Vec<f32>,
    /// Previous token for token shift in time mixing: `[hidden_size]`.
    pub time_mix_prev: Vec<f32>,
    /// Previous token for token shift in channel mixing: `[hidden_size]`.
    pub channel_mix_prev: Vec<f32>,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_size: usize,
}

impl RwkvLayerState {
    pub fn new(hidden_size: usize, num_heads: usize, head_size: usize) -> Self {
        RwkvLayerState {
            wkv_state: vec![0.0f32; num_heads * head_size * head_size],
            time_mix_prev: vec![0.0f32; hidden_size],
            channel_mix_prev: vec![0.0f32; hidden_size],
            hidden_size,
            num_heads,
            head_size,
        }
    }

    /// Zero out all state buffers.
    pub fn reset(&mut self) {
        self.wkv_state.iter_mut().for_each(|v| *v = 0.0);
        self.time_mix_prev.iter_mut().for_each(|v| *v = 0.0);
        self.channel_mix_prev.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Aggregated recurrent state for an entire RWKV model (all layers).
pub struct RwkvState {
    pub layers: Vec<RwkvLayerState>,
    position: usize,
}

impl RwkvState {
    pub fn new(num_layers: usize, hidden_size: usize, num_heads: usize, head_size: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| RwkvLayerState::new(hidden_size, num_heads, head_size))
            .collect();
        RwkvState {
            layers,
            position: 0,
        }
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
            .map(|l| {
                (l.wkv_state.len() + l.time_mix_prev.len() + l.channel_mix_prev.len())
                    * std::mem::size_of::<f32>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_state_initialised_to_zero() {
        let s = RwkvLayerState::new(64, 2, 32);
        assert!(s.wkv_state.iter().all(|&v| v == 0.0));
        assert!(s.time_mix_prev.iter().all(|&v| v == 0.0));
        assert!(s.channel_mix_prev.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn layer_state_sizes_are_correct() {
        let s = RwkvLayerState::new(64, 2, 32);
        assert_eq!(s.wkv_state.len(), 2 * 32 * 32);
        assert_eq!(s.time_mix_prev.len(), 64);
        assert_eq!(s.channel_mix_prev.len(), 64);
    }

    #[test]
    fn layer_state_reset_clears_values() {
        let mut s = RwkvLayerState::new(64, 2, 32);
        s.wkv_state[0] = 1.0;
        s.time_mix_prev[0] = 2.0;
        s.channel_mix_prev[0] = 3.0;
        s.reset();
        assert!(s.wkv_state.iter().all(|&v| v == 0.0));
        assert!(s.time_mix_prev.iter().all(|&v| v == 0.0));
        assert!(s.channel_mix_prev.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rwkv_state_position_tracking() {
        let mut rs = RwkvState::new(2, 64, 2, 32);
        assert_eq!(rs.position(), 0);
        rs.advance(5);
        assert_eq!(rs.position(), 5);
        rs.advance(3);
        assert_eq!(rs.position(), 8);
        rs.reset();
        assert_eq!(rs.position(), 0);
    }

    #[test]
    fn rwkv_state_memory_bytes_is_nonzero() {
        let rs = RwkvState::new(2, 64, 2, 32);
        // 2 layers * (2*32*32 + 64 + 64) * 4 bytes = 2 * (2048 + 128) * 4 = 17408
        let expected = 2 * (2 * 32 * 32 + 64 + 64) * 4;
        assert_eq!(rs.memory_bytes(), expected);
    }

    #[test]
    fn rwkv_state_reset_clears_all_layers() {
        let mut rs = RwkvState::new(3, 64, 2, 32);
        for layer in &mut rs.layers {
            layer.wkv_state[0] = 99.0;
            layer.time_mix_prev[0] = 88.0;
        }
        rs.reset();
        for layer in &rs.layers {
            assert!(layer.wkv_state.iter().all(|&v| v == 0.0));
            assert!(layer.time_mix_prev.iter().all(|&v| v == 0.0));
        }
    }
}
