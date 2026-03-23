/// Per-channel min/max calibration.
///
/// Computes `scale[ch] = max(|w[ch, :]|) / 127` for each output channel.
pub struct MinMaxCalibration;

impl MinMaxCalibration {
    /// Compute per-channel scale factors from weight values.
    ///
    /// `weights` is `[channels, channel_size]` row-major.
    /// Returns one scale per channel.
    pub fn compute_scales(
        weights: &[f32],
        channels: usize,
        channel_size: usize,
    ) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            channels * channel_size,
            "weights length must equal channels * channel_size"
        );
        let mut scales = Vec::with_capacity(channels);
        for ch in 0..channels {
            let row = &weights[ch * channel_size..(ch + 1) * channel_size];
            let max_abs = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            scales.push(if max_abs == 0.0 {
                1.0
            } else {
                max_abs / 127.0
            });
        }
        scales
    }
}

/// Percentile-based calibration that clips outlier channels.
///
/// Instead of using the absolute maximum (which may be a single outlier),
/// uses the value at the given percentile of absolute values per channel.
pub struct PercentileCalibration {
    pub percentile: f32,
}

impl PercentileCalibration {
    pub fn new(percentile: f32) -> Self {
        assert!(
            (0.0..=100.0).contains(&percentile),
            "percentile must be in [0, 100]"
        );
        PercentileCalibration { percentile }
    }

    /// Compute per-channel scale factors using percentile clipping.
    ///
    /// `weights` is `[channels, channel_size]` row-major.
    pub fn compute_scales(
        &self,
        weights: &[f32],
        channels: usize,
        channel_size: usize,
    ) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            channels * channel_size,
            "weights length must equal channels * channel_size"
        );
        let mut scales = Vec::with_capacity(channels);
        for ch in 0..channels {
            let row = &weights[ch * channel_size..(ch + 1) * channel_size];
            let mut abs_vals: Vec<f32> = row.iter().map(|v| v.abs()).collect();
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let idx = ((self.percentile / 100.0)
                * (abs_vals.len().saturating_sub(1)) as f32)
                .round() as usize;
            let idx = idx.min(abs_vals.len().saturating_sub(1));
            let clip_val = abs_vals[idx];

            scales.push(if clip_val == 0.0 {
                1.0
            } else {
                clip_val / 127.0
            });
        }
        scales
    }
}
