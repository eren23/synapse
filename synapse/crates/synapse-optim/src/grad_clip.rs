use crate::optimizer::Param;

/// Clips gradient of a set of parameters by total norm in-place.
///
/// Returns the total norm of all gradients before clipping.
///
/// Matches PyTorch's `torch.nn.utils.clip_grad_norm_` semantics:
/// - Computes the total L2 norm of all gradients concatenated
/// - If total_norm > max_norm, scales all gradients by max_norm / total_norm
///
/// The `norm_type` parameter specifies the type of norm (2.0 for L2, `f32::INFINITY` for max).
pub fn clip_grad_norm_(params: &mut [Param], max_norm: f32, norm_type: f32) -> f32 {
    let total_norm = compute_total_norm(params, norm_type);

    if total_norm > max_norm && total_norm > 0.0 {
        let clip_coef = max_norm / total_norm;
        for p in params.iter_mut() {
            if let Some(ref mut g) = p.grad {
                for v in g.iter_mut() {
                    *v *= clip_coef;
                }
            }
        }
    }

    total_norm
}

/// Clips gradient values of a set of parameters in-place.
///
/// Matches PyTorch's `torch.nn.utils.clip_grad_value_` semantics:
/// clamps each gradient element to `[-clip_value, clip_value]`.
pub fn clip_grad_value_(params: &mut [Param], clip_value: f32) {
    for p in params.iter_mut() {
        if let Some(ref mut g) = p.grad {
            for v in g.iter_mut() {
                *v = v.clamp(-clip_value, clip_value);
            }
        }
    }
}

/// Compute the total norm of all gradients across parameters.
fn compute_total_norm(params: &[Param], norm_type: f32) -> f32 {
    if norm_type == f32::INFINITY {
        // Max norm: largest absolute value across all gradient elements
        params
            .iter()
            .filter_map(|p| p.grad.as_ref())
            .flat_map(|g| g.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max)
    } else if norm_type == f32::NEG_INFINITY {
        // Min abs norm
        params
            .iter()
            .filter_map(|p| p.grad.as_ref())
            .flat_map(|g| g.iter())
            .map(|v| v.abs())
            .fold(f32::INFINITY, f32::min)
    } else {
        // General p-norm: (sum |g_i|^p)^(1/p)
        let sum: f32 = params
            .iter()
            .filter_map(|p| p.grad.as_ref())
            .flat_map(|g| g.iter())
            .map(|v| v.abs().powf(norm_type))
            .sum();
        sum.powf(1.0 / norm_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_grad_norm_l2() {
        // Two params: grad = [3.0, 4.0], total L2 norm = 5.0
        let mut params = vec![
            Param::with_grad(vec![0.0], vec![3.0]),
            Param::with_grad(vec![0.0], vec![4.0]),
        ];

        let total_norm = clip_grad_norm_(&mut params, 2.5, 2.0);
        assert!((total_norm - 5.0).abs() < 1e-6, "total norm = {}", total_norm);

        // After clipping: scale factor = 2.5/5.0 = 0.5
        assert!(
            (params[0].grad.as_ref().unwrap()[0] - 1.5).abs() < 1e-6,
            "got {}",
            params[0].grad.as_ref().unwrap()[0]
        );
        assert!(
            (params[1].grad.as_ref().unwrap()[0] - 2.0).abs() < 1e-6,
            "got {}",
            params[1].grad.as_ref().unwrap()[0]
        );

        // Verify new total norm
        let new_norm = compute_total_norm(&params, 2.0);
        assert!(
            (new_norm - 2.5).abs() < 1e-6,
            "new norm = {}",
            new_norm
        );
    }

    #[test]
    fn test_clip_grad_norm_no_clip_needed() {
        let mut params = vec![Param::with_grad(vec![0.0, 0.0], vec![1.0, 1.0])];
        let total_norm = clip_grad_norm_(&mut params, 10.0, 2.0);

        // norm = sqrt(2) ≈ 1.414, max_norm = 10.0, no clipping
        assert!((total_norm - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((params[0].grad.as_ref().unwrap()[0] - 1.0).abs() < 1e-7);
        assert!((params[0].grad.as_ref().unwrap()[1] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_clip_grad_norm_max_norm_type() {
        let mut params = vec![Param::with_grad(vec![0.0, 0.0], vec![3.0, -5.0])];
        let total_norm = clip_grad_norm_(&mut params, 2.0, f32::INFINITY);

        // max norm = 5.0, clip to 2.0: scale = 2.0/5.0 = 0.4
        assert!((total_norm - 5.0).abs() < 1e-6);
        assert!(
            (params[0].grad.as_ref().unwrap()[0] - 1.2).abs() < 1e-6,
            "got {}",
            params[0].grad.as_ref().unwrap()[0]
        );
        assert!(
            (params[0].grad.as_ref().unwrap()[1] - (-2.0)).abs() < 1e-6,
            "got {}",
            params[0].grad.as_ref().unwrap()[1]
        );
    }

    #[test]
    fn test_clip_grad_norm_multi_param() {
        // Multiple params with various sizes
        let mut params = vec![
            Param::with_grad(vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 2.0]),
            Param::with_grad(vec![0.0, 0.0], vec![3.0, 4.0]),
        ];
        // L2 norm = sqrt(1+4+4+9+16) = sqrt(34) ≈ 5.831
        let total_norm = clip_grad_norm_(&mut params, 1.0, 2.0);
        assert!((total_norm - 34.0_f32.sqrt()).abs() < 1e-5);

        // After clipping, total norm should be ≈ 1.0
        let new_norm = compute_total_norm(&params, 2.0);
        assert!(
            (new_norm - 1.0).abs() < 1e-5,
            "new norm = {}",
            new_norm
        );
    }

    #[test]
    fn test_clip_grad_value() {
        let mut params = vec![Param::with_grad(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![5.0, -3.0, 0.5, -0.1],
        )];

        clip_grad_value_(&mut params, 1.0);
        let g = params[0].grad.as_ref().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-7);
        assert!((g[1] - (-1.0)).abs() < 1e-7);
        assert!((g[2] - 0.5).abs() < 1e-7);
        assert!((g[3] - (-0.1)).abs() < 1e-7);
    }

    #[test]
    fn test_clip_grad_value_no_grad() {
        let mut params = vec![Param::new(vec![1.0, 2.0])];
        // Should not panic even when no grad is set
        clip_grad_value_(&mut params, 1.0);
        assert!(params[0].grad.is_none());
    }

    #[test]
    fn test_clip_grad_norm_preserves_direction() {
        let mut params = vec![Param::with_grad(vec![0.0, 0.0], vec![6.0, 8.0])];
        // norm = 10.0, clip to 5.0: scale = 0.5
        clip_grad_norm_(&mut params, 5.0, 2.0);
        let g = params[0].grad.as_ref().unwrap();
        // Direction should be preserved: ratio g[0]/g[1] = 6/8 = 0.75
        assert!((g[0] / g[1] - 0.75).abs() < 1e-6);
    }
}
