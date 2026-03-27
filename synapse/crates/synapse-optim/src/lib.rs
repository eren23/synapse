pub mod adam;
pub mod grad_clip;
pub mod lr_scheduler;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;

pub use adam::{adamw, Adam};
pub use grad_clip::{clip_grad_norm_, clip_grad_value_};
pub use lr_scheduler::{CosineAnnealingLR, LinearWarmup, ReduceLROnPlateau, StepLR};
pub use optimizer::{Optimizer, Param, ParamGroup, StateDict};
pub use rmsprop::RMSProp;
pub use sgd::SGD;

#[cfg(test)]
mod tests {
    use super::*;

    // ── Integration: SGD + scheduler ────────────────────────────────────

    #[test]
    fn test_sgd_with_step_lr() {
        let mut params = vec![Param::with_grad(vec![1.0], vec![1.0])];
        let mut opt = SGD::new(0.1);
        let mut sched = StepLR::new(0.1, 2, 0.5);

        for epoch in 0..6 {
            params[0].grad = Some(vec![1.0]);
            opt.lr = sched.get_lr();
            opt.step(&mut params);
            if epoch < 5 {
                sched.step();
            }
        }
        // Param should have decreased from 1.0
        assert!(params[0].data[0] < 1.0);
    }

    // ── Integration: Adam + gradient clipping ───────────────────────────

    #[test]
    fn test_adam_with_grad_clipping() {
        let mut params = vec![Param::with_grad(
            vec![1.0, 2.0, 3.0],
            vec![100.0, -200.0, 150.0],
        )];

        // Clip gradients first
        let norm_before = clip_grad_norm_(&mut params, 1.0, 2.0);
        assert!(norm_before > 1.0);

        // Now step with clipped gradients
        let mut opt = Adam::new(0.001);
        opt.step(&mut params);

        // Parameters should have moved only slightly due to clipping
        assert!((params[0].data[0] - 0.999).abs() < 0.01);
    }

    // ── Integration: full training loop simulation ──────────────────────

    #[test]
    fn test_training_loop_sgd_cosine() {
        let n = 100;
        let mut params = vec![Param::new(vec![5.0; n])];
        let mut opt = SGD::new(0.1).momentum(0.9);
        let mut sched = CosineAnnealingLR::new(0.1, 50);

        for _ in 0..50 {
            // Simulate gradient pointing toward zero
            let grad: Vec<f32> = params[0].data.iter().map(|&x| x).collect();
            params[0].grad = Some(grad);

            clip_grad_norm_(&mut params, 5.0, 2.0);

            opt.lr = sched.get_lr();
            opt.step(&mut params);
            opt.zero_grad(&mut params);
            sched.step();
        }

        // Parameters should have moved toward 0
        for &v in &params[0].data {
            assert!(v.abs() < 5.0, "param = {}", v);
        }
    }

    // ── Benchmark: optimizer step on 1M params ──────────────────────────

    #[test]
    fn bench_sgd_step_1m() {
        let n = 1_000_000;
        let data = vec![1.0; n];
        let grad = vec![0.01; n];
        let mut params = vec![Param::with_grad(data, grad)];
        let mut opt = SGD::new(0.01).momentum(0.9).weight_decay(1e-4);

        let start = std::time::Instant::now();
        for _ in 0..10 {
            params[0].grad = Some(vec![0.01; n]);
            opt.step(&mut params);
        }
        let elapsed = start.elapsed();

        // Just verify it completes in reasonable time (< 5s for 10 steps on 1M params)
        assert!(elapsed.as_secs() < 5, "SGD 1M params took {:?}", elapsed);
        eprintln!(
            "bench_sgd_step_1m: 10 steps on {} params in {:?} ({:.2} us/step)",
            n,
            elapsed,
            elapsed.as_micros() as f64 / 10.0,
        );
    }

    #[test]
    fn bench_adam_step_1m() {
        let n = 1_000_000;
        let data = vec![1.0; n];
        let grad = vec![0.01; n];
        let mut params = vec![Param::with_grad(data, grad)];
        let mut opt = Adam::new(0.001).weight_decay(1e-4);

        let start = std::time::Instant::now();
        for _ in 0..10 {
            params[0].grad = Some(vec![0.01; n]);
            opt.step(&mut params);
        }
        let elapsed = start.elapsed();

        assert!(elapsed.as_secs() < 5, "Adam 1M params took {:?}", elapsed);
        eprintln!(
            "bench_adam_step_1m: 10 steps on {} params in {:?} ({:.2} us/step)",
            n,
            elapsed,
            elapsed.as_micros() as f64 / 10.0,
        );
    }
}
