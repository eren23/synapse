//! Noise schedulers for the diffusion denoising process.
//!
//! A scheduler controls how noise is added and removed across timesteps.
//! Different schedulers trade off quality vs speed:
//! - DDPM: original, 1000 steps, high quality
//! - DDIM: deterministic, 20-50 steps, faster
//! - Euler/DPM: modern, 20-30 steps, best quality/speed

/// Trait for noise schedulers.
pub trait NoiseScheduler {
    /// Total number of training timesteps.
    fn num_train_timesteps(&self) -> usize;

    /// Generate the inference timestep schedule.
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Get the current timestep schedule.
    fn timesteps(&self) -> &[usize];

    /// Compute the previous sample given current sample + model prediction.
    fn step(
        &self,
        model_output: &[f32],
        timestep: usize,
        sample: &[f32],
    ) -> Vec<f32>;

    /// Add noise to clean samples at a given timestep.
    fn add_noise(
        &self,
        original: &[f32],
        noise: &[f32],
        timestep: usize,
    ) -> Vec<f32>;
}

/// DDPM (Denoising Diffusion Probabilistic Models) scheduler.
///
/// The original diffusion scheduler from Ho et al. 2020.
/// Uses 1000 timesteps with linear beta schedule.
pub struct DDPMScheduler {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    timesteps: Vec<usize>,
}

impl DDPMScheduler {
    pub fn new(num_train_timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        Self {
            num_train_timesteps,
            beta_start,
            beta_end,
            timesteps: Vec::new(),
        }
    }
}

impl NoiseScheduler for DDPMScheduler {
    fn num_train_timesteps(&self) -> usize {
        self.num_train_timesteps
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect();
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(&self, _model_output: &[f32], _timestep: usize, _sample: &[f32]) -> Vec<f32> {
        todo!("DDPM step not yet implemented")
    }

    fn add_noise(&self, _original: &[f32], _noise: &[f32], _timestep: usize) -> Vec<f32> {
        todo!("DDPM add_noise not yet implemented")
    }
}

/// DDIM (Denoising Diffusion Implicit Models) scheduler.
///
/// Deterministic variant of DDPM that allows fewer inference steps (20-50).
pub struct DDIMScheduler {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub eta: f64,
    timesteps: Vec<usize>,
}

impl DDIMScheduler {
    pub fn new(num_train_timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        Self {
            num_train_timesteps,
            beta_start,
            beta_end,
            eta: 0.0,
            timesteps: Vec::new(),
        }
    }
}

impl NoiseScheduler for DDIMScheduler {
    fn num_train_timesteps(&self) -> usize {
        self.num_train_timesteps
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect();
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(&self, _model_output: &[f32], _timestep: usize, _sample: &[f32]) -> Vec<f32> {
        todo!("DDIM step not yet implemented")
    }

    fn add_noise(&self, _original: &[f32], _noise: &[f32], _timestep: usize) -> Vec<f32> {
        todo!("DDIM add_noise not yet implemented")
    }
}
