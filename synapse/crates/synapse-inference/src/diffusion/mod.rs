//! Diffusion model support (stub — not yet implemented).
//!
//! This module provides the scaffolding for image generation via diffusion
//! models (Stable Diffusion, SDXL, Flux, etc.). The architecture follows
//! the same pattern as the LLM inference pipeline:
//!
//! 1. Load model weights (UNet, VAE, text encoder)
//! 2. Configure a noise scheduler (DDPM, DDIM, Euler, etc.)
//! 3. Run the denoising loop via `DiffusionPipeline`
//!
//! All types compile and are importable but forward methods return `todo!()`.

pub mod pipeline;
pub mod scheduler;
pub mod unet;

pub use pipeline::DiffusionPipeline;
pub use scheduler::{DDIMScheduler, DDPMScheduler, NoiseScheduler};
pub use unet::UNet;
