//! Diffusion model support.
//!
//! Two flavours:
//!
//! ## Image diffusion (UNet-based)
//! Scaffolding for image generation via Stable Diffusion, SDXL, Flux, etc.
//! Types compile and are importable but forward methods return `todo!()`.
//!
//! ## Diffusion LLM (text)
//! Non-autoregressive text generation via iterative denoising of a masked
//! token sequence. A bidirectional transformer predicts all tokens in
//! parallel, and the most confident predictions are unmasked each step.

// ── Image diffusion (stubs) ──────────────────────────────────────────
pub mod pipeline;
pub mod scheduler;
pub mod unet;

pub use pipeline::DiffusionPipeline;
pub use scheduler::{DDIMScheduler, DDPMScheduler, NoiseScheduler};
pub use unet::UNet;

// ── Diffusion LLM ────────────────────────────────────────────────────
pub mod config;
pub mod schedule;
pub mod model;

pub use config::DiffusionLLMConfig;
pub use schedule::{MaskSchedule, unmask_by_confidence};
pub use model::DiffusionModel;
