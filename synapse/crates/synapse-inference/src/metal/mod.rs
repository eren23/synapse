mod buffer;
mod device;
pub mod dispatch;
pub mod gpu_buffers;
pub mod gpu_forward;
pub mod hybrid_gpu_buffers;
pub mod hybrid_gpu_forward;
pub mod lewm_forward;

pub use buffer::BufferPool;
pub use device::{MetalBackend, MetalError};
pub use dispatch::ComputeBackend;
pub use gpu_buffers::MetalModelBuffers;
pub use lewm_forward::MetalLeWMState;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
