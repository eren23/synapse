mod buffer;
mod device;

pub use buffer::BufferPool;
pub use device::{MetalBackend, MetalError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::device::KERNEL_NAMES;

    #[test]
    fn metal_backend_creation() {
        match MetalBackend::new() {
            Ok(backend) => {
                assert!(!backend.device_name().is_empty());
                assert_eq!(backend.pipeline_count(), KERNEL_NAMES.len());
                for name in KERNEL_NAMES {
                    assert!(
                        backend.pipeline(name).is_some(),
                        "Missing pipeline: {name}"
                    );
                }
            }
            Err(MetalError::NoDevice) => {
                assert!(!MetalBackend::is_available());
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn buffer_pool_reuses_matching_size() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(MetalError::NoDevice) => {
                eprintln!("Skipping: no Metal GPU available");
                return;
            }
            Err(e) => panic!("Unexpected error: {e}"),
        };
        let mut pool = BufferPool::new(&backend.device);

        let data = vec![1.0f32; 256];
        let buf = pool.get_or_create(&data);
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.reused_count(), 0);

        pool.release(buf);
        assert_eq!(pool.free_count(), 1);

        // Same size -> reuse
        let _buf2 = pool.get_or_create(&data);
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.reused_count(), 1);
        assert_eq!(pool.free_count(), 0);

        // Different size -> new allocation
        let small = vec![1.0f32; 64];
        let _buf3 = pool.get_or_create(&small);
        assert_eq!(pool.allocated_count(), 2);
        assert_eq!(pool.reused_count(), 1);
    }

    #[test]
    fn no_memory_leak_100_iterations() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(MetalError::NoDevice) => {
                eprintln!("Skipping: no Metal GPU available");
                return;
            }
            Err(e) => panic!("Unexpected error: {e}"),
        };
        let mut pool = BufferPool::new(&backend.device);

        let data = vec![1.0f32; 1024];

        for _ in 0..100 {
            let buf = pool.get_or_create(&data);
            pool.release(buf);
        }

        // Only 1 buffer ever allocated, reused 99 times
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.reused_count(), 99);
        assert_eq!(pool.free_count(), 1);
    }
}
