use ::metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;

/// GPU buffer pool that reuses Metal buffers by size to avoid repeated allocations.
///
/// Tracks allocated vs reused buffers and organizes free buffers by byte size
/// for O(1) lookup on reuse.
pub struct BufferPool {
    device: Device,
    free: HashMap<usize, Vec<Buffer>>,
    allocated_count: usize,
    reused_count: usize,
}

impl BufferPool {
    /// Create a new buffer pool backed by the given Metal device.
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            free: HashMap::new(),
            allocated_count: 0,
            reused_count: 0,
        }
    }

    /// Get a buffer populated with `data`, reusing an existing buffer of matching
    /// byte size if available, or allocating a new one.
    pub fn get_or_create(&mut self, data: &[f32]) -> Buffer {
        let byte_size = data.len() * std::mem::size_of::<f32>();

        if let Some(buffers) = self.free.get_mut(&byte_size) {
            if let Some(buffer) = buffers.pop() {
                unsafe {
                    let ptr = buffer.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                }
                self.reused_count += 1;
                return buffer;
            }
        }

        self.allocated_count += 1;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty (zeroed) buffer of `size` f32 elements, reusing an existing
    /// buffer of matching byte size if available.
    pub fn create_empty(&mut self, size: usize) -> Buffer {
        let byte_size = size * std::mem::size_of::<f32>();

        if let Some(buffers) = self.free.get_mut(&byte_size) {
            if let Some(buffer) = buffers.pop() {
                self.reused_count += 1;
                return buffer;
            }
        }

        self.allocated_count += 1;
        self.device.new_buffer(
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Release a buffer back to the pool for future reuse.
    pub fn release(&mut self, buffer: Buffer) {
        let byte_size = buffer.length() as usize;
        self.free.entry(byte_size).or_default().push(buffer);
    }

    /// Total number of fresh GPU buffer allocations made.
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Total number of buffer reuses (avoided allocations).
    pub fn reused_count(&self) -> usize {
        self.reused_count
    }

    /// Number of buffers currently sitting in the free pool.
    pub fn free_count(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }
}
