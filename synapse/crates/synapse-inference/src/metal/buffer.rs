use ::metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;

/// GPU buffer pool that reuses Metal buffers by size to avoid repeated allocations.
///
/// Tracks allocated vs reused buffers and organizes free buffers by byte size
/// for O(1) lookup on reuse.
pub struct BufferPool {
    device: Device,
    free: HashMap<usize, Vec<Buffer>>,
    /// Cache of pre-transposed weight buffers keyed by source data pointer.
    /// Weights never change during inference, so we transpose once and reuse.
    weight_cache: HashMap<usize, Buffer>,
    allocated_count: usize,
    reused_count: usize,
}

impl BufferPool {
    /// Create a new buffer pool backed by the given Metal device.
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            free: HashMap::new(),
            weight_cache: HashMap::new(),
            allocated_count: 0,
            reused_count: 0,
        }
    }

    /// Look up or create a cached pre-transposed weight buffer.
    ///
    /// On first call for a given weight pointer, transposes [n,k] → [k,n],
    /// uploads to GPU, and caches. Subsequent calls return the cached buffer.
    pub fn get_or_create_transposed_weight(
        &mut self,
        b: &[f32],
        n: usize,
        k: usize,
    ) -> &Buffer {
        let key = b.as_ptr() as usize;
        self.weight_cache.entry(key).or_insert_with(|| {
            // Transpose [n, k] → [k, n]
            let mut bt = vec![0.0f32; k * n];
            for i in 0..n {
                for j in 0..k {
                    bt[j * n + i] = b[i * k + j];
                }
            }
            self.allocated_count += 1;
            self.device.new_buffer_with_data(
                bt.as_ptr() as *const _,
                (bt.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
    }

    /// Get a reference to a previously cached transposed weight buffer.
    pub fn get_cached_weight(&self, data_ptr: usize) -> Option<&Buffer> {
        self.weight_cache.get(&data_ptr)
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
