//! Pre-allocated KV-Cache for autoregressive transformer inference.
//!
//! Each cache holds contiguous K and V buffers sized [max_seq, n_kv_heads, head_dim].
//! Tokens are appended one at a time via O(1) memcpy. Slicing returns a zero-copy
//! pointer+length view into the populated region. Reset rewinds the position counter
//! without deallocating.
//!
//! `LayerCaches` manages N independent per-layer caches with a single init/deinit.

const std = @import("std");

/// A single-layer KV-cache with pre-allocated contiguous buffers.
pub const KvCache = struct {
    k_buf: []f32,
    v_buf: []f32,
    max_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Number of tokens currently stored (next write position).
    pos: usize,

    /// Stride in floats for one token row: n_kv_heads * head_dim.
    fn tokenStride(self: *const KvCache) usize {
        return self.n_kv_heads * self.head_dim;
    }

    /// Allocate K/V buffers for one layer.
    ///
    /// Buffers are zeroed on allocation. No further allocations occur
    /// during append/slice/reset.
    pub fn create(allocator: std.mem.Allocator, max_seq: usize, n_kv_heads: usize, head_dim: usize) !KvCache {
        if (max_seq == 0 or n_kv_heads == 0 or head_dim == 0)
            return error.InvalidDimensions;

        const total = max_seq * n_kv_heads * head_dim;
        const k_buf = try allocator.alloc(f32, total);
        errdefer allocator.free(k_buf);
        const v_buf = try allocator.alloc(f32, total);

        @memset(k_buf, 0);
        @memset(v_buf, 0);

        return .{
            .k_buf = k_buf,
            .v_buf = v_buf,
            .max_seq = max_seq,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .pos = 0,
        };
    }

    /// Free K/V buffers.
    pub fn destroy(self: *KvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.k_buf);
        allocator.free(self.v_buf);
        self.* = undefined;
    }

    /// Append new K/V vectors for a single token at the current position.
    ///
    /// `k_token` and `v_token` must each have exactly n_kv_heads * head_dim elements.
    /// O(1) memcpy — no allocation.
    pub fn append(self: *KvCache, k_token: []const f32, v_token: []const f32) !void {
        const stride = self.tokenStride();
        if (k_token.len != stride or v_token.len != stride)
            return error.ShapeMismatch;
        if (self.pos >= self.max_seq)
            return error.CacheFull;

        const offset = self.pos * stride;
        @memcpy(self.k_buf[offset..][0..stride], k_token);
        @memcpy(self.v_buf[offset..][0..stride], v_token);
        self.pos += 1;
    }

    /// Slice result: zero-copy views into the populated region [0..seq_len].
    pub const Slice = struct {
        k: []const f32,
        v: []const f32,
        seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    };

    /// Return a zero-copy view of K/V from position 0 to current seq_len.
    ///
    /// The returned slices point directly into the pre-allocated buffers.
    /// They are valid until the next append or reset call.
    pub fn slice(self: *const KvCache) Slice {
        const len = self.pos * self.tokenStride();
        return .{
            .k = self.k_buf[0..len],
            .v = self.v_buf[0..len],
            .seq_len = self.pos,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
        };
    }

    /// Reset the position counter to 0. No deallocation — buffers are reused.
    pub fn reset(self: *KvCache) void {
        self.pos = 0;
    }
};

/// Per-layer management of N independent KV-caches.
///
/// All caches share the same max_seq, n_kv_heads, head_dim configuration.
pub const LayerCaches = struct {
    caches: []KvCache,
    n_layers: usize,
    allocator: std.mem.Allocator,

    /// Allocate caches for `n_layers` layers, each with identical dimensions.
    pub fn create(allocator: std.mem.Allocator, n_layers: usize, max_seq: usize, n_kv_heads: usize, head_dim: usize) !LayerCaches {
        if (n_layers == 0)
            return error.InvalidDimensions;

        const caches = try allocator.alloc(KvCache, n_layers);
        errdefer allocator.free(caches);

        var initialized: usize = 0;
        errdefer {
            for (caches[0..initialized]) |*c| c.destroy(allocator);
        }

        for (caches) |*c| {
            c.* = try KvCache.create(allocator, max_seq, n_kv_heads, head_dim);
            initialized += 1;
        }

        return .{
            .caches = caches,
            .n_layers = n_layers,
            .allocator = allocator,
        };
    }

    /// Free all layer caches.
    pub fn destroy(self: *LayerCaches) void {
        for (self.caches) |*c| c.destroy(self.allocator);
        self.allocator.free(self.caches);
        self.* = undefined;
    }

    /// Get the cache for a specific layer.
    pub fn get(self: *LayerCaches, layer: usize) *KvCache {
        return &self.caches[layer];
    }

    /// Reset all layer caches (rewind position counters).
    pub fn resetAll(self: *LayerCaches) void {
        for (self.caches) |*c| c.reset();
    }
};
