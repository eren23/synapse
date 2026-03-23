//! Correctness tests for pre-allocated KV-Cache.
//!
//! Verifies: append/slice round-trip, reset+reuse, no re-allocation after create,
//! per-layer management, edge cases, and memory leak detection.

const std = @import("std");
const testing = std.testing;
const kvcache = @import("kvcache");
const tracking = @import("tracking");

const KvCache = kvcache.KvCache;
const LayerCaches = kvcache.LayerCaches;
const TrackingAllocator = tracking.TrackingAllocator;

// ================================================================
// Helpers
// ================================================================

/// Fill a slice with deterministic values: base + i*0.01
fn fillToken(buf: []f32, base: f32) void {
    for (buf, 0..) |*v, i| {
        v.* = base + @as(f32, @floatFromInt(i)) * 0.01;
    }
}

// ================================================================
// Basic create / append / slice
// ================================================================

test "kvcache: create and initial state" {
    var cache = try KvCache.create(testing.allocator, 16, 4, 32);
    defer cache.destroy(testing.allocator);

    try testing.expectEqual(@as(usize, 16), cache.max_seq);
    try testing.expectEqual(@as(usize, 4), cache.n_kv_heads);
    try testing.expectEqual(@as(usize, 32), cache.head_dim);
    try testing.expectEqual(@as(usize, 0), cache.pos);

    const s = cache.slice();
    try testing.expectEqual(@as(usize, 0), s.seq_len);
    try testing.expectEqual(@as(usize, 0), s.k.len);
    try testing.expectEqual(@as(usize, 0), s.v.len);
}

test "kvcache: append single token and read back via slice" {
    var cache = try KvCache.create(testing.allocator, 8, 2, 4);
    defer cache.destroy(testing.allocator);

    const stride = 2 * 4; // n_kv_heads * head_dim
    var k_tok: [stride]f32 = undefined;
    var v_tok: [stride]f32 = undefined;
    fillToken(&k_tok, 1.0);
    fillToken(&v_tok, 2.0);

    try cache.append(&k_tok, &v_tok);

    try testing.expectEqual(@as(usize, 1), cache.pos);
    const s = cache.slice();
    try testing.expectEqual(@as(usize, 1), s.seq_len);
    try testing.expectEqual(@as(usize, stride), s.k.len);

    // Verify values
    for (0..stride) |i| {
        try testing.expectApproxEqAbs(k_tok[i], s.k[i], 1e-7);
        try testing.expectApproxEqAbs(v_tok[i], s.v[i], 1e-7);
    }
}

test "kvcache: append multiple tokens" {
    var cache = try KvCache.create(testing.allocator, 64, 2, 4);
    defer cache.destroy(testing.allocator);

    const stride = 2 * 4;
    const n_tokens = 32;

    for (0..n_tokens) |t| {
        var k_tok: [stride]f32 = undefined;
        var v_tok: [stride]f32 = undefined;
        fillToken(&k_tok, @as(f32, @floatFromInt(t)));
        fillToken(&v_tok, @as(f32, @floatFromInt(t)) + 100.0);
        try cache.append(&k_tok, &v_tok);
    }

    try testing.expectEqual(@as(usize, n_tokens), cache.pos);
    const s = cache.slice();
    try testing.expectEqual(@as(usize, n_tokens), s.seq_len);
    try testing.expectEqual(@as(usize, n_tokens * stride), s.k.len);

    // Verify each token's data
    for (0..n_tokens) |t| {
        const offset = t * stride;
        for (0..stride) |i| {
            const expected_k = @as(f32, @floatFromInt(t)) + @as(f32, @floatFromInt(i)) * 0.01;
            const expected_v = @as(f32, @floatFromInt(t)) + 100.0 + @as(f32, @floatFromInt(i)) * 0.01;
            try testing.expectApproxEqAbs(expected_k, s.k[offset + i], 1e-5);
            try testing.expectApproxEqAbs(expected_v, s.v[offset + i], 1e-5);
        }
    }
}

// ================================================================
// Zero-copy: slice points into pre-allocated buffer
// ================================================================

test "kvcache: slice is zero-copy (pointers into buffer)" {
    var cache = try KvCache.create(testing.allocator, 8, 2, 4);
    defer cache.destroy(testing.allocator);

    const stride = 2 * 4;
    var k_tok: [stride]f32 = undefined;
    var v_tok: [stride]f32 = undefined;
    fillToken(&k_tok, 1.0);
    fillToken(&v_tok, 2.0);
    try cache.append(&k_tok, &v_tok);

    const s = cache.slice();
    // Slice k must point directly into k_buf
    try testing.expect(s.k.ptr == cache.k_buf.ptr);
    try testing.expect(s.v.ptr == cache.v_buf.ptr);
}

// ================================================================
// Reset and reuse
// ================================================================

test "kvcache: reset clears position, reuse produces same results" {
    var cache = try KvCache.create(testing.allocator, 16, 2, 4);
    defer cache.destroy(testing.allocator);

    const stride = 2 * 4;

    // First pass: append 5 tokens
    for (0..5) |t| {
        var k_tok: [stride]f32 = undefined;
        var v_tok: [stride]f32 = undefined;
        fillToken(&k_tok, @as(f32, @floatFromInt(t)) * 10.0);
        fillToken(&v_tok, @as(f32, @floatFromInt(t)) * 10.0 + 0.5);
        try cache.append(&k_tok, &v_tok);
    }

    // Save first-pass slice data
    const s1 = cache.slice();
    var k_copy: [5 * stride]f32 = undefined;
    var v_copy: [5 * stride]f32 = undefined;
    @memcpy(&k_copy, s1.k);
    @memcpy(&v_copy, s1.v);

    // Reset
    cache.reset();
    try testing.expectEqual(@as(usize, 0), cache.pos);
    const s_empty = cache.slice();
    try testing.expectEqual(@as(usize, 0), s_empty.seq_len);

    // Second pass: same tokens
    for (0..5) |t| {
        var k_tok: [stride]f32 = undefined;
        var v_tok: [stride]f32 = undefined;
        fillToken(&k_tok, @as(f32, @floatFromInt(t)) * 10.0);
        fillToken(&v_tok, @as(f32, @floatFromInt(t)) * 10.0 + 0.5);
        try cache.append(&k_tok, &v_tok);
    }

    // Verify second pass matches first pass
    const s2 = cache.slice();
    try testing.expectEqual(s1.seq_len, s2.seq_len);
    for (0..5 * stride) |i| {
        try testing.expectApproxEqAbs(k_copy[i], s2.k[i], 1e-7);
        try testing.expectApproxEqAbs(v_copy[i], s2.v[i], 1e-7);
    }
}

// ================================================================
// No re-allocation after create (tracking allocator)
// ================================================================

test "kvcache: no allocation after create" {
    var tracker = TrackingAllocator.init(std.heap.page_allocator);
    const alloc = tracker.allocator();

    var cache = try KvCache.create(alloc, 64, 4, 32);
    defer cache.destroy(alloc);

    // Record alloc count after create
    const allocs_after_create = tracker.alloc_count;

    // Append 64 tokens — should NOT allocate
    const stride = 4 * 32;
    var k_tok: [stride]f32 = undefined;
    var v_tok: [stride]f32 = undefined;
    fillToken(&k_tok, 1.0);
    fillToken(&v_tok, 2.0);

    for (0..64) |_| {
        try cache.append(&k_tok, &v_tok);
    }

    // Slice — should NOT allocate
    _ = cache.slice();

    // Reset — should NOT allocate
    cache.reset();

    // Re-append — should NOT allocate
    for (0..32) |_| {
        try cache.append(&k_tok, &v_tok);
    }
    _ = cache.slice();

    // Verify zero allocations after create
    try testing.expectEqual(allocs_after_create, tracker.alloc_count);
}

// ================================================================
// Error cases
// ================================================================

test "kvcache: append rejects wrong-sized token" {
    var cache = try KvCache.create(testing.allocator, 8, 2, 4);
    defer cache.destroy(testing.allocator);

    var short: [4]f32 = .{ 1, 2, 3, 4 }; // needs 8
    var correct: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };

    try testing.expectError(error.ShapeMismatch, cache.append(&short, &correct));
    try testing.expectError(error.ShapeMismatch, cache.append(&correct, &short));
}

test "kvcache: append returns CacheFull when exhausted" {
    var cache = try KvCache.create(testing.allocator, 2, 1, 2);
    defer cache.destroy(testing.allocator);

    var tok: [2]f32 = .{ 1, 2 };
    try cache.append(&tok, &tok);
    try cache.append(&tok, &tok);
    try testing.expectError(error.CacheFull, cache.append(&tok, &tok));
}

test "kvcache: create rejects zero dimensions" {
    try testing.expectError(error.InvalidDimensions, KvCache.create(testing.allocator, 0, 4, 32));
    try testing.expectError(error.InvalidDimensions, KvCache.create(testing.allocator, 16, 0, 32));
    try testing.expectError(error.InvalidDimensions, KvCache.create(testing.allocator, 16, 4, 0));
}

// ================================================================
// Per-layer management
// ================================================================

test "kvcache: LayerCaches create and per-layer independence" {
    var lc = try LayerCaches.create(testing.allocator, 4, 16, 2, 4);
    defer lc.destroy();

    try testing.expectEqual(@as(usize, 4), lc.n_layers);

    const stride = 2 * 4;

    // Append different data to each layer
    for (0..4) |layer| {
        var k_tok: [stride]f32 = undefined;
        var v_tok: [stride]f32 = undefined;
        fillToken(&k_tok, @as(f32, @floatFromInt(layer)) * 100.0);
        fillToken(&v_tok, @as(f32, @floatFromInt(layer)) * 100.0 + 50.0);

        const c = lc.get(layer);
        try c.append(&k_tok, &v_tok);
    }

    // Verify each layer has independent data
    for (0..4) |layer| {
        const c = lc.get(layer);
        try testing.expectEqual(@as(usize, 1), c.pos);
        const s = c.slice();
        const expected_k0 = @as(f32, @floatFromInt(layer)) * 100.0;
        try testing.expectApproxEqAbs(expected_k0, s.k[0], 1e-5);
    }
}

test "kvcache: LayerCaches resetAll" {
    var lc = try LayerCaches.create(testing.allocator, 3, 8, 2, 4);
    defer lc.destroy();

    const stride = 2 * 4;
    var tok: [stride]f32 = undefined;
    fillToken(&tok, 1.0);

    // Append to all layers
    for (0..3) |layer| {
        try lc.get(layer).append(&tok, &tok);
        try lc.get(layer).append(&tok, &tok);
    }

    // All should have pos=2
    for (0..3) |layer| {
        try testing.expectEqual(@as(usize, 2), lc.get(layer).pos);
    }

    lc.resetAll();

    // All should have pos=0
    for (0..3) |layer| {
        try testing.expectEqual(@as(usize, 0), lc.get(layer).pos);
    }
}

test "kvcache: LayerCaches create rejects zero layers" {
    try testing.expectError(error.InvalidDimensions, LayerCaches.create(testing.allocator, 0, 16, 2, 4));
}

// ================================================================
// Memory leak detection (testing.allocator catches leaks)
// ================================================================

test "kvcache: zero memory leaks KvCache" {
    var cache = try KvCache.create(testing.allocator, 16, 4, 32);
    const stride = 4 * 32;
    var k_tok: [stride]f32 = undefined;
    var v_tok: [stride]f32 = undefined;
    fillToken(&k_tok, 1.0);
    fillToken(&v_tok, 2.0);

    for (0..10) |_| {
        try cache.append(&k_tok, &v_tok);
    }
    _ = cache.slice();
    cache.reset();
    for (0..5) |_| {
        try cache.append(&k_tok, &v_tok);
    }

    cache.destroy(testing.allocator);
    // testing.allocator validates no leaks on cleanup
}

test "kvcache: zero memory leaks LayerCaches" {
    var lc = try LayerCaches.create(testing.allocator, 4, 16, 2, 8);
    const stride = 2 * 8;
    var tok: [stride]f32 = undefined;
    fillToken(&tok, 1.0);

    for (0..4) |layer| {
        try lc.get(layer).append(&tok, &tok);
    }
    lc.resetAll();
    for (0..4) |layer| {
        try lc.get(layer).append(&tok, &tok);
    }

    lc.destroy();
    // testing.allocator validates no leaks on cleanup
}
