//! Correctness tests for fused scaled dot-product attention.
//! Compares fused vs naive implementation across shapes, causal modes, and edge cases.
//! Tolerance: max relative error <= 1e-4.

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const attention_mod = synapse.ops.attention;

// ================================================================
// Helpers
// ================================================================

/// Create a 4D tensor [batch, heads, seq, d_head] with deterministic pseudo-random values in [-1, 1].
fn makeTensor4D(allocator: std.mem.Allocator, batch: usize, heads: usize, seq: usize, d_head: usize, seed: u32) !Tensor(f32) {
    const n = batch * heads * seq * d_head;
    const storage = try Storage.create(allocator, f32, @max(n, 1));
    const data = storage.dataAs(f32);
    var s: u32 = seed;
    for (0..n) |i| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        data[i] = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
    const shape = Shape.init(&[_]usize{ batch, heads, seq, d_head });
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    return t;
}

/// Check if two floats are approximately equal (absolute OR relative tolerance).
fn isClose(actual: f32, expected: f32, tol: f32) bool {
    const diff = @abs(actual - expected);
    if (diff <= tol) return true;
    const denom = @max(@abs(expected), @abs(actual));
    if (denom < 1e-10) return diff <= tol;
    return diff / denom <= tol;
}

/// Run a full attention correctness check: fused vs naive.
fn checkAttention(
    allocator: std.mem.Allocator,
    batch: usize,
    heads: usize,
    seq: usize,
    d_head: usize,
    causal: bool,
    tol: f32,
) !void {
    const q = try makeTensor4D(allocator, batch, heads, seq, d_head, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, batch, heads, seq, d_head, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, batch, heads, seq, d_head, 256);
    defer v.release();

    const config = attention_mod.AttentionConfig{ .causal = causal, .return_weights = false };

    const fused = try attention_mod.attention(allocator, q, k, v, config);
    defer fused.release();
    const naive = try attention_mod.naiveAttention(allocator, q, k, v, config);
    defer naive.release();

    // Verify shapes match
    try testing.expectEqual(fused.output.shape.ndim, naive.output.shape.ndim);
    for (0..4) |d| {
        try testing.expectEqual(fused.output.shape.dims[d], naive.output.shape.dims[d]);
    }

    // Compare element-wise
    const fused_data = fused.output.storage.dataAs(f32);
    const naive_data = naive.output.storage.dataAs(f32);
    const numel = fused.output.numel();
    for (0..numel) |i| {
        if (!isClose(fused_data[i], naive_data[i], tol)) {
            std.debug.print(
                "MISMATCH at flat {d}: fused={d:.8} naive={d:.8} diff={e:.4}\n",
                .{ i, fused_data[i], naive_data[i], @abs(fused_data[i] - naive_data[i]) },
            );
            return error.TestUnexpectedResult;
        }
    }
}

// ================================================================
// Correctness: fused vs naive — non-causal
// ================================================================

test "attention: non-causal [1,1,8,32]" {
    try checkAttention(testing.allocator, 1, 1, 8, 32, false, 1e-4);
}

test "attention: non-causal [2,4,32,64]" {
    try checkAttention(testing.allocator, 2, 4, 32, 64, false, 1e-4);
}

test "attention: non-causal [8,8,128,32]" {
    try checkAttention(testing.allocator, 8, 8, 128, 32, false, 1e-4);
}

// ================================================================
// Correctness: fused vs naive — causal
// ================================================================

test "attention: causal [1,1,8,32]" {
    try checkAttention(testing.allocator, 1, 1, 8, 32, true, 1e-4);
}

test "attention: causal [2,4,32,64]" {
    try checkAttention(testing.allocator, 2, 4, 32, 64, true, 1e-4);
}

test "attention: causal [8,8,128,32]" {
    try checkAttention(testing.allocator, 8, 8, 128, 32, true, 1e-4);
}

// ================================================================
// Causal mask: verify future positions are EXACTLY 0.0
// ================================================================

test "attention: causal mask exact zeros" {
    const allocator = testing.allocator;
    const batch: usize = 2;
    const heads: usize = 2;
    const seq: usize = 16;
    const d_head: usize = 32;

    const q = try makeTensor4D(allocator, batch, heads, seq, d_head, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, batch, heads, seq, d_head, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, batch, heads, seq, d_head, 256);
    defer v.release();

    const config = attention_mod.AttentionConfig{ .causal = true, .return_weights = true };

    const fused = try attention_mod.attention(allocator, q, k, v, config);
    defer fused.release();
    const naive = try attention_mod.naiveAttention(allocator, q, k, v, config);
    defer naive.release();

    // Check both fused and naive weights
    const fused_w = fused.weights.?.storage.dataAs(f32);
    const naive_w = naive.weights.?.storage.dataAs(f32);

    for (0..batch) |b| {
        for (0..heads) |h| {
            const bh_offset = (b * heads + h) * seq * seq;
            for (0..seq) |i| {
                for (0..seq) |j| {
                    if (j > i) {
                        const idx = bh_offset + i * seq + j;
                        // Must be EXACTLY 0.0
                        try testing.expectEqual(@as(f32, 0.0), fused_w[idx]);
                        try testing.expectEqual(@as(f32, 0.0), naive_w[idx]);
                    }
                }
            }
        }
    }
}

// ================================================================
// Edge cases
// ================================================================

test "attention: seq_len=1" {
    try checkAttention(testing.allocator, 1, 1, 1, 32, false, 1e-4);
}

test "attention: seq_len=1 causal" {
    try checkAttention(testing.allocator, 1, 1, 1, 32, true, 1e-4);
}

test "attention: d_head=1" {
    try checkAttention(testing.allocator, 1, 1, 8, 1, false, 1e-4);
}

test "attention: d_head=1 causal" {
    try checkAttention(testing.allocator, 1, 1, 8, 1, true, 1e-4);
}

test "attention: batch=1 heads=1 minimal" {
    try checkAttention(testing.allocator, 1, 1, 4, 4, false, 1e-4);
}

// ================================================================
// Numerical stability: no inf/nan for long sequences
// ================================================================

test "attention: no inf/nan for seq=2048" {
    const allocator = testing.allocator;
    const q = try makeTensor4D(allocator, 1, 1, 2048, 32, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, 1, 1, 2048, 32, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, 1, 1, 2048, 32, 256);
    defer v.release();

    const result = try attention_mod.attention(allocator, q, k, v, .{});
    defer result.release();

    const data = result.output.storage.dataAs(f32);
    for (data[0..result.output.numel()]) |val| {
        try testing.expect(!std.math.isInf(val));
        try testing.expect(!std.math.isNan(val));
    }
}

test "attention: no inf/nan causal seq=2048" {
    const allocator = testing.allocator;
    const q = try makeTensor4D(allocator, 1, 1, 2048, 32, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, 1, 1, 2048, 32, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, 1, 1, 2048, 32, 256);
    defer v.release();

    const result = try attention_mod.attention(allocator, q, k, v, .{ .causal = true });
    defer result.release();

    const data = result.output.storage.dataAs(f32);
    for (data[0..result.output.numel()]) |val| {
        try testing.expect(!std.math.isInf(val));
        try testing.expect(!std.math.isNan(val));
    }
}

// ================================================================
// Memory leak detection (testing.allocator catches leaks)
// ================================================================

test "attention: zero memory leaks non-causal" {
    const allocator = testing.allocator;
    const q = try makeTensor4D(allocator, 2, 2, 16, 16, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, 2, 2, 16, 16, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, 2, 2, 16, 16, 256);
    defer v.release();

    const config = attention_mod.AttentionConfig{ .causal = false, .return_weights = true };
    const result = try attention_mod.attention(allocator, q, k, v, config);
    defer result.release();

    // If we reach here without leak, testing.allocator validates on cleanup
    try testing.expect(result.output.numel() == 2 * 2 * 16 * 16);
    try testing.expect(result.weights != null);
}

test "attention: zero memory leaks causal" {
    const allocator = testing.allocator;
    const q = try makeTensor4D(allocator, 2, 2, 16, 16, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, 2, 2, 16, 16, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, 2, 2, 16, 16, 256);
    defer v.release();

    const config = attention_mod.AttentionConfig{ .causal = true, .return_weights = true };
    const result = try attention_mod.attention(allocator, q, k, v, config);
    defer result.release();

    try testing.expect(result.output.numel() == 2 * 2 * 16 * 16);
    try testing.expect(result.weights != null);
}

test "attention: zero memory leaks naive" {
    const allocator = testing.allocator;
    const q = try makeTensor4D(allocator, 2, 2, 16, 16, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, 2, 2, 16, 16, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, 2, 2, 16, 16, 256);
    defer v.release();

    const config = attention_mod.AttentionConfig{ .causal = true, .return_weights = true };
    const result = try attention_mod.naiveAttention(allocator, q, k, v, config);
    defer result.release();

    try testing.expect(result.output.numel() == 2 * 2 * 16 * 16);
    try testing.expect(result.weights != null);
}
