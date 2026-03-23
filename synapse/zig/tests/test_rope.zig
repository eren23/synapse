//! Tests for Rotary Positional Embedding (RoPE):
//! - Correctness: SIMD vs scalar reference for d_head=32, 64, 128
//! - Rotation invertibility: forward + inverse = identity within 1e-5
//! - Offset parameter shifts positions correctly
//! - Edge cases: seq_len=1, d_head=2
//! - Benchmark: SIMD vs scalar on [8, 8, 128, 32], target >= 3x speedup
//! - Zero memory leaks via tracking allocator (testing.allocator)

const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;
const math = std.math;

const rope_ops = synapse.ops.rope;
const RopeTable = rope_ops.RopeTable;

// ==================== Helpers ====================

fn expectClose(a: f32, b: f32, tol: f32) !void {
    const diff = @abs(a - b);
    const max_abs = @max(@abs(a), @abs(b));
    // Relative error when values are significant, absolute otherwise
    const err = if (max_abs > 1e-8) diff / max_abs else diff;
    if (err > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, err={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, err),
        });
        return error.TestUnexpectedResult;
    }
}

/// Deterministic pseudo-random fill for reproducible tests.
fn fillDeterministic(data: []f32, seed: u32) void {
    var s = seed;
    for (data) |*v| {
        s = s *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt(@rem(@as(i32, @bitCast(s >> 16)), 1000))) * 0.001;
    }
}

/// Run SIMD vs scalar correctness check for a given d_head.
fn testCorrectnessForDhead(d_head: usize) !void {
    const batch = 2;
    const heads = 4;
    const seq = 8;
    const half_d = d_head / 2;
    const total = batch * heads * seq * d_head;
    const max_seq = seq + 10;

    var table = try RopeTable.init(testing.allocator, max_seq, d_head);
    defer table.deinit();

    const input = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(input);
    fillDeterministic(input, 42 + @as(u32, @intCast(d_head)));

    const simd_output = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(simd_output);

    const scalar_output = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(scalar_output);

    rope_ops.ropeSimd(simd_output, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);
    rope_ops.ropeScalar(scalar_output, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);

    for (0..total) |i| {
        try expectClose(simd_output[i], scalar_output[i], 1e-5);
    }
}

// ==================== Correctness Tests ====================

test "rope: SIMD matches scalar for d_head=32" {
    try testCorrectnessForDhead(32);
}

test "rope: SIMD matches scalar for d_head=64" {
    try testCorrectnessForDhead(64);
}

test "rope: SIMD matches scalar for d_head=128" {
    try testCorrectnessForDhead(128);
}

// ==================== Rotation Invertibility ====================

test "rope: forward then inverse = identity within 1e-5" {
    const d_head = 64;
    const half_d = d_head / 2;
    const batch = 2;
    const heads = 2;
    const seq = 4;
    const total = batch * heads * seq * d_head;
    const max_seq = seq + 5;

    var table = try RopeTable.init(testing.allocator, max_seq, d_head);
    defer table.deinit();

    const input = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(input);
    fillDeterministic(input, 12345);

    // Forward rotation
    const forward = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(forward);
    rope_ops.ropeSimd(forward, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);

    // Negate sin table for inverse rotation
    const neg_sin = try testing.allocator.alloc(f32, table.sin.len);
    defer testing.allocator.free(neg_sin);
    for (0..table.sin.len) |i| {
        neg_sin[i] = -table.sin[i];
    }

    // Inverse rotation: apply with negated sin
    const restored = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(restored);
    rope_ops.ropeSimd(restored, forward, table.cos, neg_sin, batch, heads, seq, d_head, half_d, 0);

    // Should recover original input
    for (0..total) |i| {
        try expectClose(restored[i], input[i], 1e-5);
    }
}

// ==================== Offset Parameter ====================

test "rope: offset shifts positions correctly" {
    const d_head = 32;
    const half_d = d_head / 2;
    const max_seq = 64;

    var table = try RopeTable.init(testing.allocator, max_seq, d_head);
    defer table.deinit();

    // Input: [1, 1, 2, d_head] — two sequence positions
    const total_2 = 1 * 1 * 2 * d_head;
    const input = try testing.allocator.alloc(f32, total_2);
    defer testing.allocator.free(input);
    fillDeterministic(input, 777);

    // Apply with pos_offset=10: positions become 10 and 11
    const out_offset10 = try testing.allocator.alloc(f32, total_2);
    defer testing.allocator.free(out_offset10);
    rope_ops.ropeSimd(out_offset10, input, table.cos, table.sin, 1, 1, 2, d_head, half_d, 10);

    // Apply the second position's data with pos_offset=11: position becomes 11
    const total_1 = 1 * 1 * 1 * d_head;
    const out_single = try testing.allocator.alloc(f32, total_1);
    defer testing.allocator.free(out_single);
    rope_ops.ropeSimd(out_single, input[d_head..], table.cos, table.sin, 1, 1, 1, d_head, half_d, 11);

    // Second position from offset=10 should match single-position offset=11
    for (0..d_head) |i| {
        try expectClose(out_offset10[d_head + i], out_single[i], 1e-5);
    }

    // Verify offset=0 gives different results than offset=10
    const out_offset0 = try testing.allocator.alloc(f32, total_2);
    defer testing.allocator.free(out_offset0);
    rope_ops.ropeSimd(out_offset0, input, table.cos, table.sin, 1, 1, 2, d_head, half_d, 0);

    var differs = false;
    for (0..d_head) |i| {
        if (@abs(out_offset0[i] - out_offset10[i]) > 1e-6) {
            differs = true;
            break;
        }
    }
    try testing.expect(differs);
}

// ==================== Edge Cases ====================

test "rope: edge case seq_len=1" {
    const d_head = 32;
    const half_d = d_head / 2;
    const total = 1 * 1 * 1 * d_head;

    var table = try RopeTable.init(testing.allocator, 4, d_head);
    defer table.deinit();

    const input = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(input);
    fillDeterministic(input, 99);

    const simd_out = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(simd_out);
    const scalar_out = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(scalar_out);

    rope_ops.ropeSimd(simd_out, input, table.cos, table.sin, 1, 1, 1, d_head, half_d, 0);
    rope_ops.ropeScalar(scalar_out, input, table.cos, table.sin, 1, 1, 1, d_head, half_d, 0);

    for (0..total) |i| {
        try expectClose(simd_out[i], scalar_out[i], 1e-5);
    }
}

test "rope: edge case d_head=2" {
    const d_head = 2;
    const half_d = 1;
    const batch = 2;
    const heads = 2;
    const seq = 4;
    const total = batch * heads * seq * d_head;

    var table = try RopeTable.init(testing.allocator, seq + 1, d_head);
    defer table.deinit();

    const input = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(input);
    fillDeterministic(input, 55);

    const simd_out = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(simd_out);
    const scalar_out = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(scalar_out);

    rope_ops.ropeSimd(simd_out, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);
    rope_ops.ropeScalar(scalar_out, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);

    for (0..total) |i| {
        try expectClose(simd_out[i], scalar_out[i], 1e-5);
    }

    // Verify rotation actually happened (not identity for non-zero positions)
    var changed = false;
    for (0..total) |i| {
        if (@abs(simd_out[i] - input[i]) > 1e-6) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);
}

// ==================== Benchmark ====================

/// Scalar RoPE with asm barrier to prevent auto-vectorization in benchmarks.
noinline fn benchRopeScalar(
    output_ptr: [*]f32,
    input_ptr: [*]const f32,
    cos_ptr: [*]const f32,
    sin_ptr: [*]const f32,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    d_head: usize,
    half_d: usize,
    pos_offset: usize,
) void {
    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                const pos = s + pos_offset;
                const base = ((b * num_heads + h) * seq_len + s) * d_head;
                const table_off = pos * half_d;
                for (0..half_d) |p| {
                    const idx0 = base + 2 * p;
                    const x0 = input_ptr[idx0];
                    const x1 = input_ptr[idx0 + 1];
                    const c = cos_ptr[table_off + p];
                    const si = sin_ptr[table_off + p];
                    output_ptr[idx0] = x0 * c - x1 * si;
                    output_ptr[idx0 + 1] = x0 * si + x1 * c;
                    asm volatile ("" ::: .{ .memory = true });
                }
            }
        }
    }
}

test "rope: benchmark SIMD vs scalar [8, 8, 128, 32]" {
    const batch = 8;
    const heads = 8;
    const seq = 128;
    const d_head = 32;
    const half_d = d_head / 2;
    const total = batch * heads * seq * d_head;
    const max_seq = seq + 1;
    const warmup = 5;
    const iters = 50;

    var table = try RopeTable.init(testing.allocator, max_seq, d_head);
    defer table.deinit();

    const input = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(input);
    fillDeterministic(input, 42);

    const output = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(output);

    var sink: f32 = 0;

    // Warmup both paths
    for (0..warmup) |_| {
        rope_ops.ropeSimd(output, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);
        sink += output[0];
        benchRopeScalar(output.ptr, input.ptr, table.cos.ptr, table.sin.ptr, batch, heads, seq, d_head, half_d, 0);
        sink += output[0];
    }

    // Benchmark scalar (with asm barriers preventing auto-vectorization)
    const scalar_start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        benchRopeScalar(output.ptr, input.ptr, table.cos.ptr, table.sin.ptr, batch, heads, seq, d_head, half_d, 0);
        sink += output[0];
        asm volatile ("" ::: .{ .memory = true });
    }
    const scalar_end = std.time.nanoTimestamp();
    const scalar_ns: u64 = @intCast(scalar_end - scalar_start);

    // Benchmark SIMD
    const simd_start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        rope_ops.ropeSimd(output, input, table.cos, table.sin, batch, heads, seq, d_head, half_d, 0);
        sink += output[0];
        asm volatile ("" ::: .{ .memory = true });
    }
    const simd_end = std.time.nanoTimestamp();
    const simd_ns: u64 = @intCast(simd_end - simd_start);

    const speedup = @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns));

    std.debug.print("\n=== RoPE Benchmark [8, 8, 128, 32] x {d} iters ===\n", .{iters});
    std.debug.print("Scalar: {d:.2} ms\n", .{@as(f64, @floatFromInt(scalar_ns)) / 1e6});
    std.debug.print("SIMD:   {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_ns)) / 1e6});
    std.debug.print("Speedup: {d:.2}x\n", .{speedup});

    try testing.expect(speedup >= 3.0);

    // Prevent sink from being optimized away
    if (sink == 0.0) std.debug.print("sink: {d}\n", .{sink});
}
