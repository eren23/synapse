const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;

const silu_mod = synapse.ops.silu;

// ==================== Helpers ====================

fn expectRelClose(a: f32, b: f32, tol: f32) !void {
    const denom = @max(@abs(a), @abs(b));
    if (denom < 1e-8) {
        // Both near zero: use absolute tolerance
        if (@abs(a - b) > tol) {
            std.debug.print("FAIL: got={e}, expected={e}, diff={e}\n", .{
                @as(f64, a), @as(f64, b), @as(f64, @abs(a - b)),
            });
            return error.TestUnexpectedResult;
        }
        return;
    }
    const rel_err = @abs(a - b) / denom;
    if (rel_err > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, rel_err={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, rel_err),
        });
        return error.TestUnexpectedResult;
    }
}

fn fillDeterministic(buf: []f32, seed: u32) void {
    var s: u32 = seed;
    for (buf) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
}

/// Reference SiLU using std.math.exp for maximum accuracy.
fn referenceSilu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

// ==================== SiLU correctness: SIMD vs reference ====================

test "silu: SIMD matches reference for [64, 3072]" {
    const allocator = testing.allocator;
    const n: usize = 64 * 3072;

    const src = try allocator.alloc(f32, n);
    defer allocator.free(src);
    fillDeterministic(src, 42);

    const dst_simd = try allocator.alloc(f32, n);
    defer allocator.free(dst_simd);
    const dst_scalar = try allocator.alloc(f32, n);
    defer allocator.free(dst_scalar);

    silu_mod.silu(dst_simd, src);
    silu_mod.siluScalar(dst_scalar, src);

    for (0..n) |i| {
        const ref = referenceSilu(src[i]);
        try expectRelClose(dst_simd[i], ref, 1e-5);
        try expectRelClose(dst_scalar[i], ref, 1e-5);
    }
}

test "silu: SIMD matches reference for [1, 1024]" {
    const allocator = testing.allocator;
    const n: usize = 1024;

    const src = try allocator.alloc(f32, n);
    defer allocator.free(src);
    fillDeterministic(src, 137);

    const dst_simd = try allocator.alloc(f32, n);
    defer allocator.free(dst_simd);
    const dst_scalar = try allocator.alloc(f32, n);
    defer allocator.free(dst_scalar);

    silu_mod.silu(dst_simd, src);
    silu_mod.siluScalar(dst_scalar, src);

    for (0..n) |i| {
        const ref = referenceSilu(src[i]);
        try expectRelClose(dst_simd[i], ref, 1e-5);
        try expectRelClose(dst_scalar[i], ref, 1e-5);
    }
}

// ==================== SiLU(0) == 0 exactly ====================

test "silu: SiLU(0) == 0 exactly" {
    const allocator = testing.allocator;
    const n: usize = 64;

    const src = try allocator.alloc(f32, n);
    defer allocator.free(src);
    @memset(src, 0.0);

    const dst = try allocator.alloc(f32, n);
    defer allocator.free(dst);

    silu_mod.silu(dst, src);

    for (0..n) |i| {
        try testing.expectEqual(@as(f32, 0.0), dst[i]);
    }
}

// ==================== No inf/nan for inputs in [-100, 100] ====================

test "silu: no inf/nan for inputs in [-100, 100]" {
    const allocator = testing.allocator;
    const n: usize = 1024;

    const src = try allocator.alloc(f32, n);
    defer allocator.free(src);

    // Fill with values spanning [-100, 100]
    for (src, 0..) |*v, i| {
        v.* = -100.0 + 200.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
    }

    const dst = try allocator.alloc(f32, n);
    defer allocator.free(dst);

    silu_mod.silu(dst, src);

    for (0..n) |i| {
        try testing.expect(!std.math.isNan(dst[i]));
        try testing.expect(!std.math.isInf(dst[i]));
    }

    // Also verify extreme edges
    src[0] = -100.0;
    src[1] = 100.0;
    silu_mod.silu(dst[0..2], src[0..2]);
    try testing.expect(!std.math.isNan(dst[0]));
    try testing.expect(!std.math.isInf(dst[0]));
    try testing.expect(!std.math.isNan(dst[1]));
    try testing.expect(!std.math.isInf(dst[1]));
}

// ==================== SwiGLU correctness: fused vs separate ====================

test "swiglu: fused matches separate for [64, 3072]" {
    const allocator = testing.allocator;
    const n: usize = 64 * 3072;

    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    fillDeterministic(gate, 42);

    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    fillDeterministic(up, 99);

    const dst_fused = try allocator.alloc(f32, n);
    defer allocator.free(dst_fused);
    const dst_separate = try allocator.alloc(f32, n);
    defer allocator.free(dst_separate);
    const tmp = try allocator.alloc(f32, n);
    defer allocator.free(tmp);

    silu_mod.swigluFused(dst_fused, gate, up);
    silu_mod.swigluSeparate(dst_separate, gate, up, tmp);

    for (0..n) |i| {
        try expectRelClose(dst_fused[i], dst_separate[i], 1e-5);
    }
}

test "swiglu: fused matches separate for [1, 1024]" {
    const allocator = testing.allocator;
    const n: usize = 1024;

    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    fillDeterministic(gate, 77);

    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    fillDeterministic(up, 200);

    const dst_fused = try allocator.alloc(f32, n);
    defer allocator.free(dst_fused);
    const dst_separate = try allocator.alloc(f32, n);
    defer allocator.free(dst_separate);
    const tmp = try allocator.alloc(f32, n);
    defer allocator.free(tmp);

    silu_mod.swigluFused(dst_fused, gate, up);
    silu_mod.swigluSeparate(dst_separate, gate, up, tmp);

    for (0..n) |i| {
        try expectRelClose(dst_fused[i], dst_separate[i], 1e-5);
    }
}

// ==================== SwiGLU correctness vs manual reference ====================

test "swiglu: fused matches silu(gate)*up computed manually" {
    const allocator = testing.allocator;
    const n: usize = 256;

    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    fillDeterministic(gate, 55);

    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);
    fillDeterministic(up, 66);

    const dst = try allocator.alloc(f32, n);
    defer allocator.free(dst);

    silu_mod.swigluFused(dst, gate, up);

    for (0..n) |i| {
        const expected = referenceSilu(gate[i]) * up[i];
        try expectRelClose(dst[i], expected, 1e-5);
    }
}

// ==================== Large |x| SwiGLU no inf/nan ====================

test "swiglu: no inf/nan for large inputs" {
    const allocator = testing.allocator;
    const n: usize = 512;

    const gate = try allocator.alloc(f32, n);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, n);
    defer allocator.free(up);

    // gate in [-100, 100], up in [-10, 10]
    for (gate, up, 0..) |*g, *u, i| {
        g.* = -100.0 + 200.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
        u.* = -10.0 + 20.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
    }

    const dst = try allocator.alloc(f32, n);
    defer allocator.free(dst);

    silu_mod.swigluFused(dst, gate, up);

    for (0..n) |i| {
        try testing.expect(!std.math.isNan(dst[i]));
        try testing.expect(!std.math.isInf(dst[i]));
    }
}
