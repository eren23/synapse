const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;

const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const Tensor = synapse.tensor.core.Tensor;
const rmsnorm = synapse.ops.rmsnorm;
const layernorm = synapse.ops.layernorm;

// ==================== Helpers ====================

fn expectClose(a: f32, b: f32, tol: f32) !void {
    if (@abs(a - b) > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, diff={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, @abs(a - b)),
        });
        return error.TestUnexpectedResult;
    }
}

fn expectRelClose(a: f32, b: f32, tol: f32) !void {
    const denom = @max(@abs(a), @abs(b));
    if (denom < 1e-8) {
        // Both near zero: use absolute tolerance
        try expectClose(a, b, tol);
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

fn makeTensor(allocator: std.mem.Allocator, dims: []const usize, values: []const f32) !Tensor(f32) {
    const shape = Shape.init(dims);
    const n = shape.numel();
    const storage = try Storage.create(allocator, f32, n);
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    @memcpy(t.storage.dataAs(f32)[0..n], values[0..n]);
    return t;
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

fn makeOnes(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const buf = try allocator.alloc(f32, n);
    for (buf) |*v| v.* = 1.0;
    return buf;
}

fn makeZeros(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const buf = try allocator.alloc(f32, n);
    @memset(buf, 0.0);
    return buf;
}

// ==================== Correctness: SIMD vs scalar reference ====================

test "rmsnorm: SIMD matches scalar for [64, 256]" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 64, 256 };
    const n: usize = 64 * 256;
    const norm_size: usize = 256;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 42);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const simd_out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer simd_out.release();
    const scalar_out = try rmsnorm.rmsNormScalar(allocator, t, 1, gamma, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectRelClose(simd_data[i], scalar_data[i], 1e-5);
    }
}

test "rmsnorm: SIMD matches scalar for [32, 1024]" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 32, 1024 };
    const n: usize = 32 * 1024;
    const norm_size: usize = 1024;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 137);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const simd_out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer simd_out.release();
    const scalar_out = try rmsnorm.rmsNormScalar(allocator, t, 1, gamma, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectRelClose(simd_data[i], scalar_data[i], 1e-5);
    }
}

test "rmsnorm: SIMD matches scalar for [1, 4096]" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 1, 4096 };
    const n: usize = 1 * 4096;
    const norm_size: usize = 4096;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 99);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const simd_out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer simd_out.release();
    const scalar_out = try rmsnorm.rmsNormScalar(allocator, t, 1, gamma, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectRelClose(simd_data[i], scalar_data[i], 1e-5);
    }
}

// ==================== Output statistics ====================

test "rmsnorm: output norm ~1.0" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 32, 256 };
    const n: usize = 32 * 256;
    const norm_size: usize = 256;
    const outer_size: usize = 32;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 42);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    const norm_f: f32 = @floatFromInt(norm_size);

    for (0..outer_size) |outer| {
        const base = outer * norm_size;

        // Check RMS of output ~1.0: sqrt(mean(out²)) ≈ 1
        var sum_sq: f32 = 0;
        for (0..norm_size) |j| {
            sum_sq += out_data[base + j] * out_data[base + j];
        }
        const rms = @sqrt(sum_sq / norm_f);
        try expectClose(rms, 1.0, 1e-4);
    }
}

// ==================== Special values ====================

test "rmsnorm: all-zeros input" {
    const allocator = testing.allocator;
    const norm_size: usize = 64;
    const outer_size: usize = 4;
    const n = outer_size * norm_size;
    const dims = [_]usize{ outer_size, norm_size };

    const values = try makeZeros(allocator, n);
    defer allocator.free(values);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    // 0 * rsqrt(0 + eps) * 1 = 0
    for (0..n) |i| {
        try expectClose(out_data[i], 0.0, 1e-6);
        try testing.expect(!std.math.isNan(out_data[i]));
        try testing.expect(!std.math.isInf(out_data[i]));
    }
}

test "rmsnorm: constant input" {
    const allocator = testing.allocator;
    const norm_size: usize = 64;
    const outer_size: usize = 4;
    const n = outer_size * norm_size;
    const dims = [_]usize{ outer_size, norm_size };

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    for (values) |*v| v.* = 5.0;

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    // RMS of constant 5.0 = 5.0, so output ≈ 5 * (1/5) = 1.0
    for (0..n) |i| {
        try expectClose(out_data[i], 1.0, 1e-4);
        try testing.expect(!std.math.isNan(out_data[i]));
        try testing.expect(!std.math.isInf(out_data[i]));
    }
}

test "rmsnorm: large values ±1000 no inf/nan" {
    const allocator = testing.allocator;
    const norm_size: usize = 256;
    const outer_size: usize = 16;
    const n = outer_size * norm_size;
    const dims = [_]usize{ outer_size, norm_size };

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 42);
    for (values) |*v| v.* *= 1000.0;

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(out_data[i]));
        try testing.expect(!std.math.isInf(out_data[i]));
    }

    // Verify output RMS ~1.0
    const norm_f: f32 = @floatFromInt(norm_size);
    for (0..outer_size) |outer| {
        const base = outer * norm_size;
        var sum_sq: f32 = 0;
        for (0..norm_size) |j| {
            sum_sq += out_data[base + j] * out_data[base + j];
        }
        const rms = @sqrt(sum_sq / norm_f);
        try expectClose(rms, 1.0, 1e-3);
    }
}

// ==================== Differs from LayerNorm ====================

test "rmsnorm: output differs from layernorm" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 16, 128 };
    const n: usize = 16 * 128;
    const norm_size: usize = 128;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 77);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const rms_out = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer rms_out.release();
    const ln_out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer ln_out.release();

    const rms_data = rms_out.storage.dataAs(f32);
    const ln_data = ln_out.storage.dataAs(f32);

    // Count how many elements differ significantly
    var diff_count: usize = 0;
    for (0..n) |i| {
        if (@abs(rms_data[i] - ln_data[i]) > 1e-4) {
            diff_count += 1;
        }
    }
    // Vast majority should differ (non-zero-mean input)
    try testing.expect(diff_count > n / 2);
}

// ==================== Memory: zero leaks ====================

test "rmsnorm: no memory leaks via testing allocator" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 4, 32 };
    const n: usize = 4 * 32;
    const norm_size: usize = 32;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 11);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);

    const out1 = try rmsnorm.rmsNorm(allocator, t, 1, gamma, 1e-5);
    defer out1.release();

    const out2 = try rmsnorm.rmsNormScalar(allocator, t, 1, gamma, 1e-5);
    defer out2.release();

    const d1 = out1.storage.dataAs(f32);
    const d2 = out2.storage.dataAs(f32);
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(d1[i]));
        try testing.expect(!std.math.isNan(d2[i]));
    }
}
