const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;

const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const Tensor = synapse.tensor.core.Tensor;
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

test "layernorm: SIMD matches scalar for [64, 256]" {
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
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const simd_out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer simd_out.release();
    const scalar_out = try layernorm.layerNormScalar(allocator, t, 1, gamma, beta, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectClose(simd_data[i], scalar_data[i], 1e-4);
    }
}

test "layernorm: SIMD matches scalar for [32, 128, 512]" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 32, 128, 512 };
    const n: usize = 32 * 128 * 512;
    const norm_size: usize = 512;

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 137);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const simd_out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer simd_out.release();
    const scalar_out = try layernorm.layerNormScalar(allocator, t, 1, gamma, beta, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectClose(simd_data[i], scalar_data[i], 1e-4);
    }
}

test "layernorm: SIMD matches scalar for [8, 16, 32, 64] norm 2 dims" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 8, 16, 32, 64 };
    const n: usize = 8 * 16 * 32 * 64;
    const norm_size: usize = 32 * 64; // last 2 dims

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    fillDeterministic(values, 99);

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const simd_out = try layernorm.layerNorm(allocator, t, 2, gamma, beta, 1e-5);
    defer simd_out.release();
    const scalar_out = try layernorm.layerNormScalar(allocator, t, 2, gamma, beta, 1e-5);
    defer scalar_out.release();

    const simd_data = simd_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectClose(simd_data[i], scalar_data[i], 1e-4);
    }
}

// ==================== Output statistics ====================

test "layernorm: output mean ~0 and variance ~1" {
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
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    const norm_f: f32 = @floatFromInt(norm_size);

    for (0..outer_size) |outer| {
        const base = outer * norm_size;

        // Check mean ~0
        var mean: f32 = 0;
        for (0..norm_size) |j| {
            mean += out_data[base + j];
        }
        mean /= norm_f;
        try testing.expect(@abs(mean) <= 1e-5);

        // Check variance ~1
        var variance: f32 = 0;
        for (0..norm_size) |j| {
            const d = out_data[base + j] - mean;
            variance += d * d;
        }
        variance /= norm_f;
        try expectClose(variance, 1.0, 1e-4);
    }
}

// ==================== Special values ====================

test "layernorm: all-zeros input" {
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
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    // (0 - 0) / sqrt(0 + eps) = 0; gamma*0 + beta = 0
    for (0..n) |i| {
        try expectClose(out_data[i], 0.0, 1e-6);
    }
}

test "layernorm: constant input (var=0 + eps)" {
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
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    // (5 - 5) / sqrt(0 + eps) = 0; gamma*0 + beta = 0
    for (0..n) |i| {
        try expectClose(out_data[i], 0.0, 1e-6);
    }
}

test "layernorm: large values ±1000 no inf/nan" {
    const allocator = testing.allocator;
    const norm_size: usize = 256;
    const outer_size: usize = 16;
    const n = outer_size * norm_size;
    const dims = [_]usize{ outer_size, norm_size };

    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    // Fill with values in [-1000, 1000]
    fillDeterministic(values, 42);
    for (values) |*v| v.* *= 1000.0;

    const t = try makeTensor(allocator, &dims, values);
    defer t.release();

    const gamma = try makeOnes(allocator, norm_size);
    defer allocator.free(gamma);
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const out = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(out_data[i]));
        try testing.expect(!std.math.isInf(out_data[i]));
    }

    // Also verify output statistics are valid
    const norm_f: f32 = @floatFromInt(norm_size);
    for (0..outer_size) |outer| {
        const base = outer * norm_size;
        var mean: f32 = 0;
        for (0..norm_size) |j| {
            mean += out_data[base + j];
        }
        mean /= norm_f;
        try testing.expect(@abs(mean) <= 1e-4);
    }
}

// ==================== Two-pass agreement ====================

test "layernorm: two-pass matches scalar reference" {
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

    const tp_out = try layernorm.layerNormTwoPass(allocator, t, 1, gamma, beta, 1e-5);
    defer tp_out.release();
    const scalar_out = try layernorm.layerNormScalar(allocator, t, 1, gamma, beta, 1e-5);
    defer scalar_out.release();

    const tp_data = tp_out.storage.dataAs(f32);
    const scalar_data = scalar_out.storage.dataAs(f32);
    for (0..n) |i| {
        try expectClose(tp_data[i], scalar_data[i], 1e-4);
    }
}

// ==================== Memory: zero leaks ====================

test "layernorm: no memory leaks via testing allocator" {
    // testing.allocator automatically detects leaks.
    // This test exercises a full create → compute → release cycle
    // for all three implementations.
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
    const beta = try makeZeros(allocator, norm_size);
    defer allocator.free(beta);

    const out1 = try layernorm.layerNorm(allocator, t, 1, gamma, beta, 1e-5);
    defer out1.release();

    const out2 = try layernorm.layerNormTwoPass(allocator, t, 1, gamma, beta, 1e-5);
    defer out2.release();

    const out3 = try layernorm.layerNormScalar(allocator, t, 1, gamma, beta, 1e-5);
    defer out3.release();

    // Verify all three produce finite values
    const d1 = out1.storage.dataAs(f32);
    const d2 = out2.storage.dataAs(f32);
    const d3 = out3.storage.dataAs(f32);
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(d1[i]));
        try testing.expect(!std.math.isNan(d2[i]));
        try testing.expect(!std.math.isNan(d3[i]));
    }
}
