const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;

const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const Tensor = synapse.tensor.core.Tensor;
const BatchNorm = synapse.ops.batchnorm.BatchNorm;

// ==================== Helpers ====================

fn expectClose(a: f32, b: f32, tol: f32) !void {
    if (@abs(a - b) > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, diff={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, @abs(a - b)),
        });
        return error.TestUnexpectedResult;
    }
}

fn makeTensor2D(allocator: std.mem.Allocator, n: usize, c: usize, values: []const f32) !Tensor(f32) {
    const storage = try Storage.create(allocator, f32, n * c);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{ n, c }));
    storage.release();
    @memcpy(t.storage.dataAs(f32)[0 .. n * c], values);
    return t;
}

// ==================== Output mean ~0 and var ~1 ====================

test "batchnorm: output mean ~0 and var ~1 (training)" {
    const N: usize = 8;
    const C: usize = 3;

    // Deterministic data with known statistics.
    var values: [N * C]f32 = undefined;
    for (0..N) |n| {
        for (0..C) |c| {
            // Channel c has values centered around c*10 with spread ~n.
            values[n * C + c] = @as(f32, @floatFromInt(c * 10)) + @as(f32, @floatFromInt(n)) - 3.5;
        }
    }

    const t = try makeTensor2D(testing.allocator, N, C, &values);
    defer t.release();

    var bn = try BatchNorm.init(testing.allocator, C, 1e-5, 0.1);
    defer bn.deinit();

    const out = try bn.forward(testing.allocator, t, true);
    defer out.release();

    const out_data = out.storage.dataAs(f32);

    // Check per-channel output mean ~0 and var ~1.
    for (0..C) |c| {
        var mean: f32 = 0;
        for (0..N) |n| {
            mean += out_data[n * C + c];
        }
        mean /= @floatFromInt(N);
        try expectClose(mean, 0.0, 1e-4);

        var variance: f32 = 0;
        for (0..N) |n| {
            const diff = out_data[n * C + c] - mean;
            variance += diff * diff;
        }
        variance /= @floatFromInt(N);
        // var(normalized) = var(x)/(var(x)+eps) ~= 1.0
        try expectClose(variance, 1.0, 1e-3);
    }
}

// ==================== Running stats updated ====================

test "batchnorm: running stats updated after training forward" {
    const N: usize = 100;
    const C: usize = 2;
    const momentum: f32 = 0.1;

    var values: [N * C]f32 = undefined;
    // Channel 0: values 1..100, Channel 1: values 101..200
    for (0..N) |n| {
        values[n * C + 0] = @as(f32, @floatFromInt(n + 1));
        values[n * C + 1] = @as(f32, @floatFromInt(n + 101));
    }

    const t = try makeTensor2D(testing.allocator, N, C, &values);
    defer t.release();

    var bn = try BatchNorm.init(testing.allocator, C, 1e-5, momentum);
    defer bn.deinit();

    // Initial running stats: mean=0, var=1.
    try testing.expectEqual(@as(f32, 0.0), bn.running_mean[0]);
    try testing.expectEqual(@as(f32, 1.0), bn.running_var[0]);

    const out = try bn.forward(testing.allocator, t, true);
    defer out.release();

    // Compute expected batch mean for channel 0: mean(1..100) = 50.5
    const expected_mean_0: f32 = 50.5;
    // running_mean = (1-0.1)*0 + 0.1*50.5 = 5.05
    try expectClose(bn.running_mean[0], (1.0 - momentum) * 0.0 + momentum * expected_mean_0, 1e-3);

    // Channel 1: mean(101..200) = 150.5
    const expected_mean_1: f32 = 150.5;
    try expectClose(bn.running_mean[1], momentum * expected_mean_1, 1e-3);

    // Running var should have been updated from initial 1.0.
    try testing.expect(bn.running_var[0] != 1.0);
    try testing.expect(bn.running_var[1] != 1.0);
}

// ==================== Inference mode ====================

test "batchnorm: inference uses running stats" {
    const N: usize = 4;
    const C: usize = 2;

    var values: [N * C]f32 = undefined;
    for (0..N) |n| {
        for (0..C) |c| {
            values[n * C + c] = @as(f32, @floatFromInt(n * C + c));
        }
    }

    const t = try makeTensor2D(testing.allocator, N, C, &values);
    defer t.release();

    var bn = try BatchNorm.init(testing.allocator, C, 1e-5, 0.1);
    defer bn.deinit();

    // Set known running stats.
    bn.running_mean[0] = 3.0;
    bn.running_mean[1] = 4.0;
    bn.running_var[0] = 4.0;
    bn.running_var[1] = 9.0;

    const out = try bn.forward(testing.allocator, t, false);
    defer out.release();

    const out_data = out.storage.dataAs(f32);
    const in_data = t.storage.dataAs(f32);

    // Verify: output = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
    // With gamma=1, beta=0.
    for (0..N) |n| {
        for (0..C) |c| {
            const x = in_data[n * C + c];
            const expected = (x - bn.running_mean[c]) / @sqrt(bn.running_var[c] + bn.eps);
            try expectClose(out_data[n * C + c], expected, 1e-5);
        }
    }

    // Running stats should NOT change in inference mode.
    try expectClose(bn.running_mean[0], 3.0, 1e-7);
    try expectClose(bn.running_var[0], 4.0, 1e-7);
}

// ==================== Welford vs two-pass agreement ====================

test "batchnorm: welford matches two-pass" {
    const batchnorm_mod = synapse.ops.batchnorm;
    const N: usize = 200;
    const C: usize = 4;

    var values: [N * C]f32 = undefined;
    // Use a simple deterministic sequence.
    for (0..N * C) |i| {
        const fi: f32 = @floatFromInt(i);
        values[i] = fi * 0.1 - 50.0;
    }

    const storage = try Storage.create(testing.allocator, f32, N * C);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{ N, C }));
    storage.release();
    defer t.release();
    @memcpy(t.storage.dataAs(f32)[0 .. N * C], &values);

    const data = t.storage.dataAs(f32);
    const stride0 = t.strides[0];
    const stride1 = t.strides[1];

    for (0..C) |c| {
        const w = batchnorm_mod.welfordMeanVar(data, t.offset, stride0, stride1, c, N);
        const tp = batchnorm_mod.twoPassMeanVar(data, t.offset, stride0, stride1, c, N);

        try expectClose(w.mean, tp.mean, 1e-3);
        try expectClose(w.variance, tp.variance, 1e-2);
    }
}

// ==================== Multiple forward passes ====================

test "batchnorm: running stats converge over multiple batches" {
    const N: usize = 16;
    const C: usize = 2;
    const momentum: f32 = 0.1;

    var bn = try BatchNorm.init(testing.allocator, C, 1e-5, momentum);
    defer bn.deinit();

    // Run 10 forward passes with same data.
    var values: [N * C]f32 = undefined;
    for (0..N) |n| {
        for (0..C) |c| {
            values[n * C + c] = @as(f32, @floatFromInt(c * 10 + n));
        }
    }
    const t = try makeTensor2D(testing.allocator, N, C, &values);
    defer t.release();

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const out = try bn.forward(testing.allocator, t, true);
        out.release();
    }

    // After many passes, running_mean should converge toward batch_mean.
    // Channel 0 batch mean = mean(0..15) = 7.5
    // Channel 1 batch mean = mean(10..25) = 17.5
    // After 10 updates: running_mean = (1-0.1)^10 * 0 + batch_mean * (1 - (1-0.1)^10)
    const decay = std.math.pow(f32, 1.0 - momentum, 10.0);
    try expectClose(bn.running_mean[0], 7.5 * (1.0 - decay), 0.5);
    try expectClose(bn.running_mean[1], 17.5 * (1.0 - decay), 0.5);
}
