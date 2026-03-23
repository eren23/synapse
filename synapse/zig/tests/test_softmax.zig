const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;
const math = std.math;

const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const Tensor = synapse.tensor.core.Tensor;
const softmax_ops = synapse.ops.softmax;

// ==================== Helpers ====================

fn expectClose(a: f32, b: f32, tol: f32) !void {
    if (math.isNan(a) or math.isNan(b)) {
        if (math.isNan(a) and math.isNan(b)) return;
        std.debug.print("FAIL: got={e}, expected={e}\n", .{ @as(f64, a), @as(f64, b) });
        return error.TestUnexpectedResult;
    }
    if (@abs(a - b) > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, diff={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, @abs(a - b)),
        });
        return error.TestUnexpectedResult;
    }
}

fn makeTensor2D(allocator: std.mem.Allocator, rows: usize, cols: usize, values: []const f32) !Tensor(f32) {
    const storage = try Storage.create(allocator, f32, rows * cols);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{ rows, cols }));
    storage.release();
    const data = t.storage.dataAs(f32);
    @memcpy(data[0 .. rows * cols], values);
    return t;
}

// ==================== softmax sums to 1.0 ====================

test "softmax: each row sums to 1.0" {
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2, 0.3, 0.4, 0.5 };
    const t = try makeTensor2D(testing.allocator, 2, 5, &values);
    defer t.release();

    const result = try softmax_ops.softmax(testing.allocator, t, 1);
    defer result.release();

    // Same shape as input.
    try testing.expectEqual(@as(usize, 2), result.shape.ndim);
    try testing.expectEqual(@as(usize, 2), result.shape.dims[0]);
    try testing.expectEqual(@as(usize, 5), result.shape.dims[1]);

    // Each row sums to 1.0.
    for (0..2) |row| {
        var sum: f32 = 0;
        for (0..5) |col| {
            const val = result.at(&[_]usize{ row, col });
            try testing.expect(val >= 0.0);
            try testing.expect(val <= 1.0);
            sum += val;
        }
        try expectClose(sum, 1.0, 1e-5);
    }
}

test "softmax: monotonicity preserved" {
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const t = try makeTensor2D(testing.allocator, 1, 5, &values);
    defer t.release();

    const result = try softmax_ops.softmax(testing.allocator, t, 1);
    defer result.release();

    // Softmax preserves ordering: larger input -> larger output.
    for (0..4) |i| {
        try testing.expect(result.at(&[_]usize{ 0, i }) < result.at(&[_]usize{ 0, i + 1 }));
    }
}

// ==================== numerical stability ====================

test "softmax: handles [-1000, 1000] without overflow" {
    const values = [_]f32{ -1000.0, -500.0, 0.0, 500.0, 1000.0 };
    const t = try makeTensor2D(testing.allocator, 1, 5, &values);
    defer t.release();

    const result = try softmax_ops.softmax(testing.allocator, t, 1);
    defer result.release();

    var sum: f32 = 0;
    for (0..5) |i| {
        const val = result.at(&[_]usize{ 0, i });
        // No inf or NaN.
        try testing.expect(!math.isNan(val));
        try testing.expect(!math.isInf(val));
        try testing.expect(val >= 0.0);
        sum += val;
    }
    try expectClose(sum, 1.0, 1e-5);

    // The largest input (1000) should dominate.
    try expectClose(result.at(&[_]usize{ 0, 4 }), 1.0, 1e-5);
}

test "softmax: uniform input gives uniform output" {
    const values = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    const t = try makeTensor2D(testing.allocator, 1, 4, &values);
    defer t.release();

    const result = try softmax_ops.softmax(testing.allocator, t, 1);
    defer result.release();

    for (0..4) |i| {
        try expectClose(result.at(&[_]usize{ 0, i }), 0.25, 1e-5);
    }
}

// ==================== softmax along axis=0 ====================

test "softmax: axis=0" {
    // 2x3 matrix, softmax along columns.
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const t = try makeTensor2D(testing.allocator, 2, 3, &values);
    defer t.release();

    const result = try softmax_ops.softmax(testing.allocator, t, 0);
    defer result.release();

    // Each column should sum to 1.0.
    for (0..3) |col| {
        var sum: f32 = 0;
        for (0..2) |row| {
            const val = result.at(&[_]usize{ row, col });
            try testing.expect(!math.isNan(val));
            sum += val;
        }
        try expectClose(sum, 1.0, 1e-5);
    }
}

// ==================== LogSoftmax ====================

test "logSoftmax: log(softmax(x)) identity" {
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const t = try makeTensor2D(testing.allocator, 1, 5, &values);
    defer t.release();

    const sm = try softmax_ops.softmax(testing.allocator, t, 1);
    defer sm.release();

    const lsm = try softmax_ops.logSoftmax(testing.allocator, t, 1);
    defer lsm.release();

    // logSoftmax should equal log(softmax).
    for (0..5) |i| {
        const expected = @log(sm.at(&[_]usize{ 0, i }));
        try expectClose(lsm.at(&[_]usize{ 0, i }), expected, 1e-5);
    }
}

test "logSoftmax: handles [-1000, 1000]" {
    const values = [_]f32{ -1000.0, 0.0, 1000.0 };
    const t = try makeTensor2D(testing.allocator, 1, 3, &values);
    defer t.release();

    const result = try softmax_ops.logSoftmax(testing.allocator, t, 1);
    defer result.release();

    for (0..3) |i| {
        const val = result.at(&[_]usize{ 0, i });
        try testing.expect(!math.isNan(val));
        try testing.expect(!math.isInf(val) or val == -math.inf(f32));
        // LogSoftmax values are always <= 0.
        try testing.expect(val <= 0.0 + 1e-5);
    }

    // The max element (1000) should have logSoftmax close to 0.
    try expectClose(result.at(&[_]usize{ 0, 2 }), 0.0, 1e-5);
}

test "logSoftmax: sum(exp(logsoftmax)) = 1.0" {
    const values = [_]f32{ 2.0, 3.0, 1.0, 5.0 };
    const t = try makeTensor2D(testing.allocator, 1, 4, &values);
    defer t.release();

    const lsm = try softmax_ops.logSoftmax(testing.allocator, t, 1);
    defer lsm.release();

    var sum: f32 = 0;
    for (0..4) |i| {
        sum += @exp(lsm.at(&[_]usize{ 0, i }));
    }
    try expectClose(sum, 1.0, 1e-5);
}

// ==================== 1D softmax ====================

test "softmax: 1D tensor" {
    const storage = try Storage.create(testing.allocator, f32, 3);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{3}));
    storage.release();
    defer t.release();

    const data = t.storage.dataAs(f32);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;

    const result = try softmax_ops.softmax(testing.allocator, t, 0);
    defer result.release();

    var sum: f32 = 0;
    for (0..3) |i| {
        sum += result.storage.dataAs(f32)[i];
    }
    try expectClose(sum, 1.0, 1e-5);
}
