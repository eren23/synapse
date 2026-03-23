const std = @import("std");
const synapse = @import("synapse");
const testing = std.testing;

const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const Tensor = synapse.tensor.core.Tensor;
const reduce = synapse.ops.reduce;

// ==================== Helpers ====================

fn makeTensor3D(allocator: std.mem.Allocator) !Tensor(f32) {
    // Create a [2,3,4] tensor with values 0..23
    const storage = try Storage.create(allocator, f32, 24);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{ 2, 3, 4 }));
    storage.release();
    const data = t.storage.dataAs(f32);
    for (0..24) |i| {
        data[i] = @floatFromInt(i);
    }
    return t;
}

fn expectClose(a: f32, b: f32, tol: f32) !void {
    if (@abs(a - b) > tol) {
        std.debug.print("FAIL: got={e}, expected={e}, diff={e}\n", .{
            @as(f64, a), @as(f64, b), @as(f64, @abs(a - b)),
        });
        return error.TestUnexpectedResult;
    }
}

// ==================== reduceSum ====================

test "reduceSum: axis=0 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceSum(testing.allocator, t, 0, false);
    defer result.release();

    // Output shape: [3,4]
    try testing.expectEqual(@as(usize, 2), result.shape.ndim);
    try testing.expectEqual(@as(usize, 3), result.shape.dims[0]);
    try testing.expectEqual(@as(usize, 4), result.shape.dims[1]);

    // result[j,k] = t[0,j,k] + t[1,j,k] = (j*4+k) + (12+j*4+k)
    for (0..3) |j| {
        for (0..4) |k| {
            const expected: f32 = @floatFromInt(j * 4 + k + 12 + j * 4 + k);
            try expectClose(result.at(&[_]usize{ j, k }), expected, 1e-5);
        }
    }
}

test "reduceSum: axis=1 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceSum(testing.allocator, t, 1, false);
    defer result.release();

    // Output shape: [2,4]
    try testing.expectEqual(@as(usize, 2), result.shape.ndim);
    try testing.expectEqual(@as(usize, 2), result.shape.dims[0]);
    try testing.expectEqual(@as(usize, 4), result.shape.dims[1]);

    // result[i,k] = sum_j t[i,j,k] = sum_j (12*i + 4*j + k) for j=0..2
    for (0..2) |i| {
        for (0..4) |k| {
            var expected: f32 = 0;
            for (0..3) |j| {
                expected += @floatFromInt(i * 12 + j * 4 + k);
            }
            try expectClose(result.at(&[_]usize{ i, k }), expected, 1e-5);
        }
    }
}

test "reduceSum: axis=2 of [2,3,4] (SIMD path)" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceSum(testing.allocator, t, 2, false);
    defer result.release();

    // Output shape: [2,3]
    try testing.expectEqual(@as(usize, 2), result.shape.ndim);
    try testing.expectEqual(@as(usize, 2), result.shape.dims[0]);
    try testing.expectEqual(@as(usize, 3), result.shape.dims[1]);

    // result[i,j] = sum_k (12*i + 4*j + k) for k=0..3 = 4*(12*i+4*j) + 6
    for (0..2) |i| {
        for (0..3) |j| {
            var expected: f32 = 0;
            for (0..4) |k| {
                expected += @floatFromInt(i * 12 + j * 4 + k);
            }
            try expectClose(result.at(&[_]usize{ i, j }), expected, 1e-5);
        }
    }
}

// ==================== keepdim ====================

test "reduceSum: axis=1 keepdim=true" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceSum(testing.allocator, t, 1, true);
    defer result.release();

    // Output shape: [2,1,4]
    try testing.expectEqual(@as(usize, 3), result.shape.ndim);
    try testing.expectEqual(@as(usize, 2), result.shape.dims[0]);
    try testing.expectEqual(@as(usize, 1), result.shape.dims[1]);
    try testing.expectEqual(@as(usize, 4), result.shape.dims[2]);

    // Values should match the non-keepdim version.
    const ref = try reduce.reduceSum(testing.allocator, t, 1, false);
    defer ref.release();

    for (0..2) |i| {
        for (0..4) |k| {
            try expectClose(
                result.at(&[_]usize{ i, 0, k }),
                ref.at(&[_]usize{ i, k }),
                1e-5,
            );
        }
    }
}

// ==================== reduceMean ====================

test "reduceMean: axis=2 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceMean(testing.allocator, t, 2, false);
    defer result.release();

    for (0..2) |i| {
        for (0..3) |j| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += @floatFromInt(i * 12 + j * 4 + k);
            }
            try expectClose(result.at(&[_]usize{ i, j }), sum / 4.0, 1e-5);
        }
    }
}

// ==================== reduceMax ====================

test "reduceMax: axis=0 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceMax(testing.allocator, t, 0, false);
    defer result.release();

    // max along axis 0: max(t[0,j,k], t[1,j,k]) = t[1,j,k] (since values increase)
    for (0..3) |j| {
        for (0..4) |k| {
            const expected: f32 = @floatFromInt(12 + j * 4 + k);
            try expectClose(result.at(&[_]usize{ j, k }), expected, 1e-5);
        }
    }
}

test "reduceMax: axis=2 of [2,3,4] (SIMD path)" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceMax(testing.allocator, t, 2, false);
    defer result.release();

    for (0..2) |i| {
        for (0..3) |j| {
            const expected: f32 = @floatFromInt(i * 12 + j * 4 + 3);
            try expectClose(result.at(&[_]usize{ i, j }), expected, 1e-5);
        }
    }
}

// ==================== reduceMin ====================

test "reduceMin: axis=2 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.reduceMin(testing.allocator, t, 2, false);
    defer result.release();

    for (0..2) |i| {
        for (0..3) |j| {
            const expected: f32 = @floatFromInt(i * 12 + j * 4);
            try expectClose(result.at(&[_]usize{ i, j }), expected, 1e-5);
        }
    }
}

// ==================== argmax ====================

test "argmax: axis=2 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.argmax(testing.allocator, t, 2, false);
    defer result.release();

    // Values increase along axis=2, so argmax is always 3.
    for (0..2) |i| {
        for (0..3) |j| {
            try expectClose(result.at(&[_]usize{ i, j }), 3.0, 1e-5);
        }
    }
}

test "argmax: axis=0 of [2,3,4]" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = try reduce.argmax(testing.allocator, t, 0, false);
    defer result.release();

    // Values increase along axis=0, so argmax is always 1.
    for (0..3) |j| {
        for (0..4) |k| {
            try expectClose(result.at(&[_]usize{ j, k }), 1.0, 1e-5);
        }
    }
}

// ==================== 1D reduce ====================

test "reduceSum: 1D tensor" {
    const storage = try Storage.create(testing.allocator, f32, 5);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{5}));
    storage.release();
    defer t.release();

    const data = t.storage.dataAs(f32);
    for (0..5) |i| {
        data[i] = @floatFromInt(i + 1);
    }

    const result = try reduce.reduceSum(testing.allocator, t, 0, false);
    defer result.release();

    // Sum of 1+2+3+4+5 = 15
    try testing.expectEqual(@as(usize, 0), result.shape.ndim);
    try expectClose(result.storage.dataAs(f32)[0], 15.0, 1e-5);
}

// ==================== invalid axis ====================

test "reduceSum: invalid axis returns error" {
    const t = try makeTensor3D(testing.allocator);
    defer t.release();

    const result = reduce.reduceSum(testing.allocator, t, 3, false);
    try testing.expectError(error.InvalidAxis, result);
}
