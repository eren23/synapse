//! Correctness tests for MaxPool2d (with argmax indices) and AvgPool2d.
//! Verifies output values and argmax indices for backward pass correctness.

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const pool_mod = synapse.ops.pool;

// ================================================================
// Helpers
// ================================================================

fn makeTensor4d(
    allocator: std.mem.Allocator,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    seed: u32,
) !Tensor(f32) {
    const numel = n * c * h * w;
    const storage = try Storage.create(allocator, f32, numel);
    const data = storage.dataAs(f32);
    var s: u32 = seed;
    for (0..numel) |i| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        data[i] = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
    const shape = Shape.init(&[_]usize{ n, c, h, w });
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    return t;
}

fn isClose(actual: f32, expected: f32, tol: f32) bool {
    const diff = @abs(actual - expected);
    if (diff <= tol) return true;
    const denom = @max(@abs(expected), @abs(actual));
    if (denom < 1e-10) return diff <= tol;
    return diff / denom <= tol;
}

// ================================================================
// MaxPool2d tests
// ================================================================

test "maxpool2d: 2x2 stride 2" {
    const alloc = testing.allocator;

    // Create a known 1x1x4x4 input
    const storage = try Storage.create(alloc, f32, 16);
    const data = storage.dataAs(f32);
    // Row-major 4x4:
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    for (0..16) |i| {
        data[i] = @as(f32, @floatFromInt(i + 1));
    }
    const shape = Shape.init(&[_]usize{ 1, 1, 4, 4 });
    const input = Tensor(f32).init(storage, shape);
    storage.release();
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    // Output should be 1x1x2x2:
    // max(1,2,5,6)=6   max(3,4,7,8)=8
    // max(9,10,13,14)=14 max(11,12,15,16)=16
    const out = result.output.storage.dataAs(f32);
    try testing.expectEqual(result.output.shape.dims[2], 2);
    try testing.expectEqual(result.output.shape.dims[3], 2);

    try testing.expect(isClose(out[0], 6.0, 1e-6));
    try testing.expect(isClose(out[1], 8.0, 1e-6));
    try testing.expect(isClose(out[2], 14.0, 1e-6));
    try testing.expect(isClose(out[3], 16.0, 1e-6));

    // Verify argmax indices point to correct positions in 4x4 plane
    // 6 is at (1,1) = flat 5,  8 at (1,3) = flat 7
    // 14 at (3,1) = flat 13, 16 at (3,3) = flat 15
    try testing.expectEqual(result.argmax[0], 5); // (1,1)
    try testing.expectEqual(result.argmax[1], 7); // (1,3)
    try testing.expectEqual(result.argmax[2], 13); // (3,1)
    try testing.expectEqual(result.argmax[3], 15); // (3,3)
}

test "maxpool2d: 2x2 stride 1" {
    const alloc = testing.allocator;

    // 1x1x3x3 input
    const storage = try Storage.create(alloc, f32, 9);
    const data = storage.dataAs(f32);
    // 1 2 3
    // 4 5 6
    // 7 8 9
    for (0..9) |i| {
        data[i] = @as(f32, @floatFromInt(i + 1));
    }
    const shape = Shape.init(&[_]usize{ 1, 1, 3, 3 });
    const input = Tensor(f32).init(storage, shape);
    storage.release();
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 2, 2, 1, 1);
    defer result.release();

    // Output: 1x1x2x2
    // max(1,2,4,5)=5  max(2,3,5,6)=6
    // max(4,5,7,8)=8  max(5,6,8,9)=9
    const out = result.output.storage.dataAs(f32);
    try testing.expect(isClose(out[0], 5.0, 1e-6));
    try testing.expect(isClose(out[1], 6.0, 1e-6));
    try testing.expect(isClose(out[2], 8.0, 1e-6));
    try testing.expect(isClose(out[3], 9.0, 1e-6));

    // Argmax: 5->(1,1)=4, 6->(1,2)=5, 8->(2,1)=7, 9->(2,2)=8
    try testing.expectEqual(result.argmax[0], 4);
    try testing.expectEqual(result.argmax[1], 5);
    try testing.expectEqual(result.argmax[2], 7);
    try testing.expectEqual(result.argmax[3], 8);
}

test "maxpool2d: 3x3 stride 1" {
    const alloc = testing.allocator;

    // 1x1x4x4 with values 1..16
    const storage = try Storage.create(alloc, f32, 16);
    const data = storage.dataAs(f32);
    for (0..16) |i| {
        data[i] = @as(f32, @floatFromInt(i + 1));
    }
    const shape = Shape.init(&[_]usize{ 1, 1, 4, 4 });
    const input = Tensor(f32).init(storage, shape);
    storage.release();
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 3, 3, 1, 1);
    defer result.release();

    // Output: 1x1x2x2
    // top-left 3x3 max = 11, top-right = 12
    // bot-left = 15, bot-right = 16
    const out = result.output.storage.dataAs(f32);
    try testing.expect(isClose(out[0], 11.0, 1e-6));
    try testing.expect(isClose(out[1], 12.0, 1e-6));
    try testing.expect(isClose(out[2], 15.0, 1e-6));
    try testing.expect(isClose(out[3], 16.0, 1e-6));

    // Argmax: 11->(2,2)=10, 12->(2,3)=11, 15->(3,2)=14, 16->(3,3)=15
    try testing.expectEqual(result.argmax[0], 10);
    try testing.expectEqual(result.argmax[1], 11);
    try testing.expectEqual(result.argmax[2], 14);
    try testing.expectEqual(result.argmax[3], 15);
}

test "maxpool2d: multi-channel" {
    const alloc = testing.allocator;

    const input = try makeTensor4d(alloc, 1, 3, 8, 8, 42);
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    try testing.expectEqual(result.output.shape.dims[0], 1);
    try testing.expectEqual(result.output.shape.dims[1], 3);
    try testing.expectEqual(result.output.shape.dims[2], 4);
    try testing.expectEqual(result.output.shape.dims[3], 4);

    // Verify argmax indices are valid and point to the actual max
    const in_data = input.dataPtr();
    const out_data = result.output.dataPtr();

    for (0..3) |c| {
        const in_plane = in_data + c * 64; // 8*8
        const out_offset = c * 16; // 4*4

        for (0..16) |idx| {
            const max_pos = result.argmax[out_offset + idx];
            try testing.expect(max_pos < 64);
            try testing.expect(isClose(out_data[out_offset + idx], in_plane[max_pos], 1e-6));
        }
    }
}

test "maxpool2d: batch" {
    const alloc = testing.allocator;

    const input = try makeTensor4d(alloc, 2, 2, 6, 6, 99);
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    try testing.expectEqual(result.output.shape.dims[0], 2);
    try testing.expectEqual(result.output.shape.dims[1], 2);
    try testing.expectEqual(result.output.shape.dims[2], 3);
    try testing.expectEqual(result.output.shape.dims[3], 3);
}

test "maxpool2d: argmax backward correctness" {
    // Verify that using argmax indices to scatter gradients is correct
    const alloc = testing.allocator;

    const input = try makeTensor4d(alloc, 1, 1, 6, 6, 77);
    defer input.release();

    const result = try pool_mod.maxPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    const in_data = input.dataPtr();
    const out_data = result.output.dataPtr();
    const h_out: usize = 3;
    const w_out: usize = 3;
    const w_in: usize = 6;

    // For each output position, verify the argmax points to the true max
    for (0..h_out) |oh| {
        for (0..w_out) |ow| {
            const out_idx = oh * w_out + ow;
            const max_idx = result.argmax[out_idx];

            // Verify the argmax position is within the pooling window
            const max_h = max_idx / w_in;
            const max_w = max_idx % w_in;
            try testing.expect(max_h >= oh * 2 and max_h < oh * 2 + 2);
            try testing.expect(max_w >= ow * 2 and max_w < ow * 2 + 2);

            // Verify it's actually the maximum
            try testing.expect(isClose(out_data[out_idx], in_data[max_idx], 1e-6));
        }
    }
}

// ================================================================
// AvgPool2d tests
// ================================================================

test "avgpool2d: 2x2 stride 2" {
    const alloc = testing.allocator;

    const storage = try Storage.create(alloc, f32, 16);
    const data = storage.dataAs(f32);
    for (0..16) |i| {
        data[i] = @as(f32, @floatFromInt(i + 1));
    }
    const shape = Shape.init(&[_]usize{ 1, 1, 4, 4 });
    const input = Tensor(f32).init(storage, shape);
    storage.release();
    defer input.release();

    const result = try pool_mod.avgPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    // avg(1,2,5,6)=3.5  avg(3,4,7,8)=5.5
    // avg(9,10,13,14)=11.5  avg(11,12,15,16)=13.5
    const out = result.storage.dataAs(f32);
    try testing.expect(isClose(out[0], 3.5, 1e-6));
    try testing.expect(isClose(out[1], 5.5, 1e-6));
    try testing.expect(isClose(out[2], 11.5, 1e-6));
    try testing.expect(isClose(out[3], 13.5, 1e-6));
}

test "avgpool2d: 3x3 stride 1" {
    const alloc = testing.allocator;

    // 1x1x4x4 with all 1s -> average should be 1.0
    const storage = try Storage.create(alloc, f32, 16);
    const data = storage.dataAs(f32);
    @memset(data, 1.0);
    const shape = Shape.init(&[_]usize{ 1, 1, 4, 4 });
    const input = Tensor(f32).init(storage, shape);
    storage.release();
    defer input.release();

    const result = try pool_mod.avgPool2d(alloc, input, 3, 3, 1, 1);
    defer result.release();

    const out = result.storage.dataAs(f32);
    for (0..4) |i| {
        try testing.expect(isClose(out[i], 1.0, 1e-6));
    }
}

test "avgpool2d: multi-channel batch" {
    const alloc = testing.allocator;

    const input = try makeTensor4d(alloc, 2, 3, 8, 8, 55);
    defer input.release();

    const result = try pool_mod.avgPool2d(alloc, input, 2, 2, 2, 2);
    defer result.release();

    try testing.expectEqual(result.shape.dims[0], 2);
    try testing.expectEqual(result.shape.dims[1], 3);
    try testing.expectEqual(result.shape.dims[2], 4);
    try testing.expectEqual(result.shape.dims[3], 4);

    // Verify manually for first element: average of 2x2 top-left block
    const in_data = input.dataPtr();
    const out_data = result.dataPtr();
    const expected = (in_data[0] + in_data[1] + in_data[8] + in_data[9]) / 4.0;
    try testing.expect(isClose(out_data[0], expected, 1e-5));
}
