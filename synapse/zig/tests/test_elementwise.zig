const std = @import("std");
const ew = @import("elementwise");
const shape_mod = @import("shape");
const Shape = shape_mod.Shape;
const testing = std.testing;
const math = std.math;

// ============================================================
// Helpers
// ============================================================

fn relError(approx: f32, exact: f32) f64 {
    if (math.isNan(approx) and math.isNan(exact)) return 0.0;
    if (math.isNan(approx) or math.isNan(exact)) return math.inf(f64);
    if (approx == exact) return 0.0;
    const abs_exact: f64 = @abs(@as(f64, exact));
    const diff: f64 = @abs(@as(f64, approx) - @as(f64, exact));
    if (abs_exact < 1e-30) return diff;
    return diff / abs_exact;
}

fn expectClose(approx: f32, exact: f32, tol: f64) !void {
    const err = relError(approx, exact);
    if (err > tol) {
        std.debug.print("FAIL: approx={e}, exact={e}, relError={e}, tol={e}\n", .{
            @as(f64, approx), @as(f64, exact), err, tol,
        });
        return error.TestUnexpectedResult;
    }
}

fn refSigmoid(x: f32) f32 {
    return @floatCast(1.0 / (1.0 + @exp(-@as(f64, x))));
}

fn refTanh(x: f32) f32 {
    const xf: f64 = @as(f64, x);
    const exp2x = @exp(2.0 * xf);
    return @floatCast((exp2x - 1.0) / (exp2x + 1.0));
}

fn refGelu(x: f32) f32 {
    const xf: f64 = @as(f64, x);
    const inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf);
    const exp2inner = @exp(2.0 * inner);
    const tanh_val = (exp2inner - 1.0) / (exp2inner + 1.0);
    return @floatCast(0.5 * xf * (1.0 + tanh_val));
}

// ============================================================
// Element-wise add
// ============================================================

test "add: basic correctness" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0 };
    var dst: [7]f32 = undefined;
    ew.add(&dst, &a, &b);
    for (0..7) |i| {
        try testing.expectEqual(a[i] + b[i], dst[i]);
    }
}

test "add: length 0" {
    var dst: [0]f32 = .{};
    ew.add(&dst, &.{}, &.{});
}

test "add: length 1000" {
    var a: [1000]f32 = undefined;
    var b: [1000]f32 = undefined;
    var dst: [1000]f32 = undefined;
    for (0..1000) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(1000 - i)) * 0.01;
    }
    ew.add(&dst, &a, &b);
    for (0..1000) |i| {
        try expectClose(dst[i], a[i] + b[i], 1e-6);
    }
}

// ============================================================
// Element-wise sub
// ============================================================

test "sub: basic correctness" {
    var a = [_]f32{ 10.0, 20.0, 30.0, 4.5, 5.5, 6.5, 7.5 };
    var b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var dst: [7]f32 = undefined;
    ew.sub(&dst, &a, &b);
    for (0..7) |i| {
        try expectClose(dst[i], a[i] - b[i], 1e-6);
    }
}

test "sub: negative results" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var dst: [4]f32 = undefined;
    ew.sub(&dst, &a, &b);
    for (0..4) |i| {
        try testing.expectEqual(a[i] - b[i], dst[i]);
    }
}

// ============================================================
// Element-wise mul
// ============================================================

test "mul: basic correctness" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var dst: [7]f32 = undefined;
    ew.mul(&dst, &a, &b);
    for (0..7) |i| {
        try expectClose(dst[i], a[i] * b[i], 1e-6);
    }
}

test "mul: zeros and negatives" {
    var a = [_]f32{ 0.0, -1.0, -2.0, 3.0 };
    var b = [_]f32{ 5.0, -3.0, 4.0, 0.0 };
    var dst: [4]f32 = undefined;
    ew.mul(&dst, &a, &b);
    try testing.expectEqual(@as(f32, 0.0), dst[0]);
    try testing.expectEqual(@as(f32, 3.0), dst[1]);
    try testing.expectEqual(@as(f32, -8.0), dst[2]);
    try testing.expectEqual(@as(f32, 0.0), dst[3]);
}

// ============================================================
// Element-wise div
// ============================================================

test "div: basic correctness" {
    var a = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0 };
    var b = [_]f32{ 2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0 };
    var dst: [7]f32 = undefined;
    ew.div(&dst, &a, &b);
    for (0..7) |i| {
        try expectClose(dst[i], a[i] / b[i], 1e-6);
    }
}

test "div: special values" {
    var a = [_]f32{ 1.0, 0.0, 1.0, -1.0 };
    var b = [_]f32{ 0.0, 0.0, math.inf(f32), math.inf(f32) };
    var dst: [4]f32 = undefined;
    ew.div(&dst, &a, &b);
    try testing.expect(math.isPositiveInf(dst[0])); // 1/0 = inf
    try testing.expect(math.isNan(dst[1])); // 0/0 = NaN
    try testing.expectEqual(@as(f32, 0.0), dst[2]); // 1/inf = 0
    try testing.expectEqual(@as(f32, -0.0), dst[3]); // -1/inf = -0
}

// ============================================================
// Fused add+relu
// ============================================================

test "addRelu: correctness against separate add+relu" {
    var a: [1000]f32 = undefined;
    var b: [1000]f32 = undefined;
    var fused: [1000]f32 = undefined;
    var separate: [1000]f32 = undefined;

    for (0..1000) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.02 - 10.0;
        b[i] = @as(f32, @floatFromInt((i * 3 + 7) % 500)) * 0.02 - 5.0;
    }

    // Fused
    ew.addRelu(&fused, &a, &b);

    // Separate
    ew.add(&separate, &a, &b);
    ew.relu(&separate, &separate);

    for (0..1000) |i| {
        try expectClose(fused[i], separate[i], 1e-6);
    }
}

test "addRelu: negative results clamped to zero" {
    var a = [_]f32{ -10.0, -5.0, 1.0, 3.0 };
    var b = [_]f32{ 1.0, -1.0, -0.5, 2.0 };
    var dst: [4]f32 = undefined;
    ew.addRelu(&dst, &a, &b);
    try testing.expectEqual(@as(f32, 0.0), dst[0]); // -10+1 = -9 -> 0
    try testing.expectEqual(@as(f32, 0.0), dst[1]); // -5-1 = -6 -> 0
    try testing.expectEqual(@as(f32, 0.5), dst[2]); // 1-0.5 = 0.5
    try testing.expectEqual(@as(f32, 5.0), dst[3]); // 3+2 = 5
}

// ============================================================
// Fused mul+add (FMA)
// ============================================================

test "mulAdd: correctness against scalar reference" {
    var a: [100]f32 = undefined;
    var b: [100]f32 = undefined;
    var c: [100]f32 = undefined;
    var dst: [100]f32 = undefined;

    for (0..100) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(100 - i)) * 0.1;
        c[i] = @as(f32, @floatFromInt(i % 50)) * 0.01;
    }

    ew.mulAdd(&dst, &a, &b, &c);
    for (0..100) |i| {
        const expected = @mulAdd(f32, a[i], b[i], c[i]);
        try expectClose(dst[i], expected, 1e-6);
    }
}

// ============================================================
// ReLU activation
// ============================================================

test "relu: positives unchanged, negatives zeroed" {
    var src = [_]f32{ -3.0, -1.0, -0.001, 0.0, 0.001, 1.0, 3.0, 100.0 };
    var dst: [8]f32 = undefined;
    ew.relu(&dst, &src);
    try testing.expectEqual(@as(f32, 0.0), dst[0]);
    try testing.expectEqual(@as(f32, 0.0), dst[1]);
    try testing.expectEqual(@as(f32, 0.0), dst[2]);
    try testing.expectEqual(@as(f32, 0.0), dst[3]);
    try testing.expectEqual(@as(f32, 0.001), dst[4]);
    try testing.expectEqual(@as(f32, 1.0), dst[5]);
    try testing.expectEqual(@as(f32, 3.0), dst[6]);
    try testing.expectEqual(@as(f32, 100.0), dst[7]);
}

test "relu: large values" {
    var src = [_]f32{ -1e30, -1e10, 1e10, 1e30 };
    var dst: [4]f32 = undefined;
    ew.relu(&dst, &src);
    try testing.expectEqual(@as(f32, 0.0), dst[0]);
    try testing.expectEqual(@as(f32, 0.0), dst[1]);
    try testing.expectEqual(@as(f32, 1e10), dst[2]);
    try testing.expectEqual(@as(f32, 1e30), dst[3]);
}

// ============================================================
// Leaky ReLU activation
// ============================================================

test "leakyRelu: positive pass-through, negative scaled" {
    const alpha: f32 = 0.01;
    var src = [_]f32{ -10.0, -1.0, 0.0, 1.0, 10.0, -0.5, 0.5, 100.0 };
    var dst: [8]f32 = undefined;
    ew.leakyRelu(&dst, &src, alpha);
    for (0..8) |i| {
        const expected: f32 = if (src[i] > 0.0) src[i] else alpha * src[i];
        try expectClose(dst[i], expected, 1e-6);
    }
}

test "leakyRelu: alpha=0 is relu" {
    var src = [_]f32{ -5.0, -1.0, 0.0, 1.0, 5.0 };
    var relu_dst: [5]f32 = undefined;
    var leaky_dst: [5]f32 = undefined;
    ew.relu(&relu_dst, &src);
    ew.leakyRelu(&leaky_dst, &src, 0.0);
    for (0..5) |i| {
        try testing.expectEqual(relu_dst[i], leaky_dst[i]);
    }
}

// ============================================================
// Sigmoid activation
// ============================================================

test "sigmoid: basic accuracy" {
    const inputs = [_]f32{ 0.0, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 0.001, -0.001 };
    var dst: [inputs.len]f32 = undefined;
    ew.sigmoid(&dst, &inputs);
    for (0..inputs.len) |i| {
        try expectClose(dst[i], refSigmoid(inputs[i]), 1e-5);
    }
}

test "sigmoid: zeros and large values" {
    var src = [_]f32{ 0.0, 50.0, -50.0, 100.0, -100.0 };
    var dst: [5]f32 = undefined;
    ew.sigmoid(&dst, &src);
    try expectClose(dst[0], 0.5, 1e-6);
    try expectClose(dst[1], 1.0, 1e-5);
    try expectClose(dst[2], 0.0, 1e-5);
    try expectClose(dst[3], 1.0, 1e-5);
    try expectClose(dst[4], 0.0, 1e-5);
}

test "sigmoid: accuracy sweep 1000" {
    var inputs: [1000]f32 = undefined;
    var dst: [1000]f32 = undefined;
    for (0..1000) |i| {
        inputs[i] = -20.0 + @as(f32, @floatFromInt(i)) * 0.04;
    }
    ew.sigmoid(&dst, &inputs);
    var max_err: f64 = 0;
    for (0..1000) |i| {
        max_err = @max(max_err, relError(dst[i], refSigmoid(inputs[i])));
    }
    try testing.expect(max_err < 1e-5);
}

// ============================================================
// Tanh activation
// ============================================================

test "tanh: basic accuracy" {
    const inputs = [_]f32{ 0.0, 1.0, -1.0, 0.5, -0.5, 5.0, -5.0, 10.0, -10.0 };
    var dst: [inputs.len]f32 = undefined;
    ew.tanh(&dst, &inputs);
    for (0..inputs.len) |i| {
        try expectClose(dst[i], refTanh(inputs[i]), 1e-5);
    }
}

test "tanh: zeros and large values" {
    var src = [_]f32{ 0.0, 50.0, -50.0, 100.0, -100.0 };
    var dst: [5]f32 = undefined;
    ew.tanh(&dst, &src);
    try expectClose(dst[0], 0.0, 1e-7);
    try expectClose(dst[1], 1.0, 1e-5);
    try expectClose(dst[2], -1.0, 1e-5);
    try expectClose(dst[3], 1.0, 1e-5);
    try expectClose(dst[4], -1.0, 1e-5);
}

test "tanh: accuracy sweep 1000" {
    var inputs: [1000]f32 = undefined;
    var dst: [1000]f32 = undefined;
    for (0..1000) |i| {
        inputs[i] = -10.0 + @as(f32, @floatFromInt(i)) * 0.02;
    }
    ew.tanh(&dst, &inputs);
    var max_err: f64 = 0;
    for (0..1000) |i| {
        max_err = @max(max_err, relError(dst[i], refTanh(inputs[i])));
    }
    try testing.expect(max_err < 1e-5);
}

// ============================================================
// GELU activation
// ============================================================

test "gelu: basic accuracy" {
    const inputs = [_]f32{ 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0 };
    var dst: [inputs.len]f32 = undefined;
    ew.gelu(&dst, &inputs);
    for (0..inputs.len) |i| {
        const abs_err = @abs(@as(f64, dst[i]) - @as(f64, refGelu(inputs[i])));
        if (abs_err > 1e-4) {
            std.debug.print("FAIL gelu: input={e}, approx={e}, exact={e}, abs_err={e}\n", .{
                @as(f64, inputs[i]), @as(f64, dst[i]), @as(f64, refGelu(inputs[i])), abs_err,
            });
            return error.TestUnexpectedResult;
        }
    }
}

test "gelu: known values" {
    // gelu(0) = 0
    var src = [_]f32{0.0};
    var dst: [1]f32 = undefined;
    ew.gelu(&dst, &src);
    try testing.expectEqual(@as(f32, 0.0), dst[0]);
}

test "gelu: negatives approach zero, positives approach identity" {
    var src = [_]f32{ -5.0, -3.0, 3.0, 5.0 };
    var dst: [4]f32 = undefined;
    ew.gelu(&dst, &src);
    // For large negative, gelu(x) -> 0
    try expectClose(dst[0], 0.0, 1e-3);
    try testing.expect(@abs(@as(f64, dst[1]) - @as(f64, refGelu(-3.0))) < 1e-4);
    // For large positive, gelu(x) -> x
    try expectClose(dst[2], refGelu(3.0), 1e-5);
    try expectClose(dst[3], 5.0, 1e-3);
}

test "gelu: accuracy sweep 1000" {
    var inputs: [1000]f32 = undefined;
    var dst: [1000]f32 = undefined;
    for (0..1000) |i| {
        inputs[i] = -5.0 + @as(f32, @floatFromInt(i)) * 0.01;
    }
    ew.gelu(&dst, &inputs);
    // Use absolute error because GELU crosses zero, causing catastrophic
    // cancellation in relative error for values near gelu(x) ≈ 0.
    var max_abs_err: f64 = 0;
    for (0..1000) |i| {
        const abs_err = @abs(@as(f64, dst[i]) - @as(f64, refGelu(inputs[i])));
        max_abs_err = @max(max_abs_err, abs_err);
    }
    try testing.expect(max_abs_err < 1e-4);
}

// ============================================================
// Broadcasting: scalar + tensor
// ============================================================

test "broadcast add: scalar + tensor" {
    const scalar_shape = Shape.init(&.{1});
    const tensor_shape = Shape.init(&.{6});
    const scalar = [_]f32{10.0};
    var tensor = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var dst: [6]f32 = undefined;

    try ew.broadcastAdd(&dst, &scalar, scalar_shape, &tensor, tensor_shape);
    for (0..6) |i| {
        try testing.expectEqual(10.0 + tensor[i], dst[i]);
    }
}

test "broadcast sub: tensor - scalar" {
    const tensor_shape = Shape.init(&.{5});
    const scalar_shape = Shape.init(&.{1});
    var tensor = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const scalar = [_]f32{5.0};
    var dst: [5]f32 = undefined;

    try ew.broadcastSub(&dst, &tensor, tensor_shape, &scalar, scalar_shape);
    for (0..5) |i| {
        try testing.expectEqual(tensor[i] - 5.0, dst[i]);
    }
}

test "broadcast mul: scalar * tensor" {
    const scalar_shape = Shape.init(&.{1});
    const tensor_shape = Shape.init(&.{4});
    const scalar = [_]f32{3.0};
    var tensor = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;

    try ew.broadcastMul(&dst, &scalar, scalar_shape, &tensor, tensor_shape);
    for (0..4) |i| {
        try testing.expectEqual(3.0 * tensor[i], dst[i]);
    }
}

test "broadcast div: tensor / scalar" {
    const tensor_shape = Shape.init(&.{4});
    const scalar_shape = Shape.init(&.{1});
    var tensor = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const scalar = [_]f32{5.0};
    var dst: [4]f32 = undefined;

    try ew.broadcastDiv(&dst, &tensor, tensor_shape, &scalar, scalar_shape);
    for (0..4) |i| {
        try testing.expectEqual(tensor[i] / 5.0, dst[i]);
    }
}

// ============================================================
// Broadcasting: [1,N] + [M,N] (row broadcast)
// ============================================================

test "broadcast add: [1,N] + [M,N]" {
    // a = [1,4]: [1, 2, 3, 4]
    // b = [3,4]: [[10,20,30,40], [50,60,70,80], [90,100,110,120]]
    // result = [3,4]: a broadcast over M rows
    const a_shape = Shape.init(&.{ 1, 4 });
    const b_shape = Shape.init(&.{ 3, 4 });
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0 };
    var dst: [12]f32 = undefined;

    try ew.broadcastAdd(&dst, &a, a_shape, &b, b_shape);

    // Row 0: [1+10, 2+20, 3+30, 4+40]
    try testing.expectEqual(@as(f32, 11.0), dst[0]);
    try testing.expectEqual(@as(f32, 22.0), dst[1]);
    try testing.expectEqual(@as(f32, 33.0), dst[2]);
    try testing.expectEqual(@as(f32, 44.0), dst[3]);
    // Row 1: [1+50, 2+60, 3+70, 4+80]
    try testing.expectEqual(@as(f32, 51.0), dst[4]);
    try testing.expectEqual(@as(f32, 62.0), dst[5]);
    try testing.expectEqual(@as(f32, 73.0), dst[6]);
    try testing.expectEqual(@as(f32, 84.0), dst[7]);
    // Row 2: [1+90, 2+100, 3+110, 4+120]
    try testing.expectEqual(@as(f32, 91.0), dst[8]);
    try testing.expectEqual(@as(f32, 102.0), dst[9]);
    try testing.expectEqual(@as(f32, 113.0), dst[10]);
    try testing.expectEqual(@as(f32, 124.0), dst[11]);
}

// ============================================================
// Broadcasting: [M,1] + [M,N] (column broadcast)
// ============================================================

test "broadcast add: [M,1] + [M,N]" {
    // a = [3,1]: [100, 200, 300]
    // b = [3,4]: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    // result = [3,4]
    const a_shape = Shape.init(&.{ 3, 1 });
    const b_shape = Shape.init(&.{ 3, 4 });
    const a = [_]f32{ 100.0, 200.0, 300.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var dst: [12]f32 = undefined;

    try ew.broadcastAdd(&dst, &a, a_shape, &b, b_shape);

    // Row 0: [100+1, 100+2, 100+3, 100+4]
    try testing.expectEqual(@as(f32, 101.0), dst[0]);
    try testing.expectEqual(@as(f32, 102.0), dst[1]);
    try testing.expectEqual(@as(f32, 103.0), dst[2]);
    try testing.expectEqual(@as(f32, 104.0), dst[3]);
    // Row 1: [200+5, 200+6, 200+7, 200+8]
    try testing.expectEqual(@as(f32, 205.0), dst[4]);
    try testing.expectEqual(@as(f32, 206.0), dst[5]);
    try testing.expectEqual(@as(f32, 207.0), dst[6]);
    try testing.expectEqual(@as(f32, 208.0), dst[7]);
    // Row 2: [300+9, 300+10, 300+11, 300+12]
    try testing.expectEqual(@as(f32, 309.0), dst[8]);
    try testing.expectEqual(@as(f32, 310.0), dst[9]);
    try testing.expectEqual(@as(f32, 311.0), dst[10]);
    try testing.expectEqual(@as(f32, 312.0), dst[11]);
}

// ============================================================
// Broadcasting: [M,N] + [1,N] (reverse row broadcast)
// ============================================================

test "broadcast mul: [M,N] * [1,N]" {
    const a_shape = Shape.init(&.{ 2, 3 });
    const b_shape = Shape.init(&.{ 1, 3 });
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = [_]f32{ 10.0, 100.0, 1000.0 };
    var dst: [6]f32 = undefined;

    try ew.broadcastMul(&dst, &a, a_shape, &b, b_shape);

    try testing.expectEqual(@as(f32, 10.0), dst[0]);
    try testing.expectEqual(@as(f32, 200.0), dst[1]);
    try testing.expectEqual(@as(f32, 3000.0), dst[2]);
    try testing.expectEqual(@as(f32, 40.0), dst[3]);
    try testing.expectEqual(@as(f32, 500.0), dst[4]);
    try testing.expectEqual(@as(f32, 6000.0), dst[5]);
}

// ============================================================
// Broadcasting: [M,N] + [M,1] (reverse column broadcast)
// ============================================================

test "broadcast sub: [M,N] - [M,1]" {
    const a_shape = Shape.init(&.{ 2, 3 });
    const b_shape = Shape.init(&.{ 2, 1 });
    const a = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };
    const b = [_]f32{ 1.0, 2.0 };
    var dst: [6]f32 = undefined;

    try ew.broadcastSub(&dst, &a, a_shape, &b, b_shape);

    // Row 0: [10-1, 20-1, 30-1]
    try testing.expectEqual(@as(f32, 9.0), dst[0]);
    try testing.expectEqual(@as(f32, 19.0), dst[1]);
    try testing.expectEqual(@as(f32, 29.0), dst[2]);
    // Row 1: [40-2, 50-2, 60-2]
    try testing.expectEqual(@as(f32, 38.0), dst[3]);
    try testing.expectEqual(@as(f32, 48.0), dst[4]);
    try testing.expectEqual(@as(f32, 58.0), dst[5]);
}

// ============================================================
// Broadcasting: same shape (fast path)
// ============================================================

test "broadcast add: same shape uses fast path" {
    const shape = Shape.init(&.{ 2, 3 });
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };
    var dst: [6]f32 = undefined;

    try ew.broadcastAdd(&dst, &a, shape, &b, shape);
    for (0..6) |i| {
        try testing.expectEqual(a[i] + b[i], dst[i]);
    }
}

// ============================================================
// Broadcasting: incompatible shapes
// ============================================================

test "broadcast add: incompatible shapes returns error" {
    const a_shape = Shape.init(&.{ 2, 3 });
    const b_shape = Shape.init(&.{ 2, 4 });
    var dst: [12]f32 = undefined;
    const result = ew.broadcastAdd(&dst, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, a_shape, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }, b_shape);
    try testing.expectError(error.IncompatibleShapes, result);
}

// ============================================================
// Broadcasting: correctness with all ops
// ============================================================

test "broadcast ops: all four ops with [1,4]+[3,4]" {
    const a_shape = Shape.init(&.{ 1, 4 });
    const b_shape = Shape.init(&.{ 3, 4 });
    const a = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    var dst_add: [12]f32 = undefined;
    var dst_sub: [12]f32 = undefined;
    var dst_mul: [12]f32 = undefined;
    var dst_div: [12]f32 = undefined;

    try ew.broadcastAdd(&dst_add, &a, a_shape, &b, b_shape);
    try ew.broadcastSub(&dst_sub, &a, a_shape, &b, b_shape);
    try ew.broadcastMul(&dst_mul, &a, a_shape, &b, b_shape);
    try ew.broadcastDiv(&dst_div, &a, a_shape, &b, b_shape);

    for (0..12) |i| {
        const a_val = a[i % 4];
        const b_val = b[i];
        try expectClose(dst_add[i], a_val + b_val, 1e-6);
        try expectClose(dst_sub[i], a_val - b_val, 1e-6);
        try expectClose(dst_mul[i], a_val * b_val, 1e-6);
        try expectClose(dst_div[i], a_val / b_val, 1e-6);
    }
}

// ============================================================
// Benchmark: fused addRelu vs separate add+relu
// ============================================================

test "bench: fused addRelu vs separate add+relu on 1M elements" {
    const N = 4_000_000; // 16MB per array — exceeds L2 cache, memory-bound
    const WARMUP = 3;
    const ITERS = 20;
    const alloc = std.heap.page_allocator;

    const a = try alloc.alloc(f32, N);
    defer alloc.free(a);
    const b_data = try alloc.alloc(f32, N);
    defer alloc.free(b_data);
    const dst = try alloc.alloc(f32, N);
    defer alloc.free(dst);
    const tmp = try alloc.alloc(f32, N);
    defer alloc.free(tmp);

    for (0..N) |i| {
        a[i] = @as(f32, @floatFromInt(i % 997)) * 0.02 - 10.0;
        b_data[i] = @as(f32, @floatFromInt((i * 7 + 13) % 991)) * 0.02 - 10.0;
    }

    // Warmup
    for (0..WARMUP) |_| {
        ew.addRelu(dst, a, b_data);
        ew.add(tmp, a, b_data);
        ew.relu(dst, tmp);
    }

    // Bench fused (best of ITERS)
    var best_fused: u64 = std.math.maxInt(u64);
    for (0..ITERS) |_| {
        var timer = try std.time.Timer.start();
        ew.addRelu(dst, a, b_data);
        const elapsed = timer.read();
        if (elapsed < best_fused) best_fused = elapsed;
    }

    // Bench separate (best of ITERS)
    var best_separate: u64 = std.math.maxInt(u64);
    for (0..ITERS) |_| {
        var timer = try std.time.Timer.start();
        ew.add(tmp, a, b_data);
        asm volatile ("" ::: .{ .memory = true });
        ew.relu(dst, tmp);
        const elapsed = timer.read();
        if (elapsed < best_separate) best_separate = elapsed;
    }

    const fused_us = @as(f64, @floatFromInt(best_fused)) / 1000.0;
    const separate_us = @as(f64, @floatFromInt(best_separate)) / 1000.0;
    const speedup = separate_us / fused_us;

    std.debug.print("\nBenchmark: addRelu {d}M elements (best of {d})\n", .{ N / 1_000_000, ITERS });
    std.debug.print("  Fused:    {d:.1} us\n", .{fused_us});
    std.debug.print("  Separate: {d:.1} us\n", .{separate_us});
    std.debug.print("  Speedup:  {d:.2}x\n", .{speedup});

    // Only enforce threshold in release mode where optimizations are effective
    const builtin = @import("builtin");
    if (builtin.mode != .Debug) {
        try testing.expect(speedup >= 1.5);
    }
}
