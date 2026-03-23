const std = @import("std");
const testing = std.testing;
const math = std.math;

const vec_ops = @import("vec_ops");
const dispatch = @import("dispatch");
const avx2 = @import("avx2");
const neon = @import("neon");
const reduce = @import("reduce");

// ============================================================
// Reference scalar implementations for comparison
// ============================================================

fn refExp(x: f32) f32 {
    return @as(f32, @floatCast(math.exp(@as(f64, x))));
}

fn refTanh(x: f32) f32 {
    const e2x = @as(f32, @floatCast(math.exp(@as(f64, 2.0 * x))));
    return (e2x - 1.0) / (e2x + 1.0);
}

fn refSigmoid(x: f32) f32 {
    return @as(f32, @floatCast(1.0 / (1.0 + math.exp(@as(f64, -@as(f64, x))))));
}

// ============================================================
// Dispatch detection tests
// ============================================================

test "dispatch: backend is valid for current architecture" {
    const backend = vec_ops.activeBackend();
    if (comptime std.Target.Cpu.Arch.isAARCH64(@import("builtin").cpu.arch)) {
        try testing.expect(backend == .neon);
    } else if (comptime @import("builtin").cpu.arch == .x86_64) {
        try testing.expect(backend == .avx2 or backend == .scalar);
    } else {
        try testing.expect(backend == .scalar);
    }
}

test "dispatch: detectBackend returns consistent results" {
    const b1 = dispatch.detectBackend();
    const b2 = dispatch.detectBackend();
    try testing.expect(b1 == b2);
}

test "dispatch: getOps returns stable pointer" {
    const ops1 = dispatch.getOps();
    const ops2 = dispatch.getOps();
    try testing.expect(ops1 == ops2);
}

// ============================================================
// vec_ops: add
// ============================================================

test "vec_ops: add basic" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var b = [_]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
    var dst: [10]f32 = undefined;

    vec_ops.add(&dst, &a, &b);

    for (0..10) |i| {
        try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-7);
    }
}

test "vec_ops: add various sizes" {
    const sizes = [_]usize{ 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100 };
    for (sizes) |size| {
        var a: [100]f32 = undefined;
        var b: [100]f32 = undefined;
        var dst: [100]f32 = undefined;

        for (0..size) |i| {
            a[i] = @as(f32, @floatFromInt(i)) * 0.1;
            b[i] = @as(f32, @floatFromInt(i)) * 0.2 + 1.0;
        }

        vec_ops.add(dst[0..size], a[0..size], b[0..size]);

        for (0..size) |i| {
            try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-7);
        }
    }
}

// ============================================================
// vec_ops: mul
// ============================================================

test "vec_ops: mul basic" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var b = [_]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
    var dst: [10]f32 = undefined;

    vec_ops.mul(&dst, &a, &b);

    for (0..10) |i| {
        try testing.expectApproxEqAbs(a[i] * b[i], dst[i], 1e-6);
    }
}

// ============================================================
// vec_ops: fma
// ============================================================

test "vec_ops: fma basic" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var c = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    var dst: [9]f32 = undefined;

    vec_ops.fma(&dst, &a, &b, &c);

    for (0..9) |i| {
        const expected = @mulAdd(f32, a[i], b[i], c[i]);
        try testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "vec_ops: fma identity" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var ones = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var zeros = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var dst: [8]f32 = undefined;

    vec_ops.fma(&dst, &a, &ones, &zeros);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(a[i], dst[i], 1e-7);
    }
}

// ============================================================
// vec_ops: exp
// ============================================================

test "vec_ops: exp known values" {
    var src = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 5.0, -5.0 };
    var dst: [9]f32 = undefined;

    vec_ops.exp(&dst, &src);

    for (0..9) |i| {
        const expected = refExp(src[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-6);
        try testing.expectApproxEqAbs(expected, dst[i], tol);
    }
}

test "vec_ops: exp edge cases" {
    var src = [_]f32{0.0};
    var dst: [1]f32 = undefined;
    vec_ops.exp(&dst, &src);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-6);
}

test "vec_ops: exp accuracy sweep" {
    var src: [64]f32 = undefined;
    var dst: [64]f32 = undefined;

    for (0..64) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.5 - 16.0;
    }

    vec_ops.exp(&dst, &src);

    for (0..64) |i| {
        const expected = refExp(src[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-7);
        try testing.expectApproxEqAbs(expected, dst[i], tol);
    }
}

// ============================================================
// vec_ops: tanh
// ============================================================

test "vec_ops: tanh known values" {
    var src = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 0.5, -0.5 };
    var dst: [9]f32 = undefined;

    vec_ops.tanh(&dst, &src);

    for (0..9) |i| {
        const expected = refTanh(src[i]);
        try testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "vec_ops: tanh symmetry" {
    var pos = [_]f32{ 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0 };
    var neg: [8]f32 = undefined;
    var dst_pos: [8]f32 = undefined;
    var dst_neg: [8]f32 = undefined;

    for (0..8) |i| {
        neg[i] = -pos[i];
    }

    vec_ops.tanh(&dst_pos, &pos);
    vec_ops.tanh(&dst_neg, &neg);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(-dst_pos[i], dst_neg[i], 1e-6);
    }
}

test "vec_ops: tanh bounds" {
    var src = [_]f32{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 50.0 };
    var dst: [8]f32 = undefined;

    vec_ops.tanh(&dst, &src);

    for (0..8) |i| {
        try testing.expect(dst[i] >= -1.0 - 1e-6);
        try testing.expect(dst[i] <= 1.0 + 1e-6);
    }
}

// ============================================================
// vec_ops: sigmoid
// ============================================================

test "vec_ops: sigmoid known values" {
    var src = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0 };
    var dst: [9]f32 = undefined;

    vec_ops.sigmoid(&dst, &src);

    try testing.expectApproxEqAbs(@as(f32, 0.5), dst[0], 1e-6);

    for (0..9) |i| {
        const expected = refSigmoid(src[i]);
        try testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "vec_ops: sigmoid symmetry" {
    var pos = [_]f32{ 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0 };
    var neg: [8]f32 = undefined;
    var dst_pos: [8]f32 = undefined;
    var dst_neg: [8]f32 = undefined;

    for (0..8) |i| {
        neg[i] = -pos[i];
    }

    vec_ops.sigmoid(&dst_pos, &pos);
    vec_ops.sigmoid(&dst_neg, &neg);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(1.0 - dst_pos[i], dst_neg[i], 1e-5);
    }
}

test "vec_ops: sigmoid bounds" {
    var src = [_]f32{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 50.0 };
    var dst: [8]f32 = undefined;

    vec_ops.sigmoid(&dst, &src);

    for (0..8) |i| {
        try testing.expect(dst[i] >= 0.0 - 1e-6);
        try testing.expect(dst[i] <= 1.0 + 1e-6);
    }
}

// ============================================================
// vec_ops: hsum
// ============================================================

test "vec_ops: hsum basic" {
    var src = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const result = vec_ops.hsum(&src);
    try testing.expectApproxEqAbs(@as(f32, 55.0), result, 1e-5);
}

test "vec_ops: hsum empty" {
    var src: [0]f32 = .{};
    const result = vec_ops.hsum(&src);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-7);
}

test "vec_ops: hsum various sizes" {
    const sizes = [_]usize{ 1, 3, 4, 7, 8, 9, 16, 17, 32, 33 };
    for (sizes) |size| {
        var src: [33]f32 = undefined;
        var expected: f32 = 0;
        for (0..size) |i| {
            src[i] = @as(f32, @floatFromInt(i + 1));
            expected += src[i];
        }
        const result = vec_ops.hsum(src[0..size]);
        try testing.expectApproxEqAbs(expected, result, 1e-4);
    }
}

// ============================================================
// vec_ops: hmax
// ============================================================

test "vec_ops: hmax basic" {
    var src = [_]f32{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0 };
    const result = vec_ops.hmax(&src);
    try testing.expectApproxEqAbs(@as(f32, 9.0), result, 1e-7);
}

test "vec_ops: hmax empty" {
    var src: [0]f32 = .{};
    const result = vec_ops.hmax(&src);
    try testing.expect(math.isNegativeInf(result));
}

test "vec_ops: hmax with negatives" {
    var src = [_]f32{ -5.0, -3.0, -10.0, -1.0, -7.0, -2.0, -8.0, -4.0, -6.0 };
    const result = vec_ops.hmax(&src);
    try testing.expectApproxEqAbs(@as(f32, -1.0), result, 1e-7);
}

test "vec_ops: hmax various sizes" {
    const sizes = [_]usize{ 1, 3, 4, 7, 8, 9, 16, 17 };
    for (sizes) |size| {
        var src: [17]f32 = undefined;
        var expected: f32 = -math.inf(f32);
        for (0..size) |i| {
            src[i] = @as(f32, @floatFromInt(i)) * 0.7 - 3.0;
            if (src[i] > expected) expected = src[i];
        }
        const result = vec_ops.hmax(src[0..size]);
        try testing.expectApproxEqAbs(expected, result, 1e-7);
    }
}

// ============================================================
// Scalar fallback correctness
// ============================================================

test "scalar fallback: add correctness" {
    const ops = dispatch.getScalarOps();
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var dst: [5]f32 = undefined;

    ops.addFn(&dst, &a, &b, 5);

    for (0..5) |i| {
        try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-7);
    }
}

test "scalar fallback: exp correctness" {
    const ops = dispatch.getScalarOps();
    var src = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0 };
    var dst: [9]f32 = undefined;

    ops.expFn(&dst, &src, 9);

    for (0..9) |i| {
        const expected = refExp(src[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-6);
        try testing.expectApproxEqAbs(expected, dst[i], tol);
    }
}

test "scalar fallback: sigmoid correctness" {
    const ops = dispatch.getScalarOps();
    var src = [_]f32{ 0.0, 1.0, -1.0, 5.0, -5.0 };
    var dst: [5]f32 = undefined;

    ops.sigmoidFn(&dst, &src, 5);

    try testing.expectApproxEqAbs(@as(f32, 0.5), dst[0], 1e-6);
    for (0..5) |i| {
        const expected = refSigmoid(src[i]);
        try testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "scalar fallback: hsum correctness" {
    const ops = dispatch.getScalarOps();
    var src = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const result = ops.hsumFn(&src, 5);
    try testing.expectApproxEqAbs(@as(f32, 15.0), result, 1e-6);
}

test "scalar fallback: hmax correctness" {
    const ops = dispatch.getScalarOps();
    var src = [_]f32{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    const result = ops.hmaxFn(&src, 5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-7);
}

// ============================================================
// AVX2 primitive tests
// ============================================================

test "avx2: add primitive" {
    const a = avx2.splat(1.0);
    const b = avx2.splat(2.0);
    const result = avx2.add(a, b);
    const arr: [8]f32 = result;
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 3.0), v, 1e-7);
}

test "avx2: mul primitive" {
    const a = avx2.splat(3.0);
    const b = avx2.splat(4.0);
    const result = avx2.mul(a, b);
    const arr: [8]f32 = result;
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 12.0), v, 1e-7);
}

test "avx2: fma primitive" {
    const a = avx2.splat(2.0);
    const b = avx2.splat(3.0);
    const c = avx2.splat(1.0);
    const result = avx2.fma(a, b, c);
    const arr: [8]f32 = result;
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 7.0), v, 1e-7);
}

test "avx2: hsum primitive" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const v = avx2.load(&data);
    try testing.expectApproxEqAbs(@as(f32, 36.0), avx2.hsum(v), 1e-5);
}

test "avx2: hmax primitive" {
    var data = [_]f32{ 1.0, 8.0, 3.0, 7.0, 2.0, 6.0, 4.0, 5.0 };
    const v = avx2.load(&data);
    try testing.expectApproxEqAbs(@as(f32, 8.0), avx2.hmax(v), 1e-7);
}

test "avx2: exp primitive" {
    const zero = avx2.splat(0.0);
    const result = avx2.expVec(zero);
    const arr: [8]f32 = result;
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
}

test "avx2: exp primitive range" {
    var data = [_]f32{ -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
    const v = avx2.load(&data);
    const result = avx2.expVec(v);
    const arr: [8]f32 = result;
    for (0..8) |i| {
        const expected = refExp(data[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-6);
        try testing.expectApproxEqAbs(expected, arr[i], tol);
    }
}

test "avx2: tanh primitive" {
    const zero = avx2.splat(0.0);
    const arr: [8]f32 = avx2.tanhVec(zero);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}

test "avx2: sigmoid primitive" {
    const zero = avx2.splat(0.0);
    const arr: [8]f32 = avx2.sigmoidVec(zero);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-6);
}

test "avx2: load and store roundtrip" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const v = avx2.load(&data);
    var out: [8]f32 = undefined;
    avx2.store(&out, v);
    for (0..8) |i| try testing.expectApproxEqAbs(data[i], out[i], 1e-7);
}

// ============================================================
// NEON primitive tests (f32x4)
// ============================================================

test "neon: add primitive" {
    const a = neon.splat(1.0);
    const b = neon.splat(2.0);
    const arr: [4]f32 = neon.add(a, b);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 3.0), v, 1e-7);
}

test "neon: mul primitive" {
    const a = neon.splat(3.0);
    const b = neon.splat(4.0);
    const arr: [4]f32 = neon.mul(a, b);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 12.0), v, 1e-7);
}

test "neon: fma primitive" {
    const a = neon.splat(2.0);
    const b = neon.splat(3.0);
    const c = neon.splat(1.0);
    const arr: [4]f32 = neon.fma(a, b, c);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 7.0), v, 1e-7);
}

test "neon: hsum primitive" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const v = neon.load(&data);
    try testing.expectApproxEqAbs(@as(f32, 10.0), neon.hsum(v), 1e-5);
}

test "neon: hmax primitive" {
    var data = [_]f32{ 1.0, 4.0, 2.0, 3.0 };
    const v = neon.load(&data);
    try testing.expectApproxEqAbs(@as(f32, 4.0), neon.hmax(v), 1e-7);
}

test "neon: exp primitive" {
    const zero = neon.splat(0.0);
    const arr: [4]f32 = neon.expVec(zero);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
}

test "neon: exp primitive range" {
    var data = [_]f32{ -3.0, -1.0, 1.0, 3.0 };
    const v = neon.load(&data);
    const arr: [4]f32 = neon.expVec(v);
    for (0..4) |i| {
        const expected = refExp(data[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-6);
        try testing.expectApproxEqAbs(expected, arr[i], tol);
    }
}

test "neon: tanh primitive" {
    const zero = neon.splat(0.0);
    const arr: [4]f32 = neon.tanhVec(zero);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}

test "neon: sigmoid primitive" {
    const zero = neon.splat(0.0);
    const arr: [4]f32 = neon.sigmoidVec(zero);
    for (arr) |v| try testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-6);
}

test "neon: load and store roundtrip" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const v = neon.load(&data);
    var out: [4]f32 = undefined;
    neon.store(&out, v);
    for (0..4) |i| try testing.expectApproxEqAbs(data[i], out[i], 1e-7);
}

// ============================================================
// NEON bulk operations: correctness vs scalar reference
// ============================================================

test "neon bulk: add correctness" {
    var a: [11]f32 = undefined;
    var b: [11]f32 = undefined;
    var dst: [11]f32 = undefined;
    for (0..11) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 1.1;
        b[i] = @as(f32, @floatFromInt(i)) * 0.7 + 0.3;
    }
    neon.bulkAdd(&dst, &a, &b, 11);
    for (0..11) |i| try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-6);
}

test "neon bulk: mul correctness" {
    var a: [7]f32 = undefined;
    var b: [7]f32 = undefined;
    var dst: [7]f32 = undefined;
    for (0..7) |i| {
        a[i] = @as(f32, @floatFromInt(i)) + 1.0;
        b[i] = 0.5;
    }
    neon.bulkMul(&dst, &a, &b, 7);
    for (0..7) |i| try testing.expectApproxEqAbs(a[i] * b[i], dst[i], 1e-6);
}

test "neon bulk: fma correctness" {
    var a: [9]f32 = undefined;
    var b: [9]f32 = undefined;
    var c: [9]f32 = undefined;
    var dst: [9]f32 = undefined;
    for (0..9) |i| {
        a[i] = @as(f32, @floatFromInt(i)) + 1.0;
        b[i] = 2.0;
        c[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }
    neon.bulkFma(&dst, &a, &b, &c, 9);
    for (0..9) |i| {
        try testing.expectApproxEqAbs(@mulAdd(f32, a[i], b[i], c[i]), dst[i], 1e-5);
    }
}

test "neon bulk: exp accuracy" {
    var src: [13]f32 = undefined;
    var dst: [13]f32 = undefined;
    for (0..13) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) - 6.0;
    }
    neon.bulkExp(&dst, &src, 13);
    for (0..13) |i| {
        const expected = refExp(src[i]);
        const tol: f32 = @max(@abs(expected) * 1e-5, 1e-6);
        try testing.expectApproxEqAbs(expected, dst[i], tol);
    }
}

test "neon bulk: tanh accuracy" {
    var src: [11]f32 = undefined;
    var dst: [11]f32 = undefined;
    for (0..11) |i| src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) - 5.0;
    neon.bulkTanh(&dst, &src, 11);
    for (0..11) |i| try testing.expectApproxEqAbs(refTanh(src[i]), dst[i], 1e-5);
}

test "neon bulk: sigmoid accuracy" {
    var src: [9]f32 = undefined;
    var dst: [9]f32 = undefined;
    for (0..9) |i| src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) - 4.0;
    neon.bulkSigmoid(&dst, &src, 9);
    for (0..9) |i| try testing.expectApproxEqAbs(refSigmoid(src[i]), dst[i], 1e-5);
}

test "neon bulk: hsum correctness" {
    var src = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    try testing.expectApproxEqAbs(@as(f32, 28.0), neon.bulkHsum(&src, 7), 1e-5);
}

test "neon bulk: hmax correctness" {
    var src = [_]f32{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    try testing.expectApproxEqAbs(@as(f32, 5.0), neon.bulkHmax(&src, 5), 1e-7);
}

// ============================================================
// Tail handling: non-multiple-of-4 lengths
// ============================================================

test "tail handling: all ops with length 0" {
    var src: [0]f32 = .{};
    var dst: [0]f32 = .{};
    vec_ops.add(&dst, &src, &src);
    vec_ops.mul(&dst, &src, &src);
    vec_ops.exp(&dst, &src);
    vec_ops.tanh(&dst, &src);
    vec_ops.sigmoid(&dst, &src);
    try testing.expectApproxEqAbs(@as(f32, 0.0), vec_ops.hsum(&src), 1e-7);
    try testing.expect(math.isNegativeInf(vec_ops.hmax(&src)));
}

test "tail handling: all ops with length 1" {
    var a = [_]f32{2.5};
    var b = [_]f32{1.5};
    var c = [_]f32{0.1};
    var dst: [1]f32 = undefined;

    vec_ops.add(&dst, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 4.0), dst[0], 1e-7);

    vec_ops.mul(&dst, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 3.75), dst[0], 1e-7);

    vec_ops.fma(&dst, &a, &b, &c);
    try testing.expectApproxEqAbs(@mulAdd(f32, 2.5, 1.5, 0.1), dst[0], 1e-6);

    vec_ops.exp(&dst, &a);
    try testing.expectApproxEqAbs(refExp(2.5), dst[0], @max(@abs(refExp(2.5)) * 1e-5, 1e-6));

    vec_ops.tanh(&dst, &a);
    try testing.expectApproxEqAbs(refTanh(2.5), dst[0], 1e-5);

    vec_ops.sigmoid(&dst, &a);
    try testing.expectApproxEqAbs(refSigmoid(2.5), dst[0], 1e-5);

    try testing.expectApproxEqAbs(@as(f32, 2.5), vec_ops.hsum(&a), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 2.5), vec_ops.hmax(&a), 1e-7);
}

test "tail handling: all ops with length 3" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    var b = [_]f32{ 0.5, 1.0, 1.5 };
    var c = [_]f32{ 0.1, 0.2, 0.3 };
    var dst: [3]f32 = undefined;

    vec_ops.add(&dst, &a, &b);
    for (0..3) |i| try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-7);

    vec_ops.mul(&dst, &a, &b);
    for (0..3) |i| try testing.expectApproxEqAbs(a[i] * b[i], dst[i], 1e-7);

    vec_ops.fma(&dst, &a, &b, &c);
    for (0..3) |i| try testing.expectApproxEqAbs(@mulAdd(f32, a[i], b[i], c[i]), dst[i], 1e-6);

    vec_ops.exp(&dst, &a);
    for (0..3) |i| {
        const expected = refExp(a[i]);
        try testing.expectApproxEqAbs(expected, dst[i], @max(@abs(expected) * 1e-5, 1e-6));
    }

    vec_ops.tanh(&dst, &a);
    for (0..3) |i| try testing.expectApproxEqAbs(refTanh(a[i]), dst[i], 1e-5);

    vec_ops.sigmoid(&dst, &a);
    for (0..3) |i| try testing.expectApproxEqAbs(refSigmoid(a[i]), dst[i], 1e-5);

    try testing.expectApproxEqAbs(@as(f32, 6.0), vec_ops.hsum(&a), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), vec_ops.hmax(&a), 1e-7);
}

test "tail handling: length 7 (4+3 tail)" {
    var a: [7]f32 = undefined;
    var b: [7]f32 = undefined;
    var dst: [7]f32 = undefined;
    for (0..7) |i| {
        a[i] = @as(f32, @floatFromInt(i)) + 1.0;
        b[i] = @as(f32, @floatFromInt(i)) * 0.5;
    }
    vec_ops.add(&dst, &a, &b);
    for (0..7) |i| try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-7);
    vec_ops.mul(&dst, &a, &b);
    for (0..7) |i| try testing.expectApproxEqAbs(a[i] * b[i], dst[i], 1e-6);
}

test "tail handling: length 1000" {
    var a: [1000]f32 = undefined;
    var b: [1000]f32 = undefined;
    var dst: [1000]f32 = undefined;
    for (0..1000) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(1000 - i)) * 0.01;
    }
    vec_ops.add(&dst, &a, &b);
    for (0..1000) |i| try testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-5);

    var expected_sum: f32 = 0;
    for (0..1000) |i| expected_sum += a[i];
    try testing.expectApproxEqAbs(expected_sum, vec_ops.hsum(&a), 0.1);

    var expected_max: f32 = a[0];
    for (1..1000) |i| if (a[i] > expected_max) {
        expected_max = a[i];
    };
    try testing.expectApproxEqAbs(expected_max, vec_ops.hmax(&a), 1e-5);
}

// ============================================================
// Special values: 0, -0, inf, -inf, NaN, subnormals
// ============================================================

test "special values: add with zeros" {
    var a = [_]f32{ 0.0, -0.0, 1.0, -1.0 };
    var b = [_]f32{ 0.0, 0.0, -0.0, -0.0 };
    var dst: [4]f32 = undefined;
    vec_ops.add(&dst, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[1], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[2], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, -1.0), dst[3], 1e-7);
}

test "special values: mul with zeros" {
    var a = [_]f32{ 0.0, -0.0, 5.0, -5.0 };
    var b = [_]f32{ 1.0, 1.0, 0.0, -0.0 };
    var dst: [4]f32 = undefined;
    vec_ops.mul(&dst, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.0), @abs(dst[1]), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.0), @abs(dst[3]), 1e-7);
}

test "special values: exp with large inputs" {
    var src = [_]f32{ 88.0, -88.0, 50.0, -50.0 };
    var dst: [4]f32 = undefined;
    vec_ops.exp(&dst, &src);
    try testing.expect(math.isFinite(dst[0]));
    try testing.expect(dst[0] > 0.0);
    try testing.expect(math.isFinite(dst[1]));
    try testing.expect(dst[1] >= 0.0);
    try testing.expect(math.isFinite(dst[2]));
    try testing.expect(math.isFinite(dst[3]));
}

test "special values: tanh saturation" {
    var src = [_]f32{ 100.0, -100.0, 50.0, -50.0 };
    var dst: [4]f32 = undefined;
    vec_ops.tanh(&dst, &src);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), dst[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), dst[3], 1e-5);
}

test "special values: sigmoid saturation" {
    var src = [_]f32{ 100.0, -100.0, 50.0, -50.0 };
    var dst: [4]f32 = undefined;
    vec_ops.sigmoid(&dst, &src);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[3], 1e-5);
}

test "special values: subnormal inputs" {
    const subnormal: f32 = @bitCast(@as(u32, 1));
    var a = [_]f32{ subnormal, subnormal, subnormal, subnormal };
    var b = [_]f32{ subnormal, subnormal, subnormal, subnormal };
    var dst: [4]f32 = undefined;

    vec_ops.add(&dst, &a, &b);
    for (0..4) |i| try testing.expect(math.isFinite(dst[i]));

    vec_ops.exp(&dst, &a);
    for (0..4) |i| try testing.expectApproxEqAbs(@as(f32, 1.0), dst[i], 1e-5);

    vec_ops.tanh(&dst, &a);
    for (0..4) |i| try testing.expectApproxEqAbs(@as(f32, 0.0), dst[i], 1e-5);

    vec_ops.sigmoid(&dst, &a);
    for (0..4) |i| try testing.expectApproxEqAbs(@as(f32, 0.5), dst[i], 1e-5);
}

test "special values: hsum with inf" {
    var src = [_]f32{ 1.0, math.inf(f32), 2.0, 3.0 };
    try testing.expect(math.isPositiveInf(vec_ops.hsum(&src)));
}

test "special values: hmax with neg inf" {
    var src = [_]f32{ -math.inf(f32), 1.0, 2.0, 3.0 };
    try testing.expectApproxEqAbs(@as(f32, 3.0), vec_ops.hmax(&src), 1e-7);
}

// ============================================================
// Transcendental accuracy: relative error <= 1e-5
// ============================================================

test "transcendental accuracy: exp within 1e-5 relative error" {
    var src: [100]f32 = undefined;
    var dst: [100]f32 = undefined;
    for (0..100) |i| src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.5 - 25.0;
    vec_ops.exp(&dst, &src);
    for (0..100) |i| {
        const expected = refExp(src[i]);
        if (@abs(expected) > 1e-30) {
            try testing.expect(@abs(dst[i] - expected) / @abs(expected) <= 1e-5);
        }
    }
}

test "transcendental accuracy: tanh within 1e-5 relative error" {
    var src: [100]f32 = undefined;
    var dst: [100]f32 = undefined;
    for (0..100) |i| src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.2 - 10.0;
    vec_ops.tanh(&dst, &src);
    for (0..100) |i| {
        const expected = refTanh(src[i]);
        if (@abs(expected) > 1e-6) {
            try testing.expect(@abs(dst[i] - expected) / @abs(expected) <= 1e-5);
        } else {
            try testing.expectApproxEqAbs(expected, dst[i], 1e-5);
        }
    }
}

test "transcendental accuracy: sigmoid within 1e-5 relative error" {
    var src: [100]f32 = undefined;
    var dst: [100]f32 = undefined;
    for (0..100) |i| src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.2 - 10.0;
    vec_ops.sigmoid(&dst, &src);
    for (0..100) |i| {
        const expected = refSigmoid(src[i]);
        if (expected > 1e-6) {
            try testing.expect(@abs(dst[i] - expected) / expected <= 1e-5);
        } else {
            try testing.expectApproxEqAbs(expected, dst[i], 1e-5);
        }
    }
}

// ============================================================
// reduce.zig: pairwise reduction tests
// ============================================================

test "reduce: pairwiseSum correctness" {
    const v: @Vector(4, f32) = .{ 1.0, 2.0, 3.0, 4.0 };
    try testing.expectApproxEqAbs(@as(f32, 10.0), reduce.pairwiseSum(v), 1e-6);
}

test "reduce: pairwiseSum with negatives" {
    const v: @Vector(4, f32) = .{ -1.0, 2.0, -3.0, 4.0 };
    try testing.expectApproxEqAbs(@as(f32, 2.0), reduce.pairwiseSum(v), 1e-6);
}

test "reduce: pairwiseMax correctness" {
    const v: @Vector(4, f32) = .{ 1.0, 4.0, 2.0, 3.0 };
    try testing.expectApproxEqAbs(@as(f32, 4.0), reduce.pairwiseMax(v), 1e-7);
}

test "reduce: pairwiseMax all negatives" {
    const v: @Vector(4, f32) = .{ -5.0, -1.0, -3.0, -2.0 };
    try testing.expectApproxEqAbs(@as(f32, -1.0), reduce.pairwiseMax(v), 1e-7);
}

test "reduce: horizontalSum various sizes" {
    const sizes = [_]usize{ 0, 1, 3, 4, 7, 8, 9, 16, 17 };
    for (sizes) |size| {
        var src: [17]f32 = undefined;
        var expected: f32 = 0;
        for (0..size) |i| {
            src[i] = @as(f32, @floatFromInt(i + 1));
            expected += src[i];
        }
        try testing.expectApproxEqAbs(expected, reduce.horizontalSum(&src, size), 1e-4);
    }
}

test "reduce: horizontalMax various sizes" {
    const sizes = [_]usize{ 1, 3, 4, 7, 8, 9, 16, 17 };
    for (sizes) |size| {
        var src: [17]f32 = undefined;
        var expected: f32 = -math.inf(f32);
        for (0..size) |i| {
            src[i] = @as(f32, @floatFromInt(i)) * 0.7 - 3.0;
            if (src[i] > expected) expected = src[i];
        }
        try testing.expectApproxEqAbs(expected, reduce.horizontalMax(&src, size), 1e-7);
    }
}

test "reduce: horizontalMax empty returns -inf" {
    var src: [1]f32 = .{0.0};
    try testing.expect(math.isNegativeInf(reduce.horizontalMax(&src, 0)));
}

test "reduce: horizontalSum 1000 elements" {
    var src: [1000]f32 = undefined;
    var expected: f32 = 0;
    for (0..1000) |i| {
        src[i] = @as(f32, @floatFromInt(i)) * 0.01;
        expected += src[i];
    }
    try testing.expectApproxEqAbs(expected, reduce.horizontalSum(&src, 1000), 0.1);
}

test "reduce: horizontalMax 1000 elements" {
    var src: [1000]f32 = undefined;
    for (0..1000) |i| src[i] = @as(f32, @floatFromInt(i)) * 0.01;
    try testing.expectApproxEqAbs(@as(f32, 9.99), reduce.horizontalMax(&src, 1000), 1e-5);
}
