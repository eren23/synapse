//! Round-trip tests for the C ABI FFI surface.
//! Verifies that every exported function works correctly when called
//! from Zig (same calling convention as C) and that invalid inputs
//! produce proper error codes rather than panics.

const std = @import("std");
const ffi = @import("ffi");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApprox = std.testing.expectApproxEqAbs;

// ============================================================
// Storage round-trip
// ============================================================

test "storage: create, data, len, retain, release" {
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(8, &storage));
    try expect(storage != null);

    // Write data through the pointer.
    var data_ptr: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data_ptr));
    for (0..8) |i| data_ptr[i] = @floatFromInt(i);

    // Verify length.
    var len: usize = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_len(storage, &len));
    try expectEqual(@as(usize, 8), len);

    // Retain + double release.
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_retain(storage));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage)); // refcount 2 -> 1
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage)); // refcount 1 -> 0 (freed)
}

test "storage: null pointer errors" {
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_storage_create(4, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_storage_retain(null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_storage_release(null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_storage_data(null, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_storage_len(null, null));
}

// ============================================================
// Tensor round-trip
// ============================================================

test "tensor: create, shape, ndim, data_ptr, contiguous, destroy" {
    // Create storage with 6 elements.
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(6, &storage));

    // Write [1,2,3,4,5,6].
    var sdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &sdata));
    for (0..6) |i| sdata[i] = @as(f32, @floatFromInt(i + 1));

    // Create 2x3 tensor.
    const dims = [_]usize{ 2, 3 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 2, &tensor));
    try expect(tensor != null);

    // Shape.
    var out_dims: [8]usize = undefined;
    var ndim: usize = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_shape(tensor, &out_dims, &ndim));
    try expectEqual(@as(usize, 2), ndim);
    try expectEqual(@as(usize, 2), out_dims[0]);
    try expectEqual(@as(usize, 3), out_dims[1]);

    // Ndim standalone.
    var ndim2: usize = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_ndim(tensor, &ndim2));
    try expectEqual(@as(usize, 2), ndim2);

    // Data pointer -- should read back the same values.
    var tdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(tensor, &tdata));
    for (0..6) |i| {
        try expectApprox(@as(f32, @floatFromInt(i + 1)), tdata[i], 1e-6);
    }

    // Contiguous check.
    var contig: c_int = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_is_contiguous(tensor, &contig));
    try expectEqual(@as(c_int, 1), contig);

    // Contiguous copy (should alias since already contiguous).
    var contig_t: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_contiguous(tensor, &contig_t));
    try expect(contig_t != null);
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(contig_t));

    // Cleanup.
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

test "tensor: null and invalid-arg errors" {
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_create(null, null, 0, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_destroy(null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_shape(null, null, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_ndim(null, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_data_ptr(null, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_is_contiguous(null, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_tensor_contiguous(null, null));

    // ndim > MAX_RANK
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(1, &storage));
    const dims = [_]usize{1} ** 9;
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_ERR_INVALID_DIMENSIONS, ffi.syn_tensor_create(storage, &dims, 9, &tensor));

    // storage too small for shape
    const big_dims = [_]usize{ 100, 100 };
    try expectEqual(ffi.SYN_ERR_INVALID_ARG, ffi.syn_tensor_create(storage, &big_dims, 2, &tensor));

    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// Arena & Pool lifecycle
// ============================================================

test "arena: create, reset, destroy" {
    var arena: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_arena_create(4096, &arena));
    try expect(arena != null);
    try expectEqual(ffi.SYN_OK, ffi.syn_arena_reset(arena));
    try expectEqual(ffi.SYN_OK, ffi.syn_arena_destroy(arena));
}

test "arena: null and invalid-arg errors" {
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_arena_create(4096, null));
    try expectEqual(ffi.SYN_ERR_INVALID_ARG, ffi.syn_arena_create(0, @as(?*?*anyopaque, @constCast(&@as(?*anyopaque, null)))));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_arena_reset(null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_arena_destroy(null));
}

test "pool: create, destroy" {
    var pool: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_pool_create(16, &pool));
    try expect(pool != null);
    try expectEqual(ffi.SYN_OK, ffi.syn_pool_destroy(pool));
}

test "pool: null and invalid-arg errors" {
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_pool_create(16, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_pool_destroy(null));
    var pool: ?*anyopaque = null;
    try expectEqual(ffi.SYN_ERR_INVALID_ARG, ffi.syn_pool_create(0, &pool));
}

// ============================================================
// Element-wise and activation round-trips
// ============================================================

test "elementwise: add, sub, mul, div" {
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var dst: [4]f32 = undefined;

    try expectEqual(ffi.SYN_OK, ffi.syn_add(&dst, &a, &b, 4));
    try expectApprox(@as(f32, 6.0), dst[0], 1e-6);
    try expectApprox(@as(f32, 10.0), dst[2], 1e-6);

    try expectEqual(ffi.SYN_OK, ffi.syn_sub(&dst, &a, &b, 4));
    try expectApprox(@as(f32, -4.0), dst[0], 1e-6);

    try expectEqual(ffi.SYN_OK, ffi.syn_mul(&dst, &a, &b, 4));
    try expectApprox(@as(f32, 5.0), dst[0], 1e-6);
    try expectApprox(@as(f32, 32.0), dst[3], 1e-6);

    try expectEqual(ffi.SYN_OK, ffi.syn_div(&dst, &a, &b, 4));
    try expectApprox(@as(f32, 0.2), dst[0], 1e-6);
}

test "elementwise: null pointer errors" {
    var dst: [4]f32 = undefined;
    var a = [_]f32{ 1, 2, 3, 4 };
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_add(null, &a, &a, 4));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_add(&dst, null, &a, 4));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_add(&dst, &a, null, 4));
    // len=0 is fine even with valid pointers
    try expectEqual(ffi.SYN_OK, ffi.syn_add(&dst, &a, &a, 0));
}

test "activations: relu, sigmoid, tanh, gelu" {
    var src = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var dst: [5]f32 = undefined;

    // ReLU
    try expectEqual(ffi.SYN_OK, ffi.syn_relu(&dst, &src, 5));
    try expectApprox(@as(f32, 0.0), dst[0], 1e-6);
    try expectApprox(@as(f32, 0.0), dst[1], 1e-6);
    try expectApprox(@as(f32, 1.0), dst[3], 1e-6);
    try expectApprox(@as(f32, 2.0), dst[4], 1e-6);

    // Sigmoid
    try expectEqual(ffi.SYN_OK, ffi.syn_sigmoid(&dst, &src, 5));
    try expectApprox(@as(f32, 0.5), dst[2], 1e-5); // sigmoid(0) = 0.5
    try expect(dst[0] < 0.2); // sigmoid(-2) ~ 0.12
    try expect(dst[4] > 0.8); // sigmoid(2)  ~ 0.88

    // Tanh
    try expectEqual(ffi.SYN_OK, ffi.syn_tanh_act(&dst, &src, 5));
    try expectApprox(@as(f32, 0.0), dst[2], 1e-6); // tanh(0) = 0

    // GELU
    try expectEqual(ffi.SYN_OK, ffi.syn_gelu(&dst, &src, 5));
    try expectApprox(@as(f32, 0.0), dst[2], 1e-5); // gelu(0) ~ 0
    try expect(dst[4] > 1.9); // gelu(2) ~ 1.95
}

// ============================================================
// SGEMM round-trip
// ============================================================

test "sgemm: 2x3 * 3x2 = 2x2" {
    // A = [[1,2,3],[4,5,6]]  (2x3, lda=3)
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // B = [[7,8],[9,10],[11,12]]  (3x2, ldb=2)
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    // C = A*B = [[58,64],[139,154]]
    var c = [_]f32{ 0, 0, 0, 0 };

    try expectEqual(ffi.SYN_OK, ffi.syn_sgemm(2, 2, 3, &a, 3, 0, &b, 2, 0, &c, 2));
    try expectApprox(@as(f32, 58.0), c[0], 1e-3);
    try expectApprox(@as(f32, 64.0), c[1], 1e-3);
    try expectApprox(@as(f32, 139.0), c[2], 1e-3);
    try expectApprox(@as(f32, 154.0), c[3], 1e-3);
}

test "sgemm: null pointer error" {
    var c = [_]f32{0} ** 4;
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_sgemm(2, 2, 2, null, 2, 0, &c, 2, 0, &c, 2));
    // m=0 is a no-op
    try expectEqual(ffi.SYN_OK, ffi.syn_sgemm(0, 2, 2, &c, 2, 0, &c, 2, 0, &c, 2));
}

// ============================================================
// Tensor reductions
// ============================================================

test "reduce: sum, max, mean on 2x3 tensor" {
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(6, &storage));
    var data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data));
    // [1,2,3; 4,5,6]
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    for (0..6) |i| data[i] = vals[i];

    const dims = [_]usize{ 2, 3 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 2, &tensor));

    // Sum along axis=1 -> [6, 15]
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_OK, ffi.syn_reduce_sum(tensor, 1, 0, &result));
        var rdata: [*]f32 = undefined;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
        try expectApprox(@as(f32, 6.0), rdata[0], 1e-5);
        try expectApprox(@as(f32, 15.0), rdata[1], 1e-5);
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    }

    // Max along axis=0 -> [4, 5, 6]
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_OK, ffi.syn_reduce_max(tensor, 0, 0, &result));
        var rdata: [*]f32 = undefined;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
        try expectApprox(@as(f32, 4.0), rdata[0], 1e-5);
        try expectApprox(@as(f32, 5.0), rdata[1], 1e-5);
        try expectApprox(@as(f32, 6.0), rdata[2], 1e-5);
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    }

    // Mean along axis=1 -> [2, 5]
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_OK, ffi.syn_reduce_mean(tensor, 1, 0, &result));
        var rdata: [*]f32 = undefined;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
        try expectApprox(@as(f32, 2.0), rdata[0], 1e-5);
        try expectApprox(@as(f32, 5.0), rdata[1], 1e-5);
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    }

    // Invalid axis -> error, not panic
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_ERR_INVALID_AXIS, ffi.syn_reduce_sum(tensor, 5, 0, &result));
    }

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// Softmax
// ============================================================

test "softmax: probabilities sum to 1" {
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(4, &storage));
    var data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data));
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;

    const dims = [_]usize{ 1, 4 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 2, &tensor));

    var result: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_softmax(tensor, 1, &result));

    var rdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
    var sum: f32 = 0;
    for (0..4) |i| {
        try expect(rdata[i] > 0.0);
        sum += rdata[i];
    }
    try expectApprox(@as(f32, 1.0), sum, 1e-5);
    // Monotonicity: larger input -> larger softmax output.
    try expect(rdata[3] > rdata[2]);
    try expect(rdata[2] > rdata[1]);

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// BatchNorm
// ============================================================

test "batchnorm: inference mode normalizes" {
    // 2x3 input, num_features=3
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(6, &storage));
    var data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data));
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    for (0..6) |i| data[i] = vals[i];

    const dims = [_]usize{ 2, 3 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 2, &tensor));

    var result: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_batchnorm(tensor, 3, 1e-5, &result));
    try expect(result != null);

    // With default running_mean=0, running_var=1, gamma=1, beta=0:
    // output = (x - 0) / sqrt(1 + eps) + 0 ~ x (nearly unchanged)
    var rdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
    try expectApprox(@as(f32, 1.0), rdata[0], 1e-3);

    // Wrong num_features -> error
    {
        var bad_result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_ERR_INVALID_ARG, ffi.syn_batchnorm(tensor, 99, 1e-5, &bad_result));
    }

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// Transpose
// ============================================================

test "transpose: 2x3 -> 3x2" {
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(6, &storage));
    var data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data));
    // [1,2,3; 4,5,6]
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    for (0..6) |i| data[i] = vals[i];

    const dims = [_]usize{ 2, 3 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 2, &tensor));

    var result: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_transpose(tensor, &result));

    // Result should be 3x2: [1,4; 2,5; 3,6]
    var ndim: usize = 0;
    var rdims: [8]usize = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_shape(result, &rdims, &ndim));
    try expectEqual(@as(usize, 2), ndim);
    try expectEqual(@as(usize, 3), rdims[0]);
    try expectEqual(@as(usize, 2), rdims[1]);

    var rdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
    try expectApprox(@as(f32, 1.0), rdata[0], 1e-6);
    try expectApprox(@as(f32, 4.0), rdata[1], 1e-6);
    try expectApprox(@as(f32, 2.0), rdata[2], 1e-6);
    try expectApprox(@as(f32, 5.0), rdata[3], 1e-6);

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// Conv2d
// ============================================================

test "conv2d: 1x1x3x3 input, 1x1x2x2 kernel, stride 1, pad 0" {
    // Input: 1 batch, 1 channel, 3x3
    var in_storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(9, &in_storage));
    var in_data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(in_storage, &in_data));
    for (0..9) |i| in_data[i] = @floatFromInt(i + 1);

    const in_dims = [_]usize{ 1, 1, 3, 3 };
    var in_tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(in_storage, &in_dims, 4, &in_tensor));

    // Kernel: 1 out-channel, 1 in-channel, 2x2 all-ones
    var k_storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(4, &k_storage));
    var k_data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(k_storage, &k_data));
    for (0..4) |i| k_data[i] = 1.0;

    const k_dims = [_]usize{ 1, 1, 2, 2 };
    var k_tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(k_storage, &k_dims, 4, &k_tensor));

    var result: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_conv2d(in_tensor, k_tensor, 1, 1, 0, 0, &result));

    // Output: 1x1x2x2
    var rdims: [8]usize = undefined;
    var ndim: usize = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_shape(result, &rdims, &ndim));
    try expectEqual(@as(usize, 4), ndim);
    try expectEqual(@as(usize, 1), rdims[0]); // batch
    try expectEqual(@as(usize, 1), rdims[1]); // channels
    try expectEqual(@as(usize, 2), rdims[2]); // h_out
    try expectEqual(@as(usize, 2), rdims[3]); // w_out

    var rdata: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
    // Top-left: 1+2+4+5 = 12
    try expectApprox(@as(f32, 12.0), rdata[0], 1e-3);
    // Top-right: 2+3+5+6 = 16
    try expectApprox(@as(f32, 16.0), rdata[1], 1e-3);

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(k_tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(in_tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(k_storage));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(in_storage));
}

// ============================================================
// Pooling
// ============================================================

test "maxpool2d and avgpool2d: 1x1x4x4" {
    var storage: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_create(16, &storage));
    var data: [*]f32 = undefined;
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_data(storage, &data));
    for (0..16) |i| data[i] = @floatFromInt(i + 1);

    const dims = [_]usize{ 1, 1, 4, 4 };
    var tensor: ?*anyopaque = null;
    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_create(storage, &dims, 4, &tensor));

    // MaxPool2d kernel=2x2 stride=2x2 -> 1x1x2x2
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_OK, ffi.syn_maxpool2d(tensor, 2, 2, 2, 2, &result));
        var rdims: [8]usize = undefined;
        var ndim: usize = 0;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_shape(result, &rdims, &ndim));
        try expectEqual(@as(usize, 2), rdims[2]); // h_out
        try expectEqual(@as(usize, 2), rdims[3]); // w_out

        var rdata: [*]f32 = undefined;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
        // max of [1,2,5,6] = 6
        try expectApprox(@as(f32, 6.0), rdata[0], 1e-5);
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    }

    // AvgPool2d kernel=2x2 stride=2x2 -> 1x1x2x2
    {
        var result: ?*anyopaque = null;
        try expectEqual(ffi.SYN_OK, ffi.syn_avgpool2d(tensor, 2, 2, 2, 2, &result));
        var rdata: [*]f32 = undefined;
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_data_ptr(result, &rdata));
        // avg of [1,2,5,6] = 3.5
        try expectApprox(@as(f32, 3.5), rdata[0], 1e-5);
        try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(result));
    }

    try expectEqual(ffi.SYN_OK, ffi.syn_tensor_destroy(tensor));
    try expectEqual(ffi.SYN_OK, ffi.syn_storage_release(storage));
}

// ============================================================
// SIMD vec ops
// ============================================================

test "vec ops: vadd, vmul, vfma, vreduce_sum, vreduce_max" {
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c_arr = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var dst: [4]f32 = undefined;

    // vadd
    try expectEqual(ffi.SYN_OK, ffi.syn_vadd(&dst, &a, &b, 4));
    try expectApprox(@as(f32, 6.0), dst[0], 1e-6);

    // vmul
    try expectEqual(ffi.SYN_OK, ffi.syn_vmul(&dst, &a, &b, 4));
    try expectApprox(@as(f32, 5.0), dst[0], 1e-6);
    try expectApprox(@as(f32, 32.0), dst[3], 1e-6);

    // vfma: dst = a*b + c
    try expectEqual(ffi.SYN_OK, ffi.syn_vfma(&dst, &a, &b, &c_arr, 4));
    try expectApprox(@as(f32, 5.1), dst[0], 1e-4);

    // vreduce_sum
    var sum: f32 = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_vreduce_sum(&a, 4, &sum));
    try expectApprox(@as(f32, 10.0), sum, 1e-5);

    // vreduce_max
    var max_val: f32 = 0;
    try expectEqual(ffi.SYN_OK, ffi.syn_vreduce_max(&a, 4, &max_val));
    try expectApprox(@as(f32, 4.0), max_val, 1e-5);
}

test "vec ops: null pointer errors" {
    var dst: [4]f32 = undefined;
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_vadd(null, &dst, &dst, 4));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_vmul(&dst, null, &dst, 4));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_vfma(&dst, &dst, &dst, null, 4));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_vreduce_sum(null, 4, null));
    try expectEqual(ffi.SYN_ERR_NULL_PTR, ffi.syn_vreduce_max(null, 4, null));
}
