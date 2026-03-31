//! C ABI FFI exports for the Synapse tensor library.
//! All functions use C calling convention (via `export`) and return syn_status_t error codes.
//! No panics escape across the FFI boundary: all Zig errors are mapped to integer status codes.
//!
//! Module design: this file uses only file-path imports (no named modules) so that
//! every source file belongs to exactly one module. Element-wise ops and SIMD vec ops
//! are implemented inline using portable @Vector to avoid importing elementwise.zig
//! and vec_ops.zig which depend on named modules.

const std = @import("std");
const builtin = @import("builtin");
const synapse = @import("synapse");
const dispatch = @import("dispatch");

// --- Internal type aliases (from synapse module) ---
const Storage = synapse.tensor.storage.Storage;
const Shape = synapse.tensor.shape.Shape;
const MAX_RANK = synapse.tensor.shape.MAX_RANK;
const TensorF32 = synapse.tensor.core.Tensor(f32);

// --- Ops (accessed through synapse module) ---
const reduce_ops = synapse.ops.reduce;
const softmax_ops = synapse.ops.softmax;
const batchnorm_ops = synapse.ops.batchnorm;
const matmul_ops = synapse.ops.matmul;
const conv_ops = synapse.ops.conv;
const pool_ops = synapse.ops.pool;
const transpose_ops = synapse.ops.transpose;
const layernorm_ops = synapse.ops.layernorm;
const attention_ops = synapse.ops.attention;
const rope_ops = synapse.ops.rope;
const rmsnorm_ops = synapse.ops.rmsnorm;
const silu_ops = synapse.ops.silu;
const quantize_ops = synapse.ops.quantize;
const qmatmul_ops = synapse.ops.qmatmul;
const kvcache_ops = synapse.ops.kvcache;
const selective_scan_ops = synapse.ops.selective_scan;
const wkv7_ops = synapse.ops.wkv7;
const projection_ops = synapse.ops.projection;

// --- Allocator modules (separate named modules, no file overlap with synapse) ---
const ArenaAllocator = @import("arena").ArenaAllocator;
const PoolAllocator = @import("pool").PoolAllocator;

// FFI pool uses 128-byte fixed slots.
const FfiPool = PoolAllocator(128);

// Use libc malloc for FFI allocations — page_allocator (mmap/munmap per call)
// is catastrophically slow for the per-call packing buffers in syn_sgemm.
const ffi_allocator = std.heap.c_allocator;

// ============================================================
// Status codes
// ============================================================

pub const SYN_OK: c_int = 0;
pub const SYN_ERR_NULL_PTR: c_int = 1;
pub const SYN_ERR_INVALID_ARG: c_int = 2;
pub const SYN_ERR_OUT_OF_MEMORY: c_int = 3;
pub const SYN_ERR_SHAPE_MISMATCH: c_int = 4;
pub const SYN_ERR_NOT_CONTIGUOUS: c_int = 5;
pub const SYN_ERR_INVALID_AXIS: c_int = 6;
pub const SYN_ERR_INVALID_DIMENSIONS: c_int = 7;
pub const SYN_ERR_INTERNAL: c_int = 8;

pub const SYN_ARCH_UNKNOWN: u32 = 0;
pub const SYN_ARCH_AARCH64: u32 = 1;
pub const SYN_ARCH_X86_64: u32 = 2;
pub const SYN_ARCH_WASM32: u32 = 3;

pub const SYN_OS_UNKNOWN: u32 = 0;
pub const SYN_OS_MACOS: u32 = 1;
pub const SYN_OS_LINUX: u32 = 2;
pub const SYN_OS_WINDOWS: u32 = 3;
pub const SYN_OS_WASM: u32 = 4;

pub const SYN_BACKEND_SCALAR: u32 = 0;
pub const SYN_BACKEND_NEON: u32 = 1;
pub const SYN_BACKEND_AVX2: u32 = 2;

pub const SYN_RUNTIME_NATIVE_PERF: u32 = 1;
pub const SYN_RUNTIME_ARM_COMPACT: u32 = 2;
pub const SYN_RUNTIME_WASM_PORTABLE: u32 = 3;

pub const SYN_SUPPORT_STABLE: u32 = 1;
pub const SYN_SUPPORT_BETA: u32 = 2;
pub const SYN_SUPPORT_EXPERIMENTAL: u32 = 3;

pub const SYN_FEATURE_SGEMM: u64 = 1 << 0;
pub const SYN_FEATURE_LAYERNORM: u64 = 1 << 1;
pub const SYN_FEATURE_RMSNORM: u64 = 1 << 2;
pub const SYN_FEATURE_FUSED_ATTENTION: u64 = 1 << 3;
pub const SYN_FEATURE_INT8_QUANT: u64 = 1 << 4;
pub const SYN_FEATURE_Q4_0_GEMV: u64 = 1 << 5;
pub const SYN_FEATURE_KV_CACHE: u64 = 1 << 6;
pub const SYN_FEATURE_GEOMETRIC_ATTENTION: u64 = 1 << 7;

pub const SYN_CAPABILITY_ABI_VERSION: u32 = 1;

pub const syn_capability_summary_t = extern struct {
    abi_version: u32,
    target_arch: u32,
    target_os: u32,
    simd_backend: u32,
    runtime_profile: u32,
    support_level: u32,
    feature_bits: u64,
};

// ============================================================
// Internal wrapper (Tensor is a value type in Zig; we heap-box it)
// ============================================================

const TensorWrapper = struct {
    tensor: TensorF32,
};

// ============================================================
// Error mapping
// ============================================================

fn mapError(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => SYN_ERR_OUT_OF_MEMORY,
        error.ShapeMismatch => SYN_ERR_SHAPE_MISMATCH,
        error.NotContiguous => SYN_ERR_NOT_CONTIGUOUS,
        error.InvalidAxis => SYN_ERR_INVALID_AXIS,
        error.InvalidDimensions => SYN_ERR_INVALID_DIMENSIONS,
        error.InvalidStride => SYN_ERR_INVALID_ARG,
        error.IncompatibleShapes => SYN_ERR_SHAPE_MISMATCH,
        error.RankTooHigh => SYN_ERR_INVALID_DIMENSIONS,
        error.InvalidNormDims => SYN_ERR_INVALID_ARG,
        error.CacheFull => SYN_ERR_INVALID_ARG,
        else => SYN_ERR_INTERNAL,
    };
}

fn targetArchValue() u32 {
    return switch (builtin.cpu.arch) {
        .aarch64 => SYN_ARCH_AARCH64,
        .x86_64 => SYN_ARCH_X86_64,
        .wasm32 => SYN_ARCH_WASM32,
        else => SYN_ARCH_UNKNOWN,
    };
}

fn targetOsValue() u32 {
    return switch (builtin.os.tag) {
        .macos => SYN_OS_MACOS,
        .linux => SYN_OS_LINUX,
        .windows => SYN_OS_WINDOWS,
        .freestanding => if (builtin.cpu.arch == .wasm32) SYN_OS_WASM else SYN_OS_UNKNOWN,
        else => SYN_OS_UNKNOWN,
    };
}

fn simdBackendValue() u32 {
    return switch (dispatch.detectBackend()) {
        .scalar => SYN_BACKEND_SCALAR,
        .neon => SYN_BACKEND_NEON,
        .avx2 => SYN_BACKEND_AVX2,
    };
}

fn runtimeProfileValue() u32 {
    if (builtin.cpu.arch == .wasm32) return SYN_RUNTIME_WASM_PORTABLE;
    if (builtin.cpu.arch == .aarch64 and builtin.os.tag != .macos) return SYN_RUNTIME_ARM_COMPACT;
    return SYN_RUNTIME_NATIVE_PERF;
}

fn supportLevelValue(runtime_profile: u32) u32 {
    return switch (runtime_profile) {
        SYN_RUNTIME_ARM_COMPACT => SYN_SUPPORT_BETA,
        SYN_RUNTIME_WASM_PORTABLE => SYN_SUPPORT_STABLE,
        else => SYN_SUPPORT_STABLE,
    };
}

fn featureBits() u64 {
    return SYN_FEATURE_SGEMM |
        SYN_FEATURE_LAYERNORM |
        SYN_FEATURE_RMSNORM |
        SYN_FEATURE_FUSED_ATTENTION |
        SYN_FEATURE_INT8_QUANT |
        SYN_FEATURE_Q4_0_GEMV |
        SYN_FEATURE_KV_CACHE |
        SYN_FEATURE_GEOMETRIC_ATTENTION;
}

// ============================================================
// Pointer conversion helpers
// ============================================================

inline fn ptrToStorage(p: ?*anyopaque) ?*Storage {
    return @ptrCast(@alignCast(p orelse return null));
}

inline fn ptrToTensor(p: ?*anyopaque) ?*TensorWrapper {
    return @ptrCast(@alignCast(p orelse return null));
}

inline fn ptrToArena(p: ?*anyopaque) ?*ArenaAllocator {
    return @ptrCast(@alignCast(p orelse return null));
}

inline fn ptrToPool(p: ?*anyopaque) ?*FfiPool {
    return @ptrCast(@alignCast(p orelse return null));
}

inline fn ptrToKvCache(p: ?*anyopaque) ?*kvcache_ops.KvCache {
    return @ptrCast(@alignCast(p orelse return null));
}

/// Wrap a returned Tensor(f32) value in a heap-allocated TensorWrapper.
/// On allocation failure the tensor is released to avoid leaking storage.
fn wrapTensor(t: TensorF32, out_ptr: *?*anyopaque) c_int {
    const tw = ffi_allocator.create(TensorWrapper) catch {
        t.release();
        return SYN_ERR_OUT_OF_MEMORY;
    };
    tw.tensor = t;
    out_ptr.* = @ptrCast(tw);
    return SYN_OK;
}

// ============================================================
// Capability reporting
// ============================================================

pub export fn syn_capability_summary(out: ?*syn_capability_summary_t) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const runtime_profile = runtimeProfileValue();
    out_ptr.* = .{
        .abi_version = SYN_CAPABILITY_ABI_VERSION,
        .target_arch = targetArchValue(),
        .target_os = targetOsValue(),
        .simd_backend = simdBackendValue(),
        .runtime_profile = runtime_profile,
        .support_level = supportLevelValue(runtime_profile),
        .feature_bits = featureBits(),
    };
    return SYN_OK;
}

pub export fn syn_runtime_capabilities_json(out_ptr: ?*?[*]u8, out_len: ?*usize) c_int {
    const out_buf_ptr = out_ptr orelse return SYN_ERR_NULL_PTR;
    const out_len_ptr = out_len orelse return SYN_ERR_NULL_PTR;

    const runtime_profile = runtimeProfileValue();
    const payload = .{
        .abi_version = SYN_CAPABILITY_ABI_VERSION,
        .target_arch = archName(targetArchValue()),
        .target_os = osName(targetOsValue()),
        .simd_backend = backendName(simdBackendValue()),
        .runtime_profile = runtimeProfileName(runtime_profile),
        .support_level = supportLevelName(supportLevelValue(runtime_profile)),
        .feature_bits = featureBits(),
        .features = [_][]const u8{
            "sgemm",
            "layernorm",
            "rmsnorm",
            "fused_attention",
            "int8_quant",
            "q4_0_gemv",
            "kvcache",
            "geometric_attention",
        },
    };

    var out: std.io.Writer.Allocating = .init(ffi_allocator);
    defer out.deinit();
    std.json.Stringify.value(payload, .{}, &out.writer) catch |err| {
        return mapError(err);
    };
    const json_bytes = out.written();

    const owned = ffi_allocator.alloc(u8, json_bytes.len + 1) catch |err| {
        return mapError(err);
    };
    @memcpy(owned[0..json_bytes.len], json_bytes);
    owned[json_bytes.len] = 0;

    out_buf_ptr.* = owned.ptr;
    out_len_ptr.* = json_bytes.len;
    return SYN_OK;
}

pub export fn syn_runtime_capabilities_free(ptr: ?[*]u8, len: usize) c_int {
    const buf = ptr orelse return SYN_ERR_NULL_PTR;
    ffi_allocator.free(buf[0 .. len + 1]);
    return SYN_OK;
}

fn archName(value: u32) []const u8 {
    return switch (value) {
        SYN_ARCH_AARCH64 => "aarch64",
        SYN_ARCH_X86_64 => "x86_64",
        SYN_ARCH_WASM32 => "wasm32",
        else => "unknown",
    };
}

fn osName(value: u32) []const u8 {
    return switch (value) {
        SYN_OS_MACOS => "macos",
        SYN_OS_LINUX => "linux",
        SYN_OS_WINDOWS => "windows",
        SYN_OS_WASM => "wasm",
        else => "unknown",
    };
}

fn backendName(value: u32) []const u8 {
    return switch (value) {
        SYN_BACKEND_NEON => "neon",
        SYN_BACKEND_AVX2 => "avx2",
        else => "scalar",
    };
}

fn runtimeProfileName(value: u32) []const u8 {
    return switch (value) {
        SYN_RUNTIME_ARM_COMPACT => "arm_compact",
        SYN_RUNTIME_WASM_PORTABLE => "wasm_portable",
        else => "native_perf",
    };
}

fn supportLevelName(value: u32) []const u8 {
    return switch (value) {
        SYN_SUPPORT_BETA => "beta",
        SYN_SUPPORT_EXPERIMENTAL => "experimental",
        else => "stable",
    };
}

// ============================================================
// Portable @Vector SIMD helpers (inline, no external deps)
// ============================================================

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

inline fn vecAdd(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = va + vb;
    }
    while (i < len) : (i += 1) dst[i] = a[i] + b[i];
}

inline fn vecSub(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = va - vb;
    }
    while (i < len) : (i += 1) dst[i] = a[i] - b[i];
}

inline fn vecMul(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = va * vb;
    }
    while (i < len) : (i += 1) dst[i] = a[i] * b[i];
}

inline fn vecDiv(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = va / vb;
    }
    while (i < len) : (i += 1) dst[i] = a[i] / b[i];
}

inline fn vecFma(dst: []f32, a: []const f32, b: []const f32, c: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        const vc: F32x4 = c[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = @mulAdd(F32x4, va, vb, vc);
    }
    while (i < len) : (i += 1) dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
}

inline fn vecRelu(dst: []f32, src: []const f32) void {
    const len = dst.len;
    const zero: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = @max(v, zero);
    }
    while (i < len) : (i += 1) dst[i] = @max(src[i], 0.0);
}

inline fn scalarExp(x: f32) f32 {
    const clamped = @max(@min(x, @as(f32, 88.0)), @as(f32, -88.0));
    const ln2: f32 = 0.6931471805599453;
    const ln2_inv: f32 = 1.4426950408889634;
    const n_float = @round(clamped * ln2_inv);
    const r = clamped - n_float * ln2;
    var p: f32 = 1.0 / 120.0;
    p = @mulAdd(f32, p, r, 1.0 / 24.0);
    p = @mulAdd(f32, p, r, 1.0 / 6.0);
    p = @mulAdd(f32, p, r, 0.5);
    p = @mulAdd(f32, p, r, 1.0);
    p = @mulAdd(f32, p, r, 1.0);
    const n_int: i32 = @intFromFloat(n_float);
    const biased: u32 = @bitCast(n_int + @as(i32, 127));
    const pow2: f32 = @bitCast(biased << 23);
    return p * pow2;
}

inline fn expVec(x: F32x4) F32x4 {
    const ln2: F32x4 = @splat(0.6931471805599453);
    const ln2_inv: F32x4 = @splat(1.4426950408889634);
    const one: F32x4 = @splat(1.0);
    const clamped = @max(@min(x, @as(F32x4, @splat(88.0))), @as(F32x4, @splat(-88.0)));
    const n_float: F32x4 = @round(clamped * ln2_inv);
    const r: F32x4 = clamped - n_float * ln2;
    var p: F32x4 = @splat(@as(f32, 1.0 / 120.0));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 24.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 6.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 0.5))));
    p = @mulAdd(F32x4, p, r, one);
    p = @mulAdd(F32x4, p, r, one);
    const n_int: @Vector(VEC_LEN, i32) = @intFromFloat(n_float);
    const biased: @Vector(VEC_LEN, i32) = n_int + @as(@Vector(VEC_LEN, i32), @splat(@as(i32, 127)));
    const biased_u: @Vector(VEC_LEN, u32) = @bitCast(biased);
    const pow2: F32x4 = @bitCast(biased_u << @as(@Vector(VEC_LEN, u5), @splat(23)));
    return p * pow2;
}

inline fn tanhVec(x: F32x4) F32x4 {
    const one: F32x4 = @splat(1.0);
    const clamped = @max(@min(x, @as(F32x4, @splat(10.0))), @as(F32x4, @splat(-10.0)));
    const exp2x = expVec(clamped + clamped);
    return (exp2x - one) / (exp2x + one);
}

inline fn scalarTanh(x: f32) f32 {
    const clamped = @max(@min(x, @as(f32, 10.0)), @as(f32, -10.0));
    const exp2x = scalarExp(clamped + clamped);
    return (exp2x - 1.0) / (exp2x + 1.0);
}

inline fn vecSigmoid(dst: []f32, src: []const f32) void {
    const len = dst.len;
    const one: F32x4 = @splat(1.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = one / (one + expVec(-v));
    }
    while (i < len) : (i += 1) dst[i] = 1.0 / (1.0 + scalarExp(-src[i]));
}

inline fn vecTanh(dst: []f32, src: []const f32) void {
    const len = dst.len;
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = tanhVec(v);
    }
    while (i < len) : (i += 1) dst[i] = scalarTanh(src[i]);
}

inline fn vecGelu(dst: []f32, src: []const f32) void {
    const len = dst.len;
    const half: F32x4 = @splat(0.5);
    const one: F32x4 = @splat(1.0);
    const sqrt_2_over_pi: F32x4 = @splat(0.7978845608028654);
    const coeff: F32x4 = @splat(0.044715);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const x: F32x4 = src[i..][0..VEC_LEN].*;
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        (dst.ptr + i)[0..VEC_LEN].* = half * x * (one + tanhVec(inner));
    }
    while (i < len) : (i += 1) {
        const x = src[i];
        const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        dst[i] = 0.5 * x * (1.0 + scalarTanh(inner));
    }
}

inline fn vecHsum(src: []const f32) f32 {
    const len = src.len;
    var acc: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        acc += v;
    }
    var s: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) s += src[i];
    return s;
}

inline fn vecHmax(src: []const f32) f32 {
    const len = src.len;
    var acc: F32x4 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        acc = @max(acc, v);
    }
    var m: f32 = @reduce(.Max, acc);
    while (i < len) : (i += 1) m = @max(m, src[i]);
    return m;
}

// ============================================================
// Storage FFI
// ============================================================

pub export fn syn_storage_create(count: usize, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const storage = Storage.create(ffi_allocator, f32, count) catch return SYN_ERR_OUT_OF_MEMORY;
    out_ptr.* = @ptrCast(storage);
    return SYN_OK;
}

pub export fn syn_storage_retain(s: ?*anyopaque) c_int {
    const storage = ptrToStorage(s) orelse return SYN_ERR_NULL_PTR;
    _ = storage.retain();
    return SYN_OK;
}

pub export fn syn_storage_release(s: ?*anyopaque) c_int {
    const storage = ptrToStorage(s) orelse return SYN_ERR_NULL_PTR;
    storage.release();
    return SYN_OK;
}

pub export fn syn_storage_data(s: ?*anyopaque, out: ?*[*]f32) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const storage = ptrToStorage(s) orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = storage.dataAs(f32).ptr;
    return SYN_OK;
}

pub export fn syn_storage_len(s: ?*anyopaque, out: ?*usize) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const storage = ptrToStorage(s) orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = storage.byteLen() / @sizeOf(f32);
    return SYN_OK;
}

// ============================================================
// Tensor FFI
// ============================================================

pub export fn syn_tensor_create(
    storage_ptr: ?*anyopaque,
    dims_ptr: ?[*]const usize,
    ndim: usize,
    out: ?*?*anyopaque,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const storage = ptrToStorage(storage_ptr) orelse return SYN_ERR_NULL_PTR;
    const dims = dims_ptr orelse return SYN_ERR_NULL_PTR;
    if (ndim > MAX_RANK) return SYN_ERR_INVALID_DIMENSIONS;

    const shape = Shape.init(dims[0..ndim]);
    if (shape.numel() * @sizeOf(f32) > storage.byteLen()) return SYN_ERR_INVALID_ARG;

    const tw = ffi_allocator.create(TensorWrapper) catch return SYN_ERR_OUT_OF_MEMORY;
    tw.tensor = TensorF32.init(storage, shape);
    out_ptr.* = @ptrCast(tw);
    return SYN_OK;
}

pub export fn syn_tensor_destroy(t: ?*anyopaque) c_int {
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    tw.tensor.release();
    ffi_allocator.destroy(tw);
    return SYN_OK;
}

pub export fn syn_tensor_shape(
    t: ?*anyopaque,
    out_dims: ?[*]usize,
    out_ndim: ?*usize,
) c_int {
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    if (out_ndim) |p| p.* = tw.tensor.shape.ndim;
    if (out_dims) |d| {
        for (0..tw.tensor.shape.ndim) |i| d[i] = tw.tensor.shape.dims[i];
    }
    return SYN_OK;
}

pub export fn syn_tensor_ndim(t: ?*anyopaque, out: ?*usize) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = tw.tensor.shape.ndim;
    return SYN_OK;
}

pub export fn syn_tensor_data_ptr(t: ?*anyopaque, out: ?*[*]f32) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = tw.tensor.dataPtr();
    return SYN_OK;
}

pub export fn syn_tensor_is_contiguous(t: ?*anyopaque, out: ?*c_int) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = if (tw.tensor.isContiguous()) 1 else 0;
    return SYN_OK;
}

pub export fn syn_tensor_contiguous(t: ?*anyopaque, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;

    if (tw.tensor.isContiguous()) {
        const new_tw = ffi_allocator.create(TensorWrapper) catch return SYN_ERR_OUT_OF_MEMORY;
        new_tw.tensor = TensorF32.init(tw.tensor.storage, tw.tensor.shape);
        out_ptr.* = @ptrCast(new_tw);
        return SYN_OK;
    }

    const numel = tw.tensor.numel();
    const new_storage = Storage.create(ffi_allocator, f32, numel) catch return SYN_ERR_OUT_OF_MEMORY;
    const dst = new_storage.dataAs(f32);
    const src = tw.tensor.storage.dataAs(f32);
    const ndim = tw.tensor.shape.ndim;
    var indices = [_]usize{0} ** MAX_RANK;

    for (0..numel) |i| {
        var off: usize = tw.tensor.offset;
        for (0..ndim) |d| off += indices[d] * tw.tensor.strides[d];
        dst[i] = src[off];
        var d: usize = ndim;
        while (d > 0) {
            d -= 1;
            indices[d] += 1;
            if (indices[d] < tw.tensor.shape.dims[d]) break;
            indices[d] = 0;
        }
    }

    const new_tw = ffi_allocator.create(TensorWrapper) catch {
        new_storage.release();
        return SYN_ERR_OUT_OF_MEMORY;
    };
    new_tw.tensor = TensorF32.init(new_storage, tw.tensor.shape);
    new_storage.release();
    out_ptr.* = @ptrCast(new_tw);
    return SYN_OK;
}

// ============================================================
// Arena allocator FFI
// ============================================================

pub export fn syn_arena_create(region_capacity: usize, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (region_capacity == 0) return SYN_ERR_INVALID_ARG;
    const arena = ffi_allocator.create(ArenaAllocator) catch return SYN_ERR_OUT_OF_MEMORY;
    arena.* = ArenaAllocator.init(ffi_allocator, region_capacity);
    out_ptr.* = @ptrCast(arena);
    return SYN_OK;
}

pub export fn syn_arena_reset(a: ?*anyopaque) c_int {
    const arena = ptrToArena(a) orelse return SYN_ERR_NULL_PTR;
    arena.reset();
    return SYN_OK;
}

pub export fn syn_arena_destroy(a: ?*anyopaque) c_int {
    const arena = ptrToArena(a) orelse return SYN_ERR_NULL_PTR;
    arena.deinit();
    ffi_allocator.destroy(arena);
    return SYN_OK;
}

// ============================================================
// Pool allocator FFI (128-byte fixed slots)
// ============================================================

pub export fn syn_pool_create(count: usize, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (count == 0) return SYN_ERR_INVALID_ARG;
    const pool = ffi_allocator.create(FfiPool) catch return SYN_ERR_OUT_OF_MEMORY;
    pool.* = FfiPool.init(ffi_allocator, count) catch {
        ffi_allocator.destroy(pool);
        return SYN_ERR_OUT_OF_MEMORY;
    };
    out_ptr.* = @ptrCast(pool);
    return SYN_OK;
}

pub export fn syn_pool_destroy(p: ?*anyopaque) c_int {
    const pool = ptrToPool(p) orelse return SYN_ERR_NULL_PTR;
    pool.deinit();
    ffi_allocator.destroy(pool);
    return SYN_OK;
}

// ============================================================
// SGEMM FFI
// ============================================================

pub export fn syn_sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: ?[*]const f32,
    lda: usize,
    trans_a: c_int,
    b: ?[*]const f32,
    ldb: usize,
    trans_b: c_int,
    c: ?[*]f32,
    ldc: usize,
) c_int {
    const a_ptr = a orelse return SYN_ERR_NULL_PTR;
    const b_ptr = b orelse return SYN_ERR_NULL_PTR;
    const c_ptr = c orelse return SYN_ERR_NULL_PTR;
    if (m == 0 or n == 0 or k == 0) return SYN_OK;

    // M=1 fast path: GEMV uses no packing buffers
    if (m == 1 and trans_a == 0) {
        var dummy: [1]f32 = .{0};
        matmul_ops.sgemmTiled(m, n, k, a_ptr, lda, false, b_ptr, ldb, trans_b != 0, c_ptr, ldc, &dummy, &dummy);
        return SYN_OK;
    }

    const eff_kc = @min(matmul_ops.KC, k);
    const eff_mc = ((@min(matmul_ops.MC, m) + matmul_ops.MR - 1) / matmul_ops.MR) * matmul_ops.MR;
    const eff_nc = ((@min(matmul_ops.NC, n) + matmul_ops.NR - 1) / matmul_ops.NR) * matmul_ops.NR;

    const pa = ffi_allocator.alloc(f32, eff_mc * eff_kc) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pa);
    const pb = ffi_allocator.alloc(f32, eff_nc * eff_kc) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pb);

    matmul_ops.sgemmTiled(m, n, k, a_ptr, lda, trans_a != 0, b_ptr, ldb, trans_b != 0, c_ptr, ldc, pa.ptr, pb.ptr);
    return SYN_OK;
}

// ============================================================
// Element-wise operations (flat f32 arrays, portable @Vector SIMD)
// ============================================================

pub export fn syn_add(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecAdd(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

pub export fn syn_sub(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecSub(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

pub export fn syn_mul(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecMul(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

pub export fn syn_div(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecDiv(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

// ============================================================
// Activation functions (flat f32 arrays, portable @Vector SIMD)
// ============================================================

pub export fn syn_relu(dst: ?[*]f32, src: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const s = src orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecRelu(d[0..len], s[0..len]);
    return SYN_OK;
}

pub export fn syn_sigmoid(dst: ?[*]f32, src: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const s = src orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecSigmoid(d[0..len], s[0..len]);
    return SYN_OK;
}

pub export fn syn_tanh_act(dst: ?[*]f32, src: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const s = src orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecTanh(d[0..len], s[0..len]);
    return SYN_OK;
}

pub export fn syn_gelu(dst: ?[*]f32, src: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const s = src orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecGelu(d[0..len], s[0..len]);
    return SYN_OK;
}

// ============================================================
// Tensor reductions
// ============================================================

pub export fn syn_reduce_sum(t: ?*anyopaque, axis: usize, keepdim: c_int, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    const result = reduce_ops.reduceSum(ffi_allocator, tw.tensor, axis, keepdim != 0) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

pub export fn syn_reduce_max(t: ?*anyopaque, axis: usize, keepdim: c_int, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    const result = reduce_ops.reduceMax(ffi_allocator, tw.tensor, axis, keepdim != 0) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

pub export fn syn_reduce_mean(t: ?*anyopaque, axis: usize, keepdim: c_int, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(t) orelse return SYN_ERR_NULL_PTR;
    const result = reduce_ops.reduceMean(ffi_allocator, tw.tensor, axis, keepdim != 0) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Softmax
// ============================================================

pub export fn syn_softmax(input: ?*anyopaque, axis: usize, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const result = softmax_ops.softmax(ffi_allocator, tw.tensor, axis) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Batch normalization (inference mode, default gamma=1 beta=0)
// ============================================================

pub export fn syn_batchnorm(input: ?*anyopaque, num_features: usize, eps: f32, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    if (tw.tensor.shape.ndim != 2) return SYN_ERR_INVALID_DIMENSIONS;
    if (tw.tensor.shape.dims[1] != num_features) return SYN_ERR_INVALID_ARG;

    var bn = batchnorm_ops.BatchNorm.init(ffi_allocator, num_features, eps, 0.1) catch return SYN_ERR_OUT_OF_MEMORY;
    defer bn.deinit();
    const result = bn.forward(ffi_allocator, tw.tensor, false) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Conv2d (NCHW layout)
// ============================================================

pub export fn syn_conv2d(
    input: ?*anyopaque,
    kernel: ?*anyopaque,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out: ?*?*anyopaque,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const k_tw = ptrToTensor(kernel) orelse return SYN_ERR_NULL_PTR;
    const result = conv_ops.conv2d(ffi_allocator, in_tw.tensor, k_tw.tensor, stride_h, stride_w, pad_h, pad_w) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Pooling (NCHW layout)
// ============================================================

pub export fn syn_maxpool2d(
    input: ?*anyopaque,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    out: ?*?*anyopaque,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const result = pool_ops.maxPool2d(ffi_allocator, in_tw.tensor, kernel_h, kernel_w, stride_h, stride_w) catch |err| return mapError(err);
    result.allocator.free(result.argmax);
    return wrapTensor(result.output, out_ptr);
}

pub export fn syn_avgpool2d(
    input: ?*anyopaque,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    out: ?*?*anyopaque,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const result = pool_ops.avgPool2d(ffi_allocator, in_tw.tensor, kernel_h, kernel_w, stride_h, stride_w) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Transpose (2D only)
// ============================================================

pub export fn syn_transpose(input: ?*anyopaque, out: ?*?*anyopaque) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const result = transpose_ops.transpose2d(ffi_allocator, in_tw.tensor) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Raw SIMD vector operations (portable @Vector, auto-dispatched)
// ============================================================

pub export fn syn_vadd(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecAdd(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

pub export fn syn_vmul(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecMul(d[0..len], ap[0..len], bp[0..len]);
    return SYN_OK;
}

pub export fn syn_vfma(dst: ?[*]f32, a: ?[*]const f32, b: ?[*]const f32, c: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    const cp = c orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    vecFma(d[0..len], ap[0..len], bp[0..len], cp[0..len]);
    return SYN_OK;
}

pub export fn syn_vreduce_sum(src: ?[*]const f32, len: usize, out: ?*f32) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (len == 0) {
        out_ptr.* = 0.0;
        return SYN_OK;
    }
    const s = src orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = vecHsum(s[0..len]);
    return SYN_OK;
}

pub export fn syn_vreduce_max(src: ?[*]const f32, len: usize, out: ?*f32) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (len == 0) {
        out_ptr.* = -std.math.inf(f32);
        return SYN_OK;
    }
    const s = src orelse return SYN_ERR_NULL_PTR;
    out_ptr.* = vecHmax(s[0..len]);
    return SYN_OK;
}

// ============================================================
// Layer normalization (trailing dims, gamma/beta affine)
// ============================================================

pub export fn syn_layernorm_forward(
    out: ?*?*anyopaque,
    input: ?*anyopaque,
    gamma: ?*anyopaque,
    beta: ?*anyopaque,
    normalized_dim: usize,
    eps: f32,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const gamma_tw = ptrToTensor(gamma) orelse return SYN_ERR_NULL_PTR;
    const beta_tw = ptrToTensor(beta) orelse return SYN_ERR_NULL_PTR;

    if (!in_tw.tensor.isContiguous()) return SYN_ERR_NOT_CONTIGUOUS;

    const ndim = in_tw.tensor.shape.ndim;
    if (normalized_dim == 0 or normalized_dim > ndim) return SYN_ERR_INVALID_ARG;

    // Compute norm_size and validate gamma/beta lengths
    var norm_size: usize = 1;
    for ((ndim - normalized_dim)..ndim) |d| norm_size *= in_tw.tensor.shape.dims[d];
    if (gamma_tw.tensor.numel() != norm_size or beta_tw.tensor.numel() != norm_size)
        return SYN_ERR_SHAPE_MISMATCH;

    const gamma_data = gamma_tw.tensor.storage.dataAs(f32);
    const beta_data = beta_tw.tensor.storage.dataAs(f32);
    const g_off = gamma_tw.tensor.offset;
    const b_off = beta_tw.tensor.offset;

    const result = layernorm_ops.layerNorm(
        ffi_allocator,
        in_tw.tensor,
        normalized_dim,
        gamma_data[g_off .. g_off + norm_size],
        beta_data[b_off .. b_off + norm_size],
        eps,
    ) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// Scaled dot-product attention
// ============================================================

pub export fn syn_scaled_dot_product_attention(
    out: ?*?*anyopaque,
    attn_weights_out: ?*?*anyopaque,
    query: ?*anyopaque,
    key: ?*anyopaque,
    value: ?*anyopaque,
    scale: f32,
    causal: c_int,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const q_tw = ptrToTensor(query) orelse return SYN_ERR_NULL_PTR;
    const k_tw = ptrToTensor(key) orelse return SYN_ERR_NULL_PTR;
    const v_tw = ptrToTensor(value) orelse return SYN_ERR_NULL_PTR;
    _ = scale; // Scale is derived internally from d_head

    const config = attention_ops.AttentionConfig{
        .causal = causal != 0,
        .return_weights = attn_weights_out != null,
    };

    const attn_result = attention_ops.attention(
        ffi_allocator,
        q_tw.tensor,
        k_tw.tensor,
        v_tw.tensor,
        config,
    ) catch |err| return mapError(err);

    // Wrap output tensor
    const out_status = wrapTensor(attn_result.output, out_ptr);
    if (out_status != SYN_OK) {
        if (attn_result.weights) |w| w.release();
        return out_status;
    }

    // Wrap weights if requested
    if (attn_weights_out) |w_ptr| {
        if (attn_result.weights) |w| {
            const w_tw = ffi_allocator.create(TensorWrapper) catch {
                w.release();
                // Undo output wrap
                const o_tw: *TensorWrapper = @ptrCast(@alignCast(out_ptr.*));
                o_tw.tensor.release();
                ffi_allocator.destroy(o_tw);
                out_ptr.* = null;
                return SYN_ERR_OUT_OF_MEMORY;
            };
            w_tw.tensor = w;
            w_ptr.* = @ptrCast(w_tw);
        } else {
            w_ptr.* = null;
        }
    } else {
        if (attn_result.weights) |w| w.release();
    }

    return SYN_OK;
}

// ============================================================
// Rotary positional embedding (RoPE)
// ============================================================

pub export fn syn_rope_forward(
    out: ?*?*anyopaque,
    input: ?*anyopaque,
    cos_table: ?*anyopaque,
    sin_table: ?*anyopaque,
    offset: usize,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const cos_tw = ptrToTensor(cos_table) orelse return SYN_ERR_NULL_PTR;
    const sin_tw = ptrToTensor(sin_table) orelse return SYN_ERR_NULL_PTR;

    // Input must be 4D: [batch, heads, seq, d_head]
    if (in_tw.tensor.shape.ndim != 4) return SYN_ERR_INVALID_DIMENSIONS;
    if (!in_tw.tensor.isContiguous()) return SYN_ERR_NOT_CONTIGUOUS;

    const batch = in_tw.tensor.shape.dims[0];
    const heads = in_tw.tensor.shape.dims[1];
    const seq_len = in_tw.tensor.shape.dims[2];
    const d_head = in_tw.tensor.shape.dims[3];

    if (d_head < 2 or d_head % 2 != 0) return SYN_ERR_INVALID_ARG;
    const half_d = d_head / 2;

    // Validate cos/sin tables cover the needed positions
    const cos_numel = cos_tw.tensor.numel();
    const sin_numel = sin_tw.tensor.numel();
    const needed = (seq_len + offset) * half_d;
    if (cos_numel < needed or sin_numel < needed) return SYN_ERR_SHAPE_MISMATCH;

    // Allocate output
    const numel = in_tw.tensor.numel();
    const out_storage = Storage.create(ffi_allocator, f32, numel) catch return SYN_ERR_OUT_OF_MEMORY;
    const out_tensor = TensorF32.init(out_storage, in_tw.tensor.shape);
    out_storage.release();

    const in_data = in_tw.tensor.storage.dataAs(f32);
    const out_data = out_tensor.storage.dataAs(f32);
    const cos_data = cos_tw.tensor.storage.dataAs(f32);
    const sin_data = sin_tw.tensor.storage.dataAs(f32);

    rope_ops.ropeSimd(
        out_data[out_tensor.offset .. out_tensor.offset + numel],
        in_data[in_tw.tensor.offset .. in_tw.tensor.offset + numel],
        cos_data[cos_tw.tensor.offset .. cos_tw.tensor.offset + cos_numel],
        sin_data[sin_tw.tensor.offset .. sin_tw.tensor.offset + sin_numel],
        batch,
        heads,
        seq_len,
        d_head,
        half_d,
        offset,
    );

    return wrapTensor(out_tensor, out_ptr);
}

// ============================================================
// Causal attention mask
// ============================================================

pub export fn syn_causal_mask(out: ?*?*anyopaque, seq_len: usize) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (seq_len == 0) return SYN_ERR_INVALID_ARG;

    const numel = seq_len * seq_len;
    const storage = Storage.create(ffi_allocator, f32, numel) catch return SYN_ERR_OUT_OF_MEMORY;
    const mask = TensorF32.init(storage, Shape.init(&[_]usize{ seq_len, seq_len }));
    storage.release();

    const data = mask.storage.dataAs(f32);
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            data[i * seq_len + j] = if (j <= i) 0.0 else -std.math.inf(f32);
        }
    }

    return wrapTensor(mask, out_ptr);
}

// ============================================================
// RMS normalization (trailing dims, gamma affine)
// ============================================================

pub export fn syn_rmsnorm_forward(
    out: ?*?*anyopaque,
    input: ?*anyopaque,
    gamma: ?*anyopaque,
    num_norm_dims: usize,
    eps: f32,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const in_tw = ptrToTensor(input) orelse return SYN_ERR_NULL_PTR;
    const gamma_tw = ptrToTensor(gamma) orelse return SYN_ERR_NULL_PTR;

    if (!in_tw.tensor.isContiguous()) return SYN_ERR_NOT_CONTIGUOUS;

    const ndim = in_tw.tensor.shape.ndim;
    if (num_norm_dims == 0 or num_norm_dims > ndim) return SYN_ERR_INVALID_ARG;

    // Compute norm_size and validate gamma length
    var norm_size: usize = 1;
    for ((ndim - num_norm_dims)..ndim) |d| norm_size *= in_tw.tensor.shape.dims[d];
    if (gamma_tw.tensor.numel() != norm_size) return SYN_ERR_SHAPE_MISMATCH;

    const gamma_data = gamma_tw.tensor.storage.dataAs(f32);
    const g_off = gamma_tw.tensor.offset;

    const result = rmsnorm_ops.rmsNorm(
        ffi_allocator,
        in_tw.tensor,
        num_norm_dims,
        gamma_data[g_off .. g_off + norm_size],
        eps,
    ) catch |err| return mapError(err);
    return wrapTensor(result, out_ptr);
}

// ============================================================
// SiLU activation (flat f32 arrays)
// ============================================================

pub export fn syn_silu(dst: ?[*]f32, src: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const s = src orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    silu_ops.silu(d[0..len], s[0..len]);
    return SYN_OK;
}

// ============================================================
// Fused SwiGLU: dst[i] = silu(gate[i]) * up[i] (flat f32 arrays)
// ============================================================

pub export fn syn_swiglu(dst: ?[*]f32, gate: ?[*]const f32, up: ?[*]const f32, len: usize) c_int {
    const d = dst orelse return SYN_ERR_NULL_PTR;
    const g = gate orelse return SYN_ERR_NULL_PTR;
    const u = up orelse return SYN_ERR_NULL_PTR;
    if (len == 0) return SYN_OK;
    silu_ops.swigluFused(d[0..len], g[0..len], u[0..len]);
    return SYN_OK;
}

// ============================================================
// Per-channel INT8 quantization
// ============================================================

pub export fn syn_quantize_per_channel_int8(
    data: ?[*]const f32,
    channels: usize,
    channel_size: usize,
    out: ?[*]i8,
    scales: ?[*]f32,
) c_int {
    const d = data orelse return SYN_ERR_NULL_PTR;
    const o = out orelse return SYN_ERR_NULL_PTR;
    const s = scales orelse return SYN_ERR_NULL_PTR;
    if (channels == 0 or channel_size == 0) return SYN_ERR_INVALID_ARG;
    quantize_ops.quantizePerChannelInt8(d, channels, channel_size, o, s);
    return SYN_OK;
}

// ============================================================
// Per-channel INT8 dequantization
// ============================================================

pub export fn syn_dequantize_per_channel_int8(
    data: ?[*]const i8,
    channels: usize,
    channel_size: usize,
    out: ?[*]f32,
    scales: ?[*]const f32,
) c_int {
    const d = data orelse return SYN_ERR_NULL_PTR;
    const o = out orelse return SYN_ERR_NULL_PTR;
    const s = scales orelse return SYN_ERR_NULL_PTR;
    if (channels == 0 or channel_size == 0) return SYN_ERR_INVALID_ARG;
    quantize_ops.dequantizePerChannelInt8(d, channels, channel_size, o, s);
    return SYN_OK;
}

// ============================================================
// INT8 quantized GEMM with per-channel scaling
// ============================================================

pub export fn syn_qgemm_int8(
    m: usize,
    n: usize,
    k: usize,
    a: ?[*]const i8,
    lda: usize,
    b: ?[*]const i8,
    ldb: usize,
    c: ?[*]f32,
    ldc: usize,
    scales_a: ?[*]const f32,
    scales_b: ?[*]const f32,
) c_int {
    const a_ptr = a orelse return SYN_ERR_NULL_PTR;
    const b_ptr = b orelse return SYN_ERR_NULL_PTR;
    const c_ptr = c orelse return SYN_ERR_NULL_PTR;
    const sa = scales_a orelse return SYN_ERR_NULL_PTR;
    const sb = scales_b orelse return SYN_ERR_NULL_PTR;
    if (m == 0 or n == 0 or k == 0) return SYN_OK;

    // M=1 fast path: GEMV uses no packing buffers, pass dummies
    if (m == 1) {
        var dummy: [1]i8 = .{0};
        qmatmul_ops.int8GemmTiled(m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, sa, sb, &dummy, &dummy);
        return SYN_OK;
    }

    const eff_kc = @min(qmatmul_ops.KC, k);
    const eff_mc = ((@min(qmatmul_ops.MC, m) + qmatmul_ops.MR - 1) / qmatmul_ops.MR) * qmatmul_ops.MR;
    const eff_nc = ((@min(qmatmul_ops.NC, n) + qmatmul_ops.NR - 1) / qmatmul_ops.NR) * qmatmul_ops.NR;

    const pa = ffi_allocator.alloc(i8, eff_mc * eff_kc) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pa);
    const pb = ffi_allocator.alloc(i8, eff_nc * eff_kc) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pb);

    qmatmul_ops.int8GemmTiled(m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, sa, sb, pa.ptr, pb.ptr);
    return SYN_OK;
}

// ============================================================
// Fused causal attention (flat arrays, single head)
// ============================================================

/// Fused causal attention for a single head on flat arrays.
///
/// Q: [seq_q, d_head], K: [seq_k, d_head], V: [seq_k, d_head]
/// Output: [seq_q, d_head]
///
/// Uses tiled Q dimension (TILE_Q=32) with online softmax.
/// K is transposed internally for the Q·K^T matmul.
pub export fn syn_fused_attention(
    seq_q: usize,
    seq_k: usize,
    d_head: usize,
    q: ?[*]const f32,
    k: ?[*]const f32,
    v: ?[*]const f32,
    out: ?[*]f32,
) c_int {
    const q_ptr = q orelse return SYN_ERR_NULL_PTR;
    const k_ptr = k orelse return SYN_ERR_NULL_PTR;
    const v_ptr = v orelse return SYN_ERR_NULL_PTR;
    const o_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (seq_q == 0 or seq_k == 0 or d_head == 0) return SYN_OK;

    const attn_ops = @import("synapse").ops.attention;
    const matmul_ops2 = @import("synapse").ops.matmul;
    const TILE_Q = attn_ops.TILE_Q;
    const MR2 = matmul_ops2.MR;
    const NR2 = matmul_ops2.NR;
    const KC2 = matmul_ops2.KC;

    // Allocate scratch: s_tile [TILE_Q, seq_k] + packing buffers
    const tq_al = ((TILE_Q + MR2 - 1) / MR2) * MR2;
    const sk_al = ((seq_k + NR2 - 1) / NR2) * NR2;
    const dh_al = ((d_head + NR2 - 1) / NR2) * NR2;
    const kc_max = @max(@min(KC2, d_head), @min(KC2, seq_k));

    const s_tile = ffi_allocator.alloc(f32, TILE_Q * seq_k) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(s_tile);
    const pa = ffi_allocator.alloc(f32, @max(tq_al * kc_max, 1)) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pa);
    const pb = ffi_allocator.alloc(f32, @max(@max(sk_al * @min(KC2, d_head), dh_al * @min(KC2, seq_k)), 1)) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pb);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    var tq: usize = 0;
    while (tq < seq_q) : (tq += TILE_Q) {
        const tqs = @min(TILE_Q, seq_q - tq);

        // Zero S_tile
        @memset(s_tile[0 .. tqs * seq_k], 0);

        // S_tile = Q_tile @ K^T  (K is [seq_k, d_head], transposed)
        matmul_ops2.sgemmTiled(
            tqs,
            seq_k,
            d_head,
            q_ptr + tq * d_head,
            d_head,
            false,
            k_ptr,
            d_head,
            true,
            s_tile.ptr,
            seq_k,
            pa.ptr,
            pb.ptr,
        );

        // Scale
        for (s_tile[0 .. tqs * seq_k]) |*val| val.* *= scale;

        // Causal mask
        for (0..tqs) |i| {
            const mask_start = tq + i + 1;
            if (mask_start < seq_k) {
                @memset(s_tile[i * seq_k + mask_start .. (i + 1) * seq_k], -std.math.inf(f32));
            }
        }

        // Online softmax per row
        for (0..tqs) |i| {
            const rb = i * seq_k;
            var mx: f32 = -std.math.inf(f32);
            var se: f32 = 0.0;
            for (0..seq_k) |j2| {
                const x = s_tile[rb + j2];
                if (x > mx) {
                    se = se * @exp(mx - x) + 1.0;
                    mx = x;
                } else {
                    se += @exp(x - mx);
                }
            }
            const inv = 1.0 / se;
            for (0..seq_k) |j2| {
                s_tile[rb + j2] = @exp(s_tile[rb + j2] - mx) * inv;
            }
        }

        // O_tile = S_tile @ V (V is [seq_k, d_head], not transposed)
        for (0..tqs) |i| {
            @memset((o_ptr + (tq + i) * d_head)[0..d_head], 0);
        }
        matmul_ops2.sgemmTiled(
            tqs,
            d_head,
            seq_k,
            s_tile.ptr,
            seq_k,
            false,
            v_ptr,
            d_head,
            false,
            o_ptr + tq * d_head,
            d_head,
            pa.ptr,
            pb.ptr,
        );
    }

    return SYN_OK;
}

/// Bidirectional fused attention (no causal mask). For ViT/JEPA/CLIP encoders.
pub export fn syn_fused_attention_bidi(
    seq_q: usize,
    seq_k: usize,
    d_head: usize,
    q: ?[*]const f32,
    k: ?[*]const f32,
    v: ?[*]const f32,
    out: ?[*]f32,
) c_int {
    const q_ptr = q orelse return SYN_ERR_NULL_PTR;
    const k_ptr = k orelse return SYN_ERR_NULL_PTR;
    const v_ptr = v orelse return SYN_ERR_NULL_PTR;
    const o_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (seq_q == 0 or seq_k == 0 or d_head == 0) return SYN_OK;

    const attn_ops = @import("synapse").ops.attention;
    const matmul_ops2 = @import("synapse").ops.matmul;
    const TILE_Q = attn_ops.TILE_Q;
    const MR2 = matmul_ops2.MR;
    const NR2 = matmul_ops2.NR;
    const KC2 = matmul_ops2.KC;

    const tq_al = ((TILE_Q + MR2 - 1) / MR2) * MR2;
    const sk_al = ((seq_k + NR2 - 1) / NR2) * NR2;
    const dh_al = ((d_head + NR2 - 1) / NR2) * NR2;
    const kc_max = @max(@min(KC2, d_head), @min(KC2, seq_k));

    const s_tile = ffi_allocator.alloc(f32, TILE_Q * seq_k) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(s_tile);
    const pa = ffi_allocator.alloc(f32, @max(tq_al * kc_max, 1)) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pa);
    const pb = ffi_allocator.alloc(f32, @max(@max(sk_al * @min(KC2, d_head), dh_al * @min(KC2, seq_k)), 1)) catch return SYN_ERR_OUT_OF_MEMORY;
    defer ffi_allocator.free(pb);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    var tq: usize = 0;
    while (tq < seq_q) : (tq += TILE_Q) {
        const tqs = @min(TILE_Q, seq_q - tq);
        @memset(s_tile[0 .. tqs * seq_k], 0);

        matmul_ops2.sgemmTiled(
            tqs,
            seq_k,
            d_head,
            q_ptr + tq * d_head,
            d_head,
            false,
            k_ptr,
            d_head,
            true,
            s_tile.ptr,
            seq_k,
            pa.ptr,
            pb.ptr,
        );

        for (s_tile[0 .. tqs * seq_k]) |*val| val.* *= scale;

        // NO causal mask — bidirectional: all positions attend to all

        for (0..tqs) |i| {
            const rb = i * seq_k;
            var mx: f32 = -std.math.inf(f32);
            var se: f32 = 0.0;
            for (0..seq_k) |j2| {
                const x = s_tile[rb + j2];
                if (x > mx) {
                    se = se * @exp(mx - x) + 1.0;
                    mx = x;
                } else {
                    se += @exp(x - mx);
                }
            }
            const inv = 1.0 / se;
            for (0..seq_k) |j2| {
                s_tile[rb + j2] = @exp(s_tile[rb + j2] - mx) * inv;
            }
        }

        for (0..tqs) |i| {
            @memset((o_ptr + (tq + i) * d_head)[0..d_head], 0);
        }
        matmul_ops2.sgemmTiled(
            tqs,
            d_head,
            seq_k,
            s_tile.ptr,
            seq_k,
            false,
            v_ptr,
            d_head,
            false,
            o_ptr + tq * d_head,
            d_head,
            pa.ptr,
            pb.ptr,
        );
    }

    return SYN_OK;
}

// ============================================================
// Geometric Attention (distance-biased attention for 3D point clouds)
// ============================================================

pub export fn syn_geometric_attention(
    n: usize,
    d: usize,
    pos_dim: usize,
    q: ?[*]const f32,
    k: ?[*]const f32,
    v: ?[*]const f32,
    positions: ?[*]const f32,
    out: ?[*]f32,
    sigma: f32,
) c_int {
    const q_ptr = q orelse return SYN_ERR_NULL_PTR;
    const k_ptr = k orelse return SYN_ERR_NULL_PTR;
    const v_ptr = v orelse return SYN_ERR_NULL_PTR;
    const pos_ptr = positions orelse return SYN_ERR_NULL_PTR;
    const o_ptr = out orelse return SYN_ERR_NULL_PTR;
    if (n == 0 or d == 0) return SYN_OK;

    const geo_attn = @import("synapse").ops.geometric_attention;
    geo_attn.geometricAttention(n, d, pos_dim, q_ptr, k_ptr, v_ptr, pos_ptr, o_ptr, sigma);
    return SYN_OK;
}

// Q4_0 GEMV (4-bit quantized matrix-vector multiply)
// ============================================================

pub export fn syn_q4_0_gemv(
    n: usize,
    k: usize,
    a: ?[*]const f32,
    b_q4: ?[*]const u8,
    c: ?[*]f32,
) c_int {
    const a_ptr = a orelse return SYN_ERR_NULL_PTR;
    const b_ptr = b_q4 orelse return SYN_ERR_NULL_PTR;
    const c_ptr = c orelse return SYN_ERR_NULL_PTR;
    if (n == 0 or k == 0) return SYN_OK;
    if (k % 32 != 0) return SYN_ERR_SHAPE_MISMATCH;

    qmatmul_ops.q4_0GemvRow(n, k, a_ptr, b_ptr, c_ptr);
    return SYN_OK;
}

// ============================================================
// KV-Cache management
// ============================================================

pub export fn syn_kvcache_create(
    max_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: ?*?*anyopaque,
) c_int {
    const out_ptr = out orelse return SYN_ERR_NULL_PTR;
    const cache = ffi_allocator.create(kvcache_ops.KvCache) catch return SYN_ERR_OUT_OF_MEMORY;
    cache.* = kvcache_ops.KvCache.create(ffi_allocator, max_seq, n_kv_heads, head_dim) catch |err| {
        ffi_allocator.destroy(cache);
        return mapError(err);
    };
    out_ptr.* = @ptrCast(cache);
    return SYN_OK;
}

pub export fn syn_kvcache_destroy(cache: ?*anyopaque) c_int {
    const c = ptrToKvCache(cache) orelse return SYN_ERR_NULL_PTR;
    c.destroy(ffi_allocator);
    ffi_allocator.destroy(c);
    return SYN_OK;
}

pub export fn syn_kvcache_append(
    cache: ?*anyopaque,
    k_token: ?[*]const f32,
    v_token: ?[*]const f32,
    stride: usize,
) c_int {
    const c = ptrToKvCache(cache) orelse return SYN_ERR_NULL_PTR;
    const kp = k_token orelse return SYN_ERR_NULL_PTR;
    const vp = v_token orelse return SYN_ERR_NULL_PTR;
    c.append(kp[0..stride], vp[0..stride]) catch |err| return mapError(err);
    return SYN_OK;
}

pub export fn syn_kvcache_slice(
    cache: ?*anyopaque,
    k_out: ?*[*]const f32,
    v_out: ?*[*]const f32,
    seq_len_out: ?*usize,
) c_int {
    const c = ptrToKvCache(cache) orelse return SYN_ERR_NULL_PTR;
    const s = c.slice();
    if (k_out) |p| p.* = s.k.ptr;
    if (v_out) |p| p.* = s.v.ptr;
    if (seq_len_out) |p| p.* = s.seq_len;
    return SYN_OK;
}

pub export fn syn_kvcache_reset(cache: ?*anyopaque) c_int {
    const c = ptrToKvCache(cache) orelse return SYN_ERR_NULL_PTR;
    c.reset();
    return SYN_OK;
}

pub export fn syn_kvcache_truncate(cache: ?*anyopaque, new_len: usize) c_int {
    const c = ptrToKvCache(cache) orelse return SYN_ERR_NULL_PTR;
    c.truncateTo(new_len);
    return SYN_OK;
}

// ============================================================
// Selective Scan (Mamba SSM kernel)
// ============================================================

pub export fn syn_selective_scan_step(
    x: ?[*]const f32,
    delta: ?[*]const f32,
    a_log: ?[*]const f32,
    b: ?[*]const f32,
    c: ?[*]const f32,
    d_skip: ?[*]const f32,
    state: ?[*]f32,
    y: ?[*]f32,
    d_inner: usize,
    d_state: usize,
) c_int {
    const xp = x orelse return SYN_ERR_NULL_PTR;
    const dp = delta orelse return SYN_ERR_NULL_PTR;
    const ap = a_log orelse return SYN_ERR_NULL_PTR;
    const bp = b orelse return SYN_ERR_NULL_PTR;
    const cp = c orelse return SYN_ERR_NULL_PTR;
    const ds = d_skip orelse return SYN_ERR_NULL_PTR;
    const sp = state orelse return SYN_ERR_NULL_PTR;
    const yp = y orelse return SYN_ERR_NULL_PTR;
    selective_scan_ops.selectiveScanStep(xp, dp, ap, bp, cp, ds, sp, yp, d_inner, d_state);
    return SYN_OK;
}

pub export fn syn_selective_scan_seq(
    xs: ?[*]const f32,
    deltas: ?[*]const f32,
    a_log: ?[*]const f32,
    bs: ?[*]const f32,
    cs: ?[*]const f32,
    d_skip: ?[*]const f32,
    state: ?[*]f32,
    ys: ?[*]f32,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) c_int {
    const xp = xs orelse return SYN_ERR_NULL_PTR;
    const dp = deltas orelse return SYN_ERR_NULL_PTR;
    const ap = a_log orelse return SYN_ERR_NULL_PTR;
    const bp = bs orelse return SYN_ERR_NULL_PTR;
    const cp = cs orelse return SYN_ERR_NULL_PTR;
    const ds = d_skip orelse return SYN_ERR_NULL_PTR;
    const sp = state orelse return SYN_ERR_NULL_PTR;
    const yp = ys orelse return SYN_ERR_NULL_PTR;
    selective_scan_ops.selectiveScanSeq(xp, dp, ap, bp, cp, ds, sp, yp, seq_len, d_inner, d_state);
    return SYN_OK;
}

// ============================================================
// WKV7 (RWKV-7 recurrence kernel)
// ============================================================

pub export fn syn_wkv7_step(
    r: ?[*]const f32,
    k: ?[*]const f32,
    v: ?[*]const f32,
    w: ?[*]const f32,
    a: ?[*]const f32,
    state: ?[*]f32,
    out: ?[*]f32,
    head_size: usize,
) c_int {
    const rp = r orelse return SYN_ERR_NULL_PTR;
    const kp = k orelse return SYN_ERR_NULL_PTR;
    const vp = v orelse return SYN_ERR_NULL_PTR;
    const wp = w orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const sp = state orelse return SYN_ERR_NULL_PTR;
    const op = out orelse return SYN_ERR_NULL_PTR;
    wkv7_ops.wkv7Step(rp, kp, vp, wp, ap, sp, op, head_size);
    return SYN_OK;
}

pub export fn syn_wkv7_seq(
    r: ?[*]const f32,
    k: ?[*]const f32,
    v: ?[*]const f32,
    w: ?[*]const f32,
    a: ?[*]const f32,
    state: ?[*]f32,
    out: ?[*]f32,
    seq_len: usize,
    head_size: usize,
) c_int {
    const rp = r orelse return SYN_ERR_NULL_PTR;
    const kp = k orelse return SYN_ERR_NULL_PTR;
    const vp = v orelse return SYN_ERR_NULL_PTR;
    const wp = w orelse return SYN_ERR_NULL_PTR;
    const ap = a orelse return SYN_ERR_NULL_PTR;
    const sp = state orelse return SYN_ERR_NULL_PTR;
    const op = out orelse return SYN_ERR_NULL_PTR;
    wkv7_ops.wkv7Seq(rp, kp, vp, wp, ap, sp, op, seq_len, head_size);
    return SYN_OK;
}

// ============================================================
// Projection GEMV with fused bias
// ============================================================

pub export fn syn_projection_gemv_bias(
    m: usize,
    n: usize,
    k: usize,
    input: ?[*]const f32,
    weight: ?[*]const f32,
    bias: ?[*]const f32,
    output: ?[*]f32,
) c_int {
    const inp = input orelse return SYN_ERR_NULL_PTR;
    const wp = weight orelse return SYN_ERR_NULL_PTR;
    const op = output orelse return SYN_ERR_NULL_PTR;
    if (m == 0 or n == 0 or k == 0) return SYN_OK;

    projection_ops.projectionGemvBias(m, n, k, inp, wp, bias, op);
    return SYN_OK;
}
