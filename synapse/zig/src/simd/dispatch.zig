//! Runtime CPU feature detection and SIMD dispatch.
//! Detects the optimal SIMD backend (AVX2, NEON, or scalar) at runtime
//! and provides a function pointer table for vectorized operations.

const std = @import("std");
const builtin = @import("builtin");
const avx2 = @import("avx2");
const neon = @import("neon");

pub const Backend = enum {
    scalar,
    neon,
    avx2,
};

// Function pointer types for the dispatch table
pub const BinaryFn = *const fn ([*]f32, [*]const f32, [*]const f32, usize) void;
pub const TernaryFn = *const fn ([*]f32, [*]const f32, [*]const f32, [*]const f32, usize) void;
pub const UnaryFn = *const fn ([*]f32, [*]const f32, usize) void;
pub const ReduceFn = *const fn ([*]const f32, usize) f32;

pub const VecOps = struct {
    addFn: BinaryFn,
    mulFn: BinaryFn,
    fmaFn: TernaryFn,
    expFn: UnaryFn,
    tanhFn: UnaryFn,
    sigmoidFn: UnaryFn,
    hsumFn: ReduceFn,
    hmaxFn: ReduceFn,
    backend: Backend,
};

var cached_ops: ?VecOps = null;

/// Returns the dispatch table, initializing on first call.
pub fn getOps() *const VecOps {
    if (cached_ops) |*ops| return ops;
    cached_ops = initOps();
    return &cached_ops.?;
}

/// Detects the best available SIMD backend for the current CPU.
pub fn detectBackend() Backend {
    if (comptime builtin.cpu.arch == .aarch64) {
        return .neon;
    }
    if (comptime builtin.cpu.arch == .x86_64) {
        return if (x86.hasAvx2()) .avx2 else .scalar;
    }
    return .scalar;
}

/// Returns the scalar dispatch table (for testing).
pub fn getScalarOps() VecOps {
    return scalarTable();
}

fn initOps() VecOps {
    return switch (detectBackend()) {
        .avx2 => avx2Table(),
        .neon => neonTable(),
        .scalar => scalarTable(),
    };
}

fn avx2Table() VecOps {
    return .{
        .addFn = avx2.bulkAdd,
        .mulFn = avx2.bulkMul,
        .fmaFn = avx2.bulkFma,
        .expFn = avx2.bulkExp,
        .tanhFn = avx2.bulkTanh,
        .sigmoidFn = avx2.bulkSigmoid,
        .hsumFn = avx2.bulkHsum,
        .hmaxFn = avx2.bulkHmax,
        .backend = .avx2,
    };
}

fn neonTable() VecOps {
    return .{
        .addFn = neon.bulkAdd,
        .mulFn = neon.bulkMul,
        .fmaFn = neon.bulkFma,
        .expFn = neon.bulkExp,
        .tanhFn = neon.bulkTanh,
        .sigmoidFn = neon.bulkSigmoid,
        .hsumFn = neon.bulkHsum,
        .hmaxFn = neon.bulkHmax,
        .backend = .neon,
    };
}

fn scalarTable() VecOps {
    return .{
        .addFn = scalarAdd,
        .mulFn = scalarMul,
        .fmaFn = scalarFma,
        .expFn = scalarExpBulk,
        .tanhFn = scalarTanhBulk,
        .sigmoidFn = scalarSigmoidBulk,
        .hsumFn = scalarHsum,
        .hmaxFn = scalarHmax,
        .backend = .scalar,
    };
}

// ============================================================
// x86-64 CPU feature detection (only compiled on x86_64)
// ============================================================

const x86 = if (builtin.cpu.arch == .x86_64) struct {
    const CpuidResult = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

    fn hasAvx2() bool {
        // Check max supported CPUID leaf
        const leaf0 = doCpuid(0, 0);
        if (leaf0.eax < 7) return false;

        // AVX2: CPUID.7.0:EBX[5]
        const leaf7 = doCpuid(7, 0);
        if (leaf7.ebx & (1 << 5) == 0) return false;

        // OSXSAVE: CPUID.1:ECX[27]
        const leaf1 = doCpuid(1, 0);
        if (leaf1.ecx & (1 << 27) == 0) return false;

        // XCR0 bits 1+2 (SSE + AVX state saved by OS)
        const xcr0 = doXgetbv(0);
        return (xcr0 & 0x6) == 0x6;
    }

    fn doCpuid(leaf: u32, subleaf: u32) CpuidResult {
        var eax: u32 = undefined;
        var ebx: u32 = undefined;
        var ecx: u32 = undefined;
        var edx: u32 = undefined;
        asm volatile ("cpuid"
            : [_eax] "={eax}" (eax),
              [_ebx] "={ebx}" (ebx),
              [_ecx] "={ecx}" (ecx),
              [_edx] "={edx}" (edx),
            : [_leaf] "{eax}" (leaf),
              [_subleaf] "{ecx}" (subleaf),
        );
        return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
    }

    fn doXgetbv(index: u32) u64 {
        var lo: u32 = undefined;
        var hi: u32 = undefined;
        asm volatile ("xgetbv"
            : [_lo] "={eax}" (lo),
              [_hi] "={edx}" (hi),
            : [_idx] "{ecx}" (index),
        );
        return (@as(u64, hi) << 32) | lo;
    }
} else struct {
    fn hasAvx2() bool {
        return false;
    }
};

// ============================================================
// Scalar fallback implementations
// ============================================================

fn scalarAdd(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] + b[i];
    }
}

fn scalarMul(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] * b[i];
    }
}

fn scalarFma(dst: [*]f32, a: [*]const f32, b: [*]const f32, c: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

fn scalarExpBulk(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = scalarExp(src[i]);
    }
}

fn scalarTanhBulk(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = scalarTanhVal(src[i]);
    }
}

fn scalarSigmoidBulk(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = scalarSigmoidVal(src[i]);
    }
}

fn scalarHsum(src: [*]const f32, len: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += src[i];
    }
    return sum;
}

fn scalarHmax(src: [*]const f32, len: usize) f32 {
    if (len == 0) return -std.math.inf(f32);
    var max_val: f32 = src[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (src[i] > max_val) max_val = src[i];
    }
    return max_val;
}

// Scalar math helpers
fn scalarExp(x: f32) f32 {
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

fn scalarTanhVal(x: f32) f32 {
    const clamped = @max(@min(x, @as(f32, 10.0)), @as(f32, -10.0));
    const exp2x = scalarExp(clamped + clamped);
    return (exp2x - 1.0) / (exp2x + 1.0);
}

fn scalarSigmoidVal(x: f32) f32 {
    return 1.0 / (1.0 + scalarExp(-x));
}

