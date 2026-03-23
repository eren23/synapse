//! Benchmark: INT8 quantized GEMM vs f32 GEMM on 512×512, 100 iterations.
//!
//! Pass criteria:
//! - INT8 GEMM >= 2x throughput vs f32 naive GEMM
//! - INT8 GEMM output within 1% relative error of f32
//! - End-to-end (quantize + INT8 GEMM + dequantize) >= 1.5x vs f32 naive GEMM

const std = @import("std");
const synapse = @import("synapse");
const quantize = synapse.ops.quantize;
const qmatmul = synapse.ops.qmatmul;

const N_SIZE: usize = 512;
const WARMUP: usize = 3;
const ITERS: usize = 100;

const FLOPS_PER_ITER: u64 = 2 * N_SIZE * N_SIZE * N_SIZE;
const TOTAL_FLOPS: u64 = FLOPS_PER_ITER * ITERS;

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // Allocate f32 matrices
    const a_f32 = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(a_f32);
    const b_f32 = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(b_f32);
    const c_f32 = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(c_f32);
    const c_int8 = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(c_int8);

    // INT8 data
    const a_i8 = try allocator.alloc(i8, N_SIZE * N_SIZE);
    defer allocator.free(a_i8);
    const b_i8 = try allocator.alloc(i8, N_SIZE * N_SIZE);
    defer allocator.free(b_i8);
    const scales_a = try allocator.alloc(f32, N_SIZE);
    defer allocator.free(scales_a);
    const scales_b = try allocator.alloc(f32, N_SIZE);
    defer allocator.free(scales_b);

    // Packing buffers for INT8 tiled GEMM
    const packed_a = try allocator.alloc(i8, qmatmul.MC * qmatmul.KC);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(i8, qmatmul.KC * N_SIZE);
    defer allocator.free(packed_b);

    // Fill with deterministic data
    fillData(a_f32, 42);
    fillData(b_f32, 137);

    // Pre-quantize for the pure INT8 GEMM benchmark
    quantize.quantizePerChannelInt8(a_f32.ptr, N_SIZE, N_SIZE, a_i8.ptr, scales_a.ptr);
    quantize.quantizePerColumnInt8(b_f32.ptr, N_SIZE, N_SIZE, b_i8.ptr, scales_b.ptr);

    print("=== INT8 Quantized GEMM Benchmark: {d}x{d} x {d} iters ===\n", .{ N_SIZE, N_SIZE, ITERS });
    print("FLOPs per iter: {d:.2} M\n", .{@as(f64, @floatFromInt(FLOPS_PER_ITER)) / 1e6});
    print("\n", .{});

    // ---- Warmup ----
    var sink: f32 = 0;
    for (0..WARMUP) |_| {
        qmatmul.int8GemmTiled(
            N_SIZE, N_SIZE, N_SIZE,
            a_i8.ptr, N_SIZE, b_i8.ptr, N_SIZE,
            c_int8.ptr, N_SIZE,
            scales_a.ptr, scales_b.ptr,
            packed_a.ptr, packed_b.ptr,
        );
        sink += c_int8[0];
    }

    // ---- Benchmark f32 naive GEMM ----
    const f32_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        qmatmul.naiveF32Gemm(
            N_SIZE, N_SIZE, N_SIZE,
            a_f32.ptr, N_SIZE,
            b_f32.ptr, N_SIZE,
            c_f32.ptr, N_SIZE,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const f32_end = std.time.nanoTimestamp();
    const f32_ns: u64 = @intCast(f32_end - f32_start);
    sink += c_f32[0];

    // ---- Benchmark INT8 tiled GEMM (pre-quantized) ----
    const int8_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        qmatmul.int8GemmTiled(
            N_SIZE, N_SIZE, N_SIZE,
            a_i8.ptr, N_SIZE, b_i8.ptr, N_SIZE,
            c_int8.ptr, N_SIZE,
            scales_a.ptr, scales_b.ptr,
            packed_a.ptr, packed_b.ptr,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const int8_end = std.time.nanoTimestamp();
    const int8_ns: u64 = @intCast(int8_end - int8_start);
    sink += c_int8[0];

    // ---- Benchmark end-to-end: quantize + INT8 GEMM ----
    // Allocate separate buffers for end-to-end to include quantization time
    const e2e_a_i8 = try allocator.alloc(i8, N_SIZE * N_SIZE);
    defer allocator.free(e2e_a_i8);
    const e2e_b_i8 = try allocator.alloc(i8, N_SIZE * N_SIZE);
    defer allocator.free(e2e_b_i8);
    const e2e_scales_a = try allocator.alloc(f32, N_SIZE);
    defer allocator.free(e2e_scales_a);
    const e2e_scales_b = try allocator.alloc(f32, N_SIZE);
    defer allocator.free(e2e_scales_b);

    const e2e_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        // Quantize A per-row
        quantize.quantizePerChannelInt8(a_f32.ptr, N_SIZE, N_SIZE, e2e_a_i8.ptr, e2e_scales_a.ptr);
        // Quantize B per-column
        quantize.quantizePerColumnInt8(b_f32.ptr, N_SIZE, N_SIZE, e2e_b_i8.ptr, e2e_scales_b.ptr);
        // INT8 tiled GEMM
        qmatmul.int8GemmTiled(
            N_SIZE, N_SIZE, N_SIZE,
            e2e_a_i8.ptr, N_SIZE, e2e_b_i8.ptr, N_SIZE,
            c_int8.ptr, N_SIZE,
            e2e_scales_a.ptr, e2e_scales_b.ptr,
            packed_a.ptr, packed_b.ptr,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const e2e_end = std.time.nanoTimestamp();
    const e2e_ns: u64 = @intCast(e2e_end - e2e_start);
    sink += c_int8[0];

    // ---- Correctness check ----
    qmatmul.naiveF32Gemm(N_SIZE, N_SIZE, N_SIZE, a_f32.ptr, N_SIZE, b_f32.ptr, N_SIZE, c_f32.ptr, N_SIZE);
    qmatmul.int8GemmTiled(
        N_SIZE, N_SIZE, N_SIZE,
        a_i8.ptr, N_SIZE, b_i8.ptr, N_SIZE,
        c_int8.ptr, N_SIZE,
        scales_a.ptr, scales_b.ptr,
        packed_a.ptr, packed_b.ptr,
    );

    // Frobenius norm relative error (standard for GEMM accuracy)
    var sum_sq_diff: f64 = 0;
    var sum_sq_ref: f64 = 0;
    for (0..N_SIZE * N_SIZE) |i| {
        const diff: f64 = @as(f64, c_int8[i]) - @as(f64, c_f32[i]);
        sum_sq_diff += diff * diff;
        const ref_val: f64 = @as(f64, c_f32[i]);
        sum_sq_ref += ref_val * ref_val;
    }
    const frob_error: f64 = if (sum_sq_ref > 0) @sqrt(sum_sq_diff / sum_sq_ref) else 0;

    // ---- Report ----
    const f32_ms = @as(f64, @floatFromInt(f32_ns)) / 1e6;
    const int8_ms = @as(f64, @floatFromInt(int8_ns)) / 1e6;
    const e2e_ms = @as(f64, @floatFromInt(e2e_ns)) / 1e6;
    const int8_speedup = @as(f64, @floatFromInt(f32_ns)) / @as(f64, @floatFromInt(int8_ns));
    const e2e_speedup = @as(f64, @floatFromInt(f32_ns)) / @as(f64, @floatFromInt(e2e_ns));
    const f32_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(f32_ns));
    const int8_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(int8_ns));

    print("f32 naive:   {d:.2} ms  ({d:.2} GFLOPS)\n", .{ f32_ms, f32_gflops });
    print("INT8 tiled:  {d:.2} ms  ({d:.2} GFLOPS equiv)\n", .{ int8_ms, int8_gflops });
    print("End-to-end:  {d:.2} ms\n", .{e2e_ms});
    print("\n", .{});
    print("INT8 speedup:     {d:.2}x\n", .{int8_speedup});
    print("End-to-end:       {d:.2}x\n", .{e2e_speedup});
    print("Frobenius rel error: {e:.4}\n", .{frob_error});
    print("\n", .{});

    // ---- Pass/fail checks ----
    var passed = true;

    if (frob_error > 0.01) {
        print("FAIL: Frobenius relative error {e:.4} > 1%\n", .{frob_error});
        passed = false;
    } else {
        print("PASS: correctness within 1% (Frobenius: {e:.4})\n", .{frob_error});
    }

    if (int8_speedup < 2.0) {
        print("FAIL: INT8 speedup {d:.2}x < required 2.0x\n", .{int8_speedup});
        passed = false;
    } else {
        print("PASS: INT8 speedup {d:.2}x >= 2.0x\n", .{int8_speedup});
    }

    if (e2e_speedup < 1.5) {
        print("FAIL: end-to-end speedup {d:.2}x < required 1.5x\n", .{e2e_speedup});
        passed = false;
    } else {
        print("PASS: end-to-end speedup {d:.2}x >= 1.5x\n", .{e2e_speedup});
    }

    // Prevent sink from being optimized away
    if (sink == 0.0) print("sink: {d}\n", .{sink});

    if (!passed) std.process.exit(1);
}

fn fillData(data: []f32, seed: u32) void {
    var s: u32 = seed;
    for (data) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
}
