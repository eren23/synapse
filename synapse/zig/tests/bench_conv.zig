//! Benchmark: 3x3 conv on 64x64x3 input, 32 filters, 100 iterations.
//! Compares naive 4-loop vs im2col+GEMM.
//! Pass criteria: im2col+GEMM >= 8x vs naive 4-loop. Correctness within 1e-4.

const std = @import("std");
const synapse = @import("synapse");
const conv_mod = synapse.ops.conv;
const matmul_mod = synapse.ops.matmul;
const naive_conv = @import("naive_conv");

// Problem dimensions
const BATCH: usize = 1;
const C_IN: usize = 3;
const C_OUT: usize = 32;
const H_IN: usize = 64;
const W_IN: usize = 64;
const KH: usize = 3;
const KW: usize = 3;
const STRIDE_H: usize = 1;
const STRIDE_W: usize = 1;
const PAD_H: usize = 1;
const PAD_W: usize = 1;

const H_OUT: usize = (H_IN + 2 * PAD_H - KH) / STRIDE_H + 1; // 64
const W_OUT: usize = (W_IN + 2 * PAD_W - KW) / STRIDE_W + 1; // 64

const IN_SIZE: usize = BATCH * C_IN * H_IN * W_IN;
const K_SIZE: usize = C_OUT * C_IN * KH * KW;
const OUT_SIZE: usize = BATCH * C_OUT * H_OUT * W_OUT;

// im2col GEMM dimensions
const GEMM_M: usize = C_OUT;
const GEMM_K: usize = C_IN * KH * KW;
const GEMM_N: usize = H_OUT * W_OUT;
const COL_SIZE: usize = GEMM_K * GEMM_N;

const WARMUP: usize = 3;
const ITERS: usize = 100;

// FLOPs: 2 * C_out * C_in * KH * KW * H_out * W_out per sample
const FLOPS_PER_ITER: u64 = 2 * C_OUT * C_IN * KH * KW * H_OUT * W_OUT * BATCH;
const TOTAL_FLOPS: u64 = FLOPS_PER_ITER * ITERS;

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // Allocate flat arrays
    const input = try allocator.alloc(f32, IN_SIZE);
    defer allocator.free(input);
    const kernel = try allocator.alloc(f32, K_SIZE);
    defer allocator.free(kernel);
    const out_naive = try allocator.alloc(f32, OUT_SIZE);
    defer allocator.free(out_naive);
    const out_gemm = try allocator.alloc(f32, OUT_SIZE);
    defer allocator.free(out_gemm);

    // im2col and packing buffers
    const col_buf = try allocator.alloc(f32, COL_SIZE);
    defer allocator.free(col_buf);

    const eff_kc = @min(matmul_mod.KC, GEMM_K);
    const eff_mc = ((@min(matmul_mod.MC, GEMM_M) + matmul_mod.MR - 1) / matmul_mod.MR) * matmul_mod.MR;
    const eff_nc = ((@min(matmul_mod.NC, GEMM_N) + matmul_mod.NR - 1) / matmul_mod.NR) * matmul_mod.NR;
    const packed_a = try allocator.alloc(f32, eff_mc * eff_kc);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, eff_nc * eff_kc);
    defer allocator.free(packed_b);

    // Fill with deterministic data
    fillData(input, 42);
    fillData(kernel, 137);

    print("=== Conv2d Benchmark: 3x3 on {d}x{d}x{d}, {d} filters, {d} iters ===\n", .{ H_IN, W_IN, C_IN, C_OUT, ITERS });
    print("FLOPs per iter: {d:.2} M\n", .{@as(f64, @floatFromInt(FLOPS_PER_ITER)) / 1e6});
    print("\n", .{});

    // ---- Warmup ----
    var sink: f32 = 0;
    for (0..WARMUP) |_| {
        conv_mod.im2colGemmBatch(
            input.ptr, kernel.ptr, out_gemm.ptr,
            C_IN, C_OUT, H_IN, W_IN,
            KH, KW, H_OUT, W_OUT,
            STRIDE_H, STRIDE_W, PAD_H, PAD_W,
            col_buf.ptr, packed_a.ptr, packed_b.ptr,
        );
        sink += out_gemm[0];
    }

    // ---- Benchmark naive ----
    const naive_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        naive_conv.naiveConv2dRaw(
            input.ptr, kernel.ptr, out_naive.ptr,
            BATCH, C_IN, C_OUT, H_IN, W_IN,
            KH, KW, H_OUT, W_OUT,
            STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const naive_end = std.time.nanoTimestamp();
    const naive_ns: u64 = @intCast(naive_end - naive_start);
    sink += out_naive[0];

    // ---- Benchmark im2col+GEMM ----
    const gemm_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        conv_mod.im2colGemmBatch(
            input.ptr, kernel.ptr, out_gemm.ptr,
            C_IN, C_OUT, H_IN, W_IN,
            KH, KW, H_OUT, W_OUT,
            STRIDE_H, STRIDE_W, PAD_H, PAD_W,
            col_buf.ptr, packed_a.ptr, packed_b.ptr,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const gemm_end = std.time.nanoTimestamp();
    const gemm_ns: u64 = @intCast(gemm_end - gemm_start);
    sink += out_gemm[0];

    // ---- Verify correctness ----
    naive_conv.naiveConv2dRaw(
        input.ptr, kernel.ptr, out_naive.ptr,
        BATCH, C_IN, C_OUT, H_IN, W_IN,
        KH, KW, H_OUT, W_OUT,
        STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    );
    conv_mod.im2colGemmBatch(
        input.ptr, kernel.ptr, out_gemm.ptr,
        C_IN, C_OUT, H_IN, W_IN,
        KH, KW, H_OUT, W_OUT,
        STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        col_buf.ptr, packed_a.ptr, packed_b.ptr,
    );

    var max_diff: f32 = 0;
    for (0..OUT_SIZE) |i| {
        const diff = @abs(out_gemm[i] - out_naive[i]);
        const denom = @max(@abs(out_naive[i]), @as(f32, 1.0));
        const rel = diff / denom;
        if (rel > max_diff) max_diff = rel;
    }

    // ---- Report ----
    const naive_ms = @as(f64, @floatFromInt(naive_ns)) / 1e6;
    const gemm_ms = @as(f64, @floatFromInt(gemm_ns)) / 1e6;
    const speedup = @as(f64, @floatFromInt(naive_ns)) / @as(f64, @floatFromInt(gemm_ns));
    const gemm_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(gemm_ns));
    const naive_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(naive_ns));

    print("Naive:      {d:.2} ms  ({d:.3} GFLOPS)\n", .{ naive_ms, naive_gflops });
    print("im2col+GEMM: {d:.2} ms  ({d:.3} GFLOPS)\n", .{ gemm_ms, gemm_gflops });
    print("Speedup:    {d:.2}x\n", .{speedup});
    print("Max relative error: {e:.2}\n", .{max_diff});
    print("\n", .{});

    // ---- Pass/fail ----
    var passed = true;

    if (max_diff > 1e-4) {
        print("FAIL: max relative error {e:.2} > 1e-4\n", .{max_diff});
        passed = false;
    } else {
        print("PASS: correctness within 1e-4\n", .{});
    }

    if (speedup < 8.0) {
        print("FAIL: speedup {d:.2}x < required 8.0x\n", .{speedup});
        passed = false;
    } else {
        print("PASS: speedup {d:.2}x >= 8.0x\n", .{speedup});
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
