//! Benchmark: Fused vs naive scaled dot-product attention on [8,8,128,32], 50 iterations.
//! Also compares causal vs non-causal overhead.
//! Pass criteria: fused >= 3x speedup, causal overhead <= 10%.

const std = @import("std");
const synapse = @import("synapse");

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const attention_mod = synapse.ops.attention;

const BATCH: usize = 8;
const HEADS: usize = 8;
const SEQ_LEN: usize = 128;
const D_HEAD: usize = 32;
const WARMUP: usize = 3;
const ITERS: usize = 50;

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // Create Q, K, V tensors
    const q = try makeTensor4D(allocator, BATCH, HEADS, SEQ_LEN, D_HEAD, 42);
    defer q.release();
    const k = try makeTensor4D(allocator, BATCH, HEADS, SEQ_LEN, D_HEAD, 137);
    defer k.release();
    const v = try makeTensor4D(allocator, BATCH, HEADS, SEQ_LEN, D_HEAD, 256);
    defer v.release();

    print("=== Attention Benchmark: [{d},{d},{d},{d}] x {d} iters ===\n", .{ BATCH, HEADS, SEQ_LEN, D_HEAD, ITERS });
    print("\n", .{});

    var sink: f32 = 0;

    // ---- Warmup fused ----
    for (0..WARMUP) |_| {
        const result = try attention_mod.attention(allocator, q, k, v, .{});
        sink += result.output.storage.dataAs(f32)[0];
        result.release();
    }

    // ---- Benchmark naive non-causal ----
    const naive_nc_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        const result = try attention_mod.naiveAttention(allocator, q, k, v, .{});
        sink += result.output.storage.dataAs(f32)[0];
        result.release();
        asm volatile ("" ::: .{ .memory = true });
    }
    const naive_nc_end = std.time.nanoTimestamp();
    const naive_nc_ns: u64 = @intCast(naive_nc_end - naive_nc_start);

    // ---- Benchmark fused non-causal ----
    const fused_nc_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        const result = try attention_mod.attention(allocator, q, k, v, .{});
        sink += result.output.storage.dataAs(f32)[0];
        result.release();
        asm volatile ("" ::: .{ .memory = true });
    }
    const fused_nc_end = std.time.nanoTimestamp();
    const fused_nc_ns: u64 = @intCast(fused_nc_end - fused_nc_start);

    // ---- Benchmark fused causal ----
    const fused_c_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        const result = try attention_mod.attention(allocator, q, k, v, .{ .causal = true });
        sink += result.output.storage.dataAs(f32)[0];
        result.release();
        asm volatile ("" ::: .{ .memory = true });
    }
    const fused_c_end = std.time.nanoTimestamp();
    const fused_c_ns: u64 = @intCast(fused_c_end - fused_c_start);

    // ---- Benchmark naive causal ----
    const naive_c_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        const result = try attention_mod.naiveAttention(allocator, q, k, v, .{ .causal = true });
        sink += result.output.storage.dataAs(f32)[0];
        result.release();
        asm volatile ("" ::: .{ .memory = true });
    }
    const naive_c_end = std.time.nanoTimestamp();
    const naive_c_ns: u64 = @intCast(naive_c_end - naive_c_start);

    // ---- Correctness spot check ----
    const fused_r = try attention_mod.attention(allocator, q, k, v, .{});
    defer fused_r.release();
    const naive_r = try attention_mod.naiveAttention(allocator, q, k, v, .{});
    defer naive_r.release();

    const fused_data = fused_r.output.storage.dataAs(f32);
    const naive_data = naive_r.output.storage.dataAs(f32);
    const numel = fused_r.output.numel();
    var max_diff: f32 = 0;
    for (0..numel) |i| {
        const diff = @abs(fused_data[i] - naive_data[i]);
        const denom = @max(@abs(naive_data[i]), @as(f32, 1.0));
        const rel = diff / denom;
        if (rel > max_diff) max_diff = rel;
    }

    // ---- Report ----
    const naive_nc_ms = @as(f64, @floatFromInt(naive_nc_ns)) / 1e6;
    const fused_nc_ms = @as(f64, @floatFromInt(fused_nc_ns)) / 1e6;
    const fused_c_ms = @as(f64, @floatFromInt(fused_c_ns)) / 1e6;
    const naive_c_ms = @as(f64, @floatFromInt(naive_c_ns)) / 1e6;
    const speedup_nc = @as(f64, @floatFromInt(naive_nc_ns)) / @as(f64, @floatFromInt(fused_nc_ns));
    const speedup_c = @as(f64, @floatFromInt(naive_c_ns)) / @as(f64, @floatFromInt(fused_c_ns));
    const causal_overhead_fused = (fused_c_ms - fused_nc_ms) / fused_nc_ms * 100.0;
    const causal_overhead_naive = (naive_c_ms - naive_nc_ms) / naive_nc_ms * 100.0;

    print("--- Non-causal ---\n", .{});
    print("Naive:   {d:.2} ms\n", .{naive_nc_ms});
    print("Fused:   {d:.2} ms\n", .{fused_nc_ms});
    print("Speedup: {d:.2}x\n", .{speedup_nc});
    print("\n", .{});

    print("--- Causal ---\n", .{});
    print("Naive:   {d:.2} ms\n", .{naive_c_ms});
    print("Fused:   {d:.2} ms\n", .{fused_c_ms});
    print("Speedup: {d:.2}x\n", .{speedup_c});
    print("\n", .{});

    print("--- Causal overhead ---\n", .{});
    print("Fused:   {d:.1}%\n", .{causal_overhead_fused});
    print("Naive:   {d:.1}%\n", .{causal_overhead_naive});
    print("\n", .{});

    print("Max relative error: {e:.2}\n", .{max_diff});
    print("\n", .{});

    // ---- Pass/fail checks ----
    var passed = true;

    if (max_diff > 1e-4) {
        print("FAIL: max relative error {e:.2} > 1e-4\n", .{max_diff});
        passed = false;
    } else {
        print("PASS: correctness within 1e-4\n", .{});
    }

    if (speedup_nc < 3.0) {
        print("FAIL: non-causal speedup {d:.2}x < required 3.0x\n", .{speedup_nc});
        passed = false;
    } else {
        print("PASS: non-causal speedup {d:.2}x >= 3.0x\n", .{speedup_nc});
    }

    if (causal_overhead_fused > 10.0) {
        print("FAIL: fused causal overhead {d:.1}% > 10%\n", .{causal_overhead_fused});
        passed = false;
    } else {
        print("PASS: fused causal overhead {d:.1}% <= 10%\n", .{causal_overhead_fused});
    }

    // Prevent sink from being optimized away
    if (sink == 0.0) print("sink: {d}\n", .{sink});

    if (!passed) std.process.exit(1);
}

fn makeTensor4D(allocator: std.mem.Allocator, batch: usize, heads: usize, seq: usize, d_head: usize, seed: u32) !Tensor(f32) {
    const n = batch * heads * seq * d_head;
    const storage = try Storage.create(allocator, f32, @max(n, 1));
    const data = storage.dataAs(f32);
    var s: u32 = seed;
    for (0..n) |i| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        data[i] = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
    const shape = Shape.init(&[_]usize{ batch, heads, seq, d_head });
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    return t;
}
