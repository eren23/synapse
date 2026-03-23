//! Benchmark: KV-cache incremental attention vs full recompute.
//!
//! Generates 256 tokens. For each new token position t:
//! - Cached: attention(Q[1, d], K_cached[0..t, d], V_cached[0..t, d]) — O(t*d)
//! - Full:   attention(Q[t, d], K[t, d], V[t, d]) — O(t^2 * d)
//!
//! Pass criteria: cached generation >= 3x faster than full recompute at seq_len=256.

const std = @import("std");
const synapse = @import("synapse");
const kvcache = synapse.ops.kvcache;

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const attention_mod = synapse.ops.attention;
const KvCache = kvcache.KvCache;

const N_KV_HEADS: usize = 8;
const D_HEAD: usize = 64;
const SEQ_LEN: usize = 256;
const WARMUP: usize = 2;
const ITERS: usize = 5;

/// Deterministic pseudo-random fill.
fn prngFill(buf: []f32, seed: u32) void {
    var s: u32 = seed;
    for (buf) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
}

/// Make a 4D tensor [1, heads, seq, d_head] from a flat buffer.
fn makeTensor(allocator: std.mem.Allocator, heads: usize, seq: usize, d_head: usize, seed: u32) !Tensor(f32) {
    const n = heads * seq * d_head;
    const storage = try Storage.create(allocator, f32, @max(n, 1));
    prngFill(storage.dataAs(f32)[0..n], seed);
    const t = Tensor(f32).init(storage, Shape.init(&[_]usize{ 1, heads, seq, d_head }));
    storage.release();
    return t;
}

/// Single-query attention using cached K/V: computes attention for one query token
/// against cached K/V of length seq_len. Returns output of shape [d_head * heads].
fn cachedSingleQueryAttention(
    q_row: []const f32, // [n_kv_heads * d_head]
    k_cache: []const f32, // [seq_len * n_kv_heads * d_head]
    v_cache: []const f32, // [seq_len * n_kv_heads * d_head]
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    output: []f32, // [n_heads * d_head]
    scratch: []f32, // [seq_len] scores scratch
) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    for (0..n_heads) |h| {
        const q_head = q_row[h * d_head ..][0..d_head];
        const out_head = output[h * d_head ..][0..d_head];

        // Compute scores: q . k[t] for each cached position
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |t| {
            const k_head = k_cache[t * n_heads * d_head + h * d_head ..][0..d_head];
            var dot: f32 = 0;
            for (0..d_head) |d| {
                dot += q_head[d] * k_head[d];
            }
            dot *= scale;
            scratch[t] = dot;
            if (dot > max_score) max_score = dot;
        }

        // Softmax
        var sum_exp: f32 = 0;
        for (0..seq_len) |t| {
            scratch[t] = @exp(scratch[t] - max_score);
            sum_exp += scratch[t];
        }
        const inv_sum = 1.0 / sum_exp;
        for (0..seq_len) |t| {
            scratch[t] *= inv_sum;
        }

        // Weighted sum of V
        @memset(out_head, 0);
        for (0..seq_len) |t| {
            const v_head = v_cache[t * n_heads * d_head + h * d_head ..][0..d_head];
            const w = scratch[t];
            for (0..d_head) |d| {
                out_head[d] += w * v_head[d];
            }
        }
    }
}

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    const stride = N_KV_HEADS * D_HEAD;

    // Pre-generate all Q/K/V token data
    var all_q: [SEQ_LEN * stride]f32 = undefined;
    var all_k: [SEQ_LEN * stride]f32 = undefined;
    var all_v: [SEQ_LEN * stride]f32 = undefined;
    prngFill(&all_q, 42);
    prngFill(&all_k, 137);
    prngFill(&all_v, 256);

    print("=== KV-Cache Benchmark: seq_len={d}, heads={d}, d_head={d}, {d} iters ===\n\n", .{ SEQ_LEN, N_KV_HEADS, D_HEAD, ITERS });

    var sink: f32 = 0;

    // ---- Warmup ----
    for (0..WARMUP) |_| {
        var cache = try KvCache.create(allocator, SEQ_LEN, N_KV_HEADS, D_HEAD);
        defer cache.destroy(allocator);

        var out_buf: [stride]f32 = undefined;
        var scratch: [SEQ_LEN]f32 = undefined;

        for (0..SEQ_LEN) |t| {
            try cache.append(all_k[t * stride ..][0..stride], all_v[t * stride ..][0..stride]);
            const s = cache.slice();
            cachedSingleQueryAttention(all_q[t * stride ..][0..stride], s.k, s.v, s.seq_len, N_KV_HEADS, D_HEAD, &out_buf, &scratch);
            sink += out_buf[0];
        }
    }

    // ---- Benchmark cached (incremental) ----
    var cached_ns_total: u64 = 0;
    for (0..ITERS) |_| {
        var cache = try KvCache.create(allocator, SEQ_LEN, N_KV_HEADS, D_HEAD);
        defer cache.destroy(allocator);

        var out_buf: [stride]f32 = undefined;
        var scratch: [SEQ_LEN]f32 = undefined;

        const start = std.time.nanoTimestamp();
        for (0..SEQ_LEN) |t| {
            try cache.append(all_k[t * stride ..][0..stride], all_v[t * stride ..][0..stride]);
            const s = cache.slice();
            cachedSingleQueryAttention(all_q[t * stride ..][0..stride], s.k, s.v, s.seq_len, N_KV_HEADS, D_HEAD, &out_buf, &scratch);
            sink += out_buf[0];
        }
        const end = std.time.nanoTimestamp();
        cached_ns_total += @intCast(end - start);
        asm volatile ("" ::: .{ .memory = true });
    }

    // ---- Benchmark full recompute ----
    var full_ns_total: u64 = 0;
    for (0..ITERS) |_| {
        const start = std.time.nanoTimestamp();
        for (1..SEQ_LEN + 1) |t| {
            // Build full Q/K/V tensors of length t, run full attention
            const q_tensor = try makeTensor(allocator, N_KV_HEADS, t, D_HEAD, 42 +% @as(u32, @intCast(t)));
            defer q_tensor.release();
            const k_tensor = try makeTensor(allocator, N_KV_HEADS, t, D_HEAD, 137 +% @as(u32, @intCast(t)));
            defer k_tensor.release();
            const v_tensor = try makeTensor(allocator, N_KV_HEADS, t, D_HEAD, 256 +% @as(u32, @intCast(t)));
            defer v_tensor.release();

            const result = try attention_mod.naiveAttention(allocator, q_tensor, k_tensor, v_tensor, .{});
            sink += result.output.storage.dataAs(f32)[0];
            result.release();
        }
        const end = std.time.nanoTimestamp();
        full_ns_total += @intCast(end - start);
        asm volatile ("" ::: .{ .memory = true });
    }

    const cached_ms = @as(f64, @floatFromInt(cached_ns_total)) / @as(f64, @floatFromInt(ITERS)) / 1e6;
    const full_ms = @as(f64, @floatFromInt(full_ns_total)) / @as(f64, @floatFromInt(ITERS)) / 1e6;
    const speedup = @as(f64, @floatFromInt(full_ns_total)) / @as(f64, @floatFromInt(cached_ns_total));

    print("--- Results (avg per generation) ---\n", .{});
    print("Cached (incremental): {d:.2} ms\n", .{cached_ms});
    print("Full recompute:       {d:.2} ms\n", .{full_ms});
    print("Speedup:              {d:.2}x\n\n", .{speedup});

    // Prevent sink optimization
    if (sink == 0.0) print("sink: {d}\n", .{sink});

    // ---- Pass/fail ----
    if (speedup < 3.0) {
        print("FAIL: speedup {d:.2}x < required 3.0x\n", .{speedup});
        std.process.exit(1);
    } else {
        print("PASS: speedup {d:.2}x >= 3.0x\n", .{speedup});
    }
}
