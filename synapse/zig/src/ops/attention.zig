//! Fused scaled dot-product attention kernel.
//!
//! Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_head)) × V
//!
//! Two implementations:
//! - Fused: single function with tiled seq_q, keeping intermediates in L1/L2 cache
//! - Naive: separate matmul → scale → mask → softmax → matmul calls

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");
const matmul_mod = @import("matmul.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

const MR = matmul_mod.MR;
const NR = matmul_mod.NR;
const KC = matmul_mod.KC;

/// Tile size for the query sequence dimension in fused attention.
/// Keeps intermediate S_tile [TILE_Q, seq_k] in L2 cache.
pub const TILE_Q: usize = 32;

pub const AttentionConfig = struct {
    causal: bool = false,
    return_weights: bool = false,
};

pub const AttentionResult = struct {
    output: Tensor(f32),
    weights: ?Tensor(f32),

    pub fn release(self: AttentionResult) void {
        self.output.release();
        if (self.weights) |w| w.release();
    }
};

/// Fused scaled dot-product attention with tiled memory access.
///
/// Q: [batch, heads, seq_q, d_head], K/V: [batch, heads, seq_k, d_head]
/// Output: [batch, heads, seq_q, d_head], optional weights: [batch, heads, seq_q, seq_k]
///
/// Tiles over seq_q to keep intermediate score matrix in L1/L2 cache.
/// Uses tiled SGEMM for matmuls and online softmax for normalization.
pub fn attention(
    allocator: std.mem.Allocator,
    q: Tensor(f32),
    k: Tensor(f32),
    v: Tensor(f32),
    config: AttentionConfig,
) !AttentionResult {
    const dims = try validateShapes(q, k, v);
    const batch = dims.batch;
    const heads = dims.heads;
    const seq_q = dims.seq_q;
    const d_head = dims.d_head;
    const seq_k = dims.seq_k;

    const out_numel = batch * heads * seq_q * d_head;
    const out_storage = try Storage.create(allocator, f32, @max(out_numel, 1));
    const output = Tensor(f32).init(out_storage, Shape.init(&[_]usize{ batch, heads, seq_q, d_head }));
    out_storage.release();
    errdefer output.release();

    var weights: ?Tensor(f32) = null;
    if (config.return_weights) {
        const w_numel = batch * heads * seq_q * seq_k;
        const w_storage = try Storage.create(allocator, f32, @max(w_numel, 1));
        weights = Tensor(f32).init(w_storage, Shape.init(&[_]usize{ batch, heads, seq_q, seq_k }));
        w_storage.release();
    }
    errdefer if (weights) |w| w.release();

    if (out_numel == 0 or d_head == 0 or seq_k == 0)
        return .{ .output = output, .weights = weights };

    // Scratch: S_tile [TILE_Q, seq_k]
    const s_tile = try allocator.alloc(f32, TILE_Q * seq_k);
    defer allocator.free(s_tile);

    // Packing buffers for sgemmTiled (sized for both matmuls per tile)
    const tq_al = ((TILE_Q + MR - 1) / MR) * MR;
    const sk_al = ((seq_k + NR - 1) / NR) * NR;
    const dh_al = ((d_head + NR - 1) / NR) * NR;
    const kc_max = @max(@min(KC, d_head), @min(KC, seq_k));
    const pa_size = @max(tq_al * kc_max, 1);
    const pb_size = @max(@max(sk_al * @min(KC, d_head), dh_al * @min(KC, seq_k)), 1);

    const packed_a = try allocator.alloc(f32, pa_size);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, pb_size);
    defer allocator.free(packed_b);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
    const q_data = q.storage.dataAs(f32).ptr;
    const k_data = k.storage.dataAs(f32).ptr;
    const v_data = v.storage.dataAs(f32).ptr;
    const o_data = output.storage.dataAs(f32).ptr;
    const w_data: ?[*]f32 = if (weights) |w| w.storage.dataAs(f32).ptr else null;

    const o_bh_size = seq_q * d_head;
    const w_bh_size = seq_q * seq_k;

    for (0..batch) |b| {
        for (0..heads) |h| {
            const q_bh = q_data + q.offset + b * q.strides[0] + h * q.strides[1];
            const k_bh = k_data + k.offset + b * k.strides[0] + h * k.strides[1];
            const v_bh = v_data + v.offset + b * v.strides[0] + h * v.strides[1];
            const o_bh = o_data + (b * heads + h) * o_bh_size;
            const w_bh: ?[*]f32 = if (w_data) |wd| wd + (b * heads + h) * w_bh_size else null;

            var tq: usize = 0;
            while (tq < seq_q) : (tq += TILE_Q) {
                const tqs = @min(TILE_Q, seq_q - tq);

                // Zero S_tile before accumulation
                @memset(s_tile[0 .. tqs * seq_k], 0);

                // S_tile[tqs, seq_k] = Q_tile[tqs, d_head] @ K^T[d_head, seq_k]
                matmul_mod.sgemmTiled(
                    tqs,
                    seq_k,
                    d_head,
                    q_bh + tq * q.strides[2],
                    q.strides[2],
                    false,
                    k_bh,
                    k.strides[2],
                    true,
                    s_tile.ptr,
                    seq_k,
                    packed_a.ptr,
                    packed_b.ptr,
                );

                // Scale: S /= sqrt(d_head)
                for (s_tile[0 .. tqs * seq_k]) |*val| val.* *= scale;

                // Causal mask: S[i][j] = -inf where j > query_position
                if (config.causal) {
                    for (0..tqs) |i| {
                        const mask_start = tq + i + 1;
                        if (mask_start < seq_k) {
                            const row_start = i * seq_k + mask_start;
                            const row_end = (i + 1) * seq_k;
                            @memset(s_tile[row_start..row_end], -std.math.inf(f32));
                        }
                    }
                }

                // Online softmax per row
                for (0..tqs) |i| {
                    const rb = i * seq_k;
                    var mx: f32 = -std.math.inf(f32);
                    var se: f32 = 0.0;
                    for (0..seq_k) |j| {
                        const x = s_tile[rb + j];
                        if (x > mx) {
                            se = se * @exp(mx - x) + 1.0;
                            mx = x;
                        } else {
                            se += @exp(x - mx);
                        }
                    }
                    const inv = 1.0 / se;
                    for (0..seq_k) |j| {
                        s_tile[rb + j] = @exp(s_tile[rb + j] - mx) * inv;
                    }
                }

                // Store weights if requested
                if (w_bh) |wb| {
                    for (0..tqs) |i| {
                        @memcpy(
                            (wb + (tq + i) * seq_k)[0..seq_k],
                            s_tile[i * seq_k ..][0..seq_k],
                        );
                    }
                }

                // Zero output tile, then O_tile = S_tile @ V
                for (0..tqs) |i| {
                    @memset((o_bh + (tq + i) * d_head)[0..d_head], 0);
                }
                matmul_mod.sgemmTiled(
                    tqs,
                    d_head,
                    seq_k,
                    s_tile.ptr,
                    seq_k,
                    false,
                    v_bh,
                    v.strides[2],
                    false,
                    o_bh + tq * d_head,
                    d_head,
                    packed_a.ptr,
                    packed_b.ptr,
                );
            }
        }
    }

    return .{ .output = output, .weights = weights };
}

/// Naive scaled dot-product attention: separate matmul → scale → mask → softmax → matmul.
/// Used as correctness and benchmark baseline.
pub noinline fn naiveAttention(
    allocator: std.mem.Allocator,
    q: Tensor(f32),
    k: Tensor(f32),
    v: Tensor(f32),
    config: AttentionConfig,
) !AttentionResult {
    const dims = try validateShapes(q, k, v);
    const batch = dims.batch;
    const heads = dims.heads;
    const seq_q = dims.seq_q;
    const d_head = dims.d_head;
    const seq_k = dims.seq_k;

    const out_numel = batch * heads * seq_q * d_head;
    const out_storage = try Storage.create(allocator, f32, @max(out_numel, 1));
    const output = Tensor(f32).init(out_storage, Shape.init(&[_]usize{ batch, heads, seq_q, d_head }));
    out_storage.release();
    errdefer output.release();

    var weights: ?Tensor(f32) = null;
    if (config.return_weights) {
        const w_numel = batch * heads * seq_q * seq_k;
        const w_storage = try Storage.create(allocator, f32, @max(w_numel, 1));
        weights = Tensor(f32).init(w_storage, Shape.init(&[_]usize{ batch, heads, seq_q, seq_k }));
        w_storage.release();
    }
    errdefer if (weights) |w| w.release();

    if (out_numel == 0 or d_head == 0 or seq_k == 0)
        return .{ .output = output, .weights = weights };

    const s_buf = try allocator.alloc(f32, seq_q * seq_k);
    defer allocator.free(s_buf);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
    const q_data = q.storage.dataAs(f32).ptr;
    const k_data = k.storage.dataAs(f32).ptr;
    const v_data = v.storage.dataAs(f32).ptr;
    const o_data = output.storage.dataAs(f32).ptr;
    const w_data: ?[*]f32 = if (weights) |w| w.storage.dataAs(f32).ptr else null;

    const o_bh_size = seq_q * d_head;
    const w_bh_size = seq_q * seq_k;

    for (0..batch) |b| {
        for (0..heads) |h| {
            const q_bh = q_data + q.offset + b * q.strides[0] + h * q.strides[1];
            const k_bh = k_data + k.offset + b * k.strides[0] + h * k.strides[1];
            const v_bh = v_data + v.offset + b * v.strides[0] + h * v.strides[1];
            const o_bh = o_data + (b * heads + h) * o_bh_size;

            // Step 1: S = Q @ K^T (naive triple-loop)
            matmul_mod.naiveSgemm(
                seq_q,
                seq_k,
                d_head,
                q_bh,
                q.strides[2],
                false,
                k_bh,
                k.strides[2],
                true,
                s_buf.ptr,
                seq_k,
            );

            // Step 2: Scale
            for (s_buf[0 .. seq_q * seq_k]) |*val| val.* *= scale;

            // Step 3: Causal mask
            if (config.causal) {
                for (0..seq_q) |i| {
                    if (i + 1 < seq_k) {
                        @memset(s_buf[i * seq_k + i + 1 .. (i + 1) * seq_k], -std.math.inf(f32));
                    }
                }
            }

            // Step 4: Softmax per row (3-pass: max, exp+sum, normalize)
            for (0..seq_q) |i| {
                const rb = i * seq_k;
                var mx: f32 = -std.math.inf(f32);
                for (0..seq_k) |j| {
                    if (s_buf[rb + j] > mx) mx = s_buf[rb + j];
                }
                var se: f32 = 0.0;
                for (0..seq_k) |j| {
                    s_buf[rb + j] = @exp(s_buf[rb + j] - mx);
                    se += s_buf[rb + j];
                }
                const inv = 1.0 / se;
                for (0..seq_k) |j| {
                    s_buf[rb + j] *= inv;
                }
            }

            // Copy weights
            if (w_data) |wd| {
                const wb = wd + (b * heads + h) * w_bh_size;
                @memcpy(wb[0 .. seq_q * seq_k], s_buf[0 .. seq_q * seq_k]);
            }

            // Step 5: O = S @ V (naive triple-loop)
            matmul_mod.naiveSgemm(
                seq_q,
                d_head,
                seq_k,
                s_buf.ptr,
                seq_k,
                false,
                v_bh,
                v.strides[2],
                false,
                o_bh,
                d_head,
            );
        }
    }

    return .{ .output = output, .weights = weights };
}

// ================================================================
// Internal helpers
// ================================================================

fn validateShapes(q: Tensor(f32), k: Tensor(f32), v: Tensor(f32)) !struct {
    batch: usize,
    heads: usize,
    seq_q: usize,
    d_head: usize,
    seq_k: usize,
} {
    if (q.shape.ndim != 4 or k.shape.ndim != 4 or v.shape.ndim != 4)
        return error.InvalidDimensions;

    const batch = q.shape.dims[0];
    const heads = q.shape.dims[1];
    const seq_q = q.shape.dims[2];
    const d_head = q.shape.dims[3];
    const seq_k = k.shape.dims[2];

    if (k.shape.dims[0] != batch or k.shape.dims[1] != heads) return error.ShapeMismatch;
    if (v.shape.dims[0] != batch or v.shape.dims[1] != heads) return error.ShapeMismatch;
    if (v.shape.dims[2] != seq_k) return error.ShapeMismatch;
    if (k.shape.dims[3] != d_head or v.shape.dims[3] != d_head) return error.ShapeMismatch;

    return .{ .batch = batch, .heads = heads, .seq_q = seq_q, .d_head = d_head, .seq_k = seq_k };
}
