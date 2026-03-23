//! Rotary Positional Embedding (RoPE) with SIMD vectorization.
//! Applies complex rotation to pairs of tensor elements using precomputed cos/sin tables.
//! Supports position offset for KV-cache scenarios.
//!
//! For each pair (x[2i], x[2i+1]) at sequence position `pos`:
//!   x_rot[2i]   = x[2i] * cos(pos*θ_i) - x[2i+1] * sin(pos*θ_i)
//!   x_rot[2i+1] = x[2i] * sin(pos*θ_i) + x[2i+1] * cos(pos*θ_i)
//! where θ_i = 1 / (10000 ^ (2i / d_head))

const std = @import("std");
const math = std.math;

// ============================================================
// Types
// ============================================================

/// Precomputed cos/sin tables for RoPE.
/// Layout: [max_seq_len][half_d] in row-major order.
pub const RopeTable = struct {
    cos: []f32,
    sin: []f32,
    max_seq_len: usize,
    half_d: usize,
    allocator: std.mem.Allocator,

    /// Generate cos/sin tables.
    /// theta_i = 1 / (10000 ^ (2i / d_head))
    /// cos_table[pos, i] = cos(pos * theta_i)
    /// sin_table[pos, i] = sin(pos * theta_i)
    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, d_head: usize) !RopeTable {
        std.debug.assert(d_head >= 2 and d_head % 2 == 0);
        std.debug.assert(max_seq_len > 0);

        const half_d = d_head / 2;
        const table_size = max_seq_len * half_d;

        const cos_buf = try allocator.alloc(f32, table_size);
        errdefer allocator.free(cos_buf);
        const sin_buf = try allocator.alloc(f32, table_size);

        const d_head_f: f32 = @floatFromInt(d_head);

        for (0..max_seq_len) |pos| {
            const pos_f: f32 = @floatFromInt(pos);
            for (0..half_d) |i| {
                const exp = @as(f32, @floatFromInt(2 * i)) / d_head_f;
                const theta = 1.0 / math.pow(f32, 10000.0, exp);
                const angle = pos_f * theta;
                const idx = pos * half_d + i;
                cos_buf[idx] = @cos(angle);
                sin_buf[idx] = @sin(angle);
            }
        }

        return .{
            .cos = cos_buf,
            .sin = sin_buf,
            .max_seq_len = max_seq_len,
            .half_d = half_d,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RopeTable) void {
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
    }
};

// ============================================================
// SIMD Implementation (Primary)
// ============================================================

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// Shuffle masks for deinterleave/reinterleave of paired elements.
// Deinterleave {a0,a1,a2,a3} x {b0,b1,b2,b3} -> evens {a0,a2,b0,b2}, odds {a1,a3,b1,b3}
const deinterleave_even: @Vector(4, i32) = .{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) };
const deinterleave_odd: @Vector(4, i32) = .{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) };
// Reinterleave {e0,e1,e2,e3} x {o0,o1,o2,o3} -> lo {e0,o0,e1,o1}, hi {e2,o2,e3,o3}
const interleave_lo: @Vector(4, i32) = .{ 0, ~@as(i32, 0), 1, ~@as(i32, 1) };
const interleave_hi: @Vector(4, i32) = .{ 2, ~@as(i32, 2), 3, ~@as(i32, 3) };

/// SIMD-vectorized RoPE. Processes 4 pairs (8 elements) per iteration
/// using deinterleave-rotate-reinterleave with @shuffle.
///
/// Input/output layout: [batch, heads, seq, d_head] contiguous row-major.
/// cos_table/sin_table layout: [max_seq, half_d] contiguous row-major.
pub fn ropeSimd(
    output: []f32,
    input: []const f32,
    cos_table: []const f32,
    sin_table: []const f32,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    d_head: usize,
    half_d: usize,
    pos_offset: usize,
) void {
    // Use raw pointers to avoid per-element bounds checks in the hot loop
    const in_ptr = input.ptr;
    const out_ptr = output.ptr;
    const cos_ptr = cos_table.ptr;
    const sin_ptr = sin_table.ptr;

    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                const pos = s + pos_offset;
                const base = ((b * num_heads + h) * seq_len + s) * d_head;
                const table_off = pos * half_d;

                var p: usize = 0;
                // SIMD: process VEC_LEN pairs (2*VEC_LEN elements) per iteration
                while (p + VEC_LEN <= half_d) : (p += VEC_LEN) {
                    const off = base + 2 * p;
                    const toff = table_off + p;

                    // Load 2*VEC_LEN contiguous input elements
                    const v0: F32x4 = (in_ptr + off)[0..VEC_LEN].*;
                    const v1: F32x4 = (in_ptr + off + VEC_LEN)[0..VEC_LEN].*;

                    // Deinterleave into even/odd components
                    const even = @shuffle(f32, v0, v1, deinterleave_even);
                    const odd = @shuffle(f32, v0, v1, deinterleave_odd);

                    // Load cos/sin for these pairs
                    const cos_v: F32x4 = (cos_ptr + toff)[0..VEC_LEN].*;
                    const sin_v: F32x4 = (sin_ptr + toff)[0..VEC_LEN].*;

                    // Complex rotation (plain arithmetic to match scalar rounding)
                    const rot_even = even * cos_v - odd * sin_v;
                    const rot_odd = even * sin_v + odd * cos_v;

                    // Reinterleave and store
                    const out0 = @shuffle(f32, rot_even, rot_odd, interleave_lo);
                    const out1 = @shuffle(f32, rot_even, rot_odd, interleave_hi);

                    (out_ptr + off)[0..VEC_LEN].* = out0;
                    (out_ptr + off + VEC_LEN)[0..VEC_LEN].* = out1;
                }

                // Scalar tail for remaining pairs
                while (p < half_d) : (p += 1) {
                    const idx0 = base + 2 * p;
                    const x0 = input[idx0];
                    const x1 = input[idx0 + 1];
                    const c = cos_table[table_off + p];
                    const si = sin_table[table_off + p];
                    output[idx0] = x0 * c - x1 * si;
                    output[idx0 + 1] = x0 * si + x1 * c;
                }
            }
        }
    }
}

// ============================================================
// Scalar Reference Implementation (Benchmark Baseline)
// ============================================================

/// Scalar RoPE — no SIMD, one pair at a time. Used as correctness
/// reference and benchmark baseline.
pub fn ropeScalar(
    output: []f32,
    input: []const f32,
    cos_table: []const f32,
    sin_table: []const f32,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    d_head: usize,
    half_d: usize,
    pos_offset: usize,
) void {
    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                const pos = s + pos_offset;
                const base = ((b * num_heads + h) * seq_len + s) * d_head;
                const table_off = pos * half_d;

                for (0..half_d) |p| {
                    const idx0 = base + 2 * p;
                    const x0 = input[idx0];
                    const x1 = input[idx0 + 1];
                    const c = cos_table[table_off + p];
                    const si = sin_table[table_off + p];
                    output[idx0] = x0 * c - x1 * si;
                    output[idx0 + 1] = x0 * si + x1 * c;
                }
            }
        }
    }
}
