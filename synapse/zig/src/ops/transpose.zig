//! Cache-oblivious matrix transpose.
//!
//! B[j, i] = A[i, j] where A is M x N (row-major), B is N x M (row-major).
//!
//! Uses recursive decomposition: split the larger dimension in half until
//! both dimensions fit in L1 cache, then transpose directly. This naturally
//! adapts to any cache hierarchy without explicit tuning.

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

/// Base case block size. When both dimensions are <= BLOCK, use direct loops.
/// Chosen so that src block + dst block fit comfortably in L1 (typically 32-64 KB).
const BLOCK: usize = 64;

// ================================================================
// Public Tensor API
// ================================================================

/// Transpose a 2D tensor: result[j, i] = input[i, j].
/// input: 2D tensor [M, N]
/// Returns: 2D tensor [N, M]
pub fn transpose2d(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
) !Tensor(f32) {
    if (input.shape.ndim != 2) return error.InvalidDimensions;

    const m = input.shape.dims[0];
    const n = input.shape.dims[1];

    const out_shape = Shape.init(&[_]usize{ n, m });
    const numel = m * n;
    const out_storage = try Storage.create(allocator, f32, if (numel == 0) 1 else numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (numel == 0) return result;

    transposeRaw(input.dataPtr(), n, result.dataPtr(), m, m, n);

    return result;
}

// ================================================================
// Raw API
// ================================================================

/// Cache-oblivious transpose on flat row-major arrays.
/// src: M x N with leading dimension lda (>= N)
/// dst: N x M with leading dimension ldb (>= M)
pub fn transposeRaw(
    src: [*]const f32,
    lda: usize,
    dst: [*]f32,
    ldb: usize,
    m: usize,
    n: usize,
) void {
    transposeBlock(src, lda, dst, ldb, 0, 0, m, n);
}

// ================================================================
// Recursive cache-oblivious implementation
// ================================================================

fn transposeBlock(
    src: [*]const f32,
    lda: usize,
    dst: [*]f32,
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
) void {
    if (rows == 0 or cols == 0) return;

    if (rows <= BLOCK and cols <= BLOCK) {
        // Base case: direct transpose with sequential writes
        for (0..rows) |di| {
            const src_row = src + (row_start + di) * lda + col_start;
            for (0..cols) |dj| {
                dst[(col_start + dj) * ldb + (row_start + di)] = src_row[dj];
            }
        }
        return;
    }

    if (rows >= cols) {
        // Split rows
        const half = rows / 2;
        transposeBlock(src, lda, dst, ldb, row_start, col_start, half, cols);
        transposeBlock(src, lda, dst, ldb, row_start + half, col_start, rows - half, cols);
    } else {
        // Split cols
        const half = cols / 2;
        transposeBlock(src, lda, dst, ldb, row_start, col_start, rows, half);
        transposeBlock(src, lda, dst, ldb, row_start, col_start + half, rows, cols - half);
    }
}
