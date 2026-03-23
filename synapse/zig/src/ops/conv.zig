//! Conv2d: im2col + GEMM-based forward convolution, with naive 4-loop fallback.
//!
//! Input layout: NCHW (batch, channels_in, height, width)
//! Kernel layout: (C_out, C_in, KH, KW)
//! Output layout: NCHW (batch, channels_out, H_out, W_out)
//!
//! im2col transforms input patches into a column matrix, then uses tiled SGEMM.
//! Direct fallback for 1x1 kernels with stride=1, pad=0 (skip im2col).
//! Naive 4-loop implementation provided for benchmark comparison.

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
const MC = matmul_mod.MC;
const KC = matmul_mod.KC;
const NC = matmul_mod.NC;

// ================================================================
// Public Tensor API
// ================================================================

/// Conv2d forward using im2col + GEMM. Falls back to direct GEMM for 1x1 kernels.
/// input:  4D tensor [N, C_in, H, W]
/// kernel: 4D tensor [C_out, C_in, KH, KW]
/// Returns: 4D tensor [N, C_out, H_out, W_out]
pub fn conv2d(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel: Tensor(f32),
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) !Tensor(f32) {
    if (input.shape.ndim != 4) return error.InvalidDimensions;
    if (kernel.shape.ndim != 4) return error.InvalidDimensions;
    if (input.shape.dims[1] != kernel.shape.dims[1]) return error.ShapeMismatch;
    if (stride_h == 0 or stride_w == 0) return error.InvalidStride;

    const batch = input.shape.dims[0];
    const c_in = input.shape.dims[1];
    const h_in = input.shape.dims[2];
    const w_in = input.shape.dims[3];
    const c_out = kernel.shape.dims[0];
    const kh = kernel.shape.dims[2];
    const kw = kernel.shape.dims[3];

    if (h_in + 2 * pad_h < kh) return error.InvalidDimensions;
    if (w_in + 2 * pad_w < kw) return error.InvalidDimensions;

    const h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    const w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;

    const out_shape = Shape.init(&[_]usize{ batch, c_out, h_out, w_out });
    const out_numel = batch * c_out * h_out * w_out;
    const out_storage = try Storage.create(allocator, f32, if (out_numel == 0) 1 else out_numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (out_numel == 0) return result;

    // 1x1 direct path: skip im2col when kernel is 1x1, stride=1, pad=0
    if (kh == 1 and kw == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0) {
        try conv2dDirect1x1(allocator, input, kernel, result, batch, c_in, c_out, h_in, w_in);
        return result;
    }

    // im2col + GEMM path
    try conv2dIm2col(
        allocator, input, kernel, result,
        batch, c_in, c_out, h_in, w_in,
        kh, kw, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w,
    );
    return result;
}

/// Naive 4-loop convolution for benchmark baseline. Same interface as conv2d.
pub noinline fn conv2dNaive(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel: Tensor(f32),
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) !Tensor(f32) {
    if (input.shape.ndim != 4) return error.InvalidDimensions;
    if (kernel.shape.ndim != 4) return error.InvalidDimensions;
    if (input.shape.dims[1] != kernel.shape.dims[1]) return error.ShapeMismatch;
    if (stride_h == 0 or stride_w == 0) return error.InvalidStride;

    const batch = input.shape.dims[0];
    const c_in = input.shape.dims[1];
    const h_in = input.shape.dims[2];
    const w_in = input.shape.dims[3];
    const c_out = kernel.shape.dims[0];
    const kh = kernel.shape.dims[2];
    const kw = kernel.shape.dims[3];

    if (h_in + 2 * pad_h < kh) return error.InvalidDimensions;
    if (w_in + 2 * pad_w < kw) return error.InvalidDimensions;

    const h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    const w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;

    const out_shape = Shape.init(&[_]usize{ batch, c_out, h_out, w_out });
    const out_numel = batch * c_out * h_out * w_out;
    const out_storage = try Storage.create(allocator, f32, if (out_numel == 0) 1 else out_numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (out_numel == 0) return result;

    const in_data = input.dataPtr();
    const k_data = kernel.dataPtr();
    const out_data = result.dataPtr();

    naiveConv2dRaw(
        in_data, k_data, out_data,
        batch, c_in, c_out, h_in, w_in,
        kh, kw, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w,
    );

    return result;
}

// ================================================================
// im2col transform
// ================================================================

/// Transform input patches into column matrix for GEMM-based convolution.
/// Output: (C_in * KH * KW) rows x (H_out * W_out) cols, row-major.
pub fn im2col(
    dst: [*]f32,
    src: [*]const f32,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
) void {
    const col_cols = h_out * w_out;

    for (0..c_in) |c| {
        for (0..kh) |fh| {
            for (0..kw) |fw| {
                const row = c * kh * kw + fh * kw + fw;
                const row_base = row * col_cols;

                for (0..h_out) |oh| {
                    const ih_signed: isize = @as(isize, @intCast(oh * stride_h + fh)) - @as(isize, @intCast(pad_h));

                    for (0..w_out) |ow| {
                        const iw_signed: isize = @as(isize, @intCast(ow * stride_w + fw)) - @as(isize, @intCast(pad_w));
                        const col = oh * w_out + ow;

                        if (ih_signed >= 0 and ih_signed < @as(isize, @intCast(h_in)) and
                            iw_signed >= 0 and iw_signed < @as(isize, @intCast(w_in)))
                        {
                            const ih: usize = @intCast(ih_signed);
                            const iw: usize = @intCast(iw_signed);
                            dst[row_base + col] = src[c * h_in * w_in + ih * w_in + iw];
                        } else {
                            dst[row_base + col] = 0;
                        }
                    }
                }
            }
        }
    }
}

// ================================================================
// Internal implementations
// ================================================================

/// im2col + GEMM convolution for a batch of inputs.
fn conv2dIm2col(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel: Tensor(f32),
    result: Tensor(f32),
    batch: usize,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) !void {
    const gemm_m = c_out;
    const gemm_k = c_in * kh * kw;
    const gemm_n = h_out * w_out;

    // Allocate im2col buffer: K x N
    const col_buf = try allocator.alloc(f32, gemm_k * gemm_n);
    defer allocator.free(col_buf);

    // Allocate GEMM packing buffers
    const eff_kc = @min(KC, gemm_k);
    const eff_mc = ((@min(MC, gemm_m) + MR - 1) / MR) * MR;
    const eff_nc = ((@min(NC, gemm_n) + NR - 1) / NR) * NR;
    const packed_a = try allocator.alloc(f32, eff_mc * eff_kc);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, eff_nc * eff_kc);
    defer allocator.free(packed_b);

    const in_data = input.dataPtr();
    const k_data = kernel.dataPtr();
    const out_data = result.dataPtr();

    const in_batch_stride = c_in * h_in * w_in;
    const out_batch_stride = c_out * h_out * w_out;

    for (0..batch) |n| {
        // im2col for this batch sample
        im2col(
            col_buf.ptr,
            in_data + n * in_batch_stride,
            c_in, h_in, w_in,
            kh, kw,
            stride_h, stride_w,
            pad_h, pad_w,
            h_out, w_out,
        );

        // Zero output for this batch (sgemmTiled accumulates)
        @memset((out_data + n * out_batch_stride)[0..out_batch_stride], 0);

        // GEMM: output[M,N] = kernel[M,K] * col[K,N]
        matmul_mod.sgemmTiled(
            gemm_m, gemm_n, gemm_k,
            k_data, gemm_k, false,
            col_buf.ptr, gemm_n, false,
            out_data + n * out_batch_stride, gemm_n,
            packed_a.ptr, packed_b.ptr,
        );
    }
}

/// Direct 1x1 convolution: skip im2col, GEMM on reshaped input directly.
fn conv2dDirect1x1(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel: Tensor(f32),
    result: Tensor(f32),
    batch: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) !void {
    const spatial = h * w;
    const gemm_m = c_out;
    const gemm_k = c_in;
    const gemm_n = spatial;

    const eff_kc = @min(KC, gemm_k);
    const eff_mc = ((@min(MC, gemm_m) + MR - 1) / MR) * MR;
    const eff_nc = ((@min(NC, gemm_n) + NR - 1) / NR) * NR;
    const packed_a = try allocator.alloc(f32, eff_mc * eff_kc);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, eff_nc * eff_kc);
    defer allocator.free(packed_b);

    const in_data = input.dataPtr();
    const k_data = kernel.dataPtr();
    const out_data = result.dataPtr();

    for (0..batch) |n| {
        const in_offset = n * c_in * spatial;
        const out_offset = n * c_out * spatial;

        @memset((out_data + out_offset)[0 .. c_out * spatial], 0);

        // GEMM: output[C_out, H*W] = kernel[C_out, C_in] * input[C_in, H*W]
        matmul_mod.sgemmTiled(
            gemm_m, gemm_n, gemm_k,
            k_data, gemm_k, false,
            in_data + in_offset, gemm_n, false,
            out_data + out_offset, gemm_n,
            packed_a.ptr, packed_b.ptr,
        );
    }
}

// ================================================================
// Raw API (for benchmarks without tensor overhead)
// ================================================================

/// Raw naive conv2d on flat NCHW arrays.
pub noinline fn naiveConv2dRaw(
    in_data: [*]const f32,
    k_data: [*]const f32,
    out_data: [*]f32,
    batch: usize,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    for (0..batch) |n| {
        for (0..c_out) |oc| {
            for (0..h_out) |oh| {
                for (0..w_out) |ow| {
                    var sum: f32 = 0;
                    for (0..c_in) |ic| {
                        for (0..kh) |fh| {
                            for (0..kw) |fw| {
                                const ih_signed: isize = @as(isize, @intCast(oh * stride_h + fh)) - @as(isize, @intCast(pad_h));
                                const iw_signed: isize = @as(isize, @intCast(ow * stride_w + fw)) - @as(isize, @intCast(pad_w));
                                if (ih_signed >= 0 and ih_signed < @as(isize, @intCast(h_in)) and
                                    iw_signed >= 0 and iw_signed < @as(isize, @intCast(w_in)))
                                {
                                    const ih: usize = @intCast(ih_signed);
                                    const iw: usize = @intCast(iw_signed);
                                    sum += in_data[n * c_in * h_in * w_in + ic * h_in * w_in + ih * w_in + iw] *
                                        k_data[oc * c_in * kh * kw + ic * kh * kw + fh * kw + fw];
                                }
                            }
                        }
                    }
                    out_data[n * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow] = sum;
                }
            }
        }
    }
}

/// Raw im2col+GEMM for one batch. Caller provides pre-allocated buffers.
pub fn im2colGemmBatch(
    in_batch: [*]const f32,
    k_data: [*]const f32,
    out_batch: [*]f32,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    col_buf: [*]f32,
    packed_a: [*]f32,
    packed_b: [*]f32,
) void {
    const gemm_m = c_out;
    const gemm_k = c_in * kh * kw;
    const gemm_n = h_out * w_out;

    im2col(col_buf, in_batch, c_in, h_in, w_in, kh, kw, stride_h, stride_w, pad_h, pad_w, h_out, w_out);
    @memset(out_batch[0 .. gemm_m * gemm_n], 0);
    matmul_mod.sgemmTiled(
        gemm_m, gemm_n, gemm_k,
        k_data, gemm_k, false,
        col_buf, gemm_n, false,
        out_batch, gemm_n,
        packed_a, packed_b,
    );
}
