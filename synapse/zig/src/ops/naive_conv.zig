//! Naive 4-loop convolution extracted for separate compilation at Debug optimization.
//! Used as benchmark baseline for im2col+GEMM comparison.

comptime {
    if (@import("builtin").mode != .Debug) @compileError("naive_conv must be compiled at Debug optimization");
}

/// Raw naive conv2d on flat NCHW arrays. Deliberately unoptimized scalar loop.
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
    @setFloatMode(.strict);
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
                    // Compiler barrier prevents reordering across loop iterations
                    asm volatile ("" ::: .{ .memory = true });
                }
            }
        }
    }
}
