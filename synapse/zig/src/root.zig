pub const tensor = struct {
    pub const storage = @import("tensor/storage.zig");
    pub const shape = @import("tensor/shape.zig");
    pub const core = @import("tensor/tensor.zig");
    pub const view = @import("tensor/view.zig");
    pub const iterator = @import("tensor/iterator.zig");
};

pub const ops = struct {
    pub const reduce = @import("ops/reduce.zig");
    pub const softmax = @import("ops/softmax.zig");
    pub const batchnorm = @import("ops/batchnorm.zig");
    pub const layernorm = @import("ops/layernorm.zig");
    pub const matmul = @import("ops/matmul.zig");
    pub const conv = @import("ops/conv.zig");
    pub const pool = @import("ops/pool.zig");
    pub const transpose = @import("ops/transpose.zig");
    pub const rope = @import("ops/rope.zig");
    pub const attention = @import("ops/attention.zig");
};
