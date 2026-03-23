const std = @import("std");
const testing = std.testing;

const synapse = @import("synapse");
const Storage = synapse.tensor.storage.Storage;
const shape_mod = synapse.tensor.shape;
const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;
const broadcastShapes = shape_mod.broadcastShapes;
const isCompatible = shape_mod.isCompatible;
const Tensor = synapse.tensor.core.Tensor;
const view = synapse.tensor.view;
const Range = view.Range;
const TensorIterator = synapse.tensor.iterator.TensorIterator;

// ── Storage tests ──────────────────────────────────────────────────────

test "storage: create has ref count 1" {
    const storage = try Storage.create(testing.allocator, f32, 16);
    defer storage.release();
    try testing.expectEqual(@as(u32, 1), storage.refCount());
}

test "storage: retain increments ref count" {
    const storage = try Storage.create(testing.allocator, f32, 16);
    const s2 = storage.retain();
    try testing.expectEqual(@as(u32, 2), storage.refCount());
    s2.release();
    storage.release();
}

test "storage: release frees when ref count reaches 0" {
    // testing.allocator is a tracking allocator — it will fail
    // the test if any allocation is leaked.
    const storage = try Storage.create(testing.allocator, f32, 8);
    storage.release();
}

test "storage: multiple retain/release cycle" {
    const storage = try Storage.create(testing.allocator, u8, 32);
    _ = storage.retain();
    _ = storage.retain();
    try testing.expectEqual(@as(u32, 3), storage.refCount());
    storage.release();
    try testing.expectEqual(@as(u32, 2), storage.refCount());
    storage.release();
    try testing.expectEqual(@as(u32, 1), storage.refCount());
    storage.release();
}

test "storage: dataAs returns correct typed slice" {
    const storage = try Storage.create(testing.allocator, f32, 4);
    defer storage.release();
    const data = storage.dataAs(f32);
    try testing.expectEqual(@as(usize, 4), data.len);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;
    try testing.expectEqual(@as(f32, 1.0), data[0]);
    try testing.expectEqual(@as(f32, 4.0), data[3]);
}

test "storage: 64-byte alignment" {
    const storage = try Storage.create(testing.allocator, f32, 16);
    defer storage.release();
    const addr = @intFromPtr(storage.data.ptr);
    try testing.expectEqual(@as(usize, 0), addr % 64);
}

test "storage: byteLen matches expected size" {
    const storage = try Storage.create(testing.allocator, f64, 10);
    defer storage.release();
    try testing.expectEqual(@as(usize, 80), storage.byteLen()); // 10 * 8
}

test "storage: dataAs with u8" {
    const storage = try Storage.create(testing.allocator, u8, 64);
    defer storage.release();
    const data = storage.dataAs(u8);
    try testing.expectEqual(@as(usize, 64), data.len);
    // Zero-initialized
    for (data) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
}

// ── Shape: numel & strides ─────────────────────────────────────────────

test "shape: numel 3D" {
    const s = Shape.init(&.{ 2, 3, 4 });
    try testing.expectEqual(@as(usize, 24), s.numel());
}

test "shape: numel scalar (ndim=0)" {
    const s = Shape.init(&.{});
    try testing.expectEqual(@as(usize, 1), s.numel());
}

test "shape: numel 1D" {
    const s = Shape.init(&.{7});
    try testing.expectEqual(@as(usize, 7), s.numel());
}

test "shape: contiguous strides 1D" {
    const s = Shape.init(&.{5});
    const strides = s.contiguousStrides();
    try testing.expectEqual(@as(usize, 1), strides[0]);
}

test "shape: contiguous strides 3D" {
    const s = Shape.init(&.{ 2, 3, 4 });
    const strides = s.contiguousStrides();
    try testing.expectEqual(@as(usize, 12), strides[0]);
    try testing.expectEqual(@as(usize, 4), strides[1]);
    try testing.expectEqual(@as(usize, 1), strides[2]);
}

test "shape: contiguous strides 4D" {
    const s = Shape.init(&.{ 2, 3, 4, 5 });
    const strides = s.contiguousStrides();
    try testing.expectEqual(@as(usize, 60), strides[0]); // 3*4*5
    try testing.expectEqual(@as(usize, 20), strides[1]); // 4*5
    try testing.expectEqual(@as(usize, 5), strides[2]); // 5
    try testing.expectEqual(@as(usize, 1), strides[3]);
}

test "shape: equality" {
    const a = Shape.init(&.{ 2, 3, 4 });
    const b = Shape.init(&.{ 2, 3, 4 });
    const c = Shape.init(&.{ 2, 3, 5 });
    try testing.expect(a.eql(b));
    try testing.expect(!a.eql(c));
}

// ── Broadcast tests (NumPy-compatible, 12 valid + 2 invalid) ───────────

test "broadcast: [3,1] + [1,4] = [3,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 3, 1 }), Shape.init(&.{ 1, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 3, 4 })));
}

test "broadcast: [5,3,1] + [1,4] = [5,3,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 5, 3, 1 }), Shape.init(&.{ 1, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 5, 3, 4 })));
}

test "broadcast: [1] + [4] = [4]" {
    const result = try broadcastShapes(Shape.init(&.{1}), Shape.init(&.{4}));
    try testing.expect(result.eql(Shape.init(&.{4})));
}

test "broadcast: [4] + [4] = [4]" {
    const result = try broadcastShapes(Shape.init(&.{4}), Shape.init(&.{4}));
    try testing.expect(result.eql(Shape.init(&.{4})));
}

test "broadcast: [3,4] + [4] = [3,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 3, 4 }), Shape.init(&.{4}));
    try testing.expect(result.eql(Shape.init(&.{ 3, 4 })));
}

test "broadcast: [2,1,3] + [1,5,1] = [2,5,3]" {
    const result = try broadcastShapes(Shape.init(&.{ 2, 1, 3 }), Shape.init(&.{ 1, 5, 1 }));
    try testing.expect(result.eql(Shape.init(&.{ 2, 5, 3 })));
}

test "broadcast: [1,1] + [5,4] = [5,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 1, 1 }), Shape.init(&.{ 5, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 5, 4 })));
}

test "broadcast: [6,1,3] + [5,3] = [6,5,3]" {
    const result = try broadcastShapes(Shape.init(&.{ 6, 1, 3 }), Shape.init(&.{ 5, 3 }));
    try testing.expect(result.eql(Shape.init(&.{ 6, 5, 3 })));
}

test "broadcast: [2,3,4] + [3,4] = [2,3,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 2, 3, 4 }), Shape.init(&.{ 3, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 2, 3, 4 })));
}

test "broadcast: [1,3,1] + [2,1,4] = [2,3,4]" {
    const result = try broadcastShapes(Shape.init(&.{ 1, 3, 1 }), Shape.init(&.{ 2, 1, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 2, 3, 4 })));
}

test "broadcast: scalar + [3,4] = [3,4]" {
    const result = try broadcastShapes(Shape.init(&.{}), Shape.init(&.{ 3, 4 }));
    try testing.expect(result.eql(Shape.init(&.{ 3, 4 })));
}

test "broadcast: [4,1,6] + [1,6] = [4,1,6]" {
    const result = try broadcastShapes(Shape.init(&.{ 4, 1, 6 }), Shape.init(&.{ 1, 6 }));
    try testing.expect(result.eql(Shape.init(&.{ 4, 1, 6 })));
}

// ── Invalid broadcast detection ────────────────────────────────────────

test "broadcast: incompatible [3] + [4] fails" {
    try testing.expectError(error.IncompatibleShapes, broadcastShapes(Shape.init(&.{3}), Shape.init(&.{4})));
}

test "broadcast: incompatible [2,1] + [3,4] fails" {
    // dim 0 from right: 1 vs 4 → 4 (broadcast)
    // dim 1 from right: 2 vs 3 → error
    try testing.expectError(error.IncompatibleShapes, broadcastShapes(Shape.init(&.{ 2, 1 }), Shape.init(&.{ 3, 4 })));
}

test "broadcast: isCompatible helper" {
    try testing.expect(isCompatible(Shape.init(&.{ 3, 1 }), Shape.init(&.{ 1, 4 })));
    try testing.expect(!isCompatible(Shape.init(&.{3}), Shape.init(&.{4})));
}

// ── Tensor: creation and element access ────────────────────────────────

test "tensor: create and numel" {
    const storage = try Storage.create(testing.allocator, f32, 24);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3, 4 }));
    defer t.release();
    try testing.expectEqual(@as(usize, 24), t.numel());
}

test "tensor: at and set" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    // Write values: row-major layout
    // [[10, 20, 30], [40, 50, 60]]
    t.set(&.{ 0, 0 }, 10.0);
    t.set(&.{ 0, 1 }, 20.0);
    t.set(&.{ 0, 2 }, 30.0);
    t.set(&.{ 1, 0 }, 40.0);
    t.set(&.{ 1, 1 }, 50.0);
    t.set(&.{ 1, 2 }, 60.0);

    try testing.expectEqual(@as(f32, 10.0), t.at(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 30.0), t.at(&.{ 0, 2 }));
    try testing.expectEqual(@as(f32, 40.0), t.at(&.{ 1, 0 }));
    try testing.expectEqual(@as(f32, 60.0), t.at(&.{ 1, 2 }));
}

test "tensor: isContiguous for fresh tensor" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t.release();
    try testing.expect(t.isContiguous());
}

test "tensor: strides match row-major" {
    const storage = try Storage.create(testing.allocator, f32, 24);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3, 4 }));
    defer t.release();
    try testing.expectEqual(@as(usize, 12), t.strides[0]);
    try testing.expectEqual(@as(usize, 4), t.strides[1]);
    try testing.expectEqual(@as(usize, 1), t.strides[2]);
}

test "tensor: shared storage ref counting" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release(); // ref from create

    const t1 = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t1.release();
    try testing.expectEqual(@as(u32, 2), storage.refCount());

    const t2 = Tensor(f32).init(storage, Shape.init(&.{ 4, 3 }));
    defer t2.release();
    try testing.expectEqual(@as(u32, 3), storage.refCount());
}

// ── View: reshape ──────────────────────────────────────────────────────

test "view: reshape preserves data and shares storage" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    t.set(&.{ 0, 0 }, 1.0);
    t.set(&.{ 0, 1 }, 2.0);
    t.set(&.{ 0, 2 }, 3.0);
    t.set(&.{ 1, 0 }, 4.0);
    t.set(&.{ 1, 1 }, 5.0);
    t.set(&.{ 1, 2 }, 6.0);

    const reshaped = try view.reshape(f32, t, Shape.init(&.{ 3, 2 }));
    defer reshaped.release();

    // Same storage pointer (zero-copy)
    try testing.expectEqual(t.storagePtr(), reshaped.storagePtr());

    // Data accessible through new shape
    try testing.expectEqual(@as(f32, 1.0), reshaped.at(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 2.0), reshaped.at(&.{ 0, 1 }));
    try testing.expectEqual(@as(f32, 3.0), reshaped.at(&.{ 1, 0 }));
    try testing.expectEqual(@as(f32, 6.0), reshaped.at(&.{ 2, 1 }));
}

test "view: reshape to 1D" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t.release();

    const flat = try view.reshape(f32, t, Shape.init(&.{12}));
    defer flat.release();
    try testing.expectEqual(@as(usize, 12), flat.numel());
    try testing.expect(flat.isContiguous());
}

test "view: reshape fails on numel mismatch" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();
    try testing.expectError(error.ShapeMismatch, view.reshape(f32, t, Shape.init(&.{ 2, 4 })));
}

// ── View: transpose ────────────────────────────────────────────────────

test "view: transpose produces correct strides" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();
    // Original strides: [3, 1]

    const tr = try view.transpose(f32, t, 0, 1);
    defer tr.release();

    // Shape swapped: [3, 2]
    try testing.expectEqual(@as(usize, 3), tr.shape.dims[0]);
    try testing.expectEqual(@as(usize, 2), tr.shape.dims[1]);

    // Strides swapped: [1, 3]
    try testing.expectEqual(@as(usize, 1), tr.strides[0]);
    try testing.expectEqual(@as(usize, 3), tr.strides[1]);

    // Same storage (zero-copy)
    try testing.expectEqual(t.storagePtr(), tr.storagePtr());
}

test "view: transpose preserves data access" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    // [[0, 1, 2], [3, 4, 5]]
    for (0..6) |i| {
        const row = i / 3;
        const col = i % 3;
        t.set(&.{ row, col }, @floatFromInt(i));
    }

    const tr = try view.transpose(f32, t, 0, 1);
    defer tr.release();

    // Transposed: [[0, 3], [1, 4], [2, 5]]
    try testing.expectEqual(@as(f32, 0.0), tr.at(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 3.0), tr.at(&.{ 0, 1 }));
    try testing.expectEqual(@as(f32, 1.0), tr.at(&.{ 1, 0 }));
    try testing.expectEqual(@as(f32, 4.0), tr.at(&.{ 1, 1 }));
    try testing.expectEqual(@as(f32, 2.0), tr.at(&.{ 2, 0 }));
    try testing.expectEqual(@as(f32, 5.0), tr.at(&.{ 2, 1 }));
}

test "view: transposed tensor is not contiguous" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    const tr = try view.transpose(f32, t, 0, 1);
    defer tr.release();
    try testing.expect(!tr.isContiguous());
}

test "view: transpose invalid axis" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();
    try testing.expectError(error.InvalidAxis, view.transpose(f32, t, 0, 2));
}

// ── View: slice ────────────────────────────────────────────────────────

test "view: slice produces correct sub-view" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t.release();

    // Fill: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    for (0..12) |i| {
        const row = i / 4;
        const col = i % 4;
        t.set(&.{ row, col }, @floatFromInt(i));
    }

    // Slice [1:3, 1:3] → [[5,6],[9,10]]
    const sl = try view.slice(f32, t, &.{
        Range{ .start = 1, .end = 3 },
        Range{ .start = 1, .end = 3 },
    });
    defer sl.release();

    // Same storage (zero-copy)
    try testing.expectEqual(t.storagePtr(), sl.storagePtr());

    // Shape is [2, 2]
    try testing.expectEqual(@as(usize, 2), sl.shape.dims[0]);
    try testing.expectEqual(@as(usize, 2), sl.shape.dims[1]);

    // Strides unchanged from original [4, 1]
    try testing.expectEqual(@as(usize, 4), sl.strides[0]);
    try testing.expectEqual(@as(usize, 1), sl.strides[1]);

    // Offset = 0 + 1*4 + 1*1 = 5
    try testing.expectEqual(@as(usize, 5), sl.offset);

    // Values
    try testing.expectEqual(@as(f32, 5.0), sl.at(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 6.0), sl.at(&.{ 0, 1 }));
    try testing.expectEqual(@as(f32, 9.0), sl.at(&.{ 1, 0 }));
    try testing.expectEqual(@as(f32, 10.0), sl.at(&.{ 1, 1 }));
}

test "view: slice single row" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t.release();

    for (0..12) |i| {
        const row = i / 4;
        const col = i % 4;
        t.set(&.{ row, col }, @floatFromInt(i));
    }

    // Slice row 2: [2:3, 0:4] → [[8,9,10,11]]
    const sl = try view.slice(f32, t, &.{
        Range{ .start = 2, .end = 3 },
        Range{ .start = 0, .end = 4 },
    });
    defer sl.release();

    try testing.expectEqual(@as(usize, 4), sl.numel());
    try testing.expectEqual(@as(f32, 8.0), sl.at(&.{ 0, 0 }));
    try testing.expectEqual(@as(f32, 11.0), sl.at(&.{ 0, 3 }));
}

test "view: slice out of bounds" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();
    try testing.expectError(error.IndexOutOfBounds, view.slice(f32, t, &.{
        Range{ .start = 0, .end = 2 },
        Range{ .start = 1, .end = 4 }, // end > dim
    }));
}

// ── Iterator: contiguous ───────────────────────────────────────────────

test "iterator: visits all elements contiguous in order" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    for (0..6) |i| {
        const row = i / 3;
        const col = i % 3;
        t.set(&.{ row, col }, @floatFromInt(i));
    }

    var iter = TensorIterator(f32).init(t);
    var i: usize = 0;
    while (iter.next()) |val| {
        try testing.expectEqual(@as(f32, @floatFromInt(i)), val);
        i += 1;
    }
    try testing.expectEqual(@as(usize, 6), i);
}

test "iterator: reset restarts iteration" {
    const storage = try Storage.create(testing.allocator, f32, 4);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{4}));
    defer t.release();

    for (0..4) |i| {
        t.set(&.{i}, @floatFromInt(i));
    }

    var iter = TensorIterator(f32).init(t);

    // Exhaust
    while (iter.next()) |_| {}
    try testing.expect(iter.next() == null);

    // Reset and iterate again
    iter.reset();
    var sum: f32 = 0;
    while (iter.next()) |v| {
        sum += v;
    }
    try testing.expectEqual(@as(f32, 6.0), sum); // 0+1+2+3
}

// ── Iterator: non-contiguous (transposed) ──────────────────────────────

test "iterator: transposed tensor visits correct order" {
    const storage = try Storage.create(testing.allocator, f32, 6);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 2, 3 }));
    defer t.release();

    // [[0, 1, 2], [3, 4, 5]]
    for (0..6) |i| {
        t.set(&.{ i / 3, i % 3 }, @floatFromInt(i));
    }

    const tr = try view.transpose(f32, t, 0, 1);
    defer tr.release();

    // Transposed shape [3, 2], row-major iteration:
    // [0,0]=0, [0,1]=3, [1,0]=1, [1,1]=4, [2,0]=2, [2,1]=5
    const expected = [_]f32{ 0, 3, 1, 4, 2, 5 };
    var iter = TensorIterator(f32).init(tr);
    var i: usize = 0;
    while (iter.next()) |val| {
        try testing.expectEqual(expected[i], val);
        i += 1;
    }
    try testing.expectEqual(@as(usize, 6), i);
}

// ── Iterator: sliced tensor ────────────────────────────────────────────

test "iterator: sliced tensor visits correct elements" {
    const storage = try Storage.create(testing.allocator, f32, 12);
    defer storage.release();
    const t = Tensor(f32).init(storage, Shape.init(&.{ 3, 4 }));
    defer t.release();

    for (0..12) |i| {
        t.set(&.{ i / 4, i % 4 }, @floatFromInt(i));
    }

    // Slice [1:3, 1:3] → [[5,6],[9,10]]
    const sl = try view.slice(f32, t, &.{
        Range{ .start = 1, .end = 3 },
        Range{ .start = 1, .end = 3 },
    });
    defer sl.release();

    const expected = [_]f32{ 5, 6, 9, 10 };
    var iter = TensorIterator(f32).init(sl);
    var i: usize = 0;
    while (iter.next()) |val| {
        try testing.expectEqual(expected[i], val);
        i += 1;
    }
    try testing.expectEqual(@as(usize, 4), i);
}

// ── Benchmark: contiguous vs strided iterator on 1M elements ───────────

test "bench: iterator throughput contiguous vs strided 1M" {
    const N: usize = 1_000_000;
    const storage = try Storage.create(testing.allocator, f32, N);
    defer storage.release();

    // Fill with data
    const raw_data = storage.dataAs(f32);
    for (raw_data, 0..) |*d, i| {
        d.* = @floatFromInt(i % 1000);
    }

    // ── Raw pointer walk baseline ──
    var raw_timer = try std.time.Timer.start();
    var sum_raw: f32 = 0;
    for (raw_data) |v| {
        sum_raw += v;
    }
    const raw_ns = raw_timer.read();

    // ── Contiguous iterator ──
    const t_contig = Tensor(f32).init(storage, Shape.init(&.{N}));
    defer t_contig.release();

    var contig_timer = try std.time.Timer.start();
    var iter_c = TensorIterator(f32).init(t_contig);
    var sum_c: f32 = 0;
    while (iter_c.next()) |v| {
        sum_c += v;
    }
    const contig_ns = contig_timer.read();

    // ── Strided (transposed) iterator ──
    const t_2d = Tensor(f32).init(storage, Shape.init(&.{ 1000, 1000 }));
    defer t_2d.release();
    const t_strided = try view.transpose(f32, t_2d, 0, 1);
    defer t_strided.release();

    var strided_timer = try std.time.Timer.start();
    var iter_s = TensorIterator(f32).init(t_strided);
    var sum_s: f32 = 0;
    while (iter_s.next()) |v| {
        sum_s += v;
    }
    const strided_ns = strided_timer.read();

    // Print results
    const raw_ms = @as(f64, @floatFromInt(raw_ns)) / 1_000_000.0;
    const contig_ms = @as(f64, @floatFromInt(contig_ns)) / 1_000_000.0;
    const strided_ms = @as(f64, @floatFromInt(strided_ns)) / 1_000_000.0;
    const ratio = contig_ms / @max(raw_ms, 0.001);

    std.debug.print("\n  Raw pointer:  {d:.2} ms\n", .{raw_ms});
    std.debug.print("  Contiguous:   {d:.2} ms (ratio vs raw: {d:.2}x)\n", .{ contig_ms, ratio });
    std.debug.print("  Strided:      {d:.2} ms\n", .{strided_ms});

    // Sanity check: all sums should be positive and equal
    try testing.expect(sum_raw > 0);
    try testing.expectEqual(sum_raw, sum_c);
    try testing.expect(sum_s > 0);
}
