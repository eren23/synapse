const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library
    const lib_mod = b.addModule("synapse", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "synapse",
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    // Allocator modules
    const arena_mod = b.addModule("arena", .{
        .root_source_file = b.path("src/alloc/arena.zig"),
        .target = target,
        .optimize = optimize,
    });

    const pool_mod = b.addModule("pool", .{
        .root_source_file = b.path("src/alloc/pool.zig"),
        .target = target,
        .optimize = optimize,
    });

    const aligned_mod = b.addModule("aligned", .{
        .root_source_file = b.path("src/alloc/aligned.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tracking_mod = b.addModule("tracking", .{
        .root_source_file = b.path("src/alloc/tracking.zig"),
        .target = target,
        .optimize = optimize,
    });

    // SIMD modules
    const avx2_mod = b.addModule("avx2", .{
        .root_source_file = b.path("src/simd/avx2.zig"),
        .target = target,
        .optimize = optimize,
    });

    const neon_mod = b.addModule("neon", .{
        .root_source_file = b.path("src/simd/neon.zig"),
        .target = target,
        .optimize = optimize,
    });

    const dispatch_mod = b.addModule("dispatch", .{
        .root_source_file = b.path("src/simd/dispatch.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "avx2", .module = avx2_mod },
            .{ .name = "neon", .module = neon_mod },
        },
    });

    const vec_ops_mod = b.addModule("vec_ops", .{
        .root_source_file = b.path("src/simd/vec_ops.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "dispatch", .module = dispatch_mod },
        },
    });

    const reduce_mod = b.addModule("reduce", .{
        .root_source_file = b.path("src/simd/reduce.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Tensor shape module (standalone for ops)
    const shape_mod = b.addModule("shape", .{
        .root_source_file = b.path("src/tensor/shape.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Elementwise ops module
    const elementwise_mod = b.addModule("elementwise", .{
        .root_source_file = b.path("src/ops/elementwise.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "shape", .module = shape_mod },
        },
    });

    // ================================================================
    // FFI module & libsynapse_zig.a static library
    // Cross-compile: zig build -Dtarget=aarch64-linux-gnu
    //                zig build -Dtarget=x86_64-linux-gnu
    // ================================================================

    const ffi_mod = b.addModule("ffi", .{
        .root_source_file = b.path("src/ffi/exports.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "synapse", .module = lib_mod },
            .{ .name = "arena", .module = arena_mod },
            .{ .name = "pool", .module = pool_mod },
            .{ .name = "dispatch", .module = dispatch_mod },
        },
    });

    const ffi_lib = b.addLibrary(.{
        .name = "synapse_zig",
        .root_module = ffi_mod,
    });
    b.installArtifact(ffi_lib);

    // Tensor unit tests
    const tensor_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_tensor.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);

    // Allocator unit tests
    const alloc_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_alloc.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "arena", .module = arena_mod },
                .{ .name = "pool", .module = pool_mod },
                .{ .name = "aligned", .module = aligned_mod },
                .{ .name = "tracking", .module = tracking_mod },
            },
        }),
    });
    const run_alloc_tests = b.addRunArtifact(alloc_tests);

    // SIMD unit tests
    const simd_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_simd.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vec_ops", .module = vec_ops_mod },
                .{ .name = "dispatch", .module = dispatch_mod },
                .{ .name = "avx2", .module = avx2_mod },
                .{ .name = "neon", .module = neon_mod },
                .{ .name = "reduce", .module = reduce_mod },
            },
        }),
    });
    const run_simd_tests = b.addRunArtifact(simd_tests);

    // Elementwise unit tests
    const elementwise_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_elementwise.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "elementwise", .module = elementwise_mod },
                .{ .name = "shape", .module = shape_mod },
            },
        }),
    });
    const run_elementwise_tests = b.addRunArtifact(elementwise_tests);

    // Reduce ops unit tests
    const reduce_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_reduce.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_reduce_ops_tests = b.addRunArtifact(reduce_ops_tests);

    // Softmax ops unit tests
    const softmax_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_softmax.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_softmax_ops_tests = b.addRunArtifact(softmax_ops_tests);

    // Batchnorm ops unit tests
    const batchnorm_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_batchnorm.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_batchnorm_ops_tests = b.addRunArtifact(batchnorm_ops_tests);

    // LayerNorm ops unit tests
    const layernorm_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_layernorm.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_layernorm_ops_tests = b.addRunArtifact(layernorm_ops_tests);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_tensor_tests.step);
    test_step.dependOn(&run_alloc_tests.step);
    test_step.dependOn(&run_simd_tests.step);
    test_step.dependOn(&run_elementwise_tests.step);
    test_step.dependOn(&run_reduce_ops_tests.step);
    test_step.dependOn(&run_softmax_ops_tests.step);
    test_step.dependOn(&run_batchnorm_ops_tests.step);
    test_step.dependOn(&run_layernorm_ops_tests.step);

    // Allocator tests only (separate step)
    const alloc_test_step = b.step("test-alloc", "Run allocator unit tests only");
    alloc_test_step.dependOn(&run_alloc_tests.step);

    // SIMD tests only (separate step)
    const simd_test_step = b.step("test-simd", "Run SIMD unit tests only");
    simd_test_step.dependOn(&run_simd_tests.step);

    // Elementwise tests only (separate step)
    const elementwise_test_step = b.step("test-elementwise", "Run elementwise unit tests only");
    elementwise_test_step.dependOn(&run_elementwise_tests.step);

    // Ops tests only (separate steps)
    const reduce_ops_test_step = b.step("test-reduce", "Run reduce ops unit tests only");
    reduce_ops_test_step.dependOn(&run_reduce_ops_tests.step);

    const softmax_ops_test_step = b.step("test-softmax", "Run softmax ops unit tests only");
    softmax_ops_test_step.dependOn(&run_softmax_ops_tests.step);

    const batchnorm_ops_test_step = b.step("test-batchnorm", "Run batchnorm ops unit tests only");
    batchnorm_ops_test_step.dependOn(&run_batchnorm_ops_tests.step);

    const layernorm_ops_test_step = b.step("test-layernorm", "Run layernorm ops unit tests only");
    layernorm_ops_test_step.dependOn(&run_layernorm_ops_tests.step);

    // Matmul ops unit tests
    const matmul_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_matmul.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_matmul_ops_tests = b.addRunArtifact(matmul_ops_tests);
    test_step.dependOn(&run_matmul_ops_tests.step);

    const matmul_ops_test_step = b.step("test-matmul", "Run matmul ops unit tests only");
    matmul_ops_test_step.dependOn(&run_matmul_ops_tests.step);

    // Conv2d unit tests
    const conv_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_conv.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_conv_ops_tests = b.addRunArtifact(conv_ops_tests);
    test_step.dependOn(&run_conv_ops_tests.step);

    const conv_ops_test_step = b.step("test-conv", "Run conv2d ops unit tests only");
    conv_ops_test_step.dependOn(&run_conv_ops_tests.step);

    // Pool unit tests
    const pool_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_pool.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_pool_ops_tests = b.addRunArtifact(pool_ops_tests);
    test_step.dependOn(&run_pool_ops_tests.step);

    const pool_ops_test_step = b.step("test-pool", "Run pooling ops unit tests only");
    pool_ops_test_step.dependOn(&run_pool_ops_tests.step);

    // RoPE ops unit tests
    const rope_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_rope.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_rope_ops_tests = b.addRunArtifact(rope_ops_tests);
    test_step.dependOn(&run_rope_ops_tests.step);

    const rope_ops_test_step = b.step("test-rope", "Run RoPE ops unit tests only");
    rope_ops_test_step.dependOn(&run_rope_ops_tests.step);

    // Attention ops unit tests
    const attention_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_attention.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_attention_ops_tests = b.addRunArtifact(attention_ops_tests);
    test_step.dependOn(&run_attention_ops_tests.step);

    const attention_ops_test_step = b.step("test-attention", "Run attention ops unit tests only");
    attention_ops_test_step.dependOn(&run_attention_ops_tests.step);

    // RMSNorm ops unit tests
    const rmsnorm_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_rmsnorm.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_rmsnorm_ops_tests = b.addRunArtifact(rmsnorm_ops_tests);
    test_step.dependOn(&run_rmsnorm_ops_tests.step);

    const rmsnorm_ops_test_step = b.step("test-rmsnorm", "Run RMSNorm ops unit tests only");
    rmsnorm_ops_test_step.dependOn(&run_rmsnorm_ops_tests.step);

    // SiLU ops unit tests
    const silu_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_silu.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_silu_ops_tests = b.addRunArtifact(silu_ops_tests);
    test_step.dependOn(&run_silu_ops_tests.step);

    const silu_ops_test_step = b.step("test-silu", "Run SiLU ops unit tests only");
    silu_ops_test_step.dependOn(&run_silu_ops_tests.step);

    // Quantize ops unit tests
    const quantize_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_quantize.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_quantize_ops_tests = b.addRunArtifact(quantize_ops_tests);
    test_step.dependOn(&run_quantize_ops_tests.step);

    const quantize_ops_test_step = b.step("test-quantize", "Run quantize ops unit tests only");
    quantize_ops_test_step.dependOn(&run_quantize_ops_tests.step);

    // Quantized matmul ops unit tests
    const qmatmul_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_qmatmul.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_qmatmul_ops_tests = b.addRunArtifact(qmatmul_ops_tests);
    test_step.dependOn(&run_qmatmul_ops_tests.step);

    const qmatmul_ops_test_step = b.step("test-qmatmul", "Run quantized matmul ops unit tests only");
    qmatmul_ops_test_step.dependOn(&run_qmatmul_ops_tests.step);

    // KV-Cache module
    const kvcache_mod = b.addModule("kvcache", .{
        .root_source_file = b.path("src/ops/kvcache.zig"),
        .target = target,
        .optimize = optimize,
    });

    // KV-Cache unit tests
    const kvcache_ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_kvcache.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "kvcache", .module = kvcache_mod },
                .{ .name = "tracking", .module = tracking_mod },
            },
        }),
    });
    const run_kvcache_ops_tests = b.addRunArtifact(kvcache_ops_tests);
    test_step.dependOn(&run_kvcache_ops_tests.step);

    const kvcache_ops_test_step = b.step("test-kvcache", "Run KV-Cache ops unit tests only");
    kvcache_ops_test_step.dependOn(&run_kvcache_ops_tests.step);

    // Tensor unit tests (standalone)
    const tensor_test_step = b.step("test-tensor", "Run tensor unit tests only");
    tensor_test_step.dependOn(&run_tensor_tests.step);

    // FFI round-trip tests
    const ffi_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_ffi.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ffi", .module = ffi_mod },
            },
        }),
    });
    const run_ffi_tests = b.addRunArtifact(ffi_tests);
    test_step.dependOn(&run_ffi_tests.step);

    const ffi_test_step = b.step("test-ffi", "Run FFI round-trip tests only");
    ffi_test_step.dependOn(&run_ffi_tests.step);

    // Allocator benchmarks
    const bench = b.addExecutable(.{
        .name = "bench_alloc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_alloc.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "arena", .module = arena_mod },
                .{ .name = "pool", .module = pool_mod },
            },
        }),
    });
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run allocator benchmarks");
    bench_step.dependOn(&run_bench.step);

    // SIMD benchmarks — all modules compiled with ReleaseFast
    const bench_avx2_mod = b.addModule("bench_avx2", .{
        .root_source_file = b.path("src/simd/avx2.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_neon_mod = b.addModule("bench_neon", .{
        .root_source_file = b.path("src/simd/neon.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_dispatch_mod = b.addModule("bench_dispatch", .{
        .root_source_file = b.path("src/simd/dispatch.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "avx2", .module = bench_avx2_mod },
            .{ .name = "neon", .module = bench_neon_mod },
        },
    });
    const bench_simd = b.addExecutable(.{
        .name = "bench_simd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_simd.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "dispatch", .module = bench_dispatch_mod },
                .{ .name = "neon", .module = bench_neon_mod },
            },
        }),
    });
    b.installArtifact(bench_simd);

    const run_bench_simd = b.addRunArtifact(bench_simd);
    const bench_simd_step = b.step("bench-simd", "Run SIMD benchmarks");
    bench_simd_step.dependOn(&run_bench_simd.step);

    // Elementwise benchmarks — compiled with ReleaseFast
    const bench_shape_mod = b.createModule(.{
        .root_source_file = b.path("src/tensor/shape.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_elementwise_mod = b.createModule(.{
        .root_source_file = b.path("src/ops/elementwise.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "shape", .module = bench_shape_mod },
        },
    });
    const bench_elementwise = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_elementwise.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "elementwise", .module = bench_elementwise_mod },
                .{ .name = "shape", .module = bench_shape_mod },
            },
        }),
    });
    const run_bench_elementwise = b.addRunArtifact(bench_elementwise);
    const bench_elementwise_step = b.step("bench-elementwise", "Run elementwise benchmarks (ReleaseFast)");
    bench_elementwise_step.dependOn(&run_bench_elementwise.step);

    // Reduce benchmarks (SIMD reduce_sum + Welford vs two-pass)
    const bench_reduce = b.addExecutable(.{
        .name = "bench_reduce",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_reduce.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(bench_reduce);

    const run_bench_reduce = b.addRunArtifact(bench_reduce);
    const bench_reduce_step = b.step("bench-reduce", "Run reduce + batchnorm benchmarks");
    bench_reduce_step.dependOn(&run_bench_reduce.step);

    // Matmul benchmarks — compiled with ReleaseFast
    const bench_synapse_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_matmul = b.addExecutable(.{
        .name = "bench_matmul",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_matmul.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    b.installArtifact(bench_matmul);

    const run_bench_matmul = b.addRunArtifact(bench_matmul);
    const bench_matmul_step = b.step("bench-matmul", "Run SGEMM benchmarks");
    bench_matmul_step.dependOn(&run_bench_matmul.step);

    // Conv2d benchmarks — compiled with ReleaseFast
    // Naive conv compiled at Debug to prevent auto-vectorization (true scalar baseline)
    const naive_conv_mod = b.createModule(.{
        .root_source_file = b.path("src/ops/naive_conv.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const bench_conv = b.addExecutable(.{
        .name = "bench_conv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_conv.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
                .{ .name = "naive_conv", .module = naive_conv_mod },
            },
        }),
    });
    b.installArtifact(bench_conv);

    const run_bench_conv = b.addRunArtifact(bench_conv);
    const bench_conv_step = b.step("bench-conv", "Run Conv2d benchmarks");
    bench_conv_step.dependOn(&run_bench_conv.step);

    // LayerNorm benchmarks — compiled with ReleaseFast
    const bench_layernorm = b.addExecutable(.{
        .name = "bench_layernorm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_layernorm.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(bench_layernorm);

    const run_bench_layernorm = b.addRunArtifact(bench_layernorm);
    const bench_layernorm_step = b.step("bench-layernorm", "Run LayerNorm benchmarks");
    bench_layernorm_step.dependOn(&run_bench_layernorm.step);

    // RoPE benchmarks — compiled with ReleaseFast
    const bench_rope = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_rope.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    const run_bench_rope = b.addRunArtifact(bench_rope);
    const bench_rope_step = b.step("bench-rope", "Run RoPE benchmarks (ReleaseFast)");
    bench_rope_step.dependOn(&run_bench_rope.step);

    // Attention benchmarks — compiled with ReleaseFast
    const bench_attention = b.addExecutable(.{
        .name = "bench_attention",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_attention.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    b.installArtifact(bench_attention);

    const run_bench_attention = b.addRunArtifact(bench_attention);
    const bench_attention_step = b.step("bench-attention", "Run attention benchmarks");
    bench_attention_step.dependOn(&run_bench_attention.step);

    // RMSNorm benchmarks — compiled with ReleaseFast
    const bench_rmsnorm = b.addExecutable(.{
        .name = "bench_rmsnorm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_rmsnorm.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(bench_rmsnorm);

    const run_bench_rmsnorm = b.addRunArtifact(bench_rmsnorm);
    const bench_rmsnorm_step = b.step("bench-rmsnorm", "Run RMSNorm benchmarks");
    bench_rmsnorm_step.dependOn(&run_bench_rmsnorm.step);

    // SiLU/SwiGLU benchmarks — compiled with ReleaseFast
    const bench_silu = b.addExecutable(.{
        .name = "bench_silu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_silu.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(bench_silu);

    const run_bench_silu = b.addRunArtifact(bench_silu);
    const bench_silu_step = b.step("bench-silu", "Run SiLU/SwiGLU benchmarks");
    bench_silu_step.dependOn(&run_bench_silu.step);

    // INT8 Quantized GEMM benchmarks — compiled with ReleaseFast
    const bench_qmatmul = b.addExecutable(.{
        .name = "bench_qmatmul",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_qmatmul.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    b.installArtifact(bench_qmatmul);

    const run_bench_qmatmul = b.addRunArtifact(bench_qmatmul);
    const bench_qmatmul_step = b.step("bench-qmatmul", "Run INT8 quantized GEMM benchmarks");
    bench_qmatmul_step.dependOn(&run_bench_qmatmul.step);

    // KV-Cache benchmarks — compiled with ReleaseFast
    const bench_kvcache = b.addExecutable(.{
        .name = "bench_kvcache",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_kvcache.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    b.installArtifact(bench_kvcache);

    const run_bench_kvcache = b.addRunArtifact(bench_kvcache);
    const bench_kvcache_step = b.step("bench-kvcache", "Run KV-Cache benchmarks");
    bench_kvcache_step.dependOn(&run_bench_kvcache.step);

    // Fused LEWM layer unit tests (inline tests in fused_lewm_layer.zig)
    const fused_lewm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_fused_lewm.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "synapse", .module = lib_mod },
            },
        }),
    });
    const run_fused_lewm_tests = b.addRunArtifact(fused_lewm_tests);
    test_step.dependOn(&run_fused_lewm_tests.step);

    const fused_lewm_test_step = b.step("test-fused-lewm", "Run fused LEWM layer unit tests only");
    fused_lewm_test_step.dependOn(&run_fused_lewm_tests.step);

    // Fused LEWM layer benchmarks — standard vs ESP-fused, compiled with ReleaseFast
    const bench_fused_lewm = b.addExecutable(.{
        .name = "bench_fused_lewm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/bench_fused_lewm.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "synapse", .module = bench_synapse_mod },
            },
        }),
    });
    b.installArtifact(bench_fused_lewm);

    const run_bench_fused_lewm = b.addRunArtifact(bench_fused_lewm);
    const bench_fused_lewm_step = b.step("bench-fused-lewm", "Run fused LEWM layer benchmarks (standard vs ESP-fused)");
    bench_fused_lewm_step.dependOn(&run_bench_fused_lewm.step);
}
