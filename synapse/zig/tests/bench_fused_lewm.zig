//! Benchmark: Standard vs ESP-fused LEWM predictor layer.
//!
//! Compares the standard separate-loop approach (mode=0) against
//! ESP-style single-pass fused bias+GELU and bias+gated_residual (mode=1).
//!
//! Tests both single-step (seq_len=3) and 50-step rollout (seq_len=150).

const std = @import("std");
const fused = @import("synapse").ops.fused_lewm_layer;
const rollout = @import("synapse").ops.fused_lewm_rollout;

// PushT config: 192d hidden, 1024 inner, 16 heads, 2048 inter
const HIDDEN: usize = 192;
const NUM_HEADS: usize = 16;
const INNER_DIM: usize = 1024;
const HEAD_DIM: usize = INNER_DIM / NUM_HEADS;
const INTER: usize = 2048;
const LAYERS: usize = 6;

const WARMUP: usize = 5;
const SINGLE_ITERS: usize = 200;
const ROLLOUT_STEPS: usize = 50;
const ROLLOUT_SEQ_LEN: usize = ROLLOUT_STEPS * 3;
const ROLLOUT_ITERS: usize = 10;

fn fillDeterministic(buf: []f32, seed: u32) void {
    var s = seed;
    for (buf) |*v| {
        s = s *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt(@rem(@as(i32, @bitCast(s >> 16)), 1000))) * 0.001;
    }
}

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    print("\n", .{});
    print("=================================================================\n", .{});
    print("  LEWM Fused Kernel Benchmark: Standard vs ESP-Fused\n", .{});
    print("  Config: hidden={d}, inner={d}, heads={d}, inter={d}, layers={d}\n", .{ HIDDEN, INNER_DIM, NUM_HEADS, INTER, LAYERS });
    print("=================================================================\n", .{});

    // -- Allocate weights for one layer (reused for all 6 layers in bench) --
    const adaln_w = try allocator.alloc(f32, 6 * HIDDEN * HIDDEN);
    const adaln_b = try allocator.alloc(f32, 6 * HIDDEN);
    const attn_norm_w = try allocator.alloc(f32, HIDDEN);
    const to_qkv = try allocator.alloc(f32, 3 * INNER_DIM * HIDDEN);
    const attn_out_w = try allocator.alloc(f32, HIDDEN * INNER_DIM);
    const attn_out_b = try allocator.alloc(f32, HIDDEN);
    const mlp_norm_w = try allocator.alloc(f32, HIDDEN);
    const mlp_up_w = try allocator.alloc(f32, INTER * HIDDEN);
    const mlp_up_b = try allocator.alloc(f32, INTER);
    const mlp_down_w = try allocator.alloc(f32, HIDDEN * INTER);
    const mlp_down_b = try allocator.alloc(f32, HIDDEN);

    fillDeterministic(adaln_w, 10);
    fillDeterministic(adaln_b, 11);
    for (attn_norm_w) |*v| v.* = 1.0;
    fillDeterministic(to_qkv, 12);
    fillDeterministic(attn_out_w, 13);
    fillDeterministic(attn_out_b, 14);
    for (mlp_norm_w) |*v| v.* = 1.0;
    fillDeterministic(mlp_up_w, 15);
    fillDeterministic(mlp_up_b, 16);
    fillDeterministic(mlp_down_w, 17);
    fillDeterministic(mlp_down_b, 18);

    // -- Scratch buffers (sized for rollout = max) --
    const max_seq = ROLLOUT_SEQ_LEN;
    const scratch_dim = @max(max_seq * HIDDEN, max_seq * INTER);

    const mod_buf = try allocator.alloc(f32, 6 * HIDDEN);
    const normed_buf = try allocator.alloc(f32, scratch_dim);
    const qkv_buf = try allocator.alloc(f32, max_seq * 3 * INNER_DIM);
    const attn_buf = try allocator.alloc(f32, max_seq * INNER_DIM);
    const proj_buf = try allocator.alloc(f32, scratch_dim);

    // Sequence buffers (one for each mode, so we can compare)
    const seq_std = try allocator.alloc(f32, max_seq * HIDDEN);
    const seq_esp = try allocator.alloc(f32, max_seq * HIDDEN);
    const conditioning = try allocator.alloc(f32, HIDDEN);
    fillDeterministic(conditioning, 99);

    // ================================================================
    // 1. Correctness check: both modes must produce identical output
    // ================================================================
    print("\n--- Correctness Check (single step, seq_len=3) ---\n", .{});
    const single_seq = 3;
    fillDeterministic(seq_std[0 .. single_seq * HIDDEN], 42);
    @memcpy(seq_esp[0 .. single_seq * HIDDEN], seq_std[0 .. single_seq * HIDDEN]);

    // Run standard (mode=0)
    for (0..LAYERS) |_| {
        fused.lewmPredictorLayerV2(
            seq_std.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
            adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
            mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 0,
        );
    }

    // Run ESP-fused (mode=1)
    for (0..LAYERS) |_| {
        fused.lewmPredictorLayerV2(
            seq_esp.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
            adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
            mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 1,
        );
    }

    var max_diff: f32 = 0;
    var sum_sq_diff: f64 = 0;
    var sum_sq_ref: f64 = 0;
    for (0..single_seq * HIDDEN) |i| {
        const d = seq_std[i] - seq_esp[i];
        const abs_d = @abs(d);
        if (abs_d > max_diff) max_diff = abs_d;
        sum_sq_diff += @as(f64, d) * @as(f64, d);
        sum_sq_ref += @as(f64, seq_std[i]) * @as(f64, seq_std[i]);
    }
    const rel_err = if (sum_sq_ref > 0) @sqrt(sum_sq_diff / sum_sq_ref) else 0.0;

    print("  max_abs_diff = {e:.6}\n", .{max_diff});
    print("  relative_err = {e:.6}\n", .{rel_err});
    if (max_diff > 1e-4) {
        print("  FAIL: outputs diverge!\n", .{});
        return;
    }
    print("  PASS: outputs match within 1e-4\n", .{});

    // ================================================================
    // 2. Single-step benchmark (seq_len=3, 6 layers)
    // ================================================================
    print("\n--- Single-Step Benchmark (seq_len=3, {d} layers, {d} iters) ---\n", .{ LAYERS, SINGLE_ITERS });

    // Warmup
    for (0..WARMUP) |_| {
        fillDeterministic(seq_std[0 .. single_seq * HIDDEN], 42);
        for (0..LAYERS) |_| {
            fused.lewmPredictorLayerV2(
                seq_std.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 0,
            );
        }
    }

    // Standard mode
    var timer = try std.time.Timer.start();
    for (0..SINGLE_ITERS) |_| {
        fillDeterministic(seq_std[0 .. single_seq * HIDDEN], 42);
        for (0..LAYERS) |_| {
            fused.lewmPredictorLayerV2(
                seq_std.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 0,
            );
        }
    }
    const std_single_ns = timer.read();
    const std_single_ms = @as(f64, @floatFromInt(std_single_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(SINGLE_ITERS));

    // Warmup ESP
    for (0..WARMUP) |_| {
        fillDeterministic(seq_esp[0 .. single_seq * HIDDEN], 42);
        for (0..LAYERS) |_| {
            fused.lewmPredictorLayerV2(
                seq_esp.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 1,
            );
        }
    }

    // ESP-fused mode
    timer = try std.time.Timer.start();
    for (0..SINGLE_ITERS) |_| {
        fillDeterministic(seq_esp[0 .. single_seq * HIDDEN], 42);
        for (0..LAYERS) |_| {
            fused.lewmPredictorLayerV2(
                seq_esp.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 1,
            );
        }
    }
    const esp_single_ns = timer.read();
    const esp_single_ms = @as(f64, @floatFromInt(esp_single_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(SINGLE_ITERS));

    const single_speedup = std_single_ms / esp_single_ms;
    print("  Standard:  {d:.3} ms/step\n", .{std_single_ms});
    print("  ESP-fused: {d:.3} ms/step\n", .{esp_single_ms});
    print("  Speedup:   {d:.3}x\n", .{single_speedup});

    // ================================================================
    // 3. 50-step rollout benchmark (50 × seq_len=3, 6 layers each)
    //    Simulates sequential predict_next calls (attention limited to seq_len<=16)
    // ================================================================
    print("\n--- 50-Step Rollout Benchmark ({d} steps x seq_len=3, {d} layers, {d} iters) ---\n", .{ ROLLOUT_STEPS, LAYERS, ROLLOUT_ITERS });

    // Warmup standard
    for (0..WARMUP) |_| {
        for (0..ROLLOUT_STEPS) |_| {
            fillDeterministic(seq_std[0 .. single_seq * HIDDEN], 42);
            for (0..LAYERS) |_| {
                fused.lewmPredictorLayerV2(
                    seq_std.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                    adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                    mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                    mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 0,
                );
            }
        }
    }

    // Standard rollout
    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        for (0..ROLLOUT_STEPS) |_| {
            fillDeterministic(seq_std[0 .. single_seq * HIDDEN], 42);
            for (0..LAYERS) |_| {
                fused.lewmPredictorLayerV2(
                    seq_std.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                    adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                    mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                    mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 0,
                );
            }
        }
    }
    const std_roll_ns = timer.read();
    const std_roll_ms = @as(f64, @floatFromInt(std_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    // Warmup ESP rollout
    for (0..WARMUP) |_| {
        for (0..ROLLOUT_STEPS) |_| {
            fillDeterministic(seq_esp[0 .. single_seq * HIDDEN], 42);
            for (0..LAYERS) |_| {
                fused.lewmPredictorLayerV2(
                    seq_esp.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                    adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                    mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                    mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 1,
                );
            }
        }
    }

    // ESP-fused rollout
    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        for (0..ROLLOUT_STEPS) |_| {
            fillDeterministic(seq_esp[0 .. single_seq * HIDDEN], 42);
            for (0..LAYERS) |_| {
                fused.lewmPredictorLayerV2(
                    seq_esp.ptr, conditioning.ptr, single_seq, HIDDEN, NUM_HEADS, INNER_DIM, INTER,
                    adaln_w.ptr, adaln_b.ptr, attn_norm_w.ptr, to_qkv.ptr, attn_out_w.ptr, attn_out_b.ptr,
                    mlp_norm_w.ptr, mlp_up_w.ptr, mlp_up_b.ptr, mlp_down_w.ptr, mlp_down_b.ptr,
                    mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, 1,
                );
            }
        }
    }
    const esp_roll_ns = timer.read();
    const esp_roll_ms = @as(f64, @floatFromInt(esp_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    const roll_speedup = std_roll_ms / esp_roll_ms;
    print("  Standard:  {d:.1} ms/rollout ({d:.3} ms/step)\n", .{ std_roll_ms, std_roll_ms / ROLLOUT_STEPS });
    print("  ESP-fused: {d:.1} ms/rollout ({d:.3} ms/step)\n", .{ esp_roll_ms, esp_roll_ms / ROLLOUT_STEPS });
    print("  Speedup:   {d:.3}x\n", .{roll_speedup});

    // ================================================================
    // 4. Fused Rollout Benchmark (all steps batched, seq_len=150)
    // ================================================================
    print("\n--- Fused Rollout Benchmark ({d} steps batched, seq_len={d}, {d} layers, {d} iters) ---\n", .{ ROLLOUT_STEPS, ROLLOUT_SEQ_LEN, LAYERS, ROLLOUT_ITERS });

    // Build per-layer weight pointer arrays (all layers share same weights for bench)
    var adaln_ws: [LAYERS][*]const f32 = undefined;
    var adaln_bs_arr: [LAYERS][*]const f32 = undefined;
    var attn_norm_ws: [LAYERS][*]const f32 = undefined;
    var to_qkvs: [LAYERS][*]const f32 = undefined;
    var attn_out_ws: [LAYERS][*]const f32 = undefined;
    var attn_out_bs_arr: [LAYERS][*]const f32 = undefined;
    var mlp_norm_ws: [LAYERS][*]const f32 = undefined;
    var mlp_up_ws: [LAYERS][*]const f32 = undefined;
    var mlp_up_bs_arr: [LAYERS][*]const f32 = undefined;
    var mlp_down_ws: [LAYERS][*]const f32 = undefined;
    var mlp_down_bs_arr: [LAYERS][*]const f32 = undefined;

    for (0..LAYERS) |i| {
        adaln_ws[i] = adaln_w.ptr;
        adaln_bs_arr[i] = adaln_b.ptr;
        attn_norm_ws[i] = attn_norm_w.ptr;
        to_qkvs[i] = to_qkv.ptr;
        attn_out_ws[i] = attn_out_w.ptr;
        attn_out_bs_arr[i] = attn_out_b.ptr;
        mlp_norm_ws[i] = mlp_norm_w.ptr;
        mlp_up_ws[i] = mlp_up_w.ptr;
        mlp_up_bs_arr[i] = mlp_up_b.ptr;
        mlp_down_ws[i] = mlp_down_w.ptr;
        mlp_down_bs_arr[i] = mlp_down_b.ptr;
    }

    // Allocate scores buffer for dynamic attention
    const scores_buf = try allocator.alloc(f32, ROLLOUT_SEQ_LEN * ROLLOUT_SEQ_LEN);

    // Allocate GEMM packing buffers (sized for the largest GEMM in the rollout)
    const max_n = 3 * INNER_DIM; // QKV projection is widest
    const max_k = @max(HIDDEN, @max(INNER_DIM, INTER)); // FFN down K=INTER is largest
    const pack_sizes = rollout.packBufSizes(ROLLOUT_SEQ_LEN, max_n, max_k);
    const packed_a_buf = try allocator.alloc(f32, pack_sizes.a);
    const packed_b_buf = try allocator.alloc(f32, pack_sizes.b);

    // Sequence buffer for fused rollout
    const seq_fused = try allocator.alloc(f32, max_seq * HIDDEN);
    const seq_fused_esp = try allocator.alloc(f32, max_seq * HIDDEN);

    // -- Fused rollout with FUSED_ROLLOUT flag (0x01) --
    // Warmup
    for (0..WARMUP) |_| {
        fillDeterministic(seq_fused, 42);
        rollout.lewmRolloutFused(
            seq_fused.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs_arr, &attn_norm_ws, &to_qkvs, &attn_out_ws, &attn_out_bs_arr,
            &mlp_norm_ws, &mlp_up_ws, &mlp_up_bs_arr, &mlp_down_ws, &mlp_down_bs_arr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            packed_a_buf.ptr, packed_b_buf.ptr,
            rollout.FUSED_ROLLOUT,
        );
    }

    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        fillDeterministic(seq_fused, 42);
        rollout.lewmRolloutFused(
            seq_fused.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs_arr, &attn_norm_ws, &to_qkvs, &attn_out_ws, &attn_out_bs_arr,
            &mlp_norm_ws, &mlp_up_ws, &mlp_up_bs_arr, &mlp_down_ws, &mlp_down_bs_arr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            packed_a_buf.ptr, packed_b_buf.ptr,
            rollout.FUSED_ROLLOUT,
        );
    }
    const fused_roll_ns = timer.read();
    const fused_roll_ms = @as(f64, @floatFromInt(fused_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    // -- Fused rollout with FUSED_ROLLOUT | ESP_FUSED | SHARED_ADALN (0x13) --
    // Warmup
    for (0..WARMUP) |_| {
        fillDeterministic(seq_fused_esp, 42);
        rollout.lewmRolloutFused(
            seq_fused_esp.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs_arr, &attn_norm_ws, &to_qkvs, &attn_out_ws, &attn_out_bs_arr,
            &mlp_norm_ws, &mlp_up_ws, &mlp_up_bs_arr, &mlp_down_ws, &mlp_down_bs_arr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            packed_a_buf.ptr, packed_b_buf.ptr,
            rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN,
        );
    }

    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        fillDeterministic(seq_fused_esp, 42);
        rollout.lewmRolloutFused(
            seq_fused_esp.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs_arr, &attn_norm_ws, &to_qkvs, &attn_out_ws, &attn_out_bs_arr,
            &mlp_norm_ws, &mlp_up_ws, &mlp_up_bs_arr, &mlp_down_ws, &mlp_down_bs_arr,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            packed_a_buf.ptr, packed_b_buf.ptr,
            rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN,
        );
    }
    const fused_esp_roll_ns = timer.read();
    const fused_esp_roll_ms = @as(f64, @floatFromInt(fused_esp_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    const fused_vs_seq = std_roll_ms / fused_roll_ms;
    const fused_esp_vs_seq = std_roll_ms / fused_esp_roll_ms;
    print("  Sequential baseline:     {d:.1} ms/rollout\n", .{std_roll_ms});
    print("  Fused (0x01):            {d:.1} ms/rollout  ({d:.1}x vs sequential)\n", .{ fused_roll_ms, fused_vs_seq });
    print("  Fused+ESP+ADALN (0x13):  {d:.1} ms/rollout  ({d:.1}x vs sequential)\n", .{ fused_esp_roll_ms, fused_esp_vs_seq });

    // ================================================================
    // Summary
    // ================================================================
    print("\n=================================================================\n", .{});
    print("  Summary\n", .{});
    print("=================================================================\n", .{});
    print("  Single step (seq=3):        std={d:.3}ms  esp={d:.3}ms  ({d:.1}%)\n", .{
        std_single_ms,
        esp_single_ms,
        (single_speedup - 1.0) * 100.0,
    });
    print("  Rollout 50 sequential:      std={d:.1}ms  esp={d:.1}ms  ({d:.1}%)\n", .{
        std_roll_ms,
        esp_roll_ms,
        (roll_speedup - 1.0) * 100.0,
    });
    print("  Fused rollout 50 (batched): std={d:.1}ms  esp={d:.1}ms\n", .{
        fused_roll_ms,
        fused_esp_roll_ms,
    });
    print("  Fused vs sequential:        {d:.1}x (std)  {d:.1}x (esp)\n", .{
        fused_vs_seq,
        fused_esp_vs_seq,
    });
    print("=================================================================\n\n", .{});
}
