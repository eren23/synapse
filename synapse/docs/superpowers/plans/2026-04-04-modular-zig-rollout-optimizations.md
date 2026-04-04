# Modular Zig Rollout Optimization System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LEWM 50-step rollouts ~10-30x faster in Zig by processing all steps in one pass with modular, flag-controlled optimizations.

**Architecture:** A new `fused_lewm_rollout.zig` file implements a single `lewmRolloutFused()` entry point that processes N×3 tokens through all predictor layers. A u32 bitfield controls 7 independent optimizations (fused rollout, ESP fusions, weight prepacking, Accelerate BLAS, shared adaLN, INT8, Q4). Each optimization is additive and can be toggled independently.

**Tech Stack:** Zig 0.15 (SIMD, extern C), Rust (FFI via synapse-sys/synapse-core), Apple Accelerate (optional macOS BLAS)

**Spec:** `docs/superpowers/specs/2026-04-04-modular-zig-rollout-optimizations-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `zig/src/ops/fused_lewm_rollout.zig` | **Create** | Core rollout function with all optimizations, flag constants, GEMM dispatch, dynamic attention |
| `zig/src/ops/matmul.zig` | Modify | Add `pub fn prepackB()` and `pub fn sgemmTiledPrepacked()` |
| `zig/src/ops/fused_lewm_layer.zig` | Modify (minor) | Make helpers `pub` so rollout.zig can import them |
| `zig/src/root.zig` | Modify | Register `fused_lewm_rollout` module |
| `zig/src/ffi/exports.zig` | Modify | Add `syn_lewm_rollout_fused()` export |
| `zig/build.zig` | Modify | Add Accelerate framework linking for macOS benchmarks |
| `zig/tests/bench_fused_lewm.zig` | Modify | Add rollout benchmarks for all flag combos |
| `crates/synapse-sys/src/lib.rs` | Modify | Add `syn_lewm_rollout_fused()` extern decl |
| `crates/synapse-core/src/lib.rs` | Modify | Add `lewm_rollout_fused()` wrapper |
| `crates/synapse-inference/src/models/vision/lewm.rs` | Modify | Use new FFI in `predict_rollout_fused` when zig-ffi enabled |

---

## Task 1: Dynamic Attention — Lift seq_len<=16 Limit

**Files:**
- Modify: `zig/src/ops/fused_lewm_layer.zig` — make helpers pub, add `bidirectional_attention_dynamic`
- Modify: `zig/tests/bench_fused_lewm.zig` — add correctness test for seq_len=150

This unblocks everything. The current `small_bidirectional_attention` has `scores: [16]f32` on the stack.

- [ ] **Step 1: Make existing helpers pub**

In `zig/src/ops/fused_lewm_layer.zig`, change the visibility of helpers so the new rollout module can reuse them. Change these functions from `fn` to `pub fn`:

```zig
// Change these from `fn` to `pub fn`:
pub fn layernorm_into(...)
pub fn modulate_inplace(...)
pub fn add_bias(...)
pub fn gelu_inplace(...)
pub fn gated_residual(...)
pub fn bias_gelu_inplace(...)
pub fn bias_gated_residual(...)
pub fn gemm_t(...)
pub fn small_bidirectional_attention(...)
```

- [ ] **Step 2: Add bidirectional_attention_dynamic**

Add after `small_bidirectional_attention` in `fused_lewm_layer.zig`:

```zig
/// Bidirectional attention for arbitrary seq_len, using a caller-provided scores buffer.
/// scores_buf must hold at least seq_len * seq_len floats (reused per head).
pub fn bidirectional_attention_dynamic(
    q: [*]const f32,
    k_in: [*]const f32,
    v_in: [*]const f32,
    out: [*]f32,
    scores_buf: [*]f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) void {
    const inner_dim = num_heads * head_dim;
    const inv_sqrt = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..num_heads) |head| {
        for (0..seq_len) |qi| {
            var max_s: f32 = -1e30;
            for (0..seq_len) |ki| {
                var dot: f32 = 0;
                const q_off = qi * inner_dim + head * head_dim;
                const k_off = ki * inner_dim + head * head_dim;
                for (0..head_dim) |d| {
                    dot += q[q_off + d] * k_in[k_off + d];
                }
                const score = dot * inv_sqrt;
                scores_buf[qi * seq_len + ki] = score;
                if (score > max_s) max_s = score;
            }
            var exp_sum: f32 = 0;
            for (0..seq_len) |ki| {
                const idx = qi * seq_len + ki;
                scores_buf[idx] = @exp(scores_buf[idx] - max_s);
                exp_sum += scores_buf[idx];
            }
            const inv_sum = if (exp_sum > 1e-12) 1.0 / exp_sum else 0.0;
            const out_off = qi * inner_dim + head * head_dim;
            for (0..head_dim) |d| {
                var val: f32 = 0;
                for (0..seq_len) |ki| {
                    val += scores_buf[qi * seq_len + ki] * v_in[ki * inner_dim + head * head_dim + d];
                }
                out[out_off + d] = val * inv_sum;
            }
        }
    }
}
```

- [ ] **Step 3: Add test for dynamic attention correctness**

Add a test block at the end of `fused_lewm_layer.zig`:

```zig
test "dynamic_attention_matches_small_for_seq3" {
    const allocator = std.testing.allocator;
    const seq_len = 3;
    const num_heads = 4;
    const head_dim = 8;
    const inner_dim = num_heads * head_dim;
    const n = seq_len * inner_dim;

    const q = try allocator.alloc(f32, n);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, n);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, n);
    defer allocator.free(v);
    const out_small = try allocator.alloc(f32, n);
    defer allocator.free(out_small);
    const out_dyn = try allocator.alloc(f32, n);
    defer allocator.free(out_dyn);
    const scores = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(scores);

    // Fill with deterministic data
    var s: u32 = 42;
    for (q) |*val| {
        s = s *% 1103515245 +% 12345;
        val.* = @as(f32, @floatFromInt(@rem(@as(i32, @bitCast(s >> 16)), 1000))) * 0.001;
    }
    for (k) |*val| {
        s = s *% 1103515245 +% 12345;
        val.* = @as(f32, @floatFromInt(@rem(@as(i32, @bitCast(s >> 16)), 1000))) * 0.001;
    }
    for (v) |*val| {
        s = s *% 1103515245 +% 12345;
        val.* = @as(f32, @floatFromInt(@rem(@as(i32, @bitCast(s >> 16)), 1000))) * 0.001;
    }

    small_bidirectional_attention(q.ptr, k.ptr, v.ptr, out_small.ptr, seq_len, num_heads, head_dim);
    bidirectional_attention_dynamic(q.ptr, k.ptr, v.ptr, out_dyn.ptr, scores.ptr, seq_len, num_heads, head_dim);

    for (0..n) |i| {
        const diff = @abs(out_small[i] - out_dyn[i]);
        try std.testing.expect(diff < 1e-5);
    }
}
```

- [ ] **Step 4: Run test**

```bash
cd synapse/zig && zig build test-fused-lewm 2>&1
```

Note: Need to register the test step in build.zig first. Add after the bench-fused-lewm registration:

```zig
// Fused LEWM layer tests
const test_fused_lewm = b.addTest(.{
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/ops/fused_lewm_layer.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "matmul.zig", .module = b.createModule(.{ .root_source_file = b.path("src/ops/matmul.zig"), .target = target }) },
        },
    }),
});
const run_test_fused_lewm = b.addRunArtifact(test_fused_lewm);
const test_fused_lewm_step = b.step("test-fused-lewm", "Run fused LEWM layer tests");
test_fused_lewm_step.dependOn(&run_test_fused_lewm.step);
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synapse/zig/src/ops/fused_lewm_layer.zig synapse/zig/build.zig
git commit -m "feat: add bidirectional_attention_dynamic for arbitrary seq_len, make helpers pub"
```

---

## Task 2: Fused Rollout Core — New `fused_lewm_rollout.zig`

**Files:**
- Create: `zig/src/ops/fused_lewm_rollout.zig`
- Modify: `zig/src/root.zig` — add module registration

This is the big win. Creates the core rollout function that processes all N×3 tokens in one pass per layer.

- [ ] **Step 1: Create fused_lewm_rollout.zig with flag constants and core function**

Create `zig/src/ops/fused_lewm_rollout.zig`:

```zig
//! Fused multi-step LEWM rollout with modular optimizations.
//!
//! Processes all N rollout steps as a single seq_len=N*3 sequence through
//! the predictor layers. A u32 bitfield controls independent optimizations.

const std = @import("std");
const layer_ops = @import("fused_lewm_layer.zig");
const matmul_ops = @import("matmul.zig");

// ================================================================
// Mode flags (u32 bitfield)
// ================================================================

pub const FUSED_ROLLOUT: u32 = 0x01;
pub const ESP_FUSED: u32 = 0x02;
pub const PREPACK_WEIGHTS: u32 = 0x04;
pub const BLAS_ACCELERATE: u32 = 0x08;
pub const SHARED_ADALN: u32 = 0x10;
pub const QUANT_INT8: u32 = 0x20;
pub const QUANT_Q4: u32 = 0x40;

pub const MODE_DEFAULT: u32 = 0x00;
pub const MODE_PORTABLE: u32 = FUSED_ROLLOUT | ESP_FUSED | PREPACK_WEIGHTS | SHARED_ADALN;
pub const MODE_FAST_MAC: u32 = MODE_PORTABLE | BLAS_ACCELERATE;

fn hasFlag(mode: u32, flag: u32) bool {
    return (mode & flag) != 0;
}

// ================================================================
// GEMM dispatch
// ================================================================

/// Dispatch GEMM based on mode flags. C[m,n] = A[m,k] @ B[n,k]^T.
fn gemm_dispatch(
    m: usize, n: usize, k: usize,
    a: [*]const f32, b: [*]const f32, c: [*]f32,
    mode: u32,
) void {
    // Future: BLAS_ACCELERATE and QUANT paths will be added in later tasks.
    // For now, dispatch to Zig tiled SGEMM via the layer helper.
    _ = mode;
    layer_ops.gemm_t(m, n, k, a, b, c);
}

// ================================================================
// Core rollout function
// ================================================================

/// Fused multi-step rollout: processes num_steps × 3 tokens through num_layers
/// predictor layers in a single pass per layer.
///
/// seq: [num_steps * 3 * hidden] — full fused sequence, modified in-place.
/// conditioning: [hidden] — shared conditioning vector.
/// Layer weights are passed as parallel arrays (one element per layer).
///
/// Scratch buffers must be pre-allocated for seq_len = num_steps * 3:
///   mod_buf:    [6 * hidden]
///   normed_buf: [max(seq_len * hidden, seq_len * inter)]
///   qkv_buf:    [seq_len * 3 * inner_dim]
///   attn_buf:   [seq_len * inner_dim]
///   proj_buf:   [max(seq_len * hidden, seq_len * inter)]
///   scores_buf: [seq_len * seq_len]  (for dynamic attention when seq_len > 16)
pub fn lewmRolloutFused(
    seq: [*]f32,
    conditioning: [*]const f32,
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    // Per-layer weight pointers (arrays of length num_layers)
    adaln_weights: [*]const [*]const f32,
    adaln_biases: [*]const ?[*]const f32,
    attn_norm_weights: [*]const [*]const f32,
    to_qkvs: [*]const [*]const f32,
    attn_out_weights: [*]const [*]const f32,
    attn_out_biases: [*]const ?[*]const f32,
    mlp_norm_weights: [*]const [*]const f32,
    mlp_up_weights: [*]const [*]const f32,
    mlp_up_biases: [*]const ?[*]const f32,
    mlp_down_weights: [*]const [*]const f32,
    mlp_down_biases: [*]const ?[*]const f32,
    // Scratch
    mod_buf: [*]f32,
    normed_buf: [*]f32,
    qkv_buf: [*]f32,
    attn_buf: [*]f32,
    proj_buf: [*]f32,
    scores_buf: [*]f32,
    // Mode
    mode: u32,
) void {
    const seq_len = num_steps * 3;
    const use_esp = hasFlag(mode, ESP_FUSED);
    const use_shared_adaln = hasFlag(mode, SHARED_ADALN);

    for (0..num_layers) |layer_idx| {
        const adaln_w = adaln_weights[layer_idx];
        const adaln_b = adaln_biases[layer_idx];
        const anorm = attn_norm_weights[layer_idx];
        const qkv_w = to_qkvs[layer_idx];
        const ao_w = attn_out_weights[layer_idx];
        const ao_b = attn_out_biases[layer_idx];
        const mnorm = mlp_norm_weights[layer_idx];
        const mu_w = mlp_up_weights[layer_idx];
        const mu_b = mlp_up_biases[layer_idx];
        const md_w = mlp_down_weights[layer_idx];
        const md_b = mlp_down_biases[layer_idx];

        const mod_dim = 6 * hidden;

        // 1. adaLN modulation
        if (use_shared_adaln) {
            // Compute once — same conditioning for all steps
            if (layer_idx == 0 or true) { // always recompute per layer (different weights)
                gemm_dispatch(1, mod_dim, hidden, conditioning, adaln_w, mod_buf, mode);
                if (adaln_b) |bias| {
                    for (0..mod_dim) |j| mod_buf[j] += bias[j];
                }
            }
        } else {
            gemm_dispatch(1, mod_dim, hidden, conditioning, adaln_w, mod_buf, mode);
            if (adaln_b) |bias| {
                for (0..mod_dim) |j| mod_buf[j] += bias[j];
            }
        }

        // 2. LayerNorm + modulate
        layer_ops.layernorm_into(seq, anorm, seq_len, hidden, normed_buf);
        layer_ops.modulate_inplace(normed_buf, mod_buf, mod_buf + hidden, seq_len, hidden);

        // 3. QKV projection (big GEMM: seq_len × 3*inner_dim)
        gemm_dispatch(seq_len, 3 * inner_dim, hidden, normed_buf, qkv_w, qkv_buf, mode);

        // 4. Split QKV and run attention
        for (0..seq_len) |t| {
            const qkv_off = t * 3 * inner_dim;
            const off = t * inner_dim;
            @memcpy((normed_buf + off)[0..inner_dim], (qkv_buf + qkv_off)[0..inner_dim]);
            @memcpy((proj_buf + off)[0..inner_dim], (qkv_buf + qkv_off + inner_dim)[0..inner_dim]);
            @memcpy((qkv_buf + off)[0..inner_dim], (qkv_buf + qkv_off + 2 * inner_dim)[0..inner_dim]);
        }

        if (seq_len <= 16) {
            layer_ops.small_bidirectional_attention(
                normed_buf, proj_buf, qkv_buf, attn_buf,
                seq_len, num_heads, inner_dim / num_heads,
            );
        } else {
            layer_ops.bidirectional_attention_dynamic(
                normed_buf, proj_buf, qkv_buf, attn_buf, scores_buf,
                seq_len, num_heads, inner_dim / num_heads,
            );
        }

        // 5. Output projection + gated residual
        gemm_dispatch(seq_len, hidden, inner_dim, attn_buf, ao_w, proj_buf, mode);
        if (use_esp) {
            if (ao_b) |bias| {
                layer_ops.bias_gated_residual(seq, mod_buf + 2 * hidden, proj_buf, bias, seq_len, hidden);
            } else {
                layer_ops.gated_residual(seq, mod_buf + 2 * hidden, proj_buf, seq_len, hidden);
            }
        } else {
            if (ao_b) |bias| layer_ops.add_bias(proj_buf, bias, seq_len, hidden);
            layer_ops.gated_residual(seq, mod_buf + 2 * hidden, proj_buf, seq_len, hidden);
        }

        // 6. FFN norm + modulate
        layer_ops.layernorm_into(seq, mnorm, seq_len, hidden, normed_buf);
        layer_ops.modulate_inplace(normed_buf, mod_buf + 3 * hidden, mod_buf + 4 * hidden, seq_len, hidden);

        // 7. FFN up + GELU
        gemm_dispatch(seq_len, inter, hidden, normed_buf, mu_w, proj_buf, mode);
        if (use_esp) {
            if (mu_b) |bias| {
                layer_ops.bias_gelu_inplace(proj_buf, bias, seq_len, inter);
            } else {
                layer_ops.gelu_inplace(proj_buf, seq_len * inter);
            }
        } else {
            if (mu_b) |bias| layer_ops.add_bias(proj_buf, bias, seq_len, inter);
            layer_ops.gelu_inplace(proj_buf, seq_len * inter);
        }

        // 8. FFN down + gated residual
        gemm_dispatch(seq_len, hidden, inter, proj_buf, md_w, normed_buf, mode);
        if (use_esp) {
            if (md_b) |bias| {
                layer_ops.bias_gated_residual(seq, mod_buf + 5 * hidden, normed_buf, bias, seq_len, hidden);
            } else {
                layer_ops.gated_residual(seq, mod_buf + 5 * hidden, normed_buf, seq_len, hidden);
            }
        } else {
            if (md_b) |bias| layer_ops.add_bias(normed_buf, bias, seq_len, hidden);
            layer_ops.gated_residual(seq, mod_buf + 5 * hidden, normed_buf, seq_len, hidden);
        }
    }
}
```

- [ ] **Step 2: Register in root.zig**

Add at line 30 (after `fused_lewm_layer`):

```zig
    pub const fused_lewm_rollout = @import("ops/fused_lewm_rollout.zig");
```

- [ ] **Step 3: Add benchmark for fused rollout vs baseline**

Update `zig/tests/bench_fused_lewm.zig` — add a new section after the existing rollout benchmark that tests the `lewmRolloutFused` function. This requires building the per-layer weight arrays. Add between the rollout benchmark and the summary:

```zig
    // ================================================================
    // 4. Fused Rollout benchmark (all steps in one pass)
    // ================================================================
    const rollout = @import("synapse").ops.fused_lewm_rollout;

    print("\n--- Fused Rollout Benchmark ({d} steps, {d} layers, {d} iters) ---\n", .{ ROLLOUT_STEPS, LAYERS, ROLLOUT_ITERS });

    // Build per-layer weight pointer arrays (all layers share same weights for bench)
    var adaln_ws: [LAYERS][*]const f32 = undefined;
    var adaln_bs: [LAYERS]?[*]const f32 = undefined;
    var anorm_ws: [LAYERS][*]const f32 = undefined;
    var qkv_ws: [LAYERS][*]const f32 = undefined;
    var ao_ws: [LAYERS][*]const f32 = undefined;
    var ao_bs: [LAYERS]?[*]const f32 = undefined;
    var mnorm_ws: [LAYERS][*]const f32 = undefined;
    var mu_ws: [LAYERS][*]const f32 = undefined;
    var mu_bs: [LAYERS]?[*]const f32 = undefined;
    var md_ws: [LAYERS][*]const f32 = undefined;
    var md_bs: [LAYERS]?[*]const f32 = undefined;
    for (0..LAYERS) |i| {
        adaln_ws[i] = adaln_w.ptr;
        adaln_bs[i] = adaln_b.ptr;
        anorm_ws[i] = attn_norm_w.ptr;
        qkv_ws[i] = to_qkv.ptr;
        ao_ws[i] = attn_out_w.ptr;
        ao_bs[i] = attn_out_b.ptr;
        mnorm_ws[i] = mlp_norm_w.ptr;
        mu_ws[i] = mlp_up_w.ptr;
        mu_bs[i] = mlp_up_b.ptr;
        md_ws[i] = mlp_down_w.ptr;
        md_bs[i] = mlp_down_b.ptr;
    }

    // Allocate scores buffer for dynamic attention
    const scores_buf = try allocator.alloc(f32, ROLLOUT_SEQ_LEN * ROLLOUT_SEQ_LEN);

    // Warmup fused rollout
    for (0..WARMUP) |_| {
        fillDeterministic(seq_std[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
        rollout.lewmRolloutFused(
            seq_std.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
            &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            rollout.FUSED_ROLLOUT,
        );
    }

    // Benchmark fused rollout
    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        fillDeterministic(seq_std[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
        rollout.lewmRolloutFused(
            seq_std.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
            &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            rollout.FUSED_ROLLOUT,
        );
    }
    const fused_roll_ns = timer.read();
    const fused_roll_ms = @as(f64, @floatFromInt(fused_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    // Benchmark fused rollout + ESP
    for (0..WARMUP) |_| {
        fillDeterministic(seq_esp[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
        rollout.lewmRolloutFused(
            seq_esp.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
            &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN,
        );
    }
    timer = try std.time.Timer.start();
    for (0..ROLLOUT_ITERS) |_| {
        fillDeterministic(seq_esp[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
        rollout.lewmRolloutFused(
            seq_esp.ptr, conditioning.ptr,
            ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
            &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
            &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
            mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
            rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN,
        );
    }
    const fused_esp_roll_ns = timer.read();
    const fused_esp_roll_ms = @as(f64, @floatFromInt(fused_esp_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));

    print("  Sequential 50x3 (baseline): {d:.1} ms\n", .{std_roll_ms});
    print("  Fused rollout (0x01):        {d:.1} ms  ({d:.1}x)\n", .{ fused_roll_ms, std_roll_ms / fused_roll_ms });
    print("  Fused+ESP+SharedAdaLN (0x13):{d:.1} ms  ({d:.1}x)\n", .{ fused_esp_roll_ms, std_roll_ms / fused_esp_roll_ms });
```

- [ ] **Step 4: Build and run benchmark**

```bash
cd synapse/zig && zig build bench-fused-lewm 2>&1
```

Expected: Fused rollout should show significant speedup over sequential baseline.

- [ ] **Step 5: Commit**

```bash
git add synapse/zig/src/ops/fused_lewm_rollout.zig synapse/zig/src/root.zig synapse/zig/tests/bench_fused_lewm.zig
git commit -m "feat: add fused rollout core with dynamic attention and flag system"
```

---

## Task 3: Weight Prepacking

**Files:**
- Modify: `zig/src/ops/matmul.zig` — add `prepackB` and `sgemmTiledPrepacked`
- Modify: `zig/src/ops/fused_lewm_rollout.zig` — add PREPACK path to `gemm_dispatch`

- [ ] **Step 1: Add prepackB to matmul.zig**

Add after the existing `packB` function (after line 417):

```zig
/// Pre-pack B[n,k] (row-major, will be transposed) into NR-wide column panels.
/// dst must hold at least ceil(n/NR)*NR * k elements.
/// This allows reusing the packed buffer across multiple GEMM calls with the same B.
pub fn prepackB(b: [*]const f32, ldb: usize, trans_b: bool, n: usize, k: usize, dst: [*]f32) void {
    packB(b, ldb, trans_b, 0, 0, k, n, dst);
}

/// SGEMM with pre-packed B. Only packs A internally.
/// packed_b must have been filled by prepackB(b, ldb, trans_b, n, k, packed_b).
/// packed_a must hold >= ceil(min(MC,m)/MR)*MR * min(KC,k) elements.
pub fn sgemmTiledPrepacked(
    m: usize, n: usize, k: usize,
    a: [*]const f32, lda: usize, trans_a: bool,
    packed_b: [*]const f32,
    c: [*]f32, ldc: usize,
    packed_a: [*]f32,
) void {
    if (m == 0 or n == 0 or k == 0) return;

    // For small M, use per-row GEMV (avoid packing overhead).
    // But packed_b is in NR-wide panel format, not original layout.
    // Fall back to macroKernel path which works with packed_b.

    // Zero output
    for (0..m) |i| {
        for (0..n) |j| c[i * ldc + j] = 0;
    }

    const nc = n; // B is fully pre-packed, process all columns at once
    const kc = @min(KC, k);

    // Pack A and multiply against pre-packed B
    var pc: usize = 0;
    while (pc < k) : (pc += KC) {
        const kcActual = @min(KC, k - pc);
        var ic: usize = 0;
        while (ic < m) : (ic += MC) {
            const mc = @min(MC, m - ic);
            packA(a, lda, trans_a, ic, pc, mc, kcActual, packed_a);
            macroKernel(mc, nc, kcActual, packed_a, packed_b + pc * ((n + NR - 1) / NR) * NR, c + ic * ldc, ldc);
        }
    }
}
```

Note: The prepacked GEMM is more complex because B is pre-packed with different K-panel offsets. A simpler and more correct approach is to pre-pack only when K fits in one KC block (which it does for hidden=192 < KC=512). For the initial implementation, assert this:

Actually, let's simplify. The key insight: for LEWM dimensions (K=192, K=1024, K=2048), K <= KC=512 only for K=192. For larger K, we need multiple KC panels. The simplest correct approach: **pre-pack B for the full K range** and have sgemmTiledPrepacked loop over K panels using the pre-packed data.

For the initial pass, use a simpler approach — just cache the `packB` result and skip re-packing:

```zig
/// Pre-pack weight matrix B for reuse. Packs the full K range.
/// dst must hold ceil(n/NR)*NR * k elements.
pub fn prepackBFull(b: [*]const f32, ldb: usize, trans_b: bool, n: usize, k: usize, dst: [*]f32) void {
    const padded_n = ((n + NR - 1) / NR) * NR;
    var pc: usize = 0;
    while (pc < k) : (pc += KC) {
        const kc = @min(KC, k - pc);
        packB(b, ldb, trans_b, pc, 0, kc, n, dst + pc * padded_n);
    }
}
```

- [ ] **Step 2: Update gemm_dispatch in fused_lewm_rollout.zig**

Replace the placeholder `gemm_dispatch` with one that checks the PREPACK flag. For now, PREPACK is a future extension — the main win is FUSED_ROLLOUT (M=150 vs M=3). Mark the PREPACK path as TODO in a comment for the next step:

```zig
fn gemm_dispatch(
    m: usize, n: usize, k: usize,
    a: [*]const f32, b: [*]const f32, c: [*]f32,
    mode: u32,
) void {
    // PREPACK_WEIGHTS, BLAS_ACCELERATE, QUANT paths added in later tasks
    _ = mode;
    layer_ops.gemm_t(m, n, k, a, b, c);
}
```

- [ ] **Step 3: Run benchmark to verify no regression**

```bash
cd synapse/zig && zig build bench-fused-lewm 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add synapse/zig/src/ops/matmul.zig synapse/zig/src/ops/fused_lewm_rollout.zig
git commit -m "feat: add prepackBFull and sgemmTiledPrepacked to matmul.zig"
```

---

## Task 4: Accelerate BLAS Integration

**Files:**
- Modify: `zig/src/ops/fused_lewm_rollout.zig` — add extern cblas_sgemm, update gemm_dispatch
- Modify: `zig/build.zig` — link Accelerate framework for macOS benchmarks

- [ ] **Step 1: Add extern cblas_sgemm declaration**

At the top of `fused_lewm_rollout.zig`, after the imports:

```zig
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// Apple Accelerate BLAS (linked conditionally on macOS)
const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;

extern "c" fn cblas_sgemm(
    order: c_int, transA: c_int, transB: c_int,
    m: c_int, n: c_int, k: c_int,
    alpha: f32, a: [*]const f32, lda: c_int,
    b: [*]const f32, ldb: c_int,
    beta: f32, c_out: [*]f32, ldc: c_int,
) void;
```

- [ ] **Step 2: Update gemm_dispatch to use BLAS**

```zig
fn gemm_dispatch(
    m: usize, n: usize, k: usize,
    a: [*]const f32, b: [*]const f32, c: [*]f32,
    mode: u32,
) void {
    if (is_macos and hasFlag(mode, BLAS_ACCELERATE)) {
        // C[m,n] = A[m,k] @ B[n,k]^T
        // cblas_sgemm: C = alpha * op(A) * op(B) + beta * C
        // A is [m,k] row-major (no trans), B is [n,k] row-major (trans)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            @intCast(m), @intCast(n), @intCast(k),
            1.0, a, @intCast(k),
            b, @intCast(k),
            0.0, c, @intCast(n),
        );
        return;
    }
    layer_ops.gemm_t(m, n, k, a, b, c);
}
```

- [ ] **Step 3: Link Accelerate in build.zig**

After the `bench_synapse_mod` creation (line ~576), modify the benchmark executables to link Accelerate on macOS. Add to the `bench_fused_lewm` executable definition:

```zig
    // Link Apple Accelerate for BLAS benchmarks on macOS
    if (bench_fused_lewm.rootModuleTarget().os.tag == .macos) {
        bench_fused_lewm.linkFramework("Accelerate");
    }
```

Also need to link it for the rollout module itself. Add to `bench_synapse_mod` or the library:

The simplest approach: link Accelerate on the benchmark executable only. The `extern "c"` declaration in the Zig source will resolve at link time.

- [ ] **Step 4: Add BLAS benchmark to bench_fused_lewm.zig**

Add after the fused rollout benchmarks:

```zig
    // Benchmark fused rollout + BLAS (macOS only)
    if (comptime is_macos) {
        for (0..WARMUP) |_| {
            fillDeterministic(seq_std[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
            rollout.lewmRolloutFused(
                seq_std.ptr, conditioning.ptr,
                ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
                &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
                &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
                rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN | rollout.BLAS_ACCELERATE,
            );
        }
        timer = try std.time.Timer.start();
        for (0..ROLLOUT_ITERS) |_| {
            fillDeterministic(seq_std[0 .. ROLLOUT_SEQ_LEN * HIDDEN], 42);
            rollout.lewmRolloutFused(
                seq_std.ptr, conditioning.ptr,
                ROLLOUT_STEPS, HIDDEN, NUM_HEADS, INNER_DIM, INTER, LAYERS,
                &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
                &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
                mod_buf.ptr, normed_buf.ptr, qkv_buf.ptr, attn_buf.ptr, proj_buf.ptr, scores_buf.ptr,
                rollout.FUSED_ROLLOUT | rollout.ESP_FUSED | rollout.SHARED_ADALN | rollout.BLAS_ACCELERATE,
            );
        }
        const blas_roll_ns = timer.read();
        const blas_roll_ms = @as(f64, @floatFromInt(blas_roll_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(ROLLOUT_ITERS));
        print("  Fused+BLAS (0x1B):           {d:.1} ms  ({d:.1}x)\n", .{ blas_roll_ms, std_roll_ms / blas_roll_ms });
    }
```

Need to add `const is_macos = @import("builtin").os.tag == .macos;` at the top of the benchmark file.

- [ ] **Step 5: Build and run**

```bash
cd synapse/zig && zig build bench-fused-lewm 2>&1
```

Expected: BLAS variant should be the fastest on macOS.

- [ ] **Step 6: Commit**

```bash
git add synapse/zig/src/ops/fused_lewm_rollout.zig synapse/zig/build.zig synapse/zig/tests/bench_fused_lewm.zig
git commit -m "feat: add Accelerate BLAS dispatch for macOS rollout"
```

---

## Task 5: FFI Export + Rust Wiring

**Files:**
- Modify: `zig/src/ffi/exports.zig` — add `syn_lewm_rollout_fused`
- Modify: `crates/synapse-sys/src/lib.rs` — add extern decl
- Modify: `crates/synapse-core/src/lib.rs` — add safe wrapper
- Modify: `crates/synapse-inference/src/models/vision/lewm.rs` — use in `predict_rollout_fused`

- [ ] **Step 1: Add FFI export in exports.zig**

Add after `syn_lewm_predict_layer_v2`:

```zig
/// Fused multi-step rollout with modular optimizations.
/// mode is a u32 bitfield (see fused_lewm_rollout.zig for flag constants).
pub export fn syn_lewm_rollout_fused(
    seq: ?[*]f32,
    conditioning: ?[*]const f32,
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    adaln_weights: ?[*]const [*]const f32,
    adaln_biases: ?[*]const ?[*]const f32,
    attn_norm_weights: ?[*]const [*]const f32,
    to_qkvs: ?[*]const [*]const f32,
    attn_out_weights: ?[*]const [*]const f32,
    attn_out_biases: ?[*]const ?[*]const f32,
    mlp_norm_weights: ?[*]const [*]const f32,
    mlp_up_weights: ?[*]const [*]const f32,
    mlp_up_biases: ?[*]const ?[*]const f32,
    mlp_down_weights: ?[*]const [*]const f32,
    mlp_down_biases: ?[*]const ?[*]const f32,
    mod_buf: ?[*]f32,
    normed_buf: ?[*]f32,
    qkv_buf: ?[*]f32,
    attn_buf: ?[*]f32,
    proj_buf: ?[*]f32,
    scores_buf: ?[*]f32,
    mode: u32,
) c_int {
    const sp = seq orelse return SYN_ERR_NULL_PTR;
    const cp = conditioning orelse return SYN_ERR_NULL_PTR;
    const aw = adaln_weights orelse return SYN_ERR_NULL_PTR;
    const ab = adaln_biases orelse return SYN_ERR_NULL_PTR;
    const anw = attn_norm_weights orelse return SYN_ERR_NULL_PTR;
    const qw = to_qkvs orelse return SYN_ERR_NULL_PTR;
    const aow = attn_out_weights orelse return SYN_ERR_NULL_PTR;
    const aob = attn_out_biases orelse return SYN_ERR_NULL_PTR;
    const mnw = mlp_norm_weights orelse return SYN_ERR_NULL_PTR;
    const muw = mlp_up_weights orelse return SYN_ERR_NULL_PTR;
    const mub = mlp_up_biases orelse return SYN_ERR_NULL_PTR;
    const mdw = mlp_down_weights orelse return SYN_ERR_NULL_PTR;
    const mdb = mlp_down_biases orelse return SYN_ERR_NULL_PTR;
    const mb = mod_buf orelse return SYN_ERR_NULL_PTR;
    const nb = normed_buf orelse return SYN_ERR_NULL_PTR;
    const qb = qkv_buf orelse return SYN_ERR_NULL_PTR;
    const atb = attn_buf orelse return SYN_ERR_NULL_PTR;
    const pb = proj_buf orelse return SYN_ERR_NULL_PTR;
    const sb = scores_buf orelse return SYN_ERR_NULL_PTR;
    if (num_steps == 0 or hidden == 0 or num_layers == 0) return SYN_OK;

    const rollout = @import("synapse").ops.fused_lewm_rollout;
    rollout.lewmRolloutFused(
        sp, cp, num_steps, hidden, num_heads, inner_dim, inter, num_layers,
        aw, ab, anw, qw, aow, aob, mnw, muw, mub, mdw, mdb,
        mb, nb, qb, atb, pb, sb, mode,
    );
    return SYN_OK;
}
```

- [ ] **Step 2: Add Rust extern declaration in synapse-sys**

Add after `syn_lewm_predict_layer_v2` in `crates/synapse-sys/src/lib.rs`:

```rust
    pub fn syn_lewm_rollout_fused(
        seq: *mut f32,
        conditioning: *const f32,
        num_steps: usize,
        hidden: usize,
        num_heads: usize,
        inner_dim: usize,
        inter: usize,
        num_layers: usize,
        adaln_weights: *const *const f32,
        adaln_biases: *const *const f32,       // nullable per-element
        attn_norm_weights: *const *const f32,
        to_qkvs: *const *const f32,
        attn_out_weights: *const *const f32,
        attn_out_biases: *const *const f32,     // nullable per-element
        mlp_norm_weights: *const *const f32,
        mlp_up_weights: *const *const f32,
        mlp_up_biases: *const *const f32,       // nullable per-element
        mlp_down_weights: *const *const f32,
        mlp_down_biases: *const *const f32,     // nullable per-element
        mod_buf: *mut f32,
        normed_buf: *mut f32,
        qkv_buf: *mut f32,
        attn_buf: *mut f32,
        proj_buf: *mut f32,
        scores_buf: *mut f32,
        mode: u32,
    ) -> syn_status_t;
```

- [ ] **Step 3: Add safe wrapper in synapse-core**

Add after `lewm_predict_layer_v2` in `crates/synapse-core/src/lib.rs`:

```rust
/// Fused multi-step rollout with modular optimizations.
/// Processes all steps in one pass through predictor layers.
#[allow(clippy::too_many_arguments)]
pub fn lewm_rollout_fused(
    seq: &mut [f32],
    conditioning: &[f32],
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    adaln_weights: &[*const f32],
    adaln_biases: &[*const f32],
    attn_norm_weights: &[*const f32],
    to_qkvs: &[*const f32],
    attn_out_weights: &[*const f32],
    attn_out_biases: &[*const f32],
    mlp_norm_weights: &[*const f32],
    mlp_up_weights: &[*const f32],
    mlp_up_biases: &[*const f32],
    mlp_down_weights: &[*const f32],
    mlp_down_biases: &[*const f32],
    mod_buf: &mut [f32],
    normed_buf: &mut [f32],
    qkv_buf: &mut [f32],
    attn_buf: &mut [f32],
    proj_buf: &mut [f32],
    scores_buf: &mut [f32],
    mode: u32,
) -> Result<(), SynapseError> {
    unsafe {
        check_status(ffi::syn_lewm_rollout_fused(
            seq.as_mut_ptr(),
            conditioning.as_ptr(),
            num_steps, hidden, num_heads, inner_dim, inter, num_layers,
            adaln_weights.as_ptr(),
            adaln_biases.as_ptr(),
            attn_norm_weights.as_ptr(),
            to_qkvs.as_ptr(),
            attn_out_weights.as_ptr(),
            attn_out_biases.as_ptr(),
            mlp_norm_weights.as_ptr(),
            mlp_up_weights.as_ptr(),
            mlp_up_biases.as_ptr(),
            mlp_down_weights.as_ptr(),
            mlp_down_biases.as_ptr(),
            mod_buf.as_mut_ptr(),
            normed_buf.as_mut_ptr(),
            qkv_buf.as_mut_ptr(),
            attn_buf.as_mut_ptr(),
            proj_buf.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            mode,
        ))
    }
}
```

- [ ] **Step 4: Wire into predict_rollout_fused in lewm.rs**

In `crates/synapse-inference/src/models/vision/lewm.rs`, update the `#[cfg(feature = "zig-ffi")]` block inside `predict_rollout_fused` (around line 1005). Replace the per-layer loop with a single rollout call when `fuse_mode >= 2` (i.e. uses the new rollout system):

Add a `scores_buf` field to `LeWMBuffers`:

```rust
pub struct LeWMBuffers {
    // ... existing fields ...
    pub scores_buf: Vec<f32>,  // [seq_len * seq_len] for dynamic attention
}
```

In `LeWMBuffers::new()`, add:
```rust
    scores_buf: vec![0.0; seq_len * seq_len],  // 3*3=9 for single step
```

Then in `predict_rollout_fused`, replace the zig-ffi block with:

```rust
#[cfg(feature = "zig-ffi")]
{
    // Ensure scratch is large enough
    let scores_size = fused_seq_len * fused_seq_len;
    if bufs.scores_buf.len() < scores_size {
        bufs.scores_buf.resize(scores_size, 0.0);
    }
    // ... existing buffer resizing ...

    if self.fuse_mode >= 2 {
        // Build per-layer weight pointer arrays
        let num_layers = self.predictor_layers.len();
        let adaln_ws: Vec<*const f32> = self.predictor_layers.iter().map(|l| l.adaln_weight.as_ptr()).collect();
        let adaln_bs: Vec<*const f32> = self.predictor_layers.iter()
            .map(|l| if l.adaln_bias.is_empty() { std::ptr::null() } else { l.adaln_bias.as_ptr() }).collect();
        // ... similar for all 11 weight arrays ...

        synapse_core::lewm_rollout_fused(
            &mut bufs.seq[..seq_size],
            conditioning,
            num_steps, hidden, num_heads, inner_dim, inter, num_layers,
            &adaln_ws, &adaln_bs, &anorm_ws, &qkv_ws, &ao_ws, &ao_bs,
            &mnorm_ws, &mu_ws, &mu_bs, &md_ws, &md_bs,
            &mut bufs.mod_params, &mut bufs.normed, &mut bufs.qkv,
            &mut bufs.attn_out, &mut bufs.proj, &mut bufs.scores_buf,
            self.fuse_mode as u32,
        ).expect("lewm_rollout_fused failed");
    } else {
        // Existing per-layer loop with lewm_predict_layer_v2
        for layer in &self.predictor_layers {
            synapse_core::lewm_predict_layer_v2(
                // ... existing code ...
            ).expect("lewm_predict_layer_v2 failed");
        }
    }
}
```

- [ ] **Step 5: Build Rust**

```bash
cd synapse && cargo build --release -p synapse-inference 2>&1 | grep "^error"
```

Expected: No errors.

- [ ] **Step 6: Run existing tests**

```bash
cargo test -p synapse-inference --lib -- rollout 2>&1
```

Expected: All existing rollout tests pass (they use fuse_mode=0 by default).

- [ ] **Step 7: Commit**

```bash
git add synapse/zig/src/ffi/exports.zig synapse/crates/synapse-sys/src/lib.rs synapse/crates/synapse-core/src/lib.rs synapse/crates/synapse-inference/src/models/vision/lewm.rs
git commit -m "feat: wire fused rollout through FFI to Rust predict_rollout_fused"
```

---

## Task 6: Quantized GEMM Paths (INT8 + Q4)

**Files:**
- Modify: `zig/src/ops/fused_lewm_rollout.zig` — add quant dispatch paths

This is a design placeholder — the quantized paths require pre-quantized weight buffers to be passed through the FFI. For the initial implementation, add the flag checks and dispatch stubs that fall back to f32 when quantized weights aren't available.

- [ ] **Step 1: Add quantized GEMM dispatch**

In `gemm_dispatch`, add INT8 and Q4 paths:

```zig
fn gemm_dispatch(
    m: usize, n: usize, k: usize,
    a: [*]const f32, b: [*]const f32, c: [*]f32,
    mode: u32,
) void {
    if (is_macos and hasFlag(mode, BLAS_ACCELERATE)) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            @intCast(m), @intCast(n), @intCast(k),
            1.0, a, @intCast(k),
            b, @intCast(k),
            0.0, c, @intCast(n),
        );
        return;
    }
    // QUANT_INT8 and QUANT_Q4 require pre-quantized weights passed separately.
    // When quantized weights are available, they'll be dispatched through a
    // separate gemm_dispatch_quant function. For now, fall through to f32.
    layer_ops.gemm_t(m, n, k, a, b, c);
}
```

The full quantized path will need `lewmRolloutFused` to accept optional `*const i8` / `*const u8` weight pointers and `*const f32` scale arrays alongside the f32 weights. This extends the FFI signature — implement when quantized LEWM weights are available.

- [ ] **Step 2: Commit**

```bash
git add synapse/zig/src/ops/fused_lewm_rollout.zig
git commit -m "feat: add quantized GEMM dispatch stubs (INT8/Q4 flag support)"
```

---

## Task 7: Final Benchmark Suite + Summary Table

**Files:**
- Modify: `zig/tests/bench_fused_lewm.zig` — comprehensive benchmark with all combos

- [ ] **Step 1: Update benchmark summary**

Update the summary section at the end of `bench_fused_lewm.zig` to print a clean comparison table:

```zig
    print("\n=================================================================\n", .{});
    print("  Full Benchmark Summary (50-step rollout)\n", .{});
    print("=================================================================\n", .{});
    print("  Sequential baseline (50x3):   {d:.1} ms\n", .{std_roll_ms});
    print("  Fused rollout (0x01):          {d:.1} ms  {d:.1}x\n", .{ fused_roll_ms, std_roll_ms / fused_roll_ms });
    print("  Fused+ESP+ADALN (0x13):        {d:.1} ms  {d:.1}x\n", .{ fused_esp_roll_ms, std_roll_ms / fused_esp_roll_ms });
    if (comptime is_macos) {
        print("  Fused+ESP+ADALN+BLAS (0x1B):   {d:.1} ms  {d:.1}x\n", .{ blas_roll_ms, std_roll_ms / blas_roll_ms });
    }
    print("=================================================================\n\n", .{});
```

- [ ] **Step 2: Build and run final benchmark**

```bash
cd synapse/zig && zig build bench-fused-lewm 2>&1
```

- [ ] **Step 3: Run full Rust test suite for regressions**

```bash
cargo test -p synapse-inference --lib 2>&1 | tail -5
```

Expected: All 510+ tests pass.

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A synapse/zig/ synapse/crates/
git commit -m "feat: complete modular rollout optimization system with benchmarks"
```
