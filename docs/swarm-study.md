# Swarm Execution Study

An analysis of how the attocode swarm orchestrated building the Synapse
framework across two phases: Phase 1 (core framework, 18 tasks) and
Phase 2 (transformer stack, 15 tasks).

---

## The Swarm Configuration

```yaml
Roles:
  orchestrator (1)  — decomposes the goal into tasks, builds DAG
  impl workers (5)  — parallel code writers with shared read-only workspace
  judge (1)         — reviews work quality (threshold 0.75)
  merger (1)        — applies non-conflicting changes

Settings:
  max tasks: 50, max depth: 15
  task timeout: 1200s (20 min per task)
  max retries: 5 per task
  budget: $50 USD
```

---

## Phase 1: Core Framework (18 tasks, ~35K lines)

### Decomposition

The orchestrator took the goal document and produced 18 tasks in
**~4.5 minutes**. It scored **0.80/1.00** — a good decomposition with
only 4 warnings (mostly about file existence for to-be-created files).

The DAG had **10 levels** deep — meaning 10 sequential waves at minimum,
even with perfect parallelism. The tasks mapped directly to the goal's
18-task breakdown (the orchestrator followed the human-designed task
structure closely).

### The DAG

```
WAVE 1:  task-1 (storage/shape)
         task-2 (allocators)
         task-3 (SIMD NEON)        ← retried 4 times
         task-4 (SIMD AVX2)        ← retried 1 time

WAVE 2:  task-5 (tensor core)

WAVE 3:  task-7 (elementwise)
         task-8 (reduce/softmax)   ← retried 1 time

WAVE 4:  task-6 (tiled matmul)     ← retried 3 times (complex!)

WAVE 5:  task-9 (conv2d/pooling)   ← retried 1 time

WAVE 6:  task-10 (FFI exports)     ← retried 1 time

WAVE 7:  task-11 (Rust FFI bridge)

WAVE 8:  task-12 (autograd core)
         task-15 (data pipeline)

WAVE 9:  task-13 (autograd ops)    ← retried 1 time
         task-14 (optimizers)
         task-16 (graph optim)     ← retried 1 time

WAVE 10: task-17 (NN layers)
         task-18 (training + examples)
```

### Key Observations

**Parallel efficiency: 24%** — low, and the swarm itself flagged this:
> "Low parallel efficiency — review dependency graph for unnecessary
> sequential constraints"

This happened because the dependency chain was very deep (10 levels).
Many tasks had hard sequential dependencies: you can't build the tensor
core without storage, can't build matmul without SIMD, can't build
the FFI bridge without all Zig ops, etc.

**26 concurrency adjustments** — the orchestrator dynamically adjusted
how many agents ran in parallel, likely based on resource availability
and task completion patterns.

**8 tasks needed retries** (out of 18):
| Task | What | Retries | Likely cause |
|------|------|---------|-------------|
| task-3 | SIMD NEON | 4 | Complex SIMD intrinsics, agent timeout |
| task-4 | SIMD AVX2 | 1 | Similar complexity to NEON |
| task-6 | Tiled matmul | 3 | Hardest task — GOTO BLAS tiled GEMM |
| task-8 | Reduce/softmax | 1 | Welford algorithm complexity |
| task-9 | Conv2d | 1 | im2col + GEMM orchestration |
| task-10 | FFI exports | 1 | Many functions to export |
| task-13 | Autograd ops | 1 | Many backward implementations |
| task-16 | Graph optim | 1 | Complex fusion pattern matching |

The pattern: **Zig SIMD and low-level kernel tasks retry the most.**
task-3 (NEON intrinsics) took 4 retries — this was the single hardest
task for agents. Writing SIMD intrinsics with correct tail handling,
special values, and platform-specific register usage is genuinely
difficult even for humans.

**17 batches** — the orchestrator sent work in 17 waves. Early batches
were larger (4 tasks in batch 1), later ones were mostly single tasks
as the dependency chain narrowed.

**Total duration: 13,570s (~3.8 hours)** for ~35K lines of Zig + Rust.
That's about 2.6 lines per second of working time, including all
retries and sequential waiting.

---

## Phase 2: Transformer Stack (15 tasks, ~13K lines)

### Decomposition

15 tasks in **4.4 minutes**. Scored **0.55/1.00** — lower than Phase 1.
Why? **9 decomposition warnings**, including:

- 32 total issues detected
- Multiple **file overlap warnings**: tasks targeting the same files
  without dependency edges
  - task-6 and task-7 both write to `ops/mod.rs`
  - task-8, task-10, and task-11 all write to `synapse-nn/src/lib.rs`
  - task-9 and task-10 both write to `synapse-nn/src/lib.rs`

These overlaps are what triggered AST conflict detection during execution.

### The DAG

```
WAVE 1:  task-1 (LayerNorm kernel)    ← retried 1 time
         task-2 (attention kernel)    ← retried 1 time
         task-3 (RoPE kernel)

WAVE 2:  task-4 (Zig FFI exports)

WAVE 3:  task-5 (Rust FFI bindings)

WAVE 4:  task-6 (autograd attention)  ← serialized (AST conflict)
         task-7 (autograd LN/RoPE)    ← serialized (AST conflict)

WAVE 5:  task-8 (positional encoding) ← serialized (AST conflict)
         task-10 (LayerNorm module)   ← serialized (AST conflict)

WAVE 6:  task-9 (MultiHeadAttention)

WAVE 7:  task-11 (transformer blocks)
         task-12 (text utilities)

WAVE 8:  task-13 (graph fusion)
         task-14 (examples)

WAVE 9:  task-15 (E2E tests)         ← retried 3 times
```

### AST Conflict Detection — The Swarm Is Swarming

This is the most interesting behavior. The orchestrator made two
**parallel_safety_split** decisions:

**Decision 1:**
```
Serialized 2 tasks due to AST conflicts
  Parallel: []
  Serialized: ['task-6', 'task-7']
```

Both task-6 (autograd attention backward) and task-7 (autograd
LayerNorm/RoPE backward) need to modify `synapse-autograd/src/ops/mod.rs`
to register their new modules. If they ran in parallel, both would
try to add `pub mod attention;` and `pub mod layernorm;` to the same
file, creating a merge conflict.

The orchestrator detected this via AST analysis — it parsed the target
files, identified that both tasks would modify the same AST nodes
(module declarations), and forced them to run sequentially.

**Decision 2:**
```
Serialized 2 tasks due to AST conflicts
  Parallel: []
  Serialized: ['task-8', 'task-10']
```

Same pattern: task-8 (positional encoding) and task-10 (LayerNorm module)
both modify `synapse-nn/src/lib.rs` to add `pub mod positional;` and
`pub mod layernorm;`. Serialized to avoid conflict.

**Why this matters:** Without AST conflict detection, these tasks would
have run in parallel, both modified the same files, and the merger would
have had to resolve conflicts — possibly introducing bugs. The
orchestrator prevented this proactively.

### Retries

| Task | What | Retries | Notes |
|------|------|---------|-------|
| task-1 | Zig LayerNorm | 1 | SIMD kernel — consistent with Phase 1 pattern |
| task-2 | Zig attention | 1 | Fused kernel — complex memory tiling |
| task-15 | E2E tests | 3 | Integration test — depends on everything working together |

task-15 (E2E integration tests) retried 3 times — this makes sense
because it's the final integration task that must wire everything
together. If any previous task left rough edges, task-15 is where
they surface.

### Parallel Efficiency: 50%

Better than Phase 1's 24%, but still limited. The improvement came from:
- Fewer dependency levels (9 vs 10)
- More tasks that could run in parallel at early waves
- But AST conflict serialization ate into the parallelism

**12 concurrency adjustments** during execution.

**Total duration: 10,225s (~2.8 hours)** for ~13K lines.

---

## Comparing the Two Runs

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Tasks | 18 | 15 |
| Lines produced | ~35,000 | ~13,000 |
| Duration | 3.8 hours | 2.8 hours |
| Decomposition score | 0.80 | 0.55 |
| Parallel efficiency | 24% | 50% |
| Concurrency adjustments | 26 | 12 |
| Tasks needing retries | 8 (44%) | 3+2 serialized (33%) |
| Decomposition time | ~4.5 min | ~4.4 min |
| Batches | 17 | ~9 |

### What the orchestrator got right

1. **Task decomposition tracked the goal closely.** The human-written
   goal defined waves and task boundaries; the orchestrator mostly
   followed them. It didn't try to be clever — it respected the
   dependency structure.

2. **AST conflict detection prevented merge disasters.** Without it,
   tasks writing to the same `mod.rs` or `lib.rs` files would have
   conflicted. The orchestrator correctly identified these at the
   file-and-AST level, not just filename level.

3. **Retry strategy worked.** Every task that failed eventually
   succeeded within 5 attempts. No tasks were permanently blocked.

4. **Batch sizing adapted.** Early batches were larger (4 tasks in
   Phase 1 wave 1), shrinking as dependencies narrowed. This maximized
   early parallelism.

### What could be improved

1. **Parallel efficiency is low** (24-50%). The dependency chains are
   inherently deep for this kind of project (you can't write the
   training loop before the autograd engine), but the orchestrator
   could potentially identify more parallelizable work within waves.

2. **SIMD tasks consistently need retries.** The Zig SIMD kernel tasks
   (NEON intrinsics, tiled matmul, fused attention) are the hardest
   for agents. These might benefit from splitting into smaller subtasks
   or providing more reference code in the goal.

3. **Decomposition quality dropped in Phase 2** (0.80 → 0.55). The
   file overlap warnings show the orchestrator didn't fully track which
   files each task would modify. Better file-level dependency tracking
   during decomposition could prevent the need for runtime AST conflict
   serialization.

4. **The final integration task (task-15/18) always struggles.** In both
   phases, the last task needed retries. This is the "works on my machine"
   problem at scale — when everything needs to work together for the
   first time, integration issues surface.

---

## Signals That "The Swarm Is Swarming"

These are indicators of genuine multi-agent coordination, not just
sequential task execution:

1. **AST conflict serialization** — the orchestrator dynamically
   reorders tasks based on code-level analysis of what files and AST
   nodes each task will modify. This is runtime adaptation, not static
   planning.

2. **Concurrency adjustments (26 in Phase 1, 12 in Phase 2)** — the
   orchestrator is actively tuning how many agents run simultaneously
   based on system load and task characteristics.

3. **Batch spawning** — multiple agents launched simultaneously
   (4 in Phase 1 batch 1, parallel work in Phase 2), each writing
   independent code modules that later get merged.

4. **Automatic retries with reasoning** — when a task fails, the
   orchestrator decides to retry with adjusted parameters, not just
   blindly repeat.

5. **File version tracking** — the `versions.json` ledger tracks
   content hashes for every file across tasks, enabling conflict
   detection and safe merging.

6. **Critical path identification** — the postmortem identifies which
   tasks were on the critical path, useful for optimizing future runs.

---

## The Numbers

```
Total lines written by swarm:  ~48,000 (Phase 1 + Phase 2)
Total wall time:               ~6.6 hours
Total tasks:                   33
Tasks needing retries:         11 (33%)
Max retries for single task:   4 (SIMD NEON intrinsics)
Files created:                 ~170
Test files:                    ~30
All tests passing:             33/33 ✓
```
