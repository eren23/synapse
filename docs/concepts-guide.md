# Synapse Concepts Guide

A plain-language guide to the techniques, patterns, and math used across
all three phases of the Synapse framework. No Rust/Zig syntax — just the
ideas, why they matter, and how they connect.

---

## Table of Contents

1. [Tensors — The Universal Data Container](#1-tensors)
2. [SIMD — Making Math Go Fast](#2-simd)
3. [Memory Allocators — Arena, Pool, and Why malloc Is Slow](#3-memory-allocators)
4. [Matrix Multiplication — The One Operation That Rules ML](#4-matrix-multiplication)
5. [Automatic Differentiation — How Machines Learn](#5-automatic-differentiation)
6. [Neural Network Layers — Building Blocks](#6-neural-network-layers)
7. [Attention — The Transformer's Secret Weapon](#7-attention)
8. [Positional Encoding — Teaching Order to Transformers](#8-positional-encoding)
9. [Normalization — Keeping Numbers Well-Behaved](#9-normalization)
10. [Optimizers — How Weights Get Updated](#10-optimizers)
11. [The Transformer — Putting It All Together](#11-the-transformer)
12. [KV-Cache — Why Inference Is Different From Training](#12-kv-cache)
13. [Quantization — Shrinking Models Without Losing Much](#13-quantization)
14. [Grouped Query Attention — A Modern Efficiency Trick](#14-grouped-query-attention)
15. [SwiGLU — The Gated FFN Everyone Uses Now](#15-swiglu)
16. [Graph IR and Operator Fusion — Compiler Tricks for ML](#16-graph-ir-and-operator-fusion)
17. [FFI — Making Two Languages Talk](#17-ffi)
18. [Tokenization — From Text to Numbers](#18-tokenization)
19. [Sampling — How LLMs Pick the Next Word](#19-sampling)
20. [The Component Registry Pattern — One Engine, Many Models](#20-component-registry)

---

## 1. Tensors

**What it is:** A tensor is a multi-dimensional array of numbers. That's it.

| Name | Dimensions | Example |
|------|-----------|---------|
| Scalar | 0D | `42.0` |
| Vector | 1D | `[1.0, 2.0, 3.0]` — 3 elements |
| Matrix | 2D | 64 rows × 784 columns — a batch of 64 images |
| 3D Tensor | 3D | `[batch, sequence, features]` — text data |
| 4D Tensor | 4D | `[batch, channels, height, width]` — image data |

**Why it matters:** Every piece of data in ML — images, text, audio, weights,
gradients — is stored as a tensor. The entire framework is built around
efficiently creating, moving, and doing math on tensors.

**Shape and strides:** A tensor's *shape* is its dimensions (e.g., `[64, 784]`).
*Strides* tell you how many elements to skip to move one step along each
dimension. This trick allows operations like transpose and reshape without
copying any data — you just change the strides.

**Broadcasting:** When you add a `[64, 784]` tensor and a `[784]` tensor,
the smaller one is "broadcast" — logically repeated 64 times to match.
No actual copying happens; the math just pretends the data is repeated.

---

## 2. SIMD

**What it is:** Single Instruction, Multiple Data. Instead of adding two
numbers at a time, you add 4 (or 8, or 16) numbers in one CPU instruction.

```
Without SIMD (scalar):          With SIMD (NEON, 4-wide):
  a[0] + b[0] → c[0]             a[0..3] + b[0..3] → c[0..3]  ← one instruction
  a[1] + b[1] → c[1]
  a[2] + b[2] → c[2]
  a[3] + b[3] → c[3]
  (4 instructions)                (1 instruction, ~4x faster)
```

**ARM NEON:** Apple Silicon (M1/M2/M3/M4) uses NEON, which processes 4
floats at once (128-bit registers).

**x86 AVX2:** Intel/AMD chips use AVX2, processing 8 floats at once
(256-bit registers).

**Dispatch:** At startup, the code detects which CPU it's running on and
picks the fastest available instruction set. If neither NEON nor AVX2
is available, it falls back to plain scalar loops.

**Why it matters:** ML is dominated by elementwise operations on huge
arrays. SIMD gives you a 4-8x speedup for free on operations like
add, multiply, activation functions, and reductions.

---

## 3. Memory Allocators

**The problem:** In ML training, each step creates thousands of temporary
tensors (activations, gradients, intermediates). Calling `malloc` and
`free` thousands of times per step is slow because the OS has to find
free memory, update bookkeeping, and handle fragmentation.

### Arena Allocator

Think of it like a notepad. You write sequentially, page after page.
When you're done with the whole notepad, you flip back to page 1 and
start over. You never erase individual pages.

```
Step 1: [tensor1][tensor2][tensor3][tensor4]...
         ↑ bump pointer moves forward

Step 2: [reset to start]
         ↑ O(1) — just move pointer back to beginning
```

**Why it's fast:** Allocation is just incrementing a pointer (O(1)).
Deallocation is resetting one number (O(1)). No fragmentation, no
bookkeeping per object.

**Used for:** Temporary tensors during a training step. Allocate forward,
reset at step end.

### Pool Allocator

Pre-allocates a fixed number of same-sized slots. Like a parking garage
with numbered spots — acquiring a spot is instant (grab from free list),
releasing is instant (add back to free list).

**Used for:** Tensor buffers that are always the same size (e.g., layer
activations that are always `[batch, hidden_size]`).

### Aligned Allocator

SIMD instructions require data to be aligned to specific boundaries
(e.g., 64 bytes). If your data starts at a random memory address, SIMD
either crashes or falls back to slow paths. The aligned allocator
ensures every allocation starts at a 64-byte boundary.

---

## 4. Matrix Multiplication

**Why it's the most important operation:** Almost everything in neural
networks boils down to multiplying matrices:

- Linear layer: `output = input × weights`
- Attention scores: `scores = queries × keys^T`
- Convolution (via im2col): rearrange patches into a matrix, then multiply

A single forward pass through a 28-layer model does ~100+ matrix multiplications.
Speed here determines everything.

### Tiled (Blocked) GEMM

Naive matrix multiply has terrible cache behavior — it jumps around
memory randomly. Tiled GEMM splits the matrices into small blocks
(tiles) that fit in CPU cache:

```
Instead of:                  Tiled approach:
  For each row of A:           For each 8×8 block:
    For each col of B:           Load A-tile into L1 cache
      dot product                Load B-tile into L1 cache
  (cache misses everywhere)      Multiply (everything in cache!)
```

**L1/L2/L3 blocking:** The tiles are sized to fit specific cache levels:
- L1 (32-64 KB): 8×8 micro-kernel — the innermost loop
- L2 (256 KB-1 MB): MC×KC macro-tiles — medium blocks
- L3 (4-32 MB): NC partition — outer blocking

**Packing:** Before multiplying, the tiles are copied into contiguous
memory in the order they'll be accessed. This costs a little time upfront
but eliminates all cache misses during the actual multiply.

**GOTO BLAS approach:** Named after Kazushige Goto, this is the standard
algorithm used by every high-performance math library. Our Zig implementation
follows this pattern.

---

## 5. Automatic Differentiation

**The big idea:** Neural networks learn by adjusting their weights to
reduce error. To know *which direction* to adjust each weight, you need
the *gradient* — how much the error changes when you wiggle that weight.

### Forward Pass vs Backward Pass

```
Forward pass (compute output):
  input → [Layer 1] → [Layer 2] → [Layer 3] → loss

Backward pass (compute gradients):
  loss → [Layer 3 grad] → [Layer 2 grad] → [Layer 1 grad] → weight updates
```

### The Chain Rule

If `loss = f(g(h(x)))`, then:
```
d(loss)/dx = f'(g(h(x))) × g'(h(x)) × h'(x)
```

Each layer knows its own derivative. Backpropagation chains them together,
multiplying backwards through the network. This is the chain rule from
calculus, applied mechanically.

### Computation Graph

Each operation records itself in a graph (like a recipe):
```
a ──┐
    ├── mul ── z ── add ── loss
b ──┘              │
c ─────────────────┘
```

To compute gradients, walk this graph backwards. At each node, apply
the chain rule.

### Why It's Called "Automatic"

You don't write gradient formulas by hand. Each operation (add, multiply,
matmul, relu, etc.) comes with a pre-written backward function. The
framework chains them together automatically.

**GradFn trait:** Every operation implements a `backward()` function
that, given the gradient flowing in from above, computes the gradient
for each of its inputs.

---

## 6. Neural Network Layers

### Linear (Dense / Fully-Connected)

The simplest layer: `output = input × weight + bias`

Takes a vector of N numbers, produces a vector of M numbers.
The weight matrix `[N, M]` is what the network learns.

### Activation Functions

Without activations, stacking linear layers is pointless — multiple
linear transforms collapse into a single linear transform. Activations
add non-linearity, which is what lets networks learn complex patterns.

| Function | Formula | When to use |
|----------|---------|-------------|
| ReLU | `max(0, x)` | Default for hidden layers. Dead simple, fast. |
| Sigmoid | `1 / (1 + e^(-x))` | Squashes to [0, 1]. Used for probabilities. |
| Tanh | `(e^x - e^(-x)) / (e^x + e^(-x))` | Squashes to [-1, 1]. |
| GELU | `x × Φ(x)` (Gaussian CDF) | GPT-2 used this. Smooth version of ReLU. |
| SiLU / Swish | `x × sigmoid(x)` | Modern default (LLaMA, Qwen). Smooth, no dead zone. |
| Softmax | `e^(xi) / Σ e^(xj)` | Converts logits → probabilities (sums to 1). |

### Dropout

During training, randomly zero out a fraction of neurons (e.g., 20%).
This prevents the network from relying too much on any single neuron,
which reduces overfitting. During inference, dropout is turned off.

### Embedding

Converts integer IDs (like token IDs) into dense vectors. It's just a
lookup table: `embedding[token_id]` returns a vector of `hidden_size` floats.
The table values are learned during training.

---

## 7. Attention

**The problem attention solves:** In a sequence (like a sentence), each
word should be able to "look at" other words to understand context.
"The bank of the river" vs "The bank approved the loan" — "bank" needs
to see surrounding words to know its meaning.

### Scaled Dot-Product Attention

Three vectors per position:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

```
1. Compute attention scores:  scores = Q × K^T / √(d_head)
2. Apply softmax:             weights = softmax(scores)
3. Weighted sum of values:    output = weights × V
```

The `/ √(d_head)` scaling prevents the dot products from getting too
large, which would make softmax produce nearly-one-hot outputs (killing
gradient flow).

### Multi-Head Attention (MHA)

Instead of one big attention, split Q/K/V into multiple "heads" (e.g., 8).
Each head attends to different aspects of the input independently. Then
concatenate and project back.

```
Input [batch, seq, 512]
  → split into 8 heads of [batch, seq, 64]
  → each head does its own attention
  → concatenate back to [batch, seq, 512]
  → one more linear projection
```

**Why:** Different heads learn to attend to different things — one might
focus on syntax, another on semantics, another on position.

### Causal Masking

For language models (generating text left-to-right), each position can
only attend to positions before it (and itself). You can't peek at the
future. This is enforced by setting future positions to `-infinity`
before softmax, which makes their attention weights exactly 0.

```
Position:  1  2  3  4
    1:     ✓  ✗  ✗  ✗
    2:     ✓  ✓  ✗  ✗
    3:     ✓  ✓  ✓  ✗
    4:     ✓  ✓  ✓  ✓
```

### Fused Attention

Normally: matmul → scale → mask → softmax → matmul, each writing
intermediate results to memory. Fused attention does all steps in one
pass, keeping intermediates in fast CPU cache. Same output, ~3x faster.

---

## 8. Positional Encoding

**The problem:** Attention treats inputs as a *set*, not a *sequence*.
"dog bites man" and "man bites dog" would produce identical outputs
without positional information.

### Sinusoidal (Original Transformer)

Add a fixed pattern of sine and cosine waves at different frequencies
to the input. Each position gets a unique "fingerprint" of wave values.

### Rotary Position Embedding (RoPE)

Instead of adding position to the input, RoPE *rotates* the query and
key vectors by an angle proportional to their position.

```
At position p, rotate dimension pairs by angle p × θ:
  [x₀, x₁] → [x₀ cos(pθ) - x₁ sin(pθ), x₀ sin(pθ) + x₁ cos(pθ)]
```

**Why it's better:** The dot product between a query at position i and
a key at position j depends only on the *relative distance* (i-j), not
absolute positions. This means the model generalizes better to different
sequence lengths.

**RoPE theta:** The base frequency. Higher theta = the rotation
frequencies spread over more positions. Qwen3 uses θ=1,000,000 (very
high), which helps with long contexts.

### Multi-Dimensional RoPE (mRoPE)

Qwen 3.5 uses this for multimodal (vision + text). Instead of one set
of rotations, the dimensions are split into sections (e.g., [11, 11, 10])
with separate rotations for height, width, and temporal dimensions.

### Partial Rotary Factor

Qwen 3.5 only applies RoPE to 25% of dimensions (`partial_rotary_factor=0.25`).
The rest keep their absolute position information. A compromise between
relative and absolute position encoding.

---

## 9. Normalization

**The problem:** During training, each layer's input distribution shifts
as previous layers' weights change. This makes training unstable and slow.
Normalization keeps the numbers well-behaved.

### Batch Normalization (BatchNorm)

Normalize across the *batch* dimension: for each feature, compute
mean and variance across all samples in the batch, then normalize.

```
output = (x - mean_batch) / sqrt(var_batch + eps) × gamma + beta
```

**Limitation:** Depends on batch size. Doesn't work well for sequences
or inference (batch=1).

### Layer Normalization (LayerNorm)

Normalize across the *feature* dimension: for each sample, compute
mean and variance across all features.

```
output = (x - mean_features) / sqrt(var_features + eps) × gamma + beta
```

**Used by:** GPT-2, BERT, Phase 2 transformers.

### RMS Normalization (RMSNorm)

Like LayerNorm but simpler — no mean subtraction, just divide by the
root-mean-square:

```
output = x / sqrt(mean(x²) + eps) × gamma
```

**Why everyone uses it now:** Slightly faster than LayerNorm (saves one
reduction operation — no mean computation), works just as well in practice.
Used by LLaMA, Qwen3, Mistral — basically all modern LLMs.

### Welford's Algorithm

Computing variance naively requires two passes: first compute the mean,
then compute variance using that mean. Welford's algorithm does it in
*one pass*, updating mean and variance incrementally as it reads each
element. This halves the memory traffic.

---

## 10. Optimizers

After computing gradients, optimizers decide how to update the weights.

### SGD (Stochastic Gradient Descent)

Simplest: `weight -= learning_rate × gradient`

With momentum: remember the previous update direction and add a fraction
of it to the current update. Like a ball rolling downhill that maintains
some velocity.

### Adam

The most popular optimizer. Maintains two running averages per weight:
- **First moment (m):** running average of gradients (which direction)
- **Second moment (v):** running average of squared gradients (how bumpy)

Update: `weight -= lr × m / (sqrt(v) + eps)`

This automatically adapts the learning rate for each weight — weights
with large, consistent gradients get smaller updates; weights with
small, noisy gradients get larger updates.

### Learning Rate Schedulers

The learning rate typically changes during training:
- **Warmup:** Start very small, increase linearly for the first N steps.
  Prevents the model from diverging early when weights are random.
- **Cosine decay:** After warmup, decrease the learning rate following
  a cosine curve down to near-zero.
- **Step decay:** Drop the learning rate by a factor every N epochs.

### Gradient Clipping

If gradients get too large (exploding gradients), training becomes
unstable. Gradient clipping scales down the gradient vector if its
norm exceeds a threshold. Like putting a speed limit on weight updates.

---

## 11. The Transformer

The architecture behind GPT, BERT, LLaMA, Qwen, and essentially every
modern language model. It's a stack of identical layers.

### One Transformer Layer

```
input
  │
  ├─→ Normalize (RMSNorm)
  │     │
  │     ↓
  │   Multi-Head Attention ← Q, K, V all come from same input (self-attention)
  │     │
  │◄────┘  (add residual — skip connection)
  │
  ├─→ Normalize (RMSNorm)
  │     │
  │     ↓
  │   Feed-Forward Network (SwiGLU)
  │     │
  │◄────┘  (add residual — skip connection)
  │
  ↓
output (same shape as input)
```

### Residual Connections (Skip Connections)

The `+` in `output = x + attention(norm(x))`. The input bypasses the
attention block and gets added back. This is critical for training
deep networks because:
- Gradients can flow directly through the skip path
- At initialization (random weights), the layer is approximately an
  identity function — it starts by doing nothing and gradually learns

### Pre-Norm vs Post-Norm

- **Post-norm (original):** `output = norm(x + attention(x))`
- **Pre-norm (modern):** `output = x + attention(norm(x))`

Pre-norm is more stable for training deep networks. All modern LLMs
(GPT-2+, LLaMA, Qwen) use pre-norm.

### Complete Causal Language Model

```
Token IDs [batch, seq]
  → Embedding lookup [batch, seq, hidden]
  → + Positional encoding (or RoPE applied inside attention)
  → Transformer Layer 1
  → Transformer Layer 2
  → ...
  → Transformer Layer N
  → Final RMSNorm
  → Linear projection to vocabulary [batch, seq, vocab_size]
  → Softmax → probabilities for next token
```

**Qwen3-0.6B:** 28 layers, hidden=1024, 16 attention heads, vocab=151,936.
That's it. The magic is in the data and training, not the architecture.

---

## 12. KV-Cache

**The problem with autoregressive generation:** To generate token N,
the model needs to attend to all previous tokens (1 through N-1).
Without caching, generating 100 tokens means:

```
Token 1: forward pass over [1]           = 1 computation
Token 2: forward pass over [1, 2]        = 2 computations
Token 3: forward pass over [1, 2, 3]     = 3 computations
...
Token 100: forward pass over [1..100]    = 100 computations
Total: 1 + 2 + ... + 100 = 5,050 computations  (O(n²))
```

### The Fix

The Key and Value projections for tokens 1 through N-1 don't change
when you add token N. So cache them:

```
Token 1: compute K₁, V₁, store in cache. Full attention.
Token 2: compute K₂, V₂, append to cache. Attend to [K₁K₂], [V₁V₂].
Token 3: compute K₃, V₃, append to cache. Attend to [K₁K₂K₃], [V₁V₂V₃].
...
Token 100: compute K₁₀₀, V₁₀₀, append. Attend to all cached.
Total: 100 single-token computations  (O(n))
```

**Prefill vs Decode:**
- **Prefill:** Process the entire prompt in one forward pass (like training).
  Fast because you can do full parallel attention.
- **Decode:** Generate one token at a time, using KV-cache. Each step is
  small but sequential.

**Memory cost:** For Qwen3-0.6B with 2048 tokens:
```
Per layer: 2 × 2048 × 8 KV-heads × 128 head_dim × 4 bytes = 16 MB
28 layers: 28 × 16 MB = 448 MB just for KV-cache
```

This is why max sequence length matters — longer sequences need more
cache memory.

---

## 13. Quantization

**The idea:** Neural network weights are stored as 32-bit floats (4 bytes
each). But they don't need that precision. Storing them as 8-bit integers
(1 byte each) cuts memory by 4x and speeds up computation because
integer math is faster.

### How INT8 Works

For each group of weights (e.g., one row of a matrix):
1. Find the max absolute value: `scale = max(|w|) / 127`
2. Quantize: `w_int8 = round(w / scale)`
3. Store the int8 values + the scale factor

To use the weight: `w_approx = w_int8 × scale` (dequantize)

### Per-Channel Quantization

Instead of one scale for the entire matrix, use one scale per row
(or column). This is more accurate because different rows can have
very different value ranges.

### Weight-Only Quantization

The simplest approach (what we implement):
- **Weights:** stored as INT8
- **Activations:** remain as f32
- **Computation:** dequantize weights on-the-fly, or use INT8 GEMM
  and scale the output

This gives ~2x memory reduction and ~2x speedup with minimal accuracy loss.

### More Aggressive Quantization (Future)

| Format | Bits | Memory | Accuracy Loss |
|--------|------|--------|---------------|
| f32 | 32 | Baseline | None |
| f16/bf16 | 16 | 2x less | Negligible |
| INT8 | 8 | 4x less | <1% typically |
| INT4 | 4 | 8x less | 1-3% |
| GPTQ/AWQ | 4 | 8x less | <1% (smarter quantization) |

---

## 14. Grouped Query Attention (GQA)

**Standard MHA:** Each attention head has its own Q, K, and V projections.
With 16 heads, that's 16 Q matrices, 16 K matrices, 16 V matrices.

**The memory problem:** During inference, the KV-cache stores K and V
for every head. With 16 heads and long sequences, this gets huge.

**GQA solution:** Share K and V across groups of Q heads.

```
Standard MHA (16 heads):
  Q: 16 heads    K: 16 heads    V: 16 heads
  Q₁→K₁,V₁  Q₂→K₂,V₂  ... Q₁₆→K₁₆,V₁₆

GQA (16 Q heads, 8 KV heads):
  Q: 16 heads    K: 8 heads     V: 8 heads
  Q₁,Q₂→K₁,V₁   Q₃,Q₄→K₂,V₂   ...  Q₁₅,Q₁₆→K₈,V₈
```

**The extreme:** MQA (Multi-Query Attention) uses just 1 KV head shared
by all Q heads. Very memory efficient but slightly less capable.

**Qwen3-0.6B:** 16 Q heads, 8 KV heads (2 Q heads per KV group).
This halves the KV-cache memory compared to full MHA.

---

## 15. SwiGLU

**Standard FFN (GPT-2 era):**
```
output = W₂ × GELU(W₁ × input)
```
Two weight matrices, one activation.

**SwiGLU (modern):**
```
output = W_down × (SiLU(W_gate × input) ⊙ (W_up × input))
```

Three weight matrices. The `W_gate` path goes through SiLU activation,
then element-wise multiplies (`⊙`) with the `W_up` path. This "gating"
mechanism lets the network learn to selectively pass or block information.

**Why it works better:** The gate can learn to zero out irrelevant
features, making the FFN more selective. Empirically, models trained
with SwiGLU converge to better quality than standard FFN.

**The cost:** One extra weight matrix (3 instead of 2). To keep total
parameters similar, the intermediate dimension is typically set to
`2/3 × 4 × hidden_size` instead of `4 × hidden_size`. Qwen3-0.6B:
hidden=1024, intermediate=3072 (which is 3 × 1024, matching this ratio).

---

## 16. Graph IR and Operator Fusion

### What's a Graph IR?

IR = Intermediate Representation. Before executing operations, the
framework builds a graph of what needs to be computed:

```
Input → MatMul → Add(bias) → ReLU → MatMul → Softmax → Output
```

This graph is like a recipe. Before cooking, you can optimize the recipe.

### Operator Fusion

Some operations can be combined into one, avoiding intermediate memory
writes:

**Before fusion (3 operations, 2 intermediate buffers):**
```
temp1 = matmul(x, w)       ← write to memory
temp2 = temp1 + bias       ← read from memory, write to memory
output = relu(temp2)       ← read from memory, write to memory
```

**After fusion (1 fused operation, 0 intermediate buffers):**
```
output = fused_matmul_bias_relu(x, w, bias)  ← everything in registers
```

This matters because memory bandwidth (not compute) is usually the
bottleneck. Fusion reduces memory traffic.

### Other Optimization Passes

- **Constant folding:** If both inputs to an operation are known at
  compile time, compute the result once and store it.
- **Dead code elimination:** Remove operations whose outputs are never
  used by anything.
- **Memory scheduling:** Reorder operations to minimize peak memory
  usage (process nodes that free the most memory first).

### Conv + BatchNorm Fusion

A classic optimization. During inference, BatchNorm is just a linear
transform: `y = gamma × (x - mean) / sqrt(var + eps) + beta`. This
can be folded directly into the convolution weights, eliminating
the BatchNorm operation entirely.

---

## 17. FFI (Foreign Function Interface)

**The problem:** Zig writes the fast math kernels. Rust writes the
safe ML framework. They need to talk to each other.

**The solution:** The C ABI (Application Binary Interface). Both Zig
and Rust can speak C. So:

1. Zig exports functions with C calling convention
2. A C header file (`synapse.h`) declares the function signatures
3. Rust declares matching `extern "C"` functions
4. At link time, Rust calls Zig functions directly (no runtime overhead)

### Opaque Handles

Zig-owned objects (tensors, allocators) are passed to Rust as opaque
pointers — Rust can't see inside them, only pass them back to Zig
functions. This is safe because:
- Zig manages the memory
- Rust can't corrupt Zig's internal state
- If Zig changes its struct layout, Rust doesn't care

### Error Handling Across FFI

Zig uses error unions. Rust uses `Result`. C uses neither. So at the
FFI boundary, all errors become integer status codes:
```
SYN_OK = 0, SYN_ERR_ALLOC = 1, SYN_ERR_SHAPE = 2, ...
```
Rust converts these back to `Result::Err(SynapseError::...)`.

**The golden rule:** No panics cross the FFI boundary. A Zig panic
during a Rust call would crash the process with no recovery. So every
FFI function catches errors and returns status codes.

---

## 18. Tokenization

**The problem:** Neural networks work with numbers, not text. Tokenization
converts text into a sequence of integer IDs.

### Word-Level (Naive)

Split on spaces. "Hello world" → ["Hello", "world"] → [42, 137]

**Problem:** "unhappiness" is one token. New words that aren't in the
vocabulary become `<UNK>`. Vocabulary is huge (millions of words).

### Byte-Pair Encoding (BPE)

Start with individual characters, then iteratively merge the most
frequent pair:

```
Training:
  Corpus: "low lower lowest"
  Step 1: most frequent pair is ('l', 'o') → merge to 'lo'
  Step 2: most frequent pair is ('lo', 'w') → merge to 'low'
  Step 3: most frequent pair is ('e', 'r') → merge to 'er'
  ... (repeat until desired vocab size)

Encoding "lowest":
  → ['low', 'est']     (if 'low' and 'est' are in vocabulary)
  → [487, 1823]        (look up IDs)
```

**Why BPE is great:**
- Common words become single tokens ("the", "is")
- Rare words get split into subwords ("unforgettable" → "un", "forget", "table")
- No `<UNK>` — everything can be expressed as a sequence of known subwords
- Fixed vocabulary size (e.g., 32K, 128K, 152K tokens)

**Qwen3 vocabulary:** 151,936 tokens. Includes regular text, code,
multilingual characters, and special tokens.

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<PAD>` | Padding shorter sequences to equal length |
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence (signals "stop generating") |
| `<UNK>` | Unknown token (fallback) |
| `<|im_start|>` | Qwen chat: start of message |
| `<|im_end|>` | Qwen chat: end of message |

---

## 19. Sampling

After the model produces logits (raw scores for each token in the
vocabulary), we need to pick the next token. This is where sampling
strategies come in.

### Greedy (argmax)

Pick the token with the highest score. Deterministic — same input
always gives same output. Simple but can be repetitive and boring.

### Temperature

Divide logits by temperature T before softmax:
- `T = 0`: equivalent to greedy (all probability on top token)
- `T = 1`: original distribution
- `T > 1`: flatter distribution (more random, more creative)
- `T < 1`: sharper distribution (more focused, more predictable)

### Top-K

Keep only the K highest-scoring tokens, set the rest to zero, then
sample from the remaining K. Prevents very unlikely tokens from ever
being chosen.

### Top-P (Nucleus Sampling)

Sort tokens by probability, keep tokens until cumulative probability
reaches P. If the top token already has 95% probability, only that
one is kept. If probabilities are spread out, many tokens are kept.
More adaptive than fixed Top-K.

### Repetition Penalty

Divide the logits of tokens that already appeared in the output by
a penalty factor (e.g., 1.1). Discourages the model from repeating
itself.

### Typical Pipeline

```
logits
  → temperature scaling (T=0.7)
  → top-k filtering (K=50)
  → top-p filtering (P=0.9)
  → repetition penalty (1.1)
  → softmax → probabilities
  → sample from distribution
  → next token
```

---

## 20. The Component Registry Pattern

**The problem:** We want one inference engine that runs Qwen3, LLaMA,
Mistral, and future models. But they use different attention types,
normalizations, FFN variants, etc.

**The solution:** Define a trait (interface) for each component, then
pick the implementation at runtime based on a config file.

```
trait NormVariant {
    fn forward(&self, input: &Tensor) -> Tensor;
}

struct RMSNorm { weight, eps }       // implements NormVariant
struct LayerNorm { weight, bias, eps } // implements NormVariant
```

The `DecoderLayer` doesn't know or care which norm it's using:
```
struct DecoderLayer {
    norm: Box<dyn NormVariant>,  // could be RMSNorm or LayerNorm
    attention: Box<dyn AttentionVariant>,  // could be GQA, MHA, or SlidingWindow
    ffn: Box<dyn FFNVariant>,  // could be SwiGLU, GELU, or GeGLU
}
```

**Adding a new model:**
1. If it uses the same components (RMSNorm, GQA, SwiGLU) → just write
   a config JSON with the right parameters. Zero new code.
2. If it uses a new component (e.g., linear attention) → implement the
   trait, register it in the factory. Everything else stays the same.

**Adding a new component type** (like linear attention for Qwen 3.5):
```
struct LinearAttention { ... }  // implements AttentionVariant

// Register in factory:
"linear_attention" => Box::new(LinearAttention::new(config))
```

That's it. The decoder layer, generation pipeline, KV-cache, and everything
else works unchanged because they only interact with components through
the trait interface.

**This is why we call it "one engine, many models."**

---

## Glossary

| Term | Meaning |
|------|---------|
| **Autograd** | Automatic differentiation — computing gradients without manual math |
| **Batch** | Multiple samples processed together for efficiency |
| **bf16** | Brain Float 16 — 16-bit float with same exponent range as f32, less precision |
| **Causal** | Can only look at past, not future (left-to-right generation) |
| **d_model** | Hidden size / embedding dimension |
| **d_ff** | Feed-forward network intermediate size |
| **d_head** | Dimension per attention head (d_model / num_heads) |
| **Decoder** | In LLMs, the part that generates text token by token |
| **Encoder** | Processes the full input at once (BERT, not GPT) |
| **Epoch** | One full pass through the training dataset |
| **FFN** | Feed-Forward Network — the non-attention part of a transformer layer |
| **GeLU** | Gaussian Error Linear Unit activation |
| **GEMM** | General Matrix Multiplication |
| **GGUF** | GPT-Generated Unified Format — llama.cpp's weight format |
| **GQA** | Grouped Query Attention |
| **Inference** | Running a trained model to get predictions (no learning) |
| **KV-Cache** | Stored Key/Value tensors from previous tokens |
| **Logits** | Raw model output before softmax (unnormalized scores) |
| **Loss** | How wrong the model's output is (lower = better) |
| **LR** | Learning rate — how big each weight update step is |
| **MHA** | Multi-Head Attention |
| **MoE** | Mixture of Experts — multiple FFNs, router picks which to use |
| **mRoPE** | Multi-dimensional Rotary Position Embedding |
| **MTP** | Multi-Token Prediction — predicting N next tokens at once |
| **Prefill** | Processing the entire input prompt in one pass |
| **RMSNorm** | Root Mean Square Normalization |
| **RoPE** | Rotary Position Embedding |
| **Safetensors** | HuggingFace's safe, fast weight file format |
| **SiLU** | Sigmoid Linear Unit = x × sigmoid(x) |
| **SwiGLU** | SiLU-gated GLU (Gated Linear Unit) |
| **Token** | A subword unit (roughly 0.75 words per token for English) |
| **Vocab** | The set of all tokens the model knows |
