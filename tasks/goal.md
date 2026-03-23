# Swarm Goal

Synapse — High-Performance Neural Network Training Framework (Zig + Rust)

Build a complete neural network training framework with a Zig-powered tensor
engine (SIMD-vectorized ops, custom allocators, comptime shape checking) and
a Rust ML layer (autograd, layers, optimizers, data pipeline, training loop)
connected via C ABI FFI. **Synapse** provides hand-tuned SIMD kernels for
ARM NEON and x86 AVX2, tiled matrix multiplication, arena/pool memory
allocators, reverse-mode automatic differentiation, operator fusion, and a
PyTorch-like training API. Implemented as a Zig static library (~12,000 lines)
plus a Rust workspace with eight crates (~18,000 lines), targeting ~30,000
total lines with comprehensive unit tests, benchmark tests with hard pass/fail
performance thresholds, and end-to-end training examples.

**CRITICAL RULE: Every task MUST include its own tests. No implementation
without tests. Every benchmark MUST have a hard pass/fail threshold. If a
benchmark does not meet its threshold, the task FAILS.**

---

## 0) Project Overview

Synapse is a from-scratch neural network framework optimized for raw
performance on modern hardware. The Zig layer handles all numerically
intensive work (tensor storage, SIMD-vectorized operations, cache-optimized
matrix multiplication, custom memory allocators), while the Rust layer
provides safe, ergonomic abstractions for building and training neural
networks (automatic differentiation, layer modules, optimizers, data loading).

### Why Zig + Rust?

- **Zig comptime**: Compile-time tensor shape checking catches dimension
  mismatches before runtime. Comptime generics allow loop unrolling and
  bounds-check elimination in hot kernels.
- **Zig SIMD intrinsics**: Direct access to ARM NEON and x86 AVX2 vector
  instructions without library dependencies.
- **Zig allocators**: First-class custom allocators (arena for per-step
  allocation, pool for fixed-size tensors) with leak detection.
- **Rust safety**: Safe autograd engine with Arc-based tensor ownership,
  trait-based module system, fearless concurrency in data loading.
- **FFI bridge**: Clean C ABI boundary with opaque handles and status codes,
  no panics crossing the FFI.

### Sample Usage (Rust API)

```rust
use synapse::prelude::*;

fn main() -> Result<()> {
    // Create tensors (backed by Zig SIMD-aligned storage)
    let x = Tensor::randn(&[64, 784], DType::F32)?;  // batch of 64, 784 features
    let y = Tensor::zeros(&[64, 10], DType::F32)?;    // 10-class labels

    // Define a model
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true)?),
        Box::new(ReLU),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(256, 128, true)?),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, true)?),
    ]);

    // Optimizer and loss
    let mut optimizer = Adam::new(model.parameters(), AdamConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        weight_decay: 1e-4,
        ..Default::default()
    });
    let loss_fn = CrossEntropyLoss::new();

    // Training step (autograd handles backward pass)
    let input = Variable::new(x, true);
    let target = Variable::new(y, false);
    let output = model.forward(&input)?;
    let loss = loss_fn.forward(&output, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()?;

    println!("Loss: {}", loss.tensor().data::<f32>()[0]);
    Ok(())
}
```

### Sample Usage (Zig Kernel — Tiled MatMul)

```zig
const synapse = @import("synapse");
const Tensor = synapse.tensor.Tensor;
const matmul = synapse.ops.matmul;

pub fn benchmark_matmul() !void {
    var arena = synapse.alloc.ArenaAllocator.init(
        std.heap.page_allocator, 64 * 1024 * 1024,
    );
    defer arena.deinit();

    const a = try Tensor(f32).zeros(&arena.allocator(), &.{512, 512});
    const b = try Tensor(f32).zeros(&arena.allocator(), &.{512, 512});
    var c = try Tensor(f32).zeros(&arena.allocator(), &.{512, 512});

    // Tiled GEMM with 8x8 NEON micro-kernel, L1/L2 blocking
    matmul.sgemm(.{
        .M = 512, .N = 512, .K = 512,
        .alpha = 1.0, .beta = 0.0,
    }, a.data_ptr(), b.data_ptr(), c.data_ptr_mut());
}
```

---

## 1) Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              Synapse                         │
                     │                                             │
  Rust API ─────────►│  ┌────────────┐  ┌──────────┐  ┌─────────┐ │
                     │  │synapse-train│  │synapse-nn│  │syn-data │ │
                     │  └─────┬──────┘  └────┬─────┘  └────┬────┘ │
                     │        │              │              │      │
                     │  ┌─────▼──────────────▼──────┐      │      │
                     │  │    synapse-autograd        │      │      │
                     │  │  (Variable, GradFn, tape)  │      │      │
                     │  └─────────────┬─────────────┘      │      │
                     │        ┌───────┤                     │      │
                     │  ┌─────▼─────┐ │ ┌──────────────┐   │      │
                     │  │syn-optim  │ │ │ synapse-graph │   │      │
                     │  └───────────┘ │ │  (IR, fusion) │   │      │
                     │                │ └──────────────┘   │      │
                     │  ┌─────────────▼────────────────────▼────┐ │
                     │  │           synapse-core                 │ │
                     │  │    (Tensor<T>, DType, Shape, Error)    │ │
                     │  └───────────────────┬───────────────────┘ │
                     │                      │                      │
                     │  ┌───────────────────▼───────────────────┐ │
                     │  │          synapse-sys (FFI)             │ │
                     │  │    extern "C" { syn_* functions }      │ │
                     │  └───────────────────┬───────────────────┘ │
                     │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
                     │      C ABI BOUNDARY   │                     │
                     │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
                     │  ┌───────────────────▼───────────────────┐ │
                     │  │       libsynapse_zig.a (Zig)          │ │
                     │  │                                       │ │
                     │  │  ┌────────┐ ┌───────┐ ┌──────┐       │ │
                     │  │  │tensor/ │ │ alloc/│ │ simd/│       │ │
                     │  │  │storage │ │ arena │ │ neon │       │ │
                     │  │  │shape   │ │ pool  │ │ avx2 │       │ │
                     │  │  │view    │ │aligned│ │disp. │       │ │
                     │  │  └────────┘ └───────┘ └──────┘       │ │
                     │  │  ┌────────────────────────────┐       │ │
                     │  │  │           ops/             │       │ │
                     │  │  │  matmul  elementwise  conv │       │ │
                     │  │  │  reduce  softmax  batchnorm│       │ │
                     │  │  │  pool    transpose         │       │ │
                     │  │  └────────────────────────────┘       │ │
                     │  └───────────────────────────────────────┘ │
                     └─────────────────────────────────────────────┘
```

---

## 2) Directory Structure

```
synapse/
├── build.zig                          # Zig build system root
├── Cargo.toml                         # Rust workspace root
├── synapse.h                          # Generated C header (FFI boundary)
│
├── zig/                               # ~12,000 lines Zig
│   ├── build.zig                      # Zig library build config
│   ├── src/
│   │   ├── root.zig                   # Library entry point
│   │   │
│   │   ├── tensor/
│   │   │   ├── storage.zig            # Dense buffer, 64-byte aligned, ref-counting
│   │   │   ├── tensor.zig             # Tensor struct: shape, strides, offset, storage ptr
│   │   │   ├── shape.zig              # Comptime shape checking, broadcasting rules
│   │   │   ├── iterator.zig           # Strided iteration (contiguous + non-contiguous)
│   │   │   └── view.zig              # Reshape, transpose, slice (zero-copy views)
│   │   │
│   │   ├── alloc/
│   │   │   ├── arena.zig              # Arena allocator for training batches
│   │   │   ├── pool.zig               # Fixed-size slab pool for tensor buffers
│   │   │   ├── aligned.zig            # SIMD-aligned allocation (64-byte)
│   │   │   └── tracking.zig           # Allocation tracking + leak detection
│   │   │
│   │   ├── simd/
│   │   │   ├── neon.zig               # ARM NEON intrinsics (primary target)
│   │   │   ├── avx2.zig               # x86 AVX2 intrinsics (secondary)
│   │   │   ├── dispatch.zig           # Runtime CPU detection + dispatch
│   │   │   ├── vec_ops.zig            # Vectorized add, mul, fma, exp, tanh, sigmoid
│   │   │   └── reduce.zig            # Vectorized sum, max, min reductions
│   │   │
│   │   ├── ops/
│   │   │   ├── matmul.zig             # Tiled GEMM: 8x8 micro-kernel, L1/L2 blocking
│   │   │   ├── conv.zig               # im2col + GEMM convolution
│   │   │   ├── elementwise.zig        # Fused element-wise ops (add, mul, relu, etc.)
│   │   │   ├── reduce.zig             # Sum, mean, max, min along axes
│   │   │   ├── softmax.zig            # Numerically stable softmax (online algorithm)
│   │   │   ├── batchnorm.zig          # Batch normalization (Welford single-pass)
│   │   │   ├── pool.zig               # MaxPool2d, AvgPool2d
│   │   │   └── transpose.zig          # Cache-oblivious transpose
│   │   │
│   │   └── ffi/
│   │       └── exports.zig            # C ABI exported functions
│   │
│   └── tests/
│       ├── test_tensor.zig
│       ├── test_alloc.zig
│       ├── test_simd.zig
│       ├── test_matmul.zig
│       ├── test_conv.zig
│       ├── test_elementwise.zig
│       ├── test_reduce.zig
│       ├── test_softmax.zig
│       ├── test_batchnorm.zig
│       ├── test_pool.zig
│       ├── bench_matmul.zig           # MUST pass: >=5x vs naive
│       ├── bench_simd.zig             # MUST pass: >=4x vs scalar
│       ├── bench_alloc.zig            # MUST pass: >=3x arena, >=5x pool
│       ├── bench_conv.zig             # MUST pass: >=8x vs naive
│       └── bench_reduce.zig           # MUST pass: >=3x vs scalar
│
├── crates/                            # ~18,000 lines Rust
│   ├── synapse-sys/                   # Raw FFI bindings
│   │   ├── Cargo.toml
│   │   ├── build.rs                   # Invokes zig build, links static lib
│   │   └── src/lib.rs                 # extern "C" declarations
│   │
│   ├── synapse-core/                  # Safe Rust tensor wrapper
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tensor.rs              # Safe Tensor<T> wrapping Zig storage
│   │       ├── dtype.rs               # DType enum (F32, F64, I32, I64, Bool)
│   │       ├── device.rs              # Device abstraction (CPU only)
│   │       ├── error.rs               # SynapseError hierarchy
│   │       └── shape.rs               # Dynamic shape + broadcasting
│   │
│   ├── synapse-autograd/              # Automatic differentiation engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── variable.rs            # Variable = Tensor + gradient + graph node
│   │       ├── graph.rs               # DAG of operations, topological sort
│   │       ├── backward.rs            # Reverse-mode AD, gradient accumulation
│   │       ├── function.rs            # GradFn trait (forward + backward)
│   │       ├── no_grad.rs             # Gradient context manager (thread-local)
│   │       ├── grad_check.rs          # Numerical gradient checking
│   │       └── ops/
│   │           ├── mod.rs
│   │           ├── arithmetic.rs      # Add, Sub, Mul, Div backward
│   │           ├── matmul.rs          # MatMul backward
│   │           ├── reduce.rs          # Sum, Mean backward
│   │           ├── activation.rs      # ReLU, Sigmoid, Tanh, GELU backward
│   │           ├── reshape.rs         # Reshape, Transpose, View backward
│   │           ├── conv.rs            # Conv2d backward
│   │           ├── pool.rs            # MaxPool, AvgPool backward
│   │           ├── batchnorm.rs       # BatchNorm backward
│   │           ├── softmax.rs         # Softmax, LogSoftmax backward
│   │           └── loss.rs            # MSE, CrossEntropy backward
│   │
│   ├── synapse-optim/                 # Optimizers
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── optimizer.rs           # Optimizer trait
│   │       ├── sgd.rs                 # SGD + momentum + Nesterov
│   │       ├── adam.rs                # Adam + AdamW
│   │       ├── rmsprop.rs             # RMSProp
│   │       ├── lr_scheduler.rs        # StepLR, CosineAnnealing, WarmupLR
│   │       └── grad_clip.rs           # Gradient clipping (norm + value)
│   │
│   ├── synapse-nn/                    # Neural network layers
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── module.rs              # Module trait (forward, parameters, train/inference)
│   │       ├── sequential.rs          # Sequential container
│   │       ├── linear.rs              # Dense / fully-connected
│   │       ├── conv.rs                # Conv1d, Conv2d
│   │       ├── batchnorm.rs           # BatchNorm1d, BatchNorm2d
│   │       ├── dropout.rs             # Dropout, Dropout2d
│   │       ├── activation.rs          # ReLU, Sigmoid, Tanh, GELU, Softmax modules
│   │       ├── pool.rs                # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   │       ├── flatten.rs             # Flatten layer
│   │       ├── embedding.rs           # Embedding lookup table
│   │       ├── rnn.rs                 # LSTM, GRU cells
│   │       └── init.rs                # Weight init (Xavier, Kaiming)
│   │
│   ├── synapse-data/                  # Data loading pipeline
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dataset.rs             # Dataset trait
│   │       ├── dataloader.rs          # Batching, shuffling, multi-threaded prefetch
│   │       ├── sampler.rs             # Sequential, Random, WeightedRandom
│   │       ├── transform.rs           # Transform trait + Normalize, RandomCrop
│   │       └── collate.rs             # Collation (stack tensors into batch)
│   │
│   ├── synapse-train/                 # Training loop + serialization
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── trainer.rs             # Training loop with hooks
│   │       ├── metrics.rs             # Accuracy, loss tracking, moving averages
│   │       ├── checkpoint.rs          # Model save/load (binary format)
│   │       ├── progress.rs            # Progress bar, ETA estimation
│   │       └── callback.rs            # EarlyStopping, ModelCheckpoint
│   │
│   └── synapse-graph/                 # Graph optimization + operator fusion
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── ir.rs                  # Intermediate representation of compute graph
│           ├── pass.rs                # Optimization pass trait
│           ├── fusion.rs              # Operator fusion (matmul+bias+relu, conv+bn)
│           ├── dead_code.rs           # Dead node elimination
│           ├── constant_fold.rs       # Constant folding
│           └── scheduler.rs           # Memory-optimal execution ordering
│
├── tests/                             # Integration tests
│   ├── integration/
│   │   ├── mnist_e2e.rs               # Full MNIST training (MUST: >90% in 3 epochs)
│   │   ├── autograd_correctness.rs    # Numerical gradient checks
│   │   ├── ffi_roundtrip.rs           # Zig<->Rust tensor roundtrip
│   │   └── graph_optimization.rs      # Fusion + optimization correctness
│   └── benchmarks/
│       ├── matmul_bench.rs            # End-to-end matmul through FFI
│       ├── training_throughput.rs     # MUST: >=5000 samples/sec on MLP
│       └── memory_bench.rs            # Peak memory tracking
│
└── examples/
    ├── xor.rs                         # Minimal XOR (MUST: loss < 0.01 in 1000 steps)
    ├── mnist.rs                       # MNIST digit classification
    └── cifar10.rs                     # CIFAR-10 with simple CNN
```

---

## 3) FFI Boundary Specification

The FFI boundary uses opaque handles, C-compatible enums, and status codes.
**No panics may cross the FFI boundary.** All Zig errors are converted to
status codes. Rust wraps these in `Result<T, SynapseError>`.

### Error Codes

```c
typedef enum {
    SYN_OK = 0,
    SYN_ERR_ALLOC = 1,
    SYN_ERR_SHAPE = 2,
    SYN_ERR_INDEX = 3,
    SYN_ERR_DTYPE = 4,
    SYN_ERR_INVALID = 5,
} syn_status_t;

typedef enum {
    SYN_F32 = 0,
    SYN_F64 = 1,
    SYN_I32 = 2,
    SYN_I64 = 3,
} syn_dtype_t;
```

### Opaque Handles

```c
typedef struct syn_storage_t syn_storage_t;   // Zig-owned byte buffer
typedef struct syn_tensor_t  syn_tensor_t;    // Zig-owned tensor metadata
typedef struct syn_arena_t   syn_arena_t;     // Zig-owned arena allocator
typedef struct syn_pool_t    syn_pool_t;      // Zig-owned pool allocator
```

### Exported Functions

```c
// --- Storage ---
syn_status_t syn_storage_create(syn_storage_t** out, size_t nbytes);
syn_status_t syn_storage_create_pool(syn_storage_t** out, syn_pool_t* pool, size_t nbytes);
void         syn_storage_retain(syn_storage_t* s);
void         syn_storage_release(syn_storage_t* s);
void*        syn_storage_data(syn_storage_t* s);
size_t       syn_storage_len(syn_storage_t* s);

// --- Tensor ---
syn_status_t syn_tensor_create(syn_tensor_t** out, syn_storage_t* storage,
                               syn_dtype_t dtype, const size_t* shape,
                               const int64_t* strides, size_t ndim, size_t offset);
void         syn_tensor_destroy(syn_tensor_t* t);
const size_t* syn_tensor_shape(syn_tensor_t* t);
size_t       syn_tensor_ndim(syn_tensor_t* t);
void*        syn_tensor_data_ptr(syn_tensor_t* t);
int          syn_tensor_is_contiguous(syn_tensor_t* t);
syn_status_t syn_tensor_contiguous(syn_tensor_t** out, syn_tensor_t* t);

// --- Allocators ---
syn_status_t syn_arena_create(syn_arena_t** out, size_t capacity);
void         syn_arena_reset(syn_arena_t* a);
void         syn_arena_destroy(syn_arena_t* a);
syn_status_t syn_pool_create(syn_pool_t** out, size_t slot_size, size_t count);
void         syn_pool_destroy(syn_pool_t* p);

// --- Matrix Operations ---
syn_status_t syn_sgemm(float alpha, const syn_tensor_t* a, const syn_tensor_t* b,
                        float beta, syn_tensor_t* c);

// --- Elementwise Operations ---
syn_status_t syn_elementwise_add(syn_tensor_t* out, const syn_tensor_t* a,
                                  const syn_tensor_t* b);
syn_status_t syn_elementwise_mul(syn_tensor_t* out, const syn_tensor_t* a,
                                  const syn_tensor_t* b);
syn_status_t syn_elementwise_sub(syn_tensor_t* out, const syn_tensor_t* a,
                                  const syn_tensor_t* b);
syn_status_t syn_elementwise_div(syn_tensor_t* out, const syn_tensor_t* a,
                                  const syn_tensor_t* b);

// --- Activations ---
syn_status_t syn_relu(syn_tensor_t* out, const syn_tensor_t* input);
syn_status_t syn_sigmoid(syn_tensor_t* out, const syn_tensor_t* input);
syn_status_t syn_tanh_op(syn_tensor_t* out, const syn_tensor_t* input);
syn_status_t syn_gelu(syn_tensor_t* out, const syn_tensor_t* input);

// --- Reductions ---
syn_status_t syn_reduce_sum(syn_tensor_t* out, const syn_tensor_t* input,
                             const size_t* axes, size_t naxes, int keepdim);
syn_status_t syn_reduce_max(syn_tensor_t* out, const syn_tensor_t* input,
                             const size_t* axes, size_t naxes, int keepdim);
syn_status_t syn_reduce_mean(syn_tensor_t* out, const syn_tensor_t* input,
                              const size_t* axes, size_t naxes, int keepdim);

// --- Complex Operations ---
syn_status_t syn_softmax(syn_tensor_t* out, const syn_tensor_t* input, size_t axis);
syn_status_t syn_batchnorm_forward(syn_tensor_t* out, const syn_tensor_t* input,
                                    const syn_tensor_t* gamma, const syn_tensor_t* beta,
                                    syn_tensor_t* running_mean, syn_tensor_t* running_var,
                                    float eps, float momentum, int training);
syn_status_t syn_conv2d_forward(syn_tensor_t* out, const syn_tensor_t* input,
                                 const syn_tensor_t* weight, const syn_tensor_t* bias,
                                 size_t pad_h, size_t pad_w,
                                 size_t stride_h, size_t stride_w);
syn_status_t syn_maxpool2d_forward(syn_tensor_t* out, syn_tensor_t* indices,
                                    const syn_tensor_t* input,
                                    size_t kh, size_t kw, size_t sh, size_t sw);
syn_status_t syn_avgpool2d_forward(syn_tensor_t* out, const syn_tensor_t* input,
                                    size_t kh, size_t kw, size_t sh, size_t sw);
syn_status_t syn_transpose(syn_tensor_t* out, const syn_tensor_t* input,
                            size_t dim0, size_t dim1);

// --- Raw SIMD vector ops (for Rust to call on raw buffers) ---
void  syn_vadd_f32(float* dst, const float* a, const float* b, size_t len);
void  syn_vmul_f32(float* dst, const float* a, const float* b, size_t len);
void  syn_vfma_f32(float* dst, const float* a, const float* b, const float* c, size_t len);
float syn_vreduce_sum_f32(const float* src, size_t len);
float syn_vreduce_max_f32(const float* src, size_t len);
```

---

## 4) Rust Trait Boundaries

### synapse-core: Tensor

```rust
pub struct Tensor {
    storage: Arc<RawStorage>,   // wraps syn_storage_t* via Drop
    dtype: DType,
    shape: Shape,
    strides: Vec<isize>,
    offset: usize,
}

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType) -> Result<Tensor>;
    pub fn ones(shape: &[usize], dtype: DType) -> Result<Tensor>;
    pub fn randn(shape: &[usize], dtype: DType) -> Result<Tensor>;
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize]) -> Result<Tensor>;
    pub fn shape(&self) -> &[usize];
    pub fn dtype(&self) -> DType;
    pub fn numel(&self) -> usize;
    pub fn data<T: Element>(&self) -> &[T];
    pub fn data_mut<T: Element>(&mut self) -> &mut [T];
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor>;
    pub fn add(&self, other: &Tensor) -> Result<Tensor>;
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor>;
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor>;
    pub fn contiguous(&self) -> Result<Tensor>;
}
```

### synapse-autograd: Variable + GradFn

```rust
pub struct Variable {
    tensor: Tensor,
    grad: RefCell<Option<Tensor>>,
    grad_fn: Option<Arc<dyn GradFn>>,
    requires_grad: bool,
    id: usize,
}

pub trait GradFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    fn inputs(&self) -> &[VariableId];
}

impl Variable {
    pub fn backward(&self) -> Result<()>;
    pub fn grad(&self) -> Option<Tensor>;
    pub fn detach(&self) -> Variable;
    pub fn no_grad<F: FnOnce() -> R, R>(f: F) -> R;
}
```

### synapse-nn: Module

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    fn parameters(&self) -> Vec<&Variable>;
    fn parameters_mut(&mut self) -> Vec<&mut Variable>;
    fn set_training(&mut self, mode: bool);
    fn is_training(&self) -> bool;
    fn name(&self) -> &str;
}
```

### synapse-optim: Optimizer

```rust
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn add_param_group(&mut self, params: Vec<Variable>);
    fn state_dict(&self) -> HashMap<String, Tensor>;
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> Result<()>;
}
```

### synapse-data: Dataset + DataLoader

```rust
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)>;
}

pub struct DataLoader { /* ... */ }
impl DataLoader {
    pub fn new(dataset: Arc<dyn Dataset>, batch_size: usize) -> DataLoaderBuilder;
}
impl Iterator for DataLoader {
    type Item = Result<(Tensor, Tensor)>;
}
```

### synapse-graph: OptimizationPass

```rust
pub trait OptimizationPass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut ComputeGraph) -> Result<bool>;
}
```

---

## 5) Optimization Targets

**These are HARD pass/fail thresholds. If a benchmark does not meet its
target, the task FAILS. Every task that has a benchmark threshold must
include both the naive baseline implementation AND the optimized
implementation, and the benchmark must compare them.**

| # | Module | Metric | Threshold | How to Measure |
|---|--------|--------|-----------|----------------|
| 1 | `ops/matmul.zig` | Tiled GEMM vs naive triple-loop | **>=5x** on 512x512 f32 | `bench_matmul.zig`: 100 iterations, compare tiled vs naive |
| 2 | `ops/matmul.zig` | Absolute GFLOPS | **>=2 GFLOPS** single-core | Calculate: 2*M*N*K / time_ns |
| 3 | `simd/vec_ops.zig` | NEON vadd vs scalar | **>=4x** on 1M f32 elements | `bench_simd.zig`: NEON path vs scalar loop |
| 4 | `simd/vec_ops.zig` | NEON vmul vs scalar | **>=4x** on 1M f32 elements | Same benchmark |
| 5 | `simd/reduce.zig` | NEON reduce_sum vs scalar | **>=3x** on 1M f32 elements | `bench_reduce.zig` |
| 6 | `alloc/arena.zig` | Arena alloc vs malloc | **>=3x** for 10K mixed-size allocs | `bench_alloc.zig` |
| 7 | `alloc/pool.zig` | Pool acquire/release vs malloc | **>=5x** for 10K fixed-size | `bench_alloc.zig` |
| 8 | `ops/conv.zig` | im2col+GEMM vs naive 4-loop | **>=8x** on 3x3 conv, 64x64x3, 32 filters | `bench_conv.zig` |
| 9 | `ops/softmax.zig` | Online stable vs naive 2-pass | **>=2x** on [256, 1000] | Benchmark in `test_softmax.zig` |
| 10 | `ops/batchnorm.zig` | Welford 1-pass vs 2-pass | **>=1.5x** on batch of 256 | Benchmark in `test_batchnorm.zig` |
| 11 | `synapse-graph` fusion | Fused matmul+bias+relu | **>=1.3x** vs unfused on 256x256 | `graph_optimization.rs` integration test |
| 12 | `synapse-autograd` | Graph overhead | **<=5%** overhead vs raw forward (no grad) | Benchmark forward+backward overhead on 10-layer MLP |
| 13 | `synapse-train` | MNIST throughput | **>=5000 samples/sec** on Apple M-series | `training_throughput.rs`: full training loop |
| 14 | `synapse-train` | XOR convergence | **loss < 0.01** in 1000 steps | `examples/xor.rs` |
| 15 | `synapse-train` | MNIST accuracy | **>90%** in 3 epochs | `mnist_e2e.rs` integration test |

### Correctness Thresholds (non-negotiable)

| Module | Requirement |
|--------|-------------|
| All SIMD ops | Max relative error **<= 1e-5** vs scalar reference for transcendentals (exp, tanh, sigmoid) |
| MatMul | Max relative error **<= 1e-4** for 512x512 (FP summation order tolerance) |
| All autograd ops | `grad_check` passes: analytical vs numerical gradient relative error **< 1e-3** |
| Softmax | **No inf/nan** on inputs in [-1000, 1000] |
| FFI boundary | **Zero panics** crossing FFI (all errors as status codes) |
| Memory | **Zero leaks** detected by Zig tracking allocator in all test scenarios |

---

## 6) Task Decomposition — 18 Tasks

**CRITICAL RULES FOR EVERY TASK:**
1. Every task MUST write tests alongside implementation. No code without tests.
2. Every benchmark task MUST include both naive baseline AND optimized implementation.
3. Every task MUST list its pass/fail criteria. The judge uses these to accept/reject.
4. Dependencies must be respected. A task cannot start until its dependencies are complete.

### Dependency Graph

```
WAVE 1 (fully parallel, no dependencies):
  Task 1: Zig Storage+Shape
  Task 2: Zig Allocators
  Task 3: Zig SIMD NEON
  Task 4: Zig SIMD AVX2+Dispatch

WAVE 2 (depends on Wave 1):
  Task 5: Zig Tensor Core ──────────── depends: Task 1
  Task 6: Zig Tiled MatMul ─────────── depends: Task 3 or 4, Task 5
  Task 7: Zig Elementwise+Activations  depends: Task 4, Task 5
  Task 8: Zig Reduce+Softmax+BN ────── depends: Task 4, Task 5
  Task 9: Zig Conv2d+Pooling ───────── depends: Task 5, Task 6

WAVE 3 (FFI serialization point):
  Task 10: Zig FFI Exports ─────────── depends: Tasks 5-9
  Task 11: Rust FFI Bridge+Core ────── depends: Task 10

WAVE 4 (Rust ML crates, many parallel):
  Task 12: Rust Autograd Core ──────── depends: Task 11
  Task 13: Rust Autograd Ops ───────── depends: Task 12
  Task 14: Rust Optimizers ─────────── depends: Task 12 (parallel with 13)
  Task 15: Rust Data Pipeline ──────── depends: Task 11 (parallel with 12-14)
  Task 16: Rust Graph Optimization ─── depends: Task 12 (parallel with 13-14)

WAVE 5 (integration):
  Task 17: Rust NN Layers ──────────── depends: Task 13
  Task 18: Rust Training+Examples ──── depends: Tasks 14, 15, 16, 17
```

---

### Task 1: Zig Tensor Storage + Shape

**Implement:**
- `zig/src/tensor/storage.zig`: Dense contiguous byte buffer with 64-byte
  alignment, atomic reference counting (retain/release), typed data
  accessors via comptime generics.
- `zig/src/tensor/shape.zig`: Comptime shape validation, broadcasting rule
  computation (`broadcast_shapes`), stride calculation from shape
  (`contiguous_strides`), numel, compatibility checks.

**Tests (mandatory):**
- `test_tensor.zig` (partial): storage create/retain/release cycle, ref count
  reaches 0 and frees. Shape broadcast rules: [3,1]+[1,4]=[3,4],
  [5,3,1]+[1,4]=[5,3,4]. Invalid broadcast detection. Contiguous stride calc.
- Storage creation throughput (10K creates) as informational benchmark.

**Pass/fail:**
- All unit tests pass.
- Storage correctly frees when ref count hits 0 (no leaks via tracking allocator).
- Broadcast rules match NumPy semantics for at least 10 test cases.

**Dependencies:** None.

---

### Task 2: Zig Memory Allocators

**Implement:**
- `zig/src/alloc/arena.zig`: Region-based arena with O(1) reset. Bump
  allocation. Configurable backing capacity.
- `zig/src/alloc/pool.zig`: Fixed-size slab allocator. Free-list based
  acquire/release. Pre-allocates slots.
- `zig/src/alloc/aligned.zig`: Wrapper ensuring 64-byte alignment for SIMD.
- `zig/src/alloc/tracking.zig`: Debug allocator wrapper that counts
  allocs/frees and detects leaks.

**Tests (mandatory):**
- `test_alloc.zig`: Arena alloc, reset, re-alloc. Pool acquire/release/
  reacquire. Alignment verification (assert ptr % 64 == 0). Tracking
  allocator leak detection.
- `bench_alloc.zig`: Arena vs std.heap.page_allocator (10K mixed-size),
  Pool vs malloc/free (10K fixed-size).

**Pass/fail:**
- **Arena: >=3x throughput** vs page_allocator for 10K allocations.
- **Pool: >=5x throughput** vs malloc/free for fixed-size acquire/release.
- No leaks detected by tracking allocator in all test scenarios.
- All unit tests pass.

**Dependencies:** None.

---

### Task 3: Zig SIMD Intrinsics — NEON

**Implement:**
- `zig/src/simd/neon.zig`: ARM NEON implementations of vector add, multiply,
  FMA, exp (polynomial approximation), tanh, sigmoid on f32x4 registers.
  Handles tail elements (non-multiple-of-4 lengths).
- `zig/src/simd/reduce.zig` (NEON path): Horizontal sum, horizontal max
  using NEON pairwise instructions.

**Tests (mandatory):**
- `test_simd.zig`: Correctness vs scalar reference for all ops. Edge cases:
  length 0, 1, 3, 4, 7, 1000. Special values: 0, -0, inf, -inf, NaN,
  subnormals. Exp/tanh/sigmoid accuracy within 1e-5 relative error.
- `bench_simd.zig`: NEON vadd vs scalar on 1M elements. Must include BOTH
  the scalar baseline AND the NEON implementation in the benchmark.

**Pass/fail:**
- All correctness tests pass (max relative error <= 1e-5 for transcendentals).
- **NEON path >=4x throughput** vs scalar loop for vadd_f32 on 1M elements.
- Tail handling correct for all non-aligned lengths.

**Dependencies:** None.

---

### Task 4: Zig SIMD Intrinsics — AVX2 + Dispatch

**Implement:**
- `zig/src/simd/avx2.zig`: AVX2 implementations (f32x8) of the same ops as
  NEON: add, mul, FMA, exp, tanh, sigmoid, horizontal sum/max.
- `zig/src/simd/dispatch.zig`: Runtime CPU feature detection. Function pointer
  table dispatching to NEON, AVX2, or scalar fallback.
- `zig/src/simd/vec_ops.zig`: Public API that calls through dispatch table.

**Tests (mandatory):**
- `test_simd.zig` (extended): Same correctness suite as NEON but for AVX2.
  Dispatch correctly selects backend. Scalar fallback produces correct results.
- Benchmark: AVX2 vadd vs scalar on 1M elements (on x86 hardware).

**Pass/fail:**
- All correctness tests pass.
- Dispatch correctly detects CPU on current platform.
- AVX2 path >=4x vs scalar (when on x86). Scalar fallback always correct.

**Dependencies:** None. Can run in parallel with Task 3.

---

### Task 5: Zig Tensor Core

**Implement:**
- `zig/src/tensor/tensor.zig`: Tensor struct wrapping Storage with
  shape/strides/offset. Typed element access (at/set), numel,
  is_contiguous check.
- `zig/src/tensor/view.zig`: Zero-copy reshape (validates numel match),
  transpose (swaps strides/shape), slice (adjusts offset+shape+strides).
- `zig/src/tensor/iterator.zig`: Strided multi-dimensional iterator for
  both contiguous (pointer walk) and non-contiguous (index computation).

**Tests (mandatory):**
- `test_tensor.zig` (extended): Create tensor, read/write elements. Reshape
  preserves data. Transpose produces correct strides. Slice produces correct
  sub-view. Iterator visits all elements in correct order.
- Benchmark: Iterator throughput contiguous vs strided on 1M elements.

**Pass/fail:**
- All index operations correct.
- Contiguous iterator throughput within 10% of raw pointer walk.
- Reshape/transpose/slice do not copy data (same storage pointer).

**Dependencies:** Task 1.

---

### Task 6: Zig Tiled MatMul

**Implement:**
- `zig/src/ops/matmul.zig`: Full SGEMM with:
  - 8x8 micro-kernel using NEON FMA (or 8x8 AVX2 on x86)
  - L1 tiling: MR=8, NR=8 micro-tiles
  - L2 tiling: MC=256, KC=512 macro-tiles
  - L3 tiling: NC=4096
  - Packing of A and B for cache locality
  - Support for trans_a, trans_b
  - Edge cleanup for non-tile-multiple dimensions
  - **MUST also include naive triple-loop implementation for benchmark comparison**

**Tests (mandatory):**
- `test_matmul.zig`: Correctness vs naive triple-loop for sizes: 1x1, 8x8,
  16x16, 64x64, 128x128, 512x512, 1024x1024, non-square (32x64 * 64x48).
  Transposed variants. Tolerance 1e-4 for large sizes.
- `bench_matmul.zig`: 512x512 SGEMM, 100 iterations. Naive vs tiled.

**Pass/fail:**
- Correctness within 1e-4 relative error for all sizes.
- **>=5x speedup** vs naive triple-loop on 512x512.
- **>=2 GFLOPS** on Apple M-series.

**Dependencies:** Task 3 or 4 (SIMD), Task 5 (Tensor).

---

### Task 7: Zig Elementwise + Activation Ops

**Implement:**
- `zig/src/ops/elementwise.zig`: Vectorized element-wise add, sub, mul, div.
  Fused operations: add+relu, mul+add (FMA). Broadcasting support for common
  cases (scalar broadcast, last-dim broadcast).
- Activation functions: relu, leaky_relu, sigmoid, tanh, gelu (approximate).

**Tests (mandatory):**
- `test_elementwise.zig`: Correctness for all ops against scalar reference.
  Broadcasting: scalar+tensor, [1,N]+[M,N], [M,1]+[M,N]. Activation
  correctness including negatives, zeros, large values.
- Benchmark: Fused add+relu vs separate add-then-relu on 1M elements.

**Pass/fail:**
- All ops correct within 1e-6.
- **Fused add+relu >=1.5x** vs separate (memory bandwidth bound).
- Broadcasting produces correct results for all tested shape pairs.

**Dependencies:** Task 4 (dispatch), Task 5 (Tensor).

---

### Task 8: Zig Reduction + Softmax + BatchNorm Ops

**Implement:**
- `zig/src/ops/reduce.zig`: Sum, mean, max, min, argmax along arbitrary axes.
  Keepdim support. SIMD-accelerated inner loops.
- `zig/src/ops/softmax.zig`: Online numerically-stable softmax (single-pass
  max+sum via online trick). LogSoftmax variant.
- `zig/src/ops/batchnorm.zig`: Welford single-pass mean+variance. Forward
  with running stats update. Training vs inference mode.

**Tests (mandatory):**
- `test_reduce.zig`: Reduce correctness along each axis of a 3D tensor.
- `test_softmax.zig`: Softmax sums to 1.0. Handles [-1000, 1000] without
  overflow/underflow. LogSoftmax correctness.
- `test_batchnorm.zig`: Output has mean~0, var~1. Running stats updated.
- `bench_reduce.zig`: SIMD reduce_sum vs scalar on 1M elements.

**Pass/fail:**
- **Reduce >=3x** vs scalar.
- Softmax numerically stable: **no inf/nan** on inputs in [-1000, 1000].
- **BatchNorm single-pass >=1.5x** vs two-pass.

**Dependencies:** Task 4 (SIMD), Task 5 (Tensor).

---

### Task 9: Zig Conv2d + Pooling Ops

**Implement:**
- `zig/src/ops/conv.zig`: im2col transformation + GEMM-based conv2d forward.
  Padding, strides. Direct fallback for 1x1 kernels.
  **MUST include naive 4-loop implementation for benchmark comparison.**
- `zig/src/ops/pool.zig`: MaxPool2d (with argmax indices for backward),
  AvgPool2d. Stride and kernel size parameters.
- `zig/src/ops/transpose.zig`: Cache-oblivious matrix transpose.

**Tests (mandatory):**
- `test_conv.zig`: Correctness vs naive 4-loop on 3x3, 5x5, 1x1 kernels.
  Various padding/stride combos.
- `test_pool.zig`: Pooling correctness + argmax indices correct.
- `bench_conv.zig`: 3x3 conv on 64x64x3, 32 filters. Naive vs im2col+GEMM.

**Pass/fail:**
- Correctness within 1e-4.
- **im2col+GEMM conv >=8x** vs naive 4-loop.
- Pooling argmax indices correct for backward pass.

**Dependencies:** Task 5 (Tensor), Task 6 (MatMul for GEMM).

---

### Task 10: Zig FFI Exports

**Implement:**
- `zig/src/ffi/exports.zig`: All `export fn` declarations with C calling
  convention. Wraps every Zig function with error-code returns, opaque
  pointer handles, and null-safety checks. No panics escape.
- `zig/build.zig`: Build config producing `libsynapse_zig.a` static library.
  Cross-compilation support for aarch64 and x86_64.

**Tests (mandatory):**
- Round-trip test in Zig: create storage, create tensor, run matmul, read
  result, destroy. All via exported C ABI functions.
- Verify no panics escape (test with invalid inputs producing proper error codes).

**Pass/fail:**
- All FFI functions compile to valid C ABI.
- **No panics** escape across FFI boundary.
- `zig build` produces `libsynapse_zig.a` for current platform.

**Dependencies:** Tasks 5, 6, 7, 8, 9 (all Zig ops).

---

### Task 11: Rust FFI Bridge + Core Tensor

**Implement:**
- `crates/synapse-sys/build.rs`: Invokes `zig build -Doptimize=ReleaseFast`,
  links static library. Generates Rust extern declarations.
- `crates/synapse-sys/src/lib.rs`: Raw `extern "C"` block matching synapse.h.
- `crates/synapse-core/`: Safe `Tensor` wrapper with `Arc<RawStorage>` that
  calls `syn_storage_release` on Drop. DType, Shape, Error types.

**Tests (mandatory):**
- `ffi_roundtrip.rs`: Create tensors from Rust, verify shape/data. Matmul
  through FFI matches reference. Drop releases Zig storage. Error propagation.
- Memory leak test: create/destroy 10K tensors, verify no leaks.

**Pass/fail:**
- Tensor creation/destruction has **no memory leaks**.
- MatMul through FFI matches NumPy reference within 1e-4.
- All Zig error codes correctly map to Rust `Result::Err`.

**Dependencies:** Task 10.

---

### Task 12: Rust Autograd Engine — Core

**Implement:**
- `crates/synapse-autograd/src/variable.rs`: Variable wrapping Tensor +
  optional gradient. Unique ID via AtomicUsize. `requires_grad` flag.
- `crates/synapse-autograd/src/graph.rs`: Computation graph as Vec of nodes.
  Each node stores GradFn + parent indices. Topological sort for backward.
- `crates/synapse-autograd/src/backward.rs`: Reverse-mode AD. Walk graph in
  reverse topo order. Accumulate gradients (handle fan-out by summing).
- `crates/synapse-autograd/src/function.rs`: GradFn trait.
- `crates/synapse-autograd/src/no_grad.rs`: Thread-local flag to disable tracking.
- `crates/synapse-autograd/src/grad_check.rs`: Numerical gradient checking.

**Tests (mandatory):**
- Simple graph: y = a*b + c, backward produces correct da, db, dc.
- Fan-out: y = x + x, dy/dx = 2.
- Diamond: z = (x+y) * (x-y), correct dz/dx and dz/dy.
- no_grad context prevents graph construction.
- grad_check: numerical vs analytical within 1e-4 for basic ops.

**Pass/fail:**
- Correct gradients for linear, fan-out, and diamond graphs.
- grad_check passes for all tested ops.
- no_grad properly disables graph tracking.

**Dependencies:** Task 11.

---

### Task 13: Rust Autograd Ops

**Implement:**
- `crates/synapse-autograd/src/ops/arithmetic.rs`: Add, Sub, Mul, Div
  forward+backward with broadcasting gradient reduction.
- `crates/synapse-autograd/src/ops/matmul.rs`: dA = dOut @ B^T, dB = A^T @ dOut.
- `crates/synapse-autograd/src/ops/reduce.rs`: Sum backward (expand), Mean backward.
- `crates/synapse-autograd/src/ops/activation.rs`: ReLU, Sigmoid, Tanh, GELU backward.
- `crates/synapse-autograd/src/ops/reshape.rs`: Reshape, Transpose backward.
- `crates/synapse-autograd/src/ops/conv.rs`: Conv2d backward (im2col grad,
  weight grad, input grad via col2im).
- `crates/synapse-autograd/src/ops/pool.rs`: MaxPool backward (scatter via
  saved indices), AvgPool backward.
- `crates/synapse-autograd/src/ops/batchnorm.rs`: Input, gamma, beta gradients.
- `crates/synapse-autograd/src/ops/softmax.rs`: Softmax, LogSoftmax backward.
- `crates/synapse-autograd/src/ops/loss.rs`: MSELoss, CrossEntropyLoss.

**Tests (mandatory):**
- For EVERY op: `grad_check` (numerical vs analytical). This is non-negotiable.
- MatMul gradient shapes correct for non-square matrices.
- Broadcasting gradient reduction correct.
- Benchmark: Forward+backward on 10-layer MLP, batch 64.

**Pass/fail:**
- **grad_check passes for ALL ops** (relative error < 1e-3).
- No gradient shape mismatches.
- **Forward+backward overhead <=5%** vs forward-only (graph overhead).

**Dependencies:** Task 12.

---

### Task 14: Rust Optimizers

**Implement:**
- `crates/synapse-optim/src/optimizer.rs`: Optimizer trait.
- `crates/synapse-optim/src/sgd.rs`: SGD + momentum + Nesterov + weight decay.
- `crates/synapse-optim/src/adam.rs`: Adam with bias correction. AdamW.
- `crates/synapse-optim/src/rmsprop.rs`: RMSProp with centered variant.
- `crates/synapse-optim/src/lr_scheduler.rs`: StepLR, CosineAnnealingLR,
  LinearWarmup, ReduceLROnPlateau.
- `crates/synapse-optim/src/grad_clip.rs`: clip_grad_norm_, clip_grad_value_.

**Tests (mandatory):**
- SGD step on known gradient produces expected parameter update.
- Adam state (m, v) correctly maintained across steps. Bias correction
  correct for first 10 steps.
- LR schedulers produce correct learning rates at each step.
- Gradient clipping respects max norm.
- Benchmark: optimizer.step() time for 1M parameters.

**Pass/fail:**
- SGD update matches PyTorch reference for 5-step sequence within 1e-6.
- Adam matches PyTorch reference for 10-step sequence within 1e-6.
- Gradient clipping correctly bounds norm.
- All schedulers produce correct LR values.

**Dependencies:** Task 12 (Variable). Parallel with Task 13.

---

### Task 15: Rust Data Pipeline

**Implement:**
- `crates/synapse-data/src/dataset.rs`: Dataset trait. InMemoryDataset.
  TensorDataset wrapping (features, labels) pair.
- `crates/synapse-data/src/sampler.rs`: SequentialSampler, RandomSampler (seed),
  WeightedRandomSampler.
- `crates/synapse-data/src/collate.rs`: Stack tensors along new batch dimension.
  Pad sequences to max length.
- `crates/synapse-data/src/dataloader.rs`: Configurable batch_size, shuffle,
  drop_last. Multi-threaded prefetching with double-buffering.
- `crates/synapse-data/src/transform.rs`: Normalize(mean, std),
  RandomHorizontalFlip, ToTensor.

**Tests (mandatory):**
- DataLoader iterates correct number of batches.
- Shuffle produces different order each epoch but all elements exactly once.
- Batch tensors have correct shape [batch, ...].
- Transforms modify data correctly.
- Benchmark: DataLoader throughput (batches/sec) with 4 prefetch threads.

**Pass/fail:**
- Correct batch count: ceil(N/batch_size) or floor if drop_last.
- Shuffle covers all elements exactly once per epoch.
- Prefetch: next batch ready before previous finishes (timing-verified).

**Dependencies:** Task 11 (synapse-core). Parallel with Tasks 12-14.

---

### Task 16: Rust Graph Optimization + Operator Fusion

**Implement:**
- `crates/synapse-graph/src/ir.rs`: Graph IR: nodes (Op, Constant, Parameter,
  Input), edges. Node metadata (shape, dtype).
- `crates/synapse-graph/src/pass.rs`: OptimizationPass trait.
- `crates/synapse-graph/src/fusion.rs`: Fuse matmul+bias+relu. Fuse conv+batchnorm
  (fold BN weights into conv). Fuse sequential element-wise ops.
- `crates/synapse-graph/src/dead_code.rs`: Remove unreachable nodes.
- `crates/synapse-graph/src/constant_fold.rs`: Constant subgraph computation.
- `crates/synapse-graph/src/scheduler.rs`: Memory-optimal execution order
  via liveness analysis.

**Tests (mandatory):**
- Fusion correctly merges nodes (graph node count decreases).
- Dead code elimination removes unused branches.
- Constant folding computes correctly.
- Scheduler respects data dependencies.
- Benchmark: Fused matmul+bias+relu vs unfused on 256x256.

**Pass/fail:**
- **Fused matmul+bias+relu >=1.3x** vs unfused.
- Conv+BN fusion numerically identical (within 1e-5).
- Dead code elimination preserves semantics.
- Scheduler produces valid topological order.

**Dependencies:** Task 12. Parallel with Tasks 13-14.

---

### Task 17: Rust Neural Network Layers

**Implement:**
- `crates/synapse-nn/src/module.rs`: Module trait + ModuleList.
- `crates/synapse-nn/src/sequential.rs`: Sequential container.
- `crates/synapse-nn/src/linear.rs`: Linear(in, out, bias). Xavier init.
- `crates/synapse-nn/src/conv.rs`: Conv2d. Kaiming init.
- `crates/synapse-nn/src/batchnorm.rs`: BatchNorm1d, BatchNorm2d.
- `crates/synapse-nn/src/dropout.rs`: Dropout (mask during training, identity during inference).
- `crates/synapse-nn/src/activation.rs`: ReLU, Sigmoid, Tanh, GELU, Softmax.
- `crates/synapse-nn/src/pool.rs`: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.
- `crates/synapse-nn/src/flatten.rs`: Flatten(start_dim, end_dim).
- `crates/synapse-nn/src/embedding.rs`: Embedding lookup (sparse gradient).
- `crates/synapse-nn/src/rnn.rs`: LSTMCell, GRUCell.
- `crates/synapse-nn/src/init.rs`: Xavier uniform/normal, Kaiming uniform/normal.

**Tests (mandatory):**
- All output shapes match expected formulas for every layer type.
- Sequential chains correctly.
- parameters() returns correct parameter count.
- Training/inference mode toggles through Sequential.
- Dropout zeros ~p fraction (p +/- 5% over 1000 trials).
- Benchmark: Forward pass of 5-layer CNN on 32x32x3 input.

**Pass/fail:**
- All output shapes correct.
- Correct parameter counts.
- Dropout rate within p +/- 5%.
- Mode propagation works.

**Dependencies:** Task 13.

---

### Task 18: Rust Training Loop + Serialization + Examples

**Implement:**
- `crates/synapse-train/src/trainer.rs`: for each epoch, for each batch:
  forward -> loss -> backward -> step. Validation loop (no_grad). Hooks.
- `crates/synapse-train/src/metrics.rs`: RunningMean, Accuracy (top-1, top-5),
  ConfusionMatrix.
- `crates/synapse-train/src/checkpoint.rs`: Save/load model state_dict
  (parameter name -> tensor bytes). Version header.
- `crates/synapse-train/src/progress.rs`: Epoch/batch progress with ETA.
- `crates/synapse-train/src/callback.rs`: EarlyStopping, ModelCheckpoint.
- `examples/xor.rs`: 2-layer MLP solving XOR.
- `examples/mnist.rs`: MLP or CNN on MNIST.
- `examples/cifar10.rs`: Simple CNN on CIFAR-10.

**Tests (mandatory):**
- Trainer runs 1 epoch without error.
- Checkpoint save/load roundtrip produces identical parameters.
- EarlyStopping triggers after patience exceeded.
- Metrics track correctly.
- `mnist_e2e.rs`: Train MNIST 3 epochs, verify accuracy.
- `training_throughput.rs`: Measure samples/sec.

**Pass/fail:**
- **XOR converges to loss < 0.01** in 1000 steps.
- **MNIST reaches >90% accuracy** in 3 epochs.
- Checkpoint roundtrip is **bit-exact**.
- **Training throughput >=5000 samples/sec** on MLP (256-128-10), batch 64.
- EarlyStopping and callbacks work correctly.

**Dependencies:** Tasks 14, 15, 16, 17.

---

## 7) Success Metrics

| Metric | Target |
|--------|--------|
| Tasks completed | 18/18 |
| Unit tests | All pass, 100% pass rate |
| Benchmark thresholds | All 15 hard thresholds met |
| Memory safety | Zero leaks (Zig tracking allocator + Rust Drop) |
| FFI safety | Zero panics crossing boundary |
| Autograd correctness | grad_check passes for all ops |
| Numerical stability | No inf/nan in softmax, stable training |
| XOR convergence | loss < 0.01 in 1000 steps |
| MNIST accuracy | >90% in 3 epochs |
| MNIST throughput | >=5000 samples/sec |
| Total lines | ~30,000 (+/- 3K) |
| Test coverage | Every module has unit tests |
| Benchmark coverage | Every perf-critical module has pass/fail benchmark |

---

## 8) Key Architectural Decisions

1. **Opaque handles across FFI.** Zig-owned objects are exposed as opaque
   pointers (`*syn_storage_t`). Rust cannot depend on Zig struct layout.
   This allows independent evolution of Zig internals.

2. **Status codes, not panics, across FFI.** Every FFI function returns
   `syn_status_t`. Zig `@panic` is caught at the boundary and converted
   to `SYN_ERR_INVALID`. Rust wraps in `Result<T, SynapseError>`.

3. **Storage ref-counting owned by Zig.** Zig manages atomic reference
   counts. Rust's `Arc<RawStorage>` calls `syn_storage_retain` on clone
   and `syn_storage_release` on Drop. Single source of truth for memory.

4. **Comptime shape checking internal to Zig.** Micro-kernels and tiling
   use comptime generics for unrolling/bounds elimination. FFI uses
   dynamic shapes (runtime `size_t*`). Rust uses dynamic shapes.

5. **SIMD dispatch at init time.** Function pointer table populated once
   at library init. Per-call overhead is a single indirect call, not a
   branch on every operation.

6. **Arena allocator for training steps.** Each training step allocates
   intermediate tensors (activations, gradients). Arena `reset()` frees
   everything in O(1) at step end, avoiding thousands of `free()` calls.

7. **Autograd uses Wengert tape (dynamic graph).** Variables record ops
   to a thread-local tape. `backward()` replays in reverse. Tape cleared
   after backward. Matches PyTorch's dynamic approach.

8. **Graph optimization is optional.** `synapse-graph` can inspect and
   optimize a recorded computation. But the default path (direct autograd
   + Zig calls) works without it. Graph optimization is additive.

9. **Naive baselines required.** Every benchmark MUST include the naive
   implementation alongside the optimized one. Speedup is measured as
   `naive_time / optimized_time`. This prevents gaming benchmarks.

---

## 9) Build and Run Instructions

### Prerequisites
- Zig 0.13+ (or latest stable)
- Rust 1.75+ (edition 2021)
- Apple Silicon Mac (primary) or x86_64 Linux (secondary)

### Build
```bash
# Build Zig static library
cd synapse/zig && zig build -Doptimize=ReleaseFast

# Build Rust workspace (automatically invokes Zig via build.rs)
cd synapse && cargo build --release

# Run all tests
cd synapse/zig && zig build test
cd synapse && cargo test --all

# Run benchmarks
cd synapse/zig && zig build bench
cd synapse && cargo bench
```

### Run Examples
```bash
cargo run --example xor --release
cargo run --example mnist --release
cargo run --example cifar10 --release
```
