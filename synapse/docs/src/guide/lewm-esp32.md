# LEWM World Model on ESP32

This guide covers the LEWM (Latent Encoder World Model) architecture, the LQ40 binary format, model conversion, quantization strategies, and the available model variants for ESP32-P4 deployment.

> **New**: The [Hybrid ALAL Encoder](hybrid-alal.md) guide covers the 64d hybrid architecture that achieves **3.9 MB binary, 152ms predict, 922ms encode** on ESP32-P4. See also the [Optimization Journey](../architecture/optimization-journey.md) for the full 5.2x speedup story.

## LEWM Architecture Overview

LEWM is a vision-based world model that learns latent dynamics from image observations. It supports two encoder architectures:

1. **Standard ViT** (192d hidden): Used by baseline and elastic variants
2. **Hybrid ALAL** (64d hidden): Alternating full/linear attention, 3.4x fewer params

### Encoder (ViT)

A Vision Transformer that converts a 224x224 RGB image into a compact latent vector.

```
Image (224x224x3) -> Patch Embedding (16x16 grid = 256 patches)
    -> CLS token prepended (+ optional meta tokens for hybrid = 261 tokens)
    -> Positional Embedding
    -> N x Encoder Layers (LayerNorm -> Attention -> FFN)
    -> CLS token extraction -> LayerNorm -> Optional encoder.proj
    -> Projection Head -> Latent vector (64d, 96d, or 192d)
```

Each encoder layer:
- **LayerNorm** with learnable weight + bias
- **Multi-Head Self-Attention** (bidirectional, 257x257 attention matrix)
  - Q, K, V linear projections (INT8 quantized on device)
  - Scaled dot-product attention: softmax(QK^T / sqrt(d)) x V
  - Output projection
- **Feed-Forward Network** (FFN)
  - Up-projection to 4x hidden dim
  - GELU activation
  - Down-projection back to hidden dim
- **Residual connections** around both attention and FFN blocks

### Predictor (adaLN Transformer)

A lightweight transformer that predicts the next latent state given a current state and action.

```
Latent (96d or 192d) + Action (2d) -> Action Encoder (MLP)
    -> Conditioning = Latent + Action embedding
    -> N x Predictor Layers (adaLN -> Attention -> adaLN -> FFN)
    -> Output Projection
    -> Next Latent (96d or 192d)
```

Each predictor layer uses **Adaptive Layer Normalization (adaLN)**:
- A linear layer generates 6 modulation vectors from conditioning: (scale1, shift1, gate1, scale2, shift2, gate2)
- These modulate the normalized attention and FFN outputs
- All predictor weights use Q4 quantization (4-bit)

### Action Encoder

A small MLP that maps raw action vectors (e.g., 2D robot actions) to the latent dimension:

```
Action (2d) -> Conv1D -> MLP(act_dim -> 4*latent -> latent) -> Action Embedding
```

## Model Variants

| Variant | Latent Dim | Encoder Layers | Predictor Layers | Format | Size | predict_next | encode |
|---------|-----------|----------------|-----------------|--------|------|-------------|--------|
| **Hybrid ALAL 64d** | 64 | 4 (ALAL) | 4 | INT8+Q4 | 3.9 MB | **145 ms** | **817 ms** |
| **Full 192d** | 192 | 6 | 6 | INT8+Q4 | ~10 MB | 828 ms | ~10,000 ms |
| **Slim 96d full** | 96 | 4 | 4 | INT8+Q4 | ~5 MB | 583 ms | 6,416 ms |
| **Slim 96d Q4** | 96 | 4 | 4 | Q4-pred only | ~3 MB | 583 ms | N/A |
| **Slim 48d** | 48 | 2 | 2 | INT8+Q4 | ~2 MB | ~300 ms* | ~3,000 ms* |

\* Projected, not yet tested on hardware.

The **Hybrid ALAL 64d** is the recommended variant — smallest binary, fastest inference, with fused ops + exp LUT + aligned PIE alloc (2026-04-03).

The config is parsed dynamically from the LQ40 header -- any variant works without code changes.

## LQ40 Binary Format

LQ40 is a custom binary format designed for efficient loading on microcontrollers. No JSON parsing at weight-load time, no file system needed.

### Layout

```
Offset  Size    Content
----------------------------------------------
0       4       Magic: "LQ40" (4 ASCII bytes)
4       4       config_len: uint32 LE (length of JSON config)
8       N       JSON config string (N = config_len bytes)
8+N     ...     Weight data (binary, format depends on mode)
```

### Config JSON

The JSON config encodes model architecture and quantization mode:

```json
{
  "mode": "full",
  "latent_dim": 96,
  "encoder_hidden": 96,
  "encoder_layers": 4,
  "encoder_heads": 2,
  "encoder_inter": 384,
  "predictor_hidden": 96,
  "predictor_layers": 4,
  "predictor_heads": 2,
  "predictor_inter": 384,
  "image_size": 224,
  "patch_size": 14,
  "channels": 3,
  "action_dim": 2,
  "has_input_proj": true,
  "has_cond_proj": true
}
```

### Mode Values

| Mode | Description | Encoder Weights | Predictor Weights |
|------|-------------|----------------|-------------------|
| `"full"` | INT8 encoder + Q4 predictor | Per-channel INT8 with f32 scales | Q4 nibble-packed with per-block f32 scales |
| `"q4-pred"` | Q4 predictor only (no encoder) | N/A | Q4 nibble-packed |
| `"wanda20-q4"` | 20% WANDA-pruned Q4 | N/A | Sparse Q4 with bitmap |
| `"wanda40-q4"` | 40% WANDA-pruned Q4 | N/A | Sparse Q4 with bitmap |

### Weight Data Layout

**INT8 weights** (encoder):
```
For each linear layer:
  [out_features x in_features bytes]  -- INT8 weight matrix, row-major
  [out_features x 4 bytes]            -- f32 per-output-channel scales
```

**Q4 weights** (predictor):
```
For each linear layer:
  [num_blocks x 2 bytes]     -- f16 per-block scales (32 elements per block)
  [num_blocks x 16 bytes]    -- nibble-packed weights (2 per byte, low|high)
```

Each Q4 nibble encodes a value in `[-8, 7]` (unsigned 0-15 with offset -8).

## Checkpoint Conversion Pipeline

### Step 1: PyTorch .ckpt to safetensors

Convert crucible LEWM checkpoints to the standard safetensors format:

```bash
python3 scripts/convert_lewm_ckpt.py \
  --input model.ckpt \
  --output /tmp/lewm-converted/
```

This produces:
- `lejepa_weights.safetensors` -- All tensor weights in f32
- `config.json` -- Auto-inferred model configuration

The converter uses a **stub unpickler** that handles missing packages (`jepa`, `module`) by substituting stub classes. It walks the nn.Module tree recursively, extracts all tensors, flattens the key hierarchy, and infers model config from weight shapes.

**Security note**: The converter loads serialized model objects. Only use with trusted checkpoint files from known sources (e.g., your own W&B artifacts).

For batch conversion:
```bash
python3 scripts/convert_lewm_ckpt.py \
  --input-dir ~/Downloads/ \
  --output-dir /tmp/lewm-variants/
```

### Step 2: safetensors to LQ40 Binary

The Rust crate handles quantization and LQ40 serialization:

```bash
# Build the converter
cargo build -p synapse-inference --release

# The LQ40 binaries are pre-built and available in:
ls synapse/web/lewm-compress-demo/*.bin
```

Available pre-built binaries:
- `lewm-full.bin` -- Full INT8+Q4 model (192d, 6+6 layers)
- `lewm-q4-pred.bin` -- Q4 predictor only
- `lewm-slim-96d-q4.bin` -- Slim 96d Q4 predictor
- `lewm-slim-96d-full.bin` -- Slim 96d INT8+Q4
- `lewm-slim-96d-f32.bin` -- Slim 96d full precision (reference only)
- `lewm-wanda20-q4.bin` -- 20% pruned variant
- `lewm-wanda40-q4.bin` -- 40% pruned variant

### Step 3: Embed in ESP32 Build

```bash
# Copy desired model into the build
cp web/lewm-compress-demo/lewm-slim-96d-full.bin \
   synapse-esp32/esp-idf-app/main/model.bin
```

The model is embedded in flash via `CMakeLists.txt`:

```cmake
target_add_binary_data(${COMPONENT_TARGET} "model.bin" BINARY)
```

This makes it available as a linker symbol in C:

```c
extern const uint8_t _binary_model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t _binary_model_bin_end[] asm("_binary_model_bin_end");
```

## Quantization Strategies

### INT8 (Encoder Weights)

Used for all encoder linear layers (Q, K, V, O projections + FFN up/down).

- **Per-channel quantization**: Each output channel has its own f32 scale factor
- **Symmetric**: Values in `[-128, 127]`, zero point = 0
- **Compute**: PIE SIMD does 16-wide INT8 multiply-accumulate in hardware
- **Activation quantization**: Done per-row at inference time (dynamic quantization)

```
f32_weight -> clamp(-128, 127) / scale -> int8_weight
At inference: int8_result * input_scale * weight_scale -> f32_output
```

### Q4 (Predictor Weights)

Used for all predictor linear layers (adaLN, attention, FFN).

- **Per-block quantization**: Every 32 elements share one f32 scale
- **Asymmetric nibble**: 4-bit values `[0, 15]` stored as nibble pairs, decoded as `value - 8` giving `[-8, 7]`
- **Compute**: Dequant to INT8 on-the-fly, then PIE SIMD dot product

```
f32_weight -> quantize to 4-bit per block of 32 -> nibble_pack
At inference: unpack nibbles -> int8 [-8,7] -> PIE dot -> * input_scale * block_scale -> f32
```

### Shared QKV Quantization (Optimization)

The encoder's Q, K, V projections all operate on the same input (the layer-normalized sequence). Instead of quantizing the input three times:

```c
// Before: 3x redundant quantization
quantize_row_int8(normed, ...);  // for Q
quantize_row_int8(normed, ...);  // for K (same input!)
quantize_row_int8(normed, ...);  // for V (same input!)

// After: quantize once, reuse
quantize_row_int8(normed, hidden, qkv_i8, &qkv_scale);
int8linear_forward_prequant(&layer->w_q, qkv_i8, qkv_scale, ...);  // Q
int8linear_forward_prequant(&layer->w_k, qkv_i8, qkv_scale, ...);  // K
int8linear_forward_prequant(&layer->w_v, qkv_i8, qkv_scale, ...);  // V
```

Saves ~10 ms per encoder layer.

### WANDA Pruning (Experimental)

WANDA (Weights AND Activations) prunes weights based on the product of weight magnitude and activation norm:

- **WANDA 20%**: Prunes 20% of weights, ~80% model quality retained
- **WANDA 40%**: Prunes 40% of weights, smaller model, more quality loss
- Pruned weights are stored with a bitmap indicating which blocks are non-zero
- Skip zero blocks during compute for proportional speedup

## Memory Layout on Device

| Data | Location | Size | Notes |
|------|----------|------|-------|
| Model weights (Q4+INT8) | Flash, loaded to PSRAM | 5-10 MB | Embedded in partition |
| Current layer weights | L2 cache | up to 768 KB | Hardware-managed cache over PSRAM |
| Activation vectors | SRAM heap | 2-8 KB | Per-token working memory |
| INT8 dequant scratch | SRAM heap | 4 KB | Per-row temporary |
| GELU LUT | TCM (8 KB) | 4 KB | 1024-entry lookup table |
| Attention scores | SRAM heap | 264 KB max | 257x257x4 bytes per layer |
| PIE accumulators | Registers | 40 bits | Hardware XACC register |

Total PSRAM available after model load: ~22-27 MB free (of 32 MB).

## Inference Pipeline on Device

### predict_next(latent, action)

```
1. Encode action: action(2d) -> MLP -> action_emb(latent_dim)
2. Conditioning: latent + action_emb
3. Input projection: latent -> predictor_hidden (if latent_dim != predictor_hidden)
4. For each predictor layer:
   a. adaLN linear: conditioning -> 6 modulation vectors (Q4 GEMV)
   b. LayerNorm + modulate (scale1, shift1)
   c. QKV projection + attention + output projection (Q4 GEMV)
   d. Gate1 * attention_output + residual
   e. LayerNorm + modulate (scale2, shift2)
   f. FFN up -> GELU -> FFN down (Q4 GEMV)
   g. Gate2 * FFN_output + residual
5. Output projection -> next_latent
```

Timing (hybrid ALAL 64d, PIE + fused ops): **145 ms** per step.

### encode(image)

```
1. Patch embedding: image(224x224x3) -> 256 patches x hidden_dim
2. Prepend CLS token (learnable) -> 257 tokens
3. Add positional embeddings
4. For each encoder layer:
   a. LayerNorm + bias
   b. Shared QKV quantization (f32 -> INT8, once)
   c. Q, K, V projections (PIE INT8 GEMV)
   d. Dual-core attention: split 257 queries across 2 cores
      - PIE INT8 dot products for QK^T
      - Softmax per query
      - Score-weighted V summation
   e. Output projection (PIE INT8 GEMV)
   f. Residual add
   g. LayerNorm + bias
   h. FFN: up-proj -> GELU(LUT) -> down-proj (PIE INT8 GEMV)
   i. Residual add
5. Extract CLS token -> LayerNorm -> Projection head -> latent
```

Timing (hybrid ALAL 64d, PIE + dual-core + fused ops): **817 ms**.

### rollout(latent, actions[])

Simply chains `predict_next` for each action step:

```
latent_0 = initial_latent
for i in 0..N:
    latent_{i+1} = predict_next(latent_i, actions[i])
return [latent_1, latent_2, ..., latent_N]
```

Timing: **~145 ms x N steps** (e.g., 435 ms for 3 steps).

### rollout_fused(latent, actions[])

Fused multi-step rollout: encodes all actions, builds one N×3-token fused sequence,
runs all predictor layers **once**. Returns one predicted latent state per action.

```
1. Encode all actions: action -> action_emb (same encoder for all steps)
2. Build fused sequence: [z, a0, 0, z, a1, 0, ...] (positional embeddings added)
3. Run predictor layers once (all N steps processed in parallel via bidirectional attention)
4. Extract targets at positions 2, 5, 8, ... and project each
```

**Note on fused vs sequential:** Steps 1+ in fused differ from sequential by design —
fused uses bidirectional attention across all steps (parallel future hypotheses), while
sequential is strictly autoregressive. Step-0 is identical (same z_start + a0, cos_sim = 1.000).

**ESP32 limitation:** Currently limited to **50 steps max** (`MAX_PREDICTOR_SEQ_LEN=150`).

| Variant | Sequential 3-step | Fused 3-step | Speedup |
|---------|------------------|--------------|---------|
| Slim 96d (Q4) | 462 ms | 279 ms | **1.66x** |
| Full 192d (Q4) | ~900 ms | ~540 ms | **1.66x** |

## Host Testing

Run model tests on your development machine without ESP32 hardware:

```bash
# Run all 31 host tests (default feature: host-test)
cargo test -p synapse-esp32

# Run with specific model
cargo test -p synapse-esp32 -- --test-threads=1
```

Host tests validate:
- LQ40 binary parsing (magic, config, weight extraction)
- Model inference parity (predict, encode, rollout)
- Quantization correctness (INT8, Q4 round-trip)
- Config detection (auto-detect mode from LQ40 header)
