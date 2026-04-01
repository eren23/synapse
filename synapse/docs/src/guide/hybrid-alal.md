# Hybrid ALAL Encoder

The hybrid ALAL (Attention-Linear-Attention-Linear) encoder is a compact 64d vision encoder that alternates full softmax attention and kernel-trick linear attention blocks. It produces the same quality trajectories as the 192d ViT encoder at **3.4x fewer parameters** and **4.6x faster encode** on ESP32-P4.

## Architecture

```
Image [224x224x3]
  |
  Patch Embedding (256 patches x 588d -> 64d)    [INT8 batch GEMM on ESP32]
  |
  CLS token [1, 64]
  Meta tokens [4, 64]                             [hybrid-specific]
  Position Embedding [261, 64]                     [256 patches + 1 CLS + 4 meta]
  |
  Block 0 (A): Full multi-head attention (fused QKV, with bias, softmax)
  Block 1 (L): Kernel-trick linear attention (separate Q/K/V, no bias, ELU+1)
  Block 2 (A): Full attention
  Block 3 (L): Kernel-trick linear attention
  |
  Final LayerNorm -> CLS token extraction
  Encoder output projection (Linear, BN folded at convert time)
  |
  Projector: 64d -> 2048 (GELU) -> 64d
```

### A Blocks (Full Attention)

Standard bidirectional multi-head attention with softmax normalization:

```
scores = softmax(Q @ K^T / sqrt(d))
out = scores @ V
```

Weights: fused QKV projection `[3*64, 64]` with bias. Uses INT8 PIE SIMD on ESP32 for QK^T dot products.

### L Blocks (Linear Attention)

Kernel-trick O(nd^2) attention that avoids building the full n x n score matrix:

```
KV = phi(K)^T @ V          [d, d] -- computed once per head
k_sum = sum(phi(K))         [d]    -- computed once per head
out[q] = phi(Q[q]) @ KV / (phi(Q[q]) . k_sum)
```

where `phi(x) = ELU(x) + 1` (always positive, enables factorization).

Weights: separate Q, K, V projections `[64, 64]` without bias. The absence of bias is used as the detection signal for L blocks at inference time.

**Complexity**: O(nd^2) vs O(n^2d) for softmax attention. With n=261 tokens and d=64: ~2.1M MACs vs ~8.7M MACs per head.

## Conversion Pipeline

### 1. Convert checkpoint

The converter auto-detects hybrid ALAL checkpoints by the presence of `encoder.blocks.*` keys:

```bash
python3 scripts/convert_lewm_ckpt.py \
  --input hybrid_ALAL_64d_4e_4p_1ep_weights.ckpt \
  --output /tmp/hybrid_alal/
```

The converter:
- Remaps `encoder.blocks.N.*` to standard ViT naming (`encoder.encoder.layer.N.*`)
- Splits fused QKV `in_proj_weight [192, 64]` into separate Q/K/V `[64, 64]` for A blocks
- Folds BatchNorm into the encoder output projection Linear layer
- Sets `encoder_type: "hybrid"` and `meta_tokens: 4` in config.json

### 2. Export to LQ40

```bash
cargo run -p synapse --release --example export_lewm_q4 -- \
  --checkpoint /tmp/hybrid_alal/lejepa_weights.safetensors \
  --config /tmp/hybrid_alal/config.json \
  --mode full \
  --output hybrid_alal.bin
```

Output: **3.9 MB** LQ40 binary (INT8 encoder + Q4 predictor).

The LQ40 binary includes hybrid-specific weights after the standard sections:
- `meta_token` (f32, 256 floats)
- `enc_proj_weight` (f32, 4096 floats)
- `enc_proj_bias` (f32, 64 floats)

### 3. Host comparison

```bash
cargo run -p synapse --release --example lewm_compare_variants -- \
  /tmp/hybrid_alal /tmp/baseline /tmp/elastic
```

### 4. Flash to ESP32

```bash
cp hybrid_alal.bin synapse-esp32/esp-idf-app/main/model.bin
cd synapse-esp32/esp-idf-app
source ~/.espressif/esp-idf/v5.4/export.sh
idf.py build && idf.py -p /dev/cu.usbmodem* flash
```

## ESP32-P4 Performance

| Metric | 96d ViT (old) | 64d Hybrid ALAL |
|--------|--------------|-----------------|
| Binary size | 9.8 MB | **3.9 MB** |
| predict_next | 583 ms | **152 ms** |
| encode | 6,416 ms | **922 ms** |
| 3-step rollout | 1,748 ms | **460 ms** |
| Free PSRAM | 24.4 MB | 29.5 MB |

### Encoder layer breakdown

| Layer | Type | Attention | FFN | Total |
|-------|------|-----------|-----|-------|
| 0 | A (softmax) | 94 ms | 100 ms | 215 ms |
| 1 | L (kernel-trick) | 58 ms | 100 ms | 178 ms |
| 2 | A (softmax) | 94 ms | 100 ms | 215 ms |
| 3 | L (kernel-trick) | 58 ms | 100 ms | 178 ms |

Plus: patch embedding 50 ms, overhead 86 ms.

### Key ESP32 optimizations

- **Batch patch embedding**: All 256 patches extracted into one `[256, 588]` buffer, projected via a single INT8 GEMM with PIE SIMD + dual-core (470ms -> 50ms, 9.4x)
- **Kernel-trick linear attention**: L blocks use heap-allocated KV matrix (avoids stack overflow on 16KB main task stack), ELU+1 feature map, O(nd^2) compute
- **Auto-detection**: L blocks detected at runtime by empty Q/K/V biases -- no config flag needed

## Quality

Step-to-step cosine similarity: **0.993-0.994** (best of all three 64d variants).

The hybrid encoder produces lower-magnitude latents (z L2 = 0.28 vs 0.90 for baseline) but the predictor compensates within a few steps. Endpoint L2 after 20 rollout steps converges to the same range (~0.82) as baseline.

Quantization quality: cos(f32, INT8+Q4) = **0.999** -- negligible degradation.
