# LEWM Slim Architecture Experiments — Handover Guide

## What Was Built

This session added multi-architecture LEWM slim model support to synapse. The slim models use a **latent bottleneck**: the ViT encoder and DiT predictor internally run at 192d (same as baseline), but the latent representation is compressed to 48-192d via learned projection layers (`input_proj`, `cond_proj`).

### Components delivered:

| Component | Purpose |
|-----------|---------|
| `scripts/convert_lewm_ckpt.py` | Converts crucible `.ckpt` to safetensors + config.json |
| `LeWMConfig::from_json()` | Auto-detect model config from JSON |
| `input_proj` / `cond_proj` in LeWorldModel | Projection layers bridging latent_dim to predictor_hidden |
| Zig projection kernel (`zig/src/ops/projection.zig`) | SIMD-optimized small-K GEMV for projections |
| All quantization variants updated | Q4, INT8, ternary, full Q4 all support projections |
| `examples/lewm_slim_vs_baseline.rs` | Multi-model native benchmark |
| `export_lewm_q4 --config` flag | Export slim models to LQ40 binary |
| WASM demo slim support | RealLeWM, RealLeWMQ4, RealLeWMFullQ4 all handle slim |
| Slim model selector in demo HTML | Load & benchmark slim models in browser |

---

## Architecture

### Slim model structure

All slim variants share the same encoder/predictor **internal** dimensions (192d). The "slim" is in the latent space:

```
Image [224x224x3]
  |
  v
ViT Encoder (192d, N encoder layers)
  |
  v
Projector: 192d -> 1024 (GELU) -> latent_dim    [e.g. 96d]
  |
  v
Action Encoder: 10d -> 384 (GELU) -> latent_dim  [96d]
  |
  v
Build sequence: [z_t, a_embed, zeros] at latent_dim
  + pos_embedding [3, latent_dim]
  |
  v
input_proj: [3, latent_dim] -> [3, 192]          ← NEW for slim
cond_proj:  [latent_dim] -> [192]                 ← NEW for slim
  |
  v
DiT Predictor (192d, N predictor layers, adaLN)
  |
  v
pred_proj: 192d -> 1024 (GELU) -> latent_dim     [96d]
  |
  v
Output: z_{t+1} [latent_dim]
```

When `latent_dim == predictor_hidden` (192, baseline), `input_proj` and `cond_proj` are empty and skipped.

### W&B variants

Project: `eren23/crucible-lewm`

| Variant | Latent | Enc | Pred | Artifact name |
|---------|--------|-----|------|---------------|
| lewm_slim_48d_2e_2p | 48 | 2 | 2 | `lewm_slim_48d_2e_2p_epoch_1` |
| lewm_slim_64d_3e_3p | 64 | 3 | 3 | `lewm_slim_64d_3e_3p_epoch_1` |
| lewm_slim_96d_2e_3p | 96 | 2 | 3 | `lewm_slim_96d_2e_3p_epoch_1` |
| lewm_slim_96d_4e_4p | 96 | 4 | 4 | `lewm_slim_96d_4e_4p_epoch_1` |
| lewm_slim_128d_4e_4p | 128 | 4 | 4 | `lewm_slim_128d_4e_4p_epoch_1` |
| lewm_slim_192d_4e_4p | 192 | 4 | 4 | `lewm_slim_192d_4e_4p_epoch_1` |
| lewm (baseline) | 192 | 6 | 6 | existing at `/tmp/lewm-pusht/` |

---

## How to Run Experiments

### 1. Download checkpoint from W&B

```bash
pip install wandb
wandb login
python -c "
import wandb
api = wandb.Api()
art = api.artifact('eren23/crucible-lewm/lewm_slim_96d_4e_4p_epoch_1:latest')
art.download()
"
```

### 2. Convert to safetensors

```bash
python3 synapse/scripts/convert_lewm_ckpt.py \
  --input ~/Downloads/lewm_slim_96d_4e_4p_epoch_1_object.ckpt \
  --output weights/slim_96d_4e_4p/
```

Output: `weights/slim_96d_4e_4p/lejepa_weights.safetensors` + `config.json`

Batch conversion:
```bash
python3 synapse/scripts/convert_lewm_ckpt.py \
  --input-dir ~/Downloads/ \
  --output-dir weights/
```

### 3. Native benchmark

```bash
cd synapse

# Single model vs baseline
cargo run --release --example lewm_slim_vs_baseline -- \
  --slim ../weights/slim_96d_4e_4p/ \
  --baseline /tmp/lewm-pusht/pusht/lejepa_weights.safetensors

# All models in a directory
cargo run --release --example lewm_slim_vs_baseline -- \
  --models-dir ../weights/ \
  --baseline /tmp/lewm-pusht/pusht/lejepa_weights.safetensors
```

Output: comparison table with f32 size, Q4 size, Q4-full size, speed, cosine similarity.

### 4. Export for WASM demo

```bash
# Q4 predictor (f32 encoder + Q4 predictor)
cargo run --release --example export_lewm_q4 -- \
  --checkpoint ../weights/slim_96d_4e_4p/lejepa_weights.safetensors \
  --config ../weights/slim_96d_4e_4p/config.json \
  --mode q4-pred \
  --output web/lewm-compress-demo/lewm-slim-96d-q4.bin

# Full Q4 (INT8 encoder + Q4 predictor, smallest)
cargo run --release --example export_lewm_q4 -- \
  --checkpoint ../weights/slim_96d_4e_4p/lejepa_weights.safetensors \
  --config ../weights/slim_96d_4e_4p/config.json \
  --mode full \
  --output web/lewm-compress-demo/lewm-slim-96d-full.bin
```

### 5. Export f32 reference for WASM comparison

```python
# synapse/scripts/export_slim_f32_wasm.py (or inline)
from safetensors import safe_open
import struct, json

st = safe_open('weights/slim_96d_4e_4p/lejepa_weights.safetensors', framework='pt', device='cpu')
# Skip BatchNorm keys (projector.net.1.*, pred_proj.net.1.*)
skip = {k for k in st.keys() if '.net.1.' in k and ('projector' in k or 'pred_proj' in k)}
keys = sorted(k for k in st.keys() if k not in skip)

tensor_map, data_parts, float_offset = {}, [], 0
for k in keys:
    t = st.get_tensor(k)
    flat = t.float().contiguous().numpy().tobytes()
    tensor_map[k] = {'shape': list(t.shape), 'offset': float_offset, 'len': len(flat)//4}
    data_parts.append(flat)
    float_offset += len(flat)//4

header = json.dumps(tensor_map).encode()
with open('synapse/web/lewm-compress-demo/lewm-slim-96d-f32.bin', 'wb') as f:
    f.write(struct.pack('<I', len(header)) + header + b''.join(data_parts))
```

### 6. Build WASM and test in browser

```bash
cd synapse/synapse-wasm
wasm-pack build --target web --release --out-dir pkg
# Then open http://localhost:8080/web/lewm-compress-demo/
```

### 7. ESP32-P4 deployment (target)

The smallest models (48d/2e2p at Q4-full) should fit in ~3MB. Use the same export pipeline:
```bash
cargo run --release --example export_lewm_q4 -- \
  --checkpoint ../weights/slim_48d_2e_2p/lejepa_weights.safetensors \
  --config ../weights/slim_48d_2e_2p/config.json \
  --mode full \
  --output esp32/lewm-tiny.bin
```

The ESP32 inference code in `synapse-esp32/` uses the same LQ40 loading path.

### ESP32-P4 status as of 2026-03-31

Done on real hardware:

- ESP-IDF C app boots, loads model, connects WiFi, serves HTTP on ESP32-P4 with 32 MB PSRAM @ 200 MHz.
- Slim `q4-pred` predictor parity is confirmed against the Rust host reference.
- Full `INT8+Q4` predictor parity is confirmed against the Rust host reference.
- Full `INT8+Q4` `encode(image)` and `encode(image) + predict_next(action)` execute on-device.
- Deterministic patch embedding matches the Rust host probe at the printed precision.
- Short rollout smoke tests complete on-board without the task watchdog firing.
- **WiFi HTTP inference server** live via esp_hosted (ESP32-C6 companion over SDIO).
- **Companion web dashboard** with predict/rollout/encode controls and trajectory visualization.
- **PSRAM at 200 MHz** (`CONFIG_IDF_EXPERIMENTAL_FEATURES=y` required in sdkconfig).
- **PIE SIMD kernels**: INT8 GEMV, Q4 GEMV, attention QK^T dot products (esp.vmulas.s8.xacc).
- **Dual-core attention**: Core 1 worker handles second half of 257 query tokens.
- **GELU LUT**: 1024-entry lookup table replacing tanhf().
- **Tiled GEMV**: weights-outer loop order for PSRAM cache reuse.
- **4 on-boot PIE self-tests** (32, 192, 768 elements + Q4 block).

Final benchmarks with slim-96d-full model (4 encoder + 4 predictor layers):

| Operation | Scalar baseline | With all PIE + dual-core | Speedup |
|-----------|----------------|-------------------------|---------|
| predict_next | 3,037 ms | **583 ms** | 5.2x |
| encode(image) | 81,818 ms | **6,416 ms** | 12.8x |

What is still not done on ESP32-P4:

- Encoder parity is near-match rather than exact-match (acceptable for INT8 quantized path).
- No camera or real image input (test image is deterministic).
- No 48d/2e2p slim model tested on hardware yet (code supports it, needs checkpoint export).

---

## Key Files

| File | Role |
|------|------|
| **Pipeline** | |
| `scripts/convert_lewm_ckpt.py` | .ckpt to safetensors converter |
| `examples/export_lewm_q4.rs` | safetensors to LQ40 binary exporter (supports `--config`) |
| `examples/lewm_slim_vs_baseline.rs` | Multi-model native benchmark |
| **Core model** | |
| `crates/synapse-inference/src/models/vision/lewm.rs` | LeWorldModel with projection support, `LeWMConfig::from_json()` |
| `crates/synapse-inference/src/quantization/vision/q4_lewm.rs` | Q4 + CachedQ4 with projections |
| `crates/synapse-inference/src/quantization/vision/int8_lewm.rs` | INT8 with projections |
| `crates/synapse-inference/src/quantization/vision/full_q_lewm.rs` | FullQ + Q4Full with projections |
| `crates/synapse-inference/src/quantization/vision/ternary_lewm.rs` | Ternary with projections |
| **Zig kernel** | |
| `zig/src/ops/projection.zig` | SIMD projection GEMV kernel |
| `crates/synapse-inference/src/ops/projection.rs` | Rust dispatch (Zig FFI or pure-Rust fallback) |
| **WASM** | |
| `synapse-wasm/src/lib.rs` | RealLeWM, RealLeWMQ4, RealLeWMFullQ4 with slim support |
| **Demo** | |
| `web/lewm-compress-demo/index.html` | Browser demo with slim model selector |
| `web/lewm-compress-demo/lewm-slim-*.bin` | Pre-exported slim model binaries |

---

## Current Benchmark Results (96d/4e/4p vs baseline)

| Model | f32 Size | f32 Speed | Q4 Size | Q4 cos | Q4f Size |
|-------|----------|-----------|---------|--------|----------|
| baseline 192d/6e/6p | 54.6 MB | 30 ms | 18.3 MB | 0.9957 | 9.8 MB |
| slim 96d/4e/4p | 36.8 MB | 18 ms | 12.6 MB | 0.9982 | 6.5 MB |

Slim is 1.9x faster at f32, 31% smaller at Q4, and has better Q4 quality (fewer layers = less quantization error accumulation).

WASM demo comparison (slim Q4-full vs slim f32): cos ~0.94 (the INT8 encoder adds noise).

---

## Known Issues / Limitations

1. **Metal GPU fused shaders** have hardcoded `H=192, INNER=1024`. Slim models fall back to non-fused Metal path (still works, slightly slower).

2. **BatchNorm in projectors**: The LQ40 export format doesn't preserve BatchNorm weights from the projector/pred_proj. The WASM f32 reference binary also excludes them for consistency. This means the projector acts as Linear+GELU only (no BN normalization). Doesn't affect Q4 vs f32 comparison since both skip BN.

3. **Ternary quantization**: Uses `seq_len = latent/hidden` instead of `seq_len = 3`. This is a pre-existing architectural difference in how ternary stores sequences.

4. **WASM code duplication**: The slim projection logic is implemented separately in RealLeWM, RealLeWMQ4, and RealLeWMFullQ4. The WASM crate doesn't use the synapse-inference structs directly (it has its own pure-Rust implementation for WASM compat).

---

## What's Next

1. **Download and benchmark ALL variants** — the 48d, 64d, 128d, 192d models are training. Download each, convert, benchmark, find the Pareto curve.

2. **ESP32-P4 end-to-end path** — predictor-only parity and scalar end-to-end encoder execution are done. The next milestone is tightening encoder parity, then replacing the built-in smoke image with a real input path.

3. **FPGA hardwired weights** — the existing `fpga/` experiment can use slim model Q4 weights. Fewer layers = fewer LUTs = fits smaller FPGAs.

4. **Train longer** — current models are epoch 1. Training to convergence (100 epochs like the baseline expert) will improve quality. The architecture comparison should be redone with fully-trained models.

5. **Cross-architecture comparison** — compare slim LEWM vs other architectures (SSMs, DeltaNet) at similar parameter counts.

6. **Pruning + slim** — combine Wanda pruning with slim architectures for maximum compression. The 48d/2e2p + Q4 + Wanda40% could be under 2MB.

---

## ESP32-P4 Remaining Work Checklist

- Tighten `encode(image)` parity against the Rust host reference or codify an acceptable tolerance.
- Keep the stage probes for:
  - `patch0_embed`
  - `patch0_with_pos`
  - `layer0_cls`
  - `cls_norm`
- Add a board-side transport for inputs and outputs:
  - serial command protocol first
  - optional WiFi/HTTP after parity is stable
- Replace the built-in deterministic smoke image with that transport-fed input path.
- Export and test the smallest actual target model locally:
  - `48d/2e2p`
  - likely `full` or heavily-pruned `q4-pred`
- Benchmark current scalar board latency and memory for:
  - slim `q4-pred`
  - slim `full`
  - full `INT8+Q4`
  - current `full` scalar end-to-end is about `71.1 s` for `encode(image)` and `74.1 s` for `encode(image) + predict_next(action)`
- Only after the above, add PIE kernels for:
  - Q4/INT8 GEMV hot loops
  - layernorm / vecops
  - optional activation LUTs
