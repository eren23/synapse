# ESP-IDF LEWM Inference Server for ESP32-P4

WiFi-connected HTTP inference server running on ESP32-P4 with embedded LEWM world model. Serves a browser companion dashboard and exposes JSON API endpoints for predict, rollout, and encode.

## Current Status (2026-03-31)

**Working on real Waveshare ESP32-P4-WIFI6 hardware:**

- Full LEWM model (INT8 encoder + Q4 predictor) loaded from embedded LQ40 blob
- WiFi connected via ESP32-C6 companion chip (esp_hosted + esp_wifi_remote over SDIO)
- HTTP server on port 80 with companion web dashboard
- PSRAM running at 200 MHz (requires `CONFIG_IDF_EXPERIMENTAL_FEATURES=y`)
- 32 MB PSRAM detected, ~33.5 MB free after model load
- Smoke tests run on boot: predict_next, rollout, encode, encode+predict

### Benchmark Results (PSRAM @ 200 MHz, scalar C, no PIE)

| Operation | Latency | Notes |
|-----------|---------|-------|
| predict_next | 3,009 ms | 6 adaLN layers, 3 tokens, Q4 GEMV |
| rollout (3 steps) | 9,028 ms | ~3,009 ms/step |
| encode(image) | 70,913 ms | 6 ViT layers, 257 tokens, INT8 GEMV |
| encode + predict | 73,921 ms | End-to-end |

### Parity (board vs Rust host reference)

| Stage | Board | Host | Match |
|-------|-------|------|-------|
| predict_next | sum=3.144316 l2=11.316569 | same | exact |
| encode(image) | sum=0.299396 l2=13.786463 | sum=0.300385 l2=13.786430 | near (scalar drift) |

## HTTP API

All endpoints include CORS headers for browser access.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Companion web dashboard (embedded HTML) |
| GET | `/status` | Model info, memory, WiFi IP (JSON) |
| POST | `/predict` | `{latent:[...], action:[...]}` → `{next_latent:[...], latency_ms}` |
| POST | `/rollout` | `{latent:[...], actions:[[...],...]}` → `{trajectory:[[...],...], latency_ms}` |
| POST | `/encode` | Raw f32 binary body (602KB) → `{latent:[...], latency_ms}` |
| OPTIONS | `/predict`, `/rollout`, `/encode` | CORS preflight |

## How to Build & Flash

### Prerequisites

- ESP-IDF v5.4 installed (`~/.espressif/esp-idf/v5.4/`)
- Waveshare ESP32-P4-WIFI6 board connected via USB
- If `idf.py` Python deps fail with ruamel.yaml errors, downgrade setuptools: `pip install 'setuptools<81'` in the ESP-IDF venv, and pin `ruamel.yaml==0.17.40`

### Steps

```bash
# 1. Source ESP-IDF (use bash, not fish -- fish needs export.fish)
source ~/.espressif/esp-idf/v5.4/export.sh

# 2. Navigate to this directory
cd synapse/synapse-esp32/esp-idf-app

# 3. Replace placeholder model with a real LQ40 blob
cp ../../web/lewm-compress-demo/lewm-full.bin main/model.bin
# Or for slim: cp ../../web/lewm-compress-demo/lewm-slim-96d-q4.bin main/model.bin

# 4. Set WiFi credentials in main/app_main.c
#    Edit WIFI_SSID and WIFI_PASS defines near bottom of file

# 5. Clean build (first time or after sdkconfig changes)
rm -f sdkconfig
idf.py set-target esp32p4
idf.py build

# 6. Flash and monitor
idf.py -p /dev/cu.usbmodem* flash
# Monitor in a separate terminal (idf.py monitor needs TTY):
idf.py -p /dev/cu.usbmodem* monitor
```

### Quick Rebuild (no config changes)

```bash
source ~/.espressif/esp-idf/v5.4/export.sh
idf.py build && idf.py -p /dev/cu.usbmodem* flash
```

### Verify Host Reference

```bash
# Generate golden payload to compare against board output
cargo run -p synapse-esp32 --example lewm_golden -- --model web/lewm-compress-demo/lewm-full.bin --steps 3
```

## Architecture

```
Browser (same WiFi)
  ├── GET /           → companion dashboard (index.html)
  ├── POST /predict   → JSON latent+action → next_latent
  └── POST /rollout   → JSON latent+actions → trajectory
        │
        │  WiFi (esp_hosted, C6 over SDIO)
        │
ESP32-P4 (360 MHz dual-core RISC-V, 32MB PSRAM @ 200MHz)
  ├── app_main.c      → boot, model load, WiFi, HTTP server
  ├── wifi.c          → STA connection via esp_hosted
  ├── http_server.c   → ESP-IDF httpd, cJSON, CORS
  ├── index.html      → embedded companion web UI
  └── model.bin       → LQ40 blob (embedded in flash)
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `main/app_main.c` | ~2150 | Model loading, inference (predict_next, encode_image), boot sequence |
| `main/http_server.c` | ~650 | HTTP endpoints, JSON parsing, CORS |
| `main/http_server.h` | 33 | ServerConfig struct, API |
| `main/wifi.c` | 116 | WiFi STA init (esp_hosted + esp_wifi_remote) |
| `main/wifi.h` | 22 | wifi_init_sta(), wifi_get_ip() |
| `main/index.html` | 696 | Companion dashboard: status, predict, rollout, trajectory viz |
| `main/CMakeLists.txt` | 9 | Build config, component deps |
| `main/idf_component.yml` | 5 | Managed deps: esp_wifi_remote, esp_hosted |
| `main/model.bin` | ~10MB | Embedded LQ40 model blob |
| `sdkconfig.defaults` | 18 | PSRAM 200MHz, flash 32MB, custom partitions, WiFi |
| `partitions.csv` | 4 | 16MB factory partition for large model binary |

## sdkconfig.defaults

```
CONFIG_IDF_EXPERIMENTAL_FEATURES=y          # Required for PSRAM 200MHz
CONFIG_ESPTOOLPY_FLASHSIZE_32MB=y           # 32MB NOR flash
CONFIG_SPIRAM=y                             # Enable PSRAM
CONFIG_SPIRAM_SPEED_20M=n                   # Explicitly unset 20MHz default
CONFIG_SPIRAM_SPEED_200M=y                  # 200MHz PSRAM (10x bandwidth)
CONFIG_PARTITION_TABLE_CUSTOM=y             # Custom partitions.csv
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
CONFIG_ESP_WIFI_REMOTE_LIBRARY_HOSTED=y     # WiFi via esp_hosted
CONFIG_SLAVE_IDF_TARGET_ESP32C6=y           # C6 companion chip
CONFIG_HTTPD_MAX_REQ_HDR_LEN=1024           # HTTP server tuning
CONFIG_HTTPD_MAX_URI_LEN=512
CONFIG_LWIP_TCP_SND_BUF_DEFAULT=32768       # TCP buffers for image upload
CONFIG_LWIP_TCP_WND_DEFAULT=32768
```

**Critical**: `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` is required because `SPIRAM_SPEED_200M` depends on it in ESP-IDF v5.4's Kconfig. Without it, PSRAM silently falls back to 20 MHz.

## PSRAM Speed History

| Config | Boot Speed | encode latency |
|--------|-----------|----------------|
| Default (no SPIRAM config) | 0 MHz (disabled) | N/A (model won't load) |
| CONFIG_SPIRAM=y (default speed) | 20 MHz | 81,818 ms |
| +CONFIG_SPIRAM_SPEED_200M=y (without experimental) | 20 MHz (silently ignored) | 81,818 ms |
| +CONFIG_IDF_EXPERIMENTAL_FEATURES=y | **200 MHz** | **70,913 ms** |

## Companion Web Dashboard

The embedded `index.html` provides:
- Device status (IP, model info, heap/PSRAM)
- Test image generation (PushT scene, ImageNet-normalized)
- Predict/rollout controls with action sliders and presets
- Latent heatmap visualization
- 2D trajectory canvas (first two latent dims)
- Hybrid WASM mode toggle (for browser-side encode)
- Activity log

## Next Steps

1. **PIE SIMD kernels** -- 16-wide INT8 MAC for GEMV hot loops (esp.vmac.s8). Target: 5-8x speedup on predict_next and encode.
2. **Slim model testing** -- Flash slim-96d-q4 or slim-48d-2e2p for faster inference.
3. **Encoder parity** -- Tighten or codify tolerance for scalar C vs Rust host drift.
4. **Camera input** -- Replace deterministic test image with real camera or image upload.
