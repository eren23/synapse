# ESP32-P4 Deployment Guide

This guide covers deploying LEWM world model inference on the ESP32-P4 microcontroller — from toolchain installation to running a WiFi HTTP inference server on real hardware.

## Hardware

### Target Board

**Waveshare ESP32-P4-WiFi6** ([waveshare.com/esp32-p4-wifi6.htm](https://www.waveshare.com/esp32-p4-wifi6.htm))

| Feature | Specification |
|---------|--------------|
| **Chip** | ESP32-P4, revision v1.3, dual-core RISC-V RV32IMAFC @ 400 MHz |
| **PSRAM** | 32 MB, AP vendor, gen 4, HEX (X16) mode @ 200 MHz |
| **Flash** | 32 MB NOR, SPI DIO @ 80 MHz |
| **USB** | Single USB-C — USB-JTAG/Serial (native). Shows as `/dev/cu.usbmodem*` on macOS |
| **WiFi** | Via companion ESP32-C6 chip over SDIO (WiFi 6 + BLE 5) |
| **Other** | MIPI-CSI camera, MIPI-DSI display, SD card, mic, speaker headers |
| **Power** | ~78 mW (23.88 mA) active |

### ESP32-P4 Core Architecture

| Feature | Detail |
|---------|--------|
| CPU | Dual-core RISC-V RV32IMAFCZc @ 400 MHz |
| ISA Extensions | M (hardware multiply) + A (atomics) + F (FPU) + C (compressed) + Zc + Xhwlp (hardware loop) + **Xai (PIE SIMD)** |
| Internal SRAM | 768 KB L2MEM (configurable as cache or scratchpad) |
| TCM | 8 KB zero-wait Tightly Coupled Memory |
| DMA | GDMA-AHB (SRAM, 3+3 ch) + GDMA-AXI (SRAM+PSRAM, 3+3 ch) |

The **PIE (Processor Instruction Extensions)** are custom RISC-V SIMD instructions baked into the CPU — 16-wide INT8 multiply-accumulate per cycle at 400 MHz. This is distinct from PPA (Pixel Processing Accelerator), which only handles image transformations and is not useful for inference.

## Prerequisites

- macOS (Apple Silicon) or Linux x86_64
- Python 3.8+
- Rust nightly toolchain (for Rust components, optional for C-only path)

## Installation

### Step 1: Install ESP-IDF

The C application uses ESP-IDF v5.4 directly. This is the **production path** for ESP32-P4.

```bash
# Clone ESP-IDF v5.4
mkdir -p ~/.espressif
cd ~/.espressif
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git esp-idf/v5.4

# Run the install script for ESP32-P4
cd esp-idf/v5.4
./install.sh esp32p4
```

#### Python Dependency Workaround

ESP-IDF's Python dependency checker has known issues with modern setuptools:

1. `setuptools >= 82` removed `pkg_resources` — fix: `pip install "setuptools<81"` in the ESP-IDF venv
2. `ruamel.yaml` is a namespace package that `pkg_resources` can't find — pin: `pip install ruamel.yaml==0.17.40`
3. If the checker still fails, comment out `__build_check_python()` in `~/.espressif/esp-idf/v5.4/tools/cmake/build.cmake` line ~502. All Python deps are actually installed — only the checker is broken.

### Step 2: Install Rust Toolchain (Optional)

Only needed if you want to run host-side tests or the Rust model layer.

```bash
# Install espup (ESP Rust toolchain manager)
cargo install espup --locked

# Install ESP32-P4 RISC-V targets
espup install --targets esp32p4

# Install nightly + rust-src (needed for build-std)
rustup toolchain install nightly --component rust-src

# Install flashing tools
cargo install espflash --locked
cargo install ldproxy --locked
```

> **Note**: No environment sourcing is needed — `export-esp.sh` is empty for RISC-V targets.

### Step 3: Install Serial Tools

```bash
# esptool (for flashing — more reliable than espflash for P4)
pip install esptool

# pyserial (for serial monitoring)
pip install pyserial
```

## Project Structure

```
synapse/synapse-esp32/esp-idf-app/
├── CMakeLists.txt              # Top-level ESP-IDF project
├── sdkconfig.defaults          # SDK configuration (PSRAM 200MHz, WiFi placeholders)
├── sdkconfig.credentials       # WiFi SSID/pass (gitignored — create locally)
├── partitions.csv              # Custom partition table (16MB factory)
└── main/
    ├── CMakeLists.txt          # Component: sources, deps, binary data
    ├── Kconfig.projbuild       # Menuconfig: WiFi SSID/pass
    ├── idf_component.yml       # Managed deps: esp_wifi_remote, esp_hosted
    ├── lewm_types.h            # All struct/typedef definitions
    ├── app_main.c              # Boot, WiFi, HTTP server start (~100 lines)
    ├── model_loader.c/.h       # LQ40 parsing, weight deserialization (~940 lines)
    ├── inference.c/.h          # predict_next, encode_image, layer forward (~730 lines)
    ├── kernels.c/.h            # Math ops, LUTs, attention, GEMV, allocators (~890 lines)
    ├── dual_core.c/.h          # Core 1 worker, dispatch, semaphores (~120 lines)
    ├── smoke_tests.c/.h        # Boot-time smoke tests (~250 lines)
    ├── pie_gemv.c/.h           # PIE SIMD assembly + self-tests
    ├── http_server.c/.h        # HTTP endpoints, JSON, CORS (~780 lines)
    ├── wifi.c/.h               # WiFi STA via esp_hosted (C6 companion)
    ├── model.bin               # Embedded LQ40 model blob (~4 MB)
    └── index.html              # Companion web dashboard (~696 lines)
```

## Configuration

### sdkconfig.defaults

These settings are critical for performance and correctness:

```ini
# PSRAM at 200 MHz (10x bandwidth vs 20 MHz default)
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_200M=y
CONFIG_SPIRAM_USE_MALLOC=y
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=4096

# CRITICAL: Required for SPIRAM_SPEED_200M to take effect
# Without this, PSRAM silently falls back to 20 MHz!
CONFIG_IDF_EXPERIMENTAL_FEATURES=y

# 32 MB flash
CONFIG_ESPTOOLPY_FLASHSIZE_32MB=y

# Custom partition table (16 MB factory for large model binary)
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"

# WiFi via esp_hosted (C6 companion over SDIO)
CONFIG_ESP_WIFI_REMOTE_LIBRARY_HOSTED=y
CONFIG_SLAVE_IDF_TARGET_ESP32C6=y

# HTTP server tuning
CONFIG_HTTPD_MAX_REQ_HDR_LEN=1024
CONFIG_HTTPD_MAX_URI_LEN=512

# TCP buffers for image upload (602 KB raw f32 image)
CONFIG_LWIP_TCP_SND_BUF_DEFAULT=32768
CONFIG_LWIP_TCP_WND_DEFAULT=32768

# Memory
CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384
CONFIG_NEWLIB_NANO_FORMAT=y

# Logging
CONFIG_LOG_DEFAULT_LEVEL_INFO=y
CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y
```

#### PSRAM Speed History

This table illustrates why `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` is critical:

| Config | Boot Speed | encode(image) Latency |
|--------|-----------|----------------------|
| Default (no SPIRAM) | Disabled | N/A (model won't load) |
| `CONFIG_SPIRAM=y` only | 20 MHz | 81,818 ms |
| +`SPIRAM_SPEED_200M=y` (without experimental) | 20 MHz (silently ignored!) | 81,818 ms |
| +`CONFIG_IDF_EXPERIMENTAL_FEATURES=y` | **200 MHz** | **70,913 ms** |

### Partition Table

```csv
# Name,   Type, SubType, Offset,   Size,      Flags
nvs,      data, nvs,     0x9000,   0x6000,
phy_init, data, phy,     0xf000,   0x1000,
factory,  app,  factory, 0x10000,  0xFF0000,
```

The factory partition is 16 MB — large enough for the application binary with a ~10 MB embedded model blob.

## Building

### Step 1: Source ESP-IDF Environment

```bash
# IMPORTANT: Use bash, not fish. Fish needs export.fish instead.
source ~/.espressif/esp-idf/v5.4/export.sh
```

### Step 2: Select Model

Copy the desired LQ40 model binary into the build:

```bash
cd synapse/synapse-esp32/esp-idf-app

# Full model (192d, 6+6 layers, ~10 MB)
cp ../../web/lewm-compress-demo/lewm-full.bin main/model.bin

# Or slim model (96d, 4+4 layers, smaller & faster)
cp ../../web/lewm-compress-demo/lewm-slim-96d-q4.bin main/model.bin
```

### Step 3: Configure WiFi

WiFi credentials are managed via Kconfig (not hardcoded in source). Create a `sdkconfig.credentials` file (gitignored):

```bash
echo 'CONFIG_LEWM_WIFI_SSID="YourNetworkName"' > sdkconfig.credentials
echo 'CONFIG_LEWM_WIFI_PASS="YourPassword"' >> sdkconfig.credentials
```

Or use `idf.py menuconfig` → "Synapse LEWM Config" to set them interactively.

### Step 4: Build

```bash
# First-time build (or after sdkconfig changes)
rm -f sdkconfig
idf.py set-target esp32p4
cat sdkconfig.credentials >> sdkconfig   # inject WiFi creds
idf.py build

# Quick rebuild (no config changes)
idf.py build
```

Build output will be in `build/` (~35 seconds for incremental builds).

## Flashing

### Method 1: idf.py (Recommended)

```bash
idf.py -p /dev/cu.usbmodem* flash
```

### Method 2: esptool.py (If idf.py flash fails)

```bash
python -m esptool --port /dev/cu.usbmodem* --chip esp32p4 --baud 921600 \
  --after watchdog_reset write_flash --flash_mode dio --flash_size 32MB \
  0x2000 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/synapse-lewm-esp32.bin
```

> **Note**: `espflash` (the Rust tool) has a known bug where it cannot connect to ESP32-P4's USB-JTAG serial. Use `esptool.py` or `idf.py flash` instead.

## Monitoring

### Method 1: idf.py monitor

```bash
idf.py -p /dev/cu.usbmodem* monitor
```

Press `Ctrl+]` to exit.

### Method 2: pyserial (Programmatic)

`cat /dev/cu.usbmodem*` produces garbled output because macOS raw serial I/O doesn't handle USB CDC-ACM properly. Use pyserial:

```bash
python -c "
import serial, re
ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=0.5)
while True:
    chunk = ser.read(4096)
    if chunk:
        text = chunk.decode('utf-8', errors='replace')
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)  # Strip ANSI colors
        print(text, end='', flush=True)
"
```

### Method 3: screen (Interactive)

```bash
screen /dev/cu.usbmodem2101 115200
```

## Boot Sequence

On power-up, the ESP32-P4 performs these steps:

1. **Hardware init** — PSRAM detection (32 MB @ 200 MHz), flash config, dual-core startup
2. **Model loading** — Parse LQ40 blob from embedded flash: magic header, JSON config, weight data
3. **Weight setup** — Transpose INT8 weights to `[out][in_padded]` layout for PIE-friendly access
4. **PIE self-tests** — Run 4 validation tests (32, 192, 768 element dot products + Q4 block dot)
5. **Dual-core init** — Start Core 1 worker task pinned to CPU 1
6. **WiFi connect** — esp_hosted STA mode via C6 companion over SDIO
7. **Smoke tests** — predict_next, rollout(3), encode(image), encode+predict pipeline
8. **HTTP server** — Start on port 80, serve dashboard and API endpoints
9. **Ready** — Log IP address and memory stats

Example boot log:

```
I (synapse-lewm) Model: lewm-slim-96d-full (INT8+Q4)
I (synapse-lewm) Config: 96d latent, 4 encoder layers, 4 predictor layers
I (pie-test) OK test_32: result=-2720
I (pie-test) OK test_192: result=-90566
I (pie-test) OK test_768: result=119888
I (pie-test) OK test_q4: ref=-0.4625 pie=-0.4750 err=0.0125
I (pie-test) All PIE self-tests passed
I (synapse-lewm) Dual-core worker started on Core 1
I (wifi) WiFi connected, IP: 192.168.1.42
I (synapse-lewm) predict_next: 583 ms
I (synapse-lewm) rollout(3): 1748 ms
I (synapse-lewm) encode: 6416 ms
I (synapse-lewm) HTTP server started on port 80
```

## HTTP API

Once running, the device serves a REST API on port 80:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Companion web dashboard (embedded HTML) |
| `GET` | `/status` | Model info, memory stats, WiFi IP (JSON) |
| `POST` | `/predict` | Single-step prediction |
| `POST` | `/rollout` | Multi-step trajectory rollout |
| `POST` | `/encode` | Image encoding (ViT encoder) |
| `OPTIONS` | `*` | CORS preflight |

### Predict

```bash
curl -X POST http://192.168.1.42/predict \
  -H "Content-Type: application/json" \
  -d '{"latent": [0.1, 0.2, ...], "action": [0.5, 0.3]}'
```

Response:
```json
{
  "next_latent": [0.12, 0.19, ...],
  "latency_ms": 583
}
```

### Rollout

```bash
curl -X POST http://192.168.1.42/rollout \
  -H "Content-Type: application/json" \
  -d '{"latent": [0.1, ...], "actions": [[0.5, 0.3], [0.4, 0.2], [0.6, 0.1]]}'
```

Response:
```json
{
  "trajectory": [[0.12, ...], [0.14, ...], [0.15, ...]],
  "latency_ms": 1748
}
```

### Encode

```bash
# Send raw f32 image bytes (224x224x3 = 602,112 bytes)
curl -X POST http://192.168.1.42/encode \
  --data-binary @image.f32raw
```

Response:
```json
{
  "latent": [0.3, -0.1, ...],
  "latency_ms": 6416
}
```

## Companion Web Dashboard

The embedded `index.html` at `GET /` provides a full browser UI:

- **Device status** — IP, model info, heap/PSRAM usage
- **Test image generation** — PushT scene with ImageNet normalization
- **Predict/rollout controls** — Action sliders, presets, step count
- **Latent heatmap** — Visualization of the latent vector
- **2D trajectory canvas** — First two latent dimensions plotted over rollout steps
- **Hybrid WASM mode** — Toggle to run encoding in browser via WASM, prediction on device
- **Activity log** — Request/response timing

## Verifying Parity

Compare device output against the host reference:

```bash
# Generate golden reference on host
cargo run -p synapse-esp32 --example lewm_golden -- \
  --model web/lewm-compress-demo/lewm-full.bin --steps 3
```

Expected parity:

| Stage | Board | Host | Match |
|-------|-------|------|-------|
| predict_next | sum=3.144316, l2=11.316569 | same | Exact |
| encode(image) | sum=0.299396, l2=13.786463 | sum=0.300385, l2=13.786430 | Near (scalar float drift) |

The encoder shows minor drift due to accumulated floating-point differences between the C scalar path and Rust reference. The predictor matches exactly because both use the same Q4 quantized compute path.

## Troubleshooting

### PSRAM stuck at 20 MHz

**Symptom**: Boot log shows PSRAM but encode takes ~80 seconds.

**Fix**: Ensure `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` is in `sdkconfig.defaults`. Without it, `SPIRAM_SPEED_200M` is silently ignored by Kconfig. Do a clean build:

```bash
rm -f sdkconfig
idf.py set-target esp32p4
idf.py build
```

### espflash can't connect

**Symptom**: `espflash flash` says "Error while connecting to device."

**Fix**: Use `esptool.py` or `idf.py flash` instead. This is a known espflash bug with ESP32-P4's USB-JTAG serial.

### Serial monitoring shows garbled output

**Symptom**: `cat /dev/cu.usbmodem*` shows garbage characters.

**Fix**: USB CDC-ACM requires proper serial handling. Use `idf.py monitor`, `screen`, or pyserial (see Monitoring section above).

### Python dependency check fails

**Symptom**: `idf.py build` fails with ruamel.yaml or pkg_resources errors.

**Fix**: In the ESP-IDF Python venv:
```bash
pip install "setuptools<81"
pip install ruamel.yaml==0.17.40
```

If it still fails, comment out `__build_check_python()` in `~/.espressif/esp-idf/v5.4/tools/cmake/build.cmake`.

### Rust std programs crash (newlib abort)

**Symptom**: Rust binary boots then immediately `abort()` at `__retarget_lock_init_recursive`.

**Root cause**: Newlib's global constructors run `global_stdio_init()` before ESP-IDF's `esp_newlib_init()` sets up lock function pointers. The lock pointer is NULL, causing abort. This affects **all** Rust std programs on ESP32-P4.

**Workaround**: Use the C app path (`esp-idf-app/`) instead of Rust `esp-idf-svc`. The C path is the production path and has full feature parity.

**Tracking**: Waiting for upstream fix in `esp-rs/esp-idf-sys`.

## Version Matrix

| Component | Version | Status |
|-----------|---------|--------|
| ESP-IDF | v5.4 | Works (C app) |
| ESP-IDF | v5.3.2 | Works (C app), bootloader accepts v1.3 silicon |
| ESP-IDF | v5.3.0 | Rejects chip (bootloader only supports rev v0.1-v0.99) |
| esptool.py | 4.11.0 | Works |
| espflash | latest | Bug (can't connect to P4 USB-JTAG) |
| esp-idf-sys (Rust) | 0.36+ | Compiles, but runtime crash (newlib) |
| esp-idf-hal (Rust) | 0.45-0.46 | Broken (ldo.rs bug) |
| Rust nightly | any | Crashes (pre-main newlib init) |
