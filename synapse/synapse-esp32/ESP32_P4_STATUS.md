# ESP32-P4 Porting Status

Last updated: 2026-04-03

## Board

- **Board**: Waveshare ESP32-P4-WiFi6 (https://www.waveshare.com/esp32-p4-wifi6.htm)
- **Chip**: ESP32-P4, revision v1.3 (eco2), dual-core RISC-V RV32IMAFC @ 400 MHz
- **PSRAM**: 32 MB, AP vendor, generation 4, X16 (HEX) mode, running at 200 MHz (requires `CONFIG_IDF_EXPERIMENTAL_FEATURES=y`)
- **Flash**: 32 MB NOR, SPI DIO @ 80 MHz
- **USB**: Single USB-C port — USB-JTAG/Serial (native, no CH340/CP2102). Shows as `/dev/cu.usbmodem*` on macOS
- **WiFi**: Via companion ESP32-C6 chip over SDIO (WiFi 6 + BLE 5)
- **Other**: MIPI-CSI camera, MIPI-DSI display, SD card, mic, speaker headers
- **Second USB**: Unsoldered RX/TX pads (not populated on this board variant)

## What Works

1. **Cross-compilation**: `cargo +nightly build --target riscv32imafc-esp-espidf` succeeds with `esp-idf-sys v0.36+`
2. **Flashing via esptool.py**: `python -m esptool --chip esp32p4 write_flash` works perfectly
3. **C hello_world**: ESP-IDF v5.3.2 C example builds, flashes, and runs correctly:
   - "Hello world!" printed
   - "esp32p4 chip with 2 CPU core(s), silicon revision v1.3, 32MB external flash"
   - "Minimum free heap size: 608568 bytes"
   - Restarts every 10 seconds as designed
4. **Host testing**: `cargo test -p synapse-esp32` (31 tests) all pass on macOS
5. **LQ40 binary loading**: `QuantizedQ4LeWM::from_lq40_bytes()` implemented and tested
6. **ESP-IDF C app** (`esp-idf-app/`): production path for ESP32-P4 hardware
   - WiFi HTTP inference server (esp_hosted + esp_wifi_remote)
   - PIE SIMD kernels (esp.vmulas.s8.xacc, 16-wide INT8 MAC)
   - Dual-core attention (Core 1 worker for query token parallelism)
   - GELU LUT (1024 entries), tiled GEMV, PSRAM 200 MHz
   - Slim-96d-full: predict 583ms, encode 6,416ms (12.8x vs scalar baseline)
7. **Hybrid ALAL 64d encoder support** (2026-04-01):
   - New 64d custom encoder: alternating full-attention / linear-attention blocks
   - Meta token support (4 extra tokens in encoder sequence, 261 total)
   - Encoder output projection (Linear + folded BatchNorm)
   - Kernel-trick O(nd²) linear attention for L blocks (ELU+1 feature map)
   - PIE SIMD batch patch embedding (256 patches in one INT8 GEMM)
   - Binary size: 3.9 MB (vs 9.8 MB for 96d)
   - **Hybrid-64d-full: predict 152ms, encode 922ms, 3-step rollout 460ms**

### ESP32-P4 Performance Timeline

| Config | predict_next | encode | enc + 3 predict | Binary |
|--------|-------------|--------|-----------------|--------|
| 96d slim (2026-03-31) | 583 ms | 6,416 ms | 7,165 ms | 9.8 MB |
| 64d baseline (2026-04-01) | 443 ms | 4,198 ms | 5,527 ms | 10.9 MB |
| 64d hybrid ALAL | 152 ms | 1,392 ms | 1,852 ms | 3.9 MB |
| + skip softmax L blocks | 152 ms | 1,364 ms | 1,824 ms | 3.9 MB |
| + PIE batch patch embed | 152 ms | 922 ms | 1,382 ms | 3.9 MB |
| + kernel-trick attention | 152 ms | 922 ms | 1,382 ms | 3.9 MB |
| + fused ops + exp LUT (2026-04-03) | **143 ms** | **782 ms** | **1,211 ms** | **3.9 MB** |

Encode breakdown (hybrid ALAL, after optimization):
- Patch embedding: 52ms (INT8 batch GEMM)
- Layer 0 (A, softmax): 191ms (norm 2, qkv 11, attn 82, oproj 7, ffn 89)
- Layer 1 (L, kernel-trick): 161ms (norm 2, qkv 10, attn 53, oproj 6, ffn 89)
- Layer 2 (A, softmax): 192ms
- Layer 3 (L, kernel-trick): 161ms
- Overhead (norms, projectors): ~25ms

Optimizations applied (2026-04-03):
- Exp LUT (256-entry) replacing expf() in softmax and elu_plus1
- Fused bias+GELU and bias+residual loops (encoder + predictor)
- Nested (token, col) loops replacing i%dim modulo
- Reciprocal multiply in softmax (1 div instead of N)
- -O3 compiler flag, vTaskDelay→esp_task_wdt_reset
- 50-step fused rollout (MAX_PREDICTOR_SEQ_LEN=150)

## What Does NOT Work

### Rust std programs crash during newlib stdio initialization

**Symptom**: Boot log shows normal ESP-IDF startup, then immediately after `cpu_start: Multicore app`:
```
abort() was called at PC 0x4ff01b61 on core 0
```
followed by a register dump and reboot.

**Crash backtrace** (decoded via `riscv32-esp-elf-addr2line`):
```
abort()
  <- __assert_func
    <- __ubsan_include
      <- __retarget_lock_init_recursive   <-- THIS IS THE ROOT CAUSE
        <- global_stdio_init.part.0  (findfp.c)
          <- __sinit
            <- _vfprintf_r
```

**Root cause**: Rust's `build-std` compiles its own newlib. Newlib has global constructors (`.init_array`) that run `global_stdio_init` which calls `__retarget_lock_init_recursive`. This function requires ESP-IDF's `esp_newlib_init()` to have already set up the lock function pointers. But `esp_newlib_init()` runs in a later init stage (`esp_system_init_fn_init_components0`). The lock function pointer is NULL -> `abort()`.

This affects ALL Rust std programs on ESP32-P4, not just synapse. Confirmed by building a minimal 3-line Rust project (just `link_patches` + `log::info!` + loop) — same crash.

### espflash cannot connect to USB-JTAG

**Symptom**: `espflash flash` says "Error while connecting to device" even though the port exists.

**Workaround**: Use `esptool.py` instead:
```bash
python -m esptool --port /dev/cu.usbmodem* --chip esp32p4 --baud 921600 \
  --after watchdog_reset write_flash --flash_mode dio --flash_size 32MB \
  0x2000 $BUILD/bootloader/bootloader.bin \
  0x8000 $BUILD/partition_table/partition-table.bin \
  0x10000 /tmp/synapse-esp32.bin
```

### PSRAM speed stuck at 20 MHz (FIXED)

**Root cause**: `SPIRAM_SPEED_200M` depends on `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` in ESP-IDF v5.4. Without it, the Kconfig choice silently falls back to 20 MHz. **Fix**: add `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` to `sdkconfig.defaults`.

### Serial monitoring from non-TTY

`cat /dev/cu.usbmodem*` produces garbled output because macOS raw serial I/O doesn't handle USB CDC-ACM properly. Solutions:
- `screen /dev/cu.usbmodem* 115200` (interactive)
- Python `serial.Serial()` from pyserial (programmatic — this is how we captured all crash logs)

## Toolchain Setup (macOS Apple Silicon)

### Installation
```bash
# 1. Install espup (ESP Rust toolchain manager)
cargo install espup --locked

# 2. Install ESP32-P4 RISC-V targets
espup install --targets esp32p4
# Installs: riscv32imc, riscv32imac, riscv32imafc targets

# 3. No environment sourcing needed (export-esp.sh is empty for RISC-V)

# 4. Install nightly + rust-src (needed for build-std)
rustup toolchain install nightly --component rust-src

# 5. Install flashing tools
cargo install espflash --locked
cargo install ldproxy --locked

# 6. Run ESP-IDF install script for Python deps
cd ~/.espressif/esp-idf/v5.3.2 && ./install.sh esp32p4
```

### Python dependency check issue

ESP-IDF's `idf_tools.py check-python-dependencies` fails because:
1. `setuptools >= 82` removed `pkg_resources` (fix: `pip install "setuptools<70"`)
2. `ruamel.yaml` is a namespace package that `pkg_resources` can't find via normal dist metadata
3. Even after fixing both, the checker cascades through packages unreliably

**Workaround**: Comment out `__build_check_python()` in `~/.espressif/esp-idf/v5.3.2/tools/cmake/build.cmake` line 502. All Python deps are actually installed correctly — only the checker is broken.

### ESP-IDF DSI header issue (v5.3.0 only)

`esp_lcd_mipi_dsi.h` has `struct extra_flags` which causes a bindgen "redefinition" error. Fixed by renaming to anonymous `struct`. Not an issue in v5.3.2.

## Complete Version Matrix

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **esp-idf-sys** | 0.35.0 | BROKEN | Passes `--target=riscv32-none` to GCC (GCC doesn't understand Clang flag) |
| **esp-idf-sys** | 0.36.1 | COMPILES | "Using clang from ESP-IDF when available" — fixes the GCC flag issue |
| **esp-idf-sys** | 0.37.2 | COMPILES | Same fix, newer APIs |
| **esp-idf-hal** | 0.44.x | N/A | Pairs with sys 0.35 (which is broken) |
| **esp-idf-hal** | 0.45.2 | BROKEN | `ldo.rs:100`: `new_bitfield_1()` takes 2 args but called with 3. Missing `bypass` field in v5.3.x bindings |
| **esp-idf-hal** | 0.46.2 | BROKEN | Same ldo.rs issue (master == v0.46.2, no fix committed) |
| **esp-idf-svc** | 0.49.x | N/A | Pairs with hal 0.44 / sys 0.35 |
| **esp-idf-svc** | 0.50.x | BLOCKED | Requires hal 0.45 which is broken |
| **esp-idf-svc** | 0.52.x | BLOCKED | Requires hal 0.46 which is broken |
| **ESP-IDF** | v5.3.0 | REJECTS CHIP | Bootloader only supports rev v0.1-v0.99, board is v1.3 |
| **ESP-IDF** | v5.3.2 | WORKS (C) | C hello_world runs. Rust crashes at newlib init. Bootloader accepts v1.3 |
| **ESP-IDF** | v5.4.0 | TOOLCHAIN MISMATCH | Needs `esp-14.2.0` GCC but `espup` installs `esp-13.2.0` |
| **Rust nightly** | 1.93.0 (2025-12-01) | CRASHES | Same newlib abort |
| **Rust nightly** | 1.96.0 (2026-03-29) | CRASHES | Same newlib abort |
| **embuild** | 0.33 | OK | With `features = ["espidf"]` |
| **ldproxy** | 0.3.4 | OK | Linker wrapper for ESP-IDF |
| **espflash** | latest | BUG | Cannot connect to ESP32-P4 USB-JTAG serial. Use esptool.py instead |
| **esptool.py** | 4.11.0 | OK | Connects, flashes, resets correctly |

## Attempted Workarounds (all failed for the newlib crash)

1. **Different nightly versions** (1.93, 1.96) — same crash
2. **With/without esp-idf-svc** — `link_patches()` from either sys or svc doesn't help
3. **PSRAM enabled/disabled** — crashes either way (faster without PSRAM)
4. **Single-core mode** (`CONFIG_FREERTOS_UNICORE=y`) — crashes even faster
5. **Nano newlib** (`CONFIG_NEWLIB_NANO_FORMAT=y`) — same abort
6. **Minimal code** (just `puts` + `vTaskDelay`, no Rust stdio) — still crashes because newlib global constructors run before main
7. **Raw ESP_LOG FFI** (bypass Rust log crate) — doesn't help, crash is pre-main
8. **Patched esp-idf-hal ldo.rs** (removed 3rd arg) — compiles but same runtime crash

## Build & Flash Commands

```bash
# Build (from synapse/ directory)
cargo +nightly build -p synapse-esp32 \
  --target riscv32imafc-esp-espidf \
  --features esp32 --release

# Create flashable image
espflash save-image --chip esp32p4 \
  target/riscv32imafc-esp-espidf/release/synapse-esp32 \
  /tmp/synapse-esp32.bin

# Flash (3-part: bootloader + partition table + app)
BUILD=$(ls -d target/riscv32imafc-esp-espidf/release/build/esp-idf-sys-*/out/build | head -1)
python -m esptool --port /dev/cu.usbmodem* --chip esp32p4 --baud 921600 \
  --after watchdog_reset write_flash --flash_mode dio --flash_size 32MB \
  0x2000 "$BUILD/bootloader/bootloader.bin" \
  0x8000 "$BUILD/partition_table/partition-table.bin" \
  0x10000 /tmp/synapse-esp32.bin

# Monitor (via pyserial, since raw cat doesn't work with USB CDC-ACM)
python -c "
import serial, time, re
ser = serial.Serial('/dev/cu.usbmodem*', 115200, timeout=0.5)
while True:
    chunk = ser.read(4096)
    if chunk:
        text = chunk.decode('utf-8', errors='replace')
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        print(text, end='', flush=True)
"
```

## Architecture Notes

The `synapse-esp32` crate is structured for dual-mode operation:
- `--features host-test` (default): Builds and tests on Mac/Linux without ESP32 hardware
- `--features esp32`: Builds for real ESP32-P4 target with `esp-idf-svc` bindings

The inference engine uses `pure-rust` feature (no Zig FFI) since Zig kernels don't target ESP-IDF.
Model weights stay in Q4 format in PSRAM (~12-17 MB) — cannot dequantize to f32 (would need ~54 MB).

## Next Steps

1. **Track esp-rs/esp-idf-sys and esp-rs/esp-idf-hal** for ESP32-P4 newlib fix and ldo.rs fix
2. **Try `no_std` bare metal** with `esp-hal` crate — avoids newlib entirely, but requires rewriting without `std`
3. **Try ESP-IDF v5.4** when `espup` updates its toolchain to `esp-14.2.0`
4. **Consider contributing upstream**: Fix newlib init ordering in esp-idf-sys (ensure `esp_newlib_init()` runs before `.init_array`)
5. **File bug reports** on esp-rs/esp-idf-sys with this document as evidence
