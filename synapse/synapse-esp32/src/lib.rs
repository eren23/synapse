//! Synapse ESP32-P4: multi-model inference on a $10 RISC-V microcontroller.
//!
//! Supported models:
//!   - LeWM (world model): encode, predict, rollout
//!   - Mamba Q4 (language model): text generation
//!   - RWKV-7 Q4 (language model): text generation
//!
//! Architecture:
//!   Phone camera / text -> WiFi HTTP -> ESP32 inference -> JSON response
//!
//! Build for host testing:  cargo build -p synapse-esp32
//! Build for ESP32-P4:      cargo build --target riscv32imc-esp-espidf --features esp32

pub mod model;
pub mod server;
