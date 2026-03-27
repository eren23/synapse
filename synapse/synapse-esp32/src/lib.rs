//! Synapse ESP32-P4: LEWM inference on a $10 RISC-V microcontroller.
//!
//! Architecture:
//!   Phone camera -> WiFi HTTP -> ESP32 LEWM inference -> JSON response
//!
//! Build for host testing:  cargo build -p synapse-esp32
//! Build for ESP32-P4:      cargo build --target riscv32imc-esp-espidf --features esp32

pub mod model;
pub mod server;
