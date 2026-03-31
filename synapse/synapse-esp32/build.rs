fn main() {
    // Only emit ESP-IDF sysenv when cross-compiling for espidf targets.
    // Host builds (cargo test -p synapse-esp32) skip this entirely.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("espidf") {
        embuild::espidf::sysenv::output();
    }
}
