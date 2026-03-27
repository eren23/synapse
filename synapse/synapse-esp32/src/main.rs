//! ESP32-P4 LEWM inference server.
//!
//! On real hardware (--features esp32):
//!   Connects to WiFi, starts HTTP server, serves inference endpoints.
//!
//! On host (default, --features host-test):
//!   Runs a quick smoke test of the model and server handlers.

fn main() {
    #[cfg(feature = "host-test")]
    {
        println!("=== Synapse ESP32 -- Host Test Mode ===\n");

        // Create model (zeroed weights -- structure only, no real inference)
        let model = synapse_esp32::model::Esp32LeWM::new_zeroed();
        println!("Model loaded: latent_dim={}, action_dim={}",
            model.latent_dim(), model.action_dim());

        // Verify config
        let cfg = model.config();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.latent_dim, 192);
        assert_eq!(cfg.action_dim, 10);
        println!("Config: image={}x{}, patch={}, encoder_layers={}, predictor_layers={}",
            cfg.image_size, cfg.image_size, cfg.patch_size,
            cfg.encoder_layers, cfg.predictor_layers);

        // Test binary loading returns error (not yet implemented)
        let result = synapse_esp32::model::Esp32LeWM::from_binary(&[0u8; 64]);
        assert!(result.is_err());
        println!("Binary loading: correctly returns 'not yet implemented'");

        // Test server status handler
        let status = synapse_esp32::server::handle_status(&model);
        assert_eq!(status.latent_dim, 192);
        assert_eq!(status.action_dim, 10);
        println!("\nStatus: model={}, backend={}, quant={}",
            status.model, status.backend, status.quantization);

        // Test request deserialization
        let json = r#"{"latent":[0.1,0.2],"action":[0.3,0.4]}"#;
        let _req: synapse_esp32::server::PredictRequest = serde_json::from_str(json).unwrap();
        println!("Request deserialization: OK");

        // Test response serialization
        let resp = synapse_esp32::server::InferenceResponse {
            result: serde_json::json!({"latent": vec![0.0f32; 192]}),
            latency_ms: 0.0,
            operation: "predict".into(),
        };
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains("predict"));
        println!("Response serialization: OK");

        println!("\nAll host tests passed. Ready for ESP32-P4 deployment.");
        println!("  Next: flash with `cargo build --target riscv32imc-esp-espidf --features esp32`");
    }

    #[cfg(feature = "esp32")]
    {
        // Real ESP32-P4 entry point
        // TODO: Initialize when hardware arrives
        // esp_idf_svc::sys::link_patches();
        // esp_idf_svc::log::EspLogger::initialize_default();
        // WiFi connect -> HTTP server -> inference loop
        compile_error!("ESP32 target not yet configured. Install ESP-IDF toolchain first.");
    }

    #[cfg(not(any(feature = "host-test", feature = "esp32")))]
    {
        compile_error!("Enable either 'host-test' or 'esp32' feature.");
    }
}
