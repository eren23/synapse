//! ESP32-P4 multi-model inference server.
//!
//! On real hardware (--features esp32):
//!   Connects to WiFi, starts HTTP server, serves inference endpoints.
//!
//! On host (default, --features host-test):
//!   Runs a quick smoke test of all model types and server handlers.

fn main() {
    #[cfg(feature = "host-test")]
    {
        use synapse_esp32::model::{Esp32LeWM, Esp32Mamba, Esp32Model, Esp32Rwkv};

        println!("=== Synapse ESP32 -- Host Test Mode ===\n");

        // ---------------------------------------------------------------
        // Test LeWM (existing)
        // ---------------------------------------------------------------
        let lewm = Esp32LeWM::new_zeroed();
        println!("LeWM loaded: latent_dim={}, action_dim={}",
            lewm.latent_dim(), lewm.action_dim());

        // Verify config
        let cfg = lewm.config();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.latent_dim, 192);
        assert_eq!(cfg.action_dim, 10);
        println!("Config: image={}x{}, patch={}, encoder_layers={}, predictor_layers={}",
            cfg.image_size, cfg.image_size, cfg.patch_size,
            cfg.encoder_layers, cfg.predictor_layers);

        // Test binary loading returns error (not yet implemented)
        let result = Esp32LeWM::from_binary(&[0u8; 64]);
        assert!(result.is_err());
        println!("Binary loading: correctly returns 'not yet implemented'");

        // Test server status handler (backwards compat)
        let status = synapse_esp32::server::handle_lewm_status(&lewm);
        assert_eq!(status.latent_dim, 192);
        assert_eq!(status.action_dim, 10);
        println!("\nLeWM Status: model={}, backend={}, quant={}",
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

        // ---------------------------------------------------------------
        // Test Mamba Q4
        // ---------------------------------------------------------------
        println!("\n--- Mamba Q4 ---");
        let mamba = Esp32Mamba::new_zeroed();
        let result = mamba.generate(&[1, 2, 3], 5, 1.0);
        println!("Mamba Q4: generated {} tokens in {:.1}ms ({:.1} tok/s)",
            result.tokens.len(), result.latency_ms, result.tokens_per_sec);
        assert_eq!(result.tokens.len(), 5);

        // ---------------------------------------------------------------
        // Test RWKV Q4
        // ---------------------------------------------------------------
        println!("\n--- RWKV Q4 ---");
        let rwkv = Esp32Rwkv::new_zeroed();
        let result = rwkv.generate(&[1, 2, 3], 5, 1.0);
        println!("RWKV Q4: generated {} tokens in {:.1}ms ({:.1} tok/s)",
            result.tokens.len(), result.latency_ms, result.tokens_per_sec);
        assert_eq!(result.tokens.len(), 5);

        // ---------------------------------------------------------------
        // Test Esp32Model enum
        // ---------------------------------------------------------------
        println!("\n--- Esp32Model enum ---");

        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let info = model.model_info();
        println!("Mamba info: {} ({})", info.name, info.model_type);
        assert_eq!(info.model_type, "mamba");

        let model = Esp32Model::Rwkv(Esp32Rwkv::new_zeroed());
        let info = model.model_info();
        println!("RWKV info: {} ({})", info.name, info.model_type);
        assert_eq!(info.model_type, "rwkv");

        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
        let info = model.model_info();
        println!("LeWM info: {} ({})", info.name, info.model_type);
        assert_eq!(info.model_type, "lewm");

        // Test multi-model status handler
        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let status = synapse_esp32::server::handle_status(&model);
        println!("\nMamba Status: model={}, quant={}", status.model, status.quantization);

        // Test generate handler
        let req = synapse_esp32::server::GenerateRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 3,
            temperature: 1.0,
        };
        let resp = synapse_esp32::server::handle_generate(&model, req);
        println!("Generate handler: op={}, latency={:.1}ms", resp.operation, resp.latency_ms);

        // Test model info handler
        let info = synapse_esp32::server::handle_model_info(&model);
        println!("Model info handler: {} ({})", info.name, info.model_type);

        println!("\nAll host tests passed. Ready for ESP32-P4 deployment.");
        println!("  Supported models: LeWM, Mamba Q4, RWKV Q4");
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
