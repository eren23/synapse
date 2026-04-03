idf_component_register(
    SRCS "app_main.c" "wifi.c" "http_server.c" "pie_gemv.c"
         "dual_core.c" "kernels.c" "model_loader.c" "inference.c" "smoke_tests.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_http_server esp_event nvs_flash json esp_timer
)

# Aggressive optimization for performance-critical inference code
target_compile_options(${COMPONENT_TARGET} PRIVATE -O3)


target_add_binary_data(${COMPONENT_TARGET} "model.bin" BINARY)
target_add_binary_data(${COMPONENT_TARGET} "index.html" BINARY)
