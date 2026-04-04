// Re-export inline tests from fused_lewm_layer.zig via synapse module.
const synapse = @import("synapse");
test {
    _ = synapse.ops.fused_lewm_layer;
}
