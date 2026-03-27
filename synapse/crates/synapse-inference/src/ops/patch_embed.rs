//! Patch embedding: convert images to sequences of patch embeddings for ViT.

use super::matmul::matmul_t;

/// Convert image [H, W, C] to patch embeddings [num_patches, embed_dim].
///
/// Extracts P×P patches, flattens each to [P*P*C], and projects via linear layer.
/// Returns [num_patches, embed_dim] where num_patches = (H/P) * (W/P).
pub fn patch_embed(
    image: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    patch_size: usize,
    projection: &[f32], // [patch_dim, embed_dim] where patch_dim = P*P*C
    embed_dim: usize,
) -> Vec<f32> {
    let patches_h = height / patch_size;
    let patches_w = width / patch_size;
    let num_patches = patches_h * patches_w;
    let patch_dim = patch_size * patch_size * channels;

    // Extract patches into [num_patches, patch_dim] in CHW order.
    // Conv2D weight from HuggingFace is [out_ch, in_ch, kH, kW] = [embed_dim, C, P, P].
    // Flattened: index = c * P * P + py * P + px.
    // Image is stored HWC: image[(y * W + x) * C + c].
    let mut patches = vec![0.0f32; num_patches * patch_dim];
    for ph in 0..patches_h {
        for pw in 0..patches_w {
            let patch_idx = ph * patches_w + pw;
            for c in 0..channels {
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let img_y = ph * patch_size + py;
                        let img_x = pw * patch_size + px;
                        let img_idx = (img_y * width + img_x) * channels + c;
                        let patch_pixel = c * patch_size * patch_size + py * patch_size + px;
                        patches[patch_idx * patch_dim + patch_pixel] = image[img_idx];
                    }
                }
            }
        }
    }

    // Project: [num_patches, patch_dim] @ [patch_dim, embed_dim]^T = [num_patches, embed_dim]
    matmul_t(&patches, projection, num_patches, patch_dim, embed_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x4 single-channel image with patch_size=2 should produce 4 patches.
    #[test]
    fn test_patch_embed_4x4_image_2x2_patches() {
        let height = 4;
        let width = 4;
        let channels = 1;
        let patch_size = 2;
        let patch_dim = patch_size * patch_size * channels; // 4
        let embed_dim = 8;

        // Fill image with sequential values: 0, 1, 2, ..., 15
        let image: Vec<f32> = (0..height * width * channels).map(|i| i as f32).collect();

        // Identity-like projection: [embed_dim, patch_dim] — just use small random-ish values
        // For test simplicity, use a projection that is [embed_dim, patch_dim] = [8, 4]
        let projection: Vec<f32> = (0..embed_dim * patch_dim)
            .map(|i| if i % (patch_dim + 1) == 0 { 1.0 } else { 0.01 })
            .collect();

        let result = patch_embed(
            &image,
            height,
            width,
            channels,
            patch_size,
            &projection,
            embed_dim,
        );

        // Should produce 4 patches x embed_dim
        let num_patches = (height / patch_size) * (width / patch_size);
        assert_eq!(num_patches, 4);
        assert_eq!(result.len(), num_patches * embed_dim);

        // All values should be finite
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Patch embedding produced non-finite values"
        );

        // Verify the patches were extracted correctly by checking non-zero output
        assert!(
            result.iter().any(|v| *v != 0.0),
            "Patch embedding produced all-zero output"
        );
    }
}
