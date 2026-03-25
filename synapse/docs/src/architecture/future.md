# Expansion Roadmap

This page outlines the architecture extensions needed to support model families beyond autoregressive causal LMs.

## Encoder-Decoder Models (T5, BART)

Encoder-decoder architectures process the input through a bidirectional encoder, then decode autoregressively with cross-attention to encoder outputs.

**What's reusable:**
- Decoder layer structure (self-attention + FFN) is identical to the current `DecoderLayer`
- RMSNorm, SiLU/SwiGLU, and GEMV kernels apply unchanged
- KV cache for decoder self-attention works as-is
- Weight loading (safetensors/GGUF) and quantization pipelines

**What's new:**
- Encoder forward pass: full-sequence bidirectional self-attention (no causal mask)
- Cross-attention sublayer in each decoder layer: Q from decoder, K/V from encoder output
- Encoder KV cache: computed once during "prefill" and reused across all decode steps
- Metal shaders: `attention` kernel needs a mode flag to disable the causal mask

**Estimated effort:** Medium. The decoder path requires one additional cross-attention dispatch per layer. The encoder is a simplified decoder (no causal mask, no KV cache updates).

## Diffusion Models (Stable Diffusion, Flux)

Diffusion models iteratively denoise a latent tensor through a UNet (or transformer) conditioned on text embeddings from CLIP.

**What's reusable:**
- GEMV/matmul kernels for linear projections within UNet blocks
- Metal command buffer pipeline and buffer pooling
- Existing stubs at `crates/synapse-inference/src/diffusion/` (pipeline, UNet, scheduler)

**What's new:**
- UNet architecture: ResNet blocks, spatial self-attention, cross-attention to text conditioning
- VAE encoder/decoder for latent-space conversion
- CLIP text encoder (transformer encoder, no causal mask)
- Noise schedulers (DDPM, DDIM, Euler) -- stubs exist in `diffusion/scheduler.rs`
- 2D spatial operations: conv2d, group normalization, upsampling/downsampling

**Estimated effort:** Large. UNet requires conv2d Metal kernels (not yet implemented), and the iterative denoising loop (20-50 steps) demands careful memory management for intermediate feature maps.

## Vision Transformers (ViT, CLIP)

Vision transformers process images as sequences of patch embeddings through a standard transformer encoder.

**What's reusable:**
- Self-attention and FFN sublayers (identical to decoder layers without causal mask)
- RMSNorm/LayerNorm, GELU activations
- Metal matmul and attention kernels

**What's new:**
- Patch embedding layer: split image into patches, linear projection + position embedding
- No causal mask: global bidirectional attention (simpler than the current causal kernel)
- Class token (`[CLS]`) handling for classification tasks
- Image preprocessing: resize, normalize, patch extraction

**Estimated effort:** Small. A ViT is structurally simpler than the current causal LM -- it's a subset of the encoder-decoder encoder path.

## Mixture of Experts (Mixtral, DeepSeek-MoE)

MoE replaces the dense FFN in each decoder layer with a set of expert FFNs and a learned router that selects the top-k experts per token.

**What's reusable:**
- Attention sublayer is unchanged (GQA, RoPE, KV cache)
- Individual expert FFNs use the same SwiGLU + Linear structure as `DecoderLayer.ffn`
- Weight loading, quantization, and Metal infrastructure

**What's new:**
- Router network: small linear layer that scores all experts per token, selects top-k (typically k=2)
- Sparse dispatch: route each token to its selected experts, combine outputs with router weights
- Memory management: 8-64 expert weight sets per layer (large parameter count but sparse activation)
- Load balancing loss (training only): auxiliary loss to prevent expert collapse

**Estimated effort:** Medium. The core change is in `DecoderLayer::forward` to add router + sparse FFN dispatch. Metal kernels need a batched/scatter GEMV for expert routing.

## Multi-Modal Models (LLaVA, Qwen-VL)

Multi-modal models combine a vision encoder (ViT/CLIP) with an LLM decoder via cross-modal projection.

**What's reusable:**
- LLM decoder path is the existing `CausalLM` pipeline
- All Metal kernels, KV cache, and generation pipeline
- ViT encoder (once implemented, see above)

**What's new:**
- Vision encoder integration: CLIP or SigLIP image encoder
- Cross-modal projection: linear or MLP layer mapping vision features to LLM embedding space
- Mixed-modality input sequence: interleaved image tokens and text tokens
- Image preprocessing and token injection into the embedding sequence

**Estimated effort:** Medium. Most of the work is in the vision encoder (shared with ViT support) and the projection layer. The LLM side requires minimal changes -- image features are injected as additional embeddings before the first decoder layer.
