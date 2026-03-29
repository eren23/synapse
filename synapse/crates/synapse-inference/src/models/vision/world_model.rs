//! World Model (Latent Dynamics) for planning in latent space.
//!
//! Architecture: ViT visual encoder → latent state projection → dynamics
//! transformer (state + action) → predicted future states.
//! Enables multi-step rollout for model-based planning.

use std::collections::HashMap;

use crate::config::{AttentionConfig, FFNConfig, NormConfig};
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::registry::{create_attention, create_ffn, create_norm, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

use super::vit::{EncoderLayer, ViTConfig, ViTModel};

/// Configuration for a World Model.
#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    /// Visual encoder configuration.
    pub encoder: ViTConfig,
    /// Latent state dimension.
    pub latent_dim: usize,
    /// Dynamics transformer depth.
    pub dynamics_num_layers: usize,
    /// Dynamics transformer attention heads.
    pub dynamics_num_heads: usize,
    /// Dynamics transformer hidden size.
    pub dynamics_hidden_size: usize,
    /// Action space dimension.
    pub action_dim: usize,
}

impl WorldModelConfig {
    /// Dynamics head dimension.
    pub fn dynamics_head_dim(&self) -> usize {
        self.dynamics_hidden_size / self.dynamics_num_heads
    }

    /// Dynamics FFN intermediate size — 4x hidden by default.
    pub fn dynamics_intermediate_size(&self) -> usize {
        self.dynamics_hidden_size * 4
    }
}

/// A latent state vector in the world model's embedding space.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Latent state embedding: `[latent_dim]`.
    pub embedding: Vec<f32>,
}

/// Output from a world model rollout.
pub struct WorldModelOutput {
    /// Predicted future states: `[num_steps]` latent states.
    pub states: Vec<LatentState>,
}

/// World Model: encoder → dynamics predictor → planning in latent space.
pub struct WorldModel {
    pub config: WorldModelConfig,
    /// ViT visual encoder.
    pub encoder: ViTModel,
    /// Project encoder CLS embedding to latent dim: `[encoder_embed_dim, latent_dim]`.
    pub state_proj: AlignedBuffer,
    /// Action embedding matrix: `[action_dim, dynamics_hidden]`.
    pub action_embed: AlignedBuffer,
    /// Dynamics transformer layers (bidirectional over state+action tokens).
    pub dynamics_layers: Vec<EncoderLayer>,
    /// Dynamics final norm.
    pub dynamics_norm: Box<dyn NormVariant>,
    /// Dynamics final norm weight.
    pub dynamics_norm_weight: AlignedBuffer,
    /// Project dynamics output to latent dim: `[dynamics_hidden, latent_dim]`.
    pub output_proj: AlignedBuffer,
}

impl WorldModel {
    /// Build a World Model from config with zeroed weights.
    pub fn from_config(config: &WorldModelConfig) -> Self {
        let norm_config = NormConfig::LayerNorm { eps: 1e-6 };
        let dyn_h = config.dynamics_hidden_size;
        let dyn_head_dim = config.dynamics_head_dim();
        let dyn_inter = config.dynamics_intermediate_size();

        let attn_config = AttentionConfig::Bidirectional {
            num_heads: config.dynamics_num_heads,
            head_dim: dyn_head_dim,
        };
        let ffn_config = FFNConfig::GELU {
            intermediate_size: dyn_inter,
        };

        let mut dynamics_layers = Vec::with_capacity(config.dynamics_num_layers);
        for _ in 0..config.dynamics_num_layers {
            dynamics_layers.push(EncoderLayer {
                attn_norm: create_norm(&norm_config),
                attention: create_attention(&attn_config),
                ffn_norm: create_norm(&norm_config),
                ffn: create_ffn(&ffn_config),
                hidden_size: dyn_h,
                attn_norm_weight: AlignedBuffer::new_zeroed(0),
                w_q: AlignedBuffer::new_zeroed(0),
                w_k: AlignedBuffer::new_zeroed(0),
                w_v: AlignedBuffer::new_zeroed(0),
                w_o: AlignedBuffer::new_zeroed(0),
                ffn_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_up: AlignedBuffer::new_zeroed(0),
                ffn_down: AlignedBuffer::new_zeroed(0),
                q_bias: AlignedBuffer::new_zeroed(0),
                k_bias: AlignedBuffer::new_zeroed(0),
                v_bias: AlignedBuffer::new_zeroed(0),
                o_bias: AlignedBuffer::new_zeroed(0),
                ffn_up_bias: AlignedBuffer::new_zeroed(0),
                ffn_down_bias: AlignedBuffer::new_zeroed(0),
                attn_norm_bias: AlignedBuffer::new_zeroed(0),
                ffn_norm_bias: AlignedBuffer::new_zeroed(0),
            });
        }

        // Vision encoder with no classifier head
        let mut enc_config = config.encoder.clone();
        enc_config.num_classes = 0;
        let encoder = ViTModel::from_config(&enc_config);

        WorldModel {
            config: config.clone(),
            encoder,
            state_proj: AlignedBuffer::new_zeroed(0),
            action_embed: AlignedBuffer::new_zeroed(0),
            dynamics_layers,
            dynamics_norm: create_norm(&norm_config),
            dynamics_norm_weight: AlignedBuffer::new_zeroed(0),
            output_proj: AlignedBuffer::new_zeroed(0),
        }
    }

    /// Load weights into the ViT encoder from source tensors using a name mapper.
    ///
    /// Delegates to `ViTModel::load_weights()` which already handles the full
    /// ViT weight loading pattern. Use `WeightMapper::dinov2()` for DINOv2
    /// or `WeightMapper::vit()` for standard ViT checkpoints.
    pub fn load_encoder_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<crate::models::lm::LoadResult, WeightError> {
        self.encoder.load_weights(weights, mapper)
    }

    /// Encode an observation (image) to a latent state.
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> LatentState {
        let cfg = &self.config;
        let enc_dim = cfg.encoder.hidden_size;
        let lat_dim = cfg.latent_dim;

        // Run ViT forward to get CLS embedding
        let vit_output = self.encoder.forward_image(image, h, w);
        let cls_embed = &vit_output.embeddings; // [enc_dim]

        // Project to latent dimension
        let latent = if !self.state_proj.is_empty() {
            matmul_t(cls_embed, &self.state_proj, 1, enc_dim, lat_dim)
        } else {
            let mut proj = vec![0.0f32; lat_dim];
            let copy_dim = enc_dim.min(lat_dim);
            proj[..copy_dim].copy_from_slice(&cls_embed[..copy_dim]);
            proj
        };

        LatentState { embedding: latent }
    }

    /// Predict the next latent state given current state and action.
    pub fn predict_next(&self, state: &LatentState, action: &[f32]) -> LatentState {
        let cfg = &self.config;
        let dyn_h = cfg.dynamics_hidden_size;
        let lat_dim = cfg.latent_dim;
        let act_dim = cfg.action_dim;

        // 1. Embed the state into dynamics hidden space
        //    State token: project latent_dim → dynamics_hidden
        //    We reuse state_proj transposed conceptually, but use a separate pathway.
        //    For the dynamics input, we zero-pad or truncate the latent embedding.
        let mut state_token = vec![0.0f32; dyn_h];
        let copy_dim = lat_dim.min(dyn_h);
        state_token[..copy_dim].copy_from_slice(&state.embedding[..copy_dim]);

        // 2. Embed the action into dynamics hidden space
        let action_token = if !self.action_embed.is_empty() {
            matmul_t(action, &self.action_embed, 1, act_dim, dyn_h)
        } else {
            let mut tok = vec![0.0f32; dyn_h];
            let copy_a = act_dim.min(dyn_h);
            tok[..copy_a].copy_from_slice(&action[..copy_a]);
            tok
        };

        // 3. Build dynamics input: [state_token, action_token] → seq_len=2
        let seq_len = 2;
        let mut x = vec![0.0f32; seq_len * dyn_h];
        x[..dyn_h].copy_from_slice(&state_token);
        x[dyn_h..2 * dyn_h].copy_from_slice(&action_token);

        // 4. Run through dynamics transformer layers
        for layer in &self.dynamics_layers {
            x = layer.forward(&x, seq_len);
        }

        // 5. Apply final dynamics norm
        let normed = apply_norm(&x, &self.dynamics_norm_weight, &*self.dynamics_norm, dyn_h);

        // 6. Take the state token (position 0) and project to latent dim
        let state_out = &normed[..dyn_h];
        let next_latent = if !self.output_proj.is_empty() {
            matmul_t(state_out, &self.output_proj, 1, dyn_h, lat_dim)
        } else {
            let mut proj = vec![0.0f32; lat_dim];
            let copy_out = dyn_h.min(lat_dim);
            proj[..copy_out].copy_from_slice(&state_out[..copy_out]);
            proj
        };

        LatentState {
            embedding: next_latent,
        }
    }

    /// Multi-step rollout: predict a sequence of future states from actions.
    pub fn rollout(&self, initial: &LatentState, actions: &[Vec<f32>]) -> WorldModelOutput {
        let mut states = Vec::with_capacity(actions.len());
        let mut current = initial.clone();
        for action in actions {
            current = self.predict_next(&current, action);
            states.push(current.clone());
        }
        WorldModelOutput { states }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    fn test_world_model_config() -> WorldModelConfig {
        WorldModelConfig {
            encoder: ViTConfig {
                image_size: 8,
                patch_size: 4,
                channels: 3,
                hidden_size: 32,
                num_layers: 2,
                num_heads: 4,
                intermediate_size: 64,
                num_classes: 0,
            },
            latent_dim: 16,
            dynamics_num_layers: 2,
            dynamics_num_heads: 4,
            dynamics_hidden_size: 32,
            action_dim: 4,
        }
    }

    fn build_test_world_model(cfg: &WorldModelConfig) -> WorldModel {
        let enc_h = cfg.encoder.hidden_size;
        let enc_inter = cfg.encoder.intermediate_size;
        let dyn_h = cfg.dynamics_hidden_size;
        let dyn_inter = cfg.dynamics_intermediate_size();
        let lat_dim = cfg.latent_dim;
        let act_dim = cfg.action_dim;
        let patch_dim = cfg.encoder.patch_size * cfg.encoder.patch_size * cfg.encoder.channels;
        let enc_seq_len = cfg.encoder.seq_len();

        let mut model = WorldModel::from_config(cfg);

        // Encoder weights
        model.encoder.patch_proj = AlignedBuffer::from_slice(&gen_weights(enc_h * patch_dim, 1));
        model.encoder.cls_token = AlignedBuffer::from_slice(&gen_weights(enc_h, 2));
        model.encoder.pos_embed = AlignedBuffer::from_slice(&gen_weights(enc_seq_len * enc_h, 3));
        model.encoder.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);

        for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_inter, s + 6));
        }

        // State projection: [enc_dim, latent_dim]
        model.state_proj = AlignedBuffer::from_slice(&gen_weights(enc_h * lat_dim, 500));

        // Action embedding: [action_dim, dyn_hidden]
        model.action_embed = AlignedBuffer::from_slice(&gen_weights(act_dim * dyn_h, 501));

        // Output projection: [dyn_hidden, latent_dim]
        model.output_proj = AlignedBuffer::from_slice(&gen_weights(dyn_h * lat_dim, 502));

        // Dynamics norm
        model.dynamics_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; dyn_h]);

        // Dynamics layers
        for (i, layer) in model.dynamics_layers.iter_mut().enumerate() {
            let s = (i as u32 + 10) * 200;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; dyn_h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(dyn_h * dyn_h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(dyn_h * dyn_h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(dyn_h * dyn_h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(dyn_h * dyn_h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; dyn_h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(dyn_inter * dyn_h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(dyn_h * dyn_inter, s + 6));
        }

        model
    }

    #[test]
    fn test_world_model_encode_produces_latent() {
        let cfg = test_world_model_config();
        let model = build_test_world_model(&cfg);

        let image: Vec<f32> =
            (0..cfg.encoder.image_size * cfg.encoder.image_size * cfg.encoder.channels)
                .map(|i| (i as f32) / 255.0)
                .collect();

        let state = model.encode(&image, cfg.encoder.image_size, cfg.encoder.image_size);

        assert_eq!(
            state.embedding.len(),
            cfg.latent_dim,
            "Latent state should have latent_dim elements"
        );
        assert!(
            state.embedding.iter().all(|v| v.is_finite()),
            "World model encode produced non-finite latent state"
        );
    }

    #[test]
    fn test_world_model_rollout_correct_length() {
        let cfg = test_world_model_config();
        let model = build_test_world_model(&cfg);

        let image: Vec<f32> =
            (0..cfg.encoder.image_size * cfg.encoder.image_size * cfg.encoder.channels)
                .map(|i| (i as f32) / 255.0)
                .collect();

        let initial = model.encode(&image, cfg.encoder.image_size, cfg.encoder.image_size);

        // 5 actions → should produce 5 predicted states
        let actions: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                (0..cfg.action_dim)
                    .map(|j| ((i * cfg.action_dim + j) as f32) * 0.1)
                    .collect()
            })
            .collect();

        let output = model.rollout(&initial, &actions);

        assert_eq!(
            output.states.len(),
            5,
            "Rollout with 5 actions should produce 5 states"
        );

        // All states should have correct dimension and be finite
        for (i, state) in output.states.iter().enumerate() {
            assert_eq!(
                state.embedding.len(),
                cfg.latent_dim,
                "State {i} should have latent_dim elements"
            );
            assert!(
                state.embedding.iter().all(|v| v.is_finite()),
                "State {i} contains non-finite values"
            );
        }
    }

    #[test]
    fn test_world_model_load_encoder_weights() {
        use crate::weight_loading::{RawTensor, WeightMapper};

        let cfg = test_world_model_config();
        let mut model = WorldModel::from_config(&cfg);
        let enc_h = cfg.encoder.hidden_size;
        let enc_inter = cfg.encoder.intermediate_size;
        let patch_dim = cfg.encoder.patch_size * cfg.encoder.patch_size * cfg.encoder.channels;
        let enc_seq_len = cfg.encoder.seq_len();

        // Build fake weight dict with ViT naming
        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let rt = |len: usize, seed: u32| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(len, seed)),
                shape: vec![len],
            }
        };
        let ones = |len: usize| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&vec![1.0f32; len]),
                shape: vec![len],
            }
        };

        weights.insert(
            "vit.embeddings.patch_embeddings.projection.weight".into(),
            rt(enc_h * patch_dim, 1),
        );
        weights.insert(
            "vit.embeddings.patch_embeddings.projection.bias".into(),
            rt(enc_h, 2),
        );
        weights.insert("vit.embeddings.cls_token".into(), rt(enc_h, 3));
        weights.insert(
            "vit.embeddings.position_embeddings".into(),
            rt(enc_seq_len * enc_h, 4),
        );
        weights.insert("vit.layernorm.weight".into(), ones(enc_h));
        weights.insert("vit.layernorm.bias".into(), rt(enc_h, 6));

        for i in 0..cfg.encoder.num_layers {
            let s = (i as u32 + 1) * 100;
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.query.weight"),
                rt(enc_h * enc_h, s + 1),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.query.bias"),
                rt(enc_h, s + 2),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.key.weight"),
                rt(enc_h * enc_h, s + 3),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.key.bias"),
                rt(enc_h, s + 4),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.value.weight"),
                rt(enc_h * enc_h, s + 5),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.attention.value.bias"),
                rt(enc_h, s + 6),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.output.dense.weight"),
                rt(enc_h * enc_h, s + 7),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.attention.output.dense.bias"),
                rt(enc_h, s + 8),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.intermediate.dense.weight"),
                rt(enc_inter * enc_h, s + 9),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.intermediate.dense.bias"),
                rt(enc_inter, s + 10),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.output.dense.weight"),
                rt(enc_h * enc_inter, s + 11),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.output.dense.bias"),
                rt(enc_h, s + 12),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.layernorm_before.weight"),
                ones(enc_h),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.layernorm_before.bias"),
                rt(enc_h, s + 14),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.layernorm_after.weight"),
                ones(enc_h),
            );
            weights.insert(
                format!("vit.encoder.layer.{i}.layernorm_after.bias"),
                rt(enc_h, s + 16),
            );
        }

        let mapper = WeightMapper::vit();
        let result = model
            .load_encoder_weights(weights, &mapper)
            .expect("load failed");

        // Verify encoder weights loaded
        assert!(
            !model.encoder.patch_proj.is_empty(),
            "encoder patch_proj should be loaded"
        );
        assert!(
            !model.encoder.cls_token.is_empty(),
            "encoder cls_token should be loaded"
        );
        assert!(
            !model.encoder.pos_embed.is_empty(),
            "encoder pos_embed should be loaded"
        );
        assert!(
            !model.encoder.final_norm_weight.is_empty(),
            "encoder final_norm should be loaded"
        );
        assert!(
            !model.encoder.layers[0].w_q.is_empty(),
            "encoder layer 0 w_q should be loaded"
        );

        assert!(
            result.unexpected.is_empty(),
            "Should have no unmapped keys, got: {:?}",
            result.unexpected
        );
    }
}

// ── Real-time Rollout API ────────────────────────────────────────────

/// Real-time world model rollout for robotics and planning.
///
/// Maintains a latent state that evolves through action-conditioned
/// dynamics predictions. Designed for <10ms per step latency.
///
/// ```ignore
/// let mut rollout = RealtimeRollout::new(world_model);
/// rollout.reset(&image, 224, 224);  // encode initial observation
///
/// loop {
///     let action = controller.get_action(&rollout.state());
///     let next = rollout.step(&action);
///     // next.embedding is the predicted latent state
/// }
/// ```
pub struct RealtimeRollout {
    model: WorldModel,
    current_state: LatentState,
    step_count: usize,
}

impl RealtimeRollout {
    /// Create a new rollout session from a world model.
    pub fn new(model: WorldModel) -> Self {
        let latent_dim = model.config.latent_dim;
        Self {
            model,
            current_state: LatentState {
                embedding: vec![0.0f32; latent_dim],
            },
            step_count: 0,
        }
    }

    /// Reset the rollout by encoding a new observation (image).
    pub fn reset(&mut self, image: &[f32], h: usize, w: usize) {
        self.current_state = self.model.encode(image, h, w);
        self.step_count = 0;
    }

    /// Reset with a pre-encoded latent state.
    pub fn reset_with_state(&mut self, state: LatentState) {
        self.current_state = state;
        self.step_count = 0;
    }

    /// Single dynamics step: predict next state given current state + action.
    /// Returns the new latent state. This is the hot path — target <10ms.
    pub fn step(&mut self, action: &[f32]) -> &LatentState {
        self.current_state = self.model.predict_next(&self.current_state, action);
        self.step_count += 1;
        &self.current_state
    }

    /// Plan ahead: rollout N steps with given actions without modifying current state.
    /// Returns predicted trajectory without advancing the internal state.
    pub fn plan(&self, actions: &[Vec<f32>]) -> Vec<LatentState> {
        let output = self.model.rollout(&self.current_state, actions);
        output.states
    }

    /// Current latent state.
    pub fn state(&self) -> &LatentState {
        &self.current_state
    }

    /// Number of dynamics steps taken since last reset.
    pub fn steps(&self) -> usize {
        self.step_count
    }

    /// Access the underlying world model.
    pub fn model(&self) -> &WorldModel {
        &self.model
    }
}
