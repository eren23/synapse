//! Reference correctness tests for Synapse SSM kernels.
//!
//! Each test uses hand-computed expected values with tight tolerances to
//! verify mathematical correctness of the kernel implementations.

use synapse_inference::ssm::{
    deltanet_step, selective_scan_seq, wkv_step, MambaBlock, MambaConfig, MambaModel,
};
use synapse_inference::quantization::TernaryLinear;
use synapse_inference::generation::{GenerationConfig, GenerationPipeline};
use synapse_inference::model::ModelState;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Selective Scan Reference Values
// ─────────────────────────────────────────────────────────────────────────────
//
// The selective scan recurrence (one step, channel i, state j):
//   h[i,j] = exp(delta[i] * A[i,j]) * h[i,j] + delta[i] * B[j] * x[i]
//   y[i]   = sum_j( C[j] * h[i,j] ) + D[i] * x[i]
//
// Test parameters: d_inner=2, d_state=2, seq_len=2, all A=-1, D=0.
// exp(-0.5) ≈ 0.60653.
//
// ── Step 1: x=[1.0,2.0], delta=[0.5,0.5], B=[1.0,0.5], C=[1.0,1.0] ──────────
//   h[0,0] = exp(-0.5)*0 + 0.5*1.0*1.0 = 0.5
//   h[0,1] = exp(-0.5)*0 + 0.5*0.5*1.0 = 0.25
//   h[1,0] = exp(-0.5)*0 + 0.5*1.0*2.0 = 1.0
//   h[1,1] = exp(-0.5)*0 + 0.5*0.5*2.0 = 0.5
//   y[0] = 1.0*0.5 + 1.0*0.25 = 0.75
//   y[1] = 1.0*1.0 + 1.0*0.5  = 1.5
//
// ── Step 2: x=[0.5,1.0], delta=[0.5,0.5], B=[0.5,1.0], C=[1.0,0.5] ──────────
//   h[0,0] = 0.60653*0.5  + 0.5*0.5*0.5   = 0.30327 + 0.125   = 0.42827
//   h[0,1] = 0.60653*0.25 + 0.5*1.0*0.5   = 0.15163 + 0.25    = 0.40163
//   h[1,0] = 0.60653*1.0  + 0.5*0.5*1.0   = 0.60653 + 0.25    = 0.85653
//   h[1,1] = 0.60653*0.5  + 0.5*1.0*1.0   = 0.30327 + 0.5     = 0.80327
//   y[2] = 1.0*0.42827 + 0.5*0.40163 = 0.42827 + 0.20082 = 0.62909
//   y[3] = 1.0*0.85653 + 0.5*0.80327 = 0.85653 + 0.40163 = 1.25816

#[test]
fn test_selective_scan_reference_values() {
    let d_inner = 2;
    let d_state = 2;

    // x: [step1: [1.0, 2.0], step2: [0.5, 1.0]] — shape [seq_len * d_inner]
    let x     = vec![1.0f32, 2.0, 0.5, 1.0];
    let delta = vec![0.5f32, 0.5, 0.5, 0.5];
    // a_log[i * d_state + j] — all -1.0 → exp(-0.5) ≈ 0.60653
    let a_log = vec![-1.0f32, -1.0, -1.0, -1.0]; // [d_inner * d_state]
    // B per step: [step1: [1.0, 0.5], step2: [0.5, 1.0]]
    let b = vec![1.0f32, 0.5, 0.5, 1.0];
    // C per step: [step1: [1.0, 1.0], step2: [1.0, 0.5]]
    let c = vec![1.0f32, 1.0, 1.0, 0.5];
    // D (skip connection) = zero so it doesn't interfere
    let d = vec![0.0f32, 0.0];

    let mut state = vec![0.0f32; d_inner * d_state];
    let y = selective_scan_seq(&x, &delta, &a_log, &b, &c, &d, &mut state);

    assert_eq!(y.len(), 4, "output should have seq_len * d_inner = 4 elements");

    // Step 1 outputs — tight tolerance, computed exactly from initial zero state.
    assert!(
        (y[0] - 0.75).abs() < 1e-5,
        "y[0] = {} expected 0.75",
        y[0]
    );
    assert!(
        (y[1] - 1.5).abs() < 1e-5,
        "y[1] = {} expected 1.5",
        y[1]
    );

    // Step 2 outputs — exp(-0.5) introduces minor f32 rounding; use 1e-4 tolerance.
    // Expected: y[2] = 0.42827 + 0.20082 = 0.62909
    //           y[3] = 0.85653 + 0.40163 = 1.25816
    assert!(
        (y[2] - 0.62909).abs() < 1e-4,
        "y[2] = {} expected ≈ 0.62909",
        y[2]
    );
    assert!(
        (y[3] - 1.25816).abs() < 1e-4,
        "y[3] = {} expected ≈ 1.25816",
        y[3]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: WKV Reference Values
// ─────────────────────────────────────────────────────────────────────────────
//
// The WKV kernel for one head (head_size=2):
//   output[d] = sum_j( state[d,j] * r[j] )   ← output computed FIRST (pre-update)
//   state[d,j] = exp(w[d]) * state[d,j] + k[d] * v[j]  ← then state updated
//
// ── Step 1: r=[1.0,0.0], k=[1.0,0.5], v=[2.0,1.0], w=[-0.5,-0.5] ────────────
//   Initial state = zeros → output = [0, 0]
//   State update (exp(-0.5)≈0.60653):
//     state[0,0] = 0.60653*0 + 1.0*2.0 = 2.0
//     state[0,1] = 0.60653*0 + 1.0*1.0 = 1.0
//     state[1,0] = 0.60653*0 + 0.5*2.0 = 1.0
//     state[1,1] = 0.60653*0 + 0.5*1.0 = 0.5
//
// ── Step 2: r=[0.5,0.5], k=[1.0,1.0], v=[1.0,1.0], w=[-0.5,-0.5] ─────────────
//   Output (from state after step 1):
//     o[0] = state[0,0]*r[0] + state[0,1]*r[1] = 2.0*0.5 + 1.0*0.5 = 1.5
//     o[1] = state[1,0]*r[0] + state[1,1]*r[1] = 1.0*0.5 + 0.5*0.5 = 0.75
//   State update (exp(-0.5)≈0.60653):
//     state[0,0] = 0.60653*2.0 + 1.0*1.0 = 1.21306 + 1.0 = 2.21306
//     state[0,1] = 0.60653*1.0 + 1.0*1.0 = 0.60653 + 1.0 = 1.60653
//     state[1,0] = 0.60653*1.0 + 1.0*1.0 = 0.60653 + 1.0 = 1.60653
//     state[1,1] = 0.60653*0.5 + 1.0*1.0 = 0.30327 + 1.0 = 1.30327

#[test]
fn test_wkv_step_reference_values() {
    let head_size = 2;
    let mut wkv_state = vec![0.0f32; head_size * head_size];

    // ── Step 1 ────────────────────────────────────────────────────────────────
    let r1 = vec![1.0f32, 0.0];
    let k1 = vec![1.0f32, 0.5];
    let v1 = vec![2.0f32, 1.0];
    let w  = vec![-0.5f32, -0.5]; // decay in log-space, negative

    let out1 = wkv_step(&r1, &k1, &v1, &w, &mut wkv_state, head_size);

    // Output is read BEFORE the state update, so from zero state → both zero.
    assert_eq!(out1.len(), head_size);
    assert!(
        out1[0].abs() < 1e-7,
        "step1 out[0] = {} expected 0.0 (zero initial state)",
        out1[0]
    );
    assert!(
        out1[1].abs() < 1e-7,
        "step1 out[1] = {} expected 0.0 (zero initial state)",
        out1[1]
    );

    // After step 1, state = k ⊗ v (no decay from zero state):
    //   state[0,0]=2.0  state[0,1]=1.0  state[1,0]=1.0  state[1,1]=0.5
    assert!(
        (wkv_state[0] - 2.0).abs() < 1e-6,
        "state[0,0] = {} expected 2.0",
        wkv_state[0]
    );
    assert!(
        (wkv_state[1] - 1.0).abs() < 1e-6,
        "state[0,1] = {} expected 1.0",
        wkv_state[1]
    );
    assert!(
        (wkv_state[2] - 1.0).abs() < 1e-6,
        "state[1,0] = {} expected 1.0",
        wkv_state[2]
    );
    assert!(
        (wkv_state[3] - 0.5).abs() < 1e-6,
        "state[1,1] = {} expected 0.5",
        wkv_state[3]
    );

    // ── Step 2 ────────────────────────────────────────────────────────────────
    let r2 = vec![0.5f32, 0.5];
    let k2 = vec![1.0f32, 1.0];
    let v2 = vec![1.0f32, 1.0];

    let out2 = wkv_step(&r2, &k2, &v2, &w, &mut wkv_state, head_size);

    assert_eq!(out2.len(), head_size);
    // o[0] = 2.0*0.5 + 1.0*0.5 = 1.5
    assert!(
        (out2[0] - 1.5).abs() < 1e-5,
        "step2 out[0] = {} expected 1.5",
        out2[0]
    );
    // o[1] = 1.0*0.5 + 0.5*0.5 = 0.75
    assert!(
        (out2[1] - 0.75).abs() < 1e-5,
        "step2 out[1] = {} expected 0.75",
        out2[1]
    );

    // Verify state after step 2 (exp(-0.5) ≈ 0.60653):
    //   state[0,0] = 0.60653*2.0 + 1.0 ≈ 2.21306
    //   state[0,1] = 0.60653*1.0 + 1.0 ≈ 1.60653
    let exp_neg_half = (-0.5f32).exp();
    let expected_s00 = exp_neg_half * 2.0 + 1.0;
    let expected_s01 = exp_neg_half * 1.0 + 1.0;
    assert!(
        (wkv_state[0] - expected_s00).abs() < 1e-5,
        "state[0,0] = {} expected {}",
        wkv_state[0],
        expected_s00
    );
    assert!(
        (wkv_state[1] - expected_s01).abs() < 1e-5,
        "state[0,1] = {} expected {}",
        wkv_state[1],
        expected_s01
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: DeltaNet Reference Values
// ─────────────────────────────────────────────────────────────────────────────
//
// The Gated DeltaNet recurrence (one head, head_dim=2):
//   S_t = alpha * S_{t-1} + beta * outer(v, k)   ← state updated FIRST
//   o_t = S_t @ q                                  ← output reads updated state
//
// ── Step 1: q=[1.0,0.0], k=[1.0,0.5], v=[2.0,1.0], alpha=0.9, beta=0.5 ──────
//   S = 0.9*[[0,0],[0,0]] + 0.5 * outer([2,1],[1,0.5])
//     = 0.5 * [[2*1, 2*0.5], [1*1, 1*0.5]]
//     = [[1.0, 0.5], [0.5, 0.25]]
//   o = S @ q = [[1.0,0.5],[0.5,0.25]] @ [1.0,0.0]
//     = [1.0*1+0.5*0, 0.5*1+0.25*0]
//     = [1.0, 0.5]
//
// NOTE: deltanet_step updates state BEFORE computing output, so even with zero
// initial state the first output is non-zero (it reflects the new outer product).

#[test]
fn test_deltanet_step_reference_values() {
    let head_dim = 2;
    let mut memory = vec![0.0f32; head_dim * head_dim];

    let q = vec![1.0f32, 0.0];
    let k = vec![1.0f32, 0.5];
    let v = vec![2.0f32, 1.0];
    let alpha = 0.9f32;
    let beta  = 0.5f32;

    let out = deltanet_step(&q, &k, &v, alpha, beta, &mut memory, head_dim);

    assert_eq!(out.len(), head_dim);

    // Expected:
    //   memory after update = [[1.0, 0.5], [0.5, 0.25]] (flat: [1.0, 0.5, 0.5, 0.25])
    //   o[0] = 1.0*1.0 + 0.5*0.0 = 1.0
    //   o[1] = 0.5*1.0 + 0.25*0.0 = 0.5
    assert!(
        (out[0] - 1.0).abs() < 1e-6,
        "deltanet out[0] = {} expected 1.0",
        out[0]
    );
    assert!(
        (out[1] - 0.5).abs() < 1e-6,
        "deltanet out[1] = {} expected 0.5",
        out[1]
    );

    // Verify state matrix (row-major: memory[d*head_dim + j])
    // memory[0,0] = beta * v[0] * k[0] = 0.5 * 2.0 * 1.0 = 1.0
    // memory[0,1] = beta * v[0] * k[1] = 0.5 * 2.0 * 0.5 = 0.5
    // memory[1,0] = beta * v[1] * k[0] = 0.5 * 1.0 * 1.0 = 0.5
    // memory[1,1] = beta * v[1] * k[1] = 0.5 * 1.0 * 0.5 = 0.25
    assert!(
        (memory[0] - 1.0).abs() < 1e-6,
        "memory[0,0] = {} expected 1.0",
        memory[0]
    );
    assert!(
        (memory[1] - 0.5).abs() < 1e-6,
        "memory[0,1] = {} expected 0.5",
        memory[1]
    );
    assert!(
        (memory[2] - 0.5).abs() < 1e-6,
        "memory[1,0] = {} expected 0.5",
        memory[2]
    );
    assert!(
        (memory[3] - 0.25).abs() < 1e-6,
        "memory[1,1] = {} expected 0.25",
        memory[3]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Ternary GEMV Reference Values
// ─────────────────────────────────────────────────────────────────────────────
//
// Weights (2x3 matrix, stored row-major):
//   Row 0: [1.0, -1.0, 0.1]
//   Row 1: [-0.8, 0.9, -0.1]
// x = [1.0, 2.0, 3.0]
//
// Ternarization (per-row, threshold = 0.5 * mean_abs):
//   Row 0: mean_abs = (1.0+1.0+0.1)/3 = 0.7, threshold = 0.35
//     1.0  > 0.35 → +1  (nonzero_sum += 1.0)
//    -1.0 < -0.35 → -1  (nonzero_sum += 1.0)
//     0.1: |0.1| < 0.35 → 0
//     nonzero_count=2, scale = 2.0/2 = 1.0
//     output[0] = 1.0 * (+1*1.0 + -1*2.0 + 0*3.0) = 1.0*(1-2) = -1.0
//
//   Row 1: mean_abs = (0.8+0.9+0.1)/3 = 0.6, threshold = 0.3
//    -0.8 < -0.3 → -1  (nonzero_sum += 0.8)
//     0.9  > 0.3 → +1  (nonzero_sum += 0.9)
//    -0.1: |-0.1| < 0.3 → 0
//     nonzero_count=2, scale = 1.7/2 = 0.85
//     output[1] = 0.85 * (-1*1.0 + 1*2.0 + 0*3.0) = 0.85*(2-1) = 0.85

#[test]
fn test_ternary_gemv_reference_values() {
    let out_features = 2;
    let in_features  = 3;

    // Row 0: [1.0, -1.0, 0.1], Row 1: [-0.8, 0.9, -0.1]
    let weights = vec![1.0f32, -1.0, 0.1, -0.8, 0.9, -0.1];
    let x = vec![1.0f32, 2.0, 3.0];

    let layer = TernaryLinear::from_f32(&weights, out_features, in_features);
    // m=1 (single row of input)
    let output = layer.forward(&x, 1);

    assert_eq!(output.len(), out_features, "output should have 2 elements");

    // Row 0: ternary = [+1, -1, 0], scale = 1.0 → output[0] = 1.0*(-1.0) = -1.0
    assert!(
        (output[0] - (-1.0)).abs() < 1e-5,
        "output[0] = {} expected -1.0",
        output[0]
    );

    // Row 1: ternary = [-1, +1, 0], scale = 0.85 → output[1] = 0.85*1.0 = 0.85
    assert!(
        (output[1] - 0.85).abs() < 1e-5,
        "output[1] = {} expected 0.85",
        output[1]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: MambaModel generation is deterministic
// ─────────────────────────────────────────────────────────────────────────────
//
// Builds a tiny MambaModel with known pseudo-random weights, runs prefill
// twice with an identical prompt, and verifies the logits are bit-identical.
// This acts as a regression guard for the full generation pipeline.

fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
            (f32::from_bits(bits) - 1.5) * 0.2
        })
        .collect()
}

fn build_tiny_mamba() -> MambaModel {
    let config = MambaConfig::tiny_test();
    let d_model = config.d_model;
    let d_inner = config.d_inner();
    let d_state = config.d_state;
    let d_conv  = config.d_conv;
    let vocab   = config.vocab_size;

    let embed_tokens      = pseudo_random_vec(100, vocab * d_model);
    let final_norm_weight = vec![1.0f32; d_model];
    let lm_head_weight    = pseudo_random_vec(200, vocab * d_model);

    let mut blocks = Vec::new();
    for layer_idx in 0..config.num_layers {
        let s = (layer_idx as u64 + 1) * 1000;
        blocks.push(MambaBlock {
            d_model,
            d_inner,
            d_state,
            d_conv,
            norm_weight:    vec![1.0f32; d_model],
            norm_eps:        config.norm_eps as f32,
            in_proj_weight:  pseudo_random_vec(s + 1, 2 * d_inner * d_model),
            in_proj_bias:    vec![],
            conv1d_weight:   pseudo_random_vec(s + 2, d_inner * d_conv),
            conv1d_bias:     vec![0.0f32; d_inner],
            x_proj_weight:   pseudo_random_vec(s + 3, (2 * d_state + 1) * d_inner),
            dt_proj_weight:  pseudo_random_vec(s + 4, d_inner),
            dt_proj_bias:    vec![0.0f32; d_inner],
            a_log: pseudo_random_vec(s + 5, d_inner * d_state)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            d_param:        vec![1.0f32; d_inner],
            out_proj_weight: pseudo_random_vec(s + 6, d_model * d_inner),
            out_proj_bias:   vec![],
        });
    }

    MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
}

#[test]
fn test_mamba_generation_deterministic() {
    let model  = build_tiny_mamba();
    let prompt = [1u32, 2, 3, 4];

    // Run 1
    model.reset_state();
    let out1 = model.prefill(&prompt);

    // Run 2 with identical state
    model.reset_state();
    let out2 = model.prefill(&prompt);

    assert_eq!(
        out1.logits.len(),
        out2.logits.len(),
        "both runs should produce the same number of logits"
    );

    for (i, (&a, &b)) in out1.logits.iter().zip(out2.logits.iter()).enumerate() {
        assert_eq!(
            a, b,
            "logit[{i}] is not bit-identical across two runs: {a} vs {b}"
        );
    }

    // Sanity: logits should be finite and non-zero
    let any_nonzero = out1.logits.iter().any(|&v| v != 0.0);
    assert!(any_nonzero, "logits should not all be zero");

    for (i, &v) in out1.logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: GenerationPipeline end-to-end with MambaModel
// ─────────────────────────────────────────────────────────────────────────────
//
// Builds a tiny MambaModel, wraps it in a GenerationPipeline, and runs greedy
// generation with a Recurrent state. Verifies:
//   - output.token_ids contains prompt + generated tokens
//   - All generated tokens are valid (< vocab_size)
//   - Output is deterministic (run twice, same result)
//   - num_generated_tokens > 0

#[test]
fn test_mamba_generation_pipeline_end_to_end() {
    let model = build_tiny_mamba();
    let vocab_size = model.config.vocab_size;
    let prompt = vec![1u32, 2, 3, 4];
    let max_new = 5usize;

    let pipeline = GenerationPipeline::new(&model);

    // ── Run 1 ─────────────────────────────────────────────────────────────────
    model.reset_state();
    let mut state1 = ModelState::Recurrent;
    let config1 = GenerationConfig {
        max_new_tokens: max_new,
        seed: Some(42),
        ..Default::default()
    };
    let output1 = pipeline.generate(&prompt, config1, Some(&mut state1));

    // token_ids must be prompt + generated
    assert_eq!(
        output1.token_ids.len(),
        prompt.len() + output1.num_generated_tokens,
        "token_ids length must equal prompt length + num_generated_tokens"
    );

    // Prompt tokens are preserved at the front
    assert_eq!(
        &output1.token_ids[..prompt.len()],
        prompt.as_slice(),
        "first tokens in output must be the prompt"
    );

    // All generated tokens must be valid vocab IDs
    for (i, &tok) in output1.token_ids[prompt.len()..].iter().enumerate() {
        assert!(
            (tok as usize) < vocab_size,
            "generated token[{i}] = {tok} is out of vocab range (vocab_size={vocab_size})"
        );
    }

    // At least one token was generated
    assert!(
        output1.num_generated_tokens > 0,
        "num_generated_tokens must be > 0"
    );

    // ── Run 2 (determinism check) ─────────────────────────────────────────────
    model.reset_state();
    let mut state2 = ModelState::Recurrent;
    let config2 = GenerationConfig {
        max_new_tokens: max_new,
        seed: Some(42),
        ..Default::default()
    };
    let output2 = pipeline.generate(&prompt, config2, Some(&mut state2));

    assert_eq!(
        output1.token_ids, output2.token_ids,
        "greedy generation must be deterministic across two runs"
    );
    assert_eq!(
        output1.num_generated_tokens, output2.num_generated_tokens,
        "num_generated_tokens must match across two runs"
    );
}
