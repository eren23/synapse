//! SIMD vs naive throughput benchmark: measures speedup from wiring Zig SIMD kernels.
//!
//! Thresholds (release mode):
//! - SGEMM [1024,1024]×[1024,3072]: SIMD >= 4× naive
//! - RMSNorm [1,1024]:               SIMD >= 4× naive
//! - SwiGLU [1,3072]:                SIMD >= 2× naive
//! - Decoder layer end-to-end:        SIMD >= 2× naive
//!
//! Debug thresholds are 5× lower.

use std::hint::black_box;
use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::models::ModelBuilder;
use synapse_inference::weight_loading::AlignedBuffer;

// ── Helpers ──────────────────────────────────────────────────────────

/// Deterministic pseudo-random f32 values in [-1, 1] via xorshift64.
fn pseudo_rand(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Return median of a mutable slice of durations (seconds).
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    if n % 2 == 1 {
        times[n / 2]
    } else {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    }
}

/// Threshold multiplier: 1.0 in release, 0.2 (5× lower) in debug.
fn threshold_scale() -> f64 {
    if cfg!(debug_assertions) {
        0.2
    } else {
        1.0
    }
}

// ── Naive implementations ───────────────────────────────────────────
// These mirror the pub(crate) functions in decoder_layer.rs but are
// accessible from integration tests.  `black_box` prevents LLVM from
// auto-vectorising the scalar loops, giving a fair comparison.

/// Naive triple-loop y = A * B^T.  A:[m,k], B:[n,k] → y:[m,n].
#[inline(never)]
fn matmul_t_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for d in 0..k {
                sum = black_box(sum + a_row[d] * b_row[d]);
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// SIMD matmul via Zig FFI: y = A * B^T.
fn matmul_t_simd(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let status = unsafe {
        synapse_sys::syn_sgemm(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            0,
            b.as_ptr(),
            k,
            1,
            out.as_mut_ptr(),
            n,
        )
    };
    assert_eq!(status, synapse_sys::SYN_OK, "syn_sgemm failed: {status}");
    out
}

/// Naive scalar RMS normalization. `black_box` prevents auto-vectorisation.
#[inline(never)]
fn rmsnorm_naive(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mut ms = 0.0f32;
        for j in 0..hidden_size {
            ms += slice[j] * slice[j];
            ms = black_box(ms);
        }
        ms /= hidden_size as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = black_box(slice[j] * scale) * weight[j];
        }
    }
    out
}

/// SIMD RMS normalization via Zig FFI (vmul + vreduce_sum).
fn rmsnorm_simd(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    unsafe {
        for i in 0..n {
            let off = i * hidden_size;
            let row_ptr = x.as_ptr().add(off);
            let out_ptr = out.as_mut_ptr().add(off);

            synapse_sys::syn_vmul(out_ptr, row_ptr, row_ptr, hidden_size);

            let mut sum_sq = 0.0f32;
            synapse_sys::syn_vreduce_sum(out_ptr, hidden_size, &mut sum_sq);

            let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

            synapse_sys::syn_vmul(out_ptr, row_ptr, weight.as_ptr(), hidden_size);

            for j in 0..hidden_size {
                *out_ptr.add(j) *= scale;
            }
        }
    }
    out
}

/// Naive SwiGLU: silu(gate) ⊙ up.
///
/// Each intermediate is wrapped in `black_box` and processed via explicit
/// index loop to prevent LLVM from vectorising exp/div with SIMD intrinsics.
#[inline(never)]
fn swiglu_naive(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let len = gate.len();
    let mut out = vec![0.0f32; len];
    for i in 0..len {
        let g = black_box(gate[i]);
        let u = black_box(up[i]);
        let neg_g = black_box(-g);
        let exp_val = black_box(neg_g.exp());
        let denom = black_box(1.0 + exp_val);
        let silu = black_box(g / denom);
        out[i] = black_box(silu * u);
    }
    out
}

/// SIMD SwiGLU via Zig FFI.
fn swiglu_simd(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; gate.len()];
    let status = unsafe {
        synapse_sys::syn_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), gate.len())
    };
    assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
    out
}

/// Naive softmax over a mutable slice.
fn softmax_naive(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// Full naive decoder-layer forward (scalar only, no FFI).
///
/// Implements pre-norm: rmsnorm → attention(QKV proj + causal MHA) → residual → rmsnorm → SwiGLU FFN → residual.
#[inline(never)]
fn decoder_layer_naive_forward(
    x: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    eps: f32,
    attn_norm_w: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    ffn_norm_w: &[f32],
    ffn_gate: &[f32],
    ffn_up: &[f32],
    ffn_down: &[f32],
) -> Vec<f32> {
    let h = hidden_size;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // 1. Attention sub-layer
    let normed = rmsnorm_naive(x, attn_norm_w, eps, h);
    let q = matmul_t_naive(&normed, w_q, seq_len, h, q_dim);
    let k = matmul_t_naive(&normed, w_k, seq_len, h, kv_dim);
    let v = matmul_t_naive(&normed, w_v, seq_len, h, kv_dim);

    // Headwise RMS norm on Q and K
    let q = headwise_rmsnorm_naive(&q, q_norm_w, seq_len, num_heads, head_dim, eps);
    let k = headwise_rmsnorm_naive(&k, k_norm_w, seq_len, num_kv_heads, head_dim, eps);

    // Causal multi-head attention with GQA
    let mut attn_output = vec![0.0f32; seq_len * q_dim];
    for head in 0..num_heads {
        let kv_head = head / groups;
        for t in 0..seq_len {
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for s in 0..=t {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot +=
                        q[t * q_dim + head * head_dim + d] * k[s * kv_dim + kv_head * head_dim + d];
                }
                scores[s] = dot * scale;
            }
            softmax_naive(&mut scores[..=t]);
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..=t {
                    sum += scores[s] * v[s * kv_dim + kv_head * head_dim + d];
                }
                attn_output[t * q_dim + head * head_dim + d] = sum;
            }
        }
    }

    let attn_proj = matmul_t_naive(&attn_output, w_o, seq_len, q_dim, h);
    let mut residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

    // 2. FFN sub-layer (SwiGLU)
    let normed = rmsnorm_naive(&residual, ffn_norm_w, eps, h);
    let gate = matmul_t_naive(&normed, ffn_gate, seq_len, h, intermediate_size);
    let up = matmul_t_naive(&normed, ffn_up, seq_len, h, intermediate_size);
    let hidden = swiglu_naive(&gate, &up);
    let ffn_out = matmul_t_naive(&hidden, ffn_down, seq_len, intermediate_size, h);

    for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
        *r += f;
    }
    residual
}

/// Naive headwise RMS normalization.
#[inline(never)]
fn headwise_rmsnorm_naive(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    if weight.is_empty() {
        return x.to_vec();
    }
    let mut out = vec![0.0f32; x.len()];
    let stride = heads * head_dim;
    for row in 0..rows {
        let row_offset = row * stride;
        for head in 0..heads {
            let head_offset = row_offset + head * head_dim;
            let slice = &x[head_offset..head_offset + head_dim];
            let ms = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let scale_val = 1.0 / (ms + eps).sqrt();
            for idx in 0..head_dim {
                out[head_offset + idx] = slice[idx] * scale_val * weight[idx];
            }
        }
    }
    out
}

// ── Benchmark runner ────────────────────────────────────────────────

const WARMUP: usize = 3;
const ITERATIONS: usize = 15;

/// Run a benchmark: warmup, then N iterations, return median time in seconds.
fn bench<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64());
    }
    median(&mut times)
}

// ── Test 1: SGEMM ───────────────────────────────────────────────────

#[test]
fn simd_sgemm_4x_vs_naive() {
    let m = 1024;
    let k = 1024;
    let n = 3072;

    let a = pseudo_rand(m * k, 42);
    let b = pseudo_rand(n * k, 137);

    let naive_median = bench(|| {
        black_box(matmul_t_naive(&a, &b, m, k, n));
    });

    let simd_median = bench(|| {
        black_box(matmul_t_simd(&a, &b, m, k, n));
    });

    let speedup = naive_median / simd_median;
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let naive_gflops = flops / naive_median / 1e9;
    let simd_gflops = flops / simd_median / 1e9;

    eprintln!(
        "SGEMM [{m}×{k}] × [{k}×{n}]:\n  \
         naive: {:.3}ms ({:.2} GFLOPS)\n  \
         SIMD:  {:.3}ms ({:.2} GFLOPS)\n  \
         speedup: {:.2}×",
        naive_median * 1e3,
        naive_gflops,
        simd_median * 1e3,
        simd_gflops,
        speedup,
    );

    let threshold = 4.0 * threshold_scale();
    assert!(
        speedup >= threshold,
        "SGEMM SIMD speedup {speedup:.2}× < {threshold:.1}× threshold"
    );
}

// ── Test 2: RMSNorm ─────────────────────────────────────────────────

#[test]
fn simd_rmsnorm_4x_vs_naive() {
    let hidden_size = 1024;
    let x = pseudo_rand(hidden_size, 99);
    let weight = pseudo_rand(hidden_size, 200);
    let eps = 1e-6f32;

    let naive_median = bench(|| {
        black_box(rmsnorm_naive(&x, &weight, eps, hidden_size));
    });

    let simd_median = bench(|| {
        black_box(rmsnorm_simd(&x, &weight, eps, hidden_size));
    });

    let speedup = naive_median / simd_median;
    // Bandwidth: read x (4B) + read weight (4B) + write out (4B) = 12 bytes/element
    let bytes = hidden_size as f64 * 12.0;
    let naive_gbs = bytes / naive_median / 1e9;
    let simd_gbs = bytes / simd_median / 1e9;

    eprintln!(
        "RMSNorm [1×{hidden_size}]:\n  \
         naive: {:.3}ms ({:.2} GB/s)\n  \
         SIMD:  {:.3}ms ({:.2} GB/s)\n  \
         speedup: {:.2}×",
        naive_median * 1e3,
        naive_gbs,
        simd_median * 1e3,
        simd_gbs,
        speedup,
    );

    let threshold = 4.0 * threshold_scale();
    assert!(
        speedup >= threshold,
        "RMSNorm SIMD speedup {speedup:.2}× < {threshold:.1}× threshold"
    );
}

// ── Test 3: SwiGLU ──────────────────────────────────────────────────

#[test]
fn simd_swiglu_2x_vs_naive() {
    let len = 3072;
    let gate = pseudo_rand(len, 300);
    let up = pseudo_rand(len, 400);

    let naive_median = bench(|| {
        black_box(swiglu_naive(&gate, &up));
    });

    let simd_median = bench(|| {
        black_box(swiglu_simd(&gate, &up));
    });

    let speedup = naive_median / simd_median;
    // Bandwidth: read gate (4B) + read up (4B) + write dst (4B) = 12 bytes/element
    let bytes = len as f64 * 12.0;
    let naive_gbs = bytes / naive_median / 1e9;
    let simd_gbs = bytes / simd_median / 1e9;

    eprintln!(
        "SwiGLU [1×{len}]:\n  \
         naive: {:.3}ms ({:.2} GB/s)\n  \
         SIMD:  {:.3}ms ({:.2} GB/s)\n  \
         speedup: {:.2}×",
        naive_median * 1e3,
        naive_gbs,
        simd_median * 1e3,
        simd_gbs,
        speedup,
    );

    let threshold = 2.0 * threshold_scale();
    assert!(
        speedup >= threshold,
        "SwiGLU SIMD speedup {speedup:.2}× < {threshold:.1}× threshold"
    );
}

// ── Test 4: Decoder layer end-to-end ────────────────────────────────

fn bench_model_config() -> ModelConfig {
    ModelConfig {
        name: "SIMDBench".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 1024,
            num_layers: 1,
            vocab_size: 128,
            max_sequence_length: 64,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 2816,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 64,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn gen_aligned(len: usize, seed: u64) -> AlignedBuffer {
    AlignedBuffer::from_vec(pseudo_rand(len, seed))
}

fn ones_aligned(len: usize) -> AlignedBuffer {
    AlignedBuffer::from_vec(vec![1.0f32; len])
}

#[test]
fn simd_decoder_layer_vs_naive_e2e() {
    let cfg = bench_model_config();
    let h = cfg.architecture.hidden_size;
    let num_heads = cfg.attention.num_heads();
    let num_kv_heads = cfg.attention.num_kv_heads();
    let head_dim = cfg.attention.head_dim();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let inter = cfg.ffn.intermediate_size();
    let eps = 1e-6f32;
    let seq_len = 4;

    // Build SIMD decoder layer via model builder
    let mut model = ModelBuilder::from_config(&cfg);
    {
        let layer = &mut model.layers[0];
        layer.attn_norm_weight = ones_aligned(h);
        layer.w_q = gen_aligned(q_dim * h, 10);
        layer.w_k = gen_aligned(kv_dim * h, 20);
        layer.w_v = gen_aligned(kv_dim * h, 30);
        layer.w_o = gen_aligned(h * q_dim, 40);
        layer.q_norm_weight = gen_aligned(head_dim, 50);
        layer.k_norm_weight = gen_aligned(head_dim, 60);
        layer.ffn_norm_weight = ones_aligned(h);
        layer.ffn_gate = gen_aligned(inter * h, 70);
        layer.ffn_up = gen_aligned(inter * h, 80);
        layer.ffn_down = gen_aligned(h * inter, 90);
    }

    // Extract raw weight slices for the naive path
    let layer = &model.layers[0];
    let attn_norm_w: Vec<f32> = layer.attn_norm_weight.to_vec();
    let w_q: Vec<f32> = layer.w_q.to_vec();
    let w_k: Vec<f32> = layer.w_k.to_vec();
    let w_v: Vec<f32> = layer.w_v.to_vec();
    let w_o: Vec<f32> = layer.w_o.to_vec();
    let q_norm_w: Vec<f32> = layer.q_norm_weight.to_vec();
    let k_norm_w: Vec<f32> = layer.k_norm_weight.to_vec();
    let ffn_norm_w: Vec<f32> = layer.ffn_norm_weight.to_vec();
    let fg: Vec<f32> = layer.ffn_gate.to_vec();
    let fu: Vec<f32> = layer.ffn_up.to_vec();
    let fd: Vec<f32> = layer.ffn_down.to_vec();

    let x = pseudo_rand(seq_len * h, 1000);

    let naive_median = bench(|| {
        black_box(decoder_layer_naive_forward(
            &x,
            seq_len,
            h,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
            eps,
            &attn_norm_w,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &q_norm_w,
            &k_norm_w,
            &ffn_norm_w,
            &fg,
            &fu,
            &fd,
        ));
    });

    let simd_median = bench(|| {
        black_box(layer.forward(&x, seq_len, &model.rope_cos, &model.rope_sin));
    });

    let speedup = naive_median / simd_median;

    eprintln!(
        "Decoder layer (h={h}, heads={num_heads}, inter={inter}, seq={seq_len}):\n  \
         naive: {:.3}ms\n  \
         SIMD:  {:.3}ms\n  \
         speedup: {:.2}×",
        naive_median * 1e3,
        simd_median * 1e3,
        speedup,
    );

    let threshold = 2.0 * threshold_scale();
    assert!(
        speedup >= threshold,
        "Decoder layer SIMD speedup {speedup:.2}× < {threshold:.1}× threshold"
    );
}
