//! GPU-accelerated Code WM encoder — single Metal command buffer, zero CPU-GPU sync.
//!
//! All encoder_loops encoded into ONE command buffer. GPU processes them back-to-back.
//! For 128-d models, CPU BLAS outperforms GPU due to dispatch overhead. This path
//! becomes competitive at model_dim >= 256 where GPU parallelism overcomes overhead.

use ::metal::{Buffer, Device, MTLResourceOptions, MTLSize};

use super::device::MetalBackend;
use crate::models::vision::code_wm::CodeWorldModel;

pub struct MetalCodeWMWeights {
    pub norm1_w: Buffer, pub norm1_b: Buffer,
    pub attn_in_w: Buffer, pub attn_in_b: Buffer,
    pub attn_out_w: Buffer, pub attn_out_b: Buffer,
    pub norm2_w: Buffer, pub norm2_b: Buffer,
    pub mlp_up_w: Buffer, pub mlp_up_b: Buffer,
    pub mlp_down_w: Buffer, pub mlp_down_b: Buffer,
}

pub struct MetalCodeWMState {
    pub weights: MetalCodeWMWeights,
    pub seq: Buffer, pub normed: Buffer,
    pub q: Buffer, pub k: Buffer, pub v: Buffer,
    pub attn_out: Buffer, pub proj: Buffer,
    pub buf_hidden: Buffer, pub buf_mlp_hidden: Buffer,
    pub buf_seq_len: Buffer, pub buf_head_dim: Buffer,
    pub buf_stride: Buffer, pub buf_eps: Buffer,
    pub buf_total_sh: Buffer, pub buf_total_smlp: Buffer,
    pub hidden: usize, pub mlp_hidden: usize,
    pub num_heads: usize, pub head_dim: usize,
    pub seq_len: usize, pub encoder_loops: usize,
}

fn up(device: &Device, data: &[f32]) -> Buffer {
    if data.is_empty() { return al(device, 1); }
    device.new_buffer_with_data(data.as_ptr() as *const _, (data.len() * 4) as u64, MTLResourceOptions::StorageModeShared)
}
fn al(device: &Device, n: usize) -> Buffer {
    device.new_buffer((n.max(1) * 4) as u64, MTLResourceOptions::StorageModeShared)
}
fn cu(device: &Device, v: u32) -> Buffer {
    let d = [f32::from_bits(v)];
    device.new_buffer_with_data(d.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared)
}
fn cf(device: &Device, v: f32) -> Buffer {
    device.new_buffer_with_data(&v as *const f32 as *const _, 4, MTLResourceOptions::StorageModeShared)
}

impl MetalCodeWMState {
    pub fn from_model(model: &CodeWorldModel, seq_len: usize, backend: &MetalBackend) -> Self {
        let d = &backend.device;
        let h = model.config.model_dim;
        let mh = model.config.mlp_hidden;
        let b = &model.encoder_block;
        MetalCodeWMState {
            weights: MetalCodeWMWeights {
                norm1_w: up(d, &b.norm1.weight), norm1_b: up(d, &b.norm1.bias),
                attn_in_w: up(d, &b.attn_in_proj.weight), attn_in_b: up(d, &b.attn_in_proj.bias),
                attn_out_w: up(d, &b.attn_out_proj.weight), attn_out_b: up(d, &b.attn_out_proj.bias),
                norm2_w: up(d, &b.norm2.weight), norm2_b: up(d, &b.norm2.bias),
                mlp_up_w: up(d, &b.mlp_up.weight), mlp_up_b: up(d, &b.mlp_up.bias),
                mlp_down_w: up(d, &b.mlp_down.weight), mlp_down_b: up(d, &b.mlp_down.bias),
            },
            seq: al(d, seq_len*h), normed: al(d, seq_len*h),
            q: al(d, seq_len*h), k: al(d, seq_len*h), v: al(d, seq_len*h),
            attn_out: al(d, seq_len*h), proj: al(d, seq_len*mh.max(h)),
            buf_hidden: cu(d, h as u32), buf_mlp_hidden: cu(d, mh as u32),
            buf_seq_len: cu(d, seq_len as u32),
            buf_head_dim: cu(d, model.config.head_dim as u32),
            buf_stride: cu(d, h as u32),
            buf_eps: cf(d, model.config.layernorm_eps),
            buf_total_sh: cu(d, (seq_len*h) as u32),
            buf_total_smlp: cu(d, (seq_len*mh) as u32),
            hidden: h, mlp_hidden: mh,
            num_heads: model.config.num_heads, head_dim: model.config.head_dim,
            seq_len, encoder_loops: model.config.encoder_loops,
        }
    }
}

// Dispatch helpers
fn d1(cmd: &::metal::CommandBufferRef, pl: &::metal::ComputePipelineState, bufs: &[(&Buffer, u64)], n: usize) {
    let e = cmd.new_compute_command_encoder();
    e.set_compute_pipeline_state(pl);
    for (i, (b, o)) in bufs.iter().enumerate() { e.set_buffer(i as u64, Some(b), *o); }
    e.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256.min(n) as u64, 1, 1));
    e.end_encoding();
}

fn dln(cmd: &::metal::CommandBufferRef, pl: &::metal::ComputePipelineState,
       x: &Buffer, g: &Buffer, b: &Buffer, out: &Buffer, n: &Buffer, eps: &Buffer, rows: usize) {
    let e = cmd.new_compute_command_encoder();
    e.set_compute_pipeline_state(pl);
    e.set_buffer(0, Some(x), 0); e.set_buffer(1, Some(g), 0);
    e.set_buffer(2, Some(b), 0); e.set_buffer(3, Some(out), 0);
    e.set_buffer(4, Some(n), 0); e.set_buffer(5, Some(eps), 0);
    e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    e.end_encoding();
}

fn dgv(cmd: &::metal::CommandBufferRef, pl: &::metal::ComputePipelineState,
       a: &Buffer, b: &Buffer, bo: u64, c: &Buffer,
       m: &Buffer, nb: &Buffer, k: &Buffer, n: usize) {
    let e = cmd.new_compute_command_encoder();
    e.set_compute_pipeline_state(pl);
    e.set_buffer(0, Some(a), 0); e.set_buffer(1, Some(b), bo);
    e.set_buffer(2, Some(c), 0); e.set_buffer(3, Some(m), 0);
    e.set_buffer(4, Some(nb), 0); e.set_buffer(5, Some(k), 0);
    e.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256.min(n) as u64, 1, 1));
    e.end_encoding();
}

fn dat(cmd: &::metal::CommandBufferRef, pl: &::metal::ComputePipelineState,
       q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer,
       sl: &Buffer, hd: &Buffer, st: &Buffer, slen: usize, ho: u64) {
    let e = cmd.new_compute_command_encoder();
    e.set_compute_pipeline_state(pl);
    e.set_buffer(0, Some(q), ho); e.set_buffer(1, Some(k), ho);
    e.set_buffer(2, Some(v), ho); e.set_buffer(3, Some(out), ho);
    e.set_buffer(4, Some(sl), 0); e.set_buffer(5, Some(hd), 0); e.set_buffer(6, Some(st), 0);
    e.dispatch_thread_groups(MTLSize::new(slen as u64, 1, 1), MTLSize::new(256, 1, 1));
    e.end_encoding();
}

/// Run CodeWM encoder fully on GPU. Single command buffer, zero CPU-GPU sync.
pub fn code_wm_encode_metal(
    st: &MetalCodeWMState, input: &[f32], be: &MetalBackend,
) -> Vec<f32> {
    let h = st.hidden; let mh = st.mlp_hidden; let s = st.seq_len;
    let w = &st.weights;

    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), st.seq.contents() as *mut f32, s*h); }

    let ln = be.pipeline("layernorm_wb").expect("layernorm_wb");
    let gv = be.pipeline("gemv3_t").expect("gemv3_t");
    let ab = be.pipeline("add_bias").expect("add_bias");
    let ge = be.pipeline("gelu_inplace").expect("gelu_inplace");
    let ea = be.pipeline("elementwise_add").expect("elementwise_add");
    let at = be.pipeline("attention_bidi").expect("attention_bidi");

    let cmd = be.command_queue.new_command_buffer();

    for _ in 0..st.encoder_loops {
        dln(cmd, ln, &st.seq, &w.norm1_w, &w.norm1_b, &st.normed, &st.buf_hidden, &st.buf_eps, s);

        // Q/K/V via gemv3_t with row offsets into [3h, h] weight
        let wk = (h*h*4) as u64; let wv = (2*h*h*4) as u64;
        let bk = (h*4) as u64;   let bv = (2*h*4) as u64;
        dgv(cmd, gv, &st.normed, &w.attn_in_w, 0,  &st.q, &st.buf_seq_len, &st.buf_hidden, &st.buf_hidden, h);
        dgv(cmd, gv, &st.normed, &w.attn_in_w, wk, &st.k, &st.buf_seq_len, &st.buf_hidden, &st.buf_hidden, h);
        dgv(cmd, gv, &st.normed, &w.attn_in_w, wv, &st.v, &st.buf_seq_len, &st.buf_hidden, &st.buf_hidden, h);
        d1(cmd, ab, &[(&st.q, 0), (&w.attn_in_b, 0),  (&st.buf_hidden, 0), (&st.buf_total_sh, 0)], s*h);
        d1(cmd, ab, &[(&st.k, 0), (&w.attn_in_b, bk), (&st.buf_hidden, 0), (&st.buf_total_sh, 0)], s*h);
        d1(cmd, ab, &[(&st.v, 0), (&w.attn_in_b, bv), (&st.buf_hidden, 0), (&st.buf_total_sh, 0)], s*h);

        for hd in 0..st.num_heads {
            dat(cmd, at, &st.q, &st.k, &st.v, &st.attn_out,
                &st.buf_seq_len, &st.buf_head_dim, &st.buf_stride, s, (hd*st.head_dim*4) as u64);
        }

        dgv(cmd, gv, &st.attn_out, &w.attn_out_w, 0, &st.proj, &st.buf_seq_len, &st.buf_hidden, &st.buf_hidden, h);
        d1(cmd, ab, &[(&st.proj, 0), (&w.attn_out_b, 0), (&st.buf_hidden, 0), (&st.buf_total_sh, 0)], s*h);
        d1(cmd, ea, &[(&st.seq, 0), (&st.proj, 0), (&st.seq, 0), (&st.buf_total_sh, 0)], s*h);

        dln(cmd, ln, &st.seq, &w.norm2_w, &w.norm2_b, &st.normed, &st.buf_hidden, &st.buf_eps, s);

        dgv(cmd, gv, &st.normed, &w.mlp_up_w, 0, &st.proj, &st.buf_seq_len, &st.buf_mlp_hidden, &st.buf_hidden, mh);
        d1(cmd, ab, &[(&st.proj, 0), (&w.mlp_up_b, 0), (&st.buf_mlp_hidden, 0), (&st.buf_total_smlp, 0)], s*mh);
        d1(cmd, ge, &[(&st.proj, 0), (&st.buf_total_smlp, 0)], s*mh);

        dgv(cmd, gv, &st.proj, &w.mlp_down_w, 0, &st.normed, &st.buf_seq_len, &st.buf_hidden, &st.buf_mlp_hidden, h);
        d1(cmd, ab, &[(&st.normed, 0), (&w.mlp_down_b, 0), (&st.buf_hidden, 0), (&st.buf_total_sh, 0)], s*h);
        d1(cmd, ea, &[(&st.seq, 0), (&st.normed, 0), (&st.seq, 0), (&st.buf_total_sh, 0)], s*h);
    }

    cmd.commit();
    cmd.wait_until_completed();

    let mut out = vec![0.0f32; s*h];
    unsafe { std::ptr::copy_nonoverlapping(st.seq.contents() as *const f32, out.as_mut_ptr(), s*h); }
    out
}
