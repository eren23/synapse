use crate::ir::{Graph, NodeId, NodeKind, OpKind};
use crate::pass::OptimizationPass;

/// Fuses the multi-head attention pattern into a single FusedAttention node.
///
/// Detected pattern:
///   q = MatMul(input, W_q)
///   k = MatMul(input, W_k)
///   k_t = Transpose(k)
///   scores = MatMul(q, k_t)
///   scaled = Mul(scores, scale_factor)
///   weights = Softmax(scaled)
///   v = MatMul(input, W_v)
///   attended = MatMul(weights, v)
///   output = MatMul(attended, W_o)
///
/// Replaces with: FusedAttention(input, W_q, W_k, W_v, scale_factor, W_o)
pub struct FuseAttention;

impl FuseAttention {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FuseAttention {
    fn name(&self) -> &str {
        "fuse_attention"
    }

    fn run(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Strategy: find Softmax nodes and work outward to match the full pattern.
        let softmax_ids: Vec<NodeId> = graph
            .node_ids()
            .into_iter()
            .filter(|&id| {
                graph
                    .node(id)
                    .map(|n| matches!(n.kind, NodeKind::Op(OpKind::Softmax)))
                    .unwrap_or(false)
            })
            .collect();

        for softmax_id in softmax_ids {
            if let Some(did_fuse) = try_fuse_attention(graph, softmax_id) {
                if did_fuse {
                    changed = true;
                }
            }
        }

        changed
    }
}

fn try_fuse_attention(graph: &mut Graph, softmax_id: NodeId) -> Option<bool> {
    let softmax_node = graph.node(softmax_id)?;
    if softmax_node.inputs.len() != 1 {
        return Some(false);
    }

    // ── Step 1: Softmax input should be Mul (scale step) ──
    let mul_id = softmax_node.inputs[0];
    let mul_node = graph.node(mul_id)?;
    if !matches!(mul_node.kind, NodeKind::Op(OpKind::Mul)) {
        return Some(false);
    }
    if !graph.has_single_user(mul_id) || mul_node.inputs.len() != 2 {
        return Some(false);
    }

    // One input of Mul is the score MatMul, the other is the scale factor
    let (score_mm_id, scale_id) = if is_matmul(graph, mul_node.inputs[0]) {
        (mul_node.inputs[0], mul_node.inputs[1])
    } else if is_matmul(graph, mul_node.inputs[1]) {
        (mul_node.inputs[1], mul_node.inputs[0])
    } else {
        return Some(false);
    };

    if !graph.has_single_user(score_mm_id) {
        return Some(false);
    }

    // ── Step 2: Score MatMul inputs are Q projection and transposed K ──
    let score_mm = graph.node(score_mm_id)?;
    if score_mm.inputs.len() != 2 {
        return Some(false);
    }
    let q_proj_id = score_mm.inputs[0];
    let k_t_id = score_mm.inputs[1];

    // Q must be a MatMul (projection)
    if !is_matmul(graph, q_proj_id) || !graph.has_single_user(q_proj_id) {
        return Some(false);
    }

    // K^T must be a Transpose
    let k_t_node = graph.node(k_t_id)?;
    if !matches!(k_t_node.kind, NodeKind::Op(OpKind::Transpose)) {
        return Some(false);
    }
    if !graph.has_single_user(k_t_id) || k_t_node.inputs.len() != 1 {
        return Some(false);
    }

    // Transpose input must be K projection (MatMul)
    let k_proj_id = k_t_node.inputs[0];
    if !is_matmul(graph, k_proj_id) || !graph.has_single_user(k_proj_id) {
        return Some(false);
    }

    // ── Step 3: Q and K projections must share the same input tensor ──
    let q_proj = graph.node(q_proj_id)?;
    let k_proj = graph.node(k_proj_id)?;
    if q_proj.inputs.len() != 2 || k_proj.inputs.len() != 2 {
        return Some(false);
    }
    let input_id = q_proj.inputs[0];
    if k_proj.inputs[0] != input_id {
        return Some(false);
    }
    let w_q_id = q_proj.inputs[1];
    let w_k_id = k_proj.inputs[1];

    // ── Step 4: Softmax -> value aggregation MatMul ──
    if !graph.has_single_user(softmax_id) {
        return Some(false);
    }
    let attn_mm_users = graph.users(softmax_id);
    if attn_mm_users.len() != 1 {
        return Some(false);
    }
    let attn_mm_id = attn_mm_users[0];

    let attn_mm = graph.node(attn_mm_id)?;
    if !matches!(attn_mm.kind, NodeKind::Op(OpKind::MatMul)) {
        return Some(false);
    }
    if attn_mm.inputs.len() != 2 || attn_mm.inputs[0] != softmax_id {
        return Some(false);
    }

    // The other input is V projection (MatMul from same input)
    let v_proj_id = attn_mm.inputs[1];
    if !is_matmul(graph, v_proj_id) || !graph.has_single_user(v_proj_id) {
        return Some(false);
    }
    let v_proj = graph.node(v_proj_id)?;
    if v_proj.inputs.len() != 2 || v_proj.inputs[0] != input_id {
        return Some(false);
    }
    let w_v_id = v_proj.inputs[1];

    // ── Step 5: Value aggregation -> output projection MatMul ──
    if !graph.has_single_user(attn_mm_id) {
        return Some(false);
    }
    let out_proj_users = graph.users(attn_mm_id);
    if out_proj_users.len() != 1 {
        return Some(false);
    }
    let out_proj_id = out_proj_users[0];

    let out_proj = graph.node(out_proj_id)?;
    if !matches!(out_proj.kind, NodeKind::Op(OpKind::MatMul)) {
        return Some(false);
    }
    if out_proj.inputs.len() != 2 || out_proj.inputs[0] != attn_mm_id {
        return Some(false);
    }
    let w_o_id = out_proj.inputs[1];
    let output_meta = out_proj.meta.clone();

    // ── Full pattern matched! Create fused node. ──
    // Inputs: [input, W_q, W_k, W_v, scale, W_o]
    let fused_id = graph.add_node(
        NodeKind::Op(OpKind::FusedAttention),
        vec![input_id, w_q_id, w_k_id, w_v_id, scale_id, w_o_id],
        output_meta,
        "fused_attention",
    );

    graph.replace_all_uses(out_proj_id, fused_id);
    graph.remove_node(out_proj_id);
    graph.remove_node(attn_mm_id);
    graph.remove_node(softmax_id);
    graph.remove_node(mul_id);
    graph.remove_node(score_mm_id);
    graph.remove_node(k_t_id);
    graph.remove_node(q_proj_id);
    graph.remove_node(k_proj_id);
    graph.remove_node(v_proj_id);

    Some(true)
}

fn is_matmul(graph: &Graph, id: NodeId) -> bool {
    graph
        .node(id)
        .map(|n| matches!(n.kind, NodeKind::Op(OpKind::MatMul)))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use std::collections::HashMap;

    /// Build a self-attention graph:
    ///   q = MatMul(input, W_q)
    ///   k = MatMul(input, W_k)
    ///   k_t = Transpose(k)
    ///   scores = MatMul(q, k_t)
    ///   scaled = Mul(scores, scale)
    ///   weights = Softmax(scaled)
    ///   v = MatMul(input, W_v)
    ///   attended = MatMul(weights, v)
    ///   output = MatMul(attended, W_o)
    fn build_attention_graph(
        seq_len: usize,
        d_model: usize,
        d_k: usize,
        d_v: usize,
        d_out: usize,
    ) -> (Graph, NodeId, Vec<NodeId>) {
        let mut g = Graph::new();

        let input = g.add_node(
            NodeKind::Input("input".into()),
            vec![],
            NodeMeta::new(vec![seq_len, d_model], DType::F32),
            "input",
        );
        let w_q = g.add_node(
            NodeKind::Parameter("W_q".into()),
            vec![],
            NodeMeta::new(vec![d_model, d_k], DType::F32),
            "W_q",
        );
        let w_k = g.add_node(
            NodeKind::Parameter("W_k".into()),
            vec![],
            NodeMeta::new(vec![d_model, d_k], DType::F32),
            "W_k",
        );
        let w_v = g.add_node(
            NodeKind::Parameter("W_v".into()),
            vec![],
            NodeMeta::new(vec![d_model, d_v], DType::F32),
            "W_v",
        );
        let scale_val = 1.0 / (d_k as f32).sqrt();
        let scale = g.add_node(
            NodeKind::Constant(vec![scale_val; seq_len * seq_len]),
            vec![],
            NodeMeta::new(vec![seq_len, seq_len], DType::F32),
            "scale",
        );
        let w_o = g.add_node(
            NodeKind::Parameter("W_o".into()),
            vec![],
            NodeMeta::new(vec![d_v, d_out], DType::F32),
            "W_o",
        );

        // Q, K, V projections
        let q = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![input, w_q],
            NodeMeta::new(vec![seq_len, d_k], DType::F32),
            "q_proj",
        );
        let k = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![input, w_k],
            NodeMeta::new(vec![seq_len, d_k], DType::F32),
            "k_proj",
        );
        let k_t = g.add_node(
            NodeKind::Op(OpKind::Transpose),
            vec![k],
            NodeMeta::new(vec![d_k, seq_len], DType::F32),
            "k_transpose",
        );
        let v = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![input, w_v],
            NodeMeta::new(vec![seq_len, d_v], DType::F32),
            "v_proj",
        );

        // Attention computation
        let scores = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![q, k_t],
            NodeMeta::new(vec![seq_len, seq_len], DType::F32),
            "scores",
        );
        let scaled = g.add_node(
            NodeKind::Op(OpKind::Mul),
            vec![scores, scale],
            NodeMeta::new(vec![seq_len, seq_len], DType::F32),
            "scaled_scores",
        );
        let weights = g.add_node(
            NodeKind::Op(OpKind::Softmax),
            vec![scaled],
            NodeMeta::new(vec![seq_len, seq_len], DType::F32),
            "attn_weights",
        );
        let attended = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![weights, v],
            NodeMeta::new(vec![seq_len, d_v], DType::F32),
            "attended",
        );

        // Output projection
        let output = g.add_node(
            NodeKind::Op(OpKind::MatMul),
            vec![attended, w_o],
            NodeMeta::new(vec![seq_len, d_out], DType::F32),
            "output_proj",
        );
        g.mark_output(output);

        let param_ids = vec![input, w_q, w_k, w_v, scale, w_o];
        (g, output, param_ids)
    }

    #[test]
    fn test_fuse_attention_pattern() {
        let (mut g, _output, _params) = build_attention_graph(2, 4, 3, 3, 4);

        // 6 inputs/params + 9 ops = 15 nodes
        assert_eq!(g.node_count(), 15);

        let pass = FuseAttention::new();
        let changed = pass.run(&mut g);
        assert!(changed);

        // 9 ops replaced by 1 fused node => 15 - 9 + 1 = 7
        assert_eq!(g.node_count(), 7);

        let output_id = g.outputs()[0];
        let fused = g.node(output_id).unwrap();
        assert!(matches!(fused.kind, NodeKind::Op(OpKind::FusedAttention)));
        assert_eq!(fused.inputs.len(), 6); // input, W_q, W_k, W_v, scale, W_o
    }

    #[test]
    fn test_fuse_attention_idempotent() {
        let (mut g, _, _) = build_attention_graph(2, 4, 3, 3, 4);

        let pass = FuseAttention::new();
        pass.run(&mut g);
        let count_after_first = g.node_count();

        let changed = pass.run(&mut g);
        assert!(!changed, "Second run should not change anything");
        assert_eq!(g.node_count(), count_after_first);
    }

    #[test]
    fn test_fuse_attention_numerically_identical() {
        let seq_len = 2;
        let d_model = 4;
        let d_k = 3;
        let d_v = 3;
        let d_out = 4;

        let (g_unfused, _, param_ids) =
            build_attention_graph(seq_len, d_model, d_k, d_v, d_out);

        let input_data: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32) * 0.1 + 0.05)
            .collect();
        let w_q_data: Vec<f32> = (0..d_model * d_k)
            .map(|i| (i as f32) * 0.02 - 0.1)
            .collect();
        let w_k_data: Vec<f32> = (0..d_model * d_k)
            .map(|i| (i as f32) * 0.03 + 0.05)
            .collect();
        let w_v_data: Vec<f32> = (0..d_model * d_v)
            .map(|i| (i as f32) * 0.01 - 0.02)
            .collect();
        let w_o_data: Vec<f32> = (0..d_v * d_out)
            .map(|i| (i as f32) * 0.04 + 0.01)
            .collect();

        let mut inputs = HashMap::new();
        inputs.insert(param_ids[0], input_data);
        inputs.insert(param_ids[1], w_q_data);
        inputs.insert(param_ids[2], w_k_data);
        inputs.insert(param_ids[3], w_v_data);
        // scale is a constant embedded in the graph, no need to pass
        inputs.insert(param_ids[5], w_o_data);

        let unfused_output_id = g_unfused.outputs()[0];
        let unfused_result = g_unfused.execute(&inputs);
        let unfused_out = unfused_result[&unfused_output_id].clone();

        let mut g_fused = g_unfused.clone();
        FuseAttention::new().run(&mut g_fused);

        let fused_output_id = g_fused.outputs()[0];
        let fused_result = g_fused.execute(&inputs);
        let fused_out = fused_result[&fused_output_id].clone();

        assert_eq!(unfused_out.len(), fused_out.len());
        for (i, (u, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
            assert!(
                (u - f).abs() < 1e-4,
                "Attention fusion mismatch at {}: unfused={}, fused={}, diff={}",
                i, u, f, (u - f).abs()
            );
        }
    }
}
