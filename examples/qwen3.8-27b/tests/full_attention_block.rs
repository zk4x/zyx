// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end full_attention block test (Qwen3.5-27B, layer 3).
//! Mirrors vLLM's Qwen3NextAttention.forward (full_attention branch):
//!   input_rmsnorm -> qkv_proj (q+gate fused) -> q,k split ->
//!   qk_norm (per-head, with weight) -> rope -> attention (with output gate) ->
//!   o_proj -> residual_add -> output.
//!
//! Real weights from blk.3.

use qwen3_8_27b::{
    HEADS, HEAD_DIM, HIDDEN, KV_HEADS, M_PAD, ROT_DIM, S,
    attention_kernel, gemm_kernel, input_rmsnorm_kernel, pad_kernel,
    qk_norm_kernel, residual_add_kernel, rope_kernel,
};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn full_attention_block() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_full_attention_block.safetensors")?;
    let input = goldens["input"].to(dev)?;            // [S, HIDDEN]
    let w_q = goldens["w_q"].to(dev)?;                // [HIDDEN, 2*H*HEAD_DIM] (q+gate fused)
    let w_k = goldens["w_k"].to(dev)?;                // [HIDDEN, KV*HEAD_DIM]
    let w_v = goldens["w_v"].to(dev)?;                // [HIDDEN, KV*HEAD_DIM]
    let w_o = goldens["w_o"].to(dev)?;                // [HIDDEN, H*HEAD_DIM]
    let attn_norm = goldens["attn_norm"].to(dev)?;  // [HIDDEN]
    let q_norm_w = goldens["q_norm_w"].to(dev)?;    // [HEAD_DIM]
    let k_norm_w = goldens["k_norm_w"].to(dev)?;    // [HEAD_DIM]
    let cos = goldens["cos"].to(dev)?;                // [S, ROT_DIM]
    let sin = goldens["sin"].to(dev)?;                // [S, ROT_DIM]
    let expected = &goldens["output"];                 // [S, HIDDEN]

    // 1. Input RMSNorm
    let rn = input_rmsnorm_kernel().compile()?;
    let x_norm = rn.forward(&[&input, &attn_norm], vec![[S, HIDDEN]])?.remove(0);
    x_norm.sync()?;

    // Pad [S, HIDDEN] -> [M_PAD, HIDDEN] for gemm (needs R%16=0)
    let pad_in = pad_kernel(S, M_PAD, HIDDEN).compile()?;
    let x = pad_in.forward(&[&input], vec![[M_PAD, HIDDEN]])?.remove(0);
    x.sync()?;
    // Re-run rmsnorm on padded input
    let x = rn.forward(&[&x, &attn_norm], vec![[M_PAD, HIDDEN]])?.remove(0);
    x.sync()?;
    let _ = x_norm;

    // 2. QKV projections: split w_q into q_proj and gate_proj.
    //    w_q is [QKV_TOTAL, HIDDEN] = [2*H*D, HIDDEN] = [q_part, gate_part] stacked.
    //    For the kernel we project them separately. gemm requires R%16=0, so we
    //    use M_PAD=16 as the row count.
    let h_total = HEADS * HEAD_DIM;
    let kv_total = KV_HEADS * HEAD_DIM;
    let q_w_only = goldens["q_w_only"].to(dev)?;
    let gate_w_only = goldens["gate_w_only"].to(dev)?;
    let q_proj = gemm_kernel(M_PAD, HIDDEN, h_total).compile()?;
    let k_proj = gemm_kernel(M_PAD, HIDDEN, kv_total).compile()?;
    let v_proj = gemm_kernel(M_PAD, HIDDEN, kv_total).compile()?;
    let q_flat = q_proj.forward(&[&x, &q_w_only], vec![[S, h_total]])?.remove(0);
    q_flat.sync()?;
    eprintln!("q_flat ok");
    let gate_flat = q_proj.forward(&[&x, &gate_w_only], vec![[S, h_total]])?.remove(0);
    gate_flat.sync()?;
    eprintln!("gate_flat ok");
    let k_out = k_proj.forward(&[&x, &w_k], vec![[S, kv_total]])?.remove(0);
    k_out.sync()?;
    let v_out = v_proj.forward(&[&x, &w_v], vec![[S, kv_total]])?.remove(0);
    v_out.sync()?;

    // Reshape to [H, S, D] / [KV, S, D]
    let q = q_flat.reshape([HEADS, S, HEAD_DIM])?;
    let gate = gate_flat.reshape([HEADS, S, HEAD_DIM])?;
    let v = v_out.reshape([KV_HEADS, S, HEAD_DIM])?;

    // 4. Q/K norm (per-head, with weight)
    let qk = qk_norm_kernel(HEADS, KV_HEADS, HEAD_DIM).compile()?;
    let qk_out = qk.forward(&[&q, &k_out, &q_norm_w, &k_norm_w],
        vec![[HEADS, S, HEAD_DIM], [KV_HEADS, S, HEAD_DIM]])?;
    qk_out[0].sync()?;
    qk_out[1].sync()?;
    let q_n = &qk_out[0];
    let k_n = &qk_out[1];

    // 5. RoPE on Q and K
    let q_n_flat = q_n.reshape([HEADS * S, HEAD_DIM])?;
    let k_n_flat = k_n.reshape([KV_HEADS * S, HEAD_DIM])?;
    let rq = rope_kernel(S, HEADS, HEAD_DIM, ROT_DIM).compile()?;
    let rk = rope_kernel(S, KV_HEADS, HEAD_DIM, ROT_DIM).compile()?;
    let q_roped = rq.forward(&[&q_n_flat, &cos, &sin], vec![[HEADS * S, HEAD_DIM]])?.remove(0);
    q_roped.sync()?;
    let k_roped = rk.forward(&[&k_n_flat, &cos, &sin], vec![[KV_HEADS * S, HEAD_DIM]])?.remove(0);
    k_roped.sync()?;
    let q_final = q_roped.reshape([HEADS, S, HEAD_DIM])?;
    let k_final = k_roped.reshape([KV_HEADS, S, HEAD_DIM])?;

    // 6. Attention (q, k, v, gate) — gate is the output gate
    let att = attention_kernel(S, HEADS, KV_HEADS, HEAD_DIM).compile()?;
    let att_out = att.forward(&[&q_final, &k_final, &v, &gate_flat],
        vec![[S, HEADS * HEAD_DIM]])?.remove(0);
    att_out.sync()?;

    // Pad [S, H*D] -> [M_PAD, H*D] for o_proj gemm (R%16=0)
    let pad_attn = pad_kernel(S, M_PAD, HEADS * HEAD_DIM).compile()?;
    let att_out_p = pad_attn.forward(&[&att_out], vec![[M_PAD, HEADS * HEAD_DIM]])?.remove(0);
    att_out_p.sync()?;

    // 7. Output projection (input is [M_PAD, H*D] after pad)
    let go = gemm_kernel(M_PAD, HEADS * HEAD_DIM, HIDDEN).compile()?;
    let out_proj = go.forward(&[&att_out_p, &w_o], vec![[M_PAD, HIDDEN]])?.remove(0);
    out_proj.sync()?;

    // 8. Residual add
    let ra = residual_add_kernel().compile()?;
    let final_out = ra.forward(&[&input, &out_proj], vec![[M_PAD, HIDDEN]])?.remove(0);
    final_out.sync()?;

    // Take first S rows
    let out_full: Vec<f32> = final_out.to_vec()?;
    let s_size = (S as usize) * (HIDDEN as usize);
    let v: Vec<f32> = out_full[..s_size].to_vec();
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for i in 0..v.len() {
        max_err = max_err.max((v[i] - exp[i]).abs());
    }
    eprintln!("full_attention_block max_err {max_err}");
    assert!(max_err < 1e-2, "full_attention_block max_err {max_err}");
    Ok(())
}
