// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end linear_attention block test (Qwen3.5-27B, layer 0).
//! Forward: input -> pad -> [gemm_qkv, gemm_z, gemm_ba] -> conv_silu -> delta_core ->
//! rmsnorm -> pad -> gemm_o -> output.
//! Real weights from blk.0 (linear_attention layer).

use qwen3_8_27b::{
    conv_silu_kernel, delta_core_kernel, gemm_kernel, pad_kernel, rmsnorm_kernel, CONV_DIM,
    DT_RANK, HIDDEN, M_PAD, S, VAL_DIM, VD, VH,
};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn linear_attention_block() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load(
        "/home/x/Dev/rust/zyx/examples/data/qwen3_linear_attention_block.safetensors",
    )?;
    let input = goldens["input"].to(dev)?; // [S, HIDDEN]
    let w_qkv = goldens["w_qkv"].to(dev)?; // [CONV_DIM, HIDDEN]
    let w_z = goldens["w_z"].to(dev)?; // [VAL_DIM, HIDDEN]
    let w_b = goldens["w_b"].to(dev)?; // [DT_RANK, HIDDEN]
    let w_a = goldens["w_a"].to(dev)?; // [DT_RANK, HIDDEN]
    let conv_w = goldens["conv_w"].to(dev)?; // [CONV_DIM, CK=4]
    let ss_a = goldens["ss_a"].to(dev)?; // [VH]
    let ss_dt_bias = goldens["ss_dt_bias"].to(dev)?; // [VH]
    let ss_norm_w = goldens["ss_norm_w"].to(dev)?; // [VD]
    let w_o = goldens["w_o"].to(dev)?; // [HIDDEN, VAL_DIM]
    let expected = &goldens["output"]; // [S, HIDDEN]

    // 1. Pad input [S, HIDDEN] F32 -> [M_PAD, HIDDEN] F16
    let pad_in = pad_kernel(S, M_PAD, HIDDEN).compile()?;
    let pinned = pad_in.forward(&[&input], vec![[M_PAD, HIDDEN]])?.remove(0);
    pinned.sync()?;

    // 2. QKV/Z/B/A projections (gemm takes F16, but pinned is F16)
    let qkv = gemm_kernel(M_PAD, HIDDEN, CONV_DIM).compile()?;
    let z = gemm_kernel(M_PAD, HIDDEN, VAL_DIM).compile()?;
    let ba = gemm_kernel(M_PAD, HIDDEN, DT_RANK).compile()?;

    let qkv_out = qkv
        .forward(&[&pinned, &w_qkv], vec![[M_PAD, CONV_DIM]])?
        .remove(0);
    qkv_out.sync()?;
    let z_out = z
        .forward(&[&pinned, &w_z], vec![[M_PAD, VAL_DIM]])?
        .remove(0);
    z_out.sync()?;
    let b_out = ba
        .forward(&[&pinned, &w_b], vec![[M_PAD, DT_RANK]])?
        .remove(0);
    b_out.sync()?;
    let a_out = ba
        .forward(&[&pinned, &w_a], vec![[M_PAD, DT_RANK]])?
        .remove(0);
    a_out.sync()?;

    // 3. Conv + SiLU on qkv (F32)
    let cs = conv_silu_kernel().compile()?;
    let mixed = cs
        .forward(&[&qkv_out, &conv_w], vec![[M_PAD, CONV_DIM]])?
        .remove(0);
    mixed.sync()?;

    // 4. Delta core: mixed + b + a + ealog + dtb -> [VH, S, VD] (F32)
    let dc = delta_core_kernel().compile()?;
    let core = dc
        .forward(
            &[&mixed, &b_out, &a_out, &ss_a, &ss_dt_bias],
            vec![[VH, S, VD]],
        )?
        .remove(0);
    core.sync()?;

    // 5. Gated RMSNorm: core + z + nw -> [S, VAL_DIM] (F32)
    let rn = rmsnorm_kernel().compile()?;
    let normed = rn
        .forward(&[&core, &z_out, &ss_norm_w], vec![[S, VAL_DIM]])?
        .remove(0);
    normed.sync()?;

    // 6. Pad normed [S, VAL_DIM] -> [M_PAD, VAL_DIM] F16
    let pad_norm = pad_kernel(S, M_PAD, VAL_DIM).compile()?;
    let normed_p = pad_norm
        .forward(&[&normed], vec![[M_PAD, VAL_DIM]])?
        .remove(0);
    normed_p.sync()?;

    // 7. Output projection: gemm(M_PAD, VAL_DIM, HIDDEN)
    let go = gemm_kernel(M_PAD, VAL_DIM, HIDDEN).compile()?;
    let out_padded = go
        .forward(&[&normed_p, &w_o], vec![[M_PAD, HIDDEN]])?
        .remove(0);
    out_padded.sync()?;

    // 8. Take first S rows as final output
    let out_full: Vec<f32> = out_padded.to_vec()?;
    let s_size = (S as usize) * (HIDDEN as usize);
    let v: Vec<f32> = out_full[..s_size].to_vec();
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for i in 0..v.len() {
        max_err = max_err.max((v[i] - exp[i]).abs());
    }
    eprintln!("linear_attention_block max_err {max_err}");
    assert!(max_err < 1e-2, "linear_attention_block max_err {max_err}");
    Ok(())
}
