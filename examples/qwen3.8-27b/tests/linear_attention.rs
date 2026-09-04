// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Linear-attention (GatedDeltaNet) reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_linear_attention.safetensors` from
//! `tests/linear_attention_ref.py`. Run the dump first:
//! `cd tests && python3.12 linear_attention_ref.py`.
//!
//! Replicates the torch fallback path: in-projections, depthwise causal
//! conv1d + SiLU, dt/beta/gate computation, the single-chunk gated delta
//! rule (seq 6 < chunk 64), gated RMSNorm, out projection.

use std::ops::Not;

use zyx::kernel::Dev;
use zyx::{DType, Tensor, ZyxError};
use zyx_nn::Linear;

const S: i64 = 6; // seq len
const KH: i64 = 2; // key heads
const VH: i64 = 2; // value heads
const KD: i64 = 8;
const VD: i64 = 8;
const KEY_DIM: i64 = 16;
const VAL_DIM: i64 = 16;
const CONV_DIM: i64 = 48;
const CK: i64 = 4; // conv kernel
const CH: i64 = 64; // chunk size

fn l2norm(x: &Tensor) -> Result<Tensor, ZyxError> {
    let d = x.rank() as i32 - 1;
    let v = (x * x).sum_keepdim([d])?;
    Ok(x * (v + Tensor::from(1e-6f32)).rsqrt())
}

fn silu(x: &Tensor) -> Tensor {
    x * x.sigmoid()
}

#[test]
fn linear_attention() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let pqkv = Linear { weight: goldens["in_proj_qkv"].to(dev)?, bias: None };
    let pz = Linear { weight: goldens["in_proj_z"].to(dev)?, bias: None };
    let pb = Linear { weight: goldens["in_proj_b"].to(dev)?, bias: None };
    let pa = Linear { weight: goldens["in_proj_a"].to(dev)?, bias: None };
    let po = Linear { weight: goldens["out_proj"].to(dev)?, bias: None };
    let conv_w = goldens["conv"].to(dev)?;
    let dt_bias = goldens["dt_bias"].to(dev)?;
    let a_log = goldens["a_log"].to(dev)?;
    let norm_w = goldens["norm_weight"].to(dev)?;
    let input = goldens["input"].to(dev)?;
    let expected = goldens["output"].to_vec::<f32>()?;

    // Depthwise causal conv1d (kernel 4, left pad 3) + SiLU, truncated to S.
    let mixed = pqkv.forward(&input)?.transpose(1, 2)?;
    let xp = Tensor::cat([&Tensor::zeros([1i64, CONV_DIM, CK - 1i64], DType::F32).to(dev)?, &mixed], 2)?;
    let mut conv = Tensor::zeros([1i64, CONV_DIM, S], DType::F32).to(dev)?;
    for k in 0..CK {
        let wk = conv_w.narrow(2, k, 1i64)?.reshape([CONV_DIM, 1i64])?.unsqueeze(0)?;
        let slice = xp.narrow(2, k, S)?;
        conv = conv + slice * wk;
    }
    let mixed = silu(&conv).narrow(2, 0i64, S)?.transpose(1, 2)?;

    let q = mixed.narrow(2, 0i64, KEY_DIM)?.reshape([1i64, S, KH, KD])?;
    let k = mixed.narrow(2, KEY_DIM, KEY_DIM)?.reshape([1i64, S, KH, KD])?;
    let v = mixed.narrow(2, 2i64 * KEY_DIM, VAL_DIM)?.reshape([1i64, S, VH, VD])?;
    let z = pz.forward(&input)?.reshape([1i64, S, VH, VD])?;
    let b = pb.forward(&input)?;
    let a = pa.forward(&input)?;

    let beta = b.sigmoid();
    let g = -(a_log.exp() * (a + dt_bias).softplus(1.0, 20.0));

    // Single-chunk gated delta rule (S < CH): pad seq to CH.
    let pad = Tensor::zeros([1i64, VH, CH - S, KD], DType::F32).to(dev)?;
    let pad_v = Tensor::zeros([1i64, VH, CH - S, VD], DType::F32).to(dev)?;
    let pad_g = Tensor::zeros([1i64, VH, CH - S], DType::F32).to(dev)?;
    let q = Tensor::cat([&l2norm(&q)?.transpose(1, 2)?, &pad], 2)? * (1.0f32 / (KD as f32).sqrt());
    let k = Tensor::cat([&l2norm(&k)?.transpose(1, 2)?, &pad], 2)?;
    let v = Tensor::cat([&v.transpose(1, 2)?, &pad_v], 2)?;
    let beta = Tensor::cat([&beta.transpose(1, 2)?, &pad_g], 2)?;
    let g = Tensor::cat([&g.transpose(1, 2)?, &pad_g], 2)?;
    let beta1 = beta.unsqueeze(3)?;

    let v_beta = &v * &beta1;
    let k_beta = &k * &beta1;
    let q5 = q.reshape([1i64, VH, 1i64, CH, KD])?;
    let k5 = k.reshape([1i64, VH, 1i64, CH, KD])?;
    let v5 = v_beta.reshape([1i64, VH, 1i64, CH, VD])?;
    let kb5 = k_beta.reshape([1i64, VH, 1i64, CH, KD])?;
    let g4 = g.reshape([1i64, VH, 1i64, CH])?.cumsum(-1)?;

    // Tril / triu masks from arange comparisons.
    let rows = Tensor::arange(0i64, CH, 1i64)?.reshape([CH, 1i64])?.to(dev)?;
    let cols = Tensor::arange(0i64, CH, 1i64)?.to(dev)?;
    let triu_strict = cols.cmpgt(&rows)?;
    let triu_incl = cols.cmplt(&rows)?.not();
    let tril_f = triu_strict.clone().not().cast(DType::F32);

    let gdiff = g4.unsqueeze(4)? - g4.unsqueeze(3)?;
    let decay = (gdiff * &tril_f).exp() * &tril_f;
    let mut attn = k5.matmul(k5.transpose(3, 4)?)?;
    attn = (&attn * &decay).to(dev)?;
    attn = triu_incl.where_(Tensor::zeros_like(&attn), &attn)?;
    // Materialize for the row-wise triangular solve: assign needs a
    // movement-only kernel, and the fused mul above disqualifies it.
    let attn = attn.contiguous()?;
    for i in 1i64..CH {
        let row = attn.narrow(4, 0i64, i)?.narrow(3, i, 1i64)?;
        let sub = attn.narrow(4, 0i64, i)?.narrow(3, 0i64, i)?;
        let corr = (&row * &sub).sum_keepdim([3])?;
        attn.narrow(4, 0i64, i)?.narrow(3, i, 1i64)?.assign(&row + &corr)?;
    }
    let attn = attn + Tensor::eye(CH, DType::F32).to(dev)?;
    let value = attn.matmul(&v5)?.reshape([1i64, VH, 1i64, CH, VD])?;
    let k_cumdecay = attn.matmul(&(kb5 * g4.exp().unsqueeze(4)?))?;

    // Single chunk: recurrent state update + output.
    let qi = q5.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, KD])?;
    let ki = k5.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, KD])?;
    let vi = value.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, VD])?;
    let kcd = k_cumdecay.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, VD])?;
    let dec = decay.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, CH])?;
    let gc = g.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH])?;
    let g_last = g4.narrow(3, CH - 1i64, 1i64)?.reshape([1i64, VH, 1i64])?;
    let mut _state = Tensor::zeros([1i64, VH, KD, VD], DType::F32).to(dev)?;

    let a2 = (qi.matmul(ki.transpose(2, 3)?)? * dec).to(dev)?;
    let a2 = triu_strict.where_(Tensor::zeros_like(&a2), &a2)?;
    let v_prime = kcd.matmul(&_state)?;
    let v_new = &vi - &v_prime;
    let ge = gc.exp().unsqueeze(3)?;
    let core = (qi * ge).matmul(&_state)? + a2.matmul(&v_new)?;
    _state = _state * g_last.exp() + (ki * (g_last - gc).exp().unsqueeze(3)?).transpose(2, 3)?.matmul(&v_new)?;

    // Slice back to S, gated RMSNorm, out projection.
    let core = core.narrow(2, 0i64, S)?.transpose(1, 2)?;
    let core_f = core.reshape([S * VH, VD])?;
    let z_f = z.reshape([S * VH, VD])?;
    let var = (&core_f * &core_f).mean_keepdim([-1])?;
    let normed = &core_f * (var + Tensor::from(1e-6f32)).rsqrt() * &norm_w * silu(&z_f);
    let out = po.forward(normed.reshape([1i64, S, VAL_DIM])?)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((val - exp).abs() < 1e-3, "out[{i}] = {val}, expected {exp}");
    }
    Ok(())
}
