// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Linear attention reference (eager Tensor chain) — will be fused by egraph to ~10 kernels.
//! Goldens from `examples/data/qwen3_8b_linear_attention.safetensors`.

use std::ops::Not;

use zyx::{DType, Dev, Tape, Tensor, ZyxError};
use zyx_nn::Linear;

fn silu(x: &Tensor) -> Tensor {
    x.clone() * x.sigmoid()
}

fn l2norm(x: &Tensor) -> Result<Tensor, ZyxError> {
    let eps = 1e-6f32;
    let var = (x.clone() * x.clone()).mean_keepdim([-1])?;
    Ok(x * (var + Tensor::from(eps)).rsqrt())
}

#[test]
//#[ignore]
fn linear_attention() -> Result<(), ZyxError> {
    // Local variables (not consts), matching qwen3.8-27b geometry.
    let s = 6i64;
    let ch = 64i64;
    let ck = 4i64;
    //let hidden = 5120i64;
    let conv_dim = 10240i64;
    let key_dim = 2048i64;
    let val_dim = 6144i64;
    let kh = 16i64;
    let kd = 128i64;
    let vh = 48i64;
    let vd = 128i64;

    let goldens =
        Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_8b_linear_attention.safetensors")?;
    let pqkv = Linear {
        weight: goldens["in_proj_qkv"].clone(),
        bias: None,
    };
    let pz = Linear {
        weight: goldens["in_proj_z"].clone(),
        bias: None,
    };
    let pb = Linear {
        weight: goldens["in_proj_b"].clone(),
        bias: None,
    };
    let pa = Linear {
        weight: goldens["in_proj_a"].clone(),
        bias: None,
    };
    let po = Linear {
        weight: goldens["out_proj"].clone(),
        bias: None,
    };
    let conv_w = goldens["conv"].clone();
    let dt_bias = goldens["dt_bias"].clone();
    let a_log = goldens["a_log"].clone();
    let norm_w = goldens["norm_weight"].clone();
    let input = goldens["input"].clone();
    let expected = goldens["output"].to_vec::<f32>()?;

    let tape = Tape::new([
        &input,
        &pqkv.weight,
        &pz.weight,
        &pb.weight,
        &pa.weight,
        &po.weight,
        &conv_w,
        &dt_bias,
        &a_log,
        &norm_w,
    ])?;

    // Depthwise causal conv1d (kernel 4, left pad 3) + SiLU, truncated to S.
    let mixed = pqkv.forward(&input)?.transpose(1, 2)?;
    let xp = Tensor::cat(
        [
            &Tensor::zeros([1i64, conv_dim, ck - 1i64], DType::F32),
            &mixed,
        ],
        2,
    )?;
    let mut conv = Tensor::zeros([1i64, conv_dim, s], DType::F32);
    for k in 0..ck {
        let wk = conv_w
            .narrow(2, k, 1i64)?
            .reshape([conv_dim, 1i64])?
            .unsqueeze(0)?;
        let slice = xp.narrow(2, k, s)?;
        conv = conv + slice * wk;
    }
    let mixed = silu(&conv).narrow(2, 0i64, s)?.transpose(1, 2)?;

    let q = mixed.narrow(2, 0i64, key_dim)?.reshape([1i64, s, kh, kd])?;
    let k = mixed
        .narrow(2, key_dim, key_dim)?
        .reshape([1i64, s, kh, kd])?;
    fn to_v_heads(x: &Tensor, s: i64, kh: i64, vh: i64, kd: i64) -> Result<Tensor, ZyxError> {
        x.unsqueeze(3)?
            .expand([1i64, s, kh, vh / kh, kd])?
            .reshape([1i64, s, vh, kd])
    }
    let q = to_v_heads(&q, s, kh, vh, kd)?;
    let k = to_v_heads(&k, s, kh, vh, kd)?;
    let v = mixed
        .narrow(2, 2i64 * key_dim, val_dim)?
        .reshape([1i64, s, vh, vd])?;
    let z = pz.forward(&input)?.reshape([1i64, s, vh, vd])?;
    let b = pb.forward(&input)?;
    let a = pa.forward(&input)?;

    let beta = b.sigmoid();
    let g = -(a_log.exp() * (a + dt_bias).softplus(1.0, 20.0));

    // Single-chunk gated delta rule (S < CH): pad seq to CH.
    let pad = Tensor::zeros([1i64, vh, ch - s, kd], DType::F32);
    let pad_v = Tensor::zeros([1i64, vh, ch - s, vd], DType::F32);
    let pad_g = Tensor::zeros([1i64, vh, ch - s], DType::F32);
    let q = Tensor::cat([&l2norm(&q)?.transpose(1, 2)?, &pad], 2)? * (1.0f32 / (kd as f32).sqrt());
    let k = Tensor::cat([&l2norm(&k)?.transpose(1, 2)?, &pad], 2)?;
    let v = Tensor::cat([&v.transpose(1, 2)?, &pad_v], 2)?;
    let beta = Tensor::cat([&beta.transpose(1, 2)?, &pad_g], 2)?;
    let g = Tensor::cat([&g.transpose(1, 2)?, &pad_g], 2)?;
    let beta1 = beta.unsqueeze(3)?;

    let v_beta = &v * &beta1;
    let k_beta = &k * &beta1;
    let q5 = q.reshape([1i64, vh, 1i64, ch, kd])?;
    let k5 = k.reshape([1i64, vh, 1i64, ch, kd])?;
    let v5 = v_beta.reshape([1i64, vh, 1i64, ch, vd])?;
    let kb5 = k_beta.reshape([1i64, vh, 1i64, ch, kd])?;
    let g4 = g.reshape([1i64, vh, 1i64, ch])?.cumsum(-1)?;

    let rows = Tensor::arange(0i64, ch, 1i64)?.reshape([ch, 1i64])?;
    let cols = Tensor::arange(0i64, ch, 1i64)?;
    let triu_strict = cols.cmpgt(&rows)?;
    let triu_incl = cols.cmplt(&rows)?.not();
    let tril_f = triu_strict.clone().not().cast(DType::F32);

    let gdiff = g4.unsqueeze(4)? - g4.unsqueeze(3)?;
    let decay = (gdiff * &tril_f).exp() * &tril_f;
    let mut attn = k5.matmul(k5.transpose(3, 4)?)?;
    attn = &attn * &decay;
    attn = triu_incl.where_(Tensor::zeros_like(&attn), &attn)?;

    let eye = Tensor::eye(ch, DType::F32);
    let attn = attn + eye;
    let value = attn.matmul(&v5)?.reshape([1i64, vh, 1i64, ch, vd])?;
    let k_cumdecay = attn.matmul(&(kb5 * g4.exp().unsqueeze(4)?))?;

    let qi = q5.narrow(2, 0i64, 1i64)?.reshape([1i64, vh, ch, kd])?;
    let ki = k5.narrow(2, 0i64, 1i64)?.reshape([1i64, vh, ch, kd])?;
    let vi = value.narrow(2, 0i64, 1i64)?.reshape([1i64, vh, ch, vd])?;
    let kcd = k_cumdecay
        .narrow(2, 0i64, 1i64)?
        .reshape([1i64, vh, ch, vd])?;
    let dec = decay.narrow(2, 0i64, 1i64)?.reshape([1i64, vh, ch, ch])?;
    let gc = g.clone();
    let g_last = g4.narrow(3, ch - 1i64, 1i64)?.reshape([1i64, vh, 1i64])?;
    let g_last_expanded = g_last.reshape([1i64, vh, 1i64, 1i64])?;
    let mut _state = Tensor::zeros([1i64, vh, kd, vd], DType::F32);

    let a2 = qi.matmul(ki.transpose(2, 3)?)? * dec;
    let a2 = triu_strict.where_(Tensor::zeros_like(&a2), &a2)?;
    let v_prime = kcd.matmul(&_state)?;
    let v_new = &vi - &v_prime;
    let ge = gc.exp().unsqueeze(3)?;
    let core = (qi * ge).matmul(&_state)? + a2.matmul(&v_new)?;
    _state = _state * g_last_expanded.exp()
        + (ki * (g_last - gc).exp().unsqueeze(3)?)
            .transpose(2, 3)?
            .matmul(&v_new)?;

    let core = core.narrow(2, 0i64, s)?.transpose(1, 2)?;
    let core_f = core.reshape([s * vh, vd])?;
    let z_f = z.reshape([s * vh, vd])?;
    let var = (&core_f * &core_f).mean_keepdim([-1])?;
    let normed = &core_f * (var + Tensor::from(1e-6f32)).rsqrt() * &norm_w * silu(&z_f);
    let out_t = po.forward(normed.reshape([1i64, s, val_dim])?)?;

    tape.realize([&out_t])?;

    let out = out_t.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((val - exp).abs() < 1e-3, "out[{i}] = {val}, expected {exp}");
    }
    Ok(())
}
