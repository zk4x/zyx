// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
const KH: i64 = 16; // key heads
const VH: i64 = 48; // value heads
const KD: i64 = 128;
const VD: i64 = 128;
const KEY_DIM: i64 = 2048; // KH * KD
const VAL_DIM: i64 = 6144; // VH * VD
const CONV_DIM: i64 = 10240; // KEY_DIM * 2 + VAL_DIM
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
    let pqkv = Linear {
        weight: goldens["in_proj_qkv"].to(dev)?,
        bias: None,
    };
    let pz = Linear {
        weight: goldens["in_proj_z"].to(dev)?,
        bias: None,
    };
    let pb = Linear {
        weight: goldens["in_proj_b"].to(dev)?,
        bias: None,
    };
    let pa = Linear {
        weight: goldens["in_proj_a"].to(dev)?,
        bias: None,
    };
    let po = Linear {
        weight: goldens["out_proj"].to(dev)?,
        bias: None,
    };
    let conv_w = goldens["conv"].to(dev)?;
    let dt_bias = goldens["dt_bias"].to(dev)?;
    let a_log = goldens["a_log"].to(dev)?;
    let norm_w = goldens["norm_weight"].to(dev)?;
    let input = goldens["input"].to(dev)?;
    let expected = goldens["output"].to_vec::<f32>()?;

    // Depthwise causal conv1d (kernel 4, left pad 3) + SiLU, truncated to S.
    let mixed = pqkv.forward(&input)?.transpose(1, 2)?;
    let xp = Tensor::cat(
        [
            &Tensor::zeros([1i64, CONV_DIM, CK - 1i64], DType::F32).to(dev)?,
            &mixed,
        ],
        2,
    )?;
    let mut conv = Tensor::zeros([1i64, CONV_DIM, S], DType::F32).to(dev)?;
    for k in 0..CK {
        let wk = conv_w
            .narrow(2, k, 1i64)?
            .reshape([CONV_DIM, 1i64])?
            .unsqueeze(0)?;
        let slice = xp.narrow(2, k, S)?;
        conv = conv + slice * wk;
    }
    let mixed = silu(&conv).narrow(2, 0i64, S)?.transpose(1, 2)?;

    let q = mixed.narrow(2, 0i64, KEY_DIM)?.reshape([1i64, S, KH, KD])?;
    let k = mixed
        .narrow(2, KEY_DIM, KEY_DIM)?
        .reshape([1i64, S, KH, KD])?;
    // Expand k/q heads to v-heads: group h serves v-heads 3h..3h+2
    // (repeat_interleave, KH=16 → VH=48).
    fn to_v_heads(x: &Tensor) -> Result<Tensor, ZyxError> {
        x.unsqueeze(3)?
            .expand([1i64, S, KH, VH / KH, KD])?
            .reshape([1i64, S, VH, KD])
    }
    let q = to_v_heads(&q)?;
    let k = to_v_heads(&k)?;
    let v = mixed
        .narrow(2, 2i64 * KEY_DIM, VAL_DIM)?
        .reshape([1i64, S, VH, VD])?;
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
    let rows = Tensor::arange(0i64, CH, 1i64)?
        .reshape([CH, 1i64])?
        .to(dev)?;
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
        attn.narrow(4, 0i64, i)?
            .narrow(3, i, 1i64)?
            .assign(&row + &corr)?;
    }
    let attn = attn + Tensor::eye(CH, DType::F32).to(dev)?;
    let value = attn.matmul(&v5)?.reshape([1i64, VH, 1i64, CH, VD])?;
    let k_cumdecay = attn.matmul(&(kb5 * g4.exp().unsqueeze(4)?))?;

    // Single chunk: recurrent state update + output.
    let qi = q5.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, KD])?;
    let ki = k5.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, KD])?;
    let vi = value.narrow(2, 0i64, 1i64)?.reshape([1i64, VH, CH, VD])?;
    let kcd = k_cumdecay
        .narrow(2, 0i64, 1i64)?
        .reshape([1i64, VH, CH, VD])?;
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
    _state = _state * g_last.exp()
        + (ki * (g_last - gc).exp().unsqueeze(3)?)
            .transpose(2, 3)?
            .matmul(&v_new)?;

    // Slice back to S, gated RMSNorm, out projection.
    let core = core.narrow(2, 0i64, S)?.transpose(1, 2)?;
    let core_f = core.reshape([S * VH, VD])?;
    let z_f = z.reshape([S * VH, VD])?;
    let var = (&core_f * &core_f).mean_keepdim([-1])?;
    let normed = &core_f * (var + Tensor::from(1e-6f32)).rsqrt() * &norm_w * silu(&z_f);
    let out = po
        .forward(normed.reshape([1i64, S, VAL_DIM])?)?
        .to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((val - exp).abs() < 1e-3, "out[{i}] = {val}, expected {exp}");
    }
    Ok(())
}

/// Same layer, but as custom kernels ONLY — no tensor ops between them:
/// pad rows -> GEMM in-projections -> conv+SiLU -> delta-rule core ->
/// gated RMSNorm -> pad rows -> out-projection GEMM. Custom kernels all
/// come from `src/lib.rs` so every test reuses them. Golden and 1e-3
/// tolerance as in the tensor test.
#[test]
fn linear_attention_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{
        conv_silu_kernel, delta_core_kernel, gemm_kernel, pad_kernel, rmsnorm_kernel, DT_RANK,
        HIDDEN, M_PAD, S, VAL_DIM, VD, VH,
    };

    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let w_qkv = goldens["in_proj_qkv"].to(dev)?.cast(DType::F16);
    let w_z = goldens["in_proj_z"].to(dev)?.cast(DType::F16);
    let w_b = goldens["in_proj_b"].to(dev)?.cast(DType::F16);
    let w_a = goldens["in_proj_a"].to(dev)?.cast(DType::F16);
    let w_o = goldens["out_proj"].to(dev)?.cast(DType::F16);
    let conv_w = goldens["conv"].to(dev)?;
    let dt_bias = goldens["dt_bias"].to(dev)?;
    let ealog = goldens["a_log"].to(dev)?.exp();
    let norm_w = goldens["norm_weight"].to(dev)?;
    let input = goldens["input"].to(dev)?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let pad_c = pad_kernel(S, M_PAD, HIDDEN).compile()?;
    let pad_n = pad_kernel(S, M_PAD, VAL_DIM).compile()?;
    let gemm_c = gemm_kernel().compile()?;
    let conv_c = conv_silu_kernel().compile()?;
    let delta_c = delta_core_kernel().compile()?;
    let norm_c = rmsnorm_kernel().compile()?;

    // GEMM launch helper: out [R, N] = A [R, K] @ B [N, K]^T via the
    // 16-row-block kernel (rows/n/k/glens are runtime variables).
    let gemm = |a: &Tensor, b: &Tensor, rows: i64, kk: i64, nn: i64| -> Result<Tensor, ZyxError> {
        let vars = [rows, kk, nn, rows / 16, nn / 8].map(Tensor::from);
        let mut out = gemm_c.forward(
            &[&vars[0], &vars[1], &vars[2], &vars[3], &vars[4], a, b],
            vec![[rows, nn]],
        )?;
        Ok(out.remove(0))
    };

    // 1. Pad input rows 6 -> 16 (f32 -> f16).
    let pin = pad_c.forward(&[&input], vec![[M_PAD, HIDDEN]])?.remove(0);
    // 2. In-projections.
    let mixed = gemm(&pin, &w_qkv, M_PAD, HIDDEN, CONV_DIM)?;
    let z = gemm(&pin, &w_z, M_PAD, HIDDEN, VAL_DIM)?;
    let b = gemm(&pin, &w_b, M_PAD, HIDDEN, DT_RANK)?;
    let a = gemm(&pin, &w_a, M_PAD, HIDDEN, DT_RANK)?;
    // 3. Depthwise causal conv + SiLU: [16, 10240].
    let mixed_s = conv_c
        .forward(&[&mixed, &conv_w], vec![[M_PAD, CONV_DIM]])?
        .remove(0);
    // 4. Delta-rule core: [48, 6, 128].
    let core = delta_c
        .forward(&[&mixed_s, &b, &a, &ealog, &dt_bias], vec![[VH, S, VD]])?
        .remove(0);
    // 5. Gated RMSNorm: [6, 6144].
    let normed = norm_c
        .forward(&[&core, &z, &norm_w], vec![[S, VAL_DIM]])?
        .remove(0);
    // 6. Pad + out projection.
    let normed_p = pad_n.forward(&[&normed], vec![[M_PAD, VAL_DIM]])?.remove(0);
    let out_full = gemm(&normed_p, &w_o, M_PAD, VAL_DIM, HIDDEN)?;
    let out = out_full.to_vec::<f32>()?;

    assert_eq!(expected.len(), (S * HIDDEN) as usize);
    let mut bad = 0;
    for t in 0..S {
        for c in 0..HIDDEN {
            let val = out[(t * M_PAD + c) as usize];
            let exp = expected[(t * HIDDEN + c) as usize];
            if (val - exp).abs() >= 1e-3 {
                if bad < 10 {
                    println!("out[{t}, {c}] = {val}, expected {exp}");
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}

/// Per-kernel torch verification: each custom kernel in `src/lib.rs` runs
/// alone on the golden intermediates and must reproduce the torch result.
/// `linear_attention_ref.py` dumps the intermediates and asserts they
/// reproduce the hf output, so these tests pin each kernel to torch.

#[test]
fn pad_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{pad_kernel, HIDDEN, M_PAD, S, VAL_DIM};
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    // 5e-3: output is f16, values ~N(0,1) round to ~2e-3 quanta.
    let input = goldens["input"].to(dev)?.reshape([S, HIDDEN])?;
    let pad_c = pad_kernel(S, M_PAD, HIDDEN).compile()?;
    let out = pad_c.forward(&[&input], vec![[M_PAD, HIDDEN]])?.remove(0);
    let out = out.cast(DType::F32).to_vec::<f32>()?;
    let expected = goldens["pin"].to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 5e-3 {
            if bad < 10 {
                println!("pad[{i}] = {val}, expected {exp}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    // Second geometry: normed rows 6 -> 16 over VAL_DIM.
    let normed = goldens["normed"].to(dev)?;
    let pad_n = pad_kernel(S, M_PAD, VAL_DIM).compile()?;
    let out = pad_n.forward(&[&normed], vec![[M_PAD, VAL_DIM]])?.remove(0);
    let out = out.cast(DType::F32).to_vec::<f32>()?;
    let zeros = Tensor::zeros([M_PAD - S, VAL_DIM], DType::F32).to(dev)?;
    let expected = Tensor::cat([&normed, &zeros], 0)?.to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 5e-3 {
            if bad < 10 {
                println!("pad_n[{i}] = {val}, expected {exp}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}

#[test]
fn gemm_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{gemm_kernel, CONV_DIM, HIDDEN, M_PAD};
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let pin16 = goldens["pin"].to(dev)?.cast(DType::F16);
    let w = goldens["in_proj_qkv"].to(dev)?.cast(DType::F16);
    let gemm_c = gemm_kernel().compile()?;
    let vars = [M_PAD, HIDDEN, CONV_DIM, M_PAD / 16, CONV_DIM / 8].map(Tensor::from);
    let out = gemm_c
        .forward(&[&vars[0], &vars[1], &vars[2], &vars[3], &vars[4], &pin16, &w], vec![[M_PAD, CONV_DIM]])?
        .remove(0);
    let out = out.to_vec::<f32>()?;
    // mixed_href: torch half-input matmul (tf32 off), the exact MMA math.
    let expected = goldens["mixed_href"].to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 1e-2 {
            if bad < 10 {
                println!("gemm[{i}] = {val}, expected {exp}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}

#[test]
fn conv_silu_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{conv_silu_kernel, CONV_DIM, M_PAD};
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let mixed = goldens["mixed"].to(dev)?;
    let conv_w = goldens["conv"].to(dev)?;
    let conv_c = conv_silu_kernel().compile()?;
    let out = conv_c
        .forward(&[&mixed, &conv_w], vec![[M_PAD, CONV_DIM]])?
        .remove(0);
    let out = out.to_vec::<f32>()?;
    let expected = goldens["mixed_s"].to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 1e-3 {
            if bad < 10 {
                println!("conv[{i}] = {val}, expected {exp}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}

#[test]
fn delta_core_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{delta_core_kernel, S, VD, VH};
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let mixed_s = goldens["mixed_s"].to(dev)?;
    let b = goldens["b"].to(dev)?;
    let a = goldens["a"].to(dev)?;
    let ea = goldens["a_log"].to(dev)?.exp();
    let dtb = goldens["dt_bias"].to(dev)?;
    let delta_c = delta_core_kernel().compile()?;
    let out = delta_c
        .forward(&[&mixed_s, &b, &a, &ea, &dtb], vec![[VH, S, VD]])?
        .remove(0);
    let out = out.to_vec::<f32>()?;
    let expected = goldens["core"].to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    // TEMP diagnostics: group mismatches by (h, t).
    let mut seen = std::collections::BTreeMap::new();
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 1e-3 {
            let h = i / (6 * 128);
            let t = (i % (6 * 128)) / 128;
            let e = seen.entry((h, t)).or_insert((0, 0.0, 0.0));
            e.0 += 1;
            e.1 += val / exp;
            println!("TEMP delta[{i}] (h{h} t{t} c{c}) = {val}, expected {exp}", c = i % 128);
            bad += 1;
        }
    }
    for ((h, t), (n, r, _)) in &seen {
        println!("TEMP (h{h} t{t}): {n} mismatches, mean ratio {r:.5}", r = r / *n as f32);
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}

#[test]
fn rmsnorm_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::{rmsnorm_kernel, S, VAL_DIM};
    let goldens = Tensor::load("../data/qwen3_8b_linear_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let core = goldens["core"].to(dev)?;
    let z = goldens["z"].to(dev)?;
    let norm_w = goldens["norm_weight"].to(dev)?;
    let norm_c = rmsnorm_kernel().compile()?;
    let out = norm_c
        .forward(&[&core, &z, &norm_w], vec![[S, VAL_DIM]])?
        .remove(0);
    let out = out.to_vec::<f32>()?;
    let expected = goldens["normed"].to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        if (val - exp).abs() >= 1e-3 {
            if bad < 10 {
                println!("norm[{i}] = {val}, expected {exp}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches");
    Ok(())
}
