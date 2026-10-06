// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Standalone layer-0 prefill reproduction with real GGUF weights.
//!
//! One prefill step for prompt "Hello" through embedding, input RMSNorm,
//! attention (rope, causal mask, softmax, repeat_kv) and the SwiGLU MLP,
//! mirroring `src/main.rs` of this example (including its rope argument order).
//! Prints ids plus first values and stats per stage, then asserts every stage
//! is finite and every output length matches the config shapes.
//! Currently documents the layer-0 NaN divergence seen in the full example.

use std::collections::HashMap;

use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_nn::{Linear, RMSNorm};

#[test]
fn layer0() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    let base = "/home/x/Dev/rust/zyx/examples/models/llama-3.2-3b";

    // Config, read from file (no hardcoded dims).
    let cfg_text = std::fs::read_to_string(format!("{base}/config.json")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
    let hidden = cfg["hidden_size"].as_u64().unwrap() as i64;
    let inter = cfg["intermediate_size"].as_u64().unwrap() as i64;
    let nh = cfg["num_attention_heads"].as_u64().unwrap() as i64;
    let nkv = cfg["num_key_value_heads"].as_u64().unwrap() as i64;
    let eps = cfg["rms_norm_eps"].as_f64().unwrap();
    let theta = cfg["rope_theta"].as_f64().unwrap();
    let hd = cfg["head_dim"].as_u64().unwrap() as i64;
    let max_pos = cfg["max_position_embeddings"].as_u64().unwrap() as i64;
    let max_ctx = cfg["max_context"].as_u64().unwrap_or(4096) as i64;
    let s = &cfg["rope_scaling"];
    let factor = s["factor"].as_f64().unwrap();
    let high = s["high_freq_factor"].as_f64().unwrap();
    let low = s["low_freq_factor"].as_f64().unwrap();
    let orig = s["original_max_position_embeddings"].as_u64().unwrap() as f64;

    // Prompt ids, same tokenizer call as the example.
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{base}/tokenizer.json")).unwrap();
    let ids: Vec<u32> = tokenizer.encode("Hello", true).unwrap().get_ids().to_vec();
    eprintln!("LAYER0 prompt ids={ids:?}");
    assert_eq!(ids, vec![128000u32, 9906u32], "prompt ids");
    let seqlen = ids.len() as i64;

    // GGUF weights, same file as the example. No output.weight exists here,
    // so the tied head path is the only option (matches the example).
    let (_meta, mut raw) = Tensor::load_gguf(format!("{base}/Llama-3.2-3B-Instruct-f16.gguf")).unwrap();
    let embed = raw.remove("token_embd.weight").expect("token_embd.weight");
    let an_w = raw.remove("blk.0.attn_norm.weight").expect("blk.0.attn_norm.weight");
    let qw = raw.remove("blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    let kw = raw.remove("blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    let vw = raw.remove("blk.0.attn_v.weight").expect("blk.0.attn_v.weight");
    let ow = raw.remove("blk.0.attn_output.weight").expect("blk.0.attn_output.weight");
    let pn_w = raw.remove("blk.0.ffn_norm.weight").expect("blk.0.ffn_norm.weight");
    let gw = raw.remove("blk.0.ffn_gate.weight").expect("blk.0.ffn_gate.weight");
    let uw = raw.remove("blk.0.ffn_up.weight").expect("blk.0.ffn_up.weight");
    let dw = raw.remove("blk.0.ffn_down.weight").expect("blk.0.ffn_down.weight");

    // Rope tables, same math as the example (llama3 scaling), cast to F16.
    let mut inv_freq: Vec<f32> = (0..hd)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / hd as f64) as f32)
        .collect();
    for freq in inv_freq.iter_mut() {
        let wavelen = 2.0 * std::f64::consts::PI / *freq as f64;
        if wavelen > orig / low {
            *freq = (*freq as f64 / factor) as f32;
        } else if wavelen < orig / high {
        } else {
            let smooth = (orig / wavelen - low) / (high - low);
            *freq = ((1.0 - smooth) * *freq as f64 / factor + smooth * *freq as f64) as f32;
        }
    }
    let inv_len = inv_freq.len() as i64;
    let inv = Tensor::from(inv_freq).reshape([1, inv_len]).unwrap();
    let t = Tensor::arange(0u32, max_pos as u32, 1)
        .unwrap()
        .cast(DType::F32)
        .reshape([max_pos, 1])
        .unwrap();
    let freqs = t.matmul(&inv).unwrap();
    let cos = freqs.cos().cast(DType::F16);
    let sin = freqs.sin().cast(DType::F16);

    // Modules, same construction as the example (norm scales cast to F16).
    let inorm = RMSNorm { scale: an_w.cast(DType::F16), eps };
    let post_norm = RMSNorm { scale: pn_w.cast(DType::F16), eps };
    let q_proj = Linear { weight: qw, bias: None };
    let k_proj = Linear { weight: kw, bias: None };
    let v_proj = Linear { weight: vw, bias: None };
    let o_proj = Linear { weight: ow, bias: None };
    let gate_proj = Linear { weight: gw, bias: None };
    let up_proj = Linear { weight: uw, bias: None };
    let down_proj = Linear { weight: dw, bias: None };
    let cache_k = Tensor::zeros([max_ctx, nkv, hd], DType::F16).contiguous().unwrap();
    let cache_v = Tensor::zeros([max_ctx, nkv, hd], DType::F16).contiguous().unwrap();

    // Graph on one tape, mirroring the example forward.
    let tape = Tape::empty();
    tape.add(&embed).unwrap();
    tape.add(&inorm.scale).unwrap();
    tape.add(&post_norm.scale).unwrap();
    tape.add(&q_proj.weight).unwrap();
    tape.add(&k_proj.weight).unwrap();
    tape.add(&v_proj.weight).unwrap();
    tape.add(&o_proj.weight).unwrap();
    tape.add(&cos).unwrap();
    tape.add(&sin).unwrap();
    tape.add(&cache_k).unwrap();
    tape.add(&cache_v).unwrap();
    tape.add(&gate_proj.weight).unwrap();
    tape.add(&up_proj.weight).unwrap();
    tape.add(&down_proj.weight).unwrap();

    // Embedding, one-hot gather exactly as in the example.
    let input_ids = Tensor::from(ids.clone()).unsqueeze(0).unwrap();
    let [b0, s0] = input_ids.dims::<2>().unwrap();
    let [vocab_size, embed_size] = embed.dims::<2>().unwrap();
    let idx = input_ids
        .cast(DType::F32)
        .reshape([b0, s0, 1i64.into(), 1i64.into()])
        .unwrap();
    let arange = Tensor::arange(0, vocab_size.item::<i64>(), 1)
        .unwrap()
        .reshape([1i64.into(), 1i64.into(), vocab_size.clone(), 1i64.into()])
        .unwrap()
        .cast(DType::F32);
    let wm = embed.reshape([1i64.into(), 1i64.into(), vocab_size, embed_size]).unwrap();
    let one_hot = arange.equal(idx).unwrap().cast(wm.dtype());
    let xs0 = (one_hot * wm).sum([2]).unwrap();
    // (a) table-row check: direct narrows of the two embedded rows.
    let row_bos = embed.narrow(0, 128000i64, 1i64).unwrap();
    let row_hello = embed.narrow(0, 9906i64, 1i64).unwrap();

    // Block: input norm, attention, residual, post norm, MLP, residual.
    let h = inorm.forward(&xs0).unwrap();
    let [b3, seq3, _e3] = h.dims::<3>().unwrap();
    let q = q_proj.forward(&h).unwrap();
    let k = k_proj.forward(&h).unwrap();
    let v = v_proj.forward(&h).unwrap();
    let q4 = q
        .reshape([b3.clone(), seq3.clone(), nh.into(), hd.into()])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let k4 = k
        .reshape([b3.clone(), seq3.clone(), nkv.into(), hd.into()])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let v4 = v
        .reshape([b3.clone(), seq3.clone(), nkv.into(), hd.into()])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    // Same rope call order as the example: rope(cos_narrow, sin_narrow).
    let [rseq_q, _rh] = q4.rdims::<2>().unwrap();
    let pos = Tensor::variable(0i64);
    let cqq = cos.narrow(0, &pos, &rseq_q).unwrap();
    let sqq = sin.narrow(0, &pos, &rseq_q).unwrap();
    let qr = q4.rope(cqq, sqq).unwrap();
    let [rseq_k, _rhk] = k4.rdims::<2>().unwrap();
    let ck = cos.narrow(0, &pos, &rseq_k).unwrap();
    let sk = sin.narrow(0, &pos, &rseq_k).unwrap();
    let kr = k4.rope(ck, sk).unwrap();
    // KV cache write then full read, same as the example prefill.
    let k_assign = kr.clone().squeeze([0]).transpose(0, 1).unwrap();
    let v_assign = v4.clone().squeeze([0]).transpose(0, 1).unwrap();
    cache_k.narrow(0, &pos, &seq3).unwrap().assign(&k_assign).unwrap();
    cache_v.narrow(0, &pos, &seq3).unwrap().assign(&v_assign).unwrap();
    let seq_i = seq3.item::<i64>();
    let cache_len_var = Tensor::variable(seq_i);
    let kk = cache_k
        .narrow(0, 0i64, cache_len_var.clone())
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let vv = cache_v
        .narrow(0, 0i64, cache_len_var)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    // repeat_kv, same expand+reshape as the example.
    let n_rep = (nh / nkv) as usize;
    let kk = if n_rep == 1 {
        kk
    } else {
        let [rb, rh, rs, rd] = kk.dims::<4>().unwrap();
        kk.unsqueeze(2)
            .unwrap()
            .expand([-1i64, -1, n_rep as i64, -1, -1])
            .unwrap()
            .reshape([rb.clone(), rh * (n_rep as i64), rs.clone(), rd.clone()])
            .unwrap()
    };
    let vv = if n_rep == 1 {
        vv
    } else {
        let [rb, rh, rs, rd] = vv.dims::<4>().unwrap();
        vv.unsqueeze(2)
            .unwrap()
            .expand([-1i64, -1, n_rep as i64, -1, -1])
            .unwrap()
            .reshape([rb.clone(), rh * (n_rep as i64), rs.clone(), rd.clone()])
            .unwrap()
    };
    let scale = Tensor::from(1.0f32 / (hd as f32).sqrt()).cast(qr.dtype());
    let attn = qr.matmul(kk.transpose(2, 3).unwrap()).unwrap() * scale;
    let seq_len_i = seq3.item::<i64>();
    let attn = if seq_len_i <= 1 {
        attn
    } else {
        let mask: Vec<f32> = (0..seq_len_i as usize)
            .flat_map(|i| (0..seq_len_i as usize).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from(mask).reshape([seq_len_i, seq_len_i]).unwrap().cast(attn.dtype());
        attn + mask
    };
    let probs = attn.softmax([-1]).unwrap();
    let ctx = probs.matmul(&vv).unwrap();
    let ctx = ctx.transpose(1, 2).unwrap();
    let d = nh * hd;
    let ctx = ctx.reshape([b3.clone(), seq3.clone(), d.into()]).unwrap();
    let attn_out = o_proj.forward(ctx).unwrap();
    let xs_res = attn_out.clone() + xs0.clone();
    let h2 = post_norm.forward(&xs_res).unwrap();
    let gate_raw = gate_proj.forward(&h2).unwrap();
    let gate = gate_raw.clone().swish();
    let up = up_proj.forward(&h2).unwrap();
    let down = down_proj.forward(gate.clone() * up.clone()).unwrap();
    let block_out = down.clone() + xs_res.clone();

    // Realize every stage plus the caches, then dump and check each one.
    let stages: Vec<(&str, Tensor)> = vec![
        ("row_128000", row_bos.clone()),
        ("row_9906", row_hello.clone()),
        ("embed_out", xs0.clone()),
        ("l0_input_norm", h.clone()),
        ("l0_q_rope", qr.clone()),
        ("l0_k_rope_rep", kk.clone()),
        ("l0_attn_probs", probs.clone()),
        ("l0_attn_out", attn_out.clone()),
        ("l0_post_norm", h2.clone()),
        ("l0_gate", gate_raw.clone()),
        ("l0_up", up.clone()),
        ("l0_mlp_down", down.clone()),
        ("l0_block_out", block_out.clone()),
    ];
    let lens: HashMap<&str, usize> = HashMap::from([
        ("row_128000", hidden as usize),
        ("row_9906", hidden as usize),
        ("embed_out", (seqlen * hidden) as usize),
        ("l0_input_norm", (seqlen * hidden) as usize),
        ("l0_q_rope", (seqlen * nh * hd) as usize),
        ("l0_k_rope_rep", (seqlen * nh * hd) as usize),
        ("l0_attn_probs", (nh * seqlen * seqlen) as usize),
        ("l0_attn_out", (seqlen * hidden) as usize),
        ("l0_post_norm", (seqlen * hidden) as usize),
        ("l0_gate", (seqlen * inter) as usize),
        ("l0_up", (seqlen * inter) as usize),
        ("l0_mlp_down", (seqlen * hidden) as usize),
        ("l0_block_out", (seqlen * hidden) as usize),
    ]);
    let mut realize_args: Vec<&Tensor> = Vec::new();
    for (_, t) in &stages {
        realize_args.push(t);
    }
    realize_args.push(&cache_k);
    realize_args.push(&cache_v);
    tape.realize(realize_args).unwrap();
    // (a) table-row check runs before the per-stage asserts so its result
    // prints even though a later stage currently panics on NaN.
    {
        let full: Vec<f32> = xs0.clone().cast(DType::F32).try_into().unwrap();
        let rb: Vec<f32> = row_bos.clone().cast(DType::F32).try_into().unwrap();
        let rh: Vec<f32> = row_hello.clone().cast(DType::F32).try_into().unwrap();
        let h = hidden as usize;
        let mut d0 = 0f32;
        let mut d1 = 0f32;
        for i in 0..h {
            d0 = d0.max((full[i] - rb[i]).abs());
            d1 = d1.max((full[h + i] - rh[i]).abs());
        }
        eprintln!("LAYER0 rowcheck maxdiff_bos={d0} maxdiff_hello={d1}");
        assert!(d0 < 1e-3, "gathered BOS row differs from table row: {d0}");
        assert!(d1 < 1e-3, "gathered Hello row differs from table row: {d1}");
    }
    // (b) host RMSNorm reference over the same realized embed values.
    {
        let emb: Vec<f32> = xs0.clone().cast(DType::F32).try_into().unwrap();
        let got: Vec<f32> = h.clone().cast(DType::F32).try_into().unwrap();
        let sc: Vec<f32> = inorm.scale.clone().cast(DType::F32).try_into().unwrap();
        let hh = hidden as usize;
        let mut maxd = 0f32;
        for r in 0..seqlen as usize {
            let mut ms = 0f64;
            for i in 0..hh {
                let x = emb[r * hh + i] as f64;
                ms += x * x;
            }
            ms = ms / hh as f64 + eps;
            let inv = 1.0 / ms.sqrt();
            for i in 0..hh {
                let want = emb[r * hh + i] as f64 * inv * sc[i] as f64;
                maxd = maxd.max((got[r * hh + i] as f64 - want).abs() as f32);
            }
        }
        eprintln!("LAYER0 rmsref maxdiff={maxd}");
        assert!(maxd < 5e-2, "RMSNorm output differs from host reference: {maxd}");
    }
    for (name, t) in &stages {
        let v: Vec<f32> = t.clone().cast(DType::F32).try_into().unwrap();
        assert_eq!(v.len(), lens[name], "{name} length");
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut sum = 0f64;
        for &x in &v {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
            sum += x as f64;
        }
        let mean = sum / v.len().max(1) as f64;
        let first: Vec<String> = v.iter().take(8).map(|x| format!("{x}")).collect();
        eprintln!("LAYER0 {name} len={} min={mn} max={mx} mean={mean} first8=[{}]", v.len(), first.join(" "));
        assert!(v.iter().all(|x| x.is_finite()), "{name} has NaN or inf");
    }
    Ok(())
}
