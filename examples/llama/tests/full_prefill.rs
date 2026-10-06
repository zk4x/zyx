// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Full-model prefill ("Hello") through all 28 layers + final norm + tied head.
//! Asserts finite outputs and top5 against `llama_torch.py`
//! (top5 ids=[11, 0, 5127, 323, 1070]).
//! Layer-0 stages already match torch exactly (see layer0.rs).
//! `FULL_LAYERS` env overrides the layer count for bisection.

use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_nn::{Linear, RMSNorm};

#[test]
fn full_prefill() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    let base = "/home/x/Dev/rust/zyx/examples/models/llama-3.2-3b";

    let cfg_text = std::fs::read_to_string(format!("{base}/config.json")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
    let hidden = cfg["hidden_size"].as_u64().unwrap() as i64;
    let nh = cfg["num_attention_heads"].as_u64().unwrap() as i64;
    let nkv = cfg["num_key_value_heads"].as_u64().unwrap() as i64;
    let nlayers_full = cfg["num_hidden_layers"].as_u64().unwrap();
    let nlayers: u64 = std::env::var("FULL_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(nlayers_full);
    eprintln!("FULL nlayers={nlayers}");
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
    assert!(cfg["tie_word_embeddings"].as_bool().unwrap(), "tied head");

    let tokenizer = tokenizers::Tokenizer::from_file(format!("{base}/tokenizer.json")).unwrap();
    let ids: Vec<u32> = tokenizer.encode("Hello", true).unwrap().get_ids().to_vec();
    assert_eq!(ids, vec![128000u32, 9906u32], "prompt ids");
    let seqlen = ids.len() as i64;

    let (_meta, mut raw) = Tensor::load_gguf(format!("{base}/Llama-3.2-3B-Instruct-f16.gguf")).unwrap();
    let embed = raw.remove("token_embd.weight").expect("token_embd.weight");
    let out_norm_w = raw.remove("output_norm.weight").expect("output_norm.weight");

    // Rope tables, same math as layer0.rs, cast to F16.
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

    let tape = Tape::empty();
    tape.add(&embed).unwrap();
    tape.add(&out_norm_w).unwrap();
    tape.add(&cos).unwrap();
    tape.add(&sin).unwrap();

    // Per-layer weights + caches.
    let mut layers = Vec::new();
    for i in 0..nlayers {
        let p = format!("blk.{i}.");
        let an = raw.remove(&(p.clone() + "attn_norm.weight")).unwrap();
        let qw = raw.remove(&(p.clone() + "attn_q.weight")).unwrap();
        let kw = raw.remove(&(p.clone() + "attn_k.weight")).unwrap();
        let vw = raw.remove(&(p.clone() + "attn_v.weight")).unwrap();
        let ow = raw.remove(&(p.clone() + "attn_output.weight")).unwrap();
        let pn = raw.remove(&(p.clone() + "ffn_norm.weight")).unwrap();
        let gw = raw.remove(&(p.clone() + "ffn_gate.weight")).unwrap();
        let uw = raw.remove(&(p.clone() + "ffn_up.weight")).unwrap();
        let dw = raw.remove(&(p.clone() + "ffn_down.weight")).unwrap();
        let ck = Tensor::zeros([max_ctx, nkv, hd], DType::F16).contiguous().unwrap();
        let cv = Tensor::zeros([max_ctx, nkv, hd], DType::F16).contiguous().unwrap();
        for w in [&an, &qw, &kw, &vw, &ow, &pn, &gw, &uw, &dw, &ck, &cv] {
            tape.add(w).unwrap();
        }
        layers.push((an, qw, kw, vw, ow, pn, gw, uw, dw, ck, cv));
    }

    // Embedding, one-hot gather as in layer0.rs.
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
    let mut xs = (one_hot * wm).sum([2]).unwrap();

    let pos = Tensor::variable(0i64);
    let n_rep = (nh / nkv) as usize;
    let seq_len_i = seqlen;
    let mask: Vec<f32> = (0..seq_len_i as usize)
        .flat_map(|i| (0..seq_len_i as usize).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    // Hoisted out of the layer loop: one mask/scale for all layers.
    let mask_t = Tensor::from(mask).reshape([seq_len_i, seq_len_i]).unwrap();
    let scale_t = Tensor::from(1.0f32 / (hd as f32).sqrt());
    for (an, qw, kw, vw, ow, pn, gw, uw, dw, ck, cv) in &layers {
        let inorm = RMSNorm { scale: an.clone().cast(DType::F16), eps };
        let h = inorm.forward(&xs).unwrap();
        let [b3, seq3, _e3] = h.dims::<3>().unwrap();
        let q = Linear { weight: qw.clone(), bias: None }.forward(&h).unwrap();
        let k = Linear { weight: kw.clone(), bias: None }.forward(&h).unwrap();
        let v = Linear { weight: vw.clone(), bias: None }.forward(&h).unwrap();
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
        let [rseq_q, _] = q4.rdims::<2>().unwrap();
        let qr = q4.rope(cos.narrow(0, &pos, &rseq_q).unwrap(), sin.narrow(0, &pos, &rseq_q).unwrap()).unwrap();
        let [rseq_k, _] = k4.rdims::<2>().unwrap();
        let kr = k4.rope(cos.narrow(0, &pos, &rseq_k).unwrap(), sin.narrow(0, &pos, &rseq_k).unwrap()).unwrap();
        let k_assign = kr.clone().squeeze([0]).transpose(0, 1).unwrap();
        let v_assign = v4.clone().squeeze([0]).transpose(0, 1).unwrap();
        ck.narrow(0, &pos, &seq3).unwrap().assign(&k_assign).unwrap();
        cv.narrow(0, &pos, &seq3).unwrap().assign(&v_assign).unwrap();
        // Concrete length via item(), exactly as layer0.rs: the read narrow
        // takes a Variable, never the symbolic seq dim.
        let cache_len_var = Tensor::variable(seq3.item::<i64>());
        let kk = ck
            .narrow(0, 0i64, cache_len_var.clone())
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let vv = cv
            .narrow(0, 0i64, cache_len_var)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
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
        let attn = qr.matmul(kk.transpose(2, 3).unwrap()).unwrap();
        let adt = attn.dtype();
        let attn = attn * scale_t.clone().cast(adt.clone());
        let m = mask_t.clone().cast(adt);
        let probs = (attn + m).softmax([-1]).unwrap();
        let ctx = probs.matmul(&vv).unwrap().transpose(1, 2).unwrap();
        let d = nh * hd;
        let ctx = ctx.reshape([b3.clone(), seq3.clone(), d.into()]).unwrap();
        let attn_out = Linear { weight: ow.clone(), bias: None }.forward(ctx).unwrap();
        let xs_res = attn_out.clone() + xs.clone();
        let post = RMSNorm { scale: pn.clone().cast(DType::F16), eps };
        let h2 = post.forward(&xs_res).unwrap();
        let gate = Linear { weight: gw.clone(), bias: None }.forward(&h2).unwrap().swish();
        let up = Linear { weight: uw.clone(), bias: None }.forward(&h2).unwrap();
        let down = Linear { weight: dw.clone(), bias: None }.forward(gate * up).unwrap();
        xs = down + xs_res;
    }

    let final_norm = RMSNorm { scale: out_norm_w.cast(DType::F16), eps }.forward(&xs).unwrap();
    let last = final_norm.clone().narrow(1, seqlen - 1i64, 1i64).unwrap();
    // Tied head in F32, same as torch (`last.float() @ table.float().t()`).
    let logits = last.cast(DType::F32).matmul(embed.cast(DType::F32).t()).unwrap();

    tape.realize([&final_norm, &logits]).unwrap();
    let lv: Vec<f32> = logits.clone().cast(DType::F32).try_into().unwrap();
    assert!(lv.iter().all(|x| x.is_finite()), "logits have NaN or inf");
    let mut order: Vec<usize> = (0..lv.len()).collect();
    order.sort_by(|&a, &b| lv[b].partial_cmp(&lv[a]).unwrap());
    let top5: Vec<usize> = order.iter().take(5).copied().collect();
    eprintln!("FULL top5={top5:?}");
    assert_eq!(top5, vec![11, 0, 5127, 323, 1070], "top5 mismatch vs torch");
    Ok(())
}
