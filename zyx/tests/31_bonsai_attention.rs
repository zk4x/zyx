// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B full-attention core: partial RoPE, the F16 K/V cache
//! write, the GQA score/softmax/value path at the checkpoint's own geometry
//! (24 heads / 4 kv heads / head_dim 256 / rot_dim 64, cache 4096 rows).
//!
//! The sequence is `examples/bonsai-27b/src/model.rs`'s attention core, and
//! every test measures ONE of its kernels through the graph the model builds,
//! so the autotune search (and its device-side timing) is the measurement.
//! Decode is one token at `pos = 4095`; prefill is 128 tokens at `pos = 0`
//! with the causal mask.
//!
//! ```text
//! ZYX_DEV=cuda ZYX_NO_CACHE=1 cargo test --test 31_bonsai_attention -- --nocapture
//! ```
//!
//! `ZYX_DEBUG=8` prints the winning IR of each kernel, `ZYX_DEBUG=256` the
//! no-search naive IR.

use std::sync::{Arc, Mutex};

use zyx::kernel::autotune::{Measurement, set_measurement_recorder};
use zyx::{DType, Tensor, ZyxError};

/// Full-attention heads.
const N_HEAD: i64 = 24;
/// Full-attention key/value heads.
const N_KV: i64 = 4;
/// Full-attention head width.
const HEAD_DIM: i64 = 256;
/// Rotated dimensions per head (partial RoPE).
const ROT_DIM: i64 = 64;
/// `n_head / n_kv`.
const REP: i64 = N_HEAD / N_KV;
/// RoPE base frequency.
const ROPE_THETA: f64 = 10_000_000.0;
/// Cache rows (`max_context`).
const MAX_CONTEXT: i64 = 4_096;
/// Decode position: the last cache slot.
const POS: i64 = MAX_CONTEXT - 1;
/// Prefill width.
const SEQ: i64 = 128;
/// `n_head * head_dim`.
const Q_DIM: i64 = N_HEAD * HEAD_DIM;

/// Deterministic values in `[-1, 1)`.
fn seq_vals(n: i64, seed: i64) -> Vec<f32> {
    (0..n).map(|i| (((i * 17 + seed * 29) % 23) as f32 - 11.0) / 11.0).collect()
}

/// The same values as F16, so the operand is a leaf with no cast kernel in
/// front of the kernel under test.
fn f16_vals(n: i64, seed: i64) -> Vec<zyx::f16> {
    seq_vals(n, seed).into_iter().map(zyx::f16::from_f32).collect()
}

/// llama.cpp's interleaved RoPE tables: `[positions, rot_dim/2]`.
fn rope_tables(positions: i64) -> (Vec<f32>, Vec<f32>) {
    let half = ROT_DIM / 2;
    let mut cos = vec![0.0f32; (positions * half) as usize];
    let mut sin = vec![0.0f32; (positions * half) as usize];
    for pos in 0..positions {
        for pair in 0..half {
            let angle = pos as f64 * ROPE_THETA.powf(-2.0 * pair as f64 / ROT_DIM as f64);
            let (s, c) = angle.sin_cos();
            cos[(pos * half + pair) as usize] = c as f32;
            sin[(pos * half + pair) as usize] = s as f32;
        }
    }
    (cos, sin)
}

/// The GPU this class is measured on, or `None` on a CPU-only box.
fn dev() -> Option<zyx::Dev> {
    zyx::Dev::all().into_iter().find(|d| !matches!(d, zyx::Dev::C | zyx::Dev::Host))
}

/// Collects the autotune search's device-side measurements while it runs.
struct Recorder {
    rows: Arc<Mutex<Vec<(String, u64, String)>>>,
}

impl Recorder {
    fn install() -> Self {
        let rows: Arc<Mutex<Vec<(String, u64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&rows);
        set_measurement_recorder(Some(Box::new(move |m: &Measurement| {
            if let Some(nanos) = m.nanos {
                let grid = match m.grid {
                    Some(g) => format!("{g:?}"),
                    None => "?".into(),
                };
                r.lock().unwrap().push((format!("{:016x}", m.kernel_hash), nanos, grid));
            }
        })));
        Self { rows }
    }

    /// The best measured time per kernel, slowest first.
    fn report(&self, what: &str) {
        set_measurement_recorder(None);
        let rows = self.rows.lock().unwrap();
        let mut best: std::collections::BTreeMap<String, (u64, usize, String)> = Default::default();
        for (h, n, g) in rows.iter() {
            let e = best.entry(h.clone()).or_insert((u64::MAX, 0, g.clone()));
            e.0 = e.0.min(*n);
            e.1 += 1;
        }
        let mut list: Vec<_> = best.into_iter().collect();
        list.sort_by_key(|(_, (t, _, _))| std::cmp::Reverse(*t));
        for (h, (t, c, g)) in list {
            println!("{what}: kernel {h} = {:.4} ms best of {c} candidates, grid {g}", t as f64 * 1e-6);
        }
    }
}

/// `q @ k^T` at the head layout `[heads, seq, head_dim] x [heads, cache,
/// head_dim]`: the GQA scores, `heads = 24`, `kv = 4`.
fn scores(q: &Tensor, k: &Tensor) -> Result<Tensor, ZyxError> {
    let scale = Tensor::from(1.0f32 / (HEAD_DIM as f32).sqrt());
    Ok(q.cast(DType::F16).dot_dtype_output(k.transpose(1, 2)?.cast(DType::F16), DType::F32, DType::F32)? * scale)
}

/// The GQA repeat `[kv, cache, hd] -> [heads, cache, hd]`, cache length
/// symbolic so one kernel serves every position.
fn repeat_kv(x: &Tensor, cache_len: &Tensor) -> Result<Tensor, ZyxError> {
    x.unsqueeze(1)?.expand([-1i64, REP, -1, -1])?.reshape([
        Tensor::from(N_HEAD),
        cache_len.clone(),
        Tensor::from(HEAD_DIM),
    ])
}

#[test]
fn scores_decode() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let q = Tensor::from_vec(seq_vals(Q_DIM, 3), [N_HEAD, 1, HEAD_DIM])?.to(dev)?;
    let k = Tensor::from_vec(f16_vals(MAX_CONTEXT * N_KV * HEAD_DIM, 5), [MAX_CONTEXT, N_KV, HEAD_DIM])?.to(dev)?;
    let cache_len = Tensor::variable(MAX_CONTEXT);
    let kk = repeat_kv(&k.transpose(0, 1)?, &cache_len)?;
    let s = scores(&q, &kk)?;
    let v: Vec<f32> = s.try_into()?;
    std::hint::black_box(v.len());
    rec.report("scores_decode");
    Ok(())
}

#[test]
fn scores_prefill() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let q = Tensor::from_vec(seq_vals(SEQ * Q_DIM, 3), [N_HEAD, SEQ, HEAD_DIM])?.to(dev)?;
    let k = Tensor::from_vec(f16_vals(SEQ * N_KV * HEAD_DIM, 5), [SEQ, N_KV, HEAD_DIM])?.to(dev)?;
    let cache_len = Tensor::variable(SEQ);
    let kk = repeat_kv(&k.transpose(0, 1)?, &cache_len)?;
    let s = scores(&q, &kk)?;
    let v: Vec<f32> = s.try_into()?;
    std::hint::black_box(v.len());
    rec.report("scores_prefill");
    Ok(())
}

/// Scalar f32 PV reference, llama.cpp's numeric form: the f16 operands are
/// widened before the multiply (`ggml_cuda_mad` is an f32 FFMA) and the
/// products accumulate in f32, in increasing cache order.
///
/// `p` is `[N_HEAD, rows, cache]`, `v` is `[cache, N_KV, HEAD_DIM]`, the
/// result `[N_HEAD, rows, HEAD_DIM]`.
fn pv_ref(p: &[zyx::f16], v: &[zyx::f16], rows: i64, cache: i64) -> Vec<f32> {
    let mut out = vec![0.0f32; usize::try_from(N_HEAD * rows * HEAD_DIM).expect("output fits usize")];
    for h in 0..N_HEAD {
        let kv = h / REP;
        for q in 0..rows {
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..cache {
                    let pp = p[usize::try_from((h * rows + q) * cache + j).expect("p index fits")].to_f32();
                    let vv = v[usize::try_from((j * N_KV + kv) * HEAD_DIM + d).expect("v index fits")].to_f32();
                    acc += pp * vv;
                }
                out[usize::try_from((h * rows + q) * HEAD_DIM + d).expect("out index fits")] = acc;
            }
        }
    }
    out
}

/// Every PV value against [`pv_ref`]; the graph's summation order is the
/// search's, so the tolerance is the f32 sum's own noise, not the MAC's.
fn assert_pv(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        worst = worst.max(d);
        assert!(d <= 1e-3f32.max(1e-3 * w.abs()), "{what}[{i}]: {g} != {w} (|err| {d})");
    }
    println!("{what}: {} values, max |err| vs f32 reference {worst:.3e}", got.len());
}

#[test]
fn pv_decode() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let p_host = f16_vals(N_HEAD * MAX_CONTEXT, 7);
    let v_host = f16_vals(MAX_CONTEXT * N_KV * HEAD_DIM, 9);
    let p = Tensor::from_vec(p_host.clone(), [N_HEAD, 1, MAX_CONTEXT])?.to(dev)?;
    let v = Tensor::from_vec(v_host.clone(), [MAX_CONTEXT, N_KV, HEAD_DIM])?.to(dev)?;
    let cache_len = Tensor::variable(MAX_CONTEXT);
    // The model narrows the cache to `cache_len` before the repeat, so the
    // expand's axes and the reshape's are the same symbolic dims.
    let v = v.narrow(0, 0i64, cache_len.clone())?;
    let vv = repeat_kv(&v.transpose(0, 1)?, &cache_len)?;
    let out = p.cast(DType::F32).dot_dtype_output(&vv.cast(DType::F32), DType::F32, DType::F32)?;
    let vals: Vec<f32> = out.try_into()?;
    let want = pv_ref(&p_host, &v_host, 1, MAX_CONTEXT);
    assert_pv(&vals, &want, "pv_decode graph");
    rec.report("pv_decode");
    Ok(())
}

#[test]
fn pv_prefill() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let p_host = f16_vals(N_HEAD * SEQ * SEQ, 7);
    let v_host = f16_vals(SEQ * N_KV * HEAD_DIM, 9);
    let p = Tensor::from_vec(p_host.clone(), [N_HEAD, SEQ, SEQ])?.to(dev)?;
    let v = Tensor::from_vec(v_host.clone(), [SEQ, N_KV, HEAD_DIM])?.to(dev)?;
    let cache_len = Tensor::variable(SEQ);
    let v = v.narrow(0, 0i64, cache_len.clone())?;
    let vv = repeat_kv(&v.transpose(0, 1)?, &cache_len)?;
    let out = p.cast(DType::F32).dot_dtype_output(&vv.cast(DType::F32), DType::F32, DType::F32)?;
    let vals: Vec<f32> = out.try_into()?;
    let want = pv_ref(&p_host, &v_host, SEQ, SEQ);
    assert_pv(&vals, &want, "pv_prefill graph");
    rec.report("pv_prefill");
    Ok(())
}

#[test]
fn softmax_decode() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let s = Tensor::from_vec(seq_vals(N_HEAD * MAX_CONTEXT, 11), [N_HEAD, 1, MAX_CONTEXT])?.to(dev)?;
    let p = s.softmax([-1])?;
    let vals: Vec<f32> = p.try_into()?;
    std::hint::black_box(vals.len());
    rec.report("softmax_decode");
    Ok(())
}

#[test]
fn softmax_prefill() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let s = Tensor::from_vec(seq_vals(N_HEAD * SEQ * SEQ, 11), [N_HEAD, SEQ, SEQ])?.to(dev)?;
    let p = s.softmax([-1])?;
    let vals: Vec<f32> = p.try_into()?;
    std::hint::black_box(vals.len());
    rec.report("softmax_prefill");
    Ok(())
}

#[test]
fn rope_decode() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let (cos_v, sin_v) = rope_tables(MAX_CONTEXT);
    let cos = Tensor::from_vec(cos_v, [MAX_CONTEXT, ROT_DIM / 2])?.to(dev)?;
    let sin = Tensor::from_vec(sin_v, [MAX_CONTEXT, ROT_DIM / 2])?.to(dev)?;
    let q = Tensor::from_vec(seq_vals(N_HEAD * HEAD_DIM, 3), [1, N_HEAD, HEAD_DIM])?.to(dev)?;
    let k = Tensor::from_vec(seq_vals(N_KV * HEAD_DIM, 5), [1, N_KV, HEAD_DIM])?.to(dev)?;
    let pos = Tensor::variable(POS);
    for x in [&q, &k] {
        let x = x.transpose(0, 1)?.unsqueeze(0)?;
        let rotated = x.narrow(3, 0i64, ROT_DIM)?;
        let rest = x.narrow(3, ROT_DIM, HEAD_DIM - ROT_DIM)?;
        let c = cos.narrow(0, pos.clone(), 1i64)?;
        let s = sin.narrow(0, pos.clone(), 1i64)?;
        let rotated = rotated.rope(&c, &s)?;
        let out = Tensor::cat([&rotated, &rest], 3)?.squeeze([0]).transpose(0, 1)?;
        let vals: Vec<f32> = out.cast(DType::F32).try_into()?;
        std::hint::black_box(vals.len());
    }
    rec.report("rope_decode");
    Ok(())
}

#[test]
fn kv_write_decode() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    let rec = Recorder::install();
    let cache_k = Tensor::zeros([MAX_CONTEXT, N_KV, HEAD_DIM], DType::F16).to(dev)?;
    let cache_v = Tensor::zeros([MAX_CONTEXT, N_KV, HEAD_DIM], DType::F16).to(dev)?;
    let k = Tensor::from_vec(f16_vals(N_KV * HEAD_DIM, 5), [1, N_KV, HEAD_DIM])?.to(dev)?;
    let v = Tensor::from_vec(f16_vals(N_KV * HEAD_DIM, 7), [1, N_KV, HEAD_DIM])?.to(dev)?;
    let pos = Tensor::variable(POS);
    let real = Tensor::variable(1i64);
    cache_k.clone().narrow(0, pos.clone(), real.clone())?.assign(&k.cast(DType::F16))?;
    cache_v.clone().narrow(0, pos.clone(), real.clone())?.assign(&v.cast(DType::F16))?;
    let written: Vec<f32> = cache_k.cast(DType::F32).try_into()?;
    std::hint::black_box(written.len());
    rec.report("kv_write_decode");
    Ok(())
}

// ===========================================================================
// The slowest sub-kernel: PV at decode, `p[24, 1, 4096] @ v[24, 4096, 256]`
// ===========================================================================
//
// `ZYX_DEBUG=8` on `pv_decode` gives the seed the linearizer emits (the
// comment block below). Every output element is one thread:
//
// ```text
//  r0: i64 = 24                       // group_range(0) over 24 heads
//  r7: i64 = 256                      // group_range(1) over the head dim
//  for r77 in 0..cache_len {          // the whole cache, serial per thread
//    r111 = (head*256 + d + j) % (256*cache_len)   // GQA repeat index
//    r221: f16 = load v[...]                         // 2 KiB apart per step
//    r254: f16 = load p[head*cache_len + j]
//    r27: f32 = f32(r254) * f32(r221)                 // f32 MAC
//    r148: f32 += r27
//  }
//  store out[head*256 + d]
// ```
//
// 6144 threads, each summing 4096 products serially, each V load a separate
// 2-byte strided access, no vectorization. The graph's GQA repeat used to
// cost 15.06 ms: the reshape lowered to a flat-index recovery whose divisors
// were the symbolic strides `c*s` and `c` (117 ops, 12 div/mod). With
// `linearize` regrouping the reshape axes the access is
// `d + (h/rep)*256 + j*1024` (45 ops, one div by the constant `rep`) and the
// same graph is 0.46 ms; the hand-written direct-index loop
// (`jv = j*1024 + kv*256 + d`, `pv_naive`) is 3.7 ms, so the search now
// splits the clean index instead of fighting it. llama.cpp's `-fa 0` path
// runs the same
// reduction as `mul_mat_vec_f<__half, __half, 1, 128, false, false>`: one
// 128-thread block per (head, dst row), grid (256, 24), each thread striding
// the cache with half2 V loads and float2 probability loads, then a warp
// shuffle reduce and one cross-warp stage through shared memory.
//
// Our V operand is `[cache, kv, head_dim]` (head_dim contiguous), so the
// coalesced axis is `d`, not the cache. `pv_fast` is one block per
// (head, d/32) with 256 threads = 8 j-warps x 32 d-lanes (0.17 ms);
// `pv_fast_kv` is the same with the six heads of one KV head in the block, so
// each V element is loaded once and reused from a register (0.093 ms). Both
// accumulate in f32 (`ggml_cuda_mad` is a f32 FFMA) and cross the j-warps
// through `MemScope::Local`.

/// The seed PV kernel, the dumped `ZYX_DEBUG=8` shape: one thread per output
/// element, one f32 accumulator, one serial loop over the cache, the f32 MAC
/// llama.cpp's `ggml_cuda_mad` performs.
fn pv_naive(dev: zyx::Dev) -> zyx::kernel::Kernel {
    use zyx::kernel::{Kernel, MemScope};

    let mut k = Kernel::new(dev);
    let v = k.param(DType::F16);
    let p = k.param(DType::F16);
    let n = k.variable(DType::I64);
    let out = k.param_mut(DType::F32);
    let head = k.group_range(0, N_HEAD);
    let d = k.group_range(1, HEAD_DIM);
    let kv = k.div(head, REP);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    let vhd = k.const_val(N_KV * HEAD_DIM);
    let hd = k.const_val(HEAD_DIM);
    let vbase = k.mul(kv, hd);
    k.loop_over(n, |k, j| {
        let vrow = k.mad(j, vhd, vbase);
        let vidx = k.add(vrow, d);
        let v_s = k.load(v, vidx);
        let vv = k.cast(v_s, DType::F32);
        let pidx = k.mad(head, n, j);
        let p_s = k.load(p, pidx);
        let pp = k.cast(p_s, DType::F32);
        let prod = k.mul(pp, vv);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    let res = k.load(acc, 0i64);
    let oidx = k.mad(head, hd, d);
    k.store(out, res, oidx);
    k.verify();
    k
}

/// The fast PV kernel: `grid (heads, head_dim/32)`, `256` threads per block
/// (`8` j-warps of `32` d-lanes). Each thread accumulates one output
/// element's every eighth cache row with a f32 MAC; the warps' partials are
/// summed through local memory by the `jw == 0` warp.
fn pv_fast(dev: zyx::Dev) -> zyx::kernel::Kernel {
    use zyx::kernel::{Kernel, MemScope};

    const JW: i64 = 8;
    const DW: i64 = 32;
    let threads = u32::try_from(JW * DW).expect("pv_fast block fits u32");
    let mut k = Kernel::new(dev);
    let v = k.param(DType::F16);
    let p = k.param(DType::F16);
    let n = k.variable(DType::I64);
    let out = k.param_mut(DType::F32);
    // `chunk` on axis 0 so the eight 32-wide slices of one head run first:
    // their whole V slice (2 MiB) stays in L2 for the six heads that share
    // a KV head, so V crosses DRAM once instead of once per head.
    let chunk = k.group_range(0, HEAD_DIM / DW);
    let head = k.group_range(1, N_HEAD);
    let tid = k.local_range(0, threads);
    let lane = k.mod_(tid, DW);
    let jw = k.div(tid, DW);
    let chunk_off = k.mul(chunk, DW);
    let d = k.add(chunk_off, lane);
    let kv = k.div(head, REP);
    let hd = k.const_val(HEAD_DIM);
    let vhd = k.const_val(N_KV * HEAD_DIM);
    let vbase = k.mul(kv, hd);
    let zero = k.const_val(0.0f32);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    k.store(acc, zero, 0i64);
    // ceil(n / JW): the cache length is symbolic and not a multiple of 8 at
    // every decode position, so the last step masks the out-of-range j.
    let jw_last = k.const_idx(JW - 1);
    let n_pad = k.add(n, jw_last);
    let jw_const = k.const_idx(JW);
    let steps = k.div(n_pad, jw_const);
    k.loop_over(steps, |k, it| {
        let step = k.mul(it, JW);
        let j = k.add(step, jw);
        let in_range = k.cmplt(j, n);
        k.if_over(in_range, |k| {
            let vrow = k.mad(j, vhd, vbase);
            let vidx = k.add(vrow, d);
            let v_s = k.load(v, vidx);
            let vv = k.cast(v_s, DType::F32);
            let pidx = k.mad(head, n, j);
            let p_s = k.load(p, pidx);
            let pp = k.cast(p_s, DType::F32);
            let prod = k.mul(pp, vv);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, prod);
            k.store(acc, sum, 0i64);
        });
    });
    let smem = k.storage(DType::F32, MemScope::Local, i64::from(threads));
    let acc_v = k.load(acc, 0i64);
    k.store(smem, acc_v, tid);
    k.barrier();
    let zero_idx = k.const_idx(0i64);
    let first = k.equal(jw, zero_idx);
    k.if_over(first, |k| {
        let mut total = k.load(smem, lane);
        let mut w = 1i64;
        while w < JW {
            let woff = k.const_idx(w * DW);
            let at = k.add(woff, lane);
            let part = k.load(smem, at);
            total = k.add(total, part);
            w += 1;
        }
        let oidx = k.mad(head, hd, d);
        k.store(out, total, oidx);
    });
    k.verify();
    k
}

/// The same shape with the six heads of one KV head in one block: the grid
/// is `(head_dim/32, kv_heads)`, so a block loads each V element once and
/// reuses it for all six probabilities. V then crosses DRAM exactly once
/// (8.4 MiB), where [`pv_fast`]'s grid order has to keep the per-KV slice in
/// L2 to do the same.
fn pv_fast_kv(dev: zyx::Dev) -> zyx::kernel::Kernel {
    use zyx::kernel::{Kernel, MemScope};

    // One warp covers a full 32-wide d slice, so each V load is one 64-byte
    // transaction; 32 blocks (8 d-slices x 4 kv heads) x 1024 threads fill
    // the 30 SMs for one block each.
    const JW: i64 = 32;
    const DW: i64 = 32;
    let threads = u32::try_from(JW * DW).expect("pv_fast_kv block fits u32");
    let mut k = Kernel::new(dev);
    let v = k.param(DType::F16);
    let p = k.param(DType::F16);
    let n = k.variable(DType::I64);
    let out = k.param_mut(DType::F32);
    let chunk = k.group_range(0, HEAD_DIM / DW);
    let kv = k.group_range(1, N_KV);
    let tid = k.local_range(0, threads);
    let lane = k.mod_(tid, DW);
    let jw = k.div(tid, DW);
    let chunk_off = k.mul(chunk, DW);
    let d = k.add(chunk_off, lane);
    let hd = k.const_val(HEAD_DIM);
    let vhd = k.const_val(N_KV * HEAD_DIM);
    let vbase = k.mul(kv, hd);
    let head0 = k.mul(kv, REP);
    let zero = k.const_val(0.0f32);
    let regs = [0i64, 1, 2, 3, 4, 5];
    assert_eq!(regs.len() as i64, REP, "pv_fast_kv: one accumulator per repeated head");
    let acc: Vec<_> = regs
        .iter()
        .map(|_| {
            let a = k.storage(DType::F32, MemScope::Register, 1);
            k.store(a, zero, 0i64);
            a
        })
        .collect();
    let jw_last = k.const_idx(JW - 1);
    let n_pad = k.add(n, jw_last);
    let jw_const = k.const_idx(JW);
    let steps = k.div(n_pad, jw_const);
    k.loop_over(steps, |k, it| {
        let step = k.mul(it, JW);
        let j = k.add(step, jw);
        let in_range = k.cmplt(j, n);
        k.if_over(in_range, |k| {
            let vrow = k.mad(j, vhd, vbase);
            let vidx = k.add(vrow, d);
            let v_s = k.load(v, vidx);
            let vv = k.cast(v_s, DType::F32);
            for (h, &a) in acc.iter().enumerate() {
                let hc = k.const_idx(i64::try_from(h).expect("head fits i64"));
                let head = k.add(head0, hc);
                let pidx = k.mad(head, n, j);
                let p_s = k.load(p, pidx);
                let pp = k.cast(p_s, DType::F32);
                let prod = k.mul(pp, vv);
                let prev = k.load(a, 0i64);
                let sum = k.add(prev, prod);
                k.store(a, sum, 0i64);
            }
        });
    });
    let smem = k.storage(DType::F32, MemScope::Local, REP * i64::from(threads));
    for (h, &a) in acc.iter().enumerate() {
        let hc = k.const_idx(i64::try_from(h).expect("head fits i64") * i64::from(threads));
        let at = k.add(hc, tid);
        let val = k.load(a, 0i64);
        k.store(smem, val, at);
    }
    k.barrier();
    let zero_idx = k.const_idx(0i64);
    let first = k.equal(jw, zero_idx);
    k.if_over(first, |k| {
        for h in 0..REP {
            let hoff = k.const_idx(h * i64::from(threads));
            let first_at = k.add(hoff, lane);
            let mut total = k.load(smem, first_at);
            let mut w = 1i64;
            while w < JW {
                let woff = k.const_idx(h * i64::from(threads) + w * DW);
                let at = k.add(woff, lane);
                let part = k.load(smem, at);
                total = k.add(total, part);
                w += 1;
            }
            let hc = k.const_idx(h);
            let head = k.add(head0, hc);
            let oidx = k.mad(head, hd, d);
            k.store(out, total, oidx);
        }
    });
    k.verify();
    k
}

/// Declare every scalar variable param as a dimension, the way the graph does
/// when it replays a shape dim into a kernel: a length is non-negative, and it
/// has no upper bound the kernel may rely on.
fn declare_dimensions(k: &mut zyx::kernel::Kernel) {
    use zyx::kernel::{Op, ParamKind};
    let mut ids = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        if matches!(k.at(op_id), Op::Param { kind: ParamKind::Variable, .. }) {
            ids.push(op_id);
        }
        op_id = k.next_op(op_id);
    }
    for id in ids {
        k.set_param_bounds(id, (0, i64::MAX));
    }
}

/// Time one PV kernel through the search and check its values against
/// [`pv_ref`].
fn run_pv(dev: zyx::Dev, build: fn(zyx::Dev) -> zyx::kernel::Kernel, label: &str) -> Result<(), ZyxError> {
    use zyx::kernel::BeamSearch;

    let n = MAX_CONTEXT;
    let v_host = f16_vals(n * N_KV * HEAD_DIM, 9);
    let p_host = f16_vals(N_HEAD * n, 7);
    let v_t = Tensor::from_vec(v_host.clone(), [n, N_KV, HEAD_DIM])?.to(dev)?;
    let p_t = Tensor::from_vec(p_host.clone(), [N_HEAD, n])?.to(dev)?;
    let n_t = Tensor::variable(n);
    let scratch = Tensor::zeros([N_HEAD, HEAD_DIM], DType::F32).to(dev)?;

    // The seed as production hands it over: the dim ranges declared, and the
    // epilogue already applied (the search does no seed preprocessing).
    let mut seed = build(dev);
    declare_dimensions(&mut seed);
    seed.default_epilogue();

    let (winner, nanos) = BeamSearch::new().run([seed], &[&v_t, &p_t, &n_t, &scratch], &[], |_| {}, |_| 0)?;

    let out = winner.compile()?.forward(&[&v_t, &p_t, &n_t], vec![[N_HEAD, HEAD_DIM]])?.pop().unwrap();
    let got: Vec<f32> = out.try_into()?;
    let want = pv_ref(&p_host, &v_host, 1, n);
    assert_pv(&got, &want, label);
    println!("{label} [{N_HEAD} heads x {n} cache -> {HEAD_DIM}]: {:.4} ms", nanos as f64 * 1e-6);
    Ok(())
}

#[test]
fn pv_naive_ir() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    run_pv(dev, pv_naive, "pv naive")
}

#[test]
fn pv_fast_ir() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    run_pv(dev, pv_fast, "pv fast")
}

#[test]
fn pv_fast_kv_ir() -> Result<(), ZyxError> {
    let Some(dev) = dev() else { return Ok(()) };
    run_pv(dev, pv_fast_kv, "pv fast kv")
}
