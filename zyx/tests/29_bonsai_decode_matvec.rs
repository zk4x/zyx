// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B decode matvec (batch `M = 1`) on the PQ2_0 weights:
//! `ffn_gate` / `ffn_up` `[1,5120] x [5120,17408]`, `ffn_down`
//! `[1,17408] x [17408,5120]` and `lm_head` `[1,5120] x [5120,248320]`.
//!
//! The reference is llama.cpp's decode path, `mul_mat_vec_q<PQ2_0, ncols_dst=1>`:
//! one output row per block, 128 threads (`nwarps = 4`, 32 lanes each), the K
//! axis split over the threads, the activations quantized to `q8_1` by its own
//! pre-pass, the PQ2_0 code words expanded with `prmt` and multiplied with
//! `dp4a`, and the per-thread partials reduced across the four warps in shared
//! memory and then across the 32 lanes with `shfl.sync.bfly`.
//!
//! This file holds the three kernels of that path, in the order they are
//! built:
//!
//! 1. [`naive_matvec`] — the kernel the linearizer emits for the projection:
//!    one output per block, one loop over K, the PQ2_0 super-block
//!    dequantized per element (two `u8` loads for the f16 scale, one `u8`
//!    load for the code byte), an f16 weight times an f16 activation, an f32
//!    accumulator.
//! 2. [`quantize_q8_1`] — llama.cpp's activation pre-pass
//!    (`ggml/src/ggml-cuda/quantize.cu`, `quantize_q8_1`): `d = amax/127` over
//!    each 32-element group, `q = roundf(x/d)` (ties away from zero), the
//!    scale pair stored as `half2(d, sum)`, `amax == 0 -> q = 0`.
//! 3. [`vec_dot_matvec`] — the reference projection: `vec_dot_pq2_0_q8_1`
//!    (`ggml/src/ggml-cuda/vecdotq.cuh:955`) with the lane mapping of
//!    `mul_mat_vec_q` (`mmvq.cu:559`), `QI_PQ2_0 = 4` threads per 128-weight
//!    super-block, `blocks_per_iter = 32`.
//!
//! The projection reads the PQ2_0 super-block as 17 `u16` words (the f16
//! scale, then the 32 code bytes as 16 words) and the `q8_1` payload as `u32`
//! words (four `int8` each), exactly the widths the kernels load. As in
//! `26_qwen_matmul.rs`, the caller supplies those word views: `q8_1` stores
//! its payload as bytes and the projection reads them four at a time, so the
//! byte buffer is handed over as its word view.
//!
//! Values are checked against a host reference that runs the same arithmetic:
//! the naive form against an f16-rounded dequant (as the model's projection
//! casts its activation), the fast form against llama.cpp's exact integer dot
//! with the f16 scales applied per 32-element group.

use std::time::Instant;

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError, f16};

/// Residual stream width (`ffn_gate`/`ffn_up`/`lm_head` K).
const HIDDEN: i64 = 5_120;
/// SwiGLU inner width (`ffn_gate`/`ffn_up` N, `ffn_down` K).
const FFN: i64 = 17_408;
/// Vocabulary size (`lm_head` N).
const VOCAB: i64 = 248_320;
/// `q8_1` group width: one f16 scale per this many activations.
const QK: i64 = 32;
/// Weights per PQ2_0 super-block.
const WBLOCK: i64 = 128;
/// Bytes per PQ2_0 super-block: two scale bytes, then 32 code bytes.
const WROW: i64 = 34;
/// `u16` words per PQ2_0 super-block (the f16 scale, then 16 code words).
const WWORDS: i64 = 17;
/// Threads per output row: four warps of 32 lanes.
const WARPS: u32 = 4;
/// Lanes per warp.
const LANES: u32 = 32;
/// `QI_PQ2_0`: threads that share one 128-weight super-block (`QK_PQ2_0/32`).
const QI: i64 = 4;
/// Threads of one output row (`WARPS * LANES`).
const THREADS: i64 = 4 * 32;
/// `blocks_per_iter = nwarps * warp_size / qi`: super-blocks one row's threads
/// cover per pass over K.
const BPI: i64 = THREADS / QI;

/// The byte-permute table that expands a 2-bit code into its value `code - 1`:
/// byte `i` of `0x020100FF` is `i - 1`, so a code selects `-1, 0, 1, 2`.
const PRMT_TABLE: u32 = 0x0201_00FF;
/// Three-bit-per-nibble mask: bit 3 of a `prmt` selector is sign-replicate, so
/// the pair of codes packed in a nibble is reduced to its low code before the
/// table lookup (the same `and.b32 ..., 0x7777` ptxas inserts).
const PRMT_MASK: u16 = 0x7777;
/// The even/odd byte selectors of the two output words, as `__byte_perm`
/// spells them in `vec_dot_pq2_0_q8_1`.
const PRMT_EVEN: u32 = 0x5140;
/// The odd half of the same pair.
const PRMT_ODD: u32 = 0x7362;

/// Deterministic pseudo-random bytes.
fn rng_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1du64 ^ seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// A PQ2_0 weight as the checkpoint stores it: `[rows*cols/128, 34]` blocks
/// with a finite, non-zero f16 scale per super-block and random codes.
fn pq2_raw(rows: i64, cols: i64, seed: u64) -> Vec<u8> {
    let blocks = usize::try_from(rows * cols / WBLOCK).expect("block count fits usize");
    let mut raw = vec![0u8; blocks * usize::try_from(WROW).expect("row fits usize")];
    let noise = rng_bytes(blocks * 32, seed);
    for b in 0..blocks {
        // The codes only reach -1..2, so the projection's magnitude comes from
        // the scale; keep it finite and positive.
        let scale = 0.01f32 + (b % 97) as f32 * 0.001;
        let bits = f16::from_f32(scale).0;
        raw[b * 34] = (bits & 0xff) as u8;
        raw[b * 34 + 1] = (bits >> 8) as u8;
        raw[b * 34 + 2..b * 34 + 34].copy_from_slice(&noise[b * 32..b * 32 + 32]);
    }
    raw
}

/// The super-block's value bytes as `u16` words: the f16 scale, then 16 code
/// words (four 2-bit codes each), the view the projection loads.
fn pq2_words(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// The f16 scale of super-block `b`, as the projection reads it.
fn pq2_scale(raw: &[u8], b: usize) -> f32 {
    let base = b * 34;
    f16(u16::from(raw[base]) | (u16::from(raw[base + 1]) << 8)).to_f32()
}

/// `(code - 1)` of element `elem` of the packed weight, as an integer.
fn pq2_code(raw: &[u8], elem: usize) -> i32 {
    let base = elem / 128 * 34;
    let off = elem % 128;
    let code = (raw[base + 2 + off / 4] >> (2 * (off % 4))) & 3;
    i32::from(code) - 1
}

/// F32 -> F16 with the hardware's round-to-nearest-even (`cvt.rn.f16.f32`,
/// what every `cast` and every `mul.f16` in the kernels does).
///
/// [`f16::from_f32`] rounds ties *away from zero*, so an exact tie is
/// corrected here: bit 12 of the f32 mantissa set with the twelve low bits
/// clear is a tie, and the away-from-zero result then sits one ULP above the
/// even neighbour.
fn f16_rne(x: f32) -> f16 {
    let h = f16::from_f32(x);
    let bits = x.to_bits();
    if bits & 0x1fff == 0x1000 && h.0 & 1 == 1 {
        f16(h.0 - 1)
    } else {
        h
    }
}

/// [`f16_rne`] as the f32 value the kernels multiply.
fn round16_rne(x: f32) -> f32 {
    f16_rne(x).to_f32()
}

fn assert_close(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - e).abs();
        assert!(d <= abs + rel * e.abs(), "{what}[{i}]: zyx {g} != {e} (diff {d})");
    }
}

// ===========================================================================
// 1. The naive projection
// ===========================================================================

/// The kernel the linearizer emits for `[1, k] x [k, n]`: one output row per
/// block, one loop over K, the PQ2_0 super-block dequantized per element.
///
/// The dequant chain is the one `Op::Dequantize`'s lowering emits (the
/// `ZYX_DEBUG=264` dump of `27_bonsai.rs::linear_projection`): the block's two
/// scale bytes loaded as `u8` and sewn into the f16 bit pattern, the code byte
/// loaded per element and shifted by `2 * (off % 4)`, `code - 1` scaled and
/// rounded to f16, `f16 * f16 -> f16`, the product accumulated in f32.
fn naive_matvec(dev: Dev, n: i64, k: i64) -> Kernel {
    let mut kk = Kernel::new(dev);
    let act = kk.param(DType::F16);
    let w = kk.param(DType::U8);
    let c = kk.param_mut(DType::F32);
    let row = kk.group_range(0, n);
    let acc = kk.storage(DType::F32, MemScope::Register, 1);
    let zero_f = kk.const_val(0.0f32);
    kk.store(acc, zero_f, 0i64);
    let ck = kk.const_val(k);
    let c1 = kk.const_val(1i64);
    let c2 = kk.const_val(2i64);
    let c8 = kk.const_val(8u16);
    let c34 = kk.const_val(WROW);
    let c128 = kk.const_val(WBLOCK);
    let c4 = kk.const_val(4i64);
    let c2u8 = kk.const_val(2u8);
    let mask3 = kk.const_val(3u8);
    let one = kk.const_val(1.0f32);
    kk.loop_over(k, |kk, kk_loop| {
        // `elem = n*K + k`: the flat index into the row-major `[n, k]` weight.
        let elem = kk.mad(row, ck, kk_loop);
        let blk = kk.div(elem, c128);
        let off = kk.mod_(elem, c128);
        let wbase = kk.mul(blk, c34);
        // The block's f16 scale, little-endian in its first two bytes.
        let s_lo = kk.load(w, wbase);
        let s_hi_at = kk.add(wbase, c1);
        let s_hi = kk.load(w, s_hi_at);
        let s_lo16 = kk.cast(s_lo, DType::U16);
        let s_hi16 = kk.cast(s_hi, DType::U16);
        let s_hi_sh = kk.bit_shift_left(s_hi16, c8);
        let s16 = kk.bit_or(s_lo16, s_hi_sh);
        let s16b = kk.bitcast(s16, DType::F16);
        let scale = kk.cast(s16b, DType::F32);
        // The element's code byte and its 2-bit field.
        let off_byte = kk.div(off, c4);
        let byte_rel = kk.add(wbase, c2);
        let byte_idx = kk.add(byte_rel, off_byte);
        let byte = kk.load(w, byte_idx);
        let off_mod = kk.mod_(off, c4);
        let off_u8 = kk.cast(off_mod, DType::U8);
        let shift = kk.mul(off_u8, c2u8);
        let shifted = kk.bit_shift_right(byte, shift);
        let code = kk.bit_and(shifted, mask3);
        let code_f = kk.cast(code, DType::F32);
        let centered = kk.sub(code_f, one);
        let val = kk.mul(centered, scale);
        let val = kk.cast(val, DType::F16);
        let a = kk.load(act, kk_loop);
        let prod = kk.mul(val, a);
        let prod = kk.cast(prod, DType::F32);
        let prev = kk.load(acc, 0i64);
        let sum = kk.add(prod, prev);
        kk.store(acc, sum, 0i64);
    });
    let res = kk.load(acc, 0i64);
    kk.store(c, res, row);
    kk.verify();
    kk
}

#[test]
fn naive_decode_gate() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    const N: i64 = FFN;
    const K: i64 = HIDDEN;

    let raw = pq2_raw(N, K, 1);
    let act_vals: Vec<f16> = (0..K).map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0)).collect();

    let kernel = naive_matvec(dev, N, K);
    let act: Tensor = Tensor::from_vec(act_vals.clone(), [K])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(raw.clone(), [N * K / WBLOCK, WROW])?.to(dev)?;
    let out: Tensor = Tensor::zeros([N], DType::F32).to(dev)?;
    let (got, nanos) = time_kernel(kernel, &[&act, &w], &[&out], [N])?;

    let mut want = vec![0f32; usize::try_from(N).unwrap()];
    let k_us = usize::try_from(K).unwrap();
    let blocks_per_row = usize::try_from(K / WBLOCK).unwrap();
    for (r, out) in want.iter_mut().enumerate() {
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let base = (r * blocks_per_row + b) * usize::try_from(WROW).unwrap();
            let scale = pq2_scale(&raw, r * blocks_per_row + b);
            for j in 0..usize::try_from(WBLOCK).unwrap() {
                let code = (raw[base + 2 + j / 4] >> (2 * (j % 4))) & 3;
                let deq = round16_rne((f32::from(code) - 1.0) * scale);
                let prod = round16_rne(deq * act_vals[(r * k_us + b * 128 + j) % k_us].to_f32());
                acc += prod;
            }
        }
        *out = acc;
    }
    assert_close(&got, &want, 1e-3, 2e-3, "naive matvec");

    let bytes = (N * K / WBLOCK * WROW) as f64;
    println!("bonsai naive decode gate [1x{K}]x[{K}x{N}]: {:.4} ms, {:.1} GB/s packed", nanos * 1e-6, bytes / nanos);
    Ok(())
}

// ===========================================================================
// 2. The q8_1 activation pre-pass
// ===========================================================================

/// `quantize_q8_1` (`quantize.cu:55`) for one token: one block of `k/32`
/// threads, one 32-element group per thread (one lane in llama's kernel, with
/// the warp reduce replaced by the group's own max/sum along the unroll axis,
/// which is exact for `max`).
///
/// Output: the payload bytes (`I8`, `[k]`) and one `ds` word per group
/// (`U32`, `[k/32]`), its low half the f16 scale `d` and its high half the f16
/// group sum, the `half2(d, sum)` the reference stores.
fn quantize_q8_1(dev: Dev, k: i64) -> Kernel {
    let mut kk = Kernel::new(dev);
    let x = kk.param(DType::F32);
    let q8 = kk.param_mut(DType::I8);
    let ds = kk.param_mut(DType::U32);
    let groups = k / QK;
    let g = kk.local_range(0, u32::try_from(groups).expect("groups fit u32"));
    let cqk = kk.const_val(QK);
    let base = kk.mul(g, cqk);
    let l32 = kk.unroll(1, 32);
    let vidx = kk.add(base, l32);
    let vals = kk.load(x, vidx);
    let av = kk.abs(vals);
    let amax = kk.reduce_max(av, l32);
    // The group sum is the `half2`'s high half. It is its own lane so the two
    // reduce cones do not nest (the devectorizer unfolds one axis at a time).
    let l32s = kk.unroll(2, 32);
    let sidx = kk.add(base, l32s);
    let svals = kk.load(x, sidx);
    let sum = kk.reduce_sum(svals, l32s);
    let hundred = kk.const_val(127.0f32);
    let d = kk.div(amax, hundred);
    // `roundf(x/d)`, ties away from zero: trunc(qf + copysign(0.5, qf)).
    let qf = kk.div(vals, d);
    let zero_f = kk.const_val(0.0f32);
    let nonneg = kk.cmpge(qf, zero_f);
    let plus_half = kk.const_val(0.5f32);
    let minus_half = kk.const_val(-0.5f32);
    let half = kk.select(nonneg, plus_half, minus_half);
    let biased = kk.add(qf, half);
    let rounded = kk.trunc(biased);
    let zero_i = kk.const_val(0i32);
    let is_zero = kk.equal(amax, zero_f);
    let q_i = kk.cast(rounded, DType::I32);
    let q = kk.select(is_zero, zero_i, q_i);
    let qb = kk.cast(q, DType::I8);
    kk.store(q8, qb, vidx);
    // `half2(d, sum)`: low half the scale, high half the group sum.
    let d16 = kk.cast(d, DType::F16);
    let s16 = kk.cast(sum, DType::F16);
    let d_u16 = kk.bitcast(d16, DType::U16);
    let s_u16 = kk.bitcast(s16, DType::U16);
    let d32 = kk.cast(d_u16, DType::U32);
    let s32 = kk.cast(s_u16, DType::U32);
    let sh16 = kk.const_val(16u32);
    let s32_sh = kk.bit_shift_left(s32, sh16);
    let word = kk.bit_or(d32, s32_sh);
    kk.store(ds, word, g);
    kk.verify();
    kk
}

/// The host half of `quantize_q8_1`, exactly as llama.cpp's kernel computes it.
fn q8_1_host(x: &[f32]) -> (Vec<i8>, Vec<u32>) {
    let groups = x.len() / QK as usize;
    let mut payload = vec![0i8; x.len()];
    let mut ds = vec![0u32; groups];
    for g in 0..groups {
        let base = g * QK as usize;
        let mut amax = 0f32;
        let mut sum = 0f32;
        for i in 0..QK as usize {
            amax = amax.max(x[base + i].abs());
            sum += x[base + i];
        }
        let d = amax / 127.0;
        for i in 0..QK as usize {
            // `roundf` is ties-away-from-zero: `f32::round` is the same.
            let q = if amax == 0.0 { 0.0 } else { (x[base + i] / d).round() };
            payload[base + i] = q as i8;
        }
        let dbits = f16_rne(d).0;
        let sbits = f16_rne(sum).0;
        ds[g] = u32::from(dbits) | (u32::from(sbits) << 16);
    }
    (payload, ds)
}

#[test]
fn q8_1_quantize_matches_llama() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    const K: i64 = HIDDEN;
    // A group that is exactly zero (the `amax == 0` guard) and values that
    // straddle zero.
    let mut vals: Vec<f32> = (0..K).map(|i| ((i * 13 % 251) as f32 - 125.0) / 97.0).collect();
    for v in vals.iter_mut().take(QK as usize) {
        *v = 0.0;
    }
    let (payload_host, ds_host) = q8_1_host(&vals);

    let kernel = quantize_q8_1(dev, K);
    let x: Tensor = Tensor::from_vec(vals.clone(), [K])?.to(dev)?;
    let scratch_payload: Tensor = Tensor::zeros([K], DType::I8).to(dev)?;
    let scratch_ds: Tensor = Tensor::zeros([K / QK], DType::U32).to(dev)?;
    let compiled = kernel.compile()?;
    let started = Instant::now();
    let out = compiled.forward(&[&x], vec![[K], [K / QK]])?;
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let payload: Vec<i8> = out[0].clone().try_into()?;
    let ds: Vec<u32> = out[1].clone().try_into()?;
    let _ = (&scratch_payload, &scratch_ds);

    for (i, (g, h)) in payload.iter().zip(payload_host.iter()).enumerate() {
        if g != h {
            return Err(ZyxError::kernel_error(format!("q8_1 payload[{i}]: {g} != {h}").into()));
        }
    }
    // The scale half is the f16 scale; the sum half is unused by the
    // projection, so only the low half is compared.
    for (i, (g, h)) in ds.iter().zip(ds_host.iter()).enumerate() {
        if g & 0xffff != h & 0xffff {
            return Err(ZyxError::kernel_error(format!("q8_1 scale[{i}]: {:#06x} != {:#06x}", g & 0xffff, h & 0xffff).into()));
        }
    }
    println!("bonsai q8_1 [{K}]: {ms:.4} ms, payload + {} scales", K / QK);
    Ok(())
}

// ===========================================================================
// 3. The reference projection
// ===========================================================================

/// `mul_mat_vec_q<PQ2_0, 1>` (`mmvq.cu:559`) with
/// `vec_dot_pq2_0_q8_1` (`vecdotq.cuh:955`): one output row per block, 128
/// threads as four warps of 32 lanes.
///
/// Lane mapping (`QI_PQ2_0 = 4`, `blocks_per_iter = 128/4 = 32`): thread `tid`
/// owns chunk `tid % 4` of super-block `tid / 4`, then the same chunk of the
/// block `32` further on, while the block index is in range. Per chunk: four
/// `u16` code words, each expanded into two `int8` quads by `prmt`, and eight
/// `dp4a` against the chunk's eight `u32` activation words; the group's exact
/// integer sum is scaled once by `d2 * d8`.
///
/// Reduction (`mmvq.cu:763`): warps 1..3 write their per-lane partial to
/// shared memory at `(warp - 1) * 32 + lane`, a barrier, then warp 0 adds the
/// three to its own partial, reduces over the 32 lanes with
/// `shfl.sync.bfly`, and lane 0 stores.
fn vec_dot_matvec(dev: Dev, n: i64, k: i64) -> Kernel {
    let mut kk = Kernel::new(dev);
    let a8 = kk.param(DType::U32);
    let ds = kk.param(DType::U32);
    let w = kk.param(DType::U16);
    let c = kk.param_mut(DType::F32);
    let row = kk.group_range(0, n);
    let blk_count = k / WBLOCK;

    let lane = kk.local_range(0, LANES);
    let warp = kk.local_range(1, WARPS);
    let tid = kk.mad(warp, i64::from(LANES), lane);

    let cqi = kk.const_val(QI);
    let blk0 = kk.div(tid, cqi);
    let chunk = kk.mod_(tid, cqi);
    let c17 = kk.const_val(WWORDS);
    let c1 = kk.const_val(1i64);
    let two_u16 = kk.const_val(2u16);
    let mask = kk.const_val(PRMT_MASK);
    let table = kk.const_val(PRMT_TABLE);
    let sel_even = kk.const_val(PRMT_EVEN);
    let sel_odd = kk.const_val(PRMT_ODD);
    let c8 = kk.const_val(8i64);
    let c2 = kk.const_val(2i64);
    let u32_16 = kk.const_val(16u32);
    let u32_0 = kk.const_val(0u32);

    let acc = kk.storage(DType::F32, MemScope::Register, 1);
    let zero_f = kk.const_val(0.0f32);
    kk.store(acc, zero_f, 0i64);

    // `for (kbx = tid/4; kbx < blocks_per_row_x; kbx += blocks_per_iter)`:
    // a fixed `ceil(blocks / 32)` trip count, each trip guarded, so the
    // threads whose next block is out of range drop out exactly as the
    // reference's loop bound does.
    let trips = (blk_count + BPI - 1) / BPI;
    let cblk_count = kk.const_val(blk_count);
    let zero_i = kk.const_val(0i64);
    let warp_zero = kk.equal(warp, zero_i);
    let _ = &cblk_count;
    kk.loop_over(trips, |kk, it| {
        let cbpi = kk.const_val(BPI);
        let step = kk.mul(it, cbpi);
        let blk = kk.add(blk0, step);
        let in_range = kk.cmplt(blk, cblk_count);
        kk.if_over(in_range, |kk| {
            // `kbx_offset + kbx`: the row's own super-block base, then this
            // thread's block inside it.
            let row_stride = kk.const_val(blk_count * WWORDS);
            let row_words = kk.mul(row, row_stride);
            let blk_words = kk.mul(blk, c17);
            let base17 = kk.add(row_words, blk_words);
            // The super-block's f16 scale (word 0) and the group's f16 scale
            // (the low half of its `ds` word).
            let wd16 = kk.load(w, base17);
            let wd16 = kk.bitcast(wd16, DType::F16);
            let wd = kk.cast(wd16, DType::F32);
            let ds_idx = kk.mul(blk, QI);
            let ds_idx = kk.add(ds_idx, chunk);
            let dsw = kk.load(ds, ds_idx);
            let mask16 = kk.const_val(0xffffu32);
            let ds_lo = kk.bit_and(dsw, mask16);
            let ds_u16 = kk.cast(ds_lo, DType::U16);
            let ds_f16 = kk.bitcast(ds_u16, DType::F16);
            let ad = kk.cast(ds_f16, DType::F32);
            // Eight `dp4a` over the chunk's 32 elements.
            let mut sumi = kk.const_val(0u32);
            for j in 0..4i64 {
                let cj = kk.const_val(j);
                let chunk_q = kk.mul(chunk, QI);
                let word = kk.add(base17, c1);
                let word = kk.add(word, chunk_q);
                let word = kk.add(word, cj);
                let q = kk.load(w, word);
                let qe_sel = kk.bit_and(q, mask);
                let qe_sel = kk.cast(qe_sel, DType::U32);
                let q_shifted = kk.bit_shift_right(q, two_u16);
                let qo_sel = kk.bit_and(q_shifted, mask);
                let qo_sel = kk.cast(qo_sel, DType::U32);
                let qe = kk.prmt(table, table, qe_sel);
                let qo = kk.prmt(table, table, qo_sel);
                let qx = kk.prmt(qe, qo, sel_even);
                let qy = kk.prmt(qe, qo, sel_odd);
                let aw = kk.mul(ds_idx, c8);
                let two_j = kk.const_val(2 * j);
                let aw = kk.add(aw, two_j);
                let u = kk.load(a8, aw);
                let aw1 = kk.add(aw, c1);
                let v = kk.load(a8, aw1);
                sumi = kk.dp4a(u, qx, sumi);
                sumi = kk.dp4a(v, qy, sumi);
            }
            let si = kk.bitcast(sumi, DType::I32);
            let si = kk.cast(si, DType::F32);
            let scale = kk.mul(wd, ad);
            let part = kk.mul(scale, si);
            let prev = kk.load(acc, 0i64);
            let total = kk.add(prev, part);
            kk.store(acc, total, 0i64);
            let _ = (c2, u32_16, u32_0);
        });
    });

    // Warps 1..3 stage their per-lane partial; warp 0 folds and reduces.
    let sh_len = i64::from(WARPS * LANES);
    let sh = kk.storage(DType::F32, MemScope::Local, sh_len);
    let zero_i = kk.const_val(0i64);
    let warp_on = kk.cmpgt(warp, zero_i);
    kk.if_over(warp_on, |kk| {
        let wm1 = kk.sub(warp, 1i64);
        let wrow = kk.mul(wm1, i64::from(LANES));
        let at = kk.add(wrow, lane);
        let v = kk.load(acc, 0i64);
        kk.store(sh, v, at);
    });
    kk.barrier();
    kk.if_over(warp_zero, |kk| {
        let mut s = kk.load(acc, 0i64);
        for wv in 0..(WARPS - 1) {
            let cw = kk.const_val(i64::from(wv));
            let wrow = kk.mul(cw, i64::from(LANES));
            let at = kk.add(wrow, lane);
            let v = kk.load(sh, at);
            s = kk.add(s, v);
        }
        let r = kk.warp_reduce(s);
        let lane_zero = kk.equal(lane, zero_i);
        kk.if_over(lane_zero, |kk| {
            kk.store(c, r, row);
        });
    });
    kk.verify();
    kk
}

/// One compiled launch, timed by the search's device events (which take every
/// buffer, the outputs included), with its values read back from a second
/// compile.
fn time_kernel(kernel: Kernel, inputs: &[&Tensor], scratch: &[&Tensor], shape: [i64; 1]) -> Result<(Vec<f32>, f64), ZyxError> {
    let values = kernel.clone();
    let mut all: Vec<&Tensor> = inputs.to_vec();
    all.extend_from_slice(scratch);
    let (_winner, nanos) = BeamSearch::new().run([kernel], &all, &[], |_| {}, |_| 0)?;
    let compiled = values.compile()?;
    let out = compiled.forward(inputs, vec![shape])?.pop().unwrap();
    let got: Vec<f32> = out.try_into()?;
    Ok((got, nanos as f64))
}

/// llama.cpp's exact value for one output row: the integer dot per 32-element
/// group, scaled once by `f16(w scale) * f16(activation scale)` and folded in
/// f32 in group order.
fn vec_dot_ref(raw: &[u8], payload: &[i8], ds: &[u32], row: i64, k: i64) -> f32 {
    let mut acc = 0f32;
    let groups = k / QK;
    for g in 0..groups {
        let mut sumi = 0i32;
        for j in 0..QK {
            let elem = usize::try_from(row * k + g * QK + j).expect("elem fits usize");
            let q = i32::from(payload[usize::try_from(g * QK + j).expect("payload fits usize")]);
            sumi += pq2_code(raw, elem) * q;
        }
        let blk = usize::try_from((row * k + g * QK) / WBLOCK).expect("block fits usize");
        let wd = pq2_scale(raw, blk);
        let ad = f16((ds[usize::try_from(g).expect("group fits usize")] & 0xffff) as u16).to_f32();
        acc += (wd * ad) * sumi as f32;
    }
    acc
}

#[test]
fn fast_decode_gate() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    const N: i64 = FFN;
    const K: i64 = HIDDEN;

    let raw = pq2_raw(N, K, 2);
    let (payload, ds_host) = {
        let x: Vec<f32> = (0..K).map(|i| ((i * 29 % 197) as f32 - 98.0) / 131.0).collect();
        q8_1_host(&x)
    };
    let a32: Vec<u32> =
        payload.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();

    let a8: Tensor = Tensor::from_vec(a32, [K / 4])?.to(dev)?;
    let ds: Tensor = Tensor::from_vec(ds_host.clone(), [K / QK])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(pq2_words(&raw), [N * K / WBLOCK * WWORDS])?.to(dev)?;
    let out: Tensor = Tensor::zeros([N], DType::F32).to(dev)?;

    let (got, nanos) = time_kernel(vec_dot_matvec(dev, N, K), &[&a8, &ds, &w], &[&out], [N])?;

    // The integer sums are exact; only the f32 fold order across the lanes can
    // differ, so the bound is a few ULPs of the row's magnitude.
    for r in 0..N {
        let want = vec_dot_ref(&raw, &payload, &ds_host, r, K);
        let err = (got[usize::try_from(r).unwrap()] - want).abs();
        let tol = 1e-4f32.max(1e-4 * want.abs());
        if err > tol {
            return Err(ZyxError::kernel_error(
                format!("fast matvec[{r}]: {} != {want} (err {err}, tol {tol})", got[usize::try_from(r).unwrap()]).into(),
            ));
        }
    }
    let bytes = (N * K / WBLOCK * WROW) as f64;
    println!("bonsai fast decode gate [1x{K}]x[{K}x{N}]: {:.4} ms, {:.1} GB/s packed", nanos * 1e-6, bytes / nanos);
    Ok(())
}

// ===========================================================================
// 4. The int8 form the linearizer emits once the model writer adds the
//    explicit `q8_1` casts, and the optimization that makes it fast
// ===========================================================================

/// The naive int8 projection: the same per-output/per-K skeleton as
/// [`naive_matvec`], but the dot is the `q8_1` one — a `u16` code word and a
/// `u32` activation word read per element, the code and byte extracted with
/// shifts, the product accumulated exactly in `i32` per 32-element group, and
/// the group's integer sum scaled once by `f16(w scale) * f16(act scale)`.
///
/// This is the shape `opt_local_reduce` matches: the group loop is the
/// reduction loop, the inner 32-element loop is the chunk, and the two
/// register accumulators are the group sum (`i32`) and the row total (`f32`).
fn naive_int8_matvec(dev: Dev, n: i64, k: i64) -> Kernel {
    let mut kk = Kernel::new(dev);
    let a8 = kk.param(DType::U32);
    let ds = kk.param(DType::U32);
    let w = kk.param(DType::U16);
    let c = kk.param_mut(DType::F32);
    let row = kk.group_range(0, n);
    let groups = k / QK;
    let acc = kk.storage(DType::F32, MemScope::Register, 1);
    let zero_f = kk.const_val(0.0f32);
    kk.store(acc, zero_f, 0i64);

    let ckqk = kk.const_val(QK);
    let ck = kk.const_val(k);
    let c128 = kk.const_val(WBLOCK);
    let c127 = kk.const_val(WBLOCK - 1);
    let c17 = kk.const_val(WWORDS);
    let c1 = kk.const_val(1i64);
    let c3 = kk.const_val(3i64);
    let c4 = kk.const_val(4i64);
    let c8 = kk.const_val(8i64);
    let two_u16 = kk.const_val(2u16);
    let c7 = kk.const_val(7i64);
    let mask3 = kk.const_val(3u16);
    let mask_ff = kk.const_val(0xffu32);
    let mask16 = kk.const_val(0xffffu32);
    let zero_i = kk.const_val(0i32);
    let one_i = kk.const_val(1i32);
    let c8u32 = kk.const_val(8u32);
    let row_blocks = kk.const_val(k / WBLOCK * WWORDS);

    kk.loop_over(groups, |kk, g| {
        let sum_s = kk.storage(DType::I32, MemScope::Register, 1);
        kk.store(sum_s, zero_i, 0i64);
        let gbase = kk.mul(g, ckqk);
        kk.loop_over(QK, |kk, j| {
            let idx = kk.add(gbase, j);
            // The element's flat weight index and its super-block.
            let roww = kk.mul(row, ck);
            let elem = kk.add(roww, idx);
            let wblk = kk.div(elem, c128);
            let off = kk.bit_and(elem, c127);
            let wbase = kk.mul(wblk, c17);
            // The code word and the element's 2-bit field.
            let off_w = kk.div(off, c8);
            let wi = kk.add(wbase, c1);
            let wi = kk.add(wi, off_w);
            let qw = kk.load(w, wi);
            let off_2 = kk.bit_and(off, c7);
            let off_u16 = kk.cast(off_2, DType::U16);
            let sh = kk.mul(off_u16, two_u16);
            let code = kk.bit_shift_right(qw, sh);
            let code = kk.bit_and(code, mask3);
            let code_i = kk.cast(code, DType::I32);
            let code_i = kk.sub(code_i, one_i);
            // The activation byte and its `int8` value.
            let aw_i = kk.div(idx, c4);
            let aw = kk.load(a8, aw_i);
            let byte_i = kk.bit_and(idx, c3);
            let byte_u32 = kk.cast(byte_i, DType::U32);
            let sh32 = kk.mul(byte_u32, c8u32);
            let shifted = kk.bit_shift_right(aw, sh32);
            let byte = kk.bit_and(shifted, mask_ff);
            let byte_u8 = kk.cast(byte, DType::U8);
            let byte_i8 = kk.bitcast(byte_u8, DType::I8);
            let act_i = kk.cast(byte_i8, DType::I32);
            let prod = kk.mul(code_i, act_i);
            let prev = kk.load(sum_s, 0i64);
            let sum = kk.add(prev, prod);
            kk.store(sum_s, sum, 0i64);
        });
        // The group's two f16 scales: the super-block's and the activation
        // group's, applied once for the whole chunk.
        let gblk = kk.div(g, c4);
        let gb = kk.mul(row, row_blocks);
        let gbase_w = kk.mul(gblk, c17);
        let gsw = kk.add(gb, gbase_w);
        let wd16 = kk.load(w, gsw);
        let wd16 = kk.bitcast(wd16, DType::F16);
        let wd = kk.cast(wd16, DType::F32);
        let dsw = kk.load(ds, g);
        let ds_lo = kk.bit_and(dsw, mask16);
        let ds_u16 = kk.cast(ds_lo, DType::U16);
        let ds_f16 = kk.bitcast(ds_u16, DType::F16);
        let ad = kk.cast(ds_f16, DType::F32);
        let sum_i = kk.load(sum_s, 0i64);
        let sum_f = kk.cast(sum_i, DType::F32);
        let scale = kk.mul(wd, ad);
        let part = kk.mul(scale, sum_f);
        let prev = kk.load(acc, 0i64);
        let total = kk.add(prev, part);
        kk.store(acc, total, 0i64);
        let _ = (c3, c128, c127);
    });
    let res = kk.load(acc, 0i64);
    kk.store(c, res, row);
    kk.verify();
    kk
}

/// `1x` decode timing for the four projections this kernel class owns, with
/// the q8_1 operands built on the host: the fast kernel per shape.
#[test]
fn fast_all_projections() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    for (name, n, k) in [
        ("ffn_gate", FFN, HIDDEN),
        ("ffn_up", FFN, HIDDEN),
        ("ffn_down", HIDDEN, FFN),
        ("lm_head", VOCAB, HIDDEN),
    ] {
        // Weights are timed, not checked: a zero scale keeps the buffer
        // allocation on the device and the values irrelevant (nothing
        // branches on the data).
        let w: Tensor = Tensor::zeros([n * k / WBLOCK * WWORDS], DType::U16).to(dev)?;
        let kk = usize::try_from(k).unwrap();
        let (payload, ds_host) = q8_1_host(&vec![0.25f32; kk]);
        let a32: Vec<u32> =
            payload.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
        let a8: Tensor = Tensor::from_vec(a32, [k / 4])?.to(dev)?;
        let ds: Tensor = Tensor::from_vec(ds_host, [k / QK])?.to(dev)?;
        let out: Tensor = Tensor::zeros([n], DType::F32).to(dev)?;
        let (got, nanos) = time_kernel(vec_dot_matvec(dev, n, k), &[&a8, &ds, &w], &[&out], [n])?;
        assert_eq!(got.len(), usize::try_from(n).unwrap());
        let bytes = (n * k / WBLOCK * WROW) as f64;
        println!("bonsai fast {name} [1x{k}]x[{k}x{n}]: {:.4} ms, {:.1} GB/s packed", nanos * 1e-6, bytes / nanos);
    }
    Ok(())
}

/// The generalized `opt_local_reduce` must turn the naive int8 projection into
/// the reference shape: eight `dp4a` and sixteen `prmt` per 32-element group,
/// the group loop split over the configured lanes, and the two-level reduce.
///
/// The values are the host's llama-precision ones; the integer dot is exact,
/// so only the f32 fold order may differ.
#[test]
fn pass_generates_fast_matvec() -> Result<(), ZyxError> {
    use zyx::kernel::PTXOp;

    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    const N: i64 = FFN;
    const K: i64 = HIDDEN;

    let raw = pq2_raw(N, K, 3);
    let (payload, ds_host) = {
        let x: Vec<f32> = (0..K).map(|i| ((i * 53 % 181) as f32 - 90.0) / 113.0).collect();
        q8_1_host(&x)
    };
    let a32: Vec<u32> =
        payload.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
    let a8: Tensor = Tensor::from_vec(a32, [K / 4])?.to(dev)?;
    let ds: Tensor = Tensor::from_vec(ds_host.clone(), [K / QK])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(pq2_words(&raw), [N * K / WBLOCK * WWORDS])?.to(dev)?;
    let out: Tensor = Tensor::zeros([N], DType::F32).to(dev)?;

    // The seed: the naive int8 form, one thread per output row.
    let (seed_got, seed_nanos) = time_kernel(naive_int8_matvec(dev, N, K), &[&a8, &ds, &w], &[&out], [N])?;
    for r in 0..4i64 {
        let want = vec_dot_ref(&raw, &payload, &ds_host, r, K);
        assert!((seed_got[usize::try_from(r).unwrap()] - want).abs() < 1e-3, "naive int8 row {r}");
    }

    let recipe = naive_int8_matvec(dev, N, K).opt_local_reduce();
    println!("pass_generates_fast_matvec: {} configs", recipe.nconfigs());
    assert!(recipe.nconfigs() > 0, "opt_local_reduce must offer the int8 vec-dot split");

    for config in 0..recipe.nconfigs() {
        let mut kernel = naive_int8_matvec(dev, N, K);
        recipe.apply(&mut kernel, config);
        kernel.default_epilogue();
        kernel.verify();
        let dp4a =
            kernel.ops_in_order().filter(|op| matches!(op, zyx::kernel::Op::PTX(p) if matches!(**p, PTXOp::Dp4a { .. }))).count();
        let prmt =
            kernel.ops_in_order().filter(|op| matches!(op, zyx::kernel::Op::PTX(p) if matches!(**p, PTXOp::Prmt { .. }))).count();
        assert_eq!(dp4a, 8, "config {config}: eight dp4a per chunk");
        assert_eq!(prmt, 16, "config {config}: sixteen prmt per chunk");

        let (got, nanos) = time_kernel(kernel, &[&a8, &ds, &w], &[&out], [N])?;
        let mut worst = 0f32;
        for r in 0..N {
            let want = vec_dot_ref(&raw, &payload, &ds_host, r, K);
            let err = (got[usize::try_from(r).unwrap()] - want).abs();
            worst = worst.max(err);
            let tol = 1e-4f32.max(1e-4 * want.abs());
            if err > tol {
                return Err(ZyxError::kernel_error(
                    format!("pass config {config} row {r}: {} != {want} (err {err})", got[usize::try_from(r).unwrap()]).into(),
                ));
            }
        }
        let bytes = (N * K / WBLOCK * WROW) as f64;
        println!("  config {config}: {:.4} ms, {:.1} GB/s packed, worst |err| {worst:.2e}", nanos * 1e-6, bytes / nanos);
    }
    let bytes = (N * K / WBLOCK * WROW) as f64;
    println!("  seed: {:.4} ms, {:.1} GB/s packed (one thread per row, no split)", seed_nanos * 1e-6, bytes / seed_nanos);
    Ok(())
}
