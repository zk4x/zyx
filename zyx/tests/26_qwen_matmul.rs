// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! qwen3.5 (Ternary-Bonsai-2-27B) `ffn_gate` prefill projection with the
//! PQ2_0 dequant fused into the matmul, written as the *naive* custom kernel
//! the linearizer would emit, so the tiling, the staging and the tensor-core
//! instruction come from the optimization passes applied afterwards.
//!
//! The entry is the prefill one: `[128 x 5120] x [5120 x 17408]`, the width
//! `qwen prefill_gemm` measures. The weight is the checkpoint's own layout —
//! raw `[rows*cols/128, 34]` U8 super-blocks, 2 scale bytes and 32 code bytes
//! per 128 weights — and the kernel dequantizes every weight inline, exactly
//! as the fused decode kernel does:
//!
//! ```text
//! elem  = n*K + k
//! blk   = elem / 128                 // one super-block of 128 weights
//! off   = elem % 128
//! byte  = blk*34 + 2 + off/4         // 4 two-bit codes per byte
//! code  = (byte_word >> 2*(off%4)) & 3
//! scale = bits(f16) (lo | hi<<8)     // the block's first two bytes
//! w     = (f32(code) - 1) * scale    // then rounded to f16
//! acc  += f32(f16(w) * f16(act[k]))
//! ```
//!
//! Asserts the result against a host GEMM on the same f16 operands, the
//! reference dequantizing the same super-block bytes, and asserts the
//! tensorization space offers the dequant-fused form: `opt_tensorize` must be
//! able to match a product whose multiplier is a dequant chain rather than a
//! bare load, so the staged fill can dequantize while it cooperatively loads
//! the weight tile into the block's shared memory.

use std::time::Instant;

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError, f16};

/// The reference config in the tensorization space: the staged tile grid the
/// kernel is compared against, the way the llama test names the cuBLAS shape.
/// `opt_tensorize` lists the kernel's own region first, then one block per tile
/// grid in each operand orientation.
const REFERENCE_TILE: u64 = 32;

/// Prefill width: the padded sequence length of a prefill chunk.
const M: i64 = 128;
/// `ffn_gate` output features.
const N: i64 = 17408;
/// `ffn_gate` input features.
const K: i64 = 5120;
/// Weights per PQ2_0 super-block.
const BLOCK: i64 = 128;
/// Bytes per PQ2_0 super-block: 2 scale bytes, then 32 code bytes.
const ROW: i64 = 34;

#[test]
fn qwen_ffn_gate() -> Result<(), ZyxError> {
    // The optimizations read the device's warp size and budgets, so the kernel
    // is bound to a concrete device up front. Without a GPU there is no warp
    // size and no tensor core to measure: skip rather than fail.
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };

    // ---- the naive kernel the linearizer emits -------------------------
    // Two group ranges over the output, one accumulation loop over K, one
    // register accumulator, one store. The weight is dequantized inline, so
    // the product's multiplier is a chain of ops, not a load.
    let mut k = Kernel::new(dev);
    let a = k.param(DType::F16);
    let w = k.param(DType::U8);
    let c = k.param_mut(DType::F32);
    let m = k.group_range(0, M);
    let n = k.group_range(1, N);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(K, |k, k_loop| {
        // The weight's element index in the [N, K] row-major projection.
        let elem = k.mad(n, K, k_loop);
        let blk = k.div(elem, BLOCK);
        let off = k.mod_(elem, BLOCK);
        let row_start = k.mul(blk, ROW);
        // `elem % 4` is the code's position inside its byte, 2 bits apart.
        let two = k.const_val(2i64);
        let four = k.const_val(4i64);
        let off_byte = k.div(off, four);
        let byte_in_row = k.add(2i64, off_byte);
        let byte_idx = k.add(row_start, byte_in_row);
        let word = k.load(w, byte_idx);
        let off_mod = k.mod_(off, four);
        let shift = k.mul(off_mod, two);
        // The shift amount is an operand of the word's own dtype (u8), as the
        // linearized dequant chain shifts by.
        let shift = k.cast(shift, DType::U8);
        let shifted = k.bit_shift_right(word, shift);
        let mask = k.const_val(3u8);
        let code = k.bit_and(shifted, mask);
        // The block's scale: little-endian f16 in the super-block's first two
        // bytes (`f16le` in the dequantize lowering).
        let lo = k.load(w, row_start);
        let hi_idx = k.add(row_start, 1i64);
        let hi = k.load(w, hi_idx);
        let lo16 = k.cast(lo, DType::U16);
        let hi16 = k.cast(hi, DType::U16);
        let eight = k.const_val(8u16);
        let hi_shifted = k.bit_shift_left(hi16, eight);
        let bits = k.bit_or(lo16, hi_shifted);
        let scale_f16 = k.bitcast(bits, DType::F16);
        let scale = k.cast(scale_f16, DType::F32);
        let code_f32 = k.cast(code, DType::F32);
        let centered = k.sub(code_f32, 1.0f32);
        let val = k.mul(centered, scale);
        // The projection is an f16 x f16 multiply with an f32 accumulate.
        let val = k.cast(val, DType::F16);
        let act_idx = k.mad(m, K, k_loop);
        let act = k.load(a, act_idx);
        let prod = k.mul(val, act);
        let prod = k.cast(prod, DType::F32);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    let res = k.load(acc, 0i64);
    let c_idx = k.mad(m, N, n);
    k.store(c, res, c_idx);
    k.verify();

    // ---- the operands, and the reference they must reproduce ------------
    // Deterministic bytes on the host: the super-block layout the checkpoint
    // stores, dequantized here with the same arithmetic the kernel runs, so a
    // mismatch is the kernel's and not the reference's.
    let blocks = usize::try_from(N * K / BLOCK).expect("block count fits usize");
    let mut raw = vec![0u8; blocks * usize::try_from(ROW).expect("row fits usize")];
    let mut state = 0x2545_f491u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for b in 0..blocks {
        // A finite, non-zero scale: PQ2_0 codes only reach -1..1, so the scale
        // is what the projection's magnitude comes from.
        let scale = 0.01f32 + (b % 97) as f32 * 0.001;
        let bits = f16::from_f32(scale).0;
        raw[b * 34] = (bits & 0xff) as u8;
        raw[b * 34 + 1] = (bits >> 8) as u8;
        for j in 0..32 {
            raw[b * 34 + 2 + j] = next();
        }
    }

    let mut b_host = vec![0f32; usize::try_from(N * K).expect("weight fits usize")];
    for (elem, out) in b_host.iter_mut().enumerate() {
        let blk = elem / 128;
        let off = elem % 128;
        let base = blk * 34;
        let bits = u16::from(raw[base]) | (u16::from(raw[base + 1]) << 8);
        let scale = f16(bits).to_f32();
        let code = (raw[base + 2 + off / 4] >> (2 * (off % 4))) & 3;
        let val = (f32::from(code) - 1.0) * scale;
        // The kernel rounds the dequantized weight to f16 before the multiply.
        *out = f16::from_f32(val).to_f32();
    }
    let a_host: Vec<f32> = (0..usize::try_from(M * K).expect("activation fits usize"))
        .map(|i| f16::from_f32(((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).to_f32())
        .collect();

    // `sgemm` is row-major `a @ b.T`: A is the m x k `a` (row stride k), B is
    // k x n laid out as the row-major n x k weights (row stride 1, column
    // stride k).
    let mut want = vec![0f32; usize::try_from(M * N).expect("output fits usize")];
    unsafe {
        matrixmultiply::sgemm(
            usize::try_from(M).unwrap(),
            usize::try_from(K).unwrap(),
            usize::try_from(N).unwrap(),
            1.0,
            a_host.as_ptr(),
            K as isize,
            1,
            b_host.as_ptr(),
            1,
            K as isize,
            0.0,
            want.as_mut_ptr(),
            N as isize,
            1,
        )
    };

    // ---- the selected optimization -------------------------------------
    // Same as the llama test: the tensorization space is the whole thing in one
    // pass, and the reference config is applied to the seed. For this kernel
    // the space is still empty — the region matcher requires `mul(load, load)`
    // and the projection's multiplier is the PQ2_0 dequant chain — so the
    // assert states the gap the pass work has to close.
    let recipe = k.opt_tensorize();
    assert!(
        REFERENCE_TILE < recipe.nconfigs(),
        "the tensorization space must contain the reference tile: {} configs",
        recipe.nconfigs()
    );
    recipe.apply(&mut k, REFERENCE_TILE);
    k.default_epilogue();
    k.verify();

    // ---- run -----------------------------------------------------------
    let a: Tensor = Tensor::from_vec(a_host.iter().map(|&x| f16::from_f32(x)).collect::<Vec<f16>>(), [M, K])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(raw, [N * K / BLOCK, ROW])?.to(dev)?;
    // The timing is the search's device-side measurement: one warmup and one
    // timed launch, bracketed by timing events, not host wall time around a
    // queue of launches. The values come from a copy compiled on its own.
    let scratch = Tensor::zeros([M, N], DType::F32).to(dev)?;
    let values = k.clone();
    let (_winner, nanos) = BeamSearch::new().run([k], &[&a, &w, &scratch], &[], |_| {}, |_| 0)?;
    let compiled = values.compile()?;
    let out = compiled.forward(&[&a, &w], vec![[M, N]])?.pop().unwrap();
    let got: Vec<f32> = out.try_into()?;
    let nanos = nanos as f64;

    let flop = 2.0 * (M * N * K) as f64;
    // The weight is read in its packed form: every 34-byte block covers 128
    // weights, and the kernel loads one word and two scale bytes per weight.
    let bytes = (M * K) as f64 * 2.0 + (N * K / BLOCK * ROW) as f64 + (M * N) as f64 * 4.0;
    println!(
        "qwen ffn_gate {M}x{K}x{N}: {:.4} ms, {:.2} GFLOP/s, {:.1} GB/s (naive, dequant inline)",
        nanos * 1e-6,
        flop / nanos,
        bytes / nanos
    );

    // ---- the kernel must reproduce the reference ------------------------
    // Two checks. First an exact one on a sample of outputs, computed the way
    // the kernel computes: f16 weights, f16 products, f32 accumulation. That
    // is the arithmetic that decides correctness, so it is checked to f32
    // summation order.
    let n_usize = usize::try_from(N).unwrap();
    let sample = [
        (0usize, 0usize),
        (0, 57),
        (1, 100),
        (17, 4095),
        (63, 9000),
        (127, n_usize - 1),
    ];
    for &(i, j) in &sample {
        let mut exact = 0f32;
        for kk in 0..usize::try_from(K).unwrap() {
            let wv = b_host[j * usize::try_from(K).unwrap() + kk];
            let av = a_host[i * usize::try_from(K).unwrap() + kk];
            exact += f16::from_f32(wv * av).to_f32();
        }
        let got = got[i * n_usize + j];
        let err = (got - exact).abs();
        let tol = 1e-4f32.max(1e-4 * exact.abs());
        if err > tol {
            return Err(ZyxError::kernel_error(
                format!("ffn_gate disagrees with the f16-product reference at (m={i}, n={j}): {got} vs {exact}, tol {tol}")
                    .into(),
            ));
        }
    }
    // Then every element against the host GEMM. `sgemm` keeps each product in
    // f32 where the kernel rounds it to f16, so with PQ2_0's {-1,0,1} codes a
    // sum that cancels (|sum| << sum|prod|) carries that rounding as absolute
    // error: the bound is the f16 ulp of the sum of |products|, a few 1e-2
    // here. Layout, scale or index bugs are O(1), far outside it.
    let mut worst = (0f32, 0i64, 0i64);
    for i in 0..usize::try_from(M).unwrap() {
        for j in 0..n_usize {
            let got = got[i * n_usize + j];
            let want = want[i * n_usize + j];
            let err = (got - want).abs();
            let tol = 5e-2f32.max(2e-3 * want.abs());
            if err > tol {
                return Err(ZyxError::kernel_error(
                    format!("ffn_gate disagrees with the host GEMM at (m={i}, n={j}): {got} vs {want}, tol {tol}").into(),
                ));
            }
            if err > worst.0 {
                worst = (err, i as i64, j as i64);
            }
        }
    }
    println!("ffn_gate matches the host GEMM, max |err| = {:.3e} at (m={}, n={})", worst.0, worst.1, worst.2);
    Ok(())
}

// ===========================================================================
// The int8 tensor-core kernel
// ===========================================================================
//
// sm_75 (the RTX 2060) has no `m16n8k16`/`m16n8k32` int8 mma — those are
// Ampere — but it does have `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`,
// which is what llama.cpp's `mmq` runs on Turing. This is the reference kernel
// in that shape, written against the custom kernel API:
//
// - every weight is dequantized ONCE into the block's shared memory as int8
//   (`code - 1`, the exact integer payload of a PQ2_0 super-block, so the
//   tensor core sums integers with no rounding),
// - the activations are quantized to int8 per 32-element group (`q8_1`) in a
//   pre-pass, so both mma operands are int8,
// - the per-32 activation scale and the per-128 weight scale are applied to
//   the int32 accumulator in the epilogue, in f32.
//
// The fills and the fragment accesses are written as unfold (`vector`) axes
// and `compile` runs `devectorize` over them, exactly as the passes' output
// does.

/// One int8 activation scale per this many elements.
const QG: i64 = 32;
/// Activation groups per row (`K / QG`).
const KGROUP: i64 = K / QG;
/// Weight super-blocks per row (`K / BLOCK`).
const KBLK: i64 = K / BLOCK;

/// Quantize the fp16 activations to `q8_1`: one int8 scale per
/// [`QG`]-element group and one int8 value per activation element.
fn quantize_q8_kernel(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let a = k.param(DType::F16);
    let q8 = k.param_mut(DType::U8);
    let sa = k.param_mut(DType::F32);
    let row = k.group_range(0, M);
    let g = k.local_range(0, u32::try_from(KGROUP).expect("groups fit u32"));
    let kg = k.const_val(K);
    let cg = k.const_val(QG);
    let gk = k.mul(g, cg);
    let base = k.mad(row, kg, gk);
    // The 32 values of the group are one lane: one load, one maximum, one
    // quantize, one store of the group's bytes. Only the bit packing that
    // crosses lanes (in the GEMM's fill) stays serial.
    let l32 = k.unroll(1, 32);
    let vidx = k.add(base, l32);
    let raw = k.load(a, vidx);
    let vals = k.cast(raw, DType::F32);
    let av = k.abs(vals);
    let amax = k.reduce_max(av, l32);
    // `d = amax / 127`; an all-zero group quantizes to zeros rather than to a
    // division by zero (the same guard llama.cpp's quantizer has).
    let hundred = k.const_val(127.0f32);
    let zero = k.const_val(0f32);
    let cond = k.equal(amax, zero);
    let scaled = k.div(amax, hundred);
    let one_f = k.const_val(1.0f32);
    let scale = k.select(cond, one_f, scaled);
    let ckg = k.const_val(KGROUP);
    let qidx = k.mad(row, ckg, g);
    k.store(sa, scale, qidx);
    let qf = k.div(vals, scale);
    let q = k.cast(qf, DType::I32);
    let qb = k.cast(q, DType::U8);
    k.store(q8, qb, vidx);
    k.verify();
    k
}

/// The `ffn_gate` projection on int8 tensor cores: `out[M, N] = q8[M, K] @
/// w[N, K]^T`, with the PQ2_0 weights dequantized to int8 into shared memory
/// and the group scales applied to the integer accumulator.
///
/// Block: 512 threads (16 warps) over 128 rows x 64 columns, so the block's
/// shared weight tile is filled from global once and every warp reads its own
/// 16x32 window of it. A warp holds 16 rows x 32 columns: two `m8n8k16` A
/// fragments (rows 0-7 and 8-15) and four B fragments (columns 0-7, 8-15,
/// 16-23, 24-31), so one 16-wide K step is eight mma instructions.
fn gemm_s8_kernel(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let a8 = k.param(DType::U8);
    let sa = k.param(DType::F32);
    let w = k.param(DType::U8);
    let out = k.param_mut(DType::F32);

    let nblk = k.group_range(0, N / 64);
    let lane = k.local_range(0, 32);
    let warp = k.local_range(1, 16);
    let tid = k.mad(warp, 32i64, lane);
    let c1 = k.const_val(1i64);
    let c2 = k.const_val(2i64);
    let c4 = k.const_val(4i64);
    let c8 = k.const_val(8i64);
    let c16 = k.const_val(16i64);
    let c32 = k.const_val(32i64);
    let c64 = k.const_val(64i64);
    // Warp window: `mt` in 0..8 picks the 16-row band, `nt` in 0..2 the
    // 32-column band inside the block's 64 columns.
    let mt = k.bit_shift_right(warp, c1);
    let nt = k.bit_and(warp, c1);
    let m0 = k.mul(mt, c16);
    let nw = k.mul(nt, c32);
    let nbase = k.mul(nblk, c64);
    let n0 = k.add(nbase, nw);
    // mma fragment geometry: `gid = lane/4` is the fragment row (A's row, B's
    // column), `tig = lane%4` the 4-wide K group the lane holds.
    let gid = k.div(lane, c4);
    let tig = k.mod_(lane, c4);
    let tig2 = k.mul(tig, c2);
    // The parallel work as unfold lanes: the eight (A fragment, B fragment)
    // pairs one K step runs (axis 3) and the two C words of a fragment
    // (axis 2). The serial parts — super-block, 32-element group, K step —
    // stay loops and the accumulator chains through them.
    let combo = k.unroll(3, 8);
    let ai = k.div(combo, c4);
    let bj = k.mod_(combo, c4);
    let sl = k.unroll(2, 2);

    // Shared tiles, both holding the int8 mma's own operand word: 128 rows x
    // 32 packed activation words of the super-block, and the 64 weight rows'
    // 32 code bytes as their 32 int8 payload words (`code - 1`).
    let a_sh = k.storage(DType::U32, MemScope::Local, 128 * 32);
    let w_sh = k.storage(DType::U32, MemScope::Local, 64 * 32);
    // Accumulators: two int32 words per (m, n) fragment, and the fp16-free f32
    // sum they are scaled into.
    let accs = k.storage(DType::I32, MemScope::Register, 16);
    let accf = k.storage(DType::F32, MemScope::Register, 16);
    // The lane's weight scales for this super-block: four n fragments x the
    // two columns its C fragment covers.
    let swc = k.storage(DType::F32, MemScope::Register, 8);
    // One zero op for every accumulator write: the same value, so a 2-slot
    // load of freshly cleared accumulators forwards to it instead of becoming
    // an index into two separate constants.
    let zero_i = k.const_val(0i32);
    let zero_f = k.const_val(0f32);
    // Zeroing is parallel work: one store over a lane of all sixteen
    // accumulator words, unfolded by the device epilogue.
    let init = k.unroll(1, 16);
    k.store(accs, zero_i, init);
    k.store(accf, zero_f, init);

    let kgroup = k.const_val(KGROUP);
    let kblk = k.const_val(KBLK);
    let nstr = k.const_val(N);
    let row = k.const_val(ROW);
    let k128 = k.const_val(BLOCK);
    let kstr = k.const_val(K);

    k.loop_over(K / BLOCK, |k, c| {
        // ---- fill both tiles for the 128-element super-block `c` ----
        // Activations: 4096 packed words, eight per thread as two 4-word
        // vector accesses (a row of the tile is the super-block's 32 words of
        // the `[M, K/4]` view).
        // Activations: 4096 packed words from the `[M, K]` int8 view, eight per
        // thread. Both parallel axes are lanes: the eight words a thread fills,
        // and the four bytes of each word, folded into the operand word by a
        // lane reduction (the bytes occupy disjoint bit fields, so the sum is
        // the packed word).
        let chunk = k.mul(c, k128);
        {
            let w8 = k.unroll(1, 8);
            let b4 = k.unroll(2, 4);
            let tb = k.mul(tid, c8);
            let off = k.add(tb, w8);
            let r = k.div(off, c32);
            let cw = k.mod_(off, c32);
            let cw4 = k.mul(cw, c4);
            let gk = k.add(chunk, cw4);
            let gi = k.mad(r, kstr, gk);
            let bi = k.add(gi, b4);
            let byte = k.load(a8, bi);
            let byte32 = k.cast(byte, DType::U32);
            let eight = k.const_val(8u32);
            let b4u = k.cast(b4, DType::U32);
            let sh = k.mul(b4u, eight);
            let shifted = k.bit_shift_left(byte32, sh);
            let word = k.reduce_sum(shifted, b4);
            let sr = k.mul(r, c32);
            let si = k.add(sr, cw);
            k.store(a_sh, word, si);
        }
        // Weights: the super-block's 32 code bytes per row, four per thread.
        // Each code byte is four int8 payloads (`code - 1`), so the tile holds
        // the dequantized operand word, not the packed form. The unpacked
        // address is the super-block's own: `(n*KBLK + c)*34 + 2 + code_byte`,
        // the block's first two bytes being its f16 scale.
        let iw = k.unroll(1, 4);
        {
            let tb = k.mul(tid, c4);
            let e = k.add(tb, iw);
            let r = k.div(e, c32);
            let cb = k.mod_(e, c32);
            let wr = k.add(nbase, r);
            let blk = k.mul(wr, kblk);
            let blk = k.add(blk, c);
            let bi = k.mul(blk, row);
            let two = k.const_val(2i64);
            let gi = k.add(bi, two);
            let gi = k.add(gi, cb);
            let byte = k.load(w, gi);
            let mut word = k.const_val(0u32);
            for lane_i in 0..4u32 {
                let sh = k.const_val((2 * lane_i) as u8);
                let t = k.bit_shift_right(byte, sh);
                let mask = k.const_val(3u8);
                let q = k.bit_and(t, mask);
                let qw = k.cast(q, DType::U32);
                let one = k.const_val(1u32);
                let payload = k.sub(qw, one);
                let bits = k.const_val(0xffu32);
                let masked = k.bit_and(payload, bits);
                let shr = k.const_val(8 * lane_i);
                let shifted = k.bit_shift_left(masked, shr);
                word = k.bit_or(word, shifted);
            }
            let sr = k.mul(r, c32);
            let si = k.add(sr, cb);
            k.store(w_sh, word, si);
        }
        // The lane's weight scales for this super-block, hoisted out of the
        // group loop: the super-block's first two bytes are its f16 scale, one
        // per (n fragment, C column) — eight of them, one lane.
        {
            let sc = k.unroll(1, 8);
            let jj = k.div(sc, c2);
            let cc = k.mod_(sc, c2);
            let jo = k.mul(jj, c8);
            let co = k.add(tig2, cc);
            let ni = k.add(jo, co);
            let n = k.add(n0, ni);
            let blk = k.mad(n, kblk, c);
            let bi = k.mul(blk, row);
            let lo = k.load(w, bi);
            let bi1 = k.add(bi, c1);
            let hi = k.load(w, bi1);
            let lo16 = k.cast(lo, DType::U16);
            let hi16 = k.cast(hi, DType::U16);
            let eight = k.const_val(8u16);
            let hs = k.bit_shift_left(hi16, eight);
            let bits = k.bit_or(lo16, hs);
            let bf16 = k.bitcast(bits, DType::F16);
            let f = k.cast(bf16, DType::F32);
            k.store(swc, f, sc);
        }
        k.barrier();

        // ---- four 32-element groups of two 16-wide K steps ----
        for g in 0..(BLOCK / QG) {
            for s in 0..(QG / 16) {
                // The lane's operand word: four k values, the super-block's
                // 32 words per row. The 8 (A fragment, B fragment) pairs of
                // one K step are independent work, so they are one lane and
                // the epilogue unfolds it into the eight instructions.
                let ko = k.const_val(g * 8 + s * 4);
                let wo = k.add(ko, tig);
                let ar = k.mad(ai, c8, gid);
                let r = k.add(m0, ar);
                let ri = k.mul(r, c32);
                let aw = k.add(ri, wo);
                let a_frag = k.load(a_sh, aw);
                let jr = k.mad(bj, c8, gid);
                let nr = k.mul(nt, c32);
                let bn = k.add(nr, jr);
                let bi = k.mul(bn, c32);
                let bw = k.add(bi, wo);
                let b_frag = k.load(w_sh, bw);
                // The two C words of a fragment are one lane too.
                let slot_c = k.mad(combo, c2, sl);
                let c_reg = k.load(accs, slot_c);
                let d = k.wmma(
                    zyx::kernel::MMADims::m8n8k16,
                    zyx::kernel::MMALayout::row_col,
                    zyx::kernel::MMADType::s8_s8_s32_s32,
                    a_frag,
                    b_frag,
                    c_reg,
                );
                k.store(accs, d, slot_c);
            }
            // Scale this group's integer sums into the f32 accumulator and
            // clear them for the next group: `sum(int) * sa[m, g] * sw[n, g/4]`.
            let cpg = k.const_val(BLOCK / QG);
            let gbase = k.mul(c, cpg);
            let gi_off = k.const_val(g as i64);
            let gin = k.add(gbase, gi_off);
            let ar = k.mad(ai, c8, gid);
            let r = k.add(m0, ar);
            let sai = k.mad(r, kgroup, gin);
            let sav = k.load(sa, sai);
            let swi = k.mad(bj, c2, sl);
            let swv = k.load(swc, swi);
            let scale = k.mul(sav, swv);
            let slot_c = k.mad(combo, c2, sl);
            let iv = k.load(accs, slot_c);
            let fv = k.cast(iv, DType::F32);
            let prev = k.load(accf, slot_c);
            let sum = k.mad(fv, scale, prev);
            k.store(accf, sum, slot_c);
            k.store(accs, zero_i, slot_c);
        }
        // The next super-block overwrites both tiles.
        k.barrier();
    });

    // ---- scatter: C's two columns per lane, one n fragment at a time ----
    {
        let ar = k.mad(ai, c8, gid);
        let r = k.add(m0, ar);
        let slot_c = k.mad(combo, c2, sl);
        let v = k.load(accf, slot_c);
        let jo = k.mul(bj, c8);
        let co = k.add(tig2, sl);
        let no = k.add(jo, co);
        let n = k.add(n0, no);
        let oi = k.mad(r, nstr, n);
        k.store(out, v, oi);
    }
    // `c8` stays bound for the scale table's step.
    k.verify();
    k
}

/// The int8-tensor-core `ffn_gate`: the weight is dequantized into shared
/// memory as int8 (`code - 1`), the activations are quantized per 32, and the
/// integer products go through `mma.m8n8k16.s8` (Turing's int8 tensor core).
/// The reference folds the same per-group scales into the operands and runs a
/// host GEMM on the quantized values, so the only difference from the kernel
/// is f32 summation order.
#[test]
fn qwen_ffn_gate_s8() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };

    // ---- operands: the checkpoint's packed weight, fp16 activations ----
    let blocks = usize::try_from(N * K / BLOCK).expect("block count fits usize");
    let mut raw = vec![0u8; blocks * usize::try_from(ROW).expect("row fits usize")];
    let mut state = 0x2545_f491u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for b in 0..blocks {
        let scale = 0.01f32 + (b % 97) as f32 * 0.001;
        let bits = f16::from_f32(scale).0;
        raw[b * 34] = (bits & 0xff) as u8;
        raw[b * 34 + 1] = (bits >> 8) as u8;
        for j in 0..32 {
            raw[b * 34 + 2 + j] = next();
        }
    }
    // The activations, quantized here exactly as the pre-pass does.
    let a_host: Vec<f16> = (0..usize::try_from(M * K).expect("activation fits usize"))
        .map(|i| f16::from_f32(((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5))
        .collect();
    let mut q8_host = vec![0u8; usize::try_from(M * K).expect("activation fits usize")];
    let mut sa_host = vec![0f32; usize::try_from(M * KGROUP).expect("scales fit usize")];
    for m in 0..usize::try_from(M).unwrap() {
        for g in 0..usize::try_from(KGROUP).unwrap() {
            let base = m * usize::try_from(K).unwrap() + g * 32;
            let mut amax = 0f32;
            for i in 0..32 {
                amax = amax.max(a_host[base + i].to_f32().abs());
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            sa_host[m * usize::try_from(KGROUP).unwrap() + g] = scale;
            for i in 0..32 {
                let v = a_host[base + i].to_f32();
                // `cvt.rni` (the kernel's cast) rounds ties to even.
                let q = (v / scale).round_ties_even() as i32;
                q8_host[base + i] = (q & 0xff) as u8;
            }
        }
    }

    // ---- reference: the same quantized operands, scales folded in ----
    // Every 64th output column is checked: 128 x 272 outputs over the whole K.
    const STRIDE: i64 = 64;
    let cols: Vec<i64> = (0..N).step_by(usize::try_from(STRIDE).unwrap()).collect();
    let ncol = cols.len();
    let mut aref = vec![0f32; usize::try_from(M * K).unwrap()];
    for m in 0..usize::try_from(M).unwrap() {
        for kk in 0..usize::try_from(K).unwrap() {
            let byte = q8_host[m * usize::try_from(K).unwrap() + kk];
            let q = f32::from(i8::from_ne_bytes([byte]));
            aref[m * usize::try_from(K).unwrap() + kk] = q * sa_host[m * usize::try_from(KGROUP).unwrap() + kk / 32];
        }
    }
    let mut bref = vec![0f32; ncol * usize::try_from(K).unwrap()];
    for (s, &n) in cols.iter().enumerate() {
        for kk in 0..usize::try_from(K).unwrap() {
            let blk = (n * K + kk as i64) / BLOCK;
            let off = (n * K + kk as i64) % BLOCK;
            let base = usize::try_from(blk * ROW).unwrap();
            let bits = u16::from(raw[base]) | (u16::from(raw[base + 1]) << 8);
            let sw = f16(bits).to_f32();
            let code = (raw[base + 2 + usize::try_from(off).unwrap() / 4] >> (2 * (off % 4) as u32)) & 3;
            bref[s * usize::try_from(K).unwrap() + kk] = (f32::from(code) - 1.0) * sw;
        }
    }
    let mut want = vec![0f32; usize::try_from(M).unwrap() * ncol];
    unsafe {
        matrixmultiply::sgemm(
            usize::try_from(M).unwrap(),
            usize::try_from(K).unwrap(),
            ncol,
            1.0,
            aref.as_ptr(),
            K as isize,
            1,
            bref.as_ptr(),
            1,
            K as isize,
            0.0,
            want.as_mut_ptr(),
            ncol as isize,
            1,
        )
    };

    // ---- run the two kernels ----
    let a: Tensor = Tensor::from_vec(a_host.clone(), [M, K])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(raw, [N * K / BLOCK, ROW])?.to(dev)?;
    let out_t: Tensor = Tensor::from_vec(vec![0f32; usize::try_from(M * N).unwrap()], [M, N])?.to(dev)?;
    let a8_t: Tensor = Tensor::from_vec(q8_host.clone(), [M, K])?.to(dev)?;
    let sa_t: Tensor = Tensor::from_vec(sa_host.clone(), [M, KGROUP])?.to(dev)?;

    let q8_kernel = quantize_q8_kernel(dev).compile()?;
    let started = Instant::now();
    let q8_out = q8_kernel.forward(&[&a], vec![[M, K], [M, KGROUP]])?;
    let q8_dev = &q8_out[0];
    let sa_dev = &q8_out[1];
    // The pre-pass must produce the values the host reference quantized; the
    // read-back is inside the timer so the number is wall time, not a launch.
    let q8_got: Vec<u8> = q8_dev.clone().try_into()?;
    let sa_got: Vec<f32> = sa_dev.clone().try_into()?;
    let quant_nanos = started.elapsed().as_secs_f64() * 1e9;
    for (i, (g, h)) in q8_got.iter().zip(q8_host.iter()).enumerate() {
        if g != h {
            return Err(ZyxError::kernel_error(format!("q8_act word {i}: {g:#x} != {h:#x}").into()));
        }
    }
    for (i, (g, h)) in sa_got.iter().zip(sa_host.iter()).enumerate() {
        if (g - h).abs() > 1e-6 * h.abs().max(1.0) {
            return Err(ZyxError::kernel_error(format!("q8_act scale {i}: {g} != {h}").into()));
        }
    }

    // ---- the kernel, and the one pass it gets --------------------------
    // The structure above is the tensorized shape, written by hand with unfold
    // lanes: no tensorization, no recipe. The only pass is the device epilogue's
    // lane expansion, so what runs is exactly what was written.
    let gemm = gemm_s8_kernel(dev);
    gemm.verify();

    // The timing is the search's device-side measurement, the same one the
    // first test uses: one warmup and one timed launch bracketed by timing
    // events. The values come from a copy compiled on its own.
    let scratch = Tensor::zeros([M, N], DType::F32).to(dev)?;
    let values = gemm.clone();
    let (_winner, nanos) = BeamSearch::new().run([gemm], &[&a8_t, &sa_t, &w, &scratch], &[], |_| {}, |_| 0)?;
    let gemm_nanos = nanos as f64;
    let compiled = values.compile()?;
    let out = compiled.forward(&[&a8_t, &sa_t, &w], vec![[M, N]])?.pop().unwrap();

    // ---- the sampled outputs must match the reference ----
    let got: Vec<f32> = out.try_into()?;
    let mut worst = (0f32, 0i64, 0i64);
    for m in 0..usize::try_from(M).unwrap() {
        for (s, &n) in cols.iter().enumerate() {
            let got = got[m * usize::try_from(N).unwrap() + usize::try_from(n).unwrap()];
            let want = want[m * ncol + s];
            let err = (got - want).abs();
            let tol = 2e-3f32.max(2e-3 * want.abs());
            if err > tol {
                return Err(ZyxError::kernel_error(
                    format!("ffn_gate s8 disagrees at (m={m}, n={n}): {got} vs {want}, tol {tol}").into(),
                ));
            }
            if err > worst.0 {
                worst = (err, m as i64, n);
            }
        }
    }
    println!("ffn_gate s8 matches the int8 reference, max |err| = {:.3e} at (m={}, n={})", worst.0, worst.1, worst.2);
    let flop = 2.0 * (M * N * K) as f64;
    println!(
        "qwen ffn_gate {M}x{K}x{N} int8: q8_act {:.4} ms, gemm {:.4} ms, {:.2} TFLOP/s",
        quant_nanos * 1e-6,
        gemm_nanos * 1e-6,
        flop / gemm_nanos * 1e-3
    );
    let _ = out_t;
    Ok(())
}
