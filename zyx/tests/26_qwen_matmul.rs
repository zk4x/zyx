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
//! word  = w16[blk*17 + 1 + off/8]    // 8 two-bit codes per u16 word
//! code  = (word >> 2*(off%8)) & 3
//! scale = bits(f16) w16[blk*17]      // the block's first two bytes
//! w     = (f32(code) - 1) * scale    // then rounded to f16
//! acc  += f32(f16(w) * f16(act[k]))
//! ```
//!
//! The operands are taken in the width the kernel reads them: the weight is
//! declared `U16` (the 34-byte super-block as its 17 words) and the q8_1
//! payload `U32` (four int8 per word). The caller supplies those views; the
//! kernel does no implicit retyping.
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
const REFERENCE_TILE: u64 = 538;

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
    // The operands are taken in the width the kernel reads them: the q8_1
    // payload as u32 words (four int8 each) and the PQ2_0 super-block as its
    // 17 u16 words (the f16 scale, then the 32 code bytes). There is no
    // implicit retyping — the caller supplies the word views.
    let a8 = k.param(DType::U32);
    let sa = k.param(DType::F32);
    let w = k.param(DType::U16);
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
        // The 34-byte super-block as its 17 u16 words: word 0 is the f16
        // scale, words 1..17 the 32 code bytes, eight 2-bit codes per word.
        let c17 = k.const_val(17i64);
        let wrow = k.mul(blk, c17);
        let two = k.const_val(2i64);
        let eight_i = k.const_val(8i64);
        let off_word = k.div(off, eight_i);
        let word_in_row = k.add(1i64, off_word);
        let word_idx = k.add(wrow, word_in_row);
        let word = k.load(w, word_idx);
        let off_mod = k.mod_(off, eight_i);
        let shift = k.mul(off_mod, two);
        // The shift amount is an operand of the word's own dtype (u16), as the
        // linearized dequant chain shifts by.
        let shift = k.cast(shift, DType::U16);
        let shifted = k.bit_shift_right(word, shift);
        let mask = k.const_val(3u16);
        let code = k.bit_and(shifted, mask);
        // The block's scale: little-endian f16 in the super-block's first two
        // bytes (`f16le` in the dequantize lowering), one u16 load.
        let bits = k.load(w, wrow);
        let scale_f16 = k.bitcast(bits, DType::F16);
        let scale = k.cast(scale_f16, DType::F32);
        let code_f32 = k.cast(code, DType::F32);
        let centered = k.sub(code_f32, 1.0f32);
        let val = k.mul(centered, scale);
        // The projection is an f16 x f16 multiply with an f32 accumulate.
        let val = k.cast(val, DType::F16);
        // The activation's own q8_1 dequant: one int8 value and the scale of
        // its 32-element group. The payload is read as u32 words, four int8
        // each; the element's byte is the word's `elem % 4`.
        let act_idx = k.mad(m, K, k_loop);
        let four = k.const_val(4i64);
        let word_i = k.div(act_idx, four);
        let aword = k.load(a8, word_i);
        let byte_off = k.mod_(act_idx, four);
        let bo32 = k.cast(byte_off, DType::U32);
        let eight_u = k.const_val(8u32);
        let sh = k.mul(bo32, eight_u);
        let bword = k.bit_shift_right(aword, sh);
        let bmask = k.const_val(0xffu32);
        let byte_bits = k.bit_and(bword, bmask);
        let byte_u = k.cast(byte_bits, DType::U8);
        let byte = k.bitcast(byte_u, DType::I8);
        let byte_f = k.cast(byte, DType::F32);
        let group = k.div(k_loop, QG);
        let kgroup = k.const_val(KGROUP);
        let sa_idx = k.mad(m, kgroup, group);
        let sa_v = k.load(sa, sa_idx);
        let act = k.mul(byte_f, sa_v);
        let act = k.cast(act, DType::F16);
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
    // The activations, quantized to q8_1 on the host exactly as the pre-pass
    // does (one int8 value per element, one f32 scale per 32-element group),
    // then dequantized back: that pair is the projection's operand, so the
    // reference multiplies what the kernel multiplies.
    let a_f16: Vec<f16> = (0..usize::try_from(M * K).expect("activation fits usize"))
        .map(|i| f16::from_f32(((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5))
        .collect();
    let mut q8_host = vec![0i8; usize::try_from(M * K).expect("activation fits usize")];
    let mut sa_host = vec![0f32; usize::try_from(M * KGROUP).expect("scales fit usize")];
    for row in 0..usize::try_from(M).unwrap() {
        for g in 0..usize::try_from(KGROUP).unwrap() {
            let base = row * usize::try_from(K).unwrap() + g * 32;
            let mut amax = 0f32;
            for i in 0..32 {
                amax = amax.max(a_f16[base + i].to_f32().abs());
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            sa_host[row * usize::try_from(KGROUP).unwrap() + g] = scale;
            for i in 0..32 {
                let v = a_f16[base + i].to_f32();
                // `cvt.rni` (the kernel's cast) rounds ties to even.
                let q = (v / scale).round_ties_even() as i32;
                q8_host[base + i] = q as i8;
            }
        }
    }
    // The f16 operand the f16 form multiplies: the dequantized q8 value.
    let a_host: Vec<f32> = (0..usize::try_from(M * K).expect("activation fits usize"))
        .map(|i| {
            let row = i / usize::try_from(K).unwrap();
            f16::from_f32(
                f32::from(q8_host[i]) * sa_host[row * usize::try_from(KGROUP).unwrap() + (i % usize::try_from(K).unwrap()) / 32],
            )
            .to_f32()
        })
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
    // The tensorization space is the whole thing in one pass: for a matmul
    // whose operands are dequant chains it holds both forms the int8-era
    // tensor core offers — the f16 one, which evaluates each chain into an f16
    // staged tile, and the int8 one, which stages the chains' codes with their
    // per-32 scales — and the search decides between them. This config is the
    // reference kernel's shape, and it must be one of the options.
    let t0 = std::time::Instant::now();
    let recipe = k.opt_tensorize();
    eprintln!("T opt_tensorize {} ms", t0.elapsed().as_millis());
    assert!(
        REFERENCE_TILE < recipe.nconfigs(),
        "the tensorization space must contain the reference tile: {} configs",
        recipe.nconfigs()
    );
    recipe.apply(&mut k, REFERENCE_TILE);
    eprintln!("T apply {} ms", t0.elapsed().as_millis());
    k.default_epilogue();
    eprintln!("T epilogue {} ms", t0.elapsed().as_millis());
    k.verify();

    // ---- run -----------------------------------------------------------
    // The q8_1 pre-pass first: llama.cpp's quantizer is its own kernel, and the
    // projection consumes its two outputs as ordinary inputs.
    let a: Tensor = Tensor::from_vec(a_f16.clone(), [M, K])?.to(dev)?;
    let q8_kernel = quantize_q8_kernel(dev).compile()?;
    let q8_out = q8_kernel.forward(&[&a], vec![[M, K], [M, KGROUP]])?;
    let (q8_t, sa_t) = (&q8_out[0], &q8_out[1]);
    let q8_got: Vec<i8> = q8_t.clone().try_into()?;
    let sa_got: Vec<f32> = sa_t.clone().try_into()?;
    for (i, (g, h)) in q8_got.iter().zip(q8_host.iter()).enumerate() {
        if g != h {
            return Err(ZyxError::kernel_error(format!("q8 value {i}: {g} != {h}").into()));
        }
    }
    for (i, (g, h)) in sa_got.iter().zip(sa_host.iter()).enumerate() {
        if (g - h).abs() > 1e-6 * h.abs().max(1.0) {
            return Err(ZyxError::kernel_error(format!("q8 scale {i}: {g} != {h}").into()));
        }
    }
    // The word views the projection takes: the q8_1 payload as u32 words (four
    // int8 each) and the PQ2_0 super-block as 17 u16 words. The kernel does no
    // implicit retyping, so the caller packs them here.
    let a32: Vec<u32> =
        q8_host.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
    let w16: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let a32_t: Tensor = Tensor::from_vec(a32, [M * K / 4])?.to(dev)?;
    let w16_t: Tensor = Tensor::from_vec(w16, [N * K / BLOCK * 17])?.to(dev)?;
    // The timing is the search's device-side measurement: one warmup and one
    // timed launch, bracketed by timing events, not host wall time around a
    // queue of launches. The values come from a copy compiled on its own.
    let scratch = Tensor::zeros([M, N], DType::F32).to(dev)?;
    let values = k.clone();
    let (_winner, nanos) = BeamSearch::new().run([k], &[&a32_t, &sa_t, &w16_t, &scratch], &[], |_| {}, |_| 0)?;
    let compiled = values.compile()?;
    let out = compiled.forward(&[&a32_t, &sa_t, &w16_t], vec![[M, N]])?.pop().unwrap();
    let got: Vec<f32> = out.try_into()?;
    let nanos = nanos as f64;

    let flop = 2.0 * (M * N * K) as f64;
    // The packed operands: the activations are int8 with one f32 scale per 32,
    // the weight is read in its 34-byte blocks.
    let bytes = (M * K) as f64 + (M * KGROUP) as f64 * 4.0 + (N * K / BLOCK * ROW) as f64 + (M * N) as f64 * 4.0;
    println!(
        "qwen ffn_gate {M}x{K}x{N}: {:.4} ms, {:.2} GFLOP/s, {:.1} GB/s (tensorized, dequant fused)",
        nanos * 1e-6,
        flop / nanos,
        bytes / nanos
    );
    // Cold-clock floor: on an idle device the SMs sit below boost, the same
    // clock that bounds the llama gate's 17 TFLOP/s floor. The GB/s figure
    // rides the same launch and is printed, not gated.
    let tflops = flop / nanos / 1e3;
    assert!(tflops > 17.0, "ffn_gate must clear the cold-clock floor (17 TFLOP/s), got {tflops:.2}");

    // ---- the kernel must reproduce the reference ------------------------
    // The tensorized forms are not bit-identical to the scalar kernel: the f16
    // form sums f16 products in fragment order, the int8 form sums exact
    // integer products and scales each 32-group. Both round within an f16 ulp
    // (a few 1e-3 of the sum of |products|), which is the bound the host GEMM
    // below is compared at; layout, scale or index bugs are O(1), far outside
    // it.
    let n_usize = usize::try_from(N).unwrap();
    let mut worst = (0f32, 0i64, 0i64);
    for i in 0..usize::try_from(M).unwrap() {
        for j in 0..n_usize {
            let got = got[i * n_usize + j];
            let want = want[i * n_usize + j];
            let err = (got - want).abs();
            let tol = 5e-2f32.max(2e-3 * want.abs());
            if j < 8 && i < 8 {
                print!("{got:>9.4}");
            }
            if err > worst.0 {
                worst = (err, i as i64, j as i64);
            }
        }
        if i < 8 {
            print!("   |");
            for j in 0..8 {
                print!("{:>9.4}", want[i * n_usize + j]);
            }
            println!();
        }
    }
    println!("DBG worst |err| = {:.3e} at (m={}, n={})", worst.0, worst.1, worst.2);
    {
        let mut bad = 0;
        let mut rows = vec![0usize; usize::try_from(M).unwrap()];
        let mut cols = std::collections::HashMap::new();
        for i in 0..usize::try_from(M).unwrap() {
            for j in 0..n_usize {
                let got = got[i * n_usize + j];
                let want = want[i * n_usize + j];
                let tol = 5e-2f32.max(2e-3 * want.abs());
                if (got - want).abs() > tol {
                    bad += 1;
                    rows[i] += 1;
                    *cols.entry(j % 64).or_insert(0usize) += 1;
                }
            }
        }
        println!("DBG bad={bad} rows={:?}", rows.iter().enumerate().filter(|(_, c)| **c > 0).take(12).collect::<Vec<_>>());
        let mut by_col: Vec<_> = cols.into_iter().collect();
        by_col.sort_by_key(|&(c, n)| (std::cmp::Reverse(n), c));
        println!("DBG cols(n%64)={:?}", &by_col[..by_col.len().min(10)]);
    }
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
    let q8 = k.param_mut(DType::I8);
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
    let qb = k.cast(q, DType::I8);
    k.store(q8, qb, vidx);
    k.verify();
    k
}

/// The `ffn_gate` projection on int8 tensor cores: `out[M, N] = q8[M, K] @
/// w[N, K]^T`, with the PQ2_0 weights dequantized to int8 into shared memory
/// and the group scales applied to the integer sums.
///
/// The block is llama.cpp's `mmq` block for PQ2_0 on Turing, the shape its
/// ampere config table fixes (`I = 128`, `J = 128`, 256 threads): 128
/// activation rows by 128 weight rows with eight warps, each warp owning 64
/// activation rows (eight `m8n8k16` A fragments) by 32 weight rows (four B
/// fragments), so one 16-wide K step is 32 mma instructions per warp. As in
/// `mmq`'s `vec_dot_q8_0_q8_1_mma`, the int32 fragment is folded straight
/// into the f32 accumulator after each 32-element group
/// (`sum[l] += C.x[l]*dA*dB`): only the f32 sums stay live across the K loop,
/// the integer fragment is the instruction's own transient result.
fn gemm_s8_kernel(dev: Dev) -> Kernel {
    // Weight rows per block (llama.cpp's `I`) and activation rows per block
    // (`J`).
    const IB: i64 = 128;
    const JB: i64 = 128;
    // `mmq`'s PQ2_0 block config: 256 threads as 32x8.
    const NW: u32 = 8;
    // Words per row of the two shared tiles: the super-block's 32 payload
    // words, the four words of scales, then the 4-word (16-byte) tail that
    // keeps the rows of one `ldmatrix` off the same banks.
    const XS: i64 = 40;
    const YS: i64 = 36;

    let mut k = Kernel::new(dev);
    // The q8_1 payload is read as words (four int8 each) and the PQ2_0
    // super-block as 17 words (its f16 scale, then 32 code bytes) — the
    // pointer casts `mmq`'s fills make.
    let a8 = k.param(DType::U32);
    let sa = k.param(DType::F32);
    let w = k.param(DType::U16);
    let out = k.param_mut(DType::F32);

    let nb = k.group_range(0, N / IB);
    let lane = k.local_range(0, 32);
    let warp = k.local_range(1, NW);
    let tid = k.mad(warp, 32i64, lane);
    let c1 = k.const_val(1i64);
    let c2 = k.const_val(2i64);
    let c3 = k.const_val(3i64);
    let c4 = k.const_val(4i64);
    let c8 = k.const_val(8i64);
    let c16 = k.const_val(16i64);
    let c32 = k.const_val(32i64);
    let c64 = k.const_val(64i64);
    let c256 = k.const_val(256i64);
    let cxs = k.const_val(XS);
    let cys = k.const_val(YS);
    let c17 = k.const_val(17i64);
    let cib = k.const_val(IB);
    let ckb = k.const_val(KBLK);
    let ckg = k.const_val(KGROUP);
    let nstr = k.const_val(N);
    let kw = k.const_val(K / 4);
    let two = k.const_val(2u32);

    // The two shared tiles, staged as `mmq` stages them: one row per row of
    // the tile, the scales inside the row (`x_df` at word 32, `y_df` at word
    // 0), the 16-byte tail as padding.
    let xt = k.storage(DType::U32, MemScope::Local, IB * XS);
    let yt = k.storage(DType::U32, MemScope::Local, JB * YS);
    // `sum[64]`: two C words per fragment pair, live across the K loop.
    let accf = k.storage(DType::F32, MemScope::Register, 64);

    // Warp split and lane geometry: `warp/2` is the 32-row weight band,
    // `warp%2` the 8-row activation band inside each 16, `lane/4` the
    // fragment row, `lane%4` the four-int8 group of a 16-wide K step.
    let warpl = k.bit_and(warp, c1);
    let warph = k.bit_shift_right(warp, c1);
    let i0 = k.mul(warph, c32);
    let jo = k.mul(warpl, c8);
    let gid = k.div(lane, c4);
    let lanecol = k.mod_(lane, c4);
    let tig2 = k.mul(lanecol, c2);
    let lanerow = k.mod_(lane, c16);
    let lanehalf = k.div(lane, c16);
    // The in-place mma chain starts from zero and the sums from `0.0f`.
    let zero_i = k.const_val(0i32);
    let zero2 = k.stack(&[zero_i, zero_i]);
    let zero_f = k.const_val(0f32);
    let szero = k.unroll(7, 64);
    k.store(accf, zero_f, szero);

    k.loop_over(K / BLOCK, |k, kb| {
        // ---- `ggml_cuda_mmq_load_tiles_pq2_0`: the weight tile ----
        // `QI_PQ2_0 = QK_PQ2_0/32 = 4` threads per row, eight rows per warp,
        // 64 rows per pass: a thread reads the four 16-bit halves at its own
        // `kqsx * 4` (bytes `2 + kqsx*8`, inside the block's 32-byte `qs`) and
        // writes the eight words at `kqsx * 8`, so the four of them cover the
        // row's 32 words exactly. A thread turns each half into two int8 words
        // with three `prmt`s.
        {
            let it = k.unroll(0, 2);
            let j = k.unroll(1, 4);
            let lane4 = k.div(lane, c4);
            let row0 = k.mad(warp, c8, lane4);
            let step = k.mul(it, c64);
            let row = k.add(row0, step);
            let kq = k.mod_(lane, c4);
            let wrow = k.mad(nb, cib, row);
            let blk = k.mad(wrow, ckb, kb);
            let b17 = k.mul(blk, c17);
            let q0 = k.add(b17, c1);
            let qk = k.mul(kq, c4);
            let qbase = k.add(q0, qk);
            let qi = k.add(qbase, j);
            let qw = k.load(w, qi);
            // `const short * qxi` / `const int q = qxi[j]`: the half is read
            // signed (`cvt.s32.s16`), not zero-extended.
            let qs16 = k.bitcast(qw, DType::I16);
            let qi32 = k.cast(qs16, DType::I32);
            let q = k.bitcast(qi32, DType::U32);
            // `code -> code-1`: byte `i` of `0x020100FF` is `i-1`, and the
            // pair of codes in a nibble selects it (the high bit sign-extends,
            // which the table's bytes all survive).
            let table = k.const_val(0x0201_00FFu32);
            // `__byte_perm` has no sign-replicate mode, while `prmt`'s
            // selector bit 3 is sign-replicate: the compiler masks it off
            // (`and.b32 %r239, %r238, 30583`) before each inner `prmt`.
            let mask = k.const_val(0x7777u32);
            let qe_sel = k.bit_and(q, mask);
            let qe = k.prmt(table, table, qe_sel);
            let qsh = k.bit_shift_right(q, two);
            let qo_sel = k.bit_and(qsh, mask);
            let qo = k.prmt(table, table, qo_sel);
            let sel = k.const_val(0x5140u32);
            let qx = k.prmt(qe, qo, sel);
            let sel = k.const_val(0x7362u32);
            let qy = k.prmt(qe, qo, sel);
            let k8 = k.mul(kq, c8);
            let j2 = k.mul(j, c2);
            let col = k.add(k8, j2);
            let rowx = k.mul(row, cxs);
            let dst = k.add(rowx, col);
            k.store(xt, qx, dst);
            let dst1 = k.add(dst, c1);
            k.store(xt, qy, dst1);
        }
        // The block's f16 scale, one entry per row at word 32 of it.
        {
            let si = k.unroll(2, 16);
            let nw = k.const_val(i64::from(NW));
            let rows = k.mul(si, nw);
            let row = k.add(warp, rows);
            let wrow = k.mad(nb, cib, row);
            let blk = k.mad(wrow, ckb, kb);
            let b17 = k.mul(blk, c17);
            let d = k.load(w, b17);
            let d = k.bitcast(d, DType::F16);
            let d = k.cast(d, DType::F32);
            let d = k.bitcast(d, DType::U32);
            let rowx = k.mul(row, cxs);
            let at = k.add(rowx, c32);
            let at = k.add(at, lanecol);
            k.store(xt, d, at);
        }
        // ---- the y tile: `block_q8_1_mmq` is [four scales][32 words] ----
        {
            let w16 = k.unroll(3, 16);
            let m256 = k.mul(w16, c256);
            let off = k.add(tid, m256);
            let r = k.div(off, c32);
            let c = k.mod_(off, c32);
            let rw = k.mul(r, kw);
            let k32 = k.mul(kb, c32);
            let src = k.add(rw, k32);
            let src = k.add(src, c);
            let v = k.load(a8, src);
            let dst = k.mad(r, cys, c);
            let dst = k.add(dst, c4);
            k.store(yt, v, dst);
        }
        {
            let s2 = k.unroll(4, 2);
            let m256 = k.mul(s2, c256);
            let e = k.add(tid, m256);
            let r = k.div(e, c4);
            let g = k.mod_(e, c4);
            let k4 = k.mul(kb, c4);
            let gq = k.add(k4, g);
            let si = k.mad(r, ckg, gq);
            let v = k.load(sa, si);
            let v = k.bitcast(v, DType::U32);
            let dst = k.mad(r, cys, g);
            k.store(yt, v, dst);
        }
        k.barrier();

        // ---- A, the mma and the fold ----
        // The outer lane is one (sub-tile, K group): its `ldmatrix.x4` — four
        // 8x8 b16 matrices, 16 rows by the group's 32 int8 — and the two
        // weight scales of its C rows. The inner lane is the 16-row
        // activation band, `j0`: the two B words of the 8x8 `n x k` view
        // (`load_generic`), the two C columns' activation scales, the four
        // in-place `m8n8k16` of `mma_16x8_8x8`, and the fold.
        {
            // The K groups are a sequential chain: each group's fold reads the
            // sums the previous one wrote, so they are not a parallel lane.
            // The two weight sub-tiles are the same: each writes its own C rows.
            for gv in 0..4i64 {
                let g = k.const_val(gv);
                for nv in 0..2i64 {
                    let n = k.const_val(nv);
                    let n16 = k.mul(n, c16);
                    let row = k.add(i0, n16);
                    let row = k.add(row, lanerow);
                    let g8 = k.mul(g, c8);
                    let addr = k.mad(row, cxs, g8);
                    let side = k.mul(lanehalf, c4);
                    let addr = k.add(addr, side);
                    let frag = k.ldmatrix(
                        zyx::kernel::LdMatrixDims::m8n8,
                        zyx::kernel::LdMatrixNum::X4,
                        zyx::kernel::LdMatrixTrans::off,
                        zyx::kernel::LdMatrixType::b16,
                        xt,
                        addr,
                    );
                    let aw0 = k.index(frag, 0);
                    let aw1 = k.index(frag, 1);
                    let aw2 = k.index(frag, 2);
                    let aw3 = k.index(frag, 3);
                    // `dA[n][h][g]`: word `32+g` of rows `i0 + n*16 + h*8 + lane/4`.
                    let crow = k.add(i0, n16);
                    let crow = k.add(crow, gid);
                    let crowx = k.mul(crow, cxs);
                    let da0i = k.add(crowx, c32);
                    let da0i = k.add(da0i, g);
                    let da0w = k.load(xt, da0i);
                    let da0 = k.bitcast(da0w, DType::F32);
                    let drow8 = k.add(crow, c8);
                    let drow8x = k.mul(drow8, cxs);
                    let da1j = k.add(drow8x, c32);
                    let da1j = k.add(da1j, g);
                    let da1w = k.load(xt, da1j);
                    let da1 = k.bitcast(da1w, DType::F32);

                    let jt = k.unroll(9, 8);
                    let j0 = k.mul(jt, c16);
                    let jb = k.add(jo, j0);
                    let rowb = k.add(jb, gid);
                    let rowbx = k.mul(rowb, cys);
                    let colb = k.add(rowbx, c4);
                    let bcol = k.add(colb, g8);
                    let b0i = k.add(bcol, lanecol);
                    let b0 = k.load(yt, b0i);
                    let bcol4 = k.add(bcol, c4);
                    let b1i = k.add(bcol4, lanecol);
                    let b1 = k.load(yt, b1i);
                    // `dB[l]`: the activation scale of the two C columns.
                    let rd0 = k.add(jb, tig2);
                    let rd1 = k.add(rd0, c1);
                    let d0i = k.mad(rd0, cys, g);
                    let d1i = k.mad(rd1, cys, g);
                    let db0w = k.load(yt, d0i);
                    let db0 = k.bitcast(db0w, DType::F32);
                    let db1w = k.load(yt, d1i);
                    let db1 = k.bitcast(db1w, DType::F32);
                    let c00 = k.wmma(
                        zyx::kernel::MMADims::m8n8k16,
                        zyx::kernel::MMALayout::row_col,
                        zyx::kernel::MMADType::s8_s8_s32_s32,
                        aw0,
                        b0,
                        zero2,
                    );
                    let clo = k.wmma(
                        zyx::kernel::MMADims::m8n8k16,
                        zyx::kernel::MMALayout::row_col,
                        zyx::kernel::MMADType::s8_s8_s32_s32,
                        aw2,
                        b1,
                        c00,
                    );
                    let c10 = k.wmma(
                        zyx::kernel::MMADims::m8n8k16,
                        zyx::kernel::MMALayout::row_col,
                        zyx::kernel::MMADType::s8_s8_s32_s32,
                        aw1,
                        b0,
                        zero2,
                    );
                    let chi = k.wmma(
                        zyx::kernel::MMADims::m8n8k16,
                        zyx::kernel::MMALayout::row_col,
                        zyx::kernel::MMADType::s8_s8_s32_s32,
                        aw3,
                        b1,
                        c10,
                    );
                    // `sum[(j0/8 + n)*4 + l] += C.x[l] * dA[n][l/2][g] * dB[l%2]`.
                    let jt2 = k.mul(jt, c2);
                    let jn = k.add(jt2, n);
                    let base = k.mul(jn, c4);
                    let s1 = k.add(base, c1);
                    let s2 = k.add(base, c2);
                    let s3 = k.add(base, c3);
                    let ic0 = k.index(clo, 0);
                    let ic1 = k.index(clo, 1);
                    let ic2 = k.index(chi, 0);
                    let ic3 = k.index(chi, 1);
                    let v0 = k.cast(ic0, DType::F32);
                    let v1 = k.cast(ic1, DType::F32);
                    let v2 = k.cast(ic2, DType::F32);
                    let v3 = k.cast(ic3, DType::F32);
                    let k0 = k.mul(da0, db0);
                    let k1 = k.mul(da0, db1);
                    let k2 = k.mul(da1, db0);
                    let k3 = k.mul(da1, db1);
                    let p0 = k.load(accf, base);
                    let p1 = k.load(accf, s1);
                    let p2 = k.load(accf, s2);
                    let p3 = k.load(accf, s3);
                    let t0 = k.mad(v0, k0, p0);
                    let t1 = k.mad(v1, k1, p1);
                    let t2 = k.mad(v2, k2, p2);
                    let t3 = k.mad(v3, k3, p3);
                    k.store(accf, t0, base);
                    k.store(accf, t1, s1);
                    k.store(accf, t2, s2);
                    k.store(accf, t3, s3);
                }
            }
        }
        // The next super-block overwrites both tiles.
        k.barrier();
    });

    // ---- `ggml_cuda_mmq_write_back_mma`: C's two columns per lane ----
    {
        let s = k.unroll(10, 64);
        let jt = k.div(s, c8);
        let s4 = k.div(s, c4);
        let n = k.mod_(s4, c2);
        let l = k.mod_(s, c4);
        let half = k.mul(warpl, c8);
        let j0 = k.mul(jt, c16);
        let col0 = k.add(j0, half);
        let lmod = k.mod_(l, c2);
        let cpart = k.add(tig2, lmod);
        let col = k.add(col0, cpart);
        let l2 = k.div(l, c2);
        let n16 = k.mul(n, c16);
        let row = k.add(i0, n16);
        let l8 = k.mul(l2, c8);
        let row = k.add(row, l8);
        let row = k.add(row, gid);
        let nbib = k.mul(nb, cib);
        let wrow = k.add(nbib, row);
        let v = k.load(accf, s);
        let oi = k.mad(col, nstr, wrow);
        k.store(out, v, oi);
    }
    k.verify();
    k
}

/// llama.cpp's Turing `mmq` for PQ2_0, pasted from
/// `ggml/src/ggml-cuda/{mma.cuh, mmq.cuh, mmq-load-tiles.cuh,
/// mmq-vec-dot.cuh}` and specialized to this projection: src0 = the PQ2_0
/// weight (34-byte super-blocks), src1 = the q8_1 activations,
/// `dst[j*N + i] = sum_k x[i,k]*y[j,k]`.
///
/// The block is `mmq`'s config for PQ2_0 on Turing (`I = 128`, `J = 128`,
/// 256 threads as 32x8, `rows_per_warp = 32`, `ntx = 2`, sram layout Q8_0),
/// with one change: the K tile is staged one PQ2_0 super-block at a time
/// (`K_vram = 128` instead of 256), because both tiles at 256 K are 57 KB and
/// a block that has not opted into the larger carveout may only declare 48 KB
/// of static shared memory. Nothing else in the dataflow changes.
const S8_SRC: &str = r#"
#include <cuda_fp16.h>

#define QK_PQ2_0 128
#define MMQ_TILE_NE_K 32
#define QI8_0 8
#define QI8_1 8
#define XSTRIDE (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_0 + 4)
#define YSTRIDE (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define ITER_K QK_PQ2_0
#define ITILE 128
#define JTILE 128
#define NWARPS 8
#define K_ 5120
#define N_ 17408
#define NKBLK_ 40

struct block_pq2_0 {
    unsigned short d;
    unsigned char qs[32];
};

// mma.cuh: tile<I,J,int> for the NVIDIA data layout.
template <int I, int J> struct tile_int {};

template <> struct tile_int<16, 8> {
    static constexpr int ne = 16 * 8 / 32;
    int x[ne];
    static __device__ __forceinline__ int get_i(const int l) { return ((l / 2) * 8) + (threadIdx.x / 4); }
    static __device__ __forceinline__ int get_j(const int l) { return ((threadIdx.x % 4) * 2) + (l % 2); }
};

template <> struct tile_int<8, 8> {
    static constexpr int ne = 8 * 8 / 32;
    int x[ne];
    static __device__ __forceinline__ int get_i(const int l) { return threadIdx.x / 4; }
    static __device__ __forceinline__ int get_j(const int l) { return (l * 4) + (threadIdx.x % 4); }
};

// mma.cuh: load_ldmatrix (Turing), load_generic, and the int8 mma, which on
// Turing is four in-place m8n8k16 instructions.
static __device__ __forceinline__ void load_ldmatrix_16x8(tile_int<16, 8> & t, const int * __restrict__ xs0, const int stride) {
    int * xi = t.x;
    const int * xs = xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                 : "l"(xs));
}

template <int I, int J>
static __device__ __forceinline__ void load_generic(tile_int<I, J> & t, const int * __restrict__ xs0, const int stride) {
#pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
}

static __device__ __forceinline__ void mma_16x8_8x8(tile_int<16, 8> & D, const tile_int<16, 8> & A, const tile_int<8, 8> & B) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1]) : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3]) : "r"(A.x[1]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1]) : "r"(A.x[2]), "r"(B.x[1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3]) : "r"(A.x[3]), "r"(B.x[1]));
}

extern "C" __global__ void __launch_bounds__(256) k_32_8_1(
        const signed char * __restrict__ a8,   // [M, K] q8 payload
        const float * __restrict__ sa,         // [M, K/32] q8 scales
        const unsigned char * __restrict__ w,  // [N*K/128, 34] PQ2_0 blocks
        float * __restrict__ dst) {            // [M, N]
    __shared__ int x_qs[ITILE * XSTRIDE];
    __shared__ int y_qs[JTILE * YSTRIDE];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;

    const int nb = blockIdx.x;   // 128 src0 rows
    const int i0 = (warp / 2) * 32;
    const int jo = (warp % 2) * 8;

    float sum[64];
#pragma unroll
    for (int l = 0; l < 64; ++l) { sum[l] = 0.0f; }

    float * x_df  = (float *) (x_qs + MMQ_TILE_NE_K);
    int   * y_qs4 = y_qs + 4;
    float * y_df  = (float *) y_qs;

    for (int kb = 0; kb < K_ / ITER_K; ++kb) {
        __syncthreads();

        // ggml_cuda_mmq_load_tiles_pq2_0
        {
            constexpr int QI_PQ2_0 = 8;
            constexpr int threads_per_row = QI_PQ2_0;
            constexpr int nrows = 32 / threads_per_row;
            constexpr int scale_entries_per_block = QK_PQ2_0 / 32;
            constexpr int scale_entries_per_row = scale_entries_per_block;

            const int txi  = lane % threads_per_row;
            const int kbx  = txi / QI_PQ2_0;
            const int kqsx = txi % QI_PQ2_0;

            for (int i = warp * nrows + lane / threads_per_row; i < ITILE; i += NWARPS * nrows) {
                const block_pq2_0 * bxi = (const block_pq2_0 *) w + (nb * ITILE + i) * NKBLK_ + kb;
                const short * qxi = (const short *) bxi->qs + kqsx * 4;
                const int dst_offset = kbx * (scale_entries_per_block * QI8_0) + kqsx * QI8_0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int q  = qxi[j];
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    x_qs[i * XSTRIDE + dst_offset + j * 2 + 0] = qx;
                    x_qs[i * XSTRIDE + dst_offset + j * 2 + 1] = qy;
                }
            }

            const int ksx = lane % scale_entries_per_row;
#pragma unroll
            for (int i = warp; i < ITILE; i += NWARPS) {
                const block_pq2_0 * bxi = (const block_pq2_0 *) w + (nb * ITILE + i) * NKBLK_ + kb;
                x_df[i * XSTRIDE + ksx] = __half2float(__ushort_as_half(bxi->d));
            }
        }

        // the y tile: block_q8_1_mmq is [4 scales][32 ints] per 128 values
        {
#pragma unroll
            for (int w_ = 0; w_ < JTILE * MMQ_TILE_NE_K / 256; ++w_) {
                const int o = tid + w_ * 256;
                const int r = o / MMQ_TILE_NE_K;
                const int c_ = o % MMQ_TILE_NE_K;
                y_qs4[r * YSTRIDE + c_] = ((const int *) (a8 + r * K_ + kb * ITER_K))[c_];
            }
#pragma unroll
            for (int s_ = 0; s_ < JTILE * (ITER_K / 32) / 256; ++s_) {
                const int o = tid + s_ * 256;
                const int r = o / 4;
                const int g = o % 4;
                y_df[r * YSTRIDE + g] = sa[r * (K_ / 32) + kb * 4 + g];
            }
        }
        __syncthreads();

        // ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, k00 = 0
        {
            constexpr int ntx = 2;
            tile_int<16, 8> A[ntx][MMQ_TILE_NE_K / QI8_0];
            float dA[ntx][2][MMQ_TILE_NE_K / QI8_0];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                    load_ldmatrix_16x8(A[n][k01 / QI8_0], x_qs + (i0 + n * 16) * XSTRIDE + k01, XSTRIDE);
                }
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int i = i0 + n * 16 + tile_int<16, 8>::get_i(2 * l);
#pragma unroll
                    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                        dA[n][l][k01 / QI8_0] = x_df[i * XSTRIDE + k01 / QI8_0];
                    }
                }
            }

#pragma unroll
            for (int j0 = 0; j0 < JTILE; j0 += ntx * 8) {
#pragma unroll
                for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                    tile_int<8, 8> B;
                    float dB[2];
                    load_generic(B, y_qs4 + jo * YSTRIDE + j0 * YSTRIDE + k01, YSTRIDE);
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int j = j0 + tile_int<16, 8>::get_j(l);
                        dB[l] = y_df[(jo + j) * YSTRIDE + k01 / QI8_1];
                    }
#pragma unroll
                    for (int n = 0; n < ntx; ++n) {
                        tile_int<16, 8> C;
#pragma unroll
                        for (int l = 0; l < 4; ++l) { C.x[l] = 0; }
                        mma_16x8_8x8(C, A[n][k01 / QI8_0], B);
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            sum[(j0 / 8 + n) * 4 + l] += C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
                        }
                    }
                }
            }
        }
    }

    // ggml_cuda_mmq_write_back_mma
    {
        constexpr int ntx = 2;
#pragma unroll
        for (int j0 = 0; j0 < JTILE; j0 += ntx * 8) {
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int j = j0 + (warp % ntx) * 8 + tile_int<16, 8>::get_j(l);
                    const int i = i0 + n * 16 + tile_int<16, 8>::get_i(l);
                    dst[j * N_ + (nb * ITILE + i)] = sum[(j0 / 8 + n) * 4 + l];
                }
            }
        }
    }
}
"#;

/// The pasted kernel as a zyx kernel: the four buffers plus the launch
/// descriptor it needs (`push_source`).
fn gemm_s8_source(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    k.param(DType::I8);
    k.param(DType::F32);
    k.param(DType::U8);
    k.param_mut(DType::F32);
    let blocks = u32::try_from(N / 128).expect("weight-row blocks fit u32");
    k.push_source(S8_SRC, [blocks, 1, 1], [32, 8, 1]);
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
    let mut q8_host = vec![0i8; usize::try_from(M * K).expect("activation fits usize")];
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
                q8_host[base + i] = q as i8;
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
            let q = f32::from(byte);
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
    // The IR kernel reads the same buffers with the widths `mmq` casts them
    // to: the q8 payload as 32-bit words, the PQ2_0 super-block as 17 words.
    let q8_words: Vec<u32> =
        q8_host.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
    let w_words_host: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let a8_words: Tensor = Tensor::from_vec(q8_words, [M, K / 4])?.to(dev)?;
    let w_words: Tensor = Tensor::from_vec(w_words_host, [N * K / BLOCK, ROW / 2])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(raw.clone(), [N * K / BLOCK, ROW])?.to(dev)?;
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
    let q8_got: Vec<i8> = q8_dev.clone().try_into()?;
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

    // ---- the kernels, and the one pass the IR one gets ------------------
    // The IR kernel is the pasted dataflow written with the IR's own ops: no
    // tensorization, no recipe, and the only pass is the device epilogue's
    // lane expansion, so what runs is exactly what was written. The pasted one
    // is llama.cpp's CUDA source as-is, launched through `push_source`.
    let ir = gemm_s8_kernel(dev);
    ir.verify();
    let pasted = gemm_s8_source(dev);

    // The timing is the search's device-side measurement, the same one the
    // first test uses: one warmup and one timed launch bracketed by timing
    // events. The values come from a copy compiled on its own.
    let scratch = Tensor::zeros([M, N], DType::F32).to(dev)?;
    let flop = 2.0 * (M * N * K) as f64;
    for (name, gemm, ins) in [
        ("llama.cpp pasted", pasted, [&a8_t, &sa_t, &w]),
        ("IR", ir, [&a8_words, &sa_t, &w_words]),
    ] {
        let values = gemm.clone();
        let (_winner, nanos) = BeamSearch::new().run([gemm], &[ins[0], ins[1], ins[2], &scratch], &[], |_| {}, |_| 0)?;
        let gemm_nanos = nanos as f64;
        let compiled = values.compile()?;
        let out = compiled.forward(&[ins[0], ins[1], ins[2]], vec![[M, N]])?.pop().unwrap();

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
                        format!("ffn_gate s8 ({name}) disagrees at (m={m}, n={n}): {got} vs {want}, tol {tol}").into(),
                    ));
                }
                if err > worst.0 {
                    worst = (err, m as i64, n);
                }
            }
        }
        println!(
            "ffn_gate s8 ({name}) matches the int8 reference, max |err| = {:.3e} at (m={}, n={})",
            worst.0, worst.1, worst.2
        );
        println!(
            "qwen ffn_gate {M}x{K}x{N} int8 ({name}): q8_act {:.4} ms, gemm {:.4} ms, {:.2} TFLOP/s",
            quant_nanos * 1e-6,
            gemm_nanos * 1e-6,
            flop / gemm_nanos * 1e-3
        );
    }
    let _ = out_t;
    Ok(())
}
