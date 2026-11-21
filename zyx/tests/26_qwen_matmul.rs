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

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError, f16};

/// The reference config in the tensorization space: the staged tile grid the
/// kernel is compared against, the way the llama test names the cuBLAS shape.
/// `opt_tensorize` lists the kernel's own region first, then one block per tile
/// grid in each operand orientation.
const REFERENCE_TILE: u64 = 1645;

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
/// One int8 activation scale per this many elements.
const QG: i64 = 32;
/// Activation groups per row (`K / QG`).
const KGROUP: i64 = K / QG;

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
