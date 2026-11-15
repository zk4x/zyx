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

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError, f16};

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

    // ---- run -----------------------------------------------------------
    let a: Tensor = Tensor::from_vec(a_host.iter().map(|&x| f16::from_f32(x)).collect::<Vec<f16>>(), [M, K])?.to(dev)?;
    let w: Tensor = Tensor::from_vec(raw, [N * K / BLOCK, ROW])?.to(dev)?;
    // `compile` consumes the kernel; the tensorization space is read off the
    // same IR afterwards, so compile a copy.
    let runner = k.clone();
    let compiled = runner.compile()?;

    let started = Instant::now();
    let out = compiled.forward(&[&a, &w], vec![[M, N]])?.pop().unwrap();
    let got: Vec<f32> = out.try_into()?;
    let nanos = started.elapsed().as_secs_f64() * 1e9;

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

    // ---- the tensorization must offer the dequant-fused form ------------
    // The staged form is what makes this projection memory-competitive: the
    // fill dequantizes while it cooperatively loads the weight tile into
    // shared memory. The matcher must therefore accept a product whose
    // multiplier is a dequant chain, not only `mul(load, load)`.
    let recipe = k.opt_tensorize();
    println!("tensorize configs: {}", recipe.nconfigs());
    assert!(
        recipe.nconfigs() > 0,
        "the tensorization space must offer the dequant-fused form (a staged fill that dequantizes \
         the weight tile as it loads it); the matcher still requires mul(load, load)"
    );
    Ok(())
}
