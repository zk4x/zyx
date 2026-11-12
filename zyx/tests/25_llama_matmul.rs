// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama-3.2-3B `gate_proj` as a *naive* custom kernel, optimized into
//! llama.cpp's tensor-core shape.
//!
//! The kernel below is exactly what the linearizer emits for
//! `x.matmul(w.t())` on the prefill projection (128 x 3072 x 8192): two
//! group ranges over the output, one accumulation loop over K, one register
//! accumulator, one store. No tiling: the tiling, the staging and the
//! tensor-core instruction come from optimizations applied afterwards, so the
//! test isolates what the passes do and what they are worth.
//!
//! Asserts the result against a host GEMM on the same f16 operands
//! (`matrixmultiply`), and asserts the kernel is compute- and
//! bandwidth-competitive with the reference: torch 2.11 (cuBLAS,
//! `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn`) runs this exact GEMM in
//! 0.277-0.284 ms, 22.7-23.2 TFLOP/s on this card (336 GB/s peak).
//!
//! The selected config is the reference kernel's shape: a 128 x 128 output
//! tile, eight warps as a 2 x 4 grid (a 64 x 32 warp tile), operands staged
//! through two block-wide shared buffers of a two-K-step window filled with
//! 16-byte vector loads and read back with `ldmatrix.x4`. In the tensor
//! recipe that is tile `(128, 128, 128)`, staged, warp layout `(2, 4, 1)`.

use std::time::Instant;

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError};

/// The recipe config of the reference kernel's shape: `(tile * 2 + staged) *
/// layouts + layout`, i.e. `(4 * 2 + 1) * 13 + 8` — tile index 4 is
/// `(m lanes 128, n lanes 128, k lanes 128)` (16 `m16n8k8` K steps), staged,
/// warp layout index 8 is `(2, 4, 1)`.
const TORCH_TILE_8_WARPS: u64 = 125;

#[test]
fn llama_gate_proj() -> Result<(), ZyxError> {
    // gate_proj at the prefill sequence length.
    const M: i64 = 128;
    const N: i64 = 8192;
    const K: i64 = 3072;

    // The optimizations read the device's warp size and budgets, so the
    // kernel is bound to a concrete device up front.
    let dev = Dev::all()
        .into_iter()
        .find(|d| !matches!(d, Dev::C | Dev::Host))
        .ok_or_else(|| ZyxError::kernel_error("no compute device".into()))?;

    // ---- the naive kernel the linearizer emits -------------------------
    let mut k = Kernel::new(dev);
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let c = k.param_mut(DType::F32);
    let m = k.group_range(0, M);
    let n = k.group_range(1, N);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(K, |k, k_loop| {
        let a_idx = k.mad(m, K, k_loop);
        let a_load = k.load(a, a_idx);
        let b_idx = k.mad(n, K, k_loop);
        let b_load = k.load(b, b_idx);
        let prod = k.mul(a_load, b_load);
        let prod = k.cast(prod, DType::F32);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    // The projection's dtype is f16, the benchmark reads f32: the linearized
    // store casts through it.
    let res = k.load(acc, 0i64);
    let res = k.cast(res, DType::F16);
    let res = k.cast(res, DType::F32);
    let c_idx = k.mad(m, N, n);
    k.store(c, res, c_idx);
    k.verify();

    // ---- the selected optimization -------------------------------------
    // The tensor recipe is the fixed order split-M, split-N, split-K,
    // tensorize. Its config space is the product of the steps' config spaces:
    // the tile grid, the staging form, and the warp layout. This config is
    // the reference kernel's shape, and it must be one of the options.
    let recipe = k.opt_tensor_recipe();
    assert!(
        TORCH_TILE_8_WARPS < recipe.nconfigs(),
        "the recipe's config space must contain the cuBLAS shape: {} configs",
        recipe.nconfigs()
    );
    recipe.apply(&mut k, TORCH_TILE_8_WARPS);
    k.default_epilogue();
    // The device epilogue (devectorize) runs once, inside `compile`.
    k.verify();

    // ---- the operands, and the reference they must reproduce ------------
    let a = Tensor::rand([M, K], DType::F16)?.to(dev)?;
    let b = Tensor::rand([N, K], DType::F16)?.to(dev)?;
    // `sgemm` is row-major `a @ b.T`: A is the m x k `a` (row stride k), B is
    // k x n laid out as the row-major n x k `b` (row stride 1, column stride
    // k). Both sides see the same f16 operands, in f32.
    let a_host: Vec<f32> = a.clone().cast(DType::F32).try_into().unwrap();
    let b_host: Vec<f32> = b.clone().cast(DType::F32).try_into().unwrap();
    let mut want = vec![0f32; (M * N) as usize];
    unsafe {
        matrixmultiply::sgemm(
            M as usize,
            K as usize,
            N as usize,
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

    // ---- measure -------------------------------------------------------
    let compiled = k.compile()?;
    // Warmup: the first launch pays the module load and fabric init.
    let _ = compiled.forward(&[&a, &b], vec![[M, N]])?;
    const REPS: usize = 20;
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..REPS {
        last = Some(compiled.forward(&[&a, &b], vec![[M, N]])?);
    }
    // One readback drains the stream: it costs the same for every repeat, so
    // it is timed outside the loop.
    let out: Vec<f32> = last.unwrap().pop().unwrap().try_into()?;
    let nanos = t0.elapsed().as_secs_f64() / REPS as f64;

    // ---- the kernel must reproduce the reference ------------------------
    // The epilogue rounds the f32 accumulator through f16, so a kernel
    // element is the reference rounded to f16: a few f16 ulps are the whole
    // budget for the accumulation order.
    let mut worst = (0f32, 0i64, 0i64);
    for i in 0..M as usize {
        for j in 0..N as usize {
            let got = out[i * N as usize + j];
            let want = want[i * N as usize + j];
            let err = (got - want).abs();
            let tol = 1e-3f32.max(2e-3 * want.abs());
            if err > tol {
                return Err(ZyxError::kernel_error(
                    format!("gate_proj disagrees with the host GEMM at (m={i}, n={j}): {got} vs {want}, tol {tol}")
                        .into(),
                ));
            }
            if err > worst.0 {
                worst = (err, i as i64, j as i64);
            }
        }
    }
    println!("gate_proj matches the host GEMM, max |err| = {:.3e} at (m={}, n={})", worst.0, worst.1, worst.2);

    // ---- the reference's numbers ---------------------------------------
    let flop = 2.0 * (M * N * K) as f64;
    let bytes = ((M * K + N * K + M * N) as f64) * 2.0;
    let tflops = flop / nanos / 1e12;
    let gbs = bytes / nanos / 1e9;
    println!(
        "llama gate_proj {M}x{K}x{N}: {:.4} ms, {tflops:.2} TFLOP/s, {gbs:.1} GB/s",
        nanos * 1e3
    );
    assert!(tflops > 20.0, "gate_proj must beat torch's 21.5 TFLOP/s baseline (20), got {tflops:.2}");
    assert!(gbs > 170.0, "gate_proj must beat torch's 177.4 GB/s baseline (170), got {gbs:.1}");
    Ok(())
}
