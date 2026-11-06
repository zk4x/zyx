// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Hand-written m16n8k8 tensor-core GEMM in the layout llama actually uses.
//!
//! `x @ W.t()` is NT: A is `[m,k]` and B is `[n,k]`, both K-contiguous. The
//! `row_col` layout wants exactly that for B (col-major k x n == row-major
//! `[n,k]`), so the only difference from the NN kernel in
//! `06_realize::wmma_matmul` is the B fragment index: `n*K + k` instead of
//! `k*N + n`.

use zyx::kernel::{Kernel, MMADType, MMADims, MMALayout, MemScope, OpId};
use zyx::{DType, Dev, Tensor, ZyxError};

use super::bench_dev;
use super::llama::shapes::{HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, SEQ};

fn kernel(dev: Dev, m: i64, n: i64, k: i64) -> Kernel {
    let mut kernel = Kernel::new(dev);
    let a_buf = kernel.param(DType::F16);
    let b_buf = kernel.param(DType::F16);
    let c_buf = kernel.param_mut(DType::F32);

    let gidx_len = kernel.const_idx((m / 16) as u32);
    let gidy_len = kernel.const_idx((n / 8) as u32);
    let gidx = kernel.group_range(0, gidx_len);
    let gidy = kernel.group_range(1, gidy_len);
    let wid = kernel.local_range(0, 32);

    let [c0, c1, c2, c4, c8, c16] = kernel.const_idxs([0u32, 1, 2, 4, 8, 16]);
    let n_const = kernel.const_idx(n as u32);
    let k_const = kernel.const_idx(k as u32);
    let k_div_8 = kernel.const_idx((k / 8) as u32);

    // Lane -> (row, sub-column) inside the 16x8 tile, as in the hardware map.
    let row_in_tile = kernel.div(wid, c4);
    let sub_col = kernel.mod_(wid, c4);
    let col_in_tile = kernel.mul(sub_col, c2);

    let a_row = kernel.mad(gidx, c16, row_in_tile);
    let b_col = kernel.mad(gidy, c8, row_in_tile);
    let tile_base_col = kernel.mul(gidy, c8);

    let acc = kernel.zeros(DType::F32, 4);

    kernel.loop_over(k_div_8, |kernel, k_loop| {
        let k_off = kernel.mul(k_loop, c8);

        // A fragment: 4 f16, (m, k), (m, k+1), (m+8, k), (m+8, k+1).
        let a_base = kernel.mad(a_row, k_const, k_off);
        let a_base = kernel.add(a_base, col_in_tile);
        let a_load_0 = kernel.load(a_buf, a_base);
        let a_base_p1 = kernel.add(a_base, c1);
        let a_load_1 = kernel.load(a_buf, a_base_p1);
        let a_base2 = kernel.mad(c8, k_const, a_base);
        let a_load_2 = kernel.load(a_buf, a_base2);
        let a_base2_p1 = kernel.add(a_base2, c1);
        let a_load_3 = kernel.load(a_buf, a_base2_p1);
        let a_frag = kernel.stack(&[a_load_0, a_load_1, a_load_2, a_load_3]);

        // B fragment: 2 f16, (k, n) and (k+1, n) — the second element steps k,
        // which in [n,k] memory is +1.
        let b_row = kernel.add(k_off, col_in_tile);
        let b_base = kernel.mad(b_col, k_const, b_row);
        let b_load_0 = kernel.load(b_buf, b_base);
        let b_base_k = kernel.add(b_base, c1);
        let b_load_1 = kernel.load(b_buf, b_base_k);
        let b_frag = kernel.stack(&[b_load_0, b_load_1]);

        // Accumulator slots are read and written scalar-wise: a vector
        // `load_vector`/`store_vector` here would leave lane bundles for the
        // device epilogue to unfold, which no backend can compile today.
        let c_elems: Vec<OpId> = (0..4)
            .map(|i| {
                let at = kernel.const_idx(i as u32);
                kernel.load(acc, at)
            })
            .collect();
        let c_frag = kernel.stack(&c_elems);
        let d = kernel.wmma(MMADims::m16n8k8, MMALayout::row_col, MMADType::f16_f16_f16_f32, a_frag, b_frag, c_frag);
        for i in 0..4 {
            let at = kernel.const_idx(i as u32);
            let elem = kernel.index(d, i);
            kernel.store(acc, elem, at);
        }
    });

    // Scatter the accumulator: slot (m, n) offsets of the m16n8k8 C fragment.
    let c_col = kernel.add(tile_base_col, col_in_tile);
    let c_base = kernel.mad(a_row, n_const, c_col);
    let c_base2 = kernel.mad(c8, n_const, c_base);
    for (i, slot) in [(0u32, c_base), (1, kernel.add(c_base, c1)), (2, c_base2), (3, kernel.add(c_base2, c1))] {
        let at = kernel.const_idx(i);
        let elem = kernel.load(acc, at);
        kernel.store(c_buf, elem, slot);
    }
    kernel
}

fn run(dev: Dev, m: i64, n: i64, k: i64, check: bool) -> Result<(), ZyxError> {
    let compiled = kernel(dev, m, n, k).compile()?;
    let a = Tensor::rand([m, k], DType::F16)?.to(dev)?;
    let b = Tensor::rand([n, k], DType::F16)?.to(dev)?;
    let c = compiled.forward1(&[&a, &b], [m, n])?;

    let t = std::time::Instant::now();
    let _out = compiled.forward1(&[&a, &b], [m, n])?;
    dev.synchronize()?;
    let dt = t.elapsed();
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "mma-nt {m}x{n}x{k}: {:.3} ms, {:.2} GFLOP/s (weights {:.1} MiB)",
        dt.as_secs_f64() * 1e3,
        flop / dt.as_secs_f64() / 1e9,
        (n * k * 2) as f64 / (1024.0 * 1024.0),
    );

    if check {
        let a_host: Vec<f32> = a.clone().cast(DType::F32).try_into()?;
        let b_host: Vec<f32> = b.clone().cast(DType::F32).try_into()?;
        let c_host: Vec<f32> = c.cast(DType::F32).try_into()?;
        let mut worst = 0.0f32;
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;
                for t in 0..k as usize {
                    sum += a_host[i * k as usize + t] * b_host[j * k as usize + t];
                }
                worst = worst.max((c_host[i * n as usize + j] - sum).abs());
            }
        }
        println!("mma-nt {m}x{n}x{k}: max |err| = {worst}");
        assert!(worst < 1e-1, "NT wmma matmul wrong: max |err| = {worst}");
    }
    Ok(())
}

pub fn bench() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| matches!(d, Dev::Cuda(_))) else {
        println!("mma-nt needs a CUDA device with tensor cores; none available");
        return Ok(());
    };
    let _ = bench_dev();
    // Small exact check first.
    run(dev, 64, 64, 64, true)?;
    // The llama-3.2-3B prefill projections.
    for (n, k) in [
        (HEADS * HEAD_DIM, HIDDEN),     // q_proj
        (KV_HEADS * HEAD_DIM, HIDDEN),  // k/v_proj
        (INTERMEDIATE, HIDDEN),         // gate/up_proj
        (HIDDEN, INTERMEDIATE),         // down_proj
    ] {
        run(dev, SEQ, n, k, false)?;
    }
    Ok(())
}
