// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::kernel::{DeviceId, Kernel, MMADType, MMADims, MMALayout, MemLayout, Scope};
use zyx::{DType, ReduceOp, Scalar, Tensor, ZyxError};

/// Tensor-core matmul: C = A @ B where A(M×K, FP16), B(K×N, FP16), C(M×N, FP32).
///
/// Translates `~/Dev/python/matmul/kernel.cu` to the zyx kernel builder.
/// Uses the m16n8k8 WMMA instruction with one warp (32 threads) per 16×8 tile.
///
/// Requires a CUDA device with tensor cores (cc >= 7.0).
/// Run with: `AGENT=1 cargo test -p zyx --test 7_realize wmma_matmul -- --nocapture --include-ignored`
#[test]
fn wmma_matmul() -> Result<(), ZyxError> {
    let m = 1024u64;
    let n = 1024u64;
    let k = 1024u64;

    let mut kernel = Kernel::new(DeviceId::AUTO);

    // Global buffers
    let a_buf = kernel.define(DType::F16, Scope::Global, true, m * k);
    let b_buf = kernel.define(DType::F16, Scope::Global, true, k * n);
    let c_buf = kernel.define(DType::F32, Scope::Global, false, m * n);

    // Work-group (grid) indices  --  blockIdx.{x, y}
    let gidx = kernel.gidx(0, m / 16); // tile row
    let gidy = kernel.gidx(1, n / 8); // tile col

    // Thread index within work-group  --  threadIdx.x
    let wid = kernel.lidx(0, 32);

    // ---- Constants ----
    let c0 = kernel.const_idx(0u32);
    let c1 = kernel.const_idx(1u32);
    let c2 = kernel.const_idx(2u32);
    let c4 = kernel.const_idx(4u32);
    let c8 = kernel.const_idx(8u32);
    let c16 = kernel.const_idx(16u32);
    let n_const = kernel.const_idx(n as u32);
    let k_const = kernel.const_idx(k as u32);

    // wid >> 2       -> row index within A/C tile (0..7)
    let row_in_tile = kernel.div(wid, c4);
    // wid & 3        -> sub-column index within A/C tile
    let sub_col = kernel.mod_(wid, c4);
    // (wid & 3) * 2  -> column offset within A/C tile (0, 2, 4, 6)
    let col_in_tile = kernel.mul(sub_col, c2);



    // a_row = gidx * 16 + row_in_tile
    let a_row = kernel.mad(gidx, c16, row_in_tile);
    // b_col = gidy * 8 + row_in_tile
    let b_col = kernel.mad(gidy, c8, row_in_tile);
    let tile_base_col = kernel.mul(gidy, c8);

    // ---- Accumulator (float4 per thread) ----
    let acc = kernel.define(DType::F32, Scope::Register, false, 4);
    let zf = kernel.const_val(0.0f32);
    let zero_acc = kernel.vectorize(vec![zf, zf, zf, zf]);
    kernel.store(acc, zero_acc, c0, MemLayout::Vector(4));

    // ---- K loop  (k / 8 iterations) ----
    let k_loop = kernel.loop_(k / 8);

    // k * 8
    let k_off = kernel.mul(k_loop, c8);

    // ---- Load A fragment (m16 × k8 = 4 half per thread) ----
    let a_base = kernel.mad(a_row, k_const, k_off);
    let a_base = kernel.add(a_base, col_in_tile);
    let a_load_0 = kernel.load(a_buf, a_base, MemLayout::Scalar);
    let a_base_p1 = kernel.add(a_base, c1);
    let a_load_1 = kernel.load(a_buf, a_base_p1, MemLayout::Scalar);
    let a_base2 = kernel.mad(c8, k_const, a_base);
    let a_load_2 = kernel.load(a_buf, a_base2, MemLayout::Scalar);
    let a_base2_p1 = kernel.add(a_base2, c1);
    let a_load_3 = kernel.load(a_buf, a_base2_p1, MemLayout::Scalar);
    let a_frag = kernel.vectorize(vec![a_load_0, a_load_1, a_load_2, a_load_3]);

    // ---- Load B fragment (k8 × n8 = 2 half per thread) ----
    // col-major B: row = (wid%4)*2, (wid%4)*2+1, col = wid/4
    let b_row = kernel.add(k_off, col_in_tile);
    let b_base = kernel.mad(b_row, n_const, b_col);
    let b_load_0 = kernel.load(b_buf, b_base, MemLayout::Scalar);
    let b_base_n = kernel.add(b_base, n_const);
    let b_load_1 = kernel.load(b_buf, b_base_n, MemLayout::Scalar);
    let b_frag = kernel.vectorize(vec![b_load_0, b_load_1]);

    // ---- WMMA: acc = A_frag @ B_frag + acc ----
    let acc_old = kernel.load(acc, c0, MemLayout::Vector(4));
    let acc_new = kernel.wmma(
        MMADims::m16n8k8,
        MMALayout::row_col,
        MMADType::f16_f16_f16_f32,
        a_frag,
        b_frag,
        acc_old,
    );
    kernel.store(acc, acc_new, c0, MemLayout::Vector(4));

    kernel.end_loop();

    // ---- Store result to C ----
    let acc_final = kernel.load(acc, c0, MemLayout::Vector(4));
    let co = kernel.devectorize(acc_final, 0);
    let c1v = kernel.devectorize(acc_final, 1);
    let c2v = kernel.devectorize(acc_final, 2);
    let c3v = kernel.devectorize(acc_final, 3);

    let c_col = kernel.add(tile_base_col, col_in_tile);
    let c_base = kernel.mad(a_row, n_const, c_col);
    kernel.store(c_buf, co, c_base, MemLayout::Scalar);
    let c_base_p1 = kernel.add(c_base, c1);
    kernel.store(c_buf, c1v, c_base_p1, MemLayout::Scalar);
    let c_base2 = kernel.mad(c8, n_const, c_base);
    kernel.store(c_buf, c2v, c_base2, MemLayout::Scalar);
    let c_base2_p1 = kernel.add(c_base2, c1);
    kernel.store(c_buf, c3v, c_base2_p1, MemLayout::Scalar);

    // ---- Compile & run ----
    let compiled = kernel.compile()?;

    let a = Tensor::rand([m, k], DType::F16)?;
    let b = Tensor::rand([k, n], DType::F16)?;
    let a_host: Vec<f32> = a.clone().cast(DType::F32).try_into()?;
    let b_host: Vec<f32> = b.clone().cast(DType::F32).try_into()?;

    let result = compiled.forward(&[&a, &b], [m, n]);

    let c_host: Vec<f32> = result.try_into()?;

    // Reference: A @ B on CPU
    let mut ref_c = vec![0.0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for t in 0..k {
                sum += a_host[(i * k + t) as usize] * b_host[(t * n + j) as usize];
            }
            ref_c[(i * n + j) as usize] = sum;
        }
    }

    let mut max_err = 0.0f32;
    for idx in 0..(m * n) as usize {
        let err = (c_host[idx] - ref_c[idx]).abs();
        max_err = max_err.max(err);
    }
    println!("WMMA matmul max error: {max_err:.6}");
    assert!(max_err < 1.0, "WMMA matmul error too large: {max_err}");

    Ok(())
}

#[test]
fn t01() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);

    for _ in 0..1 {
        let y = x.exp2();
        x = y.log2();

        //println!("x rc = {}", x.ref_count());
        //println!("y rc = {}", y.ref_count());

        //Tensor::debug_graph();
        Tensor::realize([&x])?;
        //Tensor::debug_graph();

        //println!("x rc = {}", x.ref_count());
        //println!("y rc = {}", y.ref_count());
    }

    //Tensor::debug_graph();

    Ok(())
}

#[test]
fn t02() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);
    let z = Tensor::from(6);

    for _ in 0..20 {
        let y0 = x.exp2();
        let y1 = y0.exp2() * &z;
        let y2 = y1.exp2() + 3;
        let _y3 = y2.exp2();
        x = y2.log2();
        Tensor::realize([&x])?;
    }

    Ok(())
}

#[test]
fn t03() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);
    let z = Tensor::from(6);

    for _ in 0..200 {
        let y0 = x.exp2();
        let y1 = y0.exp2() * &z;
        let y2 = y1.exp2() + 3;
        let _y3 = y2.exp2();
        x = y2.log2();
        Tensor::realize([&x])?;
    }

    Ok(())
}

#[test]
fn t04() -> Result<(), ZyxError> {
    let input = Tensor::from([5f32, 2., -3.]);
    let target = Tensor::from([1f32, 0., 0.]);
    let loss = input.cross_entropy(target, ReduceOp::Mean)?;
    assert_eq!(loss, 0.048907f32);
    Ok(())
}

#[test]
fn t05() -> Result<(), ZyxError> {
    let x = Tensor::rand([2048, 320], DType::F32)?;
    let xdata: Vec<f32> = x.clone().try_into()?;
    let y = Tensor::rand([2048, 1], DType::F32)?;
    let ydata: Vec<f32> = y.clone().expand([2048, 320])?.try_into()?;
    let x = (x - y.expand([2048, 320])? * 1.4f32) / y.expand([2048, 320])?;
    let xvec: Vec<f32> = x.try_into()?;

    let mut i = 0;
    for ((x0, x1), x2) in xdata.into_iter().zip(ydata).zip(xvec) {
        let z = (x0 - x1 * 1.4f32) / x1;
        if !z.is_equal(x2) {
            println!("{z} != {x2} at idx={i}");
            panic!();
        }
        i += 1;
    }

    Ok(())
}

#[test]
fn pad_1() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 20, 1)?.reshape([4, 5])?;
    assert_eq!(x.rslice(3)?, [[3], [8], [13], [18]]);
    Ok(())
}

#[test]
fn t_15() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    for _ in 0..10 {
        x = &x + &x;
        //println!("{x}");
        //Tensor::plot_graph([], &format!("graph{i}"));
        //Tensor::realize([&x]).unwrap();
    }
    //println!("{x}");
    assert_eq!(x, [[2048, 3072, 1024], [2048, 4096, 1024]]);
}

#[test]
fn iter1() -> Result<(), ZyxError> {
    let mut x = Tensor::randn([64, 64], DType::F32)?;
    let y = Tensor::randn([64, 64], DType::F32)?;

    for _ in 0..20 {
        x = x.dot(&y)?.softmax([-1])?;
        Tensor::realize([&x])?;
        //println!("{}", x.is_realized());
    }

    Ok(())
}

#[test]
fn b_sftmx1() -> Result<(), ZyxError> {
    use zyx::DType;
    use zyx::Tensor;

    let shape: [usize; 2] = [2048, 320];

    let x = Tensor::rand(shape, DType::F32)?;
    let y = x.softmax([-1])?;
    let y_host: Vec<f32> = y.try_into()?;

    let x_host: Vec<f32> = x.try_into()?;
    let mut y_ref = vec![0.0f32; x_host.len()];

    for row in 0..shape[0] {
        let start = row * shape[1];
        let end = start + shape[1];
        let row_slice = &x_host[start..end];

        let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0;
        for i in 0..shape[1] {
            let e = (row_slice[i] - max).exp();
            y_ref[start + i] = e;
            sum += e;
        }

        for i in 0..shape[1] {
            y_ref[start + i] /= sum;
        }
    }

    for (a, b) in y_host.iter().zip(y_ref.iter()) {
        assert!(a.is_equal(*b), "mismatch: {a} vs {b}");
    }

    Ok(())
}

#[test]
fn b_sftmx2() -> Result<(), ZyxError> {
    use zyx::Module;
    let x = Tensor::rand([1, 320], DType::F32)?;
    let y = x.sum([-1])?;
    let y = y.expand(1024)?;
    y.realize()?;
    Ok(())
}

#[test]
fn sftmx3() -> Result<(), ZyxError> {
    use zyx::{DType, Tensor};

    let shape: [usize; 2] = [2048, 320];

    // Input
    let x = Tensor::rand(shape, DType::F32)?;

    // Compute using your implementation
    let y = x.softmax([-1])?;
    let y_host: Vec<f32> = y.try_into()?;

    // Reference implementation (stable softmax)
    let x_host: Vec<f32> = x.try_into()?;
    let mut y_ref = vec![0.0f32; x_host.len()];

    for row in 0..shape[0] {
        let start = row * shape[1];
        let end = start + shape[1];

        let row_slice = &x_host[start..end];

        // max for numerical stability
        let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // exp + sum
        let mut sum = 0.0;
        for i in 0..shape[1] {
            let e = (row_slice[i] - max).exp();
            y_ref[start + i] = e;
            sum += e;
        }

        // normalize
        for i in 0..shape[1] {
            y_ref[start + i] /= sum;
        }
    }

    // Compare
    for i in 0..y_host.len() {
        let a = y_host[i];
        let b = y_ref[i];

        let diff = (a - b).abs();

        assert!(a.is_equal(b), "Mismatch at index {i}: got {a}, expected {b}, diff={diff}");
    }

    Ok(())
}

#[test]
fn trunc_1() -> Result<(), ZyxError> {
    use zyx::Tensor;

    // Test positive numbers
    let x = Tensor::from([1.7f32, 2.3, 3.9, 4.1]);
    let y = x.trunc();
    let y_host: Vec<f32> = y.try_into()?;

    let expected = [1.0f32, 2.0, 3.0, 4.0];

    for (i, (got, exp)) in y_host.iter().zip(expected.iter()).enumerate() {
        assert!(got.is_equal(*exp), "Mismatch at index {i}: got {got}, expected {exp}");
    }

    // Test negative numbers
    let x = Tensor::from([-1.7f32, -2.3, -3.9, -4.1]);
    let y = x.trunc();
    let y_host: Vec<f32> = y.try_into()?;

    let expected = [-1.0f32, -2.0, -3.0, -4.0];

    for (i, (got, exp)) in y_host.iter().zip(expected.iter()).enumerate() {
        assert!(got.is_equal(*exp), "Mismatch at index {i}: got {got}, expected {exp}");
    }

    // Test integers (should remain unchanged)
    let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
    let y = x.trunc();
    let x_host: Vec<f32> = x.try_into()?;
    let y_host: Vec<f32> = y.try_into()?;
    assert_eq!(x_host, y_host);

    // Test mixed positive and negative
    let x = Tensor::from([1.7f32, -2.3, 3.9, -4.1]);
    let y = x.trunc();
    let y_host: Vec<f32> = y.try_into()?;

    let expected = [1.0f32, -2.0, 3.0, -4.0];

    for (i, (got, exp)) in y_host.iter().zip(expected.iter()).enumerate() {
        assert!(got.is_equal(*exp), "Mismatch at index {i}: got {got}, expected {exp}");
    }

    Ok(())
}

#[test]
fn embedding_test() -> Result<(), ZyxError> {
    // Embedding: one_hot * weight summed over vocab_size dimension
    let weight = Tensor::from([[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]);
    let input = Tensor::from([0u32, 1u32]);

    let b_size = 1u64;
    let s = 2u64;
    let vocab_size = 3u64;
    let embed_size = 2u64;

    let idx = input
        .cast(DType::F32)
        .reshape([b_size, s, 1u64, 1u64])?
        .expand([b_size, s, vocab_size, 1u64])?;
    let arange = Tensor::arange(0, vocab_size as i64, 1)?
        .reshape([1u64, 1u64, vocab_size, 1u64])?
        .cast(DType::F32)
        .expand([b_size, s, vocab_size, 1u64])?;
    let w = weight
        .reshape([1u64, 1u64, vocab_size, embed_size])?
        .expand([b_size, s, vocab_size, embed_size])?;
    let one_hot = arange.equal(&idx)?.cast(DType::F32);
    let result = (one_hot * w).sum([2])?;
    Tensor::realize([&result])?;
    Ok(())
}

#[test]
fn arange_matmul_cos() -> Result<(), ZyxError> {
    let n = 4096u64;
    let dim = 16u64;
    let inv_freq_data: Vec<f32> = (0..dim).map(|i| 0.5f32.powf(i as f32 / dim as f32)).collect();
    let inv_freq = Tensor::from(inv_freq_data.clone()).reshape([1, dim])?;
    let t = Tensor::arange(0u32, n as u32, 1)?.cast(DType::F32).reshape([n, 1])?;
    let freqs = t.matmul(&inv_freq)?;
    let cos_freqs = freqs.cos();
    Tensor::realize([&cos_freqs])?;
    let result: Vec<f32> = cos_freqs.try_into()?;
    for i in 0..n.min(10) as usize {
        for j in 0..dim as usize {
            let expected = (i as f32 * inv_freq_data[j]).cos();
            let got = result[i * dim as usize + j];
            assert!(
                got.is_equal(expected),
                "Mismatch at ({i},{j}): got {got}, expected {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn cos1() -> Result<(), ZyxError> {
    let data: [f32; 16] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657, 4.5, 6.5, 8.1, 9.1, -0.5, -0.9,
    ];
    let zdata: Vec<f32> = Tensor::from(data).cos().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        //assert_eq!(x.cos(), y);
        assert!(x.cos().is_equal(y), "{} != {y}", x.cos());
    }
    Ok(())
}
