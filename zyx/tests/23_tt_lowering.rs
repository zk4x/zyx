// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent lowering gates, on the simulator.
//!
//! Everything runs in **F32**, and that is deliberate: `ttsim` only models
//! three `(InDataFormat, OutDataFormat)` pairs that reach `Dst` -- `(0,0)`
//! FP32, `(0,4)` TF32 and `(8,8)` INT32 (`ttsim-divergence.md` rows 31 and
//! 87). Every 16-bit unpack to `Dst` is refused there, so a BF16 elementwise
//! kernel cannot run on the simulator at all. There is no implicit conversion
//! in the backend: a BF16 compute kernel is a loud codegen error, not a silent
//! widen.
use zyx::{DType, Dev, Tensor, ZyxError};

#[test]
fn sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::F32)?;

    let z = x.to(Dev::TT(0))?.sigmoid().to(Dev::C)?;

    // Host reference in F32. The SFPU `exp` is a fitted approximation, so
    // the gate is an absolute error, not bit equality.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 256);
    assert_eq!(zv.len(), 10 * 256);
    let mut bad = 0;
    let mut worst = 0f32;
    for i in 0..10 * 256 {
        let expected = 1.0 / (1.0 + (-xv[i]).exp());
        let v = zv[i];
        let err = (v - expected).abs();
        worst = worst.max(err);
        if err > 1e-3 {
            if bad < 10 {
                println!("z[{i}] = {v}, expected {expected} (err {err})");
            }
            bad += 1;
        }
    }
    println!("sigmoid bad: {bad} / {} (worst {worst})", 10 * 256);
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn add() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::F32)?;
    let y = Tensor::rand([10, 256], DType::F32)?;

    let xs = x.to(Dev::TT(0))?;
    let ys = y.to(Dev::TT(0))?;
    let z = (xs + ys).to(Dev::C)?;

    // `SFPMAD` with the constant-one register is exact F32 addition, so the
    // gate is bit equality in principle; keep a margin for the host's own
    // summation order only if it ever shows up.
    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 256);
    assert_eq!(yv.len(), 10 * 256);
    assert_eq!(zv.len(), 10 * 256);
    let mut bad = 0;
    let mut worst = 0f32;
    for i in 0..10 * 256 {
        let expected = xv[i] + yv[i];
        let v = zv[i];
        let err = (v - expected).abs();
        worst = worst.max(err);
        if err > 0.0 {
            if bad < 10 {
                println!("z[{i}] = {v}, expected {expected} (err {err})");
            }
            bad += 1;
        }
    }
    println!("add bad: {bad} / {} (worst {worst})", 10 * 256);
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn neg_sub_mul() -> Result<(), ZyxError> {
    // Three row ops of one SFPU instruction each: `NEG` is `SFPMOV` with the
    // negate modifier, `SUB` is `SFPMAD` with a negated addend, `MUL` is
    // `SFPMAD` with the negated zero register. Each is exact in F32 apart from
    // `MUL`'s one rounding, which is the host's too, so bit equality. `ABS` is
    // not here: `Tensor::abs` is `relu(x) + relu(-x)`, so it never emits
    // `UOp::Abs`. `INT_NOT` is not either: an I32 kernel stops earlier, at
    // `native unpack ... needs an F32 DST tile, got I32`.
    //
    // The operands are already on the card, so each op runs as a kernel of its
    // own; `[10, 256]` is the shape `add` uses, and the one `tt_tile` used to
    // mis-tile for a bare unary (grid `[10, 1]`, one 4 KB page per row).
    let x = Tensor::rand([10, 256], DType::F32)?;
    let y = Tensor::rand([10, 256], DType::F32)?;
    let xs = x.to(Dev::TT(0))?;
    let ys = y.to(Dev::TT(0))?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 256);

    let neg: Vec<f32> = (-&xs).to(Dev::C)?.try_into().unwrap();
    let sub: Vec<f32> = (&xs - &ys).to(Dev::C)?.try_into().unwrap();
    let mul: Vec<f32> = (&xs * &ys).to(Dev::C)?.try_into().unwrap();
    for i in 0..10 * 256 {
        assert_eq!(neg[i].to_bits(), (-xv[i]).to_bits(), "neg lane {i}");
        assert_eq!(sub[i].to_bits(), (xv[i] - yv[i]).to_bits(), "sub lane {i}");
        assert_eq!(mul[i].to_bits(), (xv[i] * yv[i]).to_bits(), "mul lane {i}");
    }

    Ok(())
}

#[test]
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;
    let y = Tensor::rand([20, 12], DType::F32)?;

    let z = x.to(Dev::TT(0))?.matmul(&y.to(Dev::TT(0))?)?.to(Dev::C)?;

    // Host reference: F32 accumulation over K=20 on the Matrix Unit.
    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(yv.len(), 20 * 12);
    assert_eq!(zv.len(), 10 * 12);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        for c in 0..12 {
            let mut acc = 0f32;
            for k in 0..20 {
                acc += xv[r * 20 + k] * yv[k * 12 + c];
            }
            let v = zv[r * 12 + c];
            let err = (v - acc).abs();
            worst = worst.max(err);
            // The Matrix Unit truncates F32 operands to TF32 in `Src`, so the
            // gate is TF32 precision, not F32.
            if err > 1e-2 {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {acc} (err {err})");
                }
                bad += 1;
            }
        }
    }
    println!("matmul bad: {bad} / {} (worst {worst})", 10 * 12);
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn sum() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.sum_keepdim([1])?.to(Dev::C)?;

    // Host reference: per-row sums over k=0..20. The transfer narrows the
    // physical tile to the logical [10, 1] shape.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        let mut acc = 0f32;
        for k in 0..20 {
            acc += xv[r * 20 + k];
        }
        let v = zv[r];
        let err = (v - acc).abs();
        worst = worst.max(err);
        if err > 1e-3 {
            if bad < 10 {
                println!("z[{r}] = {v}, expected {acc} (err {err})");
            }
            bad += 1;
        }
    }
    println!("sum bad: {bad} / 10 (worst {worst})");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn sum_sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.sum_keepdim([1])?.sigmoid().to(Dev::C)?;

    // Host reference: per-row sums over k=0..20, then sigmoid.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        let mut acc = 0f32;
        for k in 0..20 {
            acc += xv[r * 20 + k];
        }
        let expected = 1.0 / (1.0 + (-acc).exp());
        let v = zv[r];
        let err = (v - expected).abs();
        worst = worst.max(err);
        if err > 1e-3 {
            if bad < 10 {
                println!("z[{r}] = {v}, expected {expected} (err {err})");
            }
            bad += 1;
        }
    }
    println!("sum_sigmoid bad: {bad} / 10 (worst {worst})");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn max() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.max_keepdim([1])?.to(Dev::C)?;

    // Host reference: per-row maxima over k=0..20. A `SFPGT` fold is exact,
    // so any real error is a wrong program, not rounding.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        let mut m = f32::NEG_INFINITY;
        for k in 0..20 {
            m = m.max(xv[r * 20 + k]);
        }
        let v = zv[r];
        let err = (v - m).abs();
        worst = worst.max(err);
        if err > 0.0 {
            if bad < 10 {
                println!("z[{r}] = {v}, expected {m} (err {err})");
            }
            bad += 1;
        }
    }
    println!("max bad: {bad} / 10 (worst {worst})");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn sum_scalar() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.sum([0, 1])?.to(Dev::C)?;

    // Host reference: the total, left to right. The device folds the tile
    // row, then the rows in order, so the bound is the usual summation
    // error, not bit equality.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 1);
    let mut acc = 0f32;
    for v in &xv {
        acc += *v;
    }
    let err = (zv[0] - acc).abs();
    println!("sum_scalar: {} vs {} (err {err})", zv[0], acc);
    assert!(err <= 1e-3, "err {err}");

    Ok(())
}

#[test]
fn max_scalar() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.max([0, 1])?.to(Dev::C)?;

    // Host reference: the maximum over everything. Exact, like `max`.
    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 1);
    let m = xv.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let err = (zv[0] - m).abs();
    println!("max_scalar: {} vs {} (err {err})", zv[0], m);
    assert_eq!(err, 0.0);

    Ok(())
}

#[test]
fn sum_all() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;

    // The no-argument reduce (`axes = []` means every axis). It is the same
    // chain as `sum([0, 1])`: `reduce_axes` folds one axis at a time, so the
    // kernel is `Row` then `Col`, never `TileDim::Scalar`.
    let z = x.to(Dev::TT(0))?.sum_all().to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 1);
    let mut acc = 0f32;
    for v in &xv {
        acc += *v;
    }
    let err = (zv[0] - acc).abs();
    println!("sum_all: {} vs {} (err {err})", zv[0], acc);
    assert!(err <= 1e-3, "err {err}");

    Ok(())
}

#[test]
fn matmul_sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;
    let y = Tensor::rand([20, 12], DType::F32)?;

    let z = x.to(Dev::TT(0))?.matmul(&y.to(Dev::TT(0))?)?.sigmoid().to(Dev::C)?;

    // Host reference: matmul accumulation, then sigmoid.
    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(yv.len(), 20 * 12);
    assert_eq!(zv.len(), 10 * 12);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        for c in 0..12 {
            let mut acc = 0f32;
            for k in 0..20 {
                acc += xv[r * 20 + k] * yv[k * 12 + c];
            }
            let expected = 1.0 / (1.0 + (-acc).exp());
            let v = zv[r * 12 + c];
            let err = (v - expected).abs();
            worst = worst.max(err);
            // TF32 as in `matmul`, plus the fitted `exp`.
            if err > 1e-2 {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {expected} (err {err})");
                }
                bad += 1;
            }
        }
    }
    println!("matmul_sigmoid bad: {bad} / {} (worst {worst})", 10 * 12);
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn matmul_add() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::F32)?;
    let y = Tensor::rand([20, 12], DType::F32)?;
    let a = Tensor::rand([10, 20], DType::F32)?;
    let b = Tensor::rand([20, 12], DType::F32)?;

    // Two matmuls, so the second accumulator is not tile 0: this is the shape
    // that would pin `SETRWC`'s `DstVal` (four bits of 32-row halves) and the
    // row counter's restore. It does not reach them yet: the second matmul
    // emits `ZEROACC CLR_ALL`, which would wipe the first accumulator, and
    // codegen refuses that loudly. Once the clear is scoped to the
    // accumulator's own rows, this test is the `SETRWC` gate.
    let m1 = x.to(Dev::TT(0))?.matmul(&y.to(Dev::TT(0))?)?;
    let m2 = a.to(Dev::TT(0))?.matmul(&b.to(Dev::TT(0))?)?;
    let z = (m1 + m2).to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    let av: Vec<f32> = a.try_into().unwrap();
    let bv: Vec<f32> = b.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(zv.len(), 10 * 12);
    let mut bad = 0;
    let mut worst = 0f32;
    for r in 0..10 {
        for c in 0..12 {
            let mut acc = 0f32;
            for k in 0..20 {
                acc += xv[r * 20 + k] * yv[k * 12 + c];
            }
            let mut acc2 = 0f32;
            for k in 0..20 {
                acc2 += av[r * 20 + k] * bv[k * 12 + c];
            }
            let expected = acc + acc2;
            let v = zv[r * 12 + c];
            let err = (v - expected).abs();
            worst = worst.max(err);
            // TF32 in `Src`, as in `matmul`/`matmul_sigmoid`.
            if err > 1e-2 {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {expected} (err {err})");
                }
                bad += 1;
            }
        }
    }
    println!("matmul_add bad: {bad} / {} (worst {worst})", 10 * 12);
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 256], DType::F32)?;

    let z = x.to(Dev::TT(0))?.softmax([1])?.to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 256 * 256);
    assert_eq!(zv.len(), 256 * 256);

    Ok(())
}

#[test]
fn transfer_roundtrip() -> Result<(), ZyxError> {
    // Transfer-only probe: tilize to TT and untilize back, no kernel in
    // between. Exact equality is expected (tilize is a permutation plus zero
    // halo; untilize narrows to the logical shape).
    let x = Tensor::rand([10, 20], DType::F32)?;

    let z = x.to(Dev::TT(0))?.to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10 * 20);
    let mut bad = 0;
    for i in 0..10 * 20 {
        if zv[i] != xv[i] {
            if bad < 10 {
                println!("z[{i}] = {}, expected {}", zv[i], xv[i]);
            }
            bad += 1;
        }
    }
    println!("transfer_roundtrip bad: {bad} / {}", 10 * 20);
    assert_eq!(bad, 0);

    Ok(())
}
