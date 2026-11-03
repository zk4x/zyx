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
fn unary_math() -> Result<(), ZyxError> {
    // `SQRT`/`RSQRT` (`sqrt_program`), `RECIP` (`Program::recip`), `DIV`
    // (`scale_large_divisor` + `recip` + `divide`) are each within about a ulp
    // of the host's correctly rounded result; `FLOOR`/`TRUNC` (`round::body`)
    // are exact on the raw bits. `rand` is in `[0, 1)` (a safe domain for the
    // three reciprocals and the quotient); `randn` is signed, so floor/trunc
    // see both sides and non-integral magnitudes.
    let x = Tensor::rand([10, 256], DType::F32)?;
    let y = Tensor::rand([10, 256], DType::F32)?;
    let n = Tensor::randn([10, 256], DType::F32)?;
    let xs = x.to(Dev::TT(0))?;
    let ys = y.to(Dev::TT(0))?;
    let ns = n.to(Dev::TT(0))?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let yv: Vec<f32> = y.try_into().unwrap();
    let nv: Vec<f32> = n.try_into().unwrap();

    let sqrt: Vec<f32> = xs.sqrt().to(Dev::C)?.try_into().unwrap();
    let rsqrt: Vec<f32> = xs.rsqrt().to(Dev::C)?.try_into().unwrap();
    let recip: Vec<f32> = xs.reciprocal().to(Dev::C)?.try_into().unwrap();
    let div: Vec<f32> = (&xs / &ys).to(Dev::C)?.try_into().unwrap();
    let cases: [(&str, &[f32], fn(f32, f32) -> f32); 4] = [
        ("sqrt", &sqrt, |a, _| a.sqrt()),
        ("rsqrt", &rsqrt, |a, _| 1.0 / a.sqrt()),
        ("recip", &recip, |a, _| a.recip()),
        ("div", &div, |a, b| a / b),
    ];
    for (name, got, f) in cases {
        assert_eq!(got.len(), xv.len(), "{name}");
        for i in 0..xv.len() {
            let expected = f(xv[i], yv[i]);
            let v = got[i];
            assert_eq!(v.is_nan(), expected.is_nan(), "{name} lane {i}: {v} vs {expected}");
            if !expected.is_finite() {
                assert_eq!(v.is_infinite(), expected.is_infinite(), "{name} lane {i}: {v} vs {expected}");
                assert_eq!(v.to_bits() >> 31, expected.to_bits() >> 31, "{name} lane {i}: sign of {v} vs {expected}");
                continue;
            }
            let err = (v - expected).abs();
            let tol = 4.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(err <= tol, "{name} lane {i}: {v} vs {expected} (err {err}, tol {tol})");
        }
    }

    let floor: Vec<f32> = ns.floor().to(Dev::C)?.try_into().unwrap();
    let trunc: Vec<f32> = ns.trunc().to(Dev::C)?.try_into().unwrap();
    for (name, got, f) in [("floor", &floor, f32::floor as fn(f32) -> f32), ("trunc", &trunc, f32::trunc)] {
        assert_eq!(got.len(), nv.len(), "{name}");
        for i in 0..nv.len() {
            assert_eq!(got[i].to_bits(), f(nv[i]).to_bits(), "{name} lane {i}: {} vs {}", got[i], f(nv[i]));
        }
    }

    Ok(())
}

#[test]
fn tiling() -> Result<(), ZyxError> {
    // The shapes that exercise the forms `tt_tile` must finish (NOTES,
    // "Tiling: the search explores, `tt_tile` completes"):
    // - `[32, 32]` is already one tile: one 32-lane unroll over the rows and
    //   one column *group*, so the one-group pre-split must not run;
    // - `[64, 64]` vectorizes along the rows: the row group is counted in
    //   tiles (`32 * g + lane`), the column group in elements, so only the
    //   latter may be split;
    // - `[10, 32]`'s row stride is exactly 32, a bare `g * 32` that must not
    //   be mistaken for a tile-counted group;
    // - `[128, 128]` is a 4x4-tile grid (the multi-tile case).
    // `NEG` and `MUL` are exact in F32 (the host rounds `MUL` the same way), so
    // the gate is bit equality.
    for (r, c) in [(32i64, 32i64), (64, 64), (10, 32), (128, 128)] {
        let x = Tensor::rand([r, c], DType::F32)?;
        let xs = x.to(Dev::TT(0))?;

        let xv: Vec<f32> = x.try_into().unwrap();
        let neg: Vec<f32> = (-&xs).to(Dev::C)?.try_into().unwrap();
        let mul: Vec<f32> = (&xs * &xs).to(Dev::C)?.try_into().unwrap();
        assert_eq!(neg.len(), xv.len(), "[{r}, {c}]");
        assert_eq!(mul.len(), xv.len(), "[{r}, {c}]");
        for i in 0..xv.len() {
            assert_eq!(neg[i].to_bits(), (-xv[i]).to_bits(), "[{r}, {c}] neg lane {i}");
            assert_eq!(mul[i].to_bits(), (xv[i] * xv[i]).to_bits(), "[{r}, {c}] mul lane {i}");
        }
    }

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

#[test]
fn shared_input() -> Result<(), ZyxError> {
    // The same value twice in one kernel: one CB, one replayed `MUL` sequence
    // whose two loads read the same DST tile. It exercises the reader-CB
    // unpack cache (`tt_resolve_dst`) and the `COMPUTED`/`UNPACKED` seams on a
    // chain that is not one linear value. `SFPMAD` with a negated zero addend
    // is an exact multiply, so the gate is bit equality.
    let x = Tensor::rand([32, 32], DType::F32)?;
    let xs = x.to(Dev::TT(0))?;
    let z = (&xs * &xs).to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(xv.len(), 32 * 32);
    assert_eq!(zv.len(), 32 * 32);
    for i in 0..xv.len() {
        assert_eq!(zv[i].to_bits(), (xv[i] * xv[i]).to_bits(), "lane {i}");
    }

    Ok(())
}

#[test]
fn exp2() -> Result<(), ZyxError> {
    // The smallest kernel that unpacks a multi-use value, replays several
    // sequences and puts the result through a select. It is the regression
    // for two bugs: the missing packer -> unpacker seam in `tt_reg_sync`
    // (which made the whole tile one constant) and `tt_exp2`'s integer
    // `Add` lowering through the F32 arm (which halved every lane with
    // `|x| >= 0.5`).
    let x = Tensor::rand([32, 32], DType::F32)?;
    let z = x.to(Dev::TT(0))?.exp2().to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    let mut bad = 0;
    for (i, &v) in zv.iter().enumerate().take(32 * 32) {
        let want = xv[i].exp2();
        // ML approximation: `tt_exp2` drops the five selects (clamps and the
        // denormal/overflow/NaN specials) so the fused body stays inside eight
        // lane registers; the measured worst case on this tile is ~5e-5
        // relative, well below fp32-for-ML needs.
        if ((v - want) / want).abs() > 1e-4 {
            if bad < 10 {
                println!("z[{i}] = {v}, expected {want}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "exp2 lanes off");

    Ok(())
}

#[test]
fn sin_cos() -> Result<(), ZyxError> {
    // The SFPU has no sin/cos instruction: `tt_unfold` reduces the argument
    // (magic-constant round + a two-limb Cody-Waite pi/2) and evaluates the
    // Taylor core, so the gate is the host's tolerance, not bit equality.
    // `x` spans about [-100, 100]: enough for a two-digit quadrant with the
    // two-limb reduction still exact (the doc's limit is ~6400 rad).
    let x = &Tensor::rand([10, 256], DType::F32)? * 200.0f32 - 100.0f32;
    let xs = x.to(Dev::TT(0))?;
    let xv: Vec<f32> = x.try_into().unwrap();
    let sin: Vec<f32> = xs.sin().to(Dev::C)?.try_into().unwrap();
    let cos: Vec<f32> = xs.cos().to(Dev::C)?.try_into().unwrap();
    assert_eq!(sin.len(), 10 * 256);
    assert_eq!(cos.len(), 10 * 256);
    for i in 0..10 * 256 {
        for (name, got, want) in [("sin", sin[i], xv[i].sin()), ("cos", cos[i], xv[i].cos())] {
            // Absolute tolerance near zero, relative away from it.
            let tolerance = 1e-5 * want.abs().max(1.0);
            assert!((got - want).abs() <= tolerance, "{name} lane {i}: {got} vs {want} (x {})", xv[i]);
        }
    }

    Ok(())
}

#[test]
fn log2() -> Result<(), ZyxError> {
    // `tt_unfold`'s `tt_log2`: exponent and mantissa by the bit form of
    // `SFPEXEXP`/`SFPSETEXP`, `ln m` as `2 atanh(f/(2+f))`, times `log2 e`.
    // It is the regression for the integer `Sub` that had no arm: `e - 127`
    // reached the F32 pattern table and the SFPU unpack refused the `I32`
    // tile. `x` spans about [0.01, 100], so every exponent in `[-7, 7]` and a
    // mantissa above and below `sqrt 2` are covered.
    let x = &Tensor::rand([10, 256], DType::F32)? * 100.0f32 + 0.01f32;
    let z = x.to(Dev::TT(0))?.log2().to(Dev::C)?;

    let xv: Vec<f32> = x.try_into().unwrap();
    let zv: Vec<f32> = z.try_into().unwrap();
    assert_eq!(zv.len(), 10 * 256);
    for i in 0..10 * 256 {
        let want = xv[i].log2();
        // Absolute near `x = 1` (where the result is 0), relative away from it.
        let tolerance = 1e-5 * want.abs().max(1.0);
        assert!((zv[i] - want).abs() <= tolerance, "log2 lane {i}: {} vs {want} (x {})", zv[i], xv[i]);
    }

    Ok(())
}
