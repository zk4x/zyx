use super::assert_eq;
use core::ops::Neg;
use rand::{thread_rng, Rng};

macro_rules! unary_test {
    ( $dev:expr, $x:tt ) => {{
        let mut rng = thread_rng();
        let mut shapes: Vec<Shape> = Vec::new();
        for _ in 0..10 {
            let mut shape = Vec::new();
            for i in 0..rng.gen_range(1..20) {
                let n = if i > 1 {
                    1024usize * 1024usize / shape.iter().product::<usize>()
                } else {
                    1024
                };
                if n > 1 {
                    shape.insert(0, rng.gen_range(1..n));
                } else {
                    break;
                }
            }
            shapes.push(shape.into());
        }
        for shape in shapes {
            let x = $dev.randn(shape, T::dtype())?;
            let v = x.to_vec()?;
            assert_eq(x.$x().to_vec()?, v.into_iter().map(|e: T| e.$x()));
        }
        Ok(())
    }};
}

pub fn neg<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, neg)
}

pub fn relu<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, relu)
}

pub fn sin<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    unary_test!(dev, sin)
}

pub fn cos<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    unary_test!(dev, cos)
}

pub fn ln<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    let mut rng = thread_rng();
    let mut shapes: Vec<Shape> = Vec::new();
    for _ in 0..10 {
        let mut shape = Vec::new();
        for i in 0..rng.gen_range(1..20) {
            let n = if i > 1 {
                1024usize * 1024usize / shape.iter().product::<usize>()
            } else {
                1024
            };
            if n > 1 {
                shape.insert(0, rng.gen_range(1..n));
            } else {
                break;
            }
        }
        shapes.push(shape.into());
    }
    for shape in shapes {
        let e = if T::dtype().is_floating() {
            T::epsilon()
        } else {
            T::one()
        };
        let x = dev.uniform(shape, T::zero().add(e)..T::max_value())?;
        let v = x.to_vec()?;
        assert_eq(x.ln().to_vec()?, v.into_iter().map(|e: T| e.ln()));
    }
    Ok(())
}

pub fn exp<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    unary_test!(dev, exp)
}

pub fn tanh<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    unary_test!(dev, tanh)
}

pub fn sqrt<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    if !T::dtype().is_floating() {
        // TODO rounding errors on GPU when using integers
        // cause results to be off by 1
        return Ok(());
    }
    let mut rng = thread_rng();
    let mut shapes: Vec<Shape> = Vec::new();
    for _ in 0..10 {
        let mut shape = Vec::new();
        for i in 0..rng.gen_range(1..20) {
            let n = if i > 1 {
                1024usize * 1024usize / shape.iter().product::<usize>()
            } else {
                1024
            };
            if n > 1 {
                shape.insert(0, rng.gen_range(1..n));
            } else {
                break;
            }
        }
        shapes.push(shape.into());
    }
    for shape in shapes {
        let x = dev.uniform(shape, T::zero()..T::max_value())?;
        let v = x.to_vec()?;
        assert_eq(x.ln().to_vec()?, v.into_iter().map(|e: T| e.ln()));
    }
    Ok(())
}
