use super::assert_eq;
use core::ops::{Add, Mul, Sub};
use rand::{thread_rng, Rng};
use zyx::{DType, Tensor};

macro_rules! binary_test {
    ( $dev:expr, $x:tt ) => {{
        let mut rng = thread_rng();
        let mut shapes: Vec<Vec<usize>> = Vec::new();
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
        for shape in &shapes {
            // Since overflow is implementation/hardware defined, we need to limit integers
            // appropriatelly
            let (x, y) = if T::dtype().is_floating() {
                (
                    $dev.randn(shape, T::dtype())?,
                    $dev.randn(shape, T::dtype())?,
                )
            } else {
                (
                    $dev.uniform(shape, T::min_value().sqrt()..T::max_value().sqrt())?,
                    $dev.uniform(shape, T::min_value().sqrt()..T::max_value().sqrt())?,
                )
            };
            let vx = x.to_vec()?;
            let vy = y.to_vec()?;
            assert_eq(
                x.$x(y).to_vec()?,
                vx.into_iter().zip(vy).map(|(ex, ey): (T, T)| ex.$x(ey)),
            );
        }
        Ok(())
    }};
}

pub fn add(dtype: DType) {
    binary_test!(dtype, add)
}

pub fn sub(dtype: DType) {
    binary_test!(dtype, sub)
}

pub fn mul(dtype: DType) {
    binary_test!(dtype, mul)
}

pub fn div(dtype: DType) {
    // TODO why does this not work???
    /*let mut rng = thread_rng();
    let mut shapes: Vec<Shape> = Vec::new();
    for _ in 0..10 {
        let mut shape = Vec::new();
        for i in 0..rng.gen_range(1..20) {
            let n = if i > 1 {
                1024usize*1024usize/shape.iter().product::<usize>()
            } else {
                1024
            };
            if n > 1 {
                shape.insert(0, rng.gen_range(1..n));
            } else {
                break
            }
        }
        shapes.push(shape.into());
    }
    for shape in &shapes {
        // Since overflow is implementation/hardware defined, we need to limit integers
        // appropriatelly
        let (x, y) = if T::dtype().is_floating() {
            // TODO replace with where(0, 1) operator once we have it
            let mut lower = T::zero();
            for _ in 0..100 {
                lower = lower.add(T::epsilon());
            }
            (dev.uniform(shape, lower.clone()..T::max_value()),
             dev.uniform(shape, lower..T::max_value()))
        } else {
            (dev.uniform(shape, T::zero().add(T::one())..T::max_value()),
             dev.uniform(shape, T::zero().add(T::one())..T::max_value()))
        };
        let vx = x.to_vec()?;
        let vy = y.to_vec()?;
        assert_eq(x.div(y).to_vec()?, vx.into_iter().zip(vy).map(|(ex, ey): (T, T)| ex.div(ey)));
    }
    Ok(())*/
}

pub fn pow(dtype: DType) {
    let mut rng = thread_rng();
    let mut shapes: Vec<Vec<usize>> = Vec::new();
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
    for shape in &shapes {
        // Since overflow is implementation/hardware defined, we need to limit integers
        // appropriatelly
        let (x, y) = if T::dtype().is_floating() {
            (Tensor::randn(shape, dtype)?, Tensor::randn(shape, dtype)?)
        } else {
            let six = T::one()
                .add(T::one())
                .add(T::one())
                .add(T::one())
                .add(T::one())
                .add(T::one());
            (
                Tensor::uniform(shape, T::min_value().sqrt().sqrt().sqrt()..six.clone())?,
                Tensor::uniform(shape, T::min_value().sqrt().sqrt().sqrt()..six)?,
            )
        };
        let vx = x.to_vec()?;
        let vy = y.to_vec()?;
        assert_eq(
            x.pow(y).to_vec()?,
            vx.into_iter().zip(vy).map(|(ex, ey): (T, T)| ex.pow(ey)),
        );
    }
    Ok(())
}

pub fn cmplt(dtype: DType) {
    binary_test!(dtype, cmplt)
}
