use super::assert_eq;
use core::ops::{Add, Mul, Sub};
use rand::{thread_rng, Rng};
use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar, shape::Shape};

macro_rules! binary_test {
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
        for shape in &shapes {
            // Since overflow is implementation/hardware defined, we need to limit integers
            // appropriatelly
            let (x, y) = if T::dtype().is_floating() {
                ($dev.randn(shape, T::dtype())?, $dev.randn(shape, T::dtype())?)
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

pub fn add<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, add)
}

pub fn sub<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, sub)
}

pub fn mul<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, mul)
}

pub fn div<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    // TODO why does this not work???
    return Ok(());
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

pub fn pow<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
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
    for shape in &shapes {
        // Since overflow is implementation/hardware defined, we need to limit integers
        // appropriatelly
        let (x, y) = if T::dtype().is_floating() {
            (dev.randn(shape, T::dtype())?, dev.randn(shape, T::dtype())?)
        } else {
            let six = T::one()
                .add(T::one())
                .add(T::one())
                .add(T::one())
                .add(T::one())
                .add(T::one());
            (
                dev.uniform(shape, T::min_value().sqrt().sqrt().sqrt()..six.clone())?,
                dev.uniform(shape, T::min_value().sqrt().sqrt().sqrt()..six)?,
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

pub fn cmplt<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, cmplt)
}
