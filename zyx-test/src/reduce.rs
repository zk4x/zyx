use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar, shape::Shape, axes::{Axes, IntoAxes}, dtype::DType};
use rand::{thread_rng, Rng, prelude::SliceRandom};
use super::assert_eq;

pub fn sum<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
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
        let a = rng.gen_range(0..shape.rank());
        let axes = (0..shape.rank()).collect::<Vec<usize>>().choose_multiple(&mut rng, a).copied().collect::<Vec<usize>>();
        if axes.is_empty() {
            continue
        }
        let axes = axes.into_axes(a);
        let x = match T::dtype() {
            DType::F32 | DType::F64 => dev.randn(&shape, T::dtype()),
            DType::I32 => dev.uniform(&shape, i32::MIN/1024/1024..i32::MAX/1024/1024),
        };
        let v: Vec<T> = x.to_vec()?;
        let rv = x.sum(&axes).to_vec()?;
        assert_eq(rv, reduce_op(&shape, &v, &axes, &shape.clone().reduce(&axes), true));
    }
    Ok(())
}

pub fn max<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let mut rng = thread_rng();
    let mut shapes: Vec<Shape> = Vec::new();
    for _ in 0..10 {
        let mut shape = Vec::new();
        for i in 0..rng.gen_range(1..20) {
            let n = 8; // 1024usize;
            let n = if i > 1 {
                n * n / shape.iter().product::<usize>()
            } else {
                n
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
        let a = rng.gen_range(0..shape.rank());
        let axes = (0..shape.rank()).collect::<Vec<usize>>().choose_multiple(&mut rng, a).copied().collect::<Vec<usize>>();
        if axes.is_empty() {
            continue
        }
        let axes = axes.into_axes(a);
        println!("{shape}, {axes}");
        let two = T::one().add(T::one());
        let x = dev.uniform(&shape, T::min_value().div(two.clone())..T::max_value().div(two));
        let v: Vec<T> = x.to_vec()?;
        let rv = x.max(&axes).to_vec()?;
        let rv_org = reduce_op(&shape, &v, &axes, &shape.clone().reduce(&axes), false);
        //println!("{x}\n\n{}\n\n{rv_org:?}", x.max(&axes));
        assert_eq(rv, rv_org);
    }
    Ok(())
}

fn reduce_op<T: Scalar>(
    shape: &Shape,
    data: &[T],
    axes: &Axes,
    res_shape: &Shape,
    sum_reduce: bool // sum or max?
) -> Vec<T> {
    /*if axes.len() == 0 {
        return data.iter().cloned().collect()
    }*/
    // Strides of the input
    let strides = shape.strides();
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.rank()).filter(|x| !axes.contains(*x)).collect();
    // Strides of the result
    let res_strides = res_shape.strides();
    let mut res: Vec<T> = if sum_reduce {
        core::iter::repeat(T::zero())
    } else {
        core::iter::repeat(T::min_value())
    }.take(res_shape.numel()).collect();

    // Go over all data and apply sum function to correct values
    // then indices can be added just by making another vector and constantly
    // updating it (adding in case of sum) with new indices as new max/min are found
    for i in 0..shape.numel() {
        // calculate index in result
        let mut j = 0;
        for dim in &*included_dims {
            j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim];
        }
        // apply reduce function, in this case sum
        if sum_reduce {
            res[j] = Scalar::add(res[j].clone(), data[i].clone());
        } else {
            res[j] = Scalar::max(res[j].clone(), data[i].clone());
        }
    }
    res
}
