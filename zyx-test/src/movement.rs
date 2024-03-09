use super::assert_eq;
use zyx_core::{axes::Axes, backend::Backend, error::ZyxError, scalar::Scalar, shape::Shape};

pub fn reshape<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.randn([2, 4, 1, 5], T::dtype())?;
    let v: Vec<T> = x.to_vec()?;
    let y = x.reshape([8, 5]);
    assert_eq!(y.shape(), [8, 5]);
    assert_eq(y.to_vec()?, v.into_iter());
    Ok(())
}

pub fn permute<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    use zyx_core::axes::IntoAxes;
    let sh = [2, 4, 1, 5].into();
    let ax = [-3, 3, 0, 2].into_axes(4);
    let x = dev.randn(&sh, T::dtype())?;
    let v: Vec<T> = x.to_vec()?;
    let y = x.permute(&ax);
    assert_eq!(y.shape(), [4, 5, 2, 1]);
    assert_eq(y.to_vec()?, _permute(&v, sh, ax).into_iter());
    Ok(())
}

pub fn expand<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let sh = [2, 4, 1, 5].into();
    let rsh = [2, 4, 10, 5].into();
    let x = dev.randn(&sh, T::dtype())?;
    let v: Vec<T> = x.to_vec()?;
    let y = x.expand(&rsh);
    assert_eq!(y.shape(), [2, 4, 10, 5]);
    assert_eq(y.to_vec()?, _expand(v, sh, rsh).into_iter());
    Ok(())
}

pub fn pad<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[4, 3, 2, 1], [4, 2, 3, 8]])?;
    let y = x.pad([(1, 0), (0, 1)], 0);
    assert_eq!(y.shape(), [3, 5]);
    //println!("{y}");
    assert_eq!(y, [[0, 4, 3, 2, 1], [0, 4, 2, 3, 8], [0, 0, 0, 0, 0]]);
    Ok(())
}

// Ground truth function, slow but verified
fn _permute<T: Scalar>(data: &[T], shape: Shape, axes: Axes) -> Vec<T> {
    let ndim = shape.rank();
    let strides = shape.strides().permute(&axes);
    let mut acc_var = 1;
    let acc = Shape::from(
        shape
            .into_iter()
            .rev()
            .map(|x| {
                acc_var *= x;
                acc_var
            })
            .collect::<Vec<usize>>()
            .into_iter()
            .rev()
            .collect::<Vec<usize>>(),
    )
    .permute(&axes);
    let n = shape.numel();
    // temp is in reverse order
    let mut temp = vec![(0, 0); ndim]; // strides, acc_shape
    let mut begins = vec![0; ndim];
    for k in 0..ndim {
        temp[ndim - k - 1] = (strides[k], acc[k]);
    }
    let mut r_data: Vec<T> = Vec::with_capacity(n);
    let mut i = 0;
    for _ in 0..n {
        r_data.push(data[i].clone());
        for (j, (st, acc)) in temp.iter().enumerate() {
            begins[j] += st;
            i += st;
            if begins[j] < *acc {
                break;
            } else {
                i -= begins[j];
                begins[j] = 0;
            }
        }
    }
    r_data
}

// Ground truth function, slow but verified
fn _expand<T: Scalar>(mut data: Vec<T>, shape: Shape, res_shape: Shape) -> Vec<T> {
    let mut shape: Vec<usize> = shape.into();
    let mut res_shape: Vec<usize> = res_shape.into();
    // if input shape is shorter than res_shape or vice versa,
    // add necessary ones to the beginning
    while shape.len() < res_shape.len() {
        shape.insert(0, 1);
    }
    while shape.len() > res_shape.len() {
        res_shape.insert(0, 1);
    }
    let n = shape.iter().product();
    let ndims = shape.len();

    let copy_dim = |data: Vec<T>, width, times| {
        let mut res_data = Vec::with_capacity(res_shape.iter().product());
        for i in (0..n).step_by(width) {
            // copy this part of vec
            for _ in 0..times {
                res_data.extend_from_slice(&data[i..i + width]);
            }
        }
        res_data
    };

    let mut i = ndims;
    let mut width = 1;
    for (d, r) in shape
        .clone()
        .into_iter()
        .zip(res_shape.clone().into_iter())
        .rev()
    {
        i -= 1;
        if d != r {
            if d == 1 {
                data = copy_dim(data, width, r / d);
            } else {
                panic!(
                    "Incompatible input: {:?} and expand shape: {:?} on dim {:?}",
                    shape, res_shape, i
                );
            }
        }
        width *= res_shape[i];
    }
    data
}
