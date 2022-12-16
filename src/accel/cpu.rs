//! Buffer is multidimensional storage type using cpu for the calculations
//! and rayon for multithreading. It can optionally use matrixmultiply crate.
//!

use crate::{ops::{self, ConvertFrom}, shape::{Shape, BinOpShape}, dtype::ScalarType};
use core::ops::{Add, Mul};
extern crate alloc;
use alloc::{vec, sync::Arc, format};

/// Generic multidimensional buffer
/// 
/// Each buffer has a shape and data stored in vec.
/// Data is stored in row major order.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Buffer<T, Sh> {
    shape: Sh,
    data: Arc<alloc::vec::Vec<T>>,
}

impl<T, Sh> Clone for Buffer<T, Sh>
where
    Sh: Clone,
{
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T, Sh> Buffer<T, Sh> {
    /// Get Buffer's estimated memory size
    pub fn est_mem_size(&self) -> usize {
        self.data.len() * core::mem::size_of::<T>()
    }
}

// Convert between Buffers with different datatypes
impl<T, T2, Sh> ConvertFrom<Buffer<T2, Sh>> for Buffer<T, Sh>
where
    T: ConvertFrom<T2> + Send + Sync + ScalarType,
    T2: Clone + Send + Sync + ScalarType,
    Sh: Shape<D = usize>,
{
    fn cfrom(x: Buffer<T2, Sh>) -> Self {
        use rayon::prelude::*;
        use crate::ops::ConvertInto;
        Self {
            shape: x.shape,
            data: Arc::new(x.data.as_ref().par_iter().map(|x| x.clone().cinto()).collect()),
        }
    }
}

// Display Buffer
impl<T, Sh> core::fmt::Display for Buffer<T, Sh>
where
    T: core::fmt::Display + ScalarType,
    Sh: Shape<D = usize>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        extern crate alloc;
        use alloc::string::String;
        let mut res = String::new();
        if self.data.is_empty() { return f.write_str(&(res + "[]")); }
        let n = self.shape.numel();
        let ndim = Sh::N;
        //const PRECISION: usize = 3;
        // get maximal width of single value
        let mut w = 0;
        for x in self.data.as_ref().iter() {
            let l = format!("{x:w$}").len();
            if l > w { w = l; }
        }
        let d0 = self.shape.ati(-1);
        for i in 0..n {
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if i % (n/var) == 0 {
                        res += &(" ".repeat(ndim - r)+&"[".repeat(r - 1));
                        break
                    }
                    var *= self.shape.at(ndim - r);
                    r -= 1;
                }
            }
            use core::fmt::Write;
            let _ = write!(res, "{0:>1$}", self.data[i], w);
            if (i + 1) % d0 != 0usize { res += " "; }
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if (i + 1) % (n/var) == 0 {
                        res += &"]".repeat(r-1);
                        break
                    }
                    var *= self.shape.at(ndim - r);
                    r -= 1;
                }
            }
            if (i + 1) % d0 == 0usize && i != n - 1 { res += "\n"; }
        }
        f.write_str(&res)
    }
}

/// Get Buffer represented as vector.
/// It is flattened with row major order.
impl<T, Sh> ops::IntoVec<T> for Buffer<T, Sh>
where
    T: Clone + ScalarType,
    Sh: Shape<D = usize>,
{
    fn to_vec(&self) -> alloc::vec::Vec<T> {
        self.data.as_ref().clone()
    }
}

/// Create new Buffer from vec
impl<T, Sh> ops::FromVec for Buffer<T, Sh>
where
    Sh: Shape<D = usize>,
    T: Clone,
{
    type T = T;
    type Sh = Sh;

    fn from_vec(data: &[T], shape: Sh) -> Self {
        assert_eq!(shape.numel(), data.len());
        Self {
            shape,
            data: Arc::new(data.to_vec()),
        }
    }
}

/// Get Buffer's shape
impl<T, Sh> ops::GetShape for Buffer<T, Sh>
where
    Sh: Shape<D = usize>,
{
    type Output = Sh;

    fn shape(&self) -> Self::Output {
        self.shape
    }
}

/// Create new Buffer filled with zeros
impl<T, Sh> ops::Zeros for Buffer<T, Sh>
where
    T: Clone + ops::Zeros<Sh = ()> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Sh = Sh;

    fn zeros(shape: Sh) -> Self {
        let n = shape.numel();
        Self {
            shape,
            data: Arc::new(vec![T::zeros(()); n]),
        }
    }
}

/// Create new Buffer filled with ones
impl<T, Sh> ops::Ones for Buffer<T, Sh>
where
    T: Clone + ops::Ones<Sh = ()> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Sh = Sh;

    fn ones(shape: Sh) -> Self {
        let n = shape.numel();
        Self {
            shape,
            data: Arc::new(vec![T::ones(()); n]),
        }
    }
}

fn unary_op<T, Sh, F>(x: Buffer<T, Sh>, f: F) -> Buffer<T, Sh>
where
    T: Clone + Sync + Send + ScalarType,
    F: Fn(T) -> T + Sync + Send,
    Sh: Shape<D = usize>,
{
    use rayon::prelude::*;
    Buffer {
        shape: x.shape,
        data: Arc::new(match Arc::try_unwrap(x.data) {
            Ok(vec) => vec.into_par_iter().map(f).collect(),
            Err(rc) => rc.as_ref().par_iter().map(|x| f(x.clone())).collect(),
        }),
    }
}

impl<T, Sh> ops::ReLU for Buffer<T, Sh>
where
    T: Clone + Sync + Send + ops::ReLU<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn relu(self) -> Self::Output {
        unary_op(self, |x| x.relu())
    }
}

impl<T, Sh> ops::DReLU for Buffer<T, Sh>
where
    T: Clone + Sync + Send + ops::DReLU<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn drelu(self) -> Self::Output {
        unary_op(self, |x| x.drelu())
    }
}

impl<T, Sh> ops::Exp for Buffer<T, Sh>
where
    T: Clone + Sync + Send + ops::Exp<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn exp(self) -> Self::Output {
        unary_op(self, |x| x.exp())
    }
}

impl<T, Sh> ops::Ln for Buffer<T, Sh>
where
    T: Clone + Sync + Send + ops::Ln<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn ln(self) -> Self::Output {
        unary_op(self, |x| x.ln())
    }
}

impl<T, Sh> ops::Tanh for Buffer<T, Sh>
where
    T: Clone + Sync + Send + ops::Tanh<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn tanh(self) -> Self::Output {
        unary_op(self, |x| x.tanh())
    }
}

impl<T, Sh> core::ops::Neg for Buffer<T, Sh>
where
    T: Clone + Sync + Send + core::ops::Neg<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn neg(self) -> Self::Output {
        unary_op(self, |x| x.neg())
    }
}

impl<T, Sh> Buffer<T, Sh>
where
    Sh: Shape<D = usize>,
{
    fn reduce<Dims, F>(self, dims: Dims, init: T, mut f: F) -> Buffer<T, Sh>
    where
        Dims: Shape<D = i32>,
        T: Clone + ScalarType,
        F: FnMut(T, T) -> T,
    {
        use alloc::borrow::ToOwned;
        // TODO: make this multithreaded
        let mut data = Arc::try_unwrap(self.data).unwrap_or_else(|x| x.as_ref().to_owned());
        let mut shape = self.shape;

        let mut reduce_dim = |data: &[T], dim: i32| {
            let strides = shape.strides();
            let stride = strides.ati(dim);
            *shape.mut_ati(dim) = 1;
            let mut res = vec![init.clone(); shape.numel()];
            if dim == 0 || dim == -(Sh::N as i32) {
                for (i, x) in data.iter().enumerate() {
                    let idx = i % stride;
                    res[idx] = f(res[idx].clone(), x.clone());
                }
            } else {
                let width = strides.ati(dim - 1);
                for (i, x) in data.iter().enumerate() {
                    let idx = i/width*stride + i % stride;
                    res[idx] = f(res[idx].clone(), x.clone());
                }
            }
            res
        };

        if dims.is_empty() {
            for dim in 0..Sh::N {
                data = reduce_dim(&data, dim as i32);
            }
        } else {
            for i in 0..Dims::N {
                data = reduce_dim(&data, dims.at(i));
            }
        }

        Buffer {
            shape,
            data: Arc::new(data),
        }
    }
}

impl<T, Sh, Dims> ops::Sum<Dims> for Buffer<T, Sh>
where
    T: Clone + ops::Zeros<Sh = ()> + core::ops::Add<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
    Dims: Shape<D = i32>,
{
    type Output = Buffer<T, Sh>;

    fn sum(self, dims: Dims) -> Self::Output {
        self.reduce(dims, T::zeros(()), |a, b| a + b)
    }
}

impl<T, Sh, Dims> ops::Max<Dims> for Buffer<T, Sh>
where
    T: Clone + Default + ops::Min<i32, Output = T> + PartialOrd + ScalarType,
    Sh: Shape<D = usize>,
    Dims: Shape<D = i32>,
{
    type Output = Buffer<T, Sh>;
    fn max(self, dims: Dims) -> Self::Output {
        self.reduce(dims, T::min(T::default(), 0), |a, b| if a > b { a } else { b })
    }
}

impl<T, Sh, Dims> ops::Min<Dims> for Buffer<T, Sh>
where
    T: Clone + Default + ops::Max<i32, Output = T> + PartialOrd + ScalarType,
    Sh: Shape<D = usize>,
    Dims: Shape<D = i32>,
{
    type Output = Buffer<T, Sh>;
    fn min(self, dims: Dims) -> Self::Output {
        self.reduce(dims, T::max(T::default(), 0), |a, b| if a < b { a } else { b })
    }
}

impl<T, Sh, Sh2> ops::Reshape<Sh2> for Buffer<T, Sh>
where
    T: Clone +ScalarType,
    Sh: Shape<D = usize>,
    Sh2: Shape<D = usize>,
{
    type Output = Buffer<T, Sh2>;
    fn reshape(self, shape: Sh2) -> Self::Output {
        assert_eq!(self.shape.numel(), shape.numel());
        Buffer {
            shape,
            data: self.data,
        }
    }
}

impl <T, Sh, Sh2> ops::Expand<Sh2> for Buffer<T, Sh>
where
    T: Clone + ScalarType,
    Sh: Shape<D = usize>,
    Sh2: Shape<D = usize>,
{
    type Output = Buffer<T, Sh2>;
    fn expand(self, res_shape: Sh2) -> Self::Output {
        assert!(Sh2::N >= Sh::N);

        let n = self.shape.numel();
        use alloc::borrow::ToOwned;
        let mut data = Arc::try_unwrap(self.data).unwrap_or_else(|x| x.as_ref().to_owned());

        let copy_dim = |data: alloc::vec::Vec<T>, width, times| {
            let mut res_data = alloc::vec::Vec::with_capacity(res_shape.numel());
            for i in (0..n).step_by(width) {
                // copy this part of vec
                for _ in 0..times {
                    res_data.extend_from_slice(&data[i..i+width]);
                }
            }
            res_data
        };

        let mut i = Sh2::N;
        let mut width = 1;
        while i > 0 {
            i -= 1;
            let d = if Sh::N - i > 0 { self.shape.at(i) } else { 1 };
            let r = res_shape.at(i);
            if d != r {
                if d == 1 {
                    data = copy_dim(data, width, r/d);
                } else {
                    panic!("Incompatible input: {:?} and expand shape: {:?} on dim {:?}", self.shape, res_shape, i);
                }
            }
            width *= res_shape.at(i);
        }

        Buffer {
            shape: res_shape,
            data: Arc::new(data),
        }
    }
}

impl<T, Sh, Dims> ops::Permute<Dims> for Buffer<T, Sh>
where
    Sh: Shape<D = usize> + ops::Permute<Dims, Output = Sh>,
    Dims: Shape<D = i32>,
    T: Clone + ops::Zeros + ScalarType,
{
    type Output = Buffer<T, Sh>;
    fn permute(self, dims: Dims) -> Self::Output {
        let shape = self.shape.permute(dims);
        let strides = self.shape.strides().permute(dims);
        let mut acc_var = 1;
        let mut acc = Sh::ones();
        for i in 0..Sh::N {
            acc_var *= self.shape.at(Sh::N - i - 1);
            *acc.mut_at(Sh::N - i - 1) = acc_var;
        }
        acc = acc.permute(dims);
        let n = shape.numel();
        // temp is in reverse order
        let mut temp = vec![(0, 0); Sh::N]; // strides, acc_shape
        let mut begins = vec![0; Sh::N];
        for k in 0..Sh::N {
            temp[Sh::N-k-1] = (strides.at(k), acc.at(k));
        }
        let mut data = alloc::vec::Vec::with_capacity(n);
        let mut i = 0;
        for _ in  0..n {
            data.push(self.data[i].clone());
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

        Buffer {
            shape,
            data: Arc::new(data),
        }
    }
}

/*impl<T> ops::Slice for &Buffer<T, Sh>
where
    T: Clone + ops::Zeros + core::ops::Add<Output = T>,
{
    type Output = Buffer<T, Sh>;
    fn slice(self, dims: &[u8]) -> Self::Output {
        todo!()
    }
}*/

fn binary_op<T, F, XSh, YSh>(x: Buffer<T, XSh>, y: Buffer<T, YSh>, f: F) -> Buffer<T, <XSh as crate::shape::BinOpShape<YSh>>::Output>
where
    T: Sync + Send + Clone + ScalarType,
    F: Fn((T, T)) -> T + Sync + Send,
    XSh: Shape<D = usize> + crate::shape::BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    use rayon::prelude::*;
    use ops::Expand;
    use core::cmp::Ordering;
    //let mut shape = x.shape.clone();
    let mut shape = <XSh as BinOpShape<YSh>>::Output::ones();
    // TODO: fix this, so that it is not expanding, but rather using strides to not have to copy
    // stuff during expanding
    let data = Arc::new(match x.shape.numel().cmp(&y.shape.numel()) {
        Ordering::Greater => {
            for i in 0..XSh::N {
                *shape.mut_at(i) = x.shape.at(i);
            }
            match Arc::try_unwrap(x.data) {
                Ok(vec) => match Arc::try_unwrap(y.expand(x.shape).data) {
                    Ok(vec_y) => vec.into_par_iter().zip(vec_y.into_par_iter()).map(f).collect(),
                    Err(rc_y) => vec.into_par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x, y.clone()))).collect(),
                }
                Err(rc) => match Arc::try_unwrap(y.expand(x.shape).data) {
                    Ok(vec_y) => rc.as_ref().par_iter().zip(vec_y.into_par_iter()).map(|(x, y)| f((x.clone(), y))).collect(),
                    Err(rc_y) => rc.as_ref().par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x.clone(), y.clone()))).collect(),
                }
            }
        }
        Ordering::Less => {
            for i in 0..YSh::N {
                *shape.mut_at(i) = y.shape.at(i);
            }
            match Arc::try_unwrap(y.data) {
                Ok(vec) => match Arc::try_unwrap(x.expand(y.shape).data) {
                    Ok(vec_x) => vec_x.into_par_iter().zip(vec.into_par_iter()).map(f).collect(),
                    Err(rc_x) => rc_x.par_iter().zip(vec.into_par_iter()).map(|(x, y)| f((x.clone(), y))).collect(),
                },
                Err(rc) => match Arc::try_unwrap(x.expand(y.shape).data) {
                    Ok(vec_x) => vec_x.into_par_iter().zip(rc.as_ref().par_iter()).map(|(x, y)| f((x, y.clone()))).collect(),
                    Err(rc_x) => rc_x.par_iter().zip(rc.as_ref().par_iter()).map(|(x, y)| f((x.clone(), y.clone()))).collect(),
                }
            }
        }
        Ordering::Equal => {
            for i in 0..XSh::N {
                *shape.mut_at(i) = x.shape.at(i);
            }
            match Arc::try_unwrap(x.data) {
                Ok(vec) => match Arc::try_unwrap(y.data) {
                    Ok(vec_y) => vec.into_par_iter().zip(vec_y.into_par_iter()).map(f).collect(),
                    Err(rc_y) => vec.into_par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x, y.clone()))).collect(),
                },
                Err(rc) => match Arc::try_unwrap(y.data) {
                    Ok(vec_y) => rc.par_iter().zip(vec_y.into_par_iter()).map(|(x, y)| f((x.clone(), y))).collect(),
                    Err(rc_y) => rc.par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x.clone(), y.clone()))).collect(),
                }
            }
        }
    });
    Buffer {
        shape,
        data,
    }
}

impl<T, XSh, YSh> core::ops::Add<Buffer<T, YSh>> for Buffer<T, XSh>
where
    T: Clone + Sync + Send + core::ops::Add<Output = T> + ScalarType,
    XSh: Shape<D = usize> + BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    type Output = Buffer<T, <XSh as BinOpShape<YSh>>::Output>;

    fn add(self, rhs: Buffer<T, YSh>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a + b)
    }
}

use duplicate::duplicate_item;
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T, Sh> core::ops::Add<Buffer<T, Sh>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Add<Output = T> + Clone + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn add(self, rhs: Buffer<T, Sh>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: Arc::new(
                match Arc::try_unwrap(rhs.data) {
                    Ok(vec) => vec.into_par_iter().map(|y| x.clone() + y).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() + y.clone()).collect(),
                }),
        }
    }
}

impl<T, T2, Sh> core::ops::Add<T2> for Buffer<T, Sh>
where
    T2: ScalarType,
    T: Clone + Sync + Send + core::ops::Add<Output = T> + ConvertFrom<T2> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn add(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use crate::ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: Arc::new(
                match Arc::try_unwrap(self.data) {
                    Ok(vec) => vec.into_par_iter().map(|x| x + rhs.clone()).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() + rhs.clone()).collect(),
                }),
        }
    }
}

impl<T, XSh, YSh> core::ops::Sub<Buffer<T, YSh>> for Buffer<T, XSh>
where
    T: Clone + Sync + Send + core::ops::Sub<Output = T> + ScalarType,
    XSh: Shape<D = usize> + BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    type Output = Buffer<T, <XSh as BinOpShape<YSh>>::Output>;
    fn sub(self, rhs: Buffer<T, YSh>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a - b)
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T, Sh> core::ops::Sub<Buffer<T, Sh>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Sub<Output = T> + Clone + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn sub(self, rhs: Buffer<T, Sh>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: Arc::new(
                match Arc::try_unwrap(rhs.data) {
                    Ok(vec) => vec.into_par_iter().map(|y| x.clone() - y).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() - y.clone()).collect(),
                }),
        }
    }
}

impl<T, T2, Sh> core::ops::Sub<T2> for Buffer<T, Sh>
where
    T2: ScalarType,
    T: Clone + Sync + Send + core::ops::Sub<Output = T> + ConvertFrom<T2> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn sub(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use crate::ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: Arc::new(
                match Arc::try_unwrap(self.data) {
                    Ok(vec) => vec.into_par_iter().map(|x| x - rhs.clone()).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() - rhs.clone()).collect(),
                }),
        }
    }
}

impl<T, XSh, YSh> core::ops::Mul<Buffer<T, YSh>> for Buffer<T, XSh>
where
    T: Clone + Sync + Send + core::ops::Mul<Output = T> + ScalarType,
    XSh: Shape<D = usize> + BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    type Output = Buffer<T, <XSh as BinOpShape<YSh>>::Output>;
    fn mul(self, rhs: Buffer<T, YSh>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a * b)
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T, Sh> core::ops::Mul<Buffer<T, Sh>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Mul<Output = T> + Clone + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn mul(self, rhs: Buffer<T, Sh>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: Arc::new(match Arc::try_unwrap(rhs.data) {
                Ok(vec) => vec.into_par_iter().map(|y| x.clone() * y).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() * y.clone()).collect(),
            }),
        }
    }
}

impl<T, T2, Sh> core::ops::Mul<T2> for Buffer<T, Sh>
where
    T2: ScalarType,
    T: Clone + Sync + Send + core::ops::Mul<Output = T> + ops::ConvertFrom<T2> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn mul(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let y: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x * y.clone()).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() * y.clone()).collect(),
            }),
        }
    }
}

impl<T, XSh, YSh> core::ops::Div<Buffer<T, YSh>> for Buffer<T, XSh>
where
    T: Clone + Sync + Send + core::ops::Div<Output = T> + ScalarType,
    XSh: Shape<D = usize> + BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    type Output = Buffer<T, <XSh as BinOpShape<YSh>>::Output>;
    fn div(self, rhs: Buffer<T, YSh>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a / b)
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T, Sh> core::ops::Div<Buffer<T, Sh>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Div<Output = T> + Clone + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn div(self, rhs: Buffer<T, Sh>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: Arc::new(match Arc::try_unwrap(rhs.data) {
                Ok(vec) => vec.into_par_iter().map(|y| x.clone()/y).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|y| x.clone()/y.clone()).collect(),
            }),
        }
    }
}

impl<T, T2, Sh> core::ops::Div<T2> for Buffer<T, Sh>
where
    T2: ScalarType,
    T: Clone + Sync + Send + core::ops::Div<Output = T> + ConvertFrom<T2> + ScalarType,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn div(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x / rhs.clone()).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() / rhs.clone()).collect(),
            }),
        }
    }
}

impl<T, XSh, YSh> ops::Pow<Buffer<T, YSh>> for Buffer<T, XSh>
where
    T: Sync + Send + Clone + ops::Pow<Output = T> + ScalarType,
    XSh: Shape<D = usize> + BinOpShape<YSh>,
    YSh: Shape<D = usize>,
{
    type Output = Buffer<T, <XSh as BinOpShape<YSh>>::Output>;
    fn pow(self, rhs: Buffer<T, YSh>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.pow(b))
    }
}

impl<T, Sh> ops::Pow<i32> for Buffer<T, Sh>
where
    T: Sync + Send + Clone + ops::Pow<i32> + ScalarType,
    <T as ops::Pow<i32>>::Output: Send,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<<T as ops::Pow<i32>>::Output, Sh>;
    fn pow(self, rhs: i32) -> Self::Output {
        use rayon::prelude::*;
        Buffer {
            shape: self.shape,
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x.pow(rhs)).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone().pow(rhs)).collect(),
            }),
        }
    }
}

#[cfg(not(feature = "matrixmultiply"))]
impl<T, Sh> ops::MatMul<Buffer<T, Sh>> for Buffer<T, Sh>
where
    T: Sync + Send + Clone + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + core::iter::Sum + ScalarType,
    Buffer<T, Sh>: ops::Transpose<Output = Buffer<T, Sh>>,
    Sh: Shape<D = usize>,
{
    type Output = Buffer<T, Sh>;
    fn matmul(self, rhs: Buffer<T, Sh>) -> Self::Output {
        // TODO: this is about 10x (depends on hardware) slower than it should be, because it is not cache optimized.
        // TODO: implement also expanding for buffers with correct shapes.
        let s_shape = self.shape;
        let r_shape = rhs.shape;
        // if input shape is shorter than res_shape or vice versa,
        // add necessary ones to the beginning
        /*while Sh::N < Sh::N {
            s_shape.0.insert(0, 1);
        }
        while s_shape.ndim() > r_shape.ndim() {
            r_shape.0.insert(0, 1);
        }
        let ndim = s_shape.ndim();*/
        if Sh::N < 2 {
            panic!("At least one of the buffers must have 2 or more dimensions to do matrix multiplication. Current shapes: {:?}, {:?}", s_shape, r_shape);
        }
        //if s_shape[0..Sh::N-2] != r_shape[0..Sh::N-2] ||
        if s_shape.ati(-1) != r_shape.ati(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", s_shape, r_shape);
        }
        let m = s_shape.ati(-2);
        let k = s_shape.ati(-1);
        let n = r_shape.ati(-1);
        use ops::Transpose;
        let ty = rhs.transpose();
        use rayon::prelude::*;
        let x_data = self.data.as_ref();
        let ty_data = ty.data.as_ref();
        const NUM: usize = 8; //256/core::mem::size_of::<T>(); // basically SIMD length, though it is not quite that easy due to cache
        let data: alloc::vec::Vec<T> = ty_data
                .par_chunks(k)
                .map(|y_row| {
                    x_data.par_chunks(k)
                        .map(|x| {
                            x.chunks(NUM)
                                .zip(y_row.chunks(NUM))
                                .map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| a.clone() * b.clone()).sum::<T>())
                                .sum()
                        })
                        .collect::<alloc::vec::Vec<T>>()
                })
                .flatten()
                .collect();
        
        let mut shape = s_shape;
        *shape.mut_ati(-1) = m;
        *shape.mut_ati(-2) = n;

        Buffer {
            shape,
            data: Arc::new(data),
        }.transpose()
    }
}

// Let's just use matrixmultiply crate for f32 and f64
#[cfg(feature = "matrixmultiply")]
impl<Sh> ops::MatMul for Buffer<f32, Sh>
where
    Sh: Shape<D = usize>,
{
    type Output = Self;
    fn matmul(self, rhs: Self) -> Self::Output {
        let mut s_shape = self.shape;
        let mut r_shape = rhs.shape;
        // if input shape is shorter than res_shape or vice versa,
        // add necessary ones to the beginning
        while s_shape.ndim() < r_shape.ndim() {
            s_shape.0.insert(0, 1);
        }
        while s_shape.ndim() > r_shape.ndim() {
            r_shape.0.insert(0, 1);
        }
        let ndim = s_shape.ndim();
        // TODO: support for operations on more than 2 dimensions
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != r_shape.ndim() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", s_shape, r_shape);
        }
        if s_shape[0..ndim-2] != r_shape[0..ndim-2] || s_shape[-1] != r_shape[-2] {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", s_shape, r_shape);
        }
        let m = s_shape[-2];
        let k = s_shape[-1];
        let n = r_shape[-1];
        let mut data = alloc::vec::Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            matrixmultiply::sgemm(m, k, n, 1.,
                self.data.as_ptr(), k as isize, 1,
                rhs.data.as_ptr(), n as isize, 1, 0.,
                data.as_mut_ptr(), n as isize, 1);
        }

        Buffer {
            data: Arc::new(data),
            shape: [m, n].shape(),
        }
    }
}

#[cfg(feature = "matrixmultiply")]
impl<Sh> ops::MatMul for Buffer<f64, Sh>
where
    Sh: Shape<D = usize>,
{
    type Output = Self;
    fn matmul(self, rhs: Self) -> Self::Output {
        let mut s_shape = self.shape;
        let mut r_shape = rhs.shape;
        // if input shape is shorter than res_shape or vice versa,
        // add necessary ones to the beginning
        while s_shape.ndim() < r_shape.ndim() {
            s_shape.0.insert(0, 1);
        }
        while s_shape.ndim() > r_shape.ndim() {
            r_shape.0.insert(0, 1);
        }
        let ndim = s_shape.ndim();
        // TODO: support for operations on more than 2 dimensions
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != r_shape.ndim() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", s_shape, r_shape);
        }
        if s_shape[0..ndim-2] != r_shape[0..ndim-2] || s_shape[-1] != r_shape[-2] {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", s_shape, r_shape);
        }
        let m = s_shape[-2];
        let k = s_shape[-1];
        let n = r_shape[-1];
        let mut data = alloc::vec::Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            matrixmultiply::dgemm(m, k, n, 1.,
                self.data.as_ptr(), k as isize, 1,
                rhs.data.as_ptr(), n as isize, 1, 0.,
                data.as_mut_ptr(), n as isize, 1);
        }

        Buffer {
            data: Arc::new(data),
            shape: [m, n].shape(),
        }
    }
}

impl<T, Sh, Pd> ops::Conv<Pd> for Buffer<T, Sh>
where
    T: ops::Zeros<Sh = ()> + Clone + Add<Output = T> + Mul<Output = T> + ScalarType,
    Sh: Shape<D = usize>,
    Pd: Shape<D = usize>,
{
    type Output = Buffer<T, (usize, usize)>;

    fn conv(self, kernel: Self, padding: Pd) -> Self::Output {
        // TQDO: support multidimensional convolutions
        // padding must have 2 dims, it is 2d convolution
        assert_eq!(Pd::N, 2);
        // go over resulting buffer, i iterates over result
        //let ndim = self.shape.ndim();
        if Sh::N != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        // TODO: this is not correct result shape, fix it!
        let shape = (
            {
                let s = self.shape.ati(-2);
                let k = kernel.shape.ati(-2);
                let p = padding.at(0);
                (s - k + 1usize)/p
            },
            {
                let s = self.shape.ati(-1);
                let k = kernel.shape.ati(-1);
                let p = padding.at(1);
                (s - k + 1usize)/p
            },
        );
        let mut i = 0;
        let n = shape.numel();
        let mut data = alloc::vec::Vec::with_capacity(n); // result
        let self_stride = self.shape.ati(-1);
        let kernel_stride = kernel.shape.ati(-1);
        let kernel_rows = kernel.shape.ati(-2);

        let mut self_col = 0usize;
        let mut self_row = 0;
        while i < n {
            let mut sum = T::zeros(());
            let self_offset = self_row*self_stride + self_col;
            let mut row_i = 0usize;
            // repeat for each line in kernel
            while row_i < kernel_rows {
                let data_begin = self_offset + row_i * self_stride;
                sum = self.data[data_begin..self_offset+row_i*self_stride+kernel_stride].iter().zip(kernel.data[row_i*kernel_stride..(row_i+1)*kernel_stride].iter()).fold(sum, |a, (x, y)| a + x.clone() * y.clone());
                row_i += 1;
            }
            data.push(sum);

            self_col += padding.at(1);
            if self_col >= self_stride {
                self_row += padding.at(0);
                self_col = 0;
            }

            i += 1;
        }
        
        Buffer {
            shape,
            data: Arc::new(data),
        }
    }
}
