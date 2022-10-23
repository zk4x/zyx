//! Buffer is multidimensional storage type using cpu for the calculations
//! and rayon for multithreading. It can optionally use matrixmultiply crate.
//!

use crate::{ops::{self, ConvertFrom}, shape::{IntoShape, IntoDims, Shape}};
use std::ops::{Add, Mul};

// TODO: It is up to buffer to decide whether it is better to use shallow or hard copy upon cloning
// If it creates shallow copy, it needs to do the necessary reference counting
// Now Buffer can be passed by value and can implement inplace operations,
// because tensor::Variable ensures that it will not be mutated in wrong ways
// If needed, Rc can be change to Arc and RefCell to Mutex/RwLock
// Though for now everything is hard copy.
/// Generic multidimensional buffer
/// 
/// Each buffer has a shape and data stored in vec.
/// Data is stored in row major order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Buffer<T> {
    shape: Shape,
    data: Vec<T>,
}

impl<T> Buffer<T> {
    /// Get Buffer's estimated memory size
    pub fn est_mem_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>() + self.shape.ndim() * std::mem::size_of::<usize>()
    }
}

// Convert between Buffers with different datatypes
impl<T, T2> ops::ConvertFrom<Buffer<T2>> for Buffer<T>
where
    T: From<T2> + Send,
    T2: Clone + Send,
{
    fn cfrom(x: Buffer<T2>) -> Self {
        use rayon::prelude::*;
        Self {
            shape: x.shape.clone(),
            data: x.data.into_par_iter().map(|x| x.into()).collect(),
        }
    }
}

// Display Buffer
impl<T> std::fmt::Display for Buffer<T>
where
    T: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut res = String::new();
        if self.data.is_empty() { return f.write_str(&(res + "[]")); }
        let n = self.shape.numel();
        let ndim = self.shape.ndim();
        //const PRECISION: usize = 3;
        // get maximal width of single value
        let mut w = 0;
        for x in &self.data {
            let l = format!("{0:1$}", x, w).len();
            if l > w { w = l; }
        }
        let d0 = self.shape[-1];
        for i in 0..n {
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if i % (n/var) == 0 {
                        res += &(" ".repeat(ndim - r)+&"[".repeat(r - 1));
                        break
                    }
                    var *= self.shape[ndim - r];
                    r -= 1;
                }
            }
            use std::fmt::Write;
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
                    var *= self.shape[ndim - r];
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
impl<T> ops::IntoVec<T> for Buffer<T>
where
    T: Clone,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
}

/// Create new Buffer from vec
impl<T> ops::FromVec<T> for Buffer<T> {
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        assert_eq!(shape.numel(), data.len());
        Self {
            shape,
            data,
        }
    }
}

/// Get Buffer's shape
impl<T> ops::GetShape for Buffer<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

/// Create new Buffer filled with zeros
impl<T> ops::Zeros for Buffer<T>
where
    T: Clone + ops::Zeros,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        let n = shape.numel();
        Self {
            shape,
            data: vec![T::zeros(()); n],
        }
    }
}

/// Create new Buffer filled with ones
impl<T> ops::Ones for Buffer<T>
where
    T: Clone + ops::Ones,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        let n = shape.numel();
        Self {
            shape,
            data: vec![T::ones(()); n],
        }
    }
}

fn unary_op<T, F>(x: Buffer<T>, f: F) -> Buffer<T>
where
    T: Sync + Send,
    F: Fn(T) -> T + Sync + Send,
{
    use rayon::prelude::*;
    Buffer {
        shape: x.shape.clone(),
        data: x.data.into_par_iter().map(f).collect(),
    }
}

impl<T> ops::ReLU for Buffer<T>
where
    T: Sync + Send + ops::ReLU<Output = T>,
{
    type Output = Buffer<T>;
    fn relu(self) -> Self::Output {
        unary_op(self, |x| x.relu())
    }
}

impl<T> ops::DReLU for Buffer<T>
where
    T: Sync + Send + ops::DReLU<Output = T>,
{
    type Output = Buffer<T>;
    fn drelu(self) -> Self::Output {
        unary_op(self, |x| x.drelu())
    }
}

impl<T> ops::Exp for Buffer<T>
where
    T: Sync + Send + ops::Exp<Output = T>,
{
    type Output = Buffer<T>;
    fn exp(self) -> Self::Output {
        unary_op(self, |x| x.exp())
    }
}

impl<T> ops::Ln for Buffer<T>
where
    T: Sync + Send + ops::Ln<Output = T>,
{
    type Output = Buffer<T>;
    fn ln(self) -> Self::Output {
        unary_op(self, |x| x.ln())
    }
}

impl<T> ops::Tanh for Buffer<T>
where
    T: Sync + Send + ops::Tanh<Output = T>,
{
    type Output = Buffer<T>;
    fn tanh(self) -> Self::Output {
        unary_op(self, |x| x.tanh())
    }
}

impl<T> std::ops::Neg for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Neg<Output = T>,
{
    type Output = Buffer<T>;
    fn neg(self) -> Self::Output {
        unary_op(self, |x| x.neg())
    }
}

impl<T> Buffer<T> {
    fn reduce<F>(self, dims: impl IntoDims, init: T, mut f: F) -> Buffer<T>
    where
        T: Clone,
        F: FnMut(T, T) -> T,
    {
        // TODO: make this multithreaded
        let mut data = self.data.clone();
        let mut shape = self.shape.clone();
        let dims = dims.dims();
        let ndim = shape.ndim();

        let mut reduce_dim = |data: &[T], dim: i32| {
            let strides = shape.strides();
            let stride = strides[dim];
            shape[dim] = 1;
            let mut res = vec![init.clone(); shape.numel()];
            if dim == 0 || dim == -(ndim as i32) {
                for (i, x) in data.iter().enumerate() {
                    let idx = i % stride;
                    res[idx] = f(res[idx].clone(), x.clone());
                }
            } else {
                let width = strides[dim - 1];
                for (i, x) in data.iter().enumerate() {
                    let idx = i/width*stride + i % stride;
                    res[idx] = f(res[idx].clone(), x.clone());
                }
            }
            res
        };

        if dims.is_empty() {
            for dim in 0..self.shape.ndim() {
                data = reduce_dim(&data, dim as i32);
            }
        } else {
            for dim in dims {
                data = reduce_dim(&data, dim);
            }
        }

        Buffer {
            shape,
            data,
        }
    }
}

impl<T> ops::Sum for Buffer<T>
where
    T: Clone + ops::Zeros + std::ops::Add<Output = T>,
{
    type Output = Buffer<T>;
    fn sum(self, dims: impl IntoDims) -> Self::Output {
        self.reduce(dims, T::zeros(()), |a, b| a + b)
    }
}

impl<T> ops::Max for Buffer<T>
where
    T: Clone + Default + ops::Min<Output = T> + PartialOrd,
{
    type Output = Buffer<T>;
    fn max(self, dims: impl IntoDims) -> Self::Output {
        self.reduce(dims, T::min(T::default(), ()), |a, b| if a > b { a } else { b })
    }
}

impl<T> ops::Min for Buffer<T>
where
    T: Clone + Default + ops::Max<Output = T> + PartialOrd,
{
    type Output = Buffer<T>;
    fn min(self, dims: impl IntoDims) -> Self::Output {
        self.reduce(dims, T::max(T::default(), ()), |a, b| if a < b { a } else { b })
    }
}

impl<T> ops::Reshape for Buffer<T>
where
    T: Clone,
{
    type Output = Buffer<T>;
    fn reshape(self, shape: impl IntoShape) -> Self::Output {
        let shape = shape.shape();
        assert_eq!(self.shape.numel(), shape.numel());
        Buffer {
            shape,
            data: self.data,
        }
    }
}

impl<T> ops::Expand for Buffer<T>
where
    T: Clone,
{
    type Output = Buffer<T>;
    fn expand(self, res_shape: impl IntoShape) -> Self::Output {
        let mut shape = self.shape.clone();
        let mut res_shape = res_shape.shape();

        // if input shape is shorter than res_shape or vice versa,
        // add necessary ones to the beginning
        while shape.ndim() < res_shape.ndim() {
            shape.0.insert(0, 1);
        }
        while shape.ndim() > res_shape.ndim() {
            res_shape.0.insert(0, 1);
        }
        let n = shape.numel();
        let ndims = shape.ndim();
        let mut data = self.data;

        let copy_dim = |data: Vec<T>, width, times| {
            let mut res_data = Vec::with_capacity(res_shape.numel());
            for i in (0..n).step_by(width) {
                // copy this part of vec
                for _ in 0..times {
                    res_data.extend_from_slice(&data[i..i+width]);
                }
            }
            res_data
        };

        let mut i = ndims;
        let mut width = 1;
        for (d, r) in shape.clone().into_iter().zip(res_shape.clone().into_iter()).rev() {
            i -= 1;
            if d != r {
                if d == 1 {
                    data = copy_dim(data, width, r/d);
                } else {
                    panic!("Incompatible input: {:?} and expand shape: {:?} on dim {:?}",
                        shape, res_shape, i);
                }
            }
            width *= res_shape[i];
        }

        Buffer {
            shape: res_shape,
            data,
        }
    }
}

impl<T> ops::Permute for Buffer<T>
where
    T: Clone + ops::Zeros,
{
    type Output = Buffer<T>;
    fn permute(self, dims: impl IntoDims) -> Self::Output {
        // if dims.len() is not same as shape.len(), correct it
        let mut dims = dims.dims().0;
        let mut s_shape = self.shape;
        match dims.len().cmp(&s_shape.ndim()) {
            std::cmp::Ordering::Greater =>
                for _ in 0..dims.len() - s_shape.ndim() {
                    s_shape.0.insert(0, 1);
                },
            std::cmp::Ordering::Less =>
                for i in 0..s_shape.ndim() - dims.len() {
                    dims.insert(0, i as i32);
                },
            std::cmp::Ordering::Equal => {}
        }

        let ndim = s_shape.ndim();
        let shape = s_shape.clone().permute(dims.clone());
        let strides = s_shape.strides().permute(dims.clone());
        let mut acc_var = 1;
        let acc = Shape(s_shape.into_iter().rev().map(|x| { acc_var *= x; acc_var }).collect::<Vec<usize>>().into_iter().rev().collect()).permute(dims);
        let n = shape.numel();
        // temp is in reverse order
        let mut temp = vec![(0, 0); ndim]; // strides, acc_shape
        let mut begins = vec![0; ndim];
        for k in 0..ndim {
            temp[ndim-k-1] = (strides[k], acc[k]);
        }

        let mut data = Vec::with_capacity(n);
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
            data,
        }
    }
}

/*impl<T> ops::Slice for &Buffer<T>
where
    T: Clone + ops::Zeros + std::ops::Add<Output = T>,
{
    type Output = Buffer<T>;
    fn slice(self, dims: &[u8]) -> Self::Output {
        todo!()
    }
}*/

fn binary_op<T, F>(x: Buffer<T>, y: Buffer<T>, f: F) -> Buffer<T>
where
    T: Sync + Send + Clone,
    F: Fn((T, T)) -> T + Sync + Send,
{
    use rayon::prelude::*;
    use ops::Expand;
    use std::cmp::Ordering;
    let shape = x.shape.clone();
    // TODO: fix this, so that it is not expanding, but rather using strides to not have to copy
    // stuff during expanding
    let data = match x.shape.numel().cmp(&y.shape.numel()) {
        Ordering::Greater => x.data.into_par_iter().zip(y.expand(x.shape).data.into_par_iter()).map(f).collect(),
        Ordering::Less => x.expand(y.shape).data.into_par_iter().zip(y.data.into_par_iter()).map(f).collect(),
        Ordering::Equal => x.data.into_par_iter().zip(y.data.into_par_iter()).map(f).collect(),
    };
    Buffer {
        shape,
        data,
    }
}

impl<T> std::ops::Add for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Add<Output = T>,
{
    type Output = Buffer<T>;
    fn add(self, rhs: Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a + b)
    }
}

use duplicate::duplicate_item;
#[duplicate_item( dtype; [f32]; [f64]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T> std::ops::Add<Buffer<T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + std::ops::Add<Output = T> + Clone,
{
    type Output = Buffer<T>;
    fn add(self, rhs: Buffer<T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: rhs.data.into_par_iter().map(|y| x.clone() + y).collect(),
        }
    }
}

impl<T, T2> std::ops::Add<T2> for Buffer<T>
where
    T2: crate::dtype::ScalarType,
    T: Clone + Sync + Send + std::ops::Add<Output = T> + From<T2>,
{
    type Output = Buffer<T>;
    fn add(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        let rhs: T = rhs.into();
        Self {
            shape: self.shape,
            data: self.data.into_par_iter().map(|x| x + rhs.clone()).collect()
        }
    }
}

impl<T> std::ops::Sub for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Sub<Output = T>,
{
    type Output = Buffer<T>;
    fn sub(self, rhs: Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a - b)
    }
}

impl<T> std::ops::Mul for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Mul<Output = T>,
{
    type Output = Buffer<T>;
    fn mul(self, rhs: Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a * b)
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T> std::ops::Mul<Buffer<T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + std::ops::Mul<Output = T> + Clone,
{
    type Output = Buffer<T>;
    fn mul(self, rhs: Buffer<T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: rhs.data.into_par_iter().map(|y| x.clone()*y).collect(),
        }
    }
}

impl<T> std::ops::Mul<f64> for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Mul<Output = T> + ops::ConvertFrom<f64>,
{
    type Output = Buffer<T>;
    fn mul(self, rhs: f64) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: self.data.into_par_iter().map(|x| x * rhs.clone()).collect(),
        }
    }
}

impl<T> std::ops::Div for Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Div<Output = T>,
{
    type Output = Buffer<T>;
    fn div(self, rhs: Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a / b)
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<T> std::ops::Div<Buffer<T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + std::ops::Div<Output = T> + Clone,
{
    type Output = Buffer<T>;
    fn div(self, rhs: Buffer<T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            shape: rhs.shape,
            data: rhs.data.into_par_iter().map(|y| x.clone()/y).collect(),
        }
    }
}

use crate::dtype::ScalarType;
impl<T, T2> std::ops::Div<T2> for Buffer<T>
where
    T2: ScalarType,
    T: Clone + Sync + Send + std::ops::Div<Output = T> + ops::ConvertFrom<T2>,
{
    type Output = Buffer<T>;
    fn div(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            shape: self.shape,
            data: self.data.into_par_iter().map(|x| x / rhs.clone()).collect(),
        }
    }
}

impl<T> ops::Pow for Buffer<T>
where
    T: Sync + Send + Clone + ops::Pow<Output = T>,
{
    type Output = Buffer<T>;
    fn pow(self, rhs: Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.pow(b))
    }
}

impl<T> ops::Pow<i32> for Buffer<T>
where
    T: Sync + Send + Clone + ops::Pow<i32>,
{
    type Output = Buffer<<T as ops::Pow<i32>>::Output>;
    fn pow(self, rhs: i32) -> Self::Output {
        Buffer {
            shape: self.shape,
            data: self.data.into_iter().map(|x| x.pow(rhs)).collect(),
        }
    }
}

#[cfg(not(feature = "matrixmultiply"))]
impl<T> ops::MatMul for Buffer<T>
where
    T: Sync + Send + Clone + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::iter::Sum,
    Buffer<T>: ops::Transpose<Output = Buffer<T>>,
{
    type Output = Self;
    fn matmul(self, rhs: Self) -> Self::Output {
        // TODO: this is about 10x (depends on hardware) slower than it should be, because it is not cache optimized.
        // TODO: implement also expanding for buffers with correct shapes.
        let mut s_shape = self.shape;
        let mut r_shape = rhs.shape.clone();
        // if input shape is shorter than res_shape or vice versa,
        // add necessary ones to the beginning
        while s_shape.ndim() < r_shape.ndim() {
            s_shape.0.insert(0, 1);
        }
        while s_shape.ndim() > r_shape.ndim() {
            r_shape.0.insert(0, 1);
        }
        let ndim = s_shape.ndim();
        if ndim < 2 {
            panic!("You need at least one of the buffers to have 2 or more dimensions to do matrix multiplication. Current shapes: {}, {}", s_shape, r_shape);
        }
        if s_shape[0..ndim-2] != r_shape[0..ndim-2] || s_shape[-1] != r_shape[-2] {
            panic!("Incorrect x and y shapes for matmul: {}, {}", s_shape, r_shape);
        }
        let m = s_shape[-2];
        let k = s_shape[-1];
        let n = r_shape[-1];
        use ops::Transpose;
        let ty = rhs.transpose();
        use rayon::prelude::*;
        const NUM: usize = 8; // /std::mem::size_of::<T>(); // basically SIMD length
        let data: Vec<T> = ty.data
                .par_chunks(k)
                .map(|y_row| {
                    self.data.chunks(k)
                        .map(|x| {
                            x.chunks(NUM)
                                .zip(y_row.chunks(NUM))
                                .map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| a.clone() * b.clone()).sum::<T>())
                                .sum()
                        })
                        .collect::<Vec<T>>()
                })
                .flatten()
                .collect();
        
        let mut shape = s_shape;
        shape[ndim-1] = m;
        shape[ndim-2] = n;

        Buffer {
            shape,
            data,
        }.transpose()
    }
}

// Let's just use matrixmultiply crate for f32 and f64
#[cfg(feature = "matrixmultiply")]
impl ops::MatMul for Buffer<f32> {
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
        let mut data = Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            matrixmultiply::sgemm(m, k, n, 1.,
                self.data.as_ptr(), k as isize, 1,
                rhs.data.as_ptr(), n as isize, 1, 0.,
                data.as_mut_ptr(), n as isize, 1);
        }

        Buffer {
            data,
            shape: [m, n].shape(),
        }
    }
}

#[cfg(feature = "matrixmultiply")]
impl ops::MatMul for Buffer<f64> {
    type Output = Self;
    fn matmul(self, rhs: Self) -> Self::Output {
        let ndim = self.shape.ndim();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != rhs.shape.ndim() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape[-1] != rhs.shape[-2] {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        let m = self.shape[-2];
        let k = self.shape[-1];
        let n = rhs.shape[-1];
        let mut data = Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            matrixmultiply::dgemm(m, k, n, 1.,
                self.data.as_ptr(), k as isize, 1,
                rhs.data.as_ptr(), n as isize, 1, 0.,
                data.as_mut_ptr(), n as isize, 1);
        }

        Buffer {
            data,
            shape: [m, n].shape(),
        }
    }
}

impl<T> ops::Conv for Buffer<T>
where
    T: ops::Zeros + Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;
    fn conv(self, kernel: Self, padding: impl IntoShape) -> Self::Output {
        // TQDO: support multidimensional convolutions
        // padding must have 2 dims, it is 2d convolution
        let padding = padding.shape();
        assert_eq!(padding.ndim(), 2);
        // go over resulting buffer, i iterates over result
        let ndim = self.shape.ndim();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        // TODO: this is not correct result shape, fix it!
        let shape = [
            {
                let s = self.shape[-2];
                let k = kernel.shape[-2];
                let p = padding[0];
                (s - k + 1usize)/p
            },
            {
                let s = self.shape[-1];
                let k = kernel.shape[-1];
                let p = padding[1];
                (s - k + 1usize)/p
            },
        ].shape();
        let mut i = 0;
        let n = shape.numel();
        let mut data = Vec::with_capacity(n); // result
        let self_stride = self.shape[-1];
        let kernel_stride = kernel.shape[-1];
        let kernel_rows = kernel.shape[-2];

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

            self_col += padding[1];
            if self_col >= self_stride {
                self_row += padding[0];
                self_col = 0;
            }

            i += 1;
        }
        
        Self {
            shape,
            data,
        }
    }
}
