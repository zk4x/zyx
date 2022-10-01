//! Buffer is multidimensional storage type using cpu for the calculations
//! and rayon for multithreading.
//!

use crate::{ops, shape::{Shape, Dims}};

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Buffer<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> Buffer<T> {
    pub fn est_mem_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }
}

impl<T, T2> ops::ConvertFrom<&Buffer<T2>> for Buffer<T>
where
    T: From<T2>,
    T2: Clone,
{
    fn convert_from(x: &Buffer<T2>) -> Self {
        Self {
            shape: x.shape.clone(),
            data: x.data.iter().map(|x| x.clone().into()).collect(),
        }
    }
}

impl<T> std::fmt::Display for Buffer<T>
where
    T: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut res = String::new();
        if self.data.is_empty() { return f.write_str(&(res + "[]")); }
        let n = self.shape.numel();
        let ndim = self.shape.len();
        const PRECISION: usize = 3;
        // get maximal width of single value
        let mut w = 0;
        for x in &self.data {
            let l = format!("{0:1$.2$}", x, w, PRECISION).len();
            if l > w { w = l; }
        }
        let d0 = self.shape.index(-1);
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
            let _ = write!(res, "{0:>1$.2$}", self.data[i], w, PRECISION);
            if (i + 1) % d0 != 0 { res += " "; }
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
            if (i + 1) % d0 == 0 && i != n - 1 { res += "\n"; }
        }
        f.write_str(&res)
    }
}

impl<T> ops::ToVec<T> for Buffer<T>
where
    T: Clone,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T> ops::FromVec<T> for Buffer<T> {
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        debug_assert_eq!(shape.numel(), data.len());
        Self {
            shape: shape.to_vec(),
            data,
        }
    }
}

impl<T> ops::GetShape for &Buffer<T> {
    fn shape(self) -> Vec<usize> {
        self.shape.clone()
    }
}

impl<T> ops::Zeros for Buffer<T>
where
    T: Clone + ops::Zeros,
{
    fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![T::zeros(&[]); shape.numel()],
        }
    }
}

impl<T> ops::Ones for Buffer<T>
where
    T: Clone + ops::Ones,
{
    fn ones(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![T::ones(&[]); shape.numel()],
        }
    }
}

fn unary_op<T, F>(x: &Buffer<T>, f: F) -> Buffer<T>
where
    T: Sync + Send,
    F: Fn(&T) -> T + Sync + Send,
{
    use rayon::prelude::*;
    Buffer {
        shape: x.shape.clone(),
        data: x.data.par_iter().map(f).collect(),
    }
}

impl<T> ops::ReLU for &Buffer<T>
where
    T: Sync + Send,
    for<'a> &'a T: ops::ReLU<Output = T>,
{
    type Output = Buffer<T>;
    fn relu(self) -> Self::Output {
        unary_op(self, |x| x.relu())
    }
}

impl<T> ops::DReLU for &Buffer<T>
where
    T: Sync + Send,
    for<'a> &'a T: ops::DReLU<Output = T>,
{
    type Output = Buffer<T>;
    fn drelu(self) -> Self::Output {
        unary_op(self, |x| x.drelu())
    }
}

impl<T> ops::Exp for &Buffer<T>
where
    T: Sync + Send,
    for<'a> &'a T: ops::Exp<Output = T>,
{
    type Output = Buffer<T>;
    fn exp(self) -> Self::Output {
        unary_op(self, |x| x.exp())
    }
}

impl<T> ops::Ln for &Buffer<T>
where
    T: Sync + Send,
    for<'a> &'a T: ops::Ln<Output = T>,
{
    type Output = Buffer<T>;
    fn ln(self) -> Self::Output {
        unary_op(self, |x| x.ln())
    }
}

impl<T> ops::Tanh for &Buffer<T>
where
    T: Sync + Send,
    for<'a> &'a T: ops::Tanh<Output = T>,
{
    type Output = Buffer<T>;
    fn tanh(self) -> Self::Output {
        unary_op(self, |x| x.tanh())
    }
}

impl<T> std::ops::Neg for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Neg<Output = T>,
{
    type Output = Buffer<T>;
    fn neg(self) -> Self::Output {
        unary_op(self, |x| x.clone().neg())
    }
}

impl<T> Buffer<T> {
    fn reduce<F>(&self, dims: &[i32], init: T, mut f: F) -> Buffer<T>
    where
        T: Clone,
        F: FnMut(T, T) -> T,
    {
        // TODO: make this multithreaded
        let mut data = self.data.clone();
        let mut shape = self.shape.clone();
        let ndims = shape.len();

        let mut reduce_dim = |data: &[T], dim| {
            let strides = shape.strides();
            let stride = strides.index(dim);
            shape[(ndims as i32 + dim) as usize % ndims] = 1;
            let mut res = vec![init.clone(); shape.numel()];
            if dim == 0 || dim == -(ndims as i32) {
                for (i, x) in data.iter().enumerate() {
                    let idx = i % stride;
                    res[idx] = f(res[idx].clone(), x.clone());
                }
            } else {
                let width = strides.index(dim - 1);
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
                data = reduce_dim(&data, *dim);
            }
        }

        Buffer {
            shape,
            data,
        }
    }
}

impl<T> ops::Sum for &Buffer<T>
where
    T: Clone + ops::Zeros + std::ops::Add<Output = T>,
{
    type Output = Buffer<T>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        self.reduce(dims, T::zeros(&[]), |a, b| a + b)
    }
}

impl<T> ops::Max for &Buffer<T>
where
    T: Clone + Default + ops::Min<Output = T> + PartialOrd,
{
    type Output = Buffer<T>;
    fn max(self, dims: &[i32]) -> Self::Output {
        self.reduce(dims, T::min(T::default(), &[]), |a, b| if a > b { a } else { b })
    }
}

impl<T> ops::Min for &Buffer<T>
where
    T: Clone + Default + ops::Max<Output = T> + PartialOrd,
{
    type Output = Buffer<T>;
    fn min(self, dims: &[i32]) -> Self::Output {
        self.reduce(dims, T::max(T::default(), &[]), |a, b| if a < b { a } else { b })
    }
}

impl<T> ops::Reshape for &Buffer<T>
where
    T: Clone,
{
    type Output = Buffer<T>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        debug_assert_eq!(self.shape.ndim(), shape.ndim());
        debug_assert_eq!(self.shape.numel(), shape.numel());
        Buffer {
            shape: shape.to_vec(),
            data: self.data.clone(),
        }
    }
}

impl<T> ops::Expand for &Buffer<T>
where
    T: Clone,
{
    type Output = Buffer<T>;
    fn expand(self, res_shape: &[usize]) -> Self::Output {
        let shape = self.shape.clone();
        let res_shape = res_shape.dims();
        let n = shape.numel();
        let ndims = shape.len();
        let mut data = self.data.clone();

        let copy_dim = |data: Vec<T>, width, times| {
            let mut res_data = Vec::with_capacity(res_shape.numel());
            for i in (0..n).step_by(width) {
                // copy this part of vec
                for _ in 0..times {
                    for x in &data[i..i+width] {
                        res_data.push(x.clone());
                    }
                }
            }
            res_data
        };

        let mut i = ndims;
        let mut width = 1;
        for (d, r) in shape.iter().zip(res_shape.iter()).rev() {
            i -= 1;
            if *d != *r {
                if *d == 1 {
                    data = copy_dim(data, width, r/d);
                } else {
                    panic!("Incompatible input: {:?} and expand shape: {:?} on dim {:?}",
                        shape, res_shape, i);
                }
            }
            width *= res_shape[i];
        }

        Buffer {
            shape: res_shape.to_vec(),
            data,
        }
    }
}

impl<T> ops::Permute for &Buffer<T>
where
    T: Clone + ops::Zeros,
{
    type Output = Buffer<T>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        // if dims.len() is not same as shape.len(), correct it
        let mut dims = dims.to_vec();
        match dims.ndim().cmp(&self.shape.ndim()) {
            std::cmp::Ordering::Greater =>
                panic!("Input has too many dimensions"),
            std::cmp::Ordering::Less =>
                for i in 0..self.shape.ndim() - dims.ndim() {
                    dims.insert(0, i as i32);
                },
            std::cmp::Ordering::Equal => {}
        }
        let n = self.shape.numel();
        let ndims = self.shape.len();
        let mut data = vec![T::zeros(&[]); n];
        let shape = self.shape.permute(&dims);
        let strides = shape.strides();

        // calculate steps for each dimension
        let mut steps = vec![0; ndims];
        for (i, dim) in dims.iter().enumerate() {
            steps[(ndims as i32 + dim) as usize % ndims] = strides[i];
        }
        //let steps = strides.permute(dims.dims());

        let mut dim_prod = 1;
        let indexes_steps: Vec<usize> = self.shape.iter().map(|dim| { dim_prod *= dim; n/dim_prod }).collect();

        for (i, x) in self.data.iter().enumerate() {
            data[steps.iter().zip(indexes_steps.iter()).zip(self.shape.iter()).map(|((step, idx), dim)| i/idx%dim*step).sum::<usize>()] = x.clone();
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

fn binary_op<T, F>(x: &Buffer<T>, y: &Buffer<T>, f: F) -> Buffer<T>
where
    T: Sync + Send + Clone,
    F: Fn((&T, &T)) -> T + Sync + Send,
{
    use rayon::prelude::*;
    use ops::Expand;
    use std::cmp::Ordering;
    let data = match x.shape.numel().cmp(&y.shape.numel()) {
        Ordering::Greater => x.data.par_iter().zip(y.expand(&x.shape).data.par_iter()).map(f).collect(),
        Ordering::Less => x.expand(&y.shape).data.par_iter().zip(y.data.par_iter()).map(f).collect(),
        Ordering::Equal => x.data.par_iter().zip(y.data.par_iter()).map(f).collect(),
    };
    Buffer {
        shape: x.shape.clone(),
        data,
    }
}

impl<T> std::ops::Add for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Add<Output = T>,
{
    type Output = Buffer<T>;
    fn add(self, rhs: &Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.clone() + b.clone())
    }
}

impl<T> std::ops::Sub for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Sub<Output = T>,
{
    type Output = Buffer<T>;
    fn sub(self, rhs: &Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.clone() - b.clone())
    }
}

impl<T> std::ops::Mul for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Mul<Output = T>,
{
    type Output = Buffer<T>;
    fn mul(self, rhs: &Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.clone() * b.clone())
    }
}

impl<T> std::ops::Mul<f32> for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Mul<f32, Output = T>,
{
    type Output = Buffer<T>;
    fn mul(self, rhs: f32) -> Self::Output {
        use rayon::prelude::*;
        Buffer {
            shape: self.shape.clone(),
            data: self.data.par_iter().map(|x| x.clone() * rhs).collect(),
        }
    }
}

impl<T> std::ops::Div for &Buffer<T>
where
    T: Clone + Sync + Send + std::ops::Div<Output = T>,
{
    type Output = Buffer<T>;
    fn div(self, rhs: &Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.clone() / b.clone())
    }
}

impl<T> ops::Pow for &Buffer<T>
where
    T: Clone + Sync + Send + ops::Pow<Output = T>,
{
    type Output = Buffer<T>;
    fn pow(self, rhs: &Buffer<T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.clone().pow(b.clone()))
    }
}

#[cfg(not(any(feature = "matrixmultiply", feature = "cblas")))]
impl<T> ops::MatMul for &Buffer<T>
where
    T: Sync + Send + Clone + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::iter::Sum,
    for<'a> &'a Buffer<T>: ops::Transpose<Output = Buffer<T>>,
{
    type Output = Buffer<T>;
    fn matmul(self, rhs: &Buffer<T>) -> Self::Output {
        // TODO: this is about 10x (depends on hardware) slower than it should be, because it is not cache optimized.
        // TODO: implement also expanding for buffers with correct shapes.
        let ndim = self.shape.len();
        if ndim != rhs.shape.len() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape.index(-1) != rhs.shape.index(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        use ops::Transpose;
        let ty = rhs.transpose();
        use rayon::prelude::*;
        const NUM: usize = 16; // basically SIMD length
        let k = rhs.shape.index(-2);
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
        
        let mut shape = self.shape.clone();
        shape[ndim-1] = self.shape.index(-2);
        shape[ndim-2] = rhs.shape.index(-1);

        Buffer {
            shape,
            data,
        }.transpose()
    }
}

// Let's just use matrixmultiply crate for f32 and f64
#[cfg(feature = "matrixmultiply")]
impl ops::MatMul for &Buffer<f32> {
    type Output = Buffer<f32>;
    fn matmul(self, rhs: &Buffer<f32>) -> Self::Output {
        let ndim = self.shape.len();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != rhs.shape.len() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape.index(-1) != rhs.shape.index(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        let m = self.shape.index(-2);
        let k = self.shape.index(-1);
        let n = rhs.shape.index(-1);
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
            shape: vec![m, n],
        }
    }
}

#[cfg(feature = "matrixmultiply")]
impl ops::MatMul for &Buffer<f64> {
    type Output = Buffer<f64>;
    fn matmul(self, rhs: &Buffer<f64>) -> Self::Output {
        let ndim = self.shape.len();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != rhs.shape.len() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape.index(-1) != rhs.shape.index(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        let m = self.shape.index(-2);
        let k = self.shape.index(-1);
        let n = rhs.shape.index(-1);
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
            shape: vec![m, n],
        }
    }
}

#[cfg(feature = "cblas")]
impl ops::MatMul for &Buffer<f32> {
    type Output = Buffer<f32>;
    fn matmul(self, rhs: &Buffer<f32>) -> Self::Output {
        let ndim = self.shape.len();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != rhs.shape.len() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape.index(-1) != rhs.shape.index(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        let m = self.shape.index(-2);
        let k = self.shape.index(-1);
        let n = rhs.shape.index(-1);
        let mut data = Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            cblas_sys::cblas_sgemm(
                cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32, n as i32, k as i32, 1.,
                self.data.as_ptr(), 1,
                rhs.data.as_ptr(), 1, 1.,
                data.as_mut_ptr(), 1
            );
        }

        Buffer {
            data,
            shape: vec![m, n],
        }
    }
}

#[cfg(feature = "cblas")]
impl ops::MatMul for &Buffer<f64> {
    type Output = Buffer<f64>;
    fn matmul(self, rhs: &Buffer<f64>) -> Self::Output {
        let ndim = self.shape.len();
        if ndim != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        if ndim != rhs.shape.len() {
            panic!("Matmul buffers have different degrees: {:?}, {:?}", self.shape, rhs.shape);
        }
        if self.shape[0..ndim-2] != rhs.shape[0..ndim-2] || self.shape.index(-1) != rhs.shape.index(-2) {
            panic!("Incorrect x and y shapes for matmul: {:?}, {:?}", self.shape, rhs.shape);
        }
        let m = self.shape.index(-2);
        let k = self.shape.index(-1);
        let n = rhs.shape.index(-1);
        let mut data = Vec::with_capacity(m*n);
        unsafe {
            data.set_len(m*n);
            cblas_sys::cblas_dgemm(
                cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32, n as i32, k as i32, 1.,
                self.data.as_ptr(), 1,
                rhs.data.as_ptr(), 1, 1.,
                data.as_mut_ptr(), 1
            );
        }

        Buffer {
            data,
            shape: vec![m, n],
        }
    }
}

// TODO: conv2d
