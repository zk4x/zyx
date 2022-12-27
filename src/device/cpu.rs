//! Buffer is multidimensional storage type using cpu for the calculations
//! and rayon for multithreading. It can optionally use matrixmultiply crate.
//!

use crate::{ops::{self, ConvertFrom}, shape::{self, Shape, HasLastDim, ReducableBy, PermutableBy, MatMulBy, Axes}, dtype::DType};
use super::BufferFromSlice;
use core::marker::PhantomData;
extern crate alloc;
use alloc::{vec, sync::Arc};

/// CPU Device
/// 
/// When you use this device to create buffers, they are stored in system RAM and CPU is used for computations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Device {}

impl super::Device for Device {}

impl<'d, Sh, T> BufferFromSlice<'d, Buffer<Sh, T>> for Device 
where
    Sh: 'd + Shape,
    T: 'd + DType,
{
    fn slice(&'d mut self, slice: &[T]) -> Buffer<Sh, T> {
        Buffer {
            data: Arc::new(slice.to_vec()),
            shape: PhantomData,
        }
    }
}

/// Generic multidimensional buffer
/// 
/// Each buffer has a shape and data stored in vec.
/// Data is stored in row major order.
#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct Buffer<Sh, T = f32>
where
    Sh: Shape,
    T: DType,
{
    data: Arc<alloc::vec::Vec<T>>, // In the future this will be Arc<[T; Sh::NUMEL]>
    shape: PhantomData<Sh>,
}

impl<Sh, T> core::fmt::Display for Buffer<Sh, T>
where
    Sh: Shape + HasLastDim,
    T: DType + core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use super::NDBufferToString;
        f.write_str(&self.buffer_to_string())
    }
}

impl<T, Sh> Clone for Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: PhantomData,
        }
    }
}

impl<Sh, T> Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    /// Get Buffer's estimated memory size
    pub fn est_mem_size(&self) -> usize {
        self.data.len() * core::mem::size_of::<T>()
    }
}

// Convert between Buffers with different datatypes
impl<Sh, T, T2> ConvertFrom<Buffer<Sh, T2>> for Buffer<Sh, T>
where
    T: ConvertFrom<T2> + Send + Sync + DType,
    T2: Clone + Send + Sync + DType,
    Sh: Shape,
{
    fn cfrom(x: Buffer<Sh, T2>) -> Self {
        use rayon::prelude::*;
        use crate::ops::ConvertInto;
        Self {
            data: Arc::new(x.data.as_ref().par_iter().map(|x| x.clone().cinto()).collect()),
            shape: PhantomData,
        }
    }
}

/// Get Buffer's [Device]
impl<Sh, T> ops::HasDevice for Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    type Dev = Device;
    fn device(&self) -> Self::Dev {
        Device::default()
    }
}

/// Get Buffer's [DType]
impl<Sh, T> ops::HasDType for Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    type T = T;
}

/// Get Buffer's shape
impl<Sh, T> ops::HasShape for Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    type Sh = Sh;
}

impl<Sh, T> ops::ZerosLike for Buffer<Sh, T>
where
    Sh: Shape,
    T: DType + ops::Zero,
{
    fn zeros_like(&self) -> Self {
        Self {
            data: Arc::new(vec![T::zero(); Sh::numel()]),
            shape: PhantomData,
        }
    }
}

fn unary_op<T, Sh, F>(x: Buffer<Sh, T>, f: F) -> Buffer<Sh, T>
where
    T: Clone + Sync + Send + DType,
    F: Fn(T) -> T + Sync + Send,
    Sh: Shape,
{
    use rayon::prelude::*;
    Buffer {
        data: Arc::new(match Arc::try_unwrap(x.data) {
            Ok(vec) => vec.into_par_iter().map(f).collect(),
            Err(rc) => rc.as_ref().par_iter().map(|x| f(x.clone())).collect(),
        }),
        shape: PhantomData,
    }
}

impl<Sh, T> ops::ReLU for Buffer<Sh, T>
where
    T: Clone + Sync + Send + ops::ReLU<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn relu(self) -> Self::Output {
        unary_op(self, |x| x.relu())
    }
}

impl<Sh, T> ops::DReLU for Buffer<Sh, T>
where
    T: Clone + Sync + Send + ops::DReLU<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn drelu(self) -> Self::Output {
        unary_op(self, |x| x.drelu())
    }
}

impl<Sh, T> ops::Exp for Buffer<Sh, T>
where
    T: Clone + Sync + Send + ops::Exp<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn exp(self) -> Self::Output {
        unary_op(self, |x| x.exp())
    }
}

impl<Sh, T> ops::Ln for Buffer<Sh, T>
where
    T: Clone + Sync + Send + ops::Ln<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn ln(self) -> Self::Output {
        unary_op(self, |x| x.ln())
    }
}

impl<Sh, T> ops::Tanh for Buffer<Sh, T>
where
    T: Clone + Sync + Send + ops::Tanh<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn tanh(self) -> Self::Output {
        unary_op(self, |x| x.tanh())
    }
}

impl<Sh, T> core::ops::Neg for Buffer<Sh, T>
where
    T: Clone + Sync + Send + core::ops::Neg<Output = T> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn neg(self) -> Self::Output {
        unary_op(self, |x| x.neg())
    }
}

impl<Sh, T, Dims> ops::Summable<Dims> for Buffer<Sh, T>
where
    Sh: ReducableBy<Dims>,
    T: ops::Zero + core::ops::Add<Output = T> + DType,
    Sh: Shape,
    Dims: Axes,
{
    type Output = Buffer<<Sh as ReducableBy<Dims>>::Output, T>;

    fn _sum(self) -> Self::Output {
        //let res = vec![T::zeros(); <Sh as ReducableBy<Dims>>::Output::numel()];

        // Go over all data and apply sum function to correct values
        // then indices can be added just by making another vector and constantly updating it
        // with new indices as new max/min are found
        /*let mut i = 0; // i iterates over the result vec
        for x in self.data.as_ref() {
            if something {
                i -= stride at that dimension
            }
            i += 1;
        }*/



        // TODO: make this multithreaded
        //use alloc::borrow::ToOwned;
        //let mut data = Arc::try_unwrap(self.data).unwrap_or_else(|x| x.as_ref().to_owned());

        /*let reduce_dim = |data: &[T], dim: i32| {
            let strides = Sh::strides();
            let stride = strides[(Sh::RANK as i32 + dim) as usize % Sh::RANK];
            let mut res = vec![T::zeros(); <Sh as ReducableBy<Dims>>::Output::numel()];
            if dim == 0 || dim == -(Sh::RANK as i32) {
                for (i, x) in data.iter().enumerate() {
                    let idx = i % stride;
                    res[idx] = res[idx].clone() + x.clone();
                }
            } else {
                let width = strides[(Sh::RANK as i32 + dim - 1) as usize % Sh::RANK];
                for (i, x) in data.iter().enumerate() {
                    let idx = i/width*stride + i % stride;
                    res[idx] = res[idx].clone() + x.clone();
                }
            }
            res
        };

        for dim in Dims::array() {
            data = reduce_dim(&data, dim);
        }

        Buffer {
            data: Arc::new(data),
            shape: PhantomData,
        }*/
        todo!()
    }
}

impl<Sh, T, Dims> ops::Maximizable<Dims> for Buffer<Sh, T>
where
    Sh: ReducableBy<Dims>,
    T: ops::HasMin + PartialOrd + DType,
    Sh: Shape,
    Dims: Axes,
{
    type Values = Buffer<<Sh as ReducableBy<Dims>>::Output, T>;
    type Indices = Buffer<<Sh as ReducableBy<Dims>>::Output, T>;

    fn _max(self) -> (Self::Values, Self::Indices) {
        todo!()
        //self.reduce(T::min(), |a, b| if a > b { a } else { b })
    }
}

impl<Sh, T, Dims> ops::Minimizable<Dims> for Buffer<Sh, T>
where
    Sh: ReducableBy<Dims>,
    T: ops::HasMax + PartialOrd + DType,
    Sh: Shape,
    Dims: Axes,
{
    type Values = Buffer<<Sh as ReducableBy<Dims>>::Output, T>;
    type Indices = Buffer<<Sh as ReducableBy<Dims>>::Output, T>;

    fn _min(self) -> (Self::Values, Self::Indices) {
        todo!()
        //self.reduce(T::max(), |a, b| if a < b { a } else { b })
    }
}

impl<T, Sh, Sh2> ops::Reshapable<Sh2> for Buffer<Sh, T>
where
    T: Clone + DType,
    Sh: Shape,
    Sh2: Shape,
{
    type Output = Buffer<Sh2, T>;
    fn _reshape(self) -> Self::Output {
        assert_eq!(Sh::numel(), Sh2::numel());
        Buffer {
            data: self.data,
            shape: PhantomData,
        }
    }
}

impl <T, Sh, Sh2> ops::Expandable<Sh2> for Buffer<Sh, T>
where
    T: Clone + DType,
    Sh: Shape,
    Sh2: Shape,
{
    type Output = Buffer<Sh2, T>;
    fn _expand(self) -> Self::Output {
        assert!(Sh2::RANK >= Sh::RANK);

        let n = Sh::numel();
        use alloc::borrow::ToOwned;
        let mut data = Arc::try_unwrap(self.data).unwrap_or_else(|x| x.as_ref().to_owned());

        let copy_dim = |data: alloc::vec::Vec<T>, width, times| {
            let mut res_data = alloc::vec::Vec::with_capacity(Sh2::numel());
            for i in (0..n).step_by(width) {
                // copy this part of vec
                for _ in 0..times {
                    res_data.extend_from_slice(&data[i..i+width]);
                }
            }
            res_data
        };

        let mut i = Sh2::RANK;
        let mut width = 1;
        while i > 0 {
            i -= 1;
            let d = if Sh::RANK - i > 0 { Sh::at(i) } else { 1 };
            let r = Sh2::at(i);
            if d != r {
                if d == 1 {
                    data = copy_dim(data, width, r/d);
                }/* else {
                    //panic!("Incompatible input: {:?} and expand shape: {:?} on dim {:?}", self.shape, Sh2::array(), i);
                }*/
            }
            width *= Sh2::at(i);
        }

        Buffer {
            data: Arc::new(data),
            shape: PhantomData,
        }
    }
}

impl<T, Sh, Dims, const N: usize> ops::Permutable<Dims> for Buffer<Sh, T>
where
    T: ops::Zero + DType,
    Sh: Shape<AsArray = [usize; N]> + PermutableBy<Dims>,
    <Sh as PermutableBy<Dims>>::Output: Shape,
    Dims: Axes + 'static,
{
    type Output = Buffer<<Sh as PermutableBy<Dims>>::Output, T>;
    fn _permute(self) -> Self::Output {
        // permute function
        let permute = |array: &[usize]| {
            /*let n = array.len();
            let dims = Dims::array();
            let mut temp = array[index(dims[0])];
            for i in 0..Dims::RANK {
                core::mem::swap(&mut array[index(dims[i])], &mut temp);
            }
            array[index(dims[0])] = temp;*/
            if Dims::default()[-1] == -2 {
                let mut res = array.to_vec(); // TODO once stable rust supports it, this can be just array, not vecto
                res.swap(Sh::RANK - 1, Sh::RANK - 2);
                return res;
            } else if Dims::default()[-1] == -1 {
                return array.to_vec(); // TODO once stable rust supports it, this can be just array, not vecto
            }
            let mut res = vec![0; Sh::RANK]; // TODO once stable rust supports it, this can be just array, not vecto
            for (i, dim) in Dims::array().into_iter().enumerate() {
                res[i] = array[dim as usize];
            }
            res
        };
        // we permute strides and acc
        // acc is like strides but it is moved one dimension to the right,
        // so for example for Sh3
        // strides [   D1*D2,    D2,  1]
        // acc     [D0*D1*D2, D1*D2, D2]
        let strides = permute(&Sh::strides());
        let mut acc_var = 1;
        let mut acc = vec![0; Sh::RANK]; // TODO once stable rust supports it, this can be just array, not vector
        for i in 0..Sh::RANK {
            acc_var *= Sh::at(Sh::RANK - i - 1);
            acc[Sh::RANK - i - 1] = acc_var;
        }
        let acc = permute(&acc);
        // temp is in reverse order
        let mut temp = vec![(0, 0); Sh::RANK]; // (strides, acc) // TODO once stable rust supports it, this can be just array, not vector
        let mut begins = vec![0; Sh::RANK]; // TODO once stable rust supports it, this can be just array, not vector
        for k in 0..Sh::RANK {
            temp[Sh::RANK-k-1] = (strides[k], acc[k]);
        }
        // begins is array of indices over each of dimensions. They are slowly increased by strides until it reaches dimension size stored in acc
        // then we increase index in higher dimension and we go over lower dimension again kinda like revolution counter
        let mut data = alloc::vec::Vec::with_capacity(Sh::numel());
        let mut i = 0;
        for _ in  0..Sh::numel() {
            data.push(self.data[i].clone());
            for (j, (st, acc)) in temp.iter().enumerate() {
                begins[j] += st;
                i += st;
                if begins[j] < *acc {
                    break; // this happens more often than the other branch, hopefully branch predictor will be smart enough about it
                } else {
                    i -= begins[j];
                    begins[j] = 0;
                }
            }
        }

        Buffer {
            data: Arc::new(data),
            shape: PhantomData,
        }
    }
}

/*impl<T> ops::Slice for &Buffer<Sh, T>
where
    T: Clone + ops::Zeros + core::ops::Add<Output = T>,
{
    type Output = Buffer<Sh, T>;
    fn slice(self, dims: &[u8]) -> Self::Output {
        todo!()
    }
}*/

/// The idea behind this iterator is that you can provide custom shape and strides
#[derive(Debug, Clone, PartialEq)]
pub struct BufferIter<'a, T> {
    /// Shape can be changed
    pub shape: alloc::vec::Vec<usize>,
    /// Strides can be changed
    pub strides: alloc::vec::Vec<usize>,
    i: usize,
    k: usize,
    data: &'a alloc::vec::Vec<T>,
}

impl<'a, T, Sh> IntoIterator for &'a Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    type Item = T;
    type IntoIter = BufferIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        BufferIter {
            shape: Sh::array().into_iter().collect(),
            strides: Sh::strides().into_iter().collect(),
            i: 0,
            k: 0,
            data: &self.data,
        }
    }
}

impl<Sh, T> Buffer<Sh, T>
where
    Sh: Shape,
    T: DType,
{
    /// Create iterator over Buffer
    pub fn iter(&self) -> BufferIter<'_, T> {
        self.into_iter()
    }
}

impl<T> Iterator for BufferIter<'_, T>
where
    T: Clone,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.shape.len();
        let mut acc_var = 1;
        let mut acc = vec![1; rank];
        for i in 0..rank {
            acc_var *= self.shape[rank - i - 1];
            acc[rank - i - 1] = acc_var;
        }
        // temp is in reverse order
        let mut temp = vec![(0, 0); rank]; // strides, acc_shape
        let mut begins = vec![0; rank];
        for k in 0..rank {
            temp[rank-k-1] = (self.strides[k], acc[k]);
        }
        //let mut data = alloc::vec::Vec::with_capacity(Sh::numel());
        //for _ in  0..Sh::numel() {
        if self.k < self.shape.iter().product() {
            let res = self.data[self.i].clone();
            for (j, (st, acc)) in temp.iter().enumerate() {
                begins[j] += st;
                self.i += st;
                if begins[j] < *acc {
                    break;
                } else {
                    self.i -= begins[j];
                    begins[j] = 0;
                }
            }
            Some(res)
        } else {
            None
        }
    }
}

fn binary_op<T, F, XSh, YSh>(x: Buffer<XSh, T>, y: Buffer<YSh, T>, f: F) -> Buffer<XSh, T>
where
    T: Sync + Send + Clone + DType,
    F: Fn((T, T)) -> T + Sync + Send,
    XSh: Shape,
    YSh: Shape,
{
    use rayon::prelude::*;
    use core::cmp::Ordering;
    // TODO: fix this, so that it is not expanding, but rather using strides to not have to copy
    // stuff during expanding.
    // And it is also necessary to support constant shapes, because it is not possible to write requirements for expand,
    // since we don't need to expand both parameters

    let data = Arc::new(match XSh::numel().cmp(&YSh::numel()) {
        Ordering::Greater => {
            todo!()
            /*let y_it = y.into_iter();
            match Arc::try_unwrap(x.data) {
                Ok(vec) => match Arc::try_unwrap(y.expand().data) {
                    Ok(vec_y) => vec.into_par_iter().zip(vec_y.into_par_iter()).map(f).collect(),
                    Err(rc_y) => vec.into_par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x, y.clone()))).collect(),
                }
                Err(rc) => match Arc::try_unwrap(y.expand().data) {
                    Ok(vec_y) => rc.as_ref().par_iter().zip(vec_y.into_par_iter()).map(|(x, y)| f((x.clone(), y))).collect(),
                    Err(rc_y) => rc.as_ref().par_iter().zip(rc_y.par_iter()).map(|(x, y)| f((x.clone(), y.clone()))).collect(),
                }
            }*/
        }
        Ordering::Less => {
            todo!()
            /*match Arc::try_unwrap(y.data) {
                Ok(vec) => match Arc::try_unwrap(x.expand().data) {
                    Ok(vec_x) => vec_x.into_par_iter().zip(vec.into_par_iter()).map(f).collect(),
                    Err(rc_x) => rc_x.par_iter().zip(vec.into_par_iter()).map(|(x, y)| f((x.clone(), y))).collect(),
                },
                Err(rc) => match Arc::try_unwrap(x.expand().data) {
                    Ok(vec_x) => vec_x.into_par_iter().zip(rc.as_ref().par_iter()).map(|(x, y)| f((x, y.clone()))).collect(),
                    Err(rc_x) => rc_x.par_iter().zip(rc.as_ref().par_iter()).map(|(x, y)| f((x.clone(), y.clone()))).collect(),
                }
            }*/
        }
        Ordering::Equal => {
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
        data,
        shape: PhantomData,
    }
}

impl<T, XSh, YSh> core::ops::Add<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: Clone + Sync + Send + core::ops::Add<Output = T> + DType,
    XSh: Shape,
    YSh: Shape,
{
    type Output = Buffer<XSh, T>;

    fn add(self, rhs: Buffer<YSh, T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a + b)
    }
}

/*use duplicate::duplicate_item;
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<Sh, T> core::ops::Add<Buffer<Sh, T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Add<Output = T> + Clone + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn add(self, rhs: Buffer<Sh, T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            data: Arc::new(
                match Arc::try_unwrap(rhs.data) {
                    Ok(vec) => vec.into_par_iter().map(|y| x.clone() + y).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() + y.clone()).collect(),
                }),
            shape: PhantomData,
        }
    }
}*/

impl<Sh, T> core::ops::Add<T> for Buffer<Sh, T>
where
    T: DType + Sync + Send + core::ops::Add<Output = T>,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn add(self, rhs: T) -> Self::Output {
        use rayon::prelude::*;
        Self {
            data: Arc::new(
                match Arc::try_unwrap(self.data) {
                    Ok(vec) => vec.into_par_iter().map(|x| x + rhs.clone()).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() + rhs.clone()).collect(),
                }),
            shape: PhantomData,
        }
    }
}

impl<T, XSh, YSh> core::ops::Sub<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: Clone + Sync + Send + core::ops::Sub<Output = T> + DType,
    XSh: Shape,
    YSh: Shape,
{
    type Output = Buffer<XSh, T>;
    fn sub(self, rhs: Buffer<YSh, T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a - b)
    }
}

//#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<Sh, T> core::ops::Sub<Buffer<Sh, T>> for i32
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Sub<Output = T> + Clone + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn sub(self, rhs: Buffer<Sh, T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            data: Arc::new(
                match Arc::try_unwrap(rhs.data) {
                    Ok(vec) => vec.into_par_iter().map(|y| x.clone() - y).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() - y.clone()).collect(),
                }),
            shape: PhantomData,
        }
    }
}

/*impl<T, T2, Sh> core::ops::Sub<T2> for Buffer<Sh, T>
where
    T2: DType,
    T: Clone + Sync + Send + core::ops::Sub<Output = T> + ConvertFrom<T2> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn sub(self, rhs: T2) -> Self::Output {
        use rayon::prelude::*;
        use crate::ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            data: Arc::new(
                match Arc::try_unwrap(self.data) {
                    Ok(vec) => vec.into_par_iter().map(|x| x - rhs.clone()).collect(),
                    Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() - rhs.clone()).collect(),
                }),
            shape: PhantomData,
        }
    }
}*/

impl<T, XSh, YSh> core::ops::Mul<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: Clone + Sync + Send + core::ops::Mul<Output = T> + DType,
    XSh: Shape,
    YSh: Shape,
{
    type Output = Buffer<XSh, T>;
    fn mul(self, rhs: Buffer<YSh, T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a * b)
    }
}

/*#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<Sh, T> core::ops::Mul<Buffer<Sh, T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Mul<Output = T> + Clone + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn mul(self, rhs: Buffer<Sh, T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            data: Arc::new(match Arc::try_unwrap(rhs.data) {
                Ok(vec) => vec.into_par_iter().map(|y| x.clone() * y).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|y| x.clone() * y.clone()).collect(),
            }),
            shape: PhantomData,
        }
    }
}*/

impl<Sh, T> core::ops::Mul<f32> for Buffer<Sh, T>
where
    T: Clone + Sync + Send + core::ops::Mul<Output = T> + ops::ConvertFrom<f32> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn mul(self, rhs: f32) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let y: T = rhs.cinto();
        Self {
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x * y.clone()).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() * y.clone()).collect(),
            }),
            shape: PhantomData,
        }
    }
}

/*impl<Sh, T> core::ops::Mul<i32> for Buffer<Sh, T>
where
    T: Clone + Sync + Send + core::ops::Mul<Output = T> + ops::ConvertFrom<i32> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn mul(self, rhs: i32) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let y: T = rhs.cinto();
        Self {
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x * y.clone()).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() * y.clone()).collect(),
            }),
            shape: PhantomData,
        }
    }
}*/

impl<T, XSh, YSh> core::ops::Div<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: Clone + Sync + Send + core::ops::Div<Output = T> + DType,
    XSh: Shape,
    YSh: Shape,
{
    type Output = Buffer<XSh, T>;
    fn div(self, rhs: Buffer<YSh, T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a / b)
    }
}

/*#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<Sh, T> core::ops::Div<Buffer<Sh, T>> for dtype
where
    T: Sync + Send + ConvertFrom<Self> + core::ops::Div<Output = T> + Clone + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn div(self, rhs: Buffer<Sh, T>) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let x: T = self.cinto();
        Buffer {
            data: Arc::new(match Arc::try_unwrap(rhs.data) {
                Ok(vec) => vec.into_par_iter().map(|y| x.clone()/y).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|y| x.clone()/y.clone()).collect(),
            }),
            shape: PhantomData,
        }
    }
}*/

impl<Sh, T> core::ops::Div<i32> for Buffer<Sh, T>
where
    T: Sync + Send + core::ops::Div<Output = T> + ConvertFrom<i32> + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, T>;
    fn div(self, rhs: i32) -> Self::Output {
        use rayon::prelude::*;
        use ops::ConvertInto;
        let rhs: T = rhs.cinto();
        Self {
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x / rhs.clone()).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone() / rhs.clone()).collect(),
            }),
            shape: PhantomData,
        }
    }
}

impl<T, XSh, YSh> ops::Pow<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: Sync + Send + Clone + ops::Pow<Output = T> + DType,
    XSh: Shape,
    YSh: Shape,
{
    type Output = Buffer<XSh, T>;
    fn pow(self, rhs: Buffer<YSh, T>) -> Self::Output {
        binary_op(self, rhs, |(a, b)| a.pow(b))
    }
}

impl<Sh, T> ops::Pow<i32> for Buffer<Sh, T>
where
    T: Sync + Send + ops::Pow<i32> + DType,
    <T as ops::Pow<i32>>::Output: Send + DType,
    Sh: Shape,
{
    type Output = Buffer<Sh, <T as ops::Pow<i32>>::Output>;
    fn pow(self, rhs: i32) -> Self::Output {
        use rayon::prelude::*;
        Buffer {
            data: Arc::new(match Arc::try_unwrap(self.data) {
                Ok(vec) => vec.into_par_iter().map(|x| x.pow(rhs)).collect(),
                Err(rc) => rc.as_ref().par_iter().map(|x| x.clone().pow(rhs)).collect(),
            }),
            shape: PhantomData,
        }
    }
}

#[cfg(not(feature = "matrixmultiply"))]
impl<T, XSh, YSh> ops::MatMul<Buffer<YSh, T>> for Buffer<XSh, T>
where
    T: DType + Send + Sync + core::ops::Mul<Output = T> + core::iter::Sum + core::fmt::Debug,
    XSh: Shape + shape::MatMulBy<YSh>,
    YSh: Shape + HasLastDim,
{
    type Output = Buffer<<XSh as MatMulBy<YSh>>::Output, T>;
    fn matmul(self, rhs: Buffer<YSh, T>) -> Self::Output {
        // TODO: this is about 10x (depends on hardware) slower than it should be, because it is not cache optimized.
        // TODO: implement also expanding for buffers with correct shapes.
        /*if XSh::RANK < 2 {
            //panic!("First parameter in matrix multiplication must have at least 2 dimensions.");
        }*/
        // transpose function
        let transpose = |data: &[T], last_dim, n| {
            let mut res = alloc::vec::Vec::with_capacity(n);
            let mut j = 0;
            while j < last_dim {
                let mut i = j;
                while i < n {
                    res.push(data[i].clone());
                    i += last_dim;
                }
                j += 1;
            }
            res
        };
        let ty_data = transpose(rhs.data.as_ref(), YSh::LAST_DIM, YSh::numel());
        use rayon::prelude::*;
        let x_data = self.data.as_ref();
        const NUM: usize = 8; //256/core::mem::size_of::<T>(); // basically SIMD length, though it is not quite that easy due to cache
        let data: alloc::vec::Vec<T> = ty_data
            .par_chunks(XSh::LAST_DIM)
            .map(|y_row| {
                x_data.par_chunks(XSh::LAST_DIM)
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
        use crate::shape::HasLast2Dims;
        let data = transpose(&data, <XSh as MatMulBy<YSh>>::Output::LAST_DIM_2, <XSh as MatMulBy<YSh>>::Output::numel());
        Buffer {
            data: Arc::new(data),
            shape: PhantomData,
        }
    }
}

// Let's just use matrixmultiply crate for f32 and f64
#[cfg(feature = "matrixmultiply")]
impl<XSh, YSh> ops::MatMul<Buffer<f32, YSh>> for Buffer<f32, XSh>
where
    XSh: Shape + shape::HasLast2Dims + shape::MatMulBy<YSh>,
    YSh: Shape + HasLastDim,
{
    type Output = Buffer<f32, <XSh as MatMulBy<YSh>>::Output>;
    fn matmul(self, rhs: Buffer<f32, YSh>) -> Self::Output {
        // TODO: support for operations on more than 2 dimensions
        if XSh::RANK != 2 && YSh::RANK != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        let m = XSh::LAST_DIM_2;
        let k = XSh::LAST_DIM;
        let n = YSh::LAST_DIM;
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
            shape: PhantomData,
        }
    }
}

#[cfg(feature = "matrixmultiply")]
impl<XSh, YSh> ops::MatMul<Buffer<f64, YSh>> for Buffer<f64, XSh>
where
    XSh: Shape + shape::HasLast2Dims + shape::MatMulBy<YSh>,
    YSh: Shape + HasLastDim,
{
    type Output = Buffer<f64, <XSh as MatMulBy<YSh>>::Output>;
    fn matmul(self, rhs: Buffer<f64, YSh>) -> Self::Output {
        // TODO: support for operations on more than 2 dimensions
        if XSh::RANK != 2 && YSh::RANK != 2 {
            panic!("Only operations on buffers with 2 dimensions are supported.");
        }
        let m = XSh::LAST_DIM_2;
        let k = XSh::LAST_DIM;
        let n = YSh::LAST_DIM;
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
            shape: PhantomData,
        }
    }
}

/*impl<T, Sh, Pd> ops::Conv<Pd> for Buffer<Sh, T>
where
    T: ops::Zeros<Sh = Ax0> + Clone + Add<Output = T> + Mul<Output = T> + DType,
    Sh: Shape,
    Pd: Shape,
{
    type Output = Buffer<T, (usize, usize)>;

    fn conv(self, kernel: Self, padding: Pd) -> Self::Output {
        // TQDO: support multidimensional convolutions
        // padding must have 2 dims, it is 2d convolution
        assert_eq!(Pd::N, 2);
        // go over resulting buffer, i iterates over result
        //let ndim = self.shape.ndim();
        if Sh::N != 2 {
            //panic!("Only operations on buffers with 2 dimensions are supported.");
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
            data: Arc::new(data),
        }
    }
}*/

/// Get Buffer represented as vector.
/// It is flattened with row major order.
impl<Sh, T> ops::IntoVec<T> for Buffer<Sh, T>
where
    T: Clone + DType,
    Sh: Shape,
{
    fn to_vec(&self) -> alloc::vec::Vec<T> {
        self.data.as_ref().clone()
    }
}
