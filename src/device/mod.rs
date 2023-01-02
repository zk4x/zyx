pub mod cpu;
mod dtype;
pub mod opencl;

pub use dtype::{DType, SType};

extern crate alloc;

use alloc::{vec::Vec, sync::Arc};

use crate::shape::{Shape, Sh2};

// This is the new way we are going to do this
pub trait Device {
    type Buffer<T: DType>: Clone;

    fn slice<T: DType>(&self, slice: &[T]) -> Self::Buffer<T>;
}

// Self here is actual buffer, actual device storage
// This is the only operation that the accelerator needs to support :)
trait DeviceOp<T> {
    type D: Device;
    fn binary_op(
        &self,
        x_shape: Vec<usize>,
        x_strides: Vec<usize>,
        x_offset: usize,
        x_unary_ops: Vec<UnaryOp>,
        x_reduce_dims: Vec<Vec<usize>>,
        rhs: &Self,
        y_shape: Vec<usize>,
        y_strides: Vec<usize>,
        y_offset: usize,
        y_unary_ops: Vec<UnaryOp>,
        y_reduce_dims: Vec<Vec<usize>>,
        op_type: SyncOp,
        device: &Self::D,
    ) -> Self;
}

trait BufferInit<'d, T>: Device + Sized
where
    T: DType,
{
    fn zeros<S: Shape>() -> Buffer<'d, S, T, Self> {
        todo!()
    }

    fn ones<S: Shape>() -> Buffer<'d, S, T, Self> {
        todo!()
    }

    fn eye<const N: usize>() -> Buffer<'d, Sh2<N, N>, T, Self> {
        todo!()
    }

    fn randn<S: Shape>() -> Buffer<'d, S, T, Self> {
        todo!()
    }

    fn uniform<S: Shape>(low: T, high: T) -> Buffer<'d, S, T, Self> {
        todo!()
    }
}

#[derive(Debug)]
pub struct Buffer<'d, S, T = f32, D = cpu::Device>
where
    S: Shape,
    T: DType,
    D: Device,
{
    device: &'d D,
    strides: S::AsArray, // strides and offset are tracked by us, we pass shape, strides and offset to accelerators
    offset: usize,
    unary_ops: alloc::vec::Vec<UnaryOp>, // we store reduce dimensions for each of these operations, this way the whole reduce can be done in single iteration - O(n)
    reduce_dims: alloc::vec::Vec<alloc::vec::Vec<usize>>, // we store reduce dimensions for each reduce operation,
    data: Arc<D::Buffer<T>>,
}

impl<S, T, D> Clone for Buffer<'_, S, T, D>
where
    S: Shape,
    T: DType,
    D: Device,
{
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            strides: self.strides.clone(),
            offset: self.offset,
            unary_ops: self.unary_ops.clone(),
            reduce_dims: self.reduce_dims.clone(),
            data: self.data.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// These are all unary operations that are stored and evaluated lazily
// We should put many operations in here, including binary operations with constants, so that we get most out of this
// lazy evaluation.
enum UnaryOp {
    Exp,
    Inverse,
    Ln,
    Neg,
    ReLU,
    DReLU,
    Tanh,
    SumReduce,
    //MaxReduce, // TODO how do we deal with indices tracking?
    //MinReduce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SyncOp {
    MatMul,
    BinOp(BinaryOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/* These two movement ops are done automatically by us simply by tracking shape and strides.
    Expand,
    Permute,

    //Reshape, // for now we will not support reshape as it makes us create a copy in certain cases
*/

