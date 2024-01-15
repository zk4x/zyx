extern crate alloc;
use crate::axes::IntoAxes;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::{backend::Backend, node::Node};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::{Range, SubAssign};

#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub struct Id(usize);

pub fn id(id: usize) -> Id {
    Id(id)
}

impl Id {
    pub fn i(self) -> usize {
        self.0
    }
}

impl SubAssign<usize> for Id {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

pub struct Tensor<B: Backend> {
    id: Id,
    backend: B,
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        self.backend.retain(self.id);
        tensor(self.id, self.backend)
    }
}

impl<B: Backend> Drop for Tensor<B> {
    fn drop(&mut self) {
        self.backend.release(self.id);
    }
}

pub fn tensor<B: Backend>(id: Id, backend: B) -> Tensor<B> {
    Tensor { id, backend }
}

impl<B: Backend> Tensor<B> {
    // Metadata
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.backend.shape(self.id)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.backend.dtype(self.id)
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    #[must_use]
    pub fn backend(&self) -> B {
        self.backend
    }

    // Access methods
    pub fn to_vec<T: Scalar>(&self) -> Vec<T> {
        // TODO perhaps this function can return Result?
        assert_eq!(T::dtype(), self.dtype());
        self.backend.load(self.id)
    }

    // Backpropagation
    #[must_use]
    pub fn backward<'a>(
        &'a self,
        sources: impl IntoIterator<Item = &'a Tensor<B>>,
    ) -> impl Iterator<Item = Option<Tensor<B>>> + 'a
    where
        B: 'a,
    {
        let sources: Vec<&Tensor<B>> = sources.into_iter().collect();
        let grads = self
            .backend
            .backward(self.id, &sources.iter().map(|t| t.id).collect());
        sources
            .into_iter()
            .map(move |x: &Tensor<B>| grads.get(&x.id).cloned())
            .map(move |x| x.map(|x| tensor(x, self.backend)))
    }

    // Unary ops
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor<B> {
        match dtype {
            DType::F32 => self.unary_op(UOp::CastF32),
            DType::I32 => self.unary_op(UOp::CastI32),
        }
    }

    #[must_use]
    pub fn neg(&self) -> Tensor<B> {
        self.unary_op(UOp::Neg)
    }

    #[must_use]
    pub fn relu(&self) -> Tensor<B> {
        self.unary_op(UOp::ReLU)
    }

    #[must_use]
    pub fn sin(&self) -> Tensor<B> {
        self.unary_op(UOp::Sin)
    }

    #[must_use]
    pub fn cos(&self) -> Tensor<B> {
        self.unary_op(UOp::Cos)
    }

    #[must_use]
    pub fn ln(&self) -> Tensor<B> {
        self.unary_op(UOp::Ln)
    }

    #[must_use]
    pub fn exp(&self) -> Tensor<B> {
        self.unary_op(UOp::Exp)
    }

    #[must_use]
    pub fn tanh(&self) -> Tensor<B> {
        self.unary_op(UOp::Tanh)
    }

    #[must_use]
    pub fn sqrt(&self) -> Tensor<B> {
        self.unary_op(UOp::Tanh)
    }

    #[must_use]
    pub fn dropout(&self, probability: f64) -> Tensor<B> {
        // This just uses Node::Uniform
        todo!()
    }

    // Binary ops
    #[must_use]
    pub fn add(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Add)
    }

    #[must_use]
    pub fn sub(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Sub)
    }

    #[must_use]
    pub fn mul(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Mul)
    }

    #[must_use]
    pub fn div(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Div)
    }

    #[must_use]
    pub fn pow(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Pow)
    }

    #[must_use]
    pub fn cmplt(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Cmplt)
    }

    #[must_use]
    pub fn dot(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        //(self.reshape([, 1, ]) * rhs.reshape([, 1])).sum(-2)
        todo!()
    }

    // Movement ops
    #[must_use]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<B> {
        tensor(self.backend.push(Node::Reshape(self.id, shape.into())), self.backend)
    }

    #[must_use]
    pub fn expand(&self, shape: impl Into<Shape>) -> Tensor<B> {
        tensor(self.backend.push(Node::Expand(self.id, shape.into())), self.backend)
    }

    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().permute(&axes);
        tensor(self.backend.push(Node::Permute(self.id, axes, shape)), self.backend)
    }

    // Reduce ops
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        tensor(self.backend.push(Node::Sum(self.id, axes, shape)), self.backend) }

    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        tensor(self.backend.push(Node::Max(self.id, axes, shape)), self.backend)
    }

    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor<B> {
        let shape = self.shape();
        let axes = axes.into_axes(shape.rank());
        self.sum(axes.clone())/axes.iter().copied().map(|a| shape[a]).product::<usize>() as i32
    }

    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        (self - self.mean(axes.clone())).pow(2).sum(axes)
    }

    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Tensor<B> {
        self.var(axes).sqrt()
    }

    /// Constant padding
    ///
    /// This can both add and remove values from tensor. Negative padding removes values, positive padding
    /// adds values.
    ///
    /// Pad last dimension by (1, 2)
    /// ```rust
    /// #use zyx_opencl;
    /// #let dev = zyx_opencl::device().unwrap();
    /// let x = dev.tensor([[2, 3],
    ///                     [4, 1]]);
    /// let z = x.pad([(1, 2)], 0);
    /// assert_eq!(z, [[0, 2, 3, 0, 0],
    ///                [0, 4, 1, 0, 0]]);
    /// ```
    /// Pad second to last dimension by (2, -1) and last dimension by (1, 1)
    /// ```rust
    /// #use zyx_opencl;
    /// #let dev = zyx_opencl::device().unwrap();
    /// #let x = dev.tensor([[2, 3],
    /// #                    [4, 1]]);
    /// let z = x.pad([(2, -1), (1, 1)], 7);
    /// assert_eq!(z, [[7, 7, 7],
    ///                [7, 7, 2],
    ///                [7, 7, 4],
    ///                [7, 7, 7]]);
    /// ```
    ///
    /// # Panics
    /// T must be of the same dtype as Tensor's dtype, otherwise this function panics.
    #[must_use]
    pub fn pad<T: Scalar>(
        &self,
        padding: impl IntoIterator<Item = (i64, i64)>,
        value: T,
    ) -> Tensor<B> {
        assert_eq!(T::dtype(), self.dtype());
        todo!()
    }

    /// Tensors should be treated as immutable data structures.
    /// However optimizers need to update model's parameters with new values.
    /// This is the only function that mutates tensor.
    pub fn set(&mut self, x: impl IntoTensor<B>) {
        self.backend.release(self.id);
        let x = x.into_tensor(self.backend);
        // TODO assert equality of backends
        assert_eq!(self.shape(), x.shape());
        assert_eq!(self.dtype(), x.dtype());
        self.backend.retain(x.id);
        self.id = x.id;
        self.backend.set_leaf(self.id);
    }
}

enum UOp {
    CastF32,
    CastI32,
    Neg,
    ReLU,
    Sin,
    Cos,
    Ln,
    Exp,
    Tanh,
    Sqrt,
}

enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
}

// Private helper functions
impl<B: Backend> Tensor<B> {
    #[must_use]
    fn unary_op(&self, op: UOp) -> Tensor<B> {
        tensor(
            self.backend.push(match op {
                UOp::CastF32 => Node::CastF32(self.id),
                UOp::CastI32 => Node::CastI32(self.id),
                UOp::Neg => Node::Neg(self.id),
                UOp::ReLU => Node::ReLU(self.id),
                UOp::Sin => Node::Sin(self.id),
                UOp::Cos => Node::Cos(self.id),
                UOp::Ln => Node::Ln(self.id),
                UOp::Exp => Node::Exp(self.id),
                UOp::Tanh => Node::Tanh(self.id),
                UOp::Sqrt => Node::Sqrt(self.id),
            }),
            self.backend,
        )
    }

    #[must_use]
    fn binary_op(&self, rhs: impl IntoTensor<B>, op: BOp) -> Tensor<B> {
        // This does both automatic expand AND automatic casting between dtypes.
        // Both of these can be disable by changing a setting in the backend.
        todo!()
    }
}

impl<B: Backend> core::ops::Index<&[Range<i64>]> for Tensor<B> {
    type Output = ();
    fn index(&self, index: &[Range<i64>]) -> &Self::Output {
        todo!()
    }
}

impl<B: Backend> core::ops::Index<&[i64]> for Tensor<B> {
    type Output = ();
    fn index(&self, index: &[i64]) -> &Self::Output {
        let index: Vec<Range<i64>> = index.iter().copied().map(|idx| idx..idx + 1).collect();
        let index: &[Range<i64>] = &index;
        self.index(index)
    }
}

impl<B: Backend, const N: usize> core::ops::Index<[Range<i64>; N]> for Tensor<B> {
    type Output = ();
    fn index(&self, index: [Range<i64>; N]) -> &Self::Output {
        let index: &[Range<i64>] = &index;
        self.index(index)
    }
}

impl<B: Backend, const N: usize> core::ops::Index<[i64; N]> for Tensor<B> {
    type Output = ();
    fn index(&self, index: [i64; N]) -> &Self::Output {
        let index: &[i64] = &index;
        self.index(index)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Add<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Add)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Add<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Add)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Sub<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Sub)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Sub<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Sub)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Mul<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Mul)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Mul<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Mul)
    }
}


impl<B: Backend, IT: IntoTensor<B>> core::ops::Div<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Div<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Div)
    }
}
pub trait IntoTensor<B: Backend> {
    fn into_tensor(self, backend: B) -> Tensor<B>;
}

impl<B: Backend> IntoTensor<B> for Tensor<B> {
    fn into_tensor(self, _backend: B) -> Tensor<B> {
        // TODO assert self.backend == backend
        self
    }
}

impl<B: Backend> IntoTensor<B> for &Tensor<B> {
    fn into_tensor(self, _backend: B) -> Tensor<B> {
        // TODO assert self.backend == backend
        self.clone()
    }
}

impl<B: Backend> IntoTensor<B> for Vec<f32> {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(Node::IterF32(Box::new(self.into_iter()), n.into())),
            backend,
        )
    }
}

impl<B: Backend> IntoTensor<B> for &'static [f32] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(Node::IterF32(Box::new(self.into_iter().copied()), n.into())),
            backend,
        )
    }
}

impl<B: Backend> IntoTensor<B> for f32 {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterF32(Box::new([self].into_iter()), 1.into())),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize> IntoTensor<B> for [f32; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterF32(Box::new(self.into_iter()), D0.into())),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize, const D1: usize> IntoTensor<B> for [[f32; D1]; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterF32(
                Box::new(self.into_iter().flatten()),
                [D0, D1].into(),
            )),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize, const D1: usize, const D2: usize> IntoTensor<B>
    for [[[f32; D2]; D1]; D0]
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterF32(
                Box::new(self.into_iter().flatten().flatten()),
                [D0, D1, D2].into(),
            )),
            backend,
        )
    }
}

impl<B: Backend> IntoTensor<B> for Vec<i32> {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(Node::IterI32(Box::new(self.into_iter()), n.into())),
            backend,
        )
    }
}

impl<B: Backend> IntoTensor<B> for &'static [i32] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(Node::IterI32(Box::new(self.into_iter().copied()), n.into())),
            backend,
        )
    }
}

impl<B: Backend> IntoTensor<B> for i32 {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterI32(Box::new([self].into_iter()), 1.into())),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize> IntoTensor<B> for [i32; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterI32(Box::new(self.into_iter()), D0.into())),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize, const D1: usize> IntoTensor<B> for [[i32; D1]; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterI32(
                Box::new(self.into_iter().flatten()),
                [D0, D1].into(),
            )),
            backend,
        )
    }
}

impl<B: Backend, const D0: usize, const D1: usize, const D2: usize> IntoTensor<B>
for [[[i32; D2]; D1]; D0]
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(Node::IterI32(
                Box::new(self.into_iter().flatten().flatten()),
                [D0, D1, D2].into(),
            )),
            backend,
        )
    }
}
