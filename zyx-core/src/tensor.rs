extern crate alloc;
use crate::axes::IntoAxes;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::{backend::Backend, node::Node};
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::iter::repeat;
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

    /// Returns first element stored in this tensor.
    /// Usually used for tensors with exactly one element.
    /// # Panics
    /// Panics if tensor has zero elements.
    pub fn item<T: Scalar>(&self) -> T {
        self.backend.load::<T>(self.id).first().unwrap().clone()
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
        self.unary_op(UOp::Sqrt)
    }

    #[must_use]
    pub fn dropout<T: Scalar>(&self, probability: T) -> Tensor<B> {
        self * probability.into_tensor(self.backend).cmplt(tensor(
            self.backend._uniform(self.shape(), self.dtype()),
            self.backend,
        ))
    }

    #[must_use]
    pub fn sigmoid(&self) -> Tensor<B> {
        let one = 1.into_tensor(self.backend);
        &one / (&one + (-self).exp())
    }

    #[must_use]
    pub fn swish(&self) -> Tensor<B> {
        self * self.sigmoid()
    }

    #[must_use]
    pub fn leaky_relu<T: Scalar>(&self, neg_slope: T) -> Tensor<B> {
        let neg_slope = neg_slope.into_tensor(self.backend);
        self.relu() - (self * (-neg_slope)).relu()
    }

    #[must_use]
    pub fn elu<T: Scalar>(&self, alpha: T) -> Tensor<B> {
        self.relu() - (1f32.into_tensor(self.backend) - self.exp()).relu() * alpha
    }

    #[must_use]
    pub fn tan(&self) -> Tensor<B> {
        self.sin() / self.cos()
    }

    #[must_use]
    pub fn gelu(&self) -> Tensor<B> {
        self * 0.5f32
            * (((self + self.pow(3f32) * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt())
                .tanh()
                + 1f32)
    }

    #[must_use]
    pub fn quick_gelu(&self) -> Tensor<B> {
        self * (self * 1.702).sigmoid()
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
        let y = rhs.into_tensor(self.backend);
        let xshape = self.shape();
        let xrank = xshape.rank();
        let yshape = y.shape();
        let yrank = yshape.rank();
        assert_eq!(xshape[-1], yshape[-(yrank.min(2) as i64)]);
        (self.reshape(
            xshape[0..-1]
                .into_iter()
                .copied()
                .chain([1])
                .chain([xshape[-1]])
                .collect::<Vec<usize>>(),
        ) * y
            .reshape(
                yshape[0..-2]
                    .into_iter()
                    .copied()
                    .chain([1])
                    .chain([yshape[-(yrank.min(2) as i64)]])
                    .collect::<Vec<usize>>(),
            )
            .transpose())
        .sum(-1)
    }

    // Movement ops
    #[must_use]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<B> {
        tensor(
            self.backend.push(Node::Reshape(self.id, shape.into())),
            self.backend,
        )
    }

    #[must_use]
    pub fn expand(&self, shape: impl Into<Shape>) -> Tensor<B> {
        tensor(
            self.backend.push(Node::Expand(self.id, shape.into())),
            self.backend,
        )
    }

    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().permute(&axes);
        tensor(
            self.backend.push(Node::Permute(self.id, axes, shape)),
            self.backend,
        )
    }

    #[must_use]
    pub fn transpose(&self) -> Tensor<B> {
        let rank = self.rank();
        let x = if rank == 1 {
            self.reshape([1, self.shape()[0]])
        } else {
            self.clone()
        };
        let shape = x.shape();
        let mut axes: Box<[usize]> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        let axes = axes.into_axes(rank);
        let res_shape = shape.permute(&axes);
        tensor(
            self.backend.push(Node::Permute(x.id, axes, res_shape)),
            self.backend,
        )
    }

    // Reduce ops
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        let mut uniq = BTreeSet::new();
        assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot sum tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        for a in &axes {
            assert!(
                *a < shape.rank(),
                "Cannot sum tensor with shape {:?} by axes {:?}, because some axes are greater than rank.",
                self.shape(),
                axes
            );
        }
        tensor(
            self.backend.push(Node::Sum(self.id, axes, shape)),
            self.backend,
        )
    }

    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        let mut uniq = BTreeSet::new();
        assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot sum tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        for a in &axes {
            assert!(
                *a < shape.rank(),
                "Cannot sum tensor with shape {:?} by axes {:?}, because some axes are greater than rank.",
                self.shape(),
                axes
            );
        }
        tensor(
            self.backend.push(Node::Max(self.id, axes, shape)),
            self.backend,
        )
    }

    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor<B> {
        let shape = self.shape();
        let axes = axes.into_axes(shape.rank());
        self.sum(axes.clone()) / axes.iter().copied().map(|a| shape[a]).product::<usize>() as i32
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
        let mut x = self.clone();
        let mut y = rhs.into_tensor(self.backend);
        // This does both automatic expand AND automatic casting between dtypes.
        // TODO Both of these can be disable by changing a setting in the backend.
        {
            /*assert_eq!(
                graph.dtype(xid),
                graph.dtype(yid),
                "{op} parameters {xid} and {yid} have different dtypes: {} and {}",
                graph.dtype(xid),
                graph.dtype(yid)
            );*/
            // Now we just do implicit conversions. Not exactly rust style, but it's convenient.
            // We can later add option for backend to disable these implicit conversions.
            match (x.dtype(), y.dtype()) {
                (DType::F32, DType::I32) => y = y.cast(DType::F32),
                (DType::I32, DType::F32) => x = x.cast(DType::F32),
                _ => {}
            }
            let mut shapex = x.shape();
            let mut shapey = y.shape();
            let rx = shapex.rank();
            let ry = shapey.rank();
            match rx.cmp(&ry) {
                Ordering::Less => {
                    shapex = repeat(1)
                        .take(ry - rx)
                        .chain(shapex.into_iter().copied())
                        .collect::<Vec<usize>>()
                        .into();
                }
                Ordering::Greater => {
                    shapey = repeat(1)
                        .take(rx - ry)
                        .chain(shapey.into_iter().copied())
                        .collect::<Vec<usize>>()
                        .into();
                }
                Ordering::Equal => {}
            }
            let mut eshape = Vec::new();
            for (x, y) in shapex.into_iter().zip(shapey.into_iter()) {
                eshape.push(*x.max(y));
            }
            let eshape: Shape = eshape.into();
            if shapex != eshape {
                x = x.expand(eshape.clone());
            }
            if shapey != eshape {
                y = y.expand(eshape);
            }
        }
        tensor(
            match op {
                BOp::Add => x.backend.push(Node::Add(x.id, y.id)),
                BOp::Sub => x.backend.push(Node::Sub(x.id, y.id)),
                BOp::Mul => x.backend.push(Node::Mul(x.id, y.id)),
                BOp::Div => x.backend.push(Node::Div(x.id, y.id)),
                BOp::Pow => x.backend.push(Node::Pow(x.id, y.id)),
                BOp::Cmplt => x.backend.push(Node::Cmplt(x.id, y.id)),
            },
            x.backend,
        )
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

impl<B: Backend> core::ops::Neg for Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        self.unary_op(UOp::Neg)
    }
}

impl<B: Backend> core::ops::Neg for &Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        self.unary_op(UOp::Neg)
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

impl<B: Backend, T: Scalar> IntoTensor<B> for Vec<T> {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => Node::IterF32(Box::new(self.into_iter().map(T::into_f32)), n.into()),
                DType::I32 => Node::IterI32(Box::new(self.into_iter().map(T::into_i32)), n.into()),
            }),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for &'static [T] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => Node::IterF32(
                    Box::new(self.into_iter().cloned().map(T::into_f32)),
                    n.into(),
                ),
                DType::I32 => Node::IterI32(
                    Box::new(self.into_iter().cloned().map(T::into_i32)),
                    n.into(),
                ),
            }),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for T {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => {
                    Node::IterF32(Box::new([self].into_iter().map(T::into_f32)), 1.into())
                }
                DType::I32 => {
                    Node::IterI32(Box::new([self].into_iter().map(T::into_i32)), 1.into())
                }
            }),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize> IntoTensor<B> for [T; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => Node::IterF32(Box::new(self.into_iter().map(T::into_f32)), D0.into()),
                DType::I32 => Node::IterI32(Box::new(self.into_iter().map(T::into_i32)), D0.into()),
            }),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize> IntoTensor<B> for [[T; D1]; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => Node::IterF32(
                    Box::new(self.into_iter().flatten().map(T::into_f32)),
                    [D0, D1].into(),
                ),
                DType::I32 => Node::IterI32(
                    Box::new(self.into_iter().flatten().map(T::into_i32)),
                    [D0, D1].into(),
                ),
            }),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize, const D2: usize> IntoTensor<B>
    for [[[T; D2]; D1]; D0]
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.push(match T::dtype() {
                DType::F32 => Node::IterF32(
                    Box::new(self.into_iter().flatten().flatten().map(T::into_f32)),
                    [D0, D1, D2].into(),
                ),
                DType::I32 => Node::IterI32(
                    Box::new(self.into_iter().flatten().flatten().map(T::into_i32)),
                    [D0, D1, D2].into(),
                ),
            }),
            backend,
        )
    }
}
