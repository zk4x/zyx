extern crate alloc;
use crate::axes::IntoAxes;
use crate::dtype::DType;
use crate::error::ZyxError;
use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::{backend::Backend, node::Node};
use alloc::{boxed::Box, collections::BTreeSet, vec::Vec};
use core::{
    cmp::Ordering,
    iter::repeat,
    ops::{Range, SubAssign},
};

/// Id of tensor.
#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub struct Id(usize);

/// Create new id.
pub const fn id(id: usize) -> Id {
    Id(id)
}

impl Id {
    /// Convert id to usize
    pub const fn i(self) -> usize {
        self.0
    }
}

impl core::fmt::Display for Id {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl SubAssign<usize> for Id {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

/// Into i64 range, used for indexing
pub trait IntoRange: Clone {
    /// Convert self to range i64, if it is scalar, it gets converted to x..x+1
    fn into_range(self) -> Range<i64>;
}

impl IntoRange for Range<i64> {
    fn into_range(self) -> Range<i64> {
        self
    }
}

impl IntoRange for i64 {
    fn into_range(self) -> Range<i64> {
        self..self+1
    }
}

/// Implemented for objects that can be used to index tensors.
pub trait IntoIndex {
    /// Convert self to tensor index.
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>>;
}

impl<I: IntoRange> IntoIndex for &[I] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        self.iter().cloned().map(IntoRange::into_range)
    }
}

impl<I0: IntoRange, I1: IntoRange> IntoIndex for (I0, I1) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange> IntoIndex for (I0, I1, I2) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange> IntoIndex for (I0, I1, I2, I3) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range(), self.3.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange> IntoIndex for (I0, I1, I2, I3, I4) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range(), self.3.into_range(), self.4.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange, I5: IntoRange> IntoIndex for (I0, I1, I2, I3, I4, I5) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range(), self.3.into_range(), self.4.into_range(), self.5.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange, I5: IntoRange, I6: IntoRange> IntoIndex for (I0, I1, I2, I3, I4, I5, I6) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range(), self.3.into_range(), self.4.into_range(), self.5.into_range(), self.6.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange, I5: IntoRange, I6: IntoRange, I7: IntoRange> IntoIndex for (I0, I1, I2, I3, I4, I5, I6, I7) {
    fn into_index(self) -> impl IntoIterator<Item=Range<i64>> {
        [self.0.into_range(), self.1.into_range(), self.2.into_range(), self.3.into_range(), self.4.into_range(), self.5.into_range(), self.6.into_range(), self.7.into_range()].into_iter()
    }
}

/// Tensor is the core object of zyx.
/// It is multidimensional array.
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
        self.backend.release(self.id).unwrap();
    }
}

impl<B: Backend> core::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("Tensor {{ id = {:?} }}", self.id))
    }
}

impl<B: Backend> core::fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO don't print the whole tensor if it is too big
        let precision = if let Some(precision) = f.precision() {
            precision
        } else {
            3
        };
        let res = match self.dtype() {
            DType::F32 => {
                if let Ok(data) = &self.to_vec::<f32>() {
                    tensor_to_string(data, &self.shape(), precision)
                } else {
                    "f32 tensor failed to realize".into()
                }
            }
            DType::I32 => {
                if let Ok(data) = &self.to_vec::<i32>() {
                    tensor_to_string(data, &self.shape(), precision)
                } else {
                    "i32 tensor failed to realize".into()
                }
            }
        };
        f.write_str(&res)
    }
}

fn tensor_to_string<T: core::fmt::Display>(
    data: &[T],
    shape: &Shape,
    precision: usize,
) -> alloc::string::String {
    use core::fmt::Write;
    // TODO don't print whole tensor if it is big
    let n = shape.numel();
    let ndim = shape.rank();
    let mut res = alloc::string::String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let mut w = 0;
    for x in data {
        let l = alloc::format!("{x:>w$.precision$}").len();
        if l > w {
            w = l;
        }
    }
    let d0 = shape[-1];
    for (i, x) in data.iter().enumerate() {
        {
            let mut var = 1;
            let mut r = ndim;
            while r > 0 {
                if i % (n / var) == 0 {
                    res += &(" ".repeat(ndim - r) + "[".repeat(r - 1).as_str());
                    break;
                }
                var *= shape[ndim - r];
                r -= 1;
            }
        }
        let _ = write!(res, "{x:>w$.precision$}");
        if (i + 1) % d0 != 0usize {
            res += "  ";
        }
        {
            let mut var = 1;
            let mut r = ndim;
            while r > 0 {
                if (i + 1) % (n / var) == 0 {
                    res += &"]".repeat(r - 1);
                    break;
                }
                var *= shape[ndim - r];
                r -= 1;
            }
        }
        if (i + 1) % d0 == 0usize && i != n - 1 {
            res += "\n";
        }
    }
    res
}

/// Create new tensor from id and backend.
/// Used mostly internally in tensor and in backends.
pub const fn tensor<B: Backend>(id: Id, backend: B) -> Tensor<B> {
    Tensor { id, backend }
}

impl<B: Backend> Tensor<B> {
    // Metadata
    /// Tensor's unique identification.
    /// Any tensor on one backend will always have different id.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Returns the [shape](Shape) of the self tensor.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// assert_eq!(x.shape(), [2, 3]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.backend.shape(self.id)
    }

    /// Returns number of elements in the self tensor.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// assert_eq!(x.numel(), 6);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Returns the [dtype](DType) of the self tensor.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// assert_eq!(x.dtype(), zyx_opencl::DType::I32);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.backend.dtype(self.id)
    }

    /// Returns the rank of the self tensor. This is the number of tensor's dimensions.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// assert_eq!(x.rank(), 2);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Returns the [backend](Backend) of the self tensor.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// let y = x.backend().randn([2, 4, 3], zyx_opencl::DType::F32);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn backend(&self) -> B {
        self.backend
    }

    // Access methods
    /// Load tensor from backend into vector
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// let xvec: Vec<i32> = x.to_vec()?;
    /// assert_eq!(xvec, vec![2, 3, 1, 4, 1, 3]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    pub fn to_vec<T: Scalar>(&self) -> Result<Vec<T>, ZyxError> {
        if T::dtype() != self.dtype() {
            return Err(ZyxError::InvalidDType {
                expected: T::dtype(),
                found: self.dtype(),
            });
        }
        self.backend.load(self.id)
    }

    /// Returns first element stored in this tensor.
    /// Usually used for tensors with exactly one element.
    /// Error is returned if self tensor contains zero elements
    /// or if backend returns error.
    /// ```
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 1], [4, 1, 3]]);
    /// let xitem: i32 = x.item()?;
    /// assert_eq!(xitem, 2);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    pub fn item<T: Scalar>(&self) -> Result<T, ZyxError> {
        self.backend
            .load::<T>(self.id)?
            .first()
            .ok_or(ZyxError::IndexOutOfBounds { index: 0, len: 0 })
            .cloned()
    }

    // Backpropagation
    /// Returns gradients of self w.r.t. sources.
    /// ```rust
    /// let dev = zyx_opencl::device()?;
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    pub fn backward<'a>(
        &'a self,
        sources: impl IntoIterator<Item = &'a Tensor<B>>,
    ) -> Vec<Option<Tensor<B>>>
    where
        B: 'a,
    {
        let sources: Vec<&Tensor<B>> = sources.into_iter().collect();
        let grads = self
            .backend
            .backward(self.id, &sources.iter().map(|t| t.id).collect())
            .unwrap();
        sources
            .into_iter()
            .map(move |x: &Tensor<B>| grads.get(&x.id).cloned())
            .map(move |x| x.map(|x| tensor(x, self.backend))).collect()
    }

    // Unary ops
    /// Cast self into dtype.
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor<B> {
        match dtype {
            DType::F32 => self.unary_op(UOp::CastF32),
            DType::I32 => self.unary_op(UOp::CastI32),
        }
    }

    /// Returns a new tensor with the rectified linear unit function applied to the elements of self.
    #[must_use]
    pub fn relu(&self) -> Tensor<B> {
        self.unary_op(UOp::ReLU)
    }

    /// Returns a new tensor with the sine of the elements of self.
    #[must_use]
    pub fn sin(&self) -> Tensor<B> {
        self.unary_op(UOp::Sin)
    }

    /// Returns a new tensor with the cosine of the elements of self.
    #[must_use]
    pub fn cos(&self) -> Tensor<B> {
        self.unary_op(UOp::Cos)
    }

    /// Returns a new tensor with the natural logarithm of the elements of self.
    /// Due to performance reasons, this function does not check if self fits
    /// into domain of ln(x). Result on out of domain numbers is implementation
    /// defined (when x <= 0).
    #[must_use]
    pub fn ln(&self) -> Tensor<B> {
        self.unary_op(UOp::Ln)
    }

    /// Returns a new tensor with the exponential of the elements of self.
    #[must_use]
    pub fn exp(&self) -> Tensor<B> {
        self.unary_op(UOp::Exp)
    }

    /// Returns a new tensor with the hyperbolic tangent of the elements of self.
    #[must_use]
    pub fn tanh(&self) -> Tensor<B> {
        self.unary_op(UOp::Tanh)
    }

    /// Returns a new tensor with the square root of the elements of self.
    /// Due to performance reasons, this function does not check if self fits
    /// into domain of ln(x). Result on out of domain numbers is implementation
    /// defined (when x < 0).
    #[must_use]
    pub fn sqrt(&self) -> Tensor<B> {
        self.unary_op(UOp::Sqrt)
    }

    /// Returns a new tensor with each element of self randomly zeroed with given probability.
    #[must_use]
    pub fn dropout(&self, probability: impl Scalar) -> Tensor<B> {
        self * probability
            .into_tensor(self.backend)
            .cmplt(self.backend.uniform(self.shape(), 0.0..1.0))
    }

    /// Returns a new tensor with the sigmoid (logistic function) of the elements of self.
    #[must_use]
    pub fn sigmoid(&self) -> Tensor<B> {
        let one = 1.into_tensor(self.backend);
        &one / (&one + (-self).exp())
    }

    /// Returns a new tensor with the swish of the elements of self.
    #[must_use]
    pub fn swish(&self) -> Tensor<B> {
        self * self.sigmoid()
    }

    /// Returns a new tensor with the leaky relu of the elements of self.
    #[must_use]
    pub fn leaky_relu(&self, neg_slope: impl Scalar) -> Tensor<B> {
        let neg_slope = neg_slope.into_tensor(self.backend);
        self.relu() - (self * (-neg_slope)).relu()
    }

    /// Returns a new tensor with the elu of the elements of self.
    #[must_use]
    pub fn elu(&self, alpha: impl Scalar) -> Tensor<B> {
        self.relu() - (1f32.into_tensor(self.backend) - self.exp()).relu() * alpha
    }

    /// Returns a new tensor with the tangent of the elements of self.
    #[must_use]
    pub fn tan(&self) -> Tensor<B> {
        self.sin() / self.cos()
    }

    /// Returns a new tensor with the gelu of the elements of self.
    #[must_use]
    pub fn gelu(&self) -> Tensor<B> {
        self * 0.5f32
            * (((self + self.pow(3f32) * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt())
                .tanh()
                + 1f32)
    }

    /// Returns a new tensor with the quick gelu of the elements of self.
    /// ```
    ///
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.randn([2, 3, 1], zyx_opencl::DType::F32);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn quick_gelu(&self) -> Tensor<B> {
        self * (self * 1.702).sigmoid()
    }

    // Binary ops
    /// Exponentiation on self
    #[must_use]
    pub fn pow(&self, exponent: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(exponent, BOp::Pow)
    }

    /// Elementwise compare less than between self and rhs
    #[must_use]
    pub fn cmplt(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.binary_op(rhs, BOp::Cmplt)
    }

    /// Returns a new tensor with the true values replaced with if_true and the false values replaced with if_false.
    #[must_use]
    pub fn where_(&self, if_true: impl IntoTensor<B>, if_false: impl IntoTensor<B>) -> Tensor<B> {
        //let x = todo!();
        //let y = todo!();
        //let z = todo!();
        //self.backend.push(Node::Where(x, y, z)).unwrap();
        todo!()
    }

    /// Dot product (mathematical multiplication) of self and rhs
    #[must_use]
    pub fn dot(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        let y = rhs.into_tensor(self.backend);
        let xshape = self.shape();
        let yshape = y.shape();
        let yrank = yshape.rank();
        assert_eq!(
            xshape[-1],
            yshape[-(yrank.min(2) as i64)],
            "Cannot dot tensors with shapes {xshape} and {yshape}"
        );
        (self.reshape(
            xshape[0..-1]
                .iter()
                .copied()
                .chain([1])
                .chain([xshape[-1]])
                .collect::<Box<[usize]>>(),
        ) * y
            .reshape(
                yshape[0..-2]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain(yshape[-(yrank.min(2) as i64)..yrank as i64].iter().copied())
                    .collect::<Box<[usize]>>(),
            )
            .transpose())
        .sum(-1)
        .reshape(
            xshape[0..-1]
                .iter()
                .copied()
                .chain([yshape[-1]])
                .collect::<Box<[usize]>>(),
        )
    }

    // Movement ops
    /// Reshape self to shape.
    /// # Panics
    /// Following must hold:
    /// self.numel() == shape.numel()
    #[must_use]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<B> {
        let shape = shape.into();
        assert_eq!(
            self.shape().numel(),
            shape.numel(),
            "Cannot reshape tensor with shape {} to {shape}",
            self.shape()
        );
        tensor(
            self.backend.push(Node::Reshape(self.id, shape)).unwrap(),
            self.backend,
        )
    }

    /// Expand self into bigger shape
    #[must_use]
    pub fn expand(&self, shape: impl Into<Shape>) -> Tensor<B> {
        tensor(
            self.backend
                .push(Node::Expand(self.id, shape.into()))
                .unwrap(),
            self.backend,
        )
    }

    /// Constant padding
    ///
    /// This can both add and remove values from tensor. Negative padding removes values, positive padding
    /// adds values.
    ///
    /// Pad last dimension by (1, 2)
    /// ```rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3],
    ///                     [4, 1]]);
    /// let z = x.pad([(1, 2)], 0);
    /// std::println!("{}", z);
    /// assert_eq!(z, [[0, 2, 3, 0, 0],
    ///                [0, 4, 1, 0, 0]]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    /// Pad last dimension by (2, -1) and second last dimension by (1, 1)
    /// ```rust
    /// # use zyx_opencl;
    /// # let dev = zyx_opencl::device()?;
    /// # let x = dev.tensor([[2, 3],
    /// #                     [4, 1]]);
    /// let z = x.pad([(2, -1), (1, 1)], 0);
    /// println!("z: {z}");
    /// assert_eq!(z, [[0, 0, 0],
    ///                [0, 0, 2],
    ///                [0, 0, 4],
    ///                [0, 0, 0]]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    ///
    /// # Panics
    /// T must be of the same dtype as Tensor's dtype, otherwise this function panics.
    #[must_use]
    pub fn pad(
        &self,
        padding: impl IntoIterator<Item = (i64, i64)>,
        value: impl Scalar,
    ) -> Tensor<B> {
        // Cool trick :)
        fn get_dtype<T: Scalar>(_: T) -> DType {
            T::dtype()
        }
        fn get_zero<T: Scalar>(_: T) -> T {
            T::zero()
        }
        let dtype = self.dtype();
        assert_eq!(
            get_dtype(value.clone()),
            dtype,
            "Cannot pad tensor with dtype {} with value of dtype {}",
            dtype,
            get_dtype(value.clone())
        );
        // TODO asserts
        let padding: Box<[(i64, i64)]> = padding.into_iter().collect();
        let sh = self.shape();
        let psh = sh.clone().pad(&padding);
        let t0 = tensor(
            self.backend
                .push(Node::Pad(self.id, padding.clone(), psh.clone()))
                .unwrap(),
            self.backend,
        );
        let zero = get_zero(value.clone());
        if value.clone().is_equal(zero.clone()) {
            t0
        } else {
            t0 + tensor(
                self.backend
                    .push(Node::Pad(
                        self.backend.ones(sh, dtype).id,
                        padding,
                        psh.clone(),
                    ))
                    .unwrap(),
                self.backend,
            )
            .where_(zero, value)
        }
    }

    /// Reorder axes of self
    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().permute(&axes);
        tensor(
            self.backend
                .push(Node::Permute(self.id, axes, shape))
                .unwrap(),
            self.backend,
        )
    }

    /// Swap last two axes of self
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
            self.backend
                .push(Node::Permute(x.id, axes, res_shape))
                .unwrap(),
            self.backend,
        )
    }

    // Reduce ops
    /// Reduce self by summing along axes
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
            self.backend.push(Node::Sum(self.id, axes, shape)).unwrap(),
            self.backend,
        )
    }

    /// Reduce self by maximizing along axes
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
            self.backend.push(Node::Max(self.id, axes, shape)).unwrap(),
            self.backend,
        )
    }

    /// Reduce self by calculating mean along axes
    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor<B> {
        let shape = self.shape();
        let axes = axes.into_axes(shape.rank());
        self.sum(axes.clone()) / axes.iter().copied().map(|a| shape[a]).product::<usize>() as i32
    }

    /// Reduce self by calculating variance along axes
    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        (self - self.mean(axes.clone())).pow(2).sum(axes)
    }

    /// Reduce self by calculating standard deviation along axes
    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Tensor<B> {
        self.var(axes).sqrt()
    }

    /// Tensor indexing
    /// ```rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 4],
    ///                     [4, 1, 8]]);
    /// let y = x.get((-1, -3));
    /// assert_eq!(y, 4);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn get(&self, index: impl IntoIndex) -> Tensor<B> {
        // TODO asserts
        let padding: Vec<(i64, i64)> = index.into_index()
            .into_iter()
            .zip(self.shape().iter())
            .map(|(r, d)| (
                if r.start >= 0 { -r.start } else { -r.start - *d as i64 },
                if r.end > 0 { *d as i64-r.end } else { r.end }
            )).collect();
        self.pad(
            padding.into_iter().rev(),
            0,
        )
    }

    #[must_use]
    fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor<B>>, dim: i64) -> Tensor<B>
    where
        B: 'a
    {
        let tensors: Vec<&Tensor<B>> = tensors.into_iter().collect();
        let shape = tensors[0].shape();
        let rank = shape.rank();
        let dim = if dim < 0 { dim + rank as i64 } else { dim } as usize;
        // Dimension check
        for tensor in tensors {
            for (i, (d1, d2)) in shape.iter().zip(tensor.shape().iter()).enumerate() {
                if i != dim {
                    assert_eq!(*d1, *d2, "Cannot concatenate these tensors.");
                }
            }
        }
        // pad everything
        let offset = 0;
        // then sum it together
        todo!()
    }

    /// Stack multiple tensors into one
    /*#[must_use]
    pub fn stack<'a>(tensors: impl IntoIterator<Item = &'a Tensor<B>>, dim: i64) -> Tensor<B>
    where
        B: 'a
    {
        todo!()
    }*/

    /// Split into multiple tensors
    #[must_use]
    pub fn split(&self, sizes: &[i64], dim: i64) -> Vec<Tensor<B>> {
        todo!()
    }

    //#[must_use]
    //pub fn pool(&self)

    //#[must_use]
    //pub fn conv(&self)
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
            self.backend
                .push(match op {
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
                })
                .unwrap(),
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
            let mut x_shape = x.shape();
            let mut y_shape = y.shape();

            for (x, y) in x_shape.iter().zip(y_shape.iter()) {
                if x != y {
                    assert!(*x == 1 || *y == 1, "Left and right tensors have incompatible shapes for binary op: {x_shape} and {y_shape}");
                }
            }

            let rx = x_shape.rank();
            let ry = y_shape.rank();
            match rx.cmp(&ry) {
                Ordering::Less => {
                    x_shape = repeat(1)
                        .take(ry - rx)
                        .chain(x_shape.into_iter().copied())
                        .collect::<Vec<usize>>()
                        .into();
                }
                Ordering::Greater => {
                    y_shape = repeat(1)
                        .take(rx - ry)
                        .chain(y_shape.into_iter().copied())
                        .collect::<Vec<usize>>()
                        .into();
                }
                Ordering::Equal => {}
            }
            let mut eshape = Vec::new();
            for (x, y) in x_shape.into_iter().zip(y_shape.into_iter()) {
                eshape.push(*x.max(y));
            }
            let eshape: Shape = eshape.into();
            if x_shape != eshape {
                x = x.expand(eshape.clone());
            }
            if y_shape != eshape {
                y = y.expand(eshape);
            }
        }
        tensor(
            x.backend
                .push(match op {
                    BOp::Add => Node::Add(x.id, y.id),
                    BOp::Sub => Node::Sub(x.id, y.id),
                    BOp::Mul => Node::Mul(x.id, y.id),
                    BOp::Div => Node::Div(x.id, y.id),
                    BOp::Pow => Node::Pow(x.id, y.id),
                    BOp::Cmplt => Node::Cmplt(x.id, y.id),
                })
                .unwrap(),
            x.backend,
        )
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

/// Objects must implement this to be convertible into tensor
pub trait IntoTensor<B: Backend> {
    /// Convert self into tensor
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
            backend
                .push(match T::dtype() {
                    DType::F32 => {
                        Node::IterF32(Box::new(self.into_iter().map(T::into_f32)), n.into())
                    }
                    DType::I32 => {
                        Node::IterI32(Box::new(self.into_iter().map(T::into_i32)), n.into())
                    }
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for &'static [T] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        let n = self.len();
        tensor(
            backend
                .push(match T::dtype() {
                    DType::F32 => {
                        Node::IterF32(Box::new(self.iter().cloned().map(T::into_f32)), n.into())
                    }
                    DType::I32 => {
                        Node::IterI32(Box::new(self.iter().cloned().map(T::into_i32)), n.into())
                    }
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for T {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend
                .push(match T::dtype() {
                    DType::F32 => {
                        Node::IterF32(Box::new([self].into_iter().map(T::into_f32)), 1.into())
                    }
                    DType::I32 => {
                        Node::IterI32(Box::new([self].into_iter().map(T::into_i32)), 1.into())
                    }
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize> IntoTensor<B> for [T; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend
                .push(match T::dtype() {
                    DType::F32 => {
                        Node::IterF32(Box::new(self.into_iter().map(T::into_f32)), D0.into())
                    }
                    DType::I32 => {
                        Node::IterI32(Box::new(self.into_iter().map(T::into_i32)), D0.into())
                    }
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize> IntoTensor<B> for [[T; D1]; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend
                .push(match T::dtype() {
                    DType::F32 => Node::IterF32(
                        Box::new(self.into_iter().flatten().map(T::into_f32)),
                        [D0, D1].into(),
                    ),
                    DType::I32 => Node::IterI32(
                        Box::new(self.into_iter().flatten().map(T::into_i32)),
                        [D0, D1].into(),
                    ),
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize, const D2: usize> IntoTensor<B>
    for [[[T; D2]; D1]; D0]
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend
                .push(match T::dtype() {
                    DType::F32 => Node::IterF32(
                        Box::new(self.into_iter().flatten().flatten().map(T::into_f32)),
                        [D0, D1, D2].into(),
                    ),
                    DType::I32 => Node::IterI32(
                        Box::new(self.into_iter().flatten().flatten().map(T::into_i32)),
                        [D0, D1, D2].into(),
                    ),
                })
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar + PartialEq> PartialEq<T> for Tensor<B>
{
    fn eq(&self, other: &T) -> bool {
        self.numel() == 1
            && self.dtype() == T::dtype()
            && self.item::<T>().unwrap() == *other
    }
}

impl<B: Backend, T: Scalar + PartialEq, const D0: usize> PartialEq<[T; D0]>
    for Tensor<B>
{
    fn eq(&self, other: &[T; D0]) -> bool {
        self.shape() == [D0]
            && self.dtype() == T::dtype()
            && self
                .to_vec::<T>()
                .unwrap()
                .into_iter()
                .zip(other.iter())
                .all(|(x, y)| x == *y)
    }
}

impl<B: Backend, T: Scalar + PartialEq, const D0: usize, const D1: usize>
    PartialEq<[[T; D1]; D0]> for Tensor<B>
{
    fn eq(&self, other: &[[T; D1]; D0]) -> bool {
        self.shape() == [D0, D1]
            && self.dtype() == T::dtype()
            && self
                .to_vec::<T>()
                .unwrap()
                .into_iter()
                .zip(other.iter().flatten())
                .all(|(x, y)| x == *y)
    }
}

impl<
        B: Backend,
        T: Scalar + PartialEq,
        const D0: usize,
        const D1: usize,
        const D2: usize,
    > PartialEq<[[[T; D2]; D1]; D0]> for Tensor<B>
{
    fn eq(&self, other: &[[[T; D2]; D1]; D0]) -> bool {
        self.shape() == [D0, D1, D2]
            && self.dtype() == T::dtype()
            && self
                .to_vec::<T>()
                .unwrap()
                .into_iter()
                .zip(other.iter().flatten().flatten())
                .all(|(x, y)| x == *y)
    }
}
