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
    ops::{Range, SubAssign, RangeFull, RangeFrom, RangeTo, RangeInclusive, RangeToInclusive},
};
use crate::utils::SizedIterator;

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

impl IntoRange for RangeFull {
    fn into_range(self) -> Range<i64> {
        0..i64::MAX
    }
}

impl IntoRange for RangeFrom<i64> {
    fn into_range(self) -> Range<i64> {
        self.start..i64::MAX
    }
}

impl IntoRange for RangeTo<i64> {
    fn into_range(self) -> Range<i64> {
        0..self.end
    }
}

impl IntoRange for RangeInclusive<i64> {
    fn into_range(self) -> Range<i64> {
        *self.start()..*self.end() + 1
    }
}

impl IntoRange for RangeToInclusive<i64> {
    fn into_range(self) -> Range<i64> {
        0..self.end + 1
    }
}

impl IntoRange for Range<i64> {
    fn into_range(self) -> Range<i64> {
        self
    }
}

impl IntoRange for i64 {
    fn into_range(self) -> Range<i64> {
        self..self + 1
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

impl<I0: IntoRange> IntoIndex for I0 {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [self.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange> IntoIndex for (I0, I1) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [self.0.into_range(), self.1.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange> IntoIndex for (I0, I1, I2) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
        ]
        .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange> IntoIndex for (I0, I1, I2, I3) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
        ]
        .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange> IntoIndex
    for (I0, I1, I2, I3, I4)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
        ]
        .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange, I5: IntoRange>
    IntoIndex for (I0, I1, I2, I3, I4, I5)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
        ]
        .into_iter()
    }
}

impl<
        I0: IntoRange,
        I1: IntoRange,
        I2: IntoRange,
        I3: IntoRange,
        I4: IntoRange,
        I5: IntoRange,
        I6: IntoRange,
    > IntoIndex for (I0, I1, I2, I3, I4, I5, I6)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
            self.6.into_range(),
        ]
        .into_iter()
    }
}

impl<
        I0: IntoRange,
        I1: IntoRange,
        I2: IntoRange,
        I3: IntoRange,
        I4: IntoRange,
        I5: IntoRange,
        I6: IntoRange,
        I7: IntoRange,
    > IntoIndex for (I0, I1, I2, I3, I4, I5, I6, I7)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
            self.6.into_range(),
            self.7.into_range(),
        ]
        .into_iter()
    }
}

/// A range of axes that can be used for flattening tensors.
pub trait FlattenAxes {
    /// Get flatten axes
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64>;
}

impl FlattenAxes for RangeFrom<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item=i64> {
        debug_assert!(if self.start > 0 { (self.start as usize) < rank } else { ((-self.start) as usize) <= rank }, "Cannot use {self:?} as flatten axes.");
        self.start..i64::MAX
    }
}

impl FlattenAxes for RangeTo<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item=i64> {
        debug_assert!(if self.end > 0 { (self.end as usize) < rank } else { ((-self.end) as usize) <= rank }, "Cannot use {self:?} as flatten axes.");
        0..self.end
    }
}

impl FlattenAxes for RangeToInclusive<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item=i64> {
        debug_assert!(if self.end > 0 { (self.end as usize) < rank } else { ((-self.end) as usize) <= rank }, "Cannot use {self:?} as flatten axes.");
        0..self.end + 1
    }
}

impl FlattenAxes for RangeFull {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item=i64> {
        0..rank as i64
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
        //std::println!("Dropping tensor {}", self.id);
        self.backend.release(self.id).unwrap();
    }
}

impl<B: Backend> core::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
        //f.write_fmt(format_args!("Tensor {{ id = {:?} }}", self.id))
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
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "f32 tensor failed to realize".into()
                }
            }
            DType::F64 => {
                if let Ok(data) = &self.to_vec::<f64>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "f64 tensor failed to realize".into()
                }
            }
            DType::I32 => {
                if let Ok(data) = &self.to_vec::<i32>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "i32 tensor failed to realize".into()
                }
            }
        };
        f.write_fmt(format_args!("Tensor {} {}\n{res}", self.shape(), self.dtype()))
    }
}

fn tensor_to_string<T: core::fmt::Display>(
    data: &[T],
    shape: &Shape,
    precision: usize,
    width: Option<usize>,
) -> alloc::string::String {
    use core::fmt::Write;
    let n = shape.numel();
    let ndim = shape.rank();
    let mut res = alloc::string::String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let mut w = 0;
    if let Some(width) = width {
        w = width;
    } else {
        for x in data {
            let l = alloc::format!("{x:>.precision$}").len();
            if l > w {
                w = l;
            }
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
    /// All tensors on one backend will always have different ids.
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

    /// Detach gradient tape from tensor.
    /// This means that resulting tensor is a shallow copy of self,
    /// but it's gradient will be ones. Result of this operation
    /// can only be differentiated by itself.
    /// ```rust
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([2, 3]);
    /// let y = &x + &x;
    /// let g = y.backward([&x]).pop().unwrap().unwrap();
    /// assert_eq!(g, [2, 2]);
    /// let z = y.detach();
    /// let g = z.backward([&z]).pop().unwrap().unwrap();
    /// assert_eq!(g, [1, 1]);
    /// let g = z.backward([&x]).pop().unwrap();
    /// assert_eq!(g, None);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn detach(&self) -> Tensor<B> {
        // It should be possible to just be optimize this away.
        tensor(self.backend.push(Node::Detach(self.id)).unwrap(), self.backend)
    }

    /*
    /// Probably just add no_grad, that is all tensors coming from no_grad tensor
    /// are not differentiable, unless some other parameter in those ops is differentiable.
    #[must_use]
    pub fn no_grad(&self) {
        // TODO
        //self.backend.no_grad(self.id);
    }*/

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
    /// let x = dev.tensor([3., 2., 1.]);
    /// let y = x.exp() + &x;
    /// let x_grad = y.backward([&x]).into_iter().next().unwrap().unwrap();
    /// assert_eq!(x_grad, [21.0855369, 8.3890561, 3.7182818]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
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
            .map(move |x| x.map(|x| tensor(x, self.backend)))
            .collect()
    }

    // Unary ops
    /// Cast self into dtype.
    /// ```rust
    /// # use zyx_opencl::DType;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[3, 4, 2], [4, 5, 2]]);
    /// let y = x.cast(DType::F32);
    /// assert_eq!(y.dtype(), DType::F32);
    /// assert_eq!(y, [[3f32, 4., 2.], [4., 5., 2.]]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor<B> {
        tensor(self.backend.push(Node::Cast(self.id, dtype)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the rectified linear unit function applied to the elements of self.
    #[must_use]
    pub fn relu(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::ReLU(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the sine of the elements of self.
    #[must_use]
    pub fn sin(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Sin(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the cosine of the elements of self.
    #[must_use]
    pub fn cos(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Cos(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the natural logarithm of the elements of self.
    /// Due to performance reasons, this function does not check if self fits
    /// into domain of ln(x). Result on out of domain numbers is implementation
    /// defined (when x <= 0).
    #[must_use]
    pub fn ln(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Ln(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the exponential of the elements of self.
    #[must_use]
    pub fn exp(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Exp(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the hyperbolic tangent of the elements of self.
    #[must_use]
    pub fn tanh(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Tanh(self.id)).unwrap(), self.backend)
    }

    /// Returns a new tensor with the square root of the elements of self.
    /// Due to performance reasons, this function does not check if self fits
    /// into domain of ln(x). Result on out of domain numbers is implementation
    /// defined (when x < 0).
    #[must_use]
    pub fn sqrt(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Sqrt(self.id)).unwrap(), self.backend)
    }

    /// Returns 1/self
    #[must_use]
    pub fn reciprocal(&self) -> Tensor<B> {
        self.backend().ones(self.shape(), self.dtype()).unwrap() / self
    }

    /// Returns 1/self.sqrt()
    #[must_use]
    pub fn rsqrt(&self) -> Tensor<B> {
        self.reciprocal().sqrt()
    }

    /// Returns a new tensor with each element of self randomly zeroed with given probability.
    #[must_use]
    pub fn dropout(&self, probability: impl Scalar) -> Tensor<B> {
        self.backend()
            .tensor(probability).unwrap()
            .cmplt(self.backend().uniform(self.shape(), 0.0..1.0).unwrap())
            * self
    }

    /// Returns a new tensor with the absolute value of the elements of self.
    #[must_use]
    pub fn abs(&self) -> Tensor<B> {
        self.relu() + (-self).relu()
    }

    /// Returns a new tensor with the sigmoid (logistic function) of the elements of self.
    #[must_use]
    pub fn sigmoid(&self) -> Tensor<B> {
        let one = self.backend().ones(1, self.dtype()).unwrap();
        &one / (&one + (-self).exp())
    }

    /// Returns a new tensor with the swish/silu of the elements of self.
    #[must_use]
    pub fn swish(&self) -> Tensor<B> {
        self * self.sigmoid()
    }

    /// Returns a new tensor with the mish of the elements of self.
    #[must_use]
    pub fn mish(&self) -> Tensor<B> {
        self * self.softplus(1, 20).tanh()
    }

    /// Returns a new tensor with the softplus of the elements of self.
    #[must_use]
    pub fn softplus(&self, beta: impl Scalar, threshold: impl Scalar) -> Tensor<B> {
        let x = self * beta.clone();
        x.cmplt(threshold)
            .where_(((x).exp() + 1).ln() * beta.reciprocal(), x)
    }

    /// Returns a new tensor with the tangent of the elements of self.
    #[must_use]
    pub fn tan(&self) -> Tensor<B> {
        self.sin() / self.cos()
    }

    /// Returns a new tensor with the leaky relu of the elements of self.
    #[must_use]
    pub fn leaky_relu(&self, neg_slope: impl Scalar) -> Tensor<B> {
        self.relu() - (self * (-self.backend.tensor(neg_slope).unwrap())).relu()
    }

    /// Returns a new tensor with the elu of the elements of self.
    #[must_use]
    pub fn elu(&self, alpha: impl Scalar) -> Tensor<B> {
        self.relu() - (1f32.into_tensor(self.backend) - self.exp()).relu() * alpha
    }

    /// Returns a new tensor with the selu of the elements of self.
    #[must_use]
    pub fn selu(&self) -> Tensor<B> {
        1.0507009873554804934193349852946f32
            * (self.relu()
                - (1.6732632423543772848170429916717f32
                    * (self.backend.ones(1, self.dtype()).unwrap() - self.exp()))
                .relu())
    }

    /// Returns a new tensor with the celu of the elements of self.
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor<B> {
        self.relu()
            - ((self.backend.ones(1, self.dtype()).unwrap() - (self / alpha.clone()).exp()) * alpha).relu()
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
    #[must_use]
    pub fn quick_gelu(&self) -> Tensor<B> {
        self * (1.702f32 * self).sigmoid()
    }

    /// Returns a new tensor with the softmax of the elements of self.
    #[must_use]
    pub fn softmax(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let e = (self - self.max(axes.clone())).exp();
        &e / e.sum(axes)
    }

    /// Returns a new tensor with the log softmax of the elements of self.
    #[must_use]
    pub fn ln_softmax(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let m = self - self.max(axes.clone());
        &m - m.exp().sum(axes).ln()
    }

    // Loss functions, all losses are without reduce
    /// Measures the mean absolute error (MAE) between each element in the input self and target.
    #[must_use]
    pub fn l1_loss(&self, target: impl IntoTensor<B>) -> Tensor<B> {
        (self - target).abs()
    }

    /// Measures the mean squared error (MSE) between each element in the input self and target.
    #[must_use]
    pub fn mse_loss(&self, target: impl IntoTensor<B>) -> Tensor<B> {
        (self - target).pow(2)
    }

    /// Computes the cross entropy loss between self logits and target.
    /// This function expects self to contain probabilities for each class.
    #[must_use]
    pub fn cross_entropy_loss(&self, target: impl IntoTensor<B>, axes: impl IntoAxes) -> Tensor<B> {
        self.ln_softmax(axes) * target
    }

    // Binary ops
    /// Exponentiation on self
    #[must_use]
    pub fn pow(&self, exponent: impl IntoTensor<B>) -> Tensor<B> {
        let exponent = self.backend.tensor(exponent).unwrap();
        if exponent.numel() == 1 {
            let dtype = exponent.dtype();
            if !dtype.is_floating() {
                // TODO other int dtypes
                if exponent.item::<i32>().unwrap() == 2i32 {
                    return self * self
                } else if exponent.item::<i32>().unwrap() == 3i32 {
                    return self * self * self
                }
            }
        }
        if self.dtype().is_floating() {
            return (exponent * self.ln()).exp()
        }
        self.clone().binary_op(exponent, BOp::Pow)
    }

    /// Elementwise compare less than between self and rhs
    #[must_use]
    pub fn cmplt(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        self.clone().binary_op(rhs, BOp::Cmplt)
    }

    /// Returns a new tensor with the true values replaced with if_true and the false values replaced with if_false.
    #[must_use]
    pub fn where_(&self, if_true: impl IntoTensor<B>, if_false: impl IntoTensor<B>) -> Tensor<B> {
        let x = self.clone();
        let y = self.backend.tensor(if_true).unwrap();
        let z = self.backend.tensor(if_false).unwrap();
        let (x, y) = Tensor::broadcast(x, y);
        let (x, z) = Tensor::broadcast(x, z);
        let (y, z) = Tensor::broadcast(y, z);
        tensor(
            self.backend.push(Node::Where(x.id, y.id, z.id)).unwrap(),
            self.backend,
        )
    }

    /// Returns cosine_similarity between self and rhs, computed along axes.
    #[must_use]
    pub fn cosine_similarity(&self, rhs: impl IntoTensor<B>, eps: impl IntoTensor<B>) -> Tensor<B> {
        let rhs = self.backend.tensor(rhs).unwrap();
        let eps = self.backend.tensor(eps).unwrap();
        let x = self.pow(2).sqrt() * rhs.pow(2).sqrt();
        self * rhs / x.cmplt(&eps).where_(eps, x)
    }

    /// Dot product (mathematical multiplication) of self and rhs.
    /// ```rust
    /// # use zyx_opencl::DType;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[3, 4, 2], [4, 5, 2]]);
    /// let y = dev.tensor([[3], [1], [4]]);
    /// assert_eq!(x.dot(y), [[21], [25]]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn dot(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        let y = self.backend.tensor(rhs).unwrap().transpose();
        let xshape = self.shape();
        let yshape = y.shape();
        let yrank = yshape.rank();
        debug_assert_eq!(
            xshape[-1], yshape[-1],
            //yshape[-(yrank.min(2) as i64)],
            "Cannot dot tensors with shapes {xshape} and {yshape}"
        );
        let x_shape = xshape[0..-1]
            .iter()
            .copied()
            .chain([1])
            .chain([xshape[-1]])
            .collect::<Box<[usize]>>();
        let y_shape = yshape[0..-2]
            .iter()
            .copied()
            .chain([1])
            .chain(yshape[-(yrank.min(2) as i64)..yrank as i64].iter().copied())
            .collect::<Box<[usize]>>();
        //std::println!("{x_shape:?}");
        //std::println!("{y_shape:?}");
        (self.reshape(x_shape) * y.reshape(y_shape))
            .sum(-1)
            .reshape(
                xshape[0..-1]
                    .iter()
                    .copied()
                    .chain([yshape[-2]])
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
        debug_assert_eq!(
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
        let shape = shape.into();
        let sh = self.shape();
        debug_assert!(
            shape
                .iter()
                .rev()
                .enumerate()
                .all(|(i, d)| if sh.rank() > i {
                    *d == sh[sh.rank() - i - 1] || sh[sh.rank() - i - 1] == 1
                } else {
                    true
                }),
            "Can't expand tensor with shape {sh} to {shape}"
        );
        tensor(
            self.backend.push(Node::Expand(self.id, shape)).unwrap(),
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
        value: impl IntoTensor<B>,
    ) -> Tensor<B> {
        let dtype = self.dtype();
        let value = self.backend.tensor(value).unwrap();
        debug_assert_eq!(
            value.dtype(),
            dtype,
            "Cannot pad tensor with dtype {} with value of dtype {}",
            dtype,
            value.dtype()
        );
        let padding: Box<[(i64, i64)]> = padding.into_iter().collect();
        let sh = self.shape();
        debug_assert!(
            padding.len() <= sh.rank()
                && padding
                    .iter()
                    .zip(sh.iter().rev())
                    .all(|((lp, rp), d)|
                         if *lp < 0 {
                             ((-*lp) as usize) <= *d
                         } else {
                             true
                         } &&
                         if *rp < 0 {
                             ((-*rp) as usize) <= *d
                         } else {
                             true
                         }),
            "Cannot pad tensor with shape {sh} with padding {padding:?}"
        );
        let psh = sh.clone().pad(&padding);
        let t0 = tensor(
            self.backend
                .push(Node::Pad(self.id, padding.clone(), psh.clone()))
                .unwrap(),
            self.backend,
        );
        if value.numel() == 1
            && match dtype {
                DType::F32 => value.item::<f32>().unwrap().is_equal(0f32),
                DType::F64 => value.item::<f64>().unwrap().is_equal(0f64),
                DType::I32 => value.item::<i32>().unwrap().is_equal(0i32),
            }
        {
            t0
        } else {
            t0 + tensor(
                self.backend
                    .push(Node::Pad(
                        self.backend.ones(sh, dtype).unwrap().id,
                        padding,
                        psh.clone(),
                    ))
                    .unwrap(),
                self.backend,
            )
            .where_(self.backend.zeros(self.shape(), self.dtype()).unwrap(), value)
        }
    }

    /// Reorder axes of self
    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().permute(&axes);
        debug_assert!(
            axes.len() == shape.rank(),
            "Cannot permute tensor with shape {shape} with axes {axes}"
        );
        tensor(
            self.backend
                .push(Node::Permute(self.id, axes, shape))
                .unwrap(),
            self.backend,
        )
    }

    /// Swap last two axes of self.
    /// If self has rank == 1 and numel == n, then result will have shape /[n, 1/]
    #[must_use]
    pub fn transpose(&self) -> Tensor<B> {
        let mut rank = self.rank();
        let x = if rank == 1 {
            let n = self.numel();
            rank = 2;
            self.reshape([1, n])
        } else {
            self.clone()
        };
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        x.permute(axes)
    }

    /// Flatten. Joins axes into one dimension,
    #[must_use]
    pub fn flatten(&self, axes: impl FlattenAxes) -> Tensor<B> {
        let sh = self.shape();
        let n = sh.numel();
        let rank = sh.rank();
        let mut ld = 1;
        let mut first_dims = false;
        for a in axes.into_flatten_axes(rank) {
            let a = if a > 0 {
                a as usize
            } else {
                (a + rank as i64) as usize
            };
            if a == 0 {
                first_dims = true;
            }
            ld *= sh[a];
        }
        if first_dims {
            self.reshape([ld, n / ld])
        } else {
            self.reshape([n / ld, ld])
        }
    }

    // Reduce ops
    /// Reduce self by summing along axes. Shape is not squeezed.
    /// ```rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3], [4, 1]]);
    /// let z = x.sum(-1);
    /// assert_eq!(z.shape(), [2, 1]);
    /// let z = x.sum(0);
    /// assert_eq!(z.shape(), [1, 2]);
    /// let z = x.sum(..);
    /// assert_eq!(z.shape(), [1, 1]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        let mut uniq = BTreeSet::new();
        debug_assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot sum tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        tensor(
            self.backend.push(Node::Sum(self.id, axes, shape)).unwrap(),
            self.backend,
        )
    }

    /// Reduce self by maximizing along axes. Shape is not squeezed.
    /// ```rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3], [4, 1]]);
    /// let z = x.max(-1);
    /// assert_eq!(z.shape(), [2, 1]);
    /// let z = x.max(0);
    /// assert_eq!(z.shape(), [1, 2]);
    /// let z = x.max(..);
    /// assert_eq!(z.shape(), [1, 1]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor<B> {
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().reduce(&axes);
        let mut uniq = BTreeSet::new();
        debug_assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot sum tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        for a in &axes {
            debug_assert!(
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

    /// Reduce self by calculating norm along axes
    #[must_use]
    pub fn norm(&self, axes: impl IntoAxes, p: impl Scalar) -> Tensor<B> {
        self.pow(p.clone()).sum(axes).pow(p.reciprocal())
    }

    /// Reduce self by calculating product of elements along axes
    #[must_use]
    pub fn product(&self, axes: impl IntoAxes) -> Tensor<B> {
        self.ln().sum(axes).exp()
    }

    /// Get elements on diagonal of square matrix
    #[must_use]
    pub fn diagonal(&self) -> Tensor<B> {
        let n: usize = self.shape()[-1];
        self.flatten(..).pad([(0, n as i64)], 0).reshape([n, n+1]).get((.., 0))
    }

    /*
    /// QR decompose function
    #[must_use]
    fn qr_decompose(&self) -> (Tensor<B>, Tensor<B>) {
        assert_eq!(self.rank(), 2, "QR decomposition only works for 2d matrices.");
        let dtype = self.dtype();
        assert!(dtype.is_floating(), "QR decomposition only works with floating point tensors.");
        let [n, m] = self.shape().try_into().unwrap();
        let u_temp = self.get((.., 0));
        let mut q = Vec::new();
        q.push(u_temp / u_temp.norm(()));
        for i in 1..n {
            let mut u_temp = self.get((.., i));
            // TODO all those dot operations should be fused into one by using expand and reshape and such.
            for j in 0..i {
                let q_temp = q.get((.., j));
                u_temp = u_temp - self.get((.., i)).dot(&q_temp) * &q_temp;
            }
            q.push(u_temp / u_temp.norm(.., 2));
        }
        let q = Tensor::cat(q, 0);
        let r = q.dot(self);
        return (q, r)
    }*/

    /// Tensor indexing.
    ///
    /// Tensors can be indexed by tuples of any combination of values or ranges of i64.
    /// If indexing along more than 8 dimensions, use \&\[Range\<i64\>\] or \&\[i64\]
    /// ```rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device()?;
    /// let x = dev.tensor([[2, 3, 4],
    ///                     [4, 1, 8]]);
    /// let y: i32 = x.get((-1, -3)).item()?;
    /// assert_eq!(y, 4);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    #[must_use]
    pub fn get(&self, index: impl IntoIndex) -> Tensor<B> {
        // TODO asserts
        let shape = self.shape();
        let padding: Vec<(i64, i64)> = index
            .into_index()
            .into_iter()
            .zip(shape.iter())
            .map(|(r, d)| {
                (
                    if r.start >= 0 {
                        -r.start
                    } else {
                        -r.start - *d as i64
                    },
                    if r.end == i64::MAX {
                        0
                    } else if r.end > 0 {
                        -(*d as i64 - r.end)
                    } else {
                        r.end
                    },
                )
            })
            .collect();
        //std::println!("Get padding: {padding:?}");
        let n = shape.rank() - padding.len();
        self.pad(
            padding
                .into_iter()
                .chain(repeat((0, 0)).take(n))
                .collect::<Vec<(i64, i64)>>()
                .into_iter()
                .rev(),
            0,
        )
    }

    /// Concatenate multiple tensors together along dim.
    // ```rust
    // # use zyx_opencl;
    // # use zyx_opencl::Tensor;
    // let dev = zyx_opencl::device()?;
    // let x = dev.tensor([[2, 3, 4], [4, 1, 8]]);
    // let y = dev.tensor([[2, 3], [4, 1]]);
    // let z = Tensor::cat([&x, &y], -1);
    // // assert_eq!(z, []);
    // # Ok::<(), zyx_opencl::ZyxError>(())
    // ```
    #[must_use]
    pub fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor<B>>, dim: i64) -> Tensor<B>
    where
        B: 'a,
    {
        let tensors: Vec<&Tensor<B>> = tensors.into_iter().collect();
        let shape = tensors[0].shape();
        let rank = shape.rank();
        let dim = if dim < 0 { dim + rank as i64 } else { dim } as usize;
        // Dimension check
        for tensor in &tensors {
            for (i, (d1, d2)) in shape.iter().zip(tensor.shape().iter()).enumerate() {
                if i != dim {
                    debug_assert_eq!(*d1, *d2, "Cannot concatenate these tensors.");
                }
            }
        }
        let mut offset = 0i64;
        let mut res = tensors[0]
            .backend
            .zeros(tensors[0].shape(), tensors[0].dtype()).unwrap();
        for tensor in tensors {
            res = res
                + tensor.pad(
                    repeat((0i64, 0i64))
                        .take(rank - dim - 1)
                        .chain([(offset, 0i64)]),
                    0,
                );
            offset += tensor.shape()[dim] as i64;
        }
        res
    }

    // TODO Cholesky and QR solve functions that are backend accelerated

    /*
    /// Stack multiple tensors into one
    #[must_use]
    pub fn stack<'a>(tensors: impl IntoIterator<Item = &'a Tensor<B>>, dim: i64) -> Tensor<B>
    where
        B: 'a
    {
        todo!()
    }*/

    /*
    /// Split self into multiple tensors along dim with given sizes.
    // TODO example
    #[must_use]
    pub fn split(&self, sizes: &[usize], dim: i64) -> Vec<Tensor<B>> {
        // just use negative padding
        todo!()
    }*/

    //#[must_use]
    //pub fn pool(&self)

    //#[must_use]
    //pub fn conv(&self)
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
    fn binary_op(self, rhs: impl IntoTensor<B>, op: BOp) -> Tensor<B> {
        let rhs = rhs.into_tensor(self.backend);
        let (x, y) = Tensor::broadcast(self, rhs);
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

    /// Braodcasts to synchronize shapes and casts to synchronize dtypss
    /// This does both automatic expand AND automatic casting between dtypes.
    // TODO Both of these can be disable by changing a setting in the backend.
    #[must_use]
    fn broadcast(mut x: Tensor<B>, mut y: Tensor<B>) -> (Tensor<B>, Tensor<B>) {
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
            (DType::F32, DType::F64) => x = x.cast(DType::F64),
            (DType::I32, DType::F32) => x = x.cast(DType::F32),
            (DType::I32, DType::F64) => x = x.cast(DType::F64),
            (DType::F64, DType::F32) => y = y.cast(DType::F64),
            (DType::F64, DType::I32) => y = y.cast(DType::F64),
            _ => {}
        }
        let mut x_shape = x.shape();
        let mut y_shape = y.shape();

        for (x, y) in x_shape.iter().rev().zip(y_shape.iter().rev()) {
            if x != y {
                debug_assert!(
                    *x == 1 || *y == 1,
                    "Left and right tensor shapes can not be broadcasted: {x_shape} and {y_shape}"
                );
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
        (x, y)
    }
}

impl<B: Backend> core::ops::Neg for Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        tensor(self.backend.push(Node::Neg(self.id)).unwrap(), self.backend)
    }
}

impl<B: Backend> core::ops::Neg for &Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        tensor(self.backend.push(Node::Neg(self.id)).unwrap(), self.backend)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Add<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: IT) -> Self::Output {
        self.clone().binary_op(rhs, BOp::Add)
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
        self.clone().binary_op(rhs, BOp::Sub)
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
        self.clone().binary_op(rhs, BOp::Mul)
    }
}

impl<B: Backend> core::ops::Mul<Tensor<B>> for f32 {
    type Output = Tensor<B>;
    fn mul(self, rhs: Tensor<B>) -> Self::Output {
        rhs * self
    }
}

impl<B: Backend> core::ops::Mul<&Tensor<B>> for f32 {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Self::Output {
        rhs * self
    }
}

impl<B: Backend> core::ops::Mul<Tensor<B>> for f64 {
    type Output = Tensor<B>;
    fn mul(self, rhs: Tensor<B>) -> Self::Output {
        rhs * self
    }
}

impl<B: Backend> core::ops::Mul<&Tensor<B>> for f64 {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Self::Output {
        rhs * self
    }
}

impl<B: Backend> core::ops::Mul<Tensor<B>> for i32 {
    type Output = Tensor<B>;
    fn mul(self, rhs: Tensor<B>) -> Self::Output {
        rhs * self
    }
}

impl<B: Backend> core::ops::Mul<&Tensor<B>> for i32 {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Self::Output {
        rhs * self
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
        self.clone().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Div<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: IT) -> Self::Output {
        self.binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<Tensor<B>> for f32 {
    type Output = Tensor<B>;
    fn div(self, rhs: Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<&Tensor<B>> for f32 {
    type Output = Tensor<B>;
    fn div(self, rhs: &Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<Tensor<B>> for f64 {
    type Output = Tensor<B>;
    fn div(self, rhs: Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<&Tensor<B>> for f64 {
    type Output = Tensor<B>;
    fn div(self, rhs: &Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<Tensor<B>> for i32 {
    type Output = Tensor<B>;
    fn div(self, rhs: Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
    }
}

impl<B: Backend> core::ops::Div<&Tensor<B>> for i32 {
    type Output = Tensor<B>;
    fn div(self, rhs: &Tensor<B>) -> Self::Output {
        rhs.backend.tensor(self).unwrap().binary_op(rhs, BOp::Div)
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

impl<B: Backend, T: Scalar> IntoTensor<B> for Range<T>
where
    Range<T>: Iterator<Item = T> + ExactSizeIterator,
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self).unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for Vec<T> {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self).unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for &'static [T] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self.iter().cloned()).unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar> IntoTensor<B> for T {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend
                .store( [self])
                .unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize> IntoTensor<B> for [T; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self).unwrap(),
            backend,
        )
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize> IntoTensor<B> for [[T; D1]; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self.into_iter().flatten().make_sized(D0*D1)).unwrap(),
            backend,
        ).reshape([D0, D1])
    }
}

impl<B: Backend, T: Scalar, const D0: usize, const D1: usize, const D2: usize> IntoTensor<B>
    for [[[T; D2]; D1]; D0]
{
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(
            backend.store(self.into_iter().flatten().flatten().make_sized(D0*D1*D2)).unwrap(),
            backend,
        ).reshape([D0, D1, D2])
    }
}

impl<B: Backend, IT: IntoTensor<B> + Clone> PartialEq<IT> for Tensor<B> {
    fn eq(&self, other: &IT) -> bool {
        let other = self.backend.tensor(other.clone()).unwrap();
        let dtype = self.dtype();
        self.shape() == other.shape()
            && dtype == other.dtype()
            && match dtype {
                DType::F32 => self
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .zip(other.to_vec::<f32>().unwrap())
                    .all(|(x, y)| x.is_equal(y)),
                DType::F64 => self
                    .to_vec::<f64>()
                    .unwrap()
                    .into_iter()
                    .zip(other.to_vec::<f64>().unwrap())
                    .all(|(x, y)| x.is_equal(y)),
                DType::I32 => self
                    .to_vec::<i32>()
                    .unwrap()
                    .into_iter()
                    .zip(other.to_vec::<i32>().unwrap())
                    .all(|(x, y)| x.is_equal(y)),
            }
    }
}
