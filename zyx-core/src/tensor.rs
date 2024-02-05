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
        f.write_fmt(format_args!("{}", self.0))
    }
}

impl SubAssign<usize> for Id {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

/// Implemented for objects that can be used to index tensors.
pub trait IntoIndex {
    /// Convert self to tensor index.
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>>;
}

impl IntoIndex for &[Range<i64>] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        self.iter().cloned()
    }
}

impl<const N: usize> IntoIndex for [Range<i64>; N] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        self.into_iter()
    }
}

impl IntoIndex for &[i64] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        // TODO this is incorrect, what about index -1?
        self.iter().copied().map(|i| i..i + 1)
    }
}

impl<const N: usize> IntoIndex for [i64; N] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        // TODO this is incorrect, what about index -1?
        self.into_iter().map(|i| i..i + 1)
    }
}

/// Tensor is the atom of zyx.
/// Tensor is multidimensional array.
/// Tensor is immutable (with one [exception](Tensor::set)
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
    ) -> impl Iterator<Item = Option<Tensor<B>>> + 'a
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

    /// Dot product (mathematical multiplication) of self and rhs
    #[must_use]
    pub fn dot(&self, rhs: impl IntoTensor<B>) -> Tensor<B> {
        let y = rhs.into_tensor(self.backend);
        let xshape = self.shape();
        let yshape = y.shape();
        let yrank = yshape.rank();
        assert_eq!(xshape[-1], yshape[-(yrank.min(2) as i64)]);
        (self.reshape(
            xshape[0..-1]
                .iter()
                .copied()
                .chain([1])
                .chain([xshape[-1]])
                .collect::<Vec<usize>>(),
        ) * y
            .reshape(
                yshape[0..-2]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain([yshape[-(yrank.min(2) as i64)]])
                    .collect::<Vec<usize>>(),
            )
            .transpose())
        .sum(-1)
    }

    // Movement ops
    /// Reshape self to shape.
    /// # Panics
    /// Following must hold:
    /// self.numel() == shape.numel()
    #[must_use]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor<B> {
        let shape = shape.into();
        assert_eq!(self.shape().numel(), shape.numel());
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
    /// # use zyx_opencl;
    /// // let dev = zyx_opencl::device()?;
    /// let dev = zyx_opencl::device_builder()
    ///     .platform_id(1)
    ///     .build()?;
    /// let x = dev.tensor([[2, 3],
    ///                     [4, 1]]);
    /// let z = x.pad([(1, 2)], 0);
    /// std::println!("{:?}", z.shape());
    /// std::println!("{}", z);
    /// assert_eq!(z, [[0, 2, 3, 0, 0],
    ///                [0, 4, 1, 0, 0]]);
    /// # Ok::<(), zyx_opencl::ZyxError>(())
    /// ```
    /// Pad last dimension by (2, -1) and second last dimension by (1, 1)
    /// rust
    /// use zyx_opencl;
    /// let dev = zyx_opencl::device().unwrap();
    /// let x = dev.tensor([[2, 3],
    ///                     [4, 1]]);
    /// let z = x.pad([(2, -1), (1, 1)], 7);
    /// assert_eq!(z, [[7, 7, 7],
    ///                [7, 7, 2],
    ///                [7, 7, 4],
    ///                [7, 7, 7]]);
    ///
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
        assert_eq!(get_dtype(value.clone()), self.dtype());
        // TODO asserts
        // TODO add support for value
        let padding: Box<[(i64, i64)]> = padding.into_iter().collect();
        let sh = self.shape().pad(&padding);
        tensor(
            self.backend.push(Node::Pad(self.id, padding, sh)).unwrap(),
            self.backend,
        )
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
    pub fn get(&self, index: impl IntoIndex) -> Tensor<B> {
        let shape = self.shape();
        // TODO asserts
        self.pad(
            index
                .into_index()
                .into_iter()
                .enumerate()
                .map(|(a, r)| (-r.start, shape[-(a as i64)] as i64 - r.end)),
            0,
        )
    }

    /// Tensors should be treated as immutable data structures.
    /// However optimizers need to update model's parameters with new values.
    /// This is the only function that mutates tensor.
    pub fn set(&mut self, x: impl IntoTensor<B>) {
        self.backend.release(self.id).unwrap();
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

impl<B: Backend, T: Scalar + core::cmp::PartialEq, const D0: usize> PartialEq<[T; D0]>
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

impl<B: Backend, T: Scalar + core::cmp::PartialEq, const D0: usize, const D1: usize>
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
        T: Scalar + core::cmp::PartialEq,
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
