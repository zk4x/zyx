//! # Tensor
//!
//! See [Tensor].

// TODO maybe replace panics with results

extern crate alloc;
use alloc::{collections::BTreeSet, format, string::String, vec::Vec};

use crate::{
    node_id::NodeId,
    axes::IntoAxes,
    dtype::DType,
    graph::{Graph, Node},
    parameters::IntoParameters,
    shape::Shape,
    OutOfMemoryError,
};
use rclite::Rc;
use core::{cell::RefCell, cmp::Ordering};

// Precision when comparing f32 tensors.
const EPSILON: f32 = 0.00001;

/// # `IntoTensor`
/// Trait for converting values into tensor.
#[allow(clippy::module_name_repetitions)]
pub trait IntoTensor {
    /// Create tensor from self on given context.
    fn into_tensor(self, ctx: &crate::context::Context) -> Tensor;
}

/// # Tensor
///
/// Multidimensional data structure.
/// It can be single value,
/// ```rust
/// # use zyx::context::Context;
/// let mut ctx = Context::new();
/// let mut x = ctx.tensor(42);
/// x.realize();
/// assert_eq!(x, 42);
/// ```
/// vector,
/// ```rust
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// let mut x = ctx.tensor([42, 69, 10]);
/// x.realize();
/// assert_eq!(x, [42, 69, 10]);
/// ```
/// matrix,
/// ```rust
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// let mut x = ctx.tensor([[42, 69, 10], [24, 96, 1]]);
/// x.realize();
/// assert_eq!(x, [[42, 69, 10], [24, 96, 1]]);
/// ```
/// or it can have arbitrary number of dimensions.
/// ```rust
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// let mut x = ctx.randn((4, 2, 3, 1, 5));
/// assert_eq!(x.shape(), (4, 2, 3, 1, 5));
/// ```
/// You can call operations on tensors
/// ```rust
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// # let x = ctx.randn((4, 2, 3, 1, 5));
/// let y = x.exp();
/// ```
/// and calculate their gradients.
/// ```rust
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// # let mut x = ctx.randn((4, 2, 3, 1, 5));
/// # let y = x.exp();
/// y.backward(&mut x); // calculates gradient for x
/// ```
/// Tensors need to be realized before accessing.
/// ```rust
/// # use zyx::context::Context;
/// # use zyx::parameters::IntoParameters;
/// # let mut ctx = Context::new();
/// # let mut x = ctx.randn((4, 2, 3, 1, 5));
/// # let y = x.exp();
/// # y.backward(&mut x); // calculates gradient for x
/// x.realize_grad().unwrap();
/// println!("{}", x.grad().unwrap());
/// ```
#[derive(Debug)]
pub struct Tensor {
    pub(crate) data: NodeId,
    pub(crate) grad: Option<NodeId>,
    pub(crate) graph: Rc<RefCell<Graph>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.graph.borrow_mut().retain(self.data);
        if let Some(grad) = self.grad {
            self.graph.borrow_mut().retain(grad);
        }
        Self {
            data: self.data,
            grad: self.grad,
            graph: self.graph.clone(),
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let mut graph = self.graph.borrow_mut();
        graph.release(self.data);
        if let Some(grad) = self.grad {
            graph.release(grad);
        }
    }
}

impl Tensor {
    /// # Backward
    ///
    /// Calculates gradients for all sources.
    ///```
    /// # use zyx::context::Context;
    /// # use zyx::parameters::IntoParameters;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2, 3, 4]);
    /// let y = -&x;
    /// y.backward(&mut x);
    /// x.realize_grad().unwrap();
    /// assert_eq!(x.grad().unwrap(), [-1, -1, -1]);
    ///```
    #[allow(single_use_lifetimes)]
    pub fn backward<'p>(&self, sources: impl IntoParameters<'p>) {
        self.graph.borrow_mut().backward(self.data, &mut sources.into_vec());
        // TODO Maybe label gradients automatically if sources have labels
    }

    /// Cast tensor into tensor with different [`DType`].
    /// ```
    /// # use zyx::context::Context;
    /// # use zyx::dtype::DType;
    /// # let mut ctx = Context::new();
    /// let x = ctx.tensor([2, 3, 4]);
    /// assert_eq!(x.dtype(), DType::I32);
    /// let y = x.cast(DType::F32);
    /// assert_eq!(y.dtype(), DType::F32);
    /// ```
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor {
        self.new_op(Node::Cast(self.data, dtype))
    }

    /// Get tensor's context
    #[must_use]
    pub fn context(&self) -> crate::context::Context {
        crate::context::Context::from_graph(self.graph.clone())
    }

    /// Access tensor's data. This is equivalent to cloning the tensor and zeroing it's gradient.
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2., 3., 4.]);
    /// x.exp().backward(&mut x); // creates gradient for x
    /// let y = x.data();
    /// assert!(y.grad().is_none());
    /// ```
    #[must_use]
    pub fn data(&self) -> Tensor {
        self.graph.borrow_mut().retain(self.data);
        Tensor {
            data: self.data,
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Matmul operation
    /// # Panics
    /// Panics if x and y tensors have incompatible shapes.
    #[must_use]
    pub fn dot(&self, rhs: impl IntoTensor) -> Tensor {
        // always returns 2d or higher tensor, even if both inputs are 1d
        let rhs = rhs.into_tensor(&self.graph.clone().into());
        let shape = self.shape().dot(&rhs.shape());
        let x = self.transpose();
        self.new_op(Node::TDot(x.data, rhs.data, shape))
    }

    /// Dropout op
    #[must_use]
    pub fn dropout(&self, prob: f32) -> Tensor {
        let seed = if prob == 0. {
            0
        } else {
            self.graph.borrow_mut().rand_u64()
        };
        self.new_op(Node::Dropout(self.data, seed, prob))
    }

    /// Matmul operation with xhs transposed
    /// # Panics
    /// Panics if x and y tensors have incompatible shapes.
    #[must_use]
    pub fn t_dot(&self, rhs: impl IntoTensor) -> Tensor {
        let rhs = rhs.into_tensor(&self.graph.clone().into());
        let shape = self.shape().transpose().dot(&rhs.shape());
        self.new_op(Node::TDot(self.data, rhs.data, shape))
    }

    /// Get tensor's dtype
    /// ```
    /// # use zyx::context::Context;
    /// # use zyx::dtype::DType;
    /// # let mut ctx = Context::new();
    /// let x = ctx.tensor([2, 3, 4]);
    /// assert_eq!(x.dtype(), DType::I32);
    /// ```
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.graph.borrow().dtype(self.data)
    }

    /// Exp operation
    #[must_use]
    pub fn exp(&self) -> Tensor {
        self.new_op(Node::Exp(self.data))
    }

    /// Expand tensor into larger shape
    #[must_use]
    pub fn expand(&self, shape: impl Into<Shape>) -> Tensor {
        // TODO checks
        self.new_op(Node::Expand(self.data, shape.into()))
    }

    /// Access tensor's gradient.
    /// This function returns None when gradient is None,
    /// i. e. if gradient wasn't set or if it was zeroed using [`zero_grad`](Tensor::zero_grad).
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2, 3, 4]);
    /// assert!(x.grad().is_none());
    /// ```
    #[must_use]
    pub fn grad(&self) -> Option<Tensor> {
        if let Some(grad) = self.grad {
            self.graph.borrow_mut().retain(grad);
            return Some(Tensor {
                data: grad,
                grad: None,
                graph: self.graph.clone(),
            });
        }
        None
    }

    /// Get tensor's id.
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2, 3, 4]);
    /// assert_eq!(x.id(), 0);
    /// ```
    #[must_use]
    pub fn id(&self) -> usize {
        self.data.i()
    }

    /// Get tensor's label
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2, 3, 4]);
    /// assert_eq!(x.label(), None);
    /// ```
    #[must_use]
    pub fn label(&self) -> Option<String> {
        self.graph.borrow().label(self.data).cloned()
    }
    
    /// Layer norm
    #[must_use]
    pub fn layer_norm(&self, axes: impl IntoAxes) -> Tensor {
        let eps = 0.00001;
        let x = self - self.mean(axes.clone());
        &x * ((&x*&x).mean(axes) + eps).rsqrt()
    }

    /// Ln operation
    /// Natural logarithm
    #[must_use]
    pub fn ln(&self) -> Tensor {
        self.new_op(Node::Ln(self.data))
    }

    /// Reduce tensor across axes, returning max in each axes.
    /// All axes in result are kept and set to one.
    /// # Panics
    /// Panics if axes contain axis larger than tensor's rank,
    /// or if axes contain duplicates.
    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        let rank = self.rank();
        let axes = axes.into_axes(rank);
        let mut uniq = BTreeSet::new();
        assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot max tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        for a in &axes {
            assert!(
                *a < rank,
                "Cannot max tensor with shape {:?} by axes {:?}, because some axes are greater than rank.",
                self.shape(),
                axes
            );
        }
        let shape = self.shape().reduce(&axes);
        self.new_op(Node::Max(self.data, axes, shape))
    }
    
    /// Mean op
    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor {
        match self.dtype() {
            DType::F32 => self.sum(axes)/self.shape().numel() as f32,
            DType::I32 => self.sum(axes)/self.shape().numel() as i32,
        }
    }

    /// Mean square error between tensor and target.
    #[must_use]
    pub fn mse(&self, target: impl IntoTensor) -> Tensor {
        let x = self - target;
        &x*&x
    }

    /// Permute tensor's dimensions using axes.
    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor {
        // TODO checks
        let axes = axes.into_axes(self.rank());
        let shape = self.shape().permute(&axes);
        self.new_op(Node::Permute(self.data, axes, shape))
    }

    /// Pow operation
    #[must_use]
    pub fn pow(&self, rhs: impl IntoTensor) -> Tensor {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "pow",
        )
    }

    /// Get tensor's rank
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([[2, 3, 4], [5, 1, 2]]);
    /// assert_eq!(x.rank(), 2);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        self.graph.borrow().shape(self.data).rank()
    }

    /// Realize tensor
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    pub fn realize(&mut self) -> Result<(), OutOfMemoryError> {
        self.graph.borrow_mut().realize(&[self.data])
    }

    /// Realize tensor's gradient
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    pub fn realize_grad(&mut self) -> Result<(), OutOfMemoryError> {
        if let Some(grad) = self.grad {
            self.graph.borrow_mut().realize(&[grad])
        } else {
            Ok(())
        }
    }

    /// `ReLU` op
    #[must_use]
    pub fn relu(&self) -> Tensor {
        self.new_op(Node::ReLU(self.data))
    }

    /// Reshape tensor
    /// # Panics
    /// Panics if number of elements in shape is
    /// different from tensor's number of elements.
    #[must_use]
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        assert_eq!(
            self.shape().numel(),
            shape.numel(),
            "Cannot reshape {:?} into {:?}",
            self.shape(),
            shape
        );
        self.new_op(Node::Reshape(self.data, shape))
    }
    
    /// Sqrt of inverse of a tensor
    #[must_use]
    pub fn rsqrt(&self) -> Tensor {
        (1.into_tensor(&self.context()).cast(self.dtype())/self).sqrt()
    }

    /// Scaled dot product attention op
    /// Currently it is not causal
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn scaled_dot_product_attention(self, key: &Tensor, value: &Tensor, dropout: f32) -> Tensor {
        let d = libm::powf(self.shape()[-1] as f32, 0.5);
        (self.dot(key.transpose()) / d).softmax(-1).dropout(dropout).dot(value)
    }

    /// Set label
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let mut x = ctx.tensor([[2, 3, 4], [5, 1, 2]]).set_label("tensor x");
    /// assert_eq!(x.label().unwrap(), "tensor x");
    /// ```
    #[allow(clippy::return_self_not_must_use)]
    pub fn set_label(&mut self, label: &str) -> Self {
        self.graph.borrow_mut().set_label(self.data, label);
        self.clone()
    }

    /// Get tensor's shape
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let x = ctx.tensor([[2, 3, 4], [5, 1, 2]]);
    /// assert_eq!(x.shape(), (2, 3));
    /// ```
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.graph.borrow().shape(self.data).clone()
    }

    /// Sin op
    #[must_use]
    pub fn sin(&self) -> Tensor {
        self.new_op(Node::Sin(self.data))
    }

    /// Softmax op
    #[must_use]
    pub fn softmax(&self, axes: impl IntoAxes) -> Tensor {
        let x_e = self.exp();
        &x_e/x_e.sum(axes)
    }

    /// Sin op
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        self.new_op(Node::Sqrt(self.data))
    }
    
    /// Standard deviation
    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Tensor {
        self.var(axes).sqrt()
    }

    /// Reduce tensor across axes, returning sum of each axes.
    /// All axes in result are kept and set to one.
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let x = ctx.tensor([[2, 3, 4], [5, 1, 2]]);
    /// let mut y = x.sum(1);
    /// assert_eq!(y.shape(), (2, 1));
    /// y.realize().unwrap();
    /// assert_eq!(y, [[9], [8]]);
    /// ```
    /// # Panics
    /// Panics if axes contain axis larger than tensor's rank,
    /// or if axes contain duplicates.
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor {
        let rank = self.rank();
        let axes = axes.into_axes(rank);
        let mut uniq = BTreeSet::new();
        assert!(
            axes.into_iter().all(move |x| uniq.insert(x)),
            "Cannot sum tensor with shape {:?} by axes {:?}, because axes contain duplicates.",
            self.shape(),
            axes
        );
        for a in &axes {
            assert!(
                *a < rank,
                "Cannot sum tensor with shape {:?} by axes {:?}, because some axes are greater than rank.",
                self.shape(),
                axes
            );
        }
        let shape = self.shape().reduce(&axes);
        self.new_op(Node::Sum(self.data, axes, shape))
    }

    /// Hyperbolic tangent operation
    /// # Panics
    /// Panics if apply to integer dtype tensor.
    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let dtype = self.dtype();
        assert_eq!(dtype, DType::F32, "Unable to execute tanh on {dtype} input.");
        self.new_op(Node::Tanh(self.data))
    }

    /// Convert tensor to `Vec<f32>`. Returns None if this tensor isn't realized.
    /// ```
    /// # use zyx::context::Context;
    /// # let mut ctx = Context::new();
    /// let x = ctx.tensor([[2, 3, 4], [5, 1, 2]]);
    /// let mut y = x.sum(1);
    /// assert_eq!(y.shape(), (2, 1));
    /// y.realize().unwrap();
    /// assert_eq!(y, [[9], [8]]);
    /// ```
    #[must_use]
    pub fn to_vec(&self) -> Option<Vec<f32>> {
        self.graph
            .borrow_mut()
            .load_f32(self.data)
            .map(|data| data.to_vec())
    }

    /// Convert tensor to `Vec<i32>`. Returns None if this tensor isn't realized.
    #[must_use]
    pub fn to_vec_i32(&self) -> Option<Vec<i32>> {
        self.graph
            .borrow_mut()
            .load_i32(self.data)
            .map(|data| data.to_vec())
    }

    /// Transpose op
    #[must_use]
    pub fn transpose(&self) -> Tensor {
        let shape = self.shape();
        let axes = shape.transpose_axes();
        let res_shape = shape.permute(&axes);
        self.new_op(Node::Permute(self.data, axes, res_shape))
    }
    
    /// Population variance
    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Tensor {
        let x = self - self.mean(());
        (&x*&x).mean(axes)
    }

    /// Set tensor's gradient to None
    /// ```
    /// # use zyx::context::Context;
    /// let mut ctx = Context::new();
    /// let mut x = ctx.tensor([2., 3., 4.]);
    /// x.zero_grad();
    /// assert!(x.grad().is_none());
    /// ```
    pub fn zero_grad(&mut self) {
        if let Some(grad) = self.grad {
            self.graph.borrow_mut().release(grad);
        }
        self.grad = None;
    }

    // This function is the only way to mutate Tensor.
    // It is used by optimizers and for loading parameters.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn set_data(&mut self, data: Tensor) {
        debug_assert_eq!(
            self.dtype(),
            data.dtype(),
            "Internal bug, set_data data input dtype != current tensor dtype"
        );
        debug_assert_eq!(
            self.shape(),
            data.shape(),
            "Internal bug, set_data data input shape != current tensor shape"
        );
        let data = NodeId::new(data.id());
        if let Some(label) = self.label() {
            self.graph.borrow_mut().set_label(data, &label);
        }
        self.graph.borrow_mut().retain(data);
        self.graph.borrow_mut().release(self.data);
        self.data = data;
        // So that we can drop all parameters of this node after it is realized
        self.graph.borrow_mut().set_leaf(self.data);
    }

    fn new_op(&self, node: Node) -> Tensor {
        let data = self.graph.borrow_mut().push(node);
        Tensor {
            data,
            grad: None,
            graph: self.graph.clone(),
        }
    }

    fn new_binary_op(&self, mut xid: NodeId, mut yid: NodeId, op: &str) -> Tensor {
        let mut expandedx = false;
        let mut expandedy = false;
        {
            let mut graph = self.graph.borrow_mut();
            assert_eq!(
                graph.dtype(xid),
                graph.dtype(yid),
                "{} parameters {} and {} have different dtypes: {} and {}",
                op,
                xid,
                yid,
                graph.dtype(xid),
                graph.dtype(yid)
            );
            let shapex = graph.shape(xid).clone();
            let nx = shapex.numel();
            let shapey = graph.shape(yid).clone();
            let ny = shapey.numel();
            match nx.cmp(&ny) {
                Ordering::Greater => {
                    yid = graph.push(Node::Expand(yid, shapex));
                    expandedy = true;
                }
                Ordering::Less => {
                    xid = graph.push(Node::Expand(xid, shapey));
                    expandedx = true;
                }
                Ordering::Equal => {}
            }
        }
        let res = match op {
            "add" => self.new_op(Node::Add(xid, yid)),
            "sub" => self.new_op(Node::Sub(xid, yid)),
            "mul" => self.new_op(Node::Mul(xid, yid)),
            "div" => self.new_op(Node::Div(xid, yid)),
            "pow" => self.new_op(Node::Pow(xid, yid)),
            _ => panic!(),
        };
        if expandedx {
            self.graph.borrow_mut().release(xid);
        }
        if expandedy {
            self.graph.borrow_mut().release(yid);
        }
        res
    }
}

impl core::fmt::Display for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO don't print the whole tensor if it is too big
        let precision = if let Some(precision) = f.precision() {
            precision
        } else {
            3
        };
        let res = match self.dtype() {
            DType::F32 => {
                if let Some(data) = &self.to_vec() {
                    tensor_to_string(data, &self.shape(), precision)
                } else {
                    "Unrealized f32 tensor".into()
                }
            }
            DType::I32 => {
                if let Some(data) = &self.to_vec_i32() {
                    tensor_to_string(data, &self.shape(), precision)
                } else {
                    "Unrealized i32 tensor".into()
                }
            }
        };
        f.write_str(&res)
    }
}

fn tensor_to_string<T: core::fmt::Display>(data: &[T], shape: &Shape, precision: usize) -> String {
    use core::fmt::Write;
    // TODO don't print whole tensor if it is big
    let n = shape.numel();
    let ndim = shape.rank();
    let mut res = String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let mut w = 0;
    for x in data {
        let l = format!("{x:>w$.precision$}").len();
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
                    res += &(" ".repeat(ndim - r) + ("[".repeat(r - 1)).as_str());
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

impl core::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        self.new_op(Node::Neg(self.data))
    }
}

impl core::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        self.new_op(Node::Neg(self.data))
    }
}

impl<IT: IntoTensor> core::ops::Add<IT> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "add",
        )
    }
}

impl<IT: IntoTensor> core::ops::Add<IT> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "add",
        )
    }
}

impl<IT: IntoTensor> core::ops::Sub<IT> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "sub",
        )
    }
}

impl<IT: IntoTensor> core::ops::Sub<IT> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "sub",
        )
    }
}

impl<IT: IntoTensor> core::ops::Mul<IT> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "mul",
        )
    }
}

impl<IT: IntoTensor> core::ops::Mul<IT> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "mul",
        )
    }
}

impl<IT: IntoTensor> core::ops::Div<IT> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "div",
        )
    }
}

impl<IT: IntoTensor> core::ops::Div<IT> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        self.new_binary_op(
            self.data,
            rhs.into_tensor(&self.graph.clone().into()).data,
            "div",
        )
    }
}

impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        if self.shape() != 1 {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                if let Some(data) = self.to_vec() {
                    if libm::fabsf(data[0] - other) > EPSILON {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            DType::I32 => {
                return false;
            }
        }
        true
    }
}

impl<const L: usize> PartialEq<[f32; L]> for Tensor {
    fn eq(&self, other: &[f32; L]) -> bool {
        if self.shape() != L {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                if let Some(data) = self.to_vec() {
                    for (x, y) in data.iter().zip(other) {
                        if libm::fabsf(x - y) > EPSILON {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            DType::I32 => {
                return false;
            }
        }
        true
    }
}

impl<const L: usize, const M: usize> PartialEq<[[f32; L]; M]> for Tensor {
    fn eq(&self, other: &[[f32; L]; M]) -> bool {
        if self.shape() != (M, L) {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                if let Some(data) = self.to_vec() {
                    for (x, y) in data.iter().zip(other.iter().flatten()) {
                        if libm::fabsf(x - y) > EPSILON {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            DType::I32 => {
                return false;
            }
        }
        true
    }
}

impl<const L: usize, const M: usize, const N: usize> PartialEq<[[[f32; L]; M]; N]> for Tensor {
    fn eq(&self, other: &[[[f32; L]; M]; N]) -> bool {
        if self.shape() != (N, M, L) {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                if let Some(data) = self.to_vec() {
                    for (x, y) in data.iter().zip(other.iter().flatten().flatten()) {
                        if libm::fabsf(x - y) > EPSILON {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            DType::I32 => {
                return false;
            }
        }
        true
    }
}

impl PartialEq<i32> for Tensor {
    fn eq(&self, other: &i32) -> bool {
        if self.shape() != 1 {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                return false;
            }
            DType::I32 => {
                if let Some(data) = self.to_vec_i32() {
                    if data[0] != *other {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }
}

impl<const L: usize> PartialEq<[i32; L]> for Tensor {
    fn eq(&self, other: &[i32; L]) -> bool {
        if self.shape() != L {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                return false;
            }
            DType::I32 => {
                if let Some(data) = self.to_vec_i32() {
                    for (x, y) in data.iter().zip(other) {
                        if x != y {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }
}

impl<const L: usize, const M: usize> PartialEq<[[i32; L]; M]> for Tensor {
    fn eq(&self, other: &[[i32; L]; M]) -> bool {
        if self.shape() != (M, L) {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                return false;
            }
            DType::I32 => {
                if let Some(data) = self.to_vec_i32() {
                    for (x, y) in data.iter().zip(other.iter().flatten()) {
                        if x != y {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }
}

impl<const L: usize, const M: usize, const N: usize> PartialEq<[[[i32; L]; M]; N]> for Tensor {
    fn eq(&self, other: &[[[i32; L]; M]; N]) -> bool {
        if self.shape() != (N, M, L) {
            return false;
        }
        match self.dtype() {
            DType::F32 => {
                return false;
            }
            DType::I32 => {
                if let Some(data) = self.to_vec_i32() {
                    for (x, y) in data.iter().zip(other.iter().flatten().flatten()) {
                        if x != y {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }
}
