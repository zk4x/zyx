use crate::{
    DType, RT, Tensor, ZyxError,
    shape::{Dim, UAxis, into_axes},
    tensor::Axis,
};
use paste::paste;

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Mean,
    Var,
    Std,
    Max,
    Min,
    Prod,
}

impl Tensor {
    fn inverse(&self) -> Tensor {
        let dtype = self.dtype();
        if dtype.is_float() {
            -self
        } else if dtype.is_int() {
            self.bitnot()
        } else {
            !self
        }
    }

    fn reduce_impl<const KEEPDIM: bool>(
        &self,
        op: ReduceOp,
        axes: impl IntoIterator<Item = Axis>,
        dtype: Option<DType>,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        fn reduce_acc_dtype(dtype: DType) -> DType {
            if dtype.is_uint() {
                return dtype.least_upper_dtype(DType::U32);
            }
            if dtype.is_int() || dtype == DType::Bool {
                return dtype.least_upper_dtype(DType::I32);
            }
            dtype.least_upper_dtype(DType::F32)
        }

        // Determine axes
        let mut shape = self.shape();
        let rank = shape.len();
        let x_dtype = self.dtype();
        let axes: Vec<_> = axes.into_iter().collect();
        let axes_vec: Vec<UAxis> = into_axes(axes.clone(), rank)?;

        // Start with the base reduction for ops runtime supports
        let mut tensor = match op {
            ReduceOp::Sum => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().sum_reduce(x.id, axes_vec.clone()) }
            }
            ReduceOp::Max => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().max_reduce(x.id, axes_vec.clone()) }
            }
            ReduceOp::Min => {
                if let Some(dtype) = dtype {
                    self.inverse().max_dtype(axes, dtype)?.inverse()
                } else {
                    self.inverse().max(axes)?.inverse()
                }
            }
            ReduceOp::Prod => {
                if let Some(dtype) = dtype {
                    self.log2().sum_dtype(axes, dtype)?.exp2()
                } else {
                    self.log2().sum(axes)?.exp2()
                }
            }
            ReduceOp::Mean => {
                let n: i64 = axes_vec.iter().map(|&a| shape[a]).product::<Dim>().try_into().unwrap();
                let x = if let Some(dtype) = dtype {
                    self.sum_dtype(axes, dtype)?
                } else {
                    self.sum(axes)?
                };
                x / Tensor::from(n).cast(x_dtype)
            }
            ReduceOp::Var => {
                if let Some(dtype) = dtype {
                    let x = self - self.mean_keepdim_dtype(axes.clone(), dtype)?;
                    let d = Axis::try_from(axes_vec.iter().map(|&a| shape[a]).product::<usize>()).unwrap()
                        - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum_dtype(axes, dtype)? / Tensor::from(d).cast(x_dtype)
                } else {
                    let x = self - self.mean_keepdim(axes.clone())?;
                    let d = Axis::try_from(axes_vec.iter().map(|&a| shape[a]).product::<usize>()).unwrap()
                        - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum(axes)? / Tensor::from(d).cast(x_dtype)
                }
            }
            ReduceOp::Std => {
                if let Some(dtype) = dtype {
                    self.var_axes_dtype(axes, dtype)?.sqrt()
                } else {
                    self.var_axes(axes)?.sqrt()
                }
            }
        };

        if dtype.is_none() && x_dtype != tensor.dtype() {
            tensor = tensor.cast(x_dtype);
        }

        // Apply keepdim
        if KEEPDIM {
            for a in axes_vec {
                shape[a] = 1;
            }
            tensor = tensor.reshape(shape)?;
        }

        Ok(tensor)
    }
}

macro_rules! define_reduce_op {
    ($name:ident, $op_variant:expr) => {
        paste! {
            impl Tensor {
                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction over all elements.\n\n",
                    "# Examples\n",
                    "```\n",
                    "use zyx::Tensor;\n",
                    "let t = Tensor::from([1.0, 2.0, 3.0]);\n",
                    "let result = t.", stringify!($name), "_all();\n",
                    "```\n",
                )]
                #[must_use]
                pub fn [<$name _all>](&self) -> Tensor {
                    self.reduce_impl::<false>($op_variant, [], None, 1).unwrap()
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction over all elements, keeping reduced dimensions.\n\n",
                    "Reduced axes are retained with length 1.\n\n",
                    "# Examples\n",
                    "```\n",
                    "use zyx::Tensor;\n",
                    "let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);\n",
                    "let result = t.", stringify!($name), "_all_keepdim();\n",
                    "```\n",
                )]
                #[must_use]
                pub fn [<$name _all_keepdim>](&self) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], None, 1).unwrap()
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction along the specified `axes`.\n\n",
                    "# Arguments\n",
                    "* `axes` — Iterable of axes to reduce over.\n\n",
                    "# Examples\n",
                    "```\n",
                    "use zyx::Tensor;\n",
                    "let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);\n",
                    "let result = t.", stringify!($name), "_axes([0]).unwrap();\n",
                    "```\n",
                    "\n",
                    "# Errors\n",
                    "When axes are out of range\n"
                )]
                pub fn $name(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, None, 1)
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction along the specified `axes`, keeping reduced dimensions.\n\n",
                    "# Arguments\n",
                    "* `axes` — Iterable of axes to reduce over.\n",
                    "* Keeps reduced dimensions with length 1.\n\n",
                    "# Examples\n",
                    "```\n",
                    "use zyx::Tensor;\n",
                    "let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);\n",
                    "let result = t.", stringify!($name), "_axes_keepdim([1]).unwrap();\n",
                    "```\n",
                    "\n",
                    "# Errors\n",
                    "When axes are out of range\n"
                )]
                pub fn [<$name _keepdim>](&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, None, 1)
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction over all elements and casts the result to `dtype`.\n\n",
                    "# Arguments\n",
                    "* `dtype` — Desired output data type.\n\n",
                    "# Examples\n",
                    "```\n",
                    "use zyx::{Tensor, DType};\n",
                    "let t = Tensor::from([1.0, 2.0, 3.0]);\n",
                    "let result = t.", stringify!($name), "_all_dtype(DType::F64);\n",
                    "```\n",
                )]
                #[must_use]
                pub fn [<$name _all_dtype>](&self, dtype: DType) -> Tensor {
                    self.reduce_impl::<false>($op_variant, [], Some(dtype), 1).unwrap()
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction over all elements, keeping reduced dimensions,\n",
                    "and casts the result to `dtype`.\n\n",
                    "# Arguments\n",
                    "* `dtype` — Desired output data type.\n",
                )]
                #[must_use]
                pub fn [<$name _all_keepdim_dtype>](&self, dtype: DType) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], Some(dtype), 1).unwrap()
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction along specified `axes`, casting the result to `dtype`.\n\n",
                    "# Arguments\n",
                    "* `axes` — Iterable of axes to reduce over.\n",
                    "* `dtype` — Desired output data type.\n",
                    "\n",
                    "# Errors\n",
                    "When axes are out of range\n"
                )]
                pub fn [<$name _dtype>](
                    &self,
                    axes: impl IntoIterator<Item = Axis>,
                    dtype: DType
                ) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, Some(dtype), 1)
                }

                #[doc = concat!(
                    "Computes the `", stringify!($name), "` reduction along specified `axes`, keeping reduced dimensions,\n",
                    "and casts the result to `dtype`.\n\n",
                    "# Arguments\n",
                    "* `axes` — Iterable of axes to reduce over.\n",
                    "* `dtype` — Desired output data type.\n",
                    "\n",
                    "# Errors\n",
                    "When axes are out of range\n"
                )]
                pub fn [<$name _keepdim_dtype>](
                    &self,
                    axes: impl IntoIterator<Item = Axis>,
                    dtype: DType
                ) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, Some(dtype), 1)
                }
            }
        }
    };
}

define_reduce_op!(sum, ReduceOp::Sum);
define_reduce_op!(mean, ReduceOp::Mean);
define_reduce_op!(max, ReduceOp::Max);
define_reduce_op!(min, ReduceOp::Min);
define_reduce_op!(prod, ReduceOp::Prod);

/// Macro to generate reduction operations with optional axes, keepdim, dtype, and correction.
macro_rules! define_reduce_op_with_correction {
    ($name:ident, $op_variant:expr) => {
        paste! {
            impl Tensor {
                /// Computes the [$name] reduction over **all elements**.
                ///
                /// # Returns
                /// A scalar tensor containing the reduction result.
                #[must_use]
                pub fn $name(&self) -> Tensor {
                    self.reduce_impl::<false>($op_variant, [], None, 1).unwrap()
                }

                /// Computes the [$name] reduction over all elements, keeping reduced dimensions.
                ///
                /// Reduced axes are retained with length 1.
                #[must_use]
                pub fn [<$name _keepdim>](&self) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], None, 1).unwrap()
                }

                /// Computes the [$name] reduction over all elements, casting the result to `dtype`.
                #[must_use]
                pub fn [<$name _dtype>](&self, dtype: DType) -> Tensor {
                    self.reduce_impl::<false>($op_variant, [], Some(dtype), 1).unwrap()
                }

                /// Computes the [$name] reduction along specified `axes`.
                ///
                /// # Arguments
                /// * `axes` — Iterable of axes to reduce over.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes>](&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, None, 1)
                }

                /// Computes the [$name] reduction along specified `axes`, keeping reduced dimensions.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_keepdim>](&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, None, 1)
                }

                /// Computes the [$name] reduction along specified `axes`, casting the result to `dtype`.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_dtype>](&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, Some(dtype), 1)
                }

                /// Computes the [$name] reduction along specified `axes` with a `correction` factor.
                ///
                /// # Arguments
                /// * `axes` — Iterable of axes to reduce over.
                /// * `correction` — Bias correction to apply (e.g., for variance or standard deviation).
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_correction>](&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, None, correction)
                }

                /// Computes the [$name] reduction over all elements, keeping reduced dimensions and casting to `dtype`.
                #[must_use]
                pub fn [<$name _keepdim_dtype>](&self, dtype: DType) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], Some(dtype), 1).unwrap()
                }

                /// Computes the [$name] reduction over all elements, keeping reduced dimensions, with `correction`.
                #[must_use]
                pub fn [<$name _keepdim_correction>](&self, correction: Dim) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], None, correction).unwrap()
                }

                /// Computes the [$name] reduction over all elements, casting to `dtype`, with `correction`.
                #[must_use]
                pub fn [<$name _dtype_correction>](&self, dtype: DType, correction: Dim) -> Tensor {
                    self.reduce_impl::<false>($op_variant, [], Some(dtype), correction).unwrap()
                }

                /// Computes the [$name] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_keepdim_dtype>](&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, Some(dtype), 1)
                }

                /// Computes the [$name] reduction along `axes`, keeping reduced dimensions, with `correction`.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_keepdim_correction>](&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, None, correction)
                }

                /// Computes the [$name] reduction along `axes`, casting to `dtype`, with `correction`.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_dtype_correction>](&self, axes: impl IntoIterator<Item = Axis>, dtype: DType, correction: Dim) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<false>($op_variant, axes, Some(dtype), correction)
                }

                /// Computes the [$name] reduction over all elements, keeping reduced dimensions, casting to `dtype`, with `correction`.
                #[must_use]
                pub fn [<$name _keepdim_dtype_correction>](&self, dtype: DType, correction: Dim) -> Tensor {
                    self.reduce_impl::<true>($op_variant, [], Some(dtype), correction).unwrap()
                }

                /// Computes the [$name] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
                /// Includes `correction` if specified.
                ///
                /// # Errors
                /// When axes are out of range
                pub fn [<$name _axes_keepdim_dtype_correction>](&self, axes: impl IntoIterator<Item = Axis>, dtype: DType, correction: Dim) -> Result<Tensor, ZyxError> {
                    self.reduce_impl::<true>($op_variant, axes, Some(dtype), correction)
                }
            }
        }
    };
}

define_reduce_op_with_correction!(var, ReduceOp::Var);
define_reduce_op_with_correction!(std, ReduceOp::Std);
