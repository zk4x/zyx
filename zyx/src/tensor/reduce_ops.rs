// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType, RT, Tensor, ZyxError,
    kernel::BOp,
    shape::{Dim, UAxis, into_axes, into_axis},
    tensor::Axis,
};

/// Specifies how to reduce per-sample losses or values.
#[derive(Clone, Copy)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Compute the mean (average) of all values.
    Mean,
    /// No reduction, return per-sample values.
    None,
    /// Compute the variance.
    Var,
    /// Compute the standard deviation.
    Std,
    /// Take the maximum value.
    Max,
    /// Take the minimum value.
    Min,
    /// Compute the product of all values.
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

    /// Reduce implementation
    pub(crate) fn reduce_impl<const KEEPDIM: bool>(
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
        let shape = self.resolve_shape();
        // Dim tensors for CONSTRUCTION: computation must compose these so
        // symbolic dims stay symbolic (no recompile per variable value).
        // `shape` (resolved) is for user-side validation and host math only.
        let shape_dims = self.shape();
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
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Add)? }
            }
            ReduceOp::Max => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Max)? }
            }
            ReduceOp::Prod => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Mul)? }
            }
            ReduceOp::Min => {
                if let Some(dtype) = dtype {
                    self.inverse().max_dtype(axes, dtype)?.inverse()
                } else {
                    self.inverse().max(axes)?.inverse()
                }
            }
            ReduceOp::Mean => {
                // The divisor is computation — keep it SYMBOLIC: the product
                // of the reduced dims as dim tensors. Compile-time-constant
                // dims fold back to a constant in the kernel; variable-backed
                // dims become a runtime divide (no recompile per value).
                let mut n_t = Tensor::from(1i64);
                for &a in &axes_vec {
                    n_t = n_t * shape_dims[a].clone();
                }
                let x = if let Some(dtype) = dtype {
                    self.sum_dtype(axes, dtype)?
                } else {
                    self.sum(axes)?
                };
                x / n_t.cast(x_dtype)
            }
            ReduceOp::Var => {
                if let Some(dtype) = dtype {
                    let x = self - self.mean_keepdim_dtype(axes.clone(), dtype)?;
                    let shape_dims: Vec<Dim> = axes_vec.iter().map(|&a| shape[a]).collect();
                    let d =
                        Axis::try_from(shape_dims.iter().product::<Dim>() as u64).unwrap() - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum_dtype(axes, dtype)? / Tensor::from(d).cast(x_dtype)
                } else {
                    let x = self - self.mean_keepdim(axes.clone())?;
                    let shape_dims: Vec<Dim> = axes_vec.iter().map(|&a| shape[a]).collect();
                    let d =
                        Axis::try_from(shape_dims.iter().product::<Dim>() as u64).unwrap() - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum(axes)? / Tensor::from(d).cast(x_dtype)
                }
            }
            ReduceOp::Std => {
                if let Some(dtype) = dtype {
                    self.var_dtype(axes, dtype)?.sqrt()
                } else {
                    self.var(axes)?.sqrt()
                }
            }
            ReduceOp::None => self.clone(),
        };

        if dtype.is_none() && x_dtype != tensor.dtype() {
            tensor = tensor.cast(x_dtype);
        }

        // Apply keepdim — reduced axes become fresh 1 constants; kept axes
        // stay SYMBOLIC (the input's own dim tensors). Rebuilding from the
        // resolved shape would bake variable-backed dims into constants and
        // force recompiles.
        if KEEPDIM {
            let mut dims: Vec<Tensor> = Vec::with_capacity(rank);
            for (a, d) in shape_dims.iter().enumerate() {
                if axes_vec.contains(&(a as UAxis)) {
                    dims.push(Tensor::from(1i64));
                } else {
                    dims.push(d.clone());
                }
            }
            tensor = tensor.reshape(dims)?;
        }

        Ok(tensor)
    }
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `sum` reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.sum_all();
    /// ```
    #[must_use]
    pub fn sum_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Sum, [], None, 1).unwrap()
    }

    /// Compute the `sum` reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.sum_all_keepdim();
    /// ```
    #[must_use]
    pub fn sum_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Sum, [], None, 1).unwrap()
    }

    /// Compute the `sum` reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.sum([0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn sum(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Sum, axes, None, 1)
    }

    /// Compute the `sum` reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.sum_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn sum_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Sum, axes, None, 1)
    }

    /// Compute the `sum` reduction over all elements and cast the result to
    /// `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.sum_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn sum_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Sum, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `sum` reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.sum_all_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn sum_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Sum, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `sum` reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.sum_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn sum_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Sum, axes, Some(dtype), 1)
    }

    /// Compute the `sum` reduction along the given `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.sum_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn sum_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Sum, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `mean` reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.mean_all();
    /// ```
    #[must_use]
    pub fn mean_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Mean, [], None, 1).unwrap()
    }

    /// Compute the `mean` reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.mean_all_keepdim();
    /// ```
    #[must_use]
    pub fn mean_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Mean, [], None, 1).unwrap()
    }

    /// Compute the `mean` reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.mean([0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn mean(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Mean, axes, None, 1)
    }

    /// Compute the `mean` reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.mean_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn mean_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Mean, axes, None, 1)
    }

    /// Compute the `mean` reduction over all elements and cast the result to
    /// `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.mean_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn mean_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Mean, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `mean` reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.mean_all_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn mean_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Mean, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `mean` reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.mean_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn mean_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Mean, axes, Some(dtype), 1)
    }

    /// Compute the `mean` reduction along the given `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.mean_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn mean_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Mean, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// max
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `max` reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.max_all();
    /// ```
    #[must_use]
    pub fn max_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Max, [], None, 1).unwrap()
    }

    /// Compute the `max` reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.max_all_keepdim();
    /// ```
    #[must_use]
    pub fn max_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Max, [], None, 1).unwrap()
    }

    /// Compute the `max` reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.max([0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn max(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Max, axes, None, 1)
    }

    /// Compute the `max` reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.max_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn max_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Max, axes, None, 1)
    }

    /// Compute the `max` reduction over all elements and cast the result to
    /// `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.max_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn max_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Max, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `max` reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.max_all_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn max_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Max, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `max` reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.max_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn max_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Max, axes, Some(dtype), 1)
    }

    /// Compute the `max` reduction along the given `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.max_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn max_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Max, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// min
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `min` reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.min_all();
    /// ```
    #[must_use]
    pub fn min_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Min, [], None, 1).unwrap()
    }

    /// Compute the `min` reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.min_all_keepdim();
    /// ```
    #[must_use]
    pub fn min_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Min, [], None, 1).unwrap()
    }

    /// Compute the `min` reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.min([0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn min(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Min, axes, None, 1)
    }

    /// Compute the `min` reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.min_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn min_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Min, axes, None, 1)
    }

    /// Compute the `min` reduction over all elements and cast the result to
    /// `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.min_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn min_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Min, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `min` reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.min_all_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn min_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Min, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `min` reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.min_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn min_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Min, axes, Some(dtype), 1)
    }

    /// Compute the `min` reduction along the given `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.min_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn min_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Min, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// prod
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `prod` reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.prod_all();
    /// ```
    #[must_use]
    pub fn prod_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Prod, [], None, 1).unwrap()
    }

    /// Compute the `prod` reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.prod_all_keepdim();
    /// ```
    #[must_use]
    pub fn prod_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Prod, [], None, 1).unwrap()
    }

    /// Compute the `prod` reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.prod([0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn prod(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Prod, axes, None, 1)
    }

    /// Compute the `prod` reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.prod_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn prod_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Prod, axes, None, 1)
    }

    /// Compute the `prod` reduction over all elements and cast the result to
    /// `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.prod_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn prod_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Prod, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `prod` reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.prod_all_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn prod_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Prod, [], Some(dtype), 1).unwrap()
    }

    /// Compute the `prod` reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.prod_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn prod_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Prod, axes, Some(dtype), 1)
    }

    /// Compute the `prod` reduction along the given `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.prod_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn prod_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Prod, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// var
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the variance reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all();
    /// ```
    #[must_use]
    pub fn var_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], None, 1).unwrap()
    }

    /// Compute the variance reduction over all elements, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_all_keepdim();
    /// ```
    #[must_use]
    pub fn var_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], None, 1).unwrap()
    }

    /// Compute the variance reduction over all elements and cast the result
    /// to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn var_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], Some(dtype), 1).unwrap()
    }

    /// Compute the variance reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, None, 1)
    }

    /// Compute the variance reduction along the given `axes`, keeping reduced
    /// dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, None, 1)
    }

    /// Compute the variance reduction along the given `axes` and cast the
    /// result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, Some(dtype), 1)
    }

    /// Compute the variance reduction along the given `axes` with a
    /// `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_correction([1], 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, None, correction)
    }

    /// Compute the variance reduction over all elements with a `correction`
    /// factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all_correction(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the correction is out of range.
    pub fn var_all_correction(&self, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, [], None, correction)
    }

    /// Compute the variance reduction over all elements, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn var_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], Some(dtype), 1).unwrap()
    }

    /// Compute the variance reduction over all elements, keeping reduced
    /// dimensions, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all_keepdim_correction(0);
    /// ```
    #[must_use]
    pub fn var_all_keepdim_correction(&self, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], None, correction).unwrap()
    }

    /// Compute the variance reduction over all elements, cast to `dtype`,
    /// with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all_dtype_correction(DType::F64, 0);
    /// ```
    #[must_use]
    pub fn var_all_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], Some(dtype), correction).unwrap()
    }

    /// Compute the variance reduction along `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_axes_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_axes_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, Some(dtype), 1)
    }

    /// Compute the variance reduction along `axes`, keeping reduced
    /// dimensions, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_keepdim_correction([1], 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_keepdim_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, None, correction)
    }

    /// Compute the variance reduction along `axes`, cast the result to
    /// `dtype`, and apply a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_dtype_correction([1], DType::F64, 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, Some(dtype), correction)
    }

    /// Compute the variance reduction over all elements, keeping reduced
    /// dimensions, cast the result to `dtype`, and apply a `correction`
    /// factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.var_all_keepdim_dtype_correction(DType::F64, 0);
    /// ```
    #[must_use]
    pub fn var_all_keepdim_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], Some(dtype), correction).unwrap()
    }

    /// Compute the variance reduction along `axes`, keeping reduced
    /// dimensions, cast the result to `dtype`, and apply a `correction`
    /// factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.var_keepdim_dtype_correction([1], DType::F64, 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn var_keepdim_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, Some(dtype), correction)
    }
}

// ---------------------------------------------------------------------------
// std
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the standard deviation reduction over all elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all();
    /// ```
    #[must_use]
    pub fn std_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], None, 1).unwrap()
    }

    /// Compute the standard deviation reduction over all elements, keeping
    /// reduced dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_all_keepdim();
    /// ```
    #[must_use]
    pub fn std_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], None, 1).unwrap()
    }

    /// Compute the standard deviation reduction over all elements and cast
    /// the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn std_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], Some(dtype), 1).unwrap()
    }

    /// Compute the standard deviation reduction along the given `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, None, 1)
    }

    /// Compute the standard deviation reduction along the given `axes`,
    /// keeping reduced dimensions with length 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, None, 1)
    }

    /// Compute the standard deviation reduction along the given `axes` and
    /// cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, Some(dtype), 1)
    }

    /// Compute the standard deviation reduction along the given `axes` with
    /// a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_correction([1], 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, None, correction)
    }

    /// Compute the standard deviation reduction over all elements with a
    /// `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all_correction(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the correction is out of range.
    pub fn std_all_correction(&self, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, [], None, correction)
    }

    /// Compute the standard deviation reduction over all elements, keeping
    /// reduced dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_keepdim_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn std_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], Some(dtype), 1).unwrap()
    }

    /// Compute the standard deviation reduction over all elements, keeping
    /// reduced dimensions, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all_keepdim_correction(0);
    /// ```
    #[must_use]
    pub fn std_all_keepdim_correction(&self, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], None, correction).unwrap()
    }

    /// Compute the standard deviation reduction over all elements, cast the
    /// result to `dtype`, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all_dtype_correction(DType::F64, 0);
    /// ```
    #[must_use]
    pub fn std_all_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], Some(dtype), correction).unwrap()
    }

    /// Compute the standard deviation reduction along `axes`, keeping reduced
    /// dimensions, and cast the result to `dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_axes_keepdim_dtype([1], DType::F64).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_axes_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, Some(dtype), 1)
    }

    /// Compute the standard deviation reduction along `axes`, keeping reduced
    /// dimensions, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_keepdim_correction([1], 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_keepdim_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, None, correction)
    }

    /// Compute the standard deviation reduction along `axes`, cast the result
    /// to `dtype`, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_dtype_correction([1], DType::F64, 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, Some(dtype), correction)
    }

    /// Compute the standard deviation reduction over all elements, keeping
    /// reduced dimensions, cast the result to `dtype`, with a `correction`
    /// factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.std_all_keepdim_dtype_correction(DType::F64, 0);
    /// ```
    #[must_use]
    pub fn std_all_keepdim_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], Some(dtype), correction).unwrap()
    }

    /// Compute the standard deviation reduction along `axes`, keeping reduced
    /// dimensions, cast the result to `dtype`, with a `correction` factor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let y = t.std_keepdim_dtype_correction([1], DType::F64, 0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when an axis is out of range.
    pub fn std_keepdim_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, Some(dtype), correction)
    }

    /// Compute the cumulative sum reduction along the given `axis`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.cumsum(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn cumsum(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cum_reduce(axis, BOp::Add)
    }

    /// Compute the cumulative max reduction along the given `axis`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.cummax(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn cummax(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cum_reduce(axis, BOp::Max)
    }

    /// Compute the cumulative product reduction along the given `axis`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = t.cumprod(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn cumprod(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cum_reduce(axis, BOp::Mul)
    }

    /// Cumulative reduce along axis
    fn cum_reduce(&self, axis: Axis, rop: BOp) -> Result<Tensor, ZyxError> {
        let shape = self.resolve_shape();
        let uaxis = into_axis(axis, shape.len())?;
        let pl_sz = i64::try_from(shape[uaxis] - 1).unwrap();
        let mut x = self.transpose(axis, -1)?;
        x = x.rpad_zeros([(pl_sz, 0i64)])?;
        x = x.pool([shape[uaxis]], [1i64], [1i64])?;
        x = match rop {
            BOp::Add => x.sum([-1])?,
            BOp::Max => x.max([-1])?,
            BOp::Mul => x.prod([-1])?,
            _ => unreachable!(),
        };
        x = x.transpose(axis, -1)?;
        Ok(x)
    }
}
