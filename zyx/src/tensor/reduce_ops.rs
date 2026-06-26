// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, RT, Tensor, ZyxError,
    kernel::BOp,
    shape::{Dim, UAxis, into_axes},
    tensor::Axis,
};

/// Specifies how to reduce per-sample losses or values.
#[derive(Clone, Copy)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Compute the mean (average) of all values.
    Mean,
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
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Add) }
            }
            ReduceOp::Max => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Max) }
            }
            ReduceOp::Prod => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.cast(reduce_acc_dtype(x_dtype))
                };
                Tensor { id: RT.lock().reduce(x.id, axes_vec.clone(), BOp::Mul) }
            }
            ReduceOp::Min => {
                if let Some(dtype) = dtype {
                    self.inverse().max_dtype(axes, dtype)?.inverse()
                } else {
                    self.inverse().max(axes)?.inverse()
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
                    let shape_dims: Vec<u64> = axes_vec.iter().map(|&a| shape[a as usize]).collect();
                    let d = Axis::try_from(shape_dims.iter().product::<u64>()).unwrap() - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum_dtype(axes, dtype)? / Tensor::from(d).cast(x_dtype)
                } else {
                    let x = self - self.mean_keepdim(axes.clone())?;
                    let shape_dims: Vec<u64> = axes_vec.iter().map(|&a| shape[a as usize]).collect();
                    let d = Axis::try_from(shape_dims.iter().product::<u64>()).unwrap() - Axis::try_from(correction).unwrap();
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

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the `sum` reduction over all elements.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.sum_all();
    /// ```
    #[must_use]
    pub fn sum_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Sum, [], None, 1).unwrap()
    }

    /// Computes the `sum` reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.sum_all_keepdim();
    /// ```
    #[must_use]
    pub fn sum_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Sum, [], None, 1).unwrap()
    }

    /// Computes the `sum` reduction along the specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.sum([0]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn sum(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Sum, axes, None, 1)
    }

    /// Computes the `sum` reduction along the specified `axes`, keeping reduced dimensions.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * Keeps reduced dimensions with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.sum_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn sum_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Sum, axes, None, 1)
    }

    /// Computes the `sum` reduction over all elements and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    ///
    /// # Examples
    /// ```
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.sum_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn sum_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Sum, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `sum` reduction over all elements, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    #[must_use]
    pub fn sum_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Sum, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `sum` reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn sum_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Sum, axes, Some(dtype), 1)
    }

    /// Computes the `sum` reduction along specified `axes`, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn sum_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Sum, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the `mean` reduction over all elements.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.mean_all();
    /// ```
    #[must_use]
    pub fn mean_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Mean, [], None, 1).unwrap()
    }

    /// Computes the `mean` reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.mean_all_keepdim();
    /// ```
    #[must_use]
    pub fn mean_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Mean, [], None, 1).unwrap()
    }

    /// Computes the `mean` reduction along the specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.mean([0]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn mean(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Mean, axes, None, 1)
    }

    /// Computes the `mean` reduction along the specified `axes`, keeping reduced dimensions.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * Keeps reduced dimensions with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.mean_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn mean_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Mean, axes, None, 1)
    }

    /// Computes the `mean` reduction over all elements and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    ///
    /// # Examples
    /// ```
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.mean_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn mean_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Mean, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `mean` reduction over all elements, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    #[must_use]
    pub fn mean_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Mean, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `mean` reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn mean_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Mean, axes, Some(dtype), 1)
    }

    /// Computes the `mean` reduction along specified `axes`, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn mean_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Mean, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// max
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the `max` reduction over all elements.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.max_all();
    /// ```
    #[must_use]
    pub fn max_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Max, [], None, 1).unwrap()
    }

    /// Computes the `max` reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.max_all_keepdim();
    /// ```
    #[must_use]
    pub fn max_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Max, [], None, 1).unwrap()
    }

    /// Computes the `max` reduction along the specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.max([0]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn max(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Max, axes, None, 1)
    }

    /// Computes the `max` reduction along the specified `axes`, keeping reduced dimensions.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * Keeps reduced dimensions with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.max_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn max_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Max, axes, None, 1)
    }

    /// Computes the `max` reduction over all elements and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    ///
    /// # Examples
    /// ```
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.max_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn max_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Max, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `max` reduction over all elements, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    #[must_use]
    pub fn max_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Max, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `max` reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn max_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Max, axes, Some(dtype), 1)
    }

    /// Computes the `max` reduction along specified `axes`, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn max_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Max, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// min
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the `min` reduction over all elements.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.min_all();
    /// ```
    #[must_use]
    pub fn min_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Min, [], None, 1).unwrap()
    }

    /// Computes the `min` reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.min_all_keepdim();
    /// ```
    #[must_use]
    pub fn min_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Min, [], None, 1).unwrap()
    }

    /// Computes the `min` reduction along the specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.min([0]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn min(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Min, axes, None, 1)
    }

    /// Computes the `min` reduction along the specified `axes`, keeping reduced dimensions.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * Keeps reduced dimensions with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.min_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn min_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Min, axes, None, 1)
    }

    /// Computes the `min` reduction over all elements and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    ///
    /// # Examples
    /// ```
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.min_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn min_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Min, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `min` reduction over all elements, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    #[must_use]
    pub fn min_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Min, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `min` reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn min_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Min, axes, Some(dtype), 1)
    }

    /// Computes the `min` reduction along specified `axes`, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn min_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Min, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// prod
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the `prod` reduction over all elements.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.prod_all();
    /// ```
    #[must_use]
    pub fn prod_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Prod, [], None, 1).unwrap()
    }

    /// Computes the `prod` reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.prod_all_keepdim();
    /// ```
    #[must_use]
    pub fn prod_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Prod, [], None, 1).unwrap()
    }

    /// Computes the `prod` reduction along the specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.prod([0]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn prod(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Prod, axes, None, 1)
    }

    /// Computes the `prod` reduction along the specified `axes`, keeping reduced dimensions.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * Keeps reduced dimensions with length 1.
    ///
    /// # Examples
    /// ```
    /// use zyx::Tensor;
    /// let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let result = t.prod_keepdim([1]).unwrap();
    /// ```
    ///
    /// # Errors
    /// When axes are out of range
    pub fn prod_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Prod, axes, None, 1)
    }

    /// Computes the `prod` reduction over all elements and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    ///
    /// # Examples
    /// ```
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0, 2.0, 3.0]);
    /// let result = t.prod_all_dtype(DType::F64);
    /// ```
    #[must_use]
    pub fn prod_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Prod, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `prod` reduction over all elements, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `dtype` — Desired output data type.
    #[must_use]
    pub fn prod_all_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Prod, [], Some(dtype), 1).unwrap()
    }

    /// Computes the `prod` reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn prod_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Prod, axes, Some(dtype), 1)
    }

    /// Computes the `prod` reduction along specified `axes`, keeping reduced dimensions,
    /// and casts the result to `dtype`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `dtype` — Desired output data type.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn prod_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Prod, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// var
// ---------------------------------------------------------------------------

impl Tensor {
    /// Computes the [var] reduction over **all elements**.
    ///
    /// # Returns
    /// A scalar tensor containing the reduction result.
    #[must_use]
    pub fn var_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], None, 1).unwrap()
    }

    /// Computes the [var] reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    #[must_use]
    pub fn var_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], None, 1).unwrap()
    }

    /// Computes the [var] reduction over all elements, casting the result to `dtype`.
    #[must_use]
    pub fn var_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], Some(dtype), 1).unwrap()
    }

    /// Computes the [var] reduction along specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, None, 1)
    }

    /// Computes the [var] reduction along specified `axes`, keeping reduced dimensions.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, None, 1)
    }

    /// Computes the [var] reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, Some(dtype), 1)
    }

    /// Computes the [var] reduction along specified `axes` with a `correction` factor.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `correction` — Bias correction to apply.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, None, correction)
    }

    /// All correction
    ///
    /// # Errors
    /// When correction is out of range.
    pub fn var_all_correction(&self, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, [], None, correction)
    }

    /// Computes the [var] reduction over all elements, keeping reduced dimensions and casting to `dtype`.
    #[must_use]
    pub fn var_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], Some(dtype), 1).unwrap()
    }

    /// Computes the [var] reduction over all elements, keeping reduced dimensions, with `correction`.
    #[must_use]
    pub fn var_all_keepdim_correction(&self, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], None, correction).unwrap()
    }

    /// Computes the [var] reduction over all elements, casting to `dtype`, with `correction`.
    #[must_use]
    pub fn var_all_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Var, [], Some(dtype), correction).unwrap()
    }

    /// Computes the [var] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_axes_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, Some(dtype), 1)
    }

    /// Computes the [var] reduction along `axes`, keeping reduced dimensions, with `correction`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_keepdim_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Var, axes, None, correction)
    }

    /// Computes the [var] reduction along `axes`, casting to `dtype`, with `correction`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn var_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Var, axes, Some(dtype), correction)
    }

    /// Computes the [var] reduction over all elements, keeping reduced dimensions, casting to `dtype`, with `correction`.
    #[must_use]
    pub fn var_all_keepdim_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Var, [], Some(dtype), correction).unwrap()
    }

    /// Computes the [var] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
    /// Includes `correction` if specified.
    ///
    /// # Errors
    /// When axes are out of range
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
    /// Computes the [std] reduction over **all elements**.
    ///
    /// # Returns
    /// A scalar tensor containing the reduction result.
    #[must_use]
    pub fn std_all(&self) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], None, 1).unwrap()
    }

    /// Computes the [std] reduction over all elements, keeping reduced dimensions.
    ///
    /// Reduced axes are retained with length 1.
    #[must_use]
    pub fn std_all_keepdim(&self) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], None, 1).unwrap()
    }

    /// Computes the [std] reduction over all elements, casting the result to `dtype`.
    #[must_use]
    pub fn std_all_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], Some(dtype), 1).unwrap()
    }

    /// Computes the [std] reduction along specified `axes`.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, None, 1)
    }

    /// Computes the [std] reduction along specified `axes`, keeping reduced dimensions.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_keepdim(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, None, 1)
    }

    /// Computes the [std] reduction along specified `axes`, casting the result to `dtype`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, Some(dtype), 1)
    }

    /// Computes the [std] reduction along specified `axes` with a `correction` factor.
    ///
    /// # Arguments
    /// * `axes` — Iterable of axes to reduce over.
    /// * `correction` — Bias correction to apply.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, None, correction)
    }

    /// All correction
    ///
    /// # Errors
    /// When correction is out of range.
    pub fn std_all_correction(&self, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, [], None, correction)
    }

    /// Computes the [std] reduction over all elements, keeping reduced dimensions and casting to `dtype`.
    #[must_use]
    pub fn std_keepdim_dtype(&self, dtype: DType) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], Some(dtype), 1).unwrap()
    }

    /// Computes the [std] reduction over all elements, keeping reduced dimensions, with `correction`.
    #[must_use]
    pub fn std_all_keepdim_correction(&self, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], None, correction).unwrap()
    }

    /// Computes the [std] reduction over all elements, casting to `dtype`, with `correction`.
    #[must_use]
    pub fn std_all_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<false>(ReduceOp::Std, [], Some(dtype), correction).unwrap()
    }

    /// Computes the [std] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_axes_keepdim_dtype(&self, axes: impl IntoIterator<Item = Axis>, dtype: DType) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, Some(dtype), 1)
    }

    /// Computes the [std] reduction along `axes`, keeping reduced dimensions, with `correction`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_keepdim_correction(&self, axes: impl IntoIterator<Item = Axis>, correction: Dim) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, None, correction)
    }

    /// Computes the [std] reduction along `axes`, casting to `dtype`, with `correction`.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<false>(ReduceOp::Std, axes, Some(dtype), correction)
    }

    /// Computes the [std] reduction over all elements, keeping reduced dimensions, casting to `dtype`, with `correction`.
    #[must_use]
    pub fn std_all_keepdim_dtype_correction(&self, dtype: DType, correction: Dim) -> Tensor {
        self.reduce_impl::<true>(ReduceOp::Std, [], Some(dtype), correction).unwrap()
    }

    /// Computes the [std] reduction along `axes`, keeping reduced dimensions, casting to `dtype`.
    /// Includes `correction` if specified.
    ///
    /// # Errors
    /// When axes are out of range
    pub fn std_keepdim_dtype_correction(
        &self,
        axes: impl IntoIterator<Item = Axis>,
        dtype: DType,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        self.reduce_impl::<true>(ReduceOp::Std, axes, Some(dtype), correction)
    }
}
