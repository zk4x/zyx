// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType, RT, Tensor, ZyxError,
    kernel::BOp,
    shape::{Dim, UAxis, into_axes, into_axis},
    tensor::Axis,
};

/// Specifies how to reduce per-sample losses or values.
#[cfg_attr(feature = "py", pyo3::pyclass(eq, eq_int, from_py_object))]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reduction {
    /// Sum all values.
    Sum,
    /// Compute the mean (average) of all values.
    Mean,
    /// No reduction, return per-sample values.
    NoReduction,
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
    /// Logical AND reduction (true iff all values are nonzero).
    All,
    /// Logical OR reduction (true iff any value is nonzero).
    Any,
}

impl Tensor {
    /// Reduce resolved `axes` via one permute (kept axes first, summed
    /// axes trailing) followed by `reduce_last_axis` steps.
    pub(crate) fn reduce_axes(&self, axes: Vec<UAxis>, rop: BOp) -> Result<Tensor, ZyxError> {
        let rank = self.resolve_shape().len();
        let perm: Vec<Axis> =
            (0..rank as UAxis).filter(|i| !axes.contains(i)).chain(axes.iter().copied()).map(|i| i as Axis).collect();
        let mut cur = self.clone();
        if !perm.iter().copied().enumerate().all(|(i, a)| a as usize == i) {
            cur = cur.permute(perm)?;
        }
        for _ in 0..axes.len() {
            // Bind the id in its own statement: the lock guard must be dead
            // before the assignment below drops the old `cur` (Tensor::drop
            // takes the lock; a live guard self-deadlocks).
            let nid = RT.lock().reduce_last_axis(cur.id, rop)?;
            cur = Tensor { id: nid };
        }
        Ok(cur)
    }

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
        op: Reduction,
        axes: impl IntoIterator<Item = Axis>,
        dtype: Option<DType>,
        correction: Dim,
    ) -> Result<Tensor, ZyxError> {
        // Determine axes
        let shape = self.resolve_shape();
        // Dim tensors for CONSTRUCTION: computation must compose these so
        // symbolic dims stay symbolic (no recompile per variable value).
        // `shape` (resolved) is for user-side validation and host math only.
        let shape_dims = self.shape();
        let rank = shape.len();
        let x_dtype = self.dtype();
        // Bool has no sum-like definition and the framework never casts
        // under the caller: refuse loudly, the caller casts first.
        // Min/Max stay allowed (AND/OR), None is a passthrough.
        if x_dtype == DType::Bool
            && matches!(op, Reduction::Sum | Reduction::Mean | Reduction::Var | Reduction::Std | Reduction::Prod)
        {
            return Err(ZyxError::DTypeError(
                "cannot reduce bool tensor with Sum/Mean/Var/Std/Prod, cast to an integer dtype first".into(),
            ));
        }
        let axes: Vec<_> = axes.into_iter().collect();
        let axes_vec: Vec<UAxis> = into_axes(axes.clone(), rank as UAxis)?;

        // Start with the base reduction for ops runtime supports
        let mut tensor = match op {
            Reduction::Sum => {
                // Torch semantics: accumulate in the input dtype; only
                // an explicit `dtype` (sum_dtype) changes it.
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.clone()
                };
                x.reduce_axes(axes_vec.clone(), BOp::Add)?
            }
            Reduction::Max => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.clone()
                };
                x.reduce_axes(axes_vec.clone(), BOp::Max)?
            }
            Reduction::Prod => {
                let x = if let Some(dtype) = dtype {
                    self.cast(dtype)
                } else {
                    self.clone()
                };
                x.reduce_axes(axes_vec.clone(), BOp::Mul)?
            }
            Reduction::All => {
                // Torch semantics: true iff all elements are nonzero.
                // Bool tensors reduce directly; other dtypes compare
                // against zero first (the scalar takes the input dtype
                // in `broadcast`, so this works for every dtype).
                let x = if x_dtype == DType::Bool {
                    self.clone()
                } else {
                    self.ne(Tensor::from(0))?
                };
                x.reduce_axes(axes_vec.clone(), BOp::And)?
            }
            Reduction::Any => {
                // Torch semantics: true iff any element is nonzero.
                // Bool tensors reduce directly; other dtypes compare
                // against zero first (the scalar takes the input dtype
                // in `broadcast`, so this works for every dtype).
                let x = if x_dtype == DType::Bool {
                    self.clone()
                } else {
                    self.ne(Tensor::from(0))?
                };
                x.reduce_axes(axes_vec.clone(), BOp::Or)?
            }
            Reduction::Min => {
                if let Some(dtype) = dtype {
                    self.inverse().max_dtype(axes, dtype)?.inverse()
                } else {
                    self.inverse().max(axes)?.inverse()
                }
            }
            Reduction::Mean => {
                // The divisor is computation — keep it SYMBOLIC: the product
                // of the reduced dims as dim tensors. Compile-time-constant
                // dims fold back to a constant in the kernel; variable-backed
                // dims become a runtime divide (no recompile per value).
                let mut n_t = Tensor::from(1i64);
                for &a in &axes_vec {
                    n_t = n_t * shape_dims[a as usize].clone();
                }
                let x = if let Some(dtype) = dtype {
                    self.sum_dtype(axes, dtype)?
                } else {
                    self.sum(axes)?
                };
                let xd = x.dtype();
                x / n_t.cast(xd)
            }
            Reduction::Var => {
                if let Some(dtype) = dtype {
                    let x = self - self.mean_keepdim_dtype(axes.clone(), dtype)?;
                    let shape_dims: Vec<Dim> = axes_vec.iter().map(|&a| shape[a as usize]).collect();
                    let d =
                        Axis::try_from(shape_dims.iter().product::<Dim>() as u64).unwrap() - Axis::try_from(correction).unwrap();
                    (x.clone() * x).sum_dtype(axes, dtype)? / Tensor::from(d).cast(dtype)
                } else {
                    let x = self - self.mean_keepdim(axes.clone())?;
                    let shape_dims: Vec<Dim> = axes_vec.iter().map(|&a| shape[a as usize]).collect();
                    let d =
                        Axis::try_from(shape_dims.iter().product::<Dim>() as u64).unwrap() - Axis::try_from(correction).unwrap();
                    let num = (x.clone() * x).sum(axes)?;
                    let nd = num.dtype();
                    num / Tensor::from(d).cast(nd)
                }
            }
            Reduction::Std => {
                if let Some(dtype) = dtype {
                    self.var_dtype(axes, dtype)?.sqrt()
                } else {
                    self.var(axes)?.sqrt()
                }
            }
            Reduction::NoReduction => self.clone(),
        };

        if dtype.is_none() && x_dtype != tensor.dtype() && !matches!(op, Reduction::All | Reduction::Any) {
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
        self.reduce_impl::<false>(Reduction::Sum, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Sum, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Sum, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Sum, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Sum, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Sum, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Sum, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Sum, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Mean, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Mean, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Mean, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Mean, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Mean, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Mean, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Mean, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Mean, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Max, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Max, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Max, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Max, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Max, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Max, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Max, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Max, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Min, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Min, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Min, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Min, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Min, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Min, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Min, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Min, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Prod, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Prod, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Prod, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Prod, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Prod, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Prod, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Prod, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Prod, axes, Some(dtype), 1)
    }
}

// ---------------------------------------------------------------------------
// all / any
// ---------------------------------------------------------------------------

impl Tensor {
    /// Compute the `all` reduction (logical AND) over all elements.
    ///
    /// True iff every element is nonzero (torch.all semantics); bool
    /// tensors reduce directly.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([true, true, false]);
    /// assert_eq!(t.all().to_vec::<bool>().unwrap(), vec![false]);
    /// ```
    #[must_use]
    pub fn all(&self) -> Tensor {
        self.reduce_impl::<false>(Reduction::All, [], None, 1).unwrap()
    }

    /// Compute the `any` reduction (logical OR) over all elements.
    ///
    /// True iff any element is nonzero (torch.any semantics); bool
    /// tensors reduce directly.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([false, false, true]);
    /// assert_eq!(t.any().to_vec::<bool>().unwrap(), vec![true]);
    /// ```
    #[must_use]
    pub fn any(&self) -> Tensor {
        self.reduce_impl::<false>(Reduction::Any, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Var, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Var, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Var, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Var, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Var, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Var, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Var, axes, None, correction)
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
        self.reduce_impl::<false>(Reduction::Var, [], None, correction)
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
        self.reduce_impl::<true>(Reduction::Var, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Var, [], None, correction).unwrap()
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
        self.reduce_impl::<false>(Reduction::Var, [], Some(dtype), correction).unwrap()
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
        self.reduce_impl::<true>(Reduction::Var, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Var, axes, None, correction)
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
        self.reduce_impl::<false>(Reduction::Var, axes, Some(dtype), correction)
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
        self.reduce_impl::<true>(Reduction::Var, [], Some(dtype), correction).unwrap()
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
        self.reduce_impl::<true>(Reduction::Var, axes, Some(dtype), correction)
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
        self.reduce_impl::<false>(Reduction::Std, [], None, 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Std, [], None, 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Std, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<false>(Reduction::Std, axes, None, 1)
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
        self.reduce_impl::<true>(Reduction::Std, axes, None, 1)
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
        self.reduce_impl::<false>(Reduction::Std, axes, Some(dtype), 1)
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
        self.reduce_impl::<false>(Reduction::Std, axes, None, correction)
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
        self.reduce_impl::<false>(Reduction::Std, [], None, correction)
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
        self.reduce_impl::<true>(Reduction::Std, [], Some(dtype), 1).unwrap()
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
        self.reduce_impl::<true>(Reduction::Std, [], None, correction).unwrap()
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
        self.reduce_impl::<false>(Reduction::Std, [], Some(dtype), correction).unwrap()
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
        self.reduce_impl::<true>(Reduction::Std, axes, Some(dtype), 1)
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
        self.reduce_impl::<true>(Reduction::Std, axes, None, correction)
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
        self.reduce_impl::<false>(Reduction::Std, axes, Some(dtype), correction)
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
        self.reduce_impl::<true>(Reduction::Std, [], Some(dtype), correction).unwrap()
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
        self.reduce_impl::<true>(Reduction::Std, axes, Some(dtype), correction)
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
        let uaxis = into_axis(axis, shape.len() as UAxis)?;
        let pl_sz = i64::try_from(shape[uaxis as usize] - 1).unwrap();
        let mut x = self.transpose(axis, -1)?;
        // Pad with the reduction identity, not zeros: a Max scan over
        // zero-padded negatives (or a Mul scan over any zero pad) reads
        // the padding as data. Mirrors tinygrad's identity_element.
        // NOTE: blocked on branchless where_ (-inf * 0 = nan in pad lanes).
        let dtype = self.dtype();
        let identity = match rop {
            BOp::Add => Tensor::from(0i32).cast(dtype),
            BOp::Mul => Tensor::from(1i32).cast(dtype),
            BOp::Max => match dtype {
                DType::F16 | DType::F32 | DType::F64 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2 => Tensor::from(f32::NEG_INFINITY).cast(dtype),
                DType::I8 => Tensor::from(i8::MIN).cast(dtype),
                DType::I16 => Tensor::from(i16::MIN).cast(dtype),
                DType::I32 => Tensor::from(i32::MIN).cast(dtype),
                DType::I64 => Tensor::from(i64::MIN).cast(dtype),
                DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::Bool => Tensor::from(0u32).cast(dtype),
            },
            _ => unreachable!(),
        };
        let mut padding = vec![(0i64, 0i64); shape.len() - 1];
        padding.push((pl_sz, 0i64));
        x = x.pad(padding, identity)?;
        x = x.pool([shape[uaxis as usize]], [1i64], [1i64])?;
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
