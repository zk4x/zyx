// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tensor
//!
//! Tensors are at the core of all machine learning.

#![allow(clippy::fallible_impl_from)]

use crate::backend::DTypeCapability;
use crate::dtype::{Constant, DType};
use crate::error::ZyxError;
use crate::kernel::{BOp, IDX_T, UOp};
use crate::runtime::{ResolvedDim, TensorData};
use crate::scalar::{Float, Scalar};
use crate::scalar::{bf16, f8e4m3, f8e5m2, f16};
use crate::shape::{Dim, UAxis, into_axes, into_axis};
use crate::slab::SlabId;
use crate::{DebugMask, RT};
use std::fmt::{Debug, Display};
use std::iter::{once, repeat_n};
use std::ops::{Bound, Mul, Neg, Not, Range, RangeBounds};
use std::path::Path;

#[cfg(feature = "py")]
pub use index_ops::DimIndex;
pub use reduce_ops::ReduceOp;
mod binary_ops;

mod dequantize;

mod elementwise;
mod index_ops;
mod reduce_ops;
mod tilize;

/// Signed axis, when we need negative axes for indexing, reduces and so on...
pub type Axis = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u32);

impl TensorId {
    pub const fn null() -> Self {
        Self(u32::MAX)
    }

    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<usize> for TensorId {
    fn from(value: usize) -> Self {
        TensorId(value as u32)
    }
}

impl From<TensorId> for usize {
    fn from(value: TensorId) -> usize {
        value.0 as usize
    }
}

impl SlabId for TensorId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

/// Device selector and handle. Re-exported from the backend: `Copy`, names the
/// backend plus the hardware ordinal, resolves to a process-wide global device.
/// The device's memory pool is always derived from the device, never the reverse.
pub use crate::backend::Dev;

/// A tensor represents a multi-dimensional array of values. This is the primary data structure in the library.
///
/// The `Tensor` struct contains an internal identifier (`id`) that uniquely identifies each tensor.
/// Thus tensor is only 4 bytes, but it is reference counted, so it is not Copy. Clones are cheap, but require
/// locking a mutex.
///
/// ## Initialization
///
/// Tensors are initialized using [`Tensor::from`].
/// This works for initialization from arrays, vectors or scalars. Arrays can be nested.
///
/// For initialization from various random distributions, check respective associated methods.
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct Tensor {
    pub(super) id: TensorId,
}

impl Tensor {
    pub(crate) fn from_id(id: TensorId) -> Self {
        Tensor { id }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        RT.lock().retain(self.id);
        Tensor { id: self.id }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let mut rt = match RT.try_lock() {
            Ok(rt) => rt,
            Err(_poisoned) => return, // poisoned.into_inner(),
        };
        rt.release(self.id);
    }
}

impl crate::Module for Tensor {
    fn iter(&self) -> impl Iterator<Item = &Tensor> {
        once(self)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tensor> {
        once(self)
    }

    fn iter_tensors(&self) -> impl Iterator<Item = (String, &Tensor)> {
        once((format!("{}", self.id), self))
    }

    fn iter_tensors_mut(&mut self) -> impl Iterator<Item = (String, &mut Tensor)> {
        once((format!("{}", self.id), self))
    }
}

// Trait to zip tuples of iterators
trait TupleZip: Sized {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn zip(self) -> Self::IntoIter;
}

// Implementation for 2-tuples
impl<IA, IB, T> TupleZip for (IA, IB)
where
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    T: Copy,
{
    type Item = (T, T);
    type IntoIter = std::iter::Zip<IA::IntoIter, IB::IntoIter>;

    fn zip(self) -> Self::IntoIter {
        self.0.into_iter().zip(self.1)
    }
}

// Implementation for 3-tuples
impl<IA, IB, IC, T> TupleZip for (IA, IB, IC)
where
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    IC: IntoIterator<Item = T>,
    T: Copy,
{
    type Item = (T, T, T);
    type IntoIter =
        std::iter::Map<std::iter::Zip<std::iter::Zip<IA::IntoIter, IB::IntoIter>, IC::IntoIter>, fn(((T, T), T)) -> (T, T, T)>;

    fn zip(self) -> Self::IntoIter {
        self.0.into_iter().zip(self.1).zip(self.2).map(|((a, b), c)| (a, b, c))
    }
}

// Implementation for 4-tuples
impl<IA, IB, IC, ID, T> TupleZip for (IA, IB, IC, ID)
where
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    IC: IntoIterator<Item = T>,
    ID: IntoIterator<Item = T>,
    T: Copy,
{
    type Item = (T, T, T, T);
    type IntoIter = std::iter::Map<
        std::iter::Zip<std::iter::Zip<std::iter::Zip<IA::IntoIter, IB::IntoIter>, IC::IntoIter>, ID::IntoIter>,
        fn((((T, T), T), T)) -> (T, T, T, T),
    >;

    fn zip(self) -> Self::IntoIter {
        self.0.into_iter().zip(self.1).zip(self.2).zip(self.3).map(|(((a, b), c), d)| (a, b, c, d))
    }
}

impl Tensor {
    /// Return the shape of the tensor as concrete dimensions.
    ///
    /// A host-side read only: static dims come from the IR and dynamic (symbolic)
    /// dims resolve through their variables' current values. Never touches
    /// device execution. For per-dim symbolic tensors use [`Tensor::shape`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let dims = t.resolve_shape();
    /// assert_eq!(dims.len(), 2);
    /// assert_eq!(dims[0], 2);
    /// ```
    #[must_use]
    pub fn resolve_shape(&self) -> Vec<Dim> {
        RT.lock().resolve_shape(self.id)
    }

    /// Symbolic shape of this tensor: one scalar IDX_T [`Tensor`] per
    /// dimension. Static dims are constant tensors, dynamic ones are
    /// variable-backed expressions. Use this to CONSTRUCT shapes (reshape,
    /// expand, broadcast); use [`Tensor::resolve_shape`] only to DECIDE
    /// (checks, display).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let dims = t.shape();
    /// assert_eq!(dims.len(), 2);
    /// assert_eq!(dims[0].item::<i64>(), 2);
    /// ```
    #[must_use]
    pub fn shape(&self) -> Vec<Tensor> {
        let mut rt = RT.lock();
        let tids = rt.shape(self.id);
        // Each returned Tensor takes ownership of a reference to the slab dim
        // node; retain so dropping the handle does not free a node the source
        // tensor's `shape_id` still references.
        for &tid in &tids {
            rt.retain(tid);
        }
        tids.into_iter().map(|tid| Tensor { id: tid }).collect()
    }

    /// Return whether this tensor's data is currently materialized on a
    /// device.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32]);
    /// let realized = t.is_realized();
    /// ```
    #[must_use]
    pub fn is_realized(&self) -> bool {
        RT.lock().is_realized(self.id)
    }

    /// Return the device's capability for the given dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let cap = Tensor::dtype_capability(DType::F32);
    /// ```
    #[must_use]
    pub fn dtype_capability(dtype: DType) -> DTypeCapability {
        RT.lock().supports_dtype(dtype)
    }

    /// Return the first N dimensions of this tensor as scalar IDX_T dim
    /// tensors.
    ///
    /// Each entry is backed by the IR node computing that dimension: a
    /// constant for static dims, a runtime variable for dynamic ones. Use
    /// `.item()` for a concrete integer; pass the tensor directly to stay
    /// symbolic.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[2, 3, 2], [4, 5, 1]]);
    /// let [d1, d2] = t.dims().unwrap();
    /// assert_eq!(d1.item::<i64>(), 2);
    /// assert_eq!(d2.item::<i64>(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `ShapeError` when N exceeds the number of dimensions.
    #[allow(clippy::missing_panics_doc)]
    pub fn dims<const N: usize>(&self) -> Result<[Tensor; N], ZyxError> {
        let symbolic = self.shape();
        let rank = symbolic.len();
        if N > rank {
            Err(ZyxError::shape_error(format!("Requested {N} dims, but tensor only has rank of {}", rank).into()))
        } else {
            Ok(std::array::from_fn(|i| symbolic[i].clone()))
        }
    }

    /// Return the last N dimensions of this tensor as scalar IDX_T dim tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[2, 3, 2], [4, 5, 1]]);
    /// let [d2] = t.rdims().unwrap();
    /// assert_eq!(d2.item::<i64>(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `ShapeError` when N exceeds the number of dimensions.
    pub fn rdims<const N: usize>(&self) -> Result<[Tensor; N], ZyxError> {
        let shape = self.shape();

        if N > shape.len() {
            return Err(ZyxError::shape_error(format!("Requested {N} dims, but tensor only has rank of {}", shape.len()).into()));
        }

        let mut res: [Option<Tensor>; N] = std::array::from_fn(|_| None);
        for (i, d) in shape[shape.len() - N..].iter().enumerate() {
            res[i] = Some(d.clone());
        }
        Ok(res.map(|d| d.unwrap()))
    }

    /// Return the total number of elements as a scalar tensor.
    ///
    /// Fully symbolic: built by multiplying this tensor's dim tensors, so a
    /// dynamic shape yields an expression, not a concrete number. Use
    /// [`Tensor::item`] for a concrete integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[2, 3, 2], [4, 5, 1]]);
    /// assert_eq!(t.numel().item::<i32>(), 6);
    /// ```
    #[must_use]
    pub fn numel(&self) -> Tensor {
        let dims = self.shape();
        if dims.is_empty() {
            let id = RT.lock().new_constant_tensor(Constant::new(1u8));
            return Tensor { id };
        }
        let mut iter = dims.into_iter();
        let mut n = iter.next().expect("dims is non-empty");
        for d in iter {
            let id = RT.lock().binary(n.id, d.id, BOp::Mul).expect("numel: failed to build symbolic mul chain");
            n = Tensor { id };
        }
        n
    }

    /// Return the number of dimensions (rank) of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[2, 3], [4, 1]]);
    /// assert_eq!(t.rank(), 2);
    /// ```
    #[must_use]
    pub fn rank(&self) -> Dim {
        self.resolve_shape().len() as i64
    }

    /// Return the data type of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32]);
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    #[must_use]
    pub fn dtype(&self) -> DType {
        RT.lock().dtype(self.id)
    }

    /// Return the device on which the tensor lives.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32]);
    /// let dev = t.device();
    /// ```
    #[must_use]
    pub fn device(&self) -> Dev {
        RT.lock().device(self.id)
    }

    /// Return whether zyx is currently in training mode.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let training = Tensor::training();
    /// ```
    #[must_use]
    pub fn training() -> bool {
        RT.lock().training
    }

    /// Set the training mode.
    pub fn set_training(training: bool) {
        RT.lock().training = training;
    }

    /// Return whether implicit casting is enabled.
    ///
    /// Implicit casts are enabled by default.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let casts = Tensor::implicit_casts();
    /// ```
    #[must_use]
    pub fn implicit_casts() -> bool {
        RT.lock().implicit_casts
    }

    /// Set implicit casting.
    pub fn set_implicit_casts(implicit_casts: bool) {
        RT.lock().implicit_casts = implicit_casts;
    }

    /// Create a dynamic scalar tensor from a scalar value, not baked into the
    /// kernel.
    ///
    /// The value is bound per launch through backend variable slots (and
    /// excluded from program cache keys), but it is still a concrete value
    /// host-side: dim expressions over variables always fold. See the
    /// Const vs Variable mechanics documented on the symbolic `Expr` type.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let v = Tensor::variable(3.14f32);
    /// ```
    pub fn variable(x: impl Scalar) -> Tensor {
        let id = RT.lock().new_variable_tensor(x);
        Tensor { id }
    }

    /// Extract the scalar value from a scalar tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from(2.5f32);
    /// assert_eq!(t.item::<f32>(), 2.5);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    pub fn item<T: Scalar>(&self) -> T {
        let mut rt = RT.lock();
        let mut data = [T::zero(); 1];
        rt.load(self.id, &mut data).unwrap();
        data[0]
    }

    /// Copy the tensor's data to the host as a `Vec<T>`.
    ///
    /// Dtype-strict: returns a `DTypeError` if the tensor's dtype is not
    /// `T` (cast explicitly with [`Tensor::cast`] first for conversion).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let v: Vec<f32> = t.to_vec().unwrap();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `DTypeError` when the tensor dtype is not `T`.
    pub fn to_vec<T: Scalar>(&self) -> Result<Vec<T>, ZyxError> {
        let numel = self.numel().item::<Dim>() as usize;
        let mut data = vec![T::zero(); numel];
        RT.lock().load(self.id, &mut data)?;
        Ok(data)
    }

    /// Assign the value of `src` to this tensor in-place via a StoreView.
    ///
    /// A StoreView is added to `src`'s kernel writing into this tensor's
    /// buffer; materialization happens when `src`'s kernel is released.
    ///
    /// # Errors
    ///
    /// Returns a `DTypeError` if dtypes differ, a `ShapeError` if shapes
    /// differ, or `GraphTensorNotRealized` if this is an unrealized graph
    /// tensor.
    pub fn assign(self, src: impl Into<Tensor>) -> Result<(), ZyxError> {
        let src = src.into();
        RT.lock().assign(self.id, src.id)
    }

    /// Detach the tensor from the backpropagation graph.
    ///
    /// Returns a new tensor with the same data but without the graph, so the
    /// graph does not grow each iteration of a recurrent loop.
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let mut x = Tensor::randn([8, 8], DType::F32)?;
    /// let z = Tensor::randn([8], DType::F32)?;
    /// for _ in 0..100 {
    ///     x = x.detach()? + &z;
    /// }
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// May return a device error if realizing the tensor fails.
    pub fn detach(self) -> Result<Tensor, ZyxError> {
        // TODO remove realization from here
        let dims: Vec<Tensor> = self.resolve_shape().iter().map(|&d| Tensor::from(d)).collect();
        let shape = if dims.is_empty() { None } else { Some(Tensor::stack(&dims)?) };
        let shape_id = match &shape {
            Some(s) => s.id,
            None => TensorId::NULL,
        };
        let id = match self.dtype() {
            DType::BF16 => {
                let data: Vec<bf16> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::F16 => {
                let data: Vec<f16> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::F32 => {
                let data: Vec<f32> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::F64 => {
                let data: Vec<f64> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::F8E4M3 => {
                let data: Vec<f8e4m3> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::F8E5M2 => {
                let data: Vec<f8e5m2> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::U8 => {
                let data: Vec<u8> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::U16 => {
                let data: Vec<u16> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::U32 => {
                let data: Vec<u32> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::U64 => {
                let data: Vec<u64> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::I8 => {
                let data: Vec<i8> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::I16 => {
                let data: Vec<i16> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::I32 => {
                let data: Vec<i32> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::I64 => {
                let data: Vec<i64> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
            DType::Bool => {
                let data: Vec<bool> = self.try_into()?;
                RT.lock().new_host_tensor(shape_id, data.into())
            }
        }?;
        Ok(Tensor { id })
    }

    /// Create a debug guard that raises the debug mask within a block.
    ///
    /// When the guard is dropped, the mask is reset to the global state set
    /// by the `ZYX_DEBUG` env variable.
    #[must_use]
    pub fn with_debug(debug: DebugMask) -> DebugGuard {
        let guard = DebugGuard { debug: crate::debug_mask() };
        crate::set_debug_mask(debug);
        guard
    }

    /// Manually set the seed for the random number generator.
    ///
    /// Only available when the `rand` feature is enabled.
    pub fn manual_seed(seed: u64) {
        RT.lock().manual_seed(seed);
    }

    /// Create a tensor with the given shape filled with uniform random values
    /// in [0, 1) for float dtypes, or in [0, integer max] for integer dtypes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::rand([2, 3], DType::F32).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory for the
    /// tensor.
    #[allow(clippy::missing_panics_doc, reason = "all panics are checked ahead")]
    pub fn rand(shape: impl IntoIterator<Item = impl Into<Tensor>>, dtype: DType) -> Result<Tensor, ZyxError> {
        let tensors = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &tensors {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "rand shape dim must have dtype IDX_T (i64)");
            }
        }
        let shape = if tensors.is_empty() {
            None
        } else {
            Some(Tensor::stack(&tensors)?)
        };
        {
            let mut rt = RT.lock();
            let n: Dim = match &shape {
                Some(s) => {
                    let expr = match rt.tensors[s.id] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("rand: shape tid {} is not symbolic: {t:?}", s.id),
                    };
                    rt.resolve_symbolic_dims(expr).into_iter().product()
                }
                None => 1,
            };
            let shape_id = match &shape {
                Some(s) => s.id,
                None => TensorId::NULL,
            };
            if dtype.is_float() {
                // TODO later use threefry
                match dtype {
                    DType::BF16 => {
                        let data: Vec<bf16> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::F16 => {
                        let data: Vec<f16> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::F32 => {
                        let data: Vec<f32> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::F64 => {
                        let data: Vec<f64> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::F8E4M3 => {
                        let data: Vec<f8e4m3> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::F8E5M2 => {
                        let data: Vec<f8e5m2> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::U8
                    | DType::U16
                    | DType::U32
                    | DType::U64
                    | DType::I8
                    | DType::I16
                    | DType::I32
                    | DType::I64
                    | DType::Bool => panic!(),
                }
            } else {
                match dtype {
                    DType::U8 => {
                        let data: Vec<u8> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::U16 => {
                        let data: Vec<u16> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::U32 => {
                        let data: Vec<u32> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::U64 => {
                        let data: Vec<u64> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::I8 => {
                        let data: Vec<i8> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::I16 => {
                        let data: Vec<i16> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::I32 => {
                        let data: Vec<i32> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::I64 => {
                        let data: Vec<i64> = (0..n).map(|_| rt.rng.rand()).collect();
                        Ok(Tensor { id: rt.new_host_tensor(shape_id, data.into())? })
                    }
                    DType::Bool => Err(ZyxError::dtype_error("Uniform is not supported for bool".into())),
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 | DType::F8E4M3 | DType::F8E5M2 => {
                        unreachable!()
                    }
                }
            }
        }
    }

    // Initializers
    /// Create a tensor of the given shape sampled from a standard normal
    /// distribution.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::randn([2, 3], DType::F32).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory for the
    /// tensor.
    pub fn randn(shape: impl IntoIterator<Item = impl Into<Tensor>>, dtype: DType) -> Result<Tensor, ZyxError> {
        // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let dims = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &dims {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "randn shape dim must have dtype IDX_T (i64)");
            }
        }
        let mut nshape = Vec::with_capacity(dims.len() + 1);
        nshape.push(Tensor::from(2));
        nshape.extend(dims);
        let src = Tensor::rand(nshape, DType::F32)?;
        let x1 = src.slice(0)?.mul(2f32 * std::f32::consts::PI).cos();
        let x2 = (1f32 - src.slice(1)?).ln().mul(-2f32).sqrt();
        Ok((x1 * x2).cast(dtype))
    }

    /// Sample from the multinomial distribution defined by this tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([0.5f32, 0.5]);
    /// let s = t.multinomial(10, true).unwrap();
    /// assert_eq!(s.dtype(), DType::I32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    #[allow(clippy::missing_panics_doc, reason = "TODO disallow panicking")]
    pub fn multinomial(&self, num_samples: Dim, replacement: bool) -> Result<Tensor, ZyxError> {
        let sh = self.resolve_shape();
        let rank = sh.len();
        debug_assert!((1..=2).contains(&rank) && num_samples > 0, "rank={rank} must be 1 or 2");
        debug_assert!(replacement || num_samples == 1, "no replacement only supports num_samples = 1");
        let weight = if rank == 1 { self.unsqueeze(0)? } else { self.clone() };
        let cw = weight.cumsum(1)?.cast(DType::F32);
        let cdf = &cw / cw.slice((.., -1))?.unsqueeze(1)?;
        let cdf_sh = cdf.resolve_shape();
        let unif_samples = Tensor::rand([num_samples, cdf_sh[0], 1], DType::F32)?;
        let indices = unif_samples.expand([num_samples, cdf_sh[0], cdf_sh[1]])?.cmplt(cdf)?.not().sum([2])?.permute([1, 0])?;
        Ok((if rank == 1 { indices.squeeze([0]) } else { indices }).cast(DType::I32))
    }

    /// Create a tensor of the given shape sampled from a uniform distribution
    /// over the range, cast to `T`. The range start must be less than the end.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::uniform([4, 4], 0.0f32..1.0).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    pub fn uniform<T: Scalar>(
        shape: impl IntoIterator<Item = impl Into<Tensor>>,
        range: impl core::ops::RangeBounds<T>,
    ) -> Result<Tensor, ZyxError> {
        use core::ops::Bound;
        let low: f32 = match range.start_bound() {
            Bound::Included(value) | Bound::Excluded(value) => value.cast(),
            Bound::Unbounded => f32::min_value(),
        };
        let high: f32 = match range.end_bound() {
            Bound::Included(value) | Bound::Excluded(value) => value.cast(),
            Bound::Unbounded => f32::max_value(),
        };
        Ok((Tensor::rand(shape, DType::F32)? * high.sub(low) + low).cast(T::dtype()))
    }

    /// Create a tensor of the given shape of discrete uniform integers in the
    /// range [low, high).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::randint([4, 4], 0..10).unwrap();
    /// assert_eq!(t.dtype(), DType::I32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    pub fn randint<T: Scalar>(
        shape: impl IntoIterator<Item = impl Into<Tensor>>,
        range: impl core::ops::RangeBounds<T> + Clone,
    ) -> Result<Tensor, ZyxError> {
        let dims = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &dims {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "randint shape dim must have dtype IDX_T (i64)");
            }
        }
        let shape = Tensor::stack(&dims)?;
        let mut rt = RT.lock();
        let shape_expr = match rt.tensors[shape.id] {
            TensorData::Symbolic { expr, .. } => expr,
            ref t => panic!("randint: shape tid {} is not symbolic: {t:?}", shape.id),
        };
        let n: Dim = rt.resolve_symbolic_dims(shape_expr).into_iter().product();
        let data: Vec<T> = (0..n).map(|_| rt.rng.range(range.clone())).collect();
        Ok(Tensor { id: rt.new_host_tensor(shape.id, data.into())? })
    }

    /// Create a tensor of the given shape sampled from the Kaiming uniform
    /// distribution, parameterized by `a`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::kaiming_uniform([4, 4], 0.01f32).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    #[allow(clippy::missing_panics_doc)]
    pub fn kaiming_uniform<T: Float>(shape: impl IntoIterator<Item = impl Into<Tensor>>, a: T) -> Result<Tensor, ZyxError> {
        let dims: Vec<Tensor> = shape.into_iter().map(Into::into).collect();
        // Stack outside the RT lock — Tensor::stack locks internally.
        let shape_st = Tensor::stack(&dims).ok();
        let resolved: Vec<Dim> = {
            let rt = RT.lock();
            shape_st
                .as_ref()
                .map(|s| {
                    let expr = match rt.tensors[s.id] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("kaiming_uniform: shape tid {} is not symbolic: {t:?}", s.id),
                    };
                    rt.resolve_symbolic_dims(expr)
                })
                .unwrap_or_default()
        };
        let n = T::from_i64(resolved.iter().skip(1).product::<Dim>().try_into().unwrap());
        let one = T::one();
        let x = Scalar::add(one, Scalar::mul(a, a));
        let two = Scalar::add(one, one);
        let three = Scalar::add(two, one);
        let x = Scalar::div(two, x).sqrt();
        let bound = Scalar::mul(three.sqrt(), Scalar::div(x, n));
        Tensor::uniform(dims, bound.neg()..bound)
    }

    /// Create a tensor of the given shape sampled from the Glorot uniform
    /// distribution in the given dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::glorot_uniform([4, 4], DType::F32).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    #[allow(clippy::cast_precision_loss)]
    pub fn glorot_uniform(shape: impl IntoIterator<Item = impl Into<Tensor>>, dtype: DType) -> Result<Tensor, ZyxError> {
        let dims: Vec<Tensor> = shape.into_iter().map(Into::into).collect();
        // Stack outside the RT lock — Tensor::stack locks internally.
        let shape_st = Tensor::stack(&dims).ok();
        let c = {
            let rt = RT.lock();
            let resolved: Vec<Dim> = shape_st
                .as_ref()
                .map(|s| {
                    let expr = match rt.tensors[s.id] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("glorot_uniform: shape tid {} is not symbolic: {t:?}", s.id),
                    };
                    rt.resolve_symbolic_dims(expr)
                })
                .unwrap_or_default();
            6. / (resolved[0] + resolved.iter().skip(1).product::<Dim>()) as f32
        };
        let mut x = Tensor::uniform(dims, -1f32..1f32)?;
        x = x * c.pow(0.5);
        Ok(x.cast(dtype))
    }

    /// Create a tensor of the given shape filled with zeros in the given dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([2, 2], DType::F32);
    /// assert_eq!(t.to_vec::<f32>().unwrap(), vec![0.0f32, 0.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    pub fn zeros(shape: impl IntoIterator<Item = impl Into<Tensor>>, dtype: DType) -> Tensor {
        let dims = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &dims {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "zeros shape dim must have dtype IDX_T (i64)");
            }
        }
        let shape = if dims.is_empty() {
            None
        } else {
            Some(Tensor::stack(&dims).unwrap())
        };
        let id = {
            let mut rt = RT.lock();
            let shape_id = shape.as_ref().map(|s| s.id).unwrap_or(TensorId::NULL);
            rt.new_full(shape_id, dtype.zero_constant())
        };
        Tensor { id }
    }

    /// Create a tensor filled with zeros with the same shape and dtype as
    /// `input`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0]);
    /// let z = Tensor::zeros_like(t);
    /// assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0f32, 0.0]);
    /// ```
    #[must_use]
    pub fn zeros_like(input: impl Into<Tensor>) -> Tensor {
        let input = input.into();
        Tensor::zeros(input.resolve_shape(), input.dtype())
    }

    /// Create a tensor of the given shape filled with ones in the given dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::ones([2], DType::F32);
    /// assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0f32, 1.0]);
    /// ```
    #[must_use]
    pub fn ones(shape: impl IntoIterator<Item = impl Into<Tensor>>, dtype: DType) -> Tensor {
        let dims = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &dims {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "ones shape dim must have dtype IDX_T (i64)");
            }
        }
        let shape = if dims.is_empty() {
            None
        } else {
            Some(Tensor::stack(&dims).unwrap())
        };
        let id = {
            let mut rt = RT.lock();
            let shape_id = shape.as_ref().map(|s| s.id).unwrap_or(TensorId::NULL);
            rt.new_full(shape_id, dtype.one_constant())
        };
        Tensor { id }
    }

    /// Create a tensor filled with ones with the same shape and dtype as
    /// `input`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0]);
    /// let o = Tensor::ones_like(t);
    /// assert_eq!(o.to_vec::<f32>().unwrap(), vec![1.0f32, 1.0]);
    /// ```
    #[must_use]
    pub fn ones_like(input: impl Into<Tensor>) -> Tensor {
        let input = input.into();
        Tensor::ones(input.resolve_shape(), input.dtype())
    }

    /// Create a tensor of the given shape filled with `value`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::full([2], 3.14f32);
    /// assert_eq!(t.to_vec::<f32>().unwrap(), vec![3.14f32, 3.14]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    #[allow(clippy::missing_panics_doc)]
    pub fn full(shape: impl IntoIterator<Item = impl Into<Tensor>>, value: impl Scalar) -> Tensor {
        let dims = Self::cast_to_shape(shape);
        {
            let rt = RT.lock();
            for t in &dims {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "full shape dim must have dtype IDX_T (i64)");
            }
        }
        let shape = if dims.is_empty() {
            None
        } else {
            Some(Tensor::stack(&dims).unwrap())
        };
        let id = {
            let mut rt = RT.lock();
            let shape_id = shape.as_ref().map(|s| s.id).unwrap_or(TensorId::NULL);
            rt.new_full(shape_id, Constant::new(value))
        };
        Tensor { id }
    }

    /// Create a square tensor with ones on the main diagonal and zeros
    /// elsewhere, in the given dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::eye(3, DType::F32);
    /// assert_eq!(t.to_vec::<f32>().unwrap(),
    ///     vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn eye(n: Dim, dtype: DType) -> Tensor {
        Tensor::ones(vec![n, 1], dtype)
            .pad_zeros([(0i64, 0i64), (0i64, i64::try_from(n).unwrap())])
            .unwrap()
            .reshape([n + 1, n])
            .unwrap()
            .slice((..-1, ..))
            .unwrap()
    }

    /// Create a tensor of range values from `start` up to (but not including)
    /// `stop`, incrementing by `step`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::arange(0f32, 5.0, 1.0).unwrap();
    /// assert_eq!(t.to_vec::<f32>().unwrap(), vec![0.0f32, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a device error if the device cannot allocate memory.
    #[allow(clippy::missing_panics_doc)]
    pub fn arange<T: Scalar>(start: T, stop: T, step: T) -> Result<Tensor, ZyxError> {
        // if (stop-start)/step <= 0: return Tensor([], dtype=dtype, **kwargs)
        // return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs)._cumsum() + (start - step)).cast(dtype)
        //println!("Arange {start:?}, {stop:?}, {step:?}");
        let n: i64 = stop.sub(start).div(step).cast();
        let x = Tensor::full([Dim::try_from(n).unwrap()], step);
        let x = x.cumsum(0)?;
        Ok(x + start - step)
    }

    /// Create a tensor from a flat vector and the given shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    /// assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0f32, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an allocation or backend initialization error.
    pub fn from_vec<T: Scalar>(data: Vec<T>, shape: impl IntoIterator<Item = impl Into<Tensor>>) -> Result<Tensor, ZyxError> {
        let dims: Vec<Tensor> = shape.into_iter().map(Into::into).collect();
        let shape = Tensor::stack(&dims)?;
        let id = RT.lock().new_host_tensor(shape.id, data.into_boxed_slice())?;
        Ok(Tensor { id })
    }

    // unary
    /// Cast the tensor to a new dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32]);
    /// let u = t.cast(DType::F64);
    /// assert_eq!(u.to_vec::<f64>().unwrap(), vec![1.0f64]);
    /// ```
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor {
        let id = RT.lock().cast(self.id, dtype);
        return Tensor { id };
    }

    /// Reinterpret the tensor's raw bits as `dtype` without a value conversion.
    ///
    /// The current dtype and `dtype` must have equal bit widths; interpreting
    /// to `Bool` is rejected since arbitrary bit patterns are not valid `bool`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::from([1.0f32]);
    /// let u = t.bitcast(DType::U32).unwrap();
    /// assert_eq!(u.to_vec::<u32>().unwrap(), vec![0x3f800000u32]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `DTypeError` when bit widths differ or `dtype` is `Bool`, or
    /// a device error on allocation failure.
    #[allow(clippy::missing_panics_doc)]
    pub fn bitcast(&self, dtype: DType) -> Result<Tensor, ZyxError> {
        if self.dtype().bit_size() != dtype.bit_size() {
            return Err(ZyxError::dtype_error(
                format!(
                    "bitcast requires equal bit widths: {} ({} bits) -> {} ({} bits)",
                    self.dtype(),
                    self.dtype().bit_size(),
                    dtype,
                    dtype.bit_size()
                )
                .into(),
            ));
        }
        if dtype == DType::Bool {
            return Err(ZyxError::dtype_error(
                "bitcast to Bool is not allowed, arbitrary bits are not valid bool values.".into(),
            ));
        }
        let id = RT.lock().bitcast(self.id, dtype);
        Ok(Tensor { id })
    }

    /// Apply dropout to the tensor with the given probability.
    ///
    /// During training, elements are randomly zeroed with the given
    /// probability; in inference the input is returned scaled by 1/(1-p).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0]);
    /// let d = t.dropout(0.5f32);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn dropout<P: Scalar + Float>(&self, probability: P) -> Tensor {
        if Tensor::training() {
            Tensor::from(probability).cmplt(Tensor::rand(self.resolve_shape(), P::dtype()).unwrap()).unwrap() * self.clone()
        } else {
            self / P::one().sub(probability)
        }
    }

    /// Linearly interpolate between this tensor and `target` by a `weight`.
    ///
    /// Computes `input * (1 - weight) + target * weight`, returning a tensor
    /// with the same dtype as the input.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let input = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let target = Tensor::from([2.0, 4.0, 6.0]);
    /// let r = input.interpolate(&target, 0.5);
    /// assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.5f32, 3.0, 4.5]);
    /// ```
    #[must_use]
    pub fn interpolate(&self, target: &Tensor, weight: f32) -> Tensor {
        let input = self.float_cast().unwrap();
        let target = target.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Linear interpolation: input * (1 - weight) + target * weight
        let result = &input * (1.0 - weight) + &target * weight;

        result.cast(original_dtype)
    }

    /// Compute the Smooth L1 loss between this tensor and `target`.
    ///
    /// Combines L1 and L2 loss: `0.5 * (x - y)^2` when `|x - y| <= 1`, otherwise
    /// `|x - y| - 0.5`. Returns the same dtype as the input.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let p = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let t = Tensor::from([1.5, 2.5, 2.8]);
    /// let loss = p.smooth_l1_loss(&t);
    /// assert!((loss.item::<f32>() - 0.27).abs() < 1e-4);
    /// ```
    #[must_use]
    pub fn smooth_l1_loss(&self, target: &Tensor) -> Tensor {
        let input = self.float_cast().unwrap();
        let target = target.float_cast().unwrap();
        let original_dtype = self.dtype();

        let diff = &input - &target;
        let abs_diff = diff.abs();
        let mask = abs_diff.cmplt(1.0f32).unwrap();

        // Quadratic region: 0.5 * (x - y)²
        let quadratic_loss = 0.5f32 * &diff * &diff;

        // Linear region: |x - y| - 0.5
        let linear_loss = abs_diff - 0.5f32;

        // Combine based on the mask
        let loss = mask.clone() * quadratic_loss + mask.not() * linear_loss;

        // Sum all elements to get the total loss
        let total_loss = loss.sum([0]).unwrap();

        total_loss.cast(original_dtype)
    }

    /// Compute the Huber loss between this tensor and `target` with threshold
    /// `delta`.
    ///
    /// `0.5 * (x - y)^2` when `|x - y| <= delta`, otherwise
    /// `delta * |x - y| - 0.5 * delta^2`. Returns the same dtype as the input.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let p = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let t = Tensor::from([1.5, 2.5, 2.8]);
    /// let loss = p.huber_loss(&t, 1.0f32);
    /// assert!((loss.item::<f32>() - 0.27).abs() < 1e-4);
    /// ```
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn huber_loss(&self, target: &Tensor, delta: impl Scalar) -> Tensor {
        let input = self.float_cast().unwrap();
        let target = target.float_cast().unwrap();
        let original_dtype = self.dtype();

        // PyTorch huber loss formula:
        // huber_loss(x, y) = {
        //     0.5 * (x - y)²,                  if |x - y| ≤ δ
        //     δ * |x - y| - 0.5 * δ²,          otherwise
        // }

        let diff = input - target;
        let abs_diff = diff.abs();

        // Cast delta to the same dtype as input to follow PyTorch behavior
        let delta_tensor = Tensor::from(delta).float_cast().unwrap().cast(original_dtype);

        // Create mask for quadratic region (|diff| ≤ delta)
        let quadratic_mask = abs_diff.cmplt(delta_tensor.clone()).unwrap();

        // Quadratic loss: 0.5 * diff²
        let quadratic_loss = 0.5f32 * diff.clone() * diff;

        // Linear loss: delta * |diff| - 0.5 * delta²
        let linear_loss = delta_tensor.clone() * abs_diff - 0.5f32 * delta_tensor.clone() * delta_tensor;

        // Combine: use quadratic_loss where |diff| ≤ delta, linear_loss otherwise
        let result = quadratic_mask.clone() * quadratic_loss + quadratic_mask.not() * linear_loss;

        // Sum all elements to get total loss (like smooth_l1_loss does)
        let total_loss = result.sum([0]).unwrap();

        total_loss.cast(original_dtype)
    }

    // movement
    /// Expand this tensor to the given shape by broadcasting singleton
    /// dimensions.
    ///
    /// A dimension of `1` in `self` is broadcast to any target size; matching
    /// non-1 dimensions must be equal. Returns a view with the target shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([2, 3], DType::F32);
    /// assert_eq!(t.expand([4, 2, 3]).unwrap().shape(), [4, 2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if `self` cannot be expanded into the shape.
    pub fn expand<D: Into<Tensor>>(&self, shape: impl IntoIterator<Item = D>) -> Result<Tensor, ZyxError> {
        let mut tensors = Self::cast_to_shape(shape);
        // Shape/dim tensors must be IDX_T (i64) after normalization.
        let resolved: Vec<Option<i64>> = {
            let rt = RT.lock();
            for t in &tensors {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "expand dim tensor must have dtype IDX_T (i64)");
            }
            // Resolve -1 (keep dim) user-side: runtime knows nothing of sentinels.
            // Torch semantics: -1 leaves the aligned dimension unchanged; the
            // target shape may add leading dimensions (target rank >= input rank).
            tensors
                .iter()
                .map(|t| {
                    rt.resolve_symbolic(t.id).and_then(|c| match c.cast(DType::I64) {
                        crate::dtype::Constant::I64(b) => Some(i64::from_le_bytes(b)),
                        _ => None,
                    })
                })
                .collect()
        };
        if resolved.iter().any(|&d| d.is_some_and(|v| v < -1)) {
            return Err(ZyxError::shape_error("Expanded dimensions must be >= -1.".into()));
        }
        if resolved.iter().any(|&d| d == Some(-1)) {
            let own_dims = self.shape();
            let prepend = tensors.len() as i64 - own_dims.len() as i64;
            if prepend < 0 {
                return Err(ZyxError::shape_error(
                    format!("Can't expand a tensor of rank {} into a tensor of rank {}", own_dims.len(), tensors.len()).into(),
                ));
            }
            for (i, &r) in resolved.iter().enumerate() {
                if r == Some(-1) {
                    let axis = i as i64 - prepend;
                    if axis < 0 {
                        return Err(ZyxError::shape_error("The size -1 is invalid for an added dimension in expand.".into()));
                    }
                    tensors[i] = own_dims[axis as usize].clone();
                }
            }
        }
        let shape = Tensor::stack(&tensors)?;
        let id = RT.lock().expand(self.id, shape.id)?;
        Ok(Tensor { id })
    }

    /// Expand the tensor along `axis` to the new size `dim`.
    ///
    /// Replaces the given axis with `dim`, broadcasting a singleton dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[2.0f32], [3.0]]);
    /// let t2 = t.expand_axis(1, 5).unwrap();
    /// assert_eq!(t2.shape(), [2, 5]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axis is out of bounds.
    pub fn expand_axis(&self, axis: Axis, dim: Dim) -> Result<Tensor, ZyxError> {
        let rank = self.resolve_shape().len();
        let axis = into_axis(axis, rank as u32)? as usize;
        let mut dims = self.shape();
        // Only the NEW dim is a fresh constant — the user passed it concretely.
        // All other dims stay symbolic so variable-backed dims survive.
        dims[axis] = Tensor::from(dim);
        let shape = Tensor::stack(&dims)?;
        let id = RT.lock().expand(self.id, shape.id)?;
        Ok(Tensor { id })
    }

    /// Permute the axes of this tensor according to `axes`.
    ///
    /// The axes must be a permutation of the original axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::rand([3, 4], DType::I64).unwrap();
    /// let p = t.permute([1, 0]).unwrap();
    /// assert_eq!(p.shape(), [4, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axes do not match the tensor's rank.
    pub fn permute(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let axes = into_axes(axes, rank as u32)?;
        if rank != axes.len() as i64 {
            return Err(ZyxError::shape_error(
                format!("Axes has rank {}, but tensor has rank {}. It must be the same for permute.", axes.len(), rank).into(),
            ));
        }
        let id = RT.lock().permute(self.id, axes);
        Ok(Tensor { id })
    }

    /// Flip the tensor along the given axes, reversing the order of elements.
    ///
    /// Works the same as `torch.flip`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3]);
    /// let flipped = t.flip([0]).unwrap();
    /// assert_eq!(flipped.to_vec::<i32>().unwrap(), vec![3i32, 2, 1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axes list is empty or an axis is out of range.
    pub fn flip(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let mut axes: Vec<UAxis> = axes.into_iter().map(|a| into_axis(a, rank as u32)).collect::<Result<_, _>>()?;
        if axes.is_empty() {
            return Err(ZyxError::shape_error(format!("Axes must not be empty for a tensor of rank {rank}").into()));
        }
        axes.sort_unstable();
        axes.dedup();
        let id = RT.lock().flip(self.id, axes)?;
        Ok(Tensor { id })
    }

    /// Pad a single axis with zeros: `lp` zeros on the left, up to total
    /// length `len` (right padding is `len - lp - orig_len`). `lp` and `len`
    /// are scalar tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([2, 3], DType::F32);
    /// let p = t.pad_zeros_axis(0, Tensor::from(1i64), Tensor::from(6i64)).unwrap();
    /// assert_eq!(p.shape(), [6, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axis is out of range or the padding is invalid.
    #[track_caller]
    pub fn pad_zeros_axis(&self, axis: UAxis, lp: Tensor, len: Tensor) -> Result<Tensor, ZyxError> {
        let lp = lp.cast(IDX_T);
        let len = len.cast(IDX_T);
        let mut rt = RT.lock();
        debug_assert_eq!(rt.dtype(lp.id), DType::I64, "pad_zeros_axis lp must have dtype IDX_T (i64)");
        debug_assert_eq!(rt.dtype(len.id), DType::I64, "pad_zeros_axis len must have dtype IDX_T (i64)");
        let id = rt.pad_zeros(self.id, axis, lp.id, len.id);
        Ok(Tensor { id })
    }

    /// Applies `(lp, len)` zero padding to a single axis, validating against
    /// this tensor's shape.
    fn pad_axis(&self, axis: UAxis, l: i64, r: i64) -> Result<Tensor, ZyxError> {
        let shape = self.resolve_shape();
        let orig = shape[axis as usize] as i64;
        let removed = (if l < 0 { -l } else { 0 }) + (if r < 0 { -r } else { 0 });
        if orig + l + r < 0 || removed >= orig {
            return Err(ZyxError::shape_error(format!("Invalid padding left={l}, right={r} on dimension size {orig}").into()));
        }
        let lp = Tensor::from(l);
        let len = Tensor::from((orig + l + r) as i64);
        self.pad_zeros_axis(axis, lp, len)
    }

    /// Pad this tensor with zeros using per-dimension `(left, right)` padding
    /// tuples (missing higher dims are front-padded).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3]);
    /// let p = t.pad_zeros([(1, 2)]).unwrap();
    /// assert_eq!(p.to_vec::<i32>().unwrap(), vec![0i32, 1, 2, 3, 0, 0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the padding exceeds the tensor's rank or is
    /// invalid.
    #[allow(clippy::missing_panics_doc)]
    #[track_caller]
    pub fn pad_zeros(&self, padding: impl IntoIterator<Item = (i64, i64)>) -> Result<Tensor, ZyxError> {
        let mut padding: Vec<(i64, i64)> = padding.into_iter().collect();
        let rank = self.resolve_shape().len();

        if padding.len() > rank {
            return Err(ZyxError::shape_error(
                format!("Padding with {} dimensions, but tensor only has rank {rank}", padding.len()).into(),
            ));
        }
        padding.extend(std::iter::repeat_n((0i64, 0i64), rank - padding.len()));

        let mut cur = self.clone();
        for (i, &(l, r)) in padding.iter().enumerate() {
            if l != 0 || r != 0 {
                cur = cur.pad_axis(i as UAxis, l, r)?;
            }
        }
        Ok(cur)
    }

    /// Pad this tensor with zeros using per-dimension `(left, right)` padding
    /// tuples, applied in reverse order (last dim first).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3]);
    /// let p = t.rpad_zeros([(1, 2)]).unwrap();
    /// assert_eq!(p.to_vec::<i32>().unwrap(), vec![0i32, 1, 2, 3, 0, 0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the padding exceeds the tensor's rank or is
    /// invalid.
    #[allow(clippy::missing_panics_doc)]
    #[track_caller]
    pub fn rpad_zeros(&self, padding: impl IntoIterator<Item = (i64, i64)>) -> Result<Tensor, ZyxError> {
        let mut padding: Vec<(i64, i64)> = padding.into_iter().collect();
        let rank = self.resolve_shape().len();

        if padding.len() > rank {
            return Err(ZyxError::shape_error(
                format!("Padding with {} dimensions, but tensor only has rank {rank}", padding.len()).into(),
            ));
        }

        padding.extend(std::iter::repeat_n((0i64, 0i64), rank - padding.len()));
        padding.reverse();

        let mut cur = self.clone();
        for (i, &(l, r)) in padding.iter().enumerate() {
            if l != 0 || r != 0 {
                cur = cur.pad_axis(i as UAxis, l, r)?;
            }
        }
        Ok(cur)
    }

    /// Pad this tensor by a constant value, given per-dimension `(left,
    /// right)` padding tuples (negative values crop).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let x = Tensor::from([[2i32, 3], [4, 1]]);
    /// let z = x.pad([(0, 0), (1, 2)], 0i32).unwrap();
    /// assert_eq!(z.to_vec::<i32>().unwrap(),
    ///     vec![0i32, 2, 3, 0, 0, 0, 4, 1, 0, 0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the padding is invalid.
    #[allow(clippy::missing_panics_doc)]
    pub fn pad(&self, padding: impl IntoIterator<Item = (i64, i64)>, value: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let dtype = self.dtype();
        let value: Tensor = value.into();
        let padding: Vec<(i64, i64)> = padding.into_iter().collect();
        let mut sh = self.resolve_shape();
        if value.dtype() != dtype {
            return Err(ZyxError::dtype_error(
                format!("Cannot pad tensor with dtype {} with value of dtype {}", dtype, value.dtype()).into(),
            ));
        }
        if !padding.len() <= sh.len() && padding.iter().zip(sh.iter().rev()).all(|(&(lp, rp), &d)| if lp < 0 { Dim::try_from(-lp).unwrap() <= d } else { true } && if rp < 0 { Dim::try_from(-rp).unwrap() <= d } else { true }) {
            return Err(ZyxError::shape_error(format!("Cannot pad tensor with shape {sh:?} with padding {padding:?}").into()));
        }
        let t0 = self.pad_zeros(padding.clone())?;
        let ones = Tensor::ones(sh.clone(), dtype);
        crate::shape::pad(&mut sh, &padding);
        let zeros = Tensor::zeros(sh, dtype);
        Ok(t0 + ones.pad_zeros(padding)?.where_(zeros, value)?)
    }

    /// Collects a shape iterator into `Tensor`s, casting each to `IDX_T`.
    fn cast_to_shape(shape: impl IntoIterator<Item = impl Into<Tensor>>) -> Vec<Tensor> {
        shape.into_iter().map(|x| x.into().cast(IDX_T)).collect()
    }

    /// Reshape this tensor to the given shape while preserving its total
    /// number of elements.
    ///
    /// A single `-1` in the shape infers that dimension automatically. All
    /// other dimensions must be >= 1 and the total element count must match.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3, 4]);
    /// let r = t.reshape([2, 2]).unwrap();
    /// assert_eq!(r.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the new shape is incompatible.
    pub fn reshape<D: Into<Tensor>>(&self, shape: impl IntoIterator<Item = D>) -> Result<Tensor, ZyxError> {
        let mut tensors = Self::cast_to_shape(shape);
        // Shape/dim tensors must be IDX_T (i64) after normalization.
        let resolved: Vec<Option<i64>> = {
            let rt = RT.lock();
            for t in &tensors {
                debug_assert_eq!(rt.dtype(t.id), DType::I64, "reshape dim tensor must have dtype IDX_T (i64)");
            }
            // Resolve -1 (infer) user-side: runtime knows nothing of sentinels.
            tensors
                .iter()
                .map(|t| {
                    rt.resolve_symbolic(t.id).and_then(|c| match c.cast(DType::I64) {
                        crate::dtype::Constant::I64(b) => Some(i64::from_le_bytes(b)),
                        _ => None,
                    })
                })
                .collect()
        };
        let infer_count = resolved.iter().filter(|&&d| d == Some(-1)).count();
        if infer_count > 1 {
            return Err(ZyxError::shape_error("Can only infer one dimension (-1).".into()));
        }
        if resolved.iter().any(|&d| d.is_some_and(|v| v < -1)) {
            return Err(ZyxError::shape_error("Reshape dimensions must be >= -1.".into()));
        }
        // 0 is reserved as the internal inferred-dim marker; users infer with -1.
        if resolved.iter().any(|&d| d == Some(0)) {
            return Err(ZyxError::shape_error("Reshape dimensions must be nonzero; use -1 to infer a dimension.".into()));
        }
        if infer_count == 1 {
            // Symbolic inference: build `numel / product(others)` as a dim-op
            // expression over the input's dim tensors, so no concrete value is
            // ever needed at construction time (a symbolic seq dim stays
            // symbolic). Const inputs still fold later during resolution.
            let own_dims = self.shape();
            if own_dims.is_empty() {
                return Err(ZyxError::shape_error("Cannot infer dimension (-1): tensor has rank zero.".into()));
            }
            let mut numel: Option<Tensor> = None;
            for d in &own_dims {
                numel = Some(match numel {
                    None => d.clone(),
                    Some(acc) => &acc * d,
                });
            }
            let numel = numel.expect("reshape: own_dims is non-empty");
            let mut divisor: Option<Tensor> = None;
            for (t, r) in tensors.iter().zip(&resolved) {
                if *r != Some(-1) {
                    divisor = Some(match divisor {
                        None => t.clone(),
                        Some(acc) => &acc * t,
                    });
                }
            }
            // `reshape([-1])` has no other dims: inferred is just numel.
            let mut divisor_opt: Option<Tensor> = None;
            for (t, r) in tensors.iter().zip(&resolved) {
                if *r != Some(-1) {
                    // Dim arithmetic is IDX_T; normalize user-provided int widths.
                    let term = t.clone().cast(DType::I64);
                    divisor_opt = Some(match divisor_opt {
                        None => term,
                        Some(acc) => &acc * &term,
                    });
                }
            }
            let inferred = match divisor_opt {
                Some(divisor) => &numel / &divisor,
                None => numel.clone(),
            };
            for (t, r) in tensors.iter_mut().zip(&resolved) {
                if *r == Some(-1) {
                    *t = inferred.clone();
                }
            }
        }
        let shape = Tensor::stack(&tensors)?;
        let id = RT.lock().reshape(self.id, shape.id)?;
        Ok(Tensor { id })
    }

    /// Transpose (swap) the last two dimensions of this tensor.
    ///
    /// A rank-1 tensor is reshaped to shape `[n, 1]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// assert_eq!(t.t().shape(), &[2, 2]);
    /// ```
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn t(&self) -> Tensor {
        let rank = self.rank();
        if rank == 1 {
            let n = self.numel();
            return self.reshape([n, Tensor::from(1)]).unwrap();
        }
        let mut axes: Vec<Axis> = (0..Axis::try_from(rank).unwrap()).collect();
        axes.swap((rank - 1) as usize, (rank - 2) as usize);
        self.permute(axes).unwrap()
    }

    /// Transpose the two dimensions `dim0` and `dim1` of this tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[[1i32, 2]], [[3, 4]]]);
    /// let tr = t.transpose(0, -1).unwrap();
    /// assert_eq!(tr.to_vec::<i32>().unwrap(), vec![1, 3, 2, 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if an axis is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn transpose(&self, dim0: Axis, dim1: Axis) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        if (dim0 < 0 && Dim::try_from(-dim0).unwrap() > rank) || (dim0 >= 0 && Dim::try_from(dim0).unwrap() >= rank) {
            return Err(ZyxError::shape_error(
                format!("Cannot transpose dimensions {dim0} and {dim1}, {dim0} is greater than rank {rank}").into(),
            ));
        }
        if (dim1 < 0 && Dim::try_from(-dim1).unwrap() > rank) || (dim1 >= 0 && Dim::try_from(dim1).unwrap() >= rank) {
            return Err(ZyxError::shape_error(
                format!("Cannot transpose dimensions {dim0} and {dim1}, {dim1} is greater than rank {rank}").into(),
            ));
        }
        let mut axes: Vec<Axis> = (0..Axis::try_from(rank).unwrap()).collect();
        axes.swap(into_axis(dim0, rank as UAxis)? as usize, into_axis(dim1, rank as UAxis)? as usize);
        self.permute(axes)
    }

    // reduce
    /// Compute the log-softmax of this tensor along the given axes.
    ///
    /// First subtracts the max along the axes (for numerical stability), then
    /// computes `m - ln(sum(exp(m)))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let x = Tensor::from([2.0f32, 3.0, 4.0]);
    /// let v = x.ln_softmax([]).unwrap().to_vec::<f32>().unwrap();
    /// assert!((v[1] - -1.4076).abs() < 1e-4);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axes are invalid.
    #[allow(clippy::missing_panics_doc)]
    pub fn ln_softmax(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let m = self - self.max_keepdim(axes.clone())?;
        Ok(&m - m.exp().sum_keepdim(axes)?.ln())
    }

    /// Compute the softmax of this tensor along the given axes.
    ///
    /// Divides by the sum of `exp(x)` along the axes, shifted by the max for
    /// numerical stability.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let v = t.softmax([]).unwrap().to_vec::<f32>().unwrap();
    /// assert!((v[0] - 0.09003).abs() < 1e-4);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axes are invalid.
    pub fn softmax(&self, axes: impl IntoIterator<Item = Axis>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let e = (self - self.max_keepdim(axes.clone())?).exp();
        Ok(&e / e.sum_keepdim(axes)?)
    }

    // binary
    /// Compute the dot product (matmul) of this tensor and `rhs`.
    ///
    /// Contracts the last dimension of `self` with the first dimension of `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let b = Tensor::from([[5.0f32, 6.0], [7.0, 8.0]]);
    /// let c = a.dot(b).unwrap();
    /// assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0f32, 22.0, 43.0, 50.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the last/first dimensions do not match.
    pub fn dot(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let rhs = rhs.into();
        let org_y_shape = rhs.resolve_shape();
        let y = rhs.t();
        // User-side validation resolves (always resolvable, errors on
        // mismatch); the computation's reshapes compose the inputs' dim
        // tensors so symbolic dims stay symbolic (no recompile per value).
        let xres = self.resolve_shape();
        let yres = y.resolve_shape();
        let xshape = self.shape();
        let yshape = y.shape();
        let xrank = xres.len();
        let yrank = yres.len();
        if xres[xrank - 1] != yres[yrank - 1] {
            return Err(ZyxError::ShapeError(format!("Cannot dot tensors with shapes {xres:?} and {org_y_shape:?}").into()));
        }
        let mut x_shape: Vec<Tensor> = xshape[..xrank - 1].to_vec();
        x_shape.push(Tensor::from(1i64));
        x_shape.push(xshape[xrank - 1].clone());
        let mut y_shape: Vec<Tensor> = yshape[..yrank.saturating_sub(2)].to_vec();
        y_shape.push(Tensor::from(1i64));
        y_shape.extend(yshape[yrank - yrank.min(2)..].iter().cloned());
        let mut out_shape: Vec<Tensor> = xshape[..xrank - 1].to_vec();
        out_shape.push(yshape[yrank - 2].clone());
        (self.reshape(x_shape)? * y.reshape(y_shape)?).sum([-1])?.reshape(out_shape)
    }

    /// Compute the matmul of this tensor and `rhs` in the output dtype
    /// `out_dtype`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);
    /// let c = a.dot_dtype(b, DType::F32).unwrap();
    /// assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0f32, 22.0, 43.0, 50.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have incompatible shapes.
    pub fn dot_dtype(&self, rhs: impl Into<Tensor>, out_dtype: DType) -> Result<Tensor, ZyxError> {
        let rhs: Tensor = rhs.into();
        let org_y_shape = rhs.resolve_shape();
        let y = rhs.t();
        // See dot: resolve for validation, compose dim tensors for computation.
        let xres = self.resolve_shape();
        let yres = y.resolve_shape();
        let xshape = self.shape();
        let yshape = y.shape();
        let xrank = xres.len();
        let yrank = yres.len();
        if xres[xrank - 1] != yres[yrank - 1] {
            return Err(ZyxError::ShapeError(format!("Cannot dot tensors with shapes {xres:?} and {org_y_shape:?}").into()));
        }
        let mut x_shape: Vec<Tensor> = xshape[..xrank - 1].to_vec();
        x_shape.push(Tensor::from(1i64));
        x_shape.push(xshape[xrank - 1].clone());
        let mut y_shape: Vec<Tensor> = yshape[..yrank.saturating_sub(2)].to_vec();
        y_shape.push(Tensor::from(1i64));
        y_shape.extend(yshape[yrank - yrank.min(2)..].iter().cloned());
        let mut out_shape: Vec<Tensor> = xshape[..xrank - 1].to_vec();
        out_shape.push(yshape[yrank - 2].clone());
        (self.reshape(x_shape)?.cast(out_dtype) * y.reshape(y_shape)?.cast(out_dtype)).sum([-1])?.reshape(out_shape)
    }

    /// Compute the matrix multiplication of this tensor and `rhs` (alias of
    /// `dot`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let b = Tensor::from([[5.0f32, 6.0], [7.0, 8.0]]);
    /// assert_eq!(a.matmul(b).unwrap().to_vec::<f32>().unwrap(),
    ///     vec![19.0f32, 22.0, 43.0, 50.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have incompatible shapes.
    pub fn matmul(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        self.dot(rhs)
    }

    /// Element-wise raise `self` to the power `exponent`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1.0f32, 2.0]);
    /// assert_eq!(a.pow(2.0).unwrap().to_vec::<f32>().unwrap(),
    ///     vec![1.0f32, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are not broadcastable.
    pub fn pow(&self, exponent: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        //Ok((self.log2() * exponent).exp2())
        let (x, y) = Tensor::broadcast(self.clone(), exponent)?;
        let id = RT.lock().binary(x.id, y.id, BOp::Pow)?;
        Ok(Tensor { id })
    }

    /// Element-wise logical AND of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([true, true, false]);
    /// let b = Tensor::from([true, false, true]);
    /// assert_eq!(a.logical_and(b).unwrap().to_vec::<bool>().unwrap(),
    ///     vec![true, false, false]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are not broadcastable.
    pub fn logical_and(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, BOp::And)?;
        Ok(Tensor { id })
    }

    /// Element-wise logical OR of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([true, true, false]);
    /// let b = Tensor::from([true, false, true]);
    /// assert_eq!(a.logical_or(b).unwrap().to_vec::<bool>().unwrap(),
    ///     vec![true, true, true]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are not broadcastable.
    pub fn logical_or(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, BOp::Or)?;
        Ok(Tensor { id })
    }

    /// Element-wise equality of `self` and `rhs` (boolean mask of `self == rhs`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1i32, 2, 3]);
    /// let b = Tensor::from([1i32, 9, 3]);
    /// assert_eq!(a.equal(b).unwrap().to_vec::<bool>().unwrap(),
    ///     vec![true, false, true]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are not broadcastable.
    pub fn equal(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, BOp::Eq)?;
        let x = Tensor { id };
        Ok(x)
    }

    /// Element-wise inequality of `self` and `rhs` (boolean mask of
    /// `self != rhs`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1i32, 2, 3]);
    /// let b = Tensor::from([1i32, 9, 3]);
    /// assert_eq!(a.ne(b).unwrap().to_vec::<bool>().unwrap(),
    ///     vec![false, true, false]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are not broadcastable.
    pub fn ne(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, BOp::NotEq)?;
        let x = Tensor { id };
        Ok(x)
    }

    /// Boolean mask of elements of `self` that are nonzero.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([0i32, 5, -1]);
    /// assert_eq!(a.nonzero().to_vec::<bool>().unwrap(),
    ///     vec![false, true, true]);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn nonzero(&self) -> Tensor {
        let y = Tensor::from(0).cast(self.dtype()).expand(self.resolve_shape()).unwrap();
        let id = RT.lock().binary(self.id, y.id, BOp::NotEq).unwrap();
        Tensor { id }
    }

    // ternary
    /// Ternary select: element-wise `self ? if_true : if_false` (using `self`
    /// as the boolean condition).
    ///
    /// # Note
    ///
    /// Implemented branchlessly as `cond * if_true + (1 - cond) * if_false`;
    /// this yields `NaN` on a `0 * ±inf` product — do not pass infinities.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let cond = Tensor::from([true, false]);
    /// let a = cond.where_(Tensor::from([1i32, 2]), Tensor::from([3, 4])).unwrap();
    /// assert_eq!(a.to_vec::<i32>().unwrap(), vec![1, 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have non broadcastable shapes.
    // TODO: possibly for some models in the future a ternary where op will be
    // needed (graph node + kernel IR + backends); the branchless decomposition
    // below cannot support ±inf values.
    #[allow(clippy::missing_panics_doc)]
    pub fn where_(&self, if_true: impl Into<Tensor>, if_false: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let if_true = if_true.into();
        let if_false = if_false.into();
        let dtype = if_true.dtype();
        let x = self.cast(dtype);
        let (if_true, if_false) = Tensor::broadcast(if_true, if_false)?;
        Ok(x.clone() * if_true + (Tensor::ones(if_false.resolve_shape(), dtype) - x) * if_false)
    }

    // loss functions
    /// Cross-entropy loss between this tensor (logits) and `target`.
    ///
    /// `target` may be class indices or one-hot; the class axis is inferred
    /// (0 for 1D inputs, 1 for 2D+). `reduction` selects `Mean` (scalar),
    /// `Sum` (scalar), or `None` (per-sample).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, ReduceOp};
    /// let logits = Tensor::from([[1.0f32, 2.0, 0.0]]);
    /// let loss = logits.cross_entropy(Tensor::from([1i32]), ReduceOp::Mean).unwrap();
    /// assert!((loss.item::<f32>() - 0.4076).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if `self` and `target` are incompatible.
    pub fn cross_entropy(&self, target: impl Into<Tensor>, reduction: ReduceOp) -> Result<Tensor, ZyxError> {
        let target = target.into();
        let classes_dim = if self.rank() <= 1 { 0 } else { 1 };
        let target = if self.resolve_shape() != target.resolve_shape() {
            target.unsqueeze(classes_dim)?.one_hot_along_dim(self.resolve_shape()[classes_dim as usize], classes_dim)?
        } else {
            target
        };
        let ln_softmax = self.ln_softmax([classes_dim])?;
        let per_sample = (-ln_softmax * target).sum([classes_dim])?;
        match reduction {
            ReduceOp::Mean => Ok(per_sample.mean_all()),
            ReduceOp::Sum => Ok(per_sample.sum_all()),
            ReduceOp::None => Ok(per_sample),
            _ => Err(ZyxError::ParseError("invalid reduction for cross_entropy, expected Mean, Sum, or None".into())),
        }
    }

    /// Negative log-likelihood loss between this tensor (log-probabilities)
    /// and `target` class indices, with optional `weight` and
    /// `ignore_index`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, ReduceOp};
    /// let logits = Tensor::from([[2.0f32, 1.0, 0.0]]);
    /// let loss = logits.nll_loss(
    ///     Tensor::from([1i32]), None, None, ReduceOp::Mean,
    /// ).unwrap();
    /// assert!((loss.item::<f32>() + 1.0).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the shapes are incompatible.
    #[allow(clippy::missing_panics_doc)]
    pub fn nll_loss(
        &self,
        target: impl Into<Tensor>,
        weight: Option<Tensor>,
        ignore_index: Option<i64>,
        reduction: ReduceOp,
    ) -> Result<Tensor, ZyxError> {
        let target = target.into();
        let classes_dim: Axis = if self.rank() <= 1 { 0 } else { 1 };
        let _n_classes = self.resolve_shape()[classes_dim as usize];

        let weight = match weight {
            Some(w) => w.gather(0, target.flatten(..)?)?.reshape(target.resolve_shape())?,
            None => Tensor::ones(target.resolve_shape(), self.dtype()),
        };

        let masked_weight = match ignore_index {
            Some(idx) => weight * target.ne(Tensor::from(idx))?,
            None => weight,
        };

        let idx = target.unsqueeze(classes_dim)?;
        let gathered = self.gather(classes_dim, idx)?;
        let gathered = gathered.squeeze([classes_dim]);
        let nll = (-gathered) * masked_weight.clone();

        match reduction {
            ReduceOp::Mean => Ok(nll.sum_all() / masked_weight.sum_all()),
            ReduceOp::Sum => Ok(nll.sum_all()),
            ReduceOp::None => Ok(nll),
            _ => Err(ZyxError::ParseError("invalid reduction for nll_loss".into())),
        }
    }

    /// CTC (Connectionist Temporal Classification) loss between this tensor
    /// (log-probabilities, shape `[T, C]`) and `targets` (class indices, shape
    /// `[L]`), using the forward-backward algorithm in log space. `blank` is
    /// the blank label index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType, ReduceOp};
    /// let logp = Tensor::zeros([2, 2], DType::F32);
    /// let l = logp.ctc_loss(Tensor::from([0i32, 1]), 0, ReduceOp::Mean).unwrap();
    /// assert!(l.item::<f32>().is_finite());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the input shapes or dtypes are incompatible.
    #[allow(clippy::missing_panics_doc)]
    pub fn ctc_loss(&self, targets: impl Into<Tensor>, blank: i64, reduction: ReduceOp) -> Result<Tensor, ZyxError> {
        let target = targets.into();
        let shape = self.resolve_shape();
        let t_dim = shape[0];
        let l_dim = target.resolve_shape()[0];
        let n_ext: usize = (2 * l_dim + 1).try_into().unwrap();
        let dtype = self.dtype();
        let neg_inf: f32 = -1e30;

        // Build extended labels: [blank, t0, blank, t1, ..., blank, tL-1, blank]
        let target_vals: Vec<i32> = target.cast(DType::I32).try_into()?;
        let mut ext_vals: Vec<i32> = Vec::with_capacity(n_ext);
        ext_vals.push(blank as i32);
        for &t in &target_vals {
            ext_vals.push(t);
            ext_vals.push(blank as i32);
        }
        let ext_labels = Tensor::from_vec(ext_vals, [n_ext as i64])?;

        // Gather extended log-probs: [T, 2L+1]
        let ext_labels_exp = ext_labels.expand([t_dim, n_ext as i64])?;
        let lp = self.gather(1, ext_labels_exp)?;

        // log_add helper: max(a,b) + ln(1 + exp(min(a,b) - max(a,b)))
        let log_add = |a: &Tensor, b: &Tensor| -> Tensor {
            let max_val = a.maximum(b).unwrap();
            let min_val = a.minimum(b).unwrap();
            max_val.clone() + ((min_val - &max_val).exp() + 1).ln()
        };

        let neg_inf_tensor = Tensor::full([n_ext as i64], neg_inf).cast(dtype);
        let neg_inf_val = Tensor::from(neg_inf).cast(dtype);

        // Initialize alpha[0]: [-inf, ..., -inf] with first two set
        let lp0 = lp.clone().slice(0..1)?.squeeze([0]);
        let init_vals: Vec<i32> = (0..n_ext).map(|i| if i <= 1 { 1 } else { 0 }).collect();
        let init_mask = Tensor::from(init_vals).cast(dtype);
        let alpha0 = init_mask.where_(&lp0, &neg_inf_tensor)?;
        let mut alpha = vec![alpha0];

        // Skip mask: positions s >= 2 where ext_labels[s-2] != ext_labels[s]
        let labels_prev = ext_labels.clone().slice(0..((n_ext - 2) as i64))?;
        let labels_curr = ext_labels.clone().slice(2..(n_ext as i64))?;
        let skip_inner = labels_prev.ne(&labels_curr)?.cast(dtype);
        let skip_mask = if n_ext >= 3 {
            let zeros_pad = Tensor::zeros([2], dtype);
            let cat_tensors: Vec<&Tensor> = vec![&zeros_pad, &skip_inner];
            Tensor::cat(cat_tensors, 0)?
        } else {
            skip_inner
        };

        // Forward pass
        for t in 1..t_dim as usize {
            let alpha_prev = alpha.last().unwrap().clone();
            let lpt = lp.clone().slice((t as i64)..(t as i64 + 1))?.squeeze([0]);

            // Shift right by 1: pad with -inf on left
            let padded1 = alpha_prev.clone().pad([(1, 0)], neg_inf_val.clone())?;
            let shifted1 = padded1.slice(0..(n_ext as i64))?;

            // base = log_add(alpha_prev, shifted1) + lpt
            let base = log_add(&alpha_prev, &shifted1) + &lpt;

            // Shift right by 2: pad with -inf on left
            let padded2 = alpha_prev.clone().pad([(2, 0)], neg_inf_val.clone())?;
            let shifted2 = padded2.slice(0..(n_ext as i64))?;

            // skip = log_add(base, shifted2)
            let skip_val = log_add(&base, &shifted2);

            // alpha_t = where_(skip_mask, skip_val, base)
            let alpha_t = skip_mask.clone().where_(&skip_val, &base)?;
            alpha.push(alpha_t);
        }

        // Loss = -log_add(alpha[T-1, n_ext-2], alpha[T-1, n_ext-1])
        let last = alpha.last().unwrap();
        let last_two = last.slice((n_ext as i64 - 2)..(n_ext as i64))?;
        let a1 = last_two.clone().slice(0..1)?.squeeze([0]);
        let a2 = last_two.slice(1..2)?.squeeze([0]);
        let log_sum = log_add(&a1, &a2);

        match reduction {
            ReduceOp::Mean | ReduceOp::Sum => Ok(-log_sum),
            ReduceOp::None => Err(ZyxError::ParseError("CTC loss with 'None' reduction is not supported".into())),
            _ => Err(ZyxError::ParseError("invalid reduction for ctc_loss".into())),
        }
    }

    /// Triplet margin loss over `(anchor, positive, negative)` samples using the
    /// p-norm distance:
    ///
    /// `loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)`
    ///
    /// `self` is the anchor. Inputs are 2D `[N, D]` tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, ReduceOp};
    /// let anchor = Tensor::from([[0.0f32, 0.0]]);
    /// let pos = Tensor::from([[0.1f32, 0.1]]);
    /// let neg = Tensor::from([[1.0f32, 1.0]]);
    /// let l = anchor.triplet_margin_loss(&pos, &neg, 0.5, 2, false, ReduceOp::Mean).unwrap();
    /// assert!((l.item::<f32>() - 0.0).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have incompatible shapes.
    #[allow(clippy::missing_panics_doc)]
    pub fn triplet_margin_loss(
        &self,
        positive: &Tensor,
        negative: &Tensor,
        margin: f32,
        p: i32,
        swap: bool,
        reduction: ReduceOp,
    ) -> Result<Tensor, ZyxError> {
        let anchor = self;
        let dtype = anchor.dtype();
        let eps: f32 = 1e-6;

        // Helper: p-norm distance along axis 1
        let dist = |a: &Tensor, b: &Tensor| -> Result<Tensor, ZyxError> {
            let diff = a - b;
            let abs_diff = diff.abs();
            let pow_p = abs_diff.pow(Tensor::from(p as f32))?;
            let sum_p = pow_p.sum([1])?;
            let eps_t = Tensor::from(eps).cast(dtype);
            let sum_p_eps = sum_p + eps_t;
            sum_p_eps.pow(Tensor::from(1.0 / p as f32))
        };

        let dist_pos = dist(anchor, positive)?;
        let dist_neg = dist(anchor, negative)?;

        let dist_pos_final = if swap {
            let dist_pn = dist(positive, negative)?;
            dist_pos.maximum(&dist_pn)?
        } else {
            dist_pos
        };

        let loss = (dist_pos_final - &dist_neg + Tensor::from(margin)).relu();

        match reduction {
            ReduceOp::Mean => Ok(loss.mean_all()),
            ReduceOp::Sum => Ok(loss.sum_all()),
            ReduceOp::None => Ok(loss),
            _ => Err(ZyxError::ParseError("invalid reduction for triplet_margin_loss".into())),
        }
    }

    /// Shrink the tensor, cropping it to the per-dimension `(start, end)`
    /// ranges in `dims`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3, 4, 5, 6]);
    /// let s = t.shrink([(2, 5)]).unwrap();
    /// assert_eq!(s.to_vec::<i32>().unwrap(), vec![3, 4, 5]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the ranges are invalid.
    pub fn shrink<I>(&self, dims: I) -> Result<Tensor, ZyxError>
    where
        I: IntoIterator<Item = (Dim, Dim)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.rpad_zeros(
            self.resolve_shape()
                .into_iter()
                .rev()
                .zip(dims.into_iter().rev())
                .map(|(d, (s, e))| (-(s as i64), -((d - e) as i64))),
        )
    }

    /// Convert `self` (class indices) into a one-hot tensor with `num_classes`
    /// columns, appending the dimension along the last axis. If `num_classes`
    /// is `0`, it is inferred from the max index in `self` (plus one).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0i32, 2, 1]);
    /// assert_eq!(t.one_hot(3).to_vec::<i32>().unwrap(),
    ///     vec![1, 0, 0, 0, 0, 1, 0, 1, 0]);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn one_hot(&self, num_classes: Dim) -> Tensor {
        let mut num_classes = num_classes;
        if num_classes == 0 {
            num_classes = (self.max_all() + 1).item::<i64>() as i64;
        }

        let dtype = self.dtype();
        self.unsqueeze(-1)
            .unwrap()
            .one_hot_along_dim(num_classes, -1)
            .unwrap()
            .where_(Tensor::ones([1], dtype), Tensor::zeros([1], dtype))
            .unwrap()
    }

    /// Convert `self` (class indices) into a one-hot tensor along `dim`, with
    /// `num_classes` positions. The one-hot axis must already exist in `self`
    /// (callers append it first, e.g. via `unsqueeze(-1)`); the comparison
    /// broadcasts `arange` over the remaining axes.
    ///
    /// # Errors
    ///
    /// Returns a dtype error if `self` is not an integer tensor.
    pub(crate) fn one_hot_along_dim(&self, num_classes: Dim, dim: Axis) -> Result<Tensor, ZyxError> {
        if !self.dtype().is_int() {
            return Err(ZyxError::dtype_error(
                format!("_one_hot_along_dim expects integer index tensor, got {:?}", self.dtype()).into(),
            ));
        }

        let rank = self.rank();
        let dim = if dim < 0 { rank as Axis + dim } else { dim };
        let offset = rank as Axis - dim - 1;

        let dt = if num_classes > i32::MAX as i64 {
            DType::I64
        } else {
            DType::I32
        };

        let arange = Tensor::arange(0, num_classes as i64, 1)?.cast(dt);

        // Reshape to [num_classes, 1, 1, ..., 1] with `offset` ones
        let mut new_shape: Vec<Dim> = vec![num_classes];
        new_shape.extend(vec![1; offset as usize]);
        let arange = arange.reshape(new_shape)?;

        // Broadcast and compare
        self.equal(&arange)
    }

    /// Element-wise L1 (absolute difference) loss between `self` and `target`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from([2.0f32, 3.0, 4.0]);
    /// assert_eq!(a.l1_loss(b).to_vec::<f32>().unwrap(), vec![1.0f32, 1.0, 1.0]);
    /// ```
    #[must_use]
    pub fn l1_loss(&self, target: impl Into<Tensor>) -> Tensor {
        (self - target).abs()
    }

    /// Mean squared error loss between `self` and `target` (scalar mean of
    /// squared differences).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([2.0f32, 3.0]);
    /// let b = Tensor::from([4.0f32, 5.0]);
    /// assert!((a.mse_loss(b).unwrap().item::<f32>() - 4.0).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have non broadcastable shapes.
    #[track_caller]
    pub fn mse_loss(&self, target: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self, target)?;
        let id = RT.lock().binary(x.id, y.id, BOp::Sub)?;
        let x = Tensor { id };
        Ok((x.clone() * x).mean_all())
    }

    /// Binary cross-entropy loss between `self` (clamped to `[eps, 1-eps]`)
    /// and `target`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let x = Tensor::from([0.9f32, 0.1]);
    /// let t = Tensor::from([1.0f32, 0.0]);
    /// let l = x.bce_loss(t, 1e-6).unwrap();
    /// assert!((l.item::<f32>() - 0.1054).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors are not broadcastable.
    #[track_caller]
    pub fn bce_loss(&self, target: impl Into<Tensor>, eps: f32) -> Result<Tensor, ZyxError> {
        let target: Tensor = target.into();
        let x: Tensor = self.clamp(eps, 1.0 - eps)?;
        let temp: Tensor = 1 - &x;
        let loss: Tensor = -(&target * x.ln() + (1 - &target) * temp.ln());
        Ok(loss.mean_all())
    }

    /// Cosine similarity between `self` and `rhs` (dot product over the
    /// product of Euclidean norms), with `eps` guarding against zero norms.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, ReduceOp};
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from([4.0f32, 5.0, 6.0]);
    /// let s = a.cosine_similarity(b, Tensor::from([1e-9f32]), ReduceOp::Sum).unwrap();
    /// assert!((s.item::<f32>() - 0.9747).abs() < 1e-4);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors are not broadcastable, or a
    /// parse error if `reduction` is not `Sum` or `Mean`.
    pub fn cosine_similarity(
        &self,
        rhs: impl Into<Tensor>,
        eps: impl Into<Tensor>,
        reduction: ReduceOp,
    ) -> Result<Tensor, ZyxError> {
        let rhs: Tensor = rhs.into();
        let eps: Tensor = eps.into();
        let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
        match reduction {
            ReduceOp::Sum | ReduceOp::Mean => {
                let dot = (self * &rhs).sum(axes.clone())?;
                let nx = (self * self).sum(axes.clone())?.sqrt();
                let ny = (&rhs * &rhs).sum(axes)?.sqrt();
                let denom = nx * ny;
                let denom = denom.cmplt(eps.clone())?.where_(eps, denom)?;
                Ok(dot / denom)
            }
            _ => Err(ZyxError::ParseError("invalid reduction for cosine_similarity, expected Sum or Mean".into())),
        }
    }

    // misc
    /// Flatten the tensor, joining the range of `axes` into a single dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1i32, 2], [3, 4]]);
    /// assert_eq!(t.flatten(0..1).unwrap().shape(), [2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axis range is invalid.
    pub fn flatten(&self, axes: impl RangeBounds<Axis>) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let start_dim = into_axis(
            match axes.start_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim + 1,
                Bound::Unbounded => 0,
            },
            rank as UAxis,
        )?;
        let end_dim = into_axis(
            match axes.end_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim - 1,
                Bound::Unbounded => -1,
            },
            rank as UAxis,
        )? + 1;
        // The joined dim is a SYMBOLIC product (mul chain over the dim
        // expression tensors) so a variable-backed seq dim stays symbolic.
        let symbolic = self.shape();
        let mut dim_iter = symbolic[start_dim as usize..end_dim as usize].iter();
        let mut dim = match dim_iter.next() {
            Some(d) => d.clone(),
            None => Tensor::from(1),
        };
        for d in dim_iter {
            let id = RT.lock().binary(dim.id, d.id, BOp::Mul).expect("flatten: failed to build symbolic mul chain");
            dim = Tensor { id };
        }
        let new_shape: Vec<Tensor> = symbolic[..start_dim as usize]
            .to_vec()
            .into_iter()
            .chain(std::iter::once(dim))
            .chain(symbolic[end_dim as usize..].to_vec())
            .collect();
        self.reshape(new_shape)
    }

    /// Concatenate a list of tensors along `axis`.
    ///
    /// All input tensors must have matching shapes except along `axis`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1i32, 2], [3, 4]]);
    /// let b = Tensor::from([[5, 6], [7, 8]]);
    /// let c = Tensor::cat([&a, &b], 0).unwrap();
    /// assert_eq!(c.to_vec::<i32>().unwrap(),
    ///     vec![1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors cannot be concatenated along the
    /// axis.
    #[track_caller]
    pub fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, axis: Axis) -> Result<Tensor, ZyxError> {
        let tensors: Vec<&Tensor> = tensors.into_iter().collect();
        if tensors.len() < 2 {
            return Err(ZyxError::shape_error("Cat requires two or more tensors.".into()));
        }
        let shape = tensors[0].resolve_shape();
        let rank = shape.len();
        let dim: usize = (if axis < 0 {
            axis + Axis::try_from(rank).unwrap()
        } else {
            axis
        })
        .try_into()
        .unwrap();
        // Dimension check (user side — resolve, concrete comparison)
        for tensor in &tensors {
            for (i, (d1, d2)) in shape.iter().zip(tensor.resolve_shape().iter()).enumerate() {
                if i != dim && *d1 != *d2 {
                    return Err(ZyxError::shape_error("Cannot concatenate these tensors.".into()));
                }
            }
        }
        // Computation stays SYMBOLIC: the output cat-dim is the sum of the
        // inputs' dim tensors, and each input's offset is a prefix sum of the
        // same expressions — no resolved lengths are baked anywhere. Each
        // input is zero-padded along `axis` to the total length, then the
        // padded views are summed.
        let dim_t = |t: &Tensor| -> Tensor { t.shape()[dim].clone() };
        let mut total = Tensor::from(0i64);
        for t in &tensors {
            total = total + dim_t(t);
        }
        let mut lp = Tensor::from(0i64);
        let mut res: Option<Tensor> = None;
        for tensor in &tensors {
            let padded = tensor.pad_zeros_axis(dim as UAxis, lp.clone(), total.clone())?;
            res = Some(match res {
                Some(r) => r + padded,
                None => padded,
            });
            lp = lp + dim_t(tensor);
        }
        Ok(res.unwrap())
    }

    /// Remove size-1 dimensions from `self`, optionally restricted to `axes`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([1i64, 3, 1], DType::F32);
    /// assert_eq!(t.squeeze([0, 2]).shape(), [3]);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn squeeze(&self, axes: impl IntoIterator<Item = Axis>) -> Tensor {
        // The drop decision needs concrete values (`d != 1` must hold to
        // remove an axis), but the KEPT dims must be propagated SYMBOLICALLY
        // from the dim-expression tensors — rebuilding them from resolved
        // `Dim`s would bake variable-backed dims into fresh constants,
        // cutting downstream consumers off the shared variable handle
        // (they would replay a const instead of the variable arg).
        let resolved = self.resolve_shape();
        let symbolic = self.shape();
        let mut naxes = Vec::new();
        for axis in axes.into_iter().take(resolved.len()) {
            if let Ok(axis) = into_axis(axis, resolved.len() as UAxis) {
                naxes.push(axis as usize);
            }
        }
        let mut new_shape = Vec::new();
        for (a, &d) in resolved.iter().enumerate() {
            if d != 1 || !naxes.contains(&a) {
                new_shape.push(symbolic[a].clone());
            }
        }
        if new_shape.is_empty() {
            new_shape = vec![Tensor::from(1)];
        }
        self.reshape(new_shape).unwrap()
    }

    /// Insert a new size-1 dimension at position `dim` (negative counts from
    /// the end).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([2i64, 3], DType::I8);
    /// assert_eq!(t.unsqueeze(1).unwrap().shape(), [2, 1, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if `dim` is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn unsqueeze(&self, dim: Axis) -> Result<Tensor, ZyxError> {
        let rank = self.rank() as usize;
        // Dims stay SYMBOLIC — only fresh 1 constants are inserted.
        let symbolic = self.shape();
        if dim < 0 {
            if -dim > (rank + 1) as Axis {
                return Err(ZyxError::shape_error(format!("Unsqueeze dim {dim} is not possible on rank {rank} tensor.").into()));
            }
            let pos = usize::try_from(-dim).unwrap();
            let pos = rank - pos + 1;
            let new_shape: Vec<Tensor> = symbolic[..pos]
                .to_vec()
                .into_iter()
                .chain(std::iter::once(Tensor::from(1)))
                .chain(symbolic[pos..].to_vec())
                .collect();
            self.reshape(new_shape)
        } else {
            let pos = usize::try_from(dim).unwrap();
            if pos > rank {
                return Err(ZyxError::shape_error(format!("Unsqueeze dim {dim} is not possible on rank {rank} tensor.").into()));
            }
            let new_shape: Vec<Tensor> = symbolic[..pos]
                .to_vec()
                .into_iter()
                .chain(std::iter::once(Tensor::from(1)))
                .chain(symbolic[pos..].to_vec())
                .collect();
            self.reshape(new_shape)
        }
    }

    /// Index of the maximum element (flattened, 1D).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 5, 3]);
    /// assert_eq!(t.argmax().item::<i32>(), 1);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn argmax(&self) -> Tensor {
        self.flatten(..).unwrap().argmax_impl(0, false).unwrap()
    }

    /// Index of the maximum element along `axis` (shape with that axis removed).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([[1i32, 5], [3, 9]]);
    /// let a = t.argmax_axis(1).unwrap();
    /// assert_eq!(a.to_vec::<i32>().unwrap(), vec![1, 1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the axis is out of bounds.
    pub fn argmax_axis(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let _ = into_axis(axis, rank as UAxis)?;
        self.argmax_impl(axis, false)
    }

    /// Argmax
    fn argmax_impl(&self, axis: Axis, keepdim: bool) -> Result<Tensor, ZyxError> {
        // max values along the axis
        let max_vals = self.max_keepdim([axis])?;

        // mask where values equal the max
        let mask = self.equal(max_vals)?;

        // correct axis
        let shape = self.resolve_shape();
        let uaxis = into_axis(axis, shape.len() as UAxis)?;

        // create a range tensor [0, 1, 2, ...] along the axis
        let range = Tensor::arange(0, shape[uaxis as usize] as i32, 1)?;
        let mut reshape_shape = vec![1; shape.len()];
        reshape_shape[uaxis as usize] = shape[uaxis as usize];
        let reshaped_range = range.reshape(reshape_shape)?;

        // mask * range -> positions of max values
        let idx = mask * reshaped_range;

        // max along axis gives argmax
        let res = if keepdim { idx.max_keepdim([axis])? } else { idx.max([axis])? };

        Ok(res.cast(DType::I32))
    }

    /// Stack the input tensors along a new `dim` axis (each gets a size-1 dim
    /// at `dim`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1i32, 2], [3, 4]]);
    /// let b = Tensor::from([[5, 6], [7, 8]]);
    /// let c = Tensor::stack_axis([&a, &b], 0).unwrap();
    /// assert_eq!(c.to_vec::<i32>().unwrap().len(), 8);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have mismatching shapes.
    ///
    /// # See also
    ///
    /// [`unsqueeze`](Tensor::unsqueeze), [`cat`](Tensor::cat)
    #[allow(clippy::missing_panics_doc)]
    pub fn stack_axis<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: Axis) -> Result<Tensor, ZyxError> {
        let tensors: Vec<Tensor> = tensors.into_iter().cloned().collect();
        let ret = Tensor::stack(&tensors)?;
        let rank = ret.rank();
        let dim = into_axis(dim, rank as UAxis)?;
        if dim == 0 {
            Ok(ret)
        } else {
            let axes: Vec<Axis> = (1..=dim as Axis).chain(0..1).chain((dim as Axis + 1)..rank as Axis).collect();
            ret.permute(axes)
        }
    }

    /// Stack the given (identically-shaped) tensors along a new leading axis.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1i32, 2, 3]);
    /// let b = Tensor::from([4, 5, 6]);
    /// let c = Tensor::stack(&[a.clone(), b]).unwrap();
    /// assert_eq!(c.to_vec::<i32>().unwrap().len(), 6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensors have mismatching shapes.
    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, ZyxError> {
        if tensors.is_empty() {
            return Err(ZyxError::shape_error("stack: empty".into()));
        }
        let first_shape = tensors[0].resolve_shape();
        for t in tensors {
            if t.resolve_shape() != first_shape {
                return Err(ZyxError::shape_error(
                    format!("stack: all shapes must match, got {first_shape:?} and {:?}", t.resolve_shape()).into(),
                ));
            }
        }
        let ids: Vec<TensorId> = tensors.iter().map(|x| x.id).collect();
        let id = RT.lock().stack(&ids)?;
        Ok(Tensor { id })
    }

    /// Split `self` into sub-tensors of the given `sizes` along `axis`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3, 4, 5, 6]);
    /// let parts = t.split([2, 3, 1], 0).unwrap();
    /// assert_eq!(parts.len(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if `sizes` do not sum to the axis length.
    #[allow(clippy::missing_panics_doc)]
    pub fn split(&self, sizes: impl IntoIterator<Item = impl Into<Tensor>>, axis: isize) -> Result<Vec<Tensor>, ZyxError> {
        // assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        // dim = self._resolve_dim(dim)
        // if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
        // assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
        // return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])
        let sizes: Vec<Dim> = sizes.into_iter().map(|s| s.into().item::<i64>()).collect();
        let shape = self.resolve_shape();
        let rank = shape.len();
        let dim: usize = usize::try_from(if axis < 0 {
            axis + isize::try_from(rank).unwrap()
        } else {
            axis
        })
        .unwrap();
        if sizes.iter().sum::<Dim>() != shape[dim] {
            return Err(ZyxError::shape_error(
                format!(
                    "Sizes must sum exactly to {}, but got {:?}, which sums to {}",
                    shape[dim],
                    sizes,
                    sizes.iter().sum::<Dim>()
                )
                .into(),
            ));
        }

        let mut res = Vec::new();
        let mut acc_size: i64 = 0;
        for size in sizes {
            let size = size as i64;
            let mut index = Vec::new();
            for &d in shape.iter().take(dim) {
                index.push(0..d as i64);
            }
            index.push(acc_size..acc_size + size);
            //println!("Index {index:?}");
            res.push(self.slice(index)?);
            acc_size += size;
        }
        Ok(res)
    }

    /// Replace elements of `self` with `value` where `mask` is true.
    ///
    /// # Note
    ///
    /// Delegates to `where_`, so it inherits its branchless decomposition and
    /// its ±inf limitation: filling with ±inf (or `self` containing ±inf on
    /// kept elements) produces `NaN`. Use a finite value instead.
    // TODO: possibly for some models in the future a ternary where op will be
    // needed here (graph node + kernel IR + backends); then masked_fill can
    // support ±inf values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([1i32, 2, 3]);
    /// let m = Tensor::from([false, true, false]);
    /// assert_eq!(a.masked_fill(m, 0i32).unwrap().to_vec::<i32>().unwrap(),
    ///     vec![1, 0, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if `self` and `mask` are not broadcastable.
    pub fn masked_fill(&self, mask: impl Into<Tensor>, value: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        mask.into().where_(value, self.clone())
    }

    /// Triangular matrix with `1`s on and below the `diagonal` and `0`s above
    /// (shape `[r, c]`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::tri(3, 3, 0, DType::I32);
    /// assert_eq!(t.to_vec::<i32>().unwrap(),
    ///     vec![1, 1, 1, 0, 1, 1, 0, 0, 1]);
    /// ```
    #[must_use]
    #[track_caller]
    #[allow(clippy::missing_panics_doc)]
    pub fn tri(r: Dim, c: Dim, diagonal: i64, dtype: DType) -> Tensor {
        if r == 0 || c == 0 || diagonal >= c as i64 {
            return Tensor::zeros([r, c], dtype);
        }
        if r as i64 + diagonal <= 0 {
            return Tensor::ones([r, c], dtype);
        }
        let s = r + c - 1;
        let t = Tensor::ones([s, s], dtype).rpad_zeros([(0i64, s as i64)]).unwrap();
        let t = t.reshape([2 * s * s]).unwrap();
        let t = t.rpad_zeros([(0i64, -(s as i64))]).unwrap();
        let t = t.reshape([s, 2 * s - 1]).unwrap();
        let t = t.rpad_zeros([(0i64, -((2 * s - 1 - s) as i64))]).unwrap();
        if diagonal <= 0 {
            t.slice((0..r as i64, (-diagonal)..(c as i64 - diagonal))).unwrap()
        } else {
            t.slice((diagonal..(r as i64 + diagonal), 0..c as i64)).unwrap()
        }
    }

    /// Upper-triangular part of `self` (elements on/above `diagonal` kept, rest
    /// zeroed).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1i32, 2, 3], [4, 5, 6]]);
    /// let u = a.triu(0).unwrap().to_vec::<i32>().unwrap();
    /// assert_eq!(u, vec![1, 2, 3, 0, 5, 6]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensor rank is less than 2.
    pub fn triu(&self, diagonal: i64) -> Result<Tensor, ZyxError> {
        //return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal, device=self.device, dtype=dtypes.bool).where(self, self.zeros_like())
        let [r, c] = self.rdims::<2>()?;
        Tensor::tri(r.item::<Dim>(), c.item::<Dim>(), diagonal, DType::Bool).where_(self, Tensor::zeros_like(self))
    }

    /// Lower-triangular part of `self` (elements on/below `diagonal` kept, rest
    /// zeroed).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let a = Tensor::from([[1i32, 2, 3], [4, 5, 6]]);
    /// let l = a.tril(0).unwrap().to_vec::<i32>().unwrap();
    /// assert_eq!(l, vec![1, 0, 0, 4, 5, 0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensor rank is less than 2.
    pub fn tril(&self, diagonal: i64) -> Result<Tensor, ZyxError> {
        //return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal+1, device=self.device, dtype=dtypes.bool).where(self.zeros_like(), self)
        let [r, c] = self.rdims::<2>()?;
        Tensor::tri(r.item::<Dim>(), c.item::<Dim>(), diagonal + 1, DType::Bool).where_(Tensor::zeros_like(self), self)
    }

    /// Strided pooling over the last two dimensions using the given
    /// `kernel_size`, `stride`, and `dilation`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0],
    ///                        [5.0, 6.0, 7.0, 8.0],
    ///                        [9.0, 10.0, 11.0, 12.0],
    ///                        [13.0, 14.0, 15.0, 16.0]]);
    /// let p = x.pool([2, 2], [2, 2], [1, 1]).unwrap();
    /// assert_eq!(p.to_vec::<f32>().unwrap().len(), 16);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the pooling parameters are incompatible.
    #[allow(clippy::missing_panics_doc)]
    pub fn pool(
        &self,
        kernel_size: impl IntoIterator<Item = impl Into<Tensor>>,
        stride: impl IntoIterator<Item = impl Into<Tensor>>,
        dilation: impl IntoIterator<Item = impl Into<Tensor>>,
    ) -> Result<Tensor, ZyxError> {
        // What a complex function ...
        let k_: Vec<Dim> = kernel_size.into_iter().map(|s| s.into().item::<i64>()).collect();
        let stride: Vec<Dim> = stride.into_iter().map(|s| s.into().item::<i64>()).collect();
        let dilation: Vec<Dim> = dilation.into_iter().map(|s| s.into().item::<i64>()).collect();

        let shape = self.resolve_shape();
        let rank = shape.len();

        let s_: Vec<Dim> = if stride.len() == 1 {
            vec![stride[0]; k_.len()]
        } else {
            stride
        };
        let d_: Vec<Dim> = if dilation.len() == 1 {
            vec![dilation[0]; k_.len()]
        } else {
            dilation
        };
        let i_ = &shape[rank - k_.len()..];
        let o_: Vec<Dim> =
            (i_, d_.iter(), k_.iter(), s_.iter()).zip().map(|(i, d, k, s)| (*i - *d * (*k - 1) + *s - 1) / *s).collect();
        //println!("s_ {s_:?}, d_ {d_:?}, i_ {i_:?} o_ {o_:?}");
        let repeats: Vec<Dim> = repeat_n(1, rank - k_.len())
            .chain(
                k_.iter().copied().zip(i_.iter().copied()).zip(d_.iter().copied()).map(|((k, i), d)| (k * (i + d) + i - 1) / i),
            )
            .collect();
        //println!("repeats {repeats:?}");
        let pad_b: Vec<Range<i64>> = shape[..rank - k_.len()].iter().map(|&d| 0..d as i64).collect();
        let sh_b: Vec<Dim> = shape[..rank - k_.len()].into();
        let mut xup = self.repeat(repeats)?;

        // dilation
        //println!("{xup:?} before padding");
        let padding: Vec<Range<i64>> = pad_b
            .iter()
            .cloned()
            .chain(k_.iter().copied().zip(i_.iter().copied()).zip(d_.iter().copied()).map(|((k, i), d)| 0..(k * (i + d)) as i64))
            .collect();
        //println!("Padding {padding:?}");
        xup = xup.slice(padding)?;
        //println!("{xup} padded");
        let sh: Vec<Dim> = sh_b
            .iter()
            .copied()
            .chain(k_.iter().copied().zip(i_.iter().copied()).zip(d_.iter().copied()).flat_map(|((k, i), d)| [k, i + d]))
            .collect();
        //println!("Reshape {sh:?}");
        xup = xup.reshape(sh)?;

        // stride
        // padding = noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))
        // xup = xup.shrink(padding)
        let padding: Vec<Range<i64>> = pad_b
            .iter()
            .cloned()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .zip(s_.iter().copied())
                    .flat_map(|((k, o), s)| [(0..k as i64), (0..(o * s) as i64)]),
            )
            .collect();
        xup = xup.slice(padding)?;
        // sh = noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_))
        // xup = xup.reshape(sh)
        let sh: Vec<Dim> = sh_b
            .iter()
            .copied()
            .chain(k_.iter().copied().zip(o_.iter().copied()).zip(s_.iter().copied()).flat_map(|((k, o), s)| [k, o, s]))
            .collect();
        xup = xup.reshape(sh)?;
        // padding = noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))
        // xup = xup.shrink(padding)
        let padding: Vec<Range<i64>> = pad_b
            .iter()
            .cloned()
            .chain(k_.iter().copied().zip(o_.iter().copied()).flat_map(|(k, o)| [(0..k as i64), (0..o as i64), (0..1)]))
            .collect();
        xup = xup.slice(padding)?;
        // sh = noop_ + flatten((k,o) for k,o in zip(k_, o_))
        // xup = xup.reshape(sh)
        let sh: Vec<Dim> =
            sh_b.iter().copied().chain(k_.iter().copied().zip(o_.iter().copied()).flat_map(Into::<[Dim; 2]>::into)).collect();
        xup = xup.reshape(sh)?;

        // xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
        let axes: Vec<Axis> = (0..rank - k_.len())
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2 + 1))
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2))
            .map(|i| Axis::try_from(i).unwrap())
            .collect();
        xup = xup.permute(axes)?;

        Ok(xup)
    }

    /// Perform an N-dimensional convolution over `self`.
    ///
    /// `weight` has shape `[out_channels, in_channels / groups, ...]`;
    /// `stride`, `dilation`, and `padding` are given per spatial dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let t = Tensor::arange(0, 9, 1).unwrap()
    ///     .reshape([1, 1, 3, 3]).unwrap();
    /// let w = Tensor::ones([1, 1, 2, 2], DType::F32);
    /// let out = t.conv(&w, None, 1, [1, 1], [1, 1], [0, 0]).unwrap();
    /// assert_eq!(out.shape(), [1, 1, 2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensor shapes are incompatible for
    /// convolution.
    #[allow(clippy::missing_panics_doc)]
    pub fn conv(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        groups: u64,
        stride: impl IntoIterator<Item = impl Into<Tensor>>,
        dilation: impl IntoIterator<Item = impl Into<Tensor>>,
        padding: impl IntoIterator<Item = impl Into<Tensor>>,
    ) -> Result<Tensor, ZyxError> {
        fn resolve_pool_pads(padding: &[Dim], dims: usize) -> Vec<i64> {
            if padding.len() == 1 {
                vec![padding[0] as i64; 2 * dims]
            } else if padding.len() == 2 * dims {
                padding.iter().map(|&p| p as i64).collect()
            } else {
                let mut npadding: Vec<i64> = Vec::new();
                for _ in 0..2 {
                    for &p in padding {
                        npadding.push(p as i64);
                    }
                }
                npadding.reverse();
                npadding
            }
        }

        let [bs, cin_] = self.resolve_shape()[..2] else {
            return Err(ZyxError::shape_error(format!("conv requires self rank >= 2, but rank = {}", self.rank()).into()));
        };
        let wsh = weight.resolve_shape();
        let [cout, cin] = wsh[..2] else {
            return Err(ZyxError::shape_error(format!("conv requires weight rank >= 2, but rank = {}", weight.rank()).into()));
        };
        if let Some(bias) = bias
            && bias.resolve_shape().iter().product::<Dim>() != cout
        {
            return Err(ZyxError::shape_error(
                format!("Bias length {} does not match output channels {}", bias.resolve_shape().iter().product::<Dim>(), cout)
                    .into(),
            ));
        }

        let hw = wsh[2..].to_vec();

        let stride: Vec<Dim> = stride.into_iter().map(|s| s.into().item::<i64>()).collect();
        let dilation: Vec<Dim> = dilation.into_iter().map(|s| s.into().item::<i64>()).collect();
        /*if stride.len() != hw.len() || dilation.len() != hw.len() {
            return Err(ZyxError::shape_error("Stride/dilation length must match kernel spatial dimensions".into()));
        }*/

        let padding_dim: Vec<Dim> = padding.into_iter().map(|s| s.into().item::<i64>()).collect();
        let padding_: Vec<i64> = resolve_pool_pads(&padding_dim, hw.len());

        if (groups as Dim * cin != cin_) || (self.resolve_shape().len() != wsh.len()) {
            return Err(ZyxError::shape_error(
                format!(
                    "Input Tensor shape {:?} does not match the shape of the weights {:?}. ({} vs. {cin_})",
                    self.resolve_shape(),
                    wsh,
                    groups as Dim * cin
                )
                .into(),
            ));
        }

        let x = self.rpad_zeros(padding_.chunks(2).map(|x| (x[0], x[1]))).unwrap().pool(hw.clone(), stride, dilation).unwrap();
        let rcout = cout / groups as Dim;
        let xsh = x.resolve_shape();
        let oyx = &xsh[2..xsh.len() - hw.len()];

        // for now without winograd
        let shape: Vec<Dim> = [bs, groups as Dim, cin, 1].iter().chain(oyx).chain(&hw).copied().collect();
        let x = x.reshape(shape).unwrap();
        let shape: Vec<Dim> = [bs, groups as Dim, cin, rcout].iter().chain(oyx).chain(&hw).copied().collect();
        let x = x.expand(shape).unwrap();
        let mut axes = vec![0, 1, 3];
        for i in 0..oyx.len() {
            axes.push(4 + i);
        }
        axes.push(2);
        for i in 0..hw.len() {
            axes.push(4 + oyx.len() + i);
        }
        let x = x.permute(axes.iter().map(|&a| Axis::try_from(a).unwrap())).unwrap();

        let shape: Vec<Dim> =
            [1, groups as Dim, rcout].iter().chain(&vec![1; oyx.len()]).chain(&[cin]).chain(&hw).copied().collect();
        let weight = weight.reshape(shape).unwrap();
        let mut axes: Vec<Axis> = Vec::new();
        for i in 0..=oyx.len() {
            axes.push(-1 - Axis::try_from(i).unwrap());
        }
        let shape: Vec<Dim> = [bs, cout].iter().chain(oyx).copied().collect();
        let mut ret = (x * weight).sum_keepdim(axes).unwrap().reshape(shape).unwrap();

        if let Some(bias) = bias {
            let shape: Vec<Dim> =
                once(1).chain([bias.resolve_shape().iter().product::<Dim>()]).chain(repeat_n(1, hw.len())).collect();
            ret = ret + bias.reshape(shape).unwrap();
        }

        Ok(ret)
    }

    // TODO we also need these two functions for pooling
    /*def _resolve_pool_pads(self, padding:int|Sequence[int], dims:int) -> Sequence[int]:
      if not isinstance(padding, int) and not (len(padding) == 2*dims or len(padding) == dims):
        raise ValueError(f"Padding must be an int or a sequence of length {dims} or {2*dims}, but got {padding=} for {self.shape=} with {dims=}.")
      return [padding]*2*dims if isinstance(padding, int) else (padding if len(padding) == 2*dims else [p for p in padding for _ in range(2)][::-1])

    def _apply_ceil_mode(self, pads:Sequence[int], k_:tuple[sint, ...], s_:int|tuple[int, ...], d_:int|tuple[int, ...]) -> list[int]:
      (d_,s_), i_ = (make_tuple(x, len(k_)) for x in (d_,s_)), self.shape[-len(k_):]
      pads, grouped_pads = list(pads), _flat_to_grouped(pads)
      # https://arxiv.org/pdf/1603.07285 section 5.1, relationship 15.
      o_ = [ceildiv(i+pB+pA - (d*(k-1)+1), s) + 1 for i,d,k,s,(pB,pA) in zip(i_,d_,k_,s_,grouped_pads)]
      for dim,(o,i,s,k,d,(pB,pA)) in enumerate(zip(o_,i_,s_,k_,d_,grouped_pads)):
        # we have to do additional padding before `_pool` so that `o_` in `_pool` is calculated correctly
        # `s*(o-1) + (d*(k-1)+1) - (i+pB+pA)` -> last_sliding_window_start + full_kernel_size - padded_input_shape
        # we decrease padding in the case that a sliding window starts in the end padded region, thereby decreasing `o_` in `_pool`
        # `smax(s*(o-1) - (pB+i-1), 0)` -> last_sliding_window_start - (pad_before + input_size - zero_offset)
        pads[-1-dim*2] += s*(o-1) + (d*(k-1)+1) - (i+pB+pA) - smax(s*(o-1) - (pB+i-1), 0)
      return pads*/

    /// Max pooling over the last two dimensions with the given `kernel_size`,
    /// `stride`, `dilation`, and `padding`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0],
    ///                        [5.0, 6.0, 7.0, 8.0],
    ///                        [9.0, 10.0, 11.0, 12.0],
    ///                        [13.0, 14.0, 15.0, 16.0]]);
    /// let p = x.max_pool([2, 2], [2, 2], [1, 1], [(0, 0), (0, 0)], false, false).unwrap();
    /// assert_eq!(p.to_vec::<f32>().unwrap(), vec![6.0, 8.0, 14.0, 16.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the kernel size, stride, or padding is invalid.
    pub fn max_pool(
        &self,
        kernel_size: impl IntoIterator<Item = impl Into<Tensor>>,
        stride: impl IntoIterator<Item = impl Into<Tensor>>,
        dilation: impl IntoIterator<Item = impl Into<Tensor>>,
        padding: impl IntoIterator<Item = (i64, i64)>,
        ceil_mode: bool,
        return_indices: bool,
    ) -> Result<Tensor, ZyxError> {
        let kernel_size: Vec<Dim> = kernel_size.into_iter().map(|s| s.into().item::<i64>()).collect();
        let axis: Vec<Axis> = (-(kernel_size.len() as Axis)..0).collect();

        let padding: Vec<(i64, i64)> = padding.into_iter().collect();

        if ceil_mode {
            todo!("ceil mode is not implemented yet")
        }
        //if ceil_mode: pads = self._apply_ceil_mode(pads, k_, stride if stride is not None else k_, dilation)
        // TODO

        let dtype = self.dtype();
        let value: Tensor = Tensor { id: RT.lock().new_constant_tensor(dtype.min_constant()) };
        let pooled = self.pad(padding, value)?.pool(kernel_size, stride, dilation)?;

        if !return_indices {
            return pooled.max(axis);
        }

        //spatial_sz = int(math.prod(spatial_shape := self.shape[-len(k_):]))
        //idx = Tensor.arange(spatial_sz,0,-1, requires_grad=False, device=self.device).reshape(spatial_shape)
        //m = pooled == pooled.max(axis, keepdim=True)
        //idx = m * idx.pad(pads, value=dtypes.min(idx.dtype))._pool(k_, stride if stride is not None else k_, dilation)
        //return pooled.max(axis), spatial_sz - idx.max(axis)

        todo!()
    }

    /// Repeat `self` along each dimension by the counts in `repeats`. If `repeats`
    /// is shorter than the rank, it is padded with ones at the front.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from(vec![1i32, 2, 3]);
    /// assert_eq!(t.repeat([2]).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 1, 2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape error if the tensor has zero dimensions.
    #[allow(clippy::missing_panics_doc)]
    pub fn repeat(&self, repeats: impl IntoIterator<Item = impl Into<Tensor>>) -> Result<Tensor, ZyxError> {
        let repeats: Vec<Dim> = repeats.into_iter().map(|s| s.into().item::<i64>()).collect();
        let shape = self.resolve_shape();
        let rank = shape.len();
        if repeats.len() < rank {
            return Err(ZyxError::shape_error("Repeats must be greater or equal to rank of the tensor.".into()));
        }
        let base_shape: Vec<Dim> = repeat_n(1, repeats.len() - rank).chain(shape.iter().copied()).collect();
        let new_shape: Vec<Dim> = repeat_n(1i64, repeats.len() - rank).chain(shape).flat_map(|d| [1i64, d]).collect();
        let expand_shape: Vec<Dim> =
            repeats.iter().copied().zip(base_shape.iter().copied()).flat_map(Into::<[Dim; 2]>::into).collect();
        let final_shape: Vec<Dim> = repeats.iter().copied().zip(base_shape.iter().copied()).map(|(r, d)| r * d).collect();
        //println!("base_shape {base_shape:?} {new_shape:?} {expand_shape:?} {final_shape:?}");
        let mut x = self.reshape(new_shape).unwrap();
        x = x.expand(expand_shape).unwrap();
        x = x.reshape(final_shape).unwrap();
        Ok(x)
    }

    /// Apply Rotary Positional Encoding (`RoPE`) using the provided sine and
    /// cosine frequency tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let x = Tensor::ones([2, 8], DType::F32);
    /// let s = Tensor::zeros([2, 4], DType::F32);
    /// let c = Tensor::zeros([2, 4], DType::F32);
    /// let r = x.rope(s, c).unwrap();
    /// assert_eq!(r.shape(), [1, 1, 2, 8]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a shape or dtype error if the tensors are incompatible.
    pub fn rope(&self, sine_frequencies: impl Into<Tensor>, cosine_frequencies: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let sin_freqs: Tensor = sine_frequencies.into();
        let cos_freqs: Tensor = cosine_frequencies.into();
        if !RT.lock().implicit_casts {
            let dtype = self.dtype();
            let sdtype = sin_freqs.dtype();
            let cdtype = cos_freqs.dtype();
            if dtype != sdtype || dtype != cdtype {
                return Err(ZyxError::dtype_error(
                    format!(
                        "ROPE all inputs must have the same dtype self dtype {dtype}, sin_freqs {sdtype}, cos_freqs {cdtype}"
                    )
                    .into(),
                ));
            }
        }

        let sh = self.resolve_shape();
        //println!("shape={sh:?}");
        //println!("sin_freqs={:?}", sin_freqs.shape());
        //println!("cos_freqs={:?}", cos_freqs.shape());
        if sh.len() < 2 {
            return Err(ZyxError::shape_error(format!("RoPE requires input >= 2d, but current input is {}d", sh.len()).into()));
        }

        let sh_dims = self.shape();
        let seq_t = sh_dims[sh_dims.len() - 2].clone();
        let embed_t = sh_dims[sh_dims.len() - 1].clone();
        // User-side validation resolves (always resolvable); the computation
        // below composes dim tensors so symbolic dims stay symbolic.
        let sh = self.resolve_shape();
        let seq_len = sh[sh.len() - 2];
        let embed_dim = sh[sh.len() - 1];

        //let axes = 0..sh.len() as SAxis - 2;
        //println!("Squeeze axes: {axes:?}");

        if sin_freqs.resolve_shape() != [seq_len, embed_dim / 2] || cos_freqs.resolve_shape() != [seq_len, embed_dim / 2] {
            return Err(ZyxError::dtype_error(
                format!(
                    "sin_freqs and cos_freqs must have shape [seq_len, embed_dim / 2] after squeezing. \
                 However, after squeezing, sin_freqs has shape {:?} and cos_freqs has shape {:?}. \
                 Expected shapes: [{seq_len}, {}]",
                    sin_freqs.resolve_shape(),
                    cos_freqs.resolve_shape(),
                    embed_dim / 2
                )
                .into(),
            ));
        }

        // half as a dim expression; the half-slices narrow with dim-tensor
        // bounds so a symbolic embed/seq dim never gets baked.
        let half_t = embed_t / Tensor::from(2i64);
        let sin_freqs = sin_freqs.reshape([Tensor::from(1i64), Tensor::from(1i64), seq_t.clone(), half_t.clone()]).unwrap();
        let cos_freqs = cos_freqs.reshape([Tensor::from(1i64), Tensor::from(1i64), seq_t, half_t.clone()]).unwrap();

        let a = self.narrow(-1, Tensor::from(0i64), half_t.clone())?;
        let b = -self.narrow(-1, half_t.clone(), half_t)?;
        let ro = a.clone() * cos_freqs.clone() - b.clone() * sin_freqs.clone();
        let co = a * sin_freqs + b * cos_freqs;
        let r = Tensor::cat([&co, &ro], -1).unwrap(); // Concatenate along the last dimension

        Ok(r)
    }

    /// Create new tensor from file on disk.
    pub(crate) fn from_path(shape: Vec<Dim>, dtype: DType, path: impl AsRef<Path>, offset: u64) -> Result<Tensor, ZyxError> {
        let shape_st = if shape.is_empty() {
            None
        } else {
            Some(Tensor::stack(&shape.iter().map(|&d| Tensor::from(d)).collect::<Vec<_>>())?)
        };
        let id = {
            let mut rt = RT.lock();
            let shape_id = shape_st.as_ref().map(|s| s.id).unwrap_or(TensorId::NULL);
            rt.new_disk_tensor(shape_id, dtype, path.as_ref(), offset)?
        };
        Ok(Tensor { id })
    }

    /// All tensor elements as a contiguous little-endian byte vector in row-major
    /// (C) order.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i16, -2]);
    /// assert_eq!(t.to_le_bytes().unwrap(), [1, 0, 254, 255]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a realization error if `self` cannot be evaluated to concrete data.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, ZyxError> {
        Ok(match self.dtype() {
            DType::BF16 => {
                let data: Vec<bf16> = self.clone().try_into()?;
                data.into_iter().flat_map(bf16::to_le_bytes).collect()
            }
            DType::F16 => {
                let data: Vec<f16> = self.clone().try_into()?;
                data.into_iter().flat_map(f16::to_le_bytes).collect()
            }
            DType::F32 => {
                let data: Vec<f32> = self.clone().try_into()?;
                data.into_iter().flat_map(f32::to_le_bytes).collect()
            }
            DType::F64 => {
                let data: Vec<f64> = self.clone().try_into()?;
                data.into_iter().flat_map(f64::to_le_bytes).collect()
            }
            DType::F8E4M3 => {
                let data: Vec<f8e4m3> = self.clone().try_into()?;
                data.into_iter().flat_map(f8e4m3::to_le_bytes).collect()
            }
            DType::F8E5M2 => {
                let data: Vec<f8e5m2> = self.clone().try_into()?;
                data.into_iter().flat_map(f8e5m2::to_le_bytes).collect()
            }
            DType::U8 => {
                let data: Vec<u8> = self.clone().try_into()?;
                data.into_iter().flat_map(u8::to_le_bytes).collect()
            }
            DType::U16 => {
                let data: Vec<u16> = self.clone().try_into()?;
                data.into_iter().flat_map(u16::to_le_bytes).collect()
            }
            DType::U32 => {
                let data: Vec<u32> = self.clone().try_into()?;
                data.into_iter().flat_map(u32::to_le_bytes).collect()
            }
            DType::U64 => {
                let data: Vec<u64> = self.clone().try_into()?;
                data.into_iter().flat_map(u64::to_le_bytes).collect()
            }
            DType::I8 => {
                let data: Vec<i8> = self.clone().try_into()?;
                data.into_iter().flat_map(i8::to_le_bytes).collect()
            }
            DType::I16 => {
                let data: Vec<i16> = self.clone().try_into()?;
                data.into_iter().flat_map(i16::to_le_bytes).collect()
            }
            DType::I32 => {
                let data: Vec<i32> = self.clone().try_into()?;
                data.into_iter().flat_map(i32::to_le_bytes).collect()
            }
            DType::I64 => {
                let data: Vec<i64> = self.clone().try_into()?;
                data.into_iter().flat_map(i64::to_le_bytes).collect()
            }
            DType::Bool => {
                let data: Vec<bool> = self.clone().try_into()?;
                #[allow(clippy::transmute_undefined_repr)]
                unsafe {
                    std::mem::transmute::<Vec<bool>, Vec<u8>>(data)
                }
            }
        })
    }

    // Load tensor from `le_bytes` in row major order
    /*fn from_le_bytes(bytes: &[u8]) -> Result<Tensor, ZyxError> {
        let _ = bytes;
        todo!()
    }*/

    /// Move this tensor to the specified device, inserting a cross-device copy
    /// node in the graph.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// use zyx::Dev;
    /// let t = Tensor::from([1i32]);
    /// let t2 = t.to(Dev::C).unwrap();
    /// assert_eq!(t2.device(), t.device());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the device copy cannot be performed.
    pub fn to(&self, device: crate::Dev) -> Result<Tensor, ZyxError> {
        let id = RT.lock().to_device(self.id, device)?;
        Ok(Tensor { id })
    }

    /// Materialize the tensor into its own contiguous buffer. This breaks kernel
    /// fusion: downstream ops no longer fuse with the producer kernel.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1i32, 2, 3]).contiguous().unwrap();
    /// assert_eq!(t.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor cannot be made contiguous.
    pub fn contiguous(&self) -> Result<Tensor, ZyxError> {
        let id = RT.lock().contiguous(self.id)?;
        Ok(Tensor { id })
    }
}

#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct DebugGuard {
    debug: DebugMask,
}

impl Drop for DebugGuard {
    fn drop(&mut self) {
        crate::set_debug_mask(self.debug);
    }
}

impl Tensor {
    /// If self is not float, then cast it to float
    #[track_caller]
    fn float_cast(&self) -> Result<Tensor, ZyxError> {
        let dtype = self.dtype();
        if !dtype.is_float() {
            if RT.lock().implicit_casts {
                return Ok(match (dtype.bit_size() / 8) as usize {
                    2 => self.cast(DType::F16),
                    4 => self.cast(DType::F32),
                    8 => self.cast(DType::F64),
                    _ => panic!(),
                });
            }
            return Err(ZyxError::dtype_error(format!("Called function that only supports float on a tensor that is of dtype = {dtype} while implitic casts were disabled.").into()));
        }
        Ok(self.clone())
    }

    /// Braodcasts to synchronize shapes and casts to synchronize dtypss
    /// This does both automatic expand AND automatic casting between dtypes.
    // TODO Broadcasting can be disable by changing a setting in the backend.
    #[track_caller]
    fn broadcast(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Result<(Tensor, Tensor), ZyxError> {
        let mut x = x.into();
        let mut y = y.into();
        /*assert_eq!(
            graph.dtype(xid),
            graph.dtype(yid),
            "{op} parameters {xid} and {yid} have different dtypes: {} and {}",
            graph.dtype(xid),
            graph.dtype(yid)
        );*/
        // Now we just do implicit conversions. Not exactly rust style, but it's convenient.
        // We can later add option for backend to disable these implicit conversions.
        let x_dtype = x.dtype();
        let y_dtype = y.dtype();
        let x_shape = x.resolve_shape();
        let y_shape = y.resolve_shape();
        if x_dtype != y_dtype {
            // Only a rank-0 tensor (shape []) is a scalar; it is cast to the other
            // operand's dtype rather than upcasting the tensor. A shape-[1] tensor
            // is a normal 1-D tensor and participates in implicit autocast.
            let x_scalar = x_shape.is_empty();
            let y_scalar = y_shape.is_empty();
            if RT.lock().implicit_casts && !x_scalar && !y_scalar {
                let common_dtype = x_dtype.least_upper_dtype(y_dtype);
                if x_dtype != common_dtype {
                    x = x.cast(common_dtype);
                }
                if y_dtype != common_dtype {
                    y = y.cast(common_dtype);
                }
            } else if x_scalar && !y_scalar {
                x = x.cast(y_dtype);
            } else if !x_scalar && y_scalar {
                y = y.cast(x_dtype);
            } else if x_scalar && y_scalar {
                // Two scalars promote torch-style, always: with no shape
                // to prefer, least_upper_dtype picks the result dtype.
                let common_dtype = x_dtype.least_upper_dtype(y_dtype);
                if x_dtype != common_dtype {
                    x = x.cast(common_dtype);
                }
                if y_dtype != common_dtype {
                    y = y.cast(common_dtype);
                }
            } else {
                return Err(ZyxError::dtype_error(
                    format!("Implicit casting disabled, binary inputs have different dtypes: {x_dtype} and {y_dtype}").into(),
                ));
            }
        }

        for (&x, &y) in x_shape.iter().rev().zip(y_shape.iter().rev()) {
            if x != y && x != 1 && y != 1 {
                return Err(ZyxError::shape_error(
                    format!("Tensor shapes can not be broadcasted: {x_shape:?} and {y_shape:?}").into(),
                ));
            }
        }

        // The target shape is built from the operands' dim resolution —
        // `resolve_shape_without_variables` classifies every axis as
        // - Static(1) vs anything -> the other side's dim tensor
        // - Static vs Static -> max (equality checked above)
        // - Symbolic vs Symbolic -> the left side's dim tensor (folded values
        //   are already proven compatible above; a variable is const-backed,
        //   never unknown, so differing expressions are not an error)
        // - Symbolic vs Static(n != 1) -> the symbolic side's dim tensor
        //   (folded values already agree; the check defers nothing)
        let (rdx, rdy) = {
            let rt = RT.lock();
            (rt.resolve_shape_without_variables(x.id), rt.resolve_shape_without_variables(y.id))
        };
        let sdx = x.shape();
        let sdy = y.shape();
        // Right-align the shorter rank with Static(1).
        let rank = rdx.len().max(rdy.len());
        let rdim = |side: &[ResolvedDim], i: usize| {
            side.get(side.len().wrapping_sub(rank).wrapping_add(i)).copied().unwrap_or(ResolvedDim::Static(1))
        };
        fn dtensor(side: &[Tensor], i: usize, rank: usize) -> Option<&Tensor> {
            let offset = rank - side.len();
            if i < offset { None } else { Some(&side[i - offset]) }
        }
        let mut eshape_rdim: Vec<ResolvedDim> = Vec::new();
        let mut eshape_sym: Vec<Tensor> = Vec::new();
        for i in 0..rank {
            let (a, b) = (rdim(&rdx, i), rdim(&rdy, i));
            let target = match (a, b) {
                (ResolvedDim::Static(1), _) => match dtensor(&sdy, i, rank) {
                    Some(t) => t.clone(),
                    // Both sides are broadcast padding (both resolved 1).
                    None => Tensor::from(1),
                },
                (_, ResolvedDim::Static(1)) => match dtensor(&sdx, i, rank) {
                    Some(t) => t.clone(),
                    None => Tensor::from(1),
                },
                (ResolvedDim::Static(av), ResolvedDim::Static(bv)) => {
                    // Both static and non-1: equal (checked above); take the
                    // larger one's dim tensor (same const either way).
                    if av >= bv {
                        dtensor(&sdx, i, rank).expect("static dim has a dim tensor").clone()
                    } else {
                        dtensor(&sdy, i, rank).expect("static dim has a dim tensor").clone()
                    }
                }
                (ResolvedDim::Symbolic(_), ResolvedDim::Symbolic(_)) => {
                    // Different dim expressions with equal folded values
                    // (compatibility proven by the concrete check above).
                    // Propagate the left side's dim tensor.
                    dtensor(&sdx, i, rank).expect("symbolic dim has a dim tensor").clone()
                }
                (ResolvedDim::Symbolic(_), ResolvedDim::Static(_)) => {
                    // Symbolic vs static (non-1): folded values already agree
                    // (checked above). The symbolic side provides the target
                    // dim, keeping the shape symbolic downstream.
                    dtensor(&sdx, i, rank).expect("symbolic dim has a dim tensor").clone()
                }
                (ResolvedDim::Static(_), ResolvedDim::Symbolic(_)) => {
                    dtensor(&sdy, i, rank).expect("symbolic dim has a dim tensor").clone()
                }
            };
            eshape_sym.push(target);
            eshape_rdim.push(match (a, b) {
                (ResolvedDim::Static(av), ResolvedDim::Static(bv)) => ResolvedDim::Static(std::cmp::max(av, bv)),
                (ResolvedDim::Symbolic(t), _) | (_, ResolvedDim::Symbolic(t)) => ResolvedDim::Symbolic(t),
            });
        }
        //println!("Second broadcast operand {y}");
        //println!("{x_shape:?}, {eshape:?}");
        //println!("After reshape second broadcast operand {y}");
        // Expand only axes that actually differ from the operand's own dims.
        let needs_expand = |op: &[ResolvedDim]| -> bool {
            op.len() != rank
                || op.iter().enumerate().any(|(i, d)| {
                    let e = eshape_rdim[rank - op.len() + i];
                    !matches!((d, &e), (ResolvedDim::Static(a), ResolvedDim::Static(b)) if a == b)
                        && !matches!((d, &e), (ResolvedDim::Symbolic(a), ResolvedDim::Symbolic(b)) if a == b)
                })
        };
        if needs_expand(&rdx) {
            x = x.expand(eshape_sym.iter())?;
        }
        if needs_expand(&rdy) {
            y = y.expand(eshape_sym.iter())?;
        }
        Ok((x, y))
    }

    /// Tensor id
    #[must_use]
    pub const fn id(&self) -> TensorId {
        self.id
    }
}

impl TryFrom<Tensor> for bf16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [bf16::ZERO];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [f16::ZERO];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0.];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0.];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for u8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for u32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for bool {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [false];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl<T: Scalar> TryFrom<Tensor> for Vec<T> {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let numel = value.numel().item::<Dim>() as usize;
        let bytes = numel
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ZyxError::AllocationError("allocation size overflow".into()))?;
        let max_free = RT.lock().free_memory();
        if bytes as Dim > max_free {
            return Err(ZyxError::AllocationError(
                format!("cannot allocate Vec with {numel} elements ({bytes} bytes), max free memory is {max_free} bytes").into(),
            ));
        }
        let mut data = vec![T::zero(); numel as usize];
        RT.lock().load(value.id, &mut data)?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize> TryFrom<Tensor> for [T; D0] {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [T::zero(); D0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> TryFrom<Tensor> for [[T; D1]; D0] {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[T::zero(); D1]; D0];
        RT.lock().load(value.id, data.as_flattened_mut())?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> TryFrom<Tensor> for [[[T; D2]; D1]; D0] {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[[T::zero(); D2]; D1]; D0];
        RT.lock().load(value.id, data.as_flattened_mut().as_flattened_mut())?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize> TryFrom<Tensor>
    for [[[[T; D3]; D2]; D1]; D0]
{
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[[[T::zero(); D3]; D2]; D1]; D0];
        RT.lock().load(value.id, data.as_flattened_mut().as_flattened_mut().as_flattened_mut())?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> TryFrom<Tensor>
    for [[[[[T; D4]; D3]; D2]; D1]; D0]
{
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[[[[T::zero(); D4]; D3]; D2]; D1]; D0];
        RT.lock().load(value.id, data.as_flattened_mut().as_flattened_mut().as_flattened_mut().as_flattened_mut())?;
        Ok(data)
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
        //f.write_fmt(format_args!("Tensor {{ id = {:?} }}", self.id))
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO don't print the whole tensor if it is too big
        let precision = f.precision().unwrap_or(3);
        let x = self.clone();
        let res = match self.dtype() {
            DType::BF16 => {
                let data: Result<Vec<bf16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F16 => {
                let data: Result<Vec<f16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F32 => {
                let data: Result<Vec<f32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f32 tensor failed to realize {e:?}"),
                }
            }
            DType::F64 => {
                let data: Result<Vec<f64>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f64 tensor failed to realize {e:?}"),
                }
            }
            DType::F8E4M3 => {
                let data: Result<Vec<f8e4m3>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f8e4m3 tensor failed to realize {e:?}"),
                }
            }
            DType::F8E5M2 => {
                let data: Result<Vec<f8e5m2>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), precision, f.width()),
                    Err(e) => format!("f8e5m2 tensor failed to realize {e:?}"),
                }
            }
            DType::U8 => {
                let data: Result<Vec<u8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("u8 tensor failed to realize {e:?}"),
                }
            }
            DType::U16 => {
                let data: Result<Vec<u16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("u16 tensor failed to realize {e:?}"),
                }
            }
            DType::U32 => {
                let data: Result<Vec<u32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("u32 tensor failed to realize {e:?}"),
                }
            }
            DType::U64 => {
                let data: Result<Vec<u64>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("u64 tensor failed to realize {e:?}"),
                }
            }
            DType::I8 => {
                let data: Result<Vec<i8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I16 => {
                let data: Result<Vec<i16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I32 => {
                let data: Result<Vec<i32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I64 => {
                let data: Result<Vec<i64>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 0, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::Bool => {
                let data: Result<Vec<bool>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.resolve_shape(), 5, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
        };
        f.write_fmt(format_args!("{res}\ntensor {} {:?}", self.dtype(), self.resolve_shape()))
    }
}

fn tensor_to_string<T: core::fmt::Display>(data: &[T], shape: &[Dim], precision: usize, width: Option<usize>) -> String {
    use core::fmt::Write;
    let n: Dim = shape.iter().product();
    let rank = shape.len();
    let mut res = String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let w = width.unwrap_or_else(|| data.iter().map(|x| format!("{x:>.precision$}").len()).max().unwrap_or(0));
    // Rank-0 (scalar): just the value, no brackets.
    if rank == 0 {
        let _ = write!(res, "{:>w$.precision$}", data[0]);
        return res;
    }
    let d0 = shape[rank - 1];
    for (i, x) in data.iter().enumerate() {
        {
            let mut var: Dim = 1;
            let mut r = rank;
            while r > 0 {
                if (i as Dim) % (n / var) == 0 {
                    res += &(" ".repeat(rank - r) + "[".repeat(r - 1).as_str());
                    break;
                }
                var *= shape[rank - r];
                r -= 1;
            }
        }
        let _ = write!(res, "{x:>w$.precision$}");
        if !((i as Dim + 1) % d0 == 0) {
            res += "  ";
        }
        {
            let mut var: Dim = 1;
            let mut r = rank;
            while r > 0 {
                if (i as Dim + 1) % (n / var) == 0 {
                    res += &"]".repeat(r - 1);
                    break;
                }
                var *= shape[rank - r];
                r -= 1;
            }
        }
        if ((i as Dim + 1) % d0 == 0) && i as Dim != n - 1 {
            res += "\n";
        }
    }
    res
}

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl<T: Scalar> From<T> for Tensor {
    fn from(value: T) -> Self {
        let mut rt = RT.lock();
        let id = rt.new_host_tensor(TensorId::NULL, Box::new([value])).unwrap();
        Tensor { id }
    }
}

impl<T: Scalar, const D0: usize> From<[T; D0]> for Tensor {
    fn from(data: [T; D0]) -> Self {
        let shape = Tensor::stack(&[Tensor::from(D0 as Dim)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, Box::from(data)).unwrap() }
    }
}

impl<T: Scalar> From<Vec<T>> for Tensor {
    fn from(data: Vec<T>) -> Self {
        let len = data.len() as Dim;
        let shape = Tensor::stack(&[Tensor::from(len)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, data.into_boxed_slice()).unwrap() }
    }
}

impl<T: Scalar + Clone> From<Vec<Vec<T>>> for Tensor {
    fn from(data: Vec<Vec<T>>) -> Self {
        let rows = data.len() as Dim;
        let cols = data.first().map_or(0, Vec::len) as Dim;
        let flat: Vec<T> = data.into_iter().flatten().collect();
        let shape = Tensor::stack(&[Tensor::from(rows), Tensor::from(cols)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, flat.into_boxed_slice()).unwrap() }
    }
}

impl<T: Scalar + Clone> From<Vec<Vec<Vec<T>>>> for Tensor {
    fn from(data: Vec<Vec<Vec<T>>>) -> Self {
        let depth = data.len() as Dim;
        let rows = data.first().map_or(0, Vec::len) as Dim;
        let cols = data.first().and_then(|v| v.first()).map_or(0, Vec::len) as Dim;
        let flat: Vec<T> = data.into_iter().flatten().flatten().collect();
        let shape = Tensor::stack(&[Tensor::from(depth), Tensor::from(rows), Tensor::from(cols)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, flat.into_boxed_slice()).unwrap() }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> From<[[T; D1]; D0]> for Tensor {
    fn from(data: [[T; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0].as_ptr(), D0 * D1) };
        let data = Box::from(data);
        let shape = Tensor::stack(&[Tensor::from(D0 as Dim), Tensor::from(D1 as Dim)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, data).unwrap() }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> From<[[[T; D2]; D1]; D0]> for Tensor {
    fn from(data: [[[T; D2]; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0][0].as_ptr(), D0 * D1 * D2) };
        let shape = Tensor::stack(&[Tensor::from(D0 as Dim), Tensor::from(D1 as Dim), Tensor::from(D2 as Dim)]).unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, Box::from(data)).unwrap() }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize> From<[[[[T; D3]; D2]; D1]; D0]> for Tensor {
    fn from(data: [[[[T; D3]; D2]; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0][0][0].as_ptr(), D0 * D1 * D2 * D3) };
        let shape = Tensor::stack(&[
            Tensor::from(D0 as Dim),
            Tensor::from(D1 as Dim),
            Tensor::from(D2 as Dim),
            Tensor::from(D3 as Dim),
        ])
        .unwrap();
        Tensor { id: RT.lock().new_host_tensor(shape.id, Box::from(data)).unwrap() }
    }
}

impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        let data: f32 = self.clone().try_into().unwrap();
        Scalar::is_equal(data, *other)
    }
}

impl PartialEq<f64> for Tensor {
    fn eq(&self, other: &f64) -> bool {
        let data: f64 = self.clone().try_into().unwrap();
        Scalar::is_equal(data, *other)
    }
}

impl PartialEq<i32> for Tensor {
    fn eq(&self, other: &i32) -> bool {
        let data: i32 = self.clone().try_into().unwrap();
        Scalar::is_equal(data, *other)
    }
}

impl<T: Scalar> PartialEq<Vec<T>> for Tensor {
    fn eq(&self, other: &Vec<T>) -> bool {
        if self.resolve_shape() != [other.len() as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: Vec<T> = data;
                for (x, y) in data.into_iter().zip(other) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar> PartialEq<Vec<Vec<T>>> for Tensor {
    fn eq(&self, other: &Vec<Vec<T>>) -> bool {
        if self.resolve_shape() != [other.len() as Dim, other[0].len() as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: Vec<T> = data;
                for (x, y) in data.into_iter().zip(other.iter().flatten()) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar> PartialEq<Vec<Vec<Vec<T>>>> for Tensor {
    fn eq(&self, other: &Vec<Vec<Vec<T>>>) -> bool {
        if self.resolve_shape() != [other.len() as Dim, other[0].len() as Dim, other[0][0].len() as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: Vec<T> = data;
                for (x, y) in data.into_iter().zip(other.iter().flatten().flatten()) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar, const D0: usize> PartialEq<[T; D0]> for Tensor {
    fn eq(&self, other: &[T; D0]) -> bool {
        if self.resolve_shape() != [D0 as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: [T; D0] = data;
                for (x, y) in data.into_iter().zip(other) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> PartialEq<[[T; D1]; D0]> for Tensor {
    fn eq(&self, other: &[[T; D1]; D0]) -> bool {
        if self.resolve_shape() != [D0 as Dim, D1 as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: [[T; D1]; D0] = data;
                for (x, y) in data.into_iter().flatten().zip(other.iter().flatten()) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> PartialEq<[[[T; D2]; D1]; D0]> for Tensor {
    fn eq(&self, other: &[[[T; D2]; D1]; D0]) -> bool {
        if self.resolve_shape() != [D0 as Dim, D1 as Dim, D2 as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: [[[T; D2]; D1]; D0] = data;
                for (x, y) in data.into_iter().flatten().flatten().zip(other.iter().flatten().flatten()) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize> PartialEq<[[[[T; D3]; D2]; D1]; D0]>
    for Tensor
{
    fn eq(&self, other: &[[[[T; D3]; D2]; D1]; D0]) -> bool {
        if self.resolve_shape() != [D0 as Dim, D1 as Dim, D2 as Dim, D3 as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: [[[[T; D3]; D2]; D1]; D0] = data;
                for (x, y) in data.into_iter().flatten().flatten().flatten().zip(other.iter().flatten().flatten().flatten()) {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    PartialEq<[[[[[T; D4]; D3]; D2]; D1]; D0]> for Tensor
{
    fn eq(&self, other: &[[[[[T; D4]; D3]; D2]; D1]; D0]) -> bool {
        if self.resolve_shape() != [D0 as Dim, D1 as Dim, D2 as Dim, D3 as Dim, D4 as Dim] {
            return false;
        }
        match self.clone().try_into() {
            Ok(data) => {
                let data: [[[[[T; D4]; D3]; D2]; D1]; D0] = data;
                for (x, y) in data
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .flatten()
                    .zip(other.iter().flatten().flatten().flatten().flatten())
                {
                    if !Scalar::is_equal(x, *y) {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor { id: RT.lock().unary(self.id, UOp::Neg) }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor { id: RT.lock().unary(self.id, UOp::Neg) }
    }
}

impl Not for Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        self.equal(0).unwrap()
    }
}

impl Not for &Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        self.equal(0).unwrap()
    }
}
