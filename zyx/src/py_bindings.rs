// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx - updated for symbolic shapes

use crate::DebugMask;
use crate::kernel::{CompiledKernel, Kernel, MemLayout, MemScope, OpId, ParamKind};
use crate::Dev;
use crate::shape::Dim;
use crate::tape::FrozenTape;
use crate::tensor::{Axis, DebugGuard, ReduceOp};
use crate::{DType, Tape, Tensor, ZyxError};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::{
    Bound, PyAny, PyErr, PyResult,
    exceptions::{PyOSError, PyTypeError},
    pymethods,
    types::{PyAnyMethods, PyIterator, PyList, PyModule, PyModuleMethods, PyTuple},
};

impl From<ZyxError> for PyErr {
    fn from(err: ZyxError) -> Self {
        PyOSError::new_err(format!("{err:?}"))
    }
}

// helpers

fn to_tensor(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(t) = obj.extract::<Tensor>() {
        return Ok(t);
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Tensor::from(v));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Tensor::from(v));
    }
    Err(PyTypeError::new_err("expected Tensor or numeric (int/float) for dim/arg"))
}

fn parse_shape(shape: &Bound<'_, PyTuple>) -> PyResult<Vec<Tensor>> {
    // *shape where each is int or Tensor, or single list/tuple of dims
    if shape.len() == 1 {
        let first = shape.get_item(0).unwrap();
        if first.is_instance_of::<PyList>() || first.is_instance_of::<PyTuple>() {
            let iter = PyIterator::from_object(&first).unwrap();
            let mut vec = Vec::new();
            for item in iter {
                let obj = item.unwrap();
                vec.push(to_tensor(&obj)?);
            }
            return Ok(vec.into_iter().map(|t| t.cast(crate::kernel::IDX_T)).collect());
        }
    }
    let mut vec = Vec::with_capacity(shape.len());
    for item in shape.iter() {
        vec.push(to_tensor(&item)?);
    }
    Ok(vec.into_iter().map(|t| t.cast(crate::kernel::IDX_T)).collect())
}

fn parse_shape_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return parse_shape(tuple);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut v = Vec::new();
        for item in list.iter() {
            v.push(to_tensor(&item)?);
        }
        return Ok(v.into_iter().map(|t| t.cast(crate::kernel::IDX_T)).collect());
    }
    // single int/Tensor
    Ok(vec![to_tensor(obj)?.cast(crate::kernel::IDX_T)])
}

fn to_ax(axes: &Bound<'_, PyAny>) -> Vec<Axis> {
    if axes.is_none() {
        return vec![];
    }
    if let Ok(tuple) = axes.cast::<PyTuple>() {
        let mut result = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            if let Ok(ax) = item.extract::<Axis>() {
                result.push(ax);
            } else if let Ok(nested) = item.cast::<PyTuple>() {
                for nested_item in nested.iter() {
                    if let Ok(ax) = nested_item.extract::<Axis>() {
                        result.push(ax);
                    }
                }
            } else if let Ok(nested) = item.cast::<PyList>() {
                for nested_item in nested.iter() {
                    if let Ok(ax) = nested_item.extract::<Axis>() {
                        result.push(ax);
                    }
                }
            }
        }
        return result;
    }
    if let Ok(list) = axes.cast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            if let Ok(ax) = item.extract::<Axis>() {
                result.push(ax);
            }
        }
        return result;
    }
    if let Ok(single) = axes.extract::<Axis>() {
        return vec![single];
    }
    vec![]
}

fn extract_tensor_or_scalar(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(t) = obj.extract::<Tensor>() {
        return Ok(t);
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Tensor::from(v));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Tensor::from(v));
    }
    Err(PyTypeError::new_err("expected Tensor or numeric"))
}

#[pymethods]
impl Tape {
    #[new]
    pub fn py_new() -> Self {
        Tape::empty()
    }

    #[staticmethod]
    #[pyo3(name = "new")]
    pub fn py_new_with_params(params: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tensors = extract_tensor_list(params)?;
        Tape::new(&tensors).map_err(|e| e.into())
    }

    #[pyo3(name = "add")]
    pub fn add_py(&self, tensor: &Tensor) -> PyResult<()> {
        self.add(tensor).map_err(|e| e.into())
    }

    #[pyo3(name = "extend")]
    pub fn extend_py(&self, params: &Bound<'_, PyAny>) -> PyResult<()> {
        let tensors = extract_tensor_list(params)?;
        self.extend(&tensors).map_err(|e| e.into())
    }

    #[must_use]
    #[pyo3(name = "gradient")]
    pub fn gradient_py(&self, x: &Tensor, sources: &Bound<'_, PyList>) -> Vec<Tensor> {
        let sources: Vec<Tensor> =
            sources.into_iter().map(|d| d.extract::<Tensor>().expect("sources must be List(Tensor)")).collect();
        self.gradient(x, &sources)
    }

    #[pyo3(name = "realize")]
    pub fn realize_py(&mut self, tensors: &Bound<'_, PyAny>) -> PyResult<()> {
        let tensors = extract_tensor_list(tensors)?;
        let old = std::mem::replace(self, Tape::empty());
        old.realize(&tensors).map_err(|e| e.into())
    }

    #[pyo3(name = "freeze")]
    pub fn freeze_py(&mut self, outputs: &Bound<'_, PyAny>) -> PyResult<FrozenTape> {
        let tensors = extract_tensor_list(outputs)?;
        let old = std::mem::replace(self, Tape::empty());
        old.freeze(&tensors).map_err(|e| e.into())
    }
}

#[pymethods]
impl FrozenTape {
    #[pyo3(name = "replay")]
    pub fn replay_py(&self, inputs: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
        let tensors = extract_tensor_list(inputs)?;
        self.replay(&tensors).map_err(|e| e.into())
    }
}

// helper to extract Vec<Tensor> from PyAny that may be list/tuple/single
fn extract_tensor_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    if obj.is_instance_of::<PyList>() {
        let list = obj.cast::<PyList>().unwrap();
        let mut v = Vec::new();
        for item in list.iter() {
            v.push(item.extract::<Tensor>().map_err(|_| PyTypeError::new_err("expected Tensor in list"))?);
        }
        return Ok(v);
    }
    if obj.is_instance_of::<PyTuple>() {
        let tuple = obj.cast::<PyTuple>().unwrap();
        let mut v = Vec::new();
        for item in tuple.iter() {
            v.push(item.extract::<Tensor>().map_err(|_| PyTypeError::new_err("expected Tensor in tuple"))?);
        }
        return Ok(v);
    }
    if let Ok(t) = obj.extract::<Tensor>() {
        return Ok(vec![t]);
    }
    Err(PyTypeError::new_err("expected Tensor or list/tuple of Tensors"))
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (py_obj, dtype=None))]
    fn new(py_obj: &Bound<'_, PyAny>, dtype: Option<DType>) -> PyResult<Self> {
        let tensor = if let Ok(tensor) = from_numpy::<f32>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<f64>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<i8>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<i16>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<i32>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<i64>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<u8>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<u16>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<u32>(py_obj) {
            Ok(tensor)
        } else if let Ok(tensor) = from_numpy::<u64>(py_obj) {
            Ok(tensor)
        } else if let Ok(val) = py_obj.extract::<i64>() {
            Ok(Tensor::from(val))
        } else if let Ok(val) = py_obj.extract::<f64>() {
            Ok(Tensor::from(val))
        } else if let Ok(vec) = py_obj.extract::<Vec<i64>>() {
            Ok(Tensor::from(vec))
        } else if let Ok(vec) = py_obj.extract::<Vec<f64>>() {
            Ok(Tensor::from(vec))
        } else if let Ok(mat) = py_obj.extract::<Vec<Vec<i64>>>() {
            Ok(Tensor::from(mat))
        } else if let Ok(mat) = py_obj.extract::<Vec<Vec<f64>>>() {
            Ok(Tensor::from(mat))
        } else {
            Err(PyTypeError::new_err("Unsupported input type for Tensor"))
        }?;

        if let Some(target_dtype) = dtype
            && tensor.dtype() != target_dtype
        {
            return Ok(tensor.cast(target_dtype));
        }
        Ok(tensor)
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.resolve_shape();
        let np = py.import("numpy")?;
        Ok(match self.dtype() {
            DType::BF16 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => {
                let data: Vec<f32> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "float32"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::F64 => {
                let data: Vec<f64> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "float64"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U8 => {
                let data: Vec<u8> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "uint8"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U16 => {
                let data: Vec<u16> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "uint16"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U32 => {
                let data: Vec<u32> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "uint32"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U64 => {
                let data: Vec<u64> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "uint64"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I8 => {
                let data: Vec<i8> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "int8"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I16 => {
                let data: Vec<i16> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "int16"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I32 => {
                let data: Vec<i32> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "int32"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I64 => {
                let data: Vec<i64> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "int64"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::Bool => {
                let data: Vec<bool> = self.clone().try_into()?;
                np.getattr("array")?.call1((data, "bool"))?.call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
        })
    }

    #[staticmethod]
    #[pyo3(name = "manual_seed")]
    pub fn manual_seed_py(seed: u64) {
        Tensor::manual_seed(seed);
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "training")]
    pub fn training_py() -> bool {
        Tensor::training()
    }

    #[staticmethod]
    #[pyo3(name = "set_training")]
    pub fn set_training_py(training: bool) {
        Tensor::set_training(training);
    }

    // symbolic shape: returns Vec<Tensor>
    #[must_use]
    #[pyo3(name = "shape")]
    pub fn shape_py(&self) -> Vec<Tensor> {
        self.shape()
    }

    #[must_use]
    #[pyo3(name = "resolve_shape")]
    pub fn resolve_shape_py(&self) -> Vec<Dim> {
        self.resolve_shape()
    }

    #[must_use]
    #[pyo3(name = "numel")]
    pub fn numel_py(&self) -> Tensor {
        self.numel()
    }

    #[must_use]
    #[pyo3(name = "rank")]
    pub fn rank_py(&self) -> Dim {
        self.rank()
    }

    #[must_use]
    #[pyo3(name = "dtype")]
    pub fn dtype_py(&self) -> DType {
        self.dtype()
    }

    #[must_use]
    #[pyo3(name = "is_realized")]
    pub fn is_realized_py(&self) -> bool {
        self.is_realized()
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "implicit_casts")]
    pub fn implicit_casts_py() -> bool {
        Tensor::implicit_casts()
    }

    #[staticmethod]
    #[pyo3(name = "set_implicit_casts")]
    pub fn set_implicit_casts_py(implicit_casts: bool) {
        Tensor::set_implicit_casts(implicit_casts);
    }

    #[pyo3(name = "detach")]
    pub fn detach_py(&self) -> Result<Tensor, ZyxError> {
        self.clone().detach()
    }

    #[pyo3(name = "assign")]
    pub fn assign_py(&self, src: &Bound<'_, PyAny>) -> PyResult<()> {
        let src = extract_tensor_or_scalar(src).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.clone().assign(src).map_err(|e| e.into())
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "with_debug")]
    pub fn with_debug_py(debug: DebugMask) -> DebugGuard {
        Tensor::with_debug(debug)
    }

    #[staticmethod]
    #[pyo3(name = "variable", signature = (val))]
    pub fn variable_py(val: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(v) = val.extract::<i64>() {
            Ok(Tensor::variable(v))
        } else if let Ok(v) = val.extract::<f64>() {
            Ok(Tensor::variable(v))
        } else if let Ok(v) = val.extract::<f32>() {
            Ok(Tensor::variable(v))
        } else {
            Err(PyTypeError::new_err("variable expects numeric scalar"))
        }
    }

    #[staticmethod]
    #[pyo3(name = "randn", signature = (*shape, dtype=DType::F32))]
    pub fn randn_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::randn(parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?, dtype)
    }

    #[pyo3(name = "multinomial")]
    pub fn multinomial_py(&self, num_samples: Dim, replacement: bool) -> Result<Tensor, ZyxError> {
        self.multinomial(num_samples, replacement)
    }

    #[staticmethod]
    #[pyo3(name = "rand", signature = (*shape, dtype=DType::F32))]
    pub fn rand_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::rand(parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?, dtype)
    }

    #[staticmethod]
    #[pyo3(name = "uniform", signature = (*shape, dtype=DType::F32))]
    pub fn uniform_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::rand(parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?, dtype)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_", signature = (*shape, from_=-1.0, to_=1.0, dtype=DType::F32))]
    pub fn uniform_py_with_range(shape: &Bound<'_, PyTuple>, from_: f32, to_: f32, dtype: DType) -> Result<Tensor, ZyxError> {
        let tensor = Tensor::rand(parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?, dtype)?;
        let range = to_ - from_;
        let scaled = tensor * range + from_;
        Ok(scaled)
    }

    #[staticmethod]
    #[pyo3(name = "randint", signature = (*shape, low=0, high=10, dtype=DType::I64))]
    pub fn randint_py(shape: &Bound<'_, PyTuple>, low: i64, high: i64, dtype: DType) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        let res: Result<Tensor, ZyxError> = match dtype {
            DType::U8 => Tensor::randint::<u8>(shape_vec, low as u8..high as u8),
            DType::U16 => Tensor::randint::<u16>(shape_vec, low as u16..high as u16),
            DType::U32 => Tensor::randint::<u32>(shape_vec, low as u32..high as u32),
            DType::U64 => Tensor::randint::<u64>(shape_vec, low as u64..high as u64),
            DType::I8 => Tensor::randint::<i8>(shape_vec, low as i8..high as i8),
            DType::I16 => Tensor::randint::<i16>(shape_vec, low as i16..high as i16),
            DType::I32 => Tensor::randint::<i32>(shape_vec, low as i32..high as i32),
            DType::I64 => Tensor::randint::<i64>(shape_vec, low..high),
            _ => return Err(PyTypeError::new_err("randint unsupported dtype")),
        };
        res.map_err(|e| e.into())
    }

    #[staticmethod]
    #[pyo3(name = "kaiming_uniform", signature = (*shape, a=0.0, dtype=DType::F32))]
    pub fn kaiming_uniform_py(shape: &Bound<'_, PyTuple>, a: f64, dtype: DType) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Tensor::kaiming_uniform::<f32>(shape_vec, a as f32).map(|t| t.cast(dtype)).map_err(|e| e.into())
    }

    #[staticmethod]
    #[pyo3(name = "glorot_uniform", signature = (*shape, dtype=DType::F32))]
    pub fn glorot_uniform_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Tensor::glorot_uniform(shape_vec, dtype).map_err(|e| e.into())
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "zeros", signature = (*shape, dtype=DType::F32))]
    pub fn zeros_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Ok(Tensor::zeros(shape_vec, dtype))
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "ones", signature = (*shape, dtype=DType::F32))]
    pub fn ones_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Ok(Tensor::ones(shape_vec, dtype))
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "full", signature = (*shape, a))]
    pub fn full_py(shape: &Bound<'_, PyTuple>, a: f64) -> PyResult<Tensor> {
        let shape_vec = parse_shape(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Ok(Tensor::full(shape_vec, a))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_like")]
    pub fn zeros_like_py(input: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(tensor) = input.extract::<Tensor>() {
            Ok(Tensor::zeros_like(tensor))
        } else {
            Err(ZyxError::DTypeError("input must be a Tensor".into()))
        }
    }

    #[staticmethod]
    #[pyo3(name = "ones_like")]
    pub fn ones_like_py(input: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(tensor) = input.extract::<Tensor>() {
            Ok(Tensor::ones_like(tensor))
        } else {
            Err(ZyxError::DTypeError("input must be a Tensor".into()))
        }
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "eye", signature = (n, dtype=DType::F32))]
    pub fn eye_py(n: Dim, dtype: DType) -> Tensor {
        Tensor::eye(n, dtype)
    }

    #[staticmethod]
    #[pyo3(name = "arange", signature = (start=0, stop=1, step=1))]
    pub fn arange_py(start: i64, stop: i64, step: i64) -> Result<Tensor, ZyxError> {
        Tensor::arange(start, stop, step)
    }

    #[staticmethod]
    #[pyo3(name = "from_vec", signature = (data, shape))]
    pub fn from_vec_py(data: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let shape_vec = parse_shape_any(shape).map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        // try f32
        if let Ok(vec) = data.extract::<Vec<f32>>() {
            return Tensor::from_vec(vec, shape_vec).map_err(|e| e.into());
        }
        if let Ok(vec) = data.extract::<Vec<f64>>() {
            return Tensor::from_vec(vec, shape_vec).map_err(|e| e.into());
        }
        if let Ok(vec) = data.extract::<Vec<i64>>() {
            return Tensor::from_vec(vec, shape_vec).map_err(|e| e.into());
        }
        Err(PyTypeError::new_err("unsupported data for from_vec"))
    }

    #[must_use]
    #[pyo3(name = "cast")]
    pub fn cast_py(&self, dtype: DType) -> Tensor {
        self.cast(dtype)
    }

    #[pyo3(name = "bitcast")]
    pub unsafe fn bitcast_py(&self, dtype: DType) -> Result<Tensor, ZyxError> {
        unsafe { self.bitcast(dtype) }
    }

    #[must_use]
    #[pyo3(name = "dropout")]
    pub fn dropout_py(&self, probability: f32) -> Tensor {
        self.dropout(probability)
    }

    #[must_use]
    #[pyo3(name = "interpolate")]
    pub fn interpolate_py(&self, target: &Bound<'_, PyAny>, weight: f32) -> PyResult<Tensor> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        Ok(self.interpolate(&target, weight))
    }

    #[must_use]
    #[pyo3(name = "smooth_l1_loss")]
    pub fn smooth_l1_loss_py(&self, target: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        Ok(self.smooth_l1_loss(&target))
    }

    #[must_use]
    #[pyo3(name = "huber_loss")]
    pub fn huber_loss_py(&self, target: &Bound<'_, PyAny>, delta: f64) -> PyResult<Tensor> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        Ok(self.huber_loss(&target, delta))
    }

    // elementwise

    #[must_use]
    #[pyo3(name = "abs")]
    pub fn abs_py(&self) -> Tensor {
        self.abs()
    }
    #[must_use]
    #[pyo3(name = "square")]
    pub fn square_py(&self) -> Tensor {
        self.square()
    }
    #[must_use]
    #[pyo3(name = "sign")]
    pub fn sign_py(&self) -> Tensor {
        self.sign()
    }
    #[must_use]
    #[pyo3(name = "erf")]
    pub fn erf_py(&self) -> Tensor {
        self.erf()
    }
    #[must_use]
    #[pyo3(name = "erfinv")]
    pub fn erfinv_py(&self) -> Tensor {
        self.erfinv()
    }
    #[must_use]
    #[pyo3(name = "cos")]
    pub fn cos_py(&self) -> Tensor {
        self.cos()
    }
    #[must_use]
    #[pyo3(name = "cosh")]
    pub fn cosh_py(&self) -> Tensor {
        self.cosh()
    }
    #[must_use]
    #[pyo3(name = "exp")]
    pub fn exp_py(&self) -> Tensor {
        self.exp()
    }
    #[must_use]
    #[pyo3(name = "exp2")]
    pub fn exp2_py(&self) -> Tensor {
        self.exp2()
    }
    #[must_use]
    #[pyo3(name = "floor")]
    pub fn floor_py(&self) -> Tensor {
        self.floor()
    }
    #[must_use]
    #[pyo3(name = "trunc")]
    pub fn trunc_py(&self) -> Tensor {
        self.trunc()
    }
    #[must_use]
    #[pyo3(name = "log2")]
    pub fn log2_py(&self) -> Tensor {
        self.log2()
    }
    #[must_use]
    #[pyo3(name = "ln")]
    pub fn ln_py(&self) -> Tensor {
        self.ln()
    }
    #[must_use]
    #[pyo3(name = "reciprocal")]
    pub fn reciprocal_py(&self) -> Tensor {
        self.reciprocal()
    }
    #[must_use]
    #[pyo3(name = "relu")]
    pub fn relu_py(&self) -> Tensor {
        self.relu()
    }
    #[must_use]
    #[pyo3(name = "rsqrt")]
    pub fn rsqrt_py(&self) -> Tensor {
        self.rsqrt()
    }
    #[must_use]
    #[pyo3(name = "sigmoid")]
    pub fn sigmoid_py(&self) -> Tensor {
        self.sigmoid()
    }
    #[must_use]
    #[pyo3(name = "sin")]
    pub fn sin_py(&self) -> Tensor {
        self.sin()
    }
    #[must_use]
    #[pyo3(name = "sinh")]
    pub fn sinh_py(&self) -> Tensor {
        self.sinh()
    }
    #[must_use]
    #[pyo3(name = "sqrt")]
    pub fn sqrt_py(&self) -> Tensor {
        self.sqrt()
    }
    #[must_use]
    #[pyo3(name = "tan")]
    pub fn tan_py(&self) -> Tensor {
        self.tan()
    }
    #[must_use]
    #[pyo3(name = "tanh")]
    pub fn tanh_py(&self) -> Tensor {
        self.tanh()
    }
    #[must_use]
    #[pyo3(name = "gelu")]
    pub fn gelu_py(&self) -> Tensor {
        self.gelu()
    }
    #[must_use]
    #[pyo3(name = "bitnot")]
    pub fn bitnot_py(&self) -> Tensor {
        self.bitnot()
    }
    #[must_use]
    #[pyo3(name = "ceil")]
    pub fn ceil_py(&self) -> Tensor {
        self.ceil()
    }
    #[must_use]
    #[pyo3(name = "frac")]
    pub fn frac_py(&self) -> Tensor {
        self.frac()
    }
    #[must_use]
    #[pyo3(name = "isnan")]
    pub fn isnan_py(&self) -> Tensor {
        self.isnan()
    }
    #[must_use]
    #[pyo3(name = "isinf")]
    pub fn isinf_py(&self) -> Tensor {
        self.isinf()
    }
    #[must_use]
    #[pyo3(name = "log10")]
    pub fn log10_py(&self) -> Tensor {
        self.log10()
    }
    #[must_use]
    #[pyo3(name = "rad2deg")]
    pub fn rad2deg_py(&self) -> Tensor {
        self.rad2deg()
    }
    #[must_use]
    #[pyo3(name = "deg2rad")]
    pub fn deg2rad_py(&self) -> Tensor {
        self.deg2rad()
    }
    #[must_use]
    #[pyo3(name = "round")]
    pub fn round_py(&self) -> Tensor {
        self.round()
    }
    #[must_use]
    #[pyo3(name = "mish")]
    pub fn mish_py(&self) -> Tensor {
        self.mish()
    }
    #[must_use]
    #[pyo3(name = "quick_gelu")]
    pub fn quick_gelu_py(&self) -> Tensor {
        self.quick_gelu()
    }
    #[must_use]
    #[pyo3(name = "selu")]
    pub fn selu_py(&self) -> Tensor {
        self.selu()
    }
    #[must_use]
    #[pyo3(name = "hard_sigmoid")]
    pub fn hard_sigmoid_py(&self) -> Tensor {
        self.hard_sigmoid()
    }
    #[must_use]
    #[pyo3(name = "swish")]
    pub fn swish_py(&self) -> Tensor {
        self.swish()
    }

    #[pyo3(name = "log")]
    pub fn log_py(&self, base: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let base = extract_tensor_or_scalar(base).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        Ok(self.log(base))
    }

    #[must_use]
    #[pyo3(name = "leaky_relu")]
    pub fn leaky_relu_py(&self, neg_slope: &Bound<'_, PyAny>) -> Tensor {
        if let Ok(ns) = neg_slope.extract::<f64>() {
            return self.leaky_relu(ns);
        }
        if let Ok(ns) = neg_slope.extract::<i64>() {
            return self.leaky_relu(ns);
        }
        panic!("neg_slope must be numeric");
    }

    #[pyo3(name = "celu")]
    pub fn celu_py(&self, alpha: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(v) = alpha.extract::<f64>() {
            Ok(self.celu(v))
        } else if let Ok(v) = alpha.extract::<i64>() {
            Ok(self.celu(v))
        } else {
            Err(PyTypeError::new_err("alpha must be numeric"))
        }
    }

    #[pyo3(name = "elu")]
    pub fn elu_py(&self, alpha: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(v) = alpha.extract::<f64>() {
            Ok(self.elu(v))
        } else if let Ok(v) = alpha.extract::<i64>() {
            Ok(self.elu(v))
        } else {
            Err(PyTypeError::new_err("alpha must be numeric"))
        }
    }

    #[pyo3(name = "softmax")]
    pub fn softmax_py(&self, axes: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        self.softmax(to_ax(axes))
    }

    #[pyo3(name = "ln_softmax")]
    pub fn ln_softmax_py(&self, axes: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        self.ln_softmax(to_ax(axes))
    }

    #[pyo3(name = "log_softmax")]
    pub fn log_softmax_py(&self, axes: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        self.ln_softmax(to_ax(axes))
    }

    // reduce unified

    #[pyo3(name = "sum", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn sum_py(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Sum, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Sum, axes, dtype, 1)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Sum, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Sum, axes, dtype, 1)
            }
        }
    }

    #[pyo3(name = "mean", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn mean_py(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Mean, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Mean, axes, dtype, 1)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Mean, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Mean, axes, dtype, 1)
            }
        }
    }

    #[pyo3(name = "var", signature = (dim=None, keepdim=false, unbiased=true, dtype=None))]
    pub fn var_py(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        unbiased: bool,
        dtype: Option<DType>,
    ) -> Result<Tensor, ZyxError> {
        let correction: Dim = if unbiased { 1 } else { 0 };
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Var, axes, dtype, correction)
            } else {
                self.reduce_impl::<false>(ReduceOp::Var, axes, dtype, correction)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Var, axes, dtype, correction)
            } else {
                self.reduce_impl::<false>(ReduceOp::Var, axes, dtype, correction)
            }
        }
    }

    #[pyo3(name = "std", signature = (dim=None, keepdim=false, unbiased=true, dtype=None))]
    pub fn std_py(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        unbiased: bool,
        dtype: Option<DType>,
    ) -> Result<Tensor, ZyxError> {
        let correction: Dim = if unbiased { 1 } else { 0 };
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Std, axes, dtype, correction)
            } else {
                self.reduce_impl::<false>(ReduceOp::Std, axes, dtype, correction)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Std, axes, dtype, correction)
            } else {
                self.reduce_impl::<false>(ReduceOp::Std, axes, dtype, correction)
            }
        }
    }

    #[pyo3(name = "min", signature = (dim=None, keepdim=false))]
    pub fn min_py(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> Result<Tensor, ZyxError> {
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Min, axes, None, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Min, axes, None, 1)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Min, axes, None, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Min, axes, None, 1)
            }
        }
    }

    #[pyo3(name = "max", signature = (dim=None, keepdim=false))]
    pub fn max_py(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> Result<Tensor, ZyxError> {
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Max, axes, None, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Max, axes, None, 1)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Max, axes, None, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Max, axes, None, 1)
            }
        }
    }

    #[pyo3(name = "prod", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn prod_py(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        if let Some(d) = dim {
            let axes = to_ax(d);
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Prod, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Prod, axes, dtype, 1)
            }
        } else {
            let axes: Vec<Axis> = (0..self.rank() as Axis).collect();
            if keepdim {
                self.reduce_impl::<true>(ReduceOp::Prod, axes, dtype, 1)
            } else {
                self.reduce_impl::<false>(ReduceOp::Prod, axes, dtype, 1)
            }
        }
    }

    #[pyo3(name = "cumsum")]
    pub fn cumsum_py(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cumsum(axis)
    }

    #[pyo3(name = "cummax")]
    pub fn cummax_py(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cummax(axis)
    }

    #[pyo3(name = "cumprod")]
    pub fn cumprod_py(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.cumprod(axis)
    }

    #[pyo3(name = "softplus")]
    pub fn softplus_py(&self, beta: &Bound<'_, PyAny>, threshold: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(beta_val) = beta.extract::<f64>() {
            if let Ok(threshold_val) = threshold.extract::<f64>() {
                Ok(self.softplus(beta_val, threshold_val))
            } else {
                Err(PyTypeError::new_err("threshold must be numeric"))
            }
        } else {
            Err(PyTypeError::new_err("beta must be numeric"))
        }
    }

    #[pyo3(name = "isclose")]
    pub fn isclose_py(&self, other: &Bound<'_, PyAny>, rtol: f64, atol: f64) -> Result<Tensor, ZyxError> {
        if let Ok(other) = other.extract::<Tensor>() {
            self.isclose(other, rtol, atol)
        } else {
            Err(ZyxError::DTypeError("other must be a Tensor".into()))
        }
    }

    // binary / comparisons

    #[pyo3(name = "cmplt")]
    pub fn cmplt_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.cmplt(rhs)
    }

    #[pyo3(name = "cmpgt")]
    pub fn cmpgt_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.cmpgt(rhs)
    }

    #[pyo3(name = "cmpge")]
    pub fn cmpge_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        // cmpge is not directly exposed as method but via BOp; use cmpgt or equal combo
        // fallback to cmpgt + equal
        let gt = self.cmpgt(rhs.clone())?;
        let eq = self.equal(rhs)?;
        gt.logical_or(eq)
    }

    #[pyo3(name = "equal")]
    pub fn equal_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.equal(rhs)
    }

    #[pyo3(name = "ne")]
    pub fn ne_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.ne(rhs)
    }

    #[pyo3(name = "maximum")]
    pub fn maximum_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.maximum(rhs)
    }

    #[pyo3(name = "minimum")]
    pub fn minimum_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.minimum(rhs)
    }

    #[pyo3(name = "clamp")]
    pub fn clamp_py(&self, min: &Bound<'_, PyAny>, max: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let min_t = extract_tensor_or_scalar(min).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        let max_t = extract_tensor_or_scalar(max).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.clamp(min_t, max_t)
    }

    #[pyo3(name = "pow")]
    pub fn pow_py(&self, exponent: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let e = extract_tensor_or_scalar(exponent).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.pow(e)
    }

    #[pyo3(name = "logical_and")]
    pub fn logical_and_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.logical_and(rhs)
    }

    #[pyo3(name = "logical_or")]
    pub fn logical_or_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.logical_or(rhs)
    }

    #[must_use]
    #[pyo3(name = "nonzero")]
    pub fn nonzero_py(&self) -> Tensor {
        self.nonzero()
    }

    #[pyo3(name = "where_")]
    pub fn where_py(&self, if_true: &Bound<'_, PyAny>, if_false: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let t = extract_tensor_or_scalar(if_true).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        let f = extract_tensor_or_scalar(if_false).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.where_(t, f)
    }

    #[pyo3(name = "l1_loss")]
    pub fn l1_loss_py(&self, target: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        Ok(self.l1_loss(target))
    }

    #[pyo3(name = "mse_loss")]
    pub fn mse_loss_py(&self, target: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.mse_loss(target)
    }

    #[pyo3(name = "bce_loss")]
    pub fn bce_loss_py(&self, target: &Bound<'_, PyAny>, eps: f32) -> Result<Tensor, ZyxError> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.bce_loss(target, eps)
    }

    #[pyo3(name = "cosine_similarity")]
    pub fn cosine_similarity_py(&self, rhs: &Bound<'_, PyAny>, eps: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        let eps = extract_tensor_or_scalar(eps).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.cosine_similarity(rhs, eps)
    }

    #[must_use]
    #[pyo3(name = "diagonal")]
    pub fn diagonal_py(&self) -> Tensor {
        self.diagonal()
    }

    #[pyo3(name = "pad_zeros")]
    pub fn pad_zeros_py(&self, padding: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let items: Vec<i64> = padding.into_iter().map(|d| d.extract().expect("padding must be integers")).collect();
        let pairs: Vec<(i64, i64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        self.pad_zeros(pairs)
    }

    #[pyo3(name = "rpad_zeros")]
    pub fn rpad_zeros_py(&self, padding: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let items: Vec<i64> = padding.into_iter().map(|d| d.extract().expect("padding must be integers")).collect();
        let pairs: Vec<(i64, i64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        self.rpad_zeros(pairs)
    }

    #[pyo3(name = "pad")]
    pub fn pad_py(&self, padding: &Bound<'_, PyList>, value: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let items: Vec<i64> = padding.into_iter().map(|d| d.extract().expect("padding must be integers")).collect();
        let pairs: Vec<(i64, i64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        let value = extract_tensor_or_scalar(value).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.pad(pairs, value)
    }

    #[pyo3(name = "narrow")]
    pub fn narrow_py(&self, axis: Axis, start: &Bound<'_, PyAny>, length: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let start = to_tensor(start).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        let length = to_tensor(length).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.narrow(axis, start, length)
    }

    #[pyo3(name = "split")]
    pub fn split_py(&self, sizes: &Bound<'_, PyTuple>, axis: isize) -> Result<Vec<Tensor>, ZyxError> {
        // sizes can be list of int/Tensor
        let mut vec: Vec<Tensor> = Vec::new();
        for item in sizes.iter() {
            vec.push(to_tensor(&item).map_err(|_| ZyxError::ParseError("split sizes must be int/Tensor".into()))?);
        }
        // convert to Dim via Tensor shape? Actually split expects Vec<Dim> but now symbolic? check impl
        // For now pass as dims resolved? Use old Dim API via resolve
        let dims: Vec<Dim> = vec
            .iter()
            .map(|t| {
                // try to resolve const
                t.clone().item::<i64>()
            })
            .collect();
        self.split(dims, axis)
    }

    #[must_use]
    #[pyo3(name = "one_hot")]
    pub fn one_hot_py(&self, num_classes: Dim) -> Tensor {
        self.one_hot(num_classes)
    }

    #[pyo3(name = "masked_fill")]
    pub fn masked_fill_py(&self, mask: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let mask = extract_tensor_or_scalar(mask).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        let value = extract_tensor_or_scalar(value).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.masked_fill(mask, value)
    }

    #[pyo3(name = "repeat", signature = (*repeats))]
    pub fn repeat_py(&self, repeats: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        let vec: Vec<Tensor> = repeats.iter().map(|x| to_tensor(&x).unwrap().cast(crate::kernel::IDX_T)).collect();
        // repeat expects Vec<Dim> but now symbolic may be Tensor; try to use new API if available
        // fallback: convert via item
        let dims: Vec<Dim> = vec.iter().map(|t| t.clone().item::<i64>()).collect();
        self.repeat(dims)
    }

    #[pyo3(name = "reshape", signature = (*shape))]
    pub fn reshape_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        let shape_vec = parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.reshape(shape_vec)
    }

    #[pyo3(name = "transpose")]
    pub fn transpose_py(&self, dim0: Axis, dim1: Axis) -> Result<Tensor, ZyxError> {
        self.transpose(dim0, dim1)
    }

    #[must_use]
    #[pyo3(name = "t")]
    pub fn t_py(&self) -> Tensor {
        self.t()
    }

    #[pyo3(name = "permute", signature = (*axes))]
    pub fn permute_py(&self, axes: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        self.permute(to_ax(axes))
    }

    #[must_use]
    #[pyo3(name = "squeeze", signature = (axes=None))]
    pub fn squeeze_py(&self, axes: Option<&Bound<'_, PyAny>>) -> Tensor {
        let axes = axes.map(|a| to_ax(a)).unwrap_or_default();
        self.squeeze(axes)
    }

    #[pyo3(name = "unsqueeze")]
    pub fn unsqueeze_py(&self, dim: Axis) -> Result<Tensor, ZyxError> {
        self.unsqueeze(dim)
    }

    #[pyo3(name = "expand", signature = (*shape))]
    pub fn expand_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        let shape_vec = parse_shape(shape).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        self.expand(shape_vec)
    }

    #[pyo3(name = "expand_axis")]
    pub fn expand_axis_py(&self, axis: Axis, dim: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let dim_t = to_tensor(dim).map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))?;
        // expand_axis expects Dim but we support symbolic via expand
        let d: Dim = dim_t.item::<i64>();
        self.expand_axis(axis, d).map_err(|e| e.into())
    }

    #[pyo3(name = "flip")]
    pub fn flip_py(&self, axes: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        self.flip(to_ax(axes))
    }

    #[pyo3(name = "flatten", signature = (start_dim=0, end_dim=-1))]
    pub fn flatten_py(&self, start_dim: Axis, end_dim: Axis) -> Result<Tensor, ZyxError> {
        self.flatten(start_dim..=end_dim)
    }

    #[staticmethod]
    #[pyo3(name = "cat")]
    pub fn cat_py(tensors: &Bound<'_, PyAny>, axis: Axis) -> PyResult<Tensor> {
        let list = extract_tensor_list(tensors)?;
        let refs: Vec<&Tensor> = list.iter().collect();
        Tensor::cat(refs, axis).map_err(|e| e.into())
    }

    #[staticmethod]
    #[pyo3(name = "stack")]
    pub fn stack_py(tensors: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        let list = extract_tensor_list(tensors)?;
        Tensor::stack(&list).map_err(|e| e.into())
    }

    #[staticmethod]
    #[pyo3(name = "stack_axis")]
    pub fn stack_axis_py(tensors: &Bound<'_, PyAny>, dim: Axis) -> PyResult<Tensor> {
        let list = extract_tensor_list(tensors)?;
        let refs: Vec<&Tensor> = list.iter().collect();
        Tensor::stack_axis(refs, dim).map_err(|e| e.into())
    }

    #[pyo3(name = "shrink")]
    pub fn shrink_py(&self, dims: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        // shrink expects ranges, simplified: accept list of (start,end)
        // For now not fully implemented; use slice
        Err(PyTypeError::new_err("shrink not yet implemented in python"))
    }

    #[pyo3(name = "product", signature = (axes=None))]
    pub fn product_py(&self, axes: Option<&Bound<'_, PyAny>>) -> Result<Tensor, ZyxError> {
        let axes = axes.map(|a| to_ax(a)).unwrap_or_default();
        if axes.is_empty() {
            Ok(self.prod_all())
        } else {
            self.prod(axes)
        }
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        use crate::tensor::DimIndex;

        fn slice_to_dimindex(slice: &Bound<'_, PySlice>) -> PyResult<DimIndex> {
            let indices = slice.indices(isize::MAX)?;
            if indices.step != 1 {
                return Err(PyIndexError::new_err("Slice step != 1 is not supported"));
            }
            Ok(DimIndex::Range { start: indices.start as i64, end: indices.stop as i64 })
        }

        fn index_to_dimindices(idx: &Bound<'_, PyAny>) -> PyResult<Vec<DimIndex>> {
            if let Ok(i) = idx.extract::<i64>() {
                Ok(vec![DimIndex::Index(i)])
            } else if let Ok(slice) = idx.cast::<PySlice>() {
                Ok(vec![slice_to_dimindex(slice)?])
            } else if let Ok(tuple) = idx.cast::<PyTuple>() {
                let mut ranges = Vec::with_capacity(tuple.len());
                for item in tuple.iter() {
                    if let Ok(i) = item.extract::<i64>() {
                        ranges.push(DimIndex::Index(i));
                    } else if let Ok(slice) = item.cast::<PySlice>() {
                        ranges.push(slice_to_dimindex(slice)?);
                    } else {
                        return Err(PyIndexError::new_err("Tuple elements must be int or slice"));
                    }
                }
                Ok(ranges)
            } else if let Ok(list) = idx.cast::<PyList>() {
                let mut ranges = Vec::with_capacity(list.len());
                for item in list.iter() {
                    if let Ok(slice) = item.cast::<PySlice>() {
                        ranges.push(slice_to_dimindex(slice)?);
                    } else {
                        return Err(PyIndexError::new_err("List elements must be slices"));
                    }
                }
                Ok(ranges)
            } else {
                Err(PyIndexError::new_err("Unsupported index type"))
            }
        }

        let ranges = index_to_dimindices(idx)?;

        self.slice(ranges).map_err(|e| PyIndexError::new_err(format!("{e:?}")))
    }

    #[pyo3(name = "dot")]
    fn dot_py(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.dot(rhs)
    }

    #[pyo3(name = "dot_dtype")]
    fn dot_dtype_py(&self, rhs: &Bound<PyAny>, out_dtype: DType) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.dot_dtype(rhs, out_dtype)
    }

    #[pyo3(name = "matmul")]
    fn matmul_py(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.dot(rhs)
    }

    fn __matmul__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.dot(rhs)
    }

    fn __add__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        Ok(self + rhs)
    }

    fn __sub__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        Ok(self - rhs)
    }

    fn __mul__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        Ok(self * rhs)
    }

    fn __div__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        Ok(self / rhs)
    }

    fn __truediv__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        Ok(self / rhs)
    }

    fn __pow__(&self, rhs: &Bound<PyAny>, _modulo: Option<&Bound<PyAny>>) -> Result<Tensor, ZyxError> {
        let rhs = extract_tensor_or_scalar(rhs).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.pow(rhs)
    }

    fn __neg__(&self) -> Tensor {
        // use unary neg via 0 - self
        Tensor::from(0.0) - self.clone()
    }

    #[must_use]
    #[pyo3(name = "argmax")]
    pub fn argmax_py(&self) -> Tensor {
        self.argmax()
    }

    #[pyo3(name = "argmax_axis")]
    pub fn argmax_axis_py(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.argmax_axis(axis)
    }

    #[must_use]
    #[pyo3(name = "item")]
    pub fn item_py(&self) -> f64 {
        self.item::<f64>()
    }

    #[pyo3(name = "to_vec_f32")]
    pub fn to_vec_f32_py(&self) -> PyResult<Vec<f32>> {
        let v: Vec<f32> = self.clone().try_into().map_err(|e: ZyxError| PyOSError::new_err(format!("{e:?}")))?;
        Ok(v)
    }

    #[pyo3(name = "cross_entropy")]
    pub fn cross_entropy_py(&self, target: &Bound<'_, PyAny>, reduction: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let r = if let Ok(s) = reduction.extract::<String>() {
            match s.as_str() {
                "mean" => ReduceOp::Mean,
                "sum" => ReduceOp::Sum,
                _ => return Err(ZyxError::ParseError("invalid reduction".into())),
            }
        } else {
            ReduceOp::Mean
        };
        self.cross_entropy(target, r)
    }

    #[pyo3(name = "nll_loss")]
    pub fn nll_loss_py(
        &self,
        target: &Bound<'_, PyAny>,
        weight: Option<&Bound<'_, PyAny>>,
        ignore_index: Option<i64>,
        reduction: &Bound<'_, PyAny>,
    ) -> Result<Tensor, ZyxError> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let weight = match weight {
            Some(w) => Some(w.extract::<Tensor>().map_err(|_| ZyxError::DTypeError("weight must be Tensor".into()))?),
            None => None,
        };
        let r = if let Ok(s) = reduction.extract::<String>() {
            match s.as_str() {
                "mean" => ReduceOp::Mean,
                "sum" => ReduceOp::Sum,
                "none" => ReduceOp::None,
                _ => return Err(ZyxError::ParseError("invalid reduction".into())),
            }
        } else {
            ReduceOp::Mean
        };
        self.nll_loss(target, weight, ignore_index, r)
    }

    #[pyo3(name = "ctc_loss")]
    pub fn ctc_loss_py(&self, target: &Bound<'_, PyAny>, blank: i64, reduction: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let target = extract_tensor_or_scalar(target).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let r = if let Ok(s) = reduction.extract::<String>() {
            match s.as_str() {
                "mean" => ReduceOp::Mean,
                "sum" => ReduceOp::Sum,
                _ => return Err(ZyxError::ParseError("invalid reduction".into())),
            }
        } else {
            ReduceOp::Mean
        };
        self.ctc_loss(target, blank, r)
    }

    #[pyo3(name = "triplet_margin_loss")]
    pub fn triplet_margin_loss_py(
        &self,
        positive: &Bound<'_, PyAny>,
        negative: &Bound<'_, PyAny>,
        margin: f32,
        p: i32,
        swap: bool,
        reduction: &Bound<'_, PyAny>,
    ) -> Result<Tensor, ZyxError> {
        let positive = extract_tensor_or_scalar(positive).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let negative = extract_tensor_or_scalar(negative).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let r = if let Ok(s) = reduction.extract::<String>() {
            match s.as_str() {
                "mean" => ReduceOp::Mean,
                "sum" => ReduceOp::Sum,
                "none" => ReduceOp::None,
                _ => return Err(ZyxError::ParseError("invalid reduction".into())),
            }
        } else {
            ReduceOp::Mean
        };
        self.triplet_margin_loss(&positive, &negative, margin, p, swap, r)
    }

    #[pyo3(name = "one_hot_along_dim")]
    pub fn one_hot_along_dim_py(&self, num_classes: Dim, dim: Axis) -> Result<Tensor, ZyxError> {
        self.one_hot_along_dim(num_classes, dim)
    }

    #[pyo3(name = "gather")]
    pub fn gather_py(&self, axis: Axis, indices: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let indices = extract_tensor_or_scalar(indices).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.gather(axis, indices)
    }

    #[pyo3(name = "scatter")]
    pub fn scatter_py(&self, axis: Axis, indices: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let indices = extract_tensor_or_scalar(indices).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let src = extract_tensor_or_scalar(src).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.scatter(axis, indices, src)
    }

    #[pyo3(name = "index_select")]
    pub fn index_select_py(&self, dim: Axis, index: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let index = extract_tensor_or_scalar(index).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.index_select(dim, index)
    }

    #[pyo3(name = "conv")]
    pub fn conv_py(
        &self,
        weight: &Bound<'_, PyAny>,
        bias: Option<&Bound<'_, PyAny>>,
        groups: u64,
        stride: &Bound<'_, PyTuple>,
        dilation: &Bound<'_, PyTuple>,
        padding: &Bound<'_, PyTuple>,
    ) -> Result<Tensor, ZyxError> {
        let weight = weight.extract::<Tensor>().map_err(|e| ZyxError::DTypeError(format!("weight: {e}").into()))?;
        let bias = bias.and_then(|b| b.extract::<Tensor>().ok());
        // to_sh for conv expects Vec<Dim> but handle symbolic similarly
        let to_sh_vec = |t: &Bound<'_, PyTuple>| -> Result<Vec<Dim>, ZyxError> {
            t.iter().map(|x| Ok(x.extract::<i64>().map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))? as Dim)).collect()
        };
        self.conv(&weight, bias.as_ref(), groups, to_sh_vec(stride)?, to_sh_vec(dilation)?, to_sh_vec(padding)?)
    }

    #[pyo3(name = "max_pool")]
    pub fn max_pool_py(
        &self,
        kernel_shape: &Bound<'_, PyTuple>,
        stride: Option<&Bound<'_, PyTuple>>,
        _padding: Option<&Bound<'_, PyTuple>>,
    ) -> Result<Tensor, ZyxError> {
        let ks: Vec<Dim> = kernel_shape
            .iter()
            .map(|x| Ok(x.extract::<i64>().map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))? as Dim))
            .collect::<Result<_, ZyxError>>()?;
        let st = stride
            .map(|t| {
                t.iter()
                    .map(|x| Ok(x.extract::<i64>().map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))? as Dim))
                    .collect::<Result<Vec<_>, ZyxError>>()
            })
            .transpose()?
            .unwrap_or_else(|| ks.clone());
        self.max_pool(ks.clone(), st.clone(), vec![1; ks.len()], vec![(0, 0); ks.len()], false, false)
    }

    #[pyo3(name = "pool")]
    pub fn pool_py(
        &self,
        kernel_shape: &Bound<'_, PyTuple>,
        stride: Option<&Bound<'_, PyTuple>>,
        _padding: Option<&Bound<'_, PyTuple>>,
    ) -> Result<Tensor, ZyxError> {
        let ks: Vec<Dim> = kernel_shape
            .iter()
            .map(|x| Ok(x.extract::<i64>().map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))? as Dim))
            .collect::<Result<_, ZyxError>>()?;
        let st = stride
            .map(|t| {
                t.iter()
                    .map(|x| Ok(x.extract::<i64>().map_err(|e| ZyxError::ParseError(format!("{e:?}").into()))? as Dim))
                    .collect::<Result<Vec<_>, ZyxError>>()
            })
            .transpose()?
            .unwrap_or_else(|| ks.clone());
        self.pool(ks.clone(), st.clone(), vec![1; ks.len()])
    }

    #[pyo3(name = "tri")]
    #[staticmethod]
    pub fn tri_py(r: Dim, c: Dim, diagonal: i64, dtype: DType) -> Tensor {
        Tensor::tri(r, c, diagonal, dtype)
    }

    #[pyo3(name = "triu")]
    pub fn triu_py(&self, diagonal: i64) -> Result<Tensor, ZyxError> {
        self.triu(diagonal)
    }

    #[pyo3(name = "tril")]
    pub fn tril_py(&self, diagonal: i64) -> Result<Tensor, ZyxError> {
        self.tril(diagonal)
    }

    #[pyo3(name = "rope")]
    pub fn rope_py(&self, sine: &Bound<'_, PyAny>, cosine: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let s = extract_tensor_or_scalar(sine).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        let c = extract_tensor_or_scalar(cosine).map_err(|e| ZyxError::DTypeError(format!("{e:?}").into()))?;
        self.rope(s, c)
    }

    #[pyo3(name = "to")]
    pub fn to_py(&self, device: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(id) = device.extract::<usize>() {
            self.to(Dev::Cuda(id as u16))
        } else if let Ok(s) = device.extract::<String>() {
            // string like "cpu", "cuda:0" - fallback to AUTO
            let _ = s;
            self.to(Dev::Auto)
        } else {
            Err(ZyxError::ParseError("invalid device".into()))
        }
    }

    #[pyo3(name = "contiguous")]
    pub fn contiguous_py(&self) -> Result<Tensor, ZyxError> {
        self.contiguous()
    }

    #[pyo3(name = "to_le_bytes")]
    pub fn to_le_bytes_py(&self) -> Result<Vec<u8>, ZyxError> {
        self.to_le_bytes()
    }
}

// kernel bindings
#[pyo3::pyclass]
pub struct PyKernel {
    inner: Option<Kernel>,
}

#[pymethods]
impl PyKernel {
    #[new]
    #[pyo3(signature = (device=None))]
    fn new(device: Option<u32>) -> Self {
        let dev = device.map(DeviceId).unwrap_or(DeviceId::AUTO);
        Self { inner: Some(Kernel::new(dev)) }
    }

    #[pyo3(name = "compile")]
    fn compile_py(&mut self) -> PyResult<PyCompiledKernel> {
        let k = self.inner.take().ok_or_else(|| PyOSError::new_err("kernel already compiled"))?;
        let compiled = k.compile().map_err(|e| PyOSError::new_err(format!("{e:?}")))?;
        Ok(PyCompiledKernel { inner: compiled })
    }

    #[pyo3(name = "param")]
    fn param_py(&mut self, dtype: DType, kind: u8, shape: u32) -> u32 {
        let k = match kind {
            0 => ParamKind::Global,
            1 => ParamKind::GlobalMut,
            2 => ParamKind::Variable,
            _ => ParamKind::Global,
        };
        self.inner.as_mut().unwrap().param(dtype, k, OpId(shape)).0
    }

    #[pyo3(name = "add_shape")]
    fn add_shape_py(&mut self, shape: Vec<i64>) -> u32 {
        let dims: Vec<Dim> = shape.into_iter().map(|d| d as Dim).collect();
        self.inner.as_mut().unwrap().add_shape(&dims).0
    }

    #[pyo3(name = "storage")]
    fn storage_py(&mut self, dtype: DType, scope: u8, len: i64) -> u32 {
        let scope = match scope {
            0 => MemScope::Global,
            1 => MemScope::Local,
            2 => MemScope::Register,
            _ => MemScope::Global,
        };
        self.inner.as_mut().unwrap().storage(dtype, scope, len as Dim).0
    }

    #[pyo3(name = "const_val")]
    fn const_val_py(&mut self, val: f64) -> u32 {
        self.inner.as_mut().unwrap().const_val(val as f32).0
    }

    #[pyo3(name = "const_idx")]
    fn const_idx_py(&mut self, val: i64) -> u32 {
        self.inner.as_mut().unwrap().const_idx(val).0
    }

    #[pyo3(name = "group_index")]
    fn group_index_py(&mut self, axis: u32, len: u32) -> u32 {
        self.inner.as_mut().unwrap().group_range(axis, OpId(len)).0
    }

    #[pyo3(name = "local_index")]
    fn local_index_py(&mut self, axis: u32, len: u32) -> u32 {
        self.inner.as_mut().unwrap().local_range(axis, len).0
    }

    #[pyo3(name = "load")]
    fn load_py(&mut self, src: u32, index: u32, layout: u8) -> u32 {
        let layout = if layout == 0 {
            MemLayout::Scalar
        } else {
            MemLayout::Vector(layout as u16)
        };
        self.inner.as_mut().unwrap().load(OpId(src), OpId(index), layout).0
    }

    #[pyo3(name = "store")]
    fn store_py(&mut self, dst: u32, src: u32, index: u32, layout: u8) {
        let layout = if layout == 0 {
            MemLayout::Scalar
        } else {
            MemLayout::Vector(layout as u16)
        };
        self.inner.as_mut().unwrap().store(OpId(dst), OpId(src), OpId(index), layout)
    }

    #[pyo3(name = "loop_")]
    fn loop_py(&mut self, len: u32) -> u32 {
        self.inner.as_mut().unwrap().loop_(OpId(len)).0
    }

    #[pyo3(name = "end_loop")]
    fn end_loop_py(&mut self) {
        self.inner.as_mut().unwrap().end_loop()
    }

    // unary
    #[pyo3(name = "neg")]
    fn neg_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().neg(OpId(x)).0
    }
    #[pyo3(name = "exp")]
    fn exp_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().exp(OpId(x)).0
    }
    #[pyo3(name = "ln")]
    fn ln_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().ln(OpId(x)).0
    }
    #[pyo3(name = "sin")]
    fn sin_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().sin(OpId(x)).0
    }
    #[pyo3(name = "cos")]
    fn cos_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().cos(OpId(x)).0
    }
    #[pyo3(name = "sqrt")]
    fn sqrt_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().sqrt(OpId(x)).0
    }
    #[pyo3(name = "abs")]
    fn abs_py(&mut self, x: u32) -> u32 {
        self.inner.as_mut().unwrap().abs(OpId(x)).0
    }

    // binary
    #[pyo3(name = "add")]
    fn add_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().add(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "sub")]
    fn sub_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().sub(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "mul")]
    fn mul_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().mul(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "div")]
    fn div_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().div(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "max")]
    fn max_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().max(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "cmplt")]
    fn cmplt_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().cmplt(OpId(x), OpId(y)).0
    }
    #[pyo3(name = "cmpgt")]
    fn cmpgt_py(&mut self, x: u32, y: u32) -> u32 {
        self.inner.as_mut().unwrap().cmpgt(OpId(x), OpId(y)).0
    }

    #[pyo3(name = "mad")]
    fn mad_py(&mut self, x: u32, y: u32, z: u32) -> u32 {
        self.inner.as_mut().unwrap().mad(OpId(x), OpId(y), OpId(z)).0
    }
    #[pyo3(name = "cast")]
    fn cast_py(&mut self, x: u32, dtype: DType) -> u32 {
        self.inner.as_mut().unwrap().cast(OpId(x), dtype).0
    }
    #[pyo3(name = "stack")]
    fn stack_py(&mut self, ops: Vec<u32>) -> u32 {
        self.inner.as_mut().unwrap().stack(&ops.iter().map(|&o| OpId(o)).collect::<Vec<_>>()).0
    }
}

#[pyo3::pyclass]
pub struct PyCompiledKernel {
    inner: CompiledKernel,
}

#[pymethods]
impl PyCompiledKernel {
    #[pyo3(name = "forward")]
    fn forward_py(&self, inputs: &Bound<'_, PyAny>, shapes: Vec<Vec<i64>>) -> PyResult<Vec<Tensor>> {
        // inputs: list of Tensors
        let tensors = extract_tensor_list(inputs)?;
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let shapes_vec: Vec<Vec<Dim>> = shapes.into_iter().map(|v| v.into_iter().map(|d| d as Dim).collect()).collect();
        self.inner.forward(&refs, shapes_vec).map_err(|e| PyOSError::new_err(format!("{e:?}")))
    }
}

/// Re-export helper for zyx-py to register Tensor class.
pub fn register_tensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<Tape>()?;
    m.add_class::<FrozenTape>()?;
    m.add_class::<PyKernel>()?;
    m.add_class::<PyCompiledKernel>()?;
    Ok(())
}

/// Re-export helper for zyx-py to register DType class.
pub fn register_dtype(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DType>()?;
    m.add_class::<DebugMask>()?;
    Ok(())
}

/// Re-export helper for zyx-py to register Tape class.
pub fn register_tape(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // already registered via register_tensor
    Ok(())
}

fn from_numpy<T: crate::Scalar + pyo3::buffer::Element>(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let buffer = PyBuffer::<T>::get(obj)?;

    let shape: Vec<Dim> = buffer.shape().to_vec().into_iter().map(|s| Dim::try_from(s).unwrap()).collect();
    let strides: Vec<Dim> = buffer.strides().to_vec().into_iter().map(|s| Dim::try_from(s as usize).unwrap()).collect();
    let data = buffer.as_slice(obj.py()).unwrap();

    let ndim = shape.len();
    assert_eq!(strides.len(), ndim);
    assert_eq!(shape.len(), ndim);

    let total_len: Dim = shape.iter().product();
    let mut result = Vec::with_capacity(total_len as usize);

    let mut indices = vec![0usize; ndim];

    for _ in 0..total_len as usize {
        let mut offset_bytes: i64 = 0;
        for i in 0..ndim {
            let idx = indices[i];
            let s = strides[i];
            offset_bytes += (idx as i64) * (s as i64);
        }
        let element_size = std::mem::size_of::<T>() as i64;
        let index = (offset_bytes / element_size) as usize;
        result.push(data[index].get());
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] as usize {
                break;
            }
            indices[d] = 0;
        }
    }
    Ok(Tensor::from(result).reshape(shape.iter().map(|&d| Tensor::from(d)).collect::<Vec<_>>()).unwrap())
}
