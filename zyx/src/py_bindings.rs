// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx

#![allow(missing_docs)]

use crate::DebugMask;
use crate::shape::Dim;
use crate::tensor::{Axis, DebugGuard, ReduceOp};
use crate::{DType, GradientTape, Tensor, ZyxError};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::{
    Bound, PyAny, PyErr, PyResult,
    exceptions::{PyOSError, PyTypeError},
    pymethods, pymodule,
    types::{PyAnyMethods, PyIterator, PyList, PyModule, PyModuleMethods, PyTuple},
};

impl From<ZyxError> for PyErr {
    fn from(err: ZyxError) -> Self {
        PyOSError::new_err(format!("{err:?}"))
    }
}

#[pymethods]
impl GradientTape {
    #[must_use]
    #[pyo3(name = "backward")]
    pub fn gradient_py(&self, x: &Tensor, sources: &Bound<'_, PyList>) -> Vec<Option<Tensor>> {
        let sources: Vec<Tensor> = sources
            .into_iter()
            .map(|d| d.extract::<Tensor>().expect("sources must be List(Tensor)"))
            .collect();
        // In python, we cannot drop the tape ...
        self.gradient_persistent(x, &sources)
    }
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(py_obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(tensor) = from_numpy::<f32>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<f64>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<i8>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<i16>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<i32>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<i64>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<u8>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<u16>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<u32>(py_obj) {
            return Ok(tensor);
        }
        if let Ok(tensor) = from_numpy::<u64>(py_obj) {
            return Ok(tensor);
        }

        if let Ok(val) = py_obj.extract::<i64>() {
            return Ok(Tensor::from(val));
        }
        if let Ok(val) = py_obj.extract::<f64>() {
            return Ok(Tensor::from(val));
        }

        if let Ok(vec) = py_obj.extract::<Vec<i64>>() {
            return Ok(Tensor::from(vec));
        }
        if let Ok(vec) = py_obj.extract::<Vec<f64>>() {
            return Ok(Tensor::from(vec));
        }

        if let Ok(mat) = py_obj.extract::<Vec<Vec<i64>>>() {
            return Ok(Tensor::from(mat));
        }
        if let Ok(mat) = py_obj.extract::<Vec<Vec<f64>>>() {
            return Ok(Tensor::from(mat));
        }

        Err(PyTypeError::new_err("Unsupported input type for Tensor"))
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.shape();
        let np = py.import("numpy")?;
        Ok(match self.dtype() {
            DType::BF16 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => {
                let data: Vec<f32> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "float32"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::F64 => {
                let data: Vec<f64> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "float64"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U8 => {
                let data: Vec<u8> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "uint8"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U16 => {
                let data: Vec<u16> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "uint16"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U32 => {
                let data: Vec<u32> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "uint32"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::U64 => {
                let data: Vec<u64> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "uint64"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I8 => {
                let data: Vec<i8> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "int8"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I16 => {
                let data: Vec<i16> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "int16"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I32 => {
                let data: Vec<i32> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "int32"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::I64 => {
                let data: Vec<i64> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "int64"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
            DType::Bool => {
                let data: Vec<bool> = self.clone().try_into()?;
                np.getattr("array")?
                    .call1((data, "bool"))?
                    .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            }
        })
    }

    #[staticmethod]
    #[pyo3(name = "plot_dot_graph")]
    pub fn plot_dot_graph_py(tensors: &Bound<'_, PyList>, name: &str) -> Result<(), std::io::Error> {
        let tensors: Vec<Tensor> = tensors
            .into_iter()
            .map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)"))
            .collect();
        Tensor::plot_graph(&tensors, name)
    }

    #[staticmethod]
    #[pyo3(name = "manual_seed")]
    pub fn manual_seed_py(seed: u64) {
        Tensor::manual_seed(seed)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "training")]
    pub fn training_py() -> bool {
        return Tensor::training();
    }

    #[staticmethod]
    #[pyo3(name = "set_training")]
    pub fn set_training_py(training: bool) {
        Tensor::set_training(training);
    }

    #[staticmethod]
    #[pyo3(name = "realize")]
    pub fn realize_py(tensors: &Bound<'_, PyList>) -> Result<(), ZyxError> {
        let tensors: Vec<Tensor> = tensors
            .into_iter()
            .map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)"))
            .collect();
        Tensor::realize(&tensors)
    }

    #[must_use]
    #[pyo3(name = "shape")]
    pub fn shape_py(&self) -> Vec<Dim> {
        self.shape()
    }

    #[must_use]
    #[pyo3(name = "numel")]
    pub fn numel_py(&self) -> Dim {
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

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "implicit_casts")]
    pub fn implicit_casts_py() -> bool {
        Tensor::implicit_casts()
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "set_implicit_casts")]
    pub fn set_implicit_casts_py(implicit_casts: bool) {
        Tensor::set_implicit_casts(implicit_casts);
    }

    #[must_use]
    #[pyo3(name = "detach")]
    pub fn detach_py(&self) -> Result<Tensor, ZyxError> {
        self.clone().detach()
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "with_debug")]
    pub fn with_debug_py(debug: DebugMask) -> DebugGuard {
        Tensor::with_debug(debug)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "randn", signature = (*shape, dtype=DType::F32))]
    pub fn randn_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::randn(to_sh(shape)?, dtype)
    }

    #[must_use]
    #[pyo3(name = "multinomial")]
    pub fn multinomial_py(&self, num_samples: Dim, replacement: bool) -> Result<Tensor, ZyxError> {
        self.multinomial(num_samples, replacement)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "rand", signature = (*shape, dtype=DType::F32))]
    pub fn rand_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::rand(to_sh(shape)?, dtype)
    }

    /*#[staticmethod]
    #[must_use]
    #[pyo3(name = "uniform_", signature = (*shape, dtype=DType::F32))]
    pub fn uniform_py(shape: &Bound<'_, PyTuple>, from, to) -> Result<Tensor, ZyxError> {
        Tensor::uniform(to_sh(shape)?, from..to)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "kaiming_uniform", signature = (*shape, dtype=DType::F32))]
    pub fn kaiming_uniform_py(shape: &Bound<'_, PyTuple>, a) -> Result<Tensor, ZyxError> {
        Tensor::kaiming_uniform(to_sh(shape)?, a)
    }*/

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "zeros", signature = (*shape, dtype=DType::F32))]
    pub fn zeros_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Tensor {
        Tensor::zeros(to_sh(shape).unwrap(), dtype)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "ones", signature = (*shape, dtype=DType::F32))]
    pub fn ones_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Tensor {
        return Tensor::ones(to_sh(shape).unwrap(), dtype);
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "full", signature = (*shape, a))]
    pub fn full_py(shape: &Bound<'_, PyTuple>, a: f64) -> Tensor {
        Tensor::full(to_sh(shape).unwrap(), a)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "zeros_like")]
    pub fn zeros_like_py(input: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(tensor) = input.extract::<Tensor>() {
            Ok(Tensor::zeros_like(tensor))
        } else {
            Err(ZyxError::DTypeError("input must be a Tensor".into()))
        }
    }

    #[staticmethod]
    #[must_use]
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
        return Tensor::eye(n, dtype);
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "arange", signature = (start=0, stop=1, step=1))]
    pub fn arange_py(start: i64, stop: i64, step: i64) -> Result<Tensor, ZyxError> {
        Tensor::arange(start, stop, step)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "from_vec")]
    pub fn from_vec_py(data: &Bound<'_, PyList>, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        let shape_vec = to_sh(shape)?;
        if let Ok(data_vec) = data.extract::<Vec<f64>>() {
            Ok(Tensor::from_vec(data_vec, shape_vec)?)
        } else if let Ok(data_vec) = data.extract::<Vec<i64>>() {
            Ok(Tensor::from_vec(data_vec, shape_vec)?)
        } else {
            Err(ZyxError::DTypeError("data must be Vec<f64> or Vec<i64>".into()))
        }
    }

    #[must_use]
    #[pyo3(name = "abs")]
    pub fn abs_py(&self) -> Tensor {
        return self.abs();
    }

    #[must_use]
    #[pyo3(name = "cast")]
    pub fn cast_py(&self, dtype: DType) -> Tensor {
        return self.cast(dtype);
    }

    #[must_use]
    #[pyo3(name = "cos")]
    pub fn cos_py(&self) -> Tensor {
        return self.cos();
    }

    #[must_use]
    #[pyo3(name = "cosh")]
    pub fn cosh_py(&self) -> Tensor {
        return self.cosh();
    }

    #[must_use]
    #[pyo3(name = "exp")]
    pub fn exp_py(&self) -> Tensor {
        return self.exp();
    }

    #[must_use]
    #[pyo3(name = "floor")]
    pub fn floor_py(&self) -> Tensor {
        return self.floor();
    }

    #[must_use]
    #[pyo3(name = "log")]
    pub fn log_py(&self, base: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(base_tensor) = base.extract::<Tensor>() {
            Ok(self.log(base_tensor))
        } else if let Ok(base_val) = base.extract::<f64>() {
            Ok(self.log(Tensor::from(base_val)))
        } else {
            Err(PyTypeError::new_err("base must be a Tensor or numeric value"))
        }
    }

    #[must_use]
    #[pyo3(name = "log2")]
    pub fn log2_py(&self) -> Tensor {
        return self.log2();
    }

    #[must_use]
    #[pyo3(name = "reciprocal")]
    pub fn reciprocal_py(&self) -> Tensor {
        return self.reciprocal();
    }

    #[must_use]
    #[pyo3(name = "relu")]
    pub fn relu_py(&self) -> Tensor {
        return self.relu();
    }

    #[must_use]
    #[pyo3(name = "rsqrt")]
    pub fn rsqrt_py(&self) -> Tensor {
        return self.rsqrt();
    }

    #[must_use]
    #[pyo3(name = "sigmoid")]
    pub fn sigmoid_py(&self) -> Tensor {
        return self.sigmoid();
    }

    #[must_use]
    #[pyo3(name = "sin")]
    pub fn sin_py(&self) -> Tensor {
        return self.sin();
    }

    #[must_use]
    #[pyo3(name = "sinh")]
    pub fn sinh_py(&self) -> Tensor {
        return self.sinh();
    }

    #[must_use]
    #[pyo3(name = "sqrt")]
    pub fn sqrt_py(&self) -> Tensor {
        return self.sqrt();
    }

    #[must_use]
    #[pyo3(name = "tan")]
    pub fn tan_py(&self) -> Tensor {
        return self.tan();
    }

    #[must_use]
    #[pyo3(name = "tanh")]
    pub fn tanh_py(&self) -> Tensor {
        return self.tanh();
    }

    #[must_use]
    #[pyo3(name = "gelu")]
    pub fn gelu_py(&self) -> Tensor {
        return self.gelu();
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

    #[must_use]
    #[pyo3(name = "ln")]
    pub fn ln_py(&self) -> Tensor {
        return self.ln();
    }

    #[must_use]
    #[pyo3(name = "celu")]
    pub fn celu_py(&self, alpha: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(alpha_val) = alpha.extract::<f64>() {
            Ok(self.celu(alpha_val))
        } else if let Ok(alpha_val) = alpha.extract::<i64>() {
            Ok(self.celu(alpha_val))
        } else {
            Err(PyTypeError::new_err("alpha must be numeric"))
        }
    }

    #[must_use]
    #[pyo3(name = "elu")]
    pub fn elu_py(&self, alpha: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(alpha_val) = alpha.extract::<f64>() {
            Ok(self.elu(alpha_val))
        } else if let Ok(alpha_val) = alpha.extract::<i64>() {
            Ok(self.elu(alpha_val))
        } else {
            Err(PyTypeError::new_err("alpha must be numeric"))
        }
    }

    #[must_use]
    #[pyo3(name = "softmax")]
    pub fn softmax_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = axes
            .into_iter()
            .map(|d| d.extract::<Axis>().expect("axes must be integers"))
            .collect();
        self.softmax(axes)
    }

    #[must_use]
    #[pyo3(name = "log_softmax")]
    pub fn log_softmax_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = axes
            .into_iter()
            .map(|d| d.extract::<Axis>().expect("axes must be integers"))
            .collect();
        self.ln_softmax(axes)
    }

    #[must_use]
    #[pyo3(name = "sum", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn sum_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Sum, axes, dtype, 1)
        } else {
            self.reduce_impl::<false>(ReduceOp::Sum, axes, dtype, 1)
        }
    }

    #[must_use]
    #[pyo3(name = "mean", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn mean_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Mean, axes, dtype, 1)
        } else {
            self.reduce_impl::<false>(ReduceOp::Mean, axes, dtype, 1)
        }
    }

    #[must_use]
    #[pyo3(name = "var", signature = (dim=None, keepdim=false, unbiased=true, dtype=None))]
    pub fn var_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool, unbiased: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        let correction = if unbiased { 1 } else { 0 };
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Var, axes, dtype, correction)
        } else {
            self.reduce_impl::<false>(ReduceOp::Var, axes, dtype, correction)
        }
    }

    #[must_use]
    #[pyo3(name = "std", signature = (dim=None, keepdim=false, unbiased=true, dtype=None))]
    pub fn std_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool, unbiased: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        let correction = if unbiased { 1 } else { 0 };
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Std, axes, dtype, correction)
        } else {
            self.reduce_impl::<false>(ReduceOp::Std, axes, dtype, correction)
        }
    }

    #[must_use]
    #[pyo3(name = "min", signature = (dim=None, keepdim=false))]
    pub fn min_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Min, axes, None, 1)
        } else {
            self.reduce_impl::<false>(ReduceOp::Min, axes, None, 1)
        }
    }

    #[must_use]
    #[pyo3(name = "max", signature = (dim=None, keepdim=false))]
    pub fn max_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Max, axes, None, 1)
        } else {
            self.reduce_impl::<false>(ReduceOp::Max, axes, None, 1)
        }
    }

    #[must_use]
    #[pyo3(name = "prod", signature = (dim=None, keepdim=false, dtype=None))]
    pub fn prod_py(&self, dim: Option<&Bound<'_, PyList>>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = dim.map(|d| d.into_iter().map(|x| x.extract::<Axis>().expect("dim must be integers")).collect()).unwrap_or_default();
        if keepdim {
            self.reduce_impl::<true>(ReduceOp::Prod, axes, dtype, 1)
        } else {
            self.reduce_impl::<false>(ReduceOp::Prod, axes, dtype, 1)
        }
    }

    #[must_use]
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

    #[must_use]
    #[pyo3(name = "bitnot")]
    pub fn bitnot_py(&self) -> Tensor {
        return self.bitnot();
    }

    #[must_use]
    #[pyo3(name = "ceil")]
    pub fn ceil_py(&self) -> Tensor {
        return self.ceil();
    }

    #[must_use]
    #[pyo3(name = "erf")]
    pub fn erf_py(&self) -> Tensor {
        return self.erf();
    }

    #[must_use]
    #[pyo3(name = "frac")]
    pub fn frac_py(&self) -> Tensor {
        return self.frac();
    }

    #[must_use]
    #[pyo3(name = "isnan")]
    pub fn isnan_py(&self) -> Tensor {
        return self.isnan();
    }

    #[must_use]
    #[pyo3(name = "isinf")]
    pub fn isinf_py(&self) -> Tensor {
        return self.isinf();
    }

    #[must_use]
    #[pyo3(name = "log10")]
    pub fn log10_py(&self) -> Tensor {
        return self.log10();
    }

    #[must_use]
    #[pyo3(name = "rad2deg")]
    pub fn rad2deg_py(&self) -> Tensor {
        return self.rad2deg();
    }

    #[must_use]
    #[pyo3(name = "deg2rad")]
    pub fn deg2rad_py(&self) -> Tensor {
        return self.deg2rad();
    }

    #[must_use]
    #[pyo3(name = "round")]
    pub fn round_py(&self) -> Tensor {
        return self.round();
    }

    #[must_use]
    #[pyo3(name = "sign")]
    pub fn sign_py(&self) -> Tensor {
        return self.sign();
    }

    #[must_use]
    #[pyo3(name = "square")]
    pub fn square_py(&self) -> Tensor {
        return self.square();
    }

    #[must_use]
    #[pyo3(name = "trunc")]
    pub fn trunc_py(&self) -> Tensor {
        return self.trunc();
    }

    #[must_use]
    #[pyo3(name = "isclose")]
    pub fn isclose_py(&self, other: &Bound<'_, PyAny>, rtol: f64, atol: f64) -> Result<Tensor, ZyxError> {
        if let Ok(other) = other.extract::<Tensor>() {
            self.isclose(other, rtol, atol)
        } else {
            Err(ZyxError::DTypeError("other must be a Tensor".into()))
        }
    }

    // Missing unary operations
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

    // Missing comparison operations
    #[must_use]
    #[pyo3(name = "cmplt")]
    pub fn cmplt_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.cmplt(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.cmplt(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for cmplt".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "cmpgt")]
    pub fn cmpgt_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.cmpgt(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.cmpgt(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for cmpgt".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "maximum")]
    pub fn maximum_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.maximum(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.maximum(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for maximum".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "minimum")]
    pub fn minimum_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.minimum(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.minimum(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for minimum".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "equal")]
    pub fn equal_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.equal(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.equal(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for equal".into()));
        }
    }

    // Missing utility operations
    #[must_use]
    #[pyo3(name = "clamp")]
    pub fn clamp_py(&self, min: &Bound<'_, PyAny>, max: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(min_tensor) = min.extract::<Self>() {
            if let Ok(max_tensor) = max.extract::<Self>() {
                self.clamp(min_tensor, max_tensor)
            } else if let Ok(max_val) = max.extract::<f64>() {
                self.clamp(min_tensor, Tensor::from(max_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported max for clamp".into()));
            }
        } else if let Ok(min_val) = min.extract::<f64>() {
            if let Ok(max_tensor) = max.extract::<Self>() {
                self.clamp(Tensor::from(min_val), max_tensor)
            } else if let Ok(max_val) = max.extract::<f64>() {
                self.clamp(Tensor::from(min_val), Tensor::from(max_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported max for clamp".into()));
            }
        } else {
            return Err(ZyxError::DTypeError("unsupported min for clamp".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "pow")]
    pub fn pow_py(&self, exponent: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(exponent_tensor) = exponent.extract::<Self>() {
            self.pow(exponent_tensor)
        } else if let Ok(exp_val) = exponent.extract::<f64>() {
            self.pow(Tensor::from(exp_val))
        } else {
            return Err(ZyxError::DTypeError("unsupported exponent for pow".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "logical_and")]
    pub fn logical_and_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.logical_and(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.logical_and(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for logical_and".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "logical_or")]
    pub fn logical_or_py(&self, rhs: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.logical_or(rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            self.logical_or(Tensor::from(rhs))
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for logical_or".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "nonzero")]
    pub fn nonzero_py(&self) -> Tensor {
        self.nonzero()
    }

    #[must_use]
    #[pyo3(name = "where_")]
    pub fn where_py(&self, if_true: &Bound<'_, PyAny>, if_false: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(true_tensor) = if_true.extract::<Self>() {
            if let Ok(false_tensor) = if_false.extract::<Self>() {
                self.where_(true_tensor, false_tensor)
            } else if let Ok(false_val) = if_false.extract::<f64>() {
                self.where_(true_tensor, Tensor::from(false_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported if_false for where".into()));
            }
        } else if let Ok(true_val) = if_true.extract::<f64>() {
            if let Ok(false_tensor) = if_false.extract::<Self>() {
                self.where_(Tensor::from(true_val), false_tensor)
            } else if let Ok(false_val) = if_false.extract::<f64>() {
                self.where_(Tensor::from(true_val), Tensor::from(false_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported if_false for where".into()));
            }
        } else {
            return Err(ZyxError::DTypeError("unsupported if_true for where".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "l1_loss")]
    pub fn l1_loss_py(&self, target: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(target_tensor) = target.extract::<Self>() {
            Ok(self.l1_loss(target_tensor))
        } else if let Ok(target_val) = target.extract::<f64>() {
            Ok(self.l1_loss(Tensor::from(target_val)))
        } else {
            return Err(ZyxError::DTypeError("unsupported target for l1_loss".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "mse_loss")]
    pub fn mse_loss_py(&self, target: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(target_tensor) = target.extract::<Self>() {
            self.mse_loss(target_tensor)
        } else if let Ok(target_val) = target.extract::<f64>() {
            self.mse_loss(Tensor::from(target_val))
        } else {
            return Err(ZyxError::DTypeError("unsupported target for mse_loss".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "cosine_similarity")]
    pub fn cosine_similarity_py(&self, rhs: &Bound<'_, PyAny>, eps: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs_tensor) = rhs.extract::<Self>() {
            if let Ok(eps_tensor) = eps.extract::<Self>() {
                self.cosine_similarity(rhs_tensor, eps_tensor)
            } else if let Ok(eps_val) = eps.extract::<f64>() {
                self.cosine_similarity(rhs_tensor, Tensor::from(eps_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported eps for cosine_similarity".into()));
            }
        } else if let Ok(rhs_val) = rhs.extract::<f64>() {
            if let Ok(eps_tensor) = eps.extract::<Self>() {
                self.cosine_similarity(Tensor::from(rhs_val), eps_tensor)
            } else if let Ok(eps_val) = eps.extract::<f64>() {
                self.cosine_similarity(Tensor::from(rhs_val), Tensor::from(eps_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported eps for cosine_similarity".into()));
            }
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for cosine_similarity".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "diagonal")]
    pub fn diagonal_py(&self) -> Tensor {
        self.diagonal()
    }

    #[must_use]
    #[pyo3(name = "pad_zeros")]
    pub fn pad_zeros_py(&self, padding: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let items: Vec<i64> = padding
            .into_iter()
            .map(|d| d.extract().expect("padding must be integers"))
            .collect();
        let pairs: Vec<(i64, i64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        self.pad_zeros(pairs)
    }

    #[must_use]
    #[pyo3(name = "pad")]
    pub fn pad_py(&self, padding: &Bound<'_, PyList>, value: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        let items: Vec<i64> = padding
            .into_iter()
            .map(|d| d.extract().expect("padding must be integers"))
            .collect();
        let pairs: Vec<(i64, i64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        if let Ok(value_tensor) = value.extract::<Self>() {
            self.pad(pairs, value_tensor)
        } else if let Ok(value_val) = value.extract::<f64>() {
            self.pad(pairs, Tensor::from(value_val))
        } else {
            Err(ZyxError::DTypeError("value must be Tensor or numeric".into()))
        }
    }

    #[must_use]
    #[pyo3(name = "narrow")]
    pub fn narrow_py(&self, axis: Axis, start: Dim, length: Dim) -> Result<Tensor, ZyxError> {
        self.narrow(axis, start, length)
    }

    #[must_use]
    #[pyo3(name = "split")]
    pub fn split_py(&self, sizes: &Bound<'_, PyTuple>, axis: isize) -> Result<Vec<Tensor>, ZyxError> {
        self.split(to_sh(sizes)?, axis)
    }

    #[must_use]
    #[pyo3(name = "one_hot")]
    pub fn one_hot_py(&self, num_classes: Dim) -> Tensor {
        self.one_hot(num_classes)
    }

    #[must_use]
    #[pyo3(name = "masked_fill")]
    pub fn masked_fill_py(&self, mask: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(mask_tensor) = mask.extract::<Self>() {
            if let Ok(value_tensor) = value.extract::<Self>() {
                self.masked_fill(mask_tensor, value_tensor)
            } else if let Ok(value_val) = value.extract::<f64>() {
                self.masked_fill(mask_tensor, Tensor::from(value_val))
            } else {
                return Err(ZyxError::DTypeError("unsupported value for masked_fill".into()));
            }
        } else {
            return Err(ZyxError::DTypeError("unsupported mask for masked_fill".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "repeat")]
    pub fn repeat_py(&self, repeats: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.repeat(to_sh(repeats)?)
    }

    #[must_use]
    #[pyo3(name = "reshape", signature = (*shape))]
    pub fn reshape_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.reshape(to_sh(shape)?)
    }

    #[must_use]
    #[pyo3(name = "transpose")]
    pub fn transpose_py(&self, dim0: Axis, dim1: Axis) -> Result<Tensor, ZyxError> {
        self.transpose(dim0, dim1)
    }

    #[must_use]
    #[pyo3(name = "permute", signature = (*axes))]
    pub fn permute_py(&self, axes: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        let axes: Vec<Axis> = axes
            .into_iter()
            .map(|d| d.extract::<Axis>().expect("axes must be integers"))
            .collect();
        self.permute(axes)
    }

    #[must_use]
    #[pyo3(name = "squeeze", signature = (axes=None))]
    pub fn squeeze_py(&self, axes: Option<&Bound<'_, PyList>>) -> Tensor {
        match axes {
            Some(axes_list) => {
                let axes: Vec<Axis> = axes_list
                    .into_iter()
                    .map(|d| d.extract::<Axis>().expect("axes must be integers"))
                    .collect();
                self.squeeze(axes)
            }
            None => self.squeeze([]),
        }
    }

    #[must_use]
    #[pyo3(name = "unsqueeze")]
    pub fn unsqueeze_py(&self, dim: Axis) -> Result<Tensor, ZyxError> {
        self.unsqueeze(dim)
    }

    #[must_use]
    #[pyo3(name = "flatten")]
    pub fn flatten_py(&self, start_dim: Axis, end_dim: Axis) -> Result<Tensor, ZyxError> {
        self.flatten(start_dim..=end_dim)
    }

    #[must_use]
    #[pyo3(name = "expand")]
    pub fn expand_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.expand(to_sh(shape)?)
    }

    #[must_use]
    #[pyo3(name = "t")]
    pub fn t_py(&self) -> Tensor {
        return self.t();
    }

    #[must_use]
    #[pyo3(name = "to_le_bytes")]
    pub fn to_le_bytes_py(&self) -> Result<Vec<u8>, ZyxError> {
        self.to_le_bytes()
    }

    #[must_use]
    #[pyo3(name = "to_string")]
    pub fn to_string_py(&self) -> String {
        self.to_string()
    }

    #[must_use]
    #[pyo3(name = "product")]
    pub fn product_py(&self, axes: Option<&Bound<'_, PyList>>) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<Axis> = axes_list
                    .into_iter()
                    .map(|d| d.extract::<Axis>().expect("axes must be integers"))
                    .collect();
                self.prod(axes)
            }
            None => Ok(self.prod_all()), // Reduce all dimensions
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
            } else if let Ok(slice) = idx.downcast::<PySlice>() {
                Ok(vec![slice_to_dimindex(slice)?])
            } else if let Ok(tuple) = idx.downcast::<PyTuple>() {
                let mut ranges = Vec::with_capacity(tuple.len());
                for item in tuple.iter() {
                    if let Ok(i) = item.extract::<i64>() {
                        ranges.push(DimIndex::Index(i));
                    } else if let Ok(slice) = item.downcast::<PySlice>() {
                        ranges.push(slice_to_dimindex(slice)?);
                    } else {
                        return Err(PyIndexError::new_err("Tuple elements must be int or slice"));
                    }
                }
                Ok(ranges)
            } else if let Ok(list) = idx.downcast::<PyList>() {
                let mut ranges = Vec::with_capacity(list.len());
                for item in list.iter() {
                    if let Ok(slice) = item.downcast::<PySlice>() {
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

        self.slice(ranges).map_err(|e| PyIndexError::new_err(format!("{:?}", e)))
    }

    #[must_use]
    #[pyo3(name = "dot")]
    fn dot_py(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.dot(rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for dot".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "matmul")]
    fn matmul_py(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.dot(rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for matmul".into()));
        }
    }

    #[must_use]
    fn __matmul__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.dot(rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for dot".into()));
        }
    }

    #[must_use]
    fn __add__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self + rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self + rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()));
        }
    }

    #[must_use]
    fn __sub__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self - rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self - rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for sub".into()));
        }
    }

    #[must_use]
    fn __mul__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self * rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self * rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for mul".into()));
        }
    }

    #[must_use]
    fn __div__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self / rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self / rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()));
        }
    }

    #[must_use]
    #[pyo3(name = "argmax")]
    pub fn argmax_py(&self) -> Tensor {
        self.argmax()
    }

    #[must_use]
    #[pyo3(name = "argmax_axis")]
    pub fn argmax_axis_py(&self, axis: Axis) -> Result<Tensor, ZyxError> {
        self.argmax_axis(axis)
    }

    #[must_use]
    #[pyo3(name = "one_hot_along_dim")]
    pub fn one_hot_along_dim_py(&self, num_classes: Dim, dim: Axis) -> Result<Tensor, ZyxError> {
        self.one_hot_along_dim(num_classes, dim)
    }

    #[must_use]
    #[pyo3(name = "gather")]
    pub fn gather_py(&self, axis: Axis, indices: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(indices) = indices.extract::<Self>() {
            self.gather(axis, indices)
        } else {
            Err(ZyxError::DTypeError("indices must be a Tensor".into()))
        }
    }

    #[must_use]
    #[pyo3(name = "index_select")]
    pub fn index_select_py(&self, dim: Axis, index: &Bound<'_, PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(index) = index.extract::<Self>() {
            self.index_select(dim, index)
        } else {
            Err(ZyxError::DTypeError("index must be a Tensor".into()))
        }
    }

    #[must_use]
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
        self.conv(&weight, bias.as_ref(), groups, to_sh(stride)?, to_sh(dilation)?, to_sh(padding)?)
    }
}

fn to_sh(shape: &Bound<'_, PyTuple>) -> Result<Vec<Dim>, ZyxError> {
    if shape.len() == 1 {
        let first = shape.get_item(0).unwrap();

        // Check if first arg is list or tuple
        if first.is_instance_of::<PyList>() || first.is_instance_of::<PyTuple>() {
            let iter = PyIterator::from_object(&first).unwrap();
            let mut vec = Vec::new();

            for item in iter {
                let val = item.unwrap().extract::<usize>().unwrap();
                vec.push(Dim::try_from(val).map_err(|_| ZyxError::shape_error("dimension too large".into()))?);
            }

            return Ok(vec);
        }
    }

    // Otherwise treat each argument as a usize directly
    shape.as_slice().iter().map(|x| {
        let val: usize = x.extract().unwrap();
        Dim::try_from(val).map_err(|_| ZyxError::shape_error("dimension too large".into()))
    }).collect()
}

fn to_ax(axes: &Bound<'_, PyTuple>) -> Vec<Axis> {
    axes.into_iter()
        .map(|d| d.extract::<Axis>().expect("Shape must be positive integers"))
        .collect()
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<DType>()?;
    m.add_class::<GradientTape>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

fn from_numpy<T: crate::Scalar + pyo3::buffer::Element>(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let buffer = PyBuffer::<T>::get(obj)?;

    let shape: Vec<Dim> = buffer.shape().to_vec().into_iter().map(|s| Dim::try_from(s as usize).unwrap()).collect();
    let strides: Vec<Dim> = buffer.strides().to_vec().into_iter().map(|s| Dim::try_from(s as usize).unwrap()).collect();
    let data = buffer.as_slice(obj.py()).unwrap();

    let ndim = shape.len();
    assert_eq!(strides.len(), ndim);
    assert_eq!(shape.len(), ndim);

    let total_len: Dim = shape.iter().product();
    let mut result = Vec::with_capacity(total_len as usize);

    let mut indices = vec![0usize; ndim];

    for _ in 0..total_len as usize {
        // Compute flat index in strided source
        let mut offset_bytes: i64 = 0;
        for i in 0..ndim {
            let idx = indices[i];
            let s = strides[i];
            offset_bytes += (idx as i64) * (s as i64);
        }

        // Convert byte offset into index into `data`
        let element_size = std::mem::size_of::<T>() as i64;
        let index = (offset_bytes / element_size) as usize;

        result.push(data[index].get());

        // Advance indices (like an odometer)
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] as usize {
                break;
            }
            indices[d] = 0;
        }
    }

    Ok(Tensor::from(result).reshape(shape).unwrap())
}
