//! Python bindings for zyx

#![allow(missing_docs)]

use crate::shape::Dim;
use crate::tensor::DebugGuard;
use crate::DebugMask;
use crate::{DType, GradientTape, Tensor, ZyxError, tensor::SAxis};
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
        let sources: Vec<Tensor> =
            sources.into_iter().map(|d| d.extract::<Tensor>().expect("sources must be List(Tensor)")).collect();
        self.gradient(x, &sources)
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
    #[pyo3(name = "plot_dot_graph")]
    pub fn plot_dot_graph_py(tensors: &Bound<'_, PyList>, name: &str) -> Result<(), std::io::Error> {
        let tensors: Vec<Tensor> =
            tensors.into_iter().map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)")).collect();
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
        let tensors: Vec<Tensor> =
            tensors.into_iter().map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)")).collect();
        Tensor::realize(&tensors)
    }

    #[must_use]
    #[pyo3(name = "shape")]
    pub fn shape_py(&self) -> Vec<usize> {
        self.shape()
    }

    #[must_use]
    #[pyo3(name = "numel")]
    pub fn numel_py(&self) -> usize {
        self.numel()
    }

    #[must_use]
    #[pyo3(name = "rank")]
    pub fn rank_py(&self) -> usize {
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
        let shape: Vec<usize> =
            shape.into_iter().map(|d| d.extract::<usize>().expect("Shape must be positive integers")).collect();
        return Tensor::ones(shape, dtype);
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
    pub fn eye_py(n: usize, dtype: DType) -> Tensor {
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

    // TODO celu

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
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.softmax(axes)
    }

    #[must_use]
    #[pyo3(name = "log_softmax")]
    pub fn log_softmax_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.ln_softmax(axes)
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
    #[pyo3(name = "expand")]
    pub fn expand_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.expand(to_sh(shape)?)
    }

    #[must_use]
    #[pyo3(name = "reshape")]
    pub fn reshape_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.reshape(to_sh(shape)?)
    }

    #[must_use]
    #[pyo3(name = "permute")]
    pub fn permute_py(&self, axes: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.permute(to_ax(axes))
    }

    #[must_use]
    #[pyo3(name = "flatten")]
    pub fn flatten_py(&self, start_axis: isize, end_axis: isize) -> Result<Tensor, ZyxError> {
        use core::ops::Bound;
        // For PyTorch compatibility, we need to handle the axes differently
        let range = if start_axis == 0 && end_axis == -1 {
            // Flatten all dimensions
            (Bound::Unbounded, Bound::Unbounded)
        } else {
            // Flatten specific range
            let start = if start_axis < 0 { self.rank() as isize + start_axis } else { start_axis };
            let end = if end_axis < 0 { self.rank() as isize + end_axis + 1 } else { end_axis + 1 };
            (Bound::Included(start as i32), Bound::Excluded(end as i32))
        };
        self.flatten(range)
    }

    #[must_use]
    #[pyo3(name = "squeeze")]
    pub fn squeeze_py(&self, axes: Option<&Bound<'_, PyList>>) -> Tensor {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.squeeze(axes)
            }
            None => self.squeeze(vec![]), // Squeeze all dimensions of size 1
        }
    }

    #[must_use]
    #[pyo3(name = "unsqueeze")]
    pub fn unsqueeze_py(&self, dim: isize) -> Result<Tensor, ZyxError> {
        self.unsqueeze(dim)
    }

    #[must_use]
    #[pyo3(name = "transpose")]
    pub fn transpose_py(&self, dim0: isize, dim1: isize) -> Result<Tensor, ZyxError> {
        self.transpose(dim0 as SAxis, dim1 as SAxis)
    }

    #[must_use]
    #[pyo3(name = "t")]
    pub fn t_py(&self) -> Tensor {
        self.t()
    }

    #[must_use]
    #[pyo3(name = "view")]
    pub fn view_py(&self, shape: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.reshape(to_sh(shape)?)
    }

    #[must_use]
    #[pyo3(name = "max")]
    pub fn max_py(&self, axes: Option<&Bound<'_, PyList>>) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.max(axes)
            }
            None => self.max(vec![]), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "mean")]
    pub fn mean_py(&self, axes: Option<&Bound<'_, PyList>>) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.mean(axes)
            }
            None => self.mean(vec![]), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "sum")]
    pub fn sum_py(&self, axes: Option<&Bound<'_, PyList>>) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.sum(axes)
            }
            None => self.sum(vec![]), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "std", signature = (axes=None, correction=1))]
    pub fn std_py(&self, axes: Option<&Bound<'_, PyList>>, correction: usize) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.std(axes, correction)
            }
            None => self.std(vec![], correction), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "var", signature = (axes=None, correction=1))]
    pub fn var_py(&self, axes: Option<&Bound<'_, PyList>>, correction: usize) -> Result<Tensor, ZyxError> {
        match axes {
            Some(axes_list) => {
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.var(axes, correction)
            }
            None => self.var(vec![], correction), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "cumsum")]
    pub fn cumsum_py(&self, axis: isize) -> Result<Tensor, ZyxError> {
        self.cumsum(axis as SAxis)
    }

    // Missing unary operations
    #[must_use]
    #[pyo3(name = "exp2")]
    pub fn exp2_py(&self) -> Tensor {
        self.exp2()
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
        let padding: Vec<(isize, isize)> = padding.into_iter().map(|d| d.extract::<(isize, isize)>().expect("padding must be tuple of (isize, isize)")).collect();
        self.pad_zeros(padding)
    }

    #[must_use]
    #[pyo3(name = "narrow")]
    pub fn narrow_py(&self, axis: isize, start: usize, length: usize) -> Result<Tensor, ZyxError> {
        self.narrow(axis as SAxis, start, length)
    }

    #[must_use]
    #[pyo3(name = "split")]
    pub fn split_py(&self, sizes: &Bound<'_, PyTuple>, axis: isize) -> Result<Vec<Tensor>, ZyxError> {
        self.split(to_sh(sizes)?, axis)
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
                let axes: Vec<SAxis> = axes_list.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
                self.product(axes)
            }
            None => self.product(vec![]), // Reduce all dimensions
        }
    }

    #[must_use]
    #[pyo3(name = "mean_kd")]
    pub fn mean_kd_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.mean_kd(axes)
    }

    #[must_use]
    #[pyo3(name = "sum_kd")]
    pub fn sum_kd_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.sum_kd(axes)
    }

    #[must_use]
    #[pyo3(name = "max_kd")]
    pub fn max_kd_py(&self, axes: &Bound<'_, PyList>) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.max_kd(axes)
    }

    #[must_use]
    #[pyo3(name = "std_kd")]
    pub fn std_kd_py(&self, axes: &Bound<'_, PyList>, correction: usize) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.std_kd(axes, correction)
    }

    #[must_use]
    #[pyo3(name = "var_kd")]
    pub fn var_kd_py(&self, axes: &Bound<'_, PyList>, correction: usize) -> Result<Tensor, ZyxError> {
        let axes: Vec<SAxis> = axes.into_iter().map(|d| d.extract::<SAxis>().expect("axes must be integers")).collect();
        self.var_kd(axes, correction)
    }


    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        // Convert Python slice into Rust Range<isize>
        fn slice_to_range(slice: &Bound<'_, PySlice>) -> PyResult<std::ops::Range<isize>> {
            let indices = slice.indices(isize::MAX)?;
            if indices.step != 1 {
                return Err(PyIndexError::new_err("Slice step != 1 is not supported"));
            }
            Ok(indices.start..indices.stop)
        }

        // Recursively parse index into Vec<Range<isize>>
        fn index_to_ranges(idx: &Bound<'_, PyAny>) -> PyResult<Vec<std::ops::Range<isize>>> {
            if let Ok(i) = idx.extract::<isize>() {
                Ok(vec![i..i + 1])
            } else if let Ok(slice) = idx.downcast::<PySlice>() {
                Ok(vec![slice_to_range(slice)?])
            } else if let Ok(tuple) = idx.downcast::<PyTuple>() {
                let mut ranges = Vec::with_capacity(tuple.len());
                for item in tuple.iter() {
                    if let Ok(i) = item.extract::<isize>() {
                        ranges.push(i..i + 1);
                    } else if let Ok(slice) = item.downcast::<PySlice>() {
                        ranges.push(slice_to_range(slice)?);
                    } else {
                        return Err(PyIndexError::new_err("Tuple elements must be int or slice"));
                    }
                }
                Ok(ranges)
            } else if let Ok(list) = idx.downcast::<PyList>() {
                let mut ranges = Vec::with_capacity(list.len());
                for item in list.iter() {
                    if let Ok(slice) = item.downcast::<PySlice>() {
                        ranges.push(slice_to_range(slice)?);
                    } else {
                        return Err(PyIndexError::new_err("List elements must be slices"));
                    }
                }
                Ok(ranges)
            } else {
                Err(PyIndexError::new_err("Unsupported index type"))
            }
        }

        let ranges = index_to_ranges(idx)?;

        self.get(ranges).map_err(|e| PyIndexError::new_err(format!("{:?}", e)))
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
}

fn to_sh(shape: &Bound<'_, PyTuple>) -> Result<Vec<usize>, ZyxError> {
    if shape.len() == 1 {
        let first = shape.get_item(0).unwrap();

        // Check if first arg is list or tuple
        if first.is_instance_of::<PyList>() || first.is_instance_of::<PyTuple>() {
            let iter = PyIterator::from_object(&first).unwrap();
            let mut vec = Vec::new();

            for item in iter {
                let val = item.unwrap().extract::<usize>().unwrap();
                vec.push(val);
            }

            return Ok(vec);
        }
    }

    // Otherwise treat each argument as a usize directly
    Ok(shape.as_slice().iter().map(|x| x.extract::<usize>().unwrap()).collect())
}

fn to_ax(axes: &Bound<'_, PyTuple>) -> Vec<SAxis> {
    axes.into_iter().map(|d| d.extract::<SAxis>().expect("Shape must be positive integers")).collect()
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

    let shape = buffer.shape().to_vec();
    let strides = buffer.strides().to_vec();
    let data = buffer.as_slice(obj.py()).unwrap();
    let data2: Vec<T> = data.iter().map(|x| x.get()).collect();
    println!("dtype={}, shape={shape:?}, strides={strides:?}, {data2:?}", T::dtype());

    let ndim = shape.len();
    assert_eq!(strides.len(), ndim);
    assert_eq!(shape.len(), ndim);

    let total_len: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total_len);

    let mut indices = vec![0; ndim];

    for _ in 0..total_len {
        // Compute flat index in strided source
        let mut offset_bytes: isize = 0;
        for (i, &stride) in indices.iter().zip(strides.iter()) {
            offset_bytes += (*i as isize) * stride;
        }

        // Convert byte offset into index into `data`
        let element_size = std::mem::size_of::<T>() as isize;
        let index = (offset_bytes / element_size) as usize;

        result.push(data[index].get());

        // Advance indices (like an odometer)
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Ok(Tensor::from(result).reshape(shape).unwrap())
}
