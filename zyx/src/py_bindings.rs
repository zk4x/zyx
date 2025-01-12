//! Python bindings for zyx

#![allow(missing_docs)]

use pyo3::{
    exceptions::PyOSError, pymethods, pymodule, types::{PyAnyMethods, PyList, PyModule, PyModuleMethods, PyTuple}, Bound, FromPyObject, PyAny, PyErr, PyResult
};

use crate::{
    runtime::ZyxError, DType, GradientTape, Tensor
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
        self.gradient(x, &sources)
    }
}

#[pymethods]
impl Tensor {
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

    /*#[must_use]
    #[pyo3(name = "detach")]
    pub fn detach() -> Result<Tensor, ZyxError> {
        todo!()
    }*/

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "randn", signature = (*shape, dtype=DType::F32))]
    pub fn randn_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::randn(to_sh(shape)?, dtype)
    }

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "rand", signature = (*shape, dtype=DType::F32))]
    pub fn rand_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Result<Tensor, ZyxError> {
        Tensor::rand(to_sh(shape)?, dtype)
    }

    // TODO uniform

    // TODO kaiming_uniform

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
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|d| {
                d.extract::<usize>()
                    .expect("Shape must be positive integers")
            })
            .collect();
        return Tensor::ones(shape, dtype);
    }

    // TODO full

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "eye", signature = (n, dtype=DType::F32))]
    pub fn eye_py(n: usize, dtype: DType) -> Tensor {
        return Tensor::eye(n, dtype);
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

    // TODO dropout

    // TODO elu

    #[must_use]
    #[pyo3(name = "exp")]
    pub fn exp_py(&self) -> Tensor {
        return self.exp();
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

    /*#[must_use]
    #[pyo3(name = "pad")]
    pub fn pad_py(&self, padding: &Bound<'_, PyTuple>) -> Result<Tensor, ZyxError> {
        self.pad(padding)
    }*/

    #[must_use]
    #[pyo3(name = "dot")]
    fn dot_py(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            self.dot(rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()))
        }
    }

    #[must_use]
    fn __sub__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self - rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self - rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()))
        }
    }

    #[must_use]
    fn __mul__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self * rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self * rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()))
        }
    }

    #[must_use]
    fn __div__(&self, rhs: &Bound<PyAny>) -> Result<Tensor, ZyxError> {
        if let Ok(rhs) = rhs.extract::<Self>() {
            Ok(self / rhs)
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            Ok(self / rhs)
        } else {
            return Err(ZyxError::DTypeError("unsupported rhs for add".into()))
        }
    }
}

fn to_sh(shape: &Bound<'_, PyAny>) -> Result<Vec<usize>, ZyxError> {
    if shape.is_none() {
        return Err(ZyxError::ShapeError("Shape cannot be None".into()));
    }
    let tuple = shape.downcast::<PyTuple>().unwrap();
    if tuple.len().unwrap() == 1 {
        let first_element = tuple.get_item(0).unwrap();
        let dims: Vec<usize> = FromPyObject::extract_bound(&first_element).unwrap();
        Ok(dims)
    } else {
        let dims: Vec<usize> = FromPyObject::extract_bound(tuple).unwrap();
        Ok(dims)
    }
}

fn to_ax(axes: &Bound<'_, PyTuple>) -> Vec<isize> {
    axes
        .into_iter()
        .map(|d| {
            d.extract::<isize>()
                .expect("Shape must be positive integers")
        })
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
