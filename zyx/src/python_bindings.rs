//! Python bindings for zyx

use pyo3::{
    exceptions::PyOSError,
    pymethods, pymodule,
    types::{PyAnyMethods, PyList, PyModule, PyModuleMethods, PyTuple},
    Bound, PyAny, PyErr, PyResult,
};

use crate::{
    runtime::{BackendConfig, ZyxError},
    DType, Tensor,
};

impl From<ZyxError> for PyErr {
    fn from(err: ZyxError) -> Self {
        PyOSError::new_err(format!("{err:?}"))
    }
}

#[pymethods]
impl Tensor {
    #[staticmethod]
    #[pyo3(name = "plot_dot_graph")]
    pub fn plot_dot_graph_py(tensors: &Bound<'_, PyList>, name: &str) {
        let tensors: Vec<Tensor> = tensors
            .into_iter()
            .map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)"))
            .collect();
        Tensor::plot_dot_graph(&tensors, name);
    }

    #[staticmethod]
    #[pyo3(name = "configure_backends")]
    pub fn configure_backends_py(config: &BackendConfig) -> Result<(), ZyxError> {
        Tensor::configure_backends(config)
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

    #[must_use]
    #[pyo3(name = "backward")]
    pub fn backward_py(&self, sources: &Bound<'_, PyList>) -> Vec<Option<Tensor>> {
        let sources: Vec<Tensor> = sources
            .into_iter()
            .map(|d| d.extract::<Tensor>().expect("sources must be List(Tensor)"))
            .collect();
        self.backward(&sources)
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

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "randn", signature = (*shape, dtype=DType::F32))]
    pub fn randn_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Tensor {
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|d| {
                d.extract::<usize>()
                    .expect("Shape must be positive integers")
            })
            .collect();
        return Tensor::randn(shape, dtype);
    }

    // TODO uniform

    // TODO kaiming_uniform

    #[staticmethod]
    #[must_use]
    #[pyo3(name = "zeros", signature = (*shape, dtype=DType::F32))]
    pub fn zeros_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Tensor {
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|d| {
                d.extract::<usize>()
                    .expect("Shape must be positive integers")
            })
            .collect();
        return Tensor::zeros(shape, dtype);
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
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<DType>()?;
    m.add_class::<BackendConfig>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
