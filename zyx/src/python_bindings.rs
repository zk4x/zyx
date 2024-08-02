use std::vec::Vec;

use pyo3::{
    pymethods, pymodule,
    types::{PyAnyMethods, PyList, PyModule, PyModuleMethods, PyTuple},
    Bound, PyResult,
};

use crate::{DType, Device, Tensor};

#[pymethods]
impl Tensor {
    #[staticmethod]
    #[must_use]
    #[pyo3(name = "default_device")]
    pub fn default_device_py() -> Device {
        return Tensor::default_device()
    }

    #[staticmethod]
    #[pyo3(name = "plot_dot_graph")]
    pub fn plot_dot_graph_py(tensors: &Bound<'_, PyList>, name: &str) {
        let tensors: Vec<Tensor> = tensors.into_iter().map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)")).collect();
        Tensor::plot_dot_graph(&tensors, name);
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
    #[pyo3(name = "set_default_device")]
    pub fn set_default_device_py(device: Device) -> bool {
        return Tensor::set_default_device(device);
    }

    #[must_use]
    #[pyo3(name = "backward")]
    pub fn backward_py(&self, sources: &Bound<'_, PyList>) -> Vec<Option<Tensor>> {
        let sources: Vec<Tensor> = sources.into_iter().map(|d| d.extract::<Tensor>().expect("sources must be List(Tensor)")).collect();
        self.backward(&sources)
    }

    #[staticmethod]
    #[pyo3(name = "realize")]
    pub fn realize_py(tensors: &Bound<'_, PyList>) {
        let tensors: Vec<Tensor> = tensors.into_iter().map(|d| d.extract::<Tensor>().expect("tensors must be List(Tensor)")).collect();
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

    #[must_use]
    #[pyo3(name = "device")]
    pub fn device_py(&self) -> Device {
        self.device()
    }

    #[must_use]
    #[pyo3(name = "to")]
    pub fn to_py(&self, device: Device) -> Tensor {
        return self.clone().to(device);
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
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<DType>()?;
    m.add_class::<Device>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
