//! Python bindings for zyx

#![allow(missing_docs)]

use crate::{DType, GradientTape, Tensor, ZyxError, tensor::SAxis};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
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
        let shape: Vec<usize> =
            shape.into_iter().map(|d| d.extract::<usize>().expect("Shape must be positive integers")).collect();
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

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.to_string()
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
