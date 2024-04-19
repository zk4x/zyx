use std::sync::{Arc, Mutex};
use pyo3::pyclass;
use zyx_cpu::CPU;
use zyx_opencl::OpenCL;

#[pyclass]
pub struct Device();

pub enum Dev {
    CPU(CPU),
    OpenCL(OpenCL),
}
