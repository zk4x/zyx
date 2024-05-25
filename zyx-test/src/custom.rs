use zyx::{DType, Device, Tensor};

use crate::assert_eq;

pub(super) fn small_tiled_dot() {
    let n = 128;

    let x = Tensor::randn([n, n], DType::F32);
    let y_ocl = Tensor::randn([n, n], DType::F32);
    let z_ocl = x.dot(&y_ocl);

    let x = Tensor::randn([n, n], DType::F32);
    let y = Tensor::randn([n, n], DType::F32);
    assert_eq(y_ocl.to_vec::<f32>().unwrap().into_iter(), y.to_vec::<f32>().unwrap().into_iter());
    let z_cpu = x.dot(y);

    assert_eq(z_ocl.to_vec::<f32>().unwrap().into_iter(), z_cpu.to_vec::<f32>().unwrap().into_iter());
    Ok(())
}

pub(super) fn large_tiled_dot() {
    let x = Tensor::randn([1024, 2048], DType::F32).to(Device::CPU);
    let y = Tensor::randn([2048, 1024], DType::F32).to(Device::CPU);

    let x_vec: Vec<f32> = x.to_vec()?;
    let y_vec: Vec<f32> = y.to_vec()?;
    let z_vec: Vec<f32> = x.dot(y).to_vec()?;

    let x = Tensor::from(x_vec)?.reshape([1024, 2048]).to(Device::OpenCL);
    let y = Tensor::from(y_vec)?.reshape([2048, 1024]).to(Device::OpenCL);

    let z = x.dot(y).to_vec()?;

    assert_eq(z, z_vec);

    Ok(())
}

// TODO write tests for all variations of tiled reduce kernels
