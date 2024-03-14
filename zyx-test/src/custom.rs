use zyx_core::dtype::DType;
use zyx_core::error::ZyxError;
use crate::assert_eq;

pub(super) fn small_tiled_dot() -> Result<(), ZyxError> {
    let n = 128;

    let dev = zyx_opencl::device().unwrap();
    let x = dev.randn([n, n], zyx_opencl::DType::F32);
    let y_ocl = dev.randn([n, n], zyx_opencl::DType::F32);
    let z_ocl = x.dot(&y_ocl);

    let dev = zyx_cpu::device().unwrap();
    let x = dev.randn([n, n], zyx_opencl::DType::F32);
    let y = dev.randn([n, n], zyx_opencl::DType::F32);
    assert_eq(y_ocl.to_vec::<f32>().unwrap().into_iter(), y.to_vec::<f32>().unwrap().into_iter());
    let z_cpu = x.dot(y);

    assert_eq(z_ocl.to_vec::<f32>().unwrap().into_iter(), z_cpu.to_vec::<f32>().unwrap().into_iter());
    Ok(())
}

pub(super) fn large_tiled_dot() -> Result<(), ZyxError> {
    let torch = zyx_torch::device()?;
    let ocl = zyx_opencl::device()?;
    let x = torch.randn([1024, 2048], DType::F32);
    let y = torch.randn([2048, 1024], DType::F32);

    let x_vec: Vec<f32> = x.to_vec()?;
    let y_vec: Vec<f32> = y.to_vec()?;
    let z_vec: Vec<f32> = x.dot(y).to_vec()?;

    let x = ocl.tensor(x_vec).reshape([1024, 2048]);
    let y = ocl.tensor(y_vec).reshape([2048, 1024]);

    let z = x.dot(y).to_vec()?;

    assert_eq(z, z_vec);

    Ok(())
}