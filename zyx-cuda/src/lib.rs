use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

#[test]
fn t0() -> Result<(), DriverError> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let inp = dev.htod_copy(vec![1.0f32; 100])?;
    let mut out = dev.alloc_zeros::<f32>(100)?;

    let ptx = cudarc::nvrtc::compile_ptx("
        extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numel) {
            out[i] = inp[i];
        }
    }").unwrap();

    dev.load_ptx(ptx, "my_module", &["sin_kernel"]).unwrap();

    let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    println!("{out_host:?}");
    Ok(())
}

#[test]
fn t1() {
    let device = CudaDevice::new(0).unwrap();

    let a_dev: CudaSlice<f32> = device.alloc_zeros(10).unwrap();
    let b_dev: CudaSlice<f32> = device.htod_copy(vec![0.0; 10]).unwrap();
    let c_dev: CudaSlice<f32> = device.htod_sync_copy(&[1.0, 2.0, 3.0]).unwrap();

    let a_dev: CudaSlice<f32> = device.alloc_zeros(10).unwrap();
    let mut a_buf: [f32; 10] = [1.0; 10];
    device.dtoh_sync_copy_into(&a_dev, &mut a_buf);
    assert_eq!(a_buf, [0.0; 10]);
    let a_host: Vec<f32> = device.sync_reclaim(a_dev).unwrap();
    assert_eq!(&a_host, &[0.0; 10]);

    let source = "extern \"C\" __global__ void my_function(float *out) { }";

    println!("Compiling\n{source}");

    let ptx = compile_ptx(source).unwrap();

    println!("{ptx:?}");

    device.load_ptx(ptx, "module_name", &["my_function"]).unwrap();

    let func: CudaFunction = device.get_func("module_name", "my_function").unwrap();

    let mut a = device.alloc_zeros::<f32>(10).unwrap();
    let cfg = LaunchConfig::for_num_elems(10);
    unsafe { func.launch(cfg, (&mut a,)) }.unwrap();
}
