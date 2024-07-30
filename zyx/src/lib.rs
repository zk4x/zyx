#![no_std]

// This is just a personal preference
//#![deny(clippy::implicit_return)]

extern crate alloc;

use crate::runtime::Runtime;

mod device;
mod dtype;
mod mutex;
#[cfg(feature = "py")]
mod python_bindings;
mod runtime;
mod scalar;
mod shape;
mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use scalar::Scalar;
pub use shape::IntoShape;
pub use tensor::Tensor;
//pub use runtime::ZyxError;
//pub use shape::IntoAxes;

#[cfg(feature = "rand")]
const SEED: u64 = 69420;

#[cfg(feature = "std")]
extern crate std;

//#[cfg(not(feature = "std"))]
static RT: mutex::Mutex<Runtime, 1000000> = mutex::Mutex::new(Runtime::new());
//#[cfg(feature = "std")]
//static RT: std::sync::Mutex<Runtime> = std::sync::Mutex::new(Runtime::new());

// Load and save test
#[test]
fn t0() {
    use std::println;
    Tensor::set_default_device(Device::CPU);
    let x = Tensor::from([[2, 3], [4, 5]]);
    println!("{x}");
    //assert_eq!(x, [[2, 3], [4, 5]]);
}

// Unary test
#[test]
fn t1() {
    use std::println;
    Tensor::set_default_device(Device::OpenCL);
    let x = Tensor::from([[2f32, 3.], [4., 5.]]).exp();
    println!("{x}");
    //assert_eq!(x, [[2, 3], [4, 5]]);
}

#[test]
fn t2() {
    use std::println;
    //let x = Tensor::randn([2, 2], DType::F32).reshape(256).exp().expand([256, 4]);
    Tensor::set_default_device(Device::OpenCL);
    let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]])
        .expand([2, 3, 2])
        .exp()
        .ln()
        .reshape([2, 3, 2, 1]);
    //let x = Tensor::from([[[[2f32], [3.]]], [[[4.], [5.]]]]).expand([2, 3, 2, 1]);
    //println!("{x}");
    let y = Tensor::from([[2f32, 3., 1.], [4., 3., 2.]])
        .reshape([2, 3, 1, 1])
        .expand([2, 3, 2, 1]);
    //println!("{y}");
    let z = (&x + &y).expand([2, 3, 2, 2]).sum([3, 0]);
    let z = z.exp().ln().permute([1, 0]).sum(0);
    //Tensor::plot_dot_graph([&x, &y, &z], "graph0");
    //Tensor::realize([&x, &y, &z]);
    //println!("{x}\n{y}\n{z}");
    println!("{z}");

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}

#[test]
fn t3() {
    let x = Tensor::randn([1024, 1024], DType::F32).expand([1024, 1024, 1024]);
    Tensor::realize([&x]);
}

#[test]
fn t4() {
    let x = Tensor::randn([1, 1024, 1024], DType::F32);
    let y = Tensor::randn([1024, 1, 1024], DType::F32);
    let z = (x * y).sum(2);
    Tensor::realize([&z]);
}

#[test]
fn t5() {
    let x = Tensor::from([[2f32, 3.], [4., 5.]]);
    let y = x.transpose();
    let z = x.exp();
    //Tensor::plot_dot_graph([&y, &z], "graph1");
    Tensor::realize([&y, &z]);
    std::println!("{y}\n{z}");
}

#[test]
fn t6() {
    //let x = Tensor::from([[2, 3], [4, 5]]).pad_zeros([(1, 3)]);

    let x = Tensor::randn([14, 16], DType::U8);
    let x = x.get((.., 8..-2));
    std::println!("{x}");
}
