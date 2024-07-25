#![no_std]

// This is just a personal preference
//#![deny(clippy::implicit_return)]

extern crate alloc;

use crate::runtime::Runtime;

mod device;
mod dtype;
mod mutex;
mod runtime;
mod scalar;
mod shape;
mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use runtime::ZyxError;
pub use scalar::Scalar;
pub use shape::IntoShape;
pub use tensor::Tensor;
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
    Tensor::set_default_device(Device::CUDA);
    let x = Tensor::from([[2, 3], [4, 5]]);
    println!("{x}");
    //assert_eq!(x, [[2, 3], [4, 5]]);
}

#[test]
fn t1() {
    use std::println;
    //let x = Tensor::randn([2, 2], DType::F32).reshape(256).exp().expand([256, 4]);
    Tensor::set_default_device(Device::CUDA);
    /*let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]])
    .expand([2, 3, 2])
    .exp()
    .ln()
    .reshape([2, 3, 2, 1]);*/
    let x = Tensor::from([[[[2f32], [3.]]], [[[4.], [5.]]]]).expand([2, 3, 2, 1]);
    //println!("{x}");
    let y = Tensor::from([[2f32, 3., 1.], [4., 3., 2.]])
        .reshape([2, 3, 1, 1])
        .expand([2, 3, 2, 1]);
    //println!("{y}");
    let z = (&x + &y).expand([2, 3, 2, 2]).sum([3, 0]);
    Tensor::realize([&x, &y, &z]);
    println!("{x}\n{y}\n{z}");
    //println!("{z}");

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}
