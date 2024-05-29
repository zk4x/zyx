#![no_std]

extern crate alloc;

use crate::runtime::Runtime;

mod device;
mod dtype;
mod runtime;
mod scalar;
mod shape;
mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use scalar::Scalar;
pub use shape::IntoAxes;
pub use shape::IntoShape;
pub use tensor::Tensor;
pub use runtime::ZyxError;

const SEED: u64 = 69420;

static RT: spin::Mutex<Runtime> = spin::Mutex::new(Runtime::new());

#[test]
fn t0() {
    use libc_print::std_name::println;
    let x = Tensor::randn([1024, 1024], DType::F32);
    println!("{x}");

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}
