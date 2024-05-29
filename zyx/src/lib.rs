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
pub use runtime::ZyxError;
pub use scalar::Scalar;
pub use shape::IntoAxes;
pub use shape::IntoShape;
pub use tensor::Tensor;

const SEED: u64 = 69420;

static RT: spin::Mutex<Runtime> = spin::Mutex::new(Runtime::new());

#[test]
fn t0() {
    use libc_print::std_name::println;
    let x = Tensor::randn([16, 16], DType::F32).reshape(256);
    println!("{:?}", x.shape());
    Tensor::debug_graph();
    println!("{x}");

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}
