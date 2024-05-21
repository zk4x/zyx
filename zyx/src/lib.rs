#![no_std]

extern crate alloc;

use crate::runtime::Runtime;

mod tensor;
mod runtime;
mod dtype;
mod device;
mod scalar;
mod shape;

pub use tensor::Tensor;
pub use dtype::DType;
pub use device::Device;

const SEED: u64 = 69420;

static RT: spin::Mutex<Runtime> = spin::Mutex::new(Runtime::new());

#[test]
fn t0() {
    let x = Tensor::randn([1024usize, 1024], DType::F16);

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}
