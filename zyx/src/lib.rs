#![no_std]

// This is just a personal preference
//#![deny(clippy::implicit_return)]

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
pub use shape::IntoShape;
pub use tensor::Tensor;
//pub use shape::IntoAxes;

#[cfg(feature = "rand")]
const SEED: u64 = 69420;

#[cfg(feature = "std")]
extern crate std;

/*struct Mutex<T>(spin::Mutex<T>, core::sync::atomic::AtomicBool);

impl<T> Mutex<T> {
    const fn new(value: T) -> Self {
        return Self(
            spin::Mutex::new(value),
            core::sync::atomic::AtomicBool::new(false),
        );
    }
}

impl<T> Mutex<T> {
    fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
        if self.1.load(core::sync::atomic::Ordering::SeqCst) {
            panic!("Trying to lock already locked mutex. Panicking in order to avoid deadlock.");
        } else {
            self.1.store(true, core::sync::atomic::Ordering::SeqCst);
        }
        println!("Locking guard.");
        return MutexGuard(self.0.lock(), self);
    }
}

struct MutexGuard<'a, T>(spin::MutexGuard<'a, T>, &'a Mutex<T>);

impl<T> core::ops::Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        return self.0.deref();
    }
}

impl<T> core::ops::DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        return self.0.deref_mut();
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.1 .1.store(false, core::sync::atomic::Ordering::SeqCst);
        println!("Dropping guard.");
    }
}*/

type Mutex<T> = spin::Mutex<T>;

static RT: Mutex<Runtime> = Mutex::new(Runtime::new());

#[test]
fn t0() {
    use std::println;
    //let x = Tensor::randn([2, 2], DType::F32).reshape(256).exp().expand([256, 4]);
    //let x = Tensor::from([[2, 3], [4, 5]]).cast(DType::F32).exp();
    let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]]).expand([2, 3, 2]);
    //.exp()
    //.sum(1);
    //.expand([3, 2, 3, 4]);
    //let x = Tensor::from([2, 3, 4]);
    //let y = Tensor::from([7, 6, 5]);
    //let z = x + y;
    //let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]]).exp();
    //println!("{:?}", x.shape());
    //Tensor::debug_graph();
    println!("{x}");

    //let l0 = zyx_nn::Linear::new(1024, 1024, DType::F16);
}
