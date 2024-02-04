mod unary;

use zyx_core::backend::Backend;
use zyx_core::scalar::Scalar;
use zyx_opencl::ZyxError;

fn assert_eq<T: Scalar>(x: impl IntoIterator<Item = T>, y: impl IntoIterator<Item = T>) {
    let x: Vec<T> = x.into_iter().collect();
    let y: Vec<T> = y.into_iter().collect();
    assert_eq!(x.len(), y.len());
    for (i, (ex, ey)) in x.into_iter().zip(y).enumerate() {
        if !ex.clone().is_equal(ey.clone()) {
            panic!("Elements {ex:?} and {ey:?} at index {i} are not equal.");
        }
    }
}

fn run_test_fn<T: Scalar, F: Fn(B, T) -> Result<(), ZyxError>, B: Backend>(test_fn: F, backend: B, x: T) {
    println!();
    let name = std::any::type_name::<F>();
    //let b = std::any::type_name::<B>();
    print!("Running test {name} with dtype {:?} ... ", T::dtype());
    let begin = std::time::Instant::now();
    let res = test_fn(backend, x);
    let elapsed = begin.elapsed().as_nanos();
    res.unwrap_or_else(|err| panic!("Test {name} failed with error {err}"));
    println!("OK, time taken: {:.3} ms", elapsed as f32 / 1000000.);
}

macro_rules! run_test {
    ( $test:expr ) => {
        {
            let dev = zyx_opencl::device().unwrap();
            run_test_fn($test, &dev, 0f32);
            run_test_fn($test, &dev, 0i32);
        }
    }
}

fn main() {
    // TODO test for all devices
    println!("\nTesting tensor initialization");
    println!("\nTesting unary ops");
    run_test!(unary::neg);
    run_test!(unary::relu);
    run_test!(unary::sin);
    run_test!(unary::cos);
    run_test!(unary::ln);
    run_test!(unary::exp);
    run_test!(unary::tanh);
    run_test!(unary::sqrt);
    println!("\nTesting binary ops");
    println!("\nTesting movement ops");
    println!("\nTesting reduce ops");
    println!("\nTesting combination ops");
    println!("\nTesting autograd engine");
    println!("\nTesting optimizers");
    println!("\nTesting zyx-nn modules");
}
