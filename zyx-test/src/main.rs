#![allow(dead_code)]

mod autograd;
mod binary;
mod combination;
mod movement;
mod nn;
mod optimizer;
mod reduce;
mod unary;
mod custom;

use zyx_core::backend::Backend;
use zyx_core::error::ZyxError;
use zyx_core::scalar::Scalar;

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

fn run_custom_test<F: Fn() -> Result<(), ZyxError>>(test_fn: F) {
    let name: String = std::any::type_name::<F>().into();
    /*for _ in 0..4 {
        if let Some(index) = name.find(':') {
            name = name[index + 1..].into();
        }
    }*/
    //name.replace_range(name.find('&').unwrap() + 1..name.find(':').unwrap() + 2, "");
    print!("Running test {name} ... ");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let begin = std::time::Instant::now();
    let res = test_fn();
    let elapsed = begin.elapsed().as_nanos();
    res.unwrap_or_else(|err| panic!("Test {name} failed with error {err}"));
    println!("OK, time taken: {:.3} ms", elapsed as f32 / 1000000.);
}

fn run_test_fn<T: Scalar, F: Fn(B, T) -> Result<(), ZyxError>, B: Backend>(
    test_fn: F,
    backend: B,
    x: T,
) {
    let mut name: String = std::any::type_name::<F>().into();
    //let b = std::any::type_name::<B>();
    for _ in 0..4 {
        if let Some(index) = name.find(':') {
            name = name[index + 1..].into();
        }
    }
    name.replace_range(name.find('&').unwrap() + 1..name.find(':').unwrap() + 2, "");
    print!("Running test {name} ... ");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let begin = std::time::Instant::now();
    let res = test_fn(backend, x);
    let elapsed = begin.elapsed().as_nanos();
    res.unwrap_or_else(|err| panic!("Test {name} failed with error {err}"));
    println!("OK, time taken: {:.3} ms", elapsed as f32 / 1000000.);
}

macro_rules! run_test {
    ( $test:expr ) => {{
        let dev = zyx_opencl::device().unwrap();
        run_test_fn($test, &dev, 0f32);
        run_test_fn($test, &dev, 0f64);
        run_test_fn($test, &dev, 0i32);
        let dev = zyx_cpu::device().unwrap();
        run_test_fn($test, &dev, 0f32);
        run_test_fn($test, &dev, 0f64);
        run_test_fn($test, &dev, 0i32);
        let dev = zyx_torch::device().unwrap();
        run_test_fn($test, &dev, 0f32);
        run_test_fn($test, &dev, 0f64);
        run_test_fn($test, &dev, 0i32);
    }};
}

fn main() {
    println!("\nTesting custom ops");
    run_custom_test(custom::small_tiled_dot);

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
    run_test!(binary::add);
    run_test!(binary::sub);
    run_test!(binary::mul);
    run_test!(binary::div);
    //run_test!(binary::pow);
    run_test!(binary::cmplt);
    // TODO test where
    println!("\nTesting movement ops");
    // TODO more detailed movement ops tests
    run_test!(movement::reshape);
    run_test!(movement::expand);
    run_test!(movement::permute);
    run_test!(movement::pad);
    println!("\nTesting reduce ops");
    run_test!(reduce::sum);
    run_test!(reduce::max);
    println!("\nTesting combinations of ops");
    // TODO more detailed combinations of ops tests
    run_test!(combination::t0);
    run_test!(combination::t1);
    run_test!(combination::dot);
    //run_test!(combination::cat);
    //run_test!(combination::split);
    println!("\nTesting autograd engine");
    run_test!(autograd::t0);
    run_test!(autograd::t1);
    println!("\nTesting optimizers");
    run_test!(optimizer::sgd);
    //run_test!(optimizer::adam);
    println!("\nTesting nn modules");
    run_test!(nn::linear);
    //run_test!(nn::layer_norm);
    //run_test!(nn::batch_norm);
}
