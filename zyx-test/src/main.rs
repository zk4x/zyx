#![allow(dead_code)]

use zyx::{Device, Scalar, Tensor};

mod autograd;
mod binary;
mod combination;
mod movement;
mod nn;
mod optimizer;
mod reduce;
mod unary;
mod custom;
mod initialization;
mod ternary;

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

fn test_dtype<T: Scalar, F: Fn(T)>(test_fn: F) {
    for device in [Device::CUDA, Device::OpenCL, Device::WGPU, Device::CPU] {
        if Tensor::set_default_device(device) {
            let mut name: String = std::any::type_name::<F>().into();
            for _ in 0..4 {
                if let Some(index) = name.find(':') {
                    name = name[index + 1..].into();
                }
            }
            name.replace_range(name.find('&').unwrap() + 1..name.find(':').unwrap() + 2, "");
            print!("Running test {name} on device {device} ... ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let begin = std::time::Instant::now();
            test_fn();
            let elapsed = begin.elapsed().as_nanos();
            println!("OK, time taken: {:.3} ms", elapsed as f32 / 1000000.);
        } else {
            println!("Device {device} is not available.");
        }
    }
}

fn main() {
    println!("\nTesting tensor initialization");
    test_dtype::<f32, _>(initialization::randn);
    test_dtype::<f32, _>(initialization::uniform);
    test_dtype::<f32, _>(initialization::kaiming_uniform);
    test_dtype::<f32, _>(initialization::zeros);
    test_dtype::<f32, _>(initialization::ones);
    test_dtype::<f32, _>(initialization::full);
    test_dtype::<f32, _>(initialization::eye);

    println!("\nTesting unary ops");
    test_dtype::<f32, _>(unary::neg);
    test_dtype::<f32, _>(unary::relu);
    test_dtype::<f32, _>(unary::sin);
    test_dtype::<f32, _>(unary::cos);
    test_dtype::<f32, _>(unary::ln);
    test_dtype::<f32, _>(unary::exp);
    test_dtype::<f32, _>(unary::tanh);
    test_dtype::<f32, _>(unary::sqrt);

    println!("\nTesting binary ops");
    test_dtype::<f32, _>(binary::add);
    test_dtype::<f32, _>(binary::sub);
    test_dtype::<f32, _>(binary::mul);
    test_dtype::<f32, _>(binary::div);
    test_dtype::<f32, _>(binary::pow);
    test_dtype::<f32, _>(binary::cmplt);

    println!("\nTesting ternary ops");
    test_dtype::<f32, _>(ternary::_where);

    println!("\nTesting movement ops");
    // TODO more detailed movement ops tests
    test_dtype::<f32, _>(movement::reshape);
    test_dtype::<f32, _>(movement::expand);
    test_dtype::<f32, _>(movement::permute);
    test_dtype::<f32, _>(movement::pad);

    println!("\nTesting reduce ops");
    test_dtype::<f32, _>(reduce::sum);
    test_dtype::<f32, _>(reduce::max);

    println!("\nTesting custom ops");
    test_dtype::<f32, _>(custom::small_tiled_dot);

    println!("\nTesting combinations of ops");
    // TODO more detailed combinations of ops tests
    test_dtype::<f32, _>(combination::t0);
    test_dtype::<f32, _>(combination::t1);
    test_dtype::<f32, _>(combination::dot);
    test_dtype::<f32, _>(combination::cat);
    test_dtype::<f32, _>(combination::split);

    println!("\nTesting autograd engine");
    test_dtype::<f32, _>(autograd::t0);
    test_dtype::<f32, _>(autograd::t1);

    println!("\nTesting optimizers");
    test_dtype::<f32, _>(optimizer::sgd);
    test_dtype::<f32, _>(optimizer::adam);

    println!("\nTesting nn modules");
    test_dtype::<f32, _>(nn::linear);
    test_dtype::<f32, _>(nn::layer_norm);
    test_dtype::<f32, _>(nn::batch_norm);
}
