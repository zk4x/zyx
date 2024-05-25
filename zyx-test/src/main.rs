#![allow(dead_code)]

use zyx::{Device, DType, Scalar, Tensor};

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

fn test(test_fn: impl Fn(DType)) {
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64, DType::U8, DType::I8, DType::I16, DType::I32, DType::I64] {
        test_dtype(test_fn.clone(), dtype);
    }
}

fn test_dtype<F: Fn(DType)>(test_fn: F, dtype: DType) {
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
            test_fn(dtype);
            let elapsed = begin.elapsed().as_nanos();
            println!("OK, time taken: {:.3} ms", elapsed as f32 / 1000000.);
        } else {
            println!("Device {device} is not available.");
        }
    }
}

fn main() {
    println!("\nTesting tensor initialization");
    test(initialization::randn);
    test(initialization::uniform);
    test(initialization::kaiming_uniform);
    test(initialization::zeros);
    test(initialization::ones);
    test(initialization::full);
    test(initialization::eye);

    println!("\nTesting unary ops");
    test(unary::neg);
    test(unary::relu);
    test(unary::sin);
    test(unary::cos);
    test(unary::ln);
    test(unary::exp);
    test(unary::tanh);
    test(unary::sqrt);

    println!("\nTesting binary ops");
    test(binary::add);
    test(binary::sub);
    test(binary::mul);
    test(binary::div);
    test(binary::pow);
    test(binary::cmplt);

    println!("\nTesting ternary ops");
    test(ternary::_where);

    println!("\nTesting movement ops");
    // TODO more detailed movement ops tests
    test(movement::reshape);
    test(movement::expand);
    test(movement::permute);
    test(movement::pad);

    println!("\nTesting reduce ops");
    test(reduce::sum);
    test(reduce::max);

    println!("\nTesting custom ops");
    test(custom::small_tiled_dot);

    println!("\nTesting combinations of ops");
    // TODO more detailed combinations of ops tests
    test(combination::t0);
    test(combination::t1);
    test(combination::dot);
    test(combination::cat);
    test(combination::split);

    println!("\nTesting autograd engine");
    test(autograd::t0);
    test(autograd::t1);

    println!("\nTesting optimizers");
    test(optimizer::sgd);
    test(optimizer::adam);

    println!("\nTesting nn modules");
    test(nn::linear);
    test(nn::layer_norm);
    test(nn::batch_norm);
}
