use crate::runtime::Runtime;

mod dtype;
mod mutex;
#[cfg(feature = "py")]
mod python_bindings;
mod runtime;
mod scalar;
mod shape;
mod tensor;
mod index_map;

pub use dtype::DType;
pub use scalar::Scalar;
pub use shape::IntoShape;
pub use tensor::Tensor;
pub use runtime::BackendConfig;
pub use runtime::CUDAConfig;
pub use runtime::HIPConfig;
pub use runtime::OpenCLConfig;

static RT: mutex::Mutex<Runtime, 1000000000> = mutex::Mutex::new(Runtime::new());

// Load and save test
#[test]
fn t0() {
    let x = Tensor::from([[2, 3], [4, 5]]);
    println!("{x}");
    //assert_eq!(x, [[2, 3], [4, 5]]);
}

// Unary test
#[test]
fn t1() {
    let x = Tensor::from([[2f32, 3.], [4., 5.]]).exp();
    println!("{x}");
    //assert_eq!(x, [[2, 3], [4, 5]]);
}

#[test]
fn t2() {
    //let x = Tensor::randn([2, 2], DType::F32).reshape(256).exp().expand([256, 4]);
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

#[cfg(feature = "rand")]
#[test]
#[should_panic]
fn t3() {
    let x = Tensor::randn([1024, 1024], DType::F32).expand([1024, 1024, 1024]);
    Tensor::realize([&x]).unwrap();
}

#[cfg(feature = "rand")]
#[test]
fn t4() {
    let x = Tensor::uniform([1024, 1024], 0f32..1f32);
    let y = Tensor::uniform([1024, 1024], 0f32..1f32);
    //let z = (x * y).sum(2);
    let z = x.dot(y);
    Tensor::realize([&z]).unwrap();
}

#[test]
fn t5() {
    let x = Tensor::from([[2f32, 3.], [4., 5.]]);
    let y = x.t();
    let z = x.exp();
    //Tensor::plot_dot_graph([&y, &z], "graph1");
    Tensor::realize([&y, &z]).unwrap();
    println!("{y}\n{z}");
}

#[cfg(feature = "rand")]
#[test]
fn t6() {
    //let x = Tensor::from([[2, 3], [4, 5]]).pad_zeros([(1, 3)]);

    let x = Tensor::randn([14, 16], DType::U8);
    let x = x.get((.., 8..-2));
    println!("{x}");
}

#[test]
fn t7() {
    let x = Tensor::from([[2, 3], [4, 5]]);
    //let x = x.pad_zeros([(0, 1)]);
    let x = x.pad_zeros([(4, 3), (1, 2)]);
    //Tensor::plot_dot_graph([], "graph0");
    println!("{x}")
}

#[test]
fn t8() {
    let x = Tensor::ones([2, 3], DType::F32);
    println!("{x}");
}

#[test]
fn t9() {
    let mut x = Tensor::ones([1024, 1024], DType::F32);
    let y = Tensor::ones([1024, 1024], DType::F32);
    for _ in 0..10 {
        x = x.dot(&y);
    }
    println!("{x}");
}

#[test]
fn t_10() {
    let x = Tensor::eye(8, DType::I32);
    println!("{x}");
}

#[test]
fn t_11() {
    let x = Tensor::from([[2, 3, 1], [3, 4, 1]]);
    let y = Tensor::from([[2, 3], [2, 1], [4, 1]]);
    //let x = x.dot(y);
    let x = x.reshape([2, 1, 3]) * y.t().reshape([1, 2, 3]);
    let x = x.sum(2);
    println!("{x}");
}

#[test]
fn t_12() {
    let mut x = Tensor::from([2, 3, 1]);
    let w = Tensor::from([[2, 3, 2], [2, 1, 1], [4, 1, 4]]);
    let b = Tensor::from([2, 3, 5]);
    for _ in 0..10 {
        x = x.dot(&w) + &b;
        Tensor::realize([&x]).unwrap();
    }
    println!("{x}");
}

#[test]
fn t_14() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    x = x.repeat([2, 4, 1]);
    println!("{x}");
}
