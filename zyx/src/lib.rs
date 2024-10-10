#![doc = include_str!("../README.md")]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![forbid(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]

use std::{fs::File, path::Path};

use crate::runtime::Runtime;

mod dtype;
mod index_map;
mod mutex;
#[cfg(feature = "py")]
mod py_bindings;
mod runtime;
mod scalar;
mod shape;
mod tensor;

pub use dtype::DType;
pub use runtime::DeviceConfig;
pub use runtime::ZyxError;
pub use scalar::{Scalar, Float};
pub use shape::IntoShape;
pub use tensor::Tensor;

// Works, but rust does not call drop on this when exiting the program, which causes all sorts of problems ...
static RT: mutex::Mutex<Runtime, 1000000000> = mutex::Mutex::new(Runtime::new());
//static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());

/// Save tensors or modules
pub trait TensorSave {
    /// Save tensors or modules
    fn save(self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
}

impl<'a, I: IntoIterator<Item = &'a Tensor>> TensorSave for I {
    fn save(self, path: impl AsRef<Path>) -> Result<(), ZyxError> {
        use std::fmt::Write;
        use std::io::Write as IOWrite;
        let mut f = File::create(path)?;
        let mut header = String::from("{");
        let mut begin = 0;
        let tensors: Vec<&Tensor> = self.into_iter().collect();
        for tensor in &tensors {
            let dtype = tensor.dtype();
            //if let Some(label) = tensor.label() {
            //write!(header, "\"{label}\":{{").unwrap();
            //} else {
            write!(header, "\"{}\":{{", tensor.id()).unwrap();
            //}
            write!(header, "\"dtype\":\"{}\",", dtype.safetensors()).unwrap();
            let mut st_shape = format!("{:?}", tensor.shape());
            st_shape.retain(|c| !c.is_whitespace());
            write!(header, "\"shape\":{},", st_shape).unwrap();
            let size = tensor.numel() * dtype.byte_size();
            write!(header, "\"data_offsets\":[{},{}]", begin, begin + size).unwrap();
            begin += size;
            write!(header, "}},").unwrap();
        }
        header.pop();
        write!(header, "}}").unwrap();
        let header_bytes = header.as_bytes();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(header_bytes)?;
        for tensor in tensors {
            f.write_all(&tensor.to_le_bytes()?)?;
        }
        Ok(())
    }
}

/*
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
    for _ in 0..20 {
        let z = x.dot(&y);
        Tensor::realize([&z]).unwrap();
        drop(z);
        //Tensor::plot_graph([], &format!("graph{i}"));
    }
    //Tensor::plot_graph([], "graph0");
    //Tensor::realize([&z]).unwrap();
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
    let x = x.dot(y);
    //let x = x.reshape([2, 1, 3]) * y.t().reshape([1, 2, 3]);
    //let x = x.sum(2);
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

#[test]
fn t_15() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    for _ in 0..10 {
        x = &x + &x;
        println!("{x}");
        //Tensor::plot_graph([], &format!("graph{i}"));
        Tensor::realize([&x]).unwrap();
    }
    println!("{x}");
}

#[test]
fn t_16() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    let y = Tensor::from([[5, 6, 9], [4, 2, 0]]);
    let _z = x.exp2() + &y;
    x = -x * &y;
    Tensor::plot_graph([], "graph0");
    Tensor::realize([&x]).unwrap();
    Tensor::plot_graph([], "graph1");
}

#[test]
fn t_17() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    println!("{x}");
    x = x.sum([]);
    println!("{x}");
}

#[test]
fn t_18() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    let y = Tensor::from([[2, 3], [1, 2], [4, 1]]);
    x = x.dot(y).pad_zeros([(2, 1)]);
    println!("{x}");
}
*/

/*#[test]
fn t_15() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    for _ in 0..10 {
        x = &x + &x;
        //println!("{x}");
        //Tensor::plot_graph([], &format!("graph{i}"));
        Tensor::realize([&x]).unwrap();
    }
    println!("{x}");
}

#[test]
fn t_12() {
    let mut x = Tensor::from([2, 3, 1]);
    let w = Tensor::from([[2, 3, 2], [2, 1, 1], [4, 1, 4]]);
    let b = Tensor::from([2, 3, 5]);
    for _ in 0..10 {
        x = x.dot(&w) + &b;
        //Tensor::realize([&x]).unwrap();
    }
    println!("{x}");
}*/

/*#[test]
fn t1() {
    use crate::DType;
    let x = Tensor::from([0f32, 5., 1.]);
    let y = Tensor::rand([3, 5], DType::F32);
    let a = x.dot(y);
    let x = Tensor::from([0f32, 5., 1.]);
    let y = Tensor::rand([3, 5], DType::F32);
    let b = x.dot(y);
    println!("{a}, {b}");
}*/

#[test]
fn t2() {
    let x = Tensor::from([4, 2, 3]);
    let y = Tensor::from([4, 2, 3]);
    let a = x + y;
    println!("{a}");
    drop(a);
    let x = Tensor::from([4, 2, 3]);
    let y = Tensor::from([4, 2, 3]);
    let b = x + y;
    println!("{b}");
}

#[test]
fn t3() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    let tensors = x.split([2, 1], 1).unwrap();
    for t in tensors {
        println!("{t}");
    }
}

#[cfg(feature = "rand")]
#[test]
fn t4() {
    //let x = Tensor::uniform([16, 8], 0f32..1f32).unwrap();
    //let y = Tensor::uniform([8, 8], 0f32..1f32).unwrap();
    let x = Tensor::rand([1024, 1024], DType::F32).unwrap();
    let y = Tensor::rand([1024, 1024], DType::F32).unwrap();
    for _ in 0..20 {
        let z = x.dot(&y).unwrap();
        //Tensor::plot_graph([], "graph0");
        Tensor::realize([&z]).unwrap();
        //Tensor::plot_graph([], &format!("graph0"));
        //println!("{z}");
        drop(z);
        //Tensor::plot_graph([], "graph1");
        //Tensor::plot_graph([], &format!("graph"));
    }
    //Tensor::plot_graph([], "graph0");
    //Tensor::realize([&z]).unwrap();
}

#[test]
fn t5() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    println!("{}", x.get((.., 2..3)).unwrap());
}

/*#[test]
fn t6() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    let handle = std::thread::spawn(|| {
        let y = Tensor::from([2, 3]);
        println!("{y}");
    });
    println!("{x}");
    handle.join().unwrap();
}*/

/*#[test]
fn t7() -> Result<(), ZyxError> {
    use std::collections::HashMap;
    let m: HashMap<String, Tensor> = Tensor::load_gguf("mistral_7b_Q4.gguf")?;
    Ok(())
}*/
