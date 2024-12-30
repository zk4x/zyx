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
#![forbid(trivial_casts)]
#![forbid(trivial_numeric_casts)]
#![forbid(cenum_impl_drop_cast)]
#![forbid(invalid_reference_casting)]
#![deny(clippy::cast_possible_truncation)]
#![deny(clippy::cast_lossless)]
#![deny(clippy::cast_precision_loss)]
#![deny(clippy::all)]
#![deny(clippy::cast_possible_wrap)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::cast_sign_loss)]
#![deny(clippy::ptr_cast_constness)]
#![deny(clippy::pedantic)]
#![deny(clippy::fn_to_numeric_cast_any)]
#![forbid(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::as_ptr_cast_mut)]
#![deny(clippy::missing_const_for_fn)]
#![deny(clippy::nursery)]
#![allow(clippy::use_self)]
#![allow(clippy::single_call_fn)]
#![allow(clippy::similar_names)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::multiple_inherent_impl)]
//#![deny(clippy::restriction)]
#![deny(clippy::mod_module_files)]
#![allow(clippy::self_named_module_files)]
#![allow(clippy::unseparated_literal_suffix)]
#![deny(clippy::separated_literal_suffix)]
// Deny later
#![allow(clippy::single_char_lifetime_names)]
//#![deny(clippy::cargo)]

use crate::runtime::Runtime;
use std::{fs::File, path::Path};

mod backend;
mod dtype;
mod error;
mod graph;
mod ir;
mod kernel;
mod mutex;
mod node;
mod optimizer;
#[cfg(feature = "py")]
mod py_bindings;
mod rng;
mod runtime;
mod scalar;
mod scheduler;
mod shape;
mod slab;
mod tensor;
mod view;

// Constant initializable hasher because apparently noone invented that yet...
//mod chasher;

pub use dtype::DType;
pub use runtime::ZyxError;
pub use scalar::{Float, Scalar};
pub use shape::IntoShape;
pub use tensor::Tensor;

// Works, but rust does not call drop on this when exiting the program, which causes all sorts of problems ...
static RT: mutex::Mutex<Runtime, 1_000_000_000> = mutex::Mutex::new(Runtime::new());
//static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());

/// Bitflags for debugging
#[derive(Clone, Copy)]
pub struct DebugMask(u32);

impl DebugMask {
    /// Is device debugging enabled?
    #[must_use]
    pub const fn dev(&self) -> bool {
        self.0 % 2 == 1
    }

    /// Is performance debugging enabled?
    #[must_use]
    pub const fn perf(&self) -> bool {
        (self.0 >> 1) % 2 == 1
    }

    /// Is scheduler debugging enabled?
    #[must_use]
    pub const fn sched(&self) -> bool {
        (self.0 >> 2) % 2 == 1
    }

    /// Is debugging of IR enabled?
    #[must_use]
    pub const fn ir(&self) -> bool {
        (self.0 >> 3) % 2 == 1
    }

    /// Is assembly debugging enabled?
    #[must_use]
    pub const fn asm(&self) -> bool {
        (self.0 >> 4) % 2 == 1
    }
}

/// Save tensors or modules
pub trait TensorSave {
    /// Save tensors or modules
    ///
    /// # Errors
    ///
    /// Errors if tensors failed to realize or failed to save to disk.
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
            write!(header, "\"shape\":{st_shape},").unwrap();
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

/// Execution timer
/*static ET: mutex::Mutex<std::collections::BTreeMap<String, u128>, 1_000_000_000> =
    mutex::Mutex::new(std::collections::BTreeMap::new());

pub(crate) struct Timer {
    name: String,
    begin: std::time::Instant,
}

impl Timer {
    pub(crate) fn new(name: &str) -> Timer {
        let name: String = name.into();
        ET.lock().entry(name.clone()).or_insert(0);
        Timer { name, begin: std::time::Instant::now() }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        *ET.lock().get_mut(&self.name).unwrap() += self.begin.elapsed().as_micros();
        //println!("Timer took {}us", self.begin.elapsed().as_micros());
    }
}*/

/*#[test]
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
fn t1() -> Result<(), ZyxError> {
    let x = Tensor::rand([4, 2, 3], DType::F32).unwrap();
    let y = x.exp2();
    //let y = x.sum([-1]).unwrap();
    println!("{y}");
    Ok(())
}

/*#[test]
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
}*/

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

#[test]
fn causal_self_attention() -> Result<(), ZyxError> {
    let dtype = DType::F32;
    let n_embd = 4;
    let n_head = 4;
    let c_attn_weight = Tensor::from([
        [3, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [1, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [3, 1, 5, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [3, 1, 2, 3, 1, 2, 5, 8, 2, 3, 1, 3],
    ])
    .t()
    .cast(dtype);
    let c_proj_weight =
        Tensor::from([[5, 4, 2, 1], [9, 1, 5, 2], [7, 5, 6, 2], [6, 2, 7, 1]]).cast(dtype);

    let x = Tensor::from([[[1, 0, 4, 2], [2, 5, 0, 1], [0, 8, 1, 0], [5, 1, 0, 0]]]).cast(dtype);

    let [b, t, c] = x.shape()[..] else {
        return Err(ZyxError::ShapeError(
            "x must have exactly 3 dims, b, t, c".into(),
        ));
    };
    let mut splits = x.dot(c_attn_weight.t())?.split([n_embd, n_embd, n_embd], 2)?;
    let mut v = splits.pop().unwrap();
    let mut k = splits.pop().unwrap();
    let mut q = splits.pop().unwrap();

    k = k.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    q = q.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    v = v.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;

    let mut att = q.dot(k.t())? * ((1f64 / (*k.shape().last().unwrap() as f64).sqrt()) as f32);

    assert_eq!(
        att,
        [[
            [
                [147f32, 168., 189., 126.],
                [98., 112., 126., 84.],
                [77., 88., 99., 66.],
                [112., 128., 144., 96.]
            ],
            [
                [98., 112., 126., 84.],
                [112., 128., 144., 96.],
                [126., 144., 162., 108.],
                [84., 96., 108., 72.]
            ],
            [
                [910., 1040., 1170., 780.],
                [560., 640., 720., 480.],
                [735., 840., 945., 630.],
                [420., 480., 540., 360.]
            ],
            [
                [756., 756., 756., 504.],
                [864., 864., 864., 576.],
                [972., 972., 972., 648.],
                [648., 648., 648., 432.]
            ]
        ]]
    );

    att = att.softmax([1])?;
    let mut y = att.dot(v)?;

    assert_eq!(
        y,
        [[
            [[18f32], [18.], [18.], [18.]],
            [[27.], [27.], [27.], [27.]],
            [[9.], [9.], [9.], [9.]],
            [[24.], [24.], [24.], [24.]]
        ]]
    );

    y = y.transpose(1, 2)?.reshape([b, t, c])?;
    y = y.dot(c_proj_weight.t())?;

    assert_eq!(
        y,
        [[
            [18f32, 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.]
        ]]
    );

    Ok(())
}
