#![doc = include_str!("../README.md")]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![deny(missing_docs)]
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
mod kernel_cache;
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
mod chasher;
mod bar;
mod autograd;
mod static_graph;
mod optimizer;

pub(crate) type Set<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::chasher::CHasher>>;
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<crate::chasher::CHasher>>;

pub use dtype::DType;
pub use runtime::ZyxError;
pub use scalar::{Float, Scalar};
use shape::Dimension;
pub use shape::IntoShape;
pub use tensor::Tensor;
pub use autograd::GradientTape;

// Works, but rust does not call drop on this when exiting the program, which causes all sorts of problems ...
static RT: mutex::Mutex<Runtime, 1_000_000_000> = mutex::Mutex::new(Runtime::new());
//static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());

/// Bitflags for debugging
#[derive(Debug, Clone, Copy)]
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
            let size = tensor.numel() * dtype.byte_size() as Dimension;
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

// Execution timer
/*static ET: mutex::Mutex<std::collections::BTreeMap<String, (u128, u128)>, 1_000_000_000> =
    mutex::Mutex::new(std::collections::BTreeMap::new());

pub(crate) struct Timer {
    name: String,
    begin: std::time::Instant,
}

impl Timer {
    pub(crate) fn new(name: &str) -> Timer {
        let name: String = name.into();
        ET.lock().entry(name.clone()).or_insert((0, 0));
        Timer { name, begin: std::time::Instant::now() }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let mut lock = ET.lock();
        let x = lock.get_mut(&self.name).unwrap();
        x.0 += self.begin.elapsed().as_micros();
        x.1 += 1;
        //println!("Timer took {}us", self.begin.elapsed().as_micros());
    }
}*/

/*#[test]
fn t0() -> Result<(), ZyxError> {
    let x = Tensor::rand([4, 2, 3], DType::F32).unwrap();
    let y = x.exp2();
    //let y = x.sum([-1]).unwrap();
    println!("{y}");
    Ok(())
}*/

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
fn multithreading() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    let handle = std::thread::spawn(|| {
        let y = Tensor::from([2, 3]);
        println!("{y}");
    });
    println!("{x}");
    handle.join().unwrap();
}*/

/*#[test]
fn binary_cross_dependency1() -> Result<(), ZyxError> {

    let x = Tensor::from([4, 5, 1]);

    let y = Tensor::from([4, 1, 2]);

    let x1 = x.sum([])?;
    let x2 = x1.expand([3, 3])?;

    let y1 = y + &x1;
    let y2 = y1.sum([])?;
    //let y3 = y2.expand([3, 3])?;

    let x3 = x2 + &y2;

    Tensor::realize([&x1, &y2, &x3])?;

    Ok(())
}*/
