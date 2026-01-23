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
#![forbid(invalid_reference_casting)]
#![deny(clippy::cast_possible_truncation)]
#![deny(clippy::cast_lossless)]
#![deny(clippy::cast_precision_loss)]
#![deny(clippy::all)]
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
#![deny(clippy::self_named_module_files)]
#![allow(clippy::self_named_module_files)]
#![allow(clippy::unseparated_literal_suffix)]
#![deny(clippy::separated_literal_suffix)]
#![allow(clippy::unnecessary_cast)]
#![allow(trivial_numeric_casts)] // why not?, will by optimizad by the compiler anyway
#![allow(clippy::collapsible_if)]
// Deny later
#![allow(clippy::single_char_lifetime_names)]
#![forbid(clippy::cargo)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::fallible_impl_from)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_wrap)]

use crate::runtime::Runtime;

mod backend;
mod cache;
mod dtype;
mod error;
mod graph;
mod kernel;
mod mutex;
mod optimizer;
#[cfg(feature = "py")]
mod py_bindings;
mod rng;
mod runtime;
mod scalar;
mod schedule;
mod shape;
mod slab;
mod tensor;
// Constant initializable hasher because apparently noone invented that yet...
mod autograd;
mod chasher;
mod module;
mod prog_bar;
mod realize;
mod view;

type Set<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::chasher::CHasher>>;
type Map<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<crate::chasher::CHasher>>;

pub use autograd::GradientTape;
pub use dtype::DType;
pub use error::ZyxError;
pub use module::Module;
pub use scalar::{Float, Scalar};
pub use shape::IntoShape;
pub use tensor::Tensor;

// Works, but rust does not call drop on this when exiting the program, which causes all sorts of problems ...
static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());

/// Bitflags for debugging
#[cfg_attr(feature = "py", pyo3::pyclass)]
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

    /// Is kernel launch and memory movement debugging enabled?
    #[must_use]
    pub const fn kmd(&self) -> bool {
        (self.0 >> 5) % 2 == 1
    }
}

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const ORANGE: &str = "\x1b[38;5;208m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

// Execution timer
/*static ET: mutex::Mutex<std::collections::BTreeMap<String, (u128, u128)>> =
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
