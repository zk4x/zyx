// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#![doc = include_str!("../README.md")]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![warn(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]
#![forbid(invalid_reference_casting)]
/*#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::fn_to_numeric_cast_any)]
#![forbid(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::as_ptr_cast_mut)]
#![deny(clippy::missing_const_for_fn)]
#![deny(clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
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
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnested_or_patterns)]
//#![forbid(clippy::cargo)] // wgpu does not pass this
#![allow(clippy::option_if_let_else)]
#![allow(clippy::fallible_impl_from)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::into_iter_on_ref)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::needless_return)]
#![allow(clippy::as_conversions)]*/

use crate::runtime::Runtime;

mod aot;
mod backend;
mod codegen;
mod dtype;
mod error;
mod graph;
mod hashers;
pub mod kernel;
mod module;
mod mutex;
mod progress;
#[cfg(feature = "py")]
pub mod py_bindings;
mod rng;
mod runtime;
mod scalar;
mod shape;
mod slab;
mod symbolic;
mod tape;
mod tensor;
mod types;
#[cfg(feature = "viz")]
mod viz;

type Set<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::hashers::FHasher>>;
type Map<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<crate::hashers::FHasher>>;

pub use dtype::{DType, QDType};
pub use error::ZyxError;
pub use module::{GGUFMetadataValue, Module};
pub use scalar::{Float, Scalar, bf16, f8e4m3, f8e5m2, f16};
pub use tape::{FrozenTape, Tape};
pub use tensor::ReduceOp;
pub use tensor::{Dev, Tensor};

// Works, but rust does not call drop on this when exiting the program, which causes all sorts of problems ...
static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());

/// Bitflags for debugging
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
#[derive(Debug, Clone, Copy)]
pub struct DebugMask(u32);

impl DebugMask {
    /// Create a new [`DebugMask`]
    #[must_use]
    pub const fn new(x: u32) -> Self {
        Self(x)
    }

    /// Is device debugging enabled?
    #[must_use]
    pub const fn dev(&self) -> bool {
        self.0 % 2 == 1
    }

    /// Is egraph printing enabled?
    #[must_use]
    pub const fn egraph(&self) -> bool {
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

    /// Is kernel launch debugging enabled?
    #[must_use]
    pub const fn launch(&self) -> bool {
        (self.0 >> 5) % 2 == 1
    }

    /// Is memory allocation/deallocation debugging enabled?
    #[must_use]
    pub const fn memory(&self) -> bool {
        (self.0 >> 6) % 2 == 1
    }

    /// Is kernel compilation debugging enabled?
    #[must_use]
    pub const fn compile(&self) -> bool {
        (self.0 >> 7) % 2 == 1
    }

    /// Is the no-search debug path enabled (skip seed prep, epilogue and
    /// beam search; compile each seed with linearize + DCE)?
    #[must_use]
    pub const fn no_search(&self) -> bool {
        (self.0 >> 8) % 2 == 1
    }
}

/// Format launch perf (flops, global bytes read/written, nanos) as a
/// human-readable line: time, FLOP/s, read B/s, write B/s.
#[allow(unused)]
#[allow(clippy::similar_names)]
pub fn get_perf(flop: u64, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    if nanos == u64::MAX {
        return format!("INF time taken");
    }
    const fn value_unit(x: u64) -> (u64, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    //let (f, f_u) = value_unit(flop);
    //let (br, br_u) = value_unit(bytes_read);
    //let (bw, bw_u) = value_unit(bytes_written);
    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit((flop as u128 * 1_000_000 / nanos as u128 * 1000) as u64);
    let (brs, br_us) = value_unit((bytes_read as u128 * 1_000_000_000 / nanos as u128) as u64);
    let (bws, bw_us) = value_unit((bytes_written as u128 * 1_000_000_000 / nanos as u128) as u64);

    /*format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w, {}.{:02} {f_u}FLOP, {}.{:02} {br_u}B r, {}.{:02} {bw_u}B w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
        f / 100,
        f % 100,
        br / 100,
        br % 100,
        bw / 100,
        bw % 100,
    )*/

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
    )
}

static DEBUG_MASK: mutex::Mutex<Option<DebugMask>> = mutex::Mutex::new(None);

/// Returns the global debug mask, loading `ZYX_DEBUG` from the
/// environment on first access. Lives outside [`Runtime`] so any
/// thread can read it without locking the runtime.
pub(crate) fn debug_mask() -> DebugMask {
    let mut guard = DEBUG_MASK.lock();
    if let Some(mask) = *guard {
        mask
    } else {
        let mask =
            std::env::var("ZYX_DEBUG").ok().and_then(|x| x.parse::<u32>().ok()).map(DebugMask).unwrap_or(DebugMask::new(0));
        *guard = Some(mask);
        mask
    }
}

/// Sets the global debug mask (used by [`tensor::Tensor::with_debug`]).
pub(crate) fn set_debug_mask(mask: DebugMask) {
    *DEBUG_MASK.lock() = Some(mask);
}

const BOLD: &str = "\x1b[1m";
const GREY: &str = "\x1b[38;5;252m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const ORANGE: &str = "\x1b[38;5;208m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

// Execution timer
#[cfg(feature = "time")]
pub(crate) static ET: crate::mutex::Mutex<std::collections::BTreeMap<String, (u128, u128)>> =
    crate::mutex::Mutex::new(std::collections::BTreeMap::new());

#[cfg(feature = "time")]
pub(crate) struct Timer {
    name: String,
    begin: std::time::Instant,
}

#[cfg(feature = "time")]
impl Timer {
    pub(crate) fn new(name: &str) -> Timer {
        let name: String = name.into();
        ET.lock().entry(name.clone()).or_insert((0, 0));
        Timer { name, begin: std::time::Instant::now() }
    }
}

#[cfg(feature = "time")]
impl Drop for Timer {
    fn drop(&mut self) {
        let mut lock = ET.lock();
        let x = lock.get_mut(&self.name).unwrap();
        x.0 += self.begin.elapsed().as_micros();
        x.1 += 1;
    }
}
