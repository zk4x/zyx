//! Zyx compiler to IR

#![no_std]
#![forbid(unsafe_code)]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
//#![forbid(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
//#![forbid(rustdoc::missing_doc_code_examples)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]

use zyx_core::error::ZyxError;

mod tiled;
mod looped;
mod ir;

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::{collections::BTreeMap, vec::Vec};
use zyx_core::dtype::DType;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;

/// Compiled backend that holds compiler, buffers and programs
pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<Id, C::Buffer>,
}

impl<C: Compiler> CompiledBackend<C> {
    /// Initialize new compiled backend using provided compiler
    pub fn new(compiler: C) -> Self {
        Self {
            compiler,
            buffers: BTreeMap::new(),
        }
    }
}

pub struct IRKernel {}

/// Hardware information needed for applying optimizations
#[derive(Debug)]
pub struct HWInfo {
    /// Biggest kernel dimensions
    pub max_work_item_sizes: Vec<usize>,
    /// Maximum local work size threads
    pub max_work_group_size: usize,
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Is half supported?
    pub f16_support: bool,
    /// Is double supported?
    pub f64_support: bool,
    /// Is fused multiply add supported?
    pub fmadd: bool,
    /// Global (VRAM, RAM) memory size in bytes
    pub global_mem_size: usize,
    /// Maximum memory allocation for single buffer in bytes
    pub max_mem_alloc: usize,
    /// Alignment for data types in bytes
    pub mem_align: usize,
    /// Page size (base address alignment) in bytes
    pub page_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
}

/// Implement this trait for compiled backends
pub trait Compiler {
    /// Buffer holds actual values in memory
    type Buffer;
    /// Program is kernel executable on the device, can be compiled at runtime
    type Program;
    /// Hardware information
    fn hardware_info(&mut self) -> HWInfo;
    /// Allocate space for new buffer
    fn allocate_mem(&mut self, length: usize, dtype: DType) -> Result<Self::Buffer, ZyxError>;
    /// Store iter into existing buffer
    // TODO perhaps this should be async store
    fn store_mem<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        iter: impl IntoIterator<Item = T>,
    ) -> Result<(), ZyxError>;
    /// Load buffer into vec
    fn load_mem<T: Scalar>(&mut self, buffer: &Self::Buffer, length: usize) -> Result<Vec<T>, ZyxError>;
    /// Drop Buffer
    fn deallocate_mem(&mut self, buffer: &mut Self::Buffer) -> Result<(), ZyxError>;
    /// Compile ast into program
    fn compile_program(&mut self, ir: &IRKernel) -> Result<Self::Program, ZyxError>;
    /// Launch program with args
    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &[&Self::Buffer],
        flop: usize,
        bytes: usize,
    ) -> Result<(), ZyxError>;
    /// Drop Program
    fn drop_program(&mut self, program: &mut Self::Program) -> Result<(), ZyxError>;
}
