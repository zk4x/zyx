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

mod ast;
mod ir;

pub use ir::{UOp, BOp};

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::{collections::BTreeMap, vec::Vec};
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;
use crate::ast::AST;
pub use crate::ir::IRKernel;
pub use crate::ir::IROp;

/// Compiled backend that holds compiler, buffers and programs
pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<Id, C::Buffer>,
    programs: BTreeMap<AST, C::Program>,
}

/// Implement this trait for compiled backends
pub trait Compiler {
    /// Buffer holds actual values in memory
    type Buffer;
    /// Program is kernel executable on the device, can be compiled at runtime
    type Program;
    /// Store iter into buffer
    fn store<T: Scalar>(
        &mut self,
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self::Buffer, ZyxError>;
    /// Load buffer into vec
    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Result<Vec<T>, ZyxError>;
    /// Drop Buffer
    fn drop_buffer(&mut self, buffer: &mut Self::Buffer) -> Result<(), ZyxError>;
    /// Drop Program
    fn drop_program(&mut self, program: &mut Self::Program) -> Result<(), ZyxError>;
    /// Launch program with args
    fn launch(
        &mut self,
        program: &Self::Program,
        args: &[&Self::Buffer],
        flop: usize,
        bytes: usize,
    ) -> Result<Self::Buffer, ZyxError>;
    /// Compile ast into program
    fn compile(&mut self, ir: &IRKernel) -> Result<Self::Program, ZyxError>;
}
