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

use ast::Kernel;
pub use ir::{Op, IR, UOp, BOp};

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::{collections::BTreeMap, vec::Vec};
use zyx_core::axes::Axes;
use zyx_core::dtype::DType;
use zyx_core::scalar::Scalar;
use zyx_core::shape::Shape;
use zyx_core::tensor::Id;
use zyx_core::view::View;

/// Compiled backend that holds compiler, buffers and programs
pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    kernels: BTreeMap<Id, Kernel>,
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
    fn compile(&mut self, ir: &IR) -> Result<Self::Program, ZyxError>;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ASTUOp {
    Cast(DType),
    Neg,
    ReLU,
    Sin,
    Cos,
    Exp,
    Ln,
    Tanh,
    Sqrt,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ASTBOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ASTROp {
    Sum,
    Max
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ASTOp {
    Leaf(u8),
    Unary(u8, ASTUOp),
    Binary(u8, u8, ASTBOp),
    #[allow(dead_code)]
    Where(u8, u8, u8),
    Reduce(u8, ASTROp),
}

/// Abstract syntax tree that can be compiled into program.
/// Consists of kernel arguments, elementwise ops, optional reduce op
/// and elementwise ops after reduce.
/// This struct is immutable.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AST {
    /// AST argument views
    pub arg_views: Vec<View>,
    /// AST argument dtypes
    pub arg_dtypes: Vec<DType>,
    /// AST ops
    pub ops: Vec<ASTOp>,
    /// Shape of the result, this is before any reduce ops
    pub shape: Shape,
    /// DType of the result
    pub dtype: DType,
    /// Reduce axes, if any
    pub reduce_axes: Option<Axes>,
    /// DType of accumulated elements, if any
    pub reduce_dtype: Option<DType>,
}
