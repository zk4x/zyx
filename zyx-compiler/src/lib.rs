use zyx_core::error::ZyxError;

mod impls;
mod ir;

use impls::Kernel;
pub use ir::{IR, Op};

extern crate alloc;
extern crate core;

use alloc::collections::BTreeMap;
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

/// Op executable on device with compiled backend
/// usize are all IDs into ops, leafs have IDs into args
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ASTOp {
    // We are not gonna have so gigantic kernels with more than 256 arguments or ops.
    // OpenCL does not support more than 255 args.
    /// Leaf (holds data, id to kernel arg)
    Leaf(u8),
    // TODO uniform generators should also take shape into consideration
    // and repeat the same random number if this shape is expanded.
    // Shaped uniform generator of numbers from 0. to 1.
    //Uniform(View, DType),
    /// Cast into dtype unary op
    Cast(u8, DType),
    /// Neg unary op
    Neg(u8),
    /// ReLU unary op
    ReLU(u8),
    /// Sin unary op
    Sin(u8),
    /// Cos unary op
    Cos(u8),
    /// Ln unary op
    Ln(u8),
    /// Exp unary op
    Exp(u8),
    /// Tanh unary op
    Tanh(u8),
    /// Sqrt unary op
    Sqrt(u8),
    /// Addition binary op
    Add(u8, u8),
    /// Substitution binary op
    Sub(u8, u8),
    /// Multiplication binary op
    Mul(u8, u8),
    /// Division binary op
    Div(u8, u8),
    /// Exponentiation binary op
    Pow(u8, u8),
    /// Compare less than binary op
    Cmplt(u8, u8),
    /// Where op
    Where(u8, u8, u8),
    /// Sum reduce op
    Sum(u8),
    /// Max reduce op
    Max(u8),
}

/// Abstract syntax tree that can be compiled into program.
/// Consists of kernel arguments, elementwise ops, optional reduce op
/// and elementwise ops after reduce.
/// This struct is immutable.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AST {
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
