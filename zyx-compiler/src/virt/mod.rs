mod impls;

use alloc::vec::Vec;
use zyx_core::dtype::DType;
use zyx_core::node::Constant;
use zyx_core::view::View;

enum Either<L, R> {
    Left(L),
    Right(R),
}

/// Unary op
pub enum UOp {
    Noop, // Just assign
    Cast(DType),
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Tanh,
    Sqrt,
}

/// Binary op
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Max,
}

struct IndexRef {
    // to which id in indices this points
    id: usize,
}

struct MemRef {
    // which id in mems
    id: Either<usize, Constant>,
}

struct Mem {
    rc: u32,
    dtype: DType,
    // variables with 3 levels of caching, 0 is global, 1 is local, 2 is register
    scope: u8,
    view: View,
}

enum Instruction {
    // barrier with level of blocking, 0 is global, 1 is local
    Barrier(u8),
    Loop {
        // Can be either constant or index
        begin: Either<usize, IndexRef>,
        iters: usize,
        step: usize,
    },
    EndLoop,
    Unary(MemRef, UOp),
    Binary(MemRef, MemRef, BOp),
}

/// Virtual kernel representation, this gets send to device for compilation
pub struct VirtKernel {
    indices: Vec<u8>, // reference counts for indices
    mems: Vec<Mem>,
    instructions: Vec<Instruction>,
}
