mod impls;

use alloc::vec::Vec;
use zyx_core::dtype::DType;

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
    id: u8,
}

struct MemRef {
    // which id in mems
    id: u8,
    // 0 for global, 1 for local, 2 for register
    scope: u8,
}

struct Mem {
    rc: u32,
    len: u8,
    dtype: DType,
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
    Binary(MemRef, MemRef, UOp),
    Where(MemRef, MemRef, MemRef),
}

/// Virtual kernel representation, this gets send to device for compilation
pub struct VirtKernel {
    indices: Vec<u8>, // reference counts for indices
    // variables with 3 levels of caching, 0 is global, 1 is local, 2 is register
    mems: [Vec<Mem>; 3],
    instructions: Vec<Instruction>,
}
