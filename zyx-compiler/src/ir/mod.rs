mod elementwise;
mod reduce;
mod work_size;

use crate::ir::work_size::calculate_work_sizes;
use crate::{ASTBOp, ASTOp, ASTUOp, AST};
use alloc::{string::String, vec::Vec, collections::BTreeMap};
use core::fmt::{Display, Formatter};
use zyx_core::{
    dtype::DType,
    view::Index,
};

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

/// Intermediate representation for compilers
pub struct IRKernel {
    /// Global work size of the kernel
    pub global_work_size: Vec<usize>,
    /// Local work size of the kernel
    pub local_work_size: Vec<usize>,
    /// dtype, read_only, allocation size
    pub kernel_args: Vec<(DType, bool, Option<usize>)>,
    /// Vec of all instructions
    pub ops: Vec<IROp>,
}

pub enum IROp {
    Loop,
    EndLoop,
}

// Single most important function and one of the most difficult
// functions to write. All of this is cached, so take your time to optimize
// these kernels.
pub(super) fn ast_to_ir(ast: &AST, max_local_work_size: usize, max_local_memory_size: usize, max_num_registers: usize) -> IR {
    // Compile ops

    IR {
        global_work_size,
        local_work_size,
        kernel_args,
        ops,
    }
}
