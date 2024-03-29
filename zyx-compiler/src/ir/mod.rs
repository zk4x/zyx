use alloc::vec::Vec;
use zyx_core::dtype::DType;
use crate::ast::ASTOp;

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
pub(crate) fn ast_to_ir(ops: &[ASTOp], max_local_work_size: usize, max_local_memory_size: usize, max_num_registers: usize) -> IRKernel {
    // Compile ops
    for op in ops {
        match op {
            _ => {}
        }
    }

    IRKernel {
        global_work_size: Vec::new(),
        local_work_size: Vec::new(),
        kernel_args: Vec::new(),
        ops: Vec::new(),
    }
}
