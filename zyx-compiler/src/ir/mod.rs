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

/// Variable in IR
pub enum Var {
    Local { id: u8, index: String },
    Register { id: u8, index: Option<String> },
    ConstF32(f32),
    ConstF64(f64),
    ConstI32(i32),
    ConstI64(i64),
}

impl Display for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Var::Local { id, index } => f.write_fmt(format_args!("lmem{id}[{index}]")),
            Var::Register { id, index } => {
                if let Some(index) = index {
                    f.write_fmt(format_args!("rmem{id}[{index}]"))
                } else {
                    f.write_fmt(format_args!("rmem{id}"))
                }
            }
            Var::ConstF32(value) => f.write_fmt(format_args!("{value:.8}f")),
            Var::ConstF64(value) => f.write_fmt(format_args!("{value:.16}")),
            Var::ConstI32(value) => f.write_fmt(format_args!("{value}")),
            Var::ConstI64(value) => f.write_fmt(format_args!("{value}")),
        }
    }
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

/// Op for the compilers
pub enum Op {
    /// Load into res from arg at index
    LoadGlobal { res: Var, arg: u8, index: Index },
    /// Store arg into res at index
    StoreGlobal { res: u8, index: Index, arg: Var },
    /// Declare register variable id with dtype and optionally
    /// with given length (as a vector)
    DeclareVar {
        dtype: DType,
        id: u8,
        len: Option<u8>,
    },
    /// Declare local memory variable with id, dtype and given length
    DeclareLocalVar { id: u8, dtype: DType, len: usize },
    /// Initialize index id with value
    InitIndex { id: u8, value: String },
    /// Declare index
    DeclareIndex { id: u8 },
    /// Set index to value
    SetIndex { id: u8, value: String },
    /// Initialize accumulator, if sum_reduce is true,
    /// then initilize to zero, otherwise initilize to minimum
    /// of dtype.
    InitAccumulator {
        id: u8,
        dtype: DType,
        is_sum_reduce: bool,
        len: Option<u8>,
    },
    /// Unary op
    Unary { res: Var, x: Var, op: UOp },
    /// Binary op
    Binary { res: Var, x: Var, y: Var, op: BOp },
    /// Where, x is condition, y is if true, otherwise z
    Where { res: Var, x: Var, y: Var, z: Var },
    /// Loop in kernel (register/private)
    Loop {
        name: String,
        upper_bound: usize,
        step: usize,
    },
    /// If condition
    IfBlock {
        condition: String,
    },
    /// End of if condition
    EndIf,
    /// End of loop
    EndLoop,
    /// Local memory synchronization
    LocalBarrier,
}

/// Intermediate representation for compilers
pub struct IR {
    pub global_work_size: Vec<usize>,
    pub local_work_size: Vec<usize>,
    pub kernel_args: Vec<(DType, bool)>, // dtype and read_only
    pub ops: Vec<Op>,
    pub res_byte_size: usize,
}

// Single most important function and one of the most difficult
// functions to write. All of this is cached, so take your time to optimize
// these kernels.
pub(super) fn ast_to_ir(ast: &AST, max_local_work_size: usize, max_local_memory_size: usize, max_num_registers: usize) -> IR {
    // Byte size of the resulting buffer
    let res_byte_size = if let Some(reduce_axes) = &ast.reduce_axes {
        ast.shape.clone().reduce(reduce_axes).numel() * ast.dtype.byte_size()
    } else {
        ast.shape.numel() * ast.dtype.byte_size()
    };
    // TODO we should be able to partially separate optimizations from compilation
    // TODO Local memory tiling
    // TODO Register memory tiling
    // TODO Repeat these functions with different parameters (autotuning)
    let (
        arg_views,
        res_shape,
        reduce_dim,
        mut global_work_size,
        mut local_work_size,
        register_work_size,
        tiling_axes,
    ) = calculate_work_sizes(
        &ast.reduce_axes,
        &ast.shape,
        ast.arg_views.clone(),
        max_local_work_size,
        max_num_registers,
    );
    let mut kernel_args = Vec::new();
    for dtype in &ast.arg_dtypes {
        kernel_args.push((*dtype, true));
    }
    // Push result buffer as arg
    kernel_args.push((ast.dtype, false));

    // Compile ops
    let ops = if let Some(reduce_dim) = reduce_dim {
        // Whether it is 1d, 2d or 3d kernel, it can always
        // have expanded buffers, batches (optionally spread across multiple GPUs)
        // and multi-step reduces.

        // TODO tiled kernel currently does not work if local work size in dimension 1 and 2 are different
        if tiling_axes.is_empty() || local_work_size[1] != local_work_size[3] || local_work_size[2] != local_work_size[3] {
            if global_work_size.iter().product::<usize>() == reduce_dim {
                // Full reduce
                // Apply two step reduce
                let mut d = 1;
                while global_work_size[3] % (d * 2) == 0 && d < max_local_work_size {
                    d *= 2;
                }
                global_work_size[2] = d;
                local_work_size[2] = d;
                reduce::two_step_reduce::compile_reduce_kernel(
                    &ast.ops,
                    arg_views,
                    ast.arg_dtypes.clone(),
                    ast.reduce_dtype.unwrap(),
                    reduce_dim,
                    &local_work_size,
                    res_shape,
                )
            } else {
                reduce::compile_reduce_kernel(
                    &ast.ops,
                    arg_views,
                    ast.arg_dtypes.clone(),
                    ast.reduce_dtype.unwrap(),
                    reduce_dim,
                    &local_work_size,
                    res_shape,
                )
            }
        } else {
            reduce::tiled_reduce::compile_reduce_kernel(
                &ast.ops,
                arg_views,
                ast.arg_dtypes.clone(),
                ast.reduce_dtype.unwrap(),
                reduce_dim,
                &global_work_size,
                &local_work_size,
                &register_work_size,
                res_shape,
                tiling_axes,
                max_local_memory_size,
            )
        }
    } else {
        // TODO make sure that local_work_size is always bigger than register_work_size
        elementwise::compile_elementwise_kernel(ast, &local_work_size, arg_views, res_shape)
    };

    IR {
        global_work_size: if reduce_dim.is_some() { global_work_size[..global_work_size.len()-1].to_vec() } else { global_work_size },
        local_work_size: if reduce_dim.is_some() { local_work_size[..local_work_size.len()-1].to_vec() } else { local_work_size },
        kernel_args,
        ops,
        res_byte_size,
    }
}

// Same op can be applied multiple times with different register_index
fn apply_elementwise_op(res_id: u8, res_dtype: &mut DType, ast_op: &ASTOp, register_indices: &BTreeMap<u8, String>) -> Vec<Op> {
    let mut ops = Vec::new();
    // TODO put all unary ops into single function or probably macro
    match ast_op {
        ASTOp::Unary(x, op) => {
            let mut relu = false;
            let op = match op {
                ASTUOp::Cast(dtype) => {
                    *res_dtype = *dtype;
                    UOp::Cast(*dtype)
                }
                ASTUOp::Neg => UOp::Neg,
                ASTUOp::ReLU => {
                    relu = true;
                    UOp::Neg
                }
                ASTUOp::Sin => UOp::Sin,
                ASTUOp::Cos => UOp::Cos,
                ASTUOp::Exp => UOp::Exp,
                ASTUOp::Ln => UOp::Ln,
                ASTUOp::Tanh => UOp::Tanh,
                ASTUOp::Sqrt => UOp::Sqrt,
            };
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            if relu {
                ops.push(Op::Binary {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    x: Var::Register {
                        id: *x,
                        index: register_indices.get(x).cloned(),
                    },
                    y: match res_dtype {
                        DType::F32 => Var::ConstF32(0.0),
                        DType::F64 => Var::ConstF64(0.0),
                        DType::I32 => Var::ConstI32(0),
                    },
                    op: BOp::Max,
                });
            } else {
                ops.push(Op::Unary {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    x: Var::Register {
                        id: *x,
                        index: register_indices.get(x).cloned(),
                    },
                    op,
                });
            }
        }
        ASTOp::Binary(x, y, op) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Binary {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: register_indices.get(x).cloned(),
                },
                y: Var::Register {
                    id: *y,
                    index: register_indices.get(y).cloned(),
                },
                op: match op {
                    ASTBOp::Add => BOp::Add,
                    ASTBOp::Sub => BOp::Sub,
                    ASTBOp::Mul => BOp::Mul,
                    ASTBOp::Div => BOp::Div,
                    ASTBOp::Pow => BOp::Pow,
                    ASTBOp::Cmplt => BOp::Cmplt,
                },
            });
        }
        ASTOp::Where(x, y, z) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Where {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: register_indices.get(x).cloned(),
                },
                y: Var::Register {
                    id: *y,
                    index: register_indices.get(y).cloned(),
                },
                z: Var::Register {
                    id: *z,
                    index: register_indices.get(z).cloned(),
                },
            });
        }
        ASTOp::Leaf(..) | ASTOp::Reduce(..) => {
            panic!()
        }
    }
    ops
}
