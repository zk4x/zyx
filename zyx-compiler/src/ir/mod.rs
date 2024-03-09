use crate::{ASTOp, AST};
use alloc::{boxed::Box, format as f, string::String, vec::Vec};
use core::fmt::{Display, Formatter};
use zyx_core::{
    axes::{Axes, IntoAxes},
    dtype::DType,
    shape::Shape,
    view::{Index, View}
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
            Var::ConstF32(value) => {
                f.write_fmt(format_args!("{value:.8}f"))
            }
            Var::ConstF64(value) => {
                f.write_fmt(format_args!("{value:.16}"))
            }
            Var::ConstI32(value) => {
                f.write_fmt(format_args!("{value}"))
            }
            Var::ConstI64(value) => {
                f.write_fmt(format_args!("{value}"))
            }
        }
    }
}

/// Op for the compilers
pub enum Op {
    /// Load into res from arg at index
    LoadGlobal {
        res: Var,
        arg: u8,
        index: Index,
    },
    /// Store arg into res at index
    StoreGlobal {
        res: u8,
        index: Index,
        arg: Var,
    },
    /// Declare register variable id with dtype and optionally
    /// with given length (as a vector)
    DeclareVar {
        dtype: DType,
        id: u8,
        len: Option<u8>,
    },
    /// Initialize index id with value
    InitIndex {
        id: u8,
        value: String,
    },
    /// Initialize accumulator, if sum_reduce is true,
    /// then initilize to zero, otherwise initilize to minimum
    /// of dtype.
    InitAccumulator {
        id: u8,
        dtype: DType,
        is_sum_reduce: bool,
        len: Option<u8>,
    },
    /// Cast x to res_dtype
    Cast {
        res_dtype: DType,
        res: Var,
        x: Var,
    },
    /// Neg
    Neg {
        res: Var,
        x: Var,
    },
    /// Sin
    Sin {
        res: Var,
        x: Var,
    },
    /// Cos
    Cos {
        res: Var,
        x: Var,
    },
    /// Ln
    Ln {
        res: Var,
        x: Var,
    },
    /// Exp
    Exp {
        res: Var,
        x: Var,
    },
    /// Tanh
    Tanh {
        res: Var,
        x: Var,
    },
    /// Sqrt
    Sqrt {
        res: Var,
        x: Var,
    },
    /// Add
    Add {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Sub
    Sub {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Mul
    Mul {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Div
    Div {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Pow
    Pow {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Cmplt
    Cmplt {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Max
    Max {
        res: Var,
        x: Var,
        y: Var,
    },
    /// Where, x is condition, y is if true, otherwise z
    Where {
        res: Var,
        x: Var,
        y: Var,
        z: Var,
    },
    // Loop inside kernel (register/private)
    Loop {
        id: u8,
        upper_bound: usize,
        step: usize,
    },
    /// End of loop
    EndLoop,
}

/// Intermediate representation for compilers
pub struct IR {
    pub global_work_size: Vec<usize>,
    pub local_work_size: Vec<usize>,
    pub kernel_args: Vec<(DType, bool)>, // dtype and read_only
    pub ops: Vec<Op>,
    pub res_byte_size: usize,
}

/// Calculates arg_views, reduce dim size, global, local and register work sizes,
/// each across three dimensions
fn calculate_work_sizes(
    ast_reduce_axes: &Option<Axes>,
    ast_shape: &Shape,
    ast_arg_views: Vec<View>,
    max_local_work_size: usize,
    _max_num_registers: usize,
) -> (Vec<View>, Shape, Option<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let (arg_views, shape, reduce_dim) = if let Some(reduce_axes) = &ast_reduce_axes {
        let mut arg_views = ast_arg_views.clone();
        let rank = ast_shape.rank();
        let permute_axes = (0..rank as i64)
            .filter(|a| !reduce_axes.contains(*a as usize))
            .chain(reduce_axes.iter().map(|a| *a as i64))
            .collect::<Box<_>>()
            .into_axes(rank);
        let shape = if rank > 4 || reduce_axes.len() > 1 {
            let d1: usize = ast_shape
                .iter()
                .enumerate()
                .filter_map(|(a, d)| {
                    if reduce_axes.contains(a) {
                        Some(*d)
                    } else {
                        None
                    }
                })
                .product();
            let d0 = ast_shape.numel() / d1;
            let shape: Shape = [d0, d1].into();
            for view in &mut arg_views {
                *view = view.permute(&permute_axes).reshape(&shape);
            }
            shape
        } else {
            for view in &mut arg_views {
                *view = view.permute(&permute_axes);
            }
            ast_shape.permute(&permute_axes)
        };
        let reduce_dim = shape[-1];
        (arg_views, shape, Some(reduce_dim))
    } else {
        let mut arg_views = ast_arg_views.clone();
        let shape = if ast_shape.rank() > 3 {
            let n = ast_shape.numel();
            for view in &mut arg_views {
                *view = view.reshape(&[n].into());
            }
            [n].into()
        } else {
            ast_shape.clone()
        };
        (arg_views, shape, None)
    };
    let mut lws = 1;
    let rank = shape.rank();
    let mut _register_work_size = Vec::new();
    let mut global_work_size: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            if reduce_dim.is_some() && i == rank - 1 {
                None
            } else {
                let d = *d;
                /*register_work_size.push(1);
                while d % 2 == 0 && register_work_size[i] * 2 <= max_register_work_size {
                    register_work_size[i] *= 2;
                    d /= 2;
                }*/
                Some(d)
            }
        })
        .collect();
    //let mut full_reduce = false; // reduce across all axes
    if global_work_size.len() == 0 {
        //full_reduce = true;
        global_work_size.push(1);
    }
    // Runtimes are horrible at inferring local work sizes, we just have to give it our
    let local_work_size: Vec<usize> = global_work_size
        .iter()
        .rev()
        .map(|d| {
            let mut x = 1;
            while d % (x * 2) == 0 && x * lws < max_local_work_size {
                x *= 2;
            }
            lws *= x;
            x
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    (
        arg_views,
        shape,
        reduce_dim,
        global_work_size,
        local_work_size,
        _register_work_size,
    )
}

// Single most important function and one of the most difficult
// functions to write. All of this is cached, so take your time to optimize
// these kernels.
pub(super) fn ast_to_ir(ast: &AST, max_local_work_size: usize, max_num_registers: usize) -> IR {
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
    let (arg_views, res_shape, reduce_dim, global_work_size, local_work_size, _register_work_size) =
        calculate_work_sizes(
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
        compile_reduce_kernel(ast, reduce_dim, &local_work_size, arg_views, res_shape)
    } else {
        compile_elementwise_kernel(ast, &local_work_size, arg_views, res_shape)
    };

    IR {
        global_work_size,
        local_work_size,
        kernel_args,
        ops,
        res_byte_size,
    }
}

fn compile_reduce_kernel(ast: &AST, reduce_dim: usize, local_work_size: &[usize], arg_views: Vec<View>, res_shape: Shape) -> Vec<Op> {
    let mut ops = Vec::new();

    // Add indexes for ops after reduce
    for (a, d) in local_work_size.iter().enumerate() {
        ops.push(Op::InitIndex {
            id: a as u8,
            value: f!("gid{a}*{d}+lid{a}"),
        });
    }

    let mut reduce_op_i = 0;
    let mut is_sum_reduce = true;
    for (i, op) in ast.ops.iter().enumerate() {
        match op {
            ASTOp::Sum(_) => {
                reduce_op_i = i;
                break
            }
            ASTOp::Max(_) => {
                is_sum_reduce = false;
                reduce_op_i = i;
                break
            }
            _ => {}
        }
    }

    // Initiliaze accumulator
    ops.push(Op::InitAccumulator {
        id: reduce_op_i as u8,
        dtype: ast.reduce_dtype.unwrap(),
        is_sum_reduce,
        len: None,
    });

    ops.push(Op::Loop {
        id: 0,
        upper_bound: reduce_dim,
        step: 1,
    });

    // Indices in reduce loop
    ops.push(Op::InitIndex { id: local_work_size.len() as u8, value: f!("rid0") });

    // Apply AST ops before reduce
    let mut res_dtype = DType::F32;
    let mut res_id = 0;
    while res_id < reduce_op_i as u8 {
        let op = &ast.ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = ast.arg_dtypes[*id as usize];
                ops.push(Op::DeclareVar {
                    dtype: res_dtype,
                    id: res_id,
                    len: None,
                });
                let view = &arg_views[*id as usize];
                ops.push(Op::LoadGlobal {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    arg: *id,
                    index: view.cidx(),
                })
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }

    // Apply reduce op
    if is_sum_reduce {
        ops.push(Op::Add {
            res: Var::Register { id: reduce_op_i as u8, index: None },
            x: Var::Register { id: res_id - 1, index: None },
            y: Var::Register { id: reduce_op_i as u8, index: None },
        });
    } else {
        ops.push(Op::Max {
            res: Var::Register { id: reduce_op_i as u8, index: None },
            x: Var::Register { id: res_id - 1, index: None },
            y: Var::Register { id: reduce_op_i as u8, index: None },
        });
    }
    res_id += 1;

    // End reduce loop after reduce op was applied
    ops.push(Op::EndLoop);

    // Apply ops after reduce
    while res_id < ast.ops.len() as u8 {
        let op = &ast.ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = ast.arg_dtypes[*id as usize];
                ops.push(Op::DeclareVar {
                    dtype: res_dtype,
                    id: res_id,
                    len: None,
                });
                ops.push(Op::LoadGlobal {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    arg: *id,
                    index: arg_views[*id as usize].cidx(),
                })
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }

    // Store result
    ops.push(Op::StoreGlobal {
        res: ast.arg_dtypes.len() as u8,
        index: View::new(res_shape[0..-1].into()).cidx(),
        arg: Var::Register { id: res_id-1, index: None },
    });
    ops
}

fn compile_elementwise_kernel(ast: &AST, local_work_size: &[usize], arg_views: Vec<View>, res_shape: Shape) -> Vec<Op> {
    let mut ops = Vec::new();
    // Add indexes
    for (a, d) in local_work_size.iter().enumerate() {
        ops.push(Op::InitIndex {
            id: a as u8,
            value: f!("gid{a}*{d}+lid{a}"),
        });
    }
    // Compile AST ops
    let mut res_dtype = DType::F32;
    let mut res_id = 0;
    while res_id < ast.ops.len() as u8 {
        let op = &ast.ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = ast.arg_dtypes[*id as usize];
                ops.push(Op::DeclareVar {
                    dtype: res_dtype,
                    id: res_id,
                    len: None,
                });
                ops.push(Op::LoadGlobal {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    arg: *id,
                    index: arg_views[*id as usize].cidx(),
                })
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }
    // Store result
    ops.push(Op::StoreGlobal {
        res: ast.arg_dtypes.len() as u8,
        index: View::new(res_shape.clone()).cidx(),
        arg: Var::Register { id: res_id - 1, index: None },
    });
    ops
}

// Same op can be applied multiple times with different register_index
fn apply_elementwise_op(res_id: u8, res_dtype: &mut DType, ast_op: &ASTOp) -> Vec<Op> {
    let mut ops = Vec::new();
    // TODO put all unary ops into single function or probably macro
    match ast_op {
        ASTOp::Cast(x, dtype) => {
            *res_dtype = *dtype;
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Cast {
                res_dtype: *res_dtype,
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Neg(x) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Neg {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::ReLU(x) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Max {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: match res_dtype {
                    DType::F32 => Var::ConstF32(0.0),
                    DType::F64 => Var::ConstF64(0.0),
                    DType::I32 => Var::ConstI32(0),
                }
            });
        }
        ASTOp::Sin(x) => {
            // TODO use Log2 as IR op instead of ln
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Sin {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Cos(x) => {
            // TODO use Log2 as IR op instead of ln
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Cos {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Ln(x) => {
            // TODO use Log2 as IR op instead of ln
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Ln {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Exp(x) => {
            // TODO use Exp2 as IR op instead of Exp
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Exp {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Tanh(x) => {
            // TODO optimize this
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Tanh {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Sqrt(x) => {
            // TODO optimize this, small precision loss is ok
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Sqrt {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
            });
        }
        ASTOp::Add(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Add {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
            });
        }
        ASTOp::Sub(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Sub {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
            });
        }
        ASTOp::Mul(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Mul {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
            });
        }
        ASTOp::Div(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Div {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
            });
        }
        ASTOp::Pow(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Pow {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
            });
        }
        ASTOp::Cmplt(x, y) => {
            ops.push(Op::DeclareVar {
                dtype: *res_dtype,
                id: res_id,
                len: None,
            });
            ops.push(Op::Cmplt {
                res: Var::Register {
                    id: res_id,
                    index: None,
                },
                x: Var::Register {
                    id: *x,
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
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
                    index: None,
                },
                y: Var::Register {
                    id: *y,
                    index: None,
                },
                z: Var::Register {
                    id: *z,
                    index: None,
                },
            });
        }
        ASTOp::Leaf(_) | ASTOp::Sum(_) | ASTOp::Max(_) => {
            panic!()
        }
    }
    ops
}
