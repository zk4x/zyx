use std::collections::{BTreeMap, BTreeSet};
use std::prelude::rust_2015::Vec;
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::View;
use crate::{ASTOp, ASTROp, BOp, Op, UOp};
use crate::ir::{apply_elementwise_op, Var};
use alloc::{format as f, string::String};

/// This kernel does tiling in dimensions 1 and 2,
/// so it is for best for standard matmul and similar reduce ops,
/// which have input buffers expanded in those dimensions,
/// some in dimension 1, others in dimension 2.
pub(crate) fn compile_reduce_kernel(
    ast_ops: &[ASTOp],
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    reduce_dtype: DType,
    reduce_dim: usize,
    local_work_size: &[usize],
    register_work_size: &[usize],
    res_shape: Shape,
    _tiling_axes: BTreeSet<u8>,
    _max_local_memory_size: usize,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let rank = res_shape.rank();

    // Add index for batch dimension
    ops.push(Op::InitIndex {
        id: 0,
        value: f!("gid0*{}+lid0", local_work_size[0]),
    });

    let (reduce_op_i, is_sum_reduce) = ast_ops.iter().enumerate().find(|(_, op)| matches!(op, ASTOp::Reduce(..))).map(|(i, op)| {
        if let ASTOp::Reduce(_, rop) = op {
            if *rop == ASTROp::Sum {
                (i as u8, true)
            } else {
                (i as u8, false)
            }
        } else {
            panic!()
        }
    }).unwrap();

    let mut register_indices = BTreeMap::new();

    // Initiliaze accumulator
    ops.push(Op::InitAccumulator {
        id: reduce_op_i,
        dtype: reduce_dtype,
        is_sum_reduce,
        len: Some((register_work_size[1]*register_work_size[2]) as u8),
    });
    register_indices.insert(reduce_op_i, f!("rid1*{}+rid2", register_work_size[2]));

    // buffers ids and their tile sizes for each dimension
    let mut local_tiles = BTreeMap::new();

    // Declare local memory tiles
    for (id, view) in arg_views.iter().enumerate() {
        let strides = view.strides();
        match (strides[1], strides[2], strides[3]) {
            (_, _, 0usize) => {
                ops.push(Op::DeclareLocalVar {
                    id: id as u8,
                    dtype: arg_dtypes[id],
                    len: register_work_size[2]*local_work_size[2],
                });
            }
            (0usize, 0usize, _) => {
                ops.push(Op::DeclareLocalVar {
                    id: id as u8,
                    dtype: arg_dtypes[id],
                    len: local_work_size[3],
                });
                local_tiles.insert(id as u8, [1, 1, 1, local_work_size[3]]);
                // Declare appropriate register tile
                ops.push(Op::DeclareVar {
                    dtype: arg_dtypes[id],
                    id: id as u8,
                    len: None,
                });
            }
            (0usize, _, _) => {
                ops.push(Op::DeclareLocalVar {
                    id: id as u8,
                    dtype: arg_dtypes[id],
                    len: local_work_size[2]*register_work_size[2]*local_work_size[3],
                });
                local_tiles.insert(id as u8, [1, 1, local_work_size[2]*register_work_size[2], local_work_size[3]]);
                // Declare appropriate register tile
                ops.push(Op::DeclareVar {
                    dtype: arg_dtypes[id],
                    id: id as u8,
                    len: Some(register_work_size[2] as u8),
                });
            }
            (_, 0usize, _) => {
                ops.push(Op::DeclareLocalVar {
                    id: id as u8,
                    dtype: arg_dtypes[id],
                    len: local_work_size[1]*register_work_size[1]*local_work_size[3],
                });
                local_tiles.insert(id as u8, [1, local_work_size[1]*register_work_size[1], 1, local_work_size[3]]);
                // Declare appropriate register tile
                ops.push(Op::DeclareVar {
                    dtype: arg_dtypes[id],
                    id: id as u8,
                    len: None,
                });
            }
            (_, _, _) => {}
        }
    }

    // Outer reduce loop over tiles
    ops.push(Op::Loop {
        name: "lid3".into(),
        upper_bound: reduce_dim/local_work_size[3],
        step: 1,
    });

    // Load local memory tiles
    for (id, axes) in &local_tiles {
        let mut lid = String::new();
        for a in 1..=2 {
            if axes[a as usize] > 1 {
                ops.push(Op::Loop {
                    name: f!("rid{a}"),
                    upper_bound: register_work_size[a as usize],
                    step: 1,
                });
                ops.push(Op::InitIndex {
                    id: a,
                    value: f!("gid{a}*{}+lid{a}*{}+rid{a}", local_work_size[a as usize]*register_work_size[a as usize], register_work_size[a as usize]),
                });
                lid += &f!("lid{a}*{}+rid{a}*{}+", register_work_size[a as usize]*local_work_size[3], local_work_size[3]);
            }
        }
        // This is a trick, since assuption of this kernel is that local_work_size[1] == local_work_size[3]
        // && local_work_size[2] == local_work_size[3], we can use those local work sizes to load
        // reduce dimension local tile
        let local_reduce_substitute_id = if axes[1] > 1 { 2 } else { 1 };
        ops.push(Op::InitIndex {
            id: 3,
            value: f!("lid3*{}+lid{local_reduce_substitute_id}", local_work_size[3]),
        });
        ops.push(Op::LoadGlobal {
            res: Var::Local { id: *id, index: f!("{lid}lid{local_reduce_substitute_id}") },
            arg: *id,
            index: arg_views[*id as usize].cidx(),
        });
        for a in 1..=2 {
            if axes[a] > 1 {
                ops.push(Op::EndLoop);
            }
        }
    }

    // Synchronize memory after loading tiles
    ops.push(Op::LocalBarrier);

    // Reduce loop across register tiled values
    ops.push(Op::Loop {
        name: "rid3".into(),
        upper_bound: local_work_size[3],
        step: 1,
    });

    // Load all local tiles tiled in dimension 1 into registers
    ops.push(Op::Loop {
        name: "rid1".into(),
        upper_bound: register_work_size[1],
        step: 1,
    });
    for (id, axes) in &local_tiles {
        if axes[1] > 1 {
            ops.push(Op::Unary {
                res: Var::Register { id: *id, index: Some("rid1".into()) },
                x: Var::Local { id: *id, index: "".into() },
                op: UOp::Noop,
            });
        }
    }
    ops.push(Op::EndLoop);

    // Register loops (in single local memory tile)
    ops.push(Op::Loop {
        name: "rid1".into(),
        upper_bound: register_work_size[1],
        step: 1,
    });
    ops.push(Op::Loop {
        name: "rid2".into(),
        upper_bound: register_work_size[2],
        step: 1,
    });

    // Apply AST ops before reduce
    let mut res_dtype = DType::F32;
    let mut res_id = 0;
    while res_id < reduce_op_i as u8 {
        let op = &ast_ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = arg_dtypes[*id as usize];
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
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op, &register_indices));
            }
        }
        res_id += 1;
    }

    // Apply reduce op
    if is_sum_reduce {
        ops.push(Op::Binary {
            res: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            x: Var::Register {
                id: res_id - 1,
                index: None,
            },
            y: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            op: BOp::Add,
        });
    } else {
        ops.push(Op::Binary {
            res: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            x: Var::Register {
                id: res_id - 1,
                index: None,
            },
            y: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            op: BOp::Max,
        });
    }
    res_id += 1;

    // End register reduce loops
    ops.push(Op::EndLoop);
    ops.push(Op::EndLoop);
    ops.push(Op::EndLoop);

    // Synchronize memory before loading next tile
    ops.push(Op::LocalBarrier);

    // End outer reduce loop
    ops.push(Op::EndLoop);

    // Apply ops after reduce
    while res_id < ast_ops.len() as u8 {
        let op = &ast_ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = arg_dtypes[*id as usize];
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
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op, &register_indices));
            }
        }
        res_id += 1;
    }

    ops.push(Op::Loop {
        name: "rid1".into(),
        upper_bound: register_work_size[1],
        step: 1,
    });
    ops.push(Op::InitIndex {
        id: 1,
        value: f!("gid1*{}+lid1*{}+rid1", local_work_size[1]*register_work_size[1], register_work_size[1]),
    });
    ops.push(Op::Loop {
        name: "rid2".into(),
        upper_bound: register_work_size[2],
        step: 1,
    });
    ops.push(Op::InitIndex {
        id: 2,
        value: f!("gid2*{}+lid2*{}+rid2", local_work_size[2]*register_work_size[2], register_work_size[2]),
    });
    // Store result
    ops.push(Op::StoreGlobal {
        res: arg_dtypes.len() as u8,
        index: View::new(res_shape[0..-1].into()).cidx(),
        arg: Var::Register {
            id: res_id - 1,
            index: register_indices.get(&(res_id - 1)).cloned(),
        },
    });
    ops.push(Op::EndLoop);
    ops.push(Op::EndLoop);

    ops
}
