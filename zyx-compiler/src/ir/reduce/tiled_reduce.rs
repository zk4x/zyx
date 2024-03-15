use alloc::{format, vec::Vec, collections::{btree_set::BTreeSet, btree_map::BTreeMap}};
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::View;
use crate::{ASTOp, ASTROp, BOp, Op, UOp};
use crate::ir::{apply_elementwise_op, Var};


/*


We need contiguous wide loads for permuted buffers.
This is hard for buffers with multiple shapes, but for buffers with single shape,
we can just load them according to their strides[-1] dimension.

Basically we want to load them contiguously and do non-contiguous calculations
on register tiles. Or we can load them contiguously and transpose them before
assigning them to local/register memory.

For expanded buffers, we need to merge expanded axes of some buffers with non-expanded
axes of other buffers (strides[a] == 0 with strides[a] != 0).

At those dimensions, we need to reduce global work size and apply register tiling.


Example:

Let's say its batched reduce with shape:
shape [256, 1024, 512, 1024]
work dimensions (reduced due to register tiling)
global [64,  256, 128,  256]
local  [ 1,    1,  16,   16]
Where last dimension (1024) is the reduce dimension.

buffers in rows
dims in columns
last column is contiguous load stride
x marks expanded dimension
   0  1  2  3   st
0  x     x      1024*512       // permuted
1     x  x      1024
2  x        x   1
3     x         1024*512
4        x      1024*512       // permuted
5  x            1024*512*1024

Due to hardware limits, each buffer can be padded in at most 2 dimensions,
with register tile size of 4x4 and local memory tile size that is hardware dependent,
but mostly 64x64.

Which dimensions should be tiled and how will the padding work?
Buffer
0 - tile 1x3
1 - tile 0x3
2 - tile 1x2
3 - tile 2x3
4 - tile 1x3
5 - tile 2x3

So at most 2d tiles for each buffer in those dimensions that are not expanded,
going backward from reduce dimension to batch dimension.

Register tiling means up to 4x reduction in global (and possibly also local) work dimension size.

Tiles need to be loaded in such a way, that if for example when we are loading
axes 2 of buffer 2, we do not want to load axis 2 of buffer 0 and 1 (since those are expanded)
rather we want ot load axis 3.

Simply put we need to make it so that we are not going over expanded dimension more than once.
Thus load all non-expanded dimensions at once and then do single loop over expanded dimensions
to do the calculations on tiled values.

So if the tile is expanded, that dimension should be loaded outside of calculate loop.

We can have multiple load loops, but only one calculate loop.

We can do one loop over reduce dimension, which loads tiles one by one.
For those buffers that are expanded in reduce loop, we can laod them before the reduce loop.

 */


/// Returs vector of contiguous strides to enable us to have wide loads of global memory buffers
fn contiguous_load_stride(arg_views: &[View]) -> Vec<usize> {
    let mut arg_strides = Vec::new();
    for view in arg_views {
        let strides = view.strides();
        for st in strides.iter().rev() {
            if *st != 0 {
                arg_strides.push(*st);
                break;
            }
        }
    }
    arg_strides
}

fn buffer_expand_axesj(arg_views: &[View]) -> Vec<[bool; 4]> {
    let mut expand_axes = Vec::new();
    for (i, view) in arg_views.iter().enumerate() {
        expand_axes.push([false; 4]);
        let strides = view.strides();
        for (a, st) in strides.iter().enumerate() {
            if *st == 0 {
                expand_axes[i][a] = true;
            }
        }
    }
    expand_axes
}

pub(crate) fn compile_tiled_reduce(
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

    // TODO this kernel does not work if local work size in dimension 1 and 2 are different
    // so make it work with non-square tiles

    // TODO make sure we do not overuse local memory
    //let used_local_memory = 0;

    // Add indexes for ops after reduce
    for (a, d) in local_work_size.iter().enumerate() {
        if a != rank - 1 {
            ops.push(Op::InitIndex {
                id: a as u8,
                value: alloc::format!("gid{a}*{d}+lid{a}"),
            });
        }
    }

    let (reduce_op_i, is_sum_reduce) = ast_ops.iter().enumerate().find(|(_, op)| matches!(op, ASTOp::Reduce(..))).map(|(i, op)| {
        if let ASTOp::Reduce(_, rop) = op {
            if *rop == ASTROp::Sum {
                (i, true)
            } else {
                (i, false)
            }
        } else {
            panic!()
        }
    }).unwrap();

    // tile -> (axis -> tile size in that axis)
    let mut local_tiles: BTreeMap<u8, BTreeMap<u8, usize>> = BTreeMap::new();

    // Declare local memory tiles
    for (id, view) in arg_views.iter().enumerate() {
        let shape = view.shape();
        let strides = view.strides();
        std::println!("{shape} -> {strides}");
        if strides[3] == 0usize {
            let mut a = (rank-1) as u8;
            let mut dims = BTreeMap::new();
            let mut len = 1;
            for st in strides.iter().rev() {
                let dim = local_work_size[a as usize]*register_work_size[a as usize];
                len *= dim;
                if *st != 0 {
                    dims.insert(a, dim);
                    break;
                }
                a -= 1;
            }
            local_tiles.insert(id as u8, dims);
            ops.push(Op::DeclareLocalVar {
                id: id as u8,
                dtype: arg_dtypes[id],
                len,
            });
        } else if (strides[2] == 0usize && shape[1] > 1usize) || (strides[1] == 0usize && shape[1] > 1usize) {
            let mut a = (rank-1) as u8;
            let mut dims = BTreeMap::new();
            let mut len = 1;
            for st in strides.iter().rev() {
                if *st != 0 {
                    let dim = local_work_size[a as usize]*register_work_size[a as usize];
                    len *= dim;
                    dims.insert(a, dim);
                    if dims.len() > 1 {
                        break;
                    }
                }
                a -= 1;
            }
            local_tiles.insert(id as u8, dims);
            ops.push(Op::DeclareLocalVar {
                id: id as u8,
                dtype: arg_dtypes[id],
                len,
            });
        }
        // We currently do not tile buffers expanded only in batch dimension
    }

    std::println!("tiles: {local_tiles:?}");

    // Load local vectors for those buffers that are expanded in reduce dimension.
    for (id, axes) in &local_tiles {
        if axes.len() == 1 && axes.contains_key(&3) {
            ops.push(Op::LoadGlobal {
                res: Var::Local { id: *id, index: "lid2".into() },
                arg: 0,
                index: arg_views[*id as usize].cidx(),
            });
        }
    }

    // Initiliaze accumulator
    ops.push(Op::InitAccumulator {
        id: reduce_op_i as u8,
        dtype: reduce_dtype,
        is_sum_reduce,
        len: None,
    });

    // Main reduce loop over all local memory tiles
    ops.push(Op::Loop {
        id: 10,
        upper_bound: reduce_dim/local_work_size[3]/register_work_size[3],
        step: 1,
    });

    // Indices in reduce loop
    ops.push(Op::DeclareIndex {
        id: (rank - 1) as u8,
    });

    // Load local memory tiles that are not expanded in reduce dimension
    for (id, axes) in &local_tiles {
        if axes.len() != 1 || !axes.contains_key(&3) {
            let mut axes = axes.clone();
            axes.remove(&3);
            let a = axes.pop_last().unwrap().0 as usize;
            ops.push(Op::SetIndex {
                id: 3,
                value: format!("rid10*{}+lid{}", local_work_size[3], if a == 1 { 2 } else { 1 }),
            });
            ops.push(Op::LoadGlobal {
                res: Var::Local {
                    id: *id,
                    index: format!("lid{a}*{}+lid{}", local_work_size[a], if a == 1 { 2 } else { 1 }),
                },
                arg: *id,
                index: arg_views[*id as usize].cidx(),
            });
        }
    }

    // Synchronize after local memory loads
    ops.push(Op::LocalBarrier);

    // Inner loop to do the rest of reduce
    ops.push(Op::Loop {
        id: 9,
        upper_bound: local_work_size[3]*register_work_size[3],
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
                if let Some(axes) = local_tiles.get(id) {
                    let mut axes = axes.clone();
                    axes.remove(&3);
                    let a = axes.pop_last().unwrap().0 as usize;
                    ops.push(Op::Unary {
                        res: Var::Register {
                            id: res_id,
                            index: None,
                        },
                        x: Var::Local {
                            id: *id,
                            index: format!("lid{a}*{}+rid9", local_work_size[a]),
                        },
                        op: UOp::Noop,
                    });
                } else {
                    let view = &arg_views[*id as usize];
                    ops.push(Op::LoadGlobal {
                        res: Var::Register {
                            id: res_id,
                            index: None,
                        },
                        arg: *id,
                        index: view.cidx(),
                    });
                }
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
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

    // End inner reduce loop
    ops.push(Op::EndLoop);

    // Synchronize before loading next local tiles in reduce loop
    ops.push(Op::LocalBarrier);

    // End reduce loop after reduce op was applied
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
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }

    // Store result
    ops.push(Op::StoreGlobal {
        res: arg_dtypes.len() as u8,
        index: View::new(res_shape[0..-1].into()).cidx(),
        arg: Var::Register {
            id: res_id - 1,
            index: None,
        },
    });
    ops
}
