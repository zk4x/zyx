use crate::{
    DType, Map,
    dtype::Constant,
    graph::{
        BOp, ROp, UOp,
        kernel::{self, Kernel, Op, TId},
        view::RDim,
    },
    shape::Dim,
    slab::{Slab, SlabId},
};
use std::{collections::{BTreeMap, BTreeSet}, fmt::Display, hash::BuildHasherDefault};

use super::optimizer::Optimization;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IRKernel {
    /// `read_only`, dtype
    pub global_variables: Vec<(bool, DType)>,
    /// ops
    pub ops: Vec<IROp>,
}

// IR register id
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RId(u16);

impl From<usize> for RId {
    fn from(value: usize) -> Self {
        RId(value as u16)
    }
}

impl From<RId> for usize {
    fn from(value: RId) -> Self {
        value.0 as usize
    }
}

impl SlabId for RId {
    const ZERO: Self = RId(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl Display for RId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("r{}", self.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IRScope {
    Register,
    Local,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IROp {
    Const(Constant),
    Load { address: u16, offset: RId },
    //LoadTile { address, tile specs }
    Store { x: RId, address: u16, offset: RId },
    Unary { x: RId, uop: UOp },
    Cast { x: RId, dtype: DType },
    Binary { x: RId, y: RId, bop: BOp },
    MAdd { x: RId, y: RId, c: RId },
    Loop { len: Dim },
    Accumulator { init: Constant },
    AccAssign { x: RId, rop: ROp, num_loops: u32 },
    //LocalAccumulator {  },
    LocalBarrier,
}

impl IRKernel {
    pub fn debug(&self) {
        print!("fn IRKernel(");
        for (i, (read_only, dtype)) in self.global_variables.iter().enumerate() {
            print!("{}g{i} {dtype}, ", if *read_only { "" } else { "&" });
        }
        println!(")");
        let mut rid = 0;
        for op in self.ops.iter().take_while(|op| matches!(op, IROp::Loop { .. })) {
            if let IROp::Loop { len } = op {
                println!("{rid} for 0..{len}");
                rid += 1;
            } else {
                break;
            }
        }
        let mut indent = String::new();
        for op in self.ops.iter().skip(rid) {
            match op {
                IROp::Const(constant) => {
                    println!("{indent}{rid} const {constant}");
                }
                IROp::Load { address, offset } => {
                    println!("{indent}{rid} load g{address} at offset {offset}");
                }
                IROp::Store { x, address, offset } => {
                    println!("{indent}{rid} store {x} into g{address} at offset {offset}");
                }
                IROp::Unary { x, uop: op } => {
                    println!("{indent}{rid} uop.{op:?} {x}");
                }
                IROp::Cast { x, dtype } => {
                    println!("{indent}{rid} cast {x} -> {dtype}");
                }
                IROp::Binary { x, y, bop: op } => {
                    println!("{indent}{rid} bop.{op:?} {x}, {y}");
                }
                IROp::MAdd { x, y, c } => {
                    println!("{indent}{rid} madd {x} * {y} + {c}");
                }
                IROp::Loop { len } => {
                    println!("{indent}{rid} for 0..{len}");
                    indent.push_str("  ");
                }
                IROp::AccAssign { x, rop, num_loops } => {
                    match rop {
                        ROp::Sum => println!("{indent}{rid} += {x}"),
                        ROp::Max => println!("{indent}{rid} max= {x}"),
                    }
                    for _ in 0..num_loops * 2 {
                        indent.pop();
                    }
                }
                IROp::Accumulator { init } => {
                    println!("{indent}{rid} acc = {init}");
                }
                IROp::LocalBarrier => todo!(),
            }
            rid += 1;
        }
        println!();
    }
}

// convert graph kernel to IR ops
pub fn lower_to_ir(kernel_ops: Vec<Op>, opts: &Optimization) -> IRKernel {
    let mut kernel = crate::graph::kernel::Kernel {
        ops: kernel_ops,
        loads: Vec::new(),
        stores: Vec::new(),
        outputs: BTreeMap::new(),
        depends_on: BTreeSet::new(),
    };

    kernel.reshape(&opts.shape);

    // opts contains information about:
    // 1. which loops should be split
    // 2. which loops should be reordered
    // 3. which loads should be tiled
    // 4. local accumulators
    // Other optimizations are far less important.

    println!("Kernel after reshape");
    kernel.debug();

    let mut global_variables = Vec::new();
    let mut ops: Vec<IROp> = Vec::new();
    let mut reg_map: Map<TId, RId> = Map::with_hasher(BuildHasherDefault::new());
    // Must be ordered, so BTreeMap
    let mut loop_map: BTreeMap<usize, RId> = BTreeMap::new();

    // set of global vairables for deduplication
    //let mut global_vars_map = Map::with_hasher(BuildHasherDefault::new());
    for (op_id, op) in kernel.ops.iter().enumerate() {
        match op {
            &Op::Loop { len, .. } => {
                let loop_id = loop_map.len();
                loop_map.insert(loop_id, ops.len().into());
                ops.push(IROp::Loop { len });
            }
            &Op::Const { ref view, value } => {
                todo!();
            }
            &Op::Load { ref view, dtype } => {
                let address = global_variables.len() as u16;
                global_variables.push((true, dtype));

                let mut offset = ops.len().into();
                // Calculate the offset
                {
                    ops.push(IROp::Const(Constant::U64(0)));
                    for (i, &RDim { d, st, lp, rp }) in view.0[0].iter().enumerate() {
                        ops.push(IROp::Const(Constant::U64(st as u64)));
                        let y = (ops.len() - 1).into();
                        ops.push(IROp::MAdd { x: loop_map[&i], y, c: offset });
                        offset = (ops.len() - 1).into();
                    }
                }

                ops.push(IROp::Load { address, offset });

                reg_map.insert(op_id, (ops.len() - 1).into());
            }
            &Op::Store { x, ref view, dtype } => {
                let address = global_variables.len() as u16;
                global_variables.push((false, dtype));

                let mut offset = ops.len().into();
                // Calculate the offset
                {
                    ops.push(IROp::Const(Constant::U64(0)));
                    for (i, &RDim { d, st, lp, rp }) in view.0[0].iter().enumerate() {
                        ops.push(IROp::Const(Constant::U64(st as u64)));
                        let y = (ops.len() - 1).into();
                        ops.push(IROp::MAdd { x: loop_map[&i], y, c: offset });
                        offset = (ops.len() - 1).into();
                    }
                }

                ops.push(IROp::Store { x: reg_map[&x], address, offset });
            }
            &Op::Unary { x, uop } => {
                ops.push(IROp::Unary { x: reg_map[&x], uop });
                reg_map.insert(op_id, (ops.len() - 1).into());
            }
            &Op::Cast { x, dtype } => {
                ops.push(IROp::Cast { x: reg_map[&x], dtype });
                reg_map.insert(op_id, (ops.len() - 1).into());
            }
            &Op::Binary { x, y, bop } => {
                ops.push(IROp::Binary { x: reg_map[&x], y: reg_map[&y], bop });
                reg_map.insert(op_id, (ops.len() - 1).into());
            }
            Op::Accumulator { rop, dtype } => {
                let init = match rop {
                    ROp::Sum => dtype.zero_constant(),
                    ROp::Max => dtype.min_constant(),
                };
                ops.push(IROp::Accumulator { init });
                reg_map.insert(op_id, (ops.len() - 1).into());
            }
            &Op::AccAssign { x, rop, num_loops } => {
                for _ in 0..num_loops {
                    loop_map.pop_last();
                }
                ops.push(IROp::AccAssign { x: reg_map[&x], rop, num_loops });
            }
        }
    }

    // typical IR optimization passes here

    IRKernel { global_variables, ops }
}
