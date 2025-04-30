use std::hash::BuildHasherDefault;

use crate::{
    dtype::Constant, graph::{kernel::{Op, TId}, BOp, UOp}, shape::Dim, DType, Map
};

use super::optimizer::Optimization;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IRKernel {
    /// `read_only`, dtype
    pub global_variables: Vec<(bool, DType)>,
    /// ops
    pub ops: Vec<IROp>,
}

// IR register id
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RId(u16);

impl RId {
    fn from_usize(id: usize) -> Self {
        Self(u16::try_from(id).unwrap())
    }

    const fn index(self) -> usize {
        self.0 as usize
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
    Load { address: u16, offset: u16 },
    //LoadTile { address, tile specs }
    Store { address: u16, offset: u16 },
    Unary { x: u16, op: UOp },
    Cast { x: u16, dtype: DType },
    Binary { x: u16, y: u16, op: BOp },
    Loop { len: Dim },
    EndLoop,
    Accumulator { init: Constant },
    //LocalAccumulator {  },
    LocalBarrier,
}

// convert graph kernel to IR ops
pub fn lower_to_ir(kernel_ops: &[Op], opts: &Optimization) -> IRKernel {
    let mut global_variables = Vec::new();
    let mut ops = Vec::new();

    // opts contains information about:
    // 1. which loops should be split
    // 2. which loops should be reordered
    // 3. which loads should be tiled
    // 4. local accumulators
    // Other optimizations are far less important.

    let mut t_map: Map<TId, RId> = Map::with_hasher(BuildHasherDefault::new());

    // set of global vairables for deduplication
    let mut global_vars_map = Map::with_hasher(BuildHasherDefault::new());
    for op in kernel_ops {
        match op {
            &Op::Loop { len, .. } => ops.push(IROp::Loop { len }),
            &Op::Const { z, value, ref view } => {
                t_map.insert(z, RId::from_usize(ops.len()));
                ops.push(IROp::Const(value));
            }
            &Op::Load { z, x, view: ref xview, dtype: xdtype } => {
                let address = if let Some(&address) = global_vars_map.get(&x) {
                    address
                } else {
                    let address = global_variables.len() as u16;
                    global_variables.push((true, xdtype));
                    global_vars_map.insert(x, address);
                    address
                };
                t_map.insert(z, RId::from_usize(ops.len()));

                //ops.push(IROp::Load { address, offset: todo!() });
                //let zreg = ir_for_indexed_load(ops, address);
                todo!()
            }
            Op::Store { z, view: zview, dtype: zdtype, x } => {
                todo!()
            }
            Op::Accumulator { z, rop, dtype } => {
                todo!()
            }
            Op::AccAssign { rop, num_loops } => {
                todo!()
            }
            Op::Cast { z, x, dtype } => {
                todo!()
            }
            Op::Unary { z, x, uop } => {
                todo!()
            }
            Op::Binary { z, x, y, bop } => {
                todo!()
            }
        }
    }

    // typical IR optimization passes here

    IRKernel { global_variables, ops }
}
