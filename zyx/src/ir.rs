use crate::{
    dtype::Constant, kernel::Op, node::{BOp, UOp}, optimizer::Optimization, shape::Dim, DType
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Register,
    Local,
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
fn lower_to_ir(kernel_ops: &[Op], opts: Optimization) -> Vec<IROp> {
    let mut ir_ops = Vec::new();

    // opts contains information about:
    // 1. which loops should be split
    // 2. which loops should be reordered
    // 3. which loads should be tiled
    // 4. local accumulators
    // Other optimizations are far less important.

    for op in kernel_ops {
        match op {
            Op::Loop { axis, len } => todo!(),
            Op::EndLoop => todo!(),
            Op::Const { z, value, view } => todo!(),
            Op::Load { z, zscope, zview, x, xscope, xview, xdtype } => todo!(),
            Op::Store { z, zscope, zview, zdtype, x, xscope, xview } => todo!(),
            Op::Accumulator { z, rop, dtype } => todo!(),
            Op::Cast { z, x, dtype } => todo!(),
            Op::Unary { z, x, uop } => todo!(),
            Op::Binary { z, x, y, bop } => todo!(),
        }
    }

    ir_ops
}
