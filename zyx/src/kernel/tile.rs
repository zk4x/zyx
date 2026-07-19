/*use crate::{
    kernel::{Kernel, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    pub fn tile(&mut self, gidx: OpId, factor: Dim) {
        let Op::Index { len, scope: Scope::Global, axis } = self.ops[gidx].op else { return };
        if !len.is_multiple_of(factor) {
            return;
        }

        self.thread_coarse(gidx, factor);

        /*let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Cast { x, dtype } => todo!(),
                Op::Unary { x, uop } => todo!(),
                Op::Binary { x, y, bop } => todo!(),
                Op::Const(constant) => todo!(),
                Op::Define { dtype, scope, ro, len } => todo!(),
                Op::Store { dst, x, index, layout } => todo!(),
                Op::Load { src, index, layout } => todo!(),
                Op::Index { len, scope, axis } => todo!(),
                Op::Loop { len } => todo!(),
                Op::EndLoop => todo!(),
                Op::Mad { x, y, z } => todo!(),
                Op::Wmma { dims, layout, dtype, a, b, c } => todo!(),
                Op::Vectorize { ref ops } => todo!(),
                Op::Devectorize { vec, idx } => todo!(),
                Op::Barrier { scope } => todo!(),
                Op::If { condition } => todo!(),
                Op::EndIf => todo!(),
                Op::StoreView { src, dtype } => todo!(),
                Op::Reduce { x, rop, n_axes } => todo!(),
                _ => todo!(),
            }
        }*/
    }
}*/
