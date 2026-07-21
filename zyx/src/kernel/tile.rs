use crate::{
    DType, ZyxError,
    kernel::{DeviceId, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    pub fn tile(&mut self, gidx: OpId, gidy: OpId, factorx: Dim, factory: Dim) {
        let Op::Index { len: lenx, scope: Scope::Global, .. } = self.ops[gidx].op else {
            return;
        };
        let Op::Index { len: leny, scope: Scope::Global, .. } = self.ops[gidy].op else {
            return;
        };
        if !lenx.is_multiple_of(factorx) || !leny.is_multiple_of(factory) {
            return;
        }

        let mut op_id = self.head;
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
        }
    }
}

#[test]
fn tile_sin() -> Result<(), ZyxError> {
    let mut kernel = Kernel::new(DeviceId::AUTO);
    let a = kernel.define(DType::BF16, Scope::Global, false, 1024);
    let b = kernel.define(DType::BF16, Scope::Global, false, 1024);
    let gidx = kernel.gidx(0, 256);
    let gidy = kernel.gidx(1, 256);
    let stride = kernel.const_idx(256);
    let idx = kernel.mad(gidx, stride, gidy);
    let x = kernel.load(a, idx, MemLayout::Scalar);
    let x = kernel.sin(x);
    kernel.store(b, x, idx, MemLayout::Scalar);

    kernel.debug();

    kernel.tile(gidx, gidy, 32, 32);

    kernel.debug();

    kernel.verify();

    //let _compiled = kernel.compile()?;
    //compiled.forward();
    Ok(())
}
