use crate::{
    Map,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    /// Tile the iteration space by `factorx` × `factory`.
    ///
    /// Rescales `gidx` and `gidy` from element indices to tile indices,
    /// and upgrades global loads/stores to `MemLayout::Tile`.
    pub fn tile(&mut self, gidx: OpId, gidy: OpId, factorx: Dim, factory: Dim) {
        let Op::Index { len: lenx, scope: Scope::Global, axis: axisx } = self.ops[gidx].op else {
            return;
        };
        let Op::Index { len: leny, scope: Scope::Global, axis: axisy } = self.ops[gidy].op else {
            return;
        };
        if !lenx.is_multiple_of(factorx) || !leny.is_multiple_of(factory) {
            return;
        }

        let orig_lenx = lenx;

        // 1. Rescale gidx and gidy to tile indices
        self.ops[gidx].op = Op::Index { len: lenx / factorx, scope: Scope::Global, axis: axisx };
        self.ops[gidy].op = Op::Index { len: leny / factory, scope: Scope::Global, axis: axisy };

        // 2. Insert scaled = gidx * factorx after gidx
        let factorx_const = self.insert_after(gidx, Op::Const(Constant::idx(factorx)));
        let scaled_gidx = self.insert_after(factorx_const, Op::Binary { x: gidx, y: factorx_const, bop: BOp::Mul });

        // 3. Insert scaled = gidy * factory after gidy
        let factory_const = self.insert_after(gidy, Op::Const(Constant::idx(factory)));
        let scaled_gidy = self.insert_after(factory_const, Op::Binary { x: gidy, y: factory_const, bop: BOp::Mul });

        // 4. Remap all consumers of gidx → scaled_gidx, gidy → scaled_gidy
        let mut remap = Map::default();
        remap.insert(gidx, scaled_gidx);
        remap.insert(gidy, scaled_gidy);

        let mut op_id = self.head;
        while !op_id.is_null() {
            if op_id == scaled_gidx || op_id == scaled_gidy || op_id == factorx_const || op_id == factory_const {
                op_id = self.next_op(op_id);
                continue;
            }
            self.ops[op_id].op.remap_params(&remap);
            op_id = self.next_op(op_id);
        }

        // 5. Tile loads/stores to global buffers
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Load { src, index, layout } => {
                    if layout == MemLayout::Scalar {
                        if let Op::Define { scope: Scope::Global, .. } = self.ops[src].op {
                            self.ops[op_id].op = Op::Load {
                                src,
                                index,
                                layout: MemLayout::Tile { x: factorx as u16, y: factory as u16, stride: orig_lenx as u32 },
                            };
                        }
                    }
                }
                Op::Store { dst, x, index, layout } => {
                    if layout == MemLayout::Scalar {
                        if let Op::Define { scope: Scope::Global, .. } = self.ops[dst].op {
                            self.ops[op_id].op = Op::Store {
                                dst,
                                x,
                                index,
                                layout: MemLayout::Tile { x: factorx as u16, y: factory as u16, stride: orig_lenx as u32 },
                            };
                        }
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
    }
}

#[test]
fn tile_sin() -> Result<(), crate::ZyxError> {
    use crate::{DType, Float, Tensor, bf16, kernel::DeviceId};
    // Single tile (32x32 BF16 = 1 tile of 4096 bytes as Float32)
    let n = 32 * 32; // 1024
    let mut kernel = Kernel::new(DeviceId::AUTO);
    let a = kernel.define(DType::BF16, Scope::Global, true, n);
    let b = kernel.define(DType::BF16, Scope::Global, false, n);
    let gidx = kernel.gidx(0, 32);
    let gidy = kernel.gidx(1, 32);
    let stride = kernel.const_idx(32);
    let idx = kernel.mad(gidx, stride, gidy);
    let x = kernel.load(a, idx, MemLayout::Scalar);
    let x = kernel.sin(x);
    kernel.store(b, x, idx, MemLayout::Scalar);

    kernel.tile(gidx, gidy, 32, 32);
    kernel.run_always_on_optimizations();
    kernel.run_always_on_optimizations();

    kernel.debug();

    kernel.verify();

    let data: Vec<Vec<bf16>> = (0..32).map(|i| (0..32).map(|j| bf16::from_f32((i * 32 + j) as f32)).collect()).collect();
    let x = Tensor::from(data);
    let compiled = kernel.compile()?;
    let result = compiled.forward(&[&x], vec![[32, 32]])?;
    let data: Vec<bf16> = result.into_iter().next().unwrap().try_into()?;
    let expected: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32).sin()).collect();
    assert_eq!(data, expected);
    Ok(())
}
