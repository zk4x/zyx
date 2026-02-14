// For WMMA
// We need to have specific size of inner and local work sizes such
//
// 1. work_size.rs, the loads are permuted such that we can use tensor cores by putting
// the local and register loops after the inner reduce loop (in strides).
//
// 2. now we can replace the inner register loops with the tensor core instructions directly,
// nothing else to do, just verify the strides :D
//

use crate::{
    DType, graph::BOp, kernel::{Kernel, Op, OpId, Scope}, shape::Dim
};

impl Kernel {
    /// Finds loop trifectas and if possible, LICM moves these instructions before those loops
    /// and converts them into MMA instructions.
    pub fn fuse_mma(&mut self) {
        use Op::*;
        use Scope::*;
        self.swap_commutative();
        self.reassociate_commutative();
        self.loop_invariant_code_motion();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.move_constants_to_beginning();

        self.debug();

        let mut op_id = self.tail;
        let mut loop_dims = Vec::new();
        while !op_id.is_null() {
            /*if self.is_mma_store(op_id) {
                println!("Found MMA store at {op_id}");
            }*/
            match self.ops[op_id].op {
                Loop { dim, scope } => {
                    if scope == Register {
                        loop_dims.push(dim);
                        if loop_dims == vec![2, 4, 16] {
                            println!("Found the loop trifecta");
                            self.get_mma_info(op_id);
                        }
                    }
                }
                EndLoop => {
                    loop_dims.clear();
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
    }

    fn get_mma_info(&self, k_loop_id: OpId) -> bool {
        use Op::*; use Scope::*;

        let mut op_id = k_loop_id;
        //let mut indices: Vec<_> = Vec::new();
        let mut n_endloops = 0;

        let mut loop_ids = Vec::new();

        while !op_id.is_null() {
            println!("{op_id} -> {:?}", self.ops[op_id].op);
            match &self.ops[op_id].op {
                Cast { x, dtype } => todo!(),
                Unary { x, uop } => todo!(),
                Binary { x, y, bop } => todo!(),
                Const(constant) => todo!(),
                Define { dtype, scope, ro, len } => todo!(),
                Store { dst, x, index, vlen } => todo!(),
                Load { src, index, vlen } => todo!(),
                &Loop { dim, scope } => {
                    debug_assert_eq!(scope, Register);
                    loop_ids.push((op_id, dim));
                }
                EndLoop => {
                    n_endloops += 1;
                    if n_endloops == 3 {
                        return true;
                    }
                }
                Mad { x, y, z } => todo!(),
                MMA { m, n, k, c, a, b } => todo!(),
                Vectorize { ops } => todo!(),
                Devectorize { vec, idx } => todo!(),
                ConstView(_) => todo!(),
                LoadView(_) => todo!(),
                StoreView { src, dtype } => todo!(),
                Reduce { x, rop, n_axes } => todo!(),
            }

            op_id = self.next_op(op_id);
        }
        true
    }

    pub fn is_mma_store(&self, store_op_id: OpId) -> bool {
        use DType::*; use Op::*; use BOp::*; use Scope::*;
        let &Store { dst, x, index, vlen: 1 } = self.at(store_op_id) else { return false };
        {
            // Check that the index is looping over 4x2 element accumulator
            /*let &Binary { x, y, bop: Add } = self.at(index) else { return false; };
            let &Loop { dim: 2, scope: Register } = self.at(y) else { return false; };
            let &Binary { x, y, bop: Add } = self.at(x) else { return false; };
            if x != y { return false; } // it's loop + loop for stride of 2
            let &Loop { dim: 4, scope: Register } = self.at(x) else { return false; };*/
        }
        // The destination must be accumulator
        let &Define { dtype: F32, scope: Register, ro: false, len: 8 } = self.at(dst) else { return false; };
        // It must store addition
        let &Binary { x: load_x, y, bop: Add } = self.at(x) else { return false };
        // add must add f32 casted mul result to a load from an accumulator
        let &Cast { x, dtype: F32 } = self.at(y) else { return false; };
        let &Load { src, index, vlen: 1 } = self.at(load_x) else { return false; };
        {
            // Check the index
            let loop_strides = self.get_strides(index);
            println!("loop_strides={loop_strides:?}");
        }
        // right side must be f32 accumulator with length of 8
        let &Define { dtype: F32, scope: Register, ro: false, len: 8 } = self.at(src) else { return false; };
        // x must be multiply
        let &Binary { x, y, bop: Mul } = self.at(x) else { return false; };
        // inputs to that multiply must be two loads
        // x load
        let &Load { src, index, vlen: 1 } = self.at(x) else { return false; };
        // src of x load must be global arg
        let &Define { dtype: F16, scope: Scope::Global, ro: true, len: _ } = self.at(src) else { return false; };
        {
            // Check the index
            let loop_strides = self.get_strides(index);
            println!("loop_strides={loop_strides:?}");
        }
        // y load
        let &Load { src, index, vlen: 1 } = self.at(y) else { return false; };
        // src of y load must be global arg
        let &Define { dtype: F16, scope: Scope::Global, ro: true, len: _ } = self.at(src) else { return false; };
        {
            // Check the index
            let loop_strides = self.get_strides(index);
            println!("loop_strides={loop_strides:?}");
        }
        return true;
    }

    /// Returns vector of strides
    fn get_strides(&self, index_id: OpId) -> Vec<(OpId, Dim)> {
        use Op::*;
        let mut loop_strides = Vec::new();

        let mut mad_id = index_id;
        while let &Mad { x, y, z } = self.at(mad_id) {
            if let Loop { .. } = self.at(x) {
                let Const(c) = self.at(y) else { unreachable!() };
                loop_strides.push((x, c.as_dim()));
                mad_id = z;
            } else if let Loop { .. } = self.at(y) {
                let Const(c) = self.at(x) else { unreachable!() };
                loop_strides.push((y, c.as_dim()));
                mad_id = z;
            } else if let Loop { .. } = self.at(z) {
                loop_strides.push((z, 1));
                mad_id = x;
            }
            println!("mad_id={mad_id} {:?}", self.ops[mad_id].op);
        }

        loop_strides
    }
}

/*
24   LOOP REG dim=16    0..=15
92     LOOP REG dim=8    0..=7
34       MAD 7 32 12    0..=1032192
37       MAD 9 35 34    0..=1040384
40       MAD 92 38 37    0..=1047552
43       MAD 23 41 40    0..=1048560
46       MAD 24 44 43    0..=1048575
50       LOAD p49[46] len=1    0..=1048575
56       MAD 8 54 12    0..=992
59       MAD 10 44 56    0..=1023
62       MAD 23 32 59    0..=1033215
65       MAD 24 38 62    0..=1048575
69       LOAD p68[65] len=1    0..=1048575
 2       BINARY Mul 50 69
 3       CAST 2 F32
25       LOAD p21[92] len=1    0..=7
26       BINARY Add 3 25
27       STORE p21[92] <- 26 len=1    0..=7
104     END_LOOP
28   END_LOOP

34       MAD 7 32 12    0..=1032192
37       MAD 9 35 34    0..=1040384
43       MAD 23 41 40    0..=1048560
50       LOAD p49[43] vlen=4    0..=1048575 // Vector load of 4 or whatever is needed
56       MAD 8 54 12    0..=992
59       MAD 10 44 56    0..=1023
62       MAD 23 32 59    0..=1033215
65       MAD 24 38 62    0..=1048575
69       LOAD p68[65] vlen=2    0..=1048575 // Vector load of 2 or whatever is needed
25       LOAD p21[92] vlen=8    0..=7
26       MMA c=25, a=50, b=69
27       STORE p21[92] <- 26 len=8    0..=7
*/
