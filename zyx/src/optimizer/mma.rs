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
    DType,
    graph::BOp,
    kernel::{Kernel, Op, OpId, Scope},
};

impl Kernel {
    /// Turns inner loops into tensor core instructions if possible
    pub fn fuse_mma(&mut self) {
        self.swap_commutative();
        self.loop_invariant_code_motion();
        self.move_constants_to_beginning();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();

        self.debug();

        let mut op_id = self.head;
        while !op_id.is_null() {
            if self.is_mma_store(op_id) {
                println!("Found MMA store at {op_id}");
            }
            op_id = self.next_op(op_id);
        }
    }

    pub fn is_mma_store(&self, store_op_id: OpId) -> bool {
        use DType::*; use Op::*; use BOp::*; use Scope::*;
        let &Store { dst, x: store_x, index, vlen: 1 } = self.at(store_op_id) else { return false };
        {
            // Check that the index is looping over 4x2 element accumulator
            let &Binary { x, y, bop: Add } = self.at(index) else { return false; };
            let &Loop { dim: 2, scope: Register } = self.at(y) else { return false; };
            let &Binary { x, y, bop: Add } = self.at(x) else { return false; };
            if x != y { return false; } // it's loop + loop for stride of 2
            let &Loop { dim: 4, scope: Register } = self.at(x) else { return false; };
        }
        // The destination must be accumulator
        let &Define { dtype: F32, scope: Register, ro: false, len: 8 } = self.at(dst) else { return false; };
        // It must store addition
        let &Binary { x, y, bop: Add } = self.at(store_x) else { return false };
        // add must add f32 casted mul result to a load from an accumulator
        let &Cast { x, dtype: F32 } = self.at(x) else { return false; };
        let &Load { src, index, vlen: 1 } = self.at(y) else { return false; };
        {
            // Check the index
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
        }
        // y load
        let &Load { src, index, vlen: 1 } = self.at(y) else { return false; };
        // src of y load must be global arg
        let &Define { dtype: F16, scope: Scope::Global, ro: true, len: _ } = self.at(src) else { return false; };
        {
            // Check the index
        }
        return true;
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
