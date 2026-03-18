use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};

impl Kernel {
    /// Reshapes, (splits or merges) reduce from original into new_dims
    pub fn recursively_reshape(
        &mut self,
        op_id: OpId,
        n_old_dims: usize,
        new_dims: &[Dim],
        visited: &mut Set<OpId>,
        skip_last: usize,
    ) {
        if !visited.insert(op_id) {
            return;
        }
        match &mut self.ops[op_id].op {
            Op::LoadView(x) => {
                let rank = x.1.rank();
                x.1.reshape(rank - skip_last - n_old_dims..rank - skip_last, new_dims);
            }
            Op::ConstView(x) => {
                let rank = x.1.rank();
                x.1.reshape(rank - skip_last - n_old_dims..rank - skip_last, new_dims);
            }
            &mut Op::Reduce { x, n_axes, .. } => {
                let skip_last = skip_last + n_axes;
                self.recursively_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            &mut Op::Cast { x, .. } | &mut Op::Unary { x, .. } => {
                self.recursively_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            &mut Op::Binary { x, y, .. } => {
                self.recursively_reshape(x, n_old_dims, new_dims, visited, skip_last);
                self.recursively_reshape(y, n_old_dims, new_dims, visited, skip_last);
            }
            _ => {}
        }
    }

    /// Get last op in the given loop scope
    pub fn get_last_dim_op(&self, loop_id: OpId) -> OpId {
        match self.ops[loop_id].op {
            Op::Index { .. } => return self.tail,
            Op::Loop { .. } => {}
            _ => unreachable!(),
        }
        let mut loop_depth = 0;
        let mut op_id = loop_id;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        return op_id;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        op_id
    }

    /// Splits dim (index or loop) into multiple indices or loops
    pub fn split_dim(&mut self, dim_id: OpId, mut splits: Vec<Op>) {
        println!("splitting dim_id={dim_id}, splits={splits:?}");
        #[cfg(debug_assertions)]
        {
            let mut dim = 1;
            for op in splits.iter() {
                match op {
                    Op::Loop { len, .. } | Op::Index { len, .. } => dim *= len,
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            match self.ops[dim_id].op {
                Op::Index { len, .. } | Op::Loop { len, .. } => debug_assert_eq!(len, dim),
                _ => {}
            }
        }

        let last_dim_op = self.get_last_dim_op(dim_id);
        for op in &splits {
            if matches!(op, Op::Loop { .. }) {
                self.insert_after(last_dim_op, Op::EndLoop);
            }
        }

        // 12 - > 2, 2, 4
        // 0..12 - > 0..2 * st + 2 * st +  4 * st

        // Get strides
        let mut strides = Vec::new();
        let mut st = 1;
        for op in splits.iter().rev() {
            strides.push(st);
            match op {
                Op::Loop { len, .. } | Op::Index { len, .. } => st *= len,
                _ => unreachable!(),
            }
        }
        strides.reverse();
        strides.pop(); // skip stride 1
        let last_op = splits.pop().unwrap();

        // Insert splits
        let mut acc = self.insert_before(dim_id, Op::Const(Constant::idx(0)));
        for (&st, op) in strides.iter().zip(splits) {
            let x = self.insert_before(dim_id, Op::Const(Constant::idx(st as u64)));
            let y = self.insert_before(dim_id, op);
            acc = self.insert_before(dim_id, Op::Mad { x, y, z: acc });
        }

        // Replace previous op
        let y = self.insert_before(dim_id, last_op);
        self.ops[dim_id].op = Op::Binary { x: acc, y, bop: BOp::Add };
    }
}
