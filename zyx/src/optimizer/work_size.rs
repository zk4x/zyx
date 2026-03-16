use crate::{
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};

// TODO currently this is not good at all. It's too simplistic and does not try hard enough to find good default work sizes
// TODO also add permutation. It may be potentially beneficial if register and global loops are permuted.

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct WorkSizeOpt {
    gws: Vec<Dim>,
    gws_factors: Vec<Vec<[Dim; 2]>>,
    max_local_threads: Dim,
}

impl WorkSizeOpt {
    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> (Self, u32, Vec<u32>) {
        fn divisors(x: usize, limit: usize) -> Vec<usize> {
            debug_assert_ne!(x, 0);
            let mut res = Vec::new();
            let sqrt_x = (x as f64).sqrt() as usize;
            for i in 1..=sqrt_x.min(limit) {
                if x.is_multiple_of(i) {
                    res.push(i);
                }
            }
            for i in (0..res.len()).rev() {
                res.push(x / res[i]);
            }
            res
        }

        let mut gws = kernel.shape();
        // If gws > dev_info.max_global_work_dims.len(), then we join the starting dimensions
        while gws.len() > dev_info.max_global_work_dims.len() {
            let d = gws.remove(0);
            gws[0] *= d;
        }

        let mut gws_factors = Vec::new();
        //println!("max_local_work_dims = {:?}", dev_info.max_local_work_dims);
        for (d, &max_lwd) in gws.iter().copied().zip(&dev_info.max_local_work_dims) {
            let res = divisors(d, max_lwd);

            let mut factors = Vec::new();
            // here factors needs to contain all possible pairs of values in res
            for i in 0..res.len() {
                for j in 0..res.len() {
                    let a = res[i];
                    let b = res[j];
                    if a * b <= d {
                        if a <= max_lwd && b <= 16 {
                            factors.push([a, b]);
                        }
                    }
                }
            }

            gws_factors.push(factors);
        }

        let max_local_threads = dev_info.max_local_threads;
        let max_idx = gws_factors.iter().map(|gd| gd.len() as u32).product();

        //println!("gws={gws:?}, gws_factors={gws_factors:?}, max_local_threads={max_local_threads}");

        (Self { gws, gws_factors, max_local_threads }, max_idx, vec![0])
    }

    // Returns false if this index is invalid
    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let mut value = index as usize;
        let mut result: Vec<usize> = Vec::new();
        for max in self.gws_factors.iter().map(|f| f.len()).rev() {
            result.push(value % max);
            value /= max;
        }
        result.reverse();

        let mut gws = self.gws.clone();
        let mut lws = Vec::new();
        let mut rws = Vec::new();
        for (i, factors) in result.into_iter().zip(&self.gws_factors) {
            let [l, r] = factors[i];
            lws.push(l);
            rws.push(r);
        }
        if lws.iter().product::<Dim>() > self.max_local_threads {
            return false;
        }
        for ((g, l), r) in gws.iter_mut().zip(&lws).zip(&rws) {
            if !g.is_multiple_of(l * r) {
                return false;
            }
            *g /= l * r;
        }

        kernel.debug();

        debug_assert_eq!(gws.len(), lws.len());
        debug_assert_eq!(lws.len(), rws.len());

        //gws = vec![64, 128];
        //lws = vec![8, 4];
        //rws = vec![2, 2];

        let mut dim_ids = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let Op::Index { scope, axis, .. } = kernel.ops[op_id].op {
                debug_assert_eq!(scope, Scope::Global);
                dim_ids.push((op_id, axis));
            }
            op_id = kernel.next_op(op_id);
        }
        dim_ids.sort_by_key(|x| x.1);
        let dim_ids: Vec<OpId> = dim_ids.into_iter().map(|x| x.0).collect();

        let mut axis = 0;
        for (((dim_id, g), l), r) in dim_ids.into_iter().zip(gws.into_iter()).zip(lws).zip(rws) {
            let mut splits = Vec::new();
            splits.push(Op::Index { len: g, scope: Scope::Global, axis });
            splits.push(Op::Index { len: l, scope: Scope::Local, axis });
            splits.push(Op::Loop { len: r, axis });
            kernel.split_dim(dim_id, splits);
            axis += 1;
        }

        true
    }
}

impl Kernel {
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
                Op::Index { len, scope, axis } => {}
                Op::Loop { len, axis } => {
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
