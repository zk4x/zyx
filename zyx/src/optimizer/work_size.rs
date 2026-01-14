use crate::{
    backend::DeviceInfo,
    kernel::{Kernel, Op, Scope},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};

// TODO currently this is not good at all. It's too simplistic and does not try hard enough to find good work sizes

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
                        if a <= max_lwd && b <= 64 {
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
        (Self { gws, gws_factors, max_local_threads }, max_idx, vec![0, 1, 2])
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

        let shape: Vec<Dim> = gws.iter().chain(&lws).chain(&rws).copied().collect();
        let n = kernel.shape().len();

        //if n < 4 && !kernel.is_reshape_contiguous(0..n, &shape) { return false; }

        kernel.apply_movement(|view| view.reshape(0..n, &shape));

        let mut next = kernel.start;
        {
            for &dim in rws.iter().rev() {
                next = kernel.ops.push(Op::Loop { next, dim, scope: Scope::Register });
            }
            for &dim in lws.iter().rev() {
                next = kernel.ops.push(Op::Loop { next, dim, scope: Scope::Local });
            }
            for &dim in gws.iter().rev() {
                next = kernel.ops.push(Op::Loop { next, dim, scope: Scope::Global });
            }
        };
        true
    }
}
