use crate::{
    Set,
    kernel::{Kernel, Op, OpId},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopSplitOpt {
    // For each reduction op, store possible split configurations
    // [reduction_op_index][split_configuration][split_dimensions]
    reduction_splits: Vec<Vec<Vec<Dim>>>,
}

impl LoopSplitOpt {
    pub fn new(kernel: &Kernel) -> (Self, u32, Vec<u32>) {
        //return (LoopSplitOpt { reduction_splits: Vec::new() }, 10);

        let mut reduction_splits = Vec::new();

        // Find all reduction ops
        for (op_id, op) in kernel.iter_unordered() {
            if let Op::Reduce { n_axes, .. } = *op {
                // Generate all valid splits for these dimensions
                // Calculate the total product of all dimensions
                let mut reduce_dims = kernel.reduce_dims(op_id);
                reduce_dims.truncate(n_axes);
                let total_product: Dim = reduce_dims.iter().product();

                let mut options: Vec<Vec<Dim>> = Vec::new();

                // Add original
                options.push(reduce_dims);

                let defaults = [16, 8];
                for d in defaults.into_iter().chain((2..=16).filter(|x| !defaults.contains(x))) {
                    if total_product.is_multiple_of(d) {
                        options.push(vec![total_product / d, d]);
                    }
                }

                reduction_splits.push(options);
            }
        }

        let max_index = reduction_splits.iter().map(|splits| splits.len() as u32).product::<u32>();

        //println!("reduction_splits={reduction_splits:?}");
        (LoopSplitOpt { reduction_splits }, max_index, vec![1, 2])
    }

    pub fn apply_optimization(&self, mut index: u32, kernel: &mut Kernel) -> bool {
        // Check if we have any reduction splits
        if self.reduction_splits.is_empty() {
            // No reduction operations found, nothing to optimize
            return true;
        }

        let reduce_ops: Vec<OpId> =
            kernel.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).map(|(op_id, _)| op_id).collect();

        for (i, choices) in self.reduction_splits.iter().enumerate() {
            let n = choices.len() as u32;
            let idx = index % n;
            index /= n;

            let Some(&reduce_id) = reduce_ops.get(i) else { return false };

            let new_dims: &[Dim] = &self.reduction_splits[i][idx as usize];
            let Op::Reduce { x, n_axes, .. } = &mut kernel.ops[reduce_id].op else { return false };
            let x = *x;
            let n = *n_axes;
            *n_axes = new_dims.len();

            let mut visited = Set::default();
            kernel.recursively_apply_reshape(x, n, new_dims, &mut visited, 0);
        }
        true
    }
}

impl Kernel {
    /// Reshapes, (splits or merges) reduce from original into new_dims
    fn recursively_apply_reshape(
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
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            &mut Op::Cast { x, .. } | &mut Op::Unary { x, .. } => {
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            &mut Op::Binary { x, y, .. } => {
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
                self.recursively_apply_reshape(y, n_old_dims, new_dims, visited, skip_last);
            }
            _ => {}
        }
    }
}
