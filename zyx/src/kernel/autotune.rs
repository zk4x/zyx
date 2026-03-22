use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Kernel {}

impl Kernel {
    // ---------------- Kernel API ----------------
    pub fn get_cost(&self) -> u64 {
        0
    }

    pub fn unroll_loops(&self, unroll_factor: u32) -> Self {
        self.clone()
    }

    pub fn constant_folding(&self) -> Self {
        self.clone()
    }

    pub fn licm(&self) -> Self {
        self.clone()
    }

    pub fn split_loop(&self, loop_id: u32, split_dim: u32) -> Self {
        self.clone()
    }

    pub fn unroll_loops_params(&self) -> Vec<Vec<u32>> {
        vec![vec![2, 4, 8]]
    }

    pub fn split_loop_params(&self) -> Vec<Vec<u32>> {
        vec![vec![0, 1], vec![2, 4]]
    }

    pub fn hash_kernel(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    // ---------------- Cartesian product helper ----------------
    fn cartesian_product(params: &[Vec<u32>]) -> Vec<Vec<u32>> {
        params.iter().fold(vec![vec![]], |acc, p| {
            let mut res = vec![];
            for prefix in &acc {
                for &v in p {
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(v);
                    res.push(new_prefix);
                }
            }
            res
        })
    }

    // ---------------- Autotune with A* ----------------
    pub fn autotune_astar(&self, mut cost_iters: u32) -> Self {
        #[derive(Clone, Copy)]
        enum PassKind {
            UnrollLoops,
            ConstantFolding,
            Licm,
            SplitLoop,
        }
        const PASSES: &[PassKind] = &[
            PassKind::UnrollLoops,
            PassKind::ConstantFolding,
            PassKind::Licm,
            PassKind::SplitLoop,
        ];

        let mut best = self.clone();
        let mut best_cost = best.get_cost();
        cost_iters = cost_iters.saturating_sub(1);

        // min-heap by cost
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((best_cost, best.clone())));

        // track seen kernels by hash
        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(best.hash_kernel());

        // helper closure for inserting a child
        let mut try_insert = |child: Kernel| {
            if cost_iters == 0 {
                return;
            }
            let h = child.hash_kernel();
            if visited.contains(&h) {
                return;
            }

            let cost = child.get_cost();
            cost_iters -= 1;

            visited.insert(h);
            if cost < best_cost {
                best = child.clone();
                best_cost = cost;
            }

            heap.push(Reverse((cost, child)));
        };

        while cost_iters > 0 && !heap.is_empty() {
            let Reverse((_curr_cost, curr_kernel)) = heap.pop().unwrap();

            for &pass in PASSES {
                match pass {
                    PassKind::UnrollLoops => {
                        for p_comb in Self::cartesian_product(&curr_kernel.unroll_loops_params()) {
                            try_insert(curr_kernel.unroll_loops(p_comb[0]));
                        }
                    }
                    PassKind::ConstantFolding => {
                        try_insert(curr_kernel.constant_folding());
                    }
                    PassKind::Licm => {
                        try_insert(curr_kernel.licm());
                    }
                    PassKind::SplitLoop => {
                        for p_comb in Self::cartesian_product(&curr_kernel.split_loop_params()) {
                            try_insert(curr_kernel.split_loop(p_comb[0], p_comb[1]));
                        }
                    }
                }
                if cost_iters == 0 {
                    break;
                }
            }
        }

        best
    }
}
