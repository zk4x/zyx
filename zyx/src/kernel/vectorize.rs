// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Map, Set,
    kernel::{Kernel, MemLayout, Op, OpId},
    shape::Dim,
};

#[derive(Debug)]
struct LoadInfo {
    id: OpId,
    index: OpId,
}

impl Kernel {
    #[allow(unused)]
    pub fn vectorize(&mut self) {
        self.vectorize_loads();
        self.vectorize_ops();
        self.vectorize_stores();
    }

    pub fn vectorize_loads(&mut self) {
        // TODO for now this function ignores aliasing of stores and loads.
        // So later we need to make sure there are no aliasing issues

        let mut op_id = self.head;
        // Map: src id -> LoadInfo
        let mut loads: Vec<Map<OpId, Vec<LoadInfo>>> = Vec::new();
        loads.push(Map::default());
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loads.push(Map::default());
                }
                Op::Load { src, index, layout } => {
                    if layout == MemLayout::Scalar {
                        loads
                            .last_mut()
                            .unwrap()
                            .entry(src)
                            .and_modify(|e| e.push(LoadInfo { id: op_id, index }))
                            .or_insert_with(|| vec![LoadInfo { id: op_id, index }]);
                    }
                }
                Op::EndLoop => self.fun_name(&mut loads),
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        self.fun_name(&mut loads);
    }

    fn fun_name(&mut self, loads: &mut Vec<Map<OpId, Vec<LoadInfo>>>) {
        if let Some(loads) = loads.pop() {
            for (src, mut loads) in loads {
                if ![2, 4, 8, 16, 32].contains(&loads.len()) {
                    continue;
                }

                /*for load in &loads[1..] {
                    println!("{load:?} {:?}", self.get_strides(load.index));
                }*/

                loads.sort_unstable_by_key(|x| self.get_strides(x.index).len());

                // Get the base index and check offsets
                let mut base_index = None;
                let vec_len = loads.len() as Dim;
                for (base_idx, (_, vl)) in self.get_strides(loads[0].index) {
                    if vl != vec_len {
                        continue;
                    }
                    //println!("base_idx={base_idx}");
                    let mut offsets: Set<Dim> = (0..vec_len).collect();

                    if loads[1..].iter().all(|x| {
                        let strides = self.get_strides(x.index);
                        //println!("got strides={strides:?}");
                        strides.iter().any(|(&idx, (_, st))| idx == base_idx && *st == vec_len)
                            && strides.iter().any(|(&idx, (_, st))| idx.is_null() && offsets.remove(st))
                    }) {
                        if offsets.remove(&0) {
                            base_index = Some(base_idx);
                            break;
                        }
                    }
                }

                if base_index.is_some() {
                    println!("Can vectorize {} loads", loads.len());
                } else {
                    println!("Cannot vectorize loads - non-contiguous offsets");
                }

                // Now that we know offsets are continues, we can replace the loads with single vectorized load
                if let Some(base_index) = base_index {
                    let vload = self.insert_before(
                        loads[0].id,
                        Op::Load { src, index: base_index, layout: MemLayout::Vector(vec_len as u8) },
                    );
                    for (idx, load) in loads.iter().enumerate() {
                        self.ops[load.id].op = Op::Devectorize { vec: vload, idx };
                    }
                }
            }
        }
    }

    pub fn vectorize_ops(&mut self) {
        // Find all Devectorize ops and group them by their source vector
        let mut devec_map: Map<OpId, Vec<(OpId, usize)>> = Map::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            match &self.ops[op_id].op {
                Op::Devectorize { vec, idx } => {
                    devec_map.entry(*vec).or_insert_with(Vec::new).push((*vec, *idx));
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // For each vectorized load, check if we can vectorize ops after its devectorize
        for (vec_id, _devec_list) in &devec_map {
            // Find all Devectorize ops that use this vector, with their indices
            let mut devec_ops: Vec<(OpId, usize)> = Vec::new();
            let mut check_id = self.next_op(*vec_id);

            while let Op::Devectorize { vec: v, idx } = &self.ops[check_id].op {
                if v == vec_id {
                    devec_ops.push((check_id, *idx));
                    check_id = self.next_op(check_id);
                } else {
                    break;
                }
            }

            // If no devectorize ops, skip
            if devec_ops.is_empty() {
                continue;
            }

            // Find the last devectorize op and check ops after it
            let last_devec = devec_ops.last().unwrap();
            let mut check_id = self.next_op(last_devec.0);

            // Collect all elementwise ops after the last devectorize
            let mut elementwise_ops: Vec<OpId> = Vec::new();

            while !check_id.is_null() {
                match &self.ops[check_id].op {
                    Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Mad { .. } => {
                        elementwise_ops.push(check_id);
                        check_id = self.next_op(check_id);
                    }
                    Op::Load { .. } | Op::Store { .. } | Op::Vectorize { .. } => {
                        // End of elementwise chain
                        break;
                    }
                    Op::Devectorize { .. } => {
                        // Another devectorize, skip it
                        check_id = self.next_op(check_id);
                    }
                    _ => {
                        // Other ops break the chain
                        break;
                    }
                }
            }

            // If we have elementwise ops, check if they're all the same type and can be vectorized
            if !elementwise_ops.is_empty() {
                // Check if all ops are the same type (Unary, Binary, Cast, or Mad)
                let mut first_type = None;

                for &op_id in &elementwise_ops {
                    let op_type = match &self.ops[op_id].op {
                        Op::Unary { .. } => OpType::Unary,
                        Op::Binary { .. } => OpType::Binary,
                        Op::Cast { .. } => OpType::Cast,
                        Op::Mad { .. } => OpType::Mad,
                        _ => {
                            // Mixed op types, can't vectorize
                            elementwise_ops.clear();
                            break;
                        }
                    };

                    if let Some(ref first) = first_type {
                        if op_type != *first {
                            // Different op types, can't vectorize
                            elementwise_ops.clear();
                            break;
                        }
                    } else {
                        first_type = Some(op_type);
                    }
                }

                if elementwise_ops.is_empty() {
                    continue;
                }

                // All ops are the same type and have the same configuration
                // Create a single vectorized version by applying the op to the whole vector
                let mut new_vec = *vec_id;

                // Create ONE vectorized op (not one per elementwise op)
                let new_op_id = self.insert_before(
                    self.next_op(new_vec),
                    match &self.ops[elementwise_ops[0]].op {
                        Op::Unary { x, uop } => Op::Unary { x: new_vec, uop: *uop },
                        Op::Binary { x, y, bop } => Op::Binary { x: new_vec, y: *x, bop: *bop },
                        Op::Cast { x, dtype } => Op::Cast { x: new_vec, dtype: *dtype },
                        Op::Mad { x, y, z } => Op::Mad { x: new_vec, y: *y, z: *z },
                        _ => unreachable!(),
                    },
                );

                // Replace each elementwise op with a devectorize of the new vectorized result
                // elementwise_ops[i] should become Op::Devectorize { vec: new_op_id, idx: i }
                for (i, &elem_id) in elementwise_ops.iter().enumerate() {
                    // Replace the elementwise op with a devectorize of the new vectorized result
                    self.ops[elem_id].op = Op::Devectorize { vec: new_op_id, idx: i };
                }
            }
        }
    }

    pub fn vectorize_stores(&mut self) {
        // Find all Devectorize ops and group them by their source vector
        let mut devec_map: Map<OpId, Vec<(OpId, usize)>> = Map::default();
        
        let mut op_id = self.head;
        while !op_id.is_null() {
            match &self.ops[op_id].op {
                Op::Devectorize { vec, idx } => {
                    devec_map.entry(*vec).or_insert_with(Vec::new).push((*vec, *idx));
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        
        // For each vectorized load, check if the stores after its devectorize can be combined
        for (vec_id, _devec_list) in &devec_map {
            let mut devec_ops: Vec<(OpId, usize)> = Vec::new();
            let mut check_id = self.next_op(*vec_id);
            
            while let Op::Devectorize { vec: v, idx } = &self.ops[check_id].op {
                if v == vec_id {
                    devec_ops.push((check_id, *idx));
                    check_id = self.next_op(check_id);
                } else {
                    break;
                }
            }
            
            if devec_ops.is_empty() {
                continue;
            }
            
            // Find the last devectorize and check stores after it
            let last_devec = devec_ops.last().unwrap();
            let mut check_id = self.next_op(last_devec.0);
            
            // Collect all stores that use devectorized values from this vector
            let mut stores_to_combine: Vec<(OpId, usize)> = Vec::new();
            
            while !check_id.is_null() {
                match &self.ops[check_id].op {
                    Op::Store { dst, x, index, layout } => {
                        // Check if this store uses a devectorized value from vec_id
                        for (devec_id, devec_idx) in &devec_ops {
                            if let Op::Devectorize { vec: v, idx } = &self.ops[*devec_id].op {
                                if v == vec_id && *idx == *devec_idx {
                                    // This store uses a devectorized value
                                    stores_to_combine.push((check_id, *devec_idx));
                                    check_id = self.next_op(check_id);
                                    break;
                                }
                            }
                        }
                        if !stores_to_combine.is_empty() {
                            continue;
                        }
                        check_id = self.next_op(check_id);
                    }
                    Op::EndLoop | Op::Load { .. } | Op::Vectorize { .. } | Op::Devectorize { .. } => {
                        break;
                    }
                    _ => {
                        break;
                    }
                }
            }
            
            // If we have stores for all devectorized values, combine them into a single vector store
            if stores_to_combine.len() == devec_ops.len() {
                // Get the store details from the first store
                if let Some((first_store_id, _)) = stores_to_combine.first() {
                    if let Op::Store { dst, x: _, index: store_idx, layout: _ } = &self.ops[*first_store_id].op {
                        // Create a single vector store using the vectorized result
                        let vec_len = devec_ops.len() as u8;
                        
                        self.insert_before(
                            *first_store_id,
                            Op::Store {
                                dst: *dst,
                                x: *vec_id,  // Use the vectorized result directly
                                index: *store_idx,
                                layout: MemLayout::Vector(vec_len),
                            },
                        );
                        
                        // Remove the other stores (keep first one)
                        for (store_id, _) in &stores_to_combine[1..] {
                            // Mark as dead - DCE will clean up
                            self.ops[*store_id].op = Op::Store { dst: OpId::NULL, x: OpId::NULL, index: OpId::NULL, layout: MemLayout::Scalar };
                        }
                    }
                }
            }
        }
    }

    pub const fn vectorize_stores(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpType {
    Unary,
    Binary,
    Cast,
    Mad,
}
