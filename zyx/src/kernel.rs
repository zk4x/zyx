use nanoserde::{DeBin, SerBin};
use crate::{
    BLUE, CYAN, DType, GREEN, MAGENTA, Map, RED, RESET, Set, YELLOW,
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    shape::Dim,
    slab::{Slab, SlabId},
    view::View,
};
use std::{fmt::Display, hash::BuildHasherDefault};

pub const IDX_T: DType = DType::U32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub u32);

impl OpId {
    pub fn null() -> Self {
        Self(u32::MAX)
    }

    pub fn is_null(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for OpId {
    fn from(value: usize) -> Self {
        OpId(value as u32)
    }
}

impl From<OpId> for usize {
    fn from(value: OpId) -> usize {
        value.0 as usize
    }
}

impl SlabId for OpId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Kernel {
    pub ops: Slab<OpId, Op>,
    pub order: Vec<OpId>,
}

/*
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Slab<OpId, Op>,
    pub order: Vec<OpId>,
}

// This is SSA representation. All ops return immutable variables.
// The Define op can define mutable variables.
// Variables defined by define op can only be accessed with Load on Store ops,
// using their src and dst fields.
pub enum Op {
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim },
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Scope {
    Global,
    Local,
    Register,
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Scope::Global => "GLOBAL",
            Scope::Local => "LOCAL",
            Scope::Register => "REG",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Op {
    // ops that exist only in kernelizer
    ConstView { value: Constant, view: View },
    LoadView { dtype: DType, view: View },
    StoreView { src: OpId, dtype: DType },
    Reduce { x: OpId, rop: ROp, dims: Vec<Dim> },
    //MergeIndices { x: OpId, y: OpId }, // creates index for merge of loops x and y (i.e. x * y_len + y)
    //PermuteIndices(Vec<OpId>), // Permute for indices, just swapping indices around
    //PadIndex(OpId, isize, isize), // Pad index with padding
    //Unsqueeze { axis: Axis, dim: Dim } // Inserts a new loop at given axis

    // ops that exist in both
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for global stores
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,
}

impl Op {
    fn parameters(&self) -> impl Iterator<Item = OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            Op::StoreView { src, .. } => vec![*src],
            Op::Reduce { x, .. } => vec![*x],
            Op::Store { dst, x, index } => vec![*dst, *x, *index],
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            Op::Binary { x, y, .. } => vec![*x, *y],
            Op::Const(..) => vec![],
            Op::Define { .. } => vec![],
            Op::Load { src, index } => vec![*src, *index],
            Op::Loop { .. } => vec![],
            Op::EndLoop => vec![],
        }
        .into_iter()
    }

    pub fn parameters_mut(&mut self) -> impl Iterator<Item = &mut OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            Op::StoreView { src, .. } => vec![src],
            Op::Reduce { x, .. } => vec![x],
            Op::Store { dst, x, index } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Const(..) => vec![],
            Op::Define { .. } => vec![],
            Op::Load { src, index } => vec![src, index],
            Op::Loop { .. } => vec![],
            Op::EndLoop => vec![],
        }
        .into_iter()
    }

    pub fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
}

impl std::ops::Index<OpId> for Kernel {
    type Output = Op;

    fn index(&self, index: OpId) -> &Self::Output {
        &self.ops[index]
    }
}

impl std::ops::IndexMut<OpId> for Kernel {
    fn index_mut(&mut self, index: OpId) -> &mut Self::Output {
        &mut self.ops[index]
    }
}

impl Kernel {
    pub fn apply_movement(&mut self, func: impl Fn(&mut View)) {
        for op in self.ops.values_mut() {
            match op {
                Op::ConstView { view, .. } | Op::LoadView { view, .. } => {
                    func(view);
                }
                _ => {}
            }
        }
    }

    pub fn debug(&self) {
        println!();
        //println!("Kernel shape {:?}", self.shape);
        let mut indent = String::from(" ");
        for &op_id in &self.order {
            match self.ops[op_id] {
                Op::ConstView { value, ref view } => {
                    println!("{op_id:>3}{indent}{CYAN}CONST VIEW{RESET} {value} {view}")
                }
                Op::LoadView { dtype, ref view } => println!("{op_id:>3}{indent}{CYAN}LOAD VIEW{RESET} {dtype} {view}"),
                Op::StoreView { src, dtype } => println!("{op_id:>3}{indent}{CYAN}STORE VIEW{RESET} {src} {dtype}"),
                Op::Reduce { x, rop, ref dims } => {
                    println!(
                        "{op_id:>3}{indent}{RED}REDUCE{RESET} {} {x}, dims={dims:?}",
                        match rop {
                            ROp::Sum => "SUM",
                            ROp::Max => "MAX",
                        }
                    );
                }
                Op::Define { dtype, scope, ro, len } => {
                    println!("{op_id:>3}{indent}{YELLOW}DEFINE{RESET} {scope} {dtype}, len={len}, ro={ro}");
                }
                Op::Const(x) => println!("{op_id:>3}{indent}{MAGENTA}CONST{RESET} {} {x}", x.dtype()),
                Op::Load { src, index } => println!("{op_id:>3}{indent}{GREEN}LOAD{RESET} p{src}[{index}]"),
                Op::Store { dst, x: src, index } => {
                    println!("{op_id:>3}{indent}{RED}STORE{RESET} p{dst}[{index}] <- {src}")
                }
                Op::Cast { x, dtype } => println!("{op_id:>3}{indent}CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{op_id:>3}{indent}UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{op_id:>3}{indent}BINARY {bop:?} {x} {y}"),
                Op::Loop { dim, scope } => {
                    println!("{op_id:>3}{indent}{BLUE}LOOP{RESET} {scope} dim={dim}");
                    indent += " ";
                }
                Op::EndLoop => {
                    indent.pop();
                    println!("{op_id:>3}{indent}{BLUE}END_LOOP{RESET}");
                }
            }
        }
    }

    pub fn flop_mem_rw(&self) -> (u64, u64, u64) {
        let stores: Vec<OpId> =
            self.ops.iter().filter(|(_, op)| matches!(op, Op::StoreView { .. })).map(|(i, _)| i).collect();

        let mut flop = 0;
        let mut mr = 0;
        let mut mw = 0;
        let mut visited = Map::with_hasher(BuildHasherDefault::new());

        // flop, memory read, memory write, number of elements being processed
        fn recursive(x: OpId, ops: &Slab<OpId, Op>, visited: &mut Map<OpId, u64>) -> (u64, u64, u64) {
            if visited.contains_key(&x) {
                return (0, 0, 0);
            }
            let (f, r, w, n) = match &ops[x] {
                Op::ConstView { view, .. } => (0, 0, 0, view.numel() as u64),
                Op::LoadView { view, .. } => (0, view.original_numel() as u64, 0, view.numel() as u64),
                Op::StoreView { src, .. } => {
                    let (f, r, w) = recursive(*src, ops, visited);
                    let n = visited[src];
                    (f, r, w + n, 0)
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    let (f, r, w) = recursive(*x, ops, visited);
                    let n = visited[x];
                    (f + n, r, w, n)
                }
                Op::Binary { x, y, .. } => {
                    let (fx, rx, wx) = recursive(*x, ops, visited);
                    let (fy, ry, wy) = recursive(*y, ops, visited);
                    let n = visited[x];
                    debug_assert_eq!(n, visited[y]);
                    (fx + fy + n, rx + ry, wx + wy, n)
                }
                Op::Reduce { x, dims, .. } => {
                    let (mut f, r, w) = recursive(*x, ops, visited);
                    let mut n = visited[x];
                    let rd = dims.iter().product::<usize>() as u64;
                    n /= rd;
                    f += n * (rd - 1);
                    (f, r, w, n)
                }
                Op::Const(..) => unreachable!(),
                Op::Define { .. } => unreachable!(),
                Op::Load { .. } => unreachable!(),
                Op::Loop { .. } => unreachable!(),
                Op::EndLoop { .. } => unreachable!(),
                Op::Store { .. } => unreachable!(),
            };
            visited.insert(x, n);
            (f, r, w)
        }

        for store in stores {
            let (f, r, w) = recursive(store, &self.ops, &mut visited);
            flop += f;
            mr += r;
            mw += w;
        }

        //panic!("{}, {}, {}", flop, mr, mw);

        (flop, mr, mw)
    }

    pub fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x, Op::StoreView { .. }))
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x, Op::Reduce { .. }))
    }

    pub fn total_reduce_dim(&self, op: OpId) -> Dim {
        fn recurse(ops: &Slab<OpId, Op>, x: OpId, visited: &mut Set<OpId>) -> Dim {
            if visited.insert(x) {
                let mut prod: Dim = 1;
                if let Op::Reduce { dims, .. } = &ops[x] {
                    prod *= dims.iter().product::<Dim>();
                }
                for param in ops[x].parameters() {
                    prod *= recurse(ops, param, visited);
                }
                return prod;
            }
            return 1;
        }
        let mut visited = Set::default();
        recurse(&self.ops, op, &mut visited)
    }

    pub fn shape(&self) -> Vec<Dim> {
        if self.ops.values().any(|op| matches!(op, Op::Loop { .. })) {
            return self
                .ops
                .values()
                .filter_map(|op| {
                    if let Op::Loop { dim, scope } = op {
                        if matches!(scope, Scope::Global | Scope::Local) {
                            Some(*dim)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
        }
        let mut reduce_dims = 0;
        for &op_id in self.order.iter().rev() {
            match &self[op_id] {
                Op::ConstView { view, .. } | Op::LoadView { view, .. } => {
                    let mut shape = view.shape();
                    for _ in 0..reduce_dims {
                        shape.pop();
                    }
                    return shape;
                }
                Op::Reduce { dims, .. } => {
                    reduce_dims += dims.len();
                }
                _ => {}
            }
        }
        unreachable!()
    }

    pub fn close_loops(&mut self) {
        let mut loop_id = 0;
        for &op_id in &self.order {
            match self[op_id] {
                Op::Loop { .. } => loop_id += 1,
                Op::EndLoop => loop_id -= 1,
                _ => {}
            }
        }
        while loop_id > 0 {
            let op_id = self.ops.push(Op::EndLoop);
            self.order.push(op_id);
            loop_id -= 1;
        }
    }

    /// Find all Reduce ops and put them in a Loop block
    /// Add define ops and add reduce operation as BOp::Add or BOp::Max
    pub fn unfold_reduces(&mut self) {
        // This guarantees we start with innermost reduce op
        let mut reduce_op_ids: Vec<OpId> = self
            .ops
            .iter()
            .filter_map(|(id, op)| {
                if matches!(op, Op::Reduce { .. }) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        while let Some(reduce_op_id) = reduce_op_ids.pop() {
            let Op::Reduce { x, rop, ref dims } = self.ops[reduce_op_id] else { unreachable!() };
            let dims = dims.clone();

            let mut reduce_loop_ops_set = Set::default();
            let mut params = vec![x];
            let mut acc_dtype = None;
            while let Some(param) = params.pop() {
                if reduce_loop_ops_set.insert(param) {
                    params.extend(self.ops[param].parameters());
                    if acc_dtype.is_none() {
                        match self.ops[param] {
                            Op::Define { dtype, .. } => acc_dtype = Some(dtype),
                            Op::ConstView { value, .. } => acc_dtype = Some(value.dtype()),
                            Op::LoadView { dtype, .. } => acc_dtype = Some(dtype),
                            Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                            _ => {}
                        }
                    }
                }
            }
            // Sort reduce loop ops by original order
            let mut reduce_loop_ops = Vec::with_capacity(reduce_loop_ops_set.len());
            for op_id in &self.order {
                if reduce_loop_ops_set.contains(op_id) {
                    reduce_loop_ops.push(*op_id);
                }
            }

            // Remove ops from order
            self.order.retain(|op_id| !reduce_loop_ops_set.contains(op_id));

            // Create new order for loop contents
            let mut order = Vec::new();

            // Add const zero
            let const_zero = self.ops.push(Op::Const(Constant::idx(0)));
            order.push(const_zero);

            // Add accumulator
            let acc_dtype = acc_dtype.unwrap();
            let acc_init_id = self.ops.push(Op::Const(match rop {
                ROp::Sum => acc_dtype.zero_constant(),
                ROp::Max => acc_dtype.min_constant(),
            }));
            order.push(acc_init_id);
            let acc = self.ops.push(Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 });
            order.push(acc);

            // Zero the accumulator
            let zero_acc_op = self.ops.push(Op::Store { dst: acc, x: acc_init_id, index: const_zero });
            order.push(zero_acc_op);

            // Add Loops for the reduce
            for &dim in &dims {
                let loop_id = self.ops.push(Op::Loop { dim, scope: Scope::Register });
                order.push(loop_id);
            }

            // Add body of the reduce loop
            order.extend(reduce_loop_ops);

            // Add reduction operation, load from acc, accumulate, store to acc
            let load_acc = self.ops.push(Op::Load { src: acc, index: const_zero });
            order.push(load_acc);
            let binary_accumulate = self.ops.push(Op::Binary {
                x,
                y: load_acc,
                bop: match rop {
                    ROp::Sum => BOp::Add,
                    ROp::Max => BOp::Maximum,
                },
            });
            order.push(binary_accumulate);
            let store_acc = self.ops.push(Op::Store { dst: acc, x: binary_accumulate, index: const_zero });
            order.push(store_acc);

            // Close the reduce loop
            for _ in 0..dims.len() {
                let endloop_id = self.ops.push(Op::EndLoop);
                order.push(endloop_id);
            }

            // Replace old reduce op with the acc load op
            self.ops[reduce_op_id] = Op::Load { src: acc, index: const_zero };

            // Put all things back in self.order
            let reduce_i = self.order.iter().position(|&op_id| op_id == reduce_op_id).unwrap();
            self.order.splice(reduce_i..reduce_i, order);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn define_globals(&mut self) {
        let mut loads = Vec::new();
        let mut stores = Vec::new();
        for op in self.ops.values() {
            match op {
                Op::LoadView { dtype, view } => loads.push((*dtype, view.original_numel())),
                Op::StoreView { dtype, .. } => stores.push((*dtype, 0)),
                _ => {}
            }
        }
        let n_kernel_args = loads.len() + stores.len();
        let mut order = Vec::with_capacity(n_kernel_args + self.order.len());
        for (dtype, len) in loads {
            order.push(self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: true, len }));
        }
        for (dtype, len) in stores {
            order.push(self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: false, len }));
        }
        order.extend(self.order.drain(..));
        self.order = order;

        #[cfg(debug_assertions)]
        self.verify();
    }

    fn get_loops(&self, op_id: OpId) -> Vec<OpId> {
        let mut loops = Vec::new();
        for &i in &self.order {
            if i == op_id {
                return loops;
            }
            match self.ops[i] {
                Op::Loop { .. } => {
                    loops.push(i);
                }
                Op::EndLoop => {
                    loops.pop();
                }
                _ => {}
            }
        }
        loops
    }

    pub fn unfold_views(&mut self) {
        fn new_op(ops: &mut Slab<OpId, Op>, order: &mut Vec<OpId>, op: Op) -> OpId {
            let op_id = ops.push(op);
            order.push(op_id);
            op_id
        }

        self.define_globals();

        let mut global_args = Vec::new();
        let mut n_loads = 0;
        for &op_id in &self.order {
            if let Op::Define { scope, ro, .. } = self[op_id] {
                if ro {
                    n_loads += 1;
                }
                if scope == Scope::Global {
                    global_args.push(op_id);
                }
            } else {
                break;
            }
        }
        let mut load_id = 0;
        let mut store_id = 0;

        let mut i = 0;
        while i < self.order.len() {
            let op_id = self.order[i];
            match self[op_id] {
                Op::ConstView { value, ref view } => {
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let view = view.clone();
                    let axes = self.get_loops(op_id);

                    //println!("Unfolding view: {view}");
                    let ops = &mut self.ops;

                    let order = &mut Vec::new();
                    let mut pc = new_op(ops, order, Op::Const(Constant::Bool(true)));
                    let constant_zero = new_op(ops, order, Op::Const(Constant::idx(0)));

                    let mut offset;

                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        for (i, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, order, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, order, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, order, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, order, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                new_op(ops, order, Op::Const(Constant::idx(0u64)))
                            } else {
                                axes[i]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, order, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, order, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, order, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, order, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, order, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, order, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, order, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, order, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, order, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, order, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, order, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, order, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let z = new_op(ops, order, Op::Const(value));

                    let dtype = value.dtype();
                    let pcd = new_op(ops, order, Op::Cast { x: pc, dtype });

                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id] = Op::Binary { x: pcd, y: z, bop: BOp::Mul }; // this is now the new op_id

                    // Put newly created ops into correct order
                    self.order.splice(i..i, order.into_iter().map(|x| *x));

                    i += order.len();
                    continue;
                }
                Op::LoadView { dtype, ref view } => {
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1 + a2*st2 + (a3-lp3)*st3 + ...
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let view = view.clone();
                    let axes = self.get_loops(op_id);

                    //println!("Unfolding view: {view}");
                    let ops = &mut self.ops;

                    // We just record the order of all new inserted ops here
                    // and then just insrt this order into current block
                    let order = &mut Vec::new();

                    let mut pc = new_op(ops, order, Op::Const(Constant::Bool(true)));
                    let constant_zero = new_op(ops, order, Op::Const(Constant::idx(0)));
                    let mut offset = constant_zero;
                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        for (i, dim) in inner.iter().enumerate().rev() {
                            let loop_id = if let Some(old_offset) = old_offset {
                                /*let ost_c = new_op(ops, Op::Const(Constant::U32(ost)));
                                ost *= dim.d as u32;
                                let x = new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div });
                                let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
                                new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })*/
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, order, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, order, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, order, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, order, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                constant_zero
                            } else {
                                axes[i]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let padded_loop_id = if dim.lp != 0 {
                                let lp = new_op(ops, order, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, order, Op::Binary { x: loop_id, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, order, Op::Binary { x: loop_id, y: lp, bop: BOp::Add })
                                }
                            } else {
                                loop_id
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, order, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, order, Op::Binary { x: padded_loop_id, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, order, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, order, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, order, Op::Binary { x: loop_id, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, order, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, order, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, order, Op::Binary { x: loop_id, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, order, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let pcu = new_op(ops, order, Op::Cast { x: pc, dtype: IDX_T });
                    let offset = new_op(ops, order, Op::Binary { x: pcu, y: offset, bop: BOp::Mul });

                    let z = new_op(ops, order, Op::Load { src: global_args[load_id], index: offset });

                    let pcd = new_op(ops, order, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id] = Op::Binary { x: pcd, y: z, bop: BOp::Mul };

                    // Put newly created ops into correct block
                    self.order.splice(i..i, order.into_iter().map(|x| *x));

                    load_id += 1;

                    i += order.len();
                    continue;
                }
                Op::StoreView { src, .. } => {
                    let axes = self.get_loops(op_id);
                    let order = &mut Vec::new();
                    let mut index = new_op(&mut self.ops, order, Op::Const(Constant::idx(0u64)));
                    let mut st = 1;

                    let shape = {
                        let mut shape = Vec::new();
                        for &l_op_id in &self.order {
                            if l_op_id == op_id {
                                break;
                            }
                            match self[l_op_id] {
                                Op::Loop { dim, .. } => {
                                    shape.push(dim);
                                }
                                Op::EndLoop => {
                                    shape.pop();
                                }
                                _ => {}
                            }
                        }
                        shape
                    };

                    for (id, d) in shape.iter().enumerate().rev() {
                        let stride = Constant::idx(st as u64);
                        let x = if *d > 1 {
                            axes[id]
                        } else {
                            new_op(&mut self.ops, order, Op::Const(Constant::idx(0)))
                        };
                        let y = new_op(&mut self.ops, order, Op::Const(stride));
                        let x = new_op(&mut self.ops, order, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut self.ops, order, Op::Binary { x, y: index, bop: BOp::Add });
                        st *= d;
                    }

                    self.ops[op_id] = Op::Store { dst: global_args[n_loads + store_id], x: src, index };

                    // Put newly created ops into correct block
                    self.order.splice(i..i, order.into_iter().map(|x| *x));

                    store_id += 1;

                    i += order.len();
                    continue;
                }
                _ => {}
            }
            i += 1;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn unfold_pows(&mut self) {
        let mut i = 0;
        while i < self.order.len() {
            let op_id = self.order[i];
            if let Op::Binary { x, y, bop } = self.ops[op_id] {
                if bop == BOp::Pow {
                    let x = self.ops.push(Op::Unary { x, uop: UOp::Log2 });
                    self.order.insert(i, x);
                    i += 1;
                    let x = self.ops.push(Op::Binary { x, y, bop: BOp::Mul });
                    self.order.insert(i, x);
                    i += 1;
                    self.ops[op_id] = Op::Unary { x, uop: UOp::Exp2 };
                }
            }
            i += 1;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        let mut visited = Set::default();
        // We go backward from Stores and gather all needed ops, but we can't remove Loop and Define ops
        for (op_id, op) in self.ops.iter() {
            if matches!(
                op,
                Op::Store { .. } | Op::Loop { .. } | Op::Define { .. } | Op::EndLoop | Op::StoreView { .. }
            ) {
                params.push(op_id);
            }
        }
        while let Some(op_id) = params.pop() {
            if visited.insert(op_id) {
                params.extend(self[op_id].parameters());
            }
        }
        // Remove ops that are not in visited both from self.ops and self.order
        let ids: Set<OpId> = self.ops.ids().filter(|op_id| !visited.contains(op_id)).collect();
        for &op_id in &ids {
            self.ops.remove(op_id);
        }
        // Remove from self.order and loops
        self.order.retain(|op_id| !ids.contains(op_id));

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn common_subexpression_elimination(&mut self) {
        let mut unique: Vec<Map<Op, OpId>> = Vec::with_capacity(10);
        unique.push(Map::with_capacity_and_hasher(50, BuildHasherDefault::new()));
        let mut unique_loads: Vec<Map<(OpId, OpId), OpId>> = Vec::with_capacity(10);
        unique_loads.push(Map::with_capacity_and_hasher(5, BuildHasherDefault::new()));
        let mut remaps = Map::with_capacity_and_hasher(10, BuildHasherDefault::default());
        for &op_id in &self.order {
            match &self.ops[op_id] {
                Op::Define { .. } => continue,
                Op::Loop { .. } => {
                    unique.push(Map::with_capacity_and_hasher(50, BuildHasherDefault::new()));
                    unique_loads.push(Map::with_capacity_and_hasher(5, BuildHasherDefault::new()));
                }
                Op::EndLoop => {
                    unique.pop();
                    unique_loads.pop();
                }
                &Op::Load { src, index } => {
                    let local_unique = unique_loads.last_mut().unwrap();
                    if let Some(&old_op_id) = local_unique.get(&(src, index)) {
                        remaps.insert(op_id, old_op_id);
                    } else {
                        local_unique.insert((src, index), op_id);
                    }
                }
                &Op::Store { dst, .. } => {
                    let local_unique = unique_loads.last_mut().unwrap();
                    local_unique.retain(|(src, _), _| *src != dst);
                }
                op => {
                    let local_unique = unique.last_mut().unwrap();
                    if let Some(&old_op_id) = local_unique.get(op) {
                        remaps.insert(op_id, old_op_id);
                    } else {
                        local_unique.insert(op.clone(), op_id);
                    }
                }
            }
        }

        // Second pass: remap all operands
        for op in self.ops.values_mut() {
            for param in op.parameters_mut() {
                if let Some(&new_id) = remaps.get(param) {
                    *param = new_id;
                }
            }
        }

        // Third pass: remove duplicated ops from order and ops
        let mut i = 0;
        while i < self.order.len() {
            let op_id = self.order[i];
            if remaps.contains_key(&op_id) {
                self.order.remove(i);
                self.ops.remove(op_id);
            } else {
                i += 1;
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn move_constants_to_beginning(&mut self) {
        let mut first_order = Vec::new();
        let mut second_order = Vec::new();
        let mut defines_done = false;
        for &op_id in &self.order {
            if !defines_done && let Op::Define { .. } = self.ops[op_id] {
                first_order.push(op_id);
                continue;
            } else {
                defines_done = true;
            }
            if let Op::Const(_) = self.ops[op_id] {
                first_order.push(op_id);
                continue;
            }
            second_order.push(op_id);
        }
        self.order = first_order;
        self.order.extend(second_order);

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn constant_folding(&mut self) {
        fn remap(ops: &mut Slab<OpId, Op>, x: OpId, y: OpId) {
            for op in ops.values_mut() {
                for param in op.parameters_mut() {
                    if *param == x {
                        *param = y;
                    }
                }
            }
        }

        for &op_id in &self.order {
            match self.ops[op_id] {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => todo!(),
                Op::Store { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::Load { .. }
                | Op::Loop { .. }
                | Op::EndLoop => {}
                Op::Cast { x, dtype } => {
                    if let Op::Const(cx) = self.ops[x] {
                        self.ops[op_id] = Op::Const(cx.cast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(cx) = self.ops[x] {
                        self.ops[op_id] = Op::Const(cx.unary(uop));
                    }
                }
                Op::Binary { x, y, bop } => match (self.ops[x].clone(), self.ops[y].clone()) {
                    (Op::Const(cx), Op::Const(cy)) => {
                        self.ops[op_id] = Op::Const(Constant::binary(cx, cy, bop));
                    }
                    (Op::Const(cx), _) => match bop {
                        BOp::Add if cx.is_zero() => remap(&mut self.ops, op_id, y),
                        BOp::Sub if cx.is_zero() => self.ops[op_id] = Op::Unary { x: y, uop: UOp::Neg },
                        BOp::Mul if cx.is_zero() => self.ops[op_id] = Op::Const(cx.dtype().zero_constant()),
                        BOp::Mul if cx.is_one() => remap(&mut self.ops, op_id, y),
                        BOp::Mul if cx.is_two() => self.ops[op_id] = Op::Binary { x: y, y, bop: BOp::Add },
                        BOp::Div if cx.is_zero() => self.ops[op_id] = Op::Const(cx.dtype().zero_constant()),
                        BOp::Div if cx.is_one() => self.ops[op_id] = Op::Unary { x: y, uop: UOp::Reciprocal },
                        BOp::Pow if cx.is_one() => self.ops[op_id] = Op::Const(cx.dtype().one_constant()),
                        BOp::BitShiftLeft if cx.is_zero() => remap(&mut self.ops, op_id, y),
                        BOp::BitShiftRight if cx.is_zero() => remap(&mut self.ops, op_id, y),
                        _ => {}
                    },
                    (_, Op::Const(cy)) => match bop {
                        BOp::Add if cy.is_zero() => remap(&mut self.ops, op_id, x),
                        BOp::Sub if cy.is_zero() => remap(&mut self.ops, op_id, x),
                        BOp::Mul if cy.is_zero() => self.ops[op_id] = Op::Const(cy.dtype().zero_constant()),
                        BOp::Mul if cy.is_one() => remap(&mut self.ops, op_id, x),
                        BOp::Mul if cy.is_two() => self.ops[op_id] = Op::Binary { x, y: x, bop: BOp::Add },
                        BOp::Div if cy.is_zero() => panic!("Division by constant zero"),
                        BOp::Div if cy.is_one() => remap(&mut self.ops, op_id, x),
                        BOp::Mod if cy.is_zero() => panic!("Module by constant zero"),
                        BOp::Pow if cy.is_zero() => self.ops[op_id] = Op::Const(cy.dtype().one_constant()),
                        BOp::Pow if cy.is_one() => remap(&mut self.ops, op_id, x),
                        BOp::Pow if cy.is_two() => self.ops[op_id] = Op::Binary { x, y: x, bop: BOp::Mul },
                        BOp::BitShiftLeft if cy.is_zero() => remap(&mut self.ops, op_id, x),
                        BOp::BitShiftRight if cy.is_zero() => remap(&mut self.ops, op_id, x),
                        _ => {}
                    },
                    (x_op, y_op) if x_op == y_op => {
                        match bop {
                            BOp::Div => todo!(), // should be constant 1
                            BOp::Sub => todo!(), // should be constant 0
                            _ => {}
                        }
                    }
                    _ => {}
                },
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn swap_commutative(&mut self) {
        // Tracks whether a value depends on a loop index
        let mut loop_dep = Set::default();

        for &op_id in &self.order {
            let dep = match &self.ops[op_id] {
                Op::Loop { .. } => true,
                Op::Unary { x, .. } | Op::Cast { x, .. } => loop_dep.contains(x),
                Op::Binary { x, y, .. } =>
                    loop_dep.contains(x) || loop_dep.contains(y),
                Op::Load { src, index } =>
                    loop_dep.contains(src) || loop_dep.contains(index),
                Op::Store { x, index, .. } =>
                    loop_dep.contains(x) || loop_dep.contains(index),
                _ => false,
            };

            if dep {
                loop_dep.insert(op_id);
            }

            // Canonicalize commutative ops
            if let Op::Binary { x, y, bop } = &mut self.ops[op_id] {
                if bop.is_commutative() {
                    let xd = loop_dep.contains(x);
                    let yd = loop_dep.contains(y);

                    // Move loop-dependent operand to the RHS
                    if xd && !yd {
                        std::mem::swap(x, y);
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn reorder_commutative(&mut self) {
        // TODO Reorder commutative
        // Iterate:
        //   find a chain of commutative ops like add/sub
        //   reoder by moving loop index last
        // Ops can be added if they are loop invariant or exist in chain of commutative ops
        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn loop_invariant_code_motion(&mut self) {
        let mut i = self.order.len();
        let mut endloop_is = Vec::new();
        while i > 0 {
            i -= 1;
            let loop_id = self.order[i];
            if self.ops[loop_id] == Op::EndLoop {
                endloop_is.push(i);
            }
            if let Op::Loop { .. } = self.ops[loop_id] {
                let mut n_invariant_ops = 0;
                let mut op_ids_in_loop = Set::default();
                op_ids_in_loop.insert(self.order[i]); // Loop op is the primary op that breaks LICM
                for k in i + 1..endloop_is.pop().unwrap() - 1 {
                    let op_id = self.order[k];
                    let op = &self.ops[op_id];
                    if !matches!(
                        op,
                        Op::Store { .. } | Op::Load { .. } | Op::Loop { .. } | Op::EndLoop | Op::Define { .. }
                    ) && op.parameters().all(|op_id| !op_ids_in_loop.contains(&op_id))
                    {
                        let op_id = self.order.remove(k);
                        self.order.insert(i + n_invariant_ops, op_id);
                        n_invariant_ops += 1;
                    } else {
                        op_ids_in_loop.insert(op_id);
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    // Loops that don't contain stores can be deleted
    pub fn delete_empty_loops(&mut self) {
        let mut stack: Vec<(bool, Vec<OpId>)> = Vec::new();
        let mut dead = Set::default();

        for &id in &self.order {
            for s in &mut stack {
                s.1.push(id);
            }
            match self.ops[id] {
                Op::Loop { .. } => stack.push((false, vec![id])),
                Op::Store { .. } => {
                    for s in &mut stack {
                        s.0 = true
                    }
                }
                Op::EndLoop => {
                    let (has_store, ops) = stack.pop().unwrap();
                    if has_store {
                        if let Some(p) = stack.last_mut() {
                            p.1.extend(ops);
                        }
                    } else {
                        dead.extend(ops);
                    }
                }
                _ => {}
            }
        }
        self.ops.retain(|op_id| !dead.contains(op_id));
        self.order.retain(|op_id| !dead.contains(op_id));

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn verify(&self) {
        let valid_ids: Set<OpId> = self.ops.ids().collect();
        let order_ids: Set<OpId> = self.order.iter().copied().collect();
        if valid_ids != order_ids {
            self.debug();
            for &op_id in valid_ids.difference(&order_ids) {
                println!("{op_id} -> {:?}", self.ops[op_id]);
            }
            panic!(
                "ops contain ids that are unused in order: {:?}",
                valid_ids.difference(&order_ids)
            );
        }
        let mut defined: Map<OpId, usize> = Map::default();
        let mut def_loop_depth: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0usize;
        let mut seen_non_global_define = false;
        for (idx, &id) in self.order.iter().enumerate() {
            if defined.contains_key(&id) {
                self.debug();
                panic!("OpId {:?} appears multiple times in order", id);
            }
            let check_ref = |ref_id: OpId| {
                if !valid_ids.contains(&ref_id) {
                    self.debug();
                    panic!("Op {:?} references invalid OpId {:?}", id, ref_id);
                }
                let def_idx = defined.get(&ref_id).unwrap_or_else(|| {
                    self.debug();
                    panic!("Op {:?} uses OpId {:?} before it is defined", id, ref_id);
                });
                let ref_loop_depth = def_loop_depth[&ref_id];
                if ref_loop_depth > loop_depth {
                    self.debug();
                    panic!(
                        "Op {:?} at loop depth {} illegally references OpId {:?} defined at deeper loop depth {}",
                        id, loop_depth, ref_id, ref_loop_depth
                    );
                }
                *def_idx
            };
            match self.ops[id] {
                Op::Const(_) => {}
                Op::Cast { x, .. } => {
                    check_ref(x);
                }
                Op::Unary { x, .. } => {
                    check_ref(x);
                }
                Op::Binary { x, y, .. } => {
                    check_ref(x);
                    check_ref(y);
                }
                Op::Load { src, index } => {
                    check_ref(src);
                    check_ref(index);
                }
                Op::Store { dst, x, index } => {
                    check_ref(dst);
                    check_ref(x);
                    check_ref(index);
                }
                Op::Define { scope, len, ro, .. } => {
                    if ro == false && scope == Scope::Global {
                        debug_assert_eq!(len, 0)
                    }
                    let is_global = matches!(scope, Scope::Global);
                    if is_global {
                        if seen_non_global_define {
                            self.debug();
                            panic!("Global Define {:?} appears after non-global ops", id);
                        }
                    } else {
                        seen_non_global_define = true;
                    }
                }
                Op::Loop { .. } => {
                    seen_non_global_define = true;
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    if loop_depth == 0 {
                        self.debug();
                        panic!("EndLoop {:?} without matching Loop", id);
                    }
                    loop_depth -= 1;
                }
                Op::ConstView { .. } => {}
                Op::LoadView { .. } => {}
                Op::StoreView { src, .. } => {
                    check_ref(src);
                }
                Op::Reduce { x, .. } => {
                    check_ref(x);
                }
            }
            let defines_value = !matches!(self.ops[id], Op::Store { .. } | Op::EndLoop);
            if defines_value {
                defined.insert(id, idx);
                def_loop_depth.insert(id, loop_depth);
            }
        }
        if loop_depth != 0 {
            self.debug();
            panic!("Unclosed Loop: {} loops not terminated", loop_depth);
        }
    }
}

/*
/// Kernel optimization ops that may or may not be applied, that is they are beneficial for some kernels,
/// but hurt other kernels or backends.
impl Kernel {
    // Something like this to upcast local dimensions, that is to do double buffering along local dimensions
    //pub fn upcast_local(&mut self, loop_id: OpId, dim: Dim) {}

    // Split loop into multiple accumulation steps
    //pub fn multi_step_reduce(&mut self) {}

    /// Adds local memory buffer for large reduces
    pub fn grouptop(&mut self, loop_id: OpId, dim: Dim) {}

    // In tinygrad, thread splits workload across multiple CPU threads
    //pub fn thread(&mut self, loop_id)
}*/
