use nanoserde::{DeBin, SerBin};

use crate::{
    BLUE, CYAN, DType, GREEN, MAGENTA, Map, RED, RESET, Set, YELLOW,
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    shape::Dim,
    slab::{Slab, SlabId},
    view::View,
};
use std::{
    fmt::Display,
    hash::BuildHasherDefault,
};

pub const IDX_T: DType = DType::U32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(u32);

impl OpId {
    pub fn null() -> Self {
        Self(u32::MAX)
    }

    pub fn is_null(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Slab<OpId, Op>,
    pub order: Vec<OpId>,
}

// This is SSA representation. All ops return immutable variables.
// The Define op can define mutable variables.
// Variables defined by define op can only be accessed with Load on Store ops,
// using their src and dst fields.
/*pub enum Op {
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope, ops: Vec<OpId> },
}*/

impl SerBin for Kernel {
    fn ser_bin(&self, _output: &mut Vec<u8>) {
        todo!()
    }
}

impl DeBin for Kernel {
    fn de_bin(_offset: &mut usize, _bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        todo!()
    }
}

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
    //Null,

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope, ops: Vec<OpId> },
    //EndLoop,
}

impl Op {
    fn parameters(&self) -> impl Iterator<Item = OpId> {
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
        }
        .into_iter()
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
        //println!("Kernel shape {:?}", self.shape);
        let mut indent = String::from("    ");
        let mut order = self.order.clone();
        order.reverse();
        let mut end_loop_op = OpId(0);
        while let Some(op_id) = order.pop() {
            if op_id == end_loop_op {
                indent.pop();
                indent.pop();
            }
            match self.ops[op_id] {
                Op::ConstView { value, ref view } => println!("{op_id:>3}{indent}{CYAN}CONST VIEW{RESET} {value} {view}"),
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
                Op::Loop { dim, scope, ref ops } => {
                    println!("{op_id:>3}{indent}{BLUE}LOOP{RESET} {scope} dim={dim}");
                    end_loop_op = *ops.last().unwrap();
                    order.extend(ops.iter().rev());
                    indent += " ";
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

    /*pub fn shape(&self) -> Vec<Dim> {
        if self.ops.values().any(|op| matches!(op, Op::Loop { .. })) {
            return self
                .ops
                .values()
                .filter_map(|op| {
                    if let Op::Loop { dim, scope, ops } = op {
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
        for op in self.ops.iter().rev() {
            match op {
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
    }*/

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
            let Op::Reduce { x, rop, ref dims } = self[reduce_op_id] else { unreachable!() };

            // Find all relevant ops
            let mut ops = vec![x];
            let mut params = vec![x];
            while let Some(param) = params.pop() {
                params.extend(self[param].parameters());
                ops.extend(self[param].parameters());
            }
            ops.reverse();

            // Remove ops from order
            self.order.retain(|op_id| !ops.contains(op_id));

            // Put ops in a loop block
        }
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
    }

    pub fn unfold_views(&mut self) {}

    /*pub fn unfold_views(&mut self) {
        // First we generate the whole view into a new vec,
        // then we insert the vec into existing ops
        // Convert view
        fn new_op(ops: &mut Vec<Op>, op: Op) -> OpId {
            let op_id = ops.len();
            ops.push(op);
            op_id
        }

        let n_loads = self.ops.iter().filter(|op| matches!(op, Op::LoadView { .. })).count();
        let mut load_id = 0;
        let mut store_id = n_loads;
        let mut op_id = 0;
        while op_id < self.ops.len() {
            match self.ops[op_id] {
                Op::ConstView { value, ref view } => {
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let view = view.clone();

                    //println!("Unfolding view: {view}");
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let ops = &mut self.ops;
                    let axes = get_axes(&ops[0..op_id]);
                    let mut pc = new_op(ops, Op::Const(Constant::Bool(true)));
                    let constant_zero = new_op(ops, Op::Const(Constant::idx(0)));
                    #[allow(unused)] // false positive
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
                        for (a, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                new_op(ops, Op::Const(Constant::idx(0u64)))
                            } else {
                                axes[a]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    //let pcu32 = new_op(ops, Op::Cast { x: pc, dtype: DType::U32 });
                    //let offset = new_op(ops, Op::Binary { x: pcu32, y: offset, bop: BOp::Mul });

                    let z = new_op(ops, Op::Const(value));

                    // TODO process view
                    //self.ops[op_id] = Op::Const(value);
                    let dtype = value.dtype();
                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
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

                    //println!("Unfolding view: {view}");
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let ops = &mut self.ops;
                    let axes = get_axes(&ops[0..op_id]);
                    let mut pc = new_op(ops, Op::Const(Constant::Bool(true)));
                    let constant_zero = new_op(ops, Op::Const(Constant::idx(0)));
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
                        for (a, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
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
                                    let ost_c = new_op(ops, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                constant_zero
                            } else {
                                axes[a]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let pcu = new_op(ops, Op::Cast { x: pc, dtype: IDX_T });
                    let offset = new_op(ops, Op::Binary { x: pcu, y: offset, bop: BOp::Mul });

                    let z = new_op(ops, Op::Load { src: load_id, index: offset });

                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
                    load_id += 1;
                    continue;
                }
                Op::StoreView { src, .. } => {
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let axes = get_axes(&self.ops);
                    let mut index = new_op(&mut self.ops, Op::Const(Constant::idx(0u64)));
                    let mut st = 1;

                    let shape = {
                        let mut shape = Vec::new();
                        for op in &self.ops {
                            match op {
                                Op::Loop { dim, .. } => {
                                    shape.push(*dim);
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
                            new_op(&mut self.ops, Op::Const(Constant::idx(0)))
                        };
                        let y = new_op(&mut self.ops, Op::Const(stride));
                        let x = new_op(&mut self.ops, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut self.ops, Op::Binary { x, y: index, bop: BOp::Add });
                        st *= d;
                    }

                    _ = new_op(&mut self.ops, Op::Store { dst: store_id, x: src, index });

                    let n = self.ops.len();
                    /*for (i, op) in self.ops.iter().enumerate() {
                        println!("{i} -> {op:?}");
                    }
                    println!("n={n}");*/
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
                    store_id += 1;
                    continue;
                }
                _ => {}
            }
            op_id += 1;
        }
    }*/

    /*pub fn unfold_pows(&mut self) {
        let mut op_id = 0;
        while op_id < self.ops.len() {
            if let Op::Binary { x, y, bop } = self.ops[op_id] {
                if bop == BOp::Pow {
                    let mut tail: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    self.ops.push(Op::Unary { x, uop: UOp::Log2 });
                    self.ops.push(Op::Binary { x: op_id, y, bop: BOp::Mul });
                    self.ops.push(Op::Unary { x: op_id + 1, uop: UOp::Exp2 });
                    increment(&mut tail, 2, op_id..);
                    self.ops.extend(tail);
                }
            }
            op_id += 1;
        }
    }*/

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        let mut visited = Set::default();
        // We go backward from Stores and gather all needed ops, but we can't remove Loop and Define ops
        for (op_id, op) in self.ops.iter() {
            if matches!(op, Op::Store { .. } | Op::Loop { .. } | Op::Define { .. }) {
                params.push(op_id);
            }
        }
        while let Some(op_id) = params.pop() {
            params.extend(self[op_id].parameters());
            visited.extend(self[op_id].parameters());
        }
        // Remove ops that are not in visited both from self.ops and self.order
        let ids: Set<OpId> = self.ops.ids().filter(|op_id| !visited.contains(op_id)).collect();
        for &op_id in &ids {
            self.ops.remove(op_id);
        }
        // Remove from self.order and loops
        self.order.retain(|op_id| !ids.contains(op_id));
        let mut order = self.order.clone();
        order.reverse();
        while let Some(op_id) = order.pop() {
            if let Op::Loop { ops, .. } = &mut self[op_id] {
                ops.retain(|op_id| !ids.contains(op_id));
                order.extend(ops.iter().rev());
            }
        }
    }

    /*pub fn common_subexpression_elimination(&mut self) {
        let mut unique_stack: Vec<Map<Op, OpId>> = Vec::new();
        unique_stack.push(Map::with_capacity_and_hasher(10, BuildHasherDefault::new()));
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        for (op_id, op) in self.ops.iter().enumerate() {
            match op {
                Op::Loop { .. } => {
                    unique_stack.push(Map::with_capacity_and_hasher(10, BuildHasherDefault::new()));
                }
                Op::EndLoop => {
                    unique_stack.pop();
                }
                _ => {
                    for unique in &unique_stack {
                        if let Some(&id) = unique.get(op) {
                            remaps.insert(op_id, id);
                            break;
                        }
                    }

                    if !remaps.contains_key(&op_id)
                        && !matches!(
                            op,
                            Op::Define { .. } | Op::Loop { .. } | Op::EndLoop | Op::Load { .. } | Op::Store { .. }
                        )
                    {
                        unique_stack.last_mut().unwrap().insert(op.clone(), op_id);
                    }
                }
            }
        }
        remap(&mut self.ops, &remaps);
    }

    pub fn move_constants_to_beginning(&mut self) {
        let n_defines = self.ops.iter().position(|op| !matches!(op, Op::Define { .. })).unwrap();
        let tail = self.ops.split_off(n_defines);
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        let n_constants = tail.iter().filter(|op| matches!(op, Op::Const(_))).count();

        for (i, op) in tail.iter().enumerate() {
            if matches!(op, Op::Const(_)) {
                let new_index = self.ops.len();
                self.ops.push(op.clone());
                remaps.insert(i + n_defines + n_constants, new_index);
            }
        }
        self.ops.extend(tail);
        increment(
            &mut self.ops[n_defines + remaps.len()..],
            remaps.len() as isize,
            n_defines..,
        );
        remap(&mut self.ops, &remaps);
    }

    /// Constant folding
    pub fn constant_folding(&mut self) {
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        for op_id in 0..self.ops.len() {
            match self.ops[op_id] {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => unreachable!(),
                Op::Const { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Define { .. }
                | Op::Null => {}
                Op::Cast { x, dtype } => {
                    if let Op::Const(x) = self.ops[x] {
                        self.ops[op_id] = Op::Const(x.cast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(x) = self.ops[x] {
                        self.ops[op_id] = Op::Const(x.unary(uop));
                    }
                }
                Op::Binary { x, y, bop } => match (&self.ops[x], &self.ops[y]) {
                    (&Op::Const(cx), &Op::Const(cy)) => {
                        self.ops[op_id] = Op::Const(Constant::binary(cx, cy, bop));
                    }
                    (&Op::Const(cx), _) => match bop {
                        BOp::Add => {
                            if cx.is_zero() {
                                remaps.insert(op_id, y);
                            }
                        }
                        BOp::Sub => {
                            if cx.is_zero() {
                                self.ops[op_id] = Op::Unary { x: y, uop: UOp::Neg };
                            }
                        }
                        BOp::Mul => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                remaps.insert(op_id, y);
                            }
                        }
                        BOp::Div => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                self.ops[op_id] = Op::Unary { x: y, uop: UOp::Reciprocal };
                            }
                        }
                        BOp::Pow => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                remaps.insert(op_id, x);
                            } //else if cx.is_two() && cx.dtype().is_shiftable() {
                            //self.ops.insert(op_id, Op::Constant(cx.dtype().one())); // but we can't insert with remaps
                            //self.ops[op_id] = Op::Binary { x, y, bop: BOp::BitShiftLeft };
                            //}
                        }
                        BOp::Mod => todo!(),
                        BOp::Cmplt => todo!(),
                        BOp::Cmpgt => {}
                        BOp::Maximum => todo!(),
                        BOp::Or => todo!(),
                        BOp::And => todo!(),
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                        BOp::NotEq => todo!(),
                        BOp::Eq => todo!(),
                    },
                    (_, &Op::Const(cy)) => match bop {
                        BOp::Add | BOp::Sub => {
                            if cy.is_zero() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Mul => {
                            if cy.is_zero() {
                                remaps.insert(op_id, y);
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Div => {
                            if cy.is_zero() {
                                panic!("Division by zero constant.");
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Pow => {
                            if cy.is_zero() {
                                self.ops[op_id] = Op::Const(cy.dtype().one_constant());
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            } else if cy.is_two() {
                                self.ops[op_id] = Op::Binary { x, y: x, bop: BOp::Mul };
                            }
                        }
                        BOp::Mod => {
                            if cy.is_zero() {
                                panic!("Modulo by zero constant.");
                            } else if cy.is_one() {
                                self.ops[op_id] = Op::Const(cy.dtype().zero_constant());
                            }
                        }
                        BOp::Cmplt | BOp::Cmpgt | BOp::NotEq | BOp::And | BOp::Eq => {}
                        BOp::Maximum => {}
                        BOp::Or => todo!(),
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                    },
                    _ => {}
                },
            }
        }
        remap(&mut self.ops, &remaps);
    }*/
}

/*
/// Kernel optimization ops that may or may not be applied, that is they are beneficial for some kernels,
/// but hurt other kernels or backends.
impl Kernel {
    /// Take defines in loop at loop_id and multiply their size
    /// Take the loop at loop_id and move it into the loop that follows it (jam)
    pub fn loop_jam(&mut self, loop_id: OpId) {
        self.debug();
        // Get loop details
        let Op::Loop { dim, scope } = self.ops[loop_id] else { unreachable!() };
        debug_assert_eq!(scope, Scope::Register);

        // Get loop borders
        let end_loop_id = self.get_end_loop_id(loop_id);
        let inner_loop_id =
            self.ops[loop_id + 1..].iter().position(|op| matches!(op, Op::Loop { .. })).unwrap() + loop_id + 1;
        let end_inner_loop_id = self.get_end_loop_id(inner_loop_id);

        // split kernel into blocks
        let tail = self.ops.split_off(end_loop_id);
        let post_loop = self.ops.split_off(end_inner_loop_id + 1);
        let mut inner_loop = self.ops.split_off(inner_loop_id);
        let pre_loop = self.ops.split_off(loop_id);

        // Define offset that increases for each inserted op
        let mut offset = 0;

        // Expand define ops
        let mut define_map = Map::default();
        for (i, op) in pre_loop.iter().enumerate() {
            if let &Op::Define { dtype, scope, ro, len } = op {
                let new_define_id = self.ops.len();
                self.ops.push(Op::Define { dtype, scope, ro, len: len * dim });
                offset += 1;
                define_map.insert(i + loop_id, new_define_id);
            }
        }

        // Add stores of define ops before inner loops
        let loop_index = self.ops.len();
        self.ops.push(Op::Loop { dim, scope });
        offset += 1;
        for op in &pre_loop {
            if let &Op::Store { dst, x, index } = op {
                // TODO fix dst, x, index
                // for example if index is not just a constant 0, it can't be replaced,
                // it has to be multiplied by stride and added
                let dst = define_map.get(&dst).copied().unwrap_or(dst);
                self.ops.push(Op::Store { dst, x, index: loop_index });
                offset += 1;
            }
        }
        self.ops.push(Op::EndLoop);
        offset += 1;

        // ***** JAM THE LOOP *****
        // *** Put pre loop ***
        let new_inner_loop_id = self.ops.len();
        self.ops.push(inner_loop.remove(0));
        for op in &pre_loop {
            if matches!(op, Op::Define { .. } | Op::Store { .. }) {
                self.ops.push(Op::Null);
            } else {
                self.ops.push(op.clone());
            }
        }
        // Reindex pre loop
        println!(
            "new_inner_loop_id={new_inner_loop_id}, pre_loop.len()={}, offset={offset}, loop_id={loop_id}",
            pre_loop.len(),
        );
        // offset + 1 for inner loop op
        increment(&mut self.ops[new_inner_loop_id + 1..], offset + 1, loop_id..);

        // *** Put inner loop ***
        // First increment ops inside inner loop by offset (number of inserted ops)
        increment(&mut inner_loop, offset, inner_loop_id..);
        // Remap references to inner loop, now already with applied offset
        //remap(&mut inner_loop, map);
        // Increment references to pre loop, again + 1 for inner loop op itself
        increment(&mut inner_loop, offset + 1, loop_id..inner_loop_id);
        // Remap references to defines
        // remape(&mut inner_loop, defines_map);
        self.ops.extend(inner_loop);

        // Put pre loop before post loop
        // TODO fix offsets
        for op in pre_loop {
            if matches!(op, Op::Define { .. } | Op::Store { .. }) {
                self.ops.push(Op::Null);
            } else {
                self.ops.push(op.clone());
            }
        }

        // Resolve parts ofter the inner loop
        // Put post loop
        // TODO fix offsets
        self.ops.extend(post_loop);

        // Put tail
        // TODO fix offsets
        self.ops.extend(tail);

        self.debug();
        panic!();
    }

    // Something like this to upcast local dimensions, that is to do double buffering along local dimensions
    //pub fn upcast_local(&mut self, loop_id: OpId, dim: Dim) {}

    // Split loop into multiple accumulation steps
    //pub fn multi_step_reduce(&mut self) {}

    /// Adds local memory buffer for large reduces
    pub fn grouptop(&mut self, loop_id: OpId, dim: Dim) {}

    // In tinygrad, thread splits workload across multiple CPU threads
    //pub fn thread(&mut self, loop_id)

    // Loop tiling/vectorization. Tiles all loads.
    // This may not be needed, it depends how we actually implement tensor cores and double buffering
    /*pub fn loop_tile(&mut self, loop_id: OpId) {
        todo!()
    }*/

        pub fn loop_invariant_code_motion_all(&mut self) {
            let mut op_id = self.ops.len();
            while op_id > 0 {
                op_id -= 1;
                if matches!(self.ops[op_id], Op::Loop { .. }) {
                    self.loop_invariant_code_motion(op_id);
                }
            }
        }

        pub fn reorder_commutative(&mut self) {
            // TODO Reorder commutative
            // Iterate:
            //   find a chain of commutative ops like add/sub
            //   reoder by moving loop index last
        }

        pub fn loop_invariant_code_motion(&mut self, loop_id: OpId) {
            // LICM
            // Extract loop body and tail
            let end_loop_id = self.get_end_loop_id(loop_id);

            let tail = self.ops.split_off(end_loop_id);
            let mut body = self.ops.split_off(loop_id);
            // for each op in loop body - if all parameters are invariant, mark as invariant, otherwise do nothing
            let mut i = 0;
            while i < body.len() {
                // If op is invariant
                if !matches!(
                    body[i],
                    Op::Loop { .. } | Op::Store { .. } | Op::Load { .. } | Op::EndLoop | Op::Define { .. }
                ) && body[i].parameters().all(|x| x < loop_id)
                {
                    // Move op out of the loop
                    let op = body.remove(i);
                    self.ops.push(op);

                    // Increment and remap all ops in the body accordingly
                    remap_or_increment(
                        &mut body,
                        self.ops.len() + i - 1,
                        self.ops.len() - 1,
                        1,
                        self.ops.len() - 1..self.ops.len() + i - 1,
                    );
                } else {
                    i += 1;
                }
            }

            // Add body back to ops
            self.ops.extend(body);
            self.ops.extend(tail);
        }

        pub fn loop_unroll(&mut self, loop_id: OpId) {
            let Op::Loop { dim, .. } = self.ops[loop_id] else { unreachable!() };
            let end_loop_id = self.get_end_loop_id(loop_id);

            // Get tail and body
            let mut tail = self.ops.split_off(end_loop_id + 1);
            self.ops.pop();
            let loop_body = self.ops.split_off(loop_id + 1);
            self.ops.pop();

            // Repeat loop body
            let mut offset = 1;
            for idx in 0..dim {
                let mut body = loop_body.clone();
                // First index as constant
                let idx_const = self.ops.len();
                self.ops.push(Op::Const(Constant::idx(idx as u64)));

                remap_or_increment(&mut body, loop_id, idx_const, offset - 1, loop_id + 1..end_loop_id);
                offset += body.len() + 1;
                self.ops.extend(body);
            }

            // Add tail, increment ops
            let unrolled_body_size = end_loop_id - loop_id;
            let d = (unrolled_body_size * (dim - 1)) as isize - 1;
            let tail_range = end_loop_id..;
            increment(&mut tail, d, tail_range);
            self.ops.extend(tail);
        }

            /// Reshapes, (splits or merges) reduce from original into new_dims
            pub fn reshape_reduce(&mut self, reduce_id: OpId, new_dims: &[Dim]) {
                let Op::Reduce { x, ref mut dims, .. } = self.ops[reduce_id] else { return };
                let n_old_dims = dims.len();
                *dims = new_dims.into();

                let mut visited = Set::default();
                self.recursively_apply_reshape(x, n_old_dims, new_dims, &mut visited, 0);
            }

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
                match self.ops[op_id] {
                    Op::LoadView { ref mut view, .. } | Op::ConstView { ref mut view, .. } => {
                        let rank = view.rank();
                        view.reshape(rank - skip_last - n_old_dims..rank - skip_last, new_dims);
                    }
                    Op::Reduce { x, ref dims, .. } => {
                        let skip_last = skip_last + dims.len();
                        self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
                    }
                    Op::Cast { x, .. } | Op::Unary { x, .. } => {
                        self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
                    }
                    Op::Binary { x, y, .. } => {
                        self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
                        self.recursively_apply_reshape(y, n_old_dims, new_dims, visited, skip_last);
                    }
                    _ => {}
                }
            }

                // Loops that don't contain stores can be deleted
                pub fn delete_empty_loops(&mut self) {
                    // TODO make this fast by going in reverse
                    for i in 0..self.ops.len() {
                        if matches!(self.ops[i], Op::Loop { .. }) {
                            let mut contains_store = false;
                            let mut end_loop_id = 0;
                            let mut loop_level = 0;
                            for (i, op) in self.ops[i..].iter().enumerate() {
                                match op {
                                    Op::Store { .. } => {
                                        contains_store = true;
                                        break;
                                    }
                                    Op::Loop { .. } => {
                                        loop_level += 1;
                                    }
                                    Op::EndLoop => {
                                        loop_level -= 1;
                                        if loop_level == 0 {
                                            end_loop_id = i;
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            if !contains_store {
                                //panic!("Deleting from {i} to {}", i + end_loop_id);
                                for op in &mut self.ops[i..=i + end_loop_id] {
                                    *op = Op::Null
                                }
                            }
                        }
                    }
                }

                // TODO delete loops that iterate only once
                // fn delete_single_iteration_loops(&mut self) {}
}*/
