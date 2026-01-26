use crate::{
    BLUE, CYAN, DType, GREEN, MAGENTA, Map, ORANGE, RED, RESET, Set, YELLOW,
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    realize::KMKernelId,
    shape::{Dim, UAxis, permute},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};
use nanoserde::{DeBin, SerBin};
use std::{
    fmt::Display,
    hash::{BuildHasherDefault, Hash},
};

// TODO later make this dynamic u32 or u64 depending on max range
pub const IDX_T: DType = DType::U32;

#[derive(Debug, Clone)]
pub struct Kernel {
    pub outputs: Vec<TensorId>,
    pub loads: Vec<TensorId>,
    pub stores: Vec<TensorId>,
    pub ops: Slab<OpId, OpNode>,
    pub head: OpId,
    pub tail: OpId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpNode {
    pub prev: OpId,
    pub next: OpId,
    pub op: Op,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Op {
    // ops that exist in both
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    // For binary ops, next of x is y, then next of y is the binary op
    Binary { x: OpId, y: OpId, bop: BOp },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for global stores
    Store { dst: OpId, x: OpId, index: OpId, vlen: u8 },
    Load { src: OpId, index: OpId, vlen: u8 },
    Loop { dim: Dim, scope: Scope },
    EndLoop,
    // fused multiply add
    Mad { x: OpId, y: OpId, z: OpId },
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator
    // TODO would be adding d useful?
    MMA { m: u8, n: u8, k: u8, c: OpId, a: OpId, b: OpId },
    // Vectorization, YAY!
    Vectorize { ops: Vec<OpId> },
    Devectorize { vec: OpId, idx: usize }, // select a single value from a vector

    // ops that exist only in kernelizer, basically they can be eventually removed.
    // TODO Get rid of the view, use whatever ops that are needed directly
    // and then use unfold movement ops function to convert it all into indices.
    // This will make Op smaller and Copy.
    ConstView(Box<(Constant, View)>),
    LoadView(Box<(DType, View)>),
    StoreView { src: OpId, dtype: DType },
    Reduce { x: OpId, rop: ROp, n_axes: UAxis },
    //MergeIndices { x: OpId, y: OpId }, // creates index for merge of loops x and y (i.e. x * y_len + y)
    //PermuteIndices(Vec<OpId>), // Permute for indices, just swapping indices around
    //PadIndex(OpId, isize, isize), // Pad index with padding
    //Unsqueeze { axis: Axis, dim: Dim } // Inserts a new loop at given axis
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    pub fn parameters(&self) -> impl Iterator<Item = OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            &Op::StoreView { src, .. } => vec![src],
            Op::Reduce { x, .. } => vec![*x],
            &Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            Op::Const { .. } => vec![],
            Op::Define { .. } => vec![],
            &Op::Load { src, index, .. } => vec![src, index],
            Op::Loop { .. } => vec![],
            Op::EndLoop { .. } => vec![],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.clone(),
            &Op::Devectorize { vec, .. }  => vec![vec],
            &Op::MMA { a, b, c, .. } => vec![a, b, c],
        }
        .into_iter()
    }

    pub fn parameters_mut(&mut self) -> impl Iterator<Item = &mut OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            Op::StoreView { src, .. } => vec![src],
            Op::Reduce { x, .. } => vec![x],
            Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Const { .. } => vec![],
            Op::Define { .. } => vec![],
            Op::Load { src, index, .. } => vec![src, index],
            Op::Loop { .. } => vec![],
            Op::EndLoop { .. } => vec![],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.iter_mut().collect(),
            Op::Devectorize { vec, .. }  => vec![vec],
            Op::MMA { a, b, c, .. } => vec![a, b, c],
        }
        .into_iter()
    }

    pub fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    pub fn is_load(&self) -> bool {
        matches!(self, Op::Load { .. })
    }

    pub fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
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

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.head == other.head
    }
}
impl Eq for Kernel {}

impl SerBin for Kernel {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.ops.ser_bin(output);
        self.head.ser_bin(output);
        self.tail.ser_bin(output);
    }
}

impl DeBin for Kernel {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let ops = Slab::<OpId, OpNode>::de_bin(offset, bytes)?;
        let start = OpId::de_bin(offset, bytes)?;
        let end = OpId::de_bin(offset, bytes)?;
        Ok(Self { head: start, tail: end, ops, outputs: Vec::new(), loads: Vec::new(), stores: Vec::new() })
    }
}

impl Hash for Kernel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.ops.hash(state);
    }
}

impl Kernel {
    pub fn ops_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.ops.values_mut().map(|op_node| &mut op_node.op)
    }

    #[track_caller]
    pub fn at(&self, op_id: OpId) -> &Op {
        &self.ops[op_id].op
    }

    pub fn apply_movement(&mut self, func: impl Fn(&mut View)) {
        for op in self.ops_mut() {
            match op {
                Op::ConstView(x) => func(&mut x.1),
                Op::LoadView(x) => func(&mut x.1),
                _ => {}
            }
        }
    }

    pub fn debug(&self) {
        println!("\nloads={:?}", self.loads);
        println!("stores={:?}", self.stores);
        println!("outputs={:?}", self.outputs);
        //println!("Kernel shape {:?}", self.shape);
        let mut indent = String::from(" ");
        let mut ids: Map<OpId, (Dim, Dim)> = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    let value = x.0;
                    let view = &x.1;
                    println!("{op_id:>5}{indent}{CYAN}CONST VIEW{RESET} {value} {view}");
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let view = &x.1;
                    println!("{op_id:>5}{indent}{CYAN}LOAD VIEW{RESET} {dtype} {view}");
                }
                Op::StoreView { src, dtype, .. } => {
                    println!("{op_id:>5}{indent}{CYAN}STORE VIEW{RESET} {src} {dtype}");
                }
                Op::Reduce { x, rop, n_axes, .. } => {
                    println!(
                        "{op_id:>5}{indent}{RED}REDUCE{RESET} {} {x}, dims={n_axes:?}",
                        match rop {
                            ROp::Sum => "SUM",
                            ROp::Max => "MAX",
                        }
                    );
                }
                Op::Define { dtype, scope, ro, len, .. } => {
                    println!("{op_id:>5}{indent}{YELLOW}DEFINE{RESET} {scope} {dtype}, len={len}, ro={ro}");
                }
                Op::Const(value) => {
                    if value.is_positive() {
                        let Constant::U64(v) = value.cast(DType::U64) else { unreachable!() };
                        let v = usize::from_le_bytes(v);
                        ids.insert(op_id, (v, v));
                    }
                    println!("{op_id:>5}{indent}{MAGENTA}CONST{RESET} {} {value}", value.dtype());
                }
                Op::Load { src, index, vlen: len, .. } => {
                    println!(
                        "{op_id:>5}{indent}{GREEN}LOAD{RESET} p{src}[{index}] len={len:?}    {}..={}",
                        ids[&index].0, ids[&index].1
                    );
                }
                Op::Store { dst, x: src, index, vlen: len } => {
                    println!(
                        "{op_id:>5}{indent}{RED}STORE{RESET} p{dst}[{index}] <- {src} len={len}    {}..={}",
                        ids[&index].0, ids[&index].1
                    );
                }
                Op::Cast { x, dtype, .. } => {
                    if let Some((l, u)) = ids.get(&x) {
                        ids.insert(op_id, (*l, *u));
                    }
                    if let Some((l, u)) = ids.get(&op_id) {
                        println!("{op_id:>5}{indent}CAST {x} {dtype:?}    {l}..={u}");
                    } else {
                        println!("{op_id:>5}{indent}CAST {x} {dtype:?}");
                    }
                }
                Op::Unary { x, uop, .. } => {
                    if let Some((l, u)) = ids.get(&x) {
                        ids.insert(op_id, (*l, *u));
                    }
                    if let Some((l, u)) = ids.get(&op_id) {
                        println!("{op_id:>5}{indent}UNARY {uop:?} {x}    {l}..={u}");
                    } else {
                        println!("{op_id:>5}{indent}UNARY {uop:?} {x}");
                    }
                }
                Op::Binary { x, y, bop, .. } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                    {
                        ids.insert(
                            op_id,
                            match bop {
                                BOp::Add => (xl.wrapping_add(yl), xu.wrapping_add(yu)),
                                BOp::Sub => (xl.wrapping_sub(yl), xu.wrapping_sub(yu)),
                                BOp::Mul => (xl.wrapping_mul(yl), xu.wrapping_mul(yu)),
                                BOp::Div => (xl / yl, xu / yu),
                                BOp::Mod => (xl % yl, xu % yu),
                                BOp::Eq => ((xl == yl) as usize, (xu == yu) as usize),
                                BOp::NotEq => ((xl != yl) as usize, (xu != yu) as usize),
                                BOp::Cmpgt => ((xl > yl) as usize, (xu > yu) as usize),
                                BOp::Cmplt => ((xl < yl) as usize, (xu < yu) as usize),
                                BOp::And => ((xl == 1 && yl == 1) as usize, (xu == 1 && yu == 1) as usize),
                                BOp::BitShiftLeft => (xl << yl, xu << yu),
                                BOp::BitShiftRight => (xl >> yl, xu >> yu),
                                op => todo!("{:?}", op),
                            },
                        );
                    }
                    if let Some((l, u)) = ids.get(&op_id) {
                        println!("{op_id:>5}{indent}BINARY {bop:?} {x} {y}    {l}..={u}");
                    } else {
                        println!("{op_id:>5}{indent}BINARY {bop:?} {x} {y}");
                    }
                }
                Op::Mad { x, y, z } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                        && let Some(&(zl, zu)) = ids.get(&z)
                    {
                        ids.insert(
                            op_id,
                            (
                                xl.wrapping_mul(yl).wrapping_add(zl),
                                xu.wrapping_mul(yu).wrapping_add(zu),
                            ),
                        );
                    }
                    if let Some((l, u)) = ids.get(&op_id) {
                        println!("{op_id:>5}{indent}MAD {x} {y} {z}    {l}..={u}");
                    } else {
                        println!("{op_id:>5}{indent}MAD {x} {y} {z}");
                    }
                }
                Op::MMA { m, n, k, c, a, b } => {
                    println!("{op_id:>5}{indent}{ORANGE}MMA{RESET} m{m}n{n}k{k} c={c} a={a} b={b}");
                }
                Op::Loop { dim, scope, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                    println!(
                        "{op_id:>5}{indent}{BLUE}LOOP{RESET} {scope} dim={dim}    0..={}",
                        dim - 1
                    );
                    if scope == Scope::Register {
                        indent += "  ";
                    }
                }
                Op::EndLoop { .. } => {
                    if indent.len() > 1 {
                        indent.pop();
                        indent.pop();
                    }
                    println!("{op_id:>5}{indent}{BLUE}END_LOOP{RESET}");
                }
                Op::Vectorize { ref ops } => {
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = ids.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    if let Some((xl, xu)) = r {
                        ids.insert(op_id, (xl, xu));
                        println!("{op_id:>5}{indent}{ORANGE}VECTORIZE{RESET} {ops:?}    {xl}..={xu}");
                    } else {
                        println!("{op_id:>5}{indent}{ORANGE}VECTORIZE{RESET} {ops:?}");
                    }
                }
                Op::Devectorize { vec, idx } => {
                    if let Some((l, u)) = ids.get(&vec) {
                        ids.insert(op_id, (*l, *u));
                    }
                    if let Some((l, u)) = ids.get(&op_id) {
                        println!("{op_id:>5}{indent}{ORANGE}DEVECTORIZE {vec}[{idx}]    {l}..={u}");
                    } else {
                        println!("{op_id:>5}{indent}{ORANGE}DEVECTORIZE {vec}[{idx}]");
                    }
                }
            }
            op_id = self.ops[op_id].next;
        }
    }

    pub fn iter_unordered(&self) -> impl Iterator<Item = (OpId, &Op)> {
        self.ops.iter().map(|(id, node)| (id, &node.op))
    }

    pub fn flop_mem_rw(&self) -> (u64, u64, u64) {
        #[derive(Clone)]
        struct Info {
            shape: Vec<Dim>,
            flops: u64,
            mem_read: u64,
            mem_write: u64,
        }

        let mut stack: Map<OpId, Info> = Map::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let info = match self.at(op_id) {
                Op::ConstView(x) => {
                    let shape = x.1.shape();
                    Info { shape, flops: 0, mem_read: 0, mem_write: 0 }
                }
                Op::LoadView(x) => {
                    let (dtype, view) = x.as_ref();
                    let shape = view.shape();
                    let mem_read = view.original_numel() as u64 * dtype.byte_size() as u64;
                    Info { shape, flops: 0, mem_read, mem_write: 0 }
                }
                Op::StoreView { src, dtype } => {
                    let Info { shape, .. } = stack[src].clone();
                    let mem_write = shape.iter().product::<Dim>() as u64 * dtype.byte_size() as u64;
                    Info { shape, flops: 0, mem_read: 0, mem_write }
                }
                Op::Reduce { x, n_axes, .. } => {
                    let Info { mut shape, .. } = stack[x].clone();
                    let rd: Dim = shape[shape.len() - n_axes..].iter().product();
                    shape.truncate(shape.len() - n_axes);
                    let n: Dim = shape.iter().product();
                    let flops = n * (rd - 1);
                    let flops = flops as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Binary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::MMA { .. }
                | Op::Vectorize { .. }
                | Op::Devectorize { .. }
                | Op::Store { .. }
                | Op::Mad { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::Load { .. }
                | Op::Loop { .. }
                | Op::EndLoop => todo!(),
            };
            stack.insert(op_id, info);
            op_id = self.next_op(op_id);
        }

        stack.into_values().fold((0, 0, 0), |acc, info| {
            (acc.0 + info.flops, acc.1 + info.mem_read, acc.2 + info.mem_write)
        })
    }

    pub fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::StoreView { .. }))
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. }))
    }

    pub fn reduce_dims(&self, op_id: OpId) -> Vec<Dim> {
        let mut params = vec![op_id];
        let mut n_reduce_axes = 0;
        let mut visited = Set::default();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                match self.at(param) {
                    Op::ConstView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::LoadView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::Reduce { n_axes, .. } => n_reduce_axes += n_axes,
                    _ => {}
                }
                params.extend(self.at(param).parameters());
            }
        }
        unreachable!();
    }

    pub fn shape(&self) -> Vec<Dim> {
        if self.ops.values().any(|x| matches!(x.op, Op::Loop { .. })) {
            return self
                .ops
                .values()
                .filter_map(|x| {
                    if let Op::Loop { dim, scope, .. } = x.op {
                        if matches!(scope, Scope::Global | Scope::Local) {
                            Some(dim)
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
        let mut op_id = self.tail;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::ConstView(x) => {
                    let mut shape = x.1.shape();
                    shape.truncate(shape.len() - reduce_dims);
                    return shape;
                }
                Op::LoadView(x) => {
                    let mut shape = x.1.shape();
                    shape.truncate(shape.len() - reduce_dims);
                    return shape;
                }
                Op::Reduce { n_axes, .. } => {
                    reduce_dims += n_axes;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        unreachable!()
    }

    #[allow(unused)]
    pub fn is_reshape_contiguous(&self, range: std::ops::Range<UAxis>, shape: &[Dim]) -> bool {
        self.ops.values().all(|node| match &node.op {
            Op::ConstView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            Op::LoadView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            _ => true,
        })
    }

    /// Fuses multiple reduce ops together if possible
    pub fn fuse_reduces(&mut self) {
        // TODO
    }

    pub fn close_loops(&mut self) {
        let mut loop_id = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Loop { .. } => loop_id += 1,
                Op::EndLoop { .. } => loop_id -= 1,
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
        while loop_id > 0 {
            self.push_back(Op::EndLoop);
            loop_id -= 1;
        }
    }

    /// Find all Reduce ops and put them in a Loop block
    /// Add define ops and add reduce operation as BOp::Add or BOp::Max
    pub fn unfold_reduces(&mut self) {
        let mut reduce_op_ids: Vec<OpId> = self
            .iter_unordered()
            .filter_map(|(id, op)| {
                if matches!(op, Op::Reduce { .. }) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        while let Some(reduce_op_id) = reduce_op_ids.pop() {
            let Op::Reduce { x, rop, n_axes } = self.ops[reduce_op_id].op else { unreachable!() };

            let mut reduce_loop_ops_set = Set::default();
            let mut params = vec![x];
            let mut acc_dtype = None;
            while let Some(param) = params.pop() {
                if reduce_loop_ops_set.insert(param) {
                    params.extend(self.at(param).parameters());
                    if acc_dtype.is_none() {
                        match self.at(param) {
                            &Op::Define { dtype, .. } => acc_dtype = Some(dtype),
                            Op::ConstView(x) => acc_dtype = Some(x.0.dtype()),
                            Op::LoadView(x) => acc_dtype = Some(x.0),
                            &Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                            _ => {}
                        }
                    }
                }
            }
            let acc_dtype = acc_dtype.unwrap();
            // Sort reduce loop ops by original order
            let mut op_id = self.head;
            let mut loop_start = OpId::NULL;
            while !op_id.is_null() {
                if reduce_loop_ops_set.contains(&op_id) {
                    loop_start = op_id;
                    break;
                }
                op_id = self.next_op(op_id);
            }

            // Add const zero
            let const_zero = self.insert_before(loop_start, Op::Const(Constant::idx(0)));

            // Add accumulator
            let acc_init_id = self.insert_before(
                loop_start,
                Op::Const(match rop {
                    ROp::Sum => acc_dtype.zero_constant(),
                    ROp::Max => acc_dtype.min_constant(),
                }),
            );

            let acc = self.insert_before(
                loop_start,
                Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 },
            );

            // Zero the accumulator
            self.insert_before(
                loop_start,
                Op::Store { dst: acc, x: acc_init_id, index: const_zero, vlen: 1 },
            );

            // Add Loops for the reduce
            for &dim in &self.reduce_dims(reduce_op_id)[..n_axes] {
                self.insert_before(loop_start, Op::Loop { dim, scope: Scope::Register });
            }

            // Add reduction operation, load from acc, accumulate, store to acc
            let load_acc = self.insert_before(reduce_op_id, Op::Load { src: acc, index: const_zero, vlen: 1 });
            let bin_acc = self.insert_before(
                reduce_op_id,
                Op::Binary {
                    x,
                    y: load_acc,
                    bop: match rop {
                        ROp::Sum => BOp::Add,
                        ROp::Max => BOp::Maximum,
                    },
                },
            );
            self.insert_before(
                reduce_op_id,
                Op::Store { dst: acc, x: bin_acc, index: const_zero, vlen: 1 },
            );

            // Close the reduce loop
            for _ in 0..n_axes {
                self.insert_before(reduce_op_id, Op::EndLoop);
            }

            // Replace old reduce op with the acc load op
            self.ops[reduce_op_id].op = Op::Load { src: acc, index: const_zero, vlen: 1 };

            #[cfg(debug_assertions)]
            self.verify();
        }
    }

    fn new_op(&mut self, op_iter: &mut OpId, op: Op) -> OpId {
        let op_id = self.insert_after(*op_iter, op);
        *op_iter = op_id;
        op_id
    }

    pub fn unfold_views(&mut self) {
        let mut axes = Vec::new();
        let start = self.head;
        let mut op_id = self.head;
        let mut reduce_loop = OpId::NULL;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::ConstView(ref x) => {
                    let value = x.0;
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let mut view = x.1.clone();

                    let mut view_axes = axes.clone();
                    view.reverse();
                    view_axes.reverse();

                    //println!("Unfolding view: {view}");

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut pc = self.new_op(opi, Op::Const(Constant::Bool(true)));
                    let constant_zero = self.new_op(opi, Op::Const(Constant::idx(0)));

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
                                    let ost_c = self.new_op(opi, Op::Const(Constant::idx(t_ost)));
                                    self.new_op(opi, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = self.new_op(opi, Op::Const(Constant::idx(dim.d as u64)));
                                    self.new_op(opi, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                self.new_op(opi, Op::Const(Constant::idx(0u64)))
                            } else {
                                view_axes[i]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = self.new_op(opi, Op::Const(Constant::idx(dim.st as u64)));
                                let x = self.new_op(opi, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = self.new_op(opi, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let z = self.new_op(opi, Op::Const(value));

                    let dtype = value.dtype();
                    let pcd = self.new_op(opi, Op::Cast { x: pc, dtype });

                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul }; // this is now the new op_id
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1 + a2*st2 + (a3-lp3)*st3 + ...
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let mut view = x.1.clone();

                    let mut view_axes = axes.clone();
                    view.reverse();
                    view_axes.reverse();

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut pc = self.new_op(opi, Op::Const(Constant::Bool(true)));
                    let constant_zero = self.new_op(opi, Op::Const(Constant::idx(0)));
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
                                    let ost_c = self.new_op(opi, Op::Const(Constant::idx(t_ost)));
                                    self.new_op(opi, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = self.new_op(opi, Op::Const(Constant::idx(dim.d as u64)));
                                    self.new_op(opi, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                constant_zero
                            } else {
                                view_axes[i]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let padded_loop_id = if dim.lp != 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Sub })
                                } else {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Add })
                                }
                            } else {
                                loop_id
                            };

                            if dim.st != 0 {
                                let stride = self.new_op(opi, Op::Const(Constant::idx(dim.st as u64)));
                                let x = self.new_op(opi, Op::Binary { x: padded_loop_id, y: stride, bop: BOp::Mul });
                                offset = self.new_op(opi, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Cmpgt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: rp, bop: BOp::Cmplt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let pcu = self.new_op(opi, Op::Cast { x: pc, dtype: IDX_T });
                    let offset = self.new_op(opi, Op::Binary { x: pcu, y: offset, bop: BOp::Mul });

                    let src = self.insert_before(
                        start,
                        Op::Define { dtype, scope: Scope::Global, ro: true, len: view.original_numel() },
                    );
                    let z = self.new_op(opi, Op::Load { src, index: offset, vlen: 1 });

                    let pcd = self.new_op(opi, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                }
                Op::StoreView { dtype, src, .. } => {
                    // TODO make this shorter and nicer to read
                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut index = self.new_op(opi, Op::Const(Constant::idx(0u64)));

                    let mut gws = Vec::new();
                    let mut lws = Vec::new();
                    let mut rws = Vec::new();
                    let shape: Vec<Dim> = {
                        let mut shape = Vec::new();
                        let mut l_op_id = self.head;
                        while !l_op_id.is_null() {
                            if l_op_id == op_id {
                                break;
                            }
                            match *self.at(l_op_id) {
                                Op::Loop { dim, scope } => {
                                    match scope {
                                        Scope::Global => gws.push(dim),
                                        Scope::Local => lws.push(dim),
                                        Scope::Register => rws.push(dim),
                                    }
                                    shape.push(dim);
                                }
                                Op::EndLoop => {
                                    shape.pop();
                                }
                                _ => {}
                            }
                            l_op_id = self.next_op(l_op_id);
                        }
                        shape
                    };

                    let mut original_shape = shape.clone();
                    match gws.len() {
                        1 => {}
                        2 => {
                            original_shape[0] = gws[0];
                            original_shape[1] = lws[0];
                            original_shape[2] = rws[0];
                            original_shape[3] = gws[1];
                            original_shape[4] = lws[1];
                            original_shape[5] = rws[1];
                        }
                        3 => {
                            original_shape[0] = gws[0];
                            original_shape[1] = lws[0];
                            original_shape[2] = rws[0];
                            original_shape[3] = gws[1];
                            original_shape[4] = lws[1];
                            original_shape[5] = rws[1];
                            original_shape[6] = gws[2];
                            original_shape[7] = lws[2];
                            original_shape[8] = rws[2];
                        }
                        _ => unreachable!(),
                    }
                    let mut original_strides = Vec::new();
                    let mut orig_stride = 1;
                    for d in original_shape.iter().rev() {
                        original_strides.push(orig_stride);
                        orig_stride *= d;
                    }
                    original_strides.reverse();

                    // This is permute order for performance, work_size.rs applies it too
                    let (permuted_shape, permuted_strides) = match gws.len() {
                        1 => (original_shape, original_strides),
                        2 => (
                            permute(&original_shape, &[0, 3, 1, 4, 2, 5]),
                            permute(&original_strides, &[0, 3, 1, 4, 2, 5]),
                        ),
                        3 => (
                            permute(&original_shape, &[0, 3, 6, 1, 4, 7, 2, 5, 8]),
                            permute(&original_strides, &[0, 3, 6, 1, 4, 7, 2, 5, 8]),
                        ),
                        _ => unreachable!(),
                    };
                    debug_assert_eq!(permuted_shape, shape);

                    //println!("permuted_shape={permuted_shape:?}, permuted_strides={permuted_strides:?}");

                    for (id, (d, st)) in permuted_shape.into_iter().zip(permuted_strides).enumerate() {
                        let stride = Constant::idx(st as u64);
                        let x = if d > 1 {
                            axes[id]
                        } else {
                            self.new_op(opi, Op::Const(Constant::idx(0)))
                        };
                        let y = self.new_op(opi, Op::Const(stride));
                        let x = self.new_op(opi, Op::Binary { x, y, bop: BOp::Mul });
                        index = self.new_op(opi, Op::Binary { x, y: index, bop: BOp::Add });
                    }

                    let len = shape.iter().product();
                    let dst = self.insert_before(start, Op::Define { dtype, scope: Scope::Global, ro: false, len });
                    self.ops[op_id].op = Op::Store { dst, x: src, index, vlen: 1 };
                }
                Op::Loop { scope, .. } => {
                    axes.push(op_id);
                    if scope == Scope::Register && reduce_loop.is_null() {
                        if let Op::Store { dst, .. } = self.ops[self.prev_op(op_id)].op {
                            if let Op::Define { scope, .. } = self.ops[dst].op {
                                if scope == Scope::Register {
                                    reduce_loop = op_id;
                                }
                            }
                        }
                    }
                }
                Op::EndLoop => {
                    axes.pop();
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Reorder defines of global args so that stores are after loads
        let mut op_id = self.prev_op(start);
        let mut last_load = OpId::NULL;
        while !op_id.is_null() {
            let Op::Define { ro, .. } = self.ops[op_id].op else { unreachable!() };
            if ro {
                if last_load.is_null() {
                    last_load = op_id;
                }
            } else {
                if !last_load.is_null() {
                    self.move_op_after(op_id, last_load);
                }
            }
            op_id = self.prev_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn unfold_pows(&mut self) {
        let mut op_id = self.ops.first_id();
        while !op_id.is_null() {
            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                if bop == BOp::Pow {
                    let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                    let x = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                    self.ops[op_id].op = Op::Unary { x, uop: UOp::Exp2 };
                }
            }
            op_id = self.ops.next_id(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        let mut visited = Set::default();
        // We go backward from Stores and gather all needed ops, but we can't remove Loop and Define ops
        for (op_id, op) in self.iter_unordered() {
            if matches!(
                op,
                Op::Store { .. } | Op::Loop { .. } | Op::Define { .. } | Op::EndLoop { .. } | Op::StoreView { .. }
            ) {
                params.push(op_id);
            }
        }
        while let Some(op_id) = params.pop() {
            if visited.insert(op_id) {
                params.extend(self.at(op_id).parameters());
            }
        }
        //self.ops.retain(|op_id| visited.contains(op_id));
        for op_id in self.ops.ids().collect::<Vec<_>>() {
            if !visited.contains(&op_id) {
                self.remove(op_id);
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn common_subexpression_elimination(&mut self) {
        let mut unique: Vec<Map<Op, OpId>> = Vec::with_capacity(10);
        unique.push(Map::with_capacity_and_hasher(50, BuildHasherDefault::new()));
        let mut unique_loads: Vec<Map<(OpId, OpId), OpId>> = Vec::with_capacity(10);
        unique_loads.push(Map::with_capacity_and_hasher(5, BuildHasherDefault::new()));
        let mut remaps = Map::with_capacity_and_hasher(10, BuildHasherDefault::default());
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Define { .. } => {}
                Op::Loop { .. } => {
                    unique.push(Map::with_capacity_and_hasher(50, BuildHasherDefault::new()));
                    unique_loads.push(Map::with_capacity_and_hasher(5, BuildHasherDefault::new()));
                }
                Op::EndLoop => {
                    unique.pop();
                    unique_loads.pop();
                }
                &Op::Load { src, index, .. } => {
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
            for param in self.ops[op_id].op.parameters_mut() {
                if let Some(&new_id) = remaps.get(param) {
                    *param = new_id;
                }
            }
            let temp = self.next_op(op_id);
            if remaps.contains_key(&op_id) {
                self.remove(op_id);
            }
            op_id = temp;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn remove(&mut self, op_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        self.ops.remove(op_id);
    }

    pub fn insert_before(&mut self, before_id: OpId, op: Op) -> OpId {
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let prev = self.ops[before_id].prev;
        let op_node = OpNode { prev, next: before_id, op: op };
        let op_id = self.ops.push(op_node);
        self.ops[before_id].prev = op_id;
        if prev.is_null() {
            self.head = op_id;
        } else {
            self.ops[prev].next = op_id;
        }
        op_id
    }

    pub fn insert_after(&mut self, after_id: OpId, op: Op) -> OpId {
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let next = self.ops[after_id].next;
        let op_node = OpNode { prev: after_id, next, op: op };
        let op_id = self.ops.push(op_node);
        self.ops[after_id].next = op_id;
        if next.is_null() {
            self.tail = op_id;
        } else {
            self.ops[next].prev = op_id;
        }
        op_id
    }

    pub fn move_op_after(&mut self, op_id: OpId, after_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        //println!("moving op={op_id}, after={after_id}");

        // Remove
        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        // Insert
        self.ops[op_id].prev = after_id;
        let next = self.ops[after_id].next;
        self.ops[op_id].next = next;
        self.ops[after_id].next = op_id;
        if next.is_null() {
            self.tail = op_id;
        } else {
            self.ops[next].prev = op_id;
        }
    }

    pub fn move_op_before(&mut self, op_id: OpId, before_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        //println!("moving op={op_id}, before={before_id}");

        // Remove
        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        // Insert
        self.ops[op_id].next = before_id;
        let prev = self.ops[before_id].prev;
        self.ops[op_id].prev = prev;
        self.ops[before_id].prev = op_id;
        if prev.is_null() {
            self.head = op_id;
        } else {
            self.ops[prev].next = op_id;
        }
    }

    pub fn prev_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].prev
    }

    pub fn next_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].next
    }

    pub fn move_constants_to_beginning(&mut self) {
        let mut start = self.head;
        while let Op::Define { .. } = self.at(start) {
            start = self.next_op(start);
        }

        let mut op_id = start;
        let mut start = self.prev_op(start);
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Const(_) = self.at(op_id) {
                self.move_op_after(op_id, start);
                start = op_id;
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    fn remap(&mut self, x: OpId, y: OpId) {
        for op_node in self.ops.values_mut() {
            for param in op_node.op.parameters_mut() {
                if *param == x {
                    *param = y;
                }
            }
        }
    }

    // Constant folding and deletion of useless ops, etc.
    pub fn constant_folding(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => todo!(),
                Op::MMA { .. }
                | Op::Vectorize { .. } // TODO
                | Op::Devectorize { .. } // TODO
                | Op::Store { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::Load { .. }
                | Op::Loop { .. }
                | Op::EndLoop => {}
                Op::Cast { x, dtype } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.cast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.unary(uop));
                    }
                }
                Op::Binary { x, y, bop } => match (self.at(x).clone(), self.at(y).clone()) {
                    (Op::Const(cx), Op::Const(cy)) => {
                        self.ops[op_id].op = Op::Const(Constant::binary(cx, cy, bop));
                    }
                    (Op::Const(cx), _) => match bop {
                        BOp::And if cx.dtype() == DType::Bool => self.remap(op_id, y),
                        BOp::Add if cx.is_zero() => self.remap(op_id, y),
                        BOp::Sub if cx.is_zero() => self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Neg },
                        BOp::Mul if cx.is_zero() => self.ops[op_id].op = Op::Const(cx.dtype().zero_constant()),
                        BOp::Mul if cx.is_one() => self.remap(op_id, y),
                        BOp::Mul if cx.is_two() => self.ops[op_id].op = Op::Binary { x: y, y, bop: BOp::Add },
                        BOp::Mul if cx.is_power_of_two() && cx.dtype() == IDX_T => {
                            let c = self.insert_before(op_id, Op::Const(cx.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x: y, y: c, bop: BOp::BitShiftLeft };
                        }
                        BOp::Div if cx.is_zero() => self.ops[op_id].op = Op::Const(cx.dtype().zero_constant()),
                        BOp::Div if cx.is_one() => self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Reciprocal },
                        BOp::Pow if cx.is_one() => self.ops[op_id].op = Op::Const(cx.dtype().one_constant()),
                        BOp::Maximum if cx.is_minimum() => self.remap(op_id, y),
                        BOp::BitShiftLeft if cx.is_zero() => self.remap(op_id, y),
                        BOp::BitShiftRight if cx.is_zero() => self.remap(op_id, y),
                        _ => {}
                    },
                    (_, Op::Const(cy)) => match bop {
                        BOp::Sub if cy.is_zero() => self.remap(op_id, x),
                        BOp::Div if cy.is_zero() => panic!("Division by constant zero"),
                        BOp::Div if cy.is_one() => self.remap(op_id, x),
                        BOp::Div if cy.is_power_of_two() && cy.dtype() == IDX_T => {
                            let y = self.insert_before(op_id, Op::Const(cy.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::BitShiftRight };
                        }
                        BOp::Mod if cy.is_zero() => panic!("Module by constant zero"),
                        BOp::Mod if cy.is_zero() && cy.dtype() == IDX_T => {
                            let shift = Constant::binary(cy, Constant::idx(1), BOp::Sub);
                            let y = self.insert_before(op_id, Op::Const(shift));
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::BitAnd };
                        }
                        // Consecutive modulo by constant, pick smallest constant
                        BOp::Mod if cy.dtype() == IDX_T => {
                            if let Op::Binary { bop, x: xi, y: yi } = self.ops[x].op {
                                if bop == BOp::Mod
                                    && let Op::Const(ciy) = self.ops[yi].op
                                {
                                    if ciy > cy {
                                        self.ops[op_id].op = Op::Binary { x: xi, y, bop: BOp::Mod };
                                    } else {
                                        self.ops[op_id].op = Op::Binary { x: xi, y: yi, bop: BOp::Mod };
                                    }
                                }
                            }
                        }
                        BOp::Pow if cy.is_zero() => self.ops[op_id].op = Op::Const(cy.dtype().one_constant()),
                        BOp::Pow if cy.is_one() => self.remap(op_id, x),
                        BOp::Pow if cy.is_two() => self.ops[op_id].op = Op::Binary { x, y: x, bop: BOp::Mul },
                        BOp::BitShiftLeft if cy.is_zero() => self.remap(op_id, x),
                        BOp::BitShiftRight if cy.is_zero() => self.remap(op_id, x),
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
                Op::Mad { x, y, z } => {
                    match (self.at(x).clone(), self.at(y).clone(), self.at(z).clone()) {
                        (Op::Const(cx), Op::Const(cy), Op::Const(cz)) => {
                            let mul = Constant::binary(cx, cy, BOp::Mul);
                            self.ops[op_id].op = Op::Const(Constant::binary(mul, cz, BOp::Add));
                        }
                        _ => {}
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    /// Find all multiply add operations and fuse them
    pub fn fuse_mad(&mut self) {
        let mut op_id = self.head;
        let mut rcs = Map::default();
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters() {
                rcs.entry(param).and_modify(|rc| *rc += 1).or_insert(1);
            }
            if let Op::Binary { x: xo, y: yo, bop } = self.ops[op_id].op {
                if bop == BOp::Add {
                    if let Op::Binary { x, y, bop } = self.ops[xo].op {
                        if bop == BOp::Mul && rcs[&xo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: yo };
                        }
                    } else if let Op::Binary { x, y, bop } = self.ops[yo].op {
                        if bop == BOp::Mul && rcs[&yo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: xo };
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
    }

    // Eliminates accs that are not stored into in loops
    pub fn fold_accs(&mut self) {
        // Check if a define exists without a loop that stores into that define
        let mut defines = Map::default();
        let mut loop_level = 0u32;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Define { scope, .. } => {
                    if scope == Scope::Register {
                        defines.insert(op_id, loop_level);
                    }
                }
                Op::Store { dst, .. } => {
                    //println!("Store to {dst}, loop_level={loop_level}");
                    if let Some(level) = defines.get(&dst) {
                        if loop_level > *level {
                            defines.remove(&dst);
                        }
                    }
                }
                Op::Loop { .. } => {
                    loop_level += 1;
                }
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        break;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        //println!("defines: {defines:?}");
        for (define, _) in defines {
            self.fold_acc(define);
        }
    }

    pub fn fold_acc(&mut self, define_id: OpId) {
        //println!("Folding acc {define_id}");
        let Op::Define { len, .. } = self.ops[define_id].op else { unreachable!() };
        self.remove(define_id);
        let mut latest_stores = vec![OpId::NULL; len];

        let mut remaps = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match *self.at(op_id) {
                Op::Store { dst, x, index, vlen } => {
                    if vlen > 1 {
                        todo!()
                    }
                    if dst == define_id {
                        self.remove(op_id);
                        // x may have been removed as a previous load. If that was the case, the load was redundant
                        if self.ops.contains_key(x) {
                            let Op::Const(index) = self.ops[index].op else { unreachable!() };
                            let Constant::U32(index) = index else { unreachable!() };
                            latest_stores[index as usize] = x;
                            //println!("Latest stores = {latest_stores:?}");
                        }
                        op_id = next;
                        continue;
                    }
                }
                Op::Load { src, index, .. } => {
                    if src == define_id {
                        self.remove(op_id);
                        let Op::Const(index) = self.ops[index].op else { unreachable!() };
                        let Constant::U32(index) = index else { unreachable!() };
                        remaps.insert(op_id, latest_stores[index as usize]);
                        op_id = next;
                        continue;
                    }
                }
                _ => {}
            }
            self.ops[op_id].op.remap_params(&remaps);
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn swap_commutative(&mut self) {
        // Tracks whether a value depends on a loop index
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.at(op_id) {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => unreachable!(),
                Op::Devectorize { .. } | Op::MMA { .. } | Op::Vectorize { .. } => loop_depth,
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } => loop_dep[x],
                &Op::Binary { x, y, bop } => {
                    if bop.is_commutative() && !self.ops[x].op.is_const() {
                        if loop_dep[&x] > loop_dep[&y] || self.ops[y].op.is_const() {
                            //println!("Swapping {x}, {y}, loop dep {} > {}: {:?}, {:?}", loop_dep[&x], loop_dep[&y], self.ops[x].op, self.ops[y].op);
                            if let Op::Binary { x, y, .. } = &mut self.ops[op_id].op {
                                std::mem::swap(x, y);
                            }
                        }
                    }
                    loop_dep[&x].max(loop_dep[&y])
                }
                Op::Mad { x, y, z } => loop_dep[&x].max(loop_dep[&y]).max(loop_dep[&z]),
                Op::Load { .. } | Op::Store { .. } | Op::Const(_) | Op::Define { .. } => loop_depth,
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn reassociate_commutative(&mut self) {
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.at(op_id) {
                Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Reduce { .. }
                | Op::MMA { .. }  => unreachable!(),
                Op::Vectorize { ops } => {
                    let mut max = 0;
                    for op in ops {
                        max = max.max(loop_dep[op]);
                    }
                    max
                }
                Op::Devectorize { .. } => todo!(),
                Op::Mad { x, y, z } => loop_dep[x].max(loop_dep[y]).max(loop_dep[z]),
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } => loop_dep[x],
                Op::Binary { x, y, .. } => loop_dep[x].max(loop_dep[y]),
                Op::Load { .. } | Op::Store { .. } | Op::Const(_) | Op::Define { .. } => loop_depth,
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        let mut op_id = self.head;
        'a: while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { bop, .. } = self.at(op_id) {
                if !bop.is_commutative() || !bop.is_associative() {
                    op_id = next;
                    continue 'a;
                }

                // Get all the leafs
                let mut params = vec![op_id];
                let mut chain = Vec::new();
                while let Some(param) = params.pop() {
                    if let &Op::Binary { x, y, bop: t_bop } = self.at(param) {
                        if t_bop == bop {
                            params.push(x);
                            params.push(y);
                            continue;
                        }
                    }
                    chain.push(param);
                    // We have to be somewhat reasonabe about those chains
                    if chain.len() > 20 {
                        op_id = next;
                        continue 'a;
                    }
                }
                if chain.len() < 2 {
                    op_id = next;
                    continue 'a;
                }
                chain.sort_by_key(|id| loop_dep[id]);

                // Rebuild chain
                let mut prev_acc = chain[0];
                let mut j = 1;
                while j < chain.len() - 1 {
                    let op = Op::Binary { x: prev_acc, y: chain[j], bop };
                    let new_acc = self.insert_before(op_id, op);
                    prev_acc = new_acc;
                    j += 1;
                }
                self.ops[op_id].op = Op::Binary { x: prev_acc, y: chain[j], bop };
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn loop_invariant_code_motion(&mut self) {
        let mut endloop_is = Vec::new();
        let mut loop_id = self.tail;
        while !loop_id.is_null() {
            if *self.at(loop_id) == Op::EndLoop {
                endloop_is.push(loop_id);
            }
            if let Op::Loop { .. } = self.at(loop_id) {
                let mut op_ids_in_loop = Set::default();
                op_ids_in_loop.insert(loop_id); // Loop op is the primary op that breaks LICM

                let mut op_id = loop_id;
                let endloop_id = endloop_is.pop().unwrap();
                while op_id != endloop_id {
                    let op = self.at(op_id);
                    let next_op_id = self.next_op(op_id);

                    if !matches!(
                        op,
                        Op::Store { .. } | Op::Load { .. } | Op::Loop { .. } | Op::EndLoop | Op::Define { .. }
                    ) && op.parameters().all(|op_id| !op_ids_in_loop.contains(&op_id))
                    {
                        self.move_op_before(op_id, loop_id);
                    } else {
                        op_ids_in_loop.insert(op_id);
                    }

                    op_id = next_op_id;
                }
            }
            loop_id = self.prev_op(loop_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    // Loops that don't contain stores can be deleted
    pub fn delete_empty_loops(&mut self) {
        let mut stack: Vec<(bool, Vec<OpId>)> = Vec::new();
        let mut dead = Set::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            for s in &mut stack {
                s.1.push(op_id);
            }
            match self.at(op_id) {
                Op::Loop { .. } => stack.push((false, vec![op_id])),
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
            op_id = self.next_op(op_id);
        }
        for op_id in dead {
            self.remove(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn verify(&self) {
        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|vec| vec.contains(&x)) {
                self.debug();
                panic!(
                    "{op_id} {:?} uses {x} -> {:?} before declaration.",
                    self.ops[op_id].op, self.ops[x].op
                );
            }
        };

        let mut op_id = self.head;
        let mut prev: OpId;
        let mut dtypes: Map<OpId, DType> = Map::default();
        while !op_id.is_null() {
            stack.last_mut().unwrap().insert(op_id);
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    dtypes.insert(op_id, x.0.dtype());
                }
                Op::LoadView(ref x) => {
                    dtypes.insert(op_id, x.0);
                }
                Op::StoreView { src, .. } => {
                    check(op_id, src, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Store { dst, x, index, vlen: _ } => {
                    check(op_id, dst, &stack);
                    check(op_id, x, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Cast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Reduce { x, .. } | Op::Unary { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Binary { x, y, bop } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    if dtypes[&x] != dtypes[&y] {
                        self.debug();
                        panic!("Binary dtype mismatch on op={op_id}.");
                    }
                    if bop.returns_bool() {
                        dtypes.insert(op_id, DType::Bool);
                    } else {
                        dtypes.insert(op_id, dtypes[&x]);
                    }
                }
                Op::Vectorize { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            self.debug();
                            panic!("Vectorize dtype mismatch on op={op_id}.");
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Devectorize { .. } => todo!(),
                Op::MMA { c, a, b, .. } => {
                    let dtype = dtypes[&c];
                    check(op_id, c, &stack);
                    check(op_id, a, &stack);
                    check(op_id, b, &stack);
                    if dtypes[&a] != dtype || dtypes[&b] != dtype {
                        self.debug();
                        panic!("MMA dtype mismatch on op={op_id}.");
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Mad { x, y, z } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    check(op_id, z, &stack);
                    if dtypes[&x] != dtypes[&y] || dtypes[&x] != dtypes[&z] {
                        self.debug();
                        panic!("Mad dtype mismatch on op={op_id}.");
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Const(v) => {
                    dtypes.insert(op_id, v.dtype());
                }
                Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Loop { .. } => {
                    stack.push(Set::default());
                    dtypes.insert(op_id, IDX_T);
                }
                Op::EndLoop => {
                    if stack.is_empty() {
                        self.debug();
                        panic!("Endloop without matching loop.");
                    }
                    stack.pop();
                }
            }
            prev = op_id;
            op_id = self.ops[op_id].next;
            if !op_id.is_null() && self.ops[op_id].prev != prev {
                self.debug();
                panic!("Inconsistency in prev.");
            }
        }
        if stack.len() != 1 {
            self.debug();
            panic!("Wrong {} closing endloops.", stack.len());
        }
        self.check_oob();
    }

    pub fn check_oob(&self) {
        use std::collections::HashMap;
        let mut ids: Map<OpId, (usize, usize)> = HashMap::default();
        let mut defines = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    if x.is_positive() {
                        let Constant::U64(x) = x.cast(DType::U64) else { unreachable!() };
                        let v = usize::from_le_bytes(x);
                        ids.insert(op_id, (v, v));
                    }
                }
                Op::Define { len, .. } => {
                    defines.insert(op_id, len);
                }
                Op::Cast { x, .. } => {
                    if let Some((l, u)) = ids.get(&x) {
                        ids.insert(op_id, (*l, *u));
                    }
                }
                Op::Binary { x, y, bop } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                    {
                        ids.insert(
                            op_id,
                            match bop {
                                BOp::Add => (xl.wrapping_add(yl), xu.wrapping_add(yu)),
                                BOp::Sub => (xl.wrapping_sub(yl), xu.wrapping_sub(yu)),
                                BOp::Mul => (xl.wrapping_mul(yl), xu.wrapping_mul(yu)),
                                BOp::Div => (xl / yl, xu / yu),
                                BOp::Mod => (xl % yl, xu % yu),
                                BOp::Eq => ((xl == yl) as usize, (xu == yu) as usize),
                                BOp::NotEq => ((xl != yl) as usize, (xu != yu) as usize),
                                BOp::Cmpgt => ((xl > yl) as usize, (xu > yu) as usize),
                                BOp::Cmplt => ((xl < yl) as usize, (xu < yu) as usize),
                                BOp::And => ((xl == 1 && yl == 1) as usize, (xu == 1 && yu == 1) as usize),
                                BOp::BitShiftLeft => (xl << yl, xu << yu),
                                BOp::BitShiftRight => (xl >> yl, xu >> yu),
                                BOp::Pow => (xl.pow(yl as u32), xu.pow(yu as u32)),
                                op => todo!("{:?}", op),
                            },
                        );
                    }
                }
                Op::Mad { x, y, z } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                        && let Some(&(zl, zu)) = ids.get(&z)
                    {
                        ids.insert(
                            op_id,
                            (
                                xl.wrapping_mul(yl).wrapping_add(zl),
                                xu.wrapping_mul(yu).wrapping_add(zu),
                            ),
                        );
                    }
                }
                Op::Loop { dim, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                }
                Op::Load { src, index, .. } => {
                    if !ids.contains_key(&index) {
                        self.debug();
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&src] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&src]
                        );
                    }
                }
                Op::Store { dst, index, .. } => {
                    if !ids.contains_key(&index) {
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&dst] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&dst]
                        );
                    }
                }
                Op::Vectorize { ref ops } => {
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = ids.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    if let Some((xl, xu)) = r {
                        ids.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }

    pub fn push_back(&mut self, op: Op) -> OpId {
        debug_assert!(!self.ops.is_empty());
        let op_node = OpNode { prev: self.tail, next: OpId::NULL, op };
        let op_id = self.ops.push(op_node);
        self.ops[self.tail].next = op_id;
        self.tail = op_id;
        op_id
    }

    pub fn remove_first_output(&mut self, x: TensorId) {
        //println!("removing tensor {x} from kernel {kid:?}");
        let outputs = &mut self.outputs;
        outputs.iter().position(|elem| *elem == x).map(|i| outputs.remove(i));
    }

    pub fn drop_unused_ops(&mut self, visited: &Map<TensorId, (KMKernelId, OpId)>) {
        let params = self.outputs.iter().map(|tid| visited[tid].1).collect();
        let required = self.get_required_ops(params);
        let mut loaded_tensors = Vec::new();
        let mut load_index = 0;
        let loads = self.loads.clone(); // TODO remove the clone once partial borrows are working in rust
        let mut op_id = self.head;
        while !op_id.is_null() {
            let is_required = required.contains(&op_id);
            if let Op::LoadView { .. } = self.at(op_id) {
                if is_required {
                    loaded_tensors.push(loads[load_index]);
                }
                load_index += 1;
            }
            let temp = op_id;
            op_id = self.next_op(op_id);
            if !is_required {
                self.remove(temp);
            }
        }
        self.loads = loaded_tensors;
        #[cfg(debug_assertions)]
        if self.loads.len() != self.ops.values().filter(|op| matches!(op.op, Op::LoadView { .. })).count() {
            self.debug();
            panic!();
        }
    }

    pub fn get_required_ops(&self, mut params: Vec<OpId>) -> Set<OpId> {
        let mut required = Set::default();
        while let Some(param) = params.pop() {
            if required.insert(param) {
                //println!("param={param}");
                match self.at(param) {
                    Op::Reduce { x, .. } | Op::Cast { x, .. } | Op::Unary { x, .. } => {
                        params.push(*x);
                    }
                    Op::Binary { x, y, .. } => {
                        params.push(*x);
                        params.push(*y);
                    }
                    Op::Const { .. } | Op::ConstView { .. } | Op::LoadView { .. } => {}
                    Op::Vectorize { .. }
                    | Op::Devectorize { .. }
                    | Op::MMA { .. }
                    | Op::Define { .. }
                    | Op::Mad { .. }
                    | Op::StoreView { .. }
                    | Op::Load { .. }
                    | Op::Store { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop { .. } => unreachable!(),
                }
            }
        }
        required
    }
}

impl OpId {
    pub const NULL: Self = Self(u32::MAX);

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
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
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
