//! Intermediate representation that is close to assembly.
//! It is passed into different backends. Each backend
//! compiles IR into their own bytecode.

use super::{kernel::Op, node::ROp};
use crate::{
    dtype::Constant,
    kernel::TId,
    node::{BOp, UOp},
    DType,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Write},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reg {
    // This is id into kernel.registers
    Var(u16),
    Const(Constant),
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Global,
    Local,
    RegTile,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IROp {
    // Loads variable from address to variable z at give offset.
    Load {
        z: u16,
        // Address is variable in addressables
        address: u16,
        // Offset is u32 var that needs to be added to address
        offset: Reg,
    },
    // Stores variable from variable x to address as given offset.
    Store {
        // Address is variable in addressables
        address: u16,
        // Offset is u32 var that needs to be added to address
        offset: Reg,
        x: Reg,
    },
    Unary {
        z: u16,
        x: u16,
        uop: UOp,
    },
    Binary {
        z: u16,
        x: Reg,
        y: Reg,
        bop: BOp,
    },
    // z = a * b + c
    MAdd {
        z: u16,
        a: Reg,
        b: Reg,
        c: Reg,
    },
    // while id < len
    Loop {
        id: u16,
        len: usize,
    },
    EndLoop {
        id: u16,
        len: usize,
    },
    // TODO
    Barrier {
        scope: Scope,
    },
}

// TODO add vectorization
#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IRVec {
    Scalar,
    V2,
    V4,
    V8,
    V16,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IRKernel {
    // Index of var is it's Id
    // All addressable variables (those that use indexing for access)
    // These can be global args, local variables or variables in registers
    // scope, dtype, byte_size, read_only
    pub(super) addressables: Vec<(Scope, DType, usize, bool)>,
    // Registers with single value or vector stored in them
    // dtype
    pub(super) registers: Vec<DType>,
    pub(super) ops: Vec<IROp>,
}

impl IRVec {
    // TODO vectorization
    #[allow(unused)]
    pub(super) const fn len(self) -> usize {
        match self {
            IRVec::Scalar => 1,
            IRVec::V2 => 2,
            IRVec::V4 => 4,
            IRVec::V8 => 8,
            IRVec::V16 => 16,
        }
    }
}

impl Display for IRVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IRVec::Scalar => f.write_str(""),
            IRVec::V2 => f.write_char('2'),
            IRVec::V4 => f.write_char('4'),
            IRVec::V8 => f.write_char('8'),
            IRVec::V16 => f.write_str("16"),
        }
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::RegTile => "t",
            Self::Register => "r",
        })
    }
}

// Indexing also needs to be rewritten so that as much of it happens outside of the loops
// and so that it does work properly

pub struct IRCompiler {
    pub(super) ops: Vec<IROp>,
    register_map: BTreeMap<TId, Reg>,
    pointers_map: BTreeMap<(TId, Scope), u16>,
    dtypes: Vec<DType>,
}

impl IRCompiler {
    pub(super) fn load(&mut self, address: u16, offset: Reg, dtype: DType) -> u16 {
        self.dtypes.push(dtype);
        let z = u16::try_from(self.dtypes.len() - 1).unwrap();
        self.ops.push(IROp::Load { z, address, offset });
        z
    }

    fn unary_op(&mut self, x: Reg, uop: UOp) -> Reg {
        match x {
            Reg::Var(x) => {
                if let UOp::Cast(dt) = uop {
                    self.dtypes.push(dt);
                } else {
                    self.dtypes.push(self.dtypes[x as usize]);
                }
                let max_id = u16::try_from(self.dtypes.len() - 1).unwrap();
                self.ops.push(IROp::Unary { z: max_id, x, uop });
                Reg::Var(max_id)
            }
            Reg::Const(c) => Reg::Const(c.unary(uop)),
        }
    }

    fn binary_op(&mut self, x: Reg, y: Reg, bop: BOp) -> Reg {
        match x {
            Reg::Var(x) => {
                self.dtypes.push(self.dtypes[x as usize]);
            }
            Reg::Const(x) => {
                self.dtypes.push(x.dtype());
            }
        }
        match bop {
            BOp::Cmplt | BOp::Cmpgt | BOp::Or | BOp::And | BOp::NotEq => {
                *self.dtypes.last_mut().unwrap() = DType::Bool;
            }
            _ => {}
        }
        let z = u16::try_from(self.dtypes.len() - 1).unwrap();
        self.ops.push(IROp::Binary { z, x, y, bop });
        Reg::Var(z)
    }

    pub(super) fn cast(&mut self, x: Reg, dtype: DType) -> Reg {
        self.unary_op(x, UOp::Cast(dtype))
    }

    pub(super) fn and(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::And)
    }

    pub(super) fn cmplt(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Cmplt)
    }

    pub(super) fn cmpgt(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Cmpgt)
    }

    pub(super) fn add(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Add)
    }

    pub(super) fn sub(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Sub)
    }

    pub(super) fn mul(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Mul)
    }

    pub(super) fn div(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Div)
    }

    pub(super) fn mod_(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary_op(x, y, BOp::Mod)
    }

    pub(super) fn mad(&mut self, x: Reg, y: Reg, z: Reg) -> Reg {
        let t = self.mul(x, y);
        self.add(t, z)
    }

    fn vops_to_ssa_ir(
        kernel_ops: &[Op],
        args: &mut Vec<TId>,
        addressables: &mut Vec<(Scope, DType, usize, bool)>,
    ) -> IRCompiler {
        let mut c = IRCompiler {
            ops: Vec::new(),
            register_map: BTreeMap::new(),
            pointers_map: BTreeMap::new(),
            dtypes: Vec::new(),
        };

        // Declare global arguments
        for op in kernel_ops {
            match *op {
                Op::Load { z, xscope, ref xview, xdtype, .. } => {
                    if xscope == Scope::Global && !c.pointers_map.contains_key(&(z, xscope)) {
                        args.push(z);
                        let dtype = xdtype;
                        addressables.push((xscope, dtype, xview.original_numel(), true));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        c.pointers_map.insert((z, xscope), id);
                    }
                }
                Op::Store { z, zscope, ref zview, zdtype, xscope, .. } => {
                    if zscope == Scope::Global {
                        c.pointers_map
                            .entry((z, zscope))
                            .and_modify(|&mut id| {
                                assert_eq!(xscope, Scope::Register);
                                // set it to read-write
                                addressables[id as usize].3 = false;
                            })
                            .or_insert_with(|| {
                                args.push(z);
                                addressables.push((zscope, zdtype, zview.original_numel(), false));
                                u16::try_from(addressables.len() - 1).unwrap()
                            });
                    }
                }
                _ => {}
            }
        }

        // Declare local variables
        for op in kernel_ops {
            match *op {
                Op::Load { z, xscope, ref xview, xdtype, .. } => {
                    if xscope == Scope::Local && !c.pointers_map.contains_key(&(z, xscope)) {
                        addressables.push((xscope, xdtype, xview.original_numel(), true));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        c.pointers_map.insert((z, xscope), id);
                    }
                }
                Op::Store { zscope, .. } => {
                    if zscope == Scope::Local {
                        todo!()
                    }
                }
                _ => {}
            }
        }

        // Declare accumulators and get max axis id
        let mut max_axis = 0;
        for op in kernel_ops {
            match *op {
                Op::Accumulator { z, ref view, dtype, .. } => {
                    addressables.push((Scope::RegTile, dtype, view.original_numel(), false));
                    let id = u16::try_from(addressables.len() - 1).unwrap();
                    c.pointers_map.insert((z, Scope::RegTile), id);
                }
                Op::Loop { axis, .. } => max_axis = max_axis.max(u16::try_from(axis).unwrap()),
                _ => {}
            }
        }
        c.dtypes = vec![DType::U64; max_axis as usize + 1];

        // Transpiling from Kernel to IRKernel, Op -> IROp
        let mut loops = Vec::new();
        for op in kernel_ops {
            //println!("{op}");
            match op {
                &Op::Loop { axis, len } => {
                    // Axis always maps to register ids
                    let id = u16::try_from(axis).unwrap();
                    c.ops.push(IROp::Loop { id, len });
                    loops.push((id, len));
                }
                Op::EndLoop => {
                    if let Some((id, len)) = loops.pop() {
                        c.ops.push(IROp::EndLoop { id, len });
                    }
                }
                &Op::Const { z, value, ref view } => {
                    let zreg = view.ir_for_constant_load(&mut c, value);
                    c.register_map.insert(z, zreg);
                }
                &Op::Load { z, zscope, ref zview, xscope, ref xview, xdtype } => {
                    let xaddress = c.pointers_map[&(z, xscope)];
                    let zreg = xview.ir_for_indexed_load(&mut c, xaddress, xdtype);
                    c.register_map.insert(z, zreg);
                    match (zscope, xscope) {
                        (Scope::Local, Scope::Global) => {
                            let zaddress = c.pointers_map[&(z, zscope)];
                            zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                        }
                        (Scope::Register, Scope::RegTile | Scope::Local | Scope::Global) => {}
                        scopes => panic!("Invalid load scopes {scopes:?}. Internal bug."),
                    }
                }
                &Op::Store { z, zscope, ref zview, zdtype, xscope, ref xview } => {
                    match (zscope, xscope) {
                        (Scope::Local, Scope::Register) => {
                            todo!()
                        }
                        (Scope::RegTile, Scope::Register) => {
                            let zaddress = c.pointers_map[&(z, zscope)];
                            let zreg = c.register_map[&z];
                            zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                        }
                        (Scope::Global, Scope::Register | Scope::RegTile) => {
                            let zaddress = c.pointers_map[&(z, zscope)];
                            let zreg =
                                if let Some(&zaddress) = c.pointers_map.get(&(z, Scope::RegTile)) {
                                    xview.ir_for_indexed_load(&mut c, zaddress, zdtype)
                                } else {
                                    c.register_map[&z]
                                };
                            zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                        }
                        scopes => panic!("Invalid store scopes {scopes:?}"),
                    }
                }
                &Op::Accumulator { z, rop, ref view, dtype } => {
                    let address = c.pointers_map[&(z, Scope::RegTile)];
                    let acc_init = Reg::Const(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    });
                    view.ir_for_indexed_store(&mut c, address, acc_init);
                }
                &Op::Move { z, x, .. } => {
                    c.register_map.insert(z, c.register_map[&x]);
                }
                &Op::Unary { z, x, uop } => {
                    let xreg = c.register_map[&x];
                    let zreg = c.unary_op(xreg, uop);
                    c.register_map.insert(z, zreg);
                }
                &Op::Binary { z, x, y, bop } => {
                    let xreg = c.register_map[&x];
                    let yreg = c.register_map[&y];
                    let zreg = c.binary_op(xreg, yreg, bop);
                    c.register_map.insert(z, zreg);
                }
                &Op::Barrier { scope } => {
                    c.ops.push(IROp::Barrier { scope });
                }
            }
        }
        while let Some((id, len)) = loops.pop() {
            c.ops.push(IROp::EndLoop { id, len });
        }
        c
    }

    fn deduplicate_ssa(self) -> (Vec<DType>, Vec<IROp>) {
        let mut ref_counts: BTreeMap<u16, u32> = BTreeMap::new();
        // Get reference counts
        for op in &self.ops {
            match op {
                IROp::Store { x, offset, .. } => {
                    if let &Reg::Var(x) = x {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                    if let &Reg::Var(x) = offset {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
                &IROp::Load { offset: Reg::Var(x), .. } => {
                    ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                }
                &IROp::Unary { x, .. } => {
                    ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                }
                IROp::Binary { x, y, .. } => {
                    if let &Reg::Var(x) = x {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                    if let &Reg::Var(x) = y {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
                IROp::MAdd { a, b, c, .. } => {
                    if let &Reg::Var(x) = a {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                    if let &Reg::Var(x) = b {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                    if let &Reg::Var(x) = c {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
                &IROp::EndLoop { id, .. } => {
                    ref_counts.entry(id).and_modify(|rc| *rc += 1).or_insert(1);
                }
                _ => {}
            }
        }
        // Create new registers and ops but mutable, so that we don't have so many variables
        let mut registers = Vec::new();
        let mut reg_rcs = Vec::new();
        let mut ops = Vec::new();
        let mut cmp = BTreeMap::new();
        for op in self.ops {
            match op {
                IROp::Load { z, address, offset } => {
                    let zr = new_var(
                        &mut registers,
                        &mut reg_rcs,
                        self.dtypes[z as usize],
                        ref_counts[&z],
                    );
                    let offset = if let Reg::Var(offset) = offset {
                        let offset = cmp[&offset];
                        reg_rcs[offset as usize] -= 1;
                        Reg::Var(offset)
                    } else {
                        offset
                    };
                    ops.push(IROp::Load { z: zr, address, offset });
                    cmp.insert(z, zr);
                }
                IROp::Store { address, offset, x } => {
                    let xr = if let Reg::Var(x) = x {
                        let xr = cmp[&x];
                        reg_rcs[xr as usize] -= 1;
                        Reg::Var(xr)
                    } else {
                        x
                    };
                    let offset = if let Reg::Var(offset) = offset {
                        let offset = cmp[&offset];
                        reg_rcs[offset as usize] -= 1;
                        Reg::Var(offset)
                    } else {
                        offset
                    };
                    ops.push(IROp::Store { address, offset, x: xr });
                }
                IROp::Unary { z, x, uop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr =
                            new_var(&mut registers, &mut reg_rcs, self.dtypes[z as usize], zrc);
                        let xr = cmp[&x];
                        reg_rcs[xr as usize] -= 1;
                        ops.push(IROp::Unary { z: zr, x: xr, uop });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Binary { z, x, y, bop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let x = if let Reg::Var(x) = x {
                            let xr = cmp[&x];
                            reg_rcs[xr as usize] -= 1;
                            Reg::Var(xr)
                        } else {
                            x
                        };
                        let y = if let Reg::Var(x) = y {
                            let xr = cmp[&x];
                            reg_rcs[xr as usize] -= 1;
                            Reg::Var(xr)
                        } else {
                            y
                        };
                        let zr =
                            new_var(&mut registers, &mut reg_rcs, self.dtypes[z as usize], zrc);
                        ops.push(IROp::Binary { z: zr, x, y, bop });
                        cmp.insert(z, zr);
                    }
                }
                IROp::MAdd { z, a, b, c } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let a = if let Reg::Var(x) = a {
                            let xr = cmp[&x];
                            reg_rcs[xr as usize] -= 1;
                            Reg::Var(xr)
                        } else {
                            a
                        };
                        let b = if let Reg::Var(x) = b {
                            let xr = cmp[&x];
                            reg_rcs[xr as usize] -= 1;
                            Reg::Var(xr)
                        } else {
                            b
                        };
                        let c = if let Reg::Var(x) = c {
                            let xr = cmp[&x];
                            reg_rcs[xr as usize] -= 1;
                            Reg::Var(xr)
                        } else {
                            c
                        };
                        let zr =
                            new_var(&mut registers, &mut reg_rcs, self.dtypes[z as usize], zrc);
                        ops.push(IROp::MAdd { z: zr, a, b, c });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Loop { id, len } => {
                    if let Some(&zrc) = ref_counts.get(&id) {
                        let zr =
                            new_var(&mut registers, &mut reg_rcs, self.dtypes[id as usize], zrc);
                        ops.push(IROp::Loop { id: zr, len });
                        cmp.insert(id, zr);
                    }
                }
                IROp::EndLoop { id, len } => {
                    reg_rcs[cmp[&id] as usize] -= 1;
                    ops.push(IROp::EndLoop { id, len });
                }
                IROp::Barrier { scope } => ops.push(IROp::Barrier { scope }),
            }
        }
        (registers, ops)
    }

    // i.e. peephole optimization
    // TODO this should only ever fuse ops within single loop body
    fn fuse_ops(&mut self) {
        for i in 0..self.ops.len() - 1 {
            if let IROp::Binary { bop, z: z0, x: a, y: b, .. } = self.ops[i] {
                if bop == BOp::Mul {
                    for j in i + 1..self.ops.len() {
                        if let IROp::Binary { bop, z, x, y, .. } = self.ops[j] {
                            if bop == BOp::Add {
                                if Reg::Var(z0) == x {
                                    self.ops[j] = IROp::MAdd { z, a, b, c: y };
                                }
                                if Reg::Var(z0) == y {
                                    self.ops[j] = IROp::MAdd { z, a, b, c: x };
                                }
                            };
                        }
                        if matches!(self.ops[j], IROp::Loop { .. } | IROp::EndLoop { .. }) {
                            break;
                        }
                    }
                }
            }
        }
    }

    fn loop_unrolling(&mut self) {
        let mut op_i = self.ops.len();
        let mut last_end_loop = Vec::new();
        let mut iters = 0;
        while op_i > 6 {
            op_i -= 1;
            if let IROp::EndLoop { .. } = self.ops[op_i] {
                last_end_loop.push(op_i);
            }
            if let IROp::Loop { id, len } = self.ops[op_i] {
                if len < 32 {
                    let end = last_end_loop.pop().unwrap();
                    let ops: Vec<IROp> = self.ops[op_i + 1..end].into();
                    self.ops.remove(end);
                    self.ops.remove(op_i);
                    self.replace(id, Reg::Const(Constant::U64(len as u64 - 1)));
                    for i in (0..len - 1).rev() {
                        let mut ops = ops.clone();
                        while let Some(op) = ops.pop() {
                            self.ops.insert(op_i, op);
                        }
                        self.replace(id, Reg::Const(Constant::U64(i as u64)));
                    }
                    let x = ops.len() as isize * (len as isize - 1) - 2;
                    for end in &mut last_end_loop {
                        *end = (*end as isize + x) as usize;
                    }
                    iters += 1;
                    if iters > 2 {
                        return;
                    }
                }
            }
        }
        /*
        for (unsigned int r23 = 0; r23 < 1; r23 += 1) {
           for (unsigned int r24 = 0; r24 < 1; r24 += 1) {
             for (unsigned int r25 = 0; r25 < 1; r25 += 1) {
               r26 = r24 + r25;
               r27 = r23 + r26;
               r22 = p2[r27];
               p1[0] = r22;
             }
           }
         }
        */
    }

    fn global_loop_unrolling(&mut self) {
        let mut op_i = 6;
        while op_i > 0 {
            op_i -= 1;
            if let IROp::Loop { id, len } = self.ops[op_i] {
                if len == 1 {
                    self.replace(id, Reg::Const(Constant::U64(0)));
                }
            }
        }
    }

    fn loop_splitting(&mut self) {
        let _ = self;
        // TODO
    }

    fn common_subexpression_elimination(&mut self) {
        let _ = self;
        // TODO
    }

    fn dead_store_elimination(&mut self) {
        let _ = self;
        // TODO
    }

    // Loop invariant code motion and dependence analysis
    fn loop_invariant_code_motion(&mut self) {
        // Go from innermost loop to outermost loop. If there are multiple innermost loops,
        // they can be processed in parallel.
        for op_id in (6..self.ops.len()).rev() {
            if let IROp::Loop { id, .. } = self.ops[op_id] {
                let mut loop_id = op_id;
                // which variables can't be eliminated
                let mut dependents: BTreeSet<u16> = BTreeSet::from([id]);
                let mut inner_loop_counter = 0;
                let mut op_id = loop_id + 1;
                'a: loop {
                    // if operands are not in dependents, move operation before loop
                    #[allow(clippy::match_on_vec_items)]
                    let move_possible: bool = match self.ops[op_id] {
                        IROp::Load { z, offset, .. } => {
                            if let Reg::Var(offset) = offset {
                                if dependents.contains(&offset) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            }
                        }
                        IROp::Store { offset, x, .. } => {
                            let a = if let Reg::Var(offset) = offset {
                                !dependents.contains(&offset)
                            } else {
                                true
                            };
                            let b = if let Reg::Var(x) = x {
                                !dependents.contains(&x)
                            } else {
                                true
                            };
                            a && b
                        }
                        IROp::Unary { z, x, .. } => {
                            if dependents.contains(&x) {
                                dependents.insert(z);
                                false
                            } else {
                                true
                            }
                        }
                        IROp::Binary { z, x, y, .. } => {
                            let a = if let Reg::Var(x) = x {
                                if dependents.contains(&x) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let b = if let Reg::Var(y) = y {
                                if dependents.contains(&y) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            a && b
                        }
                        IROp::MAdd { .. } => todo!(),
                        IROp::Loop { .. } => {
                            inner_loop_counter += 1;
                            // This is a bit more complicated. We have to check all values
                            // in this loop block and move the loop as a whole.
                            // This is however rarely needed due to way we construct loops,
                            // so we do not need to hurry implementing this.
                            false
                        }
                        IROp::EndLoop { .. } => {
                            if inner_loop_counter == 0 {
                                break 'a;
                            }
                            inner_loop_counter -= 1;
                            false
                        }
                        IROp::Barrier { .. } => false,
                    };
                    if move_possible && inner_loop_counter == 0 {
                        let op = self.ops.remove(op_id);
                        self.ops.insert(loop_id, op);
                        loop_id += 1;
                    }
                    op_id += 1;
                }
            }
        }
    }

    fn vectorization(&mut self) {
        let _ = self;
        // TODO
    }

    // Replace all occurences of z with register x
    #[allow(clippy::match_on_vec_items)]
    fn replace(&mut self, to_replace: u16, replace_with: Reg) {
        // TODO make this non recursive
        for i in 0..self.ops.len() {
            match self.ops[i] {
                IROp::Unary { z, x, uop } => {
                    if x == to_replace {
                        if let Reg::Const(replace_with) = replace_with {
                            self.replace(z, Reg::Const(replace_with.unary(uop)));
                        }
                    }
                }
                IROp::Binary { ref mut x, ref mut y, .. } => {
                    if *x == Reg::Var(to_replace) {
                        *x = replace_with;
                    }
                    if *y == Reg::Var(to_replace) {
                        *y = replace_with;
                    }
                }
                IROp::MAdd { ref mut a, ref mut b, ref mut c, .. } => {
                    if *a == Reg::Var(to_replace) {
                        *a = replace_with;
                    }
                    if *b == Reg::Var(to_replace) {
                        *b = replace_with;
                    }
                    if *c == Reg::Var(to_replace) {
                        *c = replace_with;
                    }
                }
                IROp::Load { ref mut offset, .. } => {
                    if *offset == Reg::Var(to_replace) {
                        *offset = replace_with;
                    }
                }
                IROp::Store { ref mut offset, ref mut x, .. } => {
                    if *offset == Reg::Var(to_replace) {
                        *offset = replace_with;
                    }
                    if *x == Reg::Var(to_replace) {
                        *x = replace_with;
                    }
                }
                IROp::Loop { .. } | IROp::EndLoop { .. } | IROp::Barrier { .. } => {}
            }
        }
    }

    /// This includes elimination of useless ops, i.e. y = x*1
    /// and includes peephole optimizations
    #[allow(clippy::match_on_vec_items)]
    #[allow(clippy::single_match)]
    fn constant_folding_and_propagation(&mut self) {
        for i in 0..self.ops.len() {
            match self.ops[i] {
                IROp::Binary { z, x, y, bop } => match (x, y) {
                    (Reg::Var(_), Reg::Var(_)) => {}
                    (Reg::Var(xv), Reg::Const(yv)) => {
                        if yv.is_zero() {
                            match bop {
                                BOp::Mul | BOp::And | BOp::BitAnd => {
                                    self.replace(z, Reg::Const(yv))
                                }
                                BOp::Div | BOp::Mod => panic!("Division by zero constant"),
                                BOp::Pow | BOp::Or => {
                                    self.replace(z, Reg::Const(yv.dtype().one_constant()))
                                }
                                BOp::Add | BOp::Sub | BOp::BitXor | BOp::BitOr => {
                                    self.replace(z, Reg::Var(xv))
                                }
                                BOp::Max => self.ops[i] = IROp::Unary { z, x: xv, uop: UOp::ReLU },
                                BOp::NotEq
                                | BOp::Cmpgt
                                | BOp::Cmplt
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight => {}
                            }
                        } else if yv.is_one() {
                            match bop {
                                BOp::Mul | BOp::Div | BOp::Pow | BOp::BitAnd => {
                                    self.replace(z, Reg::Var(xv))
                                }
                                BOp::Mod => self.replace(z, Reg::Const(yv.dtype().zero_constant())),
                                BOp::BitOr => self.replace(z, Reg::Const(yv)),
                                BOp::BitXor
                                | BOp::Cmplt
                                | BOp::And
                                | BOp::Or
                                | BOp::Cmpgt
                                | BOp::Max
                                | BOp::Add
                                | BOp::Sub
                                | BOp::NotEq
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight => {}
                            }
                        } else if yv.is_two() {
                            match bop {
                                BOp::Add => {}
                                BOp::Sub => {}
                                BOp::Mul => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitShiftLeft,
                                        };
                                    }
                                }
                                BOp::Div => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitShiftRight,
                                        };
                                    }
                                }
                                BOp::Pow => {
                                    self.ops[i] = IROp::Binary {
                                        z,
                                        x: Reg::Var(xv),
                                        y: Reg::Var(xv),
                                        bop: BOp::Mul,
                                    };
                                }
                                BOp::Mod => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitAnd,
                                        };
                                    }
                                }
                                BOp::Cmplt => {}
                                BOp::Cmpgt => {}
                                BOp::Max => todo!(),
                                BOp::Or => todo!(),
                                BOp::And => todo!(),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => {}
                                BOp::BitShiftRight => {}
                            }
                        }
                    }
                    (Reg::Const(xv), Reg::Var(yv)) => {
                        if xv.is_zero() {
                            match bop {
                                BOp::Add => self.replace(z, Reg::Var(yv)),
                                BOp::Sub => self.ops[i] = IROp::Unary { z, x: yv, uop: UOp::Neg },
                                BOp::Mul | BOp::Div | BOp::Pow | BOp::Mod | BOp::And => {
                                    self.replace(z, Reg::Const(xv))
                                }
                                BOp::Cmplt => {}
                                BOp::Cmpgt => {}
                                BOp::Max => self.ops[i] = IROp::Unary { z, x: yv, uop: UOp::ReLU },
                                BOp::Or => self.replace(z, Reg::Var(yv)),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => todo!(),
                                BOp::BitShiftRight => todo!(),
                            }
                        } else if xv.is_one() {
                            match bop {
                                BOp::Add => {}
                                BOp::Sub => todo!(),
                                BOp::Mul => self.replace(z, Reg::Var(yv)),
                                BOp::Div => todo!(),
                                BOp::Pow => todo!(),
                                BOp::Mod => todo!(),
                                BOp::Cmplt => todo!(),
                                BOp::Cmpgt => todo!(),
                                BOp::Max => todo!(),
                                BOp::Or => todo!(),
                                BOp::And => todo!(),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => todo!(),
                                BOp::BitShiftRight => todo!(),
                            }
                        } else if xv.is_two() {
                            match bop {
                                BOp::Add => todo!(),
                                BOp::Sub => todo!(),
                                BOp::Mul => todo!(),
                                BOp::Div => todo!(),
                                BOp::Pow => todo!(),
                                BOp::Mod => todo!(),
                                BOp::Cmplt => todo!(),
                                BOp::Cmpgt => todo!(),
                                BOp::Max => todo!(),
                                BOp::Or => todo!(),
                                BOp::And => todo!(),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::BitShiftLeft => todo!(),
                                BOp::BitShiftRight => todo!(),
                                BOp::NotEq => todo!(),
                            }
                        }
                    }
                    (Reg::Const(x), Reg::Const(y)) => {
                        self.replace(z, Reg::Const(Constant::binary(x, y, bop)));
                    }
                },
                IROp::MAdd { .. } => todo!(),
                IROp::Unary { .. }
                | IROp::Loop { .. }
                | IROp::EndLoop { .. }
                | IROp::Load { .. }
                | IROp::Store { .. }
                | IROp::Barrier { .. } => {}
            }
        }
    }
}

fn new_var(
    registers: &mut Vec<DType>,
    reg_rcs: &mut Vec<u32>,
    ir_dtype: DType,
    ref_count: u32,
) -> u16 {
    for (i, rc) in reg_rcs.iter_mut().enumerate() {
        if *rc == 0 && registers[i] == ir_dtype {
            //registers[i] = ir_dtype;
            reg_rcs[i] = ref_count;
            return u16::try_from(i).unwrap();
        }
    }
    registers.push(ir_dtype);
    reg_rcs.push(ref_count);
    u16::try_from(registers.len() - 1).unwrap()
}

// Returns IRKernel and order in which tensors are passed to it as arguments
// Axes have the same id in registers.
impl IRKernel {
    pub(super) fn new(kernel_ops: &[Op], debug_ir: bool) -> IRKernel {
        // What we need to calculate (outputs of this function)
        // IRKernel
        let mut addressables: Vec<(Scope, DType, usize, bool)> = Vec::new();
        // Returned tensors
        let mut args = Vec::new();

        let mut compiler = IRCompiler::vops_to_ssa_ir(kernel_ops, &mut args, &mut addressables);

        // Optimizations
        // TODO perhaps it is benefitial to do this multiple times???
        compiler.global_loop_unrolling();
        compiler.loop_unrolling();
        //compiler.loop_invariant_code_motion();
        compiler.loop_splitting();
        compiler.vectorization();
        compiler.constant_folding_and_propagation();
        compiler.common_subexpression_elimination();
        compiler.dead_store_elimination();

        compiler.fuse_ops();
        if debug_ir {
            for op in &compiler.ops {
                println!("{op:?}");
            }
        }

        // TODO perhaps we can do even more optimizations with instruction scheduling
        // and register allocation? But that's a big perhaps...
        // TODO loop splitting and loop peeling

        //for op in &compiler.ops { println!("{op:?}"); }
        let (registers, ops) = compiler.deduplicate_ssa();
        //println!();
        //println!();
        //for op in &ops { println!("{op:?}"); }
        //panic!();

        /*if debug_ir {
            for op in &ops {
                println!("{op:?}");
            }
        }*/

        IRKernel { addressables, registers, ops }
    }
}
