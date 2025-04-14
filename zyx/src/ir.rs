//! Intermediate representation that is close to assembly.
//! It is passed into different backends. Each backend
//! compiles IR into their own bytecode.

use super::{kernel::Op, node::ROp};
use crate::{
    DType, DebugMask, Set,
    dtype::Constant,
    kernel::Kernel,
    node::{BOp, UOp},
    optimizer::{OptOp, Optimization},
    shape::Dim,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Write},
};

mod const_folding;
mod elementwise_opts;
mod licm;
mod loop_unrolling;
mod matmul_opts;
mod reduce_opts;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reg {
    // This is id into kernel.registers
    Var(u16),
    Const(Constant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TileScope {
    Local,
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
    Tile {
        scope: TileScope,
    },
    #[allow(unused)]
    SetLocal {
        address: u16,
        len: Dim,
        value: Constant,
    },
    Set {
        z: u16,
        value: Constant,
    },
    Cast {
        z: u16,
        x: u16,
        dtype: DType,
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
        len: Dim,
    },
    EndLoop {
        id: u16,
        len: Dim,
    },
    // This will also be needed
    /*If {
        condition: u16,
    },
    EndIf,*/
    // TODO
    Barrier {
        scope: Scope,
    },
}

impl Display for IROp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const BLUE: &str = "\x1B[34m";
        const GREEN: &str = "\x1B[32m";
        const MAGENTA: &str = "\x1B[35m";
        const RED: &str = "\x1B[31m";
        //const WHITE: &str = "\x1B[37m";
        const YELLOW: &str = "\x1B[33m";
        const RESET: &str = "\x1B[39m";
        match self {
            IROp::Load { z, address, offset } => f.write_fmt(format_args!(
                "{MAGENTA}load r{z} <- p{address}[{}]{RESET}",
                match offset {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                }
            )),
            IROp::Store { address, offset, x } => f.write_fmt(format_args!(
                "{RED}store p{address}[{}] <- r{x:?}{RESET}",
                match offset {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                }
            )),
            IROp::SetLocal { address, len, value } => {
                f.write_fmt(format_args!("set.local p{address}{{{len}}} <- {value}"))
            }
            IROp::Set { z, value } => {
                f.write_fmt(format_args!("{YELLOW}set r{z} <- {value}{RESET}"))
            }
            IROp::Cast { z, x, dtype } => f.write_fmt(format_args!("cast.{dtype} {z} <- {x}")),
            IROp::Unary { z, x, uop } => f.write_fmt(format_args!("u.{uop:?} {z} <- {x}")),
            IROp::Binary { z, x, y, bop } => f.write_fmt(format_args!(
                "r{z} <- {} {} {}",
                match x {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                },
                match bop {
                    BOp::Add => "+",
                    BOp::Sub => "-",
                    BOp::Mul => "*",
                    BOp::Div => "/",
                    BOp::Pow => "^",
                    BOp::Mod => "%",
                    BOp::Cmplt => "<",
                    BOp::Cmpgt => ">",
                    BOp::Max => "max",
                    BOp::Or => "||",
                    BOp::And => "&&",
                    BOp::BitXor => "xor",
                    BOp::BitOr => "|",
                    BOp::BitAnd => "&",
                    BOp::BitShiftLeft => "<<",
                    BOp::BitShiftRight => ">>",
                    BOp::NotEq => "!=",
                },
                match y {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                }
            )),
            IROp::MAdd { z, a, b, c } => f.write_fmt(format_args!(
                "r{z} <- {} * {} + {}",
                match a {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                },
                match b {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                },
                match c {
                    Reg::Var(x) => format!("r{x}"),
                    Reg::Const(constant) => format!("{constant}"),
                }
            )),
            IROp::Loop { id, len } => {
                f.write_fmt(format_args!("{GREEN}for r{id} in 0..{len}{RESET}"))
            }
            IROp::EndLoop { id, .. } => f.write_fmt(format_args!("endloop r{id}")),
            IROp::Barrier { scope } => f.write_fmt(format_args!("{BLUE}barrier.{scope}{RESET}")),
            IROp::Tile { scope } => f.write_fmt(format_args!("{BLUE}tile.{scope:?}{RESET}")),
        }
    }
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
            Self::Register => "r",
        })
    }
}

// Indexing also needs to be rewritten so that as much of it happens outside of the loops
// and so that it does work properly

#[derive(Clone, PartialEq, Eq)]
pub struct IRCompiler {
    pub(super) ops: Vec<IROp>,
    // address => DType,
    load_dtypes: BTreeMap<u16, DType>,
    max_id: u16,
}

impl IRCompiler {
    pub(super) fn load(&mut self, address: u16, offset: Reg, dtype: DType) -> u16 {
        self.load_dtypes.insert(address, dtype);
        self.max_id += 1;
        let z = self.max_id;
        self.ops.push(IROp::Load { z, address, offset });
        z
    }

    fn set(&mut self, value: Constant) -> Reg {
        self.max_id += 1;
        let z = self.max_id;
        self.ops.push(IROp::Set { z, value });
        Reg::Var(z)
    }

    fn unary_op(&mut self, x: Reg, uop: UOp) -> Reg {
        match x {
            Reg::Var(x) => {
                self.max_id += 1;
                let z = self.max_id;
                self.ops.push(IROp::Unary { z, x, uop });
                Reg::Var(z)
            }
            Reg::Const(c) => Reg::Const(c.unary(uop)),
        }
    }

    fn binary_op(&mut self, x: Reg, y: Reg, bop: BOp) -> Reg {
        if let Reg::Const(x) = x {
            if let Reg::Const(y) = y {
                return Reg::Const(Constant::binary(x, y, bop));
            }
        }
        self.max_id += 1;
        let z = self.max_id;
        self.ops.push(IROp::Binary { z, x, y, bop });
        Reg::Var(z)
    }

    pub(super) fn cast(&mut self, x: Reg, dtype: DType) -> Reg {
        match x {
            Reg::Var(x) => {
                self.max_id += 1;
                let z = self.max_id;
                self.ops.push(IROp::Cast { z, x, dtype });
                Reg::Var(z)
            }
            Reg::Const(c) => Reg::Const(c.cast(dtype)),
        }
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
        addressables: &mut Vec<(Scope, DType, usize, bool)>,
    ) -> IRCompiler {
        let max_id = kernel_ops
            .iter()
            .map(|op| {
                if let Op::Loop { axis, .. } = op {
                    (*axis).try_into().unwrap()
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        let mut c = IRCompiler { ops: Vec::new(), load_dtypes: BTreeMap::new(), max_id };
        let mut register_map = BTreeMap::new();
        let mut pointers_map = BTreeMap::new();

        // Declare global arguments
        for op in kernel_ops {
            match *op {
                Op::Load { z: _, xscope, x, ref xview, xdtype, zscope: _, zview: _ } => {
                    if xscope == Scope::Global && !pointers_map.contains_key(&(x, xscope)) {
                        //args.push(x);
                        let dtype = xdtype;
                        addressables.push((xscope, dtype, xview.original_numel(), true));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        pointers_map.insert((x, xscope), id);
                    }
                }
                Op::Store { z, zscope, ref zview, zdtype, x: _, xscope, xview: _ } => {
                    if zscope == Scope::Global {
                        pointers_map
                            .entry((z, zscope))
                            .and_modify(|&mut id| {
                                debug_assert_eq!(xscope, Scope::Register);
                                // set it to read-write
                                addressables[id as usize].3 = false;
                            })
                            .or_insert_with(|| {
                                //args.push(z);
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
                    if xscope == Scope::Local && !pointers_map.contains_key(&(z, xscope)) {
                        addressables.push((xscope, xdtype, xview.original_numel(), true));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        pointers_map.insert((z, xscope), id);
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
                    let (id, len) = loops.pop().unwrap();
                    c.ops.push(IROp::EndLoop { id, len });
                }
                &Op::Const { z, value, ref view } => {
                    let zreg = view.ir_for_constant_load(&mut c, value);
                    register_map.insert(z, zreg);
                }
                &Op::Load { z, zscope, ref zview, x, xscope, ref xview, xdtype } => {
                    let xaddress = pointers_map[&(x, xscope)];
                    let zreg = xview.ir_for_indexed_load(&mut c, xaddress, xdtype);
                    register_map.insert(z, zreg);
                    match (zscope, xscope) {
                        (Scope::Local, Scope::Global) => {
                            let zaddress = pointers_map[&(x, zscope)];
                            zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                        }
                        (Scope::Register, Scope::Local | Scope::Global) => {}
                        scopes => panic!("Invalid load scopes {scopes:?}. Internal bug."),
                    }
                }
                &Op::Store { z, zscope, ref zview, x, xscope, .. } => match (zscope, xscope) {
                    (Scope::Local, Scope::Register) => {
                        todo!()
                    }
                    (Scope::Global, Scope::Register) => {
                        let zaddress = pointers_map[&(z, zscope)];
                        let xreg = register_map[&x];
                        zview.ir_for_indexed_store(&mut c, zaddress, xreg);
                    }
                    scopes => panic!("Invalid store scopes {scopes:?}"),
                },
                &Op::Accumulator { z, rop, dtype } => {
                    let acc_init = match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    };
                    let zreg = c.set(acc_init);
                    register_map.insert(z, zreg);
                }
                &Op::Cast { z, x, dtype } => {
                    let xreg = register_map[&x];
                    let zreg = c.cast(xreg, dtype);
                    register_map.insert(z, zreg);
                }
                &Op::Unary { z, x, uop } => {
                    let xreg = register_map[&x];
                    let zreg = c.unary_op(xreg, uop);
                    register_map.insert(z, zreg);
                }
                &Op::Binary { z, x, y, bop } => {
                    let xreg = register_map[&x];
                    let yreg = register_map[&y];
                    if let Some(&z) = register_map.get(&z) {
                        if let Reg::Var(z) = z {
                            c.ops.push(IROp::Binary { z, x: xreg, y: yreg, bop });
                        }
                    } else {
                        let zreg = c.binary_op(xreg, yreg, bop);
                        register_map.insert(z, zreg);
                    }
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

    /// Converts from SSA form to using as few registers as possible + dead store elimination
    #[allow(clippy::cognitive_complexity)]
    fn reduce_register_use(self) -> (Vec<DType>, Vec<IROp>) {
        // iterate over all loops
        // for each loop make a map of ref counts

        let mut ref_counts: BTreeMap<u16, u32> = BTreeMap::new();
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
                &IROp::Cast { x, .. } | &IROp::Unary { x, .. } => {
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
                IROp::Loop { .. }
                | IROp::Set { .. }
                | IROp::SetLocal { .. }
                | IROp::Load { offset: Reg::Const(_), .. }
                | IROp::Barrier { .. } => {}
                _ => {}
            }
        }

        // Find all variables used inside of loops, but declared outside and not used outside.
        // Increase their ref count by one.
        let mut used: Vec<BTreeSet<u16>> = Vec::new();
        used.push(BTreeSet::new());
        let mut loop_level = 0;
        for op in self.ops.iter().rev() {
            match op {
                IROp::Load { z, offset, .. } => {
                    if !used[loop_level].contains(z) {
                        if let Some(rc) = ref_counts.get_mut(z) {
                            *rc += 1;
                        }
                    }
                    if let Reg::Var(x) = offset {
                        used[loop_level].insert(*x);
                    }
                }
                IROp::Store { offset, x, .. } => {
                    if let Reg::Var(x) = offset {
                        used[loop_level].insert(*x);
                    }
                    if let Reg::Var(x) = x {
                        used[loop_level].insert(*x);
                    }
                }
                IROp::Cast { z, x, .. } => {
                    if !used[loop_level].contains(z) {
                        if let Some(rc) = ref_counts.get_mut(z) {
                            *rc += 1;
                        }
                    }
                    used[loop_level].insert(*x);
                }
                IROp::Unary { z, x, .. } => {
                    if !used[loop_level].contains(z) {
                        if let Some(rc) = ref_counts.get_mut(z) {
                            *rc += 1;
                        }
                    }
                    used[loop_level].insert(*x);
                }
                IROp::Binary { z, x, y, .. } => {
                    if !used[loop_level].contains(z) {
                        if let Some(rc) = ref_counts.get_mut(z) {
                            *rc += 1;
                        }
                    }
                    if let Reg::Var(x) = x {
                        used[loop_level].insert(*x);
                    }
                    if let Reg::Var(x) = y {
                        used[loop_level].insert(*x);
                    }
                }
                IROp::MAdd { z, a, b, c } => {
                    if !used[loop_level].contains(z) {
                        if let Some(rc) = ref_counts.get_mut(z) {
                            *rc += 1;
                        }
                    }
                    if let Reg::Var(x) = a {
                        used[loop_level].insert(*x);
                    }
                    if let Reg::Var(x) = b {
                        used[loop_level].insert(*x);
                    }
                    if let Reg::Var(x) = c {
                        used[loop_level].insert(*x);
                    }
                }
                IROp::Loop { .. } => {
                    loop_level -= 1;
                }
                &IROp::EndLoop { .. } => {
                    used.push(BTreeSet::new());
                    loop_level += 1;
                }
                IROp::SetLocal { .. } | IROp::Set { .. } | IROp::Barrier { .. } => {}
                _ => {}
            }
        }

        //println!("Ref counts:\n{ref_counts:?}");
        let mut registers = Vec::new();
        let mut reg_rcs = Vec::new();
        let mut ops = Vec::new();
        let mut cmp = BTreeMap::new();
        for op in self.ops {
            match op {
                IROp::Load { z, address, offset } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            self.load_dtypes[&address],
                            zrc,
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
                IROp::SetLocal { address, len, value: values } => {
                    ops.push(IROp::SetLocal { address, len, value: values });
                }
                IROp::Set { z, value } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr = new_var(&mut registers, &mut reg_rcs, value.dtype(), zrc);
                        ops.push(IROp::Set { z: zr, value });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Cast { z, x, dtype } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr = new_var(&mut registers, &mut reg_rcs, dtype, zrc);
                        let xr = cmp[&x];
                        reg_rcs[xr as usize] -= 1;
                        ops.push(IROp::Cast { z: zr, x: xr, dtype });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Unary { z, x, uop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let xr = cmp[&x];
                        let dtype = registers[xr as usize];
                        let zr = new_var(&mut registers, &mut reg_rcs, dtype, zrc);
                        reg_rcs[xr as usize] -= 1;
                        ops.push(IROp::Unary { z: zr, x: xr, uop });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Binary { z, x, y, bop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        //println!("b.{bop:?} {z} <- {x:?}, {y:?}");
                        let dtype = match bop {
                            BOp::Cmplt | BOp::Cmpgt | BOp::Or | BOp::And | BOp::NotEq => {
                                DType::Bool
                            }
                            _ => match x {
                                Reg::Var(x) => registers[cmp[&x] as usize],
                                Reg::Const(constant) => constant.dtype(),
                            },
                        };
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
                        if let Some(&zr) = cmp.get(&z) {
                            ops.push(IROp::Binary { z: zr, x, y, bop });
                        } else {
                            let zr = new_var(&mut registers, &mut reg_rcs, dtype, zrc);
                            cmp.insert(z, zr);
                            ops.push(IROp::Binary { z: zr, x, y, bop });
                        }
                    }
                }
                IROp::MAdd { z, a, b, c } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let dtype = match a {
                            Reg::Var(a) => registers[cmp[&a] as usize],
                            Reg::Const(constant) => constant.dtype(),
                        };
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
                        if let Some(&zr) = cmp.get(&z) {
                            ops.push(IROp::MAdd { z: zr, a, b, c });
                        } else {
                            let zr = new_var(&mut registers, &mut reg_rcs, dtype, zrc);
                            cmp.insert(z, zr);
                            ops.push(IROp::MAdd { z: zr, a, b, c });
                        }
                    }
                }
                IROp::Loop { id, len } => {
                    let &zrc = ref_counts.get(&id).unwrap();
                    let zr = new_var(&mut registers, &mut reg_rcs, DType::U64, zrc);
                    ops.push(IROp::Loop { id: zr, len });
                    //println!("loop {id}, len {len}, {cmp:?}");
                    cmp.insert(id, zr);
                }
                IROp::EndLoop { id, len } => {
                    //println!("Endloop {id}, len {len}, {cmp:?}");
                    reg_rcs[cmp[&id] as usize] -= 1;
                    ops.push(IROp::EndLoop { id, len });
                }
                op => ops.push(op),
            }
        }
        (registers, ops)
    }

    // i.e. peephole optimization, algebraic optimization, ops merging, ...
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

    const fn common_subexpression_elimination(&self) {
        let _ = self;
        // TODO
    }

    // Replace all occurences of z with register x
    #[allow(clippy::match_on_vec_items)]
    fn replace(&mut self, to_replace: u16, replace_with: Reg, begin: usize) {
        // TODO make this non recursive
        for i in begin..self.ops.len() {
            match self.ops[i] {
                IROp::Cast { z, ref mut x, dtype } => {
                    if *x == to_replace {
                        match replace_with {
                            Reg::Var(replace_with) => *x = replace_with,
                            Reg::Const(replace_with) => {
                                self.replace(z, Reg::Const(replace_with.cast(dtype)), begin);
                            }
                        }
                    }
                }
                IROp::Unary { z, ref mut x, uop } => {
                    if *x == to_replace {
                        match replace_with {
                            Reg::Var(replace_with) => *x = replace_with,
                            Reg::Const(replace_with) => {
                                self.replace(z, Reg::Const(replace_with.unary(uop)), begin);
                            }
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
                IROp::SetLocal { .. }
                | IROp::Set { .. }
                | IROp::Loop { .. }
                | IROp::EndLoop { .. }
                | IROp::Barrier { .. } => {}
                _ => {}
            }
        }
    }

    fn debug(&self) {
        let mut first_loops = true;
        let mut indent = String::new();
        println!("IRKernel");
        for op in &self.ops {
            match op {
                IROp::Loop { .. } => {
                    println!("{indent}{op}");
                    if !first_loops {
                        indent += "  ";
                    }
                }
                IROp::EndLoop { .. } => {
                    indent.pop();
                    indent.pop();
                    println!("{indent}{op}");
                }
                _ => {
                    println!("{indent}{op}");
                    first_loops = false;
                }
            }
        }
        println!();
    }

    /*fn upcast(&mut self, axis: u16, local_acc_len: Dimension, addressables: &mut Vec<(Scope, DType, usize, bool)>) {
        let mut loop_id = None;
        let mut last_reg_acc_id = 0;
        for i in 0..self.ops.len() {
            if matches!(self.ops[i], IROp::Set { .. }) {
                last_reg_acc_id = i;
            }
            if let IROp::Loop { id, len: olen } = &mut self.ops[i] {
                if *id == axis {
                    *olen /= local_acc_len;
                    loop_id = Some(i);
                }
            }
        }
        let Some(loop_id) = loop_id else {
            return;
        };

        let local_address = addressables.len() as u16;

        // Put local acc instead of register acc
        let register_acc = self.ops.remove(last_reg_acc_id);
        let IROp::Set { z: acc_reg_id, value } = register_acc else { unreachable!() };
        self.ops.insert(
            last_reg_acc_id,
            IROp::SetLocal { address: local_address, len: local_acc_len, value },
        );

        addressables.push((Scope::Local, value.dtype(), local_acc_len, false));

        let mut num_loops = 0;
        // Find where the loop ends and put an accumulator and a second loop there.
        for i in loop_id..self.ops.len() {
            match self.ops[i] {
                IROp::Loop { id, len } => num_loops += 1,
                IROp::EndLoop { id, len } => {
                    if num_loops == 0 {
                        let i = i - 2;
                        // Last op before the end of the loop should be binary that accumulates into acc_reg_id
                        let IROp::Binary { z, x: temp_reg_id, y, bop: acc_bop } = self.ops[i - 1]
                        else {
                            unreachable!()
                        };
                        let Reg::Var(y) = y else { unreachable!() };
                        assert_eq!(z, y);
                        // Add load from local and store into local
                        self.ops.insert(
                            i - 1,
                            IROp::Load { z, address: local_address, offset: Reg::Var(5) },
                        );
                        self.ops.insert(
                            i + 1,
                            IROp::Store {
                                address: local_address,
                                offset: Reg::Var(5),
                                x: Reg::Var(z),
                            },
                        );

                        // Creates second accumulator loop
                        let Reg::Var(temp_reg_id) = temp_reg_id else { unreachable!() };
                        self.ops.insert(i + 3, register_acc);
                        self.ops.insert(i + 4, IROp::Loop { id: axis, len: local_acc_len });
                        self.ops.insert(
                            i + 5,
                            IROp::Load {
                                z: temp_reg_id,
                                address: local_address,
                                offset: Reg::Var(5),
                            },
                        );
                        self.ops.insert(
                            i + 6,
                            IROp::Binary {
                                z: acc_reg_id,
                                x: Reg::Var(temp_reg_id),
                                y: Reg::Var(acc_reg_id),
                                bop: acc_bop,
                            },
                        );
                        self.ops.insert(i + 7, IROp::EndLoop { id: axis, len: local_acc_len });
                        return;
                    } else {
                        num_loops -= 1
                    }
                }
                _ => {}
            }
        }
    }*/

    fn deduplicate(&mut self) {
        // Get all accs
        let mut accs = Set::with_hasher(Default::default());
        for op in &self.ops {
            if let IROp::Set { z, .. } = op {
                accs.insert(*z);
            }
        }
        // Keep over each op, till you find a duplicate
        let mut i = 0;
        while i < self.ops.len() {
            let mut changed = false;
            match self.ops[i] {
                IROp::Load { z, address, offset } => {
                    for j in i + 1..self.ops.len() {
                        if let IROp::Load { z: z2, address: address2, offset: offset2 } =
                            self.ops[j]
                        {
                            if address == address2 && offset == offset2 {
                                self.replace(z2, Reg::Var(z), j);
                                self.ops.remove(j);
                                changed = true;
                                break;
                            }
                        } else if matches!(self.ops[j], IROp::EndLoop { .. }) {
                            break;
                        }
                    }
                }
                IROp::Cast { z, x, dtype } => {
                    if !accs.contains(&z) {
                        for j in i + 1..self.ops.len() {
                            if let IROp::Cast { z: z2, x: x2, dtype: dtype2 } = self.ops[j] {
                                if x == x2 && dtype == dtype2 && !accs.contains(&z2) {
                                    self.replace(z2, Reg::Var(z), j);
                                    self.ops.remove(j);
                                    changed = true;
                                    break;
                                }
                            } else if matches!(self.ops[j], IROp::EndLoop { .. }) {
                                break;
                            }
                        }
                    }
                }
                IROp::Unary { z, x, uop } => {
                    if !accs.contains(&z) {
                        for j in i + 1..self.ops.len() {
                            if let IROp::Unary { z: z2, x: x2, uop: uop2 } = self.ops[j] {
                                if x == x2 && uop == uop2 && !accs.contains(&z2) {
                                    self.replace(z2, Reg::Var(z), j);
                                    self.ops.remove(j);
                                    changed = true;
                                    break;
                                }
                            } else if matches!(self.ops[j], IROp::EndLoop { .. }) {
                                break;
                            }
                        }
                    }
                }
                IROp::Binary { z, x, y, bop } => {
                    if !accs.contains(&z) {
                        for j in i + 1..self.ops.len() {
                            if let IROp::Binary { z: z2, x: x2, y: y2, bop: bop2 } = self.ops[j] {
                                if x == x2 && y == y2 && bop == bop2 && !accs.contains(&z2) {
                                    if let Reg::Var(x2) = x2 {
                                        if !accs.contains(&x2) {
                                            if let Reg::Var(y2) = y2 {
                                                if !accs.contains(&y2) {
                                                    self.replace(z2, Reg::Var(z), j);
                                                    self.ops.remove(j);
                                                    changed = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if matches!(
                                self.ops[j],
                                IROp::EndLoop { .. } | IROp::Loop { .. }
                            ) {
                                break;
                            }
                        }
                    }
                }
                IROp::MAdd { z, a, b, c } => {
                    if !accs.contains(&z) {
                        for j in i + 1..self.ops.len() {
                            if let IROp::MAdd { z: z2, a: a2, b: b2, c: c2 } = self.ops[j] {
                                if a == a2 && b == b2 && c == c2 && !accs.contains(&z2) {
                                    self.replace(z2, Reg::Var(z), j);
                                    self.ops.remove(j);
                                    changed = true;
                                    break;
                                }
                            } else if matches!(self.ops[j], IROp::EndLoop { .. }) {
                                break;
                            }
                        }
                    }
                }

                IROp::SetLocal { .. } => {}
                IROp::Set { .. } => {}
                IROp::Store { .. } => {}
                IROp::Loop { .. } => {}
                IROp::EndLoop { .. } => {}
                IROp::Barrier { .. } => {}
                _ => {}
            }
            if !changed {
                i += 1;
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
    pub(super) fn new(
        mut kernel: Kernel,
        optimization: &Optimization,
        debug: DebugMask,
    ) -> IRKernel {
        kernel.reshape(&optimization.shape);
        kernel.permute(&[0, 3, 6, 1, 4, 7, 2, 5, 8]);

        debug_assert_eq!(kernel.shape().len(), 9);

        if optimization.opt_ops.contains(&OptOp::MatmulRegisterTiling) {
            let reduce_register_dim = 4;
            for (i, op) in kernel.ops[9..].iter().enumerate() {
                if let Op::Loop { len, .. } = op {
                    kernel.split_loop(9 + i, &[len / reduce_register_dim, reduce_register_dim]);
                    break;
                }
            }
        }

        let mut addressables: Vec<(Scope, DType, usize, bool)> = Vec::new();
        let mut compiler = IRCompiler::vops_to_ssa_ir(&kernel.ops, &mut addressables);

        for opt_op in &optimization.opt_ops {
            match opt_op {
                OptOp::MatmulRegisterTiling => compiler.matmul_register_tiling(),
            }
        }

        compiler.global_loop_unrolling();
        compiler.loop_unrolling();

        let mut old_compiler = compiler.clone();
        loop {
            compiler.constant_folding_and_propagation();
            // TODO automatic reordering of additions such that we minimize dependencies
            // for loop invariant code motion
            compiler.loop_invariant_code_motion();
            //compiler.vectorization();
            compiler.constant_folding_and_propagation();
            compiler.common_subexpression_elimination();
            compiler.deduplicate();
            if compiler == old_compiler {
                break;
            } else {
                old_compiler = compiler.clone();
            }
        }

        compiler.fuse_ops();

        if debug.ir() {
            compiler.debug();
        }

        let (registers, ops) = compiler.reduce_register_use();

        IRKernel { addressables, registers, ops }
    }
}
