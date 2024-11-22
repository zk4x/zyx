//! Intermediate representation that is close to assembly.
//! It is passed into different backends. Each backend
//! compiles IR into their own bytecode.

use super::{node::ROp, scheduler::VOp};
use crate::{
    dtype::Constant,
    runtime::node::{BOp, UOp},
    tensor::TensorId,
    DType,
};
use std::{
    collections::BTreeMap,
    fmt::{Display, Write},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Reg {
    // This is id into kernel.registers
    Var(u16),
    Const(Constant),
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Scope {
    Global,
    Local,
    RegTile,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IROp {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IRDType {
    BF16(IRVec),
    F8(IRVec),
    F16(IRVec),
    F32(IRVec),
    F64(IRVec),
    U8(IRVec),
    U32(IRVec),
    I8(IRVec),
    I16(IRVec),
    I32(IRVec),
    I64(IRVec),
    Bool,
}

// TODO add vectorization
#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IRVec {
    Scalar,
    V2,
    V4,
    V8,
    V16,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct IRKernel {
    // Index of var is it's Id
    // All addressable variables (those that use indexing for access)
    // These can be global args, local variables or variables in registers
    // scope, dtype, byte_size, read_only
    pub(super) addressables: Vec<(Scope, IRDType, usize, bool)>,
    // Registers with single value or vector stored in them
    // dtype
    pub(super) registers: Vec<IRDType>,
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

impl DType {
    pub(super) const fn ir_dtype(self) -> IRDType {
        match self {
            DType::BF16 => IRDType::BF16(IRVec::Scalar),
            DType::F8 => IRDType::F8(IRVec::Scalar),
            DType::F16 => IRDType::F16(IRVec::Scalar),
            DType::F32 => IRDType::F32(IRVec::Scalar),
            DType::F64 => IRDType::F64(IRVec::Scalar),
            DType::U8 => IRDType::U8(IRVec::Scalar),
            DType::U32 => IRDType::U32(IRVec::Scalar),
            DType::I8 => IRDType::I8(IRVec::Scalar),
            DType::I16 => IRDType::I16(IRVec::Scalar),
            DType::I32 => IRDType::I32(IRVec::Scalar),
            DType::I64 => IRDType::I64(IRVec::Scalar),
            DType::Bool => IRDType::Bool,
        }
    }
}

impl IRDType {
    // TODO vectorization
    #[allow(unused)]
    pub(super) const fn byte_size(self) -> usize {
        match self {
            IRDType::Bool => 1,
            IRDType::F8(v) | IRDType::U8(v) | IRDType::I8(v) => v.len(),
            IRDType::BF16(v) | IRDType::F16(v) | IRDType::I16(v) => 2 * v.len(),
            IRDType::F32(v) | IRDType::I32(v) | IRDType::U32(v) => 4 * v.len(),
            IRDType::F64(v) | IRDType::I64(v) => 8 * v.len(),
        }
    }

    pub(super) const fn dtype(self) -> DType {
        match self {
            IRDType::BF16(_) => DType::BF16,
            IRDType::F8(_) => DType::F8,
            IRDType::F16(_) => DType::F16,
            IRDType::F32(_) => DType::F32,
            IRDType::F64(_) => DType::F64,
            IRDType::U8(_) => DType::U8,
            IRDType::U32(_) => DType::U32,
            IRDType::I8(_) => DType::I8,
            IRDType::I16(_) => DType::I16,
            IRDType::I32(_) => DType::I32,
            IRDType::I64(_) => DType::I64,
            IRDType::Bool => DType::Bool,
        }
    }
}

impl From<DType> for IRDType {
    fn from(value: DType) -> Self {
        match value {
            DType::BF16 => IRDType::BF16(IRVec::Scalar),
            DType::F8 => IRDType::F8(IRVec::Scalar),
            DType::F16 => IRDType::F16(IRVec::Scalar),
            DType::F32 => IRDType::F32(IRVec::Scalar),
            DType::F64 => IRDType::F64(IRVec::Scalar),
            DType::U8 => IRDType::U8(IRVec::Scalar),
            DType::U32 => IRDType::U32(IRVec::Scalar),
            DType::I8 => IRDType::I8(IRVec::Scalar),
            DType::I16 => IRDType::I16(IRVec::Scalar),
            DType::I32 => IRDType::I32(IRVec::Scalar),
            DType::I64 => IRDType::I64(IRVec::Scalar),
            DType::Bool => IRDType::Bool,
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

pub(super) struct IRCompiler {
    pub(super) ops: Vec<IROp>,
    register_map: BTreeMap<TensorId, Reg>,
    pointers_map: BTreeMap<(TensorId, Scope), u16>,
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

    fn vops_to_ir(
        kernel_ops: &[VOp],
        args: &mut Vec<u32>,
        addressables: &mut Vec<(Scope, IRDType, usize, bool)>,
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
                VOp::Load {
                    x,
                    xscope,
                    ref xview,
                    xdtype,
                    ..
                } => {
                    if xscope == Scope::Global && !c.pointers_map.contains_key(&(x, xscope)) {
                        args.push(x);
                        let dtype = xdtype.ir_dtype();
                        addressables.push((xscope, dtype, xview.original_numel(), true));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        c.pointers_map.insert((x, xscope), id);
                    }
                }
                VOp::Store {
                    z,
                    zscope,
                    ref zview,
                    zdtype,
                    xscope,
                    ..
                } => {
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
                                addressables.push((
                                    zscope,
                                    zdtype.ir_dtype(),
                                    zview.original_numel(),
                                    false,
                                ));
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
                VOp::Load {
                    x,
                    xscope,
                    ref xview,
                    xdtype,
                    ..
                } => {
                    if xscope == Scope::Local && !c.pointers_map.contains_key(&(x, xscope)) {
                        addressables.push((
                            xscope,
                            xdtype.ir_dtype(),
                            xview.original_numel(),
                            true,
                        ));
                        let id = u16::try_from(addressables.len() - 1).unwrap();
                        c.pointers_map.insert((x, xscope), id);
                    }
                }
                VOp::Store { zscope, .. } => {
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
                VOp::Accumulator {
                    z, ref view, dtype, ..
                } => {
                    addressables.push((
                        Scope::RegTile,
                        dtype.ir_dtype(),
                        view.original_numel(),
                        false,
                    ));
                    let id = u16::try_from(addressables.len() - 1).unwrap();
                    c.pointers_map.insert((z, Scope::RegTile), id);
                }
                VOp::Loop { axis, .. } => max_axis = max_axis.max(u16::try_from(axis).unwrap()),
                _ => {}
            }
        }
        c.dtypes = vec![DType::U32; max_axis as usize + 1];

        // Transpiling from Kernel to IRKernel, VOp -> IROp
        let mut loops = Vec::new();
        for op in kernel_ops {
            //println!("{op}");
            match op {
                &VOp::Loop { axis, len } => {
                    // Axis always maps to register ids
                    let id = u16::try_from(axis).unwrap();
                    c.ops.push(IROp::Loop { id, len });
                    loops.push((id, len));
                }
                VOp::EndLoop => {
                    if let Some((id, len)) = loops.pop() {
                        c.ops.push(IROp::EndLoop { id, len });
                    }
                }
                &VOp::Const { z, value, ref view } => {
                    let zreg = view.ir_for_constant_load(&mut c, value);
                    c.register_map.insert(z, zreg);
                }
                &VOp::Load {
                    z,
                    zscope,
                    ref zview,
                    x,
                    xscope,
                    ref xview,
                    xdtype,
                } => {
                    let xaddress = c.pointers_map[&(x, xscope)];
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
                &VOp::Store {
                    z,
                    zscope,
                    ref zview,
                    zdtype,
                    xscope,
                    ref xview,
                } => match (zscope, xscope) {
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
                        let zreg = if let Some(&zaddress) = c.pointers_map.get(&(z, Scope::RegTile))
                        {
                            xview.ir_for_indexed_load(&mut c, zaddress, zdtype)
                        } else {
                            c.register_map[&z]
                        };
                        zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                    }
                    scopes => panic!("Invalid store scopes {scopes:?}"),
                },
                &VOp::Accumulator {
                    z,
                    rop,
                    ref view,
                    dtype,
                } => {
                    let address = c.pointers_map[&(z, Scope::RegTile)];
                    let acc_init = Reg::Const(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    });
                    view.ir_for_indexed_store(&mut c, address, acc_init);
                }
                &VOp::Move { z, x, .. } => {
                    c.register_map.insert(z, c.register_map[&x]);
                }
                &VOp::Unary { z, x, uop } => {
                    let xreg = c.register_map[&x];
                    let zreg = c.unary_op(xreg, uop);
                    c.register_map.insert(z, zreg);
                }
                &VOp::Binary { z, x, y, bop } => {
                    let xreg = c.register_map[&x];
                    let yreg = c.register_map[&y];
                    let zreg = c.binary_op(xreg, yreg, bop);
                    c.register_map.insert(z, zreg);
                }
                &VOp::Barrier { scope } => {
                    c.ops.push(IROp::Barrier { scope });
                }
            }
        }
        while let Some((id, len)) = loops.pop() {
            c.ops.push(IROp::EndLoop { id, len });
        }
        c
    }

    fn into_deduplicated_ir(self) -> (Vec<IRDType>, Vec<IROp>) {
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
                &IROp::Load {
                    offset: Reg::Var(x),
                    ..
                } => {
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
                        self.dtypes[z as usize].ir_dtype(),
                        ref_counts[&z],
                    );
                    let offset = if let Reg::Var(offset) = offset {
                        let offset = cmp[&offset];
                        reg_rcs[offset as usize] -= 1;
                        Reg::Var(offset)
                    } else {
                        offset
                    };
                    ops.push(IROp::Load {
                        z: zr,
                        address,
                        offset,
                    });
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
                    ops.push(IROp::Store {
                        address,
                        offset,
                        x: xr,
                    });
                }
                IROp::Unary { z, x, uop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            self.dtypes[z as usize].ir_dtype(),
                            zrc,
                        );
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
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            self.dtypes[z as usize].ir_dtype(),
                            zrc,
                        );
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
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            self.dtypes[z as usize].ir_dtype(),
                            zrc,
                        );
                        ops.push(IROp::MAdd { z: zr, a, b, c });
                        cmp.insert(z, zr);
                    }
                }
                IROp::Loop { id, len } => {
                    if let Some(&zrc) = ref_counts.get(&id) {
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            self.dtypes[id as usize].ir_dtype(),
                            zrc,
                        );
                        ops.push(IROp::Loop { id: zr, len });
                        cmp.insert(id, zr);
                    }
                }
                IROp::EndLoop { id, len } => {
                    reg_rcs[id as usize] -= 1;
                    ops.push(IROp::EndLoop { id, len });
                }
                IROp::Barrier { scope } => ops.push(IROp::Barrier { scope }),
            }
        }
        (registers, ops)
    }

    // i.e. peephole optimization
    fn fuse_ops(&mut self) {
        for i in 0..self.ops.len() - 1 {
            if let IROp::Binary {
                bop,
                z: z0,
                x: a,
                y: b,
                ..
            } = self.ops[i]
            {
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
                    }
                }
            }
        }
    }

    fn unroll_loops(&mut self) {
        // TODO
        // simply duplicate code in required loops replacing every axis variable with constant
    }

    fn common_subexpression_elimination(&mut self) {
        // TODO
    }

    // Loop invariant code motion and dependence analysis
    fn loop_invariant_code_motion(&mut self) {
        // TODO Optimize by deduplicating ops (namely indices) and moving them before loops
        // loop invariant code motion
        // This will require automatic dependency resolution
    }

    // Replace all occurences of z with register x
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
                IROp::Binary {
                    ref mut x,
                    ref mut y,
                    ..
                } => {
                    if *x == Reg::Var(to_replace) {
                        *x = replace_with;
                    }
                    if *y == Reg::Var(to_replace) {
                        *y = replace_with;
                    }
                }
                IROp::MAdd { .. } => todo!(),
                IROp::Load { ref mut offset, .. } => {
                    if *offset == Reg::Var(to_replace) {
                        *offset = replace_with;
                    }
                }
                IROp::Store {
                    ref mut offset,
                    ref mut x,
                    ..
                } => {
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

    fn constant_propagation(&mut self) {
        for i in 0..self.ops.len() {
            match self.ops[i] {
                IROp::Unary { .. } => {}
                IROp::Binary { z, x, y, bop } => match (x, y) {
                    (Reg::Var(_), Reg::Var(_)) => {}
                    (Reg::Var(xv), Reg::Const(yv)) => {
                        if yv.is_zero() {
                            match bop {
                                BOp::Mul => self.replace(z, Reg::Const(yv)),
                                BOp::Add => self.replace(z, Reg::Var(xv)),
                                BOp::Div => panic!("Division by zero constant"),
                                _ => {}
                            }
                        }
                        if yv.is_one() {
                            match bop {
                                BOp::Mul => self.replace(z, Reg::Var(xv)),
                                _ => {}
                            }
                        }
                    }
                    (Reg::Const(xv), Reg::Var(yv)) => {
                        if xv.is_zero() {
                            match bop {
                                BOp::Add => self.replace(z, Reg::Var(yv)),
                                BOp::Mul => self.replace(z, Reg::Const(xv)),
                                BOp::Div => panic!("Division by zero constant"),
                                _ => {}
                            }
                        }
                        if xv.is_one() {
                            match bop {
                                BOp::Mul => self.replace(z, Reg::Var(yv)),
                                _ => {}
                            }
                        }
                    }
                    (Reg::Const(x), Reg::Const(y)) => {
                        self.replace(z, Reg::Const(Constant::binary(x, y, bop)));
                    }
                },
                IROp::MAdd { .. } => todo!(),
                IROp::Loop { .. }
                | IROp::EndLoop { .. }
                | IROp::Load { .. }
                | IROp::Store { .. }
                | IROp::Barrier { .. } => {}
            }
        }
    }
}

fn new_var(
    registers: &mut Vec<IRDType>,
    reg_rcs: &mut Vec<u32>,
    ir_dtype: IRDType,
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
    pub(super) fn debug(&self) {
        for op in &self.ops {
            println!("{op:?}");
        }
    }

    pub(super) fn new(kernel_ops: &[VOp]) -> (IRKernel, Vec<TensorId>) {
        // What we need to calculate (outputs of this function)
        // IRKernel
        let mut addressables: Vec<(Scope, IRDType, usize, bool)> = Vec::new();
        // Returned tensors
        let mut args = Vec::new();

        let mut compiler = IRCompiler::vops_to_ir(kernel_ops, &mut args, &mut addressables);

        // Optimizations
        compiler.unroll_loops();
        compiler.loop_invariant_code_motion();

        compiler.constant_propagation();
        compiler.common_subexpression_elimination();

        compiler.fuse_ops();
        // Optimize constants again?
        //compiler.constant_propagation();

        // TODO perhaps we can do even more optimizations with instruction scheduling
        // and register allocation? But that's a big perhaps...
        // TODO loop splitting and loop peeling

        //for op in &compiler.ops { println!("{op:?}"); }
        let (registers, ops) = compiler.into_deduplicated_ir();
        //println!();
        //println!();
        //for op in &ops { println!("{op:?}"); }
        //panic!();

        (
            IRKernel {
                addressables,
                registers,
                ops,
            },
            args,
        )
    }
}
