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
    #[allow(unused)]
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
    #[cfg(feature = "complex")]
    CF32(IRVec),
    #[cfg(feature = "complex")]
    CF64(IRVec),
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
    pub(super) fn len(&self) -> usize {
        match self {
            IRVec::Scalar => 1,
            IRVec::V2 => 2,
            IRVec::V4 => 4,
            IRVec::V8 => 8,
            IRVec::V16 => 16,
        }
    }
}

impl IRDType {
    // TODO vectorization
    #[allow(unused)]
    pub(super) fn byte_size(&self) -> usize {
        match self {
            IRDType::BF16(v) => 2 * v.len(),
            IRDType::F8(v) => v.len(),
            IRDType::F16(v) => 2 * v.len(),
            IRDType::F32(v) => 4 * v.len(),
            IRDType::F64(v) => 8 * v.len(),
            #[cfg(feature = "complex")]
            IRDType::CF32(v) => 8 * v.len(),
            #[cfg(feature = "complex")]
            IRDType::CF64(v) => 16 * v.len(),
            IRDType::U8(v) => 1 * v.len(),
            IRDType::I8(v) => 1 * v.len(),
            IRDType::I16(v) => 2 * v.len(),
            IRDType::I32(v) => 4 * v.len(),
            IRDType::I64(v) => 8 * v.len(),
            IRDType::Bool => 1,
            IRDType::U32(v) => 4 * v.len(),
        }
    }

    pub(super) fn dtype(&self) -> DType {
        match self {
            IRDType::BF16(_) => DType::BF16,
            IRDType::F8(_) => DType::F8,
            IRDType::F16(_) => DType::F16,
            IRDType::F32(_) => DType::F32,
            IRDType::F64(_) => DType::F64,
            #[cfg(feature = "complex")]
            IRDType::CF32(_) => DType::CF32,
            #[cfg(feature = "complex")]
            IRDType::CF64(_) => DType::CF64,
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
            #[cfg(feature = "complex")]
            DType::CF32 => IRDType::CF32(IRVec::Scalar),
            #[cfg(feature = "complex")]
            DType::CF64 => IRDType::CF64(IRVec::Scalar),
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
        let z = (self.dtypes.len() - 1) as u16;
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
                let max_id = (self.dtypes.len() - 1) as u16;
                self.ops.push(IROp::Unary { z: max_id, x, uop });
                Reg::Var(max_id)
            }
            Reg::Const(_) => {
                todo!("Unary op on constant")
            }
        }
    }

    fn binary_op(&mut self, x: Reg, y: Reg, bop: BOp) -> u16 {
        // TODO Constant evaluation
        match x {
            Reg::Var(x) => {
                self.dtypes.push(self.dtypes[x as usize]);
            }
            Reg::Const(c) => {
                self.dtypes.push(c.dtype());
            }
        }
        let z = (self.dtypes.len() - 1) as u16;
        self.ops.push(IROp::Binary { z, x, y, bop });
        z
    }

    pub(super) fn cast(&mut self, x: Reg, dtype: DType) -> Reg {
        self.unary_op(x, UOp::Cast(dtype))
    }

    pub(super) fn and(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::And)
    }

    pub(super) fn cmplt(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Cmplt)
    }

    pub(super) fn cmpgt(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Cmpgt)
    }

    pub(super) fn add(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Add)
    }

    pub(super) fn sub(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Sub)
    }

    pub(super) fn mul(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Mul)
    }

    pub(super) fn div(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Div)
    }

    pub(super) fn mod_(&mut self, x: Reg, y: Reg) -> u16 {
        self.binary_op(x, y, BOp::Mod)
    }

    pub(super) fn mad(&mut self, x: Reg, y: Reg, z: Reg) -> u16 {
        let t = self.mul(x, y);
        self.add(Reg::Var(t), z)
    }
}

// Returns IRKernel and order in which tensors are passed to it as arguments
// Axes have the same id in registers.
impl IRKernel {
    pub(super) fn new(kernel_ops: &[VOp]) -> (IRKernel, Vec<TensorId>) {
        // What we need to calculate (outputs of this function)
        // IRKernel
        let mut addressables: Vec<(Scope, IRDType, usize, bool)> = Vec::new();
        // Returned tensors
        let mut args = Vec::new();

        let mut c = IRCompiler {
            ops: Vec::new(),
            register_map: BTreeMap::new(),
            pointers_map: BTreeMap::new(),
            dtypes: Vec::new(),
        };

        // Declare global arguments
        for op in kernel_ops {
            match op {
                &VOp::Load {
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
                        let id = (addressables.len() - 1) as u16;
                        c.pointers_map.insert((x, xscope), id);
                    }
                }
                &VOp::Store {
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
                                (addressables.len() - 1) as u16
                            });
                    }
                }
                _ => {}
            }
        }

        // Declare local variables
        for op in kernel_ops {
            match op {
                &VOp::Load {
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
                        let id = (addressables.len() - 1) as u16;
                        c.pointers_map.insert((x, xscope), id);
                    }
                }
                &VOp::Store { zscope, .. } => {
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
            match op {
                &VOp::Accumulator {
                    z, ref view, dtype, ..
                } => {
                    addressables.push((
                        Scope::Register,
                        dtype.ir_dtype(),
                        view.original_numel(),
                        false,
                    ));
                    let id = (addressables.len() - 1) as u16;
                    c.pointers_map.insert((z, Scope::Register), id);
                }
                &VOp::Loop { axis, .. } => max_axis = max_axis.max(axis as u16),
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
                    let id = axis as u16;
                    c.ops.push(IROp::Loop { id, len });
                    loops.push((id, len));
                }
                VOp::EndLoop => {
                    if let Some((id, len)) = loops.pop() {
                        c.ops.push(IROp::EndLoop { id, len });
                    }
                }
                &VOp::Const { z, value, ref view } => {
                    let zreg = view.ir_for_constant_load(&mut c, Reg::Const(value));
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
                    let zreg = Reg::Var(xview.ir_for_indexed_load(&mut c, xaddress, xdtype));
                    c.register_map.insert(z, zreg);
                    match (zscope, xscope) {
                        (Scope::Local, Scope::Global) => {
                            let zaddress = c.pointers_map[&(z, zscope)];
                            zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                        }
                        (Scope::Register, Scope::Local) => {}
                        (Scope::Register, Scope::Global) => {}
                        _ => panic!("Invalid load scopes. Internal bug."),
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
                    (Scope::Global, Scope::Register) => {
                        let zaddress = c.pointers_map[&(z, zscope)];
                        let zreg =
                            if let Some(&zaddress) = c.pointers_map.get(&(z, Scope::Register)) {
                                Reg::Var(xview.ir_for_indexed_load(&mut c, zaddress, zdtype))
                            } else {
                                c.register_map[&z]
                            };
                        zview.ir_for_indexed_store(&mut c, zaddress, zreg);
                    }
                    _ => panic!("Invalid store scopes"),
                },
                &VOp::Accumulator {
                    z,
                    rop,
                    ref view,
                    dtype,
                } => {
                    let address = c.pointers_map[&(z, Scope::Register)];
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
                    c.register_map.insert(z, Reg::Var(zreg));
                }
                &VOp::Barrier { scope } => {
                    c.ops.push(IROp::Barrier { scope });
                }
            }
        }

        while let Some((id, len)) = loops.pop() {
            c.ops.push(IROp::EndLoop { id, len });
        }

        // TODO Optimize by deduplicating ops (namely indices) and moving them before loops
        // This will require automatic dependency resolution

        //for op in &ops { println!("{op:?}"); }
        let mut ref_counts: BTreeMap<u16, u32> = BTreeMap::new();
        // Get reference counts
        for op in &c.ops {
            match op {
                IROp::Store { x, offset, .. } => {
                    if let &Reg::Var(x) = x {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                    if let &Reg::Var(x) = offset {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
                IROp::Load { offset, .. } => {
                    if let &Reg::Var(x) = offset {
                        ref_counts.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                    }
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
        for op in c.ops {
            match op {
                IROp::Load { z, address, offset } => {
                    let zr = new_var(
                        &mut registers,
                        &mut reg_rcs,
                        c.dtypes[z as usize].ir_dtype(),
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
                    let Reg::Var(x) = x else { panic!() };
                    let xr = cmp[&x];
                    reg_rcs[xr as usize] -= 1;
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
                        x: Reg::Var(xr),
                    });
                }
                IROp::Unary { z, x, uop } => {
                    if let Some(&zrc) = ref_counts.get(&z) {
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            c.dtypes[z as usize].ir_dtype(),
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
                            c.dtypes[z as usize].ir_dtype(),
                            zrc,
                        );
                        ops.push(IROp::Binary { z: zr, x, y, bop });
                        cmp.insert(z, zr);
                    }
                }
                IROp::MAdd { z, a, b, c: co } => {
                    let Reg::Var(a) = a else { panic!() };
                    let Reg::Var(b) = b else { panic!() };
                    let Reg::Var(co) = co else { panic!() };
                    let zr = new_var(
                        &mut registers,
                        &mut reg_rcs,
                        c.dtypes[z as usize].ir_dtype(),
                        ref_counts[&z],
                    );
                    let ar = cmp[&a];
                    let br = cmp[&b];
                    let cr = cmp[&co];
                    reg_rcs[ar as usize] -= 1;
                    reg_rcs[br as usize] -= 1;
                    reg_rcs[cr as usize] -= 1;
                    ops.push(IROp::MAdd {
                        z: zr,
                        a: Reg::Var(ar),
                        b: Reg::Var(br),
                        c: Reg::Var(cr),
                    });
                    cmp.insert(z, zr);
                }
                IROp::Loop { id, len } => {
                    if let Some(&zrc) = ref_counts.get(&id) {
                        let zr = new_var(
                            &mut registers,
                            &mut reg_rcs,
                            c.dtypes[id as usize].ir_dtype(),
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

        //for op in &ops { println!("{op:?}"); }

        (
            IRKernel {
                addressables,
                registers,
                ops,
            },
            args,
        )
    }

    pub(super) fn debug(&self) {
        for op in &self.ops {
            println!("{op:?}");
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
            return i as u16;
        }
    }
    registers.push(ir_dtype);
    reg_rcs.push(ref_count);
    (registers.len() - 1) as u16
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
    pub(super) fn ir_dtype(&self) -> IRDType {
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
