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
        z: Reg,
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
    // Assign value to register z
    Set {
        z: u16,
        value: Constant,
    },
    Unary {
        z: Reg,
        x: Reg,
        uop: UOp,
    },
    Binary {
        z: Reg,
        x: Reg,
        y: Reg,
        bop: BOp,
    },
    // z = a * b + c
    MAdd {
        z: Reg,
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
    register_map: BTreeMap<TensorId, u16>,
    constant_map: BTreeMap<u16, Constant>,
    pointers_map: BTreeMap<(TensorId, Scope), u16>,
    max_id: u16,
}

impl IRCompiler {
    pub(super) fn variable(&mut self, constant: Constant) -> u16 {
        self.max_id += 1;
        self.ops.push(IROp::Set {
            z: self.max_id,
            value: constant,
        });
        self.max_id
    }

    pub(super) fn constant(&mut self, constant: Constant) -> u16 {
        self.max_id += 1;
        self.constant_map.insert(self.max_id, constant);
        self.max_id
    }

    fn unary_op(&mut self, x: u16, uop: UOp) -> u16 {
        self.max_id += 1;
        self.ops.push(IROp::Unary {
            z: Reg::Var(self.max_id),
            x: Reg::Var(x),
            uop,
        });
        self.max_id
    }

    pub(super) fn cast(&mut self, x: u16, dtype: DType) -> u16 {
        self.unary_op(x, UOp::Cast(dtype))
    }

    pub(super) fn load(&mut self, address: u16, offset: u16) -> u16 {
        self.max_id += 1;
        self.ops.push(IROp::Load {
            z: Reg::Var(self.max_id),
            address,
            offset: Reg::Var(offset),
        });
        self.max_id
    }

    fn binary_op(&mut self, x: u16, y: u16, bop: BOp) -> u16 {
        self.max_id += 1;
        self.ops.push(IROp::Binary {
            z: Reg::Var(self.max_id),
            x: Reg::Var(x),
            y: Reg::Var(y),
            bop,
        });
        self.max_id
    }

    pub(super) fn and(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::And)
    }

    pub(super) fn cmplt(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::Cmplt)
    }

    pub(super) fn cmpgt(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::Cmpgt)
    }

    pub(super) fn add(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::Add)
    }

    pub(super) fn sub(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::Sub)
    }

    pub(super) fn mul(&mut self, x: u16, y: u16) -> u16 {
        self.binary_op(x, y, BOp::Mul)
    }

    pub(super) fn mad(&mut self, x: u16, y: u16, z: u16) -> u16 {
        let t = self.mul(x, y);
        self.add(t, z)
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
            constant_map: BTreeMap::new(),
            pointers_map: BTreeMap::new(),
            max_id: 0,
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
                &VOp::Loop { axis, .. } => c.max_id = c.max_id.max(axis as u16),
                _ => {}
            }
        }

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
                    let constant = c.constant(value);
                    let zreg = view.ir_for_constant_load(&mut c, constant);
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
                                xview.ir_for_indexed_load(&mut c, zaddress, zdtype)
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
                    let acc_init = match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    };
                    let acc_init_reg = c.constant(acc_init);
                    view.ir_for_indexed_store(&mut c, address, acc_init_reg);
                }
                &VOp::Move { z, x, .. } => {
                    c.register_map.insert(z, c.register_map[&x]);
                }
                &VOp::Unary { z, x, uop, view: _ } => {
                    // TODO unary can have view, but this will probably be only used for vectorization
                    let xreg = c.register_map[&x];
                    if let Some(value) = c.constant_map.get(&xreg) {
                        let zreg = c.constant(value.unary(uop));
                        c.register_map.insert(z, zreg);
                    } else {
                        let xreg = c.register_map[&x];
                        let zreg = c.unary_op(xreg, uop);
                        c.register_map.insert(z, zreg);
                    }
                }
                &VOp::Binary {
                    z,
                    ref zview,
                    x,
                    ref xview,
                    y,
                    ref yview,
                    bop,
                } => {
                    todo!();

                    /*let dtype = graph.dtype(z).ir_dtype();
                    let id = if let Some(&Reg::Var(id)) = register_map.get(&z) {
                        id
                    } else {
                        get_empty_register(&mut registers, dtype, tensor_rcs[&z])
                    };
                    let zvar = Reg::Var(id);
                    register_map.insert(z, zvar);

                    let bin_op = IROp::Binary {
                        z: zvar,
                        x: if let Some(&address) = pointers_map.get(&(x, Scope::Register)) {
                            xview.ir_for_indexed_load(address, dtype, 0, &mut registers, &mut ops)
                        } else {
                            register_map[&x]
                        },
                        y: if let Some(&address) = pointers_map.get(&(y, Scope::Register)) {
                            yview.ir_for_indexed_load(address, dtype, 0, &mut registers, &mut ops)
                        } else {
                            register_map[&y]
                        },
                        bop,
                    };
                    ops.push(bin_op);

                    if let Reg::Var(id) = register_map[&x] {
                        registers[id as usize].1 -= 1u32;
                    }
                    if let Reg::Var(id) = register_map[&y] {
                        registers[id as usize].1 -= 1u32;
                    }

                    if let Some(&address) = pointers_map.get(&(z, Scope::Register)) {
                        zview.ir_for_indexed_store(c, address, zvar);
                    }*/
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
        todo!();

        /*(
            IRKernel {
                addressables,
                registers,
                ops: c.ops,
            },
            args,
        )*/
    }

    pub(super) fn debug(&self) {
        for op in &self.ops {
            println!("{op:?}");
        }
    }
}

/*pub(crate) fn get_empty_register(
    registers: &mut Vec<(IRDType, u32)>,
    ir_dtype: IRDType,
    rc: u32,
) -> u16 {
    if let Some(id) = registers.iter().enumerate().find_map(|(id, &(d, c))| {
        if c == 0 && ir_dtype == d {
            Some(id as u16)
        } else {
            None
        }
    }) {
        registers[id as usize] = (ir_dtype, rc);
        id
    } else {
        registers.push((ir_dtype, rc));
        (registers.len() - 1) as u16
    }
}*/

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
