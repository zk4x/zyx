use crate::{
    dtype::Constant, runtime::node::{BOp, ROp, UOp}, shape::Axis, tensor::TensorId, DType
};
use std::collections::{BTreeMap, BTreeSet};

use super::scheduler::{Kernel, VOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Var {
    Id(u16, Scope),
    Const(Constant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, bitcode::Encode, bitcode::Decode)]
pub(super) enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IROp {
    Set {
        z: u16,
        len: usize,
        value: Constant,
    },
    Load {
        z: Var,
        x: Var,
        at: Var,
        dtype: IRDType,
    },
    Store {
        z: Var,
        x: Var,
        at: Var,
        dtype: IRDType,
    },
    Unary {
        z: Var,
        x: Var,
        uop: UOp,
        // For cast this is dtype before cast
        dtype: IRDType,
    },
    Binary {
        z: Var,
        x: Var,
        y: Var,
        bop: BOp,
        dtype: IRDType,
    },
    // z = a * b + c
    MAdd {
        z: Var,
        a: Var,
        b: Var,
        c: Var,
        dtype: IRDType,
    },
    // while id < len
    While {
        id: u16,
        len: usize,
    },
    EndLoop {
        id: u16,
        len: usize,
    },
    Barrier {
        scope: Scope,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IRDType {
    #[cfg(feature = "half")]
    BF16,
    #[cfg(feature = "half")]
    F16,
    F32,
    F64,
    #[cfg(feature = "complex")]
    CF32,
    #[cfg(feature = "complex")]
    CF64,
    U8,
    I8,
    I16,
    I32,
    I64,
    Bool,
    // For indexing, usually u32
    U32,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct IRKernel {
    // Index of var is it's Id
    // len, dtype, read_only
    pub(super) addressables: Vec<(usize, IRDType, bool)>,
    // len, dtype, read_only
    pub(super) registers: Vec<(usize, IRDType, bool)>,
    pub(super) ops: Vec<IROp>,
}

impl IRDType {
    pub(super) fn byte_size(&self) -> usize {
        match self {
            #[cfg(feature = "half")]
            IRDType::F16 => 2,
            #[cfg(feature = "half")]
            IRDType::BF16 => 2,
            IRDType::F32 => 4,
            IRDType::F64 => 8,
            #[cfg(feature = "complex")]
            IRDType::CF32 => 8,
            #[cfg(feature = "complex")]
            IRDType::CF64 => 16,
            IRDType::U8 => 1,
            IRDType::I8 => 1,
            IRDType::I16 => 2,
            IRDType::I32 => 4,
            IRDType::I64 => 8,
            IRDType::Bool => 1,
            IRDType::U32 => 4,
        }
    }

    pub(super) fn dtype(&self) -> DType {
        match self {
            #[cfg(feature = "half")]
            IRDType::BF16 => DType::BF16,
            #[cfg(feature = "half")]
            IRDType::F16 => DType::F16,
            IRDType::F32 => DType::F32,
            IRDType::F64 => DType::F64,
            #[cfg(feature = "complex")]
            IRDType::CF32 => DType::CF32,
            #[cfg(feature = "complex")]
            IRDType::CF64 => DType::CF64,
            IRDType::U8 => DType::U8,
            IRDType::I8 => DType::I8,
            IRDType::I16 => DType::I16,
            IRDType::I32 => DType::I32,
            IRDType::I64 => DType::I64,
            IRDType::Bool => DType::Bool,
            IRDType::U32 => panic!(),
        }
    }
}

impl From<DType> for IRDType {
    fn from(value: DType) -> Self {
        match value {
            #[cfg(feature = "half")]
            DType::BF16 => IRDType::BF16,
            #[cfg(feature = "half")]
            DType::F16 => IRDType::F16,
            DType::F32 => IRDType::F32,
            DType::F64 => IRDType::F64,
            #[cfg(feature = "complex")]
            DType::CF32 => IRDType::CF32,
            #[cfg(feature = "complex")]
            DType::CF64 => IRDType::CF64,
            DType::U8 => IRDType::U8,
            DType::I8 => IRDType::I8,
            DType::I16 => IRDType::I16,
            DType::I32 => IRDType::I32,
            DType::I64 => IRDType::I64,
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

// Data structures will stay the same, but we will rewrite the implementation of to_ir function.
// The current version just does not work, particularly with padding. We also want to support vector datatypes,
// and local and register tiling. Neither of which currently works.
// Indexing also needs to be rewritten so that as much of it happens outside of the loops
// and so that it does work properly

// Returns IRKernel and order in which tensors are passed to it
pub(super) fn to_ir(kernel_ops: &[VOp]) -> (IRKernel, Vec<TensorId>) {
    // What we need to calculate (outputs of this function)
    let mut addressables = Vec::new();
    let mut registers = Vec::new();
    let mut ops = Vec::new();
    let mut args = Vec::new();

    // Get reference counts for all tensors and axes
    let mut tensor_rcs: BTreeMap<TensorId, u32>  = BTreeMap::new();
    let mut axes_rcs: BTreeMap<Axis, u32> = BTreeMap::new();
    for op in kernel_ops {
        match op {
            &VOp::Loop { axis, .. } => {
                axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
            }
            VOp::EndLoop => {}
            &VOp::Const { ref view, .. } => {
                // Constants are always valid, they do not need ref count
                for axis in view.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            &VOp::Load { z, zscope, x, xscope, ref view } => {
                for axis in view.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            VOp::Store { z, zscope, xscope, view } => {
                for axis in view.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            VOp::Accumulator { z, rop, view } => {
                for axis in view.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            &VOp::Move { z, x, .. } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
            }
            &VOp::Unary { z, x, uop, ref view } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
            }
            &VOp::Binary { z, x, y, bop, ref zview, ref xview, ref yview } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                tensor_rcs.entry(y).and_modify(|rc| *rc += 1).or_insert(1);
            }
        }
    }
    let tensor_rcs = tensor_rcs;

    // Allocate register space for axes.
    // This way axes also have the same id in registers.
    for (&axis, &rc) in &axes_rcs {
        registers.push((1, IRDType::U32, false, rc));
    }

    // Maps from tensors to registers
    let mut tensor_var_map: BTreeMap<TensorId, Var> = BTreeMap::new();

    fn fill_empty_register(registers: &mut Vec<(usize, IRDType, bool, u32)>, len: usize, ir_dtype: IRDType, read_only: bool, rc: u32) -> u16 {
        if let Some(id) = registers.iter().enumerate().find_map(|(id, &(l, d, r, c))| if c == 0 && l == len && ir_dtype == d && read_only == r { Some(id as u16) } else { None }) {
            registers[id as usize] = (len, ir_dtype, read_only, rc);
            id
        } else {
            registers.push((len, ir_dtype, read_only, rc));
            (registers.len() - 1) as u16
        }
    }

    // Actual transpiling from Kernel to IRKernel
    let mut last_loop = (0, 0);
    for op in kernel_ops {
        println!("{op}");
        match op {
            &VOp::Loop { axis, len } => {
                // Axis always maps to register ids
                let id = axis as u16;
                ops.push(IROp::While { id, len });
                last_loop = (id, len);
            }
            VOp::EndLoop => {
                ops.push(IROp::EndLoop { id: last_loop.0, len: last_loop.1 });
            }
            &VOp::Const { z, value, ref view } => {
                tensor_var_map.insert(z, Var::Const(value));
            }
            &VOp::Load { z, zscope, x, xscope, ref view } => {
                match zscope {
                    Scope::Global => todo!(),
                    Scope::Local => {
                        todo!()
                    }
                    Scope::Register => {
                        //registers.push((len, IRDType::F32, false));
                        // TODO
                        //tensor_var_map.insert();
                    }
                }
            }
            VOp::Store { z, zscope, xscope, view } => {
                todo!()
            }
            &VOp::Accumulator { z, rop, ref view } => {
                let ir_dtype = IRDType::F32;
                let len = view.original_numel();
                let id = fill_empty_register(&mut registers, len, ir_dtype, false, tensor_rcs[&z]);
                tensor_var_map.insert(z, Var::Id(id, Scope::Register));
            }
            VOp::Move { z, x, .. } => {
                tensor_var_map.insert(*z, tensor_var_map[x]);
                todo!()
            }
            &VOp::Unary { z, x, uop, ref view } => {
                // TODO unary can have view
                if let Var::Const(value) = tensor_var_map[&x] {
                    tensor_var_map.insert(z, Var::Const(value.unary(uop)));
                } else {
                    let ir_dtype = IRDType::F32;
                    let id = fill_empty_register(&mut registers, 1, ir_dtype, false, tensor_rcs[&z]);
                    let zvar = Var::Id(id, Scope::Register);
                    tensor_var_map.insert(z, zvar);
                    ops.push(IROp::Unary { z: zvar, x: tensor_var_map[&x], uop, dtype: ir_dtype });
                    if let Var::Id(id, _) = tensor_var_map[&x] {
                        registers[id as usize].3 -= 1u32;
                    }
                }
            }
            &VOp::Binary { z, ref zview, x, ref xview, y, ref yview, bop } => {
                // TODO binary can have view, for example if it is accumulator
                let dtype = IRDType::F32;
                let id = fill_empty_register(&mut registers, 1, dtype, false, tensor_rcs[&z]);
                let zvar = Var::Id(id, Scope::Register);
                tensor_var_map.insert(z, zvar);
                ops.push(IROp::Binary { z: zvar, x: tensor_var_map[&x], y: tensor_var_map[&y], bop, dtype });
                if let Var::Id(id, _) = tensor_var_map[&x] {
                    registers[id as usize].3 -= 1u32;
                }
                if let Var::Id(id, _) = tensor_var_map[&y] {
                    registers[id as usize].3 -= 1u32;
                }
            }
        }
    }

    (IRKernel { addressables, registers: todo!(), ops }, args)
}
