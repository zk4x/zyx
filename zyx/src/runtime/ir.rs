//! Intermediate representation that is close to assembly.
//! It is passed into different backends. Each backend
//! compiles IR into their own bytecode.

use super::{graph::Graph, node::ROp, scheduler::VOp};
use crate::{
    dtype::Constant,
    runtime::node::{BOp, UOp},
    shape::Axis,
    tensor::TensorId,
    DType,
};
use std::{
    collections::BTreeMap,
    fmt::{Display, Write},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Var {
    // This is id into kernel.registers
    Id(u16),
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
        z: Var,
        // Address is variable in addressables
        address: u16,
        // Offset is u32 var that needs to be added to address
        offset: Var,
    },
    // Stores variable from variable x to address as given offset.
    Store {
        // Address is variable in addressables
        address: u16,
        // Offset is u32 var that needs to be added to address
        offset: Var,
        x: Var,
    },
    // Assign value to register z
    Set {
        z: u16,
        value: Constant,
    },
    Unary {
        z: Var,
        x: Var,
        uop: UOp,
    },
    Binary {
        z: Var,
        x: Var,
        y: Var,
        bop: BOp,
    },
    // z = a * b + c
    MAdd {
        z: Var,
        a: Var,
        b: Var,
        c: Var,
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

// Data structures will stay the same, but we will rewrite the implementation of to_ir function.
// The current version just does not work, particularly with padding. We also want to support vector datatypes,
// and local and register tiling. Neither of which currently works.
// Indexing also needs to be rewritten so that as much of it happens outside of the loops
// and so that it does work properly

// Returns IRKernel and order in which tensors are passed to it
pub(super) fn to_ir(kernel_ops: &[VOp], graph: &Graph) -> (IRKernel, Vec<TensorId>) {
    // What we need to calculate (outputs of this function)
    let mut addressables = Vec::new();
    let mut registers = Vec::new();
    let mut ops = Vec::new();
    let mut args = Vec::new();

    // Get reference counts for all tensors and axes
    let mut tensor_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
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
            VOp::Load { zview, xview, .. } => {
                for axis in xview.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
                for axis in zview.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            VOp::Store {
                z,
                ref zview,
                ref xview,
                ..
            } => {
                for axis in xview.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
                for axis in zview.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
                tensor_rcs.entry(*z).and_modify(|rc| *rc += 1).or_insert(1);
            }
            VOp::Accumulator { view, .. } => {
                for axis in view.used_axes() {
                    axes_rcs.entry(axis).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
            &VOp::Move { x, .. } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
            }
            // TODO should we put view into axes_rcs, or should we remove axes_rcs alltogether?
            &VOp::Unary { x, .. } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
            }
            // TODO should we put view into axes_rcs, or should we remove axes_rcs alltogether?
            &VOp::Binary { x, y, .. } => {
                tensor_rcs.entry(x).and_modify(|rc| *rc += 1).or_insert(1);
                tensor_rcs.entry(y).and_modify(|rc| *rc += 1).or_insert(1);
            }
            VOp::Barrier { .. } => {}
        }
    }
    let tensor_rcs = tensor_rcs;

    // Allocate register space for axes.
    // This way axes also have the same id in registers.
    for (_, &rc) in &axes_rcs {
        registers.push((IRDType::U32(IRVec::Scalar), rc));
    }

    // Maps from tensors to registers
    let mut register_map: BTreeMap<TensorId, Var> = BTreeMap::new();
    // Map from addressables to registers
    let mut addressables_map: BTreeMap<(TensorId, Scope), u16> = BTreeMap::new();

    // Declare global arguments
    for op in kernel_ops {
        match op {
            &VOp::Load {
                x,
                xscope,
                ref xview,
                ..
            } => {
                if xscope == Scope::Global && !addressables_map.contains_key(&(x, xscope)) {
                    args.push(x);
                    let dtype = graph.dtype(x).ir_dtype();
                    addressables.push((xscope, dtype, xview.original_numel(), true));
                    let id = (addressables.len() - 1) as u16;
                    addressables_map.insert((x, xscope), id);
                }
            }
            &VOp::Store {
                z,
                zscope,
                ref zview,
                xscope,
                ..
            } => {
                if zscope == Scope::Global {
                    let dtype = graph.dtype(z).ir_dtype();
                    addressables_map
                        .entry((z, zscope))
                        .and_modify(|&mut id| {
                            assert_eq!(xscope, Scope::Register);
                            // set it to read-write
                            addressables[id as usize].3 = false;
                        })
                        .or_insert_with(|| {
                            args.push(z);
                            addressables.push((zscope, dtype, zview.original_numel(), false));
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
                ..
            } => {
                if xscope == Scope::Local && !addressables_map.contains_key(&(x, xscope)) {
                    let dtype = graph.dtype(x).ir_dtype();
                    addressables.push((xscope, dtype, xview.original_numel(), true));
                    let id = (addressables.len() - 1) as u16;
                    addressables_map.insert((x, xscope), id);
                }
            }
            &VOp::Store {
                zscope,
                ..
            } => {
                if zscope == Scope::Local {
                    todo!()
                }
            }
            _ => {}
        }
    }

    // Declare accumulators
    // TODO if we have multiple accumulators with the same size, we must reuse them
    for op in kernel_ops {
        if let &VOp::Accumulator { z, ref view, .. } = op {
            let dtype = graph.dtype(z).ir_dtype();
            addressables.push((Scope::Register, dtype, view.original_numel(), false));
            let id = (addressables.len() - 1) as u16;
            addressables_map.insert((z, Scope::Register), id);
        }
    }
    // TODO addressables can change, because they can used in multiple places, so deal with that
    // also if they are used in multiples places, then tiling is necessary, but that should
    // be handled by optimizer.

    // Actual transpiling from Kernel to IRKernel
    let mut loops = Vec::new();
    for op in kernel_ops {
        //println!("{op}");
        match op {
            &VOp::Loop { axis, len } => {
                // Axis always maps to register ids
                let id = axis as u16;
                ops.push(IROp::Loop { id, len });
                loops.push((id, len));
            }
            VOp::EndLoop => {
                if let Some((id, len)) = loops.pop() {
                    ops.push(IROp::EndLoop { id, len });
                }
            }
            &VOp::Const { z, value, ref view } => {
                let (nops, var) = view.ir_for_constant_load(value, &mut registers);
                ops.extend(nops);
                register_map.insert(z, var);
            }
            &VOp::Load {
                z,
                zscope,
                ref zview,
                x,
                xscope,
                ref xview,
            } => match (zscope, xscope) {
                (Scope::Local, Scope::Global) => {
                    let dtype = graph.dtype(z).ir_dtype();
                    let address = addressables_map[&(x, xscope)];
                    let id = xview.ir_for_indexed_load(address, dtype, tensor_rcs[&z], &mut registers, &mut ops);
                    register_map.insert(z, id);
                    let address = addressables_map[&(z, zscope)];
                    zview.ir_for_indexed_store(address, id, &mut registers, &mut ops);
                }
                (Scope::Register, Scope::Local) => {
                    let dtype = graph.dtype(z).ir_dtype();
                    let address = addressables_map[&(x, xscope)];
                    let var = xview.ir_for_indexed_load(address, dtype, tensor_rcs[&z], &mut registers, &mut ops);
                    register_map.insert(z, var);
                }
                (Scope::Register, Scope::Global) => {
                    let dtype = graph.dtype(z).ir_dtype();
                    let address = addressables_map[&(x, xscope)];
                    let var = xview.ir_for_indexed_load(address, dtype, tensor_rcs[&z], &mut registers, &mut ops);
                    register_map.insert(z, var);
                }
                _ => panic!("Invalid load scopes"),
            },
            &VOp::Store {
                z,
                zscope,
                ref zview,
                xscope,
                ref xview,
            } => match (zscope, xscope) {
                (Scope::Local, Scope::Register) => {
                    todo!()
                }
                (Scope::Global, Scope::Register) => {
                    let address = addressables_map[&(z, zscope)];
                    let dtype = graph.dtype(z).ir_dtype();
                    let x = if let Some(&address) = addressables_map.get(&(z, Scope::Register)) {
                        xview.ir_for_indexed_load(address, dtype, 0, &mut registers, &mut ops)
                    } else {
                        register_map[&z]
                    };
                    zview.ir_for_indexed_store(address, x, &mut registers, &mut ops)
                }
                _ => panic!("Invalid store scopes"),
            },
            &VOp::Accumulator { z, rop, ref view } => {
                let dtype = graph.dtype(z).ir_dtype();
                let address = addressables_map[&(z, Scope::Register)];
                let var = Var::Const(match rop {
                    ROp::Sum => dtype.dtype().zero_constant(),
                    ROp::Max => dtype.dtype().min_constant(),
                });
                view.ir_for_indexed_store(address, var, &mut registers, &mut ops);
            }
            &VOp::Move { z, x, .. } => {
                register_map.insert(z, register_map[&x]);
            }
            &VOp::Unary { z, x, uop, view: _ } => {
                // TODO unary can have view, but this will probably be only used for vectorization
                if let Var::Const(value) = register_map[&x] {
                    register_map.insert(z, Var::Const(value.unary(uop)));
                } else {
                    let dtype = graph.dtype(z).ir_dtype();
                    let id = get_empty_register(&mut registers, dtype, tensor_rcs[&z]);
                    let zvar = Var::Id(id);
                    register_map.insert(z, zvar);
                    ops.push(IROp::Unary {
                        z: zvar,
                        x: register_map[&x],
                        uop,
                    });
                    if let Var::Id(id) = register_map[&x] {
                        registers[id as usize].1 -= 1u32;
                    }
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
                // TODO binary can have view, for example if it is accumulator
                let dtype = graph.dtype(z).ir_dtype();
                let id = if let Some(&Var::Id(id)) = register_map.get(&z) {
                    id
                } else {
                    get_empty_register(&mut registers, dtype, tensor_rcs[&z])
                };
                let zvar = Var::Id(id);
                register_map.insert(z, zvar);

                let bin_op = IROp::Binary {
                    z: zvar,
                    x: if let Some(&address) = addressables_map.get(&(x, Scope::Register)) {
                        xview.ir_for_indexed_load(address, dtype, 0, &mut registers, &mut ops)
                    } else {
                        register_map[&x]
                    },
                    y: if let Some(&address) = addressables_map.get(&(y, Scope::Register)) {
                        yview.ir_for_indexed_load(address, dtype, 0, &mut registers, &mut ops)
                    } else {
                        register_map[&y]
                    },
                    bop,
                };
                ops.push(bin_op);

                if let Var::Id(id) = register_map[&x] {
                    registers[id as usize].1 -= 1u32;
                }
                if let Var::Id(id) = register_map[&y] {
                    registers[id as usize].1 -= 1u32;
                }

                if let Some(&address) = addressables_map.get(&(z, Scope::Register)) {
                    zview.ir_for_indexed_store(address, zvar, &mut registers, &mut ops);
                }
            }
            &VOp::Barrier { scope } => {
                ops.push(IROp::Barrier { scope });
            }
        }
    }

    while let Some((id, len)) = loops.pop() {
        ops.push(IROp::EndLoop { id, len });
    }

    // TODO Optimize by deduplicating ops (namely indices) and moving them before loops
    // This will require automatic dependency resolution

    //for op in &ops { println!("{op:?}"); }

    (
        IRKernel {
            addressables,
            registers: registers.into_iter().map(|(dtype, _)| dtype).collect(),
            ops,
        },
        args,
    )
}

pub(crate) fn get_empty_register(registers: &mut Vec<(IRDType, u32)>, ir_dtype: IRDType, rc: u32) -> u16 {
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
}

impl IRKernel {
    pub(super) fn debug(&self) {
        for op in &self.ops {
            println!("{op:?}");
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
