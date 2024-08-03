use core::fmt::Display;
use std::string::ToString;

use crate::{dtype::Constant, runtime::{graph::Graph, node::{BOp, ROp, UOp}, view::{Axis, StridedDim, View}}, tensor::TensorId, DType};
use super::{v::VOp, HWInfo, Scope};
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Var {
    Id(u8),
    Const(Constant),
}

#[derive(Debug, Clone, Copy)]
pub(super) enum IROp {
    Set { z: u8, len: usize, value: Constant },
    Load { z: Var, x: Var, at: Var, dtype: IRDType },
    Store { z: Var, x: Var, at: Var, dtype: IRDType },
    Unary { z: Var, x: Var, uop: UOp, dtype: IRDType },
    Binary { z: Var, x: Var, y: Var, bop: BOp, dtype: IRDType },
    // z = a * b + c
    MAdd { z: Var, a: Var, b: Var, c: Var, dtype: IRDType },
    Loop { id: u8, len: usize },
    EndLoop,
    Barrier { scope: Scope },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    // For indexing
    Idx,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IRVar {
    dtype: IRDType,
    scope: Scope,
    read_only: bool,
    len: usize,
    // TODO perhaps add offset when we do custom memory allocators offset: usize,
}

#[derive(Debug)]
pub(super) struct IRKernel {
    // Index of var is it's Id
    pub(super) vars: Vec<IRVar>,
    pub(super) ops: Vec<IROp>,
}

pub(super) fn vops_to_ir(
    vops: &[VOp],
    graph: &Graph,
    hwinfo: &HWInfo,
) -> IRKernel {
    let _ = hwinfo;
    let mut ops = Vec::new();
    let mut vars: VarMap = VarMap::new();
    let mut args = Vec::new();

    let mut max_axis = 0;

    // TODO get rid of noops

    // Get all global args and set them first
    for vop in vops {
        match vop {
            VOp::Load { x, view, .. } => {
                let _ = vars.add_var(*x, view.numel(), Scope::Global, graph.rc(*x), graph.dtype(*x).into());
            }
            VOp::Store { z, view } => {
                let _ = vars.add_var(*z, view.numel(), Scope::Global, graph.rc(*z), graph.dtype(*z).into());
            }
            _ => {}
        }
    }

    for vop in vops {
        match vop {
            VOp::Const { z, value } => {
                vars.add_const(*z, *value);
            },
            VOp::Load { z, x, view } => {
                let dtype = graph.dtype(*z).into();
                let idx = vars.generate_idx(view, &mut ops);
                let x = vars.get(*x, Scope::Global);
                let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into());
                ops.push(IROp::Load { z, x, at: idx, dtype });
                vars.remove_var(idx);
            },
            VOp::Store { z, view } => {
                let dtype = graph.dtype(*z).into();
                let idx = vars.generate_idx(view, &mut ops);
                let x = vars.get(*z, Scope::Register);
                let z = vars.get(*z, Scope::Global);
                ops.push(IROp::Store { z, x, at: idx, dtype });
                vars.remove_var(idx);
            },
            VOp::Loop { axis, dimension } => {
                let id = vars.add_axis(*axis);
                ops.push(IROp::Loop { id, len: *dimension });
                max_axis += 1;
            },
            VOp::Accumulator { z, rop, view } => {
                let dtype: DType = graph.dtype(*z).into();
                let len = view.numel();
                let Var::Id(z) = vars.add_var(*z, len, Scope::Register, graph.rc(*z), graph.dtype(*z).into()) else {panic!()};
                ops.push(IROp::Set { z, len, value: match *rop {
                    ROp::Sum => dtype.zero_constant(),
                    ROp::Max => dtype.min_constant(),
                }});
            },
            VOp::Reduce { z, x, num_axes, rop } => {
                let x_tensor = *x;
                let z_tensor = *z;
                let dtype = graph.dtype(*z).into();
                let x = vars.get(*x, Scope::Register);
                let z = vars.get(*z, Scope::Register);
                let bop = match *rop {
                    ROp::Sum => BOp::Add,
                    ROp::Max => BOp::Max,
                };
                ops.push(IROp::Binary { z, x, y: z, bop, dtype });
                vars.remove(x_tensor);
                vars.remove(z_tensor);
                for _ in 0..*num_axes {
                    ops.push(IROp::EndLoop);
                    vars.remove_axis(max_axis);
                    max_axis -= 1;
                }
            },
            VOp::Unary { z, x, uop } => {
                let x_tensor = *x;
                let dtype = graph.dtype(*z).into();
                let x = vars.get(*x, Scope::Register);
                let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into());
                ops.push(IROp::Unary { z, x, uop: *uop, dtype });
                vars.remove(x_tensor);
            },
            VOp::Binary { z, x, y, bop } => {
                let x_tensor = *x;
                let y_tensor = *y;
                let dtype = graph.dtype(*z).into();
                let x = vars.get(*x, Scope::Register);
                let y = vars.get(*y, Scope::Register);
                let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into());
                ops.push(IROp::Binary { z, x, y, bop: *bop, dtype });
                vars.remove(x_tensor);
                vars.remove(y_tensor);
            },
        }
    }

    IRKernel { vars: args, ops }
}

struct VarMap {
    // Ref count, length, scope
    vars: Vec<(u32, usize, Scope, IRDType)>,
    var_map: BTreeMap<(TensorId, Scope), Var>,
    axis_map: BTreeMap<Axis, Var>,
}

impl VarMap {
    fn new() -> VarMap {
        VarMap {
            vars: Vec::new(),
            var_map: BTreeMap::new(),
            axis_map: BTreeMap::new(),
        }
    }

    fn get(&self, tensor_id: TensorId, scope: Scope) -> Var {
        self.var_map[&(tensor_id, scope)]
    }

    /// Decrease ref count if it isn't constant
    fn remove(&mut self, tensor_id: TensorId) {
        if let Some(Var::Id(id, ..)) = self.var_map.get(&(tensor_id, Scope::Register)) {
            self.vars[*id as usize].0 -= 1;
        }
    }

    fn add_const(&mut self, x: TensorId, value: Constant) -> Var {
        let var = Var::Const(value);
        self.var_map.insert((x, Scope::Register), var);
        return var;
    }

    fn add_var(&mut self, x: TensorId, len: usize, scope: Scope, rc: u32, dtype: IRDType) -> Var {
        /*if let Some(var) = self.var_map.get(&x) {
            return *var;
        }*/
        let id = self.get_empty_id(dtype);
        self.vars[id] = (rc, len, scope, dtype);
        let var = Var::Id(id as u8);
        self.var_map.insert((x, scope), var);
        return var;
    }

    fn generate_idx(&mut self, view: &View, ops: &mut Vec<IROp>) -> Var {
        match view {
            View::None => Var::Const(Constant::I64(0)),
            View::Strided(dims) => {
                let z = self.add_index();
                ops.push(IROp::Set { z, len: 0, value: Constant::I64(0) });
                let z = Var::Id(z);
                for StridedDim { axis, stride, .. } in dims {
                    if *stride != 0 {
                        let a = self.get_axis(*axis);
                        ops.push(IROp::MAdd { z, a, b: Var::Const(Constant::I64(*stride as i64)), c: z, dtype: IRDType::Idx });
                    }
                }
                z
            },
            View::Padded(_, _) => todo!(),
        }
    }

    fn add_axis(&mut self, axis: Axis) -> u8 {
        let id = self.get_empty_id(IRDType::Idx);
        self.vars[id] = (1, 0, Scope::Register, IRDType::Idx);
        let id = id as u8;
        let var = Var::Id(id);
        self.axis_map.insert(axis, var);
        return id;
    }

    fn get_axis(&mut self, axis: Axis) -> Var {
        self.axis_map[&axis]
    }

    fn remove_axis(&mut self, axis: Axis) {
        if let Some(Var::Id(id, ..)) = self.axis_map.get(&axis) {
            self.vars[*id as usize].0 -= 1;
        }
    }

    fn remove_var(&mut self, var: Var) {
        if let Var::Id(id, ..) = var {
            self.vars[id as usize].0 -= 1;
        }
    }

    fn add_index(&mut self) -> u8 {
        let id = self.get_empty_id(IRDType::Idx);
        self.vars[id] = (1, 0, Scope::Register, IRDType::Idx);
        return id as u8;
    }

    fn get_empty_id(&mut self, dtype: IRDType) -> usize {
        if let Some(id) = self.vars.iter().position(|(rc, _, _, vdtype)| *rc == 0 && *vdtype == dtype) {
            id
        } else {
            if self.vars.len() == 255 {
                panic!("Too many variables for one kernel.");
            }
            self.vars.push((0, 0, Scope::Register, dtype));
            self.vars.len() - 1
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

impl Display for Var {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Var::Id(id) => f.write_fmt(format_args!("r{id}")),
            Var::Const(value) => f.write_fmt(format_args!("{}", value.to_string())),
        }
    }
}

use alloc::string::String;
use alloc::format as f;

pub(super) fn to_str_kernel(ir_kernel: &IRKernel) -> String {
    let mut res = String::new();
    let mut indent  = f!("  ");
    for op in &ir_kernel.ops {
        match op {
            IROp::Set { z, len, value } => {
                res += &f!("{indent}r{}{} = {};\n", *z, if *len > 0 { f!("[{len}]") } else { String::new() }, value);
            }
            IROp::Load { z, x, at, .. } => {
                res += &f!("{indent}{z} = {x}[{at}];\n");
            },
            IROp::Store { z, x, at, .. } => {
                res += &f!("{indent}{z}[{at}] = {x};\n");
            },
            IROp::Unary { z, x, uop, .. } => {
                res += &f!("{indent}{z} = {uop:?}({x});\n");
            }
            IROp::Binary { z, x, y, bop, .. } => {
                res += &f!("{indent}{z} = {bop:?}({x}, {y});\n");
            },
            IROp::MAdd { z, a, b, c, .. } => {
                res += &f!("{indent}{z} = {a} * {b} + {c};\n");
            },
            IROp::Loop { id, len } => {
                res += &f!("{indent}for (unsigned int r{id} = 0; r{id} < {len}; r{id} += 1) {{\n");
                indent += "  ";
            }
            IROp::EndLoop => {
                indent.pop();
                indent.pop();
                res += &f!("{indent}}}\n");
            }
            IROp::Barrier { scope } => {
                res += &f!("{indent}barrier(CLK_{}AL_MEM_FENCE);\n", match scope {
                    Scope::Global => "GLOB",
                    Scope::Local => "LOC",
                    Scope::Register => panic!(),
                });
            }
        }
    }
    res
}
