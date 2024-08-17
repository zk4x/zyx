use crate::runtime::view::{StridedDim, View};
use crate::shape::Axis;
use crate::{
    dtype::Constant,
    runtime::{
        graph::Graph,
        node::{BOp, ROp, UOp},
    },
    tensor::TensorId,
    DType,
};
use core::fmt::Display;
use std::{collections::BTreeMap, string::ToString};

use super::scheduler::{Kernel, VOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Var {
    Id(u8, Scope),
    Const(Constant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum IROp {
    Set {
        z: u8,
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
    Loop {
        id: u8,
        len: usize,
    },
    EndLoop,
    Barrier {
        scope: Scope,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IRDType {
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
    Idx,
}

#[derive(Debug)]
pub(crate) struct IRKernel {
    // Index of var is it's Id
    // len, dtype, read_only
    pub(super) addressables: Vec<(usize, IRDType, bool)>,
    // dtype, read_only
    pub(super) registers: Vec<(IRDType, bool)>,
    pub(super) ops: Vec<IROp>,
}

// TODO do not use rcs from graph, because we only care about rcs in single kernel,
// but graph contains rcs across multiple kernels
impl Kernel {
    /// Returns IRKernel and global arguments
    pub(super) fn to_ir(
        &self,
        graph: &Graph,
        //hwinfo: &DeviceInfo,
    ) -> (IRKernel, Vec<TensorId>) {
        let mut ops = Vec::new();

        let mut vars: VarMap = VarMap::new();
        let mut max_axis = 0;

        // Get all global args and set them first
        for vop in &self.ops {
            match &vop {
                &VOp::Load { z: _, x, view } => {
                    let dtype = graph.dtype(*x).into();
                    let _ = vars.add_var(
                        *x,
                        view.numel(),
                        Scope::Global,
                        graph.rc(*x),
                        dtype,
                        Some(*x),
                    );
                }
                VOp::Store { z, view } => {
                    let _ = vars.add_var(
                        *z,
                        view.numel(),
                        Scope::Global,
                        graph.rc(*z),
                        graph.dtype(*z).into(),
                        Some(*z),
                    );
                }
                _ => {}
            }
        }

        for vop in &self.ops {
            match vop {
                VOp::Const { z, value } => {
                    vars.add_const(*z, *value);
                }
                VOp::Noop { z, x } => {
                    vars.noop(*z, *x, graph.rc(*z));
                }
                VOp::Load { z, x, view } => {
                    let dtype = graph.dtype(*z).into();
                    let at = vars.generate_idx(view, &mut ops);
                    let x = vars.get(*x, Scope::Global);
                    let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into(), None);
                    ops.push(IROp::Load {
                        z,
                        x,
                        at,
                        dtype,
                    });
                    vars.remove_var(at);
                }
                VOp::Store { z, view } => {
                    let dtype = graph.dtype(*z).into();
                    let at = vars.generate_idx(view, &mut ops);
                    let x = vars.get(*z, Scope::Register);
                    let z = vars.get(*z, Scope::Global);
                    ops.push(IROp::Store {
                        z,
                        x,
                        at,
                        dtype,
                    });
                    vars.remove_var(at);
                }
                VOp::Loop { axis, dimension } => {
                    let id = vars.add_axis(*axis);
                    ops.push(IROp::Loop {
                        id,
                        len: *dimension,
                    });
                    max_axis += 1;
                }
                VOp::Accumulator { z, rop, view } => {
                    let dtype: DType = graph.dtype(*z).into();
                    let len = view.numel();
                    let Var::Id(z, _) = vars.add_var(
                        *z,
                        len,
                        Scope::Register,
                        graph.rc(*z),
                        graph.dtype(*z).into(),
                        None,
                    ) else {
                        panic!()
                    };
                    ops.push(IROp::Set {
                        z,
                        len,
                        value: match *rop {
                            ROp::Sum => dtype.zero_constant(),
                            ROp::Max => dtype.min_constant(),
                        },
                    });
                }
                VOp::Reduce {
                    z,
                    x,
                    num_axes,
                    rop,
                } => {
                    let dtype = graph.dtype(*z).into();
                    let bop = match *rop {
                        ROp::Sum => BOp::Add,
                        ROp::Max => BOp::Max,
                    };
                    let z_var = vars.get(*z, Scope::Register);
                    ops.push(IROp::Binary {
                        z: z_var,
                        x: vars.get(*x, Scope::Register),
                        y: z_var,
                        bop,
                        dtype,
                    });
                    vars.remove(*x);
                    //vars.remove(*z);
                    for _ in 0..*num_axes {
                        ops.push(IROp::EndLoop);
                        vars.remove_axis(max_axis);
                        max_axis -= 1;
                    }
                }
                VOp::Unary { z, x, uop } => {
                    let x_tensor = *x;
                    let dtype = graph.dtype(*z).into();
                    let x = vars.get(*x, Scope::Register);
                    let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into(), None);
                    ops.push(IROp::Unary {
                        z,
                        x,
                        uop: *uop,
                        dtype,
                    });
                    vars.remove(x_tensor);
                }
                VOp::Binary { z, x, y, bop } => {
                    let x_tensor = *x;
                    let y_tensor = *y;
                    let dtype = graph.dtype(*z).into();
                    let x = vars.get(*x, Scope::Register);
                    let y = vars.get(*y, Scope::Register);
                    let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into(), None);
                    ops.push(IROp::Binary {
                        z,
                        x,
                        y,
                        bop: *bop,
                        dtype,
                    });
                    vars.remove(x_tensor);
                    vars.remove(y_tensor);
                }
            }
        }

        let mut addressables = Vec::new();
        let mut args = Vec::new();
        for (len, dtype, read_only, tensor) in vars.addressables.into_iter() {
            addressables.push((len, dtype, read_only));
            args.push(tensor.unwrap());
        }

        (IRKernel { addressables, registers: vars.registers.into_iter().map(|(_, dtype, read_only)| (dtype, read_only)).collect(), ops }, args)
    }
}

#[derive(Debug)]
struct VarMap {
    // length, dtype, read_only
    addressables: Vec<(usize, IRDType, bool, Option<TensorId>)>,
    // Ref count, dtype, read_only
    registers: Vec<(u32, IRDType, bool)>,
    var_map: BTreeMap<(TensorId, Scope), Var>,
    axis_map: BTreeMap<Axis, Var>,
}

impl VarMap {
    fn new() -> VarMap {
        VarMap {
            addressables: Vec::new(),
            registers: Vec::new(),
            var_map: BTreeMap::new(),
            axis_map: BTreeMap::new(),
        }
    }

    fn noop(&mut self, z: TensorId, x: TensorId, z_rc: u32) {
        let var = self.var_map[&(x, Scope::Register)];
        self.var_map.insert((z, Scope::Register), var);
        if let Var::Id(id, _) = var {
            self.registers[id as usize].0 += z_rc - 1;
        }
    }

    fn get(&self, tensor_id: TensorId, scope: Scope) -> Var {
        self.var_map[&(tensor_id, scope)]
    }

    /// Decrease ref count if it isn't constant
    fn remove(&mut self, tensor_id: TensorId) {
        if let Some(Var::Id(id, scope)) = self.var_map.get(&(tensor_id, Scope::Register)) {
            match scope {
                Scope::Global | Scope::Local => {}
                Scope::Register => {
                    self.registers[*id as usize].0 -= 1;
                }
            }
        }
    }

    fn add_const(&mut self, x: TensorId, value: Constant) -> Var {
        let var = Var::Const(value);
        self.var_map.insert((x, Scope::Register), var);
        return var;
    }

    fn add_var(&mut self, x: TensorId, len: usize, scope: Scope, rc: u32, dtype: IRDType, tensor: Option<TensorId>) -> Var {
        /*if let Some(var) = self.var_map.get(&x) {
            return *var;
        }*/
        let id = self.get_empty_id(rc, len, dtype, scope, tensor);
        let var = Var::Id(id as u8, scope);
        self.var_map.insert((x, scope), var);
        return var;
    }

    fn generate_idx(&mut self, view: &View, ops: &mut Vec<IROp>) -> Var {
        match view {
            View::None => Var::Const(Constant::I64(0)),
            View::Strided(dims) => {
                let z = self.get_empty_id(1, 0, IRDType::Idx, Scope::Register, None) as u8;
                ops.push(IROp::Set {
                    z,
                    len: 0,
                    value: Constant::I64(0),
                });
                let z = Var::Id(z, Scope::Register);
                for StridedDim { axis, stride, .. } in dims {
                    if *stride != 0 {
                        let a = self.get_axis(*axis);
                        ops.push(IROp::MAdd {
                            z,
                            a,
                            b: Var::Const(Constant::I64(*stride as i64)),
                            c: z,
                            dtype: IRDType::Idx,
                        });
                    }
                }
                z
            }
            View::Padded(_, _) => todo!(),
        }
    }

    fn add_axis(&mut self, axis: Axis) -> u8 {
        let id = self.get_empty_id(1, 0, IRDType::Idx, Scope::Register, None) as u8;
        let var = Var::Id(id, Scope::Register);
        self.axis_map.insert(axis, var);
        return id;
    }

    fn get_axis(&mut self, axis: Axis) -> Var {
        self.axis_map[&axis]
    }

    fn remove_axis(&mut self, axis: Axis) {
        if let Some(Var::Id(id, ..)) = self.axis_map.get(&axis) {
            self.registers[*id as usize].0 -= 1;
        }
    }

    fn remove_var(&mut self, var: Var) {
        if let Var::Id(id, scope) = var {
            match scope {
                Scope::Global | Scope::Local => {
                    self.addressables[id as usize].0 -= 1;
                }
                Scope::Register => {
                    self.registers[id as usize].0 -= 1;
                }
            }
        }
    }

    fn get_empty_id(&mut self, rc: u32, len: usize, dtype: IRDType, scope: Scope, tensor: Option<TensorId>) -> usize {
        // This finds variable with the same dtype, however
        // we often can use registers that can hold variables of different dtypes,
        // so the dtype equality check will be disabled for some devices.
        match scope {
            Scope::Global | Scope::Local => {
                self.addressables.push((len, dtype, false, tensor));
                self.addressables.len() - 1
            }
            Scope::Register => {
                if let Some(id) = self
                    .registers
                    .iter()
                    .position(|(rc_, vdtype, _)| *rc_ == 0 && *vdtype == dtype)
                {
                    self.registers[id] = (rc, dtype, false);
                    id
                } else {
                    if self.registers.len() == 255 {
                        panic!("Too many variables for one kernel.");
                    }
                    self.registers.push((rc, dtype, false));
                    self.registers.len() - 1
                }
            }
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
            Var::Id(id, scope) => f.write_fmt(format_args!("{scope}{id}")),
            Var::Const(value) => f.write_fmt(format_args!("{}", value.to_string())),
        }
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::Register => "r",
        })
    }
}
