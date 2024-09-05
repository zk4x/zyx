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
use std::collections::BTreeMap;

use super::scheduler::{Kernel, VOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Var {
    Id(u8, Scope),
    Const(Constant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IROp {
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
    Loop {
        id: u8,
        len: usize,
    },
    EndLoop {
        id: u8,
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
    ) -> (IRKernel, Vec<(TensorId, bool)>) {
        let mut ops = Vec::new();
        let mut vars: VarMap = VarMap::new();
        let mut max_axis = 0;

        // Get all global args and set them first
        for &x in &self.inputs {
            let dtype = graph.dtype(x).into();
            let _ = vars.add_var(
                x,
                graph.shape(x).iter().product(),
                Scope::Global,
                graph.rc(x),
                dtype,
                Some(x),
                true,
            );
        }
        for &z in &self.outputs {
            let _ = vars.add_var(
                z,
                graph.shape(z).iter().product(),
                Scope::Global,
                graph.rc(z),
                graph.dtype(z).into(),
                Some(z),
                false,
            );
        }

        let mut loops = Vec::new();
        for vop in &self.ops {
            match vop {
                &VOp::Const { z, value, ref view } => {
                    let var = if view.requires_conditional_padding() {
                        vars.generate_padding(view, &mut ops, Var::Const(value), graph.rc(z), value.dtype(), None)
                    } else {
                        Var::Const(value)
                    };
                    vars.var_map.insert((z, Scope::Register), var);
                }
                VOp::Move { z, x, .. } => {
                    vars.noop(*z, *x, graph.rc(*z));
                }
                &VOp::Load { z, zscope, x, xscope, ref view } => {
                    let dtype = graph.dtype(z).into();
                    let at = vars.generate_idx(view, &mut ops);
                    let x = vars.get(x, Scope::Global);
                    let zvar = vars.add_var(z, 0, Scope::Register, graph.rc(z), graph.dtype(z).into(), None, false);
                    if view.requires_conditional_padding() {
                        let var = vars.generate_padding(view, &mut ops, zvar, graph.rc(z), graph.dtype(z), Some((x, at, dtype)));
                        vars.var_map.remove(&(z, Scope::Register));
                        vars.var_map.insert((z, Scope::Register), var);
                    } else {
                        ops.push(IROp::Load {
                            z: zvar,
                            x,
                            at,
                            dtype,
                        });
                        vars.remove_var(at);
                    }
                }
                &VOp::Store { z, zscope, xscope, ref view } => {
                    let dtype = graph.dtype(z).into();
                    let at = vars.generate_idx(view, &mut ops);
                    let x = vars.get(z, Scope::Register);
                    let z = vars.get(z, Scope::Global);
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
                    loops.push((id, *dimension));
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
                        false,
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
                        let (id, len) = loops.pop().unwrap();
                        ops.push(IROp::EndLoop { id, len });
                        vars.remove_axis(max_axis);
                        max_axis -= 1;
                    }
                }
                VOp::EndLoop => {
                    let (id, len) = loops.pop().unwrap();
                    ops.push(IROp::EndLoop { id, len });
                }
                VOp::Unary { z, x, uop } => {
                    //println!("IR Unary {uop:?} on {:?}", vars.get(*x, Scope::Register));
                    if let Var::Const(v) = vars.get(*x, Scope::Register) {
                        vars.var_map.insert((*z, Scope::Register), Var::Const(v.unary(*uop)));
                    } else {
                        let x_tensor = *x;
                        let dtype = graph.dtype(*x).into();
                        let x = vars.get(*x, Scope::Register);
                        let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into(), None, false);
                        ops.push(IROp::Unary {
                            z,
                            x,
                            uop: *uop,
                            dtype,
                        });
                        vars.remove(x_tensor);
                    }
                }
                VOp::Binary { z, x, y, bop } => {
                    if let (Var::Const(xv), Var::Const(yv)) = (vars.get(*x, Scope::Register), vars.get(*y, Scope::Register)) {
                        vars.var_map.insert((*z, Scope::Register), Var::Const(Constant::binary(xv, yv, *bop)));
                    } else {
                        let x_tensor = *x;
                        let y_tensor = *y;
                        let dtype = graph.dtype(*z).into();
                        let x = vars.get(*x, Scope::Register);
                        let y = vars.get(*y, Scope::Register);
                        let z = vars.add_var(*z, 0, Scope::Register, graph.rc(*z), graph.dtype(*z).into(), None, false);
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
        }

        while let Some((id, len)) = loops.pop() {
            ops.push(IROp::EndLoop { id, len });
        }

        let mut addressables = Vec::new();
        let mut args = Vec::new();
        for (len, dtype, read_only, tensor) in vars.addressables.into_iter() {
            addressables.push((len, dtype, read_only));
            args.push((tensor.unwrap(), read_only));
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

    fn add_var(&mut self, x: TensorId, len: usize, scope: Scope, rc: u32, dtype: IRDType, tensor: Option<TensorId>, read_only: bool) -> Var {
        /*if let Some(var) = self.var_map.get(&x) {
            return *var;
        }*/
        let id = self.get_empty_id(rc, len, dtype, scope, tensor, read_only);
        let var = Var::Id(id as u8, scope);
        self.var_map.insert((x, scope), var);
        return var;
    }

    fn generate_idx(&mut self, view: &View, ops: &mut Vec<IROp>) -> Var {
        match view {
            View::None => Var::Const(Constant::U32(0)),
            View::Strided(dims) => {
                let z = self.zero_u32_var(ops);
                let numel: usize = dims.iter().flat_map(|StridedDim { dim, stride, .. }| if *stride != 0 { Some(*dim) } else { None }).product();
                for StridedDim { axis, stride, .. } in dims {
                    if *stride != 0 && *stride != numel {
                        let a = self.get_axis(*axis);
                        ops.push(IROp::MAdd {
                            z,
                            a,
                            b: Var::Const(Constant::U32(*stride as u32)),
                            c: z,
                            dtype: IRDType::U32,
                        });
                    }
                }
                z
            }
            View::Padded(dims, padding) => {
                let z = self.zero_u32_var(ops);
                for StridedDim { axis, stride, .. } in dims {
                    if let Some((_, (lp, _))) = padding
                        .axes
                        .iter()
                        .find(|(axes, _)| axes.iter().max().unwrap() == axis)
                    {
                        println!("Padding {axis} with {lp}");
                        if *lp > 0 {
                            let t = Var::Id(self.get_empty_id(1, 0, IRDType::U32, Scope::Register, None, false) as u8, Scope::Register);
                            ops.push(IROp::Binary { z: t, x: self.get_axis(*axis), y: Var::Const(Constant::U32(*lp as u32)), bop: BOp::Sub, dtype: IRDType::U32 });
                            ops.push(IROp::MAdd { z, a: t, b: Var::Const(Constant::U32(*stride as u32)), c: z, dtype: IRDType::U32 } );
                        } else if *lp < 0 {
                            let lp = -lp;
                            let t = Var::Id(self.get_empty_id(1, 0, IRDType::U32, Scope::Register, None, false) as u8, Scope::Register);
                            ops.push(IROp::Binary { z: t, x: self.get_axis(*axis), y: Var::Const(Constant::U32(lp as u32)), bop: BOp::Add, dtype: IRDType::U32 });
                            ops.push(IROp::MAdd { z, a: t, b: Var::Const(Constant::U32(*stride as u32)), c: z, dtype: IRDType::U32 } );
                        } else {
                            ops.push(IROp::MAdd {
                                z,
                                a: self.get_axis(*axis),
                                b: Var::Const(Constant::U32(*stride as u32)),
                                c: z,
                                dtype: IRDType::U32,
                            });
                        }
                        //std::println!("dim: {dim}, paddding {lp}, {rp}");
                    } else {
                        ops.push(IROp::MAdd {
                            z,
                            a: self.get_axis(*axis),
                            b: Var::Const(Constant::U32(*stride as u32)),
                            c: z,
                            dtype: IRDType::U32,
                        });
                    }
                }
                z
            }
        }
    }

    // Takes self, view, ops and var without padding, returns var with padding applied
    // It does it all branchlessly
    // TODO this is little ugly, make it more straigthforward
    fn generate_padding(&mut self, view: &View, ops: &mut Vec<IROp>, var: Var, rc: u32, dtype: DType, load: Option<(Var, Var, IRDType)>) -> Var {
        let View::Padded(dims, padding) = view  else { panic!() };
        //std::println!("Using padded index");

        // When the padding does not apply
        let padding_condition = self.get_empty_id(1, 0, IRDType::Bool, Scope::Register, None, false) as u8;
        ops.push(IROp::Set {
            z: padding_condition,
            len: 0,
            value: Constant::Bool(false),
        });
        let padding_condition = Var::Id(padding_condition, Scope::Register);
        let mut pc = String::new();
            println!("Padding: {:?}", padding.axes);
        for StridedDim { axis, .. } in dims {
            if let Some((axes, (lp, rp))) = padding
                .axes
                .iter()
                .find(|(axes, _)| axes.iter().max().unwrap() == axis)
            {
                let mut idx = String::new();
                let mut st = 1;
                let mut dim = 1;
                for axis in axes.iter().rev() {
                    idx = format!("i{axis}*{st}+{idx}");
                    st *= dims[*axis].dim;
                    dim *= dims[*axis].dim;
                }
                idx.pop();
                if *lp > 0 {
                    pc += &format!("{idx} < {lp} || ");
                }
                if *rp > 0 {
                    pc += &format!("{idx} > {} || ", dim as isize - rp - 1);
                }

                let idx = self.zero_u32_var(ops);
                let mut st = 1;
                let mut dim = 1;
                for axis in axes.iter().rev() {
                    ops.push(IROp::MAdd {
                        z: idx,
                        a: self.get_axis(*axis),
                        b: Var::Const(Constant::U32(st as u32)),
                        c: idx,
                        dtype: IRDType::U32,
                    });
                    st *= dims[*axis].dim;
                    dim *= dims[*axis].dim;
                }

                if *lp > 0 {
                    //padding_condition += &format!("{idx} < {lp} || ");
                    let temp = Var::Id(self.get_empty_id(1, 0, IRDType::Bool, Scope::Register, None, false) as u8, Scope::Register);
                    ops.push(IROp::Binary {
                        z: temp,
                        x: idx,
                        y: Var::Const(Constant::U32(*lp as u32)),
                        dtype: IRDType::U32,
                        bop: BOp::Cmplt,
                    });
                    ops.push(IROp::Binary {
                        z: padding_condition,
                        x: temp,
                        y: padding_condition,
                        dtype: IRDType::U32,
                        bop: BOp::Or,
                    });
                    self.remove_var(temp);
                }
                if *rp > 0 {
                    //padding_condition += &format!("{idx} > {} || ", dim as isize - rp - 1);
                    let temp = Var::Id(self.get_empty_id(1, 0, IRDType::Bool, Scope::Register, None, false) as u8, Scope::Register);
                    ops.push(IROp::Binary {
                        z: temp,
                        x: idx,
                        y: Var::Const(Constant::U32((dim as isize - *rp - 1) as u32)),
                        dtype: IRDType::U32,
                        bop: BOp::Cmpgt,
                    });
                    ops.push(IROp::Binary {
                        z: padding_condition,
                        x: temp,
                        y: padding_condition,
                        dtype: IRDType::U32,
                        bop: BOp::Or,
                    });
                    self.remove_var(temp);
                }
                self.remove_var(idx);
            }
        }
        println!("Padding condition: {pc}");
        // padding_condition * 0 + !padding_condition * var
        let temp = Var::Id(self.get_empty_id(1, 0, dtype.into(), Scope::Register, None, false) as u8, Scope::Register);
        ops.push(IROp::Binary {
            z: temp,
            x: padding_condition,
            y: Var::Const(dtype.zero_constant()),
            bop: BOp::Mul,
            dtype: dtype.into(),
        });
        ops.push(IROp::Unary {
            z: padding_condition,
            x: padding_condition,
            uop: UOp::Not,
            dtype: dtype.into(),
        });
        if let Some((x, at, dtype)) = load {
            ops.push(IROp::Binary {
                z: at,
                x: padding_condition,
                y: at,
                bop: BOp::Mul,
                dtype: IRDType::U32,
            });
            ops.push(IROp::Load {
                z: var,
                x,
                at,
                dtype,
            });
            self.remove_var(at);
        }
        let temp1 = Var::Id(self.get_empty_id(1, 0, dtype.into(), Scope::Register, None, false) as u8, Scope::Register);
        ops.push(IROp::Binary {
            z: temp1,
            x: padding_condition,
            y: var,
            bop: BOp::Mul,
            dtype: dtype.into(),
        });
        let res = Var::Id(self.get_empty_id(rc, 0, dtype.into(), Scope::Register, None, false) as u8, Scope::Register);
        ops.push(IROp::Binary {
            z: res,
            x: temp,
            y: temp1,
            bop: BOp::Add,
            dtype: dtype.into(),
        });
        self.remove_var(temp);
        self.remove_var(temp1);
        self.remove_var(padding_condition);
        res
    }

    fn zero_u32_var(&mut self, ops: &mut Vec<IROp>) -> Var {
        let id = self.get_empty_id(1, 0, IRDType::U32, Scope::Register, None, false) as u8;
        ops.push(IROp::Set {
            z: id,
            len: 0,
            value: Constant::U32(0),
        });
        Var::Id(id, Scope::Register)
    }

    fn add_axis(&mut self, axis: Axis) -> u8 {
        let id = self.get_empty_id(1, 0, IRDType::U32, Scope::Register, None, false) as u8;
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

    fn get_empty_id(&mut self, rc: u32, len: usize, dtype: IRDType, scope: Scope, tensor: Option<TensorId>, read_only: bool) -> usize {
        // This finds variable with the same dtype, however
        // we often can use registers that can hold variables of different dtypes,
        // so the dtype equality check will be disabled for some devices.
        match scope {
            Scope::Global | Scope::Local => {
                self.addressables.push((len, dtype, read_only, tensor));
                self.addressables.len() - 1
            }
            Scope::Register => {
                if let Some(id) = self
                    .registers
                    .iter()
                    .position(|(rc_, vdtype, _)| *rc_ == 0 && *vdtype == dtype)
                {
                    self.registers[id] = (rc, dtype, read_only);
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

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::Register => "r",
        })
    }
}

impl IRKernel {
    pub(super) fn debug(&self) {
        println!();
        for op in &self.ops {
            println!("{op:?}");
        }
        println!()
    }
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

impl Constant {
    pub(super) fn ir_dtype(&self) -> IRDType {
        match self {
            #[cfg(feature = "half")]
            Constant::BF16(_) => IRDType::BF16,
            #[cfg(feature = "half")]
            Constant::F16(_) => IRDType::F16,
            Constant::F32(_) => IRDType::F32,
            Constant::F64(_) => IRDType::F64,
            #[cfg(feature = "complex")]
            Constant::CF32(..) => IRDType::CF32,
            #[cfg(feature = "complex")]
            Constant::CF64(..) => IRDType::CF64,
            Constant::U8(_) => IRDType::U8,
            Constant::I8(_) => IRDType::I8,
            Constant::I16(_) => IRDType::U32,
            Constant::U32(_) => IRDType::I32,
            Constant::I32(_) => IRDType::I32,
            Constant::I64(_) => IRDType::I64,
            Constant::Bool(_) => IRDType::Bool,
        }
    }
}
