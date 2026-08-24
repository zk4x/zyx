// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map,
    backend::DeviceInfo,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IdxKind, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, UOp},
    scalar::{bf16, f16},
};
use std::{fmt::Write, hash::BuildHasherDefault};

const VEC_COMPONENTS: [&str; 16] = [
    "x", "y", "z", "w", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sa", "sb",
];

impl Kernel {
    /// Compile kernel to OpenCL C source code.
    pub fn generate_opencl(&self, _device_info: &DeviceInfo, name: &str) -> Result<String, BackendError> {
        let mut global_args = String::new();
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_opencl did not finish in 10000 steps");
            }
            let op = self.at(op_id);
            if let &Op::Param { dtype, kind, .. } = op {
                match kind {
                    ParamKind::Variable => _ = writeln!(global_args, "  const {} p{op_id},", dtype.ocl()),
                    ParamKind::Global => _ = writeln!(global_args, "  __global const {}* p{op_id},", dtype.ocl()),
                    ParamKind::GlobalMut => _ = writeln!(global_args, "  __global {}* p{op_id},", dtype.ocl()),
                }
            } else {
                break;
            }
            op_id = self.next_op(op_id);
        }
        global_args.pop();
        global_args.pop();
        global_args.push('\n');

        let (dtypes, rcs) = self.compute_dtypes_and_rcs();

        let mut reg_map: Map<OpId, usize> = Map::with_capacity_and_hasher(self.ops.len().into(), BuildHasherDefault::new());
        let mut registers: Vec<((DType, MemLayout), u32, u8)> = Vec::new();

        let mut constants: Map<OpId, Constant> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut indices: Map<OpId, u8> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());

        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_opencl did not finish in 10000 steps");
            }
            match self.ops[op_id].op {
                Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::Reduce { .. }
                | Op::Move { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "OpenCL codegen: unexpected kernel op (should be unfolded)".into(),
                    });
                }
                Op::Asm { .. } => todo!(),
                Op::Const(x) => {
                    constants.insert(op_id, x);
                }
                Op::Param { .. } => {}
                Op::Storage { dtype, scope, len } => match scope {
                    MemScope::Local => {
                        _ = writeln!(source, "{indent}__local {} p{op_id}[{len}] __attribute__ ((aligned));", dtype.ocl())
                    }
                    MemScope::Register => {
                        _ = writeln!(source, "{indent}{} p{op_id}[{len}] __attribute__ ((aligned));", dtype.ocl())
                    }
                    _ => unreachable!("opencl only handles local or register storage"),
                },
                Op::Load { src, index, layout } => {
                    if rcs.contains_key(&op_id) {
                        let dtype = dtypes[&op_id];
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                        if matches!(self.ops[src].op, Op::Param { kind: ParamKind::Variable, .. }) {
                            _ = writeln!(source, "{indent}r{reg} = p{src};");
                        } else {
                            let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                            match layout {
                                MemLayout::Scalar => _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];"),
                                MemLayout::Vector(len) => {
                                    _ = writeln!(
                                        source,
                                        "{indent}r{reg} = *((__global {}*)(p{src} + {idx}));",
                                        dtype.0.ocl_vec_type(len)
                                    );
                                }
                                MemLayout::Tile { .. } => todo!(),
                            }
                        }
                    }
                }
                Op::Store { dst, src, index, layout } => {
                    let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let x = get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    match layout {
                        MemLayout::Scalar => _ = writeln!(source, "{indent}p{dst}[{idx}] = {x};"),
                        MemLayout::Vector(len) => {
                            let ocl_type = dtypes[&op_id].0.ocl_vec_type(len);
                            _ = writeln!(source, "{indent}*((__global {ocl_type}*)(p{dst} + {idx})) = {x};");
                        }
                        MemLayout::Tile { .. } => todo!(),
                    }
                }
                Op::Cast { x: xop, dtype } => {
                    let layout = dtypes[&xop].1;
                    let x = get_var(xop, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, layout), rcs[&op_id], loop_id);
                    match layout {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = writeln!(source, "{indent}r{reg}.{c} = ({}){x}.{c};", dtype.ocl());
                            }
                        }
                        _ => _ = writeln!(source, "{indent}r{reg} = ({}){x};", dtype.ocl()),
                    }
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = match uop {
                                    UOp::BitNot => writeln!(source, "{indent}r{reg}.{c} = ~{x}.{c};"),
                                    UOp::Neg => writeln!(source, "{indent}r{reg}.{c} = -{x}.{c};"),
                                    UOp::Exp => {
                                        if dtype.0 == DType::F16 {
                                            writeln!(source, "{indent}r{reg}.{c} = (half)exp((float){x}.{c});")
                                        } else {
                                            writeln!(source, "{indent}r{reg}.{c} = exp({x}.{c});")
                                        }
                                    }
                                    UOp::Exp2 => {
                                        if dtype.0 == DType::F16 {
                                            writeln!(source, "{indent}r{reg}.{c} = (half)exp2((float){x}.{c});")
                                        } else {
                                            writeln!(source, "{indent}r{reg}.{c} = exp2({x}.{c});")
                                        }
                                    }
                                    UOp::Log2 => writeln!(source, "{indent}r{reg}.{c} = log2({x}.{c});"),
                                    UOp::Reciprocal => {
                                        writeln!(source, "{indent}r{reg}.{c} = {}/{x}.{c};", dtype.0.one_constant().ocl())
                                    }
                                    UOp::Sqrt => writeln!(source, "{indent}r{reg}.{c} = sqrt({x}.{c});"),
                                    UOp::Sin => writeln!(source, "{indent}r{reg}.{c} = sin({x}.{c});"),
                                    UOp::Cos => writeln!(source, "{indent}r{reg}.{c} = cos({x}.{c});"),
                                    UOp::Floor => writeln!(source, "{indent}r{reg}.{c} = floor({x}.{c});"),
                                    UOp::Trunc => writeln!(source, "{indent}r{reg}.{c} = trunc({x}.{c});"),
                                    UOp::Ln => writeln!(source, "{indent}r{reg}.{c} = log({x}.{c});"),
                                    UOp::Abs => writeln!(source, "{indent}r{reg}.{c} = fabs({x}.{c});"),
                                };
                            }
                        }
                        MemLayout::Scalar => match uop {
                            UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                            UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                            UOp::Exp => {
                                if dtype.0 == DType::F16 {
                                    _ = writeln!(source, "{indent}r{reg} = (half)exp((float){x});");
                                } else {
                                    _ = writeln!(source, "{indent}r{reg} = exp({x});");
                                }
                            }
                            UOp::Exp2 => {
                                if dtype.0 == DType::F16 {
                                    _ = writeln!(source, "{indent}r{reg} = (half)exp2((float){x});");
                                } else {
                                    _ = writeln!(source, "{indent}r{reg} = exp2({x});");
                                }
                            }
                            UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                            UOp::Reciprocal => {
                                _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.0.one_constant().ocl());
                            }
                            UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                            UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                            UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                            UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                            UOp::Trunc => _ = writeln!(source, "{indent}r{reg} = trunc({x});"),
                            UOp::Ln => _ = writeln!(source, "{indent}r{reg} = log({x});"),
                            UOp::Abs => _ = writeln!(source, "{indent}r{reg} = fabs({x});"),
                        },
                        MemLayout::Tile { .. } => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "OpenCL codegen: Tile layout not supported for Binary".into(),
                            });
                        }
                    }
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&op_id];
                    let mut vars = String::new();
                    for &x in ops.iter() {
                        let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                        _ = write!(vars, "{x}, ");
                    }
                    vars.pop();
                    vars.pop();
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    let dtype = dtypes[&op_id];
                    let vlen = match dtype.1 {
                        MemLayout::Vector(len) => len,
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "OpenCL codegen: Vectorize requires Vector layout".into(),
                            });
                        }
                    };
                    _ = writeln!(source, "{indent}r{reg} = ({})({vars});", dtype.0.ocl_vec_type(vlen));
                }
                Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&op_id];
                    let vec = get_var(vec, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {vec}.{};", VEC_COMPONENTS[idx]);
                }
                Op::Wmma { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "OpenCL codegen does not support WMMA".into(),
                    });
                }
                Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = match bop {
                                    BOp::Add => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} + {y}.{c};"),
                                    BOp::Sub => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} - {y}.{c};"),
                                    BOp::Mul => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} * {y}.{c};"),
                                    BOp::Div => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} / {y}.{c};"),
                                    BOp::Pow => writeln!(source, "{indent}r{reg}.{c} = pow((double){x}.{c}, (double){y}.{c});"),
                                    BOp::Mod => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} % {y}.{c};"),
                                    BOp::Cmplt => writeln!(source, "{indent}r{reg}.{c} = (unsigned int)({x}.{c} < {y}.{c});"),
                                    BOp::Cmpgt => writeln!(source, "{indent}r{reg}.{c} = (unsigned int)({x}.{c} > {y}.{c});"),
                                    BOp::Cmpge => writeln!(source, "{indent}r{reg}.{c} = (unsigned int)({x}.{c} >= {y}.{c});"),
                                    BOp::Max => writeln!(source, "{indent}r{reg}.{c} = max({x}.{c}, {y}.{c});"),
                                    BOp::Or => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} || {y}.{c};"),
                                    BOp::And => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} && {y}.{c};"),
                                    BOp::BitXor => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} ^ {y}.{c};"),
                                    BOp::BitOr => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} | {y}.{c};"),
                                    BOp::BitAnd => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} & {y}.{c};"),
                                    BOp::BitShiftLeft => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} << {y}.{c};"),
                                    BOp::BitShiftRight => writeln!(source, "{indent}r{reg}.{c} = {x}.{c} >> {y}.{c};"),
                                    BOp::NotEq => writeln!(source, "{indent}r{reg}.{c} = (unsigned int)({x}.{c} != {y}.{c});"),
                                    BOp::Eq => writeln!(source, "{indent}r{reg}.{c} = (unsigned int)({x}.{c} == {y}.{c});"),
                                };
                            }
                        }
                        MemLayout::Scalar => {
                            _ = match bop {
                                BOp::Add => writeln!(source, "{indent}r{reg} = {x} + {y};"),
                                BOp::Sub => writeln!(source, "{indent}r{reg} = {x} - {y};"),
                                BOp::Mul => writeln!(source, "{indent}r{reg} = {x} * {y};"),
                                BOp::Div => writeln!(source, "{indent}r{reg} = {x} / {y};"),
                                BOp::Pow => writeln!(source, "{indent}r{reg} = pow((double){x}, (double){y});"),
                                BOp::Mod => writeln!(source, "{indent}r{reg} = {x} % {y};"),
                                BOp::Cmplt => writeln!(source, "{indent}r{reg} = {x} < {y};"),
                                BOp::Cmpgt => writeln!(source, "{indent}r{reg} = {x} > {y};"),
                                BOp::Cmpge => writeln!(source, "{indent}r{reg} = {x} >= {y};"),
                                BOp::Max => writeln!(source, "{indent}r{reg} = max({x}, {y});"),
                                BOp::Or => writeln!(source, "{indent}r{reg} = {x} || {y};"),
                                BOp::And => writeln!(source, "{indent}r{reg} = {x} && {y};"),
                                BOp::BitXor => writeln!(source, "{indent}r{reg} = {x} ^ {y};"),
                                BOp::BitOr => writeln!(source, "{indent}r{reg} = {x} | {y};"),
                                BOp::BitAnd => writeln!(source, "{indent}r{reg} = {x} & {y};"),
                                BOp::BitShiftLeft => writeln!(source, "{indent}r{reg} = {x} << {y};"),
                                BOp::BitShiftRight => writeln!(source, "{indent}r{reg} = {x} >> {y};"),
                                BOp::NotEq => writeln!(source, "{indent}r{reg} = {x} != {y};"),
                                BOp::Eq => writeln!(source, "{indent}r{reg} = {x} == {y};"),
                            }
                        }
                        MemLayout::Tile { .. } => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "OpenCL codegen: Tile layout not supported for Binary".into(),
                            });
                        }
                    }
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let z = get_var(z, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = writeln!(source, "{indent}r{reg}.{c} = {x}.{c} * {y}.{c} + {z}.{c};");
                            }
                        }
                        _ => _ = writeln!(source, "{indent}r{reg} = {x} * {y} + {z};"),
                    }
                }
                Op::Index { axis, kind: scope } => {
                    indices.insert(op_id, loop_id);
                    let id_fn = match scope {
                        IdxKind::Group(_) => "get_group_id",
                        IdxKind::Local(_) => "get_local_id",
                        IdxKind::Warp(_) => todo!(),
                    };
                    let max_idx = match scope {
                        IdxKind::Group(len_id) => {
                            self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim).unwrap().saturating_sub(1)
                        }
                        IdxKind::Local(len) => u64::from(len).saturating_sub(1),
                        IdxKind::Warp(_) => todo!(),
                    };
                    _ = writeln!(
                        source,
                        "{indent}{idx_type} idx{loop_id} = {id_fn}({axis}); // 0..={max_idx}",
                        idx_type = self.dtype(op_id).ocl(),
                    );
                    loop_id += 1;
                }
                Op::Loop { len, .. } => {
                    indices.insert(op_id, loop_id);
                    let len = get_var(len, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    _ = writeln!(
                        source,
                        "{indent}for ({idx_type} idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{",
                        idx_type = self.dtype(op_id).ocl()
                    );
                    indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                    loop_id -= 1;
                }
                Op::If { condition } => {
                    let condition = get_var(condition, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    _ = writeln!(source, "{indent}if ({condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Barrier => _ = writeln!(source, "{indent}barrier(CLK_LOCAL_MEM_FENCE);"),
            }
            op_id = self.next_op(op_id);
        }
        let mut reg_str = String::new();
        if !registers.is_empty() {
            let (dt, _, _) = registers.remove(0);
            let mut prev_dt = dt;
            _ = write!(
                reg_str,
                "{indent}{} r0",
                match dt.1 {
                    MemLayout::Scalar => dt.0.ocl().to_string(),
                    MemLayout::Vector(len) => dt.0.ocl_vec_type(len),
                    MemLayout::Tile { .. } =>
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "OpenCL codegen: Tile layout not supported in register declarations".into()
                        }),
                }
            );
            for (i, (dt, _, _)) in (1..).zip(registers) {
                if dt == prev_dt {
                    _ = write!(reg_str, ", r{i}");
                } else {
                    _ = write!(
                        reg_str,
                        ";\n{indent}{} r{i}",
                        match dt.1 {
                            MemLayout::Scalar => dt.0.ocl().to_string(),
                            MemLayout::Vector(len) => dt.0.ocl_vec_type(len),
                            MemLayout::Tile { .. } =>
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "OpenCL codegen: Tile layout not supported in register declarations".into()
                                }),
                        }
                    );
                }
                prev_dt = dt;
            }
            _ = writeln!(reg_str, ";");
        }

        let mut pragma = String::new();
        if dtypes.values().any(|&x| x.0 == DType::F16) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if dtypes.values().any(|&x| x.0 == DType::F64) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }

        Ok(format!("{pragma}__kernel void {name}(\n{global_args}) {{\n{reg_str}{source}}}\n"))
    }
}

fn new_reg(
    op_id: OpId,
    reg_map: &mut Map<OpId, usize>,
    registers: &mut Vec<((DType, MemLayout), u32, u8)>,
    dtype: (DType, MemLayout),
    rc: u32,
    current_loop_level: u8,
) -> usize {
    for (i, (dt, nrc, loop_level)) in registers.iter_mut().enumerate() {
        if *nrc == 0 && *dt == dtype && current_loop_level <= *loop_level {
            reg_map.insert(op_id, i);
            *nrc = rc;
            *loop_level = current_loop_level;
            return i;
        }
    }
    let i = registers.len();
    registers.push((dtype, rc, current_loop_level));
    reg_map.insert(op_id, i);
    i
}

fn get_var(
    op_id: OpId,
    constants: &Map<OpId, Constant>,
    indices: &Map<OpId, u8>,
    reg_map: &Map<OpId, usize>,
    registers: &mut [((DType, MemLayout), u32, u8)],
    loop_level: u8,
) -> Result<String, BackendError> {
    if let Some(c) = constants.get(&op_id) {
        Ok(c.ocl())
    } else if let Some(id) = indices.get(&op_id) {
        Ok(format!("idx{id}"))
    } else if let Some(reg) = reg_map.get(&op_id) {
        if registers[*reg].2 == loop_level {
            registers[*reg].1 -= 1;
        }
        Ok(format!("r{reg}"))
    } else {
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("OpenCL codegen: variable {op_id} not found").into(),
        })
    }
}

impl DType {
    fn ocl(self) -> &'static str {
        match self {
            Self::BF16 => todo!("bf16 should be casted to f16 or f32"),
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "uchar",
            Self::U16 => "ushort",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U32 => "uint",
            Self::U64 => "ulong",
        }
    }
    fn ocl_vec_type(self, len: u16) -> String {
        match self {
            Self::Bool => format!("uint{len}"),
            other => format!("{}{len}", other.ocl()),
        }
    }
}

impl Constant {
    fn ocl(self) -> String {
        match self {
            Self::BF16(x) => format!("{:.16}f", bf16::from_le_bytes(x)),
            Self::F16(x) => format!("(half){:.16}", f16::from_le_bytes(x)),
            Self::F32(x) => format!("{:.16}f", f32::from_le_bytes(x)),
            Self::F64(x) => format!("(double){:.16}", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("(int){x}"),
            Self::I64(x) => format!("(long){}", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{x}"),
        }
    }
}
