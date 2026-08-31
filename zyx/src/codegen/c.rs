// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map,
    backend::DeviceInfo,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, UOp},
    scalar::{bf16, f16},
};
use std::{fmt::Write, hash::BuildHasherDefault};

impl Kernel {
    /// Compile kernel to C source code.
    pub fn generate_c(&self, device_info: &DeviceInfo, has_openmp: bool, name: &str) -> Result<String, BackendError> {
        let (dtypes, rcs) = self.compute_dtypes_and_rcs();

        let mut reg_map: Map<OpId, usize> = Map::with_capacity_and_hasher(self.ops.len().into(), BuildHasherDefault::new());
        let mut registers: Vec<((DType, MemLayout), u32, u8)> = Vec::new();
        let mut constants: Map<OpId, Constant> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut indices: Map<OpId, u8> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());

        let mut loop_id: u8 = 0;
        let mut global_cast = String::new();
        let mut n_params: usize = 0;
        {
            let mut op_id = self.head;
            let mut steps_op_id = 0usize;
            while !op_id.is_null() {
                steps_op_id += 1;
                if steps_op_id > 10_000 {
                    panic!("generate_c did not finish in 10000 steps");
                }
                match self.ops[op_id].op {
                    Op::Range { kind: scope, .. } => {
                        if !matches!(scope, RangeKind::Group(_)) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "C codegen: C only supports group index".into(),
                            });
                        }
                        indices.insert(op_id, loop_id);
                        loop_id = loop_id.checked_add(1).expect("C: too many loops (>255)");
                    }
                    Op::Storage { dtype, scope: MemScope::Global, .. } => {
                        if matches!(dtype, DType::F16 | DType::BF16) {
                            _ = writeln!(global_cast, "  unsigned short* p{op_id} = (unsigned short*)args[{n_params}];");
                        } else {
                            let ct = dtype.c_type();
                            _ = writeln!(global_cast, "  {ct}* p{op_id} = ({ct}*)args[{n_params}];");
                        }
                        n_params += 1;
                    }
                    Op::Param { kind: ParamKind::Variable, dtype, .. } => {
                        if matches!(dtype, DType::F16 | DType::BF16) {
                            _ = writeln!(global_cast, "  unsigned short p{op_id} = *(unsigned short*)args[{n_params}];");
                        } else {
                            let ct = dtype.c_type();
                            _ = writeln!(global_cast, "  {ct} p{op_id} = *({ct}*)args[{n_params}];");
                        }
                        n_params += 1;
                    }
                    Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, dtype, .. } => {
                        if matches!(dtype, DType::F16 | DType::BF16) {
                            _ = writeln!(global_cast, "  unsigned short* p{op_id} = (unsigned short*)args[{n_params}];");
                        } else {
                            let ct = dtype.c_type();
                            _ = writeln!(global_cast, "  {ct}* p{op_id} = ({ct}*)args[{n_params}];");
                        }
                        n_params += 1;
                    }
                    _ => {}
                }
                op_id = self.next_op(op_id);
            }
        }

        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut index_loop_depth: u8 = 0;
        loop_id = 0;
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_c did not finish in 10000 steps");
            }
            match self.ops[op_id].op {
                Op::Range { kind: scope, .. } => {
                    let RangeKind::Group(len) = scope else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "C codegen: LocalIndex not expected".into(),
                        });
                    };
                    if index_loop_depth == 0 && has_openmp {
                        _ = writeln!(source, "{indent}#pragma omp parallel for");
                    }
                    let len = get_var(len, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    _ = writeln!(
                        source,
                        "{indent}for ({idx_type} idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{",
                        idx_type = self.dtype(op_id).c_type(),
                    );
                    indent += "  ";
                    index_loop_depth += 1;
                    loop_id += 1;
                }
                Op::Loop { len, .. } => {
                    indices.insert(op_id, loop_id);
                    let len = get_var(len, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    _ = writeln!(
                        source,
                        "{indent}for ({idx_type} idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{",
                        idx_type = self.dtype(op_id).c_type()
                    );
                    indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    if indent.len() < 2 {
                        indent = String::from("  ");
                    }
                    _ = writeln!(source, "{indent}}}");
                    loop_id -= 1;
                }
                Op::Const(x) => {
                    constants.insert(op_id, x);
                }
                Op::Load { src, index, layout } => {
                    if let Some(&rc) = rcs.get(&op_id) {
                        let dtype = dtypes[&op_id];
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rc, loop_id);
                        if matches!(self.ops[src].op, Op::Param { kind: ParamKind::Variable, .. }) {
                            match dtypes[&src].0 {
                                DType::F16 => _ = writeln!(source, "{indent}r{reg} = f16tof32(p{src});"),
                                DType::BF16 => _ = writeln!(source, "{indent}r{reg} = bf16tof32(p{src});"),
                                _ => _ = writeln!(source, "{indent}r{reg} = p{src};"),
                            }
                        } else {
                            let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                            match layout {
                                MemLayout::Scalar => match dtypes[&src].0 {
                                    DType::F16 => {
                                        _ = writeln!(source, "{indent}r{reg} = f16tof32(p{src}[{idx}]);");
                                    }
                                    DType::BF16 => {
                                        _ = writeln!(source, "{indent}r{reg} = bf16tof32(p{src}[{idx}]);");
                                    }
                                    _ => {
                                        _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];");
                                    }
                                },
                                MemLayout::Vector(len) => match dtypes[&src].0 {
                                    DType::F16 => {
                                        for i in 0..len {
                                            _ = writeln!(
                                                source,
                                                "{indent}{} = f16tof32(p{src}[{idx} + {i}]);",
                                                lane_access(&format!("r{reg}"), i as usize)
                                            );
                                        }
                                    }
                                    DType::BF16 => {
                                        for i in 0..len {
                                            _ = writeln!(
                                                source,
                                                "{indent}{} = bf16tof32(p{src}[{idx} + {i}]);",
                                                lane_access(&format!("r{reg}"), i as usize)
                                            );
                                        }
                                    }
                                    _ if !device_info.supported_vec_lens.is_empty() => {
                                        _ = writeln!(
                                            source,
                                            "{indent}r{reg} = *(({}*)(p{src} + {idx}));",
                                            dtype.0.vec_type_name(len)
                                        );
                                    }
                                    _ => {
                                        for i in 0..len {
                                            _ = writeln!(
                                                source,
                                                "{indent}{} = p{src}[{idx} + {i}];",
                                                lane_access(&format!("r{reg}"), i as usize)
                                            );
                                        }
                                    }
                                },
                                MemLayout::Tile { .. } => {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: "C codegen: Tile layout not supported for Load".into(),
                                    });
                                }
                            }
                        }
                    }
                }
                Op::Store { dst, src, index, layout } => {
                    let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let x = get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    match layout {
                        MemLayout::Scalar => match dtypes[&dst].0 {
                            DType::F16 => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = f32tof16({x});");
                            }
                            DType::BF16 => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = f32tobf16({x});");
                            }
                            _ => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = {x};");
                            }
                        },
                        MemLayout::Vector(len) => match dtypes[&dst].0 {
                            DType::F16 => {
                                for i in 0..len {
                                    _ = writeln!(
                                        source,
                                        "{indent}p{dst}[{idx} + {i}] = f32tof16({});",
                                        lane_access(&x, i as usize)
                                    );
                                }
                            }
                            DType::BF16 => {
                                for i in 0..len {
                                    _ = writeln!(
                                        source,
                                        "{indent}p{dst}[{idx} + {i}] = f32tobf16({});",
                                        lane_access(&x, i as usize)
                                    );
                                }
                            }
                            _ if !device_info.supported_vec_lens.is_empty() => {
                                let ocl_type = dtypes[&dst].0.c_type();
                                _ = writeln!(source, "{indent}*(({ocl_type}{len}*)(p{dst} + {idx})) = {x};");
                            }
                            _ => {
                                for i in 0..len {
                                    _ = writeln!(source, "{indent}p{dst}[{idx} + {i}] = {};", lane_access(&x, i as usize));
                                }
                            }
                        },
                        MemLayout::Tile { .. } => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "C codegen: Tile layout not supported for Store".into(),
                            });
                        }
                    }
                }
                Op::Cast { x, dtype } => {
                    let vlen = dtypes[&x].1;
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, vlen), rcs[&op_id], loop_id);
                    match vlen {
                        MemLayout::Vector(n) => {
                            for i in 0..n {
                                _ = writeln!(
                                    source,
                                    "{indent}{} = ({})({});",
                                    lane_access(&format!("r{reg}"), i as usize),
                                    dtype.c_type(),
                                    lane_access(&x, i as usize)
                                );
                            }
                        }
                        _ => _ = writeln!(source, "{indent}r{reg} = ({}){x};", dtype.c_type()),
                    }
                }
                Op::Bitcast { x, dtype } => {
                    let vlen = dtypes[&x].1;
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, vlen), rcs[&op_id], loop_id);
                    let byte_size = dtype.bit_size() as usize / 8;
                    match vlen {
                        MemLayout::Vector(n) => {
                            for i in 0..n {
                                _ = writeln!(
                                    source,
                                    "memcpy(&{}, &{}, {byte_size});",
                                    lane_access(&format!("r{reg}"), i as usize),
                                    lane_access(&x, i as usize)
                                );
                            }
                        }
                        _ => _ = writeln!(source, "memcpy(&r{reg}, &{x}, {byte_size});"),
                    }
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(n) => {
                            for i in 0..n {
                                let lane = lane_access(&x, i as usize);
                                let dst = lane_access(&format!("r{reg}"), i as usize);
                                match uop {
                                    UOp::BitNot => _ = writeln!(source, "{indent}{dst} = ~{lane};"),
                                    UOp::Neg => _ = writeln!(source, "{indent}{dst} = -{lane};"),
                                    UOp::Exp => _ = writeln!(source, "{indent}{dst} = exp({lane});"),
                                    UOp::Exp2 => _ = writeln!(source, "{indent}{dst} = exp2({lane});"),
                                    UOp::Ln => _ = writeln!(source, "{indent}{dst} = log({lane});"),
                                    UOp::Log2 => _ = writeln!(source, "{indent}{dst} = log2({lane});"),
                                    UOp::Reciprocal => {
                                        _ = writeln!(source, "{indent}{dst} = {}/{lane};", dtype.0.one_constant().c_code())
                                    }
                                    UOp::Sqrt => _ = writeln!(source, "{indent}{dst} = sqrt({lane});"),
                                    UOp::Sin => _ = writeln!(source, "{indent}{dst} = sin({lane});"),
                                    UOp::Cos => _ = writeln!(source, "{indent}{dst} = cos({lane});"),
                                    UOp::Floor => _ = writeln!(source, "{indent}{dst} = floor({lane});"),
                                    UOp::Trunc => _ = writeln!(source, "{indent}{dst} = trunc({lane});"),
                                    UOp::Abs => _ = writeln!(source, "{indent}{dst} = fabs({lane});"),
                                }
                            }
                        }
                        _ => match uop {
                            UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                            UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                            UOp::Exp => _ = writeln!(source, "{indent}r{reg} = exp({x});"),
                            UOp::Exp2 => _ = writeln!(source, "{indent}r{reg} = exp2({x});"),
                            UOp::Ln => _ = writeln!(source, "{indent}r{reg} = log({x});"),
                            UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                            UOp::Reciprocal => _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.0.one_constant().c_code()),
                            UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                            UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                            UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                            UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                            UOp::Trunc => _ = writeln!(source, "{indent}r{reg} = trunc({x});"),
                            UOp::Abs => _ = writeln!(source, "{indent}r{reg} = fabs({x});"),
                        },
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
                        MemLayout::Vector(n) => n,
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "C codegen: Vectorize requires Vector layout".into(),
                            });
                        }
                    };
                    _ = writeln!(source, "{indent}r{reg} = ({}){{{}}};", dtype.0.vec_type_name(vlen), vars);
                }
                Op::Wmma { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "C codegen does not support WMMA".into(),
                    });
                }
                Op::Index { vec, idx } => {
                    let dtype = dtypes[&op_id];
                    let vec = get_var(vec, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {};", lane_access(&vec, idx));
                }
                Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(n) => {
                            for i in 0..n {
                                let xl = lane_access(&x, i as usize);
                                let yl = lane_access(&y, i as usize);
                                emit_binary_op(&mut source, &indent, reg, i as usize, &xl, &yl, bop);
                            }
                        }
                        _ => emit_binary_op(&mut source, &indent, reg, usize::MAX, &x, &y, bop),
                    }
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let z = get_var(z, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {x} * {y} + {z};");
                }
                Op::If { condition } => {
                    let condition = get_var(condition, &constants, &indices, &reg_map, &mut registers, loop_id)?;
                    _ = writeln!(source, "{indent}if ({condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    if indent.len() < 2 {
                        indent = String::from("  ");
                    }
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Param { .. } => {}
                Op::Storage { dtype, scope, len } => {
                    debug_assert_eq!(scope, MemScope::Register, "C backend only supports register scoped storage");
                    _ = writeln!(source, "{indent}{} p{op_id}[{len}] __attribute__((aligned));", dtype.c_type(),);
                }
                Op::Barrier => {}
                Op::Asm { .. } => todo!(),
                Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "C codegen: ConstView/LoadView/StoreView/Move/Reduce should not appear".into(),
                    });
                }
            }
            op_id = self.next_op(op_id);
        }

        for _ in 0..index_loop_depth {
            indent.pop();
            indent.pop();
            if indent.len() < 2 {
                indent = String::from("  ");
            }
            _ = writeln!(source, "{indent}}}");
        }

        let mut reg_str = String::new();
        if !registers.is_empty() {
            let (dt, _, _) = registers[0];
            let mut prev_dt = dt;
            let prefix = "  ";
            _ = write!(
                reg_str,
                "{prefix}{} r0",
                match dt.1 {
                    MemLayout::Scalar => dt.0.c_type().into(),
                    MemLayout::Vector(len) => dt.0.vec_type_name(len),
                    MemLayout::Tile { .. } =>
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "C codegen: Tile layout not supported in register declarations".into()
                        }),
                }
            );
            for (i, (dt, _, _)) in (1..).zip(registers[1..].iter()) {
                if *dt == prev_dt {
                    _ = write!(reg_str, ", r{i}");
                } else {
                    _ = write!(
                        reg_str,
                        ";\n{prefix}{} r{i}",
                        match dt.1 {
                            MemLayout::Scalar => dt.0.c_type().into(),
                            MemLayout::Vector(len) => dt.0.vec_type_name(len),
                            MemLayout::Tile { .. } =>
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "C codegen: Tile layout not supported in register declarations".into()
                                }),
                        }
                    );
                }
                prev_dt = *dt;
            }
            _ = writeln!(reg_str, ";");
        }

        let f16_helpers = if !dtypes.values().any(|(dt, _)| matches!(dt, DType::F16 | DType::BF16)) {
            String::new()
        } else {
            r"static inline float f16tof32(unsigned short h) {
  unsigned int sign = (unsigned int)(h & 0x8000) << 16;
  unsigned int mantissa = (unsigned int)(h & 0x03FF);
  unsigned int exp = (unsigned int)((h >> 10) & 0x1F);
  unsigned int f;
  if (exp == 0) {
    if (mantissa == 0) { f = sign; }
    else {
      int e = -1; unsigned int m = mantissa;
      while ((m & 0x0400) == 0) { m <<= 1; e--; }
      f = sign | ((127 + e) << 23) | ((m & 0x03FF) << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000 | (mantissa << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (mantissa << 13);
  }
  float r; memcpy(&r, &f, sizeof(r)); return r;
}
static inline unsigned short f32tof16(float v) {
  unsigned int f; memcpy(&f, &v, sizeof(f));
  unsigned int sign = (f >> 16) & 0x8000;
  unsigned int exp = (f >> 23) & 0xFF;
  unsigned int mantissa = f & 0x007FFFFF;
  unsigned short h;
  if (exp == 0) { h = (unsigned short)sign; }
  else if (exp == 255) { h = (unsigned short)(sign | 0x7C00 | (mantissa >> 13)); }
  else {
    int new_exp = (int)exp - 127 + 15;
    if (new_exp >= 31) { h = (unsigned short)(sign | 0x7C00); }
    else if (new_exp <= 0) { h = (unsigned short)sign; }
    else { h = (unsigned short)(sign | (new_exp << 10) | (mantissa >> 13)); }
  }
  return h;
}
static inline float bf16tof32(unsigned short h) {
  unsigned int b = (unsigned int)h << 16; float r; memcpy(&r, &b, sizeof(r)); return r;
}
static inline unsigned short f32tobf16(float v) {
  unsigned int b; memcpy(&b, &v, sizeof(b)); return (unsigned short)(b >> 16);
}
"
            .to_string()
        };
        let bit_helpers = if !dtypes.values().any(|(dt, _)| matches!(dt, DType::F32 | DType::F64)) {
            String::new()
        } else {
            "static inline float u32tof32(unsigned int b) { union { unsigned int u; float f; } v; v.u = b; return v.f; }\n\
             static inline double u64tof64(unsigned long b) { union { unsigned long u; double f; } v; v.u = b; return v.f; }\n"
                .to_string()
        };
        let omp_include = if has_openmp { "#include <omp.h>\n" } else { "" };
        let mut vec_types = String::new();
        for (dt, _, _) in &registers {
            if let MemLayout::Vector(len) = dt.1 {
                let name = dt.0.vec_type_name(len);
                if !vec_types.contains(&format!("\ntypedef {} {name}", dt.0.c_type())) {
                    _ = writeln!(vec_types, "typedef {} {name} __attribute__((ext_vector_type({len})));", dt.0.c_type());
                }
            }
        }
        let nargs_check = if n_params > 0 {
            format!("  if (nargs != {n_params}) return;\n")
        } else {
            String::new()
        };
        Ok(format!(
            "#include <math.h>\n#include <stdint.h>\n#include <string.h>\n\
             {omp_include}\
             {vec_types}\
             {f16_helpers}\
             {bit_helpers}\
             void {name}(void** args, unsigned long nargs) {{\n\
             {nargs_check}\
             {global_cast}\
             {reg_str}\
             {source}}}\n"
        ))
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
        Ok(c.c_code())
    } else if let Some(&id) = indices.get(&op_id) {
        Ok(format!("idx{id}"))
    } else if let Some(&reg) = reg_map.get(&op_id) {
        if registers[reg].2 == loop_level {
            registers[reg].1 -= 1;
        }
        Ok(format!("r{reg}"))
    } else {
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("C codegen: variable {op_id} not found in constants, indices, or registers").into(),
        })
    }
}

fn lane_access(var: &str, lane: usize) -> String {
    if lane < 10 {
        format!("{var}.s{lane}")
    } else {
        // clang parses `.s10`..`.s15` as multi-lane swizzles, not decimal
        // lanes, so lanes >= 10 must use subscript indexing.
        format!("{var}[{lane}]")
    }
}

fn emit_binary_op(source: &mut String, indent: &str, reg: usize, lane: usize, x: &str, y: &str, bop: BOp) {
    let dst = if lane == usize::MAX {
        format!("r{reg}")
    } else {
        lane_access(&format!("r{reg}"), lane)
    };
    _ = match bop {
        BOp::Add => writeln!(source, "{indent}{dst} = {x} + {y};"),
        BOp::Sub => writeln!(source, "{indent}{dst} = {x} - {y};"),
        BOp::Mul => writeln!(source, "{indent}{dst} = {x} * {y};"),
        BOp::Div => writeln!(source, "{indent}{dst} = {x} / {y};"),
        BOp::Pow => writeln!(source, "{indent}{dst} = pow({x}, {y});"),
        BOp::Mod => writeln!(source, "{indent}{dst} = (int){x} % (int){y};"),
        BOp::Cmplt => writeln!(source, "{indent}{dst} = {x} < {y};"),
        BOp::Cmpgt => writeln!(source, "{indent}{dst} = {x} > {y};"),
        BOp::Cmpge => writeln!(source, "{indent}{dst} = {x} >= {y};"),
        BOp::Max => writeln!(source, "{indent}{dst} = fmax({x}, {y});"),
        BOp::Or => writeln!(source, "{indent}{dst} = {x} || {y};"),
        BOp::And => writeln!(source, "{indent}{dst} = {x} && {y};"),
        BOp::BitXor => writeln!(source, "{indent}{dst} = {x} ^ {y};"),
        BOp::BitOr => writeln!(source, "{indent}{dst} = {x} | {y};"),
        BOp::BitAnd => writeln!(source, "{indent}{dst} = {x} & {y};"),
        BOp::BitShiftLeft => writeln!(source, "{indent}{dst} = {x} << {y};"),
        BOp::BitShiftRight => writeln!(source, "{indent}{dst} = {x} >> {y};"),
        BOp::NotEq => writeln!(source, "{indent}{dst} = {x} != {y};"),
        BOp::Eq => writeln!(source, "{indent}{dst} = {x} == {y};"),
    };
}

impl DType {
    pub(crate) const fn c_type(self) -> &'static str {
        match self {
            Self::F64 => "double",
            Self::U8 | Self::Bool => "uint8_t",
            Self::U16 => "uint16_t",
            Self::U32 => "uint32_t",
            Self::U64 => "uint64_t",
            Self::I8 => "int8_t",
            Self::I16 => "int16_t",
            Self::I32 => "int32_t",
            Self::I64 => "int64_t",
            Self::F32 | Self::F16 | Self::BF16 => "float",
        }
    }

    fn vec_type_name(self, len: u16) -> String {
        format!("{}{}", self.c_type(), len).replace(' ', "_")
    }
}

impl Constant {
    pub fn c_code(self) -> String {
        match self {
            Self::F32(x) => {
                let val = f32::from_le_bytes(x);
                if val.is_finite() {
                    format!("{:.16}f", val)
                } else {
                    format!("u32tof32(0x{:08X}u)", val.to_bits())
                }
            }
            Self::F64(x) => {
                let val = f64::from_le_bytes(x);
                if val.is_finite() {
                    format!("{:.16}", val)
                } else {
                    format!("u64tof64(0x{:016X}ul)", val.to_bits())
                }
            }
            Self::U8(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{}ul", u64::from_le_bytes(x)),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::I32(x) => format!("(int){x}"),
            Self::I64(x) => format!("{}l", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{}", x as i32),
            Self::F16(x) => {
                let val = f16::from_le_bytes(x).to_f32();
                if val.is_finite() {
                    format!("{:.16}f", val)
                } else {
                    format!("u32tof32(0x{:08X}u)", val.to_bits())
                }
            }
            Self::BF16(x) => {
                let val = bf16::from_le_bytes(x).to_f32();
                if val.is_finite() {
                    format!("{:.16}f", val)
                } else {
                    format!("u32tof32(0x{:08X}u)", val.to_bits())
                }
            }
        }
    }
}
