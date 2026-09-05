// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType, Map,
    backend::gws_from_kernel,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, UOp},
    scalar::{bf16, f16},
};
use std::hash::BuildHasherDefault;

const VEC_COMPONENTS: [&str; 16] = [
    "x", "y", "z", "w", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sa", "sb",
];

/// Bit width of one MMA input element.
const fn mma_input_bits(dtype: MMADType) -> u64 {
    match dtype {
        MMADType::f16_f16_f16_f32 | MMADType::f16_f16_f16_f16 => 16,
        MMADType::s8_s8_s32_s32 => 8,
        MMADType::s4_s4_s32_s32 => 4,
        MMADType::b1_b1_s32_xor_popc | MMADType::b1_b1_s32_and_popc => 1,
    }
}

/// (input type, accumulator type, op suffix, accumulator PTX constraint, accumulator C type)
const fn mma_dtype_parts(dtype: MMADType) -> (&'static str, &'static str, &'static str, char, &'static str) {
    match dtype {
        MMADType::f16_f16_f16_f32 => ("f16", "f32", "", 'f', "float"),
        MMADType::f16_f16_f16_f16 => ("f16", "f16", "", 'r', "unsigned"),
        MMADType::s8_s8_s32_s32 => ("s8", "s32", "", 'r', "unsigned"),
        MMADType::s4_s4_s32_s32 => ("s4", "s32", "", 'r', "unsigned"),
        MMADType::b1_b1_s32_xor_popc => ("b1", "s32", ".xor.popc", 'r', "unsigned"),
        MMADType::b1_b1_s32_and_popc => ("b1", "s32", ".and.popc", 'r', "unsigned"),
    }
}

/// MMA layout as a PTX instruction modifier.
const fn mma_layout_parts(layout: MMALayout) -> &'static str {
    match layout {
        MMALayout::row_col => "row.col",
    }
}

/// Name of the generated `__device__` helper for one (dims, layout, dtype) combo.
fn mma_helper_name(dims: MMADims, layout: MMALayout, dtype: MMADType) -> String {
    let (m, n, k) = dims.decompose_mnk();
    let (in_s, acc_s, op_s, _, _) = mma_dtype_parts(dtype);
    let layout_s = mma_layout_parts(layout).replace('.', "_");
    let op_name = op_s.replace(['.', '-'], "_");
    let prefix = if op_s.is_empty() { "" } else { "_" };
    format!("wmma_m{m}n{n}k{k}_{layout_s}_{acc_s}_{in_s}_{in_s}_{acc_s}{prefix}{op_name}")
}

/// Emit a `__device__` helper wrapping one `mma.sync` instruction.
///
/// Fragments are passed as 32-bit register arrays (a, b) and accumulator array (c).
/// Per PTX ISA, A holds m*k/32 elements per thread, B holds n*k/32, C/D holds
/// m*n/32 accumulator elements; sub-32-bit elements are bit-packed into registers.
fn mma_helper(dims: MMADims, layout: MMALayout, dtype: MMADType) -> String {
    use std::fmt::Write;
    let (m, n, k) = dims.decompose_mnk();
    let (in_s, acc_s, op_s, acc_ptype, acc_ct) = mma_dtype_parts(dtype);
    let in_bits = mma_input_bits(dtype);
    let na = (m * k * in_bits / 1024) as usize;
    let nb = (n * k * in_bits / 1024) as usize;
    let nc = (if acc_s == "f16" { m * n / 64 } else { m * n / 32 }) as usize;

    // D operands: %0..%{nc-1} (in-out accumulator), then A, then B.
    let mut d_list = String::from("{");
    let mut outs = String::new();
    for i in 0..nc {
        if i > 0 {
            d_list.push(',');
            outs.push_str(", ");
        }
        _ = write!(d_list, "%{i}");
        _ = write!(outs, "\"+{acc_ptype}\"(c[{i}])");
    }
    d_list.push('}');
    let mut a_list = String::from("{");
    let mut a_ins = String::new();
    for (i, r) in (0..na).enumerate() {
        if i > 0 {
            a_list.push(',');
            a_ins.push_str(", ");
        }
        _ = write!(a_list, "%{}", nc + r);
        _ = write!(a_ins, "\"r\"(a[{r}])");
    }
    a_list.push('}');
    let mut b_list = String::from("{");
    let mut b_ins = String::new();
    for (i, r) in (0..nb).enumerate() {
        if i > 0 {
            b_list.push(',');
            b_ins.push_str(", ");
        }
        _ = write!(b_list, "%{}", nc + na + r);
        _ = write!(b_ins, "\"r\"(b[{r}])");
    }
    b_list.push('}');

    let name = mma_helper_name(dims, layout, dtype);
    let mut s = String::new();
    _ = writeln!(s, "__device__ void {name}(unsigned* a, unsigned* b, {acc_ct}* c) {{");
    _ = writeln!(
        s,
        "  asm(\"mma.sync.aligned.m{m}n{n}k{k}.{}.{acc_s}.{in_s}.{in_s}.{acc_s}{op_s} \\n\\t\"",
        mma_layout_parts(layout)
    );
    _ = writeln!(s, "    \"{d_list}, {a_list}, {b_list}, {d_list};\"");
    _ = writeln!(s, "    : {outs}");
    _ = writeln!(s, "    : {a_ins}, {b_ins});");
    _ = writeln!(s, "}}");
    s
}

impl Kernel {
    /// Compile kernel to CUDA C++ source code.
    pub fn generate_cuda(&self, name: &str) -> Result<String, BackendError> {
        use std::fmt::Write;

        // Reject group lengths that are constant and exceed the device grid limits.
        gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;

        let mut global_args = String::new();
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_cuda did not finish in 10000 steps");
            }
            let op = &self.ops[op_id].op;
            if let &Op::Param { dtype, kind, .. } = op {
                match kind {
                    ParamKind::Variable => _ = writeln!(global_args, "  {} p{op_id},", dtype.cu()),
                    ParamKind::Global => _ = writeln!(global_args, "  const {}* p{op_id},", dtype.cu()),
                    ParamKind::GlobalMut => _ = writeln!(global_args, "  {}* p{op_id},", dtype.cu()),
                }
            }
            // Walk the ENTIRE op list: Param defines may sit anywhere in the
            // kernel (merges append them after compute ops), and the argument
            // order must match the positional `loads` mapping used at launch.
            op_id = self.next_op(op_id);
        }
        global_args.pop();
        global_args.pop();
        global_args.push('\n');

        // Variable params are runtime scalars addressable directly as `p{op_id}`.
        let mut var_params: Map<OpId, DType> = Map::with_hasher(BuildHasherDefault::new());
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Param { dtype, kind: ParamKind::Variable, .. } = &self.ops[op_id].op {
                var_params.insert(op_id, *dtype);
            }
            op_id = self.next_op(op_id);
        }

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
                panic!("generate_cuda did not finish in 10000 steps");
            }
            match self.ops[op_id].op {
                Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "CUDA codegen: unexpected kernel op (should be unfolded)".into(),
                    });
                }
                Op::Asm { .. } => todo!(),
                Op::Const(x) => {
                    constants.insert(op_id, x);
                }
                Op::Param { .. } => {}
                Op::Storage { dtype, scope, len } => match scope {
                    MemScope::Local => _ = writeln!(source, "{indent}__shared__ {} p{op_id}[{len}];", dtype.cu()),
                    MemScope::Register => _ = writeln!(source, "{indent}{} p{op_id}[{len}];", dtype.cu()),
                    _ => unreachable!("cuda supports only local and register scopes"),
                },
                Op::Load { src, index, layout } => {
                    if rcs.contains_key(&op_id) {
                        let dtype = dtypes[&op_id];
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                        if matches!(self.ops[src].op, Op::Param { kind: ParamKind::Variable, .. }) {
                            _ = writeln!(source, "{indent}r{reg} = p{src};");
                        } else {
                            let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                            match layout {
                                MemLayout::Scalar => _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];"),
                                MemLayout::Vector(len) => {
                                    _ = writeln!(
                                        source,
                                        "{indent}r{reg} = *reinterpret_cast<const {}*>(&p{src}[{idx}]);",
                                        dtype.0.cu_vec_type(len)
                                    )
                                }
                                MemLayout::Tile { .. } => todo!(),
                            }
                        }
                    }
                }
                Op::Store { dst, src, index, layout } => {
                    let x = get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    match layout {
                        MemLayout::Scalar => _ = writeln!(source, "{indent}p{dst}[{idx}] = {x};"),
                        MemLayout::Vector(len) => {
                            let vec_type = dtypes[&src].0.cu_vec_type(len);
                            _ = writeln!(source, "{indent}*reinterpret_cast<{vec_type}*>(&p{dst}[{idx}]) = {x};",);
                        }
                        MemLayout::Tile { .. } => todo!(),
                    }
                }
                Op::Wmma { dims, layout, dtype, c, a, b } => {
                    let a = get_var(a, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let b = get_var(b, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let c = get_var(c, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtypes[&op_id], rcs[&op_id], loop_id);
                    let name = mma_helper_name(dims, layout, dtype);
                    let c_cast = mma_dtype_parts(dtype).4;
                    _ = writeln!(source, "{indent}{name}((unsigned*)&{a}, (unsigned*)&{b}, ({c_cast}*)&{c});");
                    _ = writeln!(source, "{indent}r{reg} = {c};");
                }
                Op::Cast { x, dtype } => {
                    let x_var = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let mem_layout = dtypes[&x].1;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, mem_layout), rcs[&op_id], loop_id);
                    if dtype == DType::BF16 {
                        _ = writeln!(source, "{indent}r{reg} = __float2bfloat16((float){x_var});");
                    } else {
                        match mem_layout {
                            MemLayout::Vector(len) => {
                                for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                    _ = writeln!(source, "{indent}r{reg}.{c} = ({}){x_var}.{c};", dtype.cu());
                                }
                            }
                            _ => _ = writeln!(source, "{indent}r{reg} = ({}){x_var};", dtype.cu()),
                        }
                    }
                }
                Op::Bitcast { x, dtype } => {
                    let x_var = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let mem_layout = dtypes[&x].1;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, mem_layout), rcs[&op_id], loop_id);
                    let byte_size = dtype.bit_size() as usize / 8;
                    match mem_layout {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = writeln!(source, "{indent}memcpy(&r{reg}.{c}, &{x_var}.{c}, {byte_size});");
                            }
                        }
                        _ => _ = writeln!(source, "{indent}memcpy(&r{reg}, &{x_var}, {byte_size});"),
                    }
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match dtype.1 {
                        MemLayout::Vector(len) => {
                            for &c in VEC_COMPONENTS.iter().take(len as usize) {
                                _ = match uop {
                                    UOp::BitNot => writeln!(source, "{indent}r{reg}.{c} = ~{x}.{c};"),
                                    UOp::Not => writeln!(source, "{indent}r{reg}.{c} = !{x}.{c};"),
                                    UOp::Neg => writeln!(source, "{indent}r{reg}.{c} = -{x}.{c};"),
                                    UOp::Exp => return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "CUDA codegen: UOp::Exp should be converted to Exp2 + mul by ln2(e) before reaching CUDA backend".into() }),
                                    UOp::Exp2 => {
                                        if dtype.0 == DType::F16 {
                                            writeln!(source, "{indent}r{reg}.{c} = (half)exp2((float){x}.{c});")
                                        } else {
                                            writeln!(source, "{indent}r{reg}.{c} = exp2({x}.{c});")
                                        }
                                    }
                                    UOp::Log2 => writeln!(source, "{indent}r{reg}.{c} = log2({x}.{c});"),
                                    UOp::Reciprocal => {
                                        writeln!(source, "{indent}r{reg}.{c} = {}/{x}.{c};", dtype.0.one_constant().cu())
                                    }
                                    UOp::Sqrt => writeln!(source, "{indent}r{reg}.{c} = sqrt({x}.{c});"),
                                    UOp::Sin => writeln!(source, "{indent}r{reg}.{c} = sin({x}.{c});"),
                                    UOp::Cos => writeln!(source, "{indent}r{reg}.{c} = cos({x}.{c});"),
                                    UOp::Floor => writeln!(source, "{indent}r{reg}.{c} = floor({x}.{c});"),
                                    UOp::Trunc => writeln!(source, "{indent}r{reg}.{c} = trunc({x}.{c});"),
                                    UOp::Ln => writeln!(source, "{indent}r{reg}.{c} = log({x}.{c});"),
                                    UOp::Abs => writeln!(source, "{indent}r{reg}.{c} = fabsf({x}.{c});"),
                                };
                            }
                        }
                        MemLayout::Scalar => match uop {
                            UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                            UOp::Not => _ = writeln!(source, "{indent}r{reg} = !{x};"),
                            UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                            UOp::Exp => return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "CUDA codegen: UOp::Exp should be converted to Exp2 + mul by ln2(e) before reaching CUDA backend".into() }),
                            UOp::Exp2 => {
                                if dtype.0 == DType::F16 {
                                    _ = writeln!(source, "{indent}r{reg} = (half)exp2((float){x});");
                                } else {
                                    _ = writeln!(source, "{indent}r{reg} = exp2({x});");
                                }
                            }
                            UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                            UOp::Reciprocal => {
                                _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.0.one_constant().cu());
                            }
                            UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                            UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                            UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                            UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                            UOp::Trunc => _ = writeln!(source, "{indent}r{reg} = trunc({x});"),
                            UOp::Ln => _ = writeln!(source, "{indent}r{reg} = log({x});"),
                            UOp::Abs => _ = writeln!(source, "{indent}r{reg} = fabsf({x});"),
                        },
                        MemLayout::Tile { .. } => return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "CUDA codegen: Tile layout not supported for Unary".into() }),
                    }
                }
                Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
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
                                    BOp::Mod if dtype.0.is_float() => {
                                        writeln!(source, "{indent}r{reg}.{c} = fmodf({x}.{c}, {y}.{c});")
                                    }
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
                                BOp::Mod if dtype.0.is_float() => writeln!(source, "{indent}r{reg} = fmodf({x}, {y});"),
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
                                context: "CUDA codegen: Tile layout not supported for Binary".into(),
                            });
                        }
                    }
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&op_id];
                    let mut vars = String::new();
                    for &x in ops.iter() {
                        let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                        _ = write!(vars, "{x}, ");
                    }
                    vars.pop();
                    vars.pop();
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {{{vars}}};");
                }
                Op::Index { vec, idx } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(vec, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {x}.{};", VEC_COMPONENTS[idx]);
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    let z = get_var(z, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
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
                Op::Range { axis, kind: scope } => {
                    indices.insert(op_id, loop_id);
                    let axis_letter = ["x", "y", "z"][axis as usize];
                    let (idx_expr, max_idx) = match scope {
                        // Dynamic dims are `-1`; the bound is only a source comment.
                        RangeKind::Group(len_id) => {
                            let max = self
                                .resolve_const(len_id)
                                .and_then(crate::dtype::Constant::as_dim)
                                .unwrap_or(-1)
                                .saturating_sub(1);
                            (format!("blockIdx.{axis_letter}"), max)
                        }
                        RangeKind::Local(len) => (format!("threadIdx.{axis_letter}"), i64::from(len).saturating_sub(1)),
                        // Lane id: the referenced thread id mod warp size. The
                        // referenced local range is emitted before (head order),
                        // so its index variable already exists.
                        RangeKind::Warp(local_id) => {
                            let lidx = indices
                                .get(&local_id)
                                .copied()
                                .expect("warp range must reference an already-emitted local range");
                            let warp_size = self.dev_info().warp_size;
                            (format!("idx{lidx} % {warp_size}"), i64::from(warp_size) - 1)
                        }
                    };
                    let idx_type = self.dtype(op_id).cu();
                    _ = writeln!(source, "{indent}{idx_type} idx{loop_id} = {idx_expr}; // 0..={max_idx}");
                    loop_id += 1;
                }
                Op::Loop { len, .. } => {
                    indices.insert(op_id, loop_id);
                    let len = get_var(len, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    _ = writeln!(
                        source,
                        "{indent}for ({idx_type} idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{",
                        idx_type = self.dtype(op_id).cu()
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
                    let condition = get_var(condition, &constants, &indices, &reg_map, &mut registers, loop_id, &var_params)?;
                    _ = writeln!(source, "{indent}if ({condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Barrier => _ = writeln!(source, "{indent}__syncthreads();"),
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
                    MemLayout::Scalar => dt.0.cu().to_string(),
                    MemLayout::Vector(len) => dt.0.cu_vec_type(len),
                    MemLayout::Tile { .. } =>
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "CUDA codegen: Tile layout not supported in register declarations".into()
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
                            MemLayout::Scalar => dt.0.cu().to_string(),
                            MemLayout::Vector(len) => dt.0.cu_vec_type(len),
                            MemLayout::Tile { .. } =>
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "CUDA codegen: Tile layout not supported in register declarations".into()
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
            pragma += "#include <cuda_fp16.h>\n";
            pragma += "struct __align__(8) half4 { half x, y, z, w; };\n";
            pragma += "struct __align__(16) half8 { half x, y, z, w, s4, s5, s6, s7; };\n";
        }
        if dtypes.values().any(|&x| x.0 == DType::BF16) {
            pragma += "#include <cuda_bf16.h>\n";
        }

        // Emit one mma helper per distinct (dims, layout, dtype) combo used by the kernel.
        let mut helper_funcs = String::new();
        let mut wmma_combos: Vec<(MMADims, MMALayout, MMADType)> = vec![];
        for (_, op) in self.iter_unordered() {
            if let Op::Wmma { dims, layout, dtype, .. } = op {
                let combo = (*dims, *layout, *dtype);
                if !wmma_combos.contains(&combo) {
                    wmma_combos.push(combo);
                }
            }
        }
        for combo in &wmma_combos {
            helper_funcs += &mma_helper(combo.0, combo.1, combo.2);
        }

        Ok(format!("{pragma}{helper_funcs}extern \"C\"\n__global__ void {name}(\n{global_args}) {{\n{reg_str}{source}}}\n\t\0"))
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
    var_params: &Map<OpId, DType>,
) -> Result<String, BackendError> {
    if var_params.contains_key(&op_id) {
        Ok(format!("p{op_id}"))
    } else if let Some(c) = constants.get(&op_id) {
        Ok(c.cu())
    } else if let Some(id) = indices.get(&op_id) {
        Ok(format!("idx{id}"))
    } else if let Some(reg) = reg_map.get(&op_id) {
        if loop_level == registers[*reg].2 {
            registers[*reg].1 -= 1;
        }
        Ok(format!("r{reg}"))
    } else {
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("CUDA codegen: variable {op_id} not found").into(),
        })
    }
}

impl DType {
    pub(super) const fn cu(&self) -> &'static str {
        match self {
            Self::BF16 => "__nv_bfloat16",
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::I8 | Self::U8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U16 => "unsigned short",
            Self::U32 => "unsigned int",
            Self::U64 => "unsigned long",
        }
    }
    pub(super) fn cu_vec_type(&self, len: u16) -> String {
        match self {
            Self::Bool => format!("uint{len}"),
            Self::U16 => format!("ushort{len}"),
            Self::U32 => format!("uint{len}"),
            Self::U64 => format!("ulong{len}"),
            other => format!("{}{len}", other.cu()),
        }
    }
}

impl Constant {
    fn cu(&self) -> String {
        fn format_precise(val: impl std::fmt::Display, decimals: usize) -> String {
            let s = format!("{val:.decimals$}");
            let s = s.trim_end_matches('0').trim_end_matches('.');
            if s.contains('.') { s.to_string() } else { format!("{s}.0") }
        }
        match self {
            &Self::BF16(x) => {
                let val: f32 = bf16::from_le_bytes(x).into();
                if val.is_finite() {
                    format!("__float2bfloat16({}f)", format_precise(val, 9))
                } else {
                    format!("__float2bfloat16(__int_as_float(0x{:08X}u))", val.to_bits())
                }
            }
            &Self::F16(x) => {
                let bits: u16 = f16::from_le_bytes(x).to_bits();
                format!("(half)0x{:04X}", bits)
            }
            &Self::F32(x) => {
                let val = f32::from_le_bytes(x);
                if val.is_finite() {
                    format!("{}f", format_precise(val, 9))
                } else {
                    format!("__int_as_float(0x{:08X}u)", val.to_bits())
                }
            }
            &Self::F64(x) => {
                let val = f64::from_le_bytes(x);
                if val.is_finite() {
                    format_precise(val, 18)
                } else {
                    format!("__longlong_as_double(0x{:016X}ull)", val.to_bits())
                }
            }
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}U"),
            &Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("(int){x}"),
            &Self::I64(x) => format!("{}", i64::from_le_bytes(x)),
            &Self::Bool(x) => format!("{}", x as i32),
        }
    }
}
