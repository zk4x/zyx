// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! WGSL code generation: kernel IR lowered to a WebGPU compute shader.
//!
//! WGSL is the only shader language a browser accepts, and it differs from
//! SPIR-V in three ways this module has to absorb:
//!
//! - **No 64-bit integers.** Every `i64`/`u64` value is emitted as
//!   `i32`/`u32`, including `IDX_T` index math. A constant that does not fit
//!   in 32 bits is a loud error rather than a truncated literal.
//! - **No push constants.** Scalar `Variable` params (symbolic dims) become
//!   members of one `uniform` struct, in param order.
//! - **32-bit storage elements only.** `bool` buffers (one byte per element in
//!   zyx) and 64-bit integer buffers cannot be represented, so they are
//!   rejected instead of being silently mis-indexed.
//!
//! The body mirrors [`Kernel::generate_spirv`]'s op walk: each value-producing
//! op becomes a `let v<op_id>` binding, so no register allocation is involved.

use std::fmt::Write;

use crate::{
    DType, Map,
    backend::{GwsDim, gws_from_kernel},
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, GPUOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, SourceBlock, UOp},
    scalar::f16,
    shape::Dim,
};

fn wgsl_err(context: String) -> BackendError {
    BackendError { status: ErrorStatus::KernelCompilation, context: context.into() }
}

/// Scalar WGSL type for a zyx dtype. 64-bit integers narrow to 32 bits.
fn wgsl_scalar(dt: DType) -> Result<&'static str, BackendError> {
    Ok(match dt {
        DType::Bool => "bool",
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => "i32",
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => "u32",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2 => {
            return Err(wgsl_err(format!("WGSL: {dt} has no WebGPU scalar type")));
        }
    })
}

/// Element type of a storage buffer. `bool` is one byte per element in zyx but
/// 32 bits in WebGPU, and 64-bit integers have no WGSL type at all, so both are
/// rejected here rather than declared with the wrong stride.
fn wgsl_buffer_element(dt: DType) -> Result<&'static str, BackendError> {
    match dt {
        DType::Bool => Err(wgsl_err("WGSL: bool storage buffers are 1 byte in zyx but 32 bits in WebGPU".into())),
        DType::I64 | DType::U64 => Err(wgsl_err("WGSL: 64-bit integer storage buffers are not representable".into())),
        _ => wgsl_scalar(dt),
    }
}

/// WGSL type of a value with a lane shape. A one-dim shape is a vector (the
/// same lane model SPIR-V emits); tiles lower as scalars.
fn wgsl_type(dt: DType, shape: &[Dim]) -> Result<String, BackendError> {
    let scalar = wgsl_scalar(dt)?;
    Ok(match shape {
        [] | [1] => scalar.to_string(),
        [len] if (2..=4).contains(len) => format!("vec{len}<{scalar}>"),
        [len] => return Err(wgsl_err(format!("WGSL: lane width {len} has no vec type"))),
        [_, _] => scalar.to_string(),
        _ => return Err(wgsl_err("WGSL: rank-3+ value shape".into())),
    })
}

/// WGSL literal for a constant, in its narrowed 32-bit type.
fn wgsl_literal(c: Constant) -> Result<String, BackendError> {
    Ok(match c {
        Constant::Bool(v) => v.to_string(),
        Constant::I8(v) => format!("{v}i"),
        Constant::I16(v) => format!("{v}i"),
        Constant::I32(v) => format!("{v}i"),
        Constant::I64(b) => {
            let v = i64::from_le_bytes(b);
            let narrowed =
                i32::try_from(v).map_err(|_| wgsl_err(format!("WGSL: i64 constant {v} does not fit the 32-bit index type")))?;
            format!("{narrowed}i")
        }
        Constant::U8(v) => format!("{v}u"),
        Constant::U16(v) => format!("{v}u"),
        Constant::U32(v) => format!("{v}u"),
        Constant::U64(b) => {
            let v = u64::from_le_bytes(b);
            let narrowed = u32::try_from(v).map_err(|_| wgsl_err(format!("WGSL: u64 constant {v} does not fit u32")))?;
            format!("{narrowed}u")
        }
        Constant::F32(b) => {
            let v = f32::from_le_bytes(b);
            if v.is_finite() {
                format!("{v:?}")
            } else {
                // WGSL has no NaN/inf literals.
                format!("bitcast<f32>({:#010x}u)", v.to_bits())
            }
        }
        Constant::F16(b) => {
            let v = f16::from_le_bytes(b).to_f32();
            if v.is_finite() {
                format!("f16({v:?})")
            } else {
                return Err(wgsl_err("WGSL: non-finite f16 constant".into()));
            }
        }
        Constant::F64(_) | Constant::BF16(_) | Constant::F8E4M3(_) | Constant::F8E5M2(_) => {
            return Err(wgsl_err("WGSL: constant dtype has no WebGPU representation".into()));
        }
    })
}

/// `0`/`1` of the given WGSL type, splat over lanes: `vec3<f32>(1.0)`.
fn typed_one(ty: &str, is_float: bool, one: bool) -> String {
    let lit = match (is_float, one) {
        (true, false) => "0.0",
        (true, true) => "1.0",
        (false, false) => "0",
        (false, true) => "1",
    };
    if ty.starts_with("vec") {
        format!("{ty}({lit})")
    } else {
        lit.to_string()
    }
}

/// WGSL conversion expression from one scalar/vector type to another.
fn cast_expr(expr: &str, src: DType, src_ty: &str, dst: DType, dst_ty: &str) -> String {
    if src_ty == dst_ty {
        // Narrow integer aliases (`i8` -> `i32`, `u16` -> `u32`) are the same
        // WGSL type: the conversion is the identity.
        return expr.to_string();
    }
    if src == DType::Bool {
        format!("select({}, {}, {expr})", typed_one(dst_ty, dst.is_float(), false), typed_one(dst_ty, dst.is_float(), true))
    } else if dst == DType::Bool {
        format!("{expr} != {}", typed_one(src_ty, src.is_float(), false))
    } else {
        format!("{dst_ty}({expr})")
    }
}

/// Shift amounts must be `u32` in WGSL, scalars and vectors alike.
fn as_u32(expr: &str, ty: &str) -> String {
    match ty {
        "u32" => expr.to_string(),
        "i32" => format!("u32({expr})"),
        _ if ty.starts_with("vec") => format!("{}({expr})", ty.replacen("i32", "u32", 1)),
        _ => format!("u32({expr})"),
    }
}

impl Kernel {
    /// WGSL render: lower to a descriptor-only kernel — head order `Params`,
    /// `Grid`, `LocalWorkSize`, `Source`. The shader text is emitted here;
    /// the backend only hands it to the driver.
    pub(crate) fn render_wgsl(&self) -> Result<Kernel, BackendError> {
        // Max over local ranges per axis; the entry-point name derives from
        // this `lws` and must match the `name` rebuilt at launch.
        let mut lws = [1u32; 3];
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 50_000 {
                return Err(wgsl_err("render_wgsl: kernel exceeds 50000 op render limit".into()));
            }
            if let Op::Range { axis, kind: RangeKind::Local(len) } = self.ops[op_id].op
                && axis < 3
            {
                lws[axis as usize] = lws[axis as usize].max(len);
            }
            op_id = self.next_op(op_id);
        }
        if lws.iter().map(|&x| u64::from(x)).product::<u64>() > u64::from(self.dev_info().max_local_threads) {
            return Err(wgsl_err("Invalid local work size.".into()));
        }
        // Constant group lengths must fit the device grid; this same vector
        // becomes the rendered `Grid` below.
        let gws_vec = gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;
        let (dtypes, _rcs) = self.compute_dtypes_and_rcs();
        // Final capability guard: placement already filters devices, but a late
        // pass can introduce an op the chosen device cannot run.
        for (dt, caps) in &self.required_capabilities() {
            let have = self.dev_info().supports_dtype(*dt);
            if !have.contains(*caps) {
                return Err(wgsl_err(format!("dtype {dt:?} does not support required ops {caps:?} on this device")));
            }
        }

        // --- Collect addressable state in one walk -------------------------
        let mut globals: Vec<(OpId, DType, bool)> = Vec::new(); // (param, dtype, read_only)
        let mut locals: Vec<(OpId, DType, Dim)> = Vec::new();
        let mut regs: Vec<(OpId, DType, Dim)> = Vec::new();
        let mut variables: Vec<(OpId, DType)> = Vec::new();
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 50_000 {
                return Err(wgsl_err("render_wgsl: kernel exceeds 50000 op render limit".into()));
            }
            match self.at(op_id) {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Global => globals.push((op_id, *dtype, true)),
                    ParamKind::GlobalMut => globals.push((op_id, *dtype, false)),
                    ParamKind::Variable => variables.push((op_id, *dtype)),
                },
                Op::Storage { dtype, scope, len, .. } => match scope {
                    MemScope::Local => locals.push((op_id, *dtype, *len)),
                    MemScope::Register => regs.push((op_id, *dtype, *len)),
                    MemScope::Global | MemScope::Circular => {
                        return Err(wgsl_err("WGSL: unexpected global/circular storage".into()));
                    }
                },
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        let mut buffer_names: Map<OpId, String> = Map::default();
        let mut variable_members: Map<OpId, (usize, DType)> = Map::default();
        let mut shader = String::new();

        let uses_f16 = dtypes.values().any(|(dt, _)| *dt == DType::F16)
            || globals.iter().any(|g| g.1 == DType::F16)
            || locals.iter().any(|l| l.1 == DType::F16)
            || regs.iter().any(|r| r.1 == DType::F16)
            || variables.iter().any(|v| v.1 == DType::F16);
        if uses_f16 {
            shader.push_str("enable f16;\n\n");
        }

        // --- Bindings ------------------------------------------------------
        for (binding, &(op, dt, read_only)) in globals.iter().enumerate() {
            let elem = wgsl_buffer_element(dt)?;
            let access = if read_only { "read" } else { "read_write" };
            let _ = writeln!(shader, "@group(0) @binding({binding}) var<storage, {access}> p{binding}: array<{elem}>;");
            buffer_names.insert(op, format!("p{binding}"));
        }

        // Scalar `Variable` params: one uniform struct, in param order.
        if !variables.is_empty() {
            let _ = writeln!(shader, "\nstruct Params {{");
            for (member, &(_, dt)) in variables.iter().enumerate() {
                let ty = if dt == DType::Bool { "i32" } else { wgsl_scalar(dt)? };
                let _ = writeln!(shader, "  m{member}: {ty},");
            }
            let _ = writeln!(shader, "}}");
            let _ = writeln!(shader, "@group(0) @binding({}) var<uniform> params: Params;", globals.len());
            for (member, &(op, dt)) in variables.iter().enumerate() {
                variable_members.insert(op, (member, dt));
            }
        }

        for (i, &(op, dt, len)) in locals.iter().enumerate() {
            let elem = wgsl_scalar(dt)?;
            let _ = writeln!(shader, "var<workgroup> l{i}: array<{elem}, {len}>;");
            buffer_names.insert(op, format!("l{i}"));
        }

        // --- Entry point ---------------------------------------------------
        let name = format!("k_lws_{}_{}_{}", lws[0], lws[1], lws[2]);
        let _ = writeln!(shader, "\n@compute @workgroup_size({}, {}, {})", lws[0], lws[1], lws[2]);
        let _ = writeln!(shader, "fn {name}(");
        let _ = writeln!(shader, "  @builtin(workgroup_id) gid: vec3<u32>,");
        let _ = writeln!(shader, "  @builtin(local_invocation_id) lid: vec3<u32>");
        let _ = writeln!(shader, ") {{");

        // Register arrays are function-scope variables.
        for (i, &(op, dt, len)) in regs.iter().enumerate() {
            let elem = wgsl_scalar(dt)?;
            let _ = writeln!(shader, "  var r{i}: array<{elem}, {len}>;");
            buffer_names.insert(op, format!("r{i}"));
        }

        // --- Function body -------------------------------------------------
        let mut depth = 1usize;
        let mut scopes: Vec<()> = Vec::new();
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 50_000 {
                return Err(wgsl_err("render_wgsl: kernel exceeds 50000 op render limit".into()));
            }
            let ind = "  ".repeat(depth);
            match self.ops[op_id].op {
                Op::Source(_) | Op::GPU(_) | Op::Spirv(_) | Op::PTX(_) | Op::Asm { .. } => {
                    return Err(wgsl_err("WGSL: unexpected meta op in kernel body".into()));
                }
                Op::Unroll { .. } => {
                    return Err(wgsl_err("WGSL: unroll lane vector must be devectorized before codegen".into()));
                }
                Op::Select { cond, if_true, if_false } => {
                    let (dt, shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = select(v{if_false}, v{if_true}, v{cond});");
                }
                Op::TT { .. }
                | Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::Reduce { .. }
                | Op::Wmma { .. }
                | Op::After { .. }
                | Op::ToDevice { .. }
                | Op::Contiguous { .. }
                | Op::Program { .. }
                | Op::Kernel(_)
                | Op::Detach { .. } => {
                    return Err(wgsl_err("WGSL: unexpected kernel op (should be unfolded)".into()));
                }
                Op::Stack { ref ops } => {
                    let (dt, shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    let args: Vec<String> = ops.iter().map(|x| format!("v{x}")).collect();
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {ty}({});", args.join(", "));
                }
                Op::Index { vec, idx } => {
                    let (dt, _) = dtypes[&op_id].clone();
                    let ty = wgsl_scalar(dt)?;
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = v{vec}[{idx}];");
                }
                Op::Const(c) => {
                    let (dt, shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    let lit = wgsl_literal(c)?;
                    // Lane vectors and tiles broadcast through the constructor.
                    let expr = if shape.len() == 1 && shape[0] > 1 {
                        format!("{ty}({lit})")
                    } else {
                        lit
                    };
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {expr};");
                }
                Op::Param { kind, .. } => {
                    if matches!(kind, ParamKind::Variable) {
                        let (member, dt) = variable_members[&op_id];
                        if dt == DType::Bool {
                            let _ = writeln!(shader, "{ind}let v{op_id}: bool = params.m{member} != 0;");
                        } else {
                            let ty = wgsl_scalar(dt)?;
                            let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = params.m{member};");
                        }
                    }
                }
                Op::Storage { .. } => {}
                Op::Copy { .. } => {
                    return Err(wgsl_err("WGSL: Copy lowering is not implemented".into()));
                }
                Op::Load { src, src_idx } => {
                    let access_shape = self.shape(op_id);
                    if !access_shape.is_empty() {
                        return Err(wgsl_err("WGSL: vector/tile loads are not supported".into()));
                    }
                    let (dt, shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    if let Some(&(member, pdt)) = variable_members.get(&src) {
                        if pdt == DType::Bool {
                            let _ = writeln!(shader, "{ind}let v{op_id}: bool = params.m{member} != 0;");
                        } else {
                            let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = params.m{member};");
                        }
                    } else {
                        let Some(name) = buffer_names.get(&src) else {
                            return Err(wgsl_err("WGSL: load from unknown variable".into()));
                        };
                        let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {name}[v{src_idx}];");
                    }
                }
                Op::Store { dst, dst_idx, src } => {
                    let access_shape = self.shape(op_id);
                    if !access_shape.is_empty() {
                        return Err(wgsl_err("WGSL: vector/tile stores are not supported".into()));
                    }
                    if variable_members.contains_key(&dst) {
                        return Err(wgsl_err("WGSL: store to a scalar Variable param".into()));
                    }
                    let Some(name) = buffer_names.get(&dst) else {
                        return Err(wgsl_err("WGSL: store to unknown variable".into()));
                    };
                    let _ = writeln!(shader, "{ind}{name}[v{dst_idx}] = v{src};");
                }
                Op::Cast { x, dtype: _ } => {
                    let (src_dt, src_shape) = dtypes[&x].clone();
                    let (dst_dt, dst_shape) = dtypes[&op_id].clone();
                    let src_ty = wgsl_type(src_dt, &src_shape)?;
                    let dst_ty = wgsl_type(dst_dt, &dst_shape)?;
                    let expr = cast_expr(&format!("v{x}"), src_dt, &src_ty, dst_dt, &dst_ty);
                    let _ = writeln!(shader, "{ind}let v{op_id}: {dst_ty} = {expr};");
                }
                Op::Bitcast { x, dtype: _ } => {
                    let (src_dt, src_shape) = dtypes[&x].clone();
                    let (dst_dt, dst_shape) = dtypes[&op_id].clone();
                    let src_ty = wgsl_type(src_dt, &src_shape)?;
                    let dst_ty = wgsl_type(dst_dt, &dst_shape)?;
                    let expr = if src_ty == dst_ty {
                        // Narrow integer aliases are the same WGSL type; bool's
                        // representation is 0/1, so this is an exact reinterpre-
                        // tation either way.
                        format!("v{x}")
                    } else if dst_dt == DType::Bool {
                        format!("v{x} != {}", typed_one(&src_ty, src_dt.is_float(), false))
                    } else if src_dt == DType::Bool {
                        format!(
                            "select({}, {}, v{x})",
                            typed_one(&dst_ty, dst_dt.is_float(), false),
                            typed_one(&dst_ty, dst_dt.is_float(), true)
                        )
                    } else {
                        format!("bitcast<{dst_ty}>(v{x})")
                    };
                    let _ = writeln!(shader, "{ind}let v{op_id}: {dst_ty} = {expr};");
                }
                Op::Unary { x, uop } => {
                    let (dt, shape) = dtypes[&x].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    let arg = format!("v{x}");
                    let expr = match uop {
                        UOp::Neg => format!("-{arg}"),
                        UOp::BitNot => format!("~{arg}"),
                        UOp::Not => format!("!{arg}"),
                        UOp::Exp2 => format!("exp2({arg})"),
                        UOp::Log2 => format!("log2({arg})"),
                        UOp::Sqrt => format!("sqrt({arg})"),
                        UOp::Rsqrt => format!("inverseSqrt({arg})"),
                        UOp::Sin => format!("sin({arg})"),
                        UOp::Cos => format!("cos({arg})"),
                        UOp::Floor => format!("floor({arg})"),
                        UOp::Trunc => format!("trunc({arg})"),
                        UOp::Abs => format!("abs({arg})"),
                        UOp::Reciprocal => format!("{} / {arg}", typed_one(&ty, dt.is_float(), true)),
                    };
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {expr};");
                }
                Op::Binary { x, y, bop } => {
                    let (dt, _) = dtypes[&x].clone();
                    let (res_dt, res_shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(res_dt, &res_shape)?;
                    if dt == DType::Bool && !matches!(bop, BOp::Or | BOp::And | BOp::Eq | BOp::NotEq) {
                        return Err(wgsl_err(format!("WGSL: unsupported bool binary op {bop:?}")));
                    }
                    let xs = format!("v{x}");
                    let ys = format!("v{y}");
                    let expr = match bop {
                        BOp::Add => format!("{xs} + {ys}"),
                        BOp::Sub => format!("{xs} - {ys}"),
                        BOp::Mul => format!("{xs} * {ys}"),
                        BOp::Div => format!("{xs} / {ys}"),
                        BOp::Mod => format!("{xs} % {ys}"),
                        BOp::Pow => format!("pow({xs}, {ys})"),
                        BOp::Max => format!("max({xs}, {ys})"),
                        BOp::Cmplt => format!("{xs} < {ys}"),
                        BOp::Cmpgt => format!("{xs} > {ys}"),
                        BOp::Cmpge => format!("{xs} >= {ys}"),
                        BOp::Eq => format!("{xs} == {ys}"),
                        BOp::NotEq => format!("{xs} != {ys}"),
                        BOp::Or => format!("{xs} || {ys}"),
                        BOp::And => format!("{xs} && {ys}"),
                        BOp::BitXor => format!("{xs} ^ {ys}"),
                        BOp::BitOr => format!("{xs} | {ys}"),
                        BOp::BitAnd => format!("{xs} & {ys}"),
                        BOp::BitShiftLeft | BOp::BitShiftRight => {
                            let (ydt, yshape) = dtypes[&y].clone();
                            let yty = wgsl_type(ydt, &yshape)?;
                            let amount = as_u32(&ys, &yty);
                            let op = if bop == BOp::BitShiftLeft { "<<" } else { ">>" };
                            format!("{xs} {op} {amount}")
                        }
                    };
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {expr};");
                }
                Op::Range { axis, kind } => {
                    let (dt, shape) = dtypes[&op_id].clone();
                    let ty = wgsl_type(dt, &shape)?;
                    let src = match kind {
                        RangeKind::Group(_) => "gid",
                        RangeKind::Local(_) => "lid",
                    };
                    let comp = match axis {
                        0 => "x",
                        1 => "y",
                        2 => "z",
                        _ => return Err(wgsl_err(format!("WGSL: work dimension {axis} exceeds 3"))),
                    };
                    let _ = writeln!(shader, "{ind}let v{op_id}: {ty} = {ty}({src}.{comp});");
                }
                Op::Loop { len } => {
                    if dtypes[&len].0 == DType::Bool {
                        let _ = writeln!(shader, "{ind}if v{len} {{");
                    } else {
                        let len_expr = match self.resolve_const(len).and_then(Constant::as_dim) {
                            Some(v) => wgsl_literal(Constant::I64(v.to_le_bytes()))?,
                            None => format!("v{len}"),
                        };
                        let _ = writeln!(
                            shader,
                            "{ind}for (var c{op_id}: i32 = 0i; c{op_id} < {len_expr}; c{op_id} = c{op_id} + 1i) {{"
                        );
                        let _ = writeln!(shader, "{ind}  let v{op_id}: i32 = c{op_id};");
                    }
                    scopes.push(());
                    depth += 1;
                }
                Op::EndLoop => {
                    if scopes.pop().is_none() {
                        return Err(wgsl_err("WGSL: EndLoop without Loop".into()));
                    }
                    depth -= 1;
                    let ind = "  ".repeat(depth);
                    let _ = writeln!(shader, "{ind}}}");
                }
                Op::Barrier => {
                    let _ = writeln!(shader, "{ind}workgroupBarrier();");
                }
            }
            op_id = self.next_op(op_id);
        }
        if !scopes.is_empty() {
            return Err(wgsl_err(format!("WGSL: {} unclosed loop scope(s)", scopes.len())));
        }
        let _ = writeln!(shader, "}}");

        if gws_vec.len() > 3 {
            return Err(wgsl_err(format!("wgsl render: grid rank {} exceeds 3D", gws_vec.len())));
        }
        let mut gws = [GwsDim::Const(1), GwsDim::Const(1), GwsDim::Const(1)];
        for (i, g) in gws_vec.into_iter().enumerate() {
            gws[i] = g;
        }
        let mut params = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Param { kind, .. } = self.ops[op_id].op {
                params.push(kind);
            }
            op_id = self.next_op(op_id);
        }
        let mut rendered = Kernel {
            ops: crate::slab::Slab::new(),
            head: OpId::NULL,
            tail: OpId::NULL,
            dev: self.dev,
            dev_info: self.dev_info.clone(),
        };
        rendered.push_back(Op::GPU(Box::new(GPUOp::Params(params.into_boxed_slice()))));
        rendered.push_back(Op::GPU(Box::new(GPUOp::Grid(gws))));
        rendered.push_back(Op::GPU(Box::new(GPUOp::LocalWorkSize(lws))));
        rendered.push_back(Op::Source(SourceBlock(shader.into_boxed_str())));
        Ok(rendered)
    }
}
