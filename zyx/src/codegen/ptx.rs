// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! PTX assembly codegen from zyx kernel IR.

use crate::{
    DType, Map,
    backend::DeviceInfo,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope, UOp},
    scalar::{bf16, f16},
    shape::Dim,
};
use std::fmt::Write;

impl DType {
    fn ptx(&self) -> &'static str {
        match self {
            Self::BF16 => todo!("BF16 PTX codegen is WIP"),
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "s8",
            Self::I16 => "s16",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::Bool => "pred",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

impl Constant {
    fn ptx(&self) -> String {
        fn format_precise(val: impl std::fmt::Display, decimals: usize) -> String {
            let s = format!("{:.*}", decimals, val);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            if s.contains('.') { s.to_string() } else { format!("{s}.0") }
        }
        match self {
            Self::BF16(x) => format!("{}f", f32::from(bf16::from_le_bytes(*x))),
            Self::F16(x) => format!("__float2half({:.6})", f16::from_le_bytes(*x).to_f32()),
            Self::F32(x) => format_precise(f32::from_le_bytes(*x), 9),
            Self::F64(x) => format_precise(f64::from_le_bytes(*x), 18),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}U"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(*x)),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{}", i64::from_le_bytes(*x)),
            Self::Bool(x) => format!("{}", if *x { 1 } else { 0 }),
        }
    }
}

struct Compiler {
    var_map: Map<OpId, u16>,
    loops: Vec<(Dim, u16, u16)>,
    header: String,
    body: String,
    indent: String,
    registers: Vec<(DType, u32)>,
    scopes: Map<OpId, Scope>,
}

impl Compiler {
    fn bop_to_ptx(&self, bop: BOp, dtype: DType) -> &'static str {
        match bop {
            BOp::Add => "add",
            BOp::Sub => "sub",
            BOp::Mul => {
                if matches!(dtype, DType::F32 | DType::F64) {
                    "mul"
                } else {
                    "mul.lo"
                }
            }
            BOp::Div => {
                if matches!(dtype, DType::F32 | DType::F64) {
                    "div.approx"
                } else {
                    "div"
                }
            }
            BOp::Pow => todo!(),
            BOp::Mod => "rem",
            BOp::Cmplt => "setp.lt",
            BOp::Cmpgt => "setp.gt",
            BOp::Max => "max",
            BOp::Or => "or",
            BOp::And => "and",
            BOp::BitXor => "xor",
            BOp::BitOr => "or",
            BOp::BitAnd => "and",
            BOp::BitShiftLeft => "shl.b32",
            BOp::BitShiftRight => "shr.b32",
            BOp::NotEq => "setp.ne",
            BOp::Eq => "setp.eq",
        }
    }

    fn uop_to_ptx(&self, uop: UOp) -> Result<&'static str, BackendError> {
        match uop {
            UOp::Neg => Ok("neg"),
            UOp::BitNot => Ok("not"),
            UOp::Exp => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: UOp::Exp should be converted to Exp2 + mul by ln2(e) before reaching PTX backend".into(),
            }),
            UOp::Exp2 => Ok("ex2.approx"),
            UOp::Log2 => Ok("lg2.approx"),
            UOp::Reciprocal => Ok("rcp.approx"),
            UOp::Sqrt => Ok("sqrt.approx"),
            UOp::Sin => Ok("sin.approx"),
            UOp::Cos => Ok("cos.approx"),
            UOp::Floor => Ok("floor.approx"),
            UOp::Trunc => Ok("trunc.approx"),
            UOp::Ln => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: UOp::Ln should be converted to Log2 + mul by ln(2) before reaching PTX backend".into(),
            }),
            UOp::Abs => Ok("abs"),
        }
    }

    fn get_scope(&self, var: OpId) -> Scope {
        self.scopes.get(&var).copied().unwrap_or(Scope::Register)
    }

    fn new_reg(&mut self, dtype: DType, rc: u32) -> u16 {
        for (i, reg) in self.registers.iter_mut().enumerate() {
            if reg.1 == 0 && reg.0 == dtype {
                reg.1 = rc;
                return i as u16;
            }
        }
        let i = self.registers.len();
        self.registers.push((dtype, rc));
        i as u16
    }

    fn new_var(&mut self, op_id: OpId, dtype: DType, rc: u32) -> u16 {
        let i = self.new_reg(dtype, rc);
        self.var_map.insert(op_id, i);
        i
    }

    fn get_var(&mut self, x: OpId) -> u16 {
        let r = self.var_map[&x];
        self.registers[r as usize].1 -= 1;
        r
    }

    fn release_reg(&mut self, x: u16) {
        self.registers[x as usize].1 -= 1;
    }
}

impl Kernel {
    /// Compile kernel to PTX assembly.
    pub fn generate_ptx(
        &self,
        cc: [i32; 2],
        _dev_info: &DeviceInfo,
        debug: bool,
    ) -> Result<(Vec<u8>, Box<str>, Vec<Dim>, Vec<Dim>), BackendError> {
        let mut comp = Compiler {
            var_map: Map::default(),
            loops: Vec::new(),
            header: String::new(),
            body: String::new(),
            indent: "  ".to_string(),
            registers: Vec::new(),
            scopes: Map::default(),
        };

        let mut gws = vec![1; 3];
        let mut lws = vec![1; 3];
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::GroupIndex { len, axis } => gws[axis as usize] = len,
                Op::LocalIndex { len, axis } => lws[axis as usize] = len,
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if lws.iter().product::<Dim>() > _dev_info.max_local_threads {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }
        let name = format!(
            "k_{}__{}",
            gws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
        )
        .into_boxed_str();

        _ = writeln!(comp.header, ".version {0}.{1}\n.target sm_{0}{1}\n.address_size 64\n.visible .entry {name}(", cc[0], cc[1]);
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Define { scope: Scope::Global, .. }) {
                writeln!(comp.header, "{}.param .u64 g{op_id},", comp.indent).unwrap();
            }
            op_id = self.next_op(op_id);
        }
        comp.header.pop();
        comp.header.pop();
        _ = writeln!(comp.header, "\n) {{");

        let mut loop_id_label_map: Map<u8, u32> = Map::default();
        let mut label = 0;

        let (dtypes, rcs) = self.compute_dtypes_and_rcs();
        let mut loop_id: u8 = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Define { dtype, scope, len, .. } => {
                    comp.scopes.insert(op_id, scope);
                    match scope {
                        Scope::Global => {
                            _ = writeln!(comp.body, "{}ld.param.u64 %p{op_id}, [g{op_id}];", comp.indent);
                        }
                        Scope::Local => todo!(),
                        Scope::Register => {
                            _ = writeln!(
                                comp.body,
                                "{}.local .align {} .{} %p{op_id}[{len}];",
                                comp.indent,
                                dtype.bit_size() / 8,
                                dtype.ptx()
                            );
                        }
                    }
                }
                Op::GroupIndex { axis, .. } => {
                    let reg = comp.new_var(op_id, IDX_T, rcs[&op_id]);
                    _ = writeln!(
                        comp.body,
                        "{}{}.u32 %r{reg}, %ctaid.{};",
                        comp.indent,
                        if IDX_T == DType::U64 { "cvt.u64" } else { "mov" },
                        ["x", "y", "z"][axis as usize],
                    );
                }
                Op::LocalIndex { axis, .. } => {
                    let reg = comp.new_var(op_id, IDX_T, rcs[&op_id]);
                    _ = writeln!(
                        comp.body,
                        "{}{}.u32 %r{reg}, %tid.{};",
                        comp.indent,
                        if IDX_T == DType::U64 { "cvt.u64" } else { "mov" },
                        ["x", "y", "z"][axis as usize],
                    );
                }
                Op::Const(ref constant) => {
                    let reg = comp.new_var(op_id, constant.dtype(), u32::MAX);
                    _ = writeln!(comp.body, "{}mov.{} %r{reg}, {};", comp.indent, constant.dtype().ptx(), constant.ptx());
                }
                Op::Load { src, index, .. } => {
                    let dtype = dtypes[&src].0;
                    match comp.get_scope(src) {
                        Scope::Global => {
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            let idx = comp.get_var(index);
                            let offset = comp.new_reg(DType::U64, 1);
                            let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                            if IDX_T == DType::U64 {
                                if offset != idx {
                                    _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                }
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u64 %address, %p{src}, %r{offset};", comp.indent);
                            comp.release_reg(offset);
                            _ = writeln!(comp.body, "{}ld.global.{} %r{reg}, [%address];", comp.indent, dtype.ptx());
                        }
                        Scope::Local => todo!(),
                        Scope::Register => {
                            let idx = comp.get_var(index);
                            let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                            _ = writeln!(comp.body, "{}ld.local.{} %r{reg}, [%p{src} + %r{idx}];", comp.indent, dtype.ptx());
                        }
                    }
                }
                Op::Store { dst, x, index, .. } => {
                    let dtype = dtypes[&x].0;
                    let byte_shift = (dtype.bit_size() / 8).ilog2();
                    let offset = comp.new_reg(DType::U64, 1);
                    match comp.get_scope(dst) {
                        Scope::Global => {
                            if dtype == DType::Bool {
                                let gstu = comp.new_reg(DType::U32, 1);
                                let idx = comp.get_var(index);
                                let x = comp.get_var(x);
                                _ = writeln!(comp.body, "{}selp.u32 %r{gstu}, 1, 0, %r{x};", comp.indent);
                                if IDX_T == DType::U64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst}, %r{offset};", comp.indent);
                                _ = writeln!(comp.body, "{}st.global.u8 [%address], %r{gstu};", comp.indent);
                                comp.release_reg(gstu);
                            } else {
                                let idx = comp.get_var(index);
                                let x = comp.get_var(x);
                                if IDX_T == DType::U64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst}, %r{offset};", comp.indent);
                                _ = writeln!(comp.body, "{}st.global.{} [%address], %r{x};", comp.indent, dtype.ptx());
                            }
                        }
                        Scope::Local => todo!(),
                        Scope::Register => {
                            let idx = comp.get_var(index);
                            let x = comp.get_var(x);
                            _ = writeln!(comp.body, "{}st.local.{} [%p{dst} + %r{idx}], %r{x};", comp.indent, dtype.ptx());
                        }
                    }
                    comp.release_reg(offset);
                }
                Op::Cast { x, dtype } => {
                    let xdtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                    match (dtype, xdtype) {
                        (DType::Bool, _) => {
                            if dtype.is_float() {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0.0;", comp.indent, xdtype.ptx());
                            } else {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0;", comp.indent, xdtype.ptx());
                            }
                        }
                        (_, DType::Bool) => {
                            if dtype == DType::F64 {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1.0, 0.0, %r{x};", comp.indent, dtype.ptx());
                            } else if dtype == DType::F32 {
                                let a = comp.new_reg(DType::F32, 0);
                                let b = comp.new_reg(DType::F32, 0);
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, %r{a}, %r{b}, %r{x};", comp.indent, dtype.ptx());
                            } else {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1, 0, %r{x};", comp.indent, dtype.ptx());
                            }
                        }
                        (DType::I32, DType::F32) => {
                            _ = writeln!(comp.body, "{}cvt.rni.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        (_, _) => {
                            _ = writeln!(comp.body, "{}cvt.rn.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                    }
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                    _ = writeln!(comp.body, "{}{}.{} %r{reg}, %r{x};", comp.indent, comp.uop_to_ptx(uop)?, dtype.ptx());
                }
                Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id].0;
                    let xr = comp.get_var(x);
                    let yr = comp.get_var(y);
                    let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{reg}, %r{xr}, %r{yr};",
                        comp.indent,
                        comp.bop_to_ptx(bop, dtype),
                        dtypes[&x].0.ptx(),
                    );
                }
                Op::Mad { x, y, z, .. } => {
                    let dtype = dtypes[&op_id].0;
                    let xr = comp.get_var(x);
                    let yr = comp.get_var(y);
                    let zr = comp.get_var(z);
                    let reg = comp.new_var(op_id, dtype, rcs[&op_id]);
                    let mul = comp.new_reg(dtype, 1);
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{mul}, %r{xr}, %r{yr};",
                        comp.indent,
                        comp.bop_to_ptx(BOp::Mul, dtype),
                        dtype.ptx(),
                    );
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{reg}, %r{mul}, %r{zr};",
                        comp.indent,
                        comp.bop_to_ptx(BOp::Add, dtype),
                        dtype.ptx(),
                    );
                    comp.release_reg(mul);
                }
                Op::Loop { len } => {
                    let dim = self.loop_len_dim(len);
                    let loop_idx = comp.new_var(op_id, IDX_T, rcs[&op_id]);
                    let loop_pred = comp.new_reg(DType::Bool, 2);
                    comp.loops.push((dim, loop_pred, loop_idx));
                    _ = writeln!(comp.body, "{}mov.{} %r{loop_idx}, 0;", comp.indent, IDX_T.ptx());
                    _ = writeln!(comp.body, "{}LOOP_{label}:", comp.indent);
                    loop_id_label_map.insert(loop_id, label);
                    label += 1;
                    comp.indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    loop_id -= 1;
                    if let Some((dim, loop_pred, loop_idx)) = comp.loops.pop() {
                        _ = writeln!(comp.body, "{}add.{} %r{loop_idx}, %r{loop_idx}, 1;", comp.indent, IDX_T.ptx());
                        writeln!(
                            comp.body,
                            "{}setp.lt.{} %r{loop_pred}, %r{loop_idx}, {};",
                            comp.indent,
                            IDX_T.ptx(),
                            Constant::idx(dim as u64).ptx()
                        )
                        .unwrap();
                        _ = writeln!(comp.body, "{}@%r{loop_pred} bra LOOP_{};", comp.indent, loop_id_label_map[&loop_id]);
                        comp.indent.pop();
                        comp.indent.pop();
                    }
                }
                Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Move { .. }
                | Op::Reduce { .. }
                | Op::Wmma { .. }
                | Op::Vectorize { .. }
                | Op::Devectorize { .. }
                | Op::If { .. }
                | Op::EndIf
                | Op::Barrier { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "PTX: unexpected kernel op (should be unfolded)".into(),
                    });
                }
            }
            op_id = self.next_op(op_id);
        }

        _ = writeln!(comp.body, "{}ret;\n}}", comp.indent);

        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Define { scope: Scope::Global, .. }) {
                _ = writeln!(comp.header, "{}.reg .s64 %p{op_id};", comp.indent);
            }
            op_id = self.next_op(op_id);
        }

        _ = writeln!(comp.header, "{}.reg .u64 %address;", comp.indent);
        for (reg_id, (dtype, _)) in comp.registers.iter().enumerate() {
            _ = writeln!(comp.header, "{}.reg .{} %r{reg_id};", comp.indent, dtype.ptx());
        }

        comp.header.push_str(&comp.body);

        if debug {
            println!("{}", comp.header);
        }
        Ok((comp.header.into_bytes(), name, gws, lws))
    }
}
